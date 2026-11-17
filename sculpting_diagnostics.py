"""Diagnose already-saved confirmation predictions without fitting or changing cuts.

Conditional random-selection references are descriptive, uncorrected diagnostics
on previously inspected data, not discovery tests or new confirmation evidence.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

BINS = np.linspace(3100, 3900, 17)


def wilson(k, n):
    if n == 0:
        return None
    z = 1.959963984540054
    rate = k / n
    center = (rate + z*z/(2*n)) / (1 + z*z/n)
    radius = z*np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def js_reference(counts, accepted, permutations, seed):
    counts, accepted = np.asarray(counts), np.asarray(accepted)
    if (counts.ndim != 1 or counts.shape != accepted.shape
            or not np.issubdtype(counts.dtype, np.integer)
            or not np.issubdtype(accepted.dtype, np.integer)
            or np.any(counts < 0) or np.any(accepted < 0)
            or np.any(accepted > counts) or permutations <= 0):
        raise ValueError('Invalid histogram or permutation count')
    n = int(accepted.sum())
    if counts.sum() == 0 or n == 0:
        return dict(status='undefined_empty_sample', observed=None, null_mean=None,
                    null_median=None, null_95_interval=None, p_upper=None)
    # Equivalent in distribution to permuting selection labels over the fixed
    # background events: histogram counts follow a multivariate hypergeometric.
    rng = np.random.default_rng(seed)
    sampled = rng.multivariate_hypergeometric(counts, n, size=permutations)
    null = jensenshannon(counts[None, :], sampled, base=2, axis=1)**2
    observed = float(jensenshannon(counts, accepted, base=2)**2)
    return dict(status='defined', observed=observed, null_mean=float(null.mean()),
                null_median=float(np.median(null)),
                null_95_interval=np.quantile(null, [.025, .975]).tolist(),
                p_upper=float((1 + np.count_nonzero(null >= observed))/(permutations+1)))


def diagnose(mass, truth, score, threshold, permutations=1999, seed=731):
    mass, truth, score = map(np.asarray, (mass, truth, score))
    if (mass.ndim != 1 or mass.shape != truth.shape or mass.shape != score.shape
            or not np.isfinite(mass).all() or not np.isfinite(score).all()
            or not np.isfinite(threshold) or not np.isin(truth, [0, 1]).all()
            or np.any((mass < 3100) | (mass >= 3900))):
        raise ValueError('Invalid prediction arrays or threshold')
    bg = truth == 0
    selected = score > threshold
    masks = {'lower_sb': bg & (mass < 3300),
             'sr': bg & (mass >= 3300) & (mass < 3700),
             'upper_sb': bg & (mass >= 3700),
             'sb': bg & ((mass < 3300) | (mass >= 3700))}
    regions = {}
    for name, mask in masks.items():
        n, k = int(mask.sum()), int((mask & selected).sum())
        regions[name] = dict(total=n, selected=k, efficiency=k/n if n else None,
                             conditional_wilson_95=wilson(k, n))
    esr, esb = regions['sr']['efficiency'], regions['sb']['efficiency']
    ratio = esr/esb if esr is not None and esb is not None and esb > 0 else None
    counts = np.histogram(mass[bg], BINS)[0]
    accepted = np.histogram(mass[bg & selected], BINS)[0]
    return dict(regions=regions, sr_to_measured_sb_ratio=ratio,
                sr_to_nominal_005_ratio=esr/.05 if esr is not None else None,
                background_total=int(bg.sum()), background_selected=int((bg & selected).sum()),
                background_score_std=float(score[bg].std()) if bg.any() else None,
                bin_total=counts.tolist(), bin_selected=accepted.tolist(),
                js=js_reference(counts, accepted, permutations, seed))


def checkpoint_diagnostic(path, strength):
    history = np.loadtxt(path, delimiter=',', ndmin=2)
    if history.shape[1] != 6 or not np.isfinite(history).all():
        raise ValueError(f'Invalid training history: {path}')
    def chosen(values):
        best, epoch = float('inf'), None
        for row, value in zip(history, values):
            if value < best-1e-6:
                best, epoch = value, int(row[0])
        return epoch
    if not np.allclose(history[:, 5], history[:, 3]+strength*history[:, 4], atol=1e-12, rtol=1e-10):
        raise ValueError(f'Objective mismatch: {path}')
    return dict(recorded_objective_epoch=chosen(history[:, 5]),
                bce_only_epoch_on_observed_trajectory=chosen(history[:, 3]),
                weighted_validation_penalty_range=[float(strength*history[:, 4].min()),
                                                   float(strength*history[:, 4].max())],
                caveat='BCE-only comparison uses observed epochs; it is not a counterfactual retraining or early-stop trajectory.')


def plot_efficiencies(records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fractions = sorted({row['fraction'] for row in records})
    fig, axes = plt.subplots(1, len(fractions), figsize=(6*len(fractions), 4), squeeze=False)
    for fraction, ax in zip(fractions, axes[0]):
        for model in ('mass_mlp', 'disco', 'efp_mlp'):
            group = [row for row in records if row['fraction'] == fraction and row['model'] == model]
            if not group:
                continue
            rates = []
            for row in group:
                total = np.asarray(row['bin_total'])
                rates.append(np.divide(row['bin_selected'], total, out=np.full(16, np.nan), where=total>0))
            rates = np.asarray(rates)
            # Omit a bin if any seed has no support rather than averaging a
            # changing set of seeds in different bins.
            valid = np.isfinite(rates).all(axis=0)
            ax.errorbar(((BINS[1:]+BINS[:-1])/2)[valid], rates[:, valid].mean(axis=0),
                        yerr=rates[:, valid].std(axis=0), fmt='.-', label=model)
        ax.axvspan(3300, 3700, alpha=.1, color='gray')
        ax.axhline(.05, color='gray', ls=':')
        ax.set(xlabel='Dijet mass [GeV]', ylabel='Background selection efficiency',
               title=f'Training fraction {fraction:.1%}')
        ax.legend()
    fig.suptitle('Previously inspected predictions; bars show seed SD, not confidence intervals', fontsize=10)
    fig.tight_layout()
    fig.savefig(output/'background_efficiency.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Root containing results_confirmation/')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--permutations', type=int, default=1999)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        parser.error('Keep diagnostic output separate from experiment artifacts')
    if args.permutations < 1:
        parser.error('permutations must be positive')
    metrics_path = run/'results_confirmation/metrics.csv'
    with metrics_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('No saved confirmation metrics')
    manifest = {}
    def fingerprint(path):
        manifest[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint(metrics_path)
    records, histories, keys = [], [], set()
    reference = None
    for row in rows:
        fraction, seed, model = float(row['fraction']), int(row['seed']), row['model']
        key = (fraction, seed, model)
        if key in keys or model not in ('mass_mlp', 'disco', 'efp_mlp'):
            raise ValueError(f'Duplicate or unsupported model key: {key}')
        keys.add(key)
        path = run/'results_confirmation'/f'f{fraction:g}_seed{seed}_{model}.npz'
        fingerprint(path)
        with np.load(path, allow_pickle=False) as saved:
            ids = saved['rows']
            if len(np.unique(ids)) != len(ids) or len(ids) != len(saved['mass']):
                raise ValueError('Invalid event IDs')
            if reference is None:
                reference = ids.copy()
            if not np.array_equal(reference, ids):
                raise ValueError('Models do not share the same evaluation rows')
            detail = diagnose(saved['mass'], saved['truth'], saved['score'], float(saved['threshold']),
                              args.permutations)
        records.append(dict(fraction=fraction, seed=seed, model=model, **detail))
        folder = (run/'results_tuning'/f'seed{seed}_f{fraction:g}_lambda0.003' if model == 'disco'
                  else run/'results_study'/f'seed{seed}_f{fraction:g}_{model}')
        history = folder/'training.csv'
        if history.exists():
            fingerprint(history)
            histories.append(dict(fraction=fraction, seed=seed, model=model,
                                  **checkpoint_diagnostic(history, .003 if model == 'disco' else 0)))
    output.mkdir(parents=True, exist_ok=True)
    result = dict(run=str(run), analysis='Exploratory diagnostics of already-inspected saved predictions',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256=manifest, numpy_version=np.__version__,
                  permutations=args.permutations, permutation_seed=731,
                  null='Uniform subset of fixed size from fixed background events; all 16 mass bins; base-2 JS',
                  caveats=['Uncorrected conditional p-values; failure to reject does not prove independence.',
                           'Wilson intervals condition on the fitted model and threshold.',
                           'Seed variability is not independent-event uncertainty.',
                           'No new inference, training, cut optimization or source-data validation.'],
                  records=records, checkpoint_diagnostics=histories)
    (output/'diagnostics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Saved-prediction sculpting diagnostics', '', f'Run: `{run}`', '',
             'Previously inspected samples; exploratory diagnostics, not independent confirmation.',
             'Region efficiencies below are means across the listed fitted seeds; ratios divide these means.',
             '', '| Fraction | Model | Seeds | Lower SB | SR | Upper SB | Combined SB | SR / measured SB |',
             '|---|---|---|---|---|---|---|---|']
    for fraction, model in sorted({(r['fraction'], r['model']) for r in records}):
        group = [r for r in records if r['fraction'] == fraction and r['model'] == model]
        values = {name: [r['regions'][name]['efficiency'] for r in group]
                  for name in ('lower_sb', 'sr', 'upper_sb', 'sb')}
        if any(v is None for val in values.values() for v in val):
            lines.append(f'| {fraction:g} | {model} | {len(group)} | unavailable | unavailable | unavailable | unavailable | unavailable |')
            continue
        means = [float(np.mean(values[name])) for name in ('lower_sb', 'sr', 'upper_sb', 'sb')]
        ratio = f'{means[1]/means[3]:.4f}' if means[3] > 0 else 'undefined'
        lines.append(f'| {fraction:g} | {model} | {len(group)} | '+' | '.join(f'{v:.5f}' for v in means)+f' | {ratio} |')
    lines += ['', '## EFP-only JS conditional null checks', '',
              f'{args.permutations} random subsets per model, preserving its actual selected count.',
              'These exploratory p-values are uncorrected across models, seeds and fractions.', '',
              '| Fraction | Seed | Observed JS | Null mean | Null 95% interval | Upper-tail p |',
              '|---|---|---|---|---|---|']
    for row in records:
        if row['model'] == 'efp_mlp':
            js = row['js']
            lines.append(f'| {row["fraction"]:g} | {row["seed"]} | {js["observed"]} | {js["null_mean"]} | {js["null_95_interval"]} | {js["p_upper"]} |')
    lines += ['', 'Full per-seed region counts, conditional Wilson intervals, histogram counts,',
              'score spread, checkpoint comparisons and input hashes are in diagnostics.json.',
              'An SR/SB efficiency difference is not a calibrated bump significance.',
              'Undefined values remain unavailable; no result is substituted.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    plot_efficiencies(records, output)
    print(f'Wrote {len(records)} diagnostic records to {output}')


if __name__ == '__main__':
    main()

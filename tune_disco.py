"""Assess prescribed lambda=0.003 using verified study fits, without retraining."""
import argparse
from pathlib import Path
from itertools import product
from artifacts import verify_run
from provenance import atomic_json, read_json, file_hash
from metrics import mean_complete, ratio
from protocol import SIC_TARGET, JS_TARGET

SEEDS = [11, 22, 33]
FRACTIONS = [.005, .01]
STRENGTHS = [.003]


def select_strength(rows, seeds=SEEDS, fractions=FRACTIONS, strengths=STRENGTHS):
    """Exact grid qualification; duplicate/missing keys never qualify."""
    expected = set(product(seeds, fractions, [0.] + list(strengths)))
    indexed = {}
    for row in rows:
        key = (row['seed'], row['fraction'], row['lambda'])
        if key in indexed or key not in expected:
            raise ValueError(f'Duplicate or unexpected comparison key: {key}')
        indexed[key] = row
    full = set(indexed) == expected
    candidates = []
    for strength in strengths:
        item = dict(lambda_value=strength, eligible=full, regimes=[])
        for fraction in fractions:
            groups = [[indexed.get((seed, fraction, lam)) for seed in seeds] for lam in (0., strength)]
            means = [[mean_complete([row[metric] if row else None for row in group])
                      for metric in ('sic', 'js_divergence')] for group in groups]
            sr, js = ratio(means[1][0], means[0][0]), ratio(means[1][1], means[0][1])
            qualifies = bool(full and sr is not None and js is not None and sr >= SIC_TARGET and js <= JS_TARGET)
            item['eligible'] &= qualifies
            # Paired leave-one-seed-out ratios reveal sensitivity to training seeds.
            influence = []
            for omitted in seeds:
                remaining = [s for s in seeds if s != omitted]
                vals = [[mean_complete([indexed.get((s, fraction, lam), {}).get(metric) for s in remaining])
                         for metric in ('sic', 'js_divergence')] for lam in (0., strength)]
                influence.append(dict(omitted_seed=omitted, sic_ratio=ratio(vals[1][0], vals[0][0]), js_ratio=ratio(vals[1][1], vals[0][1])))
            item['regimes'].append(dict(fraction=fraction, expected_seeds=list(seeds), denominator=len(seeds),
                                        sic_retained=sr, js_ratio=js, qualifies=qualifies, leave_one_seed_out=influence))
        candidates.append(item)
    return dict(assessment='prescribed_fixed_lambda' if len(strengths) == 1 else 'predeclared_search',
                prescribed_lambda=strengths[0] if len(strengths) == 1 else None,
                status='complete' if full else 'incomplete', expected_records=len(expected), actual_records=len(indexed),
                missing_keys=[list(k) for k in sorted(expected-set(indexed))], candidates=candidates,
                targets_met=bool(full and all(c['eligible'] for c in candidates)),
                interpretation='Exploratory paired comparison on tuning events; event sample shared across seeds.')


def assess(run):
    run = Path(run)
    protocol = verify_run(run)
    config = protocol['identity']['config']
    rows = read_json(run/'results/metrics.json')
    comparison = [dict(row, **{'lambda': 0. if row['model'] == 'mass_mlp' else .003})
                  for row in rows if row['model'] in ('mass_mlp', 'sideband_0.003') and row['split'] == 'tuning' and row['cut'] == .05]
    result = select_strength(comparison, config['seeds'], config['fractions'])
    result.update(run_id=protocol['run_id'], unique_fits=len(protocol['identity']['expected_fits']), new_fits=0,
                  reuse='Existing verified fits referenced exactly once; not additional independent evidence',
                  input_sha256=file_hash(run/'results/metrics.json'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from report_study import validate_output
    validate_output(args.run, args.output)
    atomic_json(args.output, assess(args.run))
    print(f'Fixed-lambda assessment: {args.output}')


if __name__ == '__main__':
    main()

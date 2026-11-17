"""Audit all inputs before atomically publishing an artifact-derived HTML report."""
import argparse
import html
import itertools
from pathlib import Path
import shutil
import tempfile
import numpy as np
from artifacts import verify_run, verify, complete
from provenance import read_json, atomic_json, identity, file_hash, source_identity
from splits import validate_roles
from metrics import calibrate, metrics_for
from protocol import BINS


def validate_output(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError('Keep derived output separate from run artifacts')


def equal_calculation(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(equal_calculation(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equal_calculation(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        return actual is not None and np.isclose(actual, expected, rtol=1e-10, atol=1e-12)
    return actual == expected


def audit_run(run):
    run = Path(run)
    protocol = verify_run(run)
    frozen = protocol['identity']
    if frozen['source'] != source_identity():
        raise ValueError('Audit calculation source differs from saved run; use the retained source snapshot/environment')
    splits = read_json(run/'splits.json')
    all_records = []
    fingerprints = {'protocol.json': file_hash(run/'protocol.json'), 'splits.json': file_hash(run/'splits.json')}
    for name in frozen['expected_fits']:
        directory = run/'fits'/name
        marker = verify(directory, protocol['run_id'])
        fingerprints[f'fits/{name}/complete.json'] = file_hash(directory/'complete.json')
        meta = marker['metadata']
        config = frozen['config']
        if meta['seed'] not in config['seeds'] or meta['fraction'] not in config['fractions']:
            raise ValueError('Unexpected fit metadata')
        from study import fit_name, specifications, inject
        expected_spec = next((s for s in specifications(config['lambdas']) if s[0] == meta['model']), None)
        if (expected_spec is None or name != fit_name(meta['seed'], meta['fraction'], meta['model'])
                or (meta['include_mass'], meta['mode'], meta['strength']) != expected_spec[1:]):
            raise ValueError('Fit identity/strength mismatch')
        roles = splits['seeds'][str(meta['seed'])]
        validate_roles(roles)
        with np.load(directory/'predictions.npz', allow_pickle=False) as p:
            for key, role in [('train_rows', 'training'), ('checkpoint_rows', 'checkpoint')]:
                rows = p[key]
                if len(np.unique(rows)) != len(rows) or not set(map(int, rows)) <= set(roles[role]):
                    raise ValueError('Fit uses wrong or overlapping role')
            thresholds = read_json(directory/'thresholds.json')
            if set(thresholds) != {str(c) for c in frozen['cuts']}:
                raise ValueError('Calibration cut grid mismatch')
            metrics = read_json(directory/'metrics.json')
            keys = [(r['split'], r['cut']) for r in metrics]
            if len(keys) != len(set(keys)) or set(keys) != set(itertools.product(['calibration', 'tuning'], frozen['cuts'])):
                raise ValueError('Missing or duplicate metric grid')
            for role in ('calibration', 'tuning'):
                if p[f'{role}_row'].tolist() != roles[role]:
                    raise ValueError('Prediction sample membership differs from manifest')
                data = dict(mass=p[f'{role}_mass'], truth=p[f'{role}_truth'])
                score = p[f'{role}_score']
                for cut in frozen['cuts']:
                    threshold = calibrate(p['calibration_mass'], p['calibration_score'], cut)
                    if threshold != thresholds[str(cut)]:
                        raise ValueError('Threshold not derived from calibration-only predictions')
                    row = next(r for r in metrics if r['split'] == role and r['cut'] == cut)
                    expected = dict(seed=meta['seed'], fraction=meta['fraction'], model=meta['model'], strength=meta['strength'],
                                    cut=cut, split=role, threshold=threshold, nominal_calibration_acceptance=cut,
                                    calibration_interpretation='in_sample_diagnostic')
                    expected.update(metrics_for(data, np.arange(len(score)), score, threshold, meta['seed']+300, config['permutations']))
                    if not equal_calculation(row, expected):
                        raise ValueError(f'Saved metric recalculation failed: {name}/{role}/{cut}')
            all_records.extend(metrics)
    if read_json(run/'results/metrics.json') != all_records:
        raise ValueError('Aggregate metrics differ from per-fit records')
    from study import verified_data, reload_scores
    data = verified_data(protocol)
    from splits import indices
    from study import inject
    # Check exact nested injection membership and prediction event contents.
    for name in frozen['expected_fits']:
        directory = run/'fits'/name
        meta = verify(directory, protocol['run_id'])['metadata']
        pools = indices(data, splits, meta['seed'])
        with np.load(directory/'predictions.npz', allow_pickle=False) as p:
            from metrics import region
            for key, role, offset in [('train_rows', 'training', 0), ('checkpoint_rows', 'checkpoint', 1)]:
                expected = data['row'][inject(pools[role], data['truth'], region(data['mass']), meta['fraction'], meta['seed']+offset)]
                if not np.array_equal(expected, p[key]):
                    raise ValueError('Injected membership differs from protocol')
            for role in ('calibration', 'tuning'):
                for field in ('mass', 'truth'):
                    if not np.array_equal(data[field][pools[role]], p[f'{role}_{field}']):
                        raise ValueError('Prediction event contents differ from verified source features')
                chosen = pools[role][:32]
                prediction = reload_scores(directory, meta, {k: v[chosen] for k, v in data.items()})
                if not np.allclose(prediction, p[f'{role}_score'][:32], atol=2e-6, rtol=0):
                    raise ValueError('CPU model reload differs from saved predictions')
    from audit_predictions import audit as independent_audit
    independent = independent_audit(run)
    return dict(independent_calculation=independent, status='passed', run_id=protocol['run_id'], fits=len(frozen['expected_fits']),
                metric_records=len(all_records), input_sha256=fingerprints,
                tolerance=dict(rtol=1e-10, atol=1e-12), limitation='Content and arithmetic audit; not certification of unknown exposure or simulator correctness')


def render(run, output, diagnostic=False):
    run, output = Path(run), Path(output)
    validate_output(run, output)
    audit = audit_run(run)  # Never create scientific output before this succeeds.
    protocol = read_json(run/'protocol.json')
    if protocol['identity']['config']['smoke'] and not diagnostic:
        raise ValueError('Synthetic runs require --diagnostic and cannot produce a scientific report')
    if output.exists():
        raise ValueError('Report output already exists; choose a new output directory')
    rows = read_json(run/'results/metrics.json')
    from tune_disco import assess
    assessment = assess(run)
    label = 'SYNTHETIC SOFTWARE CHECK — NOT PHYSICS RESULTS' if diagnostic else protocol['identity']['interpretation']
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix='.'+output.name))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        images = []
        for fraction, cut, split in itertools.product(protocol['identity']['config']['fractions'], protocol['identity']['cuts'], ['calibration', 'tuning']):
            group = [r for r in rows if (r['fraction'], r['cut'], r['split']) == (fraction, cut, split)]
            fig, ax = plt.subplots(figsize=(9, 5))
            for model in sorted({r['model'] for r in group}):
                subset = [r for r in group if r['model'] == model]
                rates = np.array([[b/a if a else np.nan for a, b in zip(r['local']['bin_total'], r['local']['bin_selected'])] for r in subset])
                valid = np.isfinite(rates).all(axis=0)
                centers = (BINS[1:] + BINS[:-1])/2
                ax.errorbar(centers[valid], rates[:, valid].mean(0), yerr=rates[:, valid].std(0), fmt='.-', label=f'{model} ({len(subset)} seeds)')
            ax.axvspan(3300, 3700, color='gray', alpha=.1)
            ax.axhline(cut, color='gray', linestyle=':')
            ax.set(xlabel='Dijet mass [GeV]', ylabel='Background acceptance', title=f'{label}\nf={fraction:g}; nominal cut={cut:g}; {split}; bars: seed SD')
            ax.legend(fontsize=8)
            fig.tight_layout()
            filename = f'efficiency_f{fraction:g}_cut{cut:g}_{split}.png'
            fig.savefig(temporary/filename, dpi=130)
            plt.close(fig)
            images.append(filename)
        def esc(value):
            return html.escape(str(value))
        def show(value):
            return 'unavailable' if value is None else (f'{value:.6g}' if isinstance(value, float) else esc(value))
        cols = ['seed', 'fraction', 'model', 'cut', 'split', 'threshold', 'sic', 'js_divergence', 'background_efficiency_sb', 'background_efficiency_sr', 'score_std']
        table = '<table><tr>'+''.join(f'<th>{esc(c)}</th>' for c in cols)+'</tr>'
        for row in rows:
            table += '<tr>'+''.join(f'<td>{show(row[c])}</td>' for c in cols)+'</tr>'
        table += '</table>'
        # Failed or undefined comparisons cannot yield a success headline.
        verdict = 'Fixed-lambda point targets met on tuning data.' if assessment['targets_met'] else 'Fixed-lambda point targets not established on tuning data.'
        details = []
        for row in rows:
            details.append(f'<details><summary>{esc(row["model"])} seed {row["seed"]}, f={row["fraction"]}, {row["split"]}, cut={row["cut"]}</summary><pre>{esc(__import__("json").dumps(dict(local=row["local"], closure=row["closure"], undefined_reasons=row["undefined_reasons"]), indent=2, allow_nan=False))}</pre></details>')
        content = f'''<!doctype html><html><head><meta charset="utf-8"><title>MLHEP {esc(label)}</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1300px}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:6px;border:1px solid #bbb}}img{{max-width:100%}}pre{{white-space:pre-wrap}}.notice{{background:#fff0cd;padding:1rem}}</style></head><body>
<h1>{esc(label)}</h1><p>Run ID: <code>{protocol['run_id']}</code></p>
<p class="notice">{esc(verdict)} This is an exploratory comparison. Calibration acceptance is an in-sample diagnostic. The dataset has unknown prior exposures beyond documented uses. Previously evaluated raw rows 200000–299999 are exploratory. No independent confirmation is claimed.</p>
<p>Audit: {audit['fits']} verified unique fits, {audit['metric_records']} metric records. Every configured seed is included. Acceptance uses strict score &gt; cut, linear calibration quantiles and base-2 JS. Undefined values remain unavailable. Bar widths show training-seed SD, not independent-event uncertainty.</p>
<p>SR/SB ratios can reflect a smooth trend; neither those ratios nor a global JS reduction establish a localized excess or removal of mass sculpting. Closure is a descriptive sideband-only linear interpolation conditional on the threshold; calibration uncertainty and a validated background fit remain required for physics claims.</p>
<h2>Every seed and operating point</h2>{table}<h2>Fixed-lambda assessment and seed influence</h2><pre>{esc(__import__('json').dumps(assessment, indent=2, allow_nan=False))}</pre>
<h2>Background efficiency</h2>{''.join(f'<img src="{name}" alt="Background efficiency by mass">' for name in images)}
<h2>Per-seed local counts, intervals and closure</h2>{''.join(details)}
<p>Full numerical values and undefined reasons: metrics.json. Training floor diagnostics and gradient telemetry: each verified fit. This report contains no bump significance or causal explanation of the penalty.</p></body></html>'''
        (temporary/'REPORT.html').write_text(content)
        atomic_json(temporary/'audit.json', audit)
        atomic_json(temporary/'metrics.json', rows)
        atomic_json(temporary/'assessment.json', assessment)
        atomic_json(temporary/'figure_manifest.json', dict(run_id=protocol['run_id'], protocol_sha256=file_hash(run/'protocol.json'), images=images))
        complete(temporary, protocol['run_id'], dict(analysis='diagnostic' if diagnostic else 'exploratory_report'))
        verify(temporary, protocol['run_id'])
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output/'REPORT.html'


def render_evaluation(run, evaluation, output, diagnostic=False):
    run, evaluation, output = map(Path, (run, evaluation, output))
    validate_output(run, output)
    validate_output(evaluation, output)
    audit_run(run)
    protocol = read_json(run/'protocol.json')
    verify(evaluation, protocol['run_id'])
    from audit_predictions import audit_replay
    audit = audit_replay(run, evaluation)
    freeze = read_json(evaluation/'protocol.json')
    if freeze['freeze_id'] != identity(freeze['protocol']) or freeze['protocol']['run_id'] != protocol['run_id']:
        raise ValueError('Evaluation freeze identity mismatch')
    if protocol['identity']['config']['smoke'] and not diagnostic:
        raise ValueError('Synthetic evaluation requires --diagnostic')
    rows = read_json(evaluation/'metrics.json')
    expected = set(itertools.product(freeze['protocol']['seeds'], freeze['protocol']['fractions'], freeze['protocol']['models'], freeze['protocol']['cuts']))
    keys = [(r['seed'],r['fraction'],r['model'],r['cut']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Incomplete evaluation grid')
    splits = read_json(run/'splits.json')
    reserved = set(freeze['protocol']['rows'])
    if reserved != set(splits['seeds'][str(freeze['protocol']['seeds'][0])]['final']):
        raise ValueError('Final reservation mismatch')
    from study import verified_data
    data = verified_data(protocol)
    lookup = {int(row): i for i,row in enumerate(data['row'])}
    selected = np.array([lookup[row] for row in freeze['protocol']['rows']])
    selected = selected[(data['truth'][selected] == 0) | ((data['mass'][selected] >= 3300) & (data['mass'][selected] < 3700))]
    for path in evaluation.glob('*.npz'):
        with np.load(path, allow_pickle=False) as p:
            for field in ('row','mass','truth'):
                if not np.array_equal(p[field],data[field][selected]):
                    raise ValueError('Evaluation differs from exact reserved sample or event contents')
    if output.exists():
        raise ValueError('Output exists; choose a new report directory')
    interpretation = read_json(evaluation/'interpretation.json')
    label = 'SYNTHETIC SOFTWARE CHECK' if diagnostic else interpretation['mode']
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        images = []
        for fraction,cut in itertools.product(freeze['protocol']['fractions'],freeze['protocol']['cuts']):
            fig,ax = plt.subplots(figsize=(9,5))
            for model in freeze['protocol']['models']:
                group = [r for r in rows if (r['fraction'],r['model'],r['cut']) == (fraction,model,cut)]
                rates = np.array([[b/a if a else np.nan for a,b in zip(r['local']['bin_total'],r['local']['bin_selected'])] for r in group])
                good = np.isfinite(rates).all(0)
                centers = np.arange(3125,3900,50)
                ax.errorbar(centers[good],rates[:,good].mean(0),yerr=rates[:,good].std(0),fmt='.-',label=model)
            ax.axvspan(3300,3700,color='gray',alpha=.1)
            ax.axhline(cut,color='gray',linestyle=':')
            ax.set(title=f'{label}; f={fraction:g}; nominal cut={cut:g}; seed SD',xlabel='Dijet mass [GeV]',ylabel='Background acceptance')
            ax.legend()
            fig.tight_layout()
            name=f'evaluation_f{fraction:g}_cut{cut:g}.png'
            fig.savefig(temp/name,dpi=130)
            plt.close(fig)
            images.append(name)
        import json
        esc = html.escape
        comparisons = read_json(evaluation/'comparisons.json')
        content = f'<!doctype html><html><head><meta charset="utf-8"><title>MLHEP {esc(label)}</title></head><body><h1>{esc(label)}</h1><p>Run ID: {protocol["run_id"]}; freeze: {freeze["freeze_id"]}</p>'
        content += '<p>Previously evaluated raw rows 200000–299999 remain historical exploratory data. No row-number rule establishes unused data. Intervals are conditional paired event-bootstrap intervals with fixed models and thresholds; seed variability is not independent-event evidence. Undefined replicates make their interval unavailable. Global JS and SR/SB ratios do not establish absence of a fake resonance.</p>'
        content += '<h2>Every configured fraction</h2><pre>'+esc(json.dumps(comparisons,indent=2,allow_nan=False))+'</pre>'
        content += ''.join(f'<img style="max-width:100%" src="{name}" alt="Evaluation background acceptance">' for name in images)
        content += '<h2>Every seed and cut, including unavailable metrics</h2><pre>'+esc(json.dumps(rows,indent=2,allow_nan=False))+'</pre></body></html>'
        (temp/'REPORT.html').write_text(content)
        atomic_json(temp/'audit.json',audit)
        atomic_json(temp/'figure_manifest.json',dict(run_id=protocol['run_id'],evaluation_complete_sha256=file_hash(evaluation/'complete.json'),images=images))
        complete(temp,protocol['run_id'],dict(analysis=label))
        verify(temp,protocol['run_id'])
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return output/'REPORT.html'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--evaluation', type=Path)
    args = p.parse_args()
    if args.audit_only:
        validate_output(args.run, args.output)
        atomic_json(args.output, audit_run(args.run))
    elif args.evaluation:
        print(render_evaluation(args.run,args.evaluation,args.output,args.diagnostic))
    else:
        print(render(args.run, args.output, args.diagnostic))


if __name__ == '__main__':
    main()

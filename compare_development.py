"""Predeclare and compare train-only transform robustness on shared development rows."""
import argparse
import html
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
from provenance import ROOT,identity,atomic_json,read_json,file_hash,source_identity,environment
from artifacts import verify_run,verify,complete
from preprocessing import VARIANTS,BASELINE
from metrics import mean_complete,ratio


def declare(output,dataset_sha256):
    import torch
    from study import parser,execution_config
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args=parser().parse_args(['--output','unused','--telemetry'])
    config=execution_config(args)
    locations={name:dict(run=str(ROOT/f'results_v21_development_200k_{name}'),
                         report=str(ROOT/f'output/v21_development_200k_{name}_report')) for name in VARIANTS}
    protocol=dict(version=1,dataset_sha256=dataset_sha256,source=source_identity(),environment=environment('cuda'),
                  base_config=config,variants=list(VARIANTS),locations=locations,
                  analysis='Predeclared exploratory robustness comparison; not a search or independent confirmation',
                  rule='Report every seed/fraction/model for each fixed transform; no event deletion or final-score access; do not select a winner from these exploratory ratios',
                  comparisons='Per-seed and ratios of complete seed means for SIC, JS and local closure. f=0 is a control; .005/.01 are the prescribed target regimes.')
    payload=dict(declaration_id=identity(protocol),protocol=protocol)
    output=Path(output)
    if output.exists():
        if read_json(output) != payload:
            raise ValueError('Development declaration changed; preserve the original')
    else:
        atomic_json(output,payload)
    return payload


def compare(declaration_path,output):
    declaration=read_json(declaration_path)
    rule=declaration['protocol']
    if declaration['declaration_id'] != identity(rule) or rule['source'] != source_identity():
        raise ValueError('Development declaration identity/source mismatch')
    runs={}
    manifest={str(declaration_path):file_hash(declaration_path)}
    for variant in rule['variants']:
        location=rule['locations'][variant]
        run,report=Path(location['run']),Path(location['report'])
        protocol=verify_run(run)
        expected_config=dict(rule['base_config'],transform=variant)
        frozen=protocol['identity']
        if (frozen['config'] != expected_config or frozen['dataset_sha256'] != rule['dataset_sha256']
                or frozen['source'] != rule['source'] or frozen['environment'] != rule['environment']
                or frozen['development_declaration_id'] != declaration['declaration_id']):
            raise ValueError('Variant configuration/provenance differs from declaration')
        verify(report,protocol['run_id'])
        audit=read_json(report/'audit.json')
        if audit['status'] != 'passed' or audit['run_id'] != protocol['run_id']:
            raise ValueError('Variant report audit missing')
        for name,digest in audit['input_sha256'].items():
            if file_hash(run/name) != digest:
                raise ValueError('Variant report refers to stale run inputs')
        if runs:
            reference=next(iter(runs.values()))['protocol']['identity']
            if frozen['split_sha256'] != reference['split_sha256'] or frozen['cache_sha256'] != reference['cache_sha256']:
                raise ValueError('Transform variants do not share exact data/splits')
        runs[variant]=dict(protocol=protocol,rows=read_json(run/'results/metrics.json'))
        manifest[str(run/'protocol.json')]=file_hash(run/'protocol.json')
        manifest[str(report/'complete.json')]=file_hash(report/'complete.json')
    records=[]
    config=rule['base_config']
    for variant in rule['variants']:
        for fraction in config['fractions']:
            for model in ['efp_mlp','mass_mlp','mass_gbdt','sideband_0.003','oracle_0.003']:
                sets=[]
                for name in [BASELINE,variant]:
                    rows=[r for r in runs[name]['rows'] if r['fraction'] == fraction and r['model'] == model and r['split'] == 'tuning' and r['cut'] == .05]
                    if len(rows) != len(config['seeds']) or {r['seed'] for r in rows} != set(config['seeds']):
                        raise ValueError('Incomplete or duplicate transform comparison')
                    sets.append(rows)
                means=[[mean_complete([row[key] for row in group]) for key in ['sic','js_divergence']] for group in sets]
                per_seed=[]
                for seed in config['seeds']:
                    base,current=[next(r for r in group if r['seed'] == seed) for group in sets]
                    per_seed.append(dict(seed=seed,baseline_sic=base['sic'],variant_sic=current['sic'],
                        baseline_js=base['js_divergence'],variant_js=current['js_divergence'],
                        baseline_closure=base.get('closure_uncertainty'),variant_closure=current.get('closure_uncertainty')))
                records.append(dict(transform=variant,fraction=fraction,model=model,cut=.05,expected_seeds=config['seeds'],
                    seed_denominator=len(config['seeds']),sic_ratio=ratio(means[1][0],means[0][0]),js_ratio=ratio(means[1][1],means[0][1]),
                    undefined_reason='missing_metric_or_zero_baseline' if ratio(means[1][0],means[0][0]) is None or ratio(means[1][1],means[0][1]) is None else None,
                    per_seed=per_seed))
    from audit_predictions import audit_transform_comparison,audit_assessment
    numerical_audit=audit_transform_comparison(records,runs,config)
    for variant,item in runs.items():
        audit_assessment(read_json(Path(rule["locations"][variant]["report"])/"assessment.json"),item["rows"],config)
    output=Path(output).resolve()
    for location in rule['locations'].values():
        if any(output.is_relative_to(Path(location[k]).resolve()) for k in ['run','report']):
            raise ValueError('Keep comparison outside immutable run/report bundles')
    result=dict(declaration_id=declaration['declaration_id'],analysis=rule['analysis'],
        independent_calculation=numerical_audit,unique_fits=sum(len(run['protocol']['identity']['expected_fits']) for run in runs.values()),
        independent_event_samples=0,source=rule['source'],input_sha256=manifest,records=records,
        limitation='All methods share inspected development events and fixed seeds. Descriptive robustness, not final confirmation, automatic winner selection, or proof of physical-model adequacy.')
    if output.exists():
        raise ValueError('Comparison output exists; preserve it and choose a new path')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name))
    try:
        atomic_json(temporary/'comparison.json',result)
        label='SYNTHETIC SOFTWARE CHECK' if config['smoke'] else 'Exploratory preprocessing robustness'
        columns=['transform','fraction','model','sic_ratio','js_ratio','seed_denominator','undefined_reason']
        table='<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr>'
        for record in records:
            table+='<tr>'+''.join('<td>'+html.escape(str(record[k]) if record[k] is not None else 'unavailable')+'</td>' for k in columns)+'</tr>'
        table+='</table>'
        text='<!doctype html><html><head><meta charset="utf-8"><title>'+label+'</title><style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}pre{white-space:pre-wrap}</style></head><body><h1>'+label+'</h1>'
        text+='<p>Declaration: '+declaration['declaration_id']+'</p><p>'+html.escape(result['limitation'])+'</p>'
        text+='<p>Every configured model, transform and fraction is retained. Ratios divide complete seed means; unavailable values and zero baseline denominators cannot qualify. This comparison selects no winning transform.</p>'+table
        text+='<h2>Every paired seed and local closure interval</h2><pre>'+html.escape(json.dumps(records,indent=2,allow_nan=False))+'</pre></body></html>'
        (temporary/'REPORT.html').write_text(text)
        complete(temporary,declaration['declaration_id'],dict(analysis=label))
        verify(temporary,declaration['declaration_id'])
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('declare')
    d.add_argument('--output',type=Path,required=True)
    d.add_argument('--dataset-sha256',required=True)
    c=sub.add_parser('compare')
    c.add_argument('--declaration',type=Path,required=True)
    c.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command == 'declare':
        declare(args.output,args.dataset_sha256)
    else:
        compare(args.declaration,args.output)


if __name__ == '__main__':
    main()

"""Separate protocol freeze and reserved-event replay/eligible independent evaluation.

The previously evaluated raw rows 200000:300000 are historical exploratory data.
This command never calls them fresh. Unknown exposure cannot qualify as unused.
"""
import argparse
from pathlib import Path
import shutil
import tempfile
import joblib
import numpy as np
import torch
from artifacts import verify_run, verify, complete, writer_lock
from provenance import read_json, atomic_json, identity, file_hash, source_identity
from exposure import load_ledger, eligible, record_access
from study import fit_name, verified_data, reload_scores
from protocol import FINAL_MODELS, CUTS, PRESCRIBED_LAMBDA
from metrics import metrics_for, region
from bootstrap import paired, measures

MODELS = list(FINAL_MODELS)


def freeze(run, destination):
    run = Path(run)
    protocol = verify_run(run)
    from report_study import audit_run, validate_output
    audit_run(run)
    validate_output(run, destination)
    splits = read_json(run/'splits.json')
    config = protocol['identity']['config']
    files = {}
    for seed in config['seeds']:
        for fraction in config['fractions']:
            for model in MODELS:
                name = fit_name(seed, fraction, model)
                path = run/'fits'/name
                marker = verify(path, protocol['run_id'])
                meta = marker['metadata']
                expected_strength = .003 if model == 'sideband_0.003' else 0.
                if meta['strength'] != expected_strength or meta['model'] != model:
                    raise ValueError('Model strength or identity mismatch')
                files[name] = file_hash(path/'complete.json')
    frozen = dict(version=2, run_id=protocol['run_id'], source=source_identity(),
                  dataset_sha256=protocol['identity']['dataset_sha256'], split_sha256=identity(splits),
                  rows=splits['seeds'][str(config['seeds'][0])]['final'], files=files,
                  seeds=config['seeds'], fractions=config['fractions'], models=MODELS,
                  prescribed_lambda=.003, lambda_history=protocol['identity']['lambda_history'],
                  bootstrap=protocol['identity']['bootstrap'], comparison='SIC ratio >=0.90; JS ratio <=0.80',
                  cuts=list(CUTS), mode='synthetic_smoke' if config['smoke'] else 'exploratory_replay',
                  limitation='No unused sample is inferred from row number. Historical rows 200000:300000 were previously evaluated.')
    payload = dict(freeze_id=identity(frozen), protocol=frozen)
    destination = Path(destination)
    if destination.exists():
        if read_json(destination) != payload:
            raise ValueError('Frozen protocol exists with different identity')
    else:
        atomic_json(destination, payload)
    return payload


def load_model(folder, metadata, data):
    return reload_scores(folder, metadata, data)


def evaluate(run, frozen_path, output, independent=False, ledger_path=None):
    run, output = Path(run), Path(output)
    protocol = verify_run(run)
    payload = read_json(frozen_path)
    frozen = payload['protocol']
    if (payload['freeze_id'] != identity(frozen) or frozen['run_id'] != protocol['run_id']
            or frozen['source'] != source_identity() or frozen['split_sha256'] != file_hash(run/'splits.json')):
        raise ValueError('Frozen protocol identity changed')
    from report_study import audit_run, validate_output
    audit_run(run)
    validate_output(run, output)
    config = protocol['identity']['config']
    dataset_hash = frozen['dataset_sha256']
    canonical_ledger = Path(protocol['ledger_path']).resolve()
    if ledger_path is not None and Path(ledger_path).resolve() != canonical_ledger:
        raise ValueError('Every evaluation must use the recorded dataset ledger; alternate ledgers are forbidden')
    ledger_path = canonical_ledger
    if independent:
        if config['smoke'] or ledger_path is None:
            raise ValueError('Independent mode requires real data and a source-audited exposure ledger')
        ledger = load_ledger(ledger_path, dataset_hash)
        eligible(ledger, frozen['rows'])
    # Every model/threshold verified before opening candidate final features.
    for name, digest in frozen['files'].items():
        if file_hash(run/'fits'/name/'complete.json') != digest:
            raise ValueError('Frozen model artifacts changed')
    with writer_lock(output.parent/(output.name+'.lock')):
        if output.exists():
            raise ValueError('Evaluation output exists; preserve it and choose a new output')
        record_access(ledger_path, dataset_hash, frozen['rows'],
                      'independent_evaluation_attempt' if independent else 'exploratory_replay_attempt', payload['freeze_id'],
                      require_unused=independent)
        data = verified_data(protocol)
        lookup = {int(row): i for i, row in enumerate(data['row'])}
        try:
            ids = np.array([lookup[row] for row in frozen['rows']])
        except KeyError as error:
            raise ValueError('Reserved rows missing from verified cache') from error
        data = {k: v[ids] for k, v in data.items()}
        # Signal outside SR never enters sensitivity/evaluation sample.
        keep = (data['truth'] == 0) | (region(data['mass']) == 1)
        data = {k: v[keep] for k, v in data.items()}
        temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix='.'+output.name))
        try:
            records, comparisons = [], []
            for fraction in frozen['fractions']:
                decisions = {}
                for seed in frozen['seeds']:
                    for name in MODELS:
                        folder = run/'fits'/fit_name(seed, fraction, name)
                        metadata = verify(folder, protocol['run_id'])['metadata']
                        score = load_model(folder, metadata, data)
                        cuts = read_json(folder/'thresholds.json')
                        with np.load(folder/'predictions.npz',allow_pickle=False) as saved:
                            calibration={k:saved['calibration_'+k].copy() for k in ('mass','truth','score')}
                        for cut in frozen['cuts']:
                            threshold = cuts[str(cut)]
                            row = dict(seed=seed, fraction=fraction, model=name, cut=cut, split='final_replay' if not independent else 'final', threshold=threshold)
                            row.update(metrics_for(data,np.arange(len(score)),score,threshold,seed+300,config['permutations'],
                                calibration=calibration,cut=cut,closure_replicates=config['closure_replicates']))
                            records.append(row)
                        decisions[(seed, name)] = score > cuts['0.05']
                        np.savez_compressed(temporary/f'{fit_name(seed, fraction, name)}.npz', **data, score=score, threshold=cuts['0.05'])
                rule = frozen['bootstrap']
                comparisons.append(dict(fraction=fraction, **paired(data['mass'], data['truth'], decisions,
                                          frozen['seeds'], rule['replicates'], rule['seed'])))
            atomic_json(temporary/'metrics.json', records)
            atomic_json(temporary/'comparisons.json', comparisons)
            atomic_json(temporary/'protocol.json', payload)
            atomic_json(temporary/'interpretation.json', dict(mode='independent_event_check' if independent else frozen['mode'],
                       limitation='Event-disjoint simulation check cannot establish independent generator/systematic behavior'))
            complete(temporary, protocol['run_id'], dict(freeze_id=payload['freeze_id']))
            verify(temporary, protocol['run_id'])
            temporary.rename(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--run', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    e = sub.add_parser('evaluate')
    e.add_argument('--run', type=Path, required=True)
    e.add_argument('--freeze', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--independent', action='store_true')
    e.add_argument('--ledger', type=Path)
    args = p.parse_args()
    if args.command == 'freeze':
        freeze(args.run, args.output)
    else:
        evaluate(args.run, args.freeze, args.output, args.independent, args.ledger)


if __name__ == '__main__':
    main()

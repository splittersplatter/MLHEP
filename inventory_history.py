"""Inventory historical source/artifact identities and conservatively retain exposures.

Legacy bundles lack full raw-content provenance. Their reported row use is
recorded conservatively against the current dataset, never certified as unused.
No legacy prediction scores are loaded or models evaluated.
"""
import argparse
from pathlib import Path
import numpy as np
from provenance import ROOT, file_hash, atomic_json, read_json, identity
from exposure import load_ledger, record_access

HISTORICAL = ('results_linux','results_linux_new','reproducible_demo')
ROW_KEYS = dict(train_rows='training',validation_rows='checkpoint_and_exploratory_evaluation',
                checkpoint_validation_rows='checkpoint',checkpoint_rows='checkpoint',test_rows='exploratory_evaluation',
                rows='exploratory_evaluation')


def inventory(dataset, destination, roots=None):
    destination=Path(destination).resolve()
    protected=[Path(dataset).resolve()]+[(ROOT/name).resolve() for name in (roots or HISTORICAL)]
    if destination.exists() or any(destination == p or destination.is_relative_to(p) for p in protected):
        raise ValueError("Inventory destination must be new and outside source/history paths")
    digest = file_hash(dataset)
    bundles=[]
    for name in roots or HISTORICAL:
        root=ROOT/name
        if not root.exists():
            bundles.append(dict(bundle=name,status='unavailable',files={},exposures={}))
            continue
        files={}
        roles={}
        errors=[]
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix not in ('.py','.sh','.json','.npz','.pt','.joblib'):
                continue
            if 'feature_cache' in path.parts or '__pycache__' in path.parts:
                continue
            relative=str(path.relative_to(root))
            files[relative]=file_hash(path)
            if path.suffix == '.npz':
                try:
                    with np.load(path,allow_pickle=False) as saved:
                        for key,purpose in ROW_KEYS.items():
                            if key in saved:
                                rows=saved[key]
                                if rows.ndim != 1 or not np.issubdtype(rows.dtype,np.integer) or (rows < 0).any():
                                    raise ValueError(f'Invalid {key}')
                                roles.setdefault(purpose,set()).update(map(int,rows))
                except (ValueError,OSError,KeyError) as error:
                    errors.append(dict(file=relative,error=str(error)))
        bundles.append(dict(bundle=name,status='inventoried_with_gaps' if errors else 'inventoried',files=files,
                            exposures={k:sorted(v) for k,v in roles.items()},errors=errors,
                            raw_content_identity='unavailable_in_legacy_manifests',
                            interpretation='Reported prior use conservatively retained; does not certify original dataset provenance'))
    report=dict(version=1,current_dataset_sha256=digest,bundles=bundles,
                history='unknown',known_confirmation_raw_range=[200000,300000],
                conservative_manual_derived_inspection_range=[0,300000],
                limitation='Legacy selected row IDs establish reported use; unseen ranges and unrecorded manual accesses remain unknown. Current data hash does not retroactively certify legacy source identity.')
    atomic_json(destination,report)
    ledger_path=ROOT/'exposure_ledger'/f'{digest}.json'
    for bundle in bundles:
        for purpose,rows in bundle['exposures'].items():
            key='legacy:'+bundle['bundle']+':'+purpose+':'+identity(rows)
            existing=load_ledger(ledger_path,digest)
            if not any(record['run_id'] == key for record in existing['records']):
                record_access(ledger_path,digest,rows,'reported_legacy_'+purpose,key)
    # The full previously evaluated range is conservative where individual rows
    # are unavailable; do not reinterpret the remaining raw range as unused.
    key='legacy:known_confirmation_raw_200000_300000'
    if not any(r['run_id'] == key for r in load_ledger(ledger_path,digest)['records']):
        record_access(ledger_path,digest,range(200000,300000),'reported_legacy_confirmation_raw_access',key)
    key="legacy:conservative_derived_artifact_inspection_0_300000"
    if not any(r["run_id"] == key for r in load_ledger(ledger_path,digest)["records"]):
        record_access(ledger_path,digest,range(0,300000),"conservative_reported_legacy_derived_artifact_inspection",key)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=inventory(args.data,args.output)
    print('Inventoried',len(report['bundles']),'historical bundles; unknown exposure remains unknown')


if __name__ == '__main__':
    main()

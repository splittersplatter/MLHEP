"""Inspect raw floor events from injected training samples only."""
import argparse
from pathlib import Path
import numpy as np
import h5py
import hdf5plugin
from provenance import read_json,file_hash,atomic_json
from artifacts import verify_run
from study import verified_data
from preprocessing import specification


def audit(run,dataset,output):
    run,dataset,output=map(Path,(run,dataset,output))
    if output.exists() or output.resolve() == dataset.resolve() or output.resolve().is_relative_to(run.resolve()):
        raise ValueError('Keep floor audit separate from experiment artifacts')
    protocol=verify_run(run)
    if protocol['identity']['config']['smoke']:
        raise ValueError('Raw-physics floor audit requires a real source-backed run')
    if file_hash(dataset) != protocol['identity']['dataset_sha256']:
        raise ValueError('Raw floor source content identity mismatch')
    cache_path=Path(protocol['cache']['path'])
    if output.resolve() == cache_path.resolve():
        raise ValueError('Floor audit cannot replace feature cache')
    recipe=specification(protocol['identity']['config']['transform'])
    floor=recipe['floor']
    data=verified_data(protocol)
    lookup={int(row):i for i,row in enumerate(data['row'])}
    train_rows=set()
    records=[]
    for name in protocol['identity']['expected_fits']:
        with np.load(run/'fits'/name/'predictions.npz',allow_pickle=False) as p:
            train_rows.update(map(int,p['train_rows']))
    floor_rows=[row for row in sorted(train_rows) if (data['x'][lookup[row]] <= floor).any()]
    width=data['x'].shape[1]//2
    with h5py.File(dataset,'r') as handle:
        values=handle['df/block0_values']
        for row in floor_rows:
            raw=values[row]
            particles=raw[:2100].reshape(700,3)
            if not np.isfinite(raw).all() or (particles[:,0] < 0).any() or raw[-1] != data['truth'][lookup[row]]:
                raise ValueError('Invalid raw floor event or truth mismatch')
            i=lookup[row]
            counts=data['constituents'][i]
            raw_count=int((particles[:,0] > 0).sum())
            if counts.sum() > raw_count:
                raise ValueError('Jet constituent accounting exceeds raw particles')
            jets=[]
            for jet in range(2):
                efp=data['x'][i,jet*width:(jet+1)*width]
                n=int(counts[jet])
                jets.append(dict(multiplicity=n,efp_exact_zero=int((efp == 0).sum()),efp_at_floor=int((efp <= floor).sum()),
                    jet_pt=float(data['jet_pt'][i,jet]),jet_eta=float(data['jet_eta'][i,jet]),
                    jet_rapidity=float(data['jet_rapidity'][i,jet]),
                    observation='Single-constituent jet has zero angular distances; consistent with vanishing nonconstant angular EFPs' if n == 1 else 'Nonzero multiplicity; graph-specific index assignments, angular geometry and numerical scale need interpretation; degree is not a distinct-prong veto'))
            records.append(dict(row=row,truth=int(data['truth'][i]),mass=float(data['mass'][i]),raw_nonpadding_particles=raw_count,
                                raw_min_positive_pt=float(particles[particles[:,0] > 0,0].min()),jets=jets))
    result=dict(run_id=protocol['run_id'],dataset_sha256=protocol['identity']['dataset_sha256'],
                protocol_sha256=file_hash(run/'protocol.json'),transform=recipe,unique_injected_training_rows=len(train_rows),
                training_floor_event_count=len(records),records=records,
                boundary='Only union of injected training rows across declared fits. Calibration, tuning and reserved-final-only rows are excluded.',
                limitation='Raw validity and multiplicity diagnostics, not a demonstrated causal effect on classifier performance; no row removed and no four-constituent veto.')
    atomic_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=audit(args.run,args.data,args.output)
    print('Inspected',result['training_floor_event_count'],'source-backed training floor events')


if __name__ == '__main__':
    main()

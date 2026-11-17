"""Phase 1: background-only CWoLa closure test with per-jet EFPs.
Run python phase1.py; see README_phase1.md for interpretation and limitations.
"""
import json
from pathlib import Path
import warnings
import h5py
import hdf5plugin  # Registers the dataset's Blosc filter.
import numpy as np
import torch
from torch import nn
import fastjet as fj
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message=".*'fork' is not available.*")
    import energyflow as ef

ROOT = Path(__file__).resolve().parent


def extract(args):
    """Bounded raw-data memory; cache only compact observables and source row IDs."""
    from provenance import file_hash, identity, atomic_json, read_json, environment
    from artifacts import writer_lock
    if args.events <= 0 or args.chunk_size <= 0 or args.degree <= 0:
        raise ValueError('Extraction counts and degree must be positive')
    before = args.data.stat()
    digest = file_hash(args.data)
    start_row = getattr(args, 'start_row', 0)
    with h5py.File(args.data, 'r') as handle:
        values = handle['df/block0_values']
        schema = dict(shape=list(values.shape), dtype=str(values.dtype), columns=2101)
    stop_row = min(start_row + args.events, schema['shape'][0])
    if start_row < 0 or stop_row <= start_row:
        raise ValueError('Invalid raw row range')
    calculator = ef.EFPSet(f'd<={args.degree}', measure='hadr', beta=1., normed=True)
    metadata = dict(feature_width=int(2*calculator.count()), version=3, dataset_sha256=digest, requested_range=[start_row, start_row+args.events],
                    actual_range=[start_row, stop_row], schema=schema, degree=args.degree,
                    feature_definition='ungroomed_antikt_R1_pt1200_mass_order_hadr_beta1_normed',
                    extraction_sha256=file_hash(Path(__file__)),
                    dependencies=environment()['packages'])
    key = identity(metadata)
    cache = Path(getattr(args, 'cache_dir', ROOT / 'feature_cache')) / f'jets_v3_{key}.npz'
    receipt = cache.with_suffix('.json')
    if cache.exists():
        record = read_json(receipt)
        if record['metadata'] != metadata or record['sha256'] != file_hash(cache):
            raise ValueError('Cache provenance or content mismatch')
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(str(saved['metadata'])) != metadata:
                raise ValueError('Embedded cache metadata mismatch')
            result = {k: saved[k] for k in FEATURE_KEYS}
        validate_features(result, metadata)
        if args.data.stat() != before:
            raise ValueError('Dataset changed during cache validation')
        args.cache_receipt = dict(metadata=metadata, sha256=record['sha256'], path=str(cache.resolve()))
        return result
    definition = fj.JetDefinition(fj.antikt_algorithm, 1.0)
    result = {k: [] for k in FEATURE_KEYS}
    with h5py.File(args.data, 'r') as handle:
        values = handle['df/block0_values']
        if values.shape[1] != 2101 or not np.array_equal(handle['df/block0_items'][:], np.arange(2101)):
            raise ValueError('Unexpected HDF5 layout: expected 700 (pt,eta,phi) triplets plus label.')
        count = min(args.events, len(values) - start_row)
        if count <= 0 or start_row < 0:
            raise ValueError("Invalid source row range")
        for start in range(0, count, args.chunk_size):
            block = values[start_row+start:start_row+min(start + args.chunk_size, count)]
            for offset, row in enumerate(block):
                if row[-1] not in (0, 1):
                    raise ValueError('Non-binary truth label')
                p = row[:2100].reshape(700, 3)
                if not np.isfinite(p).all() or (p[:, 0] < 0).any():
                    raise ValueError(f'Invalid particles at row {start + offset}')
                p = p[p[:, 0] > 0]
                pt, eta, phi = p.T
                particles = [fj.PseudoJet(float(px), float(py), float(pz), float(e))
                             for px, py, pz, e in zip(pt*np.cos(phi), pt*np.sin(phi),
                                                      pt*np.sinh(eta), pt*np.cosh(eta))]
                sequence = fj.ClusterSequence(particles, definition)
                jets = fj.sorted_by_pt(sequence.inclusive_jets())
                if len(jets) < 2 or jets[0].pt() <= 1200:
                    continue
                mass = (jets[0] + jets[1]).m()
                if not 3100 <= mass < 3900:
                    continue
                jets = sorted(jets[:2], key=lambda j: j.m())
                features = []
                for jet in jets:
                    constituents = np.array([[c.pt(), c.rap(), c.phi()] for c in jet.constituents()])
                    features.extend(calculator.compute(constituents))
                result['constituents'].append([len(j.constituents()) for j in jets])
                result['jet_pt'].append([j.pt() for j in jets])
                result['jet_eta'].append([j.eta() for j in jets])
                result['jet_rapidity'].append([j.rap() for j in jets])
                result['delta_eta'].append(abs(jets[0].eta()-jets[1].eta()))
                result['delta_rapidity'].append(abs(jets[0].rap()-jets[1].rap()))
                result['jet_mass'].append([jets[0].m(), jets[1].m()-jets[0].m()])
                result['x'].append(features)
                result['mass'].append(mass)
                result['truth'].append(row[-1])
                result['row'].append(start_row + start + offset)
            if start // args.chunk_size % 40 == 0 or start + len(block) == count:
                print(f'Processed {start + len(block)}/{count}; retained {len(result["mass"])}', flush=True)
    result = {k: np.asarray(v) for k, v in result.items()}
    if not len(result['mass']) or not np.isfinite(result['x']).all():
        raise ValueError('No selected events or invalid features')
    result['row'] = result['row'].astype(np.int64)
    result['truth'] = result['truth'].astype(np.int8)
    validate_features(result, metadata)
    after = args.data.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
        raise ValueError('Dataset changed during extraction; use a stable snapshot')
    cache.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(cache.parent / key):
        temporary = cache.with_suffix('.pending.npz')
        np.savez_compressed(temporary, **result, metadata=json.dumps(metadata, allow_nan=False))
        temporary.replace(cache)
        atomic_json(receipt, dict(metadata=metadata, sha256=file_hash(cache)))
    args.cache_receipt = dict(metadata=metadata, sha256=file_hash(cache), path=str(cache.resolve()))
    return result


FEATURE_KEYS = ('x', 'mass', 'truth', 'row', 'jet_mass', 'constituents', 'jet_pt',
                'jet_eta', 'jet_rapidity', 'delta_eta', 'delta_rapidity')


def validate_features(data, metadata):
    n = len(data['mass'])
    if not n or any(len(data[k]) != n or not np.isfinite(data[k]).all() for k in FEATURE_KEYS):
        raise ValueError('Empty, unaligned or nonfinite features')
    if (data['x'].ndim != 2 or data['x'].shape[1] != metadata['feature_width'] or (data['x'] < 0).any()
            or data['jet_mass'].shape != (n, 2) or (data['jet_mass'] < 0).any()
            or data['mass'].shape != (n,) or data['truth'].shape != (n,)
            or not np.isin(data['truth'], [0, 1]).all()
            or ((data['mass'] < 3100) | (data['mass'] >= 3900)).any()):
        raise ValueError('Invalid feature shape, truth, or mass selection')
    rows = data['row']
    if (rows.shape != (n,) or not np.issubdtype(rows.dtype, np.integer)
            or len(np.unique(rows)) != n or (np.diff(rows) <= 0).any()
            or (rows < metadata['actual_range'][0]).any() or (rows >= metadata['actual_range'][1]).any()):
        raise ValueError('Invalid raw row IDs')
    for name in ('constituents', 'jet_pt', 'jet_eta', 'jet_rapidity'):
        if data[name].shape != (n, 2):
            raise ValueError(f'Invalid {name} shape')
    if (data['constituents'] < 1).any() or not np.issubdtype(data['constituents'].dtype, np.integer):
        raise ValueError('Invalid constituent multiplicities')
    if (data['jet_pt'] <= 0).any() or data['delta_eta'].shape != (n,) or data['delta_rapidity'].shape != (n,):
        raise ValueError('Invalid jet kinematics')


class BaselineEFPNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32),
                                 nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)  # Logits: numerically stable BCEWithLogitsLoss.


from metrics import region


def main():
    # The maintained closure control uses the same five-role/provenance pipeline.
    import sys
    from study import main as study_main
    study_main(sys.argv[1:] + ['--fractions', '0'])


if __name__ == '__main__':
    main()

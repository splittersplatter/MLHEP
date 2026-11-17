"""Protocol v2: maintained CWoLa/DisCo development pipeline (final evaluation separate)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from phase1 import ROOT, extract, BaselineEFPNetwork
from metrics import disco_squared, region, metrics_for, calibrate, js_stat
from provenance import atomic_json, read_json, identity, source_identity, environment, git_state, file_hash
from artifacts import complete, verify, writer_lock
from splits import allocate, indices, validate_roles
from exposure import record_access
from protocol import BINS, CUTS


def partitions(data, seed):
    """Compatibility helper: training/checkpoint/tuning only; use persisted manifest for runs."""
    manifest = allocate(data, [seed], 'in_memory')
    roles = indices(data, manifest, seed)
    return roles['training'], roles['checkpoint'], roles['tuning']


def inject(ids, truth, sr, fraction, seed):
    if not 0 <= fraction < 1:
        raise ValueError('Invalid injection fraction')
    background = ids[truth[ids] == 0]
    signal = np.random.default_rng(seed).permutation(ids[(truth[ids] == 1) & (sr[ids] == 1)])
    count = int(round(fraction/(1-fraction)*np.sum(sr[background])))
    if count > len(signal):
        raise ValueError(f'Infeasible injection: need {count} SR signal; available {len(signal)}; SR background {int(sr[background].sum())}')
    return np.concatenate((background, signal[:count]))


def raw_features(data, include_mass):
    raw = np.log(np.maximum(data['x'], 1e-12))
    return np.column_stack((raw, np.log1p(np.maximum(data['jet_mass'], 0)))) if include_mass else raw


def prepare(data, train, include_mass):
    raw = raw_features(data, include_mass)
    mean, scale = raw[train].mean(0), raw[train].std(0)
    active = scale > 1e-8
    if not active.any() or not np.isfinite(raw).all():
        raise ValueError('No active features or invalid inputs')
    standardized = ((raw[:, active]-mean[active])/scale[active]).astype('float32')
    floor = data['x'][train] <= 1e-12
    clipped = standardized[train, :][:, :int(active[:data['x'].shape[1]].sum())][floor[:, active[:data['x'].shape[1]]]]
    diagnostics = dict(training_events=len(train), efp_floor_counts=floor.sum(0).tolist(),
                       efp_floor_rates=floor.mean(0).tolist(), constant_columns=np.flatnonzero(~active).tolist(),
                       active_feature_min=standardized[train].min(0).tolist(),
                       active_feature_max=standardized[train].max(0).tolist(),
                       clipped_standardized_range=[float(clipped.min()), float(clipped.max())] if clipped.size else None,
                       transform='log(max(EFP,1e-12)); log1p(mass); training-only standardization')
    return standardized, dict(mean=mean, scale=scale, active=active), diagnostics


def apply_preprocessing(data, preprocessing, include_mass):
    raw = raw_features(data, include_mass)
    active, mean, scale = (preprocessing[k] for k in ('active', 'mean', 'scale'))
    if raw.shape[1] != len(active) or mean.shape != scale.shape or mean.shape != active.shape or (scale[active] <= 0).any():
        raise ValueError('Incompatible preprocessing')
    x = ((raw[:, active]-mean[active])/scale[active]).astype('float32')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transformed features')
    return x


def gradient_diagnostic(model, tx, ty, tm, weights, ids, allowed, strength):
    """autograd.grad does not mutate gradients, optimizer, RNG or batch order."""
    logits = model(tx[ids])
    bce = (nn.functional.binary_cross_entropy_with_logits(logits, ty[ids], reduction='none')*weights[ty[ids].long()]).mean()
    local = np.flatnonzero(allowed[ids])[:128]
    penalty = strength*disco_squared(torch.sigmoid(logits[local]), tm[ids[local]])
    parameters = tuple(model.parameters())
    first = torch.cat([g.flatten() for g in torch.autograd.grad(bce, parameters, retain_graph=True)])
    second = torch.cat([g.flatten() for g in torch.autograd.grad(penalty, parameters)])
    a, b = first.norm().item(), second.norm().item()
    cosine = float(torch.dot(first, second).item()/(a*b)) if a*b > 0 else None
    scores = torch.sigmoid(logits).detach().cpu().numpy()
    result = dict(bce_gradient_norm=a, weighted_penalty_gradient_norm=b, cosine=cosine,
                  cosine_reason=None if cosine is not None else 'zero_gradient',
                  score_std=float(scores.std()), score_quantiles=np.quantile(scores, [0, .25, .5, .75, 1]).tolist(),
                  region_counts=np.bincount(ty[ids].cpu().numpy().astype(int), minlength=2).tolist())
    if not np.isfinite([a, b]).all() or (cosine is not None and not np.isfinite(cosine)):
        raise FloatingPointError('Nonfinite gradient telemetry')
    return result


def train_mlp(x, data, train, valid, seed, mode, strength, args):
    torch.manual_seed(seed)
    device = torch.device(getattr(args, 'device', 'cpu'))
    model = BaselineEFPNetwork(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    target = region(data['mass'])
    counts = np.bincount(target[train], minlength=2)
    if not len(valid) or (counts == 0).any():
        raise ValueError('Missing training class or checkpoint sample')
    weights = torch.tensor(len(train)/(2*counts), dtype=torch.float32, device=device)
    tx = torch.from_numpy(x).to(device)
    ty = torch.tensor(target, dtype=torch.float32, device=device)
    tm = torch.tensor(data['mass']/1000, dtype=torch.float32, device=device)
    allowed = data['truth'] == 0 if mode == 'oracle' else target == 0
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    generator = torch.Generator().manual_seed(seed+100)
    best, state, stale, history, telemetry = float('inf'), None, 0, [], []
    rng = np.random.default_rng(seed+200)
    penalty_valid = valid[allowed[valid]]
    penalty_valid = rng.choice(penalty_valid, min(512, len(penalty_valid)), replace=False)
    if strength and len(penalty_valid) < 4:
        raise ValueError('Insufficient checkpoint penalty support')
    def predict(ids):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(tx[part]) for part in np.array_split(ids, max(1, (len(ids)+1023)//1024))])
    for epoch in range(args.epochs):
        model.train()
        total_bce, total_penalty, batches = 0., 0., 0
        order = train[torch.randperm(len(train), generator=generator).numpy()]
        for begin in range(0, len(order), 512):
            ids = order[begin:begin+512]
            logits = model(tx[ids])
            bce = (criterion(logits, ty[ids])*weights[ty[ids].long()]).mean()
            penalty = logits.sum()*0
            if strength:
                local = np.flatnonzero(allowed[ids])[:128]
                penalty = disco_squared(torch.sigmoid(logits[local]), tm[ids[local]])
            loss = bce+strength*penalty
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_bce += bce.item()
            total_penalty += penalty.item()
            batches += 1
        logits = predict(valid)
        val_bce = (criterion(logits, ty[valid])*weights[ty[valid].long()]).mean().item()
        with torch.no_grad():
            val_penalty = disco_squared(torch.sigmoid(predict(penalty_valid)), tm[penalty_valid]).item() if strength else 0.
        objective = val_bce+strength*val_penalty
        if not np.isfinite(objective):
            raise FloatingPointError('Nonfinite checkpoint objective')
        improved = objective < best-1e-6
        history.append([epoch+1, total_bce/batches, total_penalty/batches, val_bce, val_penalty, objective])
        if getattr(args, 'telemetry', False):
            detail = gradient_diagnostic(model, tx, ty, tm, weights, train[:512], allowed, strength)
            telemetry.append(dict(epoch=epoch+1, checkpoint_reason='objective_improved' if improved else 'no_improvement', **detail))
        if improved:
            best, state, stale = objective, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(state)
    return model, history, telemetry


def verified_data(protocol):
    if protocol['identity']['config']['smoke']:
        return synthetic_data()
    cache = protocol['cache']
    if file_hash(cache['path']) != protocol['identity']['cache_sha256']:
        raise ValueError('Run cache checksum changed')
    from phase1 import FEATURE_KEYS, validate_features
    import json
    with np.load(cache['path'], allow_pickle=False) as p:
        if json.loads(str(p['metadata'])) != protocol['identity']['cache_metadata']:
            raise ValueError('Run cache metadata mismatch')
        data = {k: p[k] for k in FEATURE_KEYS}
    validate_features(data, protocol['identity']['cache_metadata'])
    return data


def reload_scores(folder, metadata, data):
    with np.load(folder/'preprocessing.npz', allow_pickle=False) as p:
        x = apply_preprocessing(data, p, metadata['include_mass'])
    if metadata['model'] == 'mass_gbdt':
        model = joblib.load(folder/'model.joblib')
    else:
        model = BaselineEFPNetwork(x.shape[1])
        model.load_state_dict(torch.load(folder/'model.pt', map_location='cpu', weights_only=True))
    return score_model(model, x)


def score_model(model, x):
    if isinstance(model, HistGradientBoostingClassifier):
        return model.predict_proba(x)[:, 1]
    if not len(x):
        return np.empty(0)
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(part).to(next(model.parameters()).device))).cpu().numpy()
                               for part in np.array_split(x, max(1, (len(x)+1023)//1024))])


def specifications(lambdas):
    return [('efp_mlp', False, 'none', 0.), ('mass_mlp', True, 'none', 0.),
            ('mass_gbdt', True, 'none', 0.)] + [(f'{mode}_{lam:g}', True, mode, lam) for mode in ('sideband', 'oracle') for lam in lambdas]


def fit_name(seed, fraction, name):
    return f'seed{seed}_f{fraction:g}_{name}'


def synthetic_data(events=2400):
    # Explicit labeled fixture; never substitutes for missing physics data.
    rng = np.random.default_rng(771)
    mass = np.tile([3200., 3400., 3600., 3800.], events//4)
    truth = np.tile([0, 0, 1, 0, 1, 1, 0, 1], len(mass)//8)
    x = rng.lognormal(-2, 1, (len(mass), 8))
    x[:2, 0] = 0
    x[:, -1] = 1
    return dict(x=x, mass=mass, truth=truth, row=np.arange(len(mass)),
                jet_mass=rng.uniform(10, 300, (len(mass), 2)))


def run_study(args):
    execution_source = source_identity()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    output = args.output.resolve()
    if args.smoke:
        data = synthetic_data()
        receipt = dict(sha256=identity({k: v.tolist() for k, v in data.items()}),
                       metadata=dict(dataset_sha256='synthetic_fixture_v1', actual_range=[0, len(data['row'])]))
    else:
        data = extract(args)
        receipt = args.cache_receipt
    if source_identity() != execution_source:
        raise ValueError('Source changed during extraction')
    dataset_hash = receipt['metadata']['dataset_sha256']
    ledger_path = output/'exposure.json' if args.smoke else ROOT/'exposure_ledger'/f'{dataset_hash}.json'
    manifest = allocate(data, args.seeds, dataset_hash, args.smoke)
    specs = specifications(args.lambdas)
    expected = [fit_name(s, f, spec[0]) for s in args.seeds for f in args.fractions for spec in specs]
    # Preflight every injection before training any model.
    for seed in args.seeds:
        roles = indices(data, manifest, seed)
        for fraction in args.fractions:
            for name, offset in [('training', 0), ('checkpoint', 1)]:
                inject(roles[name], data['truth'], region(data['mass']), fraction, seed+offset)
    config = {k: v for k, v in vars(args).items() if k not in ('output', 'cache_receipt', 'data', 'cache_dir')}
    frozen = dict(version=2, config=config, dataset_sha256=dataset_hash, cache_sha256=receipt['sha256'],
                  cache_metadata=receipt['metadata'], split_sha256=identity(manifest),
                  source=execution_source, environment=environment(args.device), expected_fits=expected,
                  metric_version='v2-strict-cut-base2-js', bins=BINS.tolist(),
                  thresholds='strict score > linear calibration quantile', cuts=list(CUTS),
                  minimum_support=dict(correlation=4, closure_bin=5, closure_sr=20),
                  bootstrap=dict(replicates=300, seed=773, invalid_rule='any invalid replicate makes interval unavailable'),
                  interpretation='SYNTHETIC SMOKE' if args.smoke else 'Exploratory development; prior dataset exposure is not fully known',
                  lambda_history='0.003 informed by earlier inspected results; no retrospective preselection claim')
    run_id = identity(frozen)
    with writer_lock(output):
        if (output/'protocol.json').exists():
            previous = read_json(output/'protocol.json')
            if previous['identity'] != frozen or previous['run_id'] != run_id:
                raise ValueError('Run identity changed; choose a new output directory')
            from provenance import verify_sources
            verify_sources(output/'source', frozen['source'])
            if read_json(output/'splits.json') != manifest:
                raise ValueError('Persisted splits changed')
        else:
            if any(p.name != '.writer.lock' for p in output.iterdir()):
                raise ValueError('Refuse nonempty output without v2 protocol')
            (output/'source').mkdir()
            for name in frozen['source']:
                shutil.copyfile(ROOT/name, output/'source'/name)
            atomic_json(output/'splits.json', manifest)
            atomic_json(output/'protocol.json', dict(run_id=run_id, identity=frozen, git=git_state(), input_path=str(args.data), cache=receipt, ledger_path=str(ledger_path.resolve())))
        (output/'fits').mkdir(exist_ok=True)
        records = []
        for seed in args.seeds:
            roles = indices(data, manifest, seed)
            validate_roles({k: data['row'][v] for k, v in roles.items()})
            # Reserve each role before any fit accesses it; never erase interrupted attempts.
            for role in ('training', 'checkpoint', 'calibration', 'tuning'):
                record_access(ledger_path, dataset_hash, data['row'][roles[role]], role, run_id)
            for fraction in args.fractions:
                train = inject(roles['training'], data['truth'], region(data['mass']), fraction, seed)
                valid = inject(roles['checkpoint'], data['truth'], region(data['mass']), fraction, seed+1)
                for name, include_mass, mode, strength in specs:
                    key = fit_name(seed, fraction, name)
                    directory = output/'fits'/key
                    if (directory/'complete.json').exists():
                        marker = verify(directory, run_id)
                        if marker['metadata'] != dict(seed=seed, fraction=fraction, model=name, mode=mode, strength=strength, include_mass=include_mass):
                            raise ValueError('Fit metadata mismatch')
                        records.extend(read_json(directory/'metrics.json'))
                        continue
                    print(f'Train {key} on {"cpu (scikit-learn)" if name == "mass_gbdt" else args.device}', flush=True)
                    temp = Path(tempfile.mkdtemp(dir=output/'fits', prefix='.'+key))
                    try:
                        x, preprocessing, tails = prepare(data, train, include_mass)
                        if name == 'mass_gbdt':
                            target = region(data['mass'])[train]
                            counts = np.bincount(target, minlength=2)
                            model = HistGradientBoostingClassifier(max_iter=5 if args.smoke else 150, max_leaf_nodes=15,
                                l2_regularization=1., early_stopping=False, random_state=seed)
                            model.fit(x[train], target, sample_weight=len(train)/(2*counts[target]))
                            joblib.dump(model, temp/'model.joblib')
                        else:
                            model, history, telemetry = train_mlp(x, data, train, valid, seed, mode, strength, args)
                            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, temp/'model.pt')
                            np.savetxt(temp/'training.csv', history, delimiter=',', header='epoch,train_bce,train_dcor2,val_bce,val_dcor2,val_objective')
                            atomic_json(temp/'telemetry.json', telemetry)
                        atomic_json(temp/'features.json', tails)
                        np.savez_compressed(temp/'preprocessing.npz', **preprocessing)
                        scores = {role: score_model(model, x[roles[role]]) for role in ('calibration', 'tuning')}
                        run_records, saved = [], dict(train_rows=data['row'][train], checkpoint_rows=data['row'][valid])
                        for role in ('calibration', 'tuning'):
                            ids = roles[role]
                            for field in ('row', 'mass', 'truth'):
                                saved[f'{role}_{field}'] = data[field][ids]
                            saved[f'{role}_score'] = scores[role]
                        thresholds = {}
                        for cut in frozen['cuts']:
                            threshold = calibrate(data['mass'][roles['calibration']], scores['calibration'], cut)
                            thresholds[str(cut)] = threshold
                            for role in ('calibration', 'tuning'):
                                row = dict(seed=seed, fraction=fraction, model=name, strength=strength, cut=cut,
                                           split=role, threshold=threshold, nominal_calibration_acceptance=cut,
                                           calibration_interpretation='in_sample_diagnostic')
                                row.update(metrics_for(data, roles[role], scores[role], threshold, seed+300, args.permutations))
                                run_records.append(row)
                        np.savez_compressed(temp/'predictions.npz', **saved)
                        atomic_json(temp/'thresholds.json', thresholds)
                        atomic_json(temp/'metrics.json', run_records)
                        metadata = dict(seed=seed, fraction=fraction, model=name, mode=mode, strength=strength, include_mass=include_mass)
                        if source_identity() != execution_source:
                            raise ValueError('Source changed during fit; refuse completion')
                        complete(temp, run_id, metadata)
                        verify(temp, run_id)
                        if directory.exists():
                            # Preserve an interrupted fit before publishing its verified replacement.
                            quarantine = Path(tempfile.mkdtemp(dir=output/'fits', prefix='.incomplete_'+key))
                            directory.rename(quarantine/'original')
                        temp.rename(directory)
                        records.extend(run_records)
                    finally:
                        if temp.exists():
                            shutil.rmtree(temp)
        results_temp = Path(tempfile.mkdtemp(dir=output, prefix='.results'))
        try:
            atomic_json(results_temp/'metrics.json', records)
            complete(results_temp, run_id, dict(fits=expected))
            if (output/'results').exists():
                verify(output/'results', run_id)
                if read_json(output/'results/metrics.json') != records:
                    raise ValueError('Completed results changed')
            else:
                results_temp.rename(output/'results')
        finally:
            if results_temp.exists():
                shutil.rmtree(results_temp)
    print(f'Completed {len(expected)} unique fits: {output}', flush=True)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'events_anomalydetection.h5')
    p.add_argument('--events', type=int, default=200000)
    p.add_argument('--start-row', type=int, default=0)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--cache-dir', type=Path, default=ROOT/'feature_cache')
    p.add_argument('--degree', type=int, default=4)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    p.add_argument('--fractions', type=float, nargs='+', default=[0., .005, .01])
    p.add_argument('--lambdas', type=float, nargs='+', default=[.003])
    p.add_argument('--permutations', type=int, default=199)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    p.add_argument('--telemetry', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Explicit SYNTHETIC fixture; barred from scientific reporting')
    p.add_argument('--output', type=Path, required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (min(args.events, args.chunk_size, args.degree, args.epochs, args.permutations, args.threads) <= 0
            or len(set(args.seeds)) != len(args.seeds) or len(set(args.fractions)) != len(args.fractions)
            or args.lambdas != [.003] or any(not 0 <= f < 1 for f in args.fractions)):
        p.error('Positive counts, unique seeds/fractions, valid fractions, and prescribed --lambdas 0.003 required')
    run_study(args)


if __name__ == '__main__':
    main()

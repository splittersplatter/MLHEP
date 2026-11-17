"""Independent numerical audit. Does not import training or production metric helpers."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import jensenshannon
from provenance import read_json, atomic_json


def auc(labels, score):
    labels = np.asarray(labels)
    n1, n0 = np.sum(labels == 1), np.sum(labels == 0)
    if not n1 or not n0:
        return None
    return float((rankdata(score)[labels == 1].sum() - n1*(n1+1)/2)/(n1*n0))


def js(mass, accepted):
    edges = np.arange(3100., 3901., 50.)
    a, b = np.histogram(mass, edges)[0], np.histogram(mass[accepted], edges)[0]
    return float(jensenshannon(a, b, base=2)**2) if a.sum() and b.sum() else None


def dcor(score, mass):
    score, mass = np.asarray(score, dtype=np.float64), np.asarray(mass, dtype=np.float64)
    def center(v):
        m = np.abs(v[:, None]-v[None, :])
        return m - m.mean(0) - m.mean(1)[:, None] + m.mean()
    a, b = center(score), center(mass)
    return float(max(0, (a*b).mean()) / np.sqrt(max(1e-20, (a*a).mean()*(b*b).mean())))


def calculate(mass, truth, score, threshold, seed, permutations):
    sr = (mass >= 3300) & (mass < 3700)
    bg, sg = truth == 0, (truth == 1) & sr
    selected = score > threshold
    def rate(mask):
        return float(selected[mask].mean()) if mask.any() else None
    es, eb = rate(sg), rate(bg & sr)
    observed = js(mass[bg], selected[bg])
    rng = np.random.default_rng(seed)
    null = [js(mass[bg], rng.permutation(selected[bg])) for _ in range(permutations)] if observed is not None else []
    pick = rng.choice(np.flatnonzero(bg), min(1000, int(bg.sum())), replace=False)
    return dict(signal_auc_sr=auc(truth[sr], score[sr]), bkg_region_auc=auc(sr[bg], score[bg]),
                signal_efficiency=es, background_efficiency_sr=eb, background_efficiency_sb=rate(bg & ~sr),
                sic=float(es/np.sqrt(eb)) if es is not None and eb is not None and eb > 0 else None,
                js_divergence=observed, js_null_median=float(np.median(null)) if null else None,
                js_permutation_p=float((1 + np.count_nonzero(np.array(null) >= observed))/(permutations+1)) if null else None,
                score_mass_dcor_squared=dcor(score[pick], mass[pick]/1000) if len(pick) >= 4 else None,
                score_std=float(score[bg].std()) if bg.any() else None,
                bkg_selected_sr=int(selected[bg & sr].sum()), signal_selected_sr=int(selected[sg].sum()),
                bkg_test_sr=int((bg & sr).sum()), signal_test_sr=int(sg.sum()))


def check(actual, expected, context):
    for key, value in expected.items():
        got = actual[key]
        if value is None:
            valid = got is None
        else:
            valid = got is not None and np.isfinite(got) and np.isclose(value, got, rtol=1e-10, atol=1e-12)
        if not valid:
            raise ValueError(f'Independent calculation mismatch: {context}/{key}: {got} versus {value}')


def audit(run):
    run = Path(run)
    p = read_json(run/'protocol.json')
    config = p['identity']['config']
    count = 0
    for name in p['identity']['expected_fits']:
        directory = run/'fits'/name
        with np.load(directory/'predictions.npz', allow_pickle=False) as data:
            for row in read_json(directory/'metrics.json'):
                split = row['split']
                cmass, cscore = data['calibration_mass'], data['calibration_score']
                sb = (cmass < 3300) | (cmass >= 3700)
                threshold = float(np.quantile(cscore[sb], 1-row['cut'], method='linear'))
                check(row, dict(threshold=threshold), name)
                expected = calculate(data[f'{split}_mass'], data[f'{split}_truth'], data[f'{split}_score'], threshold,
                                     row['seed']+300, config['permutations'])
                check(row, expected, name)
                # Independent local counts in all bins and regions.
                mass, truth, score = data[f'{split}_mass'], data[f'{split}_truth'], data[f'{split}_score']
                bg, selected = truth == 0, score > threshold
                edges = np.arange(3100., 3901., 50.)
                if (row['local']['bin_total'] != np.histogram(mass[bg], edges)[0].tolist()
                        or row['local']['bin_selected'] != np.histogram(mass[bg & selected], edges)[0].tolist()):
                    raise ValueError('Independent histogram mismatch')
                for region, mask in dict(lower_sb=bg & (mass < 3300), sr=bg & (mass >= 3300) & (mass < 3700),
                                         upper_sb=bg & (mass >= 3700), sb=bg & ((mass < 3300)|(mass >= 3700))).items():
                    check(row['local']['regions'][region], dict(total=int(mask.sum()), selected=int(selected[mask].sum()),
                          efficiency=float(selected[mask].mean()) if mask.any() else None), name)
                count += 1
    return dict(status='passed', records=count, calculation='Independent NumPy/rank-sum implementation of thresholds, AUC, efficiencies, SIC, JS, permutation p, dCor and local counts',
                rtol=1e-10, atol=1e-12)


def audit_replay(run, replay):
    """Independent event-bootstrap arithmetic on verified saved decisions."""
    run, replay = Path(run), Path(replay)
    protocol = read_json(replay/'protocol.json')['protocol']
    config = read_json(run/'protocol.json')['identity']['config']
    rows = read_json(replay/'metrics.json')
    comparisons = read_json(replay/'comparisons.json')
    for fraction in protocol['fractions']:
        decisions = {}
        reference = None
        for seed in protocol['seeds']:
            for model in protocol['models']:
                filename = f'seed{seed}_f{fraction:g}_{model}.npz'
                with np.load(replay/filename, allow_pickle=False) as p:
                    mass, truth, raw_rows, score = p['mass'], p['truth'], p['row'], p['score']
                    if reference is not None and not np.array_equal(reference, raw_rows):
                        raise ValueError('Replay rows differ across models')
                    reference = raw_rows.copy()
                    for cut in protocol['cuts']:
                        threshold = read_json(run/'fits'/filename.removesuffix('.npz')/'thresholds.json')[str(cut)]
                        row = next(r for r in rows if (r['seed'],r['fraction'],r['model'],r['cut']) == (seed,fraction,model,cut))
                        check(row, calculate(mass,truth,score,threshold,seed+300,config['permutations']), filename)
                    decisions[(seed,model)] = score > float(p['threshold'])
        sr = (mass >= 3300)&(mass < 3700)
        groups = [np.flatnonzero((truth == label)&(sr == region)) for label,region in [(0,0),(0,1),(1,1)]]
        def ratios(ids):
            averages = []
            for model in ['mass_mlp','sideband_0.003']:
                values = []
                for seed in protocol['seeds']:
                    selected = decisions[(seed,model)][ids]
                    m,t = mass[ids],truth[ids]
                    s = (m >= 3300)&(m < 3700)
                    bg,sg = (t == 0)&s,(t == 1)&s
                    eb = float(selected[bg].mean()) if bg.any() else None
                    es = float(selected[sg].mean()) if sg.any() else None
                    values.append([es/np.sqrt(eb) if es is not None and eb is not None and eb > 0 else None, js(m[t == 0],selected[t == 0])])
                averages.append([float(np.mean([v[i] for v in values])) if all(v[i] is not None for v in values) else None for i in [0,1]])
            return [averages[1][i]/averages[0][i] if averages[0][i] is not None and averages[0][i] > 0 and averages[1][i] is not None else None for i in [0,1]]
        target = next(r for r in comparisons if r['fraction'] == fraction)
        point = ratios(np.arange(len(mass)))
        check(target, dict(sic_ratio=point[0],js_ratio=point[1]), str(fraction))
        rule = protocol['bootstrap']
        rng = np.random.default_rng(rule['seed'])
        samples = [ratios(np.concatenate([rng.choice(group,len(group),replace=True) for group in groups]))
                   for _ in range(rule['replicates'])] if all(len(g) for g in groups) else [[None,None]]*rule['replicates']
        for i,key in enumerate(['sic','js']):
            invalid = sum(v[i] is None for v in samples)
            if target[f'invalid_{key}_replicates'] != invalid:
                raise ValueError('Bootstrap invalid count mismatch')
            expected = np.quantile([v[i] for v in samples],[.025,.975]).tolist() if not invalid else None
            actual = target[f'{key}_interval']
            if (expected is None and actual is not None) or (expected is not None and not np.allclose(expected,actual,rtol=1e-10,atol=1e-12)):
                raise ValueError('Independent bootstrap interval mismatch')
    return dict(status='passed', records=len(rows), fractions=len(comparisons), calculation='Independent paired stratified bootstrap and replay arithmetic')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replay', type=Path)
    args = p.parse_args()
    atomic_json(args.output, audit_replay(args.run,args.replay) if args.replay else audit(args.run))


if __name__ == '__main__':
    main()

"""Independent numerical audit. Does not import training or production metric helpers."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import jensenshannon
from provenance import read_json, atomic_json, identity, file_hash


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


def compare_tree(actual, expected, context='aggregate'):
    """Compare a numerical/structural subset, including nulls and array shapes."""
    if isinstance(expected, dict):
        for key,value in expected.items():
            compare_tree(actual[key],value,context+'/'+key)
    elif isinstance(expected, (list,tuple)):
        if not isinstance(actual,(list,tuple)) or len(actual) != len(expected):
            raise ValueError('Independent shape mismatch: '+context)
        for i,value in enumerate(expected):
            compare_tree(actual[i],value,context+f'/{i}')
    elif isinstance(expected,(int,float,np.number)) and not isinstance(expected,(bool,np.bool_)):
        if actual is None or not np.isfinite(actual) or not np.isclose(actual,expected,rtol=1e-10,atol=1e-12):
            raise ValueError(f'Independent numerical mismatch: {context}: {actual} vs {expected}')
    elif actual != expected:
        raise ValueError(f'Independent value mismatch: {context}: {actual} vs {expected}')


def independent_closure(mass, selected):
    edges=np.arange(3100.,3901.,50.)
    n=np.histogram(mass,edges)[0]
    k=np.histogram(np.asarray(mass)[selected],edges)[0]
    x=((edges[1:]+edges[:-1])/2-3500)/1000
    side=np.abs(x) >= .2
    inside=~side
    empty=dict(status='unavailable',residual=None,conditional_standard_error=None,diagnostic_within_3se=None)
    if (n == 0).any() or n[inside].sum() < 20 or (n[side] < 5).any():
        return dict(empty,reason='insufficient_bin_support')
    rate=k/n
    posterior=(k+.5)/(n+1)
    variance=posterior*(1-posterior)/(n+2)
    w=1/variance[side]
    s0,s1,s2=w.sum(),(w*x[side]).sum(),(w*x[side]**2).sum()
    covariance=np.array([[s2,-s1],[-s1,s0]])/(s0*s2-s1*s1)
    beta=covariance @ np.array([(w*rate[side]).sum(),(w*x[side]*rate[side]).sum()])
    predicted=beta[0]+beta[1]*x[inside]
    if ((predicted < 0)|(predicted > 1)).any():
        return dict(empty,reason='unphysical_linear_extrapolation',coefficients=beta.tolist())
    t=np.array([1.,np.average(x[inside],weights=n[inside])])
    expected=float(t @ beta)
    observed=float(k[inside].sum()/n[inside].sum())
    error=float(np.sqrt(t @ covariance @ t+(n[inside]**2*variance[inside]).sum()/n[inside].sum()**2))
    return dict(status='defined',reason=None,coefficients=beta.tolist(),covariance=covariance.tolist(),
                sr_expected=expected,sr_observed=observed,residual=observed-expected,
                conditional_standard_error=error,diagnostic_within_3se=bool(abs(observed-expected) <= 3*error))


def independent_uncertainty(calibration,mass,truth,score,cut,reps,seed,same):
    cm,cs,ct=(calibration[k] for k in ['mass','score','truth'])
    sb=(cm < 3300)|(cm >= 3700)
    result=dict(status='unavailable',replicates_requested=reps,replicates_attempted=0,invalid_replicates=None,
                residual_interval_95=None,threshold_interval_95=None,seed=seed,cut=cut,same_sample=same)
    if not sb.any():
        return dict(result,reason='no_calibration_sidebands')
    threshold=np.quantile(cs[sb],1-cut)
    bg=truth == 0
    point=independent_closure(mass[bg],score[bg] > threshold)
    result['point']=point
    if point['status'] != 'defined':
        return dict(result,reason='point_closure_'+point['reason'])
    strata=[np.flatnonzero(bg & mask) for mask in [mass < 3300,(mass >= 3300)&(mass < 3700),mass >= 3700]]
    rng=np.random.default_rng(seed)
    residuals,thresholds=[],[]
    invalid=0
    for _ in range(reps):
        ids=rng.integers(0,len(cm),len(cm))
        if not sb[ids].any():
            invalid+=1
            continue
        threshold=np.quantile(cs[ids][sb[ids]],1-cut)
        draw=ids[ct[ids] == 0] if same else np.concatenate([rng.choice(g,len(g),replace=True) for g in strata])
        fit=independent_closure(mass[draw],score[draw] > threshold)
        thresholds.append(threshold)
        if fit['status'] == 'defined':
            residuals.append(fit['residual'])
        else:
            invalid+=1
    interval=np.quantile(residuals,[.025,.975]).tolist() if invalid == 0 else None
    return dict(result,status='defined' if interval is not None else 'unavailable',reason=None if interval is not None else 'invalid_bootstrap_replicates',
        replicates_attempted=reps,invalid_replicates=invalid,residual_interval_95=interval,
        threshold_interval_95=np.quantile(thresholds,[.025,.975]).tolist() if len(thresholds) == reps else None,
        diagnostic_interval_contains_zero=bool(interval and interval[0] <= 0 <= interval[1]))


def audit_local(row,mass,truth,score,calibration,config,same=False):
    threshold=row['threshold']
    bg,passed=truth == 0,score > threshold
    detail=row['local']
    masks=dict(lower_sb=mass < 3300,sr=(mass >= 3300)&(mass < 3700),upper_sb=mass >= 3700,sb=(mass < 3300)|(mass >= 3700))
    for region,mask in masks.items():
        n=int((bg & mask).sum());k=int((bg & mask & passed).sum())
        if n:
            z=1.959963984540054
            denominator=n+z*z
            center=(k+z*z/2)/denominator
            radius=z*np.sqrt(k*(n-k)/n+z*z/4)/denominator
            interval=[float(max(0,center-radius)),float(min(1,center+radius))]
        else:
            interval=None
        compare_tree(detail['regions'][region],dict(total=n,selected=k,efficiency=k/n if n else None,conditional_wilson_95=interval))
    edges=np.arange(3100.,3901.,50.)
    counts=np.histogram(mass[bg],edges)[0]
    accepted=np.histogram(mass[bg & passed],edges)[0]
    esr=detail['regions']['sr']['efficiency'];esb=detail['regions']['sb']['efficiency']
    compare_tree(detail,dict(bin_total=counts.tolist(),bin_selected=accepted.tolist(),
        background_total=int(bg.sum()),background_selected=int((bg & passed).sum()),
        background_score_std=float(score[bg].std()) if bg.any() else None,
        sr_to_measured_sb_ratio=esr/esb if esr is not None and esb is not None and esb > 0 else None,
        sr_to_nominal_005_ratio=esr/.05 if esr is not None else None))
    if not counts.sum() or not accepted.sum():
        null=dict(status='undefined_empty_sample',observed=None,null_mean=None,null_median=None,null_95_interval=None,p_upper=None)
    else:
        draws=np.random.default_rng(row['seed']+300).multivariate_hypergeometric(counts,int(accepted.sum()),size=config['permutations'])
        values=jensenshannon(counts[None,:],draws,base=2,axis=1)**2
        observed=float(jensenshannon(counts,accepted,base=2)**2)
        null=dict(status='defined',observed=observed,null_mean=float(values.mean()),null_median=float(np.median(values)),
            null_95_interval=np.quantile(values,[.025,.975]).tolist(),p_upper=float((1+(values >= observed).sum())/(len(values)+1)))
    compare_tree(detail['js'],null,'local/JS reference')
    compare_tree(row['closure'],independent_closure(mass[bg],passed[bg]),'closure')
    if 'closure_uncertainty' in row:
        compare_tree(row['closure_uncertainty'],independent_uncertainty(calibration,mass,truth,score,row['cut'],config['closure_replicates'],row['seed']+1189,same),'closure uncertainty')


def independent_mean(values):
    return sum(values)/len(values) if values and all(v is not None and np.isfinite(v) for v in values) else None


def independent_ratio(a,b):
    return a/b if a is not None and b is not None and b > 0 else None


def audit_assessment(assessment,rows,config):
    seeds=config['seeds'];fractions=[f for f in config['fractions'] if f > 0]
    if not fractions:
        compare_tree(assessment,dict(status='control_only',targets_met=False,candidates=[]))
        return
    indexed={(r['seed'],r['fraction'],r['model']):r for r in rows if r['split'] == 'tuning' and r['cut'] == .05}
    def ratios(f,subset):
        means=[[independent_mean([indexed[(s,f,m)][metric] for s in subset]) for metric in ['sic','js_divergence']] for m in ['mass_mlp','sideband_0.003']]
        return [independent_ratio(means[1][i],means[0][i]) for i in [0,1]]
    regimes=[]
    for f in fractions:
        sr,jr=ratios(f,seeds)
        regimes.append(dict(fraction=f,expected_seeds=seeds,denominator=len(seeds),sic_retained=sr,js_ratio=jr,
            qualifies=bool(sr is not None and jr is not None and sr >= .9 and jr <= .8),
            leave_one_seed_out=[dict(omitted_seed=s,**dict(zip(['sic_ratio','js_ratio'],ratios(f,[v for v in seeds if v != s])))) for s in seeds]))
    targets=all(r['qualifies'] for r in regimes)
    compare_tree(assessment,dict(status='complete',target_fractions=fractions,targets_met=targets,expected_records=len(seeds)*len(fractions)*2,
        actual_records=len(seeds)*len(fractions)*2,missing_keys=[],new_fits=0,
        candidates=[dict(lambda_value=.003,eligible=targets,regimes=regimes)]),'fixed-lambda assessment')
    controls=[dict(r,**{'lambda':0. if r['model'] == 'mass_mlp' else .003}) for r in rows if r['fraction'] == 0 and r['model'] in ['mass_mlp','sideband_0.003'] and r['split'] == 'tuning' and r['cut'] == .05]
    compare_tree(assessment['control_records'],controls,'background controls')


def audit_transform_comparison(records,runs,config,baseline='log_floor_1e-12'):
    expected=[]
    for variant in runs:
        for f in config['fractions']:
            for model in ['efp_mlp','mass_mlp','mass_gbdt','sideband_0.003','oracle_0.003']:
                groups=[[r for r in runs[v]['rows'] if r['fraction'] == f and r['model'] == model and r['split'] == 'tuning' and r['cut'] == .05] for v in [baseline,variant]]
                means=[[independent_mean([r[k] for r in g]) for k in ['sic','js_divergence']] for g in groups]
                sr,jr=[independent_ratio(means[1][i],means[0][i]) for i in [0,1]]
                paired=[]
                for s in config['seeds']:
                    b,v=[next(r for r in g if r['seed'] == s) for g in groups]
                    paired.append(dict(seed=s,baseline_sic=b['sic'],variant_sic=v['sic'],baseline_js=b['js_divergence'],variant_js=v['js_divergence'],baseline_closure=b['closure_uncertainty'],variant_closure=v['closure_uncertainty']))
                expected.append(dict(transform=variant,fraction=f,model=model,cut=.05,expected_seeds=config['seeds'],seed_denominator=len(config['seeds']),
                    sic_ratio=sr,js_ratio=jr,undefined_reason='missing_metric_or_zero_baseline' if sr is None or jr is None else None,per_seed=paired))
    compare_tree(records,expected,'preprocessing comparison')
    return dict(status='passed',records=len(records),calculation='Independent complete-seed sums, ratios and exact paired record membership')


def envelope(run):
    """Independent grid, content and role audit, without production validators."""
    run=Path(run)
    p=read_json(run/'protocol.json')
    rule=p['identity']
    if identity(rule) != p['run_id']:
        raise ValueError('Independent protocol identity mismatch')
    split=read_json(run/'splits.json')
    if identity(split) != rule['split_sha256'] or split['dataset_sha256'] != rule['dataset_sha256']:
        raise ValueError('Independent split identity mismatch')
    config=rule['config']
    models=['efp_mlp','mass_mlp','mass_gbdt']+[f'{mode}_{lam:g}' for mode in ['sideband','oracle'] for lam in config['lambdas']]
    expected=[f'seed{s}_f{f:g}_{m}' for s in config['seeds'] for f in config['fractions'] for m in models]
    if rule['expected_fits'] != expected or not expected or len(set(expected)) != len(expected):
        raise ValueError('Independent fit grid mismatch')
    fixed=None
    universe=None
    for seed in config['seeds']:
        roles=split['seeds'][str(seed)]
        if set(roles) != {'training','checkpoint','calibration','tuning','final'}:
            raise ValueError('Independent role grid mismatch')
        seen=set()
        for name,members in roles.items():
            if not members or len(set(members)) != len(members) or seen & set(members):
                raise ValueError('Independent duplicate/overlapping roles')
            seen.update(members)
        current={k:roles[k] for k in ['calibration','tuning','final']}
        if fixed is not None and (fixed != current or seen != universe):
            raise ValueError('Independent fixed role membership changed')
        fixed,universe=current,seen
    for name,digest in rule['source'].items():
        if file_hash(run/'source'/name) != digest:
            raise ValueError('Independent source snapshot mismatch')
    if config['smoke']:
        raw_rows=np.arange(rule['cache_metadata']['actual_range'][1])
        mass=np.tile([3200.,3400.,3600.,3800.],len(raw_rows)//4)
        truth=np.tile([0,0,1,0,1,1,0,1],len(raw_rows)//8)
    else:
        cache=p['cache']
        if file_hash(cache['path']) != rule['cache_sha256']:
            raise ValueError('Independent cache checksum mismatch')
        with np.load(cache['path'],allow_pickle=False) as saved:
            raw_rows,mass,truth=(saved[k].copy() for k in ('row','mass','truth'))
    if len(set(map(int,raw_rows))) != len(raw_rows) or set(map(int,raw_rows)) != universe:
        raise ValueError('Independent cache/role universe mismatch')
    lookup={int(row):i for i,row in enumerate(raw_rows)}
    return p,split,lookup,mass,truth


def verify_files(directory,run_id):
    directory=Path(directory)
    marker=read_json(directory/'complete.json')
    if marker['run_id'] != run_id or not marker['files']:
        raise ValueError('Independent artifact identity mismatch')
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'complete.json'}
    if actual != set(marker['files']):
        raise ValueError('Independent artifact membership mismatch')
    for name,digest in marker['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(directory/name) != digest:
            raise ValueError('Independent artifact checksum mismatch')
    return marker


def audit(run):
    run = Path(run)
    p,split_manifest,lookup,source_mass,source_truth=envelope(run)
    config = p['identity']['config']
    count = 0
    aggregate=[]
    for name in p['identity']['expected_fits']:
        directory = run/'fits'/name
        marker=verify_files(directory,p['run_id'])
        meta=marker['metadata']
        roles=split_manifest['seeds'][str(meta['seed'])]
        rows=read_json(directory/'metrics.json')
        keys=[(r['split'],r['cut']) for r in rows]
        if len(keys) != 4 or set(keys) != {(s,c) for s in ['calibration','tuning'] for c in [.2,.05]}:
            raise ValueError('Independent metric grid mismatch')
        aggregate.extend(rows)
        with np.load(directory/'predictions.npz', allow_pickle=False) as data:
            for role in ['calibration','tuning']:
                raw_ids=np.array([lookup[r] for r in roles[role]])
                if data[role+'_row'].tolist() != roles[role] or not np.array_equal(data[role+'_mass'],source_mass[raw_ids]) or not np.array_equal(data[role+'_truth'],source_truth[raw_ids]):
                    raise ValueError('Independent prediction membership/content mismatch')
            sr=(source_mass >= 3300)&(source_mass < 3700)
            for key,role,offset in [('train_rows','training',0),('checkpoint_rows','checkpoint',1)]:
                pool=np.array([lookup[r] for r in roles[role]])
                bg=pool[source_truth[pool] == 0]
                signal=pool[(source_truth[pool] == 1)&sr[pool]]
                signal=np.random.default_rng(meta['seed']+offset).permutation(signal)
                amount=int(round(meta['fraction']/(1-meta['fraction'])*int(sr[bg].sum())))
                expected_rows=np.array(list(lookup))[np.concatenate([bg,signal[:amount]])]
                if not np.array_equal(data[key],expected_rows):
                    raise ValueError('Independent injection membership mismatch')
            for row in rows:
                split = row['split']
                cmass, cscore = data['calibration_mass'], data['calibration_score']
                sb = (cmass < 3300) | (cmass >= 3700)
                threshold = float(np.quantile(cscore[sb], 1-row['cut'], method='linear'))
                check(row, dict(threshold=threshold), name)
                expected = calculate(data[f'{split}_mass'], data[f'{split}_truth'], data[f'{split}_score'], threshold,
                                     row['seed']+300, config['permutations'])
                check(row, expected, name)
                calibration={k:data["calibration_"+k] for k in ["mass","truth","score"]}
                audit_local(row,data[split+"_mass"],data[split+"_truth"],data[split+"_score"],calibration,config,split == "calibration")
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
    verify_files(run/'results',p['run_id'])
    if read_json(run/'results/metrics.json') != aggregate:
        raise ValueError('Independent published aggregate differs from fit records')
    return dict(status='passed', records=count, exact_role_membership='passed', published_aggregate='passed', calculation='Independent NumPy/rank-sum implementation of thresholds, AUC, efficiencies, SIC, JS, permutation p, dCor local counts, Wilson intervals, fixed-count JS reference and sideband closure/calibration bootstrap',
                rtol=1e-10, atol=1e-12, auditor_sha256=file_hash(Path(__file__)))


def audit_replay(run, replay):
    """Independent event-bootstrap arithmetic on verified saved decisions."""
    run, replay = Path(run), Path(replay)
    training,split,lookup,source_mass,source_truth=envelope(run)
    verify_files(replay,training['run_id'])
    frozen=read_json(replay/'protocol.json')
    protocol=frozen['protocol']
    if identity(protocol) != frozen['freeze_id'] or protocol['run_id'] != training['run_id']:
        raise ValueError('Independent freeze identity mismatch')
    config = training['identity']['config']
    if protocol['rows'] != split['seeds'][str(config['seeds'][0])]['final'] or protocol['seeds'] != config['seeds'] or protocol['fractions'] != config['fractions'] or protocol['models'] != ['mass_mlp','sideband_0.003','efp_mlp'] or protocol['cuts'] != [.2,.05]:
        raise ValueError('Independent frozen sample/model grid mismatch')
    rows = read_json(replay/'metrics.json')
    comparisons = read_json(replay/'comparisons.json')
    keys=[(r['seed'],r['fraction'],r['model'],r['cut']) for r in rows]
    expected={(s,f,m,c) for s in protocol['seeds'] for f in protocol['fractions'] for m in protocol['models'] for c in protocol['cuts']}
    if len(keys) != len(set(keys)) or set(keys) != expected or len(comparisons) != len(protocol['fractions']) or {r['fraction'] for r in comparisons} != set(protocol['fractions']):
        raise ValueError('Independent replay comparison grid mismatch')
    reserved=np.array([lookup[r] for r in protocol['rows']])
    reserved=reserved[(source_truth[reserved] == 0)|((source_mass[reserved] >= 3300)&(source_mass[reserved] < 3700))]
    expected_rows=np.array(list(lookup))[reserved]
    for fraction in protocol['fractions']:
        decisions = {}
        reference = None
        for seed in protocol['seeds']:
            for model in protocol['models']:
                filename = f'seed{seed}_f{fraction:g}_{model}.npz'
                with np.load(replay/filename, allow_pickle=False) as p:
                    mass, truth, raw_rows, score = p['mass'], p['truth'], p['row'], p['score']
                    if not np.array_equal(raw_rows,expected_rows) or not np.array_equal(mass,source_mass[reserved]) or not np.array_equal(truth,source_truth[reserved]):
                        raise ValueError('Independent exact final membership/content mismatch')
                    calibration_cuts=read_json(run/'fits'/filename.removesuffix('.npz')/'thresholds.json')
                    if float(p['threshold']) != calibration_cuts['0.05']:
                        raise ValueError('Independent saved bootstrap decision threshold mismatch')
                    if reference is not None and not np.array_equal(reference, raw_rows):
                        raise ValueError('Replay rows differ across models')
                    reference = raw_rows.copy()
                    for cut in protocol['cuts']:
                        threshold = read_json(run/'fits'/filename.removesuffix('.npz')/'thresholds.json')[str(cut)]
                        row = next(r for r in rows if (r['seed'],r['fraction'],r['model'],r['cut']) == (seed,fraction,model,cut))
                        check(row, calculate(mass,truth,score,threshold,seed+300,config['permutations']), filename)
                        with np.load(run/'fits'/filename.removesuffix('.npz')/'predictions.npz',allow_pickle=False) as cal:
                            calibration={k:cal['calibration_'+k] for k in ['mass','truth','score']}
                            audit_local(row,mass,truth,score,calibration,config)
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
        point_pass=bool(point[0] is not None and point[1] is not None and point[0] >= .9 and point[1] <= .8)
        si,ji=target['sic_interval'],target['js_interval']
        interval_pass=bool(si and ji and si[0] >= .9 and ji[1] <= .8)
        if target['point_targets_met'] != point_pass or target['intervals_clear_targets'] != interval_pass:
            raise ValueError('Independent target qualification mismatch')
    return dict(status='passed', records=len(rows), fractions=len(comparisons), calculation='Independent paired stratified bootstrap and replay arithmetic')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replay', type=Path)
    args = p.parse_args()
    destination=args.output.resolve()
    for source in [args.run]+([args.replay] if args.replay else []):
        root=source.resolve()
        if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
            raise ValueError('Keep independent audit output separate from source artifacts')
    atomic_json(args.output, audit_replay(args.run,args.replay) if args.replay else audit(args.run))


if __name__ == '__main__':
    main()

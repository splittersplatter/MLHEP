"""Paired stratified event resampling; no dropping undefined replicates."""
import numpy as np
from metrics import region, js_stat, mean_complete, ratio
from protocol import BASELINE, DISCO, SIC_TARGET, JS_TARGET


def measures(mass, truth, selected):
    sr = region(mass).astype(bool)
    bg, signal = truth == 0, (truth == 1) & sr
    eb = float(selected[bg & sr].mean()) if (bg & sr).any() else None
    es = float(selected[signal].mean()) if signal.any() else None
    sic = float(es/np.sqrt(eb)) if es is not None and eb is not None and eb > 0 else None
    return sic, js_stat(mass[bg], selected[bg])


def paired(mass, truth, decisions, seeds, replicates=300, seed=773):
    from metrics import validate
    mass,truth,_ = validate(mass,truth,np.zeros(len(mass)))
    if not seeds or len(set(seeds)) != len(seeds) or replicates < 1:
        raise ValueError('Nonempty unique seed list and positive bootstrap count required')
    expected={(s,m) for s in seeds for m in (BASELINE,DISCO)}
    if not expected <= set(decisions):
        raise ValueError('Missing paired bootstrap decisions')
    for key,values in decisions.items():
        if np.asarray(values).shape != mass.shape or np.asarray(values).dtype != np.dtype(bool):
            raise ValueError('Unaligned or nonboolean bootstrap decisions')
    def point(ids):
        means = []
        for model in (BASELINE, DISCO):
            values = [measures(mass[ids], truth[ids], decisions[(s, model)][ids]) for s in seeds]
            means.append([mean_complete([v[i] for v in values]) for i in range(2)])
        return [ratio(means[1][i], means[0][i]) for i in range(2)]
    estimate = point(np.arange(len(mass)))
    strata = [np.flatnonzero((truth == label) & (region(mass) == sr)) for label, sr in [(0, 0), (0, 1), (1, 1)]]
    rng = np.random.default_rng(seed)
    samples = []
    if all(len(group) for group in strata):
        for _ in range(replicates):
            ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
            samples.append(point(ids))
    invalid = [sum(values[i] is None for values in samples) for i in range(2)] if samples else [replicates, replicates]
    intervals = [np.quantile([v[i] for v in samples], [.025, .975]).tolist()
                 if invalid[i] == 0 and len(samples) == replicates else None for i in range(2)]
    return dict(sic_ratio=estimate[0], js_ratio=estimate[1], sic_interval=intervals[0], js_interval=intervals[1],
                replicates=replicates, invalid_sic_replicates=invalid[0], invalid_js_replicates=invalid[1],
                invalid_rule='any invalid replicate makes interval unavailable', seed_denominator=len(seeds),
                point_targets_met=bool(estimate[0] is not None and estimate[1] is not None and estimate[0] >= SIC_TARGET and estimate[1] <= JS_TARGET),
                intervals_clear_targets=bool(intervals[0] and intervals[1] and intervals[0][0] >= SIC_TARGET and intervals[1][1] <= JS_TARGET),
                interpretation='Conditional on fixed models, thresholds, and truth/region counts; no training or simulator uncertainty')

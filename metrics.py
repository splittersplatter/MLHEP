"""Versioned metrics: strict score > cut, linear quantiles, base-2 JS."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import jensenshannon
from sculpting_diagnostics import diagnose
from protocol import BINS, SR_MIN, SR_MAX, MASS_MIN, MASS_MAX

VERSION = 'v2-strict-cut-base2-js'
MIN_CORRELATION = 4


def region(mass):
    return ((mass >= SR_MIN) & (mass < SR_MAX)).astype(np.int64)


def disco_squared(score, mass):
    if len(score) < 4:
        return score.sum() * 0
    def centered(v):
        distance = torch.abs(v[:, None] - v[None, :])
        return distance - distance.mean(0) - distance.mean(1)[:, None] + distance.mean()
    a, b = centered(score), centered(mass)
    denominator = torch.sqrt((a.square().mean() * b.square().mean()).clamp_min(1e-20))
    return (a*b).mean().clamp_min(0) / denominator


def validate(mass, truth, score):
    mass, truth, score = map(np.asarray, (mass, truth, score))
    if (mass.ndim != 1 or mass.shape != truth.shape or mass.shape != score.shape
            or not np.isfinite(mass).all() or not np.isfinite(score).all()
            or not np.isin(truth, [0, 1]).all() or ((mass < MASS_MIN) | (mass >= MASS_MAX)).any()):
        raise ValueError('Invalid or unaligned metric inputs')
    return mass, truth, score


def calibrate(mass, score, cut, mode='sideband', truth=None):
    mass, score = np.asarray(mass), np.asarray(score)
    if not 0 < cut < 1 or mass.shape != score.shape or mass.ndim != 1 or not np.isfinite(score).all():
        raise ValueError('Invalid calibration inputs')
    if mode == 'sideband':
        mask = region(mass) == 0
    elif mode == 'simulation_sr_background' and truth is not None:
        mask = (region(mass) == 1) & (np.asarray(truth) == 0)
    else:
        raise ValueError('Unsupported calibration mode')
    if not mask.any():
        raise ValueError('No calibration support')
    return float(np.quantile(score[mask], 1-cut, method='linear'))


def js_stat(mass, selected, bins=BINS):
    base = np.histogram(mass, bins)[0]
    accepted = np.histogram(np.asarray(mass)[selected], bins)[0]
    if not accepted.sum() or not base.sum():
        return None
    return float(jensenshannon(base, accepted, base=2)**2)


def metrics_for(data, ids, score, threshold, seed, permutations):
    mass, truth, score = validate(data['mass'][ids], data['truth'][ids], score)
    if not np.isfinite(threshold) or permutations < 1:
        raise ValueError('Invalid threshold or permutations')
    sr, background = region(mass).astype(bool), truth == 0
    signal, selected = (truth == 1) & sr, score > threshold
    bsr, bsb = background & sr, background & ~sr
    reasons = {}
    def unavailable(key, reason):
        reasons[key] = reason
        return None
    def efficiency(key, mask):
        return float(selected[mask].mean()) if mask.any() else unavailable(key, 'empty_denominator')
    def auc(key, labels, values):
        return float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else unavailable(key, 'single_or_missing_class')
    eb = efficiency('background_efficiency_sr', bsr)
    es = efficiency('signal_efficiency', signal)
    js = js_stat(mass[background], selected[background])
    if js is None:
        unavailable('js_divergence', 'no_selected_background_or_no_background')
    rng = np.random.default_rng(seed)
    null = [js_stat(mass[background], rng.permutation(selected[background])) for _ in range(permutations)] if js is not None else []
    pick = rng.choice(np.flatnonzero(background), min(1000, int(background.sum())), replace=False)
    if len(pick) < MIN_CORRELATION:
        dc = unavailable('score_mass_dcor_squared', 'insufficient_samples')
        dc_status = 'insufficient_samples'
    else:
        dc = disco_squared(torch.tensor(score[pick], dtype=torch.float64),
                           torch.tensor(mass[pick]/1000, dtype=torch.float64)).item()
        dc_status = 'constant_variable' if np.ptp(score[pick]) == 0 or np.ptp(mass[pick]) == 0 else 'defined'
    sic = es / np.sqrt(eb) if es is not None and eb is not None and eb > 0 else unavailable('sic', 'missing_class_or_zero_background_efficiency')
    detail = diagnose(mass, truth, score, threshold, permutations, seed)
    from local_closure import closure
    result = dict(metric_version=VERSION, signal_auc_sr=auc('signal_auc_sr', truth[sr], score[sr]),
                  bkg_region_auc=auc('bkg_region_auc', sr[background], score[background]),
                  signal_efficiency=es, background_efficiency_sr=eb,
                  background_efficiency_sb=efficiency('background_efficiency_sb', bsb), sic=float(sic) if sic is not None else None,
                  bkg_selected_sr=int(selected[bsr].sum()), signal_selected_sr=int(selected[signal].sum()),
                  bkg_test_sr=int(bsr.sum()), signal_test_sr=int(signal.sum()),
                  js_divergence=js, js_null_median=float(np.median(null)) if null else unavailable('js_null_median', 'undefined_observed_js'),
                  js_permutation_p=float((1 + np.sum(np.asarray(null) >= js))/(permutations+1)) if null else unavailable('js_permutation_p', 'undefined_observed_js'),
                  score_mass_dcor_squared=dc, correlation_status=dc_status,
                  score_std=float(score[background].std()) if background.any() else unavailable('score_std', 'no_background'),
                  counts=dict(total=len(mass), background=int(background.sum()), signal_sr=int(signal.sum())),
                  undefined_reasons=reasons, local=detail,
                  closure=closure(mass[background], selected[background]))
    return result


def mean_complete(values):
    if not values or any(v is None or not np.isfinite(v) for v in values):
        return None
    return float(np.mean(values))


def ratio(numerator, denominator):
    return numerator/denominator if numerator is not None and denominator is not None and denominator > 0 else None

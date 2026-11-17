"""Descriptive sideband-only linear efficiency interpolation, no bump significance.

Weighted least squares fits k/n on bin centers using sidebands only. Jeffreys
binomial variances prevent zero errors. This is a diagnostic model, not a
validated discovery background fit. calibrated_closure propagates empirical
calibration and evaluation uncertainty conditional on the fitted model.
"""
import numpy as np

from protocol import BINS, SR_MIN, SR_MAX


def closure(mass, selected):
    mass, selected = np.asarray(mass), np.asarray(selected)
    if mass.shape != selected.shape or mass.ndim != 1:
        raise ValueError('Invalid closure arrays')
    n = np.histogram(mass, BINS)[0]
    k = np.histogram(mass[selected], BINS)[0]
    centers = (BINS[1:] + BINS[:-1])/2
    sb = (centers < SR_MIN) | (centers >= SR_MAX)
    sr = ~sb
    description = dict(model='sideband_only_linear_efficiency_v1',
                       interpretation='Exploratory conditional interpolation; no bump significance or validated removal claim',
                       calibration_uncertainty='not_included; condition on frozen threshold',
                       status='unavailable', residual=None, conditional_standard_error=None,
                       diagnostic_within_3se=None)
    if np.any(n == 0) or n[sr].sum() < 20 or (n[sb] < 5).any():
        return dict(description, reason='insufficient_bin_support')
    x = np.column_stack((np.ones(16), (centers - 3500)/1000))
    rates = k/n
    p = (k+.5)/(n+1)
    variance = p*(1-p)/(n+2)
    weight = 1/variance[sb]
    cov = np.linalg.inv(x[sb].T @ (weight[:, None]*x[sb]))
    beta = cov @ (x[sb].T @ (weight*rates[sb]))
    predicted = x[sr] @ beta
    if ((predicted < 0) | (predicted > 1)).any():
        return dict(description, reason='unphysical_linear_extrapolation', coefficients=beta.tolist())
    target = np.average(x[sr], axis=0, weights=n[sr])
    expected = float(target @ beta)
    observed = float(k[sr].sum()/n[sr].sum())
    stderr = float(np.sqrt(target @ cov @ target + np.sum(n[sr]**2*variance[sr])/n[sr].sum()**2))
    return dict(description, status='defined', reason=None, coefficients=beta.tolist(),
                covariance=cov.tolist(), sr_expected=expected, sr_observed=observed,
                residual=observed-expected, conditional_standard_error=stderr,
                diagnostic_within_3se=bool(abs(observed-expected) <= 3*stderr))


def calibrated_closure(calibration, mass, truth, score, cut, replicates=199, seed=1189, same_sample=False):
    """Propagate empirical calibration and evaluation sampling with fixed models.

    Each draw re-estimates the sideband calibration quantile, then refits the
    efficiency line using evaluation sidebands only. SR outcomes are used only
    for its residual. Stratification preserves observed evaluation region counts.
    This is a conditional diagnostic, not a discovery background-fit validation.
    """
    if replicates < 1 or not 0 < cut < 1:
        raise ValueError('Invalid closure resampling protocol')
    cm,cs,ct=(np.asarray(calibration[k]) for k in ('mass','score','truth'))
    mass,truth,score=map(np.asarray,(mass,truth,score))
    if cm.ndim != 1 or cm.shape != cs.shape or cm.shape != ct.shape or mass.shape != score.shape or mass.shape != truth.shape:
        raise ValueError('Unaligned closure calibration/evaluation arrays')
    if not np.isfinite(cm).all() or not np.isfinite(cs).all() or not np.isfinite(mass).all() or not np.isfinite(score).all():
        raise ValueError('Nonfinite closure inputs')
    if same_sample and not (np.array_equal(cm,mass) and np.array_equal(cs,score) and np.array_equal(ct,truth)):
        raise ValueError('In-sample calibration closure must use identical arrays')
    sb=(cm < SR_MIN)|(cm >= SR_MAX)
    metadata=dict(version='calibrated_sideband_linear_closure_v1',cut=cut,replicates_requested=replicates,
                  seed=seed,same_sample=same_sample,replicates_attempted=0,invalid_replicates=None,
                  residual_interval_95=None,threshold_interval_95=None,status='unavailable',
                  calibration_uncertainty='empirical sideband quantile resampling included',
                  interpretation='Conditional on fixed model; empirical calibration/evaluation event uncertainty. Not a bump significance or proof of sculpting removal.')
    if not sb.any():
        return dict(metadata,reason='no_calibration_sidebands')
    threshold=float(np.quantile(cs[sb],1-cut,method='linear'))
    bg=truth == 0
    point=closure(mass[bg],score[bg] > threshold)
    if point['status'] != 'defined':
        return dict(metadata,reason='point_closure_'+point['reason'],point=point)
    labels=np.where(mass < SR_MIN,0,np.where(mass < SR_MAX,1,2))
    groups=[np.flatnonzero(bg & (labels == i)) for i in range(3)]
    rng=np.random.default_rng(seed)
    residuals=[]
    thresholds=[]
    invalid=0
    for _ in range(replicates):
        calibration_ids=rng.integers(0,len(cm),len(cm))
        draw_sb=sb[calibration_ids]
        if not draw_sb.any():
            invalid+=1
            continue
        threshold_draw=float(np.quantile(cs[calibration_ids][draw_sb],1-cut,method='linear'))
        if same_sample:
            ids=calibration_ids[ct[calibration_ids] == 0]
        else:
            ids=np.concatenate([rng.choice(group,len(group),replace=True) for group in groups])
        result=closure(mass[ids],score[ids] > threshold_draw)
        thresholds.append(threshold_draw)
        if result['status'] != 'defined':
            invalid+=1
        else:
            residuals.append(result['residual'])
    interval=np.quantile(residuals,[.025,.975]).tolist() if invalid == 0 else None
    return dict(metadata,status='defined' if interval is not None else 'unavailable',reason=None if interval is not None else 'invalid_bootstrap_replicates',
                point=point,replicates_attempted=replicates,invalid_replicates=invalid,
                residual_interval_95=interval,threshold_interval_95=np.quantile(thresholds,[.025,.975]).tolist() if len(thresholds) == replicates else None,
                diagnostic_interval_contains_zero=bool(interval and interval[0] <= 0 <= interval[1]))


def background_controls(replicates=199, seed=661):
    """Predefined, explicitly synthetic controls: flat, slope, localized step."""
    rng=np.random.default_rng(seed)
    cm=rng.uniform(3100,3900,30000)
    em=rng.uniform(3100,3900,30000)
    cu=rng.uniform(0,1,len(cm))
    eu=rng.uniform(0,1,len(em))
    cases=[]
    for name,slope,step in [('flat',0.,0.),('smooth_slope',.04,0.),('localized_sr_step',.04,.12)]:
        cs=cu+slope*(cm-3500)/800
        es=eu+slope*(em-3500)/800+step*((em >= SR_MIN)&(em < SR_MAX))
        result=calibrated_closure(dict(mass=cm,score=cs,truth=np.zeros(len(cm),dtype=int)),em,np.zeros(len(em),dtype=int),es,.05,replicates,seed+1)
        cases.append(dict(case=name,**result))
    return dict(analysis='SYNTHETIC BACKGROUND-ONLY SOFTWARE CONTROLS; not physics results',seed=seed,replicates=replicates,
                cases=cases,limitation='Checks this interpolation/resampling implementation under declared toy distributions; does not certify physical background-model coverage')

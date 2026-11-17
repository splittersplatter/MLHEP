"""Descriptive sideband-only linear efficiency interpolation, no bump significance.

Weighted least squares fits k/n on bin centers using sidebands only. Jeffreys
binomial variances prevent zero errors. This is a diagnostic model, not a
validated discovery background fit. Conditional calibration uncertainty must be
assessed separately before physics claims.
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

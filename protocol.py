"""Immutable scientific constants for protocol v2; changes require a new run."""
import numpy as np

MASS_MIN, MASS_MAX = 3100., 3900.
SR_MIN, SR_MAX = 3300., 3700.
BINS = np.linspace(MASS_MIN, MASS_MAX, 17)
BINS.setflags(write=False)
CUTS = (.2, .05)
BASELINE = 'mass_mlp'
DISCO = 'sideband_0.003'
FINAL_MODELS = (BASELINE, DISCO, 'efp_mlp')
PRESCRIBED_LAMBDA = .003
SIC_TARGET, JS_TARGET = .90, .80

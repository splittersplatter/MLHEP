> Historical protocol-v1 documentation. These commands and numerical claims describe earlier runs, not the maintained protocol. Use [README.md](README.md) for protocol v2. Previously evaluated rows 200000–299999 are exploratory.

# Phase 1: a defensible CWoLa mass-sculpting baseline

Run from the PyCharm terminal with the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe phase1.py
```

Defaults: first 50,000 raw rows, 25 epochs, CPU or CUDA automatically. Output:
`results_phase1/mass_sculpting.png`, `metrics.json`, `training.csv`, `model.pt`,
`predictions.npz`, and `config.json`. Feature extraction is cached under
`feature_cache`; rerunning training with a different seed reuses it. The dataset
must remain at `events_anomalydetection.h5`, or pass `--data PATH`.

Quick software check (not a physics result):

```powershell
.\.venv\Scripts\python.exe phase1.py --events 2000 --epochs 2 --output results_phase1_smoke
```

## What changed and why

The previous script is saved as `phase1_supervised_legacy.py`. It trained against
truth labels and calculated all-particle event mass. That is not CWoLa hunting.
The new script reconstructs anti-kt R=1 jets, takes the two highest-pt jets,
requires leading pt >1200 GeV, and orders them by increasing jet mass.
The resonant variable is mJJ = sqrt((pJ1+pJ2)^2).
It uses SR [3300,3700) GeV and adjacent SB [3100,3300) and [3700,3900) GeV.
These choices follow the paper's preprocessing and window definitions:
https://arxiv.org/html/2604.20965v1 (sections III.3 and III.4).

Features are normalized hadronic EFPs (beta=1, degree <=4), computed separately
on each jet's constituents. There are 36 per jet before removing constants.
The dijet mass is NEVER an input feature. This is a reduced EFP-only study,
not reproduction of the paper's full 1,034-feature Kitchen Sink or its GBDTs.
No subjettiness or explicit jet-mass inputs are included.

For the critical zero-signal control, truth labels filter out simulated signal.
All neural-network targets instead encode SR=1 and SB=0. Thus truth is used to
construct a controlled simulated sample, not as supervised training labels.
Splits are disjoint 60/20/20 train/validation/test, stratified by SR/SB.
Log-EFP normalization and constant-column removal use training events only.
Balanced BCE class weights use training counts. Validation loss selects the
checkpoint; validation score quantiles fix cuts before looking at the test set.
The MLP remains 128 -> 64 -> 32 -> 1, with logits and stable BCEWithLogitsLoss.

## Interpreting the output

The top plot compares normalized background mass shapes. The lower plot shows
selection efficiency against mass, with pointwise 68% Wilson binomial intervals.
A mass-independent score should have approximately flat selection efficiency.
Intervals condition on the fitted model and fixed threshold; they do not include
training variation or validation-quantile uncertainty. Empty mass bins are omitted.
The metrics include SR/SB efficiencies and SR-versus-SB AUC, NOT signal AUC.
AUC alone does not diagnose all dependence. This pipeline measures sculpting;
it does not guarantee a central bump, calculate a discovery significance, or
establish independence from a finite histogram. Small smoke tests are noisy.

Before a project conclusion, increase statistics and repeat independent seeds.
The current prefix subset is for development; check dataset ordering/sample
representativeness before final results. Do not pick a seed because it looks bumpy.
The saved row IDs let you audit event overlap. Cached features are keyed by source
path, size, modification time, degree, row count and preprocessing version.

## Phase 2 boundary

This script intentionally establishes Phase 1. DisCo should penalize dependence
between the classifier SCORE and mJJ, not merely individual feature correlations.
For a pure-background control, full-window score/mass DisCo is well defined, but
it competes directly with predicting SR/SB: a constant classifier is a trivial
solution. A null-test improvement alone cannot establish useful anomaly detection.
Next compare lambda=0 against a lambda scan on identical splits, then test small
signal injections and held-out SR signal efficiency/SIC. For mixed samples,
background-only truth-masked DisCo is an oracle simulation benchmark; sideband-only
DisCo is data-driven but does not guarantee SR decorrelation. Specify that choice
explicitly. Do not claim the original paper is invalid or a universal flaw is fixed.

Dependencies: numpy, h5py, hdf5plugin, fastjet, energyflow, torch, scikit-learn,
matplotlib. All are installed in this project's .venv. Processing is sequential,
with bounded HDF5 chunks and no worker processes. Model files contain NumPy
preprocessing arrays: load only your own checkpoints (PyTorch may require
weights_only=False for these trusted files).

Regenerate the corrected efficiency plot without retraining: `.\.venv\Scripts\python.exe phase1.py --plot-only`. Points show observed k/n; asymmetric error bars give Wilson intervals.

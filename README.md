# MLHEP: maintained CWoLa / DisCo protocol v2

The root Python modules are the maintained implementation. `results_linux/`,
`results_linux_new/`, `reproducible_demo/`, existing PDFs and their manifests are
historical evidence. Do not run a historical launch script to start a corrected
experiment. Historical source hashes are in `docs/legacy_source_inventory.json`.
Missing original training source remains unavailable; snapshots are not retroactive
proof of lineage. `CRITIQUE_REVIEW.md` records the scientific assessment;
`IMPLEMENTATION_STATUS.md` maps the implementation and remaining limitations.

## Environment and GPU training

Use a project-local Python 3.12 environment with `requirements-study.txt`.
The maintained launcher accepts `PYTHON_BIN`; it contains no path into another
project and does not install or upgrade dependencies. Actual versions, device,
CUDA runtime, GPU name, thread count and deterministic settings are recorded in
every run. The lock file records the environment used during this implementation;
CUDA wheels/drivers must be compatible with the host.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-study.txt
PYTHON_BIN=.venv/bin/python ./run_fixed_lambda.sh --output results_v2_development --telemetry
```

MLPs train on **CUDA by default**; unavailable CUDA is an error. `--device cpu`
is an explicit verification option, and `--device auto` explicitly permits
fallback. The histogram GBDT comparator uses scikit-learn on CPU. FastJet/EFP
extraction is CPU work. CPU reload and inference remain supported. Deterministic
CUDA training uses `CUBLAS_WORKSPACE_CONFIG=:4096:8`; no bitwise CPU/GPU or
cross-version identity is promised. If a sandbox hides the GPU, launch training
from the host environment rather than silently switching the scientific run to
CPU.

Default development settings: first 200,000 raw rows, ungroomed anti-kT R=1,
degree <=4, 15 epochs, seeds 11/22/33, fractions 0/.005/.01, prescribed DisCo
lambda .003, sideband and labeled oracle comparisons, EFP-only, mass MLP and
mass GBDT. Changing these creates a different run identity. Lambda .003 was
informed by earlier inspected results; this is not a retrospective preselection.

## Separation and content provenance

Stratify selected events by truth and lower SB / SR / upper SB. Fix 20% each for
calibration, tuning evaluation and reserved final evaluation across seeds.
Within the remaining 40%, each seed allocates 75% training and 25% checkpoint
validation. Exact raw IDs and counts are persisted. Training/checkpoint mixtures
use nested SR-only injections and matched pools across methods. Preprocessing
fits injected training rows only. Calibration/tuning pools retain their declared
truth/region strata; calibration cuts use all sideband calibration events and
no truth mask. Background/signal performance is evaluated using truth in the
simulation, with signal efficiency defined only inside SR. This convention is
explicit and differs from historical overlapping validation.

All roles are disjoint. Thresholds use calibration-only linear quantiles with
strict `score > cut`. Calibration acceptance is an in-sample diagnostic. Tuning
results inform choices and are exploratory. The default study never scores the
reserved final role. Reserving events now does not erase prior exposure.

Dataset SHA-256, requested/actual raw range, HDF5 schema, feature width,
extraction source and dependency versions identify schema-v3 caches. Old caches
are never silently trusted. New extraction also saves constituent counts,
jet pT/eta/rapidity and angular separations. No four-constituent veto, grooming,
angular cut or altered transform is introduced. Such physics changes require
new declared protocols, caches and comparisons.

Run identity includes configuration, source, cache, splits and environment.
Each fit has checksummed model, preprocessing, predictions, thresholds,
metrics, floor/tail diagnostics and optional gradient telemetry. `complete.json`
is written last. Changed/corrupt complete artifacts are rejected. Incomplete
fits are retrained under the same identity; their old contents are quarantined.
Exclusive writer locks prevent concurrent publishers. A stale lock after a
process is killed needs manual PID inspection before removal. Never delete a
lock belonging to a running process.

## Audited reporting and fixed-lambda assessment

```bash
.venv/bin/python report_study.py --run results_v2_development --output output/v2_development_report
.venv/bin/python tune_disco.py --run results_v2_development --output output/v2_fixed_lambda.json
.venv/bin/python audit_predictions.py --run results_v2_development --output output/v2_independent_arithmetic.json
```

Reports verify source snapshots, every expected fit, exact sample membership,
calibration cuts and numerical metrics before writing output. An independent
NumPy/rank-sum calculation checks thresholds, AUC, efficiencies, SIC, JS,
permutations, dCor and local counts. All seeds/cuts appear; incomplete or
undefined comparisons cannot qualify. Fixed-lambda assessment references the
existing study fits with zero new fits and reports leave-one-seed-out influence.
Figures and HTML are published together via an atomic directory rename, with
input hashes and an artifact marker. Output must be separate from source runs.
No Windows font paths or hard-coded empirical success claims remain.

Undefined metrics are JSON null with reasons/counts. Empty selected-background
JS also has an undefined permutation p-value. Base-2 JS, linear quantiles,
16 mass bins and strict selection are versioned. Local region efficiencies,
Wilson intervals, actual/nominal acceptance, score spread and fixed-count JS
references accompany every seed/cut/model. Sideband-only linear closure is
**descriptive**, conditional on the threshold, and does not include calibration
uncertainty. It cannot establish removal of fake resonances or a discovery
significance. Such claims require a separately validated fit and controls.

## Freeze and reserved-event evaluation

```bash
.venv/bin/python confirm_disco.py freeze --run results_v2_development --output output/v2_freeze.json
.venv/bin/python confirm_disco.py evaluate --run results_v2_development --freeze output/v2_freeze.json --output results_v2_replay
.venv/bin/python report_study.py --run results_v2_development --evaluation results_v2_replay --output output/v2_replay_report
.venv/bin/python audit_predictions.py --run results_v2_development --replay results_v2_replay --output output/v2_replay_arithmetic.json
```

Freeze is a separate command before candidate final scoring. It pins exact
reserved rows, source, models, preprocessing, cuts, comparisons and bootstrap
rules. Evaluation reserves access in the dataset ledger before loading features;
interrupted attempts remain exposed. Paired stratified resampling shares the
same event draws across all models/seeds; ratios divide seed means. Any undefined
replicate makes that metric's interval unavailable, with the invalid count saved.
Intervals condition on models, cuts and event class/region counts.

Real-data ledgers are keyed by full dataset hash under `exposure_ledger/`, shared
across output directories. History defaults to **unknown**, including beyond
raw row 300,000. `--independent --ledger <recorded-dataset-ledger>` refuses unknown
or reused samples. It cannot use an alternate empty ledger to evade earlier
access. An externally established source audit covering exact unused rows is
required; this code cannot certify unknown past use. Previously evaluated rows
200000–299999 are always historical exploratory evidence. An event-disjoint
simulation check would still not validate independent generator systematics.

## Software validation

```bash
.venv/bin/python -m unittest test_study test_sculpting_diagnostics test_integrity test_extraction test_pipeline -v
.venv/bin/python -O -m unittest test_integrity test_extraction -v
.venv/bin/python study.py --smoke --device cuda --epochs 2 --seeds 11 22 --fractions .005 .01 --permutations 19 --telemetry --output /tmp/MLHEP_SYNTHETIC_smoke
.venv/bin/python report_study.py --run /tmp/MLHEP_SYNTHETIC_smoke --diagnostic --output /tmp/MLHEP_SYNTHETIC_report
```

Synthetic fixtures are explicitly labeled and barred from scientific reporting.
Tests cover role overlap, fixed membership, stale provenance/resume, same-size
same-mtime dataset edits, corrupt caches/models, incomplete fits/grids, writer
locks, strict JSON, ties, missing classes, empty bins, undefined bootstrap,
sideband-only interpolation and telemetry invariance. The CUDA test runs when
host GPU access is available. See the implementation status for actual executed
checks and limitations rather than inferring physics success from software tests.

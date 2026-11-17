# MLHEP integrity and reproducibility fix plan

Prepared 18 November 2026; extended after the Collider ML Critique review.
Status: training/provenance fixes remain planned. Saved-prediction sculpting
diagnostics and their tests are implemented separately; no new experiment has
been run. See `CRITIQUE_REVIEW.md` for the claim-by-claim assessment.

## Evidence and boundaries

The primary target reviewed is `results_linux_new/`, with saved outputs in
`results_linux_new/previous_run_20261118/`. Root-level scripts, `results_linux/`,
and `reproducible_demo/` are additional implementations or historical bundles.
The supplied audit is the review specification; this is a current-source review,
not a commit-diff review.

Independent review script: `output/review/recheck_saved_predictions.py`.
Machine-readable evidence: `output/review/recheck_saved_predictions.json`.
It imports no training implementation, changes no experiment artifacts, and
checks saved predictions using its own metric calculation:

- 75 study prediction archives, 300 metric records: signal efficiency,
  background efficiency, SIC and base-2 JS match at rtol=1e-10, atol=1e-12.
- 18 confirmation models: the same four metrics match.
- All checkpoint-validation rows occur in evaluation validation in every
  study fit; overlap is 10,806–10,859 rows per fit.
- Six matching fixed-lambda validation pairs have identical rows and scores.
- All 54 frozen artifact hashes listed in the archived confirmation protocol
  match. Current `study.py` and `confirm_disco.py` hashes differ from that
  protocol; current `phase1.py` matches. This does not recover missing old source.
- The existing Linux `test_study` suite passes all three tests. It does not
  test checkpoint/evaluation separation, provenance, or report correctness.
- A separate synthetic diagnostic of the current `metrics_for` with constant
  zero scores and threshold 1 produced undefined JS and `js_permutation_p=0.005`
  for 199 permutations. This is a software failure example, not physics data.

These checks find no numerical discrepancy in the quantities checked. They do
not certify the dataset, training history, independence, bootstrap coverage,
all saved metrics, or scientific conclusions. Existing `output/review/*audit.json`
files were inspected as historical evidence, not treated as fresh verification.
Bootstrap intervals were not regenerated during this review. No training or new
confirmation inference was run.

## Required integrity rules

1. Every reported empirical number must trace to a named run, verified inputs,
   saved predictions and a versioned calculation. Missing evidence means
   unavailable/unverified, never a plausible replacement or zero.
2. Preserve old outputs and their manifests. A corrected run gets a new protocol
   version and output directory. Never replace old hashes to make old outputs
   appear to come from current code.
3. Report unsuccessful targets, undefined metrics, exclusions and incomplete
   runs explicitly. Never hide failing seeds or change thresholds after seeing
   evaluation results to obtain a desired claim.
4. Recalculation/replay verifies arithmetic or inference. It does not create an
   independent confirmation. A recorded hash proves identity, not correctness.

## Implementation order and acceptance checks

### 1. Establish one maintained implementation and preserve history (P0)

Files: `.gitignore`, root entry points, `results_linux_new/*.py`, launcher and
historical bundles.

Inventory code, outputs, dataset identities, run lineage and known exposures
before moving anything. Use the root-level implementation as the maintained
entry point after porting reviewed Linux device support; keep archived bundles
explicitly historical. Compare implementations before consolidation so CPU/GPU,
fixed-lambda, preprocessing and inference behavior are not accidentally lost.
Treat missing historical source as unavailable, rather than reconstructing it
and claiming an exact historical replay.

Narrow `.gitignore`: it currently excludes `results*/`, all Markdown, all HTML
and manifests. Track maintained source, protocol documentation, this plan and
dependency definitions; keep large data/models and generated results excluded.
Do not broadly unignore datasets or generated outputs. Remove the launcher's
dependency on another project's absolute Python path; accept a configured
interpreter and validate the actual environment. Retain a CPU verification path.

Acceptance: clean-checkout entry points are unambiguous; required source/docs
are visible to Git; historical artifacts retain their identities; imports resolve
to the maintained modules rather than a sibling copy.

### 2. Replace overlapping sample roles (P0)

Files: `study.py:41`, `study.py:240`, `tune_disco.py:82`, `report_study.py:141`
in the Linux copy, then equivalent maintained entry points.

Create a shared, persisted split manifest keyed by dataset content identity and
original raw row ID. Define five disjoint roles:

| Role | Permitted use |
|---|---|
| Training | Fit preprocessing and model weights; construct injected mixtures |
| Checkpoint validation | Early stopping and checkpoint objective only |
| Calibration | Estimate sideband score cuts; separate labeled SR calibration for the explicitly labeled simulation benchmark |
| Tuning evaluation | Compare fixed models/strengths using frozen calibration cuts |
| Final evaluation | Evaluate only after the complete analysis choice is frozen |

Fix the calibration, tuning and final membership across seeds and model
families. Seeds may vary the training/checkpoint split within their allocated
development pool. Fix proportions before running; record counts per truth/region
stratum and verify sufficient SR signal for every requested injection. Fail with
counts if infeasible rather than borrowing from other roles. Keep preprocessing
fit on injected training rows only. Preserve nested injection subsets and matched
baseline/DisCo pools.

Use calibration-only thresholds for tuning and final metrics. Calibration
acceptance itself is an in-sample diagnostic, not an independent performance
estimate. Tuning-evaluation results become selection data when used to choose
settings. The existing test sample is already inspected and must be labeled
exploratory in future comparisons.

Remove the implicit `eligible < 1000` fallback in production; require an explicit
synthetic/smoke mode, saved as such and barred from scientific reporting.
Save raw row IDs for every role, including checkpoint and calibration. Validate
pairwise disjointness at allocation, training, evaluation and report loading with
explicit exceptions, not Python `assert` statements that vanish under `-O`.

Acceptance: all role pairs are disjoint; changing a seed never moves a fixed
evaluation event into training/checkpointing; all model comparisons share the
declared pools; no evaluation or final scores affect fitting/calibration; negative
tests with deliberate overlap fail before training. Changing calibration scores
changes cuts; changing tuning/final scores cannot change cuts or checkpoints.

### 3. Make confirmation history and freezing explicit (P0)

Files: `confirm_disco.py:1,57,70,98,138,143`, launch scripts and report templates.

Separate replay from independent evaluation. Label rows 200000–299999 as a
previously evaluated sample in headings, plot titles, tables and conclusions,
including regenerated reports. Preserve the historical originals and attach
corrections in new report outputs rather than silently rewriting the archive.

Add an exposure ledger keyed by dataset hash and raw ranges/row IDs, recording
training, checkpointing, calibration, exploratory evaluation, final evaluation and
known manual inspection. Inventory all available runs/bundles; unknown exposure
history remains unknown. Hash changes or copied paths must not erase a known
dataset's history. Do not assume rows >=300000 are unused just because this
script does not reference them. Unused rows in the same simulation file may
provide an event-disjoint check, not an independent generator/systematics study.

Freeze a protocol artifact before opening candidate final predictions: source
snapshot, data identity, exact sample reservation, lambda=0.003, model identities,
preprocessing, thresholds, comparisons, mass bins, targets, bootstrap scheme and
seeds, treatment of undefined replicates and reporting rules. Record honestly that
0.003 is informed by earlier results; freezing now cannot prove preselection in
the past. Validate actual model metadata/strength, not just directory names or
the count `completed_fits == 6`.

Production evaluation verifies freeze identity and exposure eligibility before
scoring; reserve the sample before access and record interrupted attempts too.
Use a new sample only when history supports that claim. Otherwise provide an
explicit replay/exploratory mode and refuse the independent-confirmation label.
Keep final evaluation a separate command, outside the default training launcher.

Acceptance: reused/unknown samples cannot enter independent mode; changed
lambda, model, preprocessing or threshold invalidates a frozen protocol;
interruption cannot reset exposure; replay reports contain no unsupported
"fresh"/"independent" claims. Software checks do not certify unknown past use.

### 4. Enforce content provenance and safe resume/cache loading (P0)

Files: `phase1.py:29`, `study.py:221,245`, `tune_disco.py:43,70`, confirmation
artifact loading; proposed shared `provenance.py` and `artifacts.py`.

Stream a full SHA-256 of the input dataset. Record requested and actual raw
ranges, HDF5 schema, feature definition, extraction-source hash and relevant
dependency versions. Path, size and mtime are descriptive fields, not identity.
Hash once per invocation rather than reading the 2.7 GB file per extraction
block; detect source changes during hashing/extraction and require a stable
snapshot when concurrent writes are possible.

Resolve caches by this manifest, never by the first matching event count/degree.
Validate embedded metadata, array lengths/shapes/dtypes, finite values, truth
labels, mass range, unique row IDs and row bounds; verify the cache content hash
referenced by the run. Legacy caches without content provenance need rebuilding
or a separately documented source-backed validation, not automatic blessing.

Run identity includes full execution configuration, data/cache/split hashes,
explicit source snapshot hashes (including extraction, architecture and shared
helpers), dependency versions and CPU/GPU/determinism settings. Record Git commit
and dirty state as supplementary provenance. Retain the source snapshot itself.

Write per-fit manifests linking weights, preprocessing, predictions and metrics
with checksums and run identity. A completion marker is written last via atomic
rename after all outputs validate. A `metrics.json` file alone is insufficient.
Resume rejects changed identities, malformed JSON, missing/corrupt artifacts and
incomplete fits; incomplete fits are retrained under the same protocol in a
temporary location. Prevent simultaneous writers to the same run. Epoch-level
resume is unnecessary initially; fit-level resume is sufficient.

Acceptance: reject source edits with identical CLI options; reject dataset edits
preserving file size/mtime, wrong-dataset caches, altered splits and modified
models/predictions. Interrupted writes never count as complete. Relocation may
retain verified content identity; it must not silently redirect input lineage.

### 5. Handle undefined metrics and qualification honestly (P0)

Files: `study.py:143–175`, `tune_disco.py:16–40`, `confirm_disco.py:40` and
bootstrap/plotting/report aggregation paths.

Centralize metric definitions and version them. Preserve strict `score > cut`,
quantile convention, base-2 JS and exact bin edges in protocol metadata. Validate
aligned finite scores and event arrays. Return JSON `null` plus a reason and
supporting counts for undefined metrics: zero selected background, missing signal
or background, single-class AUC, zero ratio denominators, empty mass bins, or
insufficient correlation samples. A constant variable and an insufficient sample
are distinct cases. Undefined JS must produce an undefined permutation p-value;
it must never produce the minimum p-value through comparisons against NaN.

Use strict JSON (`allow_nan=False`) everywhere. Qualification requires the exact
expected seed/fraction/model keys, unique records and valid metrics, not just
matching counts. Keep fixed-lambda assessment separate from search/selection;
the current `selected_lambda` field is misleading for a single prescribed value.
No success verdict is produced for incomplete or undefined comparisons.

Predefine minimum statistical support before new evaluation. Count invalid
bootstrap replicates and follow the frozen rule for unavailable intervals;
never silently drop them or manufacture finite intervals. Preserve paired event
resampling across methods/seeds, ratios of means, and the stated conditional
interpretation. Report exact aggregation denominators and missing fits/metrics;
do not silently average only favorable or defined seeds.

Acceptance: all-none/all-selected cuts, tied scores, missing classes, zero
baseline JS/SIC, empty bins and incomplete candidate grids produce specified
statuses without false p-values or success labels. Strict serialization succeeds
for valid and undefined cases. Existing valid saved metrics remain numerically
consistent under the declared historical calculation version.

### 6. Generate every report from verified run artifacts (P0)

Files: `report/build_report.py:69` and throughout narrative/tables,
`report/make_figures.py`, `report_study.py`, confirmation/tuning report writers.

Require explicit run/protocol inputs. Verify their lineage before creating any
output. Derive tables, counts, chosen parameters, intervals and conditional
success/failure prose from those inputs. Distinguish old demonstration, Linux
experiment and replay by run ID in the report and figure manifest. Remove
hard-coded empirical claims from abstract, results, captions and conclusion;
retain legitimate declared protocol constants and descriptive text.

Replace root-level result assumptions with explicit resolved artifact paths;
use available portable fonts instead of `C:/Windows/Fonts`. Verify all inputs
before plotting, render in a temporary directory and publish the completed report
bundle atomically. Run split/provenance audits before writing the report, rather
than after its headline conclusions. A failed audit produces diagnostics, not a
scientific success report. Label intentional partial-run diagnostics clearly.

Acceptance: changing a synthetic input metric updates its table, plot and related
prose consistently; a failing fixture cannot yield success language; absent runs,
unverified hashes and mixed run IDs fail clearly. Every empirical claim maps to
a run field/calculation. Visually inspect generated PDF/HTML and all plots.
Synthetic report fixtures must be visibly labeled and kept outside real results.

### 7. Reuse identical fits without overstating evidence (P1)

Files: fixed-lambda study/tuning orchestration and result metadata.

Use the main study's sideband-0.003 fit when data, splits, preprocessing, seeds,
training settings, source and environment identity match exactly. Record a
reference to that fit and its validation predictions instead of training it again.
Compare against the intentional lambda=0 baseline. If identity does not match,
do not reuse automatically. Call identical retraining a reproducibility check,
never an additional independent confirmation; retain paired model comparisons.

Acceptance: a matching fit is evaluated/referenced once; one changed dependency
or scientific input blocks reuse; counts distinguish unique fits from evaluations
and replay/reproducibility attempts.

### 8. Measure local closure alongside global distortion (P0; PDF critique)

Implemented now: `sculpting_diagnostics.py` reads saved confirmation predictions
and produces per-seed lower-SB, SR, upper-SB and combined-SB counts/efficiencies,
conditional Wilson intervals, SR/measured-SB ratios, binned selection curves,
score spread, and conditional JS null distributions. Reports were generated for
both the older demo and archived Linux run in `output/review/critique_*`.
These are retrospective diagnostics on inspected data, not new confirmations.

Still required in the maintained production reporting pipeline: include these
diagnostics for every declared cut/fraction/model, preserve per-seed results,
and link their input hashes. Separate nominal calibration acceptance from
achieved evaluation acceptance. Do not interpret an SR/SB ratio as a bump
significance: unequal sideband populations and a smooth slope can change it.
Predefine a sideband-only background/efficiency interpolation and closure
procedure, with fit/calibration uncertainty and background-only controls, before
any future claim about a localized excess or fake-resonance removal. Do not add
an SR truth penalty to the practical sideband method; keep it an oracle benchmark.

Acceptance: a synthetic localized SR step is visible even if global JS improves;
an imposed smooth slope is distinguished from a localized residual by a specified
closure model; interpolation never sees selected SR outcomes while fitting;
failed local closure prevents a claim of removed mass sculpting. Any bump
significance requires a separately validated background-fit analysis.

### 9. Quantify loss dynamics and training-seed stability (P1; PDF critique)

Implemented now: saved training histories are checked against BCE + lambda*DisCo,
and checkpoint choices under the recorded objective are compared with BCE-only
choices on the same observed epochs. This is not counterfactual retraining.

Still required: add opt-in diagnostic logging of BCE and weighted penalty
gradient norms/cosine in parameter space, score standard deviation/quantiles,
class counts, train/validation components and checkpoint reason. Measure using
the same fixed diagnostic batch without optimizer steps, RNG drift, or changed
training order; test that enabling logging preserves training behavior under
the same deterministic setup. Label measurements as diagnostics, never infer
gradient magnitudes from scalar losses alone. Validate the effect of the
denominator floor near constant scores.

Keep the current fixed-lambda comparison explicit. For any later search, freeze
the full candidate grid, independent tuning sample, seed budget and aggregate
criterion before running. Report every seed and failure; assess paired differences
and leave-one-seed-out influence descriptively on existing fits. Additional seeds
are a future development experiment, not new independent event samples. Never
select seed subsets after viewing final results. Preserve EFP-only in every main
comparison; near-zero SIC is not proof of constant output without score checks.

Acceptance: gradient telemetry is measured, finite or explicitly unavailable;
its collection does not change fitted outputs; all configured seeds appear in
reports; a favorable mean driven by one seed is visible. No report claims the
small lambda has a known causal mechanism without supporting diagnostics.

### 10. Audit feature floors and version physics ablations (P0/P1; PDF critique)

Implemented now: `output/review/critique_feature_check.py` records per-feature
floor counts and the standardized range of clipped values on one saved training
split, plus a labeled synthetic counterexample to the claimed degree/prong rule.
Only 11 of 69,698 cached events have any floored EFP values, but the inspected
training split includes a clipped value around -30.2 standardized units. The
rare outliers warrant investigation; a causal effect on the model is unverified.

Still required: report train-only feature floor rates, tails and constant removal
in preprocessing artifacts. Preserve constituent multiplicities and jet pT,
rapidity/eta and angular separation in a new extraction/cache schema; they are
not present in the current cache and must not be invented. Investigate raw
floor events and preprocessing robustness using development data only. Compare
predeclared alternative transforms/floors or train-fit robust scaling as new
protocols; do not delete events or choose a transform from final performance.

Treat grooming, a dijet angular-separation cut, and dimensionless mass inputs
as separately justified physics ablations, not automatic repairs to historical
data. Specify grooming algorithm/settings, which masses/EFPs are recomputed,
which dijet mass defines regions, and whether the angular cut uses eta or rapidity.
Each change needs a new cache identity, split manifest, training, calibration and
report, with changed sample counts. Keep the documented ungroomed R=1 baseline.
The LHCO record specifies no pileup/MPI; do not attribute observed effects to
those absent simulation components. A grooming change does not guarantee
decorrelation. The cited Kitchen Sink preprocessing does not require the
critique's proposed grooming or angular cut.

Acceptance: preprocessing diagnostics identify the observed rare outliers;
synthetic sparse jets do not trigger an erroneous four-constituent veto; physics
variants cannot load incompatible caches or models; no physics variant is called
an improvement until a declared comparison has actually been run.

## Verification and release sequence

1. Preserve the current archive and establish source/data/exposure inventory.
2. Add failing regression tests for overlap, provenance rejection, no-selection
   p-value and stale-report behavior, then implement steps 1–6. Use explicitly
   labeled synthetic fixtures for these software tests, never substitute data in
   scientific output folders.
3. Keep independent saved-prediction recalculation as a separate check; do not
   make all tests call the same production metric helper. Extend the auditor to
   thresholds, AUC, dCor, permutation values, bootstrap ratios/intervals, exact
   membership and all published aggregates with recorded tolerances.
4. Test unchanged resume versus changed-code/data/split resume, corrupted caches,
   missing artifacts, interruption and simultaneous-writer rejection. Ensure
   critical validations also work with Python optimization enabled.
5. Run a small, clearly marked smoke experiment covering each model family,
   checkpointing, calibration, resume, model reload and full report generation.
   Explicitly use development/synthetic data; no candidate final sample is read.
   Verify CPU inference; test CUDA training/reload when supported. Record actual
   environment and reproducibility tolerances; do not promise bitwise equality
   across CPU/GPU or dependency changes without evidence.
6. Only then run the corrected development protocol in a new output directory.
   Keep lambda=0.003 for prescribed DisCo and lambda=0 for comparisons; report
   every configured model, fraction, seed and target outcome.
7. Freeze the complete analysis and reserve an evidence-backed unused sample
   before any independent final evaluation. If no such sample is established,
   stop the independent-confirmation stage and deliver exploratory/replay results
   with that limitation. No amount of code repair restores independence to data
   already used to inform analysis choices.

Completion means the regression checks, artifact audit and report audit pass,
the run can be reproduced under its recorded environment, and all remaining
scientific limitations are disclosed. It does not require favorable physics
results. No code or plan can guarantee correctness merely by recording hashes
or making tests pass.

## Commands actually used in this review

The available Linux environment required the existing interpreter and local
dependency directory; no packages were installed or upgraded.

```bash
# From results_linux_new/: all 3 existing tests passed.
PYTHONPATH=/home/dhruv/Desktop/projects/MLHEP/.linux-deps \
  /home/dhruv/Desktop/projects/NNQS-AIMD/.venv/bin/python -m unittest test_study -v

# From project root: read archived predictions; write only review evidence.
PYTHONPATH=.linux-deps \
  /home/dhruv/Desktop/projects/NNQS-AIMD/.venv/bin/python \
  output/review/recheck_saved_predictions.py
```

The launcher environment itself is a portability issue to fix, not the proposed
long-term setup. The first review changed only this plan and its review evidence.
The subsequent PDF review adds a diagnostic utility, five synthetic tests,
claim-by-claim assessment, and targeted Git-ignore exceptions for the two review
documents. The training implementation and historical results remain unchanged.


## Protocol-v2 implementation follow-up (18 November 2026)

The original review above remains a record of the state at review time.
Root training, provenance, splitting, metrics, assessment and reporting have since
been replaced by protocol v2. See `IMPLEMENTATION_STATUS.md` for the requirement
mapping, executed checks and remaining scientific limits, and `README.md` for
maintained commands. Historical experiment outputs have not been rewritten.

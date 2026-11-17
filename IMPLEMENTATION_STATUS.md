# Protocol-v2 implementation status

18 November 2026. Implements the integrity repairs in FIX_PLAN.md without
rewriting historical experiment artifacts or claiming independent confirmation.

| Plan area | Implementation / evidence |
|---|---|
| Maintained entry point | Root study.py and main.py; phase1.py control delegates to the same pipeline. Linux CUDA support ported; GPU is default. Root launcher uses configurable Python. Historical bundles remain untouched. |
| Five roles | splits.py persists exact raw IDs, stratifies truth and three regions, fixes calibration/tuning/final across seeds, rejects overlap and small production pools. All injections preflight before fitting. |
| Freeze / exposure | exposure.py and confirm_disco.py separate freeze/replay/independent eligibility. Unknown history defaults to refusal. Shared per-dataset ledgers preserve interrupted attempts. Prior 200000:300000 use is explicitly disclosed. |
| Cache / resume | provenance.py and artifacts.py stream dataset SHA-256, verify schema-v3 cache content, retain source snapshots, checksum fit outputs and completion markers; exclusive locks and temporary publication. Incomplete fits retrain; changed complete artifacts reject. |
| Metrics / qualification | metrics.py defines strict cuts, linear calibration quantiles and base-2 JS; null/reasons/counts for undefined values. Exact grids and leave-one-seed-out comparisons in tune_disco.py. bootstrap.py retains invalid replicate counts and makes affected intervals unavailable. |
| Reports | report_study.py audits first, derives HTML/plots from explicit run, atomically publishes hashes/figure manifest. Historical hard-coded source generators replaced; original generated PDFs remain historical. |
| Reuse | Prescribed .003 assessment reuses verified study fits, reports zero new fits, references input hash. No duplicate tuning training. |
| Local closure | Every cut/model/seed has regions, Wilson intervals, fixed-count JS reference, score spread and sideband-only linear interpolation. Synthetic slope/step check verifies fitting never sees SR selections. |
| Optimization / seed effects | Opt-in parameter-gradient norms/cosine, score quantiles, counts and checkpoint reason on fixed training batch, with no optimizer step/RNG change. Exact CPU/GPU invariance regression and seed omission influence. |
| Feature tails / physics | Training-only floor rates, constant removal and standardized tails saved per fit. Schema-v3 extraction stores multiplicities and kinematics. Sparse two-constituent jets remain valid. Physics variants need new protocols; no unsupported grooming/cut/UE explanation is imposed. |

Independent `audit_predictions.py` recalculates corrected-run cuts, AUC,
efficiencies, SIC, JS, permutation p-values, dCor, histograms and replay paired
bootstrap ratios/intervals. It does not import production metric/training helpers.
Historical `output/review/recheck_saved_predictions.py` was rerun separately:
300 study records and 18 confirmation models remain arithmetically consistent;
legacy checkpoint/evaluation overlap remains present and explicitly unrepaired.

## Scientific limits intentionally retained

No existing inspected sample has become independent through these repairs.
Source hashes cannot certify simulation correctness or recover unknown historical
source. Existing historical reports/outputs retain their identities. Known source
lineage and exposure limits are recorded in the review and source inventory.

The linear closure diagnostic conditions on the calibrated threshold; calibration
uncertainty, background-only closure ensembles and a validated discovery
background fit remain future analysis. Accordingly no maintained report claims
removal of mass sculpting, zero dependence, a bump significance, or a known causal
mechanism for the small penalty. Grooming, dimensionless masses, angular cuts and
transform robustness experiments are future predeclared development protocols,
not automatic preprocessing repairs. Additional seeds do not create independent
event samples.

The canonical 200k-row, 15-epoch development experiment is a separate recorded
scientific run; software smoke checks are not a replacement for that experiment.
No independent final evaluation is permitted until unused-sample history is
established. See the execution evidence appended below for actual completed runs.

## Completed execution evidence

All 27 host regression checks pass, including CUDA training and CPU reload.
Optimized-Python integrity/extraction checks: 13 pass, 1 sandbox CUDA skip; the
CUDA case passes separately on the host. The final limited real-data run
`results_v2_development_10k_final/` completed 45 unique fits with CUDA MLPs and
the CPU GBDT comparator. Its 180 metric records pass production and independent
arithmetic audits. Run ID: `bdc6552f4f1c04e643e188456bf34c1badeff280dc3704cc2d19142f092b1527`.

The generated report is `output/v2_development_10k_final_report/REPORT.html`.
Not all tuning targets are established; undefined ratios and failing targets are
retained. A real protocol freeze was produced without scoring final-role rows.
Synthetic tests exercised freeze/replay, independent bootstrap audit and reporting.
See `docs/VALIDATION.md`, `docs/development_validation.json` and
`docs/CODE_REVIEW.md`. Historical bundles remain unchanged.

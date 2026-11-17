# Final figure-and-table inventory

This is the report blueprint, not a drafted report. Final figures should be regenerated
with common fonts, model colours, GeV units and explicit uncertainty definitions.

| Item | Main message | Data/source | Status |
|---|---|---|---|
| Figure 1: workflow and information boundaries | Particles -> jets -> features -> CWoLa -> lambda selection -> frozen confirmation | phase1.py, study.py, tune_disco.py, confirm_disco.py | New schematic to draw; show truth access separately |
| Figure 2: baseline sculpting | Jet-mass inputs introduce a sensitivity/distortion trade-off | Original results_study/closure_f0.01.png; summary.csv and selected baseline predictions | Existing exploratory plot; relabel and simplify |
| Figure 3: lambda scan, two injection panels | Strong penalties fail; 0.003 meets the validation rule | results_tuning/validation_summary.csv and selection.json; initial strong-lambda rows from results_study/summary.csv | New publication-style plot; plot seed points and 90%/80% target lines |
| Figure 4: fresh background efficiency versus mass | Physical effect of frozen DisCo | results_confirmation/*.npz; original efficiency_f0.005.png and efficiency_f0.01.png | Existing curves; regenerate as two panels; seed SD is not an event confidence interval |
| Figure 5: fresh sensitivity versus distortion | Quantify the benefit and its cost; include simpler EFP-only reference | results_confirmation/metrics.csv and confirmation.json | New plot; show all three seed points, aggregate results and carefully labelled intervals |
| Table 1: experimental design | What each comparison establishes | Original 105-fit grid, 24-fit tuning grid, 18-model frozen confirmation | Assemble from saved configs; do not call the 18 inference evaluations training runs |
| Table 2: data accounting and truth access | Demonstrate prevention of leakage | Saved row IDs and cache metadata | Assemble counts; distinguish raw rows, selected events, injected training signal and full evaluation signal |
| Table 3: final performance | Report absolute SIC/JS and relative gains with limits | AUDITED_RESULTS.md and results_confirmation/metrics.csv | Audited numerical source ready |

Appendix: all hyperparameters; full seed/strength tables; mathematical tests;
DisCo centering and estimator formula; reproducibility commands; failure of strong
regularization; oracle benchmark versus sideband method. Preserve oracle results
in the original project (summary.csv records all runs); they are not the practical
final method.

Caption rules: name the sample (training/validation/fresh confirmation), injected
TRAINING signal fraction, score-cut calibration source, model seeds, and error-bar
meaning. Mark exploratory plots versus confirmation plots. Do not mix those samples
into one undifferentiated success claim. No plot establishes a fake discovery.

Suggested final claim: validation-selected sideband DisCo reduces mean background
JS distortion by about 45-59% while retaining about 94-107% of baseline mean SIC
on an independent simulated sample. This is a reduced EFP study, not a full Kitchen
Sink reproduction or evidence of complete independence. No report prose drafted.

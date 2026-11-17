# Audited results table

Metrics recomputed from all 18 saved confirmation prediction files. Frozen code/model/preprocessing/threshold-file SHA-256 hashes match the confirmation protocol. Fresh row IDs are identical across models, unique, and disjoint from saved training/validation IDs.

| Training signal | Model | Mean SIC | Mean JS | Mean signal efficiency | Mean background efficiency |
|---|---|---:|---:|---:|---:|
| 0.5% | mass_mlp | 2.554105 | 0.030717 | 0.627118 | 0.060261 |
| 0.5% | disco | 2.408902 | 0.016805 | 0.603144 | 0.062620 |
| 0.5% | efp_mlp | 0.914956 | 0.002118 | 0.210292 | 0.051465 |
| 1.0% | mass_mlp | 2.877388 | 0.030323 | 0.705678 | 0.060184 |
| 1.0% | disco | 3.065637 | 0.012556 | 0.726381 | 0.056440 |
| 1.0% | efp_mlp | 2.869630 | 0.002022 | 0.664470 | 0.053799 |

| Training signal | SIC retained, 95% interval | JS reduction, 95% interval |
|---|---|---|
| 0.5% | 94.31% [92.96, 95.92] | 45.29% [34.56, 50.71] |
| 1.0% | 106.54% [104.23, 108.93] | 58.59% [47.09, 61.56] |

Ratios are ratios of three-seed means, not means of per-seed ratios. SIC is a sensitivity proxy, not discovery significance. JS uses base-2 logarithms and 16 bins over 3100-3900 GeV.
Bootstrap intervals are transcribed from the saved confirmation output, not recomputed by this quick audit. They are conditional paired stratified event-bootstrap intervals for fixed models, not training/simulator uncertainty or simultaneous coverage.
Important: seed 11 at 1% injection has worse JS with DisCo; improvements are aggregate. EFP-only is a competitive simpler reference at 1%, but unstable at 0.5%.
Full numerical per-seed values: results_confirmation/metrics.csv. Full interval regeneration: confirm_disco.py (uses cached features, frozen inference, and 300 paired bootstrap replicates).
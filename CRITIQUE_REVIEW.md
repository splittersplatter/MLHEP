# Assessment of Collider ML Critique

Reviewed 18 November 2026. Source: `/home/dhruv/Downloads/Collider ML Critique.pdf`
(three pages). Its recommendations were assessed as external claims, not adopted
as instructions. The PDF discusses the older demonstration report; its numerical
claims must not be attributed to the later Linux experiment.

## What has actually been addressed

Added `sculpting_diagnostics.py` and five synthetic tests in
`test_sculpting_diagnostics.py`. Generated retrospective diagnostic reports for
all 18 saved confirmation models in each of two bundles:

- `output/review/critique_demo/REPORT.md` and `diagnostics.json`.
- `output/review/critique_linux/REPORT.md` and `diagnostics.json`.
- Each folder also contains `background_efficiency.png`.

Added `output/review/critique_feature_check.py` and its JSON evidence. It checks
existing development caches and a clearly labeled synthetic mathematical example.
Extended `FIX_PLAN.md` with production integration, local closure, loss telemetry,
seed stability, feature-tail diagnostics and separately versioned physics variants.
The two review documents now have targeted `.gitignore` exceptions.

Eight tests pass: the three existing root mathematical/split checks and five new
diagnostic tests. Training code, models, saved predictions, saved thresholds and
historical reports were not changed. No new event sample, model inference,
training, selection or independent confirmation was performed. Existing
provenance/split repairs remain planned; the diagnostic additions do not repair
past sample reuse.

## Claim-by-claim findings and response

| Critique | Assessment | Response / remaining action |
|---|---|---|
| Global JS can conceal local SR behavior | Valid limitation | Added region/bin diagnostics; require local closure before claims about fake-bump removal |
| SR acceptance is 25.2% above SB | Uses nominal 5%, not measured evaluation SB | Report both calibration target and measured acceptance; see recalculation below |
| Sideband penalty guarantees nothing inside SR | Correct | Preserve this limitation; do not replace the practical method with a truth-masked SR penalty |
| Upward slope is necessarily caused by learning larger jet masses | Plausible mechanism, not established by existing results | Require feature/score/kinematic diagnostics and controlled feature ablations |
| Tiny scalar penalty is irrelevant unless gradients blow up | Overstated | Add actual gradient telemetry; scalar loss size alone does not measure optimization impact |
| Three-seed lambda selection risks selection noise | Valid concern | Already covered partly by separation/freezing plan; added seed influence/stability requirements and full-grid reporting |
| Checkpointing effectively ignores DisCo | Not established; not universally true in saved logs | Added comparison of recorded-objective and BCE-only epoch choices |
| EFP-only is exactly at a Poisson floor and has zero sculpting | Unsupported strength of conclusion | Added conditional fixed-count JS reference checks; retain EFP-only and avoid equivalence/independence claims |
| Ungroomed jets are a preprocessing mistake; UE explains effects | Not established for this benchmark; MPI/pileup are absent by dataset specification | Keep documented baseline; grooming is a separately versioned ablation |
| Missing dijet angular cut is necessarily a mistake | Not demonstrated against the cited preprocessing | Treat a defined angular cut as a new selection requiring new extraction/training |
| Degree-4 EFPs necessarily vanish below four constituents | Incorrect general rule | Verified a counterexample; no four-constituent veto should be introduced |
| The log floor can create large standardized tails | Real concern, distinct from the incorrect EFP explanation | Measured rare clipped values and a -30.2 standardized tail; added train-only robustness work |

### 1. Local distortion: confirmed concern, narrower interpretation

Recomputed from the older demo predictions, averaging the same three frozen
models at training fraction 0.005:

| Model | Lower SB | SR | Upper SB | Combined SB | SR / measured SB |
|---|---:|---:|---:|---:|---:|
| Mass MLP | 0.03287 | 0.06026 | 0.08919 | 0.04992 | 1.2070 |
| DisCo | 0.03872 | 0.06262 | 0.07948 | 0.05106 | 1.2264 |

Ratios are ratios of mean efficiencies, not means of seed ratios. The critique's
25.2% is the excess relative to nominal 0.05; the measured combined-SB reference
gives about 22.6% for DisCo. The relative SR acceptance increases slightly versus
the baseline, despite improved global JS. But upper-SB acceptance remains larger
than SR acceptance. These summaries therefore establish residual mass dependence,
not a narrow fake resonance or a discovery significance. Unequal populations of
the sidebands also affect their pooled efficiency; a smooth trend can produce an
SR/pooled-SB difference without a localized bump.

The Linux archive is different: at 0.005 the corresponding SR/measured-SB ratios
are 1.1669 for DisCo and 1.1480 for mass MLP. Its full curves and counts are in
the separate Linux diagnostic report. Do not combine results across bundles or
treat the shared evaluation events as independent samples.

Future scientific claims need a predefined sideband interpolation/background fit
and background-only closure study, including uncertainty. This is additional
analysis, not something that can be repaired by changing a plot caption.

### 2. Optimization and checkpointing: measure before explaining

The numerical magnitude of a penalty is not its parameter-gradient magnitude.
For exact distance correlation away from the denominator floor, rescaling scores
preserves the value while changing the gradient with respect to those scores.
That makes the critique's mechanism possible, not proven for these networks.
Sigmoid derivatives, parameter gradients, finite batches and the denominator floor
also matter. A small injected signal fraction does not establish identical
background feature distributions in SR/SB.

At an individual event, unweighted BCE has derivative `sigmoid(logit) - y` with
respect to the logit; near score 0.5 this is approximately +/-0.5, not necessarily
tiny. With respect to the probability, the magnitude is approximately 2. Aggregate
gradients can cancel; that requires measurement. Class weighting changes these
derivatives by the weights.

The critique's `1/sqrt(N)` estimate is not an estimate of validation BCE error
without the per-event loss variance. Successive checkpoints are also evaluated
on the same rows, so differences are paired, not independent validation draws.

Replaying selection from the saved loss components with the code's `1e-6`
improvement rule changes the selected epoch between BCE-only and the recorded
objective in 1 of 6 demo DisCo histories (0.005, seed 22: epoch 13 versus 11).
It changes none of the 6 Linux histories. This compares observed epochs only;
counterfactual BCE-only early stopping might stop at a different time. The logs
cannot establish whether DisCo gradients caused a later sensitivity improvement.
The updated plan requires direct loss/gradient/score-spread telemetry.

Three-seed selection remains exploratory and vulnerable to training variation.
Increasing seeds after seeing results cannot retrospectively remove selection
bias. Freeze new development comparisons and keep final data separate.

### 3. JS noise: conditional reference, not an exact floor

The expression `(K-1)/(8*N*ln(2))` is a leading large-sample approximation for a
particular reference model, not a positive lower bound for every observed JS.
Here the selected histogram is a subset of its reference histogram, so those
histograms are statistically dependent. Selection counts vary by fitted model.
Sparse bins, finite populations and nonuniform bin populations also matter.

The new diagnostic uses 1,999 fixed-size random subsets of the actual background
histogram for each fitted model. Multivariate-hypergeometric draws give the same
histogram distribution as uniformly permuting selected/unselected labels over
the fixed background events. This preserves the actual selected count and the
16-bin spectrum; it does not use the rough nominal 5% count.

For the older demo, all six EFP-only upper-tail p-values exceed 0.05, from 0.0645
to 0.9145. This does not establish zero sculpting or equivalence. For the Linux
archive, seed 33 at 0.005 has p=0.0205 in an uncorrected check; it must not be
silently removed from a blanket claim. These retrospective checks span multiple
models/seeds/fractions, have Monte Carlo uncertainty, and are not calibrated
discovery statements. Failure to reject a null is not proof of independence.

Preserve EFP-only as a central comparator. Any claim of added value must compare
sensitivity, local closure and variability rather than global JS alone. A small
SIC does not itself establish collapsed constant scores.

### 4. Physics preprocessing: ablations, not automatic repairs

The [LHCO dataset record](https://zenodo.org/records/4536377) specifies no pileup
or multi-parton interactions. The critique's underlying-event explanation should
not be used as an established cause in this benchmark. Shower radiation and
other kinematic correlations remain possible contributors, not measured causes.

The [Kitchen Sink preprocessing, Section III.3](https://arxiv.org/html/2604.20965v1#S3.SS3)
describes anti-kT R=1, a leading-pT threshold and mass ordering, consistent with
the present intended baseline. That section does not specify grooming or the
proposed angular cut. This does not mean such choices are never useful in other
analyses. [Soft Drop](https://arxiv.org/abs/1402.2657) removes soft wide-angle
radiation, but changing grooming changes observables and does not guarantee
independence from dijet mass.

Grooming, dimensionless mass inputs and a precisely defined angular cut belong
in predeclared development ablations, each with new extraction/cache identity,
training, calibration and changed selection counts. Current feature caches lack
jet eta/pT and constituent counts needed for several proposed diagnostics; these
values cannot be reconstructed reliably by assumption or filled in.

### 5. EFP zeros and the log floor: separate the false explanation from the real tail

In the [EnergyFlow definition](https://energyflow.network/docs/efp/), degree
counts edges and particle indices are summed independently, including repeated
assignments. It is not a universal requirement for four distinct constituents.
The [original EFP paper](https://www.ericmetodiev.com/files/pdfs/energyflowpolynomials.pdf)
relates the nonvanishing prong requirement to graph chromatic number.

A synthetic two-constituent example with four parallel edges yields EFP=0.0128
with the installed library. This refutes the general four-constituent rule; it
does not say every graph is nonzero on two constituents.

The log floor is nevertheless worth auditing. Each inspected development cache
has 259 entries at or below 1e-12 out of 5,018,256 EFP values, including 257 exact
zeros, across 11 of 69,698 events. On the recorded seed-11, 0.005 mass-MLP
training split, 80 active-feature entries are floored; their standardized range
is approximately -30.20 to -4.99. Thus a rare tail is present. Its cause and
effect on fitted performance are unverified; constituent counts were not stored.

The plan now requires train-only floor/tail diagnostics and predeclared transform
robustness checks. It does not silently remove these events, change historical
preprocessing, or promise that an alternative transformation improves results.

## Reproduce the added checks

Use the existing Linux interpreter/dependencies; portability remains a separate
planned fix. From the project root:

```bash
PYTHONPATH=.linux-deps MPLCONFIGDIR=/tmp/mlhep-review-mpl \
  /home/dhruv/Desktop/projects/NNQS-AIMD/.venv/bin/python \
  -m unittest test_study test_sculpting_diagnostics -v

PYTHONPATH=.linux-deps MPLCONFIGDIR=/tmp/mlhep-review-mpl \
  /home/dhruv/Desktop/projects/NNQS-AIMD/.venv/bin/python sculpting_diagnostics.py \
  --run reproducible_demo --output output/review/critique_demo

PYTHONPATH=.linux-deps MPLCONFIGDIR=/tmp/mlhep-review-mpl \
  /home/dhruv/Desktop/projects/NNQS-AIMD/.venv/bin/python sculpting_diagnostics.py \
  --run results_linux_new/previous_run_20261118 --output output/review/critique_linux

PYTHONPATH=.linux-deps \
  /home/dhruv/Desktop/projects/NNQS-AIMD/.venv/bin/python \
  output/review/critique_feature_check.py
```

Diagnostic JSON records source-file checksums, exact counts, permutation settings
and explicit undefined values. These newly calculated diagnostics remain separate
from original experiment results. Checksums identify inspected artifacts; they
do not resolve legacy provenance gaps or certify the simulator.


## Protocol-v2 implementation follow-up (18 November 2026)

The original review above remains a record of the state at review time.
Root training, provenance, splitting, metrics, assessment and reporting have since
been replaced by protocol v2. See `IMPLEMENTATION_STATUS.md` for the requirement
mapping, executed checks and remaining scientific limits, and `README.md` for
maintained commands. Historical experiment outputs have not been rewritten.

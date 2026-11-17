> Historical protocol-v1 documentation. These commands and numerical claims describe earlier runs, not the maintained protocol. Use [README.md](README.md) for protocol v2. Previously evaluated rows 200000–299999 are exploratory.

# Confirm the selected DisCo strength on fresh events

Run this in PyCharm's terminal:

```powershell
.\.venv\Scripts\python.exe confirm_disco.py
```

The script uses raw source rows 200000 through 299999. No model is trained and no
threshold is adjusted. It loads three existing seeds for the ordinary mass-feature
MLP, sideband DisCo lambda=0.003, and EFP-only MLP, at both training injection
fractions (18 saved models). All models evaluate the same fresh events.

Extraction uses bounded chunks and saves separate caches every 10,000 raw events.
If interrupted, rerun the command. Completed extraction blocks are reused; an
unfinished block or inference/bootstrap stage may repeat. Do not change model
files, source data or code mid-confirmation: the frozen protocol detects changes.
No active assistant session is needed while this runs.

When complete, send results_confirmation/REPORT.md. It shows whether at EACH
signal fraction the mean SIC retains >=90% of baseline while mean raw JS is <=80%
of baseline. It also shows 95% paired event-bootstrap intervals. A point estimate
meeting the targets with intervals crossing them is less conclusive than intervals
entirely beyond the targets. These are conditional, pointwise intervals, not a
claim of discovery significance or complete decorrelation.

The EFP-only reference, all model metrics and predictions are saved alongside
background-efficiency plots. Seed error bars in those plots are standard deviations
across three frozen models, NOT independent-data confidence intervals.

Verification performed before handoff: all 18 models were loaded and their inference
checked against already-used validation predictions. Fresh-data evaluation has NOT
been run by the assistant. For a repeat of the loader check only:

```powershell
.\.venv\Scripts\python.exe confirm_disco.py --check-only
```

The bootstrap conditions on observed background counts in SR/SB and the SR signal
count. It accounts for the shared fresh events when comparing methods; it does not
include uncertainty from training or simulation. Do not tune again on this sample.

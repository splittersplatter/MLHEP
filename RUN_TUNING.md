> Historical protocol-v1 documentation. These commands and numerical claims describe earlier runs, not the maintained protocol. Use [README.md](README.md) for protocol v2. Previously evaluated rows 200000–299999 are exploratory.

# Run the smaller DisCo tuning experiment

In the PyCharm terminal, run:

```powershell
.\.venv\Scripts\python.exe tune_disco.py
```

No download or feature extraction is needed. The script requires the existing
200,000-event cache and original results_study validation predictions.
It runs 4 weaker penalties (0.001, 0.003, 0.01, 0.03), 2 SR signal fractions
(0.5%, 1%), and 3 seeds (11,22,33): 24 new sideband-DisCo fits.
Baseline validation predictions are reused, with source-row matching checked.
The code uses the existing 15-epoch limit and validation-based early stopping.
It never scores test events or reads stored test prediction arrays.

One fit was run during verification. It is saved in results_tuning and will be
skipped automatically. Interrupted runs resume using the same command.
For shorter sessions, add `--max-new-fits 4` to run at most four unfinished fits.
Do not change the script/protocol mid-run: use another output folder if changing it.

When finished, open results_tuning/REPORT.md. It lists the selected strength or
reports that none qualifies. selection.json contains the machine-readable result;
validation_summary.csv has all the metrics. Each fit retains model weights,
preprocessing, validation predictions, source row IDs and training history.

The rule is chosen before the sweep: retain >=90% of baseline mean validation SIC
AND reduce mean JS divergence by >=20%, separately at BOTH injection fractions.
Among qualifying strengths, choose the smallest mean JS/baseline ratio; break ties
with the weaker strength. These are engineering targets, not significance tests.
No selection is issued until all 24 fits finish. A no-winner outcome is valid.

Truth is used to evaluate signal sensitivity and background distortion on simulation;
the training penalty uses only sideband membership. Thresholds are the 95th percentile
of validation-sideband scores. Validation includes checkpoint selection and strength
tuning, so the outcome is exploratory and potentially optimistic. Three seeds share
much of the data. Do not interpret their differences as independent confidence bounds.

Fresh-data confirmation is deliberately deferred until a strength is selected.
Send the contents of results_tuning/REPORT.md back for interpretation. No active
assistant session is needed while Python trains.

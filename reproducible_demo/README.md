# Reproducible MLHEP demonstration

This folder contains only the selected demonstration code, frozen artifacts,
provenance, caches, and report-groundwork documents. Original project files are
unchanged. It is a verification/replay bundle, not a claim of clean-room training
reproduction. No report has been drafted.

## Run the short live demonstration
From the MLHEP PyCharm terminal:

```powershell
.\.venv\Scripts\python.exe reproducible_demo\run_gauntlet.py
```

It sequentially runs mathematical/split tests, reloads all 18 models against saved
validation predictions, and independently recomputes confirmation metrics and
ratios. It writes VERIFICATION_LOG.txt and AUDITED_RESULTS.md. No retraining occurs.

Optional longer replay (frozen inference plus 300 paired bootstrap replicates):

```powershell
.\.venv\Scripts\python.exe reproducible_demo\run_gauntlet.py --replay-confirmation
```

Cache blocks are included. The original raw HDF5 must remain at the path in
results_study/config.json because the extractor verifies source-file metadata even
when loading caches. The 2.8 GB file and virtual environment are deliberately not
copied. On this machine the dataset is ../events_anomalydetection.h5. Dependencies
are pinned in requirements-study.txt; Python version matches the project .venv.
Moving the bundle to another machine requires a documented path/provenance update,
not silently editing a frozen protocol. Clean-environment portability is untested.

## Exact order and what to show
1. FIGURE_TABLE_INVENTORY.md: experimental story and intended figures.
2. run_gauntlet.py: mathematical tests -> inference reload -> saved-result audit.
3. AUDITED_RESULTS.md: absolute values, confirmed ratios and uncertainty limits.
4. results_confirmation/metrics.csv: individual seeds, including adverse outcomes.
5. REPRODUCIBILITY_CHECKLIST.md: what has and has not been verified.
6. Optional confirmation replay: generates plots and report directly from frozen artifacts.

## Files and why they are included
- phase1.py: physics extraction, network definition, mass windows.
- study.py: shared training, preprocessing, scoring, DisCo and metric functions.
- test_study.py: mathematical/statistical and split tests.
- tune_disco.py: documented validation-only selection implementation.
- confirm_disco.py: frozen fresh-data evaluation and bootstrap implementation.
- report_study.py: analysis-generation source for the broad experiment.
- audit_results.py and run_gauntlet.py: short verification demonstration.
- requirements-study.txt: installed package versions.
- results_study: full aggregate summary/config/manifest plus ONLY the 12 selected
  baseline/EFP-only model runs needed for final comparison and inference checks.
- results_tuning: full validation summary/selection/protocol plus ONLY the six
  selected lambda=0.003 runs (not all 24 tuning checkpoints).
- results_confirmation: original protocol, metrics, intervals and 18 prediction archives.
- feature_cache: original 200k development feature cache for reload checks, plus
  ten 10k confirmation caches for replay. These are derived data, not raw events.
- FILE_MANIFEST.csv: relative filenames, sizes and SHA-256 hashes at packaging time.
  Regenerated outputs may change; frozen model/source hashes are independently checked
  against the original confirmation protocol by audit_results.py.

## Full training reproduction (separate, expensive experiment)
The original full sequence is phase1.py (optional diagnostic), study.py,
report_study.py, tune_disco.py, confirm_disco.py. Do not run training into this frozen
bundle or overwrite the original experiment. Make a separate workspace, provide the
raw dataset, run study.py --data ABSOLUTE_DATASET_PATH, then report_study.py,
tune_disco.py and confirm_disco.py. The scripts currently contain cache-path and
configuration-path assumptions; porting them to a new source path requires explicit
updates and a new manifest. This is why the present bundle claims verified artifact
replay, not one-command portable clean-room training reproduction.

The full archived experiment remains in the parent MLHEP project; this bundle omits
unused models rather than duplicating everything. The original study manifest
records historical source versions, whereas the confirmation protocol records the
specific extraction/scoring versions used for the final run.

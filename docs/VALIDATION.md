# Executed validation, 18 November 2026

The execution environment available in this workspace uses the pre-existing
`.linux-deps` packages with the existing Python 3.12.14 interpreter at
`/home/dhruv/Desktop/projects/NNQS-AIMD/.venv/bin/python`. No installation or
upgrade was performed. That path is recorded here as execution evidence; it is
not embedded in the maintained launcher or required by the software.

The initial sandbox GPU probe was unavailable. At the user's request, host
checks were run outside the sandbox: NVIDIA driver 595.99.02, RTX 3060 Laptop
GPU, PyTorch 2.15.0 CUDA 13.2. CUDA matrix multiplication and finite-result check
passed. All scientific MLP training now defaults to CUDA and fails if it is
unavailable. CPU remains an explicit option and the scikit-learn GBDT runs on CPU.

Executed host suite:

```bash
PYTHONPATH=.linux-deps MPLCONFIGDIR=/tmp/mlhep-mpl \
  /home/dhruv/Desktop/projects/NNQS-AIMD/.venv/bin/python \
  -m unittest test_study test_sculpting_diagnostics test_integrity test_extraction test_pipeline -v
```

All 27 tests passed, including CUDA training, unchanged telemetry-enabled
predictions, CPU reload, concurrent reservation, exact membership, safe resume,
strict JSON, undefined statistics and synthetic full reporting/replay/bootstrap.
The optimized-Python integrity/extraction suite also passed (14 checks, GPU check
skipped in the sandbox and separately passed on the host).

The independent historical auditor was rerun: 300 study metric records and 18
historical confirmation models remain numerically consistent for the four
quantities it checks. It still finds the documented legacy sample overlap; old
results are not relabeled as repaired or independent.

A declared limited development run was executed on the host GPU with real,
already-explored raw rows 0:10000, 15 epochs, seeds 11/22/33, fractions 0/.005/.01,
lambda .003, 199 permutations and gradient telemetry. All 45 unique fits completed.
Models include EFP-only MLP, mass MLP, mass GBDT, sideband DisCo and labeled oracle.
The full canonical 200k-row run has not been executed by this implementation task.

Final run: `results_v2_development_10k_final/`.
Final report: `output/v2_development_10k_final_report/REPORT.html`.
A preceding 10k run remains preserved under its own earlier source snapshot.
No final-role predictions have been scored in these real-data runs. No candidate
unused sample was opened for independent evaluation. The freeze/replay pathway
was exercised on labeled synthetic software fixtures only.

Machine-readable audited run evidence is in `development_validation.json`.
HTML was inspected in the in-app browser and figures checked locally. The report
shows unavailable values and failure to establish all tuning targets; no success
or independence claim was manufactured. Read these results as a pipeline and
limited development check, not as the complete physics study.

## Final software verification and handoff

The final host suite `python -m unittest discover -v` passed all **38 tests**,
with no skips, including GPU training, identical telemetry-enabled GPU outputs
and CPU reload. The optimized-Python integrity/extraction/audit suite passed
17 tests with one sandbox CUDA skip (18 total); that GPU case passed in the host
suite. New tests exercise declared floor membership, input/history preservation,
independent Wilson/JS/closure-bootstrap corruption rejection, train-only robust
scaling, shared declared transforms, and tuning/final checkpoint invariance.

Full source-backed extraction of raw rows 0:200000 completed with 69,698 selected
events and a versioned schema/content receipt. It retained a tiny signed FastJet
mass under the explicit roundoff bound; materially negative values reject.
Historical inventory covers 550, 430 and 104 files in the three bundles and
records reported role use plus conservative derived-artifact inspection. It
retains unknown historical source/exposure status. The full generated inventory
is under output/; docs/historical_exposure_summary.json records its digest.

**No complete 200k training study was started.** The user explicitly requested
that complete studies be run by them. README.md provides the baseline GPU command
and the predeclared three-transform loop. Scientific outcomes of the final source
remain unmeasured. Earlier 10k results are preserved under their original source.
No real final-role scores were opened. Software/review checks do not establish
unused-sample provenance, physical-model coverage or removal of mass sculpting.

Machine-readable final evidence: final_repair_validation.json. `git diff --check`
passes. Code review findings and their fixes are in CODE_REVIEW.md.

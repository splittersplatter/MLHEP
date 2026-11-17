# Protocol-v2 review

Baseline: initial commit `44d10b8`; scope: maintained working-tree implementation,
new helpers/tests and FIX_PLAN.md / CRITIQUE_REVIEW.md requirements.

## Standards

One hard defect: independent eligibility and reservation were separate
transactions. Fixed by checking eligibility inside the dataset-ledger lock that
records access, before loading candidate features. A concurrent-reservation
regression requires exactly one successful attempt.

Two material maintainability/audit heuristics: repeated production protocol
constants and weaker replay sample checks. Production constants now share
`protocol.py`, while independent arithmetic retains its own definitions. Replay
reports require exact reserved membership after the declared exclusion, and
mass/truth equality against verified features. Development audit additionally
reloads models on CPU with an explicit 2e-6 absolute score tolerance.

Follow-up static review found these issues addressed. GPU/CPU tensor movement
was reviewed; executed GPU and reload evidence is in VALIDATION.md.

## Spec

Two exposure-integrity findings: non-atomic eligibility/reservation (also found
on the Standards axis), and replay accepting an alternate ledger. Both are
fixed. Every evaluation uses the protocol's canonical dataset ledger; unknown
or reused exposure is refused in independent mode. Tests cover concurrent
reservation, reuse, alternate-ledger replay and interrupted-attempt persistence.

The follow-up reviewer executed all three focused exposure regressions and
reported them passing. Future closure uncertainty, physics ablations and unknown
sample independence are disclosed limitations, not unsupported success claims.

Standards: 1 hard finding and 2 heuristics addressed; Spec: 2 findings addressed.
Worst defect on each axis was the exposure reservation race, now fixed.

## Final extensions review

Spec and Standards reviewers identified a hardcoded 1e-12 floor in raw-event
auditing. It now derives and records the declared transform specification; a
fixture between 1e-12 and 1e-10 verifies changed membership/counts. Spec also
identified audit-output paths that could overwrite data/history. The new writers
require new destinations and protect source/cache/history paths, with regression
checks. Independent-audit coverage now includes Wilson intervals, conditional
JS nulls, sideband fit covariance/residuals, calibration bootstrap, fixed-lambda
and seed-omission ratios, and transform-comparison aggregates. Corruption tests
prove the independent local audit rejects edited numerical output.

Reviewers found train-only scaling, coupled calibration resampling, sideband-only
fitting, declared comparison identities and precompletion reload checks coherent.
No historical experiment artifacts were edited. Complete scientific studies are
left to the user by explicit instruction; no favorable outcome is presumed.

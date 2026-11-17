# Reproducibility checklist

## Verified in the current project environment
- [x] Custom-loss mathematical tests pass (see gauntlet output).
- [x] Frozen code, models, preprocessing and threshold files match confirmation hashes.
- [x] Metrics recomputed from 18 saved prediction archives agree with metrics.csv.
- [x] Confirmation rows are unique, shared between models, in [200000,300000), and disjoint from saved training/validation rows.
- [x] Aggregate SIC and JS ratios agree with confirmation.json.
- [x] Saved models reproduce earlier validation predictions (18 reload checks).

## Required interpretation checks
- [x] Strength was selected on validation; confirmation used frozen models/cuts.
- [x] Classification targets are SR/SB, not signal truth.
- [x] Truth used for simulation mixture construction and evaluation is disclosed.
- [x] Oracle DisCo is labelled truth-assisted; selected sideband DisCo is distinct.
- [x] Three shared-data seeds are not treated as independent test samples.
- [x] Individual failures and the EFP-only reference are retained.
- [x] Conditional event-bootstrap intervals exclude training and simulator uncertainty.

## Not yet independently verified
- [ ] Clean-environment installation and full training reproduction on another machine.
- [ ] Bitwise reproduction across different CPUs/library versions (not guaranteed).
- [ ] This quick audit does NOT recalculate bootstrap intervals; run confirmation
      replay to regenerate them. Original intervals are preserved with provenance.
- [ ] No additional simulation generator, signal topology, systematic variation,
      calibrated bump fit, discovery significance or look-elsewhere analysis.

Source-row separation is verified; detecting duplicate physical events occurring
under different source row IDs has not been performed.

Do not change frozen artifact files to make a hash check pass. Investigate mismatches.

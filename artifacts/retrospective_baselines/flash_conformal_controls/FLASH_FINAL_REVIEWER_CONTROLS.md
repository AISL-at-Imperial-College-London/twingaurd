# Final flash reviewer-control analysis

The final-test calculations use the exact frozen E5 1200-episode artifacts.
The new C0/C1 calibration scores use deterministic reconstruction of the
frozen E4 ordinary calibration futures; every regenerated operational input
is checked against the frozen E4 operational-input bundle before use.
No model is refit and no E5 trajectory is regenerated.

## Frozen-result reconstruction
- Frozen/recomputed adaptive q95: 11.0485985083
- M0_nominal_DT: certified=779/1200, false-safe=207/1200, coverage=0.000000
- M1_discrepancy_mean: certified=672/1200, false-safe=103/1200, coverage=0.000000
- M2_raw_bootstrap: certified=623/1200, false-safe=72/1200, coverage=0.241379
- M3_pooled_conformal: certified=177/1200, false-safe=5/1200, coverage=0.939933

## Reviewer-facing 95% controls
- M0_nominal_DT: availability=0.6492, false-safe=0.1725, coverage=0.0000
- C0_nominal_fixedscale_conformal: availability=0.0000, false-safe=0.0000, coverage=0.9577
- M1_discrepancy_mean: availability=0.5600, false-safe=0.0858, coverage=0.0000
- C1_corrected_fixedscale_conformal: availability=0.0408, false-safe=0.0008, coverage=0.9477
- M2_raw_bootstrap: availability=0.5192, false-safe=0.0600, coverage=0.2414
- M3_pooled_conformal: availability=0.1475, false-safe=0.0042, coverage=0.9399

Interpretation guardrails:
- C0 isolates conformal back-off around the nominal DT.
- C1 isolates discrepancy-mean correction under a fixed normalized width.
- M3 adds state/action-dependent bootstrap scale before conformal calibration.
- Fixed C0/C1 scales are the declared engineering scales, not test-tuned widths.
- The guarantee remains population-marginal; alpha is not a SIL PFD.

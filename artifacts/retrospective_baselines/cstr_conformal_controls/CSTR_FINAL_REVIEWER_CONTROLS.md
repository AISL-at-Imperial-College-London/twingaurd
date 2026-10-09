# Final CSTR reviewer-control analysis

This analysis uses the exact frozen C1 500-episode calibration population
and exact frozen C1 1000-episode untouched final-test population.
No trajectories were regenerated and no model was refit.

## Frozen-result reconstruction
- Frozen upper q95: 5.46455758694
- Nominal-DT false-safe: 394/1000
- M3 certified: 338/1000
- M3 false-safe: 0/1000
- M3 upper-trajectory coverage: 0.9570

## Reviewer-facing 95% controls
- M0_nominal_DT: availability=1.0000, false-safe=0.3940, coverage=0.0000
- C0_nominal_constant_conformal: availability=0.0000, false-safe=0.0000, coverage=0.9510
- M1_corrected_center: availability=0.6810, false-safe=0.1210, coverage=0.2330
- C1_corrected_constant_conformal: availability=0.1940, false-safe=0.0000, coverage=0.9480
- M2_raw_bootstrap_95: availability=0.3900, false-safe=0.0000, coverage=0.7460
- M3_adaptive_conformal: availability=0.3380, false-safe=0.0000, coverage=0.9570

Interpretation:
- C0 asks whether conformalising the nominal DT alone yields useful authority.
- C1 asks whether discrepancy correction improves authority under a constant-width back-off.
- M3 asks whether state/action-dependent scale improves authority beyond constant width.
- The formal guarantee remains marginal over the represented episode population.

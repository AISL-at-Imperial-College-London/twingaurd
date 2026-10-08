# Milestone B.9S + B.10 — robustness and uncertainty calibration

## Central result

The training-only robustness classification is **PREVIOUS FIT NOT CONVERGED ENOUGH**. The frozen mean identity is **B9S_seed_91005_limit_100**. `replaced_after_B9=True`; a new 240-episode final population with new seeds was generated and revealed prospectively.

Across high-quality fits, spectral radius ranged from 0.994998 to 0.995000 (SD 0.000001); selection RMSE SD was 0.0041 K. Fit selection used no B.9 final-assessment episode.

## Variance diagnosis

U0 is the frozen homoskedastic model. U1 applies the calibration-only global variance multiplier 0.9387. U2 implemented: **False**, with calibration-supported features `[]`. No latent mean parameter was altered.

The pooled standardized innovation variance is 0.931; max innovation ACF is 0.100, max squared-innovation ACF is 0.024, and Ljung–Box p=8.61e-86. Statistical rejection is interpreted with these effect sizes and the prospective calibration result.

## Prospective calibration and sharpness

The uncertainty verdict is **UNDER-CONSERVATIVE** and the best tested comparator (none passed calibration) is **U0**. Its pooled pointwise MACE is 0.369, simultaneous MACE is 0.377, average pointwise width is 0.928 K, simultaneous width is 1.762 K, and median rollout RMSE is 0.610 K. Width alone never qualifies a model.

Calibration data comprise 60 model-selection episodes (20/category); uncertainty testing uses 240 episodes (80/category). Each complete prediction snapshot was written without truth and hashed before the corresponding hidden trajectory was read.

## Runtime

Median N=50 filtering time is 0.003593 s, variance evaluation is 0.010034 s, 2,000-path propagation is 0.0388 s, and total online latency is 0.0524 s.

## Final decision

**UNCERTAINTY MODEL NEEDS REVISION**

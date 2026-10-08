# Milestone B.12 — conformal trajectory calibration

## Decision

**TRAJECTORY CALIBRATION SUFFICIENT**

The B.11 U3 predictor is frozen exactly at J=10 and M=2000. The posterior median is the predeclared centre and the predictive-scale floor is 1e-06 K. Primary B2 uses one max-standardized score per complete 240-step calibration trajectory. For level 1-alpha, the finite-sample rank is `ceil((n_cal+1)*(1-alpha))` and the corresponding ordered episode score is used.

Five hundred new calibration episodes were used to freeze all scores, the 31-point smoothed horizon ablation, pooled/Mondrian quantiles, and method definitions. A separate 150-episode diagnostic population was used only for n_cal sensitivity. Five hundred subsequently generated final episodes remained untouched until each prediction snapshot was written and hashed.

## Findings

Raw B0 95% simultaneous coverage is 0.604; pooled B2 coverage is 0.934 with Wilson CI [0.909, 0.953]. The primary 95% inflation factor is **q=5.774**. B2/B0 95% mean-width ratio is 1.74, and B2 mean width is 3.538 K.

Pooled B2 MACE across 50/80/90/95/99% is 0.024. Minimum regime-specific 95% coverage is 0.898; the Mondrian minimum is 0.904. These are empirical regime diagnostics, not conditional conformal guarantees. The primary guarantee remains marginal over the defined episode population.

Calibration-size diagnostics indicate approximately n_cal=100 episodes for stable q and diagnostic coverage under the frozen rule. Median U3 latency is 0.2885 s, conformal construction adds 0.000238 s, and total latency is 0.2888 s.

## Answers

1. The frozen mean can be calibrated without modification: yes.
2. Episode-level B2 changes 95% simultaneous coverage from 0.604 to 0.934.
3. Required 95% inflation is q=5.774; q well above one quantifies U3 scale underestimation.
4. Sharpness is quantified by the 1.74 width ratio and absolute width above.
5. Regime generalization is reported in `coverage_summary.csv`; no category-specific guarantee is claimed.
6. Calibration data efficiency is summarized above and in `calibration_size_sensitivity.csv`.
7. Online conformal overhead is 0.000238 s.
8. Action-specific safety certification readiness: yes for the next decision-layer evaluation, not as a safety guarantee by itself.

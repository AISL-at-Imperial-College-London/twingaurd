# Milestone B.11 — parameter-uncertainty propagation

## Result

**PARAMETER UNCERTAINTY PARTIALLY SUFFICIENT**

Fifty converged, stable bootstrap models were obtained from complete-episode resamples of the original 600-episode offline training population. 0 attempted fits failed the predeclared convergence/stability rule and were recorded. No calibration, selection, B.10 test, or B.11 outcome entered bootstrap fitting. Calibration used 60 newly generated episodes; after J=10 and M=2000 were frozen, final evaluation used another 240 new episodes with distinct seeds and prediction-before-reveal snapshots.

U0 pooled pointwise/simultaneous MACE are 0.367/0.386; U3 values are 0.342/0.309. U3 changes 95% coverage by +0.030 pointwise and +0.104 simultaneous. Its width ratio is 1.07, while its mean-RMSE ratio is 0.976. Extremely broad bands do not qualify as success under the frozen rule.

## Scientific questions

1. Missing uncertainty explained: MACE reductions are 6.7% pointwise and 20.1% simultaneous.
2. Between-model variance changes from 0.0000 K² at h=1 to 0.0411 K² at h=240; its final share is 22.3%.
3. Late-horizon 95% coverage changes from 0.487 to 0.529, versus h=1 values 0.958/0.954.
4. Central prediction: U0/U3 median recursive RMSE is 0.599/0.584 K; U3 MAE and horizon bias are reported in the episode and horizon tables.
5. Operational ensemble: J=10, the smallest calibration-equivalent size under the frozen J50 comparison rule.
6. Runtime: U3 median total latency is 0.1695 s; offline bootstrap fitting is separately reported and is not an online cost.
7. Safety-probability readiness: no; the remaining calibration limitation must be resolved or explicitly bounded first.

The parameter family, preprocessing, N=50 filter, latent dimension, and conditional mean formulation are unchanged from B.10. U3 adds only whole-episode-bootstrap parameter uncertainty.

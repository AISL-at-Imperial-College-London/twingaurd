# Milestone C1.6B - Multi-fault CSTR revalidation

The transferred fault-blind method retained k=1, the four observable features, whole-episode bootstrap, J=10/M=2000 prediction, standardized episode-level conformal scores, and prospective predict-decide-reveal ordering. All state-space parameters, 50 bootstrap fits, and q+=6.6786 (finite-sample rank 381/400) were learned anew from balanced multi-fault populations. The final test used the practical minimum of 600 episodes because end-to-end CSTR integration and ensemble rollout made 1,000 materially more expensive. No final episode entered fitting, model diagnostics, bootstrap, or calibration.

The predefined k=1 training/selection gate passed: selection median RMSE was 0.854 K, absolute mean prediction bias was 0.652 K, and spectral radius was 0.99500. Dimensions 2 and 3 were therefore not evaluated.

## Pooled final results

The proposed method certified 0.235 of episodes, upper-envelope coverage was 0.948, and 0 false-safe decisions gave joint probability 0.000 (Wilson 95% CI [0.000, 0.006]). Conditional false-safe rate was 0.000 (Wilson [0.000, 0.027]); median latency was 0.755 s. Nominal-DT joint false-safe was 0.442.

| Fault subgroup | N | Actual unsafe | SAFE rate | False-safe count | Upper coverage |
|---|---:|---:|---:|---:|---:|
| ua_fouling | 300 | 0.450 | 0.287 | 0 | 0.950 |
| feed_temperature_step | 300 | 0.433 | 0.183 | 0 | 0.947 |

These fault rows are post-decision subgroup diagnostics under pooled calibration, not formal per-fault guarantees. High-severity UA undercoverage **persists** at 0.850; high-severity feed-temperature coverage is similarly 0.840. This conditional undercoverage does not contradict the pooled conformal guarantee, but it is the strongest deployment limitation.

## Original C1 comparison

| Metric | Original single-fault C1 | Multi-fault C1.6B |
|---|---:|---:|
| Nominal-DT joint false-safe | 0.394 | 0.442 |
| Proposed joint false-safe | 0.000 | 0.000 |
| Proposed SAFE certification | 0.338 | 0.235 |
| Proposed upper coverage | 0.957 | 0.948 |
| Mean conformal width (K) | 2.471 | 3.875 |
| Median online latency (s) | 0.167 | 0.755 |

### MULTI-FAULT REVALIDATION SUCCESSFUL

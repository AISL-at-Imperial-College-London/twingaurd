# Frozen CSTR limitations

1. **Marginal, not conditional, validity.** Split conformal calibration controls whole-trajectory upper-envelope miscoverage marginally over the declared exchangeable episode population. It does not provide a 95% guarantee for every selected action, fault subgroup, severity, or operating state.

2. **Interpolation abstention.** In the 1,000-episode C1 test, all interpolation actions were unresolved and none were certified SAFE. C1.5A reproduced this result and found a near-boundary population with heavier calibration-score tails; it did not justify reducing the frozen quantile.

3. **High-severity subgroup undercoverage.** Multi-fault pooled upper coverage is 0.948, but high-severity UA and feed-temperature subgroup coverage is 0.850 and 0.840. The original high-severity UA weakness therefore persists and extends to the added mechanism. These are subgroup diagnostics, not formal conditional guarantees.

4. **Lower-envelope limitation.** C1 lower trajectory coverage is 0.928, below its upper coverage of 0.957. The paper must use SAFE versus NOT CERTIFIED as the primary decision and must not present strong UNSAFE certification as validated.

5. **Simulator-only evidence.** All training, calibration, and testing use the same CSTR simulator family. Numerical integration, controller structure, sensor model, and fault implementations may not represent plant/model discrepancy in hardware or another simulator.

6. **Limited fault diversity.** Final multi-fault evidence covers two mechanisms: UA fouling and an upstream feed-temperature step. It does not cover arbitrary kinetics, composition, sensor, actuator, or structural faults. A separate C1.5B result directory is absent; frozen severity evidence comes from C1.5A and the completed C1.6B severity analysis.

7. **Finite safety tests.** C1 contains 1,000 final episodes and C1.6B contains 600. Zero observed false-safe events yields Wilson upper bounds of 0.0038 and 0.0064, not zero true probability. C1.5C contains 100 decision states.

8. **Selection changes the evaluated population.** Screening and choosing the largest conformal margin can improve availability, but pooled split-conformal validity does not automatically become conditional validity for the selected action.

9. **Width and latency increase under fault diversity.** Relative to original C1, C1.6B mean conformal width rises from 2.471 to 3.875 K, median latency rises from 0.167 to 0.755 s, and SAFE certification falls from 0.338 to 0.235.

10. **Developmental ablations use multiple frozen cohorts.** Early B metrics, the B.7–B.9 final assessments, B.11 uncertainty tests, B.12 conformal tests, and C1 safety tests are not one shared episode set. `FINAL_ABLATION.csv` identifies comparability groups; numbers must not be interpreted as a single paired monotone sequence outside those groups.

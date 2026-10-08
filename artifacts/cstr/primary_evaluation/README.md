# Milestone C1 — action-specific safety certification

## Decision

**SAFETY LAYER SUCCESSFUL**

The predictor is frozen at the B.11 U3 stack (first 10 accepted bootstrap models, M=2000, posterior median centre) and the B.12 numerical scale floor of 1e-06 K. The primary physical constraint is the repository's existing hard limit, **reactor temperature <= 313.0 K at every one of 240 future seconds** (`cstr_case.SafeStateChecker`). It was fixed before C1 outcome generation.

The independent safety-calibration population contains 500 episodes. Its finite-sample 95% one-sided quantiles are q+=5.465 and q-=2.600; they were frozen before generating 1000 new final episodes. Candidate actions retain the established 0.012–0.045 L/s interpolation/boundary/extrapolation protocol. B.12's untouched population had already shown this protocol spans conservative, boundary, and unsafe outcomes at 313 K, so neither the limit nor action range was tuned to C1 test performance.

B.11/B.12 predict measured temperature. For the physical C1 constraint, evaluator-side `T_true` is recovered exactly by removing the saved 0.10 K measurement draw using its frozen RNG seed after reveal. Both measured and physical trajectories are retained; no future trajectory enters prediction or calibration before its prescribed reveal.

## Central results at 95%

| Method | SAFE rate | Joint false-safe | Conditional false-safe | UNRESOLVED |
|---|---:|---:|---:|---:|
| S0 nominal DT | 1.000 | 0.394 | 0.394 | 0.000 |
| S1 raw U3 | 0.390 | 0.000 | 0.000 | 0.442 |
| S2 symmetric B.12 | 0.338 | 0.000 | 0.000 | 0.547 |
| S3 one-sided | 0.338 | 0.000 | 0.000 | 0.493 |

The S3 joint false-safe 95% Wilson interval is [0.000, 0.004]. Its conditional false-safe interval is [0.000, 0.011] over 338 SAFE decisions. This conditional rate is empirical, not conformally guaranteed.

S3 achieved upper whole-trajectory coverage 0.957; the lower envelope achieved 0.928, below the 0.95 target. No false-UNSAFE decision occurred, but lower-envelope calibration should not be described as validated at 95%. S3's mean envelope width was 2.471 K versus 3.539 K for S2. S3 did not increase the SAFE count relative to S2; its gain was sharper envelopes and fewer unresolved decisions (0.493 versus 0.547).

## Regimes

- interpolation: SAFE rate 0.000, joint false-safe 0.000, unresolved 1.000.
- boundary: SAFE rate 0.521, joint false-safe 0.000, unresolved 0.476.
- extrapolation: SAFE rate 0.494, joint false-safe 0.000, unresolved 0.000.

These are pooled-calibration diagnostics; no category-conditional guarantee is claimed.

## Interpretation and predeclared questions

1. Nominal DT unsafe-as-safe behaviour is quantified by S0 joint false-safe=0.394.
2. Frozen latent correction changes it to S1=0.000.
3. Symmetric and one-sided conformalization change it to S2=0.000 and S3=0.000.
4. One-sided calibration is sharper than symmetric B.12 and resolves more strong-UNSAFE cases, but does not certify more SAFE actions in this population.
5. S3 certifies 0.558 of genuinely safe actions.
6. Among 286 evaluator-defined near-boundary episodes (absolute true margin <=1 K), 0.976 become UNRESOLVED; the overall unresolved rate is 0.493.
7. Empirical conditional false-safe is 0.000.
8. Marginal joint false-safe is 0.000.
9. Tightening 90% to 99% widened S3 mean envelopes from 2.179 to 2.794 K, but the discrete decisions were unchanged at 338 SAFE and zero false-safe at all three levels; unresolved increased from 0.487 to 0.494.
10. Regime results appear above and in Figure C1.9.
11. Multi-action screening found at least one certified SAFE action in 1.000 of states having an actually safe candidate. Final-test online latency median/p95/max was 0.167/0.197/0.308 s. Median filtering/rollout/conformal components were 0.0234/0.0531/0.000154 s. Screening K=1 measured median was 0.144 s; all K-specific sequential and ideal-parallel lower-bound timings are in `screening_latency.csv`.

## Guarantee scope and limitation

Under split-conformal exchangeability over the defined episode population, whole-trajectory upper-envelope miscoverage is marginally controlled. Because `SAFE and actually unsafe` is a subset of upper-envelope failure, this supports a marginal population bound on joint false-safe probability. It does **not** guarantee that every selected action has safety probability at least 1-alpha, nor does it guarantee the conditional false-safe rate. Usefulness is strongly regime-dependent: all 334 interpolation actions were UNRESOLVED despite 63.5% being physically safe. The lower-envelope coverage shortfall, selective/conditional validity, finite regime samples, and simulator-to-deployment population match are the strongest limitations. The optional LLM Action Agent experiment was not run; it is not part of the statistical guarantee or core C1 result.

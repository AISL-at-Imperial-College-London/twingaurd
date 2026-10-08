# CSTR paper claims audit

| Candidate claim | Status | Frozen evidence and permitted wording |
|---|---|---|
| Fault identity is not required. | **SUPPORTED WITH QUALIFICATION** | The operational feature sets exclude fault labels, magnitudes, changed parameters, hidden plant states, and future truth. C1.6B remains fault-blind across UA fouling and feed-temperature steps. This supports fault-blind operation for the two simulated mechanisms, not universal fault invariance. |
| The nominal DT can be unsafe after faults. | **SUPPORTED** | Nominal-DT joint false-safe probability is 0.394 in C1 and 0.442 in C1.6B. |
| Latent discrepancy modelling improves counterfactual prediction. | **SUPPORTED** | On the frozen B.9 final assessment, k=1 median recursive RMSE is 0.680 K, compared with 2.582 K for the preceding observable-conditioned F1 result. |
| Raw probabilistic uncertainty is calibrated. | **NOT SUPPORTED** | B.10 raw U0 95% simultaneous coverage is 0.525 and its go/no-go verdict is under-conservative. |
| Parameter uncertainty fully solves calibration. | **NOT SUPPORTED** | B.11 U3 improves 95% simultaneous coverage to 0.617 but remains far below 0.95; the frozen verdict is only partially sufficient. |
| Conformal calibration restores useful marginal trajectory coverage. | **SUPPORTED WITH QUALIFICATION** | B.12 B2 reaches 0.934 coverage (Wilson 0.909–0.953); C1 and C1.6B upper coverage are 0.957 and 0.948. The guarantee is marginal over the exchangeable episode population. |
| Every certified action has at least 95% individual safety probability. | **NOT SUPPORTED** | Split conformal coverage is not a conditional per-action probability guarantee. The observed conditional false-safe rate cannot establish this statement. |
| No false-safe events were observed in the frozen C1 test. | **SUPPORTED WITH QUALIFICATION** | C1 observed 0/1000 joint false-safe events (Wilson upper bound 0.0038). C1.6B also observed 0/600 (upper bound 0.0064). These are finite empirical results, not proof of zero risk. |
| Strong UNSAFE certification is validated. | **NOT SUPPORTED** | Lower-envelope coverage is 0.928 in C1 and the primary method is SAFE versus NOT CERTIFIED. Strong UNSAFE claims are outside the validated primary result. |
| Performance is identical across action regimes. | **NOT SUPPORTED** | C1 interpolation SAFE rate is 0 while boundary and extrapolation rates are 0.521 and 0.494. C1.5A confirms interpolation is close to the boundary and has 0.916 upper coverage. |
| The method repairs the DT. | **NOT SUPPORTED** | The nominal DT is never modified. The method estimates discrepancy and selectively revalidates counterfactual rollouts. |
| The method revalidates individual DT rollouts. | **SUPPORTED WITH QUALIFICATION** | It issues an action-specific SAFE decision from an action-specific envelope, prospectively. Statistical validity remains population-marginal rather than an individual conditional guarantee. |
| Multiple candidate actions improve operational usefulness. | **SUPPORTED WITH QUALIFICATION** | In C1.5C, conditional screening success rises from 0.714 at K=1 to 1.000 at K=20 and the probability of any certified action rises from 0.400 to 1.000. The evidence comprises 100 simulated decision states. |
| The result generalizes across distinct physical fault mechanisms. | **SUPPORTED WITH QUALIFICATION** | C1.6B balances UA fouling and feed-temperature steps and obtains 0.948 pooled coverage with no observed false-safe events. Only two simulated fault families were tested, and both high-severity subgroups undercover. |

## Audit rule

`SUPPORTED` denotes a direct frozen empirical result within its tested population. `SUPPORTED WITH QUALIFICATION` requires the stated population, finite-sample, or marginal-validity qualifier. `NOT SUPPORTED` means the manuscript must not make the claim affirmatively.

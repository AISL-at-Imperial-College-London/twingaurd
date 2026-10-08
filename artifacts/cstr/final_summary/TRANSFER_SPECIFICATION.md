# Transfer specification

## Process-independent methodology

1. **Nominal-DT discrepancy formulation.** Run a process-specific nominal digital twin under the observed action sequence and model the measured plant-minus-DT output discrepancy. The nominal DT remains unchanged.
2. **Fault-blind latent inference.** Filter a low-dimensional latent discrepancy state from post-event residual observations and operationally available covariates. Physical fault identity, magnitude, changed parameters, hidden states, and future plant outputs are forbidden inputs.
3. **Parameter ensemble.** Resample complete training episodes, refit the already selected model class, reject fits only under predeclared convergence/stability rules, and propagate both latent/process uncertainty and between-fit parameter uncertainty.
4. **Trajectory prediction.** For each proposed action, generate coherent future discrepancy paths, add them to the nominal-DT trajectory, and retain a predeclared center and horizon-dependent scale.
5. **Conformal trajectory calibration.** On an independent calibration population, compute one score per complete trajectory, `S_i+ = max_h[(y_i,h - m_i,h)/max(s_i,h, epsilon)]`, and use the finite-sample corrected order statistic.
6. **SAFE certification.** Construct `U_conf(h) = m(h) + q+ s(h)` and certify SAFE only if `max_h U_conf(h)` is below the process safety limit. All other outcomes are NOT CERTIFIED unless a separate lower-envelope method has independently been validated.
7. **Prospective evaluation.** Freeze the model, ensemble, score, quantile, threshold, and decision rule; then execute observations → prediction artifact → decision artifact → truth reveal on untouched test episodes.
8. **Reporting.** Report marginal trajectory coverage, joint and conditional false-safe rates with binomial intervals, abstention/certification, subgroup diagnostics without subgroup-guarantee claims, and online latency.

## Final CSTR-specific instance

These values document the frozen CSTR case and must not be copied blindly to another process.

| Quantity | Frozen CSTR value |
|---|---|
| Process | Controlled continuous stirred-tank reactor with material and energy balances implemented in `cstr_digital_twin.py` |
| Manipulated recovery input | Inlet-flow setpoint |
| Operational action range | 0.012–0.045 L/s in the screening experiment |
| Predicted/safety output | Physical reactor temperature |
| Safety threshold | `T_max = 313 K` over the full horizon |
| Sampling interval | 1 s |
| Post-fault observation count | `N = 50` |
| Prediction horizon | 240 steps (240 s) |
| Final fault families | UA fouling and feed-temperature step |
| Observable features | `[u_t, delta_u_t, nominal_DT_temperature_t, elapsed_since_detection_t]` |
| Latent dimension | `k = 1` |
| Bootstrap/prediction | 50 whole-episode bootstrap fits; online `J = 10`; `M = 2000` trajectories |
| Multi-fault conformal quantile | `q+ = 6.6785682210`, learned from 400 new multi-fault calibration episodes |
| Original single-fault C1 quantile | `q+ = 5.4645575869`, retained only for the original C1/C1.5C evidence |
| Predictive scale floor | `epsilon = 1e-6 K` |

### Multi-fault fitted state-space parameters

The frozen final multi-fault k=1 fit is:

- `A = [[0.9949994284450694]]`
- `B = [[0.0012041693335767833, -0.0025391358695666787, -0.00017198967118078347, -0.00507525827977644]]`
- `C = [[2.898525724853236]]`
- `D = [[0.011745730430564205, 0.001239175349253177, 0.026548853581278564, 1.2278636358897201]]`
- `Q = [[0.0003190307040162892]]`
- `R = 0.008284510980519473`
- initial latent mean `0.7268905070043462`
- initial covariance `[[0.000573777862921232]]`

Training-only feature means are `[0.02782602555, -0.00004748397, 309.1502624, 60.5]`; feature scales are `[0.006390721941, 0.002875422008, 0.5989835839, 34.63981332]` in the feature order above.

## Required transfer actions

A new process requires its own equations and nominal DT, operational feature audit, action and safety bounds, sampling interval, training/selection/calibration/final populations, fitted parameters, bootstrap ensemble, predictive scale assessment, and conformal quantile. The CSTR numerical parameters, feature normalization, `q+`, action bounds, and 313 K threshold are not transferable constants.

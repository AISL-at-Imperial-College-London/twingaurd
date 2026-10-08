# Flash E4 Bootstrap and Conformal Calibration

## Frozen Inputs and Scope

E4 uses the 667 complete E3 training episodes for whole-episode bootstrap,
the 143 complete E3 selection operational inputs for J selection, and only the
E2 ordinary/boundary calibration seed blocks for conformal scores. No selection
truth, final, stress, or rare-event truth is read. The frozen mean is the E3
`k=3` `NO_MODEL_PASSED_PREDECLARED_FALLBACK`; calibration does not repair its
documented mean-bias limitation.

## Bootstrap Parameter Uncertainty

Accepted 100 complete-episode bootstrap fits from
150 numerical attempts. Rejections were based only on
convergence, covariance positivity, finiteness, and spectral radius below 0.995.
No predictive-performance or test-data rejection occurred. Rejection counts:
`{"not_converged": 47, "target_already_filled": 3}`.

## Ensemble Selection

Normalized order is pressure / level / temperature. The smallest candidate
passing center-difference <=0.05 and scale-ratio 0.95-1.05 is selected.

| J | Median normalized center difference vs J100 | Median scale ratio vs J100 | Result |
|---:|---:|---:|---|
| 10 | 0.0120 / 0.0073 / 0.0098 | 0.9840 / 0.9859 / 0.9810 | PASS |
| 25 | 0.0092 / 0.0057 / 0.0074 | 1.0062 / 0.9783 / 1.0059 | PASS |
| 50 | 0.0053 / 0.0047 / 0.0046 | 1.0062 / 0.9940 / 1.0059 | PASS |
| 100 | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 1.0000 / 1.0000 | PASS |

Selected `J=10`. Every prediction uses `M=2000` coherent paths;
each path retains one bootstrap model for all 120 horizons and propagates
correlated latent/output noise.

## Pooled Whole-Trajectory Calibration

One score maximizes high pressure, high temperature, high level, and low level
over the full horizon. Quantiles use `ceil((n+1)(1-alpha))`.

| Calibration population | alpha | Complete N | Rank | q |
|---|---:|---:|---:|---:|
| calibration_ordinary | 0.10 | 447 | 404 | 9.291741 |
| calibration_ordinary | 0.05 | 447 | 426 | 11.048599 |
| calibration_ordinary | 0.01 | 447 | 444 | 15.956864 |
| calibration_boundary | 0.10 | 463 | 418 | 9.022876 |
| calibration_boundary | 0.05 | 463 | 441 | 11.349921 |
| calibration_boundary | 0.01 | 463 | 460 | 14.164733 |

Reserved-to-complete counts are ordinary 600->447
and boundary 600->463. Domain exits are
retained in the evaluator manifest and never replaced. Ordinary and nominal-DT
boundary quantiles remain separate as E2 required.

## Predictive Scale Diagnostics

| Population | Median sigma_P [bar] | Median sigma_L | Median sigma_T [K] | Mean floor-hit fraction P/L/T |
|---|---:|---:|---:|---:|
| calibration_ordinary | 0.021277 | 0.014941 | 0.927773 | 0.0004 / 0.0164 / 0.0013 |
| calibration_boundary | 0.021315 | 0.015098 | 0.927848 | 0.0000 / 0.0133 / 0.0002 |

Scale floors are 0.005 bar, 0.005 level fraction, and 0.25 K.

## Action-Conditional Extension

`d(a)=||(a-a_nominal)/(a_max-a_min)||_2` is implemented using only the candidate
action. However, E2 did not declare Mondrian bin edges or a minimum group sample
count. Therefore no group was formed and no group-specific q was estimated.
The secondary method is frozen as `UNSUPPORTED_NOT_CERTIFIED` with reason
`E2_BINS_NOT_PREDECLARED`; inventing bins after calibration would violate the
preregistration.

## Decision Semantics

For a supported pooled quantile, certify `SAFE` only when `P_upper<=0.90 bar`,
`T_upper<=365 K`, `L_upper<=0.85`, and `L_lower>=0.20` at every horizon. All
other cases, including unsupported Mondrian groups, return `NOT CERTIFIED`.
No UNSAFE certificate exists. E4 contains no test result or safety-performance
claim.

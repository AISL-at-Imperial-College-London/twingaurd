# Flash E3 Model Selection

## Scope and Leakage Boundary

Only the E2 `training` (reserved N=900; complete N=667)
and `model_selection` (reserved N=180; complete N=143)
seed blocks were generated or accessed. Calibration, final, boundary, structural-stress, and
rare-event populations were not generated or read. The nominal process bundle
and E2 preregistration hashes were verified before generation.

The fitted arrays contain only residual `[P,L,T]` and the 11 observable features:
current action, action change from nominal, and nominal-DT `[P,L,T]`. Evaluator
fault metadata is separated and never passed to fitting or prediction.

EM uses the unchanged relative-likelihood convergence tolerance `1e-5`. The
uniform iteration ceiling is 100: the initial 40-iteration execution produced
no converged candidate, so computation was continued for every k without
changing a gate, initialization, population, action, or convergence tolerance.

### Declared numerical-domain exits

233 training and 37
selection seeds left the frozen E1 property/vessel domain under their reserved
fault/action combinations. They were recorded in the evaluator manifest and
were not clipped, extrapolated, replaced, or regenerated. Only complete
trajectories enter identification and RMSE calculations. This is an unplanned
E2 population-support limitation, not a gate relaxation.

## Exact Gate Results

Normalized output order below is pressure / level / temperature.

| k | Median worst normalized RMSE | Per-output median normalized RMSE | Normalized bias | Innovation lag-1 | Spectral radius | Gate result |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.5295 | 0.2179 / 0.2472 / 0.2709 | -0.2122 / -0.4866 / -0.1980 | 0.7870 / 0.9930 / 0.8527 | 0.99500 | FAIL: worst_normalized_rmse, each_output_normalized_bias, each_output_innovation_autocorrelation |
| 2 | 0.4493 | 0.2164 / 0.2244 / 0.2733 | -0.1974 / -0.2805 / -0.1807 | 0.7985 / -0.1405 / 0.8676 | 0.99500 | FAIL: converged_finite, each_output_normalized_bias, each_output_innovation_autocorrelation |
| 3 | 0.4189 | 0.1927 / 0.2293 / 0.1903 | -0.2229 / -0.2790 / -0.1859 | 0.3449 / -0.1432 / 0.2765 | 0.99500 | FAIL: each_output_normalized_bias |

All candidates were numerically finite with positive covariance matrices and
finite 120-step counterfactual rollouts. Failed candidates: 1, 2, 3.

## Frozen Selection

### NO MODEL PASSED

Selected `k=3` with status `NO_MODEL_PASSED_PREDECLARED_FALLBACK`. Its median engineering-unit RMSE
is 0.019266 bar pressure,
0.022933 level fraction, and
0.951325 K temperature. The fitted latent
coordinates are statistical discrepancy memory only and have no physical fault
interpretation.

## Counterfactual Verification

The selected model's two pre-truth action probes change the nominal rollout by
1.58619 in the largest raw output
coordinate and change the predicted discrepancy rollout by
0.404239. Prediction APIs do
not accept fault metadata or future truth, and repeated predictions are exact
under fixed inputs.

## Transfer Limitations

The model is linear-Gaussian with stable diagonal latent dynamics and is tested
only on F1-F3 within the E2 training/selection ranges. Selection diagnostics do
not establish calibrated uncertainty, SAFE performance, fault-conditional
validity, boundary reliability, or structural-mismatch robustness. Composition
is not measured and is represented only indirectly through observable residual
history. No calibration or final-test claim is made in E3.

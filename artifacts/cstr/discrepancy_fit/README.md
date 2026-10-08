# Milestone B.9 — Latent discrepancy state-space model

## Architecture and separation

A separate library of **600** simulator episodes, each with 120 post-fault observations, was generated for offline identification. Every episode resets its latent mean and covariance. Fault metadata appears only in the evaluator manifest; the estimator receives residuals and `[u, delta_u, nominal_DT_temperature, elapsed_since_detection]` only. Shared `A,B,C,D,Q,R` parameters are frozen before any B.5 evaluation episode is filtered.

Two deterministic EM initializations were evaluated for each `k in {1,2,3,4}`, using Kalman-filter likelihood summed across independent episodes. Dynamics use the stable parameterization `a_i=0.995*tanh(theta_i)` throughout optimization. No eigenvalue was clipped. The diagonal-mode convention improves numerical identifiability, but latent coordinates remain abstract and must not be interpreted physically or compared across k.

Continuous exogenous features are standardized using the offline training library only. Initialization uses OLS for the direct observation term, stable modes spaced from 0.25 to 0.85 with deterministic seed jitter, small Gaussian input coefficients, `C_i=1/k`, `Q=0.08*s_residual^2 I`, `R=0.5*s_residual^2`, initial mean zero, and initial covariance `s_residual^2 I`. EM uses at most 15 iterations and relative likelihood tolerance `1e-4`; each stable diagonal dynamics row is optimized with L-BFGS-B in its unconstrained theta parameterization.

## Selection

Selection-only recursive RMSE values were k=1: 0.530 K, k=2: 0.662 K, k=3: 0.672 K, k=4: 0.671 K. Using only the existing 60 selection episodes, the smallest dimension within 5% of the best recursive RMSE whose innovation ACF was not materially worse was **k=1**. The primary final assessment then used the 240 untouched episode IDs once, after k and all parameters were frozen. The separate N experiment is a predeclared fixed-model diagnostic and does not feed back into selection.

## Final result

The selected latent model has median recursive RMSE **0.680 K**, mean RMSE **0.847 K**, median MAE **0.594 K**, innovation bias **+0.012 K**, and maximum innovation ACF **0.173**. Its spectral radius is **0.9950** with stability margin **0.0050**.

Optimization limitation: the selected initialization reached the 15-iteration cap with `converged=False` under the `1e-4` likelihood criterion, and its slow mode lies near the stable parameterization ceiling. The threshold-based final classification below is therefore valid for this frozen fitted candidate, but a training-only sensitivity run with a higher iteration cap is warranted before treating the parameter estimates as numerically definitive. This does not authorize refitting against final-assessment performance.

B.8 F1 has median recursive RMSE **2.582 K**, bias **-1.157 K**, and maximum ACF **0.994**. B.7B ARX(1) has median recursive RMSE **3.030 K**. The innovation comparison is descriptive because the latent value is an online Kalman innovation over N=50 whereas the frozen ARX/F1 values are held-out teacher-forced innovations over H=240.

The Ljung–Box diagnostic for selected pooled final innovations is Q=487.67, p=1.9e-98; standardized innovation variance is 0.926. The maximum ACF is substantially reduced but the large-sample Ljung–Box result still rejects literal whiteness. This is a dependence diagnostic, not a final calibration test.

## Online data efficiency and runtime

Fixed-model performance was N=10: 0.668 K, N=20: 0.641 K, N=50: 0.680 K, N=100: 0.804 K. Each N branch acts from its own post-fault decision time, so the sequence is a deployment-timing experiment rather than a monotone information-only ablation. Median N=50 filtering time is **0.003995 s** and median 2,000-trajectory open-loop propagation time is **0.0435 s**. Offline fitting took **227.79 s** for the selected initialization and is not an online cost. The largest analytical-versus-Monte-Carlo mean discrepancy across final episodes was **0.052 K**.

## Answers

1. Open-loop improvement: median RMSE reduction versus F1 is 73.7%.
2. Smallest useful latent dimension: **k=1** under the frozen selection rule.
3. Innovation ACF changes from 0.994 to **0.173**.
4. Horizon drift is quantified against both frozen baselines in `horizon_error_summary.csv` and Figures B9.4–B9.5.
5. N=10–50 inference is reported above and in `data_efficiency_summary.csv`.
6. Online Kalman filtering is computationally lightweight at the measured runtime.
7. The direct comparison with F1 and ARX(1) is in `baseline_comparison.csv`.
8. Readiness: The linear latent mean model meets every predeclared sufficiency gate and can proceed to a separate uncertainty-modelling milestone.

## Decision

**LATENT STATE SUFFICIENT**

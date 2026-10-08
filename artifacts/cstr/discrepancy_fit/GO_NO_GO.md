# Milestone B.9 go/no-go

## Verdict

**LATENT STATE SUFFICIENT**

Against frozen B.8 F1, recursive RMSE reduction is 73.7%, absolute-bias reduction is 98.9%, and maximum-ACF reduction is 82.6%. Final maximum ACF is 0.173; spectral radius is 0.9950.

The `LATENT STATE SUFFICIENT` threshold requires all five predeclared conditions: 20% RMSE reduction, 30% absolute-bias reduction, 50% ACF reduction, final ACF at most 0.30, and stable non-explosive dynamics. The linear latent mean model meets every predeclared sufficiency gate and can proceed to a separate uncertainty-modelling milestone.

Numerical caveat: the selected EM run reached its iteration cap without the declared likelihood tolerance, and its slow mode is near the stability ceiling. The verdict is the mandatory threshold classification for the frozen candidate, not evidence that latent coordinates have physical meaning or that the optimizer sensitivity question is closed.

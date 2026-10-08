from __future__ import annotations

import numpy as np

from twinguard.cstr.freeze_discrepancy_model import (
    VarianceModel,
    calibration_classification,
    fit_variance_models,
    summarize_calibration,
)


def test_global_variance_recalibration_uses_standardized_innovations() -> None:
    rng=np.random.default_rng(4);standardized=rng.normal(0,2,size=4000);arrays={"standardized":standardized,"state_abs":rng.random(4000),"delta_abs":rng.random(4000),"support":rng.random(4000)};diagnostics={name:{"rho":0.,"variance_ratio":1.} for name in ("state_abs","delta_abs","support")};models,metadata=fit_variance_models(arrays,diagnostics)
    assert [model.name for model in models]==["U0","U1"]
    assert 3.5<metadata["U1_global_scale"]<4.5


def test_u2_is_only_enabled_by_predeclared_conditional_structure_gate() -> None:
    rng=np.random.default_rng(5);arrays={"standardized":rng.normal(size=200),"state_abs":rng.random(200),"delta_abs":rng.random(200),"support":rng.random(200)};diagnostics={"state_abs":{"rho":.2,"variance_ratio":1.5},"delta_abs":{"rho":.01,"variance_ratio":2.},"support":{"rho":.2,"variance_ratio":1.1}};models,metadata=fit_variance_models(arrays,diagnostics)
    assert [model.name for model in models]==["U0","U1","U2"]
    assert metadata["U2_features"]==["state_abs"]


def test_calibration_summary_does_not_mix_pointwise_and_simultaneous_coverage() -> None:
    rows=[]
    for episode in range(12):
        for level in (.5,.8,.9,.95,.99):rows.append({"episode_id":str(episode),"action_category":"interpolation","uncertainty_model":"U0","nominal_level":level,"pointwise_coverage":level,"simultaneous_coverage":0,"pointwise_average_width_k":1.,"pointwise_median_width_k":1.,"simultaneous_average_width_k":2.,"rollout_RMSE_k":.5})
    summary,comparison=summarize_calibration(rows,50);pooled=[r for r in summary if r["action_category"]=="pooled"]
    assert all(np.isclose(r["pointwise_empirical_coverage"],r["nominal_level"]) for r in pooled)
    assert all(r["simultaneous_empirical_coverage"]==0 for r in pooled)


def test_variance_model_scaling_keeps_mean_unchanged() -> None:
    model=VarianceModel("U1",2.5);np.testing.assert_allclose(model.scale({},7),np.full(7,2.5))

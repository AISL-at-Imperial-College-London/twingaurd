from __future__ import annotations

import numpy as np

from twinguard.cstr.conformal_calibration import corrected_quantile,smooth_horizon_scale,trajectory_scores


def test_corrected_split_conformal_rank_is_used() -> None:
    scores=np.arange(1,101,dtype=float);q,k=corrected_quantile(scores,.95)
    assert k==96
    assert q==96


def test_corrected_quantile_is_infinite_when_requested_rank_exceeds_sample() -> None:
    q,k=corrected_quantile(np.arange(50.),.99)
    assert k==51 and np.isinf(q)


def test_primary_score_treats_trajectory_as_one_object() -> None:
    actual=np.array([0.,2.,1.]);center=np.zeros(3);scale=np.array([1.,2.,.5]);absolute,standardized=trajectory_scores(actual,center,scale)
    assert absolute==2.
    assert standardized==2.


def test_horizon_scale_is_smoothed_positive_and_normalized() -> None:
    values=np.tile(np.linspace(.5,2.,240),(20,1));scale=smooth_horizon_scale(values)
    assert scale.shape==(240,)
    assert np.all(scale>0)
    assert np.isclose(np.median(scale),1.)

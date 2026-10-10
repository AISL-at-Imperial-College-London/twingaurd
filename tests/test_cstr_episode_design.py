import numpy as np
import pytest

from twinguard.cstr.episode_design import (
    FROZEN_FEATURES,
    PREDICTION_KEYS_WITHOUT_TRUTH,
    EpisodeDesignConfig,
    _select_action_after_fit,
    action_support_distance,
    construct_simultaneous_bands,
    frozen_model_configuration,
    wilson_interval,
)


@pytest.mark.skip(reason="documents the superseded -B Bayesian residual model; hashes development-only source files by working-directory-relative name (also fails in the development repository)")
def test_episode_design_freezes_existing_m4_model_and_numerical_settings() -> None:
    config = EpisodeDesignConfig(episodes_per_category=1)
    config.validate()
    frozen = frozen_model_configuration(config)

    assert frozen["model_name"] == "M4_full_bayesian"
    assert tuple(frozen["feature_order_without_intercept"]) == FROZEN_FEATURES
    assert frozen["prior"]["prior_precision"] == 0.01
    assert frozen["prior"]["prior_alpha"] == 2.0
    assert frozen["prior"]["prior_beta"] == 0.25
    assert frozen["rollout_horizon_steps"] == 240
    assert frozen["calibration_experiment_M"] == 2000


def test_prediction_snapshot_schema_cannot_contain_truth() -> None:
    lowered = {name.lower() for name in PREDICTION_KEYS_WITHOUT_TRUTH}
    assert not any("actual" in name or "truth" in name or "plant" in name for name in lowered)


def test_simultaneous_band_contains_requested_fraction_of_posterior_paths() -> None:
    rng = np.random.default_rng(8)
    paths = np.cumsum(rng.normal(size=(500, 25)), axis=1)
    bands = construct_simultaneous_bands(paths, (0.50, 0.80, 0.95))

    for level in (0.50, 0.80, 0.95):
        inside = np.all(
            (paths >= bands.lower[level][None, :])
            & (paths <= bands.upper[level][None, :]),
            axis=1,
        )
        assert np.mean(inside) >= level
        assert np.mean(inside) <= level + 0.02


def test_wilson_interval_is_finite_at_boundary_rates() -> None:
    zero = wilson_interval(0, 100)
    one = wilson_interval(100, 100)
    assert zero[0] == 0.0
    assert 0.0 < zero[1] < 0.1
    assert 0.9 < one[0] < 1.0
    assert one[1] == 1.0


def test_heldout_action_generators_respect_categories_and_support_distance() -> None:
    rng = np.random.default_rng(99)
    interpolation = [_select_action_after_fit("interpolation", rng) for _ in range(30)]
    boundary = [_select_action_after_fit("boundary", rng) for _ in range(30)]
    extrapolation = [_select_action_after_fit("extrapolation", rng) for _ in range(30)]

    assert all(0.023 < value < 0.032 for value in interpolation)
    assert all(value < 0.023 or value > 0.032 for value in boundary)
    assert all(value <= 0.018 or value >= 0.038 for value in extrapolation)
    assert np.mean([action_support_distance(value) for value in extrapolation]) > np.mean(
        [action_support_distance(value) for value in interpolation]
    )

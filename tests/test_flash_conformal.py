from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from twinguard.uncertainty.flash_bootstrap import coherent_predictive_summary
from twinguard.conformal.flash_conformal import (
    conformal_envelope,
    corrected_quantile,
    mondrian_group,
    safety_decision,
    safety_direction_score,
)
from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.discrepancy.flash_state_space import FEATURE_NAMES, FlashStateSpaceParameters
from twinguard.scenarios.flash_population_design import MEASURED_OUTPUTS, SafetySpecification
from twinguard.flash.bootstrap_and_calibrate import batch_nominal_dt_margins


def _model() -> FlashStateSpaceParameters:
    twin = FlashDigitalTwin()
    action = twin.nominal_action().as_array()
    output = twin.outputs(twin.nominal_state())
    feature_mean = np.concatenate(
        [action, np.zeros(4), [output[name] for name in MEASURED_OUTPUTS]]
    )
    return FlashStateSpaceParameters(
        A=np.asarray([[0.8]]),
        B=np.zeros((1, len(FEATURE_NAMES))),
        C=np.asarray([[0.3], [0.2], [0.5]]),
        D=np.zeros((3, len(FEATURE_NAMES))),
        Q=np.asarray([[0.02]]),
        R=np.diag([0.02, 0.02, 0.02]),
        initial_mean=np.zeros(1),
        initial_covariance=np.eye(1) * 0.1,
        feature_mean=feature_mean,
        feature_scale=np.ones(len(FEATURE_NAMES)),
        residual_scale=np.asarray([0.10, 0.10, 5.0]),
        log_likelihood=0.0,
        iterations=1,
        converged=True,
        initialization_seed=1,
        fit_seconds=0.0,
        final_parameter_change=0.0,
    )


def test_one_score_covers_all_four_safety_directions() -> None:
    center = np.tile([0.6, 0.5, 350.0], (4, 1))
    scale = np.tile([0.01, 0.02, 0.5], (4, 1))
    cases = (
        ([0.63, 0.5, 350.0], "high_pressure"),
        ([0.6, 0.5, 352.0], "high_temperature"),
        ([0.6, 0.57, 350.0], "high_level"),
        ([0.6, 0.43, 350.0], "low_level"),
    )
    for values, direction in cases:
        actual = center.copy()
        actual[2] = values
        score, maximizing_direction, horizon = safety_direction_score(
            actual, center, scale
        )
        assert score > 0.0
        assert maximizing_direction == direction
        assert horizon == 2


def test_corrected_quantile_uses_finite_sample_rank() -> None:
    scores = np.arange(1.0, 101.0)
    quantile, rank = corrected_quantile(scores, 0.05)
    assert rank == 96
    assert quantile == 96.0


def test_safe_decision_requires_every_envelope_direction() -> None:
    center = np.tile([0.70, 0.50, 350.0], (120, 1))
    scale = np.tile([0.01, 0.01, 0.5], (120, 1))
    assert safety_decision(center, scale, 2.0) == "SAFE"
    for output, value in ((0, 0.89), (1, 0.84), (1, 0.21), (2, 364.5)):
        changed = center.copy()
        changed[30, output] = value
        assert safety_decision(changed, scale, 2.0) == "NOT CERTIFIED"
    assert safety_decision(center, scale, None, supported=False) == "NOT CERTIFIED"


def test_no_mondrian_group_is_invented_without_population_design_edges() -> None:
    assert mondrian_group(FlashDigitalTwin().nominal_action().as_array(), None) is None


def test_coherent_predictive_paths_are_seeded_and_keep_horizon_structure() -> None:
    model = _model()
    features = np.broadcast_to(model.feature_mean, (50, len(FEATURE_NAMES))).copy()
    residual = np.zeros((50, 3))
    future_features = np.broadcast_to(model.feature_mean, (120, len(FEATURE_NAMES))).copy()
    nominal = np.tile([0.6, 0.6, 350.0], (120, 1))
    models = [model] * 10
    first = coherent_predictive_summary(
        models,
        residual,
        features,
        future_features,
        nominal,
        ensemble_size=10,
        trajectory_count=2000,
        seed=123,
        retain_trajectories=True,
    )
    second = coherent_predictive_summary(
        models,
        residual,
        features,
        future_features,
        nominal,
        ensemble_size=10,
        trajectory_count=2000,
        seed=123,
        retain_trajectories=True,
    )
    np.testing.assert_array_equal(first.trajectories, second.trajectories)
    assert first.trajectories.shape == (2000, 120, 3)
    assert abs(np.corrcoef(first.trajectories[:, 10, 0], first.trajectories[:, 11, 0])[0, 1]) > 0.05


def test_vectorized_boundary_margins_match_frozen_scalar_dt() -> None:
    twin = FlashDigitalTwin()
    nominal = twin.nominal_action()
    actions = np.stack(
        [
            nominal.as_array(),
            replace(nominal, heat_duty_w=18_000.0).as_array(),
            replace(nominal, liquid_valve_opening=0.65, feed_flow_mol_s=0.85).as_array(),
        ]
    )
    vectorized = batch_nominal_dt_margins(twin.nominal_state().as_array(), actions)
    safety = SafetySpecification()
    scalar = []
    for action in actions:
        trajectory = FlashDigitalTwin().simulate(
            action, 1200.0, initial_state=twin.nominal_state(), sample_interval_s=10.0
        )
        future = {name: trajectory[name][1:] for name in MEASURED_OUTPUTS}
        scalar.append(safety.trajectory_margin(future))
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-10, atol=1e-10)

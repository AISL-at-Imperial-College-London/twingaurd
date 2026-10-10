"""Whole-episode bootstrap and coherent flash predictive trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from twinguard.discrepancy.flash_state_space import (
    FlashStateSpaceParameters,
    kalman_filter,
)


ENSEMBLE_CANDIDATES = (10, 25, 50, 100)
PREDICTIVE_TRAJECTORIES = 2000
PREDICTIVE_SCALE_FLOORS = np.asarray([0.005, 0.005, 0.25])


@dataclass(frozen=True)
class PredictiveSummary:
    center: np.ndarray
    scale: np.ndarray
    trajectories: np.ndarray | None = None


def coherent_predictive_summary(
    models: Sequence[FlashStateSpaceParameters],
    observed_residual: np.ndarray,
    observed_features: np.ndarray,
    future_features: np.ndarray,
    nominal_future: np.ndarray,
    *,
    ensemble_size: int,
    trajectory_count: int = PREDICTIVE_TRAJECTORIES,
    seed: int,
    retain_trajectories: bool = False,
) -> PredictiveSummary:
    """Sample coherent paths by keeping one model for every path and horizon."""

    if ensemble_size not in ENSEMBLE_CANDIDATES:
        raise ValueError("ensemble size is not an population design-predeclared candidate")
    if len(models) < ensemble_size:
        raise ValueError("insufficient accepted bootstrap models")
    if trajectory_count % ensemble_size:
        raise ValueError("trajectory count must divide evenly across ensemble models")
    per_model = trajectory_count // ensemble_size
    future_count = len(future_features)
    paths = np.empty((trajectory_count, future_count, 3))
    rng = np.random.default_rng(seed)
    offset = 0
    for model in models[:ensemble_size]:
        filtered = kalman_filter(model, observed_residual, observed_features)
        state = rng.multivariate_normal(
            filtered.filtered_mean[-1],
            filtered.filtered_covariance[-1],
            size=per_model,
        )
        standardized_features = model.standardize_features(future_features)
        for horizon, feature in enumerate(standardized_features):
            process_noise = rng.multivariate_normal(
                np.zeros(model.latent_dimension), model.Q, size=per_model
            )
            state = state @ model.A.T + feature @ model.B.T + process_noise
            observation_noise = rng.multivariate_normal(
                np.zeros(3), model.R, size=per_model
            )
            standardized_residual = (
                state @ model.C.T + feature @ model.D.T + observation_noise
            )
            paths[offset : offset + per_model, horizon] = (
                nominal_future[horizon]
                + standardized_residual * model.residual_scale
            )
        offset += per_model
    center = np.median(paths, axis=0)
    scale = np.std(paths, axis=0, ddof=1)
    return PredictiveSummary(
        center=center,
        scale=scale,
        trajectories=paths if retain_trajectories else None,
    )


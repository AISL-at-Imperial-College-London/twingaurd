"""Whole-trajectory multi-constraint conformal calibration for the flash case."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping, Sequence

import numpy as np

from twinguard.uncertainty.flash_bootstrap import PREDICTIVE_SCALE_FLOORS
from twinguard.scenarios.flash_population_design import SafetySpecification, nominal_and_effective_action_bounds


ALPHA_LEVELS = (0.10, 0.05, 0.01)
PRIMARY_ALPHA = 0.05
DIRECTION_NAMES = (
    "high_pressure",
    "high_temperature",
    "high_level",
    "low_level",
)


@dataclass(frozen=True)
class ConformalEnvelope:
    pressure_upper: np.ndarray
    temperature_upper: np.ndarray
    level_upper: np.ndarray
    level_lower: np.ndarray


def safety_direction_components(
    actual: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    actual_array = np.asarray(actual, dtype=float)
    center_array = np.asarray(center, dtype=float)
    scale_array = np.asarray(scale, dtype=float)
    if not actual_array.shape == center_array.shape == scale_array.shape:
        raise ValueError("actual, center, and scale trajectories must align")
    if actual_array.ndim != 2 or actual_array.shape[1] != 3:
        raise ValueError("flash trajectories must have shape (horizon, 3)")
    denominator = np.maximum(scale_array, PREDICTIVE_SCALE_FLOORS)
    return np.column_stack(
        [
            (actual_array[:, 0] - center_array[:, 0]) / denominator[:, 0],
            (actual_array[:, 2] - center_array[:, 2]) / denominator[:, 2],
            (actual_array[:, 1] - center_array[:, 1]) / denominator[:, 1],
            (center_array[:, 1] - actual_array[:, 1]) / denominator[:, 1],
        ]
    )


def safety_direction_score(
    actual: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[float, str, int]:
    components = safety_direction_components(actual, center, scale)
    flat_index = int(np.argmax(components))
    horizon, direction = np.unravel_index(flat_index, components.shape)
    return float(components[horizon, direction]), DIRECTION_NAMES[direction], int(horizon)


def corrected_quantile(scores: Sequence[float], alpha: float) -> tuple[float, int]:
    values = np.sort(np.asarray(scores, dtype=float))
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a nonempty finite vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    rank = min(int(ceil((len(values) + 1) * (1.0 - alpha))), len(values))
    return float(values[rank - 1]), rank


def conformal_envelope(
    center: np.ndarray,
    scale: np.ndarray,
    quantile: float,
) -> ConformalEnvelope:
    denominator = np.maximum(np.asarray(scale, dtype=float), PREDICTIVE_SCALE_FLOORS)
    center_array = np.asarray(center, dtype=float)
    return ConformalEnvelope(
        pressure_upper=center_array[:, 0] + quantile * denominator[:, 0],
        temperature_upper=center_array[:, 2] + quantile * denominator[:, 2],
        level_upper=center_array[:, 1] + quantile * denominator[:, 1],
        level_lower=center_array[:, 1] - quantile * denominator[:, 1],
    )


def safety_decision(
    center: np.ndarray,
    scale: np.ndarray,
    quantile: float | None,
    *,
    supported: bool = True,
    safety: SafetySpecification | None = None,
) -> str:
    if not supported or quantile is None:
        return "NOT CERTIFIED"
    specification = safety or SafetySpecification()
    envelope = conformal_envelope(center, scale, quantile)
    certified = bool(
        np.all(envelope.pressure_upper <= specification.pressure_max_bar)
        and np.all(envelope.temperature_upper <= specification.temperature_max_k)
        and np.all(envelope.level_upper <= specification.level_max_fraction)
        and np.all(envelope.level_lower >= specification.level_min_fraction)
    )
    return "SAFE" if certified else "NOT CERTIFIED"


def normalized_action_distance(action: Sequence[float]) -> float:
    nominal, lower, upper = nominal_and_effective_action_bounds()
    vector = np.asarray(action, dtype=float)
    if vector.shape != nominal.shape:
        raise ValueError("action has incorrect dimension")
    return float(np.linalg.norm((vector - nominal) / (upper - lower)))


def mondrian_group(
    action: Sequence[float],
    bin_edges: Sequence[float] | None,
) -> str | None:
    """Return no group when calibration bin edges are unavailable."""

    if bin_edges is None:
        return None
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("Mondrian bin edges must be strictly increasing")
    distance = normalized_action_distance(action)
    index = int(np.searchsorted(edges, distance, side="right") - 1)
    if index < 0 or index >= len(edges) - 1:
        return None
    return f"d_[{edges[index]:.6g},{edges[index + 1]:.6g})"


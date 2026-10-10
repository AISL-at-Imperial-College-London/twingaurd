"""Define flash faults, candidate actions, safety limits, and episode populations.

Operational records contain values available before each prediction.
Hidden fault identities and future outcomes stay in evaluator-only records.
Population seeds are separated by training, calibration, and evaluation role.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
from itertools import product
from typing import Mapping, Sequence

import numpy as np

from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.plants.flash_drum import FlashAction, FlashParameters, FlashState, PhysicalStateError


PROCESS_BUNDLE_SHA256 = "1bc228ee61c628bf44795a0f67c9a476973f630dcd95f9932d88c96682325624"
OBSERVATION_COUNT = 50
SAMPLING_INTERVAL_S = 10.0
OBSERVATION_WINDOW_S = OBSERVATION_COUNT * SAMPLING_INTERVAL_S
PREDICTION_HORIZON_S = 1200.0
BOUNDARY_DELTA_NORMALIZED = 0.15
BOUNDARY_LHS_SIZE = 128
BOUNDARY_TOP_RANKS = 8
ACTION_NAMES = (
    "heat_duty_w",
    "vapor_valve_opening",
    "liquid_valve_opening",
    "feed_flow_mol_s",
)
MAXIMUM_ACTION_CHANGE = np.asarray([6000.0, 0.25, 0.25, 0.25])
MEASURED_OUTPUTS = ("pressure_bar", "level_fraction", "temperature_k")
FORBIDDEN_OPERATIONAL_TOKENS = (
    "fault",
    "magnitude",
    "severity",
    "changed_parameter",
    "hidden_state",
    "future_truth",
    "future_trajectory",
    "safety_label",
)


def purpose_rng(episode_seed: int, purpose_code: int) -> np.random.Generator:
    """Create a deterministic purpose-isolated RNG from an evaluator root seed."""

    return np.random.default_rng(np.random.SeedSequence([episode_seed, purpose_code]))


def purpose_seed(episode_seed: int, purpose_code: int) -> int:
    """Return a stable integer seed for APIs that do not accept SeedSequence."""

    sequence = np.random.SeedSequence([episode_seed, purpose_code])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


class FaultFamily(str, Enum):
    VAPOR_OUTLET_DEGRADATION = "F1_vapor_outlet_degradation"
    LIQUID_OUTLET_DEGRADATION = "F2_liquid_outlet_degradation"
    FEED_COMPOSITION_SHIFT = "F3_unmeasured_feed_composition_shift"
    NONIDEAL_VLE_STRESS = "F4_nonideal_vle_structural_stress"


PRIMARY_FAULT_FAMILIES = (
    FaultFamily.VAPOR_OUTLET_DEGRADATION,
    FaultFamily.LIQUID_OUTLET_DEGRADATION,
    FaultFamily.FEED_COMPOSITION_SHIFT,
)


@dataclass(frozen=True)
class EvaluatorFaultTruth:
    """Hidden record that must only be serialized with evaluator truth."""

    family: FaultFamily
    normalized_severity: float
    signed_magnitude: float
    changed_parameter: str
    nominal_value: float
    faulty_value: float
    severity_label: str


@dataclass(frozen=True)
class OperationalEpisodeInput:
    """Strict fault-blind data available before future truth is generated."""

    episode_id: str
    observation_time_s: np.ndarray
    observed_outputs: np.ndarray
    nominal_dt_outputs: np.ndarray
    action_history: np.ndarray
    candidate_action: np.ndarray

    def validate(self) -> None:
        names = {item.name for item in fields(self)}
        for name in names:
            if any(token in name for token in FORBIDDEN_OPERATIONAL_TOKENS):
                raise ValueError(f"forbidden evaluator field in operational record: {name}")
        if self.observation_time_s.shape != (OBSERVATION_COUNT,):
            raise ValueError("operational record must contain exactly 50 observation times")
        if self.observed_outputs.shape != (OBSERVATION_COUNT, len(MEASURED_OUTPUTS)):
            raise ValueError("observed outputs must have shape (50, 3)")
        if self.nominal_dt_outputs.shape != self.observed_outputs.shape:
            raise ValueError("nominal-DT and observed output histories must align")
        if self.action_history.shape != (OBSERVATION_COUNT, len(ACTION_NAMES)):
            raise ValueError("action history must have shape (50, 4)")
        if self.candidate_action.shape != (len(ACTION_NAMES),):
            raise ValueError("candidate action must have shape (4,)")


@dataclass(frozen=True)
class SafetySpecification:
    pressure_max_bar: float = 0.90
    level_min_fraction: float = 0.20
    level_max_fraction: float = 0.85
    temperature_max_k: float = 365.0
    pressure_scale_bar: float = 0.10
    level_scale_fraction: float = 0.10
    temperature_scale_k: float = 5.0

    @classmethod
    def from_parameters(cls, parameters: FlashParameters) -> "SafetySpecification":
        return cls(
            pressure_max_bar=parameters.pressure_safety_max_bar,
            level_min_fraction=parameters.level_safety_min_fraction,
            level_max_fraction=parameters.level_safety_max_fraction,
            temperature_max_k=parameters.temperature_safety_max_k,
        )

    def violation_margins(self, trajectory: Mapping[str, Sequence[float]]) -> np.ndarray:
        pressure = np.asarray(trajectory["pressure_bar"], dtype=float)
        level = np.asarray(trajectory["level_fraction"], dtype=float)
        temperature = np.asarray(trajectory["temperature_k"], dtype=float)
        if not pressure.shape == level.shape == temperature.shape:
            raise ValueError("pressure, level, and temperature trajectories must align")
        return np.column_stack(
            [
                (pressure - self.pressure_max_bar) / self.pressure_scale_bar,
                (level - self.level_max_fraction) / self.level_scale_fraction,
                (self.level_min_fraction - level) / self.level_scale_fraction,
                (temperature - self.temperature_max_k) / self.temperature_scale_k,
            ]
        )

    def trajectory_margin(self, trajectory: Mapping[str, Sequence[float]]) -> float:
        """Return positive normalized safety margin, negative on violation."""

        return -float(np.max(self.violation_margins(trajectory)))

    def is_physically_safe(self, trajectory: Mapping[str, Sequence[float]]) -> bool:
        return bool(self.trajectory_margin(trajectory) >= 0.0)


def severity_label(normalized_severity: float) -> str:
    if not 0.0 <= normalized_severity <= 1.0:
        raise ValueError("normalized severity must lie in [0, 1]")
    if normalized_severity < 1.0 / 3.0:
        return "low"
    if normalized_severity < 2.0 / 3.0:
        return "medium"
    return "high"


def sample_primary_fault(family: FaultFamily, seed: int) -> EvaluatorFaultTruth:
    """Sample a primary hidden fault without modifying the nominal DT."""

    if family not in PRIMARY_FAULT_FAMILIES:
        raise ValueError("F4 is reserved for structural stress and is not a primary fault")
    rng = purpose_rng(seed, 211)
    severity = float(rng.uniform(0.0, 1.0))
    p = FlashParameters()
    if family is FaultFamily.VAPOR_OUTLET_DEGRADATION:
        loss = 0.10 + 0.35 * severity
        nominal = p.vapor_valve_coefficient_mol_s_bar_sqrt
        faulty = nominal * (1.0 - loss)
        return EvaluatorFaultTruth(
            family, severity, loss, "vapor_valve_coefficient_mol_s_bar_sqrt",
            nominal, faulty, severity_label(severity)
        )
    if family is FaultFamily.LIQUID_OUTLET_DEGRADATION:
        loss = 0.10 + 0.35 * severity
        nominal = p.liquid_valve_coefficient_mol_s
        faulty = nominal * (1.0 - loss)
        return EvaluatorFaultTruth(
            family, severity, loss, "liquid_valve_coefficient_mol_s",
            nominal, faulty, severity_label(severity)
        )

    shift_magnitude = 0.04 + 0.08 * severity
    # Sequential reserved F3 seeds alternate parity, giving exact sign balance.
    sign = -1.0 if seed % 2 == 0 else 1.0
    shift = sign * shift_magnitude
    nominal = p.feed_benzene_fraction
    return EvaluatorFaultTruth(
        family, severity, shift, "feed_benzene_fraction", nominal,
        nominal + shift, severity_label(severity)
    )


def faulted_plant_parameters(
    nominal_parameters: FlashParameters, truth: EvaluatorFaultTruth
) -> FlashParameters:
    """Apply F1-F3 to an independent plant parameter copy only."""

    if truth.family is FaultFamily.NONIDEAL_VLE_STRESS:
        raise NotImplementedError("F4 is reserved and must not enter population design-calibration training")
    expected = getattr(nominal_parameters, truth.changed_parameter)
    if not np.isclose(expected, truth.nominal_value, rtol=0.0, atol=1e-12):
        raise ValueError("fault record does not match the supplied nominal parameter set")
    return replace(nominal_parameters, **{truth.changed_parameter: truth.faulty_value})


def nominal_and_effective_action_bounds(
    parameters: FlashParameters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = FlashDigitalTwin(parameters)
    nominal = model.nominal_action().as_array()
    p = model.parameters
    physical = np.asarray(
        [
            p.heat_duty_bounds_w,
            p.vapor_valve_bounds,
            p.liquid_valve_bounds,
            p.feed_flow_bounds_mol_s,
        ],
        dtype=float,
    )
    lower = np.maximum(physical[:, 0], nominal - MAXIMUM_ACTION_CHANGE)
    upper = np.minimum(physical[:, 1], nominal + MAXIMUM_ACTION_CHANGE)
    return nominal, lower, upper


def validate_candidate_action(action: FlashAction | Sequence[float]) -> FlashAction:
    candidate = action if isinstance(action, FlashAction) else FlashAction.from_vector(action)
    nominal, lower, upper = nominal_and_effective_action_bounds()
    values = candidate.as_array()
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError(f"candidate action outside population design domain [{lower}, {upper}]")
    if np.any(np.abs(values - nominal) > MAXIMUM_ACTION_CHANGE + 1e-12):
        raise ValueError("candidate exceeds maximum change from nominal action")
    return candidate


def latin_hypercube_actions(count: int, seed: int) -> np.ndarray:
    """Generate a centered Latin hypercube over the configured action bounds."""

    if count < 1:
        raise ValueError("action count must be positive")
    rng = np.random.default_rng(seed)
    unit = np.empty((count, len(ACTION_NAMES)))
    centers = (np.arange(count, dtype=float) + 0.5) / count
    for dimension in range(len(ACTION_NAMES)):
        unit[:, dimension] = centers[rng.permutation(count)]
    _, lower, upper = nominal_and_effective_action_bounds()
    return lower + unit * (upper - lower)


def targeted_action_points() -> np.ndarray:
    """Return nominal, axial extremes, and all domain corners."""

    nominal, lower, upper = nominal_and_effective_action_bounds()
    points = [nominal]
    for dimension in range(len(ACTION_NAMES)):
        for value in (lower[dimension], upper[dimension]):
            point = nominal.copy()
            point[dimension] = value
            points.append(point)
    points.extend(np.asarray(values, dtype=float) for values in product(*zip(lower, upper)))
    return np.unique(np.vstack(points), axis=0)


def boundary_candidate_pool(seed: int) -> np.ndarray:
    return np.vstack(
        [latin_hypercube_actions(BOUNDARY_LHS_SIZE, seed), targeted_action_points()]
    )


@dataclass(frozen=True)
class BoundarySelection:
    action: FlashAction
    nominal_dt_margin: float
    within_boundary_band: bool
    selected_rank: int
    eligible_count: int
    pool_size: int


def select_boundary_action(
    decision_state: FlashState,
    seed: int,
    *,
    integration_step_s: float = 1.0,
) -> BoundarySelection:
    """Select using only nominal-DT predictions; no plant object is accepted."""

    pool = boundary_candidate_pool(seed)
    safety = SafetySpecification.from_parameters(FlashParameters())
    margins = np.empty(len(pool))
    for index, vector in enumerate(pool):
        twin = FlashDigitalTwin()
        try:
            prediction = twin.simulate(
                FlashAction.from_vector(vector),
                PREDICTION_HORIZON_S,
                initial_state=decision_state,
                integration_step_s=integration_step_s,
                sample_interval_s=SAMPLING_INTERVAL_S,
            )
            future_prediction = {
                key: prediction[key][1:]
                for key in MEASURED_OUTPUTS
            }
            margins[index] = safety.trajectory_margin(future_prediction)
        except PhysicalStateError:
            # A nominal trajectory outside the validated thermodynamic/vessel
            # domain is a gross DT violation, never a boundary candidate.
            margins[index] = -np.inf
    ordering = np.lexsort((np.arange(len(pool)), np.abs(margins)))
    eligible = ordering[np.abs(margins[ordering]) <= BOUNDARY_DELTA_NORMALIZED]
    candidate_order = eligible if len(eligible) else ordering[:1]
    top = candidate_order[:BOUNDARY_TOP_RANKS]
    selected_rank = int(seed % len(top))
    selected_index = int(top[selected_rank])
    return BoundarySelection(
        action=validate_candidate_action(pool[selected_index]),
        nominal_dt_margin=float(margins[selected_index]),
        within_boundary_band=bool(abs(margins[selected_index]) <= BOUNDARY_DELTA_NORMALIZED),
        selected_rank=selected_rank,
        eligible_count=int(len(eligible)),
        pool_size=int(len(pool)),
    )

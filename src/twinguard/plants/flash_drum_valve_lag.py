"""Flash plant variant with an unmodeled vapor-valve actuator lag."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twinguard.plants.flash_drum import FlashAction, FlashParameters, FlashPlant, FlashState


@dataclass(frozen=True)
class VaporValveLagTruth:
    """Evaluator-only description of the unmodelled actuator mechanism."""

    normalized_severity: float
    time_constant_s: float
    minimum_time_constant_s: float = 30.0
    maximum_time_constant_s: float = 120.0


class VaporValveLagPlant:
    """Flash plant with a first-order vapor-valve actuator absent from the DT.

    The commanded opening is operationally visible. The effective opening is a
    hidden actuator state and is never supplied to the discrepancy predictor.
    """

    def __init__(
        self,
        parameters: FlashParameters,
        initial_state: FlashState,
        time_constant_s: float,
        initial_vapor_opening: float,
    ) -> None:
        if not np.isfinite(time_constant_s) or time_constant_s <= 0.0:
            raise ValueError("vapor-valve lag time constant must be positive")
        self._plant = FlashPlant(parameters, initial_state)
        self.time_constant_s = float(time_constant_s)
        self.effective_vapor_opening = float(initial_vapor_opening)
        self._plant.validate_action(
            FlashAction(15000.0, self.effective_vapor_opening, 0.5, 1.0)
        )

    @property
    def state(self) -> FlashState:
        return self._plant.state

    @property
    def parameters(self) -> FlashParameters:
        return self._plant.parameters

    def outputs(self, state: FlashState) -> dict[str, float]:
        return self._plant.outputs(state)

    def nominal_action(self) -> FlashAction:
        return self._plant.nominal_action()

    def step(self, action: FlashAction, dt_s: float = 1.0) -> dict[str, float]:
        command = self._plant.validate_action(action)
        ratio = dt_s / self.time_constant_s
        decay = float(np.exp(-ratio))
        initial = self.effective_vapor_opening
        final = command.vapor_valve_opening + (
            initial - command.vapor_valve_opening
        ) * decay
        average = command.vapor_valve_opening + (
            initial - command.vapor_valve_opening
        ) * (1.0 - decay) / ratio
        effective = FlashAction(
            command.heat_duty_w,
            float(average),
            command.liquid_valve_opening,
            command.feed_flow_mol_s,
        )
        result = self._plant.step(effective, dt_s)
        self.effective_vapor_opening = float(final)
        return {
            **result,
            "commanded_vapor_valve_opening": command.vapor_valve_opening,
            "effective_vapor_valve_opening": self.effective_vapor_opening,
        }


def sample_vapor_valve_lag(seed: int) -> VaporValveLagTruth:
    """Sample only the new mechanism's severity from an isolated RNG stream."""

    rng = np.random.default_rng(np.random.SeedSequence([seed, 811]))
    severity = float(rng.uniform(0.0, 1.0))
    lower, upper = 30.0, 120.0
    return VaporValveLagTruth(severity, lower + severity * (upper - lower))

"""Reduced dynamic equilibrium-stage model of a benzene/toluene flash drum.

The liquid phase is perfectly mixed and the disengaged vapor is assumed to be
in instantaneous VLE with that liquid. The three integrated extensive states
are total liquid holdup, liquid benzene holdup, and liquid sensible enthalpy.
Pressure is the Raoult-law bubble pressure, so it responds algebraically to the
dynamic temperature and composition.

Property sources (NIST Chemistry WebBook, SRD 69):
* Benzene Antoine and molecular data: https://webbook.nist.gov/cgi/cbook.cgi?ID=C71432&Type=ANTOINE
* Toluene Antoine and molecular data: https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Type=ANTOINE
* Benzene vaporization enthalpy: https://webbook.nist.gov/cgi/cbook.cgi?ID=C71432&Type=HVAP-FORM1
* Toluene vaporization enthalpy: https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Type=HVAP-FORM2
* Liquid heat capacities: the corresponding NIST condensed-phase pages.

The constant liquid heat capacities are a declared reduced-model
approximation. All calculations use mol, s, K, J, W, and bar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np


BENZENE = 0
TOLUENE = 1


class PhysicalStateError(ValueError):
    """Raised when a trajectory leaves the declared physical model domain."""


@dataclass(frozen=True)
class AntoineCorrelation:
    """NIST Antoine correlation with pressure in bar and temperature in K."""

    a: float
    b_kelvin: float
    c_kelvin: float
    temperature_min_k: float
    temperature_max_k: float
    source: str

    def pressure_bar(self, temperature_k: float) -> float:
        if not self.temperature_min_k <= temperature_k <= self.temperature_max_k:
            raise PhysicalStateError(
                f"temperature {temperature_k:.6g} K is outside Antoine range "
                f"[{self.temperature_min_k}, {self.temperature_max_k}] K"
            )
        return 10.0 ** (self.a - self.b_kelvin / (temperature_k + self.c_kelvin))


@dataclass(frozen=True)
class VaporizationCorrelation:
    """NIST reduced-temperature vaporization-enthalpy correlation."""

    a_kj_mol: float
    beta: float
    critical_temperature_k: float
    temperature_min_k: float
    temperature_max_k: float
    alpha: float | None
    source: str

    def enthalpy_j_mol(self, temperature_k: float) -> float:
        if not self.temperature_min_k <= temperature_k <= self.temperature_max_k:
            raise PhysicalStateError(
                f"temperature {temperature_k:.6g} K is outside latent-heat range "
                f"[{self.temperature_min_k}, {self.temperature_max_k}] K"
            )
        reduced = temperature_k / self.critical_temperature_k
        exponential = np.exp(
            -(self.alpha if self.alpha is not None else self.beta) * reduced
        )
        return float(
            1000.0 * self.a_kj_mol * exponential * (1.0 - reduced) ** self.beta
        )


@dataclass(frozen=True)
class FlashParameters:
    """Flash-drum operating parameters and physical validity limits."""

    reference_temperature_k: float = 298.15
    benzene_cp_liquid_j_mol_k: float = 135.69
    toluene_cp_liquid_j_mol_k: float = 157.09
    benzene_cp_source: str = "NIST SRD 69; Grolier, Roux-Desgranges, et al. (1993), 298.15 K"
    toluene_cp_source: str = "NIST SRD 69; Grolier, Roux-Desgranges, et al. (1993), 298.15 K"
    liquid_capacity_mol: float = 1000.0
    feed_temperature_k: float = 340.0
    feed_benzene_fraction: float = 0.55
    downstream_pressure_bar: float = 0.20
    liquid_valve_coefficient_mol_s: float = 1.4734213784
    vapor_valve_coefficient_mol_s_bar_sqrt: float = 1.3500368345
    minimum_holdup_mol: float = 50.0
    maximum_holdup_mol: float = 1000.0
    minimum_temperature_k: float = 333.5
    maximum_temperature_k: float = 373.0
    pressure_safety_max_bar: float = 0.90
    level_safety_min_fraction: float = 0.20
    level_safety_max_fraction: float = 0.85
    temperature_safety_max_k: float = 365.0
    heat_duty_bounds_w: tuple[float, float] = (10000.0, 22000.0)
    vapor_valve_bounds: tuple[float, float] = (0.25, 0.85)
    liquid_valve_bounds: tuple[float, float] = (0.25, 0.85)
    feed_flow_bounds_mol_s: tuple[float, float] = (0.75, 1.25)
    benzene_antoine: AntoineCorrelation = AntoineCorrelation(
        a=4.72583,
        b_kelvin=1660.652,
        c_kelvin=-1.461,
        temperature_min_k=333.4,
        temperature_max_k=373.5,
        source="NIST SRD 69; Eon, Pommier, et al. (1971)",
    )
    toluene_antoine: AntoineCorrelation = AntoineCorrelation(
        a=4.07827,
        b_kelvin=1343.943,
        c_kelvin=-53.773,
        temperature_min_k=308.52,
        temperature_max_k=384.66,
        source="NIST SRD 69; Willingham, Taylor, et al. (1945)",
    )
    benzene_vaporization: VaporizationCorrelation = VaporizationCorrelation(
        a_kj_mol=47.41,
        alpha=0.1231,
        beta=0.3602,
        critical_temperature_k=562.1,
        temperature_min_k=293.0,
        temperature_max_k=469.0,
        source="NIST SRD 69; Majer and Svoboda (1985), form 1",
    )
    toluene_vaporization: VaporizationCorrelation = VaporizationCorrelation(
        a_kj_mol=53.09,
        alpha=None,
        beta=0.2774,
        critical_temperature_k=591.7,
        temperature_min_k=298.0,
        temperature_max_k=410.0,
        source="NIST SRD 69; Majer and Svoboda (1985), form 2",
    )

    def independent_copy(self) -> "FlashParameters":
        return replace(
            self,
            benzene_antoine=replace(self.benzene_antoine),
            toluene_antoine=replace(self.toluene_antoine),
            benzene_vaporization=replace(self.benzene_vaporization),
            toluene_vaporization=replace(self.toluene_vaporization),
        )


@dataclass(frozen=True)
class FlashAction:
    """Continuous action vector [Q, u_V, u_L, F]."""

    heat_duty_w: float
    vapor_valve_opening: float
    liquid_valve_opening: float
    feed_flow_mol_s: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.heat_duty_w,
                self.vapor_valve_opening,
                self.liquid_valve_opening,
                self.feed_flow_mol_s,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> "FlashAction":
        array = np.asarray(values, dtype=float)
        if array.shape != (4,) or not np.all(np.isfinite(array)):
            raise ValueError("flash action must be a finite vector of shape (4,)")
        return cls(*map(float, array))


@dataclass(frozen=True)
class FlashState:
    """Extensive state [liquid mol, liquid benzene mol, sensible enthalpy J]."""

    liquid_moles: float
    benzene_moles: float
    liquid_energy_j: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.liquid_moles, self.benzene_moles, self.liquid_energy_j],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: Sequence[float]) -> "FlashState":
        array = np.asarray(values, dtype=float)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise PhysicalStateError("flash state must be a finite vector of shape (3,)")
        return cls(*map(float, array))


ActionSchedule = FlashAction | Sequence[float] | Callable[[float], FlashAction | Sequence[float]]


class DynamicFlashModel:
    """Process-neutral mechanics shared by plant and nominal DT instances."""

    def __init__(
        self,
        parameters: FlashParameters | None = None,
        initial_state: FlashState | None = None,
    ) -> None:
        source = parameters if parameters is not None else FlashParameters()
        self.parameters = source.independent_copy()
        self.initial_state = initial_state or self.nominal_state()
        self.validate_state(self.initial_state)
        self.state = FlashState.from_array(self.initial_state.as_array().copy())

    def component_heat_capacities(self) -> np.ndarray:
        p = self.parameters
        return np.asarray(
            [p.benzene_cp_liquid_j_mol_k, p.toluene_cp_liquid_j_mol_k]
        )

    def state_from_nxt(
        self, liquid_moles: float, benzene_fraction: float, temperature_k: float
    ) -> FlashState:
        if not 0.0 < benzene_fraction < 1.0:
            raise PhysicalStateError("liquid benzene fraction must be in (0, 1)")
        benzene_moles = liquid_moles * benzene_fraction
        component_moles = np.asarray([benzene_moles, liquid_moles - benzene_moles])
        energy = float(
            np.dot(component_moles, self.component_heat_capacities())
            * (temperature_k - self.parameters.reference_temperature_k)
        )
        state = FlashState(liquid_moles, benzene_moles, energy)
        self.validate_state(state)
        return state

    def nominal_state(self) -> FlashState:
        return self.state_from_nxt(600.0, 0.45, 350.0)

    def temperature_k(self, state: FlashState | Sequence[float]) -> float:
        array = state.as_array() if isinstance(state, FlashState) else np.asarray(state)
        total, benzene, energy = map(float, array)
        toluene = total - benzene
        capacity = float(
            benzene * self.parameters.benzene_cp_liquid_j_mol_k
            + toluene * self.parameters.toluene_cp_liquid_j_mol_k
        )
        if capacity <= 0.0:
            raise PhysicalStateError("liquid heat capacity must remain positive")
        return self.parameters.reference_temperature_k + energy / capacity

    def liquid_benzene_fraction(self, state: FlashState | Sequence[float]) -> float:
        array = state.as_array() if isinstance(state, FlashState) else np.asarray(state)
        return float(array[1] / array[0])

    def saturation_pressures_bar(self, temperature_k: float) -> np.ndarray:
        p = self.parameters
        return np.asarray(
            [
                p.benzene_antoine.pressure_bar(temperature_k),
                p.toluene_antoine.pressure_bar(temperature_k),
            ]
        )

    def latent_heats_j_mol(self, temperature_k: float) -> np.ndarray:
        p = self.parameters
        return np.asarray(
            [
                p.benzene_vaporization.enthalpy_j_mol(temperature_k),
                p.toluene_vaporization.enthalpy_j_mol(temperature_k),
            ]
        )

    def vle(self, temperature_k: float, liquid_benzene_fraction: float) -> tuple[float, float]:
        psat = self.saturation_pressures_bar(temperature_k)
        liquid = np.asarray([liquid_benzene_fraction, 1.0 - liquid_benzene_fraction])
        pressure = float(np.dot(liquid, psat))
        vapor_benzene_fraction = float(liquid[0] * psat[BENZENE] / pressure)
        return pressure, vapor_benzene_fraction

    def outputs(self, state: FlashState | Sequence[float]) -> dict[str, float]:
        flash_state = state if isinstance(state, FlashState) else FlashState.from_array(state)
        temperature = self.temperature_k(flash_state)
        liquid_fraction = self.liquid_benzene_fraction(flash_state)
        pressure, vapor_fraction = self.vle(temperature, liquid_fraction)
        return {
            "pressure_bar": pressure,
            "level_fraction": flash_state.liquid_moles / self.parameters.liquid_capacity_mol,
            "temperature_k": temperature,
            "liquid_benzene_fraction": liquid_fraction,
            "vapor_benzene_fraction": vapor_fraction,
            "liquid_moles": flash_state.liquid_moles,
        }

    def validate_action(self, action: FlashAction | Sequence[float]) -> FlashAction:
        candidate = action if isinstance(action, FlashAction) else FlashAction.from_vector(action)
        p = self.parameters
        values_and_bounds = (
            (candidate.heat_duty_w, p.heat_duty_bounds_w, "heat duty"),
            (candidate.vapor_valve_opening, p.vapor_valve_bounds, "vapor valve"),
            (candidate.liquid_valve_opening, p.liquid_valve_bounds, "liquid valve"),
            (candidate.feed_flow_mol_s, p.feed_flow_bounds_mol_s, "feed flow"),
        )
        for value, bounds, name in values_and_bounds:
            if not np.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{name} {value!r} outside bounds {bounds}")
        return candidate

    def validate_state(self, state: FlashState | Sequence[float]) -> None:
        flash_state = state if isinstance(state, FlashState) else FlashState.from_array(state)
        p = self.parameters
        if not p.minimum_holdup_mol <= flash_state.liquid_moles <= p.maximum_holdup_mol:
            raise PhysicalStateError("liquid holdup left the physical vessel capacity")
        if not 0.0 < flash_state.benzene_moles < flash_state.liquid_moles:
            raise PhysicalStateError("component holdups must be positive")
        temperature = self.temperature_k(flash_state)
        if not p.minimum_temperature_k <= temperature <= p.maximum_temperature_k:
            raise PhysicalStateError("temperature left the validated property domain")

    def nominal_action(self) -> FlashAction:
        """Return the action that makes the nominal state an exact steady state."""

        state = self.nominal_state()
        output = self.outputs(state)
        x = output["liquid_benzene_fraction"]
        y = output["vapor_benzene_fraction"]
        p = self.parameters
        feed = 1.0
        vapor = feed * (p.feed_benzene_fraction - x) / (y - x)
        liquid = feed - vapor
        u_v = vapor / (
            p.vapor_valve_coefficient_mol_s_bar_sqrt
            * np.sqrt(output["pressure_bar"] - p.downstream_pressure_bar)
        )
        u_l = liquid / (
            p.liquid_valve_coefficient_mol_s * np.sqrt(output["level_fraction"])
        )
        h_feed = self.liquid_enthalpy_j_mol(
            p.feed_temperature_k, p.feed_benzene_fraction
        )
        h_liquid = self.liquid_enthalpy_j_mol(350.0, x)
        h_vapor = self.vapor_enthalpy_j_mol(350.0, y)
        heat = liquid * h_liquid + vapor * h_vapor - feed * h_feed
        return self.validate_action(FlashAction(float(heat), float(u_v), float(u_l), feed))

    def liquid_enthalpy_j_mol(self, temperature_k: float, benzene_fraction: float) -> float:
        cp = float(
            np.dot(
                [benzene_fraction, 1.0 - benzene_fraction],
                self.component_heat_capacities(),
            )
        )
        return cp * (temperature_k - self.parameters.reference_temperature_k)

    def vapor_enthalpy_j_mol(self, temperature_k: float, benzene_fraction: float) -> float:
        fractions = np.asarray([benzene_fraction, 1.0 - benzene_fraction])
        sensible = float(np.dot(fractions, self.component_heat_capacities())) * (
            temperature_k - self.parameters.reference_temperature_k
        )
        return sensible + float(np.dot(fractions, self.latent_heats_j_mol(temperature_k)))

    def flow_diagnostics(
        self, state: FlashState | Sequence[float], action: FlashAction | Sequence[float]
    ) -> dict[str, float]:
        flash_state = state if isinstance(state, FlashState) else FlashState.from_array(state)
        candidate = self.validate_action(action)
        output = self.outputs(flash_state)
        p = self.parameters
        liquid_flow = (
            p.liquid_valve_coefficient_mol_s
            * candidate.liquid_valve_opening
            * np.sqrt(max(output["level_fraction"], 0.0))
        )
        pressure_drop = max(output["pressure_bar"] - p.downstream_pressure_bar, 0.0)
        vapor_flow = (
            p.vapor_valve_coefficient_mol_s_bar_sqrt
            * candidate.vapor_valve_opening
            * np.sqrt(pressure_drop)
        )
        return {
            **output,
            "feed_flow_mol_s": candidate.feed_flow_mol_s,
            "liquid_flow_mol_s": float(liquid_flow),
            "vapor_flow_mol_s": float(vapor_flow),
            "heat_duty_w": candidate.heat_duty_w,
            "vapor_valve_opening": candidate.vapor_valve_opening,
            "liquid_valve_opening": candidate.liquid_valve_opening,
        }

    def derivatives(
        self, state: FlashState | Sequence[float], action: FlashAction | Sequence[float]
    ) -> np.ndarray:
        flash_state = state if isinstance(state, FlashState) else FlashState.from_array(state)
        self.validate_state(flash_state)
        candidate = self.validate_action(action)
        d = self.flow_diagnostics(flash_state, candidate)
        p = self.parameters
        feed = candidate.feed_flow_mol_s
        liquid = d["liquid_flow_mol_s"]
        vapor = d["vapor_flow_mol_s"]
        x = d["liquid_benzene_fraction"]
        y = d["vapor_benzene_fraction"]
        temperature = d["temperature_k"]
        total_rate = feed - liquid - vapor
        benzene_rate = feed * p.feed_benzene_fraction - liquid * x - vapor * y
        energy_rate = (
            feed * self.liquid_enthalpy_j_mol(p.feed_temperature_k, p.feed_benzene_fraction)
            + candidate.heat_duty_w
            - liquid * self.liquid_enthalpy_j_mol(temperature, x)
            - vapor * self.vapor_enthalpy_j_mol(temperature, y)
        )
        return np.asarray([total_rate, benzene_rate, energy_rate], dtype=float)

    def _rk4_step(
        self, state: FlashState, action: FlashAction, dt_s: float
    ) -> tuple[FlashState, np.ndarray]:
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("integration step must be positive and finite")
        initial = state.as_array()
        k1 = self.derivatives(initial, action)
        k2 = self.derivatives(initial + 0.5 * dt_s * k1, action)
        k3 = self.derivatives(initial + 0.5 * dt_s * k2, action)
        k4 = self.derivatives(initial + dt_s * k3, action)
        average_rate = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        final = FlashState.from_array(initial + dt_s * average_rate)
        self.validate_state(final)
        residual = final.as_array() - initial - dt_s * average_rate
        return final, residual

    def step(self, action: FlashAction | Sequence[float], dt_s: float = 1.0) -> dict[str, float]:
        candidate = self.validate_action(action)
        self.state, residual = self._rk4_step(self.state, candidate, dt_s)
        return {
            **self.flow_diagnostics(self.state, candidate),
            "material_balance_residual_mol": float(residual[0]),
            "component_balance_residual_mol": float(residual[1]),
            "energy_balance_residual_j": float(residual[2]),
        }

    def reset(self, state: FlashState | None = None) -> None:
        source = state if state is not None else self.initial_state
        self.validate_state(source)
        self.state = FlashState.from_array(source.as_array().copy())

    def simulate(
        self,
        action: ActionSchedule,
        duration_s: float,
        *,
        initial_state: FlashState | None = None,
        integration_step_s: float = 1.0,
        sample_interval_s: float = 10.0,
    ) -> dict[str, np.ndarray]:
        if duration_s <= 0.0 or sample_interval_s <= 0.0:
            raise ValueError("duration and sample interval must be positive")
        steps = int(round(duration_s / integration_step_s))
        sample_steps = int(round(sample_interval_s / integration_step_s))
        if not np.isclose(steps * integration_step_s, duration_s):
            raise ValueError("duration must be an integer multiple of integration step")
        if sample_steps < 1 or not np.isclose(
            sample_steps * integration_step_s, sample_interval_s
        ):
            raise ValueError("sample interval must be an integer multiple of integration step")
        self.reset(initial_state)
        records: list[dict[str, float]] = []
        max_residual = np.zeros(3)

        def at_time(time_s: float) -> FlashAction:
            raw = action(time_s) if callable(action) else action
            return self.validate_action(raw)

        def record(time_s: float, candidate: FlashAction) -> None:
            records.append(
                {
                    "time_s": time_s,
                    **self.flow_diagnostics(self.state, candidate),
                    "liquid_energy_j": self.state.liquid_energy_j,
                    "benzene_moles": self.state.benzene_moles,
                }
            )

        record(0.0, at_time(0.0))
        for index in range(1, steps + 1):
            time_start = (index - 1) * integration_step_s
            candidate = at_time(time_start)
            self.state, residual = self._rk4_step(
                self.state, candidate, integration_step_s
            )
            max_residual = np.maximum(max_residual, np.abs(residual))
            if index % sample_steps == 0:
                record(index * integration_step_s, at_time(index * integration_step_s))

        columns = records[0].keys()
        result = {key: np.asarray([row[key] for row in records], dtype=float) for key in columns}
        result["max_balance_residuals"] = max_residual
        return result

    def is_safe(self, state: FlashState | Sequence[float]) -> bool:
        output = self.outputs(state)
        p = self.parameters
        return bool(
            output["pressure_bar"] <= p.pressure_safety_max_bar
            and p.level_safety_min_fraction
            <= output["level_fraction"]
            <= p.level_safety_max_fraction
            and output["temperature_k"] <= p.temperature_safety_max_k
        )


class FlashPlant(DynamicFlashModel):
    """Simulate the flash drum under a supplied parameter set."""


def trajectory_max_abs_difference(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray], keys: Sequence[str]
) -> float:
    return max(float(np.max(np.abs(first[key] - second[key]))) for key in keys)

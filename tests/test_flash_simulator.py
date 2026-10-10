from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.plants.flash_drum import (
    FlashAction,
    FlashParameters,
    FlashPlant,
    PhysicalStateError,
    trajectory_max_abs_difference,
)


OUTPUT_KEYS = (
    "pressure_bar",
    "level_fraction",
    "temperature_k",
    "liquid_benzene_fraction",
    "vapor_benzene_fraction",
)


def test_nist_property_correlations_have_consistent_units_and_ranges() -> None:
    model = FlashPlant()
    p = model.parameters

    # At each normal boiling point, the pure-component vapor pressure is 1 bar.
    assert p.benzene_antoine.pressure_bar(353.25) == pytest.approx(1.0, abs=0.015)
    assert p.toluene_antoine.pressure_bar(383.75) == pytest.approx(1.0, abs=0.015)

    latent = model.latent_heats_j_mol(350.0)
    assert np.all((latent > 25_000.0) & (latent < 45_000.0))
    cp = model.component_heat_capacities()
    assert np.all((cp > 100.0) & (cp < 200.0))
    assert p.benzene_antoine.b_kelvin > 1000.0
    assert p.toluene_antoine.b_kelvin > 1000.0


def test_raoults_law_and_phase_compositions_are_exactly_consistent() -> None:
    model = FlashPlant()
    temperature = 350.0
    x_benzene = 0.45
    pressure, y_benzene = model.vle(temperature, x_benzene)
    psat = model.saturation_pressures_bar(temperature)

    assert pressure == pytest.approx(x_benzene * psat[0] + (1.0 - x_benzene) * psat[1])
    assert y_benzene * pressure == pytest.approx(x_benzene * psat[0])
    assert (1.0 - y_benzene) * pressure == pytest.approx((1.0 - x_benzene) * psat[1])
    assert x_benzene < y_benzene < 1.0


def test_nominal_state_and_action_close_all_three_dynamic_balances() -> None:
    model = FlashPlant()
    state = model.nominal_state()
    action = model.nominal_action()
    rate = model.derivatives(state, action)

    assert np.max(np.abs(rate[:2])) < 1e-12
    assert abs(rate[2]) < 1e-9


def test_instantaneous_material_component_and_energy_balance_identities() -> None:
    model = FlashPlant()
    state = model.state_from_nxt(640.0, 0.48, 352.0)
    action = FlashAction(16500.0, 0.58, 0.46, 1.08)
    d = model.flow_diagnostics(state, action)
    rate = model.derivatives(state, action)
    p = model.parameters

    expected_total = d["feed_flow_mol_s"] - d["liquid_flow_mol_s"] - d["vapor_flow_mol_s"]
    expected_benzene = (
        d["feed_flow_mol_s"] * p.feed_benzene_fraction
        - d["liquid_flow_mol_s"] * d["liquid_benzene_fraction"]
        - d["vapor_flow_mol_s"] * d["vapor_benzene_fraction"]
    )
    expected_energy = (
        d["feed_flow_mol_s"]
        * model.liquid_enthalpy_j_mol(p.feed_temperature_k, p.feed_benzene_fraction)
        + action.heat_duty_w
        - d["liquid_flow_mol_s"]
        * model.liquid_enthalpy_j_mol(d["temperature_k"], d["liquid_benzene_fraction"])
        - d["vapor_flow_mol_s"]
        * model.vapor_enthalpy_j_mol(d["temperature_k"], d["vapor_benzene_fraction"])
    )

    np.testing.assert_allclose(rate, [expected_total, expected_benzene, expected_energy])


def test_rk4_step_preserves_discrete_balance_to_roundoff() -> None:
    model = FlashPlant()
    diagnostics = model.step(model.nominal_action(), dt_s=1.0)

    assert abs(diagnostics["material_balance_residual_mol"]) < 1e-12
    assert abs(diagnostics["component_balance_residual_mol"]) < 1e-12
    assert abs(diagnostics["energy_balance_residual_j"]) < 1e-8


def test_plant_and_dt_are_independent_instances_with_numerical_parity() -> None:
    parameters = FlashParameters()
    plant = FlashPlant(parameters)
    twin = FlashDigitalTwin(parameters)
    action = plant.nominal_action()

    assert plant is not twin
    assert plant.parameters is not twin.parameters
    assert plant.parameters.benzene_antoine is not twin.parameters.benzene_antoine
    plant_result = plant.simulate(action, 600.0, sample_interval_s=10.0)
    twin_result = twin.simulate(action, 600.0, sample_interval_s=10.0)
    assert trajectory_max_abs_difference(plant_result, twin_result, OUTPUT_KEYS) < 1e-12

    changed = replace(parameters, feed_temperature_k=341.0)
    separate_plant = FlashPlant(changed)
    assert twin.parameters.feed_temperature_k == 340.0
    assert separate_plant.parameters.feed_temperature_k == 341.0


def test_fixed_seed_initial_conditions_and_trajectories_are_reproducible() -> None:
    def run(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        model = FlashPlant()
        state = model.state_from_nxt(
            600.0 + rng.uniform(-25.0, 25.0),
            0.45 + rng.uniform(-0.02, 0.02),
            350.0 + rng.uniform(-0.8, 0.8),
        )
        result = model.simulate(
            model.nominal_action(), 300.0, initial_state=state, sample_interval_s=10.0
        )
        return np.column_stack([result[key] for key in OUTPUT_KEYS])

    np.testing.assert_array_equal(run(1601), run(1601))
    assert not np.array_equal(run(1601), run(1602))


def test_perturbed_long_horizon_is_stable_positive_and_convergent() -> None:
    model = FlashPlant()
    perturbed = model.state_from_nxt(650.0, 0.47, 352.0)
    result = model.simulate(
        model.nominal_action(), 7200.0, initial_state=perturbed, sample_interval_s=60.0
    )

    assert np.all(result["liquid_moles"] > 0.0)
    assert np.all((result["liquid_benzene_fraction"] > 0.0) & (result["liquid_benzene_fraction"] < 1.0))
    assert np.all((result["vapor_benzene_fraction"] > 0.0) & (result["vapor_benzene_fraction"] < 1.0))
    assert result["temperature_k"][-1] == pytest.approx(350.0, abs=1e-4)
    assert result["level_fraction"][-1] == pytest.approx(0.6, abs=0.002)
    assert result["liquid_benzene_fraction"][-1] == pytest.approx(0.45, abs=1e-4)


def test_action_steps_have_expected_directional_responses() -> None:
    base_model = FlashPlant()
    base_action = base_model.nominal_action()

    def final_output(action: FlashAction) -> dict[str, float]:
        model = FlashPlant()
        result = model.simulate(action, 900.0, sample_interval_s=30.0)
        return {key: float(result[key][-1]) for key in OUTPUT_KEYS}

    base = final_output(base_action)
    more_heat = final_output(replace(base_action, heat_duty_w=base_action.heat_duty_w + 2500.0))
    more_vapor = final_output(replace(base_action, vapor_valve_opening=base_action.vapor_valve_opening + 0.12))
    more_liquid = final_output(replace(base_action, liquid_valve_opening=base_action.liquid_valve_opening + 0.12))
    more_feed = final_output(replace(base_action, feed_flow_mol_s=base_action.feed_flow_mol_s + 0.15))

    assert more_heat["temperature_k"] > base["temperature_k"] + 2.0
    assert more_heat["pressure_bar"] > base["pressure_bar"] + 0.04
    assert more_vapor["level_fraction"] < base["level_fraction"] - 0.02
    assert more_liquid["level_fraction"] < base["level_fraction"] - 0.05
    assert more_feed["level_fraction"] > base["level_fraction"] + 0.04


def test_action_bounds_and_property_domain_are_enforced() -> None:
    model = FlashPlant()
    nominal = model.nominal_action()
    for index, bounds in enumerate(
        (
            model.parameters.heat_duty_bounds_w,
            model.parameters.vapor_valve_bounds,
            model.parameters.liquid_valve_bounds,
            model.parameters.feed_flow_bounds_mol_s,
        )
    ):
        vector = nominal.as_array()
        vector[index] = bounds[1] + 1.0
        with pytest.raises(ValueError, match="outside bounds"):
            model.validate_action(vector)

    with pytest.raises(PhysicalStateError, match="Antoine range"):
        model.saturation_pressures_bar(400.0)

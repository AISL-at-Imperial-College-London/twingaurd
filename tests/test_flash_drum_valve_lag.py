import numpy as np

from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.plants.flash_drum import FlashAction, FlashParameters, FlashPlant
from twinguard.plants.flash_drum_valve_lag import (
    VaporValveLagPlant,
    sample_vapor_valve_lag,
)


def test_structural_lag_is_reproducible_and_within_predeclared_range():
    first = sample_vapor_valve_lag(28_000_123)
    second = sample_vapor_valve_lag(28_000_123)
    assert first == second
    assert 0.0 <= first.normalized_severity <= 1.0
    assert 30.0 <= first.time_constant_s <= 120.0


def test_structural_plant_matches_nominal_under_unchanged_command():
    parameters = FlashParameters()
    twin = FlashDigitalTwin(parameters)
    initial = twin.nominal_state()
    action = twin.nominal_action()
    healthy = FlashPlant(parameters, initial)
    lagged = VaporValveLagPlant(
        parameters, initial, time_constant_s=75.0,
        initial_vapor_opening=action.vapor_valve_opening,
    )
    for _ in range(120):
        healthy.step(action, 1.0)
        lagged.step(action, 1.0)
    assert np.array_equal(healthy.state.as_array(), lagged.state.as_array())


def test_vapor_valve_lag_is_bounded_monotone_and_absent_from_nominal_dt():
    parameters = FlashParameters()
    twin = FlashDigitalTwin(parameters)
    nominal = twin.nominal_action()
    command = FlashAction(
        nominal.heat_duty_w,
        nominal.vapor_valve_opening + 0.20,
        nominal.liquid_valve_opening,
        nominal.feed_flow_mol_s,
    )
    lagged = VaporValveLagPlant(
        parameters, twin.nominal_state(), time_constant_s=60.0,
        initial_vapor_opening=nominal.vapor_valve_opening,
    )
    effective = []
    for _ in range(30):
        lagged.step(command, 1.0)
        effective.append(lagged.effective_vapor_opening)
    assert np.all(np.diff(effective) > 0.0)
    assert nominal.vapor_valve_opening < effective[-1] < command.vapor_valve_opening
    twin.step(command, 1.0)
    assert not hasattr(twin, "effective_vapor_opening")

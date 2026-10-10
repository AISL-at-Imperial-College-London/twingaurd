"""Plant / nominal-twin separation: degradation enters only through independent plant parameter copies."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.plants.cstr import FaultConfig
from twinguard.plants.flash_drum import FlashParameters, FlashPlant
from twinguard.plants.flash_drum_valve_lag import VaporValveLagPlant, sample_vapor_valve_lag
from twinguard.scenarios.flash_population_design import (
    PRIMARY_FAULT_FAMILIES,
    faulted_plant_parameters,
    sample_primary_fault,
)


@pytest.mark.parametrize("family", PRIMARY_FAULT_FAMILIES)
def test_flash_fault_changes_only_the_plant_parameter_copy(family) -> None:
    nominal = FlashParameters()
    truth = sample_primary_fault(family, 26_000_000)
    plant_parameters = faulted_plant_parameters(nominal, truth)
    twin = FlashDigitalTwin()
    changed = [f.name for f in fields(FlashParameters)
               if not np.array_equal(np.asarray(getattr(plant_parameters, f.name), dtype=object),
                                     np.asarray(getattr(nominal, f.name), dtype=object))]
    assert changed == [truth.changed_parameter]
    assert getattr(nominal, truth.changed_parameter) == truth.nominal_value
    assert getattr(twin.parameters, truth.changed_parameter) == truth.nominal_value
    assert FlashPlant(plant_parameters).parameters is not twin.parameters


def test_structural_mismatch_is_plant_only_and_seeded() -> None:
    first = sample_vapor_valve_lag(28_000_000)
    assert first == sample_vapor_valve_lag(28_000_000)
    assert 30.0 <= first.time_constant_s <= 120.0
    # the actuator lag exists only in the plant wrapper; the nominal twin has no such mechanism
    assert "time_constant_s" in VaporValveLagPlant.__init__.__code__.co_varnames
    assert not issubclass(FlashDigitalTwin, VaporValveLagPlant)
    assert not hasattr(FlashDigitalTwin(), "time_constant_s")


def test_cstr_feed_temperature_fault_is_inert_by_default() -> None:
    config = FaultConfig()
    assert config.feed_temp_step_on is False
    assert config.feed_temp_step_K == 0.0

from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.scenarios.flash_population_design import (
    ACTION_NAMES,
    BOUNDARY_DELTA_NORMALIZED,
    FORBIDDEN_OPERATIONAL_TOKENS,
    MAXIMUM_ACTION_CHANGE,
    OBSERVATION_COUNT,
    PRIMARY_FAULT_FAMILIES,
    EvaluatorFaultTruth,
    FaultFamily,
    OperationalEpisodeInput,
    SafetySpecification,
    boundary_candidate_pool,
    faulted_plant_parameters,
    latin_hypercube_actions,
    nominal_and_effective_action_bounds,
    sample_primary_fault,
    select_boundary_action,
    validate_candidate_action,
)
from twinguard.plants.flash_drum import FlashParameters, FlashPlant
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT


def test_simulator_validation_go_is_a_hard_population_design_precondition() -> None:
    summary = (EXPERIMENT_ROOT / "simulator_validation" / "simulator_validation_summary.json").read_text(
        encoding="utf-8"
    )
    assert '"go": true' in summary


@pytest.mark.parametrize("family", PRIMARY_FAULT_FAMILIES)
def test_primary_faults_change_only_the_declared_plant_parameter(family: FaultFamily) -> None:
    nominal = FlashParameters()
    truth = sample_primary_fault(family, 22000123)
    faulty = faulted_plant_parameters(nominal, truth)
    twin = FlashDigitalTwin(nominal)

    assert isinstance(truth, EvaluatorFaultTruth)
    assert getattr(faulty, truth.changed_parameter) == truth.faulty_value
    assert getattr(twin.parameters, truth.changed_parameter) == truth.nominal_value
    changed = {
        item.name
        for item in fields(nominal)
        if getattr(nominal, item.name) != getattr(faulty, item.name)
    }
    assert changed == {truth.changed_parameter}


def test_fault_ranges_and_reserved_f4_are_enforced() -> None:
    for seed in range(20):
        f1 = sample_primary_fault(FaultFamily.VAPOR_OUTLET_DEGRADATION, seed)
        f2 = sample_primary_fault(FaultFamily.LIQUID_OUTLET_DEGRADATION, seed)
        f3 = sample_primary_fault(FaultFamily.FEED_COMPOSITION_SHIFT, seed)
        assert 0.10 <= f1.signed_magnitude <= 0.45
        assert 0.10 <= f2.signed_magnitude <= 0.45
        assert 0.04 <= abs(f3.signed_magnitude) <= 0.12
        assert 0.43 <= f3.faulty_value <= 0.67
    with pytest.raises(ValueError, match="reserved"):
        sample_primary_fault(FaultFamily.NONIDEAL_VLE_STRESS, 1)


def test_operational_schema_has_no_evaluator_or_future_truth_fields() -> None:
    names = {item.name for item in fields(OperationalEpisodeInput)}
    assert not any(token in name for name in names for token in FORBIDDEN_OPERATIONAL_TOKENS)
    record = OperationalEpisodeInput(
        episode_id="example",
        observation_time_s=np.arange(1, OBSERVATION_COUNT + 1) * 10.0,
        observed_outputs=np.zeros((OBSERVATION_COUNT, 3)),
        nominal_dt_outputs=np.zeros((OBSERVATION_COUNT, 3)),
        action_history=np.zeros((OBSERVATION_COUNT, 4)),
        candidate_action=FlashPlant().nominal_action().as_array(),
    )
    record.validate()


def test_violation_margins_implement_all_four_safety_directions() -> None:
    safety = SafetySpecification()
    boundary = {
        "pressure_bar": [0.90],
        "level_fraction": [0.20],
        "temperature_k": [365.0],
    }
    margins = safety.violation_margins(boundary)
    np.testing.assert_allclose(margins, [[0.0, -6.5, 0.0, 0.0]])
    assert safety.is_physically_safe(boundary)

    violations = {
        "pressure_bar": [0.91, 0.80, 0.80, 0.80],
        "level_fraction": [0.50, 0.86, 0.19, 0.50],
        "temperature_k": [350.0, 350.0, 350.0, 365.1],
    }
    assert np.all(np.max(safety.violation_margins(violations), axis=0) > 0.0)
    assert not safety.is_physically_safe(violations)


def test_latin_hypercube_is_reproducible_space_filling_and_in_bounds() -> None:
    first = latin_hypercube_actions(64, 23000000)
    second = latin_hypercube_actions(64, 23000000)
    other = latin_hypercube_actions(64, 23000001)
    nominal, lower, upper = nominal_and_effective_action_bounds()

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all((first > lower) & (first < upper))
    assert np.all(np.abs(first - nominal) <= MAXIMUM_ACTION_CHANGE + 1e-12)
    for row in first:
        validate_candidate_action(row)
    for dimension in range(len(ACTION_NAMES)):
        bins = np.floor((first[:, dimension] - lower[dimension]) / (upper[dimension] - lower[dimension]) * 64)
        assert set(bins.astype(int)) == set(range(64))


def test_boundary_pool_contains_lhs_nominal_axial_and_corner_points() -> None:
    pool = boundary_candidate_pool(26000000)
    nominal, lower, upper = nominal_and_effective_action_bounds()
    assert pool.shape == (153, 4)
    assert any(np.array_equal(row, nominal) for row in pool)
    assert any(np.array_equal(row, lower) for row in pool)
    assert any(np.array_equal(row, upper) for row in pool)


def test_boundary_selection_uses_nominal_dt_and_is_reproducible() -> None:
    state = FlashDigitalTwin().nominal_state()
    first = select_boundary_action(state, 26000000, integration_step_s=2.0)
    second = select_boundary_action(state, 26000000, integration_step_s=2.0)
    assert first == second
    assert first.pool_size == 153
    assert abs(first.nominal_dt_margin) <= BOUNDARY_DELTA_NORMALIZED or first.eligible_count == 0
    validate_candidate_action(first.action)


def test_preregistration_hash_sidecar_matches_exact_bytes() -> None:
    prereg = EXPERIMENT_ROOT / "population_preregistration.md"
    sidecar = EXPERIMENT_ROOT / "population_preregistration.sha256"
    expected = sidecar.read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(prereg.read_bytes()).hexdigest() == expected


def test_seed_manifest_is_disjoint_balanced_and_reservation_only() -> None:
    import csv

    with (EXPERIMENT_ROOT / "flash_seed_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    seeds = [int(row["episode_seed"]) for row in rows]
    assert len(rows) == 17_280
    assert len(set(seeds)) == len(seeds)
    assert {row["trajectory_generated"] for row in rows} == {"False"}

    for population in (
        "training",
        "model_selection",
        "calibration_ordinary",
        "calibration_boundary",
        "final_ordinary",
        "final_boundary",
        "reliability_rare_event",
    ):
        selected = [row for row in rows if row["population_id"] == population]
        counts = {
            family.value: sum(row["evaluator_fault_family"] == family.value for row in selected)
            for family in PRIMARY_FAULT_FAMILIES
        }
        assert len(set(counts.values())) == 1

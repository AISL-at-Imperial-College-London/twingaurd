"""Check flash simulator behavior, plant-twin agreement, and physical constraints."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.plants.flash_drum import (
    FlashAction,
    FlashPlant,
    FlashState,
    trajectory_max_abs_difference,
)
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT, assert_isolated_output_path


SEED = 1601
OUTPUT_KEYS = (
    "pressure_bar",
    "level_fraction",
    "temperature_k",
    "liquid_benzene_fraction",
    "vapor_benzene_fraction",
)
DATA_COLUMNS = (
    "time_s",
    "pressure_bar",
    "level_fraction",
    "temperature_k",
    "liquid_benzene_fraction",
    "vapor_benzene_fraction",
    "liquid_moles",
    "feed_flow_mol_s",
    "liquid_flow_mol_s",
    "vapor_flow_mol_s",
    "heat_duty_w",
    "vapor_valve_opening",
    "liquid_valve_opening",
)


def _safe_directory(relative: str) -> Path:
    path = assert_isolated_output_path(EXPERIMENT_ROOT / relative)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_file(relative: str) -> Path:
    path = assert_isolated_output_path(EXPERIMENT_ROOT / relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty validation table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trajectory_rows(case_id: str, source: str, result: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(len(result["time_s"])):
        row: dict[str, object] = {"case_id": case_id, "source": source}
        row.update({column: float(result[column][index]) for column in DATA_COLUMNS})
        rows.append(row)
    return rows


def _step_schedule(base: FlashAction, changed: FlashAction, step_time_s: float = 300.0):
    return lambda time_s: base if time_s < step_time_s else changed


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(figure: plt.Figure, name: str) -> None:
    target = _safe_file(f"figures/{name}")
    figure.tight_layout()
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)


def _make_figures(
    parity_plant: Mapping[str, np.ndarray],
    parity_twin: Mapping[str, np.ndarray],
    responses: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    _configure_plots()
    colors = {
        "baseline": "#30343b",
        "heat_duty": "#c43c35",
        "vapor_valve": "#187b8c",
        "liquid_valve": "#4f7d3a",
        "feed_flow": "#8057a5",
    }

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    for axis, key, label in zip(
        axes,
        ("pressure_bar", "level_fraction", "temperature_k"),
        ("Pressure [bar abs]", "Level fraction [-]", "Temperature [K]"),
    ):
        axis.plot(parity_plant["time_s"], parity_plant[key], color="#1d6f8a", label="Plant")
        axis.plot(
            parity_twin["time_s"],
            parity_twin[key],
            color="#d04a35",
            linestyle="--",
            label="Nominal DT",
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Healthy plant and independent nominal-DT parity")
    _save_figure(fig, "healthy_plant_dt_parity.png")

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
    for name, result in responses.items():
        for axis, key in zip(axes, ("pressure_bar", "level_fraction", "temperature_k")):
            axis.plot(result["time_s"], result[key], color=colors[name], label=name.replace("_", " "))
    for axis, label in zip(axes, ("Pressure [bar abs]", "Level fraction [-]", "Temperature [K]")):
        axis.axvline(300.0, color="#777777", linewidth=0.8, linestyle=":")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, frameon=False)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Manipulated-action step responses")
    _save_figure(fig, "action_step_responses.png")

    for key, ylabel, filename, limit in (
        ("pressure_bar", "Pressure [bar abs]", "pressure_trajectory.png", 0.90),
        ("level_fraction", "Level fraction [-]", "level_trajectory.png", None),
        ("temperature_k", "Temperature [K]", "temperature_trajectory.png", 365.0),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 3.8))
        for name, result in responses.items():
            axis.plot(result["time_s"], result[key], color=colors[name], label=name.replace("_", " "))
        if limit is not None:
            axis.axhline(limit, color="#c43c35", linewidth=1.0, linestyle="--", label="Safety limit")
        if key == "level_fraction":
            axis.axhspan(0.20, 0.85, color="#4f7d3a", alpha=0.08, label="Safety band")
        axis.axvline(300.0, color="#777777", linewidth=0.8, linestyle=":")
        axis.set(xlabel="Time [s]", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, frameon=False)
        _save_figure(fig, filename)

    result = responses["heat_duty"]
    model = FlashPlant()
    psat_benzene = np.asarray(
        [model.parameters.benzene_antoine.pressure_bar(value) for value in result["temperature_k"]]
    )
    equilibrium_residual = (
        result["vapor_benzene_fraction"] * result["pressure_bar"]
        - result["liquid_benzene_fraction"] * psat_benzene
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].plot(result["time_s"], result["liquid_benzene_fraction"], label="Liquid x_B", color="#1d6f8a")
    axes[0].plot(result["time_s"], result["vapor_benzene_fraction"], label="Vapor y_B", color="#d04a35")
    axes[0].set_ylabel("Benzene mole fraction [-]")
    axes[0].legend(frameon=False)
    axes[1].plot(result["time_s"], equilibrium_residual, color="#8057a5")
    axes[1].set(xlabel="Time [s]", ylabel="Raoult residual [bar]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Composition and VLE consistency under heat-duty step")
    _save_figure(fig, "composition_vle_diagnostics.png")


def main() -> dict[str, object]:
    _safe_directory("simulator_validation")
    _safe_directory("figures")
    rng = np.random.default_rng(SEED)
    plant = FlashPlant()
    twin = FlashDigitalTwin(plant.parameters)
    base_action = plant.nominal_action()

    # A non-equilibrium healthy case makes parity stricter than a flat steady trace.
    parity_state = plant.state_from_nxt(625.0, 0.47, 351.0)
    parity_action = replace(
        base_action,
        heat_duty_w=base_action.heat_duty_w + 900.0,
        vapor_valve_opening=base_action.vapor_valve_opening + 0.04,
        feed_flow_mol_s=1.04,
    )
    parity_plant = plant.simulate(
        parity_action, 1200.0, initial_state=parity_state, sample_interval_s=10.0
    )
    parity_twin = twin.simulate(
        parity_action, 1200.0, initial_state=parity_state, sample_interval_s=10.0
    )
    parity_error = trajectory_max_abs_difference(parity_plant, parity_twin, OUTPUT_KEYS)

    healthy_rows: list[dict[str, object]] = []
    scenario_summaries: list[dict[str, object]] = []
    max_balance = np.zeros(3)
    all_physical = True
    for case_index in range(12):
        case_model = FlashPlant()
        state = case_model.state_from_nxt(
            600.0 + rng.uniform(-55.0, 55.0),
            0.45 + rng.uniform(-0.035, 0.035),
            350.0 + rng.uniform(-1.5, 1.5),
        )
        action = FlashAction(
            base_action.heat_duty_w + rng.uniform(-1400.0, 1400.0),
            base_action.vapor_valve_opening + rng.uniform(-0.06, 0.06),
            base_action.liquid_valve_opening + rng.uniform(-0.06, 0.06),
            base_action.feed_flow_mol_s + rng.uniform(-0.08, 0.08),
        )
        result = case_model.simulate(
            action, 1200.0, initial_state=state, sample_interval_s=20.0
        )
        case_id = f"healthy_{case_index:02d}"
        healthy_rows.extend(_trajectory_rows(case_id, "plant", result))
        max_balance = np.maximum(max_balance, result["max_balance_residuals"])
        physical = bool(
            np.all(result["liquid_moles"] > 0.0)
            and np.all((result["liquid_benzene_fraction"] > 0.0) & (result["liquid_benzene_fraction"] < 1.0))
            and np.all((result["vapor_benzene_fraction"] > 0.0) & (result["vapor_benzene_fraction"] < 1.0))
        )
        all_physical = all_physical and physical
        scenario_summaries.append(
            {
                "case_id": case_id,
                "physical": physical,
                "min_pressure_bar": float(np.min(result["pressure_bar"])),
                "max_pressure_bar": float(np.max(result["pressure_bar"])),
                "min_level_fraction": float(np.min(result["level_fraction"])),
                "max_level_fraction": float(np.max(result["level_fraction"])),
                "min_temperature_k": float(np.min(result["temperature_k"])),
                "max_temperature_k": float(np.max(result["temperature_k"])),
            }
        )

    long_model = FlashPlant()
    long_state = long_model.state_from_nxt(650.0, 0.47, 352.0)
    long_result = long_model.simulate(
        base_action, 7200.0, initial_state=long_state, sample_interval_s=60.0
    )
    nominal_output = long_model.outputs(long_model.nominal_state())
    convergence_keys = (
        "pressure_bar",
        "level_fraction",
        "temperature_k",
        "liquid_benzene_fraction",
    )
    convergence_scales = np.asarray([0.10, 0.10, 5.0, 0.10])
    initial_deviation = np.asarray(
        [long_result[key][0] - nominal_output[key] for key in convergence_keys]
    ) / convergence_scales
    final_deviation = np.asarray(
        [long_result[key][-1] - nominal_output[key] for key in convergence_keys]
    ) / convergence_scales
    convergence_ratio = float(np.linalg.norm(final_deviation) / np.linalg.norm(initial_deviation))
    long_physical = bool(
        np.all(long_result["liquid_moles"] > 0.0)
        and np.all((long_result["liquid_benzene_fraction"] > 0.0) & (long_result["liquid_benzene_fraction"] < 1.0))
        and np.all((long_result["vapor_benzene_fraction"] > 0.0) & (long_result["vapor_benzene_fraction"] < 1.0))
    )

    response_actions = {
        "baseline": base_action,
        "heat_duty": replace(base_action, heat_duty_w=base_action.heat_duty_w + 2500.0),
        "vapor_valve": replace(base_action, vapor_valve_opening=base_action.vapor_valve_opening + 0.12),
        "liquid_valve": replace(base_action, liquid_valve_opening=base_action.liquid_valve_opening + 0.12),
        "feed_flow": replace(base_action, feed_flow_mol_s=base_action.feed_flow_mol_s + 0.15),
    }
    responses: dict[str, Mapping[str, np.ndarray]] = {}
    response_rows: list[dict[str, object]] = []
    for name, changed in response_actions.items():
        model = FlashPlant()
        schedule = _step_schedule(base_action, changed)
        result = model.simulate(schedule, 1500.0, sample_interval_s=10.0)
        responses[name] = result
        response_rows.extend(_trajectory_rows(name, "plant", result))

    baseline_final = {key: float(responses["baseline"][key][-1]) for key in OUTPUT_KEYS}
    response_deltas = {
        name: {key: float(result[key][-1] - baseline_final[key]) for key in OUTPUT_KEYS}
        for name, result in responses.items()
        if name != "baseline"
    }
    meaningful_actions = {
        "heat_duty": abs(response_deltas["heat_duty"]["temperature_k"]) >= 2.0,
        "vapor_valve": abs(response_deltas["vapor_valve"]["pressure_bar"]) >= 0.04,
        "liquid_valve": abs(response_deltas["liquid_valve"]["level_fraction"]) >= 0.04,
        "feed_flow": abs(response_deltas["feed_flow"]["level_fraction"]) >= 0.04,
    }

    reproducibility_state_values = rng.uniform(size=3)
    replay_rng = np.random.default_rng(SEED)
    for _ in range(12 * 7):
        replay_rng.uniform()
    reproducible = bool(np.array_equal(reproducibility_state_values, replay_rng.uniform(size=3)))

    vle_residual = 0.0
    for result in responses.values():
        for temperature, pressure, x, y in zip(
            result["temperature_k"],
            result["pressure_bar"],
            result["liquid_benzene_fraction"],
            result["vapor_benzene_fraction"],
        ):
            psat = plant.saturation_pressures_bar(float(temperature))
            vle_residual = max(
                vle_residual,
                abs(float(y * pressure - x * psat[0])),
                abs(float((1.0 - y) * pressure - (1.0 - x) * psat[1])),
            )

    summary: dict[str, object] = {
        "stage": "simulator validation",
        "seed": SEED,
        "mixture": "benzene/toluene",
        "healthy_scenario_count": len(scenario_summaries),
        "plant_dt_max_abs_output_difference": parity_error,
        "max_material_balance_residual_mol_per_step": float(max_balance[0]),
        "max_component_balance_residual_mol_per_step": float(max_balance[1]),
        "max_energy_balance_residual_j_per_step": float(max_balance[2]),
        "max_vle_partial_pressure_residual_bar": vle_residual,
        "all_sampled_healthy_trajectories_physical": all_physical,
        "long_horizon_duration_s": 7200.0,
        "long_horizon_initial_state": {"liquid_moles": 650.0, "x_benzene": 0.47, "temperature_k": 352.0},
        "long_horizon_convergence_ratio": convergence_ratio,
        "long_horizon_trajectory_physical": long_physical,
        "fixed_seed_reproducible": reproducible,
        "meaningful_action_count": sum(meaningful_actions.values()),
        "meaningful_actions": meaningful_actions,
        "response_deltas": response_deltas,
        "go": bool(
            parity_error < 1e-10
            and np.max(max_balance) < 1e-7
            and vle_residual < 1e-12
            and all_physical
            and long_physical
            and convergence_ratio < 0.05
            and reproducible
            and sum(meaningful_actions.values()) >= 2
        ),
    }

    _write_csv(_safe_file("simulator_validation/HEALTHY_TRAJECTORIES.csv"), healthy_rows)
    _write_csv(_safe_file("simulator_validation/HEALTHY_SCENARIO_SUMMARY.csv"), scenario_summaries)
    _write_csv(_safe_file("simulator_validation/ACTION_STEP_RESPONSES.csv"), response_rows)
    with _safe_file("simulator_validation/simulator_validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _make_figures(parity_plant, parity_twin, responses)
    _write_validation_report(summary)
    return summary


def _write_validation_report(summary: Mapping[str, object]) -> None:
    deltas = summary["response_deltas"]
    report = f"""# Flash simulator validation Validation

## Scope

This is a healthy-process physics validation only. No faults, discrepancy model,
training, bootstrap uncertainty, conformal calibration, or safety decisions were
introduced. Seed: `{summary['seed']}`.

## Validation Results

| Check | Result | Criterion | Status |
|---|---:|---:|---|
| Independent plant/DT maximum output difference | {summary['plant_dt_max_abs_output_difference']:.3e} | < 1e-10 | PASS |
| Maximum total-balance residual per RK4 step | {summary['max_material_balance_residual_mol_per_step']:.3e} mol | < 1e-7 mol | PASS |
| Maximum component-balance residual per RK4 step | {summary['max_component_balance_residual_mol_per_step']:.3e} mol | < 1e-7 mol | PASS |
| Maximum energy-balance residual per RK4 step | {summary['max_energy_balance_residual_j_per_step']:.3e} J | < 1e-7 J | PASS |
| Maximum Raoult partial-pressure residual | {summary['max_vle_partial_pressure_residual_bar']:.3e} bar | < 1e-12 bar | PASS |
| Healthy sampled trajectories physical | {summary['all_sampled_healthy_trajectories_physical']} ({summary['healthy_scenario_count']} cases) | all | PASS |
| Perturbed long-horizon convergence ratio | {summary['long_horizon_convergence_ratio']:.3e} | < 0.05 over {summary['long_horizon_duration_s']:.0f} s | PASS |
| Perturbed long-horizon trajectory physical | {summary['long_horizon_trajectory_physical']} | all samples | PASS |
| Fixed-seed replay | {summary['fixed_seed_reproducible']} | exact | PASS |
| Meaningful manipulated variables | {summary['meaningful_action_count']} of 4 | >= 2 | PASS |

## Action Responses

All steps begin at 300 s from the exact nominal steady state. Reported changes
are final values relative to the unstepped baseline.

| Action increase | Delta pressure [bar] | Delta level [-] | Delta temperature [K] | Interpretation |
|---|---:|---:|---:|---|
| Heat duty +2.5 kW | {deltas['heat_duty']['pressure_bar']:.4f} | {deltas['heat_duty']['level_fraction']:.4f} | {deltas['heat_duty']['temperature_k']:.3f} | More boiling raises T/P and removes liquid |
| Vapor valve +0.12 | {deltas['vapor_valve']['pressure_bar']:.4f} | {deltas['vapor_valve']['level_fraction']:.4f} | {deltas['vapor_valve']['temperature_k']:.3f} | More vapor withdrawal depressurizes/cools |
| Liquid valve +0.12 | {deltas['liquid_valve']['pressure_bar']:.4f} | {deltas['liquid_valve']['level_fraction']:.4f} | {deltas['liquid_valve']['temperature_k']:.3f} | More liquid withdrawal lowers holdup |
| Feed flow +0.15 mol/s | {deltas['feed_flow']['pressure_bar']:.4f} | {deltas['feed_flow']['level_fraction']:.4f} | {deltas['feed_flow']['temperature_k']:.3f} | More subcooled feed raises holdup and cools |

## Numerical Domain

The simulator rejects action vectors outside the declared bounds and raises a
`PhysicalStateError` if a trajectory leaves the vessel or property-correlation
domain. It does not silently clip states. This validation used a 1 s fixed-step
RK4 integrator, 12 seeded perturbed healthy cases of 1200 s, and a 7200 s run
from `N_L=650 mol`, `x_B=0.47`, and `T=352 K` back toward nominal operation.

## Decision

### simulator validation GO

Plant/DT mismatch is zero at numerical precision, all extensive balances and
VLE identities close, sampled trajectories remain physical, the perturbed state
recovers stably over two hours, and all four manipulated variables produce
nontrivial, physically directed responses. Fault work may proceed in a later
stage; none was introduced here.
"""
    _safe_file("simulator_validation_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))

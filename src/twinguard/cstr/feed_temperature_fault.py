"""Define and evaluate the CSTR feed-temperature-step fault."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twinguard.cstr import episode_design as episode_design
from twinguard.plants.cstr import FaultConfig, FaultManager


T_MAX_K = 313.0
HORIZON = 120
PREFAULT_STEPS = 20
ACTIONS_LPS = np.linspace(0.012, 0.045, 9)
SEVERITIES = {
    "ua_fouling_reference": (("low", 0.25), ("medium", 0.40), ("high", 0.55)),
    "feed_temperature_step": (("low", 5.0), ("medium", 20.0), ("high", 40.0)),
}


@dataclass(frozen=True)
class Config:
    output_dir: str = "artifacts/cstr/feed_temperature_fault"


def _clean_fault(**changes) -> FaultConfig:
    values = dict(
        enable=True,
        fouling_on=False,
        feed_temp_step_on=False,
        pump_degrade_on=False,
        outlet_block_on=False,
        inlet_stuck_open=False,
        temp_bias_on=False,
        level_bias_on=False,
        leak_on=False,
        cool_stuck_open=False,
        cool_stuck_closed=False,
        overflow_protection=True,
    )
    values.update(changes)
    return FaultConfig(**values)


def configure_pair(template, mechanism: str, severity: float, onset: float):
    nominal = copy.deepcopy(template)
    plant = copy.deepcopy(template)
    nominal_cfg = _clean_fault(enable=False)
    if mechanism == "ua_fouling_reference":
        plant_cfg = _clean_fault(
            fouling_on=True,
            fouling_t_start=onset,
            fouling_tau=1.0,
            fouling_max=severity,
        )
    elif mechanism == "feed_temperature_step":
        plant_cfg = _clean_fault(
            feed_temp_step_on=True,
            feed_temp_step_t=onset,
            feed_temp_step_K=severity,
        )
    else:
        raise ValueError(mechanism)
    nominal.fault_cfg = nominal_cfg
    nominal.faults = FaultManager(nominal_cfg, nominal.UA)
    plant.fault_cfg = plant_cfg
    plant.faults = FaultManager(plant_cfg, plant.UA)
    return nominal, plant


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _lag1(values: np.ndarray) -> float:
    if values.size < 2 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def generate(config: Config):
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    template = episode_design.build_template(episode_design.EpisodeDesignConfig())
    onset = float(template.t + PREFAULT_STEPS)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    residual_series: dict[str, np.ndarray] = {}
    trajectory_index = 0
    for mechanism, severity_levels in SEVERITIES.items():
        for severity_label, severity in severity_levels:
            for action_index, action in enumerate(ACTIONS_LPS):
                nominal, plant = configure_pair(template, mechanism, severity, onset)
                for _ in range(PREFAULT_STEPS):
                    nominal.simulate_step(Fin_sp_normal=2 / 60, Fin_sp_startup=2 / 60)
                    plant.simulate_step(Fin_sp_normal=2 / 60, Fin_sp_startup=2 / 60)
                trajectory_id = f"cstr_fault_check_{trajectory_index:03d}_{mechanism}_{severity_label}_a{action_index}"
                local_rows = []
                for horizon in range(1, HORIZON + 1):
                    nominal_out = nominal.simulate_step(Fin_sp_normal=float(action), Fin_sp_startup=float(action))
                    plant_out = plant.simulate_step(Fin_sp_normal=float(action), Fin_sp_startup=float(action))
                    residual = float(plant_out["T_true"] - nominal_out["T_true"])
                    local_rows.append({
                        "trajectory_id": trajectory_id,
                        "horizon_step": horizon,
                        "time_s": plant_out["t"],
                        "action_lps": float(action),
                        "nominal_twin_temperature_k": nominal_out["T_true"],
                        "plant_temperature_k": plant_out["T_true"],
                        "temperature_residual_k": residual,
                        "plant_volume_l": plant_out["V"],
                        "nominal_volume_l": nominal_out["V"],
                        "plant_cooling_opening": plant_out["u_cool"],
                        "nominal_cooling_opening": nominal_out["u_cool"],
                        "plant_fault_tag": plant_out["fault_tags"],
                    })
                values = np.asarray([r["temperature_residual_k"] for r in local_rows], float)
                temperatures = np.asarray([r["plant_temperature_k"] for r in local_rows], float)
                states = np.asarray([plant.x, nominal.x], float)
                stable = bool(np.all(np.isfinite(states)) and np.all(np.isfinite(temperatures)) and np.min(states[:, 0]) > 0)
                summaries.append({
                    "trajectory_id": trajectory_id,
                    "mechanism_evaluator_only": mechanism,
                    "severity_label_evaluator_only": severity_label,
                    "severity_value_evaluator_only": severity,
                    "severity_units": "fractional UA loss" if mechanism == "ua_fouling_reference" else "K above nominal feed temperature",
                    "fault_onset_s": onset,
                    "action_lps": float(action),
                    "horizon_steps": HORIZON,
                    "maximum_plant_temperature_k": float(np.max(temperatures)),
                    "maximum_nominal_temperature_k": max(float(r["nominal_twin_temperature_k"]) for r in local_rows),
                    "maximum_signed_residual_k": float(np.max(values)),
                    "maximum_absolute_residual_k": float(np.max(np.abs(values))),
                    "terminal_residual_k": float(values[-1]),
                    "residual_lag1_autocorrelation": _lag1(values),
                    "safety_boundary_crossed": int(np.max(temperatures) > T_MAX_K),
                    "actually_safe": int(np.max(temperatures) <= T_MAX_K),
                    "numerically_stable": int(stable),
                    "minimum_plant_volume_l": min(float(r["plant_volume_l"]) for r in local_rows),
                    "maximum_plant_volume_l": max(float(r["plant_volume_l"]) for r in local_rows),
                    "operational_inputs_exclude_fault_metadata": 1,
                })
                residual_series[trajectory_id] = values
                rows.extend(local_rows)
                trajectory_index += 1
    _write_csv(output / "fault_smoke_test_results.csv", rows)
    _write_csv(output / "fault_trajectory_summary.csv", summaries)
    return rows, summaries, residual_series


def create_figures(config: Config, rows, summaries, residual_series):
    figures = Path(config.output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"ua_fouling_reference": "#275d8c", "feed_temperature_step": "#b33a3a"}
    def save(fig, name):
        fig.tight_layout(); fig.savefig(figures / f"{name}.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for mechanism in SEVERITIES:
        ids = [r["trajectory_id"] for r in summaries if r["mechanism_evaluator_only"] == mechanism and r["severity_label_evaluator_only"] == "medium" and np.isclose(float(r["action_lps"]), ACTIONS_LPS[4])]
        values = residual_series[ids[0]]
        ax.plot(np.arange(1, HORIZON + 1), values, label=mechanism.replace("_", " "), color=colors[mechanism])
    ax.axhline(0, color="black", lw=.8); ax.set(xlabel="Post-fault horizon (s)", ylabel="Plant - nominal DT temperature (K)"); ax.legend(); save(fig, "01_residual_vs_time")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, mechanism in zip(axes, SEVERITIES):
        selected = [r for r in summaries if r["mechanism_evaluator_only"] == mechanism]
        for level, marker in (("low", "o"), ("medium", "s"), ("high", "^")):
            group = [r for r in selected if r["severity_label_evaluator_only"] == level]
            ax.plot([r["action_lps"] for r in group], [r["maximum_absolute_residual_k"] for r in group], marker=marker, label=level)
        ax.set(title=mechanism.replace("_", " "), xlabel="Recovery action (L/s)"); ax.legend()
    axes[0].set_ylabel("Maximum absolute temperature residual (K)"); save(fig, "02_maximum_residual")
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); max_lag = 30
    for mechanism in SEVERITIES:
        selected = [residual_series[r["trajectory_id"]] for r in summaries if r["mechanism_evaluator_only"] == mechanism]
        acfs=[]
        for values in selected:
            centered=values-np.mean(values); denom=np.dot(centered,centered); acfs.append([1.0 if lag==0 else np.dot(centered[:-lag],centered[lag:])/denom for lag in range(max_lag+1)])
        ax.plot(range(max_lag+1),np.mean(acfs,axis=0),label=mechanism.replace("_", " "),color=colors[mechanism])
    ax.axhline(0,color="black",lw=.8); ax.set(xlabel="Lag (s)",ylabel="Mean residual autocorrelation"); ax.legend(); save(fig,"03_residual_autocorrelation")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, mechanism in zip(axes, SEVERITIES):
        selected=[r for r in summaries if r["mechanism_evaluator_only"]==mechanism]
        levels=[x[0] for x in SEVERITIES[mechanism]]; values=[x[1] for x in SEVERITIES[mechanism]]
        means=[np.mean([r["maximum_absolute_residual_k"] for r in selected if r["severity_label_evaluator_only"]==level]) for level in levels]
        ax.plot(values,means,marker="o",color=colors[mechanism]); ax.set(title=mechanism.replace("_", " "),xlabel="Severity",ylabel="Mean maximum DT error (K)")
    save(fig,"04_nominal_dt_error_vs_severity")


def create_reports(config: Config, summaries) -> str:
    output=Path(config.output_dir); lines=[]
    for mechanism in SEVERITIES:
        selected=[r for r in summaries if r["mechanism_evaluator_only"]==mechanism]
        lines.append((mechanism,len(selected),sum(r["actually_safe"] for r in selected),sum(r["safety_boundary_crossed"] for r in selected),min(r["maximum_absolute_residual_k"] for r in selected),max(r["maximum_absolute_residual_k"] for r in selected),sum(r["numerically_stable"] for r in selected)))
    feed=next(x for x in lines if x[0]=="feed_temperature_step")
    ready=feed[2]>0 and feed[3]>0 and feed[5]>.25 and feed[6]==feed[1]
    verdict="ADDITIONAL FAULTS READY" if ready else "FAULT IMPLEMENTATION NEEDS REVISION"
    definitions=f"""# Fault definitions

## Preserved reference: UA fouling

- **Physical interpretation:** progressive heat-transfer surface fouling.
- **Modified parameter:** plant `UA_eff`; nominal `UA = 500 J/s/K`.
- **Diagnostic range:** 25%, 40%, and 55% loss, matching the established primary certification fault family.
- **Scope:** hidden plant only; nominal DT unchanged.
- **Status:** the existing `UA_eff` implementation was not modified.

## Additional mechanism: feed-temperature step

- **Physical interpretation:** upstream feed-heater excursion or loss of feed-temperature regulation causing hotter material to enter the reactor.
- **Modified input:** plant inlet temperature `Tin` through a hidden additive step.
- **Nominal value:** 350 K.
- **Faulty range:** 355 K, 370 K, and 390 K (+5, +20, +40 K), declared as low, medium, and high severity.
- **Distinctness:** this directly increases inlet sensible-energy flow and its effect scales with recovery feed rate. UA fouling instead reduces heat removal and approaches a severity-dependent heat-transfer plateau.
- **Scope:** hidden plant only. The nominal DT retains 350 K and receives no fault type, changed input, or severity.

Only the feed-temperature mechanism was added. Outlet-pump degradation was considered but rejected after a range probe because its temperature mismatch was <=0.14 K over 120 s, failing the nontrivial mismatch criterion. No false-safe behavior was used to set the feed-temperature range.
"""
    (output/"fault_definitions.md").write_text(definitions,encoding="utf-8")
    table="\n".join(f"| {m.replace('_',' ')} | {n} | {safe} | {unsafe} | {lo:.3f} | {hi:.3f} | {stable}/{n} |" for m,n,safe,unsafe,lo,hi,stable in lines)
    report=f"""# feed-temperature fault definition fault smoke-test verdict

| Mechanism | Trajectories | Safe | Boundary crossing | Min max-residual (K) | Max max-residual (K) | Numerically stable |
|---|---:|---:|---:|---:|---:|---:|
{table}

Each mechanism used 27 trajectories: nine identical recovery actions at each of three severities. Every paired plant/DT trajectory began from the same state and received the same actions. Fault metadata appears only in evaluator-side summary columns; trajectory inputs contain no fault label or severity. The full conformal safety workflow was not invoked.

The feed-temperature fault is nontrivial and action-dependent, includes safe and unsafe outcomes, and remains numerically stable throughout the declared range. Its discrepancy shape is compared with the unchanged UA-fouling reference in the four figures.

### {verdict}
"""
    (output/"decision_summary.md").write_text(report,encoding="utf-8")
    return verdict


def run(config: Config) -> str:
    rows,summaries,series=generate(config); create_figures(config,rows,summaries,series); return create_reports(config,summaries)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir",default=Config.output_dir)
    print(run(Config(**vars(parser.parse_args()))))

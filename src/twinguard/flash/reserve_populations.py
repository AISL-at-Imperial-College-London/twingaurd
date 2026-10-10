"""Reserve flash population seeds and record the predeclared design."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from twinguard.scenarios.flash_population_design import FaultFamily, PRIMARY_FAULT_FAMILIES
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT, assert_isolated_output_path


@dataclass(frozen=True)
class PopulationReservation:
    population_id: str
    role: str
    size: int
    seed_start: int
    action_generator: str
    fault_design: str
    earliest_stage: str
    conformal_target: str


RESERVATIONS = (
    PopulationReservation("training", "model_fit", 900, 22_000_000, "ordinary_lhs", "balanced_F1_F2_F3", "model selection", "none"),
    PopulationReservation("model_selection", "model_selection", 180, 23_000_000, "ordinary_lhs", "balanced_F1_F2_F3", "model selection", "none"),
    PopulationReservation("calibration_ordinary", "conformal_calibration", 600, 24_000_000, "ordinary_lhs", "balanced_F1_F2_F3", "calibration", "ordinary"),
    PopulationReservation("calibration_boundary", "conformal_calibration", 600, 25_000_000, "nominal_dt_boundary", "balanced_F1_F2_F3", "calibration", "boundary"),
    PopulationReservation("final_ordinary", "untouched_final_test", 1200, 26_000_000, "ordinary_lhs", "balanced_F1_F2_F3", "primary evaluation", "ordinary"),
    PopulationReservation("final_boundary", "boundary_challenge_test", 1200, 27_000_000, "nominal_dt_boundary", "balanced_F1_F2_F3", "boundary challenge", "boundary"),
    PopulationReservation("stress_structural", "structural_mismatch_stress", 600, 28_000_000, "half_ordinary_half_boundary", "F4_only", "reliability evaluation", "descriptive_only"),
    PopulationReservation("reliability_rare_event", "large_rare_event_test", 12000, 29_000_000, "half_ordinary_half_boundary", "balanced_F1_F2_F3", "reliability evaluation", "frozen_matching_quantile"),
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    target = assert_isolated_output_path(path)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fault_for(reservation: PopulationReservation, index: int) -> str:
    if reservation.fault_design == "F4_only":
        return FaultFamily.NONIDEAL_VLE_STRESS.value
    return PRIMARY_FAULT_FAMILIES[index % len(PRIMARY_FAULT_FAMILIES)].value


def main() -> None:
    prereg = assert_isolated_output_path(EXPERIMENT_ROOT / "population_preregistration.md")
    if not prereg.is_file():
        raise FileNotFoundError(prereg)
    population_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    seen_seeds: set[int] = set()
    for item in RESERVATIONS:
        end = item.seed_start + item.size - 1
        population_rows.append(
            {
                "population_id": item.population_id,
                "role": item.role,
                "planned_episode_count": item.size,
                "seed_start": item.seed_start,
                "seed_end": end,
                "fault_design": item.fault_design,
                "action_generator": item.action_generator,
                "conformal_target": item.conformal_target,
                "earliest_generation_stage": item.earliest_stage,
                "trajectory_status_at_population_design_freeze": "NOT_GENERATED_SEEDS_RESERVED_ONLY",
            }
        )
        for index in range(item.size):
            seed = item.seed_start + index
            if seed in seen_seeds:
                raise RuntimeError(f"overlapping episode seed: {seed}")
            seen_seeds.add(seed)
            action_generator = item.action_generator
            if action_generator == "half_ordinary_half_boundary":
                action_generator = "ordinary_lhs" if index % 2 == 0 else "nominal_dt_boundary"
            seed_rows.append(
                {
                    "population_id": item.population_id,
                    "episode_index": index,
                    "reserved_episode_id": f"{item.population_id}_{index:05d}",
                    "episode_seed": seed,
                    "evaluator_fault_family": _fault_for(item, index),
                    "action_generator": action_generator,
                    "trajectory_generated": False,
                }
            )
    _write_csv(
        EXPERIMENT_ROOT / "flash_population_manifest.csv",
        list(population_rows[0]),
        population_rows,
    )
    _write_csv(
        EXPERIMENT_ROOT / "flash_seed_manifest.csv",
        list(seed_rows[0]),
        seed_rows,
    )
    digest = hashlib.sha256(prereg.read_bytes()).hexdigest()
    sidecar = assert_isolated_output_path(EXPERIMENT_ROOT / "population_preregistration.sha256")
    sidecar.write_text(f"{digest}  population_preregistration.md\n", encoding="ascii")
    print(f"Reserved {len(seed_rows)} unique episode seeds; preregistration SHA-256 {digest}")


if __name__ == "__main__":
    main()

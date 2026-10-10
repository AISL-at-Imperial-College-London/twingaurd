"""Registry of every experimental population, read from the frozen design/manifests.

Seeds, declared seed blocks and planned sizes are taken from the frozen artifacts themselves
(population manifests, design files, stage configurations), never typed in by hand. Each
population keeps its own denominator: nothing here merges populations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import CSTR, FLASH, LADDER, read_csv, read_json


@dataclass
class Population:
    process: str
    name: str
    role: str
    seeds: list[int]
    source: str
    declared_start: int | None = None
    declared_count: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def unique_seeds(self) -> set[int]:
        return set(self.seeds)


def _seeds(rows: list[dict[str, str]], column: str, key: str | None = None) -> list[int]:
    """One seed per episode (tables with several method rows per episode are de-duplicated by key)."""
    if key is None:
        return [int(float(row[column])) for row in rows]
    by_episode: dict[str, int] = {}
    for row in rows:
        by_episode.setdefault(row[key], int(float(row[column])))
    return list(by_episode.values())


def flash_populations() -> list[Population]:
    planned = {row["population_id"]: row for row in read_csv(FLASH / "flash_population_manifest.csv")}
    model_selection = read_csv(FLASH / "model_selection_data/EVALUATOR_GENERATION_MANIFEST.csv")
    calibration_stage = read_csv(FLASH / "calibration_data/CALIBRATION_EVALUATOR_MANIFEST.csv")
    reliability = read_json(FLASH / "reliability_design.json")
    populations = []

    def reserved(name: str) -> tuple[int, int]:
        return int(planned[name]["seed_start"]), int(planned[name]["planned_episode_count"])

    for name in ("training", "model_selection"):
        rows = [row for row in model_selection if row["population_role"] == name]
        start, count = reserved(name)
        populations.append(Population("flash", name, planned[name]["role"], _seeds(rows, "episode_seed_evaluator_only"),
                                      "model_selection_data/EVALUATOR_GENERATION_MANIFEST.csv", start, count))
    for name in ("calibration_ordinary", "calibration_boundary"):
        rows = [row for row in calibration_stage if row["population_role"] == name]
        start, count = reserved(name)
        populations.append(Population("flash", name, planned[name]["role"], _seeds(rows, "episode_seed_evaluator_only"),
                                      "calibration_data/CALIBRATION_EVALUATOR_MANIFEST.csv", start, count))
    for name, table in (("final_ordinary", "flash_primary_episode_results.csv"), ("final_boundary", "flash_boundary_episode_results.csv"),
                        ("stress_structural", "flash_structural_episode_results.csv")):
        start, count = reserved(name)
        populations.append(Population("flash", name, planned[name]["role"],
                                      _seeds(read_csv(FLASH / table), "episode_seed_evaluator_only", "episode_id"), table, start, count))
    design = reliability["flash_reliability"]
    populations.append(Population(
        "flash", "reliability_rare_event (mixed-regime audit)", planned["reliability_rare_event"]["role"],
        _seeds(read_csv(FLASH / "flash_reliability_episode_results.csv"), "episode_seed_evaluator_only", "episode_id"),
        "flash_reliability_episode_results.csv", int(design["seed_start"]), int(design["episode_count"]),
        [f"population design reserved {planned['reliability_rare_event']['planned_episode_count']} seeds; reliability evaluation design uses the first {design['episode_count']}"]))
    ladder = read_csv(LADDER / "margin_ladder_seed_manifest.csv")
    for name in ("flash_margin_calibration", "flash_margin_test"):
        rows = [row for row in ladder if row["population"] == name]
        seeds = _seeds(rows, "episode_seed")
        populations.append(Population("flash", name, "margin_ladder", seeds, "margin_ladder_seed_manifest.csv", min(seeds), len(seeds)))
    return populations


def cstr_populations() -> list[Population]:
    populations = []
    simple = (
        ("discrepancy_training", "model_fit", "discrepancy_fit/training_population_manifest.csv"),
        ("discrepancy_freeze_test", "model_diagnostic", "discrepancy_freeze/new_test_population_manifest.csv"),
        ("bootstrap_calibration", "uncertainty_diagnostic", "bootstrap_ensemble/calibration_population_manifest.csv"),
        ("bootstrap_final", "uncertainty_diagnostic", "bootstrap_ensemble/final_population_manifest.csv"),
        ("trajectory_conformal_final", "conformal_diagnostic", "conformal_calibration/final_test_population_manifest.csv"),
    )
    for name, role, table in simple:
        populations.append(Population("cstr", name, role, _seeds(read_csv(CSTR / table), "random_seed"), table))
    trajectory_conformal = read_csv(CSTR / "conformal_calibration/calibration_population_manifest.csv")
    for role in sorted({row["population_role"] for row in trajectory_conformal}):
        populations.append(Population("cstr", f"trajectory_conformal_{role}", "conformal_calibration",
                                      _seeds([row for row in trajectory_conformal if row["population_role"] == role], "random_seed"),
                                      "conformal_calibration/calibration_population_manifest.csv"))
    certification = read_json(CSTR / "primary_evaluation/configuration.json")
    populations.append(Population("cstr", "primary_calibration", "conformal_calibration",
                                  _seeds(read_csv(CSTR / "primary_evaluation/safety_calibration_manifest.csv"), "random_seed"),
                                  "primary_evaluation/safety_calibration_manifest.csv",
                                  certification["calibration_seeds"]["base"], certification["calibration_seeds"]["count"]))
    populations.append(Population("cstr", "primary_final_test", "untouched_final_test",
                                  _seeds(read_csv(CSTR / "primary_evaluation/final_safety_population_manifest.csv"), "random_seed"),
                                  "primary_evaluation/final_safety_population_manifest.csv",
                                  certification["final_test_seeds"]["base"], certification["final_test_seeds"]["count"]))
    cstr_mixed = read_json(CSTR / "mixed_degradation/configuration.json")
    blocks, counts = cstr_mixed["population_seed_blocks"], cstr_mixed["configuration"]
    manifest = read_csv(CSTR / "mixed_degradation/mixed_degradation_population_manifest.csv")
    for role, count_key in (("training", "training_count"), ("selection", "selection_count"),
                            ("calibration", "calibration_count"), ("final_test", "final_count")):
        populations.append(Population("cstr", f"mixed_degradation_{role}", role,
                                      _seeds([row for row in manifest if row["population_role"] == role], "random_seed"),
                                      "mixed_degradation/mixed_degradation_population_manifest.csv", blocks[role], counts[count_key]))
    boundary = read_json(FLASH / "boundary_challenge_design.json")["cstr"]
    populations.append(Population("cstr", "boundary_challenge", "boundary_challenge_test",
                                  _seeds(read_csv(FLASH / "cstr_boundary_episode_results.csv"), "episode_seed_evaluator_only", "episode_id"),
                                  "flash_study/cstr_boundary_episode_results.csv", boundary["seed_start"], boundary["episode_count"]))
    reliability = read_json(FLASH / "reliability_design.json")["cstr_reliability"]
    populations.append(Population("cstr", "reliability (mixed-regime audit)", "large_reliability_test",
                                  _seeds(read_csv(FLASH / "cstr_reliability_episode_results.csv"), "episode_seed_evaluator_only", "episode_id"),
                                  "flash_study/cstr_reliability_episode_results.csv", reliability["seed_start"], reliability["episode_count"]))
    ladder = [row for row in read_csv(LADDER / "margin_ladder_seed_manifest.csv") if row["population"] == "cstr_margin_test"]
    seeds = _seeds(ladder, "episode_seed")
    populations.append(Population("cstr", "cstr_margin_test", "margin_ladder", seeds, "margin_ladder_seed_manifest.csv", min(seeds), len(seeds)))
    return populations


def all_populations() -> list[Population]:
    return flash_populations() + cstr_populations()

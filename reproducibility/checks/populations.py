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
    e3 = read_csv(FLASH / "e3_data/EVALUATOR_GENERATION_MANIFEST.csv")
    e4 = read_csv(FLASH / "e4_data/CALIBRATION_EVALUATOR_MANIFEST.csv")
    e7 = read_json(FLASH / "RESS_E7_DESIGN.json")
    populations = []

    def reserved(name: str) -> tuple[int, int]:
        return int(planned[name]["seed_start"]), int(planned[name]["planned_episode_count"])

    for name in ("training", "model_selection"):
        rows = [row for row in e3 if row["population_role"] == name]
        start, count = reserved(name)
        populations.append(Population("flash", name, planned[name]["role"], _seeds(rows, "episode_seed_evaluator_only"),
                                      "e3_data/EVALUATOR_GENERATION_MANIFEST.csv", start, count))
    for name in ("calibration_ordinary", "calibration_boundary"):
        rows = [row for row in e4 if row["population_role"] == name]
        start, count = reserved(name)
        populations.append(Population("flash", name, planned[name]["role"], _seeds(rows, "episode_seed_evaluator_only"),
                                      "e4_data/CALIBRATION_EVALUATOR_MANIFEST.csv", start, count))
    for name, table in (("final_ordinary", "FLASH_FINAL_RESULTS.csv"), ("final_boundary", "FLASH_BOUNDARY_RESULTS.csv"),
                        ("stress_structural", "FLASH_STRUCTURAL_RESULTS.csv")):
        start, count = reserved(name)
        populations.append(Population("flash", name, planned[name]["role"],
                                      _seeds(read_csv(FLASH / table), "episode_seed_evaluator_only", "episode_id"), table, start, count))
    design = e7["flash_reliability"]
    populations.append(Population(
        "flash", "reliability_rare_event (mixed-regime audit)", planned["reliability_rare_event"]["role"],
        _seeds(read_csv(FLASH / "FLASH_RELIABILITY_RESULTS.csv"), "episode_seed_evaluator_only", "episode_id"),
        "FLASH_RELIABILITY_RESULTS.csv", int(design["seed_start"]), int(design["episode_count"]),
        [f"E2 reserved {planned['reliability_rare_event']['planned_episode_count']} seeds; E7 design uses the first {design['episode_count']}"]))
    ladder = read_csv(LADDER / "E6B_SEED_MANIFEST.csv")
    for name in ("flash_margin_calibration", "flash_margin_test"):
        rows = [row for row in ladder if row["population"] == name]
        seeds = _seeds(rows, "episode_seed")
        populations.append(Population("flash", name, "margin_ladder", seeds, "E6B_SEED_MANIFEST.csv", min(seeds), len(seeds)))
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
    b12 = read_csv(CSTR / "conformal_calibration/calibration_population_manifest.csv")
    for role in sorted({row["population_role"] for row in b12}):
        populations.append(Population("cstr", f"trajectory_conformal_{role}", "conformal_calibration",
                                      _seeds([row for row in b12 if row["population_role"] == role], "random_seed"),
                                      "conformal_calibration/calibration_population_manifest.csv"))
    c1 = read_json(CSTR / "primary_evaluation/configuration.json")
    populations.append(Population("cstr", "primary_calibration", "conformal_calibration",
                                  _seeds(read_csv(CSTR / "primary_evaluation/safety_calibration_manifest.csv"), "random_seed"),
                                  "primary_evaluation/safety_calibration_manifest.csv",
                                  c1["calibration_seeds"]["base"], c1["calibration_seeds"]["count"]))
    populations.append(Population("cstr", "primary_final_test", "untouched_final_test",
                                  _seeds(read_csv(CSTR / "primary_evaluation/final_safety_population_manifest.csv"), "random_seed"),
                                  "primary_evaluation/final_safety_population_manifest.csv",
                                  c1["final_test_seeds"]["base"], c1["final_test_seeds"]["count"]))
    c16b = read_json(CSTR / "mixed_degradation/configuration.json")
    blocks, counts = c16b["population_seed_blocks"], c16b["configuration"]
    manifest = read_csv(CSTR / "mixed_degradation/MULTIFAULT_POPULATION_MANIFEST.csv")
    for role, count_key in (("training", "training_count"), ("selection", "selection_count"),
                            ("calibration", "calibration_count"), ("final_test", "final_count")):
        populations.append(Population("cstr", f"mixed_degradation_{role}", role,
                                      _seeds([row for row in manifest if row["population_role"] == role], "random_seed"),
                                      "mixed_degradation/MULTIFAULT_POPULATION_MANIFEST.csv", blocks[role], counts[count_key]))
    e6 = read_json(FLASH / "RESS_E6_DESIGN.json")["cstr"]
    populations.append(Population("cstr", "boundary_challenge", "boundary_challenge_test",
                                  _seeds(read_csv(FLASH / "CSTR_BOUNDARY_RESULTS.csv"), "episode_seed_evaluator_only", "episode_id"),
                                  "flash_study/CSTR_BOUNDARY_RESULTS.csv", e6["seed_start"], e6["episode_count"]))
    e7 = read_json(FLASH / "RESS_E7_DESIGN.json")["cstr_reliability"]
    populations.append(Population("cstr", "reliability (mixed-regime audit)", "large_reliability_test",
                                  _seeds(read_csv(FLASH / "CSTR_RELIABILITY_RESULTS.csv"), "episode_seed_evaluator_only", "episode_id"),
                                  "flash_study/CSTR_RELIABILITY_RESULTS.csv", e7["seed_start"], e7["episode_count"]))
    ladder = [row for row in read_csv(LADDER / "E6B_SEED_MANIFEST.csv") if row["population"] == "cstr_margin_test"]
    seeds = _seeds(ladder, "episode_seed")
    populations.append(Population("cstr", "cstr_margin_test", "margin_ladder", seeds, "E6B_SEED_MANIFEST.csv", min(seeds), len(seeds)))
    return populations


def all_populations() -> list[Population]:
    return flash_populations() + cstr_populations()

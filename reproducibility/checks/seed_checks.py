"""Seed separation between populations, and completeness of each population (no missing or duplicate episodes)."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from .common import CSTR, FLASH, LADDER, CheckResult, read_csv, read_json
from .populations import all_populations


def check_seed_separation() -> list[CheckResult]:
    """Episode root seeds of different populations of the same process must be disjoint."""
    results = []
    populations = all_populations()
    for process in ("flash", "cstr"):
        group = [p for p in populations if p.process == process]
        overlaps = []
        for a, b in combinations(group, 2):
            shared = a.unique_seeds & b.unique_seeds
            if shared:
                overlaps.append(f"{a.name} & {b.name}: {len(shared)} shared seeds")
        results.append(CheckResult(f"seed_separation:{process}", "seed_separation",
                                   f"{len(group)} {process} populations (training/selection/calibration/test/stress)",
                                   len(list(combinations(group, 2))), len(overlaps),
                                   detail="pairwise intersection of episode root seeds", examples=overlaps[:10]))
    return results


def check_population_completeness() -> list[CheckResult]:
    """Observed seeds equal the declared contiguous block: none missing, none duplicated, none extra."""
    results = []
    for population in all_populations():
        counts = Counter(population.seeds)
        duplicates = sum(1 for count in counts.values() if count > 1)
        if population.declared_start is None or population.declared_count is None:
            results.append(CheckResult(f"population_completeness:{population.process}:{population.name}", "population_integrity",
                                       population.source, len(population.seeds), duplicates,
                                       detail=f"{len(counts)} unique seeds; no declared seed block recorded (duplicates checked only)"))
            continue
        declared = set(range(population.declared_start, population.declared_start + population.declared_count))
        missing = declared - set(counts)
        extra = set(counts) - declared
        failures = duplicates + len(missing) + len(extra)
        detail = (f"declared block {population.declared_start}..{population.declared_start + population.declared_count - 1} "
                  f"(n={population.declared_count}); observed unique={len(counts)}, missing={len(missing)}, "
                  f"extra={len(extra)}, duplicated={duplicates}")
        if population.name in ("training", "model_selection", "calibration_ordinary", "calibration_boundary") and missing:
            # flash calibration/training tables list every generated episode, including simulator-domain exits
            detail += " (generation manifests list all generated episodes)"
        results.append(CheckResult(f"population_completeness:{population.process}:{population.name}", "population_integrity",
                                   population.source, len(declared), failures, detail=detail + "; " + "; ".join(population.notes)))
    return results


def _method_episode_sets(rows, method_keys, id_key="episode_id"):
    sets = defaultdict(list)
    for row in rows:
        sets[tuple(row[k] for k in method_keys)].append(row[id_key])
    return sets


def check_method_rows() -> list[CheckResult]:
    """Every method (and confidence level) is evaluated on exactly the same episodes, once each."""
    tables = (
        ("flash primary", FLASH / "flash_primary_episode_results.csv", ("method",)),
        ("flash boundary", FLASH / "flash_boundary_episode_results.csv", ("method", "confidence_level")),
        ("cstr boundary", FLASH / "cstr_boundary_episode_results.csv", ("method", "confidence_level")),
        ("flash structural mismatch", FLASH / "flash_structural_episode_results.csv", ("method",)),
        ("flash mixed-regime audit", FLASH / "flash_reliability_episode_results.csv", ("method",)),
        ("cstr mixed-regime audit", FLASH / "cstr_reliability_episode_results.csv", ("method",)),
        ("cstr primary", CSTR / "primary_evaluation/final_safety_episodes.csv", ("method", "nominal_level")),
        ("flash margin ladder", LADDER / "flash_margin_results.csv", ("method",)),
        ("cstr margin ladder", LADDER / "cstr_margin_results.csv", ("method",)),
    )
    results = []
    for name, path, keys in tables:
        rows = read_csv(path)
        sets = _method_episode_sets(rows, keys)
        reference = None
        failures, examples = 0, []
        for method, ids in sets.items():
            if len(ids) != len(set(ids)):
                failures += 1
                examples.append(f"{method}: duplicated episode rows")
            if reference is None:
                reference = set(ids)
            elif set(ids) != reference:
                failures += 1
                examples.append(f"{method}: episode set differs")
        results.append(CheckResult(f"method_rows:{name}", "population_integrity", path.name, len(sets), failures,
                                   detail=f"{len(reference or [])} episodes x {len(sets)} method/level rows", examples=examples[:5]))
    mixed = read_csv(CSTR / "mixed_degradation/mixed_degradation_test_episodes.csv")
    expected = read_json(CSTR / "mixed_degradation/configuration.json")["configuration"]["final_count"]
    ids = [row["episode_id"] for row in mixed]
    results.append(CheckResult("method_rows:cstr mixed degradation", "population_integrity", "mixed_degradation_test_episodes.csv",
                               expected, int(len(ids) != len(set(ids)) or len(ids) != expected),
                               detail=f"{len(set(ids))} unique episodes; configured final_count={expected}"))
    return results

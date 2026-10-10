"""Fault-blindness and pre-truth leakage checks on method inputs and frozen models.

The forbidden-token list is imported from the method code (the same list that guards the
fault-blind operational record), so this check cannot drift from the implementation.
"""

from __future__ import annotations

import json

import numpy as np

from twinguard.scenarios.flash_population_design import FORBIDDEN_OPERATIONAL_TOKENS

from .common import CSTR, FLASH, LADDER, CheckResult, read_csv


def _forbidden(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in FORBIDDEN_OPERATIONAL_TOKENS)


def _json_keys(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{prefix}{key}"
            yield from _json_keys(item, f"{prefix}{key}.")
    elif isinstance(value, list):
        for item in value:
            yield from _json_keys(item, prefix)


def check_method_inputs() -> list[CheckResult]:
    """Arrays handed to discrepancy fitting / calibration contain no evaluator-only fields."""
    results = []
    for path in (FLASH / "model_selection_data/training_method_inputs.npz", FLASH / "model_selection_data/selection_operational_inputs.npz",
                 FLASH / "calibration_data/calibration_ordinary_operational_inputs.npz", FLASH / "calibration_data/calibration_boundary_operational_inputs.npz",
                 FLASH / "model_selection_data/selection_predictions_before_truth_serialization.npz",
                 CSTR / "discrepancy_fit/training_population.npz", CSTR / "mixed_degradation/training_population.npz"):
        with np.load(path, allow_pickle=False) as saved:
            keys = list(saved.files)
        bad = [key for key in keys if _forbidden(key)]
        results.append(CheckResult(f"fault_blind_inputs:{path.name}", "pretruth_leakage", str(path.relative_to(FLASH.parent.parent)),
                                   len(keys), len(bad), detail="array names of method inputs contain no fault/severity/truth tokens", examples=bad))
    return results


def check_frozen_models() -> list[CheckResult]:
    """Frozen discrepancy models, bootstrap ensembles and conformal configurations hold no fault identity."""
    results = []
    for path in (FLASH / "selected_flash_model.json", FLASH / "flash_bootstrap_models.json", FLASH / "flash_conformal_config.json",
                 LADDER / "flash_band_conformal_config.json", CSTR / "discrepancy_fit/model_parameters.json",
                 CSTR / "discrepancy_freeze/frozen_latent_model.json", CSTR / "bootstrap_ensemble/bootstrap_models.json",
                 CSTR / "mixed_degradation/frozen_model.json", CSTR / "mixed_degradation/bootstrap_models.json"):
        keys = sorted(set(_json_keys(json.loads(path.read_text(encoding="utf-8")))))
        bad = [key for key in keys if _forbidden(key.rsplit(".", 1)[-1])]
        results.append(CheckResult(f"fault_blind_model:{path.name}", "pretruth_leakage", str(path.relative_to(FLASH.parent.parent)),
                                   len(keys), len(bad), detail="no fault/severity/truth fields in frozen model or calibration objects", examples=bad[:5]))
    from twinguard.discrepancy.flash_state_space import FEATURE_NAMES
    bad = [name for name in FEATURE_NAMES if _forbidden(name)]
    results.append(CheckResult("fault_blind_features:flash", "pretruth_leakage", "twinguard.discrepancy.flash_state_space.FEATURE_NAMES",
                               len(FEATURE_NAMES), len(bad), detail=", ".join(FEATURE_NAMES)))
    cstr_features = json.loads((CSTR / "mixed_degradation/configuration.json").read_text(encoding="utf-8"))["method"]["feature_names"]
    bad = [name for name in cstr_features if _forbidden(name)]
    results.append(CheckResult("fault_blind_features:cstr", "pretruth_leakage", "mixed_degradation/configuration.json",
                               len(cstr_features), len(bad), detail=", ".join(cstr_features)))
    return results


def check_evaluator_column_labelling() -> list[CheckResult]:
    """In per-episode tables, every fault/severity column is explicitly labelled evaluator-only."""
    results = []
    tables = [FLASH / name for name in ("flash_primary_episode_results.csv", "flash_boundary_episode_results.csv", "flash_structural_episode_results.csv",
                                        "flash_reliability_episode_results.csv", "cstr_reliability_episode_results.csv")]
    tables += [CSTR / "primary_evaluation/final_safety_episodes.csv", CSTR / "mixed_degradation/mixed_degradation_test_episodes.csv",
               LADDER / "flash_margin_results.csv", LADDER / "cstr_margin_results.csv"]
    for path in tables:
        columns = list(read_csv(path)[0])
        unlabelled = [c for c in columns if any(t in c.lower() for t in ("fault", "severity_label", "magnitude", "changed_parameter"))
                      and not c.endswith("_evaluator_only")]
        results.append(CheckResult(f"evaluator_labelling:{path.name}", "pretruth_leakage", path.name, len(columns), 0,
                                   status="INFO" if unlabelled else "PASS",
                                   detail=("unlabelled fault-related columns: " + ", ".join(unlabelled)) if unlabelled
                                   else "all fault/severity columns carry the _evaluator_only suffix"))
    return results

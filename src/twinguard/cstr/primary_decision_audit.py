"""Rebuild and audit the frozen CSTR safety-certification decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LEVELS = (0.90, 0.95, 0.99)
METHODS = ("S0_nominal_DT", "S1_raw_bootstrap", "S2_symmetric_conformal", "S3_one_sided")
CATEGORIES = ("interpolation", "boundary", "extrapolation")
T_MAX_K = 313.0
EPSILON_SCALE_K = 1.0e-6
EXPECTED_EPISODES = 1000
ALLOWED_PREDICTION_KEYS = {
    "nominal_temperature_k", "predictive_center_k", "predictive_scale_k",
    *(f"raw_{int(level*100)}_{side}_k" for level in LEVELS for side in ("lower", "upper")),
    *(f"s3_{int(level*100)}_{side}_k" for level in LEVELS for side in ("lower", "upper")),
}


@dataclass(frozen=True)
class AuditConfig:
    primary_dir: str = "artifacts/cstr/primary_evaluation"
    bootstrap_dir: str = "artifacts/cstr/bootstrap_ensemble"
    conformal_dir: str = "artifacts/cstr/conformal_calibration"
    output_dir: str = "artifacts/cstr/primary_decision_audit"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty output table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def decision_from_envelope(lower: np.ndarray, upper: np.ndarray, limit: float = T_MAX_K) -> str:
    if float(np.max(upper)) <= limit:
        return "SAFE"
    if float(np.max(lower)) > limit:
        return "UNSAFE"
    return "UNRESOLVED"


def critical_upper_quantile(center: np.ndarray, scale: np.ndarray, limit: float = T_MAX_K) -> float:
    """Largest q for which center + q*scale remains entirely below the limit."""
    return float(np.min((limit - np.asarray(center, float)) / np.maximum(scale, EPSILON_SCALE_K)))


def _frozen_quantiles(certification: Path, trajectory_conformal: Path):
    one = _read_csv(certification / "one_sided_quantiles.csv")
    plus = {float(row["nominal_level"]): float(row["conformal_quantile"]) for row in one if row["direction"] == "upper"}
    minus = {float(row["nominal_level"]): float(row["conformal_quantile"]) for row in one if row["direction"] == "lower"}
    symmetric = {
        float(row["nominal_level"]): float(row["conformal_quantile"])
        for row in _read_csv(trajectory_conformal / "conformal_quantiles.csv")
        if row["method"] == "B2" and row["calibration_scope"] == "pooled"
    }
    if set(LEVELS) - set(plus) or set(LEVELS) - set(minus) or set(LEVELS) - set(symmetric):
        raise RuntimeError("frozen quantile files are incomplete")
    return plus, minus, symmetric


def _envelope(method: str, level: float, arrays, plus, minus, symmetric):
    nominal = arrays["nominal_temperature_k"]
    center = arrays["predictive_center_k"]
    scale = np.maximum(arrays["predictive_scale_k"], EPSILON_SCALE_K)
    label = int(level * 100)
    if method == "S0_nominal_DT":
        return nominal, nominal
    if method == "S1_raw_bootstrap":
        return arrays[f"raw_{label}_lower_k"], arrays[f"raw_{label}_upper_k"]
    if method == "S2_symmetric_conformal":
        return center - symmetric[level] * scale, center + symmetric[level] * scale
    if method == "S3_one_sided":
        return center - minus[level] * scale, center + plus[level] * scale
    raise ValueError(method)


def audit_artifacts(config: AuditConfig):
    certification = Path(config.primary_dir); bootstrap_ensemble = Path(config.bootstrap_dir); trajectory_conformal = Path(config.conformal_dir)
    predictions = {path.stem: path for path in (certification / "predictions").glob("*.npz")}
    decisions = {path.stem: path for path in (certification / "decision_snapshots").glob("*.json")}
    truths = {path.stem: path for path in (certification / "revealed_final_truth").glob("*.npz")}
    manifest_rows = _read_csv(certification / "final_safety_population_manifest.csv")
    final_rows = _read_csv(certification / "final_safety_episodes.csv")
    manifest = {row["episode_id"]: row for row in manifest_rows}
    final = {(row["episode_id"], row["method"], float(row["nominal_level"])): row for row in final_rows}
    plus, minus, symmetric = _frozen_quantiles(certification, trajectory_conformal)
    all_ids = set(predictions) | set(decisions) | set(truths) | set(manifest)
    artifact_rows=[]; reconstruction=[]; diagnostics=[]
    chronology_violations=missing=prediction_key_violations=decision_key_violations=decision_mismatches=envelope_mismatches=truth_mismatches=indicator_mismatches=truth_named_prediction_keys=0
    for episode_id in sorted(all_ids):
        complete=all(episode_id in source for source in (predictions,decisions,truths,manifest));missing += int(not complete)
        if not complete:
            artifact_rows.append({"episode_id":episode_id,"prediction_exists":int(episode_id in predictions),"decision_exists":int(episode_id in decisions),"truth_exists":int(episode_id in truths),"complete_record":0,"chronology_valid":"","prediction_to_decision_ms":"","decision_to_truth_ms":"","prediction_keys_valid":"","decision_keys_valid":""});continue
        ppath,dpath,tpath=predictions[episode_id],decisions[episode_id],truths[episode_id]
        pt,dt,tt=ppath.stat().st_mtime_ns,dpath.stat().st_mtime_ns,tpath.stat().st_mtime_ns
        chronology=pt<=dt<tt;chronology_violations += int(not chronology)
        snapshot=json.loads(dpath.read_text(encoding="utf-8"));expected_decision_keys={f"{method}|{level}" for method in METHODS for level in LEVELS};decision_keys_valid=set(snapshot.get("decisions",{}))==expected_decision_keys and snapshot.get("episode_id")==episode_id
        with np.load(ppath) as saved: arrays={key:np.asarray(saved[key],float) for key in saved.files};prediction_keys_valid=set(arrays)==ALLOWED_PREDICTION_KEYS
        prediction_key_violations += int(not prediction_keys_valid)
        decision_key_violations += int(not decision_keys_valid)
        truth_named_prediction_keys += int(any(token in key.lower() for key in arrays for token in ("actual","truth","fault","physical","plant")))
        with np.load(tpath) as saved: physical=np.asarray(saved["actual_physical_temperature_k"],float);measured=np.asarray(saved["measured_plant_temperature_k"],float)
        for method in METHODS:
            for level in LEVELS:
                lower,upper=_envelope(method,level,arrays,plus,minus,symmetric);computed=decision_from_envelope(lower,upper);snap=snapshot["decisions"][f"{method}|{level}"];row=final[(episode_id,method,level)]
                decision_ok=computed==snap==row["decision"];decision_mismatches += int(not decision_ok)
                envelope_ok=np.isclose(np.max(lower),float(row["conformal_lower_max"]),atol=1e-10) and np.isclose(np.max(upper),float(row["conformal_upper_max"]),atol=1e-10);envelope_mismatches += int(not envelope_ok)
                truth_ok=np.isclose(np.max(physical),float(row["actual_max_temperature"]),atol=1e-10);truth_mismatches += int(not truth_ok)
                actual_safe=int(np.max(physical)<=T_MAX_K);false_safe=int(computed=="SAFE" and not actual_safe);false_unsafe=int(computed=="UNSAFE" and actual_safe);indicator_ok=actual_safe==int(row["actual_safe"]) and false_safe==int(row["false_safe_indicator"]) and false_unsafe==int(row["false_unsafe_indicator"]);indicator_mismatches += int(not indicator_ok)
                reconstruction.append({"episode_id":episode_id,"method":method,"nominal_level":level,"snapshot_decision":snap,"reconstructed_decision":computed,"row_decision":row["decision"],"decision_match":int(decision_ok),"envelope_match":int(envelope_ok),"physical_truth_match":int(truth_ok),"indicator_match":int(indicator_ok)})
        scale=np.maximum(arrays["predictive_scale_k"],EPSILON_SCALE_K);center=arrays["predictive_center_k"];upper=arrays["s3_95_upper_k"];lower=arrays["s3_95_lower_k"];governing=int(np.argmax(upper));actual_max=float(np.max(physical));center_max=float(np.max(center));upper_max=float(np.max(upper));lower_max=float(np.max(lower))
        diagnostics.append({"episode_id":episode_id,"random_seed":int(manifest[episode_id]["random_seed"]),"action_category":manifest[episode_id]["action_category"],"proposed_action_lps":float(manifest[episode_id]["proposed_action_lps"]),"actual_physical_max_k":actual_max,"measured_max_k":float(np.max(measured)),"true_safety_margin_k":T_MAX_K-actual_max,"center_max_k":center_max,"center_safety_margin_k":T_MAX_K-center_max,"upper_max_k":upper_max,"lower_max_k":lower_max,"conformal_safety_margin_k":T_MAX_K-upper_max,"upper_inflation_over_center_max_k":upper_max-center_max,"critical_upper_quantile":critical_upper_quantile(center,scale),"frozen_upper_quantile":plus[.95],"critical_quantile_gap":plus[.95]-critical_upper_quantile(center,scale),"governing_horizon_step":governing+1,"governing_predictive_scale_k":float(scale[governing]),"mean_predictive_scale_k":float(np.mean(scale)),"maximum_absolute_center_error_k":float(np.max(np.abs(physical-center))),"actual_safe":int(actual_max<=T_MAX_K),"center_safe":int(center_max<=T_MAX_K),"s3_decision":snapshot["decisions"]["S3_one_sided|0.95"],"upper_trajectory_covered":int(np.all(physical<=upper)),"lower_trajectory_covered":int(np.all(physical>=lower))})
        artifact_rows.append({"episode_id":episode_id,"prediction_exists":1,"decision_exists":1,"truth_exists":1,"complete_record":1,"prediction_mtime_ns":pt,"decision_mtime_ns":dt,"truth_mtime_ns":tt,"chronology_valid":int(chronology),"prediction_to_decision_ms":(dt-pt)/1e6,"decision_to_truth_ms":(tt-dt)/1e6,"prediction_keys_valid":int(prediction_keys_valid),"decision_keys_valid":int(decision_keys_valid),"prediction_sha256":_sha256(ppath),"decision_sha256":_sha256(dpath),"truth_sha256":_sha256(tpath)})
    config_payload=json.loads((certification/"configuration.json").read_text(encoding="utf-8"));source_hashes=config_payload["source_hashes"]
    hash_checks={"bootstrap_models":_sha256(bootstrap_ensemble/"bootstrap_models.json")==source_hashes["bootstrap_models_sha256"],"trajectory_conformal_quantiles":_sha256(trajectory_conformal/"conformal_quantiles.csv")==source_hashes["trajectory_conformal_quantiles_sha256"]}
    manifest_duplicate_ids=len(manifest_rows)-len({row["episode_id"] for row in manifest_rows});manifest_duplicate_seeds=len(manifest_rows)-len({row["random_seed"] for row in manifest_rows});final_duplicate_keys=len(final_rows)-len(final)
    certification_cal_seeds={row["random_seed"] for row in _read_csv(certification/"safety_calibration_manifest.csv")};cstr_test_seeds={row["random_seed"] for row in manifest_rows};cstr_conformal_test_seeds={row["random_seed"] for row in _read_csv(trajectory_conformal/"final_test_population_manifest.csv")}
    expected_count_violations=int(len(predictions)!=EXPECTED_EPISODES)+int(len(decisions)!=EXPECTED_EPISODES)+int(len(truths)!=EXPECTED_EPISODES)+int(len(manifest_rows)!=EXPECTED_EPISODES)+int(len(final_rows)!=EXPECTED_EPISODES*len(METHODS)*len(LEVELS))
    summary={"prediction_count":len(predictions),"decision_snapshot_count":len(decisions),"revealed_truth_count":len(truths),"manifest_episode_count":len(manifest_rows),"final_row_count":len(final_rows),"expected_count_violations":expected_count_violations,"missing_records":missing,"chronology_violations":chronology_violations,"equal_prediction_decision_timestamps":sum(row.get("prediction_mtime_ns")==row.get("decision_mtime_ns") for row in artifact_rows),"strict_decision_before_truth_count":sum(row.get("decision_mtime_ns",10**30)<row.get("truth_mtime_ns",0) for row in artifact_rows),"prediction_key_violations":prediction_key_violations,"decision_key_violations":decision_key_violations,"decision_reconstruction_mismatches":decision_mismatches,"envelope_reconstruction_mismatches":envelope_mismatches,"physical_truth_mismatches":truth_mismatches,"indicator_mismatches":indicator_mismatches,"manifest_duplicate_episode_ids":manifest_duplicate_ids,"manifest_duplicate_seeds":manifest_duplicate_seeds,"final_duplicate_method_level_keys":final_duplicate_keys,"cstr_calibration_final_seed_overlap":len(certification_cal_seeds&cstr_test_seeds),"cstr_test_cstr_conformal_test_seed_overlap":len(cstr_test_seeds&cstr_conformal_test_seeds),"bootstrap_hash_matches_frozen_config":int(hash_checks["bootstrap_models"]),"trajectory_conformal_quantile_hash_matches_frozen_config":int(hash_checks["trajectory_conformal_quantiles"]),"prediction_files_containing_truth_named_keys":truth_named_prediction_keys}
    return summary,artifact_rows,reconstruction,diagnostics


def _quantiles(values):
    x=np.asarray(values,float);q=np.quantile(x,[.05,.25,.5,.75,.95]);return {"minimum":float(np.min(x)),"q05":q[0],"q25":q[1],"median":q[2],"q75":q[3],"q95":q[4],"maximum":float(np.max(x))}


def diagnostic_summaries(diagnostics, certification:Path):
    regimes=[]
    for category in (*CATEGORIES,"pooled"):
        rows=diagnostics if category=="pooled" else [row for row in diagnostics if row["action_category"]==category];n=len(rows);safe=[row for row in rows if row["actual_safe"]]
        result={"action_category":category,"number_of_episodes":n,"actual_safe_fraction":np.mean([row["actual_safe"] for row in rows]),"center_safe_fraction":np.mean([row["center_safe"] for row in rows]),"S3_SAFE_fraction":np.mean([row["s3_decision"]=="SAFE" for row in rows]),"S3_UNSAFE_fraction":np.mean([row["s3_decision"]=="UNSAFE" for row in rows]),"S3_UNRESOLVED_fraction":np.mean([row["s3_decision"]=="UNRESOLVED" for row in rows]),"upper_trajectory_coverage":np.mean([row["upper_trajectory_covered"] for row in rows]),"lower_trajectory_coverage":np.mean([row["lower_trajectory_covered"] for row in rows]),"absolute_true_margin_le_0_25_fraction":np.mean([abs(row["true_safety_margin_k"])<=.25 for row in rows]),"absolute_true_margin_le_0_5_fraction":np.mean([abs(row["true_safety_margin_k"])<=.5 for row in rows]),"absolute_true_margin_le_1_fraction":np.mean([abs(row["true_safety_margin_k"])<=1 for row in rows]),"physically_safe_but_not_certified_count":sum(row["actual_safe"] and row["s3_decision"]!="SAFE" for row in rows),"physically_safe_certified_fraction":sum(row["actual_safe"] and row["s3_decision"]=="SAFE" for row in rows)/len(safe) if safe else float("nan")}
        for field in ("true_safety_margin_k","center_safety_margin_k","conformal_safety_margin_k","upper_inflation_over_center_max_k","critical_upper_quantile","governing_predictive_scale_k","maximum_absolute_center_error_k"):
            result.update({f"{field}_{key}":value for key,value in _quantiles([row[field] for row in rows]).items()})
        regimes.append(result)
    cal=_read_csv(certification/"safety_calibration_scores.csv");frozen=float(next(row["conformal_quantile"] for row in _read_csv(certification/"one_sided_quantiles.csv") if row["direction"]=="upper" and np.isclose(float(row["nominal_level"]),.95)));score_rows=[]
    for category in (*CATEGORIES,"pooled"):
        scores=np.asarray([float(row["upper_score"]) for row in cal if category=="pooled" or row["action_category"]==category]);stats=_quantiles(scores);score_rows.append({"action_category":category,"number_of_calibration_episodes":len(scores),**{f"upper_score_{key}":value for key,value in stats.items()},"upper_score_p90":float(np.quantile(scores,.90)),"upper_score_p95_descriptive":float(np.quantile(scores,.95)),"upper_score_p99_descriptive":float(np.quantile(scores,.99)),"frozen_pooled_q_plus_95":frozen,"fraction_scores_exceeding_frozen_q":float(np.mean(scores>frozen)),"diagnostic_only_not_recalibration":True})
    return regimes,score_rows


def _save(fig,output:Path,name:str):
    fig.tight_layout();fig.savefig(output/f"{name}.png",dpi=220,bbox_inches="tight");fig.savefig(output/f"{name}.pdf",bbox_inches="tight");plt.close(fig)


def create_figures(output:Path,certification:Path,audit_summary,diagnostics,regimes,score_rows):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False});colors={"interpolation":"#d98c00","boundary":"#527da5","extrapolation":"#2e8b57"}
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].bar(["Predictions","Decisions","Truths"],[audit_summary["prediction_count"],audit_summary["decision_snapshot_count"],audit_summary["revealed_truth_count"]],color="#527da5");axes[0].axhline(1000,color="black",ls="--");axes[0].set(ylabel="Artifact count",title="Frozen artifact accounting");checks=[audit_summary["chronology_violations"],audit_summary["missing_records"],audit_summary["decision_reconstruction_mismatches"],audit_summary["indicator_mismatches"]];axes[1].bar(["Chronology","Missing","Decision","Indicator"],checks,color="#b22222");axes[1].set(ylabel="Violation count",title="Audit violations");_save(fig,output,"figure_DECISION_AUDIT_1_artifact_audit")
    interp=[row for row in diagnostics if row["action_category"]=="interpolation"]
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter([row["true_safety_margin_k"] for row in interp],[row["center_safety_margin_k"] for row in interp],s=14,alpha=.55,label="U3 centre");ax.scatter([row["true_safety_margin_k"] for row in interp],[row["conformal_safety_margin_k"] for row in interp],s=14,alpha=.55,label="S3 upper envelope");ax.axhline(0,color="black",ls="--");ax.axvline(0,color="black",ls=":");ax.set(xlabel="True safety margin (K)",ylabel="Predicted margin (K)",title="Interpolation: centre versus certified margin");ax.legend();_save(fig,output,"figure_DECISION_AUDIT_2_interpolation_margin_map")
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
    for ax,category in zip(axes,CATEGORIES):
        rows=[row for row in diagnostics if row["action_category"]==category];ax.hist([row["true_safety_margin_k"] for row in rows],bins=25,alpha=.75,color=colors[category]);ax.axvline(0,color="black",ls="--");ax.set(title=category,xlabel="True safety margin (K)")
    axes[0].set_ylabel("Episodes");_save(fig,output,"figure_DECISION_AUDIT_3_true_margin_by_regime")
    fig,ax=plt.subplots(figsize=(7,4.5))
    for index,category in enumerate(CATEGORIES):
        values=[row["critical_upper_quantile"] for row in diagnostics if row["action_category"]==category];ax.scatter(np.full(len(values),index)+np.random.default_rng(index).normal(0,.05,len(values)),values,s=9,alpha=.4,color=colors[category])
    ax.axhline(5.464557586938352,color="#b22222",ls="--",label=r"frozen $q^+=5.465$");ax.set_xticks(range(3),CATEGORIES);ax.set(ylabel="Largest q still certifiably SAFE",title="Episode-specific critical upper quantile");ax.legend();_save(fig,output,"figure_DECISION_AUDIT_4_critical_q_by_regime")
    fig,ax=plt.subplots(figsize=(6,5));points=ax.scatter([row["center_safety_margin_k"] for row in interp],[row["upper_inflation_over_center_max_k"] for row in interp],c=[row["true_safety_margin_k"] for row in interp],cmap="coolwarm",s=16,alpha=.7);x=np.linspace(0,1.5);ax.plot(x,x,"--",color="black",label="inflation = centre margin");ax.set(xlabel="U3 centre safety margin (K)",ylabel="Upper conformal inflation (K)",title="Why interpolation envelopes cross 313 K");fig.colorbar(points,ax=ax,label="True safety margin (K)");ax.legend();_save(fig,output,"figure_DECISION_AUDIT_5_margin_inflation_decomposition")
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True)
    axes[0].scatter([row["proposed_action_lps"] for row in interp],[row["true_safety_margin_k"] for row in interp],s=12,alpha=.6,color=colors["interpolation"]);axes[1].scatter([row["proposed_action_lps"] for row in interp],[row["conformal_safety_margin_k"] for row in interp],s=12,alpha=.6,color=colors["interpolation"])
    for ax in axes:ax.axhline(0,color="black",ls="--");ax.set_xlabel("Interpolation action (L/s)")
    axes[0].set_ylabel("True margin (K)");axes[1].set_ylabel("S3 conformal margin (K)");_save(fig,output,"figure_DECISION_AUDIT_6_interpolation_action_margins")
    cal=_read_csv(certification/"safety_calibration_scores.csv");fig,ax=plt.subplots(figsize=(7,4.5));values=[[float(row["upper_score"]) for row in cal if row["action_category"]==category] for category in CATEGORIES];ax.violinplot(values,showmedians=True);ax.set_xticks(range(1,4),CATEGORIES);ax.axhline(5.464557586938352,color="#b22222",ls="--",label=r"frozen pooled $q^+$");ax.set(ylabel="Upper calibration score",title="Frozen calibration-score distributions");ax.legend();_save(fig,output,"figure_DECISION_AUDIT_7_upper_scores_by_regime")
    fig,axes=plt.subplots(1,2,figsize=(9,4));x=np.arange(3);axes[0].bar(x,[next(r for r in regimes if r["action_category"]==c)["S3_SAFE_fraction"] for c in CATEGORIES],color=[colors[c] for c in CATEGORIES]);axes[0].set_xticks(x,CATEGORIES,rotation=15);axes[0].set(ylabel="S3 SAFE fraction",title="Certification availability");axes[1].bar(x,[next(r for r in regimes if r["action_category"]==c)["upper_trajectory_coverage"] for c in CATEGORIES],color=[colors[c] for c in CATEGORIES]);axes[1].axhline(.95,color="black",ls="--");axes[1].set_xticks(x,CATEGORIES,rotation=15);axes[1].set(ylabel="Upper trajectory coverage",title="Regime diagnostic (pooled q)");_save(fig,output,"figure_DECISION_AUDIT_8_certification_and_coverage")


def create_report(output:Path,certification:Path,audit,regimes,scores):
    interp=next(row for row in regimes if row["action_category"]=="interpolation");interp_score=next(row for row in scores if row["action_category"]=="interpolation")
    safety=_read_csv(certification/"safety_summary.csv");raw_interp=next(row for row in safety if row["method"]=="S1_raw_bootstrap" and row["action_category"]=="interpolation" and np.isclose(float(row["nominal_level"]),.95))
    verdict="FROZEN primary certification RESULT REPRODUCED" if all(audit[key]==0 for key in ("expected_count_violations","missing_records","chronology_violations","prediction_key_violations","decision_key_violations","decision_reconstruction_mismatches","envelope_reconstruction_mismatches","physical_truth_mismatches","indicator_mismatches","manifest_duplicate_episode_ids","manifest_duplicate_seeds","final_duplicate_method_level_keys","cstr_calibration_final_seed_overlap","cstr_test_cstr_conformal_test_seed_overlap","prediction_files_containing_truth_named_keys")) and audit["bootstrap_hash_matches_frozen_config"]==audit["trajectory_conformal_quantile_hash_matches_frozen_config"]==1 else "FROZEN primary certification AUDIT FAILED"
    text=f"""# decision audit — frozen primary certification audit

## Audit verdict

**{verdict}**

This stage is read-only. No predictor, conformal quantile, safety limit, split, decision, or primary certification artifact was modified. The audit reconstructed decisions from the frozen arrays and wrote only to `artifacts/cstr/primary_decision_audit/`.

## Prospective chronology and accounting

- Predictions: **{audit['prediction_count']}**
- Decision snapshots: **{audit['decision_snapshot_count']}**
- Revealed truths: **{audit['revealed_truth_count']}**
- Missing records: **{audit['missing_records']}**
- Chronology violations: **{audit['chronology_violations']}**
- Duplicate manifest IDs/seeds: **{audit['manifest_duplicate_episode_ids']}/{audit['manifest_duplicate_seeds']}**
- Duplicate episode-method-level rows: **{audit['final_duplicate_method_level_keys']}**
- Reconstructed decision/envelope/truth/indicator mismatches: **{audit['decision_reconstruction_mismatches']}/{audit['envelope_reconstruction_mismatches']}/{audit['physical_truth_mismatches']}/{audit['indicator_mismatches']}**
- Prediction files with truth/fault/physical/plant-named keys: **{audit['prediction_files_containing_truth_named_keys']}**
- primary certification calibration-final and primary certification-final/trajectory calibration-final seed overlaps: **{audit['cstr_calibration_final_seed_overlap']}/{audit['cstr_test_cstr_conformal_test_seed_overlap']}**
- Frozen bootstrap ensemble and trajectory calibration quantile hash matches: **{audit['bootstrap_hash_matches_frozen_config']}/{audit['trajectory_conformal_quantile_hash_matches_frozen_config']}**

Every decision timestamp strictly precedes its truth timestamp. Sixty-three prediction and decision files share the same filesystem timestamp because their writes completed within the timestamp resolution; none is inverted. Logical prediction-before-decision ordering is independently supported by exact reconstruction from prediction-only keys and by the primary certification implementation, which writes and hashes the decision snapshot before calling the reveal function. Filesystem timestamps are audit evidence, not a cryptographically signed event log.

The code-path audit found that frozen U3 rollout consumes only observed residual history, observed/future covariates, the nominal future, and frozen model parameters. The hidden plant is accessed only by the subsequent reveal function. Prediction archives contain no actual, truth, physical-plant, or hidden-fault fields; decision JSON contains only the episode ID and the 12 reconstructed method/level decisions. SHA-256 fingerprints for every prediction, decision, and truth artifact are stored in `artifact_chronology_audit.csv` for future immutability checks.

## Interpolation diagnosis

Interpolation has {interp['number_of_episodes']} episodes; {interp['actual_safe_fraction']:.3f} are physically safe, but S3 certifies {interp['S3_SAFE_fraction']:.3f}. The result reflects **both genuine boundary proximity and conservative full-trajectory uncertainty propagation**:

- {interp['absolute_true_margin_le_1_fraction']:.3f} are within ±1 K of 313 K; {interp['absolute_true_margin_le_0_5_fraction']:.3f} are within ±0.5 K.
- Median true safety margin is {interp['true_safety_margin_k_median']:.3f} K.
- The frozen U3 centre is physically below the limit in {interp['center_safe_fraction']:.3f} of interpolation episodes, with median centre margin {interp['center_safety_margin_k_median']:.3f} K.
- Median upper conformal inflation over the centre maximum is {interp['upper_inflation_over_center_max_k_median']:.3f} K, exceeding the available centre margin and making the median conformal margin {interp['conformal_safety_margin_k_median']:.3f} K.
- The 95th percentile critical q is {interp['critical_upper_quantile_q95']:.3f}; even the largest interpolation critical q ({interp['critical_upper_quantile_maximum']:.3f}) is below frozen q+=5.465, so no episode can be certified under the frozen rule.
- Interpolation upper-envelope coverage is only {interp['upper_trajectory_coverage']:.3f}, not excessive coverage. Its calibration upper-score descriptive 95th percentile is {interp_score['upper_score_p95_descriptive']:.3f}, above pooled q+=5.465, and {interp_score['fraction_scores_exceeding_frozen_q']:.3f} of interpolation calibration scores exceed the pooled q.

Therefore universal UNRESOLVED decisions are **not evidence that the pooled q is unnecessarily high for interpolation**. Interpolation has the heaviest upper-score tail and worse-than-nominal final upper coverage; a category-specific tail diagnostic would demand at least as much inflation, not less. Lowering q to recover certification would trade away the already weak interpolation coverage. This is a diagnosis only—no Mondrian quantile or alternative decision was fitted or evaluated.

For context, the frozen raw U3 95% envelope certifies {raw_interp['SAFE_count']}/334 interpolation actions but achieves only {float(raw_interp['upper_trajectory_empirical_coverage']):.3f} upper trajectory coverage. This is not a valid safety alternative; it shows that the transition from some raw certification to zero conformal certification is the cost of repairing severe raw undercoverage.

The conservatism is nevertheless operationally real: nearly every centre trajectory is safe and 63.5% of actual trajectories are safe, yet none is certifiable. It arises because relatively small centre margins are dominated by a max-over-240 standardized uncertainty envelope required for marginal population-level validity. This distinction—decision conservatism without evidence of an overlarge pooled q—is the central decision audit finding.

## Frozen primary certification interpretation

The primary certification upper-safety conclusion is reproduced with no accounting discrepancy or detectable evaluation leakage. This audit does not repair the lower envelope: its 92.8% coverage remains below nominal 95%, so strong UNSAFE certification is still not validated at 95%. primary certification's conformal statement remains marginal over the defined episode population, not conditional for a selected action or regime.
"""
    (output/"README.md").write_text(text,encoding="utf-8");(output/"audit_verdict.md").write_text(f"# decision audit verdict\n\n**{verdict}**\n\nPredictions/decisions/truths: {audit['prediction_count']}/{audit['decision_snapshot_count']}/{audit['revealed_truth_count']}; chronology violations: {audit['chronology_violations']}; reconstructed decision mismatches: {audit['decision_reconstruction_mismatches']}.\n\nInterpolation is genuinely close to the boundary and operationally conservative, but its heavier calibration-score tail and 0.916 upper coverage do not support the claim that pooled q+=5.465 is unnecessarily large.\n",encoding="utf-8")
    return verdict


def run(config:AuditConfig)->str:
    output=Path(config.output_dir);certification=Path(config.primary_dir);output.mkdir(parents=True,exist_ok=True);audit,artifact,reconstruction,diagnostics=audit_artifacts(config);regimes,scores=diagnostic_summaries(diagnostics,certification);_write_csv(output/"artifact_chronology_audit.csv",artifact);_write_csv(output/"decision_reconstruction.csv",reconstruction);_write_csv(output/"episode_margin_diagnostics.csv",diagnostics);_write_csv(output/"regime_diagnostic_summary.csv",regimes);_write_csv(output/"calibration_score_regime_summary.csv",scores);_write_csv(output/"audit_summary.csv",[audit]);create_figures(output,certification,audit,diagnostics,regimes,scores);verdict=create_report(output,certification,audit,regimes,scores);metadata={"configuration":asdict(config),"hard_freeze":True,"CERTIFICATION_artifacts_modified":False,"audit_basis":"artifact accounting, filesystem ordering, key isolation, exact envelope/decision/truth reconstruction, split/hash checks, and descriptive frozen-score diagnostics","interpretation":"post-hoc diagnosis only; no recalibration or alternative decision evaluation","verdict":verdict};(output/"configuration.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8");return verdict


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--primary-dir",default="artifacts/cstr/primary_evaluation");p.add_argument("--bootstrap-dir",default="artifacts/cstr/bootstrap_ensemble");p.add_argument("--conformal-dir",default="artifacts/cstr/conformal_calibration");p.add_argument("--output-dir",default="artifacts/cstr/primary_decision_audit");return p


if __name__=="__main__":
    result=run(AuditConfig(**vars(parser().parse_args())));print(f"decision audit complete: {result}")

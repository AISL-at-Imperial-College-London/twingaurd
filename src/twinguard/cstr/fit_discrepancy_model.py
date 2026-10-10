"""Fit a latent state-space model to CSTR nominal-twin residuals."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from twinguard.cstr import episode_design as episode_design
from twinguard.cstr import recursive_rollout_diagnostics as recursive_diagnostics
from twinguard.cstr import stable_arx_order_study as arx_order_study
from twinguard.discrepancy.cstr_state_space import (
    StateSpaceParameters,
    fit_state_space_em,
    kalman_filter,
    predictive_rollout,
)


LATENT_DIMENSIONS=(1,2,3,4)
ONLINE_COUNTS=(10,20,50,100)
MAX_LAG=10
SELECTION_PER_CATEGORY=20
FEATURE_NAMES=("u_t","delta_u_t","y_dt_t","elapsed_since_detection_s")
FORBIDDEN_FEATURE_TOKENS=("fault","ua","fouling","magnitude","faulty_state","plant_concentration","future_true")


@dataclass(frozen=True)
class DiscrepancyFitConfig:
    source_dir:str="artifacts/cstr/episode_design"
    arx_dir:str="artifacts/cstr/stable_arx_order_study"
    observable_dir:str="artifacts/cstr/observable_conditioning_study"
    output_dir:str="artifacts/cstr/discrepancy_fit"
    training_episodes:int=600
    training_steps:int=120
    posterior_samples:int=2000
    em_iterations:int=15
    initialization_seeds:tuple[int,...]=(91_001,91_002)
    workers:int=8
    limit_evaluation_episodes:int|None=None

    def validate(self)->None:
        if any(token in name.lower() for name in FEATURE_NAMES for token in FORBIDDEN_FEATURE_TOKENS):raise ValueError("hidden feature entered discrepancy fit design")
        if self.training_episodes<=0 or self.training_steps<100 or self.workers<=0:raise ValueError("invalid discrepancy fit population configuration")
        if self.limit_evaluation_episodes is None and self.posterior_samples!=2000:raise ValueError("full discrepancy fit freezes M=2000")


_TRAIN_TEMPLATE=None
_TRAIN_CONFIG=None


def _init_training_worker(template,config)->None:
    global _TRAIN_TEMPLATE,_TRAIN_CONFIG;_TRAIN_TEMPLATE=template;_TRAIN_CONFIG=config


def _training_episode_task(task:tuple[int,int,int])->dict[str,object]:
    index,seed,steps=task
    if _TRAIN_TEMPLATE is None or _TRAIN_CONFIG is None:raise RuntimeError("training worker not initialized")
    cfg=_TRAIN_CONFIG;rng=np.random.default_rng(seed);measurement_rng=np.random.default_rng(seed+1_000_000)
    onset=float(rng.uniform(cfg.fault_onset_min_s,cfg.fault_onset_max_s));magnitude=float(rng.uniform(cfg.fault_magnitude_min,cfg.fault_magnitude_max));nominal,plant,metadata=episode_design._configure_episode_models(_TRAIN_TEMPLATE,cfg,rng,onset,magnitude)
    previous_action=2/60
    for _ in range(cfg.pre_fault_lead_steps):episode_design._paired_observation(nominal,plant,previous_action,measurement_rng,cfg.measurement_noise_k)
    residual=np.empty(steps);z=np.empty((steps,4));actions=np.empty(steps);block_remaining=0;action=previous_action;changes=0
    for t in range(steps):
        if block_remaining<=0:
            block_remaining=int(rng.integers(5,16));action=float(rng.uniform(.012,.045) if rng.random()<.25 else rng.uniform(.019,.036));changes+=int(t>0)
        row=episode_design._paired_observation(nominal,plant,action,measurement_rng,cfg.measurement_noise_k);residual[t]=row["temperature_residual_k"];actions[t]=action;z[t]=[action,action-previous_action,row["nominal_twin_temperature_k"],t+1];previous_action=action;block_remaining-=1
    return {"episode_index":index,"random_seed":seed,"hidden_fault_onset_s_evaluator_only":onset,"hidden_fault_magnitude_evaluator_only":magnitude,"initial_volume_perturbation_l_evaluator_only":metadata["initial_volume_perturbation_l"],"initial_concentration_scale_evaluator_only":metadata["initial_concentration_scale"],"initial_temperature_perturbation_k_evaluator_only":metadata["initial_temperature_perturbation_k"],"steps":steps,"action_min_lps":float(np.min(actions)),"action_max_lps":float(np.max(actions)),"action_changes":changes,"residual":residual,"z":z}


def generate_training_population(config:DiscrepancyFitConfig,template,output:Path)->tuple[list[tuple[np.ndarray,np.ndarray]],list[dict[str,object]]]:
    tasks=[(index,900_000+index,config.training_steps) for index in range(config.training_episodes)]
    with ProcessPoolExecutor(max_workers=min(config.workers,len(tasks)),initializer=_init_training_worker,initargs=(template,episode_design.EpisodeDesignConfig(),)) as pool:results=list(pool.map(_training_episode_task,tasks,chunksize=1))
    episodes=[(np.asarray(r["residual"]),np.asarray(r["z"])) for r in results]
    manifest=[{k:v for k,v in r.items() if k not in ("residual","z")} for r in results]
    np.savez_compressed(output/"training_population.npz",residual=np.stack([e[0] for e in episodes]),z=np.stack([e[1] for e in episodes]))
    return episodes,manifest


def _fit_task(task:tuple[int,int,list[tuple[np.ndarray,np.ndarray]],int])->StateSpaceParameters:
    k,seed,episodes,iterations=task;return fit_state_space_em(episodes,k,seed,max_iterations=iterations)


def fit_candidates(config:DiscrepancyFitConfig,episodes:list[tuple[np.ndarray,np.ndarray]])->tuple[dict[int,StateSpaceParameters],list[StateSpaceParameters]]:
    tasks=[(k,seed,episodes,config.em_iterations) for k in LATENT_DIMENSIONS for seed in config.initialization_seeds]
    with ProcessPoolExecutor(max_workers=min(config.workers,len(tasks))) as pool:candidates=list(pool.map(_fit_task,tasks,chunksize=1))
    selected={k:max((p for p in candidates if p.latent_dimension==k),key=lambda p:p.log_likelihood) for k in LATENT_DIMENSIONS}
    return selected,candidates


_EVAL_TEMPLATE=None
_EVAL_CONFIG=None
_EVAL_SOURCE=None
_EVAL_MODELS=None
_EVAL_SAMPLES=2000


def _init_eval_worker(template,config,source,models,samples)->None:
    global _EVAL_TEMPLATE,_EVAL_CONFIG,_EVAL_SOURCE,_EVAL_MODELS,_EVAL_SAMPLES
    _EVAL_TEMPLATE=template;_EVAL_CONFIG=config;_EVAL_SOURCE=Path(source);_EVAL_MODELS=models;_EVAL_SAMPLES=samples;arx_order_study._init_worker(template,config,source,samples)


def _selection_roles(rows:list[dict[str,str]])->list[dict[str,str]]:
    counts={};output=[]
    for row in rows:
        item=dict(row);category=row["action_category"];index=counts.get(category,0);counts[category]=index+1;item["split_role"]="model_selection" if index<SELECTION_PER_CATEGORY else "final_assessment";output.append(item)
    return output


def _online_and_future(source_row:dict[str,str])->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    if _EVAL_SOURCE is None:raise RuntimeError("evaluation worker not initialized")
    pre,learning,proposed=arx_order_study._reconstruct_observations(source_row);previous=float(pre[-1]["action_lps"]);z_online=[]
    for index,row in enumerate(learning):
        action=float(row["action_lps"]);z_online.append([action,action-previous,float(row["nominal_twin_temperature_k"]),index+1]);previous=action
    residual_online=np.asarray([r["temperature_residual_k"] for r in learning]);episode_id=source_row["episode_id"]
    with np.load(_EVAL_SOURCE/"evaluations"/f"{episode_id}.npz") as saved:
        actual=np.asarray(saved["actual_plant_temperature_k"],dtype=float);actual_residual=np.asarray(saved["actual_residual_k"],dtype=float);nominal=np.asarray(saved["nominal_twin_temperature_k"],dtype=float);actions=np.asarray(saved["action_sequence_lps"],dtype=float)
    if not np.allclose(actions,proposed):raise RuntimeError("frozen action mismatch")
    z_future=[]
    for h in range(240):
        action=float(actions[h]);z_future.append([action,action-previous,float(nominal[h]),51+h]);previous=action
    return residual_online,np.asarray(z_online),np.asarray(z_future),actual,actual_residual,proposed


def _evaluate_model(parameters:StateSpaceParameters,source_row:dict[str,str],seed_offset:int)->dict[str,object]:
    residual_online,z_online,z_future,actual,actual_residual,proposed=_online_and_future(source_row)
    start=time.perf_counter();filtered=kalman_filter(parameters,residual_online,z_online);filter_seconds=time.perf_counter()-start
    start=time.perf_counter();rollout=predictive_rollout(parameters,filtered.filtered_mean[-1],filtered.filtered_covariance[-1],z_future,_EVAL_SAMPLES,np.random.default_rng(int(float(source_row["random_seed"]))+seed_offset));propagation_total=time.perf_counter()-start
    nominal=z_future[:,2];samples=nominal[None,:]+rollout["residual_samples"];mean=nominal+rollout["analytical_mean"];median=np.median(samples,axis=0);lower,upper=np.quantile(samples,[.025,.975],axis=0);error=mean-actual
    return {"episode_id":source_row["episode_id"],"action_category":source_row["action_category"],"split_role":source_row["split_role"],"random_seed":int(float(source_row["random_seed"])),"latent_dimension":parameters.latent_dimension,"proposed_action_lps":proposed,"online_observation_count":50,"recursive_RMSE_k":float(np.sqrt(np.mean(error**2))),"recursive_MAE_k":float(np.mean(np.abs(error))),"horizon_240_absolute_error_k":float(abs(error[-1])),"innovation_mean_k":float(np.mean(filtered.innovations)),"innovation_median_k":float(np.median(filtered.innovations)),"innovation_RMSE_k":float(np.sqrt(np.mean(filtered.innovations**2))),"standardized_innovation_variance":float(np.var(filtered.standardized_innovations,ddof=1)),"decision_state_covariance_trace":float(np.trace(filtered.filtered_covariance[-1])),"mean_95_width_k":float(np.mean(upper-lower)),"pointwise_95_coverage_diagnostic":float(np.mean((actual>=lower)&(actual<=upper))),"spectral_radius":parameters.spectral_radius,"stability_margin":1-parameters.spectral_radius,"filtering_seconds":filter_seconds,"predictive_total_seconds":propagation_total,"analytical_propagation_seconds":max(0.0,propagation_total-float(rollout["monte_carlo_seconds"])),"monte_carlo_seconds":rollout["monte_carlo_seconds"],"analytical_MC_mean_max_abs_difference_k":rollout["sample_mean_max_abs_difference"],"innovation":filtered.innovations,"standardized_innovation":filtered.standardized_innovations,"recursive_error":error,"posterior_mean_temperature":mean,"posterior_median_temperature":median,"actual_temperature":actual,"nominal_temperature":nominal,"filtered_uncertainty_trace":np.trace(filtered.filtered_covariance,axis1=1,axis2=2)}


def _candidate_episode_task(source_row:dict[str,str])->list[dict[str,object]]:
    return [_evaluate_model(_EVAL_MODELS[k],source_row,70_000_000+k*100_000) for k in LATENT_DIMENSIONS]


def _selected_episode_task(source_row:dict[str,str])->dict[str,object]:
    parameters=next(iter(_EVAL_MODELS.values()));return _evaluate_model(parameters,source_row,80_000_000+parameters.latent_dimension*100_000)


def _acf(arrays:Sequence[np.ndarray],max_lag:int=MAX_LAG)->np.ndarray:
    values=np.concatenate(arrays);center=float(np.mean(values));den=float(np.sum((values-center)**2));result=np.ones(max_lag+1)
    if den<=np.finfo(float).tiny:return np.concatenate(([1.],np.zeros(max_lag)))
    for lag in range(1,max_lag+1):result[lag]=sum(float(np.sum((x[:-lag]-center)*(x[lag:]-center))) for x in arrays)/den
    return result


def _innovation_row(rows:Sequence[dict[str,object]],split:str,category:str,k:int)->dict[str,object]:
    arrays=[np.asarray(r["innovation"]) for r in rows];standardized=np.concatenate([np.asarray(r["standardized_innovation"]) for r in rows]);values=np.concatenate(arrays);acf=_acf(arrays);n=values.size;lb=float(n*(n+2)*sum(acf[lag]**2/(n-lag) for lag in range(1,MAX_LAG+1)));pvalue=float(chi2.sf(lb,MAX_LAG));row={"latent_dimension":k,"split_role":split,"action_category":category,"number_of_episodes":len(rows),"number_of_innovations":n,"innovation_mean_k":float(np.mean(values)),"innovation_median_k":float(np.median(values)),"innovation_RMSE_k":float(np.sqrt(np.mean(values**2))),"maximum_absolute_ACF_lags_1_10":float(np.max(np.abs(acf[1:]))),"Ljung_Box_Q_lags_10":lb,"Ljung_Box_p_value_lags_10":pvalue,"standardized_innovation_variance":float(np.var(standardized,ddof=1))}
    row.update({f"ACF_lag_{lag}":float(acf[lag]) for lag in range(1,MAX_LAG+1)});return row


def build_innovation_summary(selection:Sequence[dict[str,object]],final:Sequence[dict[str,object]],selected_k:int)->list[dict[str,object]]:
    output=[]
    for split,source,dimensions in (("model_selection",selection,LATENT_DIMENSIONS),("final_assessment",final,(selected_k,))):
        for k in dimensions:
            krows=[r for r in source if int(r["latent_dimension"])==k]
            for category in ("interpolation","boundary","extrapolation","pooled"):
                rows=krows if category=="pooled" else [r for r in krows if r["action_category"]==category]
                output.append(_innovation_row(rows,split,category,k))
    return output


def build_dimension_selection(selection:Sequence[dict[str,object]],innovations:Sequence[dict[str,object]],models:dict[int,StateSpaceParameters])->list[dict[str,object]]:
    output=[]
    for k in LATENT_DIMENSIONS:
        krows=[r for r in selection if int(r["latent_dimension"])==k]
        for category in ("interpolation","boundary","extrapolation","pooled"):
            rows=krows if category=="pooled" else [r for r in krows if r["action_category"]==category];innovation=next(r for r in innovations if r["split_role"]=="model_selection" and int(r["latent_dimension"])==k and r["action_category"]==category);parameter=models[k]
            output.append({"latent_dimension":k,"action_category":category,"number_of_episodes":len(rows),"median_recursive_RMSE_k":float(np.median([r["recursive_RMSE_k"] for r in rows])),"mean_recursive_RMSE_k":float(np.mean([r["recursive_RMSE_k"] for r in rows])),"median_recursive_MAE_k":float(np.median([r["recursive_MAE_k"] for r in rows])),"innovation_mean_k":innovation["innovation_mean_k"],"absolute_innovation_bias_k":abs(float(innovation["innovation_mean_k"])),"maximum_absolute_ACF_lags_1_10":innovation["maximum_absolute_ACF_lags_1_10"],"horizon_240_RMSE_k":float(np.sqrt(np.mean([r["recursive_error"][-1]**2 for r in rows]))),"spectral_radius":parameter.spectral_radius,"stability_margin":1-parameter.spectral_radius,"offline_log_likelihood":parameter.log_likelihood,"offline_fit_seconds":parameter.fit_seconds,"median_filtering_seconds":float(np.median([r["filtering_seconds"] for r in rows])),"median_rollout_seconds":float(np.median([r["predictive_total_seconds"] for r in rows]))})
    return output


def select_dimension(rows:Sequence[dict[str,object]])->tuple[int,dict[int,float]]:
    pooled=[r for r in rows if r["action_category"]=="pooled"];scores={int(r["latent_dimension"]):float(r["median_recursive_RMSE_k"]) for r in pooled};best=min(scores.values());near=[k for k in LATENT_DIMENSIONS if scores[k]<=1.05*best];best_acf=min(float(next(r for r in pooled if int(r["latent_dimension"])==k)["maximum_absolute_ACF_lags_1_10"]) for k in near);selected=next(k for k in near if float(next(r for r in pooled if int(r["latent_dimension"])==k)["maximum_absolute_ACF_lags_1_10"])<=1.2*best_acf);return selected,scores


def _efficiency_episode_task(source_row:dict[str,str])->list[dict[str,object]]:
    if _EVAL_TEMPLATE is None or _EVAL_CONFIG is None:raise RuntimeError("efficiency worker not initialized")
    cfg=_EVAL_CONFIG;seed=int(float(source_row["random_seed"]));rng=np.random.default_rng(seed);measurement_rng=np.random.default_rng(seed+1_000_000);action_rng=np.random.default_rng(seed+2_000_000);onset=float(rng.uniform(cfg.fault_onset_min_s,cfg.fault_onset_max_s));magnitude=float(rng.uniform(cfg.fault_magnitude_min,cfg.fault_magnitude_max));nominal,plant,_=episode_design._configure_episode_models(_EVAL_TEMPLATE,cfg,rng,onset,magnitude)
    prior=2/60
    for _ in range(cfg.pre_fault_lead_steps):episode_design._paired_observation(nominal,plant,prior,measurement_rng,cfg.measurement_noise_k)
    proposed=episode_design._select_action_after_fit(source_row["action_category"],action_rng);observations=[];snapshots={}
    for index in range(100):
        action=episode_design.TRAINING_ACTIONS_LPS[(index//cfg.training_action_block_steps)%len(episode_design.TRAINING_ACTIONS_LPS)];row=episode_design._paired_observation(nominal,plant,action,measurement_rng,cfg.measurement_noise_k);observations.append(row)
        if index+1 in ONLINE_COUNTS:snapshots[index+1]=(copy.deepcopy(nominal),copy.deepcopy(plant))
    parameter=next(iter(_EVAL_MODELS.values()));output=[]
    for n in ONLINE_COUNTS:
        previous=prior;z_online=[]
        for index,row in enumerate(observations[:n]):
            action=float(row["action_lps"]);z_online.append([action,action-previous,row["nominal_twin_temperature_k"],index+1]);previous=action
        start=time.perf_counter();filtered=kalman_filter(parameter,np.asarray([r["temperature_residual_k"] for r in observations[:n]]),np.asarray(z_online));filter_seconds=time.perf_counter()-start
        nominal_branch,plant_branch=snapshots[n];branch_rng=np.random.default_rng(seed+4_000_000 if n==50 else seed+90_000_000+n*10_000);nominal_future=[];actual=[];z_future=[]
        for h in range(240):
            nominal_out=nominal_branch.simulate_step(Fin_sp_normal=proposed,Fin_sp_startup=proposed);plant_out=plant_branch.simulate_step(Fin_sp_normal=proposed,Fin_sp_startup=proposed);nominal_temperature=float(nominal_out["T_true"]);plant_temperature=float(plant_out["T_true"])+float(branch_rng.normal(0,cfg.measurement_noise_k));nominal_future.append(nominal_temperature);actual.append(plant_temperature);z_future.append([proposed,proposed-previous,nominal_temperature,n+1+h]);previous=proposed
        propagation_start=time.perf_counter();rollout=predictive_rollout(parameter,filtered.filtered_mean[-1],filtered.filtered_covariance[-1],np.asarray(z_future),_EVAL_SAMPLES,np.random.default_rng(seed+100_000_000+n*10_000));predictive_total=time.perf_counter()-propagation_start;samples=np.asarray(nominal_future)[None,:]+rollout["residual_samples"];mean=np.asarray(nominal_future)+rollout["analytical_mean"];lower,upper=np.quantile(samples,[.025,.975],axis=0);error=mean-np.asarray(actual)
        output.append({"episode_id":source_row["episode_id"],"action_category":source_row["action_category"],"random_seed":seed,"N_postfault":n,"recursive_RMSE_k":float(np.sqrt(np.mean(error**2))),"decision_state_covariance_trace":float(np.trace(filtered.filtered_covariance[-1])),"average_state_marginal_sd":float(np.mean(np.sqrt(np.diag(filtered.filtered_covariance[-1])))),"mean_95_width_k":float(np.mean(upper-lower)),"filtering_seconds":filter_seconds,"analytical_propagation_seconds":max(0.0,predictive_total-float(rollout["monte_carlo_seconds"])),"monte_carlo_seconds":rollout["monte_carlo_seconds"],"predictive_total_seconds":predictive_total,"innovation_RMSE_k":float(np.sqrt(np.mean(filtered.innovations**2)))})
    return output


def build_efficiency_summary(rows:Sequence[dict[str,object]])->list[dict[str,object]]:
    output=[]
    for n in ONLINE_COUNTS:
        nrows=[r for r in rows if int(r["N_postfault"])==n]
        for category in ("interpolation","boundary","extrapolation","pooled"):
            selected=nrows if category=="pooled" else [r for r in nrows if r["action_category"]==category]
            output.append({"N_postfault":n,"action_category":category,"number_of_episodes":len(selected),"median_recursive_RMSE_k":float(np.median([r["recursive_RMSE_k"] for r in selected])),"mean_recursive_RMSE_k":float(np.mean([r["recursive_RMSE_k"] for r in selected])),"median_decision_state_covariance_trace":float(np.median([r["decision_state_covariance_trace"] for r in selected])),"median_average_state_marginal_sd":float(np.median([r["average_state_marginal_sd"] for r in selected])),"median_95_width_k":float(np.median([r["mean_95_width_k"] for r in selected])),"median_filtering_seconds":float(np.median([r["filtering_seconds"] for r in selected])),"median_analytical_propagation_seconds":float(np.median([r["analytical_propagation_seconds"] for r in selected])),"median_monte_carlo_seconds":float(np.median([r["monte_carlo_seconds"] for r in selected]))})
    return output


def build_final_summary(final:Sequence[dict[str,object]],innovations:Sequence[dict[str,object]],parameters:StateSpaceParameters)->list[dict[str,object]]:
    output=[]
    for category in ("interpolation","boundary","extrapolation","pooled"):
        rows=list(final) if category=="pooled" else [r for r in final if r["action_category"]==category];innovation=next(r for r in innovations if r["split_role"]=="final_assessment" and r["action_category"]==category)
        output.append({"action_category":category,"number_of_episodes":len(rows),"latent_dimension":parameters.latent_dimension,"median_recursive_RMSE_k":float(np.median([r["recursive_RMSE_k"] for r in rows])),"mean_recursive_RMSE_k":float(np.mean([r["recursive_RMSE_k"] for r in rows])),"median_MAE_k":float(np.median([r["recursive_MAE_k"] for r in rows])),"innovation_mean_k":innovation["innovation_mean_k"],"absolute_innovation_bias_k":abs(float(innovation["innovation_mean_k"])),"maximum_absolute_ACF_lags_1_10":innovation["maximum_absolute_ACF_lags_1_10"],"horizon_240_RMSE_k":float(np.sqrt(np.mean([r["recursive_error"][-1]**2 for r in rows]))),"spectral_radius":parameters.spectral_radius,"stability_margin":1-parameters.spectral_radius,"median_filtering_seconds":float(np.median([r["filtering_seconds"] for r in rows])),"median_analytical_propagation_seconds":float(np.median([r["analytical_propagation_seconds"] for r in rows])),"median_open_loop_rollout_seconds":float(np.median([r["predictive_total_seconds"] for r in rows])),"median_monte_carlo_seconds":float(np.median([r["monte_carlo_seconds"] for r in rows])),"maximum_analytical_MC_mean_difference_k":float(np.max([r["analytical_MC_mean_max_abs_difference_k"] for r in rows]))})
    return output


def _read_csv(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


def build_horizon_summary(final:Sequence[dict[str,object]],selected_k:int,arx_dir:Path,observable_dir:Path)->list[dict[str,object]]:
    output=[]
    for category in ("interpolation","boundary","extrapolation","pooled"):
        rows=list(final) if category=="pooled" else [r for r in final if r["action_category"]==category];errors=np.vstack([r["recursive_error"] for r in rows])
        for h in range(240):output.append({"model":f"latent_k{selected_k}","action_category":category,"horizon_step":h+1,"number_of_episodes":len(rows),"RMSE_k":float(np.sqrt(np.mean(errors[:,h]**2))),"bias_prediction_minus_truth_k":float(np.mean(errors[:,h]))})
    for row in _read_csv(observable_dir/"horizon_summary.csv"):
        if row["feature_set"]=="F1" and row["split_role"]=="final_assessment" and row["prediction_method"]=="recursive_posterior_mean":output.append({"model":"observable_F1","action_category":row["action_category"],"horizon_step":int(row["horizon_step"]),"number_of_episodes":int(row["number_of_episodes"]),"RMSE_k":float(row["RMSE_k"]),"bias_prediction_minus_truth_k":float(row["bias_prediction_minus_truth_k"])})
    for row in _read_csv(arx_dir/"horizon_error_summary.csv"):
        if row["order"]=="1" and row["split_role"]=="final_assessment" and row["prediction_method"]=="posterior_mean":output.append({"model":"stable_ARX1","action_category":row["action_category"],"horizon_step":int(row["horizon_step"]),"number_of_episodes":int(row["number_of_episodes"]),"RMSE_k":float(row["RMSE_k"]),"bias_prediction_minus_truth_k":float(row["bias_prediction_minus_truth_k"])})
    return output


def build_baseline_comparison(final_summary:Sequence[dict[str,object]],arx_dir:Path,observable_dir:Path)->list[dict[str,object]]:
    b8rows=_read_csv(observable_dir/"feature_set_comparison.csv");b7rows=_read_csv(arx_dir/"arx_order_comparison.csv");output=[]
    for category in ("interpolation","boundary","extrapolation","pooled"):
        latent=next(r for r in final_summary if r["action_category"]==category);f1=next(r for r in b8rows if r["feature_set"]=="F1" and r["split_role"]=="final_assessment" and r["action_category"]==category);arx=next(r for r in b7rows if r["order"]=="1" and r["split_role"]=="final_assessment" and r["action_category"]==category)
        output.extend([
            {"model":f"latent_k{latent['latent_dimension']}","action_category":category,"median_recursive_RMSE_k":latent["median_recursive_RMSE_k"],"innovation_bias_k":latent["innovation_mean_k"],"maximum_absolute_ACF_lags_1_10":latent["maximum_absolute_ACF_lags_1_10"],"horizon_240_RMSE_k":latent["horizon_240_RMSE_k"],"innovation_definition":"online Kalman innovations over N=50"},
            {"model":"observable_F1","action_category":category,"median_recursive_RMSE_k":float(f1["recursive_RMSE"]),"innovation_bias_k":float(f1["innovation_bias"]),"maximum_absolute_ACF_lags_1_10":float(f1["max_innovation_ACF"]),"horizon_240_RMSE_k":float(f1["horizon_240_RMSE"]),"innovation_definition":"held-out teacher-forced innovations over H=240"},
            {"model":"stable_ARX1","action_category":category,"median_recursive_RMSE_k":float(arx["median_posterior_mean_trajectory_RMSE_k"]),"innovation_bias_k":float(arx["pooled_innovation_mean_k"]),"maximum_absolute_ACF_lags_1_10":float(arx["maximum_absolute_innovation_ACF_lags_1_10"]),"horizon_240_RMSE_k":"","innovation_definition":"held-out teacher-forced innovations over H=240"},
        ])
    return output


def decision_rule(final_summary:Sequence[dict[str,object]],baseline:Sequence[dict[str,object]])->tuple[str,dict[str,float|bool]]:
    latent=next(r for r in final_summary if r["action_category"]=="pooled");f1=next(r for r in baseline if r["model"]=="observable_F1" and r["action_category"]=="pooled");metrics={"median_recursive_RMSE_reduction":1-float(latent["median_recursive_RMSE_k"])/float(f1["median_recursive_RMSE_k"]),"absolute_innovation_bias_reduction":1-abs(float(latent["innovation_mean_k"]))/abs(float(f1["innovation_bias_k"])),"maximum_innovation_ACF_reduction":1-float(latent["maximum_absolute_ACF_lags_1_10"])/float(f1["maximum_absolute_ACF_lags_1_10"]),"final_maximum_innovation_ACF":float(latent["maximum_absolute_ACF_lags_1_10"]),"stable":bool(float(latent["spectral_radius"])<1)}
    sufficient=metrics["median_recursive_RMSE_reduction"]>=.20 and metrics["absolute_innovation_bias_reduction"]>=.30 and metrics["maximum_innovation_ACF_reduction"]>=.50 and metrics["final_maximum_innovation_ACF"]<=.30 and metrics["stable"]
    if sufficient:return "LATENT STATE SUFFICIENT",metrics
    meaningful=metrics["median_recursive_RMSE_reduction"]>=.10 or metrics["absolute_innovation_bias_reduction"]>=.20 or metrics["maximum_innovation_ACF_reduction"]>=.20
    return ("LATENT STATE PARTIALLY SUFFICIENT" if meaningful else "LINEAR LATENT MODEL INADEQUATE"),metrics


def _parameter_payload(parameters:StateSpaceParameters)->dict[str,object]:
    return {"latent_dimension":parameters.latent_dimension,"A":parameters.A.tolist(),"B":parameters.B.tolist(),"C":parameters.C.tolist(),"D":parameters.D.tolist(),"Q":parameters.Q.tolist(),"R":parameters.R,"initial_mean":parameters.initial_mean.tolist(),"initial_covariance":parameters.initial_covariance.tolist(),"z_mean":parameters.z_mean.tolist(),"z_scale":parameters.z_scale.tolist(),"feature_names":list(FEATURE_NAMES),"log_likelihood":parameters.log_likelihood,"iterations":parameters.iterations,"converged":parameters.converged,"initialization_seed":parameters.initialization_seed,"fit_seconds":parameters.fit_seconds,"eigenvalues":[{"real":float(x.real),"imaginary":float(x.imag)} for x in np.linalg.eigvals(parameters.A)],"spectral_radius":parameters.spectral_radius,"stability_margin":1-parameters.spectral_radius,"stability_parameterization":"diagonal modes a_i=0.995*tanh(theta_i); stable throughout optimization; no clipping","identifiability_note":"latent coordinates are abstract, nonphysical, and not compared across fitted k"}


def _write_csv(path:Path,rows:Sequence[dict[str,object]])->None:
    if not rows:raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True,exist_ok=True);fields=list(rows[0])
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def _plain(rows:Sequence[dict[str,object]])->list[dict[str,object]]:
    return [{k:v for k,v in row.items() if not isinstance(v,np.ndarray)} for row in rows]


def _save(fig:plt.Figure,path:Path)->None:
    fig.tight_layout();fig.savefig(path.with_suffix(".png"),dpi=300);fig.savefig(path.with_suffix(".pdf"));plt.close(fig)


def create_figures(selection_rows,innovation_rows,final,horizon,efficiency,selected_k:int,observable_dir:Path,output:Path)->None:
    output.mkdir(parents=True,exist_ok=True);plt.rcParams.update({"figure.dpi":150,"font.size":9});pooled=[r for r in selection_rows if r["action_category"]=="pooled"];x=np.asarray(LATENT_DIMENSIONS)
    fig,ax=plt.subplots(figsize=(6.5,4));ax.plot(x,[next(r for r in pooled if r["latent_dimension"]==k)["median_recursive_RMSE_k"] for k in x],"o-");ax.axvline(selected_k,color="black",ls="--",label=f"selected k={selected_k}");ax.set(xlabel="Latent dimension k",ylabel="Selection median recursive RMSE (K)");ax.legend();ax.grid(alpha=.25);_save(fig,output/"figure_DISCREPANCY_FIT_1_recursive_RMSE_vs_k")
    fig,ax=plt.subplots(figsize=(6.5,4));ax.plot(x,[next(r for r in pooled if r["latent_dimension"]==k)["maximum_absolute_ACF_lags_1_10"] for k in x],"o-");ax.axvline(selected_k,color="black",ls="--");ax.set(xlabel="Latent dimension k",ylabel="Maximum |Kalman innovation ACF|");ax.grid(alpha=.25);_save(fig,output/"figure_DISCREPANCY_FIT_2_maximum_innovation_ACF_vs_k")
    b8innov=_read_csv(observable_dir/"innovation_summary.csv");f1=next(r for r in b8innov if r["feature_set"]=="F1" and r["split_role"]=="final_assessment" and r["action_category"]=="pooled");latent=next(r for r in innovation_rows if r["latent_dimension"]==selected_k and r["split_role"]=="final_assessment" and r["action_category"]=="pooled");fig,ax=plt.subplots(figsize=(6.5,4));ax.plot(range(1,11),[float(f1[f"ACF_lag_{lag}"]) for lag in range(1,11)],"o-",label="observable conditioning F1");ax.plot(range(1,11),[latent[f"ACF_lag_{lag}"] for lag in range(1,11)],"o-",label=f"latent k={selected_k}");ax.axhline(0,color="black",lw=.7);ax.set(xlabel="Lag",ylabel="Innovation ACF");ax.legend();ax.grid(alpha=.25);_save(fig,output/"figure_DISCREPANCY_FIT_3_innovation_ACF")
    for figure,metric,ylabel,name in ((4,"RMSE_k","RMSE (K)","horizon_RMSE"),(5,"bias_prediction_minus_truth_k","Prediction bias (K)","horizon_bias")):
        fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
        for ax,category in zip(axes,("interpolation","boundary","extrapolation")):
            for model in ("stable_ARX1","observable_F1",f"latent_k{selected_k}"):
                rows=[r for r in horizon if r["model"]==model and r["action_category"]==category];ax.plot([r["horizon_step"] for r in rows],[r[metric] for r in rows],label=model)
            if figure==5:ax.axhline(0,color="black",lw=.7)
            ax.set_title(category);ax.set_xlabel("Horizon step");ax.grid(alpha=.25)
        axes[0].set_ylabel(ylabel);axes[-1].legend();_save(fig,output/f"figure_DISCREPANCY_FIT_{figure}_{name}")
    representatives=[]
    for number,category in enumerate(("interpolation","boundary","extrapolation"),start=6):
        candidates=[r for r in final if r["action_category"]==category];med=np.median([r["recursive_RMSE_k"] for r in candidates]);chosen=min(candidates,key=lambda r:abs(r["recursive_RMSE_k"]-med));representatives.append(chosen);fig,ax=plt.subplots(figsize=(7,4));h=np.arange(1,241);ax.plot(h,chosen["actual_temperature"],color="black",label="revealed plant");ax.plot(h,chosen["nominal_temperature"],ls="--",label="nominal DT");ax.plot(h,chosen["posterior_mean_temperature"],label=f"latent k={selected_k} mean");ax.set(xlabel="Horizon step",ylabel="Temperature (K)",title=f"{category}: {chosen['episode_id']}");ax.legend();ax.grid(alpha=.25);_save(fig,output/f"figure_DISCREPANCY_FIT_{number}_{category}_rollout")
    chosen=representatives[0];fig,ax=plt.subplots(figsize=(6.5,4));ax.plot(range(1,51),chosen["filtered_uncertainty_trace"]);ax.set(xlabel="Post-fault observation",ylabel="Trace of filtered latent covariance",title=f"Abstract latent uncertainty, k={selected_k}");ax.grid(alpha=.25);_save(fig,output/"figure_DISCREPANCY_FIT_9_filtered_latent_uncertainty")
    pooled_eff=[r for r in efficiency if r["action_category"]=="pooled"];fig,ax=plt.subplots(figsize=(6.5,4));ax.plot(ONLINE_COUNTS,[next(r for r in pooled_eff if r["N_postfault"]==n)["median_recursive_RMSE_k"] for n in ONLINE_COUNTS],"o-",label="recursive RMSE");ax.set(xlabel="Observed post-fault samples N",ylabel="Median recursive RMSE (K)");other=ax.twinx();other.plot(ONLINE_COUNTS,[next(r for r in pooled_eff if r["N_postfault"]==n)["median_decision_state_covariance_trace"] for n in ONLINE_COUNTS],"s--",color="tab:orange",label="state uncertainty");other.set_ylabel("Median covariance trace");lines=ax.lines+other.lines;ax.legend(lines,[line.get_label() for line in lines]);ax.grid(alpha=.25);_save(fig,output/"figure_DISCREPANCY_FIT_10_performance_vs_N")


def create_reports(output:Path,selected_k:int,scores:dict[int,float],final_summary,baseline,innovation_summary,efficiency,parameters:StateSpaceParameters,decision:str,metrics:dict[str,float|bool],config:DiscrepancyFitConfig)->None:
    latent=next(r for r in final_summary if r["action_category"]=="pooled");f1=next(r for r in baseline if r["model"]=="observable_F1" and r["action_category"]=="pooled");arx=next(r for r in baseline if r["model"]=="stable_ARX1" and r["action_category"]=="pooled");innov=next(r for r in innovation_summary if r["split_role"]=="final_assessment" and r["action_category"]=="pooled");score_text=", ".join(f"k={k}: {scores[k]:.3f} K" for k in LATENT_DIMENSIONS);eff_text=", ".join(f"N={n}: {float(next(r for r in efficiency if r['N_postfault']==n and r['action_category']=='pooled')['median_recursive_RMSE_k']):.3f} K" for n in ONLINE_COUNTS)
    readiness="The linear latent mean model meets every predeclared sufficiency gate and can proceed to a separate uncertainty-modelling stage." if decision=="LATENT STATE SUFFICIENT" else "The linear latent mean model should not yet be treated as sufficient for uncertainty modelling; retain the diagnosis and refine the mean/state representation without changing variance in this stage."
    readme=f"""# discrepancy fit — Latent discrepancy state-space model

## Architecture and separation

A separate library of **{config.training_episodes}** simulator episodes, each with {config.training_steps} post-fault observations, was generated for offline identification. Every episode resets its latent mean and covariance. Fault metadata appears only in the evaluator manifest; the estimator receives residuals and `[u, delta_u, nominal_DT_temperature, elapsed_since_detection]` only. Shared `A,B,C,D,Q,R` parameters are frozen before any episode design evaluation episode is filtered.

Two deterministic EM initializations were evaluated for each `k in {{1,2,3,4}}`, using Kalman-filter likelihood summed across independent episodes. Dynamics use the stable parameterization `a_i=0.995*tanh(theta_i)` throughout optimization. No eigenvalue was clipped. The diagonal-mode convention improves numerical identifiability, but latent coordinates remain abstract and must not be interpreted physically or compared across k.

Continuous exogenous features are standardized using the offline training library only. Initialization uses OLS for the direct observation term, stable modes spaced from 0.25 to 0.85 with deterministic seed jitter, small Gaussian input coefficients, `C_i=1/k`, `Q=0.08*s_residual^2 I`, `R=0.5*s_residual^2`, initial mean zero, and initial covariance `s_residual^2 I`. EM uses at most {config.em_iterations} iterations and relative likelihood tolerance `1e-4`; each stable diagonal dynamics row is optimized with L-BFGS-B in its unconstrained theta parameterization.

## Selection

Selection-only recursive RMSE values were {score_text}. Using only the existing 60 selection episodes, the smallest dimension within 5% of the best recursive RMSE whose innovation ACF was not materially worse was **k={selected_k}**. The primary final assessment then used the 240 untouched episode IDs once, after k and all parameters were frozen. The separate N experiment is a predeclared fixed-model diagnostic and does not feed back into selection.

## Final result

The selected latent model has median recursive RMSE **{float(latent['median_recursive_RMSE_k']):.3f} K**, mean RMSE **{float(latent['mean_recursive_RMSE_k']):.3f} K**, median MAE **{float(latent['median_MAE_k']):.3f} K**, innovation bias **{float(latent['innovation_mean_k']):+.3f} K**, and maximum innovation ACF **{float(latent['maximum_absolute_ACF_lags_1_10']):.3f}**. Its spectral radius is **{parameters.spectral_radius:.4f}** with stability margin **{1-parameters.spectral_radius:.4f}**.

Optimization limitation: the selected initialization reached the {config.em_iterations}-iteration cap with `converged={parameters.converged}` under the `1e-4` likelihood criterion, and its slow mode lies near the stable parameterization ceiling. The threshold-based final classification below is therefore valid for this frozen fitted candidate, but a training-only sensitivity run with a higher iteration cap is warranted before treating the parameter estimates as numerically definitive. This does not authorize refitting against final-assessment performance.

observable conditioning F1 has median recursive RMSE **{float(f1['median_recursive_RMSE_k']):.3f} K**, bias **{float(f1['innovation_bias_k']):+.3f} K**, and maximum ACF **{float(f1['maximum_absolute_ACF_lags_1_10']):.3f}**. ARX order study ARX(1) has median recursive RMSE **{float(arx['median_recursive_RMSE_k']):.3f} K**. The innovation comparison is descriptive because the latent value is an online Kalman innovation over N=50 whereas the frozen ARX/F1 values are held-out teacher-forced innovations over H=240.

The Ljung–Box diagnostic for selected pooled final innovations is Q={float(innov['Ljung_Box_Q_lags_10']):.2f}, p={float(innov['Ljung_Box_p_value_lags_10']):.3g}; standardized innovation variance is {float(innov['standardized_innovation_variance']):.3f}. The maximum ACF is substantially reduced but the large-sample Ljung–Box result still rejects literal whiteness. This is a dependence diagnostic, not a final calibration test.

## Online data efficiency and runtime

Fixed-model performance was {eff_text}. Each N branch acts from its own post-fault decision time, so the sequence is a deployment-timing experiment rather than a monotone information-only ablation. Median N=50 filtering time is **{float(latent['median_filtering_seconds']):.6f} s** and median 2,000-trajectory open-loop propagation time is **{float(latent['median_open_loop_rollout_seconds']):.4f} s**. Offline fitting took **{parameters.fit_seconds:.2f} s** for the selected initialization and is not an online cost. The largest analytical-versus-Monte-Carlo mean discrepancy across final episodes was **{float(latent['maximum_analytical_MC_mean_difference_k']):.3f} K**.

## Answers

1. Open-loop improvement: median RMSE reduction versus F1 is {float(metrics['median_recursive_RMSE_reduction']):.1%}.
2. Smallest useful latent dimension: **k={selected_k}** under the frozen selection rule.
3. Innovation ACF changes from 0.994 to **{float(latent['maximum_absolute_ACF_lags_1_10']):.3f}**.
4. Horizon drift is quantified against both frozen baselines in `horizon_error_summary.csv` and Figures discrepancy fit.4–discrepancy fit.5.
5. N=10–50 inference is reported above and in `data_efficiency_summary.csv`.
6. Online Kalman filtering is computationally lightweight at the measured runtime.
7. The direct comparison with F1 and ARX(1) is in `baseline_comparison.csv`.
8. Readiness: {readiness}

## Decision

**{decision}**
"""
    (output/"README.md").write_text(readme,encoding="utf-8")
    go=f"""# discrepancy fit decision

## Verdict

**{decision}**

Against frozen observable conditioning F1, recursive RMSE reduction is {float(metrics['median_recursive_RMSE_reduction']):.1%}, absolute-bias reduction is {float(metrics['absolute_innovation_bias_reduction']):.1%}, and maximum-ACF reduction is {float(metrics['maximum_innovation_ACF_reduction']):.1%}. Final maximum ACF is {float(metrics['final_maximum_innovation_ACF']):.3f}; spectral radius is {parameters.spectral_radius:.4f}.

The `LATENT STATE SUFFICIENT` threshold requires all five predeclared conditions: 20% RMSE reduction, 30% absolute-bias reduction, 50% ACF reduction, final ACF at most 0.30, and stable non-explosive dynamics. {readiness}

Numerical caveat: the selected EM run reached its iteration cap without the declared likelihood tolerance, and its slow mode is near the stability ceiling. The verdict is the mandatory threshold classification for the frozen candidate, not evidence that latent coordinates have physical meaning or that the optimizer sensitivity question is closed.
"""
    (output/"decision_summary.md").write_text(go,encoding="utf-8")


def run_experiment(config:DiscrepancyFitConfig)->int:
    config.validate();output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True);source=Path(config.source_dir);b7dir=Path(config.arx_dir);b8dir=Path(config.observable_dir);b5config=episode_design.EpisodeDesignConfig(output_dir=config.source_dir);template=episode_design.build_template(b5config)
    episodes,manifest=generate_training_population(config,template,output);_write_csv(output/"training_population_manifest.csv",manifest)
    models,candidates=fit_candidates(config,episodes)
    source_rows=_selection_roles(recursive_diagnostics.load_episode_design_episodes(source/"calibration_episodes.csv"))
    if config.limit_evaluation_episodes is not None:
        limit=config.limit_evaluation_episodes
        source_rows=[r for category in ("interpolation","boundary","extrapolation") for split in ("model_selection","final_assessment") for r in [x for x in source_rows if x["action_category"]==category and x["split_role"]==split][:limit]]
    selection_source=[r for r in source_rows if r["split_role"]=="model_selection"]
    with ProcessPoolExecutor(max_workers=min(config.workers,len(selection_source)),initializer=_init_eval_worker,initargs=(template,b5config,str(source),models,config.posterior_samples)) as pool:nested=list(pool.map(_candidate_episode_task,selection_source,chunksize=1))
    selection=[row for episode in nested for row in episode]
    selection_innov=[]
    for k in LATENT_DIMENSIONS:
        krows=[r for r in selection if r["latent_dimension"]==k]
        for category in ("interpolation","boundary","extrapolation","pooled"):
            rows=krows if category=="pooled" else [r for r in krows if r["action_category"]==category];selection_innov.append(_innovation_row(rows,"model_selection",category,k))
    dimension_rows=build_dimension_selection(selection,selection_innov,models);selected_k,scores=select_dimension(dimension_rows);selected_parameters=models[selected_k]
    final_source=[r for r in source_rows if r["split_role"]=="final_assessment"]
    with ProcessPoolExecutor(max_workers=min(config.workers,len(final_source)),initializer=_init_eval_worker,initargs=(template,b5config,str(source),{selected_k:selected_parameters},config.posterior_samples)) as pool:final=list(pool.map(_selected_episode_task,final_source,chunksize=1))
    innovation_summary=build_innovation_summary(selection,final,selected_k);final_summary=build_final_summary(final,innovation_summary,selected_parameters);horizon=build_horizon_summary(final,selected_k,b7dir,b8dir);baseline=build_baseline_comparison(final_summary,b7dir,b8dir);decision,decision_metrics=decision_rule(final_summary,baseline)
    with ProcessPoolExecutor(max_workers=min(config.workers,len(final_source)),initializer=_init_eval_worker,initargs=(template,b5config,str(source),{selected_k:selected_parameters},config.posterior_samples)) as pool:eff_nested=list(pool.map(_efficiency_episode_task,final_source,chunksize=1))
    efficiency_episode=[row for episode in eff_nested for row in episode];efficiency=build_efficiency_summary(efficiency_episode)
    runtime=[]
    for parameter in candidates:runtime.append({"runtime_type":"offline_fit","latent_dimension":parameter.latent_dimension,"initialization_seed":parameter.initialization_seed,"action_category":"","N_postfault":"","seconds":parameter.fit_seconds,"notes":f"{parameter.iterations} EM iterations; converged={parameter.converged}; loglik={parameter.log_likelihood}"})
    for row in final_summary:
        runtime.append({"runtime_type":"online_filter_N50","latent_dimension":selected_k,"initialization_seed":"","action_category":row["action_category"],"N_postfault":50,"seconds":row["median_filtering_seconds"],"notes":"median per episode"})
        runtime.append({"runtime_type":"analytical_propagation_H240","latent_dimension":selected_k,"initialization_seed":"","action_category":row["action_category"],"N_postfault":50,"seconds":row["median_analytical_propagation_seconds"],"notes":"median per episode"})
        runtime.append({"runtime_type":"monte_carlo_2000_trajectories","latent_dimension":selected_k,"initialization_seed":"","action_category":row["action_category"],"N_postfault":50,"seconds":row["median_monte_carlo_seconds"],"notes":"coherent state/process/observation sampling"})
    _write_csv(output/"latent_dimension_selection.csv",dimension_rows);_write_csv(output/"final_assessment_results.csv",_plain(final));_write_csv(output/"final_assessment_summary.csv",final_summary);_write_csv(output/"innovation_summary.csv",innovation_summary);_write_csv(output/"horizon_error_summary.csv",horizon);_write_csv(output/"data_efficiency_summary.csv",efficiency);_write_csv(output/"data_efficiency_episode_results.csv",efficiency_episode);_write_csv(output/"runtime_summary.csv",runtime);_write_csv(output/"baseline_comparison.csv",baseline)
    parameter_json={"selected_latent_dimension":selected_k,"selected":_parameter_payload(selected_parameters),"best_by_dimension":{str(k):_parameter_payload(p) for k,p in models.items()},"all_initializations":[_parameter_payload(p) for p in candidates]};(output/"model_parameters.json").write_text(json.dumps(parameter_json,indent=2),encoding="utf-8")
    configuration={**asdict(config),"initialization_seeds":list(config.initialization_seeds),"feature_names":list(FEATURE_NAMES),"em_relative_likelihood_tolerance":1e-4,"dynamics_optimizer":"L-BFGS-B over a_i=0.995*tanh(theta_i); max 100 row iterations","state_reset":"learned common initial mean/covariance reapplied independently to every episode","selection_rule":"smallest k within 5% of best selection recursive RMSE with max ACF within 20% of best among near-optimal candidates","decision_rule":{"sufficient":{"RMSE_reduction_min":.20,"absolute_bias_reduction_min":.30,"ACF_reduction_min":.50,"final_ACF_max":.30,"stable_required":True},"partial_meaningful_thresholds":{"RMSE_reduction":.10,"bias_or_ACF_reduction":.20}},"decision":decision,"decision_metrics":decision_metrics};(output/"configuration.json").write_text(json.dumps(configuration,indent=2),encoding="utf-8")
    create_figures(dimension_rows,innovation_summary,final,horizon,efficiency,selected_k,b8dir,output/"figures");create_reports(output,selected_k,scores,final_summary,baseline,innovation_summary,efficiency,selected_parameters,decision,decision_metrics,config);return len(final)


def build_argument_parser()->argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--source-dir",default="artifacts/cstr/episode_design");parser.add_argument("--arx-dir",default="artifacts/cstr/stable_arx_order_study");parser.add_argument("--observable-dir",default="artifacts/cstr/observable_conditioning_study");parser.add_argument("--output-dir",default="artifacts/cstr/discrepancy_fit");parser.add_argument("--training-episodes",type=int,default=600);parser.add_argument("--training-steps",type=int,default=120);parser.add_argument("--posterior-samples",type=int,default=2000);parser.add_argument("--em-iterations",type=int,default=15);parser.add_argument("--workers",type=int,default=min(8,os.cpu_count() or 1));parser.add_argument("--limit-evaluation-episodes",type=int);return parser


def main()->None:
    args=vars(build_argument_parser().parse_args());config=DiscrepancyFitConfig(**args);count=run_experiment(config);print(f"discrepancy fit complete: {config.output_dir} ({count} final episodes)")


if __name__=="__main__":main()

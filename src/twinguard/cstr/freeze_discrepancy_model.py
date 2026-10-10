"""Assess CSTR discrepancy-fit stability and calibrate predictive uncertainty."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, kurtosis, skew, spearmanr

from twinguard.cstr import episode_design as episode_design
from twinguard.cstr import recursive_rollout_diagnostics as recursive_diagnostics
from twinguard.cstr import stable_arx_order_study as arx_order_study
from twinguard.cstr import fit_discrepancy_model as discrepancy_fit
from twinguard.discrepancy.cstr_state_space import (
    StateSpaceParameters,
    fit_scalar_state_space_em_checkpoints,
    kalman_filter,
    predictive_rollout,
)


LEVELS=episode_design.COVERAGE_LEVELS
CATEGORIES=("interpolation","boundary","extrapolation")
ITERATION_LIMITS=(15,30,60,100)
SENSITIVITY_SEEDS=(91_001,91_002,91_003,91_004,91_005)
MAX_LAG=20


@dataclass(frozen=True)
class DiscrepancyFreezeConfig:
    source_dir:str="artifacts/cstr/episode_design"
    fit_dir:str="artifacts/cstr/discrepancy_fit"
    output_dir:str="artifacts/cstr/discrepancy_freeze"
    posterior_samples:int=2000
    bootstrap_samples:int=2000
    workers:int=5
    limit_calibration_episodes:int|None=None
    limit_test_episodes:int|None=None

    def validate(self)->None:
        if self.posterior_samples<=0 or self.bootstrap_samples<=0 or self.workers<=0:raise ValueError("sample and worker counts must be positive")
        if self.limit_test_episodes is None and self.posterior_samples!=2000:raise ValueError("full discrepancy freeze freezes M=2000")


@dataclass(frozen=True)
class EpisodeInputs:
    episode_id:str
    category:str
    random_seed:int
    residual_online:np.ndarray
    z_online:np.ndarray
    z_future:np.ndarray
    nominal_future:np.ndarray
    proposed_action:float
    truth_path:Path
    hidden_plant:object|None=None
    hidden_measurement_seed:int|None=None


@dataclass(frozen=True)
class VarianceModel:
    name:str
    global_scale:float
    feature_names:tuple[str,...]=()
    feature_mean:np.ndarray|None=None
    feature_scale:np.ndarray|None=None
    gamma:np.ndarray|None=None

    def scale(self,features:dict[str,np.ndarray],length:int)->np.ndarray:
        if self.name=="U0":return np.ones(length)
        if self.name=="U1":return np.full(length,self.global_scale)
        if self.gamma is None or self.feature_mean is None or self.feature_scale is None:raise RuntimeError("incomplete U2 variance model")
        x=np.column_stack([features[name] for name in self.feature_names]);xs=(x-self.feature_mean)/self.feature_scale
        return np.clip(np.exp(self.gamma[0]+xs@self.gamma[1:]),.05,20.0)


def _read_csv(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


def _write_csv(path:Path,rows:Sequence[dict[str,object]])->None:
    if not rows:raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def _parameter_from_payload(payload:dict)->StateSpaceParameters:
    return StateSpaceParameters(A=np.asarray(payload["A"],float),B=np.asarray(payload["B"],float),C=np.asarray(payload["C"],float),D=np.asarray(payload["D"],float),Q=np.asarray(payload["Q"],float),R=float(payload["R"]),initial_mean=np.asarray(payload["initial_mean"],float),initial_covariance=np.asarray(payload["initial_covariance"],float),z_mean=np.asarray(payload["z_mean"],float),z_scale=np.asarray(payload["z_scale"],float),log_likelihood=float(payload["log_likelihood"]),iterations=int(payload["iterations"]),converged=bool(payload["converged"]),initialization_seed=int(payload["initialization_seed"]),fit_seconds=float(payload["fit_seconds"]),final_parameter_change=float(payload.get("final_parameter_change",np.nan)))


def _parameter_payload(p:StateSpaceParameters)->dict[str,object]:
    result=discrepancy_fit._parameter_payload(p);result["final_parameter_change"]=p.final_parameter_change;return result


def load_original_model(fit_dir:Path)->StateSpaceParameters:
    root=json.loads((fit_dir/"model_parameters.json").read_text(encoding="utf-8"));return _parameter_from_payload(root["selected"])


def prepare_episode(row:dict[str,str],source:Path)->EpisodeInputs:
    pre,learning,proposed=arx_order_study._reconstruct_observations(row);previous=float(pre[-1]["action_lps"]);online=[]
    for index,item in enumerate(learning):
        action=float(item["action_lps"]);online.append([action,action-previous,float(item["nominal_twin_temperature_k"]),index+1]);previous=action
    truth_path=source/"evaluations"/f"{row['episode_id']}.npz"
    with np.load(truth_path) as saved:
        nominal=np.asarray(saved["nominal_twin_temperature_k"],float);actions=np.asarray(saved["action_sequence_lps"],float)
    if not np.allclose(actions,proposed):raise RuntimeError("held-out action mismatch")
    future=[]
    for h,(action,nom) in enumerate(zip(actions,nominal)):
        future.append([float(action),float(action)-previous,float(nom),51+h]);previous=float(action)
    return EpisodeInputs(row["episode_id"],row["action_category"],int(float(row["random_seed"])),np.asarray([x["temperature_residual_k"] for x in learning]),np.asarray(online),np.asarray(future),nominal,float(proposed),truth_path)


def generate_new_test_inputs(config:episode_design.EpisodeDesignConfig,template,output:Path,episodes_per_category:int=80,base_seed:int=730_000,episode_prefix:str="new_episode",truth_subdir:str="revealed_test_truth")->tuple[list[EpisodeInputs],list[dict[str,object]]]:
    """Create new observed histories while keeping every plant future latent."""
    episodes=[];manifest=[]
    for category_index,category in enumerate(CATEGORIES):
        for local_index in range(episodes_per_category):
            index=category_index*episodes_per_category+local_index;seed=base_seed+index;rng=np.random.default_rng(seed);measurement_rng=np.random.default_rng(seed+1_000_000);action_rng=np.random.default_rng(seed+2_000_000)
            onset=float(rng.uniform(config.fault_onset_min_s,config.fault_onset_max_s));magnitude=float(rng.uniform(config.fault_magnitude_min,config.fault_magnitude_max));nominal,plant,metadata=episode_design._configure_episode_models(template,config,rng,onset,magnitude);prior=2/60
            pre=[episode_design._paired_observation(nominal,plant,prior,measurement_rng,config.measurement_noise_k) for _ in range(config.pre_fault_lead_steps)];learning=[]
            for t in range(config.n_postfault):
                action=episode_design.TRAINING_ACTIONS_LPS[(t//config.training_action_block_steps)%len(episode_design.TRAINING_ACTIONS_LPS)];learning.append(episode_design._paired_observation(nominal,plant,action,measurement_rng,config.measurement_noise_k))
            proposed=episode_design._select_action_after_fit(category,action_rng);previous=float(pre[-1]["action_lps"]);online=[]
            for t,row in enumerate(learning):action=float(row["action_lps"]);online.append([action,action-previous,float(row["nominal_twin_temperature_k"]),t+1]);previous=action
            nominal_future=[];future=[]
            for h in range(config.horizon_steps):
                out=nominal.simulate_step(Fin_sp_normal=proposed,Fin_sp_startup=proposed);temp=float(out["T_true"]);nominal_future.append(temp);future.append([proposed,proposed-previous,temp,51+h]);previous=proposed
            episode_id=f"{episode_prefix}_{index:04d}_{category}";truth_path=output/truth_subdir/f"{episode_id}.npz"
            episodes.append(EpisodeInputs(episode_id,category,seed,np.asarray([x["temperature_residual_k"] for x in learning]),np.asarray(online),np.asarray(future),np.asarray(nominal_future),proposed,truth_path,plant,seed+4_000_000))
            manifest.append({"episode_id":episode_id,"action_category":category,"random_seed":seed,"hidden_fault_onset_s_evaluator_only":onset,"hidden_fault_magnitude_evaluator_only":magnitude,**{f"{k}_evaluator_only":v for k,v in metadata.items()},"truth_generated_after_prediction":True})
    return episodes,manifest


def _analytical_mean(p:StateSpaceParameters,m:np.ndarray,z_future:np.ndarray)->np.ndarray:
    state=np.asarray(m,float).copy();z=p.standardize(z_future);out=[]
    for item in z:state=p.A@state+p.B@item;out.append(float((p.C@state+p.D@item).item()))
    return np.asarray(out)


def evaluate_selection_fit(p:StateSpaceParameters,episodes:Sequence[EpisodeInputs])->dict[str,float]:
    errors=[];innovations=[];loglik=0.0
    for item in episodes:
        filtered=kalman_filter(p,item.residual_online,item.z_online);prediction=item.nominal_future+_analytical_mean(p,filtered.filtered_mean[-1],item.z_future)
        with np.load(item.truth_path) as saved:actual=np.asarray(saved["actual_plant_temperature_k"],float)
        errors.append(float(np.sqrt(np.mean((prediction-actual)**2))));innovations.append(filtered.innovations);loglik+=filtered.log_likelihood
    acf=_acf(innovations,10);values=np.concatenate(innovations)
    return {"selection_log_likelihood":float(loglik),"selection_recursive_RMSE_k":float(np.median(errors)),"selection_innovation_bias_k":float(np.mean(values)),"selection_maximum_innovation_ACF":float(np.max(np.abs(acf[1:])))}


def _sensitivity_task(task:tuple[np.ndarray,np.ndarray,int])->dict[int,StateSpaceParameters]:
    y,z,seed=task;return fit_scalar_state_space_em_checkpoints(y,z,seed,ITERATION_LIMITS)


def run_em_sensitivity(training_path:Path,selection:Sequence[EpisodeInputs],workers:int)->tuple[list[dict[str,object]],list[StateSpaceParameters],str,dict[str,object]]:
    with np.load(training_path) as saved:y=np.asarray(saved["residual"],float);z=np.asarray(saved["z"],float)
    tasks=[(y,z,seed) for seed in SENSITIVITY_SEEDS]
    if workers==1:
        checkpoints=[_sensitivity_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks))) as pool:checkpoints=list(pool.map(_sensitivity_task,tasks))
    fits=[checkpoints[s][limit] for s in range(len(SENSITIVITY_SEEDS)) for limit in ITERATION_LIMITS];rows=[]
    for p in fits:
        metrics=evaluate_selection_fit(p,selection);rows.append({"iteration_limit":p.iterations,"initialization_seed":p.initialization_seed,"train_log_likelihood":p.log_likelihood,**metrics,"EM_iterations_completed":p.iterations,"converged":p.converged,"final_parameter_change":p.final_parameter_change,"A":json.dumps(p.A.tolist()),"B":json.dumps(p.B.tolist()),"C":json.dumps(p.C.tolist()),"D":json.dumps(p.D.tolist()),"Q":json.dumps(p.Q.tolist()),"R":p.R,"spectral_radius":p.spectral_radius,"stability_margin":1-p.spectral_radius,"fit_seconds_cumulative":p.fit_seconds})
    best=max(fits,key=lambda x:x.log_likelihood);ll_cut=best.log_likelihood-max(5.0,.005*abs(best.log_likelihood));quality=[p for p in fits if p.converged and p.log_likelihood>=ll_cut]
    if not quality:quality=[p for p in fits if p.log_likelihood>=ll_cut]
    rhos=np.asarray([p.spectral_radius for p in quality]);quality_rows=[next(r for r in rows if r["initialization_seed"]==p.initialization_seed and r["iteration_limit"]==p.iterations) for p in quality];rmses=np.asarray([r["selection_recursive_RMSE_k"] for r in quality_rows])
    robust=len(quality)>=3 and np.ptp(rhos)<=.02 and np.std(rmses)/max(np.mean(rmses),1e-12)<=.10
    classification="ROBUST LATENT DYNAMICS" if robust else "OPTIMIZATION SENSITIVE"
    robustness={"classification":classification,"number_of_high_quality_fits":len(quality),"likelihood_threshold":ll_cut,"spectral_radius_mean":float(np.mean(rhos)),"spectral_radius_sd":float(np.std(rhos)),"spectral_radius_min":float(np.min(rhos)),"spectral_radius_max":float(np.max(rhos)),"selection_RMSE_mean_k":float(np.mean(rmses)),"selection_RMSE_sd_k":float(np.std(rmses))}
    return rows,fits,classification,robustness


def choose_frozen_model(original:StateSpaceParameters,fits:Sequence[StateSpaceParameters],rows:Sequence[dict[str,object]],classification:str)->tuple[StateSpaceParameters,str,bool]:
    original_metrics=evaluate_selection_fit(original,_SELECTION_CACHE)
    best=max(fits,key=lambda p:p.log_likelihood);best_row=next(r for r in rows if r["initialization_seed"]==best.initialization_seed and r["iteration_limit"]==best.iterations)
    ll_gain=best.log_likelihood-original.log_likelihood;rmse_gain=1-float(best_row["selection_recursive_RMSE_k"])/original_metrics["selection_recursive_RMSE_k"]
    replace=classification=="ROBUST LATENT DYNAMICS" and best.converged and ll_gain>max(10.,.001*abs(original.log_likelihood)) and rmse_gain>.05
    if replace:return best,"PREVIOUS FIT NOT CONVERGED ENOUGH",True
    return original,classification,False


def _acf(arrays:Sequence[np.ndarray],max_lag:int=MAX_LAG)->np.ndarray:
    values=np.concatenate(arrays);center=float(np.mean(values));den=float(np.sum((values-center)**2));answer=np.ones(max_lag+1)
    if den<=1e-15:return np.r_[1.,np.zeros(max_lag)]
    for lag in range(1,max_lag+1):answer[lag]=sum(float(np.sum((x[:-lag]-center)*(x[lag:]-center))) for x in arrays)/den
    return answer


def innovation_data(p:StateSpaceParameters,episodes:Sequence[EpisodeInputs])->tuple[list[dict[str,object]],dict[str,np.ndarray]]:
    records=[];arrays={key:[] for key in ("innovation","standardized","state_abs","state_variance","input","delta_abs","elapsed","nominal","support")};by_category={c:[] for c in CATEGORIES};std_by_category={c:[] for c in CATEGORIES}
    squared_by_category={c:[] for c in CATEGORIES}
    for item in episodes:
        with np.load(item.truth_path) as saved:future_residual=np.asarray(saved["actual_residual_k"],float)
        all_residual=np.r_[item.residual_online,future_residual];all_z=np.vstack((item.z_online,item.z_future));whole=kalman_filter(p,all_residual,all_z);sl=slice(item.residual_online.size,None);values=whole.innovations[sl];standardized=whole.standardized_innovations[sl]
        by_category[item.category].append(values);std_by_category[item.category].append(standardized);squared_by_category[item.category].append(standardized**2)
        arrays["innovation"].append(values);arrays["standardized"].append(standardized);arrays["state_abs"].append(np.abs(whole.filtered_mean[sl,0]));arrays["state_variance"].append(whole.filtered_covariance[sl,0,0]);arrays["input"].append(item.z_future[:,0]);arrays["delta_abs"].append(np.abs(item.z_future[:,1]));arrays["elapsed"].append(item.z_future[:,3]);arrays["nominal"].append(item.z_future[:,2]);arrays["support"].append(np.full(item.z_future.shape[0],episode_design.action_support_distance(item.proposed_action)))
    for key in arrays:arrays[key]=np.concatenate(arrays[key])
    for category in (*CATEGORIES,"pooled"):
        source=[x for c in CATEGORIES for x in by_category[c]] if category=="pooled" else by_category[category];values=np.concatenate(source);std=arrays["standardized"] if category=="pooled" else np.concatenate(std_by_category[category]);acf=_acf(source);sq_source=squared_by_category[category] if category!="pooled" else [x for c in CATEGORIES for x in squared_by_category[c]];sqacf=_acf(sq_source);n=values.size;lb=float(n*(n+2)*sum(acf[i]**2/(n-i) for i in range(1,MAX_LAG+1)))
        row={"action_category":category,"number_of_innovations":n,"innovation_mean_k":float(np.mean(values)),"innovation_median_k":float(np.median(values)),"innovation_variance_k2":float(np.var(values,ddof=1)),"standardized_mean":float(np.mean(std)),"standardized_variance":float(np.var(std,ddof=1)),"standardized_skewness":float(skew(std,bias=False)),"standardized_excess_kurtosis":float(kurtosis(std,bias=False)),"standardized_abs_gt_1_96_rate":float(np.mean(np.abs(std)>1.96)),"standardized_abs_gt_2_576_rate":float(np.mean(np.abs(std)>2.576)),"maximum_absolute_ACF_lags_1_20":float(np.max(np.abs(acf[1:]))),"maximum_absolute_squared_ACF_lags_1_20":float(np.max(np.abs(sqacf[1:]))),"Ljung_Box_Q_lags_20":lb,"Ljung_Box_p_value_lags_20":float(chi2.sf(lb,MAX_LAG))}
        row.update({f"ACF_lag_{i}":float(acf[i]) for i in range(1,MAX_LAG+1)});row.update({f"squared_ACF_lag_{i}":float(sqacf[i]) for i in range(1,MAX_LAG+1)});records.append(row)
    return records,arrays


def variance_diagnostics(arrays:dict[str,np.ndarray],episodes:Sequence[EpisodeInputs],p:StateSpaceParameters)->tuple[list[dict[str,object]],dict[str,dict[str,float]]]:
    target=arrays["standardized"]**2;rows=[];summary={}
    labels={"state_abs":"absolute_filtered_latent_state","state_variance":"posterior_latent_state_variance","input":"input_u_t","delta_abs":"absolute_delta_u_t","elapsed":"elapsed_time_tau","nominal":"nominal_DT_output","support":"action_support_distance"}
    for key,label in labels.items():
        x=arrays[key];rho,pvalue=spearmanr(x,target);rho=0.0 if not np.isfinite(rho) else float(rho);pvalue=1.0 if not np.isfinite(pvalue) else float(pvalue);groups=np.array_split(np.argsort(x,kind="stable"),5);variances=[]
        for index,indices in enumerate(groups):
            value=float(np.mean(target[indices]));variances.append(value);rows.append({"diagnostic_type":"binned_continuous","predictor":label,"bin_or_category":index+1,"sample_count":int(indices.size),"predictor_mean":float(np.mean(x[indices])),"empirical_standardized_innovation_variance":value,"Spearman_rho":rho,"Spearman_p_value":pvalue,"variance_ratio_max_to_min":""})
        ratio=float(max(variances)/max(min(variances),1e-12));summary[key]={"rho":float(rho),"pvalue":float(pvalue),"variance_ratio":ratio};
        for row in rows[-5:]:row["variance_ratio_max_to_min"]=ratio
    offset=0;category_values=[]
    for item in episodes:
        count=item.z_future.shape[0];category_values.extend([item.category]*count);offset+=count
    category_values=np.asarray(category_values)
    variances=[]
    for category in CATEGORIES:
        mask=category_values==category;value=float(np.mean(target[mask]));variances.append(value);rows.append({"diagnostic_type":"categorical","predictor":"action_category","bin_or_category":category,"sample_count":int(mask.sum()),"predictor_mean":"","empirical_standardized_innovation_variance":value,"Spearman_rho":"","Spearman_p_value":"","variance_ratio_max_to_min":""})
    ratio=float(max(variances)/max(min(variances),1e-12));
    for row in rows[-3:]:row["variance_ratio_max_to_min"]=ratio
    return rows,summary


def fit_variance_models(arrays:dict[str,np.ndarray],diagnostics:dict[str,dict[str,float]])->tuple[list[VarianceModel],dict[str,object]]:
    target=np.maximum(arrays["standardized"]**2,1e-8);global_scale=float(np.mean(target));models=[VarianceModel("U0",1.0),VarianceModel("U1",global_scale)]
    supported=[name for name in ("state_abs","delta_abs","support") if abs(diagnostics[name]["rho"])>=.10 and diagnostics[name]["variance_ratio"]>=1.30]
    metadata={"U1_global_scale":global_scale,"U2_implemented":bool(supported),"U2_support_rule":"|Spearman rho| >= 0.10 and max/min binned variance ratio >= 1.30","U2_features":supported}
    if supported:
        x=np.column_stack([arrays[name] for name in supported]);mean=np.mean(x,axis=0);scale=np.std(x,axis=0);scale=np.where(scale>1e-12,scale,1.);xs=(x-mean)/scale
        def objective(gamma):
            log_scale=np.clip(gamma[0]+xs@gamma[1:],-3,3);return float(np.mean(log_scale+target/np.exp(log_scale))+.002*np.sum(gamma[1:]**2))
        result=minimize(objective,np.r_[np.log(global_scale),np.zeros(len(supported))],method="L-BFGS-B");models.append(VarianceModel("U2",global_scale,tuple(supported),mean,scale,np.asarray(result.x)));metadata.update({"U2_gamma":result.x.tolist(),"U2_optimizer_success":bool(result.success),"U2_objective":float(result.fun)})
    return models,metadata


def _future_variance_features(p:StateSpaceParameters,filtered_mean:np.ndarray,z_future:np.ndarray,proposed:float)->dict[str,np.ndarray]:
    state=filtered_mean.copy();z=p.standardize(z_future);states=[]
    for item in z:state=p.A@state+p.B@item;states.append(abs(float(state[0])))
    return {"state_abs":np.asarray(states),"delta_abs":np.abs(z_future[:,1]),"support":np.full(z_future.shape[0],episode_design.action_support_distance(proposed))}


def _hash(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_test_episode(p:StateSpaceParameters,item:EpisodeInputs,models:Sequence[VarianceModel],output:Path,samples:int)->tuple[list[dict[str,object]],dict[str,object]]:
    start=time.perf_counter();filtered=kalman_filter(p,item.residual_online,item.z_online);filter_seconds=time.perf_counter()-start
    start=time.perf_counter();base=predictive_rollout(p,filtered.filtered_mean[-1],filtered.filtered_covariance[-1],item.z_future,samples,np.random.default_rng(item.random_seed+110_000_000));rollout_seconds=time.perf_counter()-start
    mean=item.nominal_future+base["analytical_mean"];features=_future_variance_features(p,filtered.filtered_mean[-1],item.z_future,item.proposed_action);payload={"nominal_temperature_k":item.nominal_future,"posterior_mean_temperature_k":mean};model_cache={};variance_seconds=0.0
    for model in models:
        tick=time.perf_counter();scale=model.scale(features,len(mean));paths=mean[None,:]+np.sqrt(scale)[None,:]*(item.nominal_future[None,:]+base["residual_samples"]-mean[None,:]);variance_seconds+=time.perf_counter()-tick;point={};sim=episode_design.construct_simultaneous_bands(paths)
        for level in LEVELS:
            tail=(1-level)/2;lower,upper=np.quantile(paths,[tail,1-tail],axis=0);label=int(level*100);payload[f"{model.name}_pointwise_{label}_lower_k"]=lower;payload[f"{model.name}_pointwise_{label}_upper_k"]=upper;payload[f"{model.name}_simultaneous_{label}_lower_k"]=sim.lower[level];payload[f"{model.name}_simultaneous_{label}_upper_k"]=sim.upper[level]
        model_cache[model.name]=(paths,sim)
    pred_path=output/"predictions"/f"{item.episode_id}.npz";pred_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pred_path,**payload);before=_hash(pred_path)
    if item.truth_path.exists():
        with np.load(item.truth_path) as revealed:actual=np.asarray(revealed["actual_plant_temperature_k"],float)
    else:
        if item.hidden_plant is None or item.hidden_measurement_seed is None:raise RuntimeError("missing hidden plant for prospective reveal")
        reveal_rng=np.random.default_rng(item.hidden_measurement_seed);actual=[]
        for _ in range(item.nominal_future.size):
            out=item.hidden_plant.simulate_step(Fin_sp_normal=item.proposed_action,Fin_sp_startup=item.proposed_action);actual.append(float(out["T_true"])+float(reveal_rng.normal(0,episode_design.EpisodeDesignConfig().measurement_noise_k)))
        actual=np.asarray(actual);item.truth_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(item.truth_path,actual_plant_temperature_k=actual,actual_residual_k=actual-item.nominal_future)
    if _hash(pred_path)!=before:raise AssertionError("prediction changed after truth reveal")
    rows=[]
    for model in models:
        paths,sim=model_cache[model.name];posterior_mean=np.mean(paths,axis=0);rmse=float(np.sqrt(np.mean((posterior_mean-actual)**2)))
        for level in LEVELS:
            label=int(level*100);lower=payload[f"{model.name}_pointwise_{label}_lower_k"];upper=payload[f"{model.name}_pointwise_{label}_upper_k"];sl=sim.lower[level];su=sim.upper[level]
            rows.append({"episode_id":item.episode_id,"action_category":item.category,"random_seed":item.random_seed,"uncertainty_model":model.name,"nominal_level":level,"pointwise_coverage":float(np.mean((actual>=lower)&(actual<=upper))),"pointwise_average_width_k":float(np.mean(upper-lower)),"pointwise_median_width_k":float(np.median(upper-lower)),"simultaneous_coverage":int(np.all((actual>=sl)&(actual<=su))),"simultaneous_average_width_k":float(np.mean(su-sl)),"rollout_RMSE_k":rmse,"prediction_saved_before_truth_reveal":True})
    runtime={"episode_id":item.episode_id,"action_category":item.category,"filtering_seconds":filter_seconds,"variance_model_evaluation_seconds":variance_seconds,"rollout_2000_paths_seconds":rollout_seconds,"total_online_latency_seconds":filter_seconds+variance_seconds+rollout_seconds}
    return rows,runtime


def _bootstrap_ci(values:np.ndarray,rng:np.random.Generator,reps:int)->tuple[float,float]:
    n=values.size;draw=np.empty(reps)
    for i in range(reps):draw[i]=np.mean(values[rng.integers(0,n,n)])
    return tuple(float(x) for x in np.quantile(draw,[.025,.975]))


def summarize_calibration(rows:Sequence[dict[str,object]],bootstrap:int)->tuple[list[dict[str,object]],list[dict[str,object]]]:
    summary=[];comparison=[];model_names=tuple(dict.fromkeys(r["uncertainty_model"] for r in rows))
    for model in model_names:
        for category in (*CATEGORIES,"pooled"):
            source=[r for r in rows if r["uncertainty_model"]==model and (category=="pooled" or r["action_category"]==category)]
            if not source:continue
            level_rows=[]
            for level in LEVELS:
                chosen=[r for r in source if r["nominal_level"]==level];point=np.asarray([r["pointwise_coverage"] for r in chosen]);sim=np.asarray([r["simultaneous_coverage"] for r in chosen]);rng=np.random.default_rng(120_000+int(level*1000)+sum(map(ord,model+category)));pci=_bootstrap_ci(point,rng,bootstrap);sci=_bootstrap_ci(sim,rng,bootstrap);pc=float(np.mean(point));sc=float(np.mean(sim))
                row={"uncertainty_model":model,"action_category":category,"nominal_level":level,"number_of_episodes":len(chosen),"pointwise_empirical_coverage":pc,"pointwise_CI_lower":pci[0],"pointwise_CI_upper":pci[1],"simultaneous_empirical_coverage":sc,"simultaneous_CI_lower":sci[0],"simultaneous_CI_upper":sci[1],"pointwise_calibration_error":abs(pc-level),"simultaneous_calibration_error":abs(sc-level),"average_predictive_width_k":float(np.mean([r["pointwise_average_width_k"] for r in chosen])),"median_predictive_width_k":float(np.median([r["pointwise_median_width_k"] for r in chosen])),"simultaneous_average_width_k":float(np.mean([r["simultaneous_average_width_k"] for r in chosen]))};summary.append(row);level_rows.append(row)
            comparison.append({"uncertainty_model":model,"action_category":category,"number_of_episodes":len(source)//len(LEVELS),"pointwise_MACE":float(np.mean([r["pointwise_calibration_error"] for r in level_rows])),"simultaneous_MACE":float(np.mean([r["simultaneous_calibration_error"] for r in level_rows])),"average_predictive_width_k":float(np.mean([r["average_predictive_width_k"] for r in level_rows])),"simultaneous_average_width_k":float(np.mean([r["simultaneous_average_width_k"] for r in level_rows])),"median_rollout_RMSE_k":float(np.median([r["rollout_RMSE_k"] for r in source]))})
    return summary,comparison


def calibration_classification(summary:Sequence[dict[str,object]],comparison:Sequence[dict[str,object]])->tuple[str,str]:
    models=tuple(dict.fromkeys(r["uncertainty_model"] for r in summary));acceptable=[]
    for model in models:
        pooled=[r for r in summary if r["uncertainty_model"]==model and r["action_category"]=="pooled"];regime=[r for r in summary if r["uncertainty_model"]==model and r["action_category"]!="pooled"]
        pm=np.mean([r["pointwise_calibration_error"] for r in pooled]);sm=np.mean([r["simultaneous_calibration_error"] for r in pooled]);major=any((r["nominal_level"]>=.9 and r["pointwise_empirical_coverage"]<r["nominal_level"]-.10) or (r["nominal_level"]>=.9 and r["simultaneous_empirical_coverage"]<r["nominal_level"]-.15) for r in regime)
        if pm<=.08 and sm<=.10 and not major:acceptable.append(model)
    if acceptable:
        selected=min(acceptable,key=lambda model:next(r["average_predictive_width_k"] for r in comparison if r["uncertainty_model"]==model and r["action_category"]=="pooled"));return selected,"CALIBRATED"
    pooled_comparison=[r for r in comparison if r["action_category"]=="pooled"];selected=min(pooled_comparison,key=lambda r:r["pointwise_MACE"]+r["simultaneous_MACE"])["uncertainty_model"];selected_rows=[r for r in summary if r["uncertainty_model"]==selected and r["action_category"]=="pooled"]
    direction=np.mean([r["pointwise_empirical_coverage"]-r["nominal_level"] for r in selected_rows]);return selected,("UNDER-CONSERVATIVE" if direction<-.03 else "OVER-CONSERVATIVE" if direction>.03 else "MIXED")


def create_figures(output:Path,sensitivity,robustness,innovation,variance,summary,episode_rows:Sequence[dict[str,object]],episodes:Sequence[EpisodeInputs],selected_model:str)->None:
    figures=output/"figures";figures.mkdir(parents=True,exist_ok=True);plt.rcParams.update({"font.size":9,"figure.dpi":150})
    def save(fig,name):fig.tight_layout();fig.savefig(figures/f"{name}.png",dpi=300);fig.savefig(figures/f"{name}.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));
    for seed in SENSITIVITY_SEEDS:
        rows=[r for r in sensitivity if r["initialization_seed"]==seed];axes[0].plot([r["iteration_limit"] for r in rows],[r["train_log_likelihood"] for r in rows],"o-",label=str(seed));axes[1].plot([r["iteration_limit"] for r in rows],[r["final_parameter_change"] for r in rows],"o-")
    axes[0].set(xlabel="EM iteration limit",ylabel="Training log likelihood");axes[0].legend(title="seed",fontsize=7);axes[1].set(xlabel="EM iteration limit",ylabel="Relative final parameter change",yscale="log");save(fig,"figure_DISCREPANCY_FREEZE_1_EM_convergence")
    fig,ax=plt.subplots(figsize=(6.5,4));conv=[r for r in sensitivity if str(r["converged"]).lower()=="true"];source=conv or sensitivity;ax.scatter([r["iteration_limit"] for r in source],[r["spectral_radius"] for r in source],c=[r["selection_recursive_RMSE_k"] for r in source],cmap="viridis");ax.axhline(.995,ls="--",color="black",label="parameterization ceiling");ax.set(xlabel="EM iteration limit",ylabel="Spectral radius",title=f"{robustness['classification']}");ax.legend();save(fig,"figure_DISCREPANCY_FREEZE_2_spectral_radius")
    pooled=next(r for r in innovation if r["action_category"]=="pooled");fig,ax=plt.subplots(figsize=(7,4));lags=np.arange(1,21);ax.plot(lags,[pooled[f"ACF_lag_{i}"] for i in lags],"o-",label="innovation");ax.plot(lags,[pooled[f"squared_ACF_lag_{i}"] for i in lags],"s-",label="squared innovation");ax.axhline(0,color="black",lw=.7);ax.set(xlabel="Lag",ylabel="ACF");ax.legend();save(fig,"figure_DISCREPANCY_FREEZE_3_innovation_ACF")
    fig,axes=plt.subplots(2,2,figsize=(10,7));names=("absolute_filtered_latent_state","absolute_delta_u_t","action_support_distance","posterior_latent_state_variance")
    for ax,name in zip(axes.ravel(),names):
        rows=[r for r in variance if r["predictor"]==name and r["diagnostic_type"]=="binned_continuous"];ax.plot([r["predictor_mean"] for r in rows],[r["empirical_standardized_innovation_variance"] for r in rows],"o-");ax.set(xlabel=name.replace("_"," "),ylabel="Empirical variance");ax.grid(alpha=.25)
    save(fig,"figure_DISCREPANCY_FREEZE_4_conditional_variance")
    for number,metric,title in ((5,"pointwise_empirical_coverage","Pointwise calibration"),(6,"simultaneous_empirical_coverage","Simultaneous trajectory calibration")):
        fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharex=True,sharey=True)
        for ax,category in zip(axes,CATEGORIES):
            for model in tuple(dict.fromkeys(r["uncertainty_model"] for r in summary)):
                rows=[r for r in summary if r["action_category"]==category and r["uncertainty_model"]==model];values=np.asarray([r[metric] for r in rows]);lower=np.asarray([r["pointwise_CI_lower" if number==5 else "simultaneous_CI_lower"] for r in rows]);upper=np.asarray([r["pointwise_CI_upper" if number==5 else "simultaneous_CI_upper"] for r in rows]);ax.errorbar([r["nominal_level"] for r in rows],values,yerr=np.vstack((values-lower,upper-values)),fmt="o-",capsize=2,label=model)
            ax.plot([.45,1],[.45,1],"k--",lw=.8);ax.set_title(category);ax.set_xlabel("Nominal coverage");ax.grid(alpha=.25)
        axes[0].set_ylabel("Empirical coverage");axes[-1].legend();fig.suptitle(title);save(fig,f"figure_DISCREPANCY_FREEZE_{number}_{'pointwise' if number==5 else 'simultaneous'}_calibration")
    fig,ax=plt.subplots(figsize=(6.5,4));pooled_rows=[r for r in summary if r["action_category"]=="pooled"]
    for model in tuple(dict.fromkeys(r["uncertainty_model"] for r in pooled_rows)):
        rows=[r for r in pooled_rows if r["uncertainty_model"]==model];ax.plot([r["average_predictive_width_k"] for r in rows],[r["pointwise_empirical_coverage"] for r in rows],"o-",label=model)
    ax.set(xlabel="Average predictive width (K)",ylabel="Pointwise empirical coverage");ax.legend();ax.grid(alpha=.25);save(fig,"figure_DISCREPANCY_FREEZE_7_coverage_vs_width")
    for number,category in enumerate(CATEGORIES,start=8):
        candidates=[r for r in episode_rows if r["uncertainty_model"]==selected_model and r["nominal_level"]==.95 and r["action_category"]==category];chosen=min(candidates,key=lambda r:abs(r["rollout_RMSE_k"]-np.median([x["rollout_RMSE_k"] for x in candidates])));item=next(x for x in episodes if x.episode_id==chosen["episode_id"])
        with np.load(output/"predictions"/f"{item.episode_id}.npz") as pred, np.load(item.truth_path) as truth:
            mean=np.asarray(pred["posterior_mean_temperature_k"]);lo=np.asarray(pred[f"{selected_model}_pointwise_95_lower_k"]);hi=np.asarray(pred[f"{selected_model}_pointwise_95_upper_k"]);actual=np.asarray(truth["actual_plant_temperature_k"])
        fig,ax=plt.subplots(figsize=(7,4));h=np.arange(1,len(mean)+1);ax.fill_between(h,lo,hi,alpha=.25,label=f"{selected_model} 95% pointwise");ax.plot(h,item.nominal_future,"--",label="nominal DT");ax.plot(h,mean,label="predictive mean");ax.plot(h,actual,color="black",label="revealed plant");ax.set(xlabel="Horizon step",ylabel="Temperature (K)",title=f"{category}: {item.episode_id}");ax.legend();save(fig,f"figure_DISCREPANCY_FREEZE_{number}_{category}_rollout")


def create_reports(output:Path,robustness:dict[str,object],frozen_identity:str,replaced:bool,variance_metadata:dict[str,object],comparison,summary,innovation,runtime,calibration_verdict:str,selected_uncertainty:str,final_decision:str)->None:
    pooled=[r for r in comparison if r["action_category"]=="pooled"];selected=next(r for r in pooled if r["uncertainty_model"]==selected_uncertainty);innov=next(r for r in innovation if r["action_category"]=="pooled");latency=next(r for r in runtime if r["action_category"]=="pooled")
    selection_wording="preferred calibrated model" if calibration_verdict=="CALIBRATED" else "best tested comparator (none passed calibration)"
    text=f"""# discrepancy freeze — robustness and uncertainty calibration

## Central result

The training-only robustness classification is **{robustness['classification']}**. The frozen mean identity is **{frozen_identity}**. `replaced_after_DISCREPANCY_FIT={replaced}`; {'a new 240-episode final population with new seeds was generated and revealed prospectively' if replaced else 'the original discrepancy fit candidate remains frozen, so the existing 240 uncertainty-unseen final episodes are retained as the discrepancy freeze test population'}.

Across high-quality fits, spectral radius ranged from {robustness['spectral_radius_min']:.6f} to {robustness['spectral_radius_max']:.6f} (SD {robustness['spectral_radius_sd']:.6f}); selection RMSE SD was {robustness['selection_RMSE_sd_k']:.4f} K. Fit selection used no discrepancy fit final-assessment episode.

## Variance diagnosis

U0 is the frozen homoskedastic model. U1 applies the calibration-only global variance multiplier {variance_metadata['U1_global_scale']:.4f}. U2 implemented: **{variance_metadata['U2_implemented']}**, with calibration-supported features `{variance_metadata['U2_features']}`. No latent mean parameter was altered.

The pooled standardized innovation variance is {innov['standardized_variance']:.3f}; max innovation ACF is {innov['maximum_absolute_ACF_lags_1_20']:.3f}, max squared-innovation ACF is {innov['maximum_absolute_squared_ACF_lags_1_20']:.3f}, and Ljung–Box p={innov['Ljung_Box_p_value_lags_20']:.3g}. Statistical rejection is interpreted with these effect sizes and the prospective calibration result.

## Prospective calibration and sharpness

The uncertainty verdict is **{calibration_verdict}** and the {selection_wording} is **{selected_uncertainty}**. Its pooled pointwise MACE is {selected['pointwise_MACE']:.3f}, simultaneous MACE is {selected['simultaneous_MACE']:.3f}, average pointwise width is {selected['average_predictive_width_k']:.3f} K, simultaneous width is {selected['simultaneous_average_width_k']:.3f} K, and median rollout RMSE is {selected['median_rollout_RMSE_k']:.3f} K. Width alone never qualifies a model.

Calibration data comprise 60 model-selection episodes (20/category); uncertainty testing uses 240 episodes (80/category). Each complete prediction snapshot was written without truth and hashed before the corresponding hidden trajectory was read.

## Runtime

Median N=50 filtering time is {latency['median_filtering_seconds']:.6f} s, variance evaluation is {latency['median_variance_model_evaluation_seconds']:.6f} s, 2,000-path propagation is {latency['median_rollout_2000_paths_seconds']:.4f} s, and total online latency is {latency['median_total_online_latency_seconds']:.4f} s.

## Final decision

**{final_decision}**
"""
    (output/"README.md").write_text(text,encoding="utf-8")
    (output/"decision_summary.md").write_text(f"# discrepancy freeze decision\n\n**{final_decision}**\n\nMean robustness: **{robustness['classification']}**. Uncertainty calibration: **{calibration_verdict}** using **{selected_uncertainty}**. Pooled pointwise/simultaneous MACE are {selected['pointwise_MACE']:.3f}/{selected['simultaneous_MACE']:.3f}; widths are {selected['average_predictive_width_k']:.3f}/{selected['simultaneous_average_width_k']:.3f} K. Remaining innovation dependence is not used by itself to demand another temporal state.\n",encoding="utf-8")


_SELECTION_CACHE:Sequence[EpisodeInputs]=()


def run(config:DiscrepancyFreezeConfig)->str:
    global _SELECTION_CACHE
    config.validate();source=Path(config.source_dir);b9dir=Path(config.fit_dir);output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    b5cfg=episode_design.EpisodeDesignConfig(output_dir=str(source));template=episode_design.build_template(b5cfg);arx_order_study._init_worker(template,b5cfg,str(source),config.posterior_samples)
    rows=discrepancy_fit._selection_roles(recursive_diagnostics.load_episode_design_episodes(source/"calibration_episodes.csv"));cal_rows=[r for r in rows if r["split_role"]=="model_selection"];test_rows=[r for r in rows if r["split_role"]=="final_assessment"]
    if config.limit_calibration_episodes is not None:cal_rows=[r for c in CATEGORIES for r in [x for x in cal_rows if x["action_category"]==c][:config.limit_calibration_episodes]]
    if config.limit_test_episodes is not None:test_rows=[r for c in CATEGORIES for r in [x for x in test_rows if x["action_category"]==c][:config.limit_test_episodes]]
    calibration=[prepare_episode(r,source) for r in cal_rows];_SELECTION_CACHE=calibration
    sensitivity,fits,classification,robustness=run_em_sensitivity(b9dir/"training_population.npz",calibration,config.workers);original=load_original_model(b9dir);frozen,classification,replaced=choose_frozen_model(original,fits,sensitivity,classification);robustness["classification"]=classification
    _write_csv(output/"em_sensitivity.csv",sensitivity);_write_csv(output/"latent_parameter_robustness.csv",[robustness])
    frozen_identity="original_DISCREPANCY_FIT_seed_91001_limit_15" if not replaced else f"B9S_seed_{frozen.initialization_seed}_limit_{frozen.iterations}"
    (output/"frozen_latent_model.json").write_text(json.dumps({"identity":frozen_identity,"original_DISCREPANCY_FIT_candidate_retained":not replaced,"selection_and_training_only_choice":True,"new_untouched_final_population_required":replaced,"parameters":_parameter_payload(frozen)},indent=2),encoding="utf-8")
    innovation,arrays=innovation_data(frozen,calibration);variance,diagnostic_summary=variance_diagnostics(arrays,calibration,frozen);models,variance_metadata=fit_variance_models(arrays,diagnostic_summary);_write_csv(output/"innovation_diagnostics.csv",innovation);_write_csv(output/"variance_diagnostics.csv",variance)
    # A replacement is never scored on the old final population.
    if replaced:
        per_category=config.limit_test_episodes if config.limit_test_episodes is not None else 80;tests,manifest=generate_new_test_inputs(b5cfg,template,output,per_category);_write_csv(output/"new_test_population_manifest.csv",manifest)
    else:tests=[prepare_episode(r,source) for r in test_rows]
    episode_rows=[];runtime_rows=[]
    for index,item in enumerate(tests):
        result,runtime=evaluate_test_episode(frozen,item,models,output,config.posterior_samples);episode_rows.extend(result);runtime_rows.append(runtime)
        if (index+1)%40==0:print(f"completed {index+1}/{len(tests)} discrepancy freeze test episodes",flush=True)
    summary,comparison=summarize_calibration(episode_rows,config.bootstrap_samples);selected_uncertainty,calibration_verdict=calibration_classification(summary,comparison)
    runtime_summary=[]
    for category in (*CATEGORIES,"pooled"):
        chosen=runtime_rows if category=="pooled" else [r for r in runtime_rows if r["action_category"]==category];runtime_summary.append({"action_category":category,"number_of_episodes":len(chosen),"median_filtering_seconds":float(np.median([r["filtering_seconds"] for r in chosen])),"median_variance_model_evaluation_seconds":float(np.median([r["variance_model_evaluation_seconds"] for r in chosen])),"median_rollout_2000_paths_seconds":float(np.median([r["rollout_2000_paths_seconds"] for r in chosen])),"median_total_online_latency_seconds":float(np.median([r["total_online_latency_seconds"] for r in chosen]))})
    _write_csv(output/"calibration_episode_results.csv",episode_rows);_write_csv(output/"calibration_summary.csv",summary);_write_csv(output/"uncertainty_model_comparison.csv",comparison);_write_csv(output/"runtime_summary.csv",runtime_summary)
    median_mean_rmse=float(np.median([r["rollout_RMSE_k"] for r in episode_rows if r["uncertainty_model"]=="U0" and r["nominal_level"]==.95]));mean_invalid=classification=="OPTIMIZATION SENSITIVE" or median_mean_rmse>2.065
    final="LATENT MEAN MODEL NEEDS REVISION" if mean_invalid else "READY FOR SAFETY PROBABILITY" if calibration_verdict=="CALIBRATED" else "UNCERTAINTY MODEL NEEDS REVISION"
    metadata={"config":asdict(config),"variance_model":variance_metadata,"calibration_verdict":calibration_verdict,"selected_uncertainty_model":selected_uncertainty,"final_decision":final};(output/"configuration.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    create_figures(output,sensitivity,robustness,innovation,variance,summary,episode_rows,tests,selected_uncertainty);create_reports(output,robustness,frozen_identity,replaced,variance_metadata,comparison,summary,innovation,runtime_summary,calibration_verdict,selected_uncertainty,final);return final


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-dir",default="artifacts/cstr/episode_design");p.add_argument("--fit-dir",default="artifacts/cstr/discrepancy_fit");p.add_argument("--output-dir",default="artifacts/cstr/discrepancy_freeze");p.add_argument("--posterior-samples",type=int,default=2000);p.add_argument("--bootstrap-samples",type=int,default=2000);p.add_argument("--workers",type=int,default=min(5,os.cpu_count() or 1));p.add_argument("--limit-calibration-episodes",type=int);p.add_argument("--limit-test-episodes",type=int);return p


if __name__=="__main__":
    result=run(DiscrepancyFreezeConfig(**vars(parser().parse_args())));print(f"discrepancy freeze complete: {result}")

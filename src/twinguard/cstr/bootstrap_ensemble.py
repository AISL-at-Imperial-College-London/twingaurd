"""Fit bootstrap ensembles of CSTR discrepancy models to estimate parameter uncertainty."""

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

from twinguard.cstr import episode_design as episode_design
from twinguard.cstr import freeze_discrepancy_model as discrepancy_freeze
from twinguard.discrepancy.cstr_state_space import StateSpaceParameters,fit_scalar_state_space_em_checkpoints,kalman_filter,predictive_rollout


LEVELS=episode_design.COVERAGE_LEVELS
CATEGORIES=discrepancy_freeze.CATEGORIES
J_VALUES=(10,20,50)
M_VALUES=(1000,2000,5000)
HORIZON=240
BOOTSTRAP_TARGET=50
BOOTSTRAP_MAX_ATTEMPTS=75
BOOTSTRAP_INIT_SEED=91_005


@dataclass(frozen=True)
class BootstrapEnsembleConfig:
    fit_dir:str="artifacts/cstr/discrepancy_fit"
    freeze_dir:str="artifacts/cstr/discrepancy_freeze"
    output_dir:str="artifacts/cstr/bootstrap_ensemble"
    bootstrap_target:int=50
    bootstrap_max_attempts:int=75
    calibration_per_category:int=20
    final_per_category:int=80
    total_trajectories:int=2000
    bootstrap_samples:int=2000
    workers:int=5
    limit_episodes:int|None=None

    def validate(self)->None:
        if self.bootstrap_target!=50 or self.bootstrap_max_attempts<self.bootstrap_target:raise ValueError("full bootstrap ensemble targets J=50 with declared supplemental attempts")
        if self.limit_episodes is None and (self.calibration_per_category!=20 or self.final_per_category!=80 or self.total_trajectories!=2000):raise ValueError("full bootstrap ensemble freezes 60 calibration, 240 final, and M=2000")


_TRAIN_Y=None
_TRAIN_Z=None


def _init_bootstrap_worker(path:str)->None:
    global _TRAIN_Y,_TRAIN_Z
    with np.load(path) as saved:_TRAIN_Y=np.asarray(saved["residual"],float);_TRAIN_Z=np.asarray(saved["z"],float)


def _bootstrap_task(task:tuple[int,int])->tuple[int,int,StateSpaceParameters|None,str]:
    replicate,seed=task
    try:
        if _TRAIN_Y is None or _TRAIN_Z is None:raise RuntimeError("bootstrap worker lacks training data")
        rng=np.random.default_rng(seed);indices=rng.integers(0,_TRAIN_Y.shape[0],_TRAIN_Y.shape[0]);fit=fit_scalar_state_space_em_checkpoints(_TRAIN_Y[indices],_TRAIN_Z[indices],BOOTSTRAP_INIT_SEED,(100,))[100]
        reason="" if fit.converged and fit.spectral_radius<1 and np.isfinite(fit.log_likelihood) else "nonconverged_or_unstable"
        return replicate,seed,(fit if not reason else None),reason
    except Exception as exc:return replicate,seed,None,f"fit_exception:{type(exc).__name__}:{exc}"


def fit_bootstrap_ensemble(training_path:Path,target:int,max_attempts:int,workers:int)->tuple[list[StateSpaceParameters],list[dict[str,object]]]:
    accepted=[];manifest=[];attempt=0
    while len(accepted)<target and attempt<max_attempts:
        batch=min(max(10,target-len(accepted)),max_attempts-attempt);tasks=[(i,1_210_000+i) for i in range(attempt,attempt+batch)]
        if workers==1:
            _init_bootstrap_worker(str(training_path));results=[_bootstrap_task(x) for x in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(workers,batch),initializer=_init_bootstrap_worker,initargs=(str(training_path),)) as pool:results=list(pool.map(_bootstrap_task,tasks))
        for replicate,seed,fit,reason in results:
            use=fit is not None and len(accepted)<target
            if use:accepted.append(fit)
            payload={"bootstrap_replicate":replicate,"bootstrap_seed":seed,"episode_resample_size":600,"initialization_seed":BOOTSTRAP_INIT_SEED,"EM_iteration_limit":100,"fit_success":fit is not None,"accepted_in_J50":use,"exclusion_reason":reason if fit is None else ("target_already_reached" if not use else ""),"train_log_likelihood":"" if fit is None else fit.log_likelihood,"EM_iterations_completed":"" if fit is None else fit.iterations,"converged":False if fit is None else fit.converged,"final_parameter_change":"" if fit is None else fit.final_parameter_change,"spectral_radius":"" if fit is None else fit.spectral_radius,"stability_margin":"" if fit is None else 1-fit.spectral_radius,"fit_seconds":"" if fit is None else fit.fit_seconds}
            if fit is not None:payload.update({name:json.dumps(getattr(fit,name).tolist()) for name in ("A","B","C","D","Q")});payload["R"]=fit.R
            else:payload.update({name:"" for name in ("A","B","C","D","Q","R")})
            manifest.append(payload)
        attempt+=batch;print(f"bootstrap attempts {attempt}: {len(accepted)} accepted",flush=True)
    if len(accepted)<target:raise RuntimeError(f"only {len(accepted)} converged stable bootstrap fits after {attempt} declared attempts")
    return accepted,manifest


def parameter_summary(models:Sequence[StateSpaceParameters])->list[dict[str,object]]:
    values={"A":np.asarray([p.A[0,0] for p in models]),"C":np.asarray([p.C[0,0] for p in models]),"Q":np.asarray([p.Q[0,0] for p in models]),"R":np.asarray([p.R for p in models]),"spectral_radius":np.asarray([p.spectral_radius for p in models]),"stability_margin":np.asarray([1-p.spectral_radius for p in models])}
    for i,name in enumerate(discrepancy_freeze.discrepancy_fit.FEATURE_NAMES):values[f"B_{name}"]=np.asarray([p.B[0,i] for p in models]);values[f"D_{name}"]=np.asarray([p.D[0,i] for p in models])
    for h in (1,10,50,100,240):values[f"A_power_{h}"]=values["A"]**h
    rows=[]
    for name,x in values.items():rows.append({"parameter":name,"number_of_models":len(x),"mean":float(np.mean(x)),"standard_deviation":float(np.std(x,ddof=1)),"median":float(np.median(x)),"q025":float(np.quantile(x,.025)),"q975":float(np.quantile(x,.975)),"minimum":float(np.min(x)),"maximum":float(np.max(x))})
    return rows


def _hash(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def _reveal(item:discrepancy_freeze.EpisodeInputs)->np.ndarray:
    if item.truth_path.exists():
        with np.load(item.truth_path) as saved:return np.asarray(saved["actual_plant_temperature_k"],float)
    if item.hidden_plant is None or item.hidden_measurement_seed is None:raise RuntimeError("hidden plant unavailable")
    rng=np.random.default_rng(item.hidden_measurement_seed);actual=[]
    for _ in range(HORIZON):
        out=item.hidden_plant.simulate_step(Fin_sp_normal=item.proposed_action,Fin_sp_startup=item.proposed_action);actual.append(float(out["T_true"])+float(rng.normal(0,episode_design.EpisodeDesignConfig().measurement_noise_k)))
    actual=np.asarray(actual);item.truth_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(item.truth_path,actual_plant_temperature_k=actual,actual_residual_k=actual-item.nominal_future);return actual


def _paths_for_model(p:StateSpaceParameters,item:discrepancy_freeze.EpisodeInputs,count:int,seed:int)->tuple[np.ndarray,np.ndarray,np.ndarray,float,float]:
    start=time.perf_counter();filtered=kalman_filter(p,item.residual_online,item.z_online);filter_seconds=time.perf_counter()-start;roll=predictive_rollout(p,filtered.filtered_mean[-1],filtered.filtered_covariance[-1],item.z_future,count,np.random.default_rng(seed));paths=item.nominal_future[None,:]+roll["residual_samples"];mean=item.nominal_future+roll["analytical_mean"];return paths,mean,np.asarray(roll["analytical_variance"]),filter_seconds,float(roll["monte_carlo_seconds"])


def _interval_payload(name:str,paths:np.ndarray)->tuple[dict[str,np.ndarray],dict[float,tuple[np.ndarray,np.ndarray]],object]:
    payload={};point={};sim=episode_design.construct_simultaneous_bands(paths)
    for level in LEVELS:
        tail=(1-level)/2;lo,hi=np.quantile(paths,[tail,1-tail],axis=0);label=int(level*100);point[level]=(lo,hi);payload[f"{name}_pointwise_{label}_lower_k"]=lo;payload[f"{name}_pointwise_{label}_upper_k"]=hi;payload[f"{name}_simultaneous_{label}_lower_k"]=sim.lower[level];payload[f"{name}_simultaneous_{label}_upper_k"]=sim.upper[level]
    return payload,point,sim


def evaluate_episode(fixed:StateSpaceParameters,ensemble:Sequence[StateSpaceParameters],item:discrepancy_freeze.EpisodeInputs,j:int,m_total:int,prediction_dir:Path,include_u0:bool=True)->tuple[list[dict[str,object]],list[dict[str,object]],dict[str,object]]:
    chosen=list(ensemble[:j]);payload={"nominal_temperature_k":item.nominal_future};details={};runtime={"episode_id":item.episode_id,"action_category":item.category,"ensemble_size_J":j,"total_trajectories":m_total}
    if include_u0:
        u0_start=time.perf_counter()
        u0paths,u0mean,u0var,u0filter,u0roll=_paths_for_model(fixed,item,m_total,item.random_seed+131_000_000);extra,point,sim=_interval_payload("U0",u0paths);payload.update(extra);payload["U0_predictive_mean_temperature_k"]=u0mean;details["U0"]=(u0paths,u0mean,point,sim,u0var,np.zeros(HORIZON));runtime.update({"U0_filtering_seconds":u0filter,"U0_trajectory_seconds":u0roll,"U0_total_seconds":u0filter+u0roll})
        runtime["U0_total_seconds"]=time.perf_counter()-u0_start
    u3_start=time.perf_counter()
    counts=np.full(j,m_total//j,int);counts[:m_total%j]+=1;paths=[];means=[];variances=[];filter_seconds=0.;roll_seconds=0.
    for index,(model,count) in enumerate(zip(chosen,counts)):
        sample,mean,var,ft,rt=_paths_for_model(model,item,int(count),item.random_seed+141_000_000+index*10_000);paths.append(sample);means.append(mean);variances.append(var);filter_seconds+=ft;roll_seconds+=rt
    u3paths=np.vstack(paths);means=np.vstack(means);variances=np.vstack(variances);u3mean=np.mean(means,axis=0);within=np.mean(variances,axis=0);between=np.var(means,axis=0,ddof=0);extra,point,sim=_interval_payload("U3",u3paths);payload.update(extra);payload["U3_predictive_mean_temperature_k"]=u3mean;details["U3"]=(u3paths,u3mean,point,sim,within,between);runtime.update({"U3_filtering_seconds":filter_seconds,"U3_trajectory_seconds":roll_seconds,"U3_total_seconds":time.perf_counter()-u3_start})
    prediction_dir.mkdir(parents=True,exist_ok=True);path=prediction_dir/f"{item.episode_id}.npz";np.savez_compressed(path,**payload);before=_hash(path);actual=_reveal(item)
    if _hash(path)!=before:raise AssertionError("prediction snapshot changed after reveal")
    rows=[];horizon=[]
    for name,(paths,mean,point,sim,within,between) in details.items():
        error=mean-actual;rmse=float(np.sqrt(np.mean(error**2)));mae=float(np.mean(np.abs(error)));sd=np.std(paths,axis=0,ddof=1)
        for level in LEVELS:
            lo,hi=point[level];rows.append({"episode_id":item.episode_id,"action_category":item.category,"random_seed":item.random_seed,"uncertainty_model":name,"ensemble_size_J":0 if name=="U0" else j,"total_trajectories":m_total,"nominal_level":level,"pointwise_coverage":float(np.mean((actual>=lo)&(actual<=hi))),"pointwise_average_width_k":float(np.mean(hi-lo)),"pointwise_median_width_k":float(np.median(hi-lo)),"simultaneous_coverage":int(np.all((actual>=sim.lower[level])&(actual<=sim.upper[level]))),"simultaneous_average_width_k":float(np.mean(sim.upper[level]-sim.lower[level])),"rollout_RMSE_k":rmse,"rollout_MAE_k":mae,"prediction_saved_before_truth_reveal":True})
        lo95,hi95=point[.95]
        for h in range(HORIZON):horizon.append({"episode_id":item.episode_id,"action_category":item.category,"uncertainty_model":name,"horizon_step":h+1,"covered_95":int(actual[h]>=lo95[h] and actual[h]<=hi95[h]),"predictive_standard_deviation_k":float(sd[h]),"squared_error_k2":float(error[h]**2),"bias_k":float(error[h]),"within_model_variance_k2":float(within[h]),"between_model_variance_k2":float(between[h]),"total_decomposed_variance_k2":float(within[h]+between[h])})
    return rows,horizon,runtime


def summarize_horizon(rows:Sequence[dict[str,object]])->tuple[list[dict[str,object]],list[dict[str,object]]]:
    summary=[];decomp=[]
    for model in ("U0","U3"):
        for category in (*CATEGORIES,"pooled"):
            source=[r for r in rows if r["uncertainty_model"]==model and (category=="pooled" or r["action_category"]==category)]
            for h in range(1,HORIZON+1):
                chosen=[r for r in source if r["horizon_step"]==h];summary.append({"uncertainty_model":model,"action_category":category,"horizon_step":h,"number_of_episodes":len(chosen),"empirical_95_coverage":float(np.mean([r["covered_95"] for r in chosen])),"mean_predictive_standard_deviation_k":float(np.mean([r["predictive_standard_deviation_k"] for r in chosen])),"realised_RMSE_k":float(np.sqrt(np.mean([r["squared_error_k2"] for r in chosen]))),"mean_bias_k":float(np.mean([r["bias_k"] for r in chosen]))})
                if model=="U3":decomp.append({"action_category":category,"horizon_step":h,"number_of_episodes":len(chosen),"mean_within_model_variance_k2":float(np.mean([r["within_model_variance_k2"] for r in chosen])),"mean_between_model_variance_k2":float(np.mean([r["between_model_variance_k2"] for r in chosen])),"mean_total_variance_k2":float(np.mean([r["total_decomposed_variance_k2"] for r in chosen])),"between_variance_fraction":float(np.mean([r["between_model_variance_k2"]/(r["total_decomposed_variance_k2"]+1e-15) for r in chosen]))})
    return summary,decomp


def sensitivity_row(rows:Sequence[dict[str,object]],runtime:Sequence[dict[str,object]],experiment:str,j:int,m:int)->dict[str,object]:
    selected=[r for r in rows if r["uncertainty_model"]=="U3" and np.isclose(r["nominal_level"],.95)];return {"experiment_type":experiment,"ensemble_size_J":j,"total_trajectories":m,"number_of_episodes":len(selected),"pointwise_95_coverage":float(np.mean([r["pointwise_coverage"] for r in selected])),"simultaneous_95_coverage":float(np.mean([r["simultaneous_coverage"] for r in selected])),"average_95_width_k":float(np.mean([r["pointwise_average_width_k"] for r in selected])),"simultaneous_95_width_k":float(np.mean([r["simultaneous_average_width_k"] for r in selected])),"median_recursive_RMSE_k":float(np.median([r["rollout_RMSE_k"] for r in selected])),"median_online_latency_seconds":float(np.median([r["U3_total_seconds"] for r in runtime]))}


def choose_j(rows:Sequence[dict[str,object]])->int:
    reference=next(r for r in rows if r["experiment_type"]=="ensemble_size" and r["ensemble_size_J"]==50)
    for j in J_VALUES:
        row=next(r for r in rows if r["experiment_type"]=="ensemble_size" and r["ensemble_size_J"]==j)
        if abs(row["pointwise_95_coverage"]-reference["pointwise_95_coverage"])<=.03 and abs(row["simultaneous_95_coverage"]-reference["simultaneous_95_coverage"])<=.05 and row["median_recursive_RMSE_k"]<=1.05*reference["median_recursive_RMSE_k"]:return j
    return 50


def decision(summary,comparison)->tuple[str,dict[str,float]]:
    pooled={r["uncertainty_model"]:r for r in comparison if r["action_category"]=="pooled"};u0=pooled["U0"];u3=pooled["U3"];s95={r["uncertainty_model"]:r for r in summary if r["action_category"]=="pooled" and np.isclose(r["nominal_level"],.95)}
    metrics={"pointwise_MACE_reduction":1-u3["pointwise_MACE"]/u0["pointwise_MACE"],"simultaneous_MACE_reduction":1-u3["simultaneous_MACE"]/u0["simultaneous_MACE"],"pointwise_95_gain":s95["U3"]["pointwise_empirical_coverage"]-s95["U0"]["pointwise_empirical_coverage"],"simultaneous_95_gain":s95["U3"]["simultaneous_empirical_coverage"]-s95["U0"]["simultaneous_empirical_coverage"],"U3_pointwise_95":s95["U3"]["pointwise_empirical_coverage"],"U3_simultaneous_95":s95["U3"]["simultaneous_empirical_coverage"],"width_ratio":u3["average_predictive_width_k"]/u0["average_predictive_width_k"],"RMSE_ratio":u3["median_rollout_RMSE_k"]/u0["median_rollout_RMSE_k"]}
    sufficient=metrics["pointwise_MACE_reduction"]>=.30 and metrics["simultaneous_MACE_reduction"]>=.30 and metrics["U3_pointwise_95"]>=.85 and metrics["U3_simultaneous_95"]>=.80 and metrics["width_ratio"]<=4 and metrics["RMSE_ratio"]<=1.10
    partial=metrics["pointwise_MACE_reduction"]>=.15 or metrics["simultaneous_MACE_reduction"]>=.15 or metrics["pointwise_95_gain"]>=.10 or metrics["simultaneous_95_gain"]>=.10
    return ("PARAMETER UNCERTAINTY SUFFICIENT" if sufficient else "PARAMETER UNCERTAINTY PARTIALLY SUFFICIENT" if partial else "STRUCTURAL UNCERTAINTY REMAINS"),metrics


def _write_csv(path:Path,rows:Sequence[dict[str,object]])->None:discrepancy_freeze._write_csv(path,rows)


def create_figures(output:Path,models,fixed,summary,horizon,decomp,sensitivity,episode_rows)->None:
    figures=output/"figures";figures.mkdir(parents=True,exist_ok=True);plt.rcParams.update({"font.size":9,"figure.dpi":150})
    def save(fig,name):fig.tight_layout();fig.savefig(figures/f"{name}.png",dpi=300);fig.savefig(figures/f"{name}.pdf");plt.close(fig)
    a=np.asarray([p.A[0,0] for p in models]);fig,ax=plt.subplots(figsize=(6.5,4));ax.hist(a,bins=14,alpha=.75);ax.axvline(fixed.A[0,0],color="black",ls="--",label="frozen U0");ax.set(xlabel="A = spectral radius",ylabel="Bootstrap models");ax.legend();save(fig,"figure_BOOTSTRAP_1_bootstrap_A")
    hs=(1,10,50,100,240);fig,ax=plt.subplots(figsize=(6.5,4));ax.boxplot([a**h for h in hs],tick_labels=hs);ax.set(xlabel="Horizon h",ylabel=r"$A^h$");save(fig,"figure_BOOTSTRAP_2_A_power_h")
    for number,metric,title in ((3,"pointwise_empirical_coverage","Pointwise calibration"),(4,"simultaneous_empirical_coverage","Simultaneous calibration")):
        fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharex=True,sharey=True)
        for ax,category in zip(axes,CATEGORIES):
            for model in ("U0","U3"):
                rows=[r for r in summary if r["action_category"]==category and r["uncertainty_model"]==model];values=np.asarray([r[metric] for r in rows]);lo=np.asarray([r["pointwise_CI_lower" if number==3 else "simultaneous_CI_lower"] for r in rows]);hi=np.asarray([r["pointwise_CI_upper" if number==3 else "simultaneous_CI_upper"] for r in rows]);ax.errorbar([r["nominal_level"] for r in rows],values,yerr=np.vstack((values-lo,hi-values)),fmt="o-",capsize=2,label=model)
            ax.plot([.45,1],[.45,1],"k--",lw=.8);ax.set_title(category);ax.set_xlabel("Nominal coverage");ax.grid(alpha=.25)
        axes[0].set_ylabel("Empirical coverage");axes[-1].legend();fig.suptitle(title);save(fig,f"figure_BOOTSTRAP_{number}_{'pointwise' if number==3 else 'simultaneous'}_calibration")
    pooled=[r for r in horizon if r["action_category"]=="pooled"];fig,ax=plt.subplots(figsize=(7,4));
    for model in ("U0","U3"):
        rows=[r for r in pooled if r["uncertainty_model"]==model];ax.plot([r["horizon_step"] for r in rows],[r["empirical_95_coverage"] for r in rows],label=model)
    ax.axhline(.95,color="black",ls="--");ax.set(xlabel="Horizon",ylabel="Empirical 95% coverage");ax.legend();save(fig,"figure_BOOTSTRAP_5_horizon_coverage")
    rows=[r for r in decomp if r["action_category"]=="pooled"];fig,ax=plt.subplots(figsize=(7,4));ax.plot([r["horizon_step"] for r in rows],[r["mean_within_model_variance_k2"] for r in rows],label="within-model");ax.plot([r["horizon_step"] for r in rows],[r["mean_between_model_variance_k2"] for r in rows],label="between-model");ax.plot([r["horizon_step"] for r in rows],[r["mean_total_variance_k2"] for r in rows],label="total");ax.set(xlabel="Horizon",ylabel=r"Variance (K$^2$)");ax.legend();save(fig,"figure_BOOTSTRAP_6_variance_decomposition")
    pooled=[r for r in summary if r["action_category"]=="pooled"];fig,ax=plt.subplots(figsize=(6.5,4));
    for model in ("U0","U3"):
        rows=[r for r in pooled if r["uncertainty_model"]==model];ax.plot([r["average_predictive_width_k"] for r in rows],[r["pointwise_empirical_coverage"] for r in rows],"o-",label=model)
    ax.set(xlabel="Average width (K)",ylabel="Pointwise coverage");ax.legend();save(fig,"figure_BOOTSTRAP_7_coverage_width")
    for number,category in enumerate(CATEGORIES,start=8):
        rows=[r for r in episode_rows if r["uncertainty_model"]=="U3" and r["action_category"]==category and np.isclose(r["nominal_level"],.95)];med=np.median([r["rollout_RMSE_k"] for r in rows]);chosen=min(rows,key=lambda r:abs(r["rollout_RMSE_k"]-med));eid=chosen["episode_id"]
        with np.load(output/"predictions"/f"{eid}.npz") as pred,np.load(output/"revealed_final_truth"/f"{eid}.npz") as truth:mean=np.asarray(pred["U3_predictive_mean_temperature_k"]);lo=np.asarray(pred["U3_pointwise_95_lower_k"]);hi=np.asarray(pred["U3_pointwise_95_upper_k"]);nom=np.asarray(pred["nominal_temperature_k"]);actual=np.asarray(truth["actual_plant_temperature_k"])
        fig,ax=plt.subplots(figsize=(7,4));h=np.arange(1,HORIZON+1);ax.fill_between(h,lo,hi,alpha=.25,label="U3 95% pointwise");ax.plot(h,nom,"--",label="nominal DT");ax.plot(h,mean,label="U3 mean");ax.plot(h,actual,color="black",label="revealed plant");ax.set(xlabel="Horizon",ylabel="Temperature (K)",title=f"{category}: {eid}");ax.legend();save(fig,f"figure_BOOTSTRAP_{number}_{category}_rollout")
    rows=[r for r in sensitivity if r["experiment_type"]=="ensemble_size"];fig,ax=plt.subplots(figsize=(6.5,4));ax.plot([r["ensemble_size_J"] for r in rows],[r["pointwise_95_coverage"] for r in rows],"o-",label="pointwise");ax.plot([r["ensemble_size_J"] for r in rows],[r["simultaneous_95_coverage"] for r in rows],"s-",label="simultaneous");ax.axhline(.95,color="black",ls="--");ax.set(xlabel="Ensemble size J",ylabel="95% coverage");ax.legend();save(fig,"figure_BOOTSTRAP_11_calibration_vs_J")
    fig,ax=plt.subplots(figsize=(6.5,4));ax.plot([r["ensemble_size_J"] for r in rows],[r["median_online_latency_seconds"] for r in rows],"o-");ax.set(xlabel="Ensemble size J",ylabel="Median online latency (s)");save(fig,"figure_BOOTSTRAP_12_latency_vs_J")


def create_reports(output:Path,manifest,param_summary,sensitivity,summary,comparison,horizon,decomp,runtime,selected_j,decision_name,metrics)->None:
    pooled={r["uncertainty_model"]:r for r in comparison if r["action_category"]=="pooled"};u0=pooled["U0"];u3=pooled["U3"];two_tank_setup=next(r for r in decomp if r["action_category"]=="pooled" and r["horizon_step"]==1);d240=next(r for r in decomp if r["action_category"]=="pooled" and r["horizon_step"]==240);h1={r["uncertainty_model"]:r for r in horizon if r["action_category"]=="pooled" and r["horizon_step"]==1};h240={r["uncertainty_model"]:r for r in horizon if r["action_category"]=="pooled" and r["horizon_step"]==240};online=next(r for r in runtime if r["runtime_type"]=="final_online" and r["uncertainty_model"]=="U3" and r["action_category"]=="pooled");failed=sum(str(r["fit_success"]).lower()!="true" for r in manifest)
    readme=f"""# bootstrap ensemble — parameter-uncertainty propagation

## Result

**{decision_name}**

Fifty converged, stable bootstrap models were obtained from complete-episode resamples of the original 600-episode offline training population. {failed} attempted fits failed the predeclared convergence/stability rule and were recorded. No calibration, selection, discrepancy freeze test, or bootstrap ensemble outcome entered bootstrap fitting. Calibration used 60 newly generated episodes; after J={selected_j} and M=2000 were frozen, final evaluation used another 240 new episodes with distinct seeds and prediction-before-reveal snapshots.

U0 pooled pointwise/simultaneous MACE are {u0['pointwise_MACE']:.3f}/{u0['simultaneous_MACE']:.3f}; U3 values are {u3['pointwise_MACE']:.3f}/{u3['simultaneous_MACE']:.3f}. U3 changes 95% coverage by {metrics['pointwise_95_gain']:+.3f} pointwise and {metrics['simultaneous_95_gain']:+.3f} simultaneous. Its width ratio is {metrics['width_ratio']:.2f}, while its mean-RMSE ratio is {metrics['RMSE_ratio']:.3f}. Extremely broad bands do not qualify as success under the frozen rule.

## Scientific questions

1. Missing uncertainty explained: MACE reductions are {metrics['pointwise_MACE_reduction']:.1%} pointwise and {metrics['simultaneous_MACE_reduction']:.1%} simultaneous.
2. Between-model variance changes from {two_tank_setup['mean_between_model_variance_k2']:.4f} K² at h=1 to {d240['mean_between_model_variance_k2']:.4f} K² at h=240; its final share is {d240['between_variance_fraction']:.1%}.
3. Late-horizon 95% coverage changes from {h240['U0']['empirical_95_coverage']:.3f} to {h240['U3']['empirical_95_coverage']:.3f}, versus h=1 values {h1['U0']['empirical_95_coverage']:.3f}/{h1['U3']['empirical_95_coverage']:.3f}.
4. Central prediction: U0/U3 median recursive RMSE is {u0['median_rollout_RMSE_k']:.3f}/{u3['median_rollout_RMSE_k']:.3f} K; U3 MAE and horizon bias are reported in the episode and horizon tables.
5. Operational ensemble: J={selected_j}, the smallest calibration-equivalent size under the frozen J50 comparison rule.
6. Runtime: U3 median total latency is {online['median_total_seconds']:.4f} s; offline bootstrap fitting is separately reported and is not an online cost.
7. Safety-probability readiness: {'yes, subject to later decision-level validation' if decision_name=='PARAMETER UNCERTAINTY SUFFICIENT' else 'no; the remaining calibration limitation must be resolved or explicitly bounded first'}.

The parameter family, preprocessing, N=50 filter, latent dimension, and conditional mean formulation are unchanged from discrepancy freeze. U3 adds only whole-episode-bootstrap parameter uncertainty.
"""
    (output/"README.md").write_text(readme,encoding="utf-8");(output/"decision_summary.md").write_text(f"# bootstrap ensemble decision\n\n**{decision_name}**\n\nPointwise/simultaneous MACE reductions: {metrics['pointwise_MACE_reduction']:.1%}/{metrics['simultaneous_MACE_reduction']:.1%}. U3 95% coverage: {metrics['U3_pointwise_95']:.3f}/{metrics['U3_simultaneous_95']:.3f}. Width and RMSE ratios: {metrics['width_ratio']:.2f}/{metrics['RMSE_ratio']:.3f}.\n",encoding="utf-8")


def run(config:BootstrapEnsembleConfig)->str:
    config.validate();output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True);b9dir=Path(config.fit_dir);b10dir=Path(config.freeze_dir);fixed=discrepancy_freeze._parameter_from_payload(json.loads((b10dir/"frozen_latent_model.json").read_text(encoding="utf-8"))["parameters"])
    models,manifest=fit_bootstrap_ensemble(b9dir/"training_population.npz",config.bootstrap_target,config.bootstrap_max_attempts,config.workers);_write_csv(output/"bootstrap_model_manifest.csv",manifest);params=parameter_summary(models);_write_csv(output/"bootstrap_parameter_summary.csv",params);(output/"bootstrap_models.json").write_text(json.dumps([discrepancy_freeze._parameter_payload(p) for p in models],indent=2),encoding="utf-8")
    b5cfg=episode_design.EpisodeDesignConfig();template=episode_design.build_template(b5cfg);cal_n=config.limit_episodes or config.calibration_per_category;final_n=config.limit_episodes or config.final_per_category
    calibration,cal_manifest=discrepancy_freeze.generate_new_test_inputs(b5cfg,template,output,cal_n,830_000,"cstr_bootstrap_calibration","revealed_calibration_truth");_write_csv(output/"calibration_population_manifest.csv",cal_manifest)
    sensitivity=[]
    for j in J_VALUES:
        rows=[];times=[]
        for item in calibration:
            result,_,runtime=evaluate_episode(fixed,models,item,j,config.total_trajectories,output/"calibration_predictions"/f"J{j}",False);rows.extend(result);times.append(runtime)
        sensitivity.append(sensitivity_row(rows,times,"ensemble_size",j,config.total_trajectories))
    selected_j=choose_j(sensitivity);diagnostic=[item for category in CATEGORIES for item in [x for x in calibration if x.category==category][:3]]
    for m in M_VALUES:
        rows=[];times=[]
        for item in diagnostic:
            result,_,runtime=evaluate_episode(fixed,models,item,selected_j,m,output/"calibration_predictions"/f"M{m}",False);rows.extend(result);times.append(runtime)
        sensitivity.append(sensitivity_row(rows,times,"monte_carlo_budget",selected_j,m))
    _write_csv(output/"ensemble_size_sensitivity.csv",sensitivity)
    final,final_manifest=discrepancy_freeze.generate_new_test_inputs(b5cfg,template,output,final_n,930_000,"cstr_bootstrap_test","revealed_final_truth");_write_csv(output/"final_population_manifest.csv",final_manifest)
    episode_rows=[];horizon_rows=[];runtime_rows=[]
    for index,item in enumerate(final):
        rows,hrows,runtime=evaluate_episode(fixed,models,item,selected_j,config.total_trajectories,output/"predictions",True);episode_rows.extend(rows);horizon_rows.extend(hrows);runtime_rows.append(runtime)
        if (index+1)%40==0:print(f"completed {index+1}/{len(final)} bootstrap ensemble final episodes",flush=True)
    summary,comparison=discrepancy_freeze.summarize_calibration(episode_rows,config.bootstrap_samples);horizon_summary,decomp=summarize_horizon(horizon_rows);decision_name,metrics=decision(summary,comparison)
    runtime=[]
    runtime.append({"runtime_type":"offline_bootstrap_fit","uncertainty_model":"U3","action_category":"pooled","ensemble_size_J":50,"number_of_episodes":600,"median_filtering_seconds":"","median_trajectory_seconds":"","median_total_seconds":float(np.sum([float(r["fit_seconds"]) for r in manifest if r["fit_seconds"]!=""]))})
    for model in ("U0","U3"):
        for category in (*CATEGORIES,"pooled"):
            chosen=runtime_rows if category=="pooled" else [r for r in runtime_rows if r["action_category"]==category];prefix=model;runtime.append({"runtime_type":"final_online","uncertainty_model":model,"action_category":category,"ensemble_size_J":0 if model=="U0" else selected_j,"number_of_episodes":len(chosen),"median_filtering_seconds":float(np.median([r[f"{prefix}_filtering_seconds"] for r in chosen])),"median_trajectory_seconds":float(np.median([r[f"{prefix}_trajectory_seconds"] for r in chosen])),"median_total_seconds":float(np.median([r[f"{prefix}_total_seconds"] for r in chosen]))})
    for row in sensitivity:runtime.append({"runtime_type":"J_or_M_sensitivity","uncertainty_model":"U3","action_category":"pooled","ensemble_size_J":row["ensemble_size_J"],"number_of_episodes":row["number_of_episodes"],"median_filtering_seconds":"","median_trajectory_seconds":"","median_total_seconds":row["median_online_latency_seconds"]})
    _write_csv(output/"calibration_episode_results.csv",episode_rows);_write_csv(output/"calibration_summary.csv",summary);_write_csv(output/"uncertainty_model_comparison.csv",comparison);_write_csv(output/"horizon_calibration_summary.csv",horizon_summary);_write_csv(output/"variance_decomposition.csv",decomp);_write_csv(output/"runtime_summary.csv",runtime)
    metadata={"configuration":asdict(config),"frozen_U0_identity":"B9S_seed_91005_limit_100","bootstrap_source":"original artifacts/cstr/discrepancy_fit/training_population.npz only","bootstrap_unit":"complete episode","acceptance":"converged by 100 iterations, finite likelihood, spectral radius < 1; attempts up to 75 declared before fitting","selected_J":selected_j,"selected_M_total":config.total_trajectories,"decision_rule":{"sufficient":"MACE reductions >=30%, U3 95% pointwise/simultaneous >=.85/.80, width <=4x, RMSE <=1.10x","partial":"either MACE reduction >=15% or 95% coverage gain >=.10"},"decision_metrics":metrics,"decision":decision_name};(output/"configuration.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    create_figures(output,models,fixed,summary,horizon_summary,decomp,sensitivity,episode_rows);create_reports(output,manifest,params,sensitivity,summary,comparison,horizon_summary,decomp,runtime,selected_j,decision_name,metrics);return decision_name


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--fit-dir",default="artifacts/cstr/discrepancy_fit");p.add_argument("--freeze-dir",default="artifacts/cstr/discrepancy_freeze");p.add_argument("--output-dir",default="artifacts/cstr/bootstrap_ensemble");p.add_argument("--workers",type=int,default=min(5,os.cpu_count() or 1));p.add_argument("--limit-episodes",type=int);p.add_argument("--total-trajectories",type=int,default=2000);p.add_argument("--bootstrap-samples",type=int,default=2000);return p


if __name__=="__main__":
    result=run(BootstrapEnsembleConfig(**vars(parser().parse_args())));print(f"bootstrap ensemble complete: {result}")

"""Calibrate CSTR discrepancy predictions with episode-level conformal scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict,dataclass
from pathlib import Path
import time
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twinguard.cstr import episode_design as episode_design
from twinguard.cstr import freeze_discrepancy_model as discrepancy_freeze
from twinguard.cstr import bootstrap_ensemble as bootstrap_ensemble


LEVELS=episode_design.COVERAGE_LEVELS
CATEGORIES=discrepancy_freeze.CATEGORIES
METHODS=("B0","B1","B2","B3H","B2M")
EPSILON_SCALE_K=1e-6
HORIZON_SMOOTHING_WINDOW=31
PRIMARY_CALIBRATION_COUNT=500
DIAGNOSTIC_COUNT=150
FINAL_COUNT=500
SIZE_VALUES=(50,100,200,300,500)
SIZE_REPETITIONS=20


@dataclass(frozen=True)
class TrajectoryConformalConfig:
    bootstrap_dir:str="artifacts/cstr/bootstrap_ensemble"
    output_dir:str="artifacts/cstr/conformal_calibration"
    calibration_count:int=500
    diagnostic_count:int=150
    final_count:int=500
    trajectories:int=2000
    ensemble_size:int=10
    bootstrap_samples:int=2000
    limit_episodes:int|None=None

    def validate(self)->None:
        if self.limit_episodes is None and (self.calibration_count,self.diagnostic_count,self.final_count,self.trajectories,self.ensemble_size)!=(500,150,500,2000,10):raise ValueError("full trajectory calibration freezes n_cal=500, n_diag=150, n_test=500, M=2000, J=10")


def corrected_quantile(scores:np.ndarray,level:float)->tuple[float,int]:
    values=np.sort(np.asarray(scores,float));k=int(np.ceil((values.size+1)*level))
    return (float("inf") if k>values.size else float(values[k-1])),k


def trajectory_scores(actual:np.ndarray,center:np.ndarray,scale:np.ndarray,epsilon:float=EPSILON_SCALE_K,horizon_scale:np.ndarray|None=None)->tuple[float,float]:
    error=np.abs(np.asarray(actual)-np.asarray(center));den=np.maximum(np.asarray(scale),epsilon)
    if horizon_scale is not None:den=den*np.asarray(horizon_scale)
    return float(np.max(error)),float(np.max(error/den))


def smooth_horizon_scale(standardized_errors:np.ndarray,window:int=HORIZON_SMOOTHING_WINDOW)->np.ndarray:
    raw=np.median(np.asarray(standardized_errors,float),axis=0);pad=window//2;padded=np.pad(raw,(pad,pad),mode="edge");smooth=np.convolve(padded,np.ones(window)/window,mode="valid");smooth=np.maximum(smooth,1e-3);return smooth/np.median(smooth)


def _write_csv(path:Path,rows:Sequence[dict[str,object]])->None:discrepancy_freeze._write_csv(path,rows)


def _hash(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_population(config:episode_design.EpisodeDesignConfig,template,output:Path,total:int,base_seed:int,prefix:str,truth_subdir:str)->tuple[list[discrepancy_freeze.EpisodeInputs],list[dict[str,object]]]:
    per=(total+2)//3;episodes,manifest=discrepancy_freeze.generate_new_test_inputs(config,template,output,per,base_seed,prefix,truth_subdir);episodes=episodes[:total];ids={x.episode_id for x in episodes};manifest=[x for x in manifest if x["episode_id"] in ids]
    if len(episodes)!=total:raise RuntimeError("population count mismatch")
    return episodes,manifest


def predict_u3(item:discrepancy_freeze.EpisodeInputs,models:Sequence,trajectories:int,path:Path)->tuple[dict[str,object],float]:
    start=time.perf_counter();j=len(models);counts=np.full(j,trajectories//j,int);counts[:trajectories%j]+=1;paths=[]
    for index,(model,count) in enumerate(zip(models,counts)):
        sample,_,_,_,_=bootstrap_ensemble._paths_for_model(model,item,int(count),item.random_seed+151_000_000+index*10_000);paths.append(sample)
    samples=np.vstack(paths);center=np.median(samples,axis=0);mean=np.mean(samples,axis=0);scale=np.std(samples,axis=0,ddof=1);sim=episode_design.construct_simultaneous_bands(samples);point={};payload={"nominal_temperature_k":item.nominal_future,"U3_posterior_median_temperature_k":center,"U3_posterior_mean_temperature_k":mean,"U3_predictive_standard_deviation_k":scale}
    for level in LEVELS:
        tail=(1-level)/2;lo,hi=np.quantile(samples,[tail,1-tail],axis=0);point[level]=(lo,hi);label=int(level*100);payload[f"B0_pointwise_{label}_lower_k"]=lo;payload[f"B0_pointwise_{label}_upper_k"]=hi;payload[f"B0_simultaneous_{label}_lower_k"]=sim.lower[level];payload[f"B0_simultaneous_{label}_upper_k"]=sim.upper[level]
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**payload);return {"center":center,"mean":mean,"scale":scale,"B0_point":point,"B0_sim":sim,"prediction_path":path},time.perf_counter()-start


def reveal_after_prediction(item:discrepancy_freeze.EpisodeInputs,prediction_path:Path)->np.ndarray:
    before=_hash(prediction_path);actual=bootstrap_ensemble._reveal(item)
    if _hash(prediction_path)!=before:raise AssertionError("prediction changed after truth reveal")
    return actual


def calibration_records(episodes:Sequence[discrepancy_freeze.EpisodeInputs],models,trajectories:int,output:Path,role:str)->tuple[list[dict[str,object]],list[dict[str,np.ndarray]],list[float]]:
    rows=[];arrays=[];times=[]
    for index,item in enumerate(episodes):
        prediction,seconds=predict_u3(item,models,trajectories,output/f"{role}_predictions"/f"{item.episode_id}.npz");actual=reveal_after_prediction(item,prediction["prediction_path"]);error=np.abs(actual-prediction["center"]);absolute,standardized=trajectory_scores(actual,prediction["center"],prediction["scale"]);rows.append({"episode_id":item.episode_id,"population_role":role,"action_category":item.category,"random_seed":item.random_seed,"absolute_max_error_score_k":absolute,"standardized_max_error_score":standardized,"horizon_normalized_score":"","minimum_predictive_sd_k":float(np.min(prediction["scale"])),"median_predictive_sd_k":float(np.median(prediction["scale"])),"prediction_saved_before_truth_reveal":True});arrays.append({"error":error,"standardized_error":error/np.maximum(prediction["scale"],EPSILON_SCALE_K),"scale":prediction["scale"]});times.append(seconds)
        if (index+1)%50==0:print(f"completed {index+1}/{len(episodes)} {role} predictions",flush=True)
    return rows,arrays,times


def build_quantiles(rows:Sequence[dict[str,object]],arrays:Sequence[dict[str,np.ndarray]],horizon_scale:np.ndarray)->tuple[list[dict[str,object]],dict[tuple[str,str,float],float]]:
    for row,array in zip(rows,arrays):row["horizon_normalized_score"]=float(np.max(array["standardized_error"]/horizon_scale))
    lookup={};output=[]
    score_fields={"B1":"absolute_max_error_score_k","B2":"standardized_max_error_score","B3H":"horizon_normalized_score"}
    for method,field in score_fields.items():
        values=np.asarray([r[field] for r in rows],float)
        for level in LEVELS:
            q,k=corrected_quantile(values,level);lookup[(method,"pooled",level)]=q;output.append({"method":method,"calibration_scope":"pooled","nominal_level":level,"miscoverage_alpha":1-level,"number_of_calibration_episodes":len(values),"corrected_rank_k":k,"conformal_quantile":q,"finite_sample_convention":"k=ceil((n+1)*(1-alpha)); kth ordered episode score"})
    for category in CATEGORIES:
        values=np.asarray([r["standardized_max_error_score"] for r in rows if r["action_category"]==category],float)
        for level in LEVELS:
            q,k=corrected_quantile(values,level);lookup[("B2M",category,level)]=q;output.append({"method":"B2M","calibration_scope":category,"nominal_level":level,"miscoverage_alpha":1-level,"number_of_calibration_episodes":len(values),"corrected_rank_k":k,"conformal_quantile":q,"finite_sample_convention":"k=ceil((n+1)*(1-alpha)); kth ordered episode score"})
    return output,lookup


def method_band(method:str,level:float,category:str,prediction:dict[str,object],quantiles:dict,horizon_scale:np.ndarray)->tuple[np.ndarray,np.ndarray,float]:
    center=np.asarray(prediction["center"]);scale=np.maximum(np.asarray(prediction["scale"]),EPSILON_SCALE_K)
    if method=="B0":return prediction["B0_sim"].lower[level],prediction["B0_sim"].upper[level],float("nan")
    scope=category if method=="B2M" else "pooled";q=quantiles[(method,scope,level)]
    half=np.full(center.size,q) if method=="B1" else q*scale*(horizon_scale if method=="B3H" else 1.)
    return center-half,center+half,q


def evaluate_final(episodes,models,trajectories,output,quantiles,horizon_scale)->tuple[list[dict[str,object]],list[dict[str,object]],list[dict[str,object]]]:
    episode_rows=[];horizon_raw=[];runtime=[]
    for index,item in enumerate(episodes):
        prediction,u3_seconds=predict_u3(item,models,trajectories,output/"predictions"/f"{item.episode_id}.npz");conformal_start=time.perf_counter();bands={}
        for method in METHODS:
            for level in LEVELS:bands[(method,level)]=method_band(method,level,item.category,prediction,quantiles,horizon_scale)
        conformal_seconds=time.perf_counter()-conformal_start;actual=reveal_after_prediction(item,prediction["prediction_path"]);error=prediction["center"]-actual;rmse=float(np.sqrt(np.mean(error**2)));mae=float(np.mean(np.abs(error)))
        for method in METHODS:
            for level in LEVELS:
                lo,hi,q=bands[(method,level)];point_lo,point_hi=prediction["B0_point"][level] if method=="B0" else (lo,hi);width=hi-lo;episode_rows.append({"episode_id":item.episode_id,"action_category":item.category,"random_seed":item.random_seed,"method":method,"nominal_level":level,"simultaneous_coverage":int(np.all((actual>=lo)&(actual<=hi))),"pointwise_coverage_diagnostic":float(np.mean((actual>=point_lo)&(actual<=point_hi))),"mean_trajectory_band_width_k":float(np.mean(width)),"median_trajectory_band_width_k":float(np.median(width)),"maximum_trajectory_band_width_k":float(np.max(width)),"integrated_width_k_steps":float(np.sum(width)),"conformal_quantile":"" if method=="B0" else q,"recursive_RMSE_k":rmse,"recursive_MAE_k":mae,"prediction_saved_before_truth_reveal":True})
            lo,hi,_=bands[(method,.95)];point_lo,point_hi=prediction["B0_point"][.95] if method=="B0" else (lo,hi);width=hi-lo
            for h in range(len(actual)):horizon_raw.append({"episode_id":item.episode_id,"action_category":item.category,"method":method,"horizon_step":h+1,"covered_pointwise_95":int(actual[h]>=point_lo[h] and actual[h]<=point_hi[h]),"band_width_95_k":float(width[h]),"squared_center_error_k2":float(error[h]**2),"center_bias_k":float(error[h])})
        runtime.append({"episode_id":item.episode_id,"action_category":item.category,"U3_prediction_seconds":u3_seconds,"conformal_band_construction_seconds":conformal_seconds,"total_online_seconds":u3_seconds+conformal_seconds})
        if (index+1)%50==0:print(f"completed {index+1}/{len(episodes)} final_test predictions",flush=True)
    return episode_rows,horizon_raw,runtime


def wilson(successes:int,n:int,z:float=1.959963984540054)->tuple[float,float]:
    p=successes/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return float(center-half),float(center+half)


def summarize(episode_rows:Sequence[dict[str,object]],horizon_raw:Sequence[dict[str,object]])->tuple[list[dict[str,object]],list[dict[str,object]],list[dict[str,object]]]:
    coverage=[];sharpness=[];horizon=[]
    for method in METHODS:
        for category in (*CATEGORIES,"pooled"):
            source=[r for r in episode_rows if r["method"]==method and (category=="pooled" or r["action_category"]==category)]
            for level in LEVELS:
                chosen=[r for r in source if np.isclose(r["nominal_level"],level)];success=sum(r["simultaneous_coverage"] for r in chosen);ci=wilson(success,len(chosen));emp=success/len(chosen);coverage.append({"method":method,"action_category":category,"nominal_level":level,"number_of_episodes":len(chosen),"simultaneous_empirical_coverage":emp,"Wilson_CI_lower":ci[0],"Wilson_CI_upper":ci[1],"target_coverage":level,"coverage_error":abs(emp-level),"pointwise_coverage_diagnostic":float(np.mean([r["pointwise_coverage_diagnostic"] for r in chosen]))});sharpness.append({"method":method,"action_category":category,"nominal_level":level,"number_of_episodes":len(chosen),"mean_trajectory_band_width_k":float(np.mean([r["mean_trajectory_band_width_k"] for r in chosen])),"median_trajectory_band_width_k":float(np.median([r["median_trajectory_band_width_k"] for r in chosen])),"maximum_trajectory_band_width_k":float(np.max([r["maximum_trajectory_band_width_k"] for r in chosen])),"mean_integrated_width_k_steps":float(np.mean([r["integrated_width_k_steps"] for r in chosen]))})
            raw=[r for r in horizon_raw if r["method"]==method and (category=="pooled" or r["action_category"]==category)]
            for h in range(1,241):
                selected=[r for r in raw if r["horizon_step"]==h];horizon.append({"method":method,"action_category":category,"horizon_step":h,"number_of_episodes":len(selected),"empirical_pointwise_95_coverage_diagnostic":float(np.mean([r["covered_pointwise_95"] for r in selected])),"mean_95_band_width_k":float(np.mean([r["band_width_95_k"] for r in selected])),"realised_center_RMSE_k":float(np.sqrt(np.mean([r["squared_center_error_k2"] for r in selected]))),"center_bias_k":float(np.mean([r["center_bias_k"] for r in selected]))})
    return coverage,sharpness,horizon


def size_sensitivity(calibration_rows,calibration_arrays,diagnostic_rows,diagnostic_arrays)->list[dict[str,object]]:
    cal_scores=np.asarray([r["standardized_max_error_score"] for r in calibration_rows]);diag_scores=np.asarray([r["standardized_max_error_score"] for r in diagnostic_rows]);diag_scale=np.asarray([np.mean(x["scale"]) for x in diagnostic_arrays]);output=[]
    for n in SIZE_VALUES:
        for repetition in range(SIZE_REPETITIONS):
            rng=np.random.default_rng(1_520_000+n*100+repetition);indices=np.arange(len(cal_scores)) if n==len(cal_scores) else rng.choice(len(cal_scores),n,replace=False);q,k=corrected_quantile(cal_scores[indices],.95);output.append({"calibration_size":n,"repetition":repetition,"subsample_seed":1_520_000+n*100+repetition,"diagnostic_episode_count":len(diag_scores),"q_0_95":q,"corrected_rank_k":k,"diagnostic_simultaneous_95_coverage":float(np.mean(diag_scores<=q)),"mean_diagnostic_95_band_width_k":float(np.mean(2*q*diag_scale))})
    return output


def runtime_summary(rows:Sequence[dict[str,object]],offline_seconds:float)->list[dict[str,object]]:
    output=[{"runtime_type":"offline_conformal_score_and_quantile","action_category":"pooled","number_of_episodes":500,"median_U3_prediction_seconds":"","median_conformal_band_seconds":"","median_total_online_seconds":"","seconds":offline_seconds}]
    for category in (*CATEGORIES,"pooled"):
        chosen=rows if category=="pooled" else [r for r in rows if r["action_category"]==category];output.append({"runtime_type":"online_final","action_category":category,"number_of_episodes":len(chosen),"median_U3_prediction_seconds":float(np.median([r["U3_prediction_seconds"] for r in chosen])),"median_conformal_band_seconds":float(np.median([r["conformal_band_construction_seconds"] for r in chosen])),"median_total_online_seconds":float(np.median([r["total_online_seconds"] for r in chosen])),"seconds":""})
    return output


def decide(coverage,sharpness)->tuple[str,dict[str,float]]:
    pooled_b2=[r for r in coverage if r["method"]=="B2" and r["action_category"]=="pooled"];mace=float(np.mean([r["coverage_error"] for r in pooled_b2]));b2_95=next(r for r in pooled_b2 if np.isclose(r["nominal_level"],.95));cats=[r for r in coverage if r["method"]=="B2" and r["action_category"] in CATEGORIES and np.isclose(r["nominal_level"],.95)];mond=[r for r in coverage if r["method"]=="B2M" and r["action_category"] in CATEGORIES and np.isclose(r["nominal_level"],.95)];b2w=next(r for r in sharpness if r["method"]=="B2" and r["action_category"]=="pooled" and np.isclose(r["nominal_level"],.95));b0w=next(r for r in sharpness if r["method"]=="B0" and r["action_category"]=="pooled" and np.isclose(r["nominal_level"],.95));metrics={"B2_pooled_MACE":mace,"B2_pooled_95_coverage":b2_95["simultaneous_empirical_coverage"],"B2_min_category_95_coverage":min(r["simultaneous_empirical_coverage"] for r in cats),"B2M_min_category_95_coverage":min(r["simultaneous_empirical_coverage"] for r in mond),"B2_to_B0_95_width_ratio":b2w["mean_trajectory_band_width_k"]/b0w["mean_trajectory_band_width_k"],"B2_mean_95_width_k":b2w["mean_trajectory_band_width_k"]}
    pooled_ok=mace<=.05 and b2_95["simultaneous_empirical_coverage"]>=.90;sharp=metrics["B2_to_B0_95_width_ratio"]<=3 and metrics["B2_mean_95_width_k"]<=5.;regime=pooled_ok and metrics["B2_min_category_95_coverage"]<.88 and metrics["B2M_min_category_95_coverage"]>=metrics["B2_min_category_95_coverage"]+.05
    if regime:return "REGIME-DEPENDENT CALIBRATION REQUIRED",metrics
    if pooled_ok and sharp and metrics["B2_min_category_95_coverage"]>=.88:return "TRAJECTORY CALIBRATION SUFFICIENT",metrics
    if pooled_ok and not sharp:return "TRAJECTORY CALIBRATION TOO CONSERVATIVE",metrics
    return "STRUCTURAL GENERALIZATION FAILURE",metrics


def create_figures(output,coverage,sharpness,horizon,cal_rows,quantile_rows,size_rows,episode_rows)->None:
    figures=output/"figures";figures.mkdir(parents=True,exist_ok=True);plt.rcParams.update({"font.size":9,"figure.dpi":150})
    def save(fig,name):fig.tight_layout();fig.savefig(figures/f"{name}.png",dpi=300);fig.savefig(figures/f"{name}.pdf");plt.close(fig)
    pooled=[r for r in coverage if r["action_category"]=="pooled"]
    fig,ax=plt.subplots(figsize=(6.5,4));
    for method in ("B0","B1","B2"):
        rows=[r for r in pooled if r["method"]==method];ax.plot([r["nominal_level"] for r in rows],[r["simultaneous_empirical_coverage"] for r in rows],"o-",label=method)
    ax.plot([.45,1],[.45,1],"k--");ax.set(xlabel="Nominal trajectory coverage",ylabel="Empirical simultaneous coverage");ax.legend();save(fig,"figure_TRAJECTORY_CONFORMAL_1_calibration_curves")
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(len(LEVELS));width=.25
    for i,method in enumerate(("B0","B1","B2")):
        rows=[r for r in pooled if r["method"]==method];ax.bar(x+(i-1)*width,[r["simultaneous_empirical_coverage"] for r in rows],width,label=method)
    ax.plot(x,LEVELS,"ko",label="target");ax.set_xticks(x,[f"{int(100*x)}%" for x in LEVELS]);ax.set(ylabel="Whole-trajectory coverage");ax.legend();save(fig,"figure_TRAJECTORY_CONFORMAL_2_trajectory_coverage")
    fig,ax=plt.subplots(figsize=(6.5,4));
    for method in ("B0","B1","B2"):
        cov=[r for r in pooled if r["method"]==method];wid=[r for r in sharpness if r["method"]==method and r["action_category"]=="pooled"];ax.plot([r["mean_trajectory_band_width_k"] for r in wid],[r["simultaneous_empirical_coverage"] for r in cov],"o-",label=method)
    ax.set(xlabel="Mean trajectory-band width (K)",ylabel="Simultaneous coverage");ax.legend();save(fig,"figure_TRAJECTORY_CONFORMAL_3_coverage_width")
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].hist([r["standardized_max_error_score"] for r in cal_rows],bins=30,alpha=.8);axes[0].set(xlabel="B2 calibration score",ylabel="Episodes");axes[1].hist([r["absolute_max_error_score_k"] for r in cal_rows],bins=30,alpha=.8);axes[1].set(xlabel="B1 score (K)",ylabel="Episodes");save(fig,"figure_TRAJECTORY_CONFORMAL_4_score_distribution")
    fig,ax=plt.subplots(figsize=(6.5,4));rows=[r for r in quantile_rows if r["method"]=="B2" and r["calibration_scope"]=="pooled"];ax.plot([r["nominal_level"] for r in rows],[r["conformal_quantile"] for r in rows],"o-");ax.axhline(1,color="black",ls="--");ax.set(xlabel="Nominal trajectory coverage",ylabel="B2 inflation quantile q");save(fig,"figure_TRAJECTORY_CONFORMAL_5_q_vs_coverage")
    fig,ax=plt.subplots(figsize=(7,4));
    for method in ("B0","B2"):
        rows=[r for r in horizon if r["method"]==method and r["action_category"]=="pooled"];ax.plot([r["horizon_step"] for r in rows],[r["empirical_pointwise_95_coverage_diagnostic"] for r in rows],label=method)
    ax.axhline(.95,color="black",ls="--");ax.set(xlabel="Horizon",ylabel="Diagnostic pointwise 95% coverage");ax.legend();save(fig,"figure_TRAJECTORY_CONFORMAL_6_horizon_coverage")
    fig,ax=plt.subplots(figsize=(7,4));
    for method in ("B0","B1","B2"):
        rows=[r for r in horizon if r["method"]==method and r["action_category"]=="pooled"];ax.plot([r["horizon_step"] for r in rows],[r["mean_95_band_width_k"] for r in rows],label=method)
    ax.set(xlabel="Horizon",ylabel="Mean 95% band width (K)");ax.legend();save(fig,"figure_TRAJECTORY_CONFORMAL_7_width_horizon")
    for number,category in enumerate(CATEGORIES,start=8):
        rows=[r for r in episode_rows if r["method"]=="B2" and r["action_category"]==category and np.isclose(r["nominal_level"],.95)];med=np.median([r["recursive_RMSE_k"] for r in rows]);chosen=min(rows,key=lambda r:abs(r["recursive_RMSE_k"]-med));eid=chosen["episode_id"]
        with np.load(output/"predictions"/f"{eid}.npz") as pred,np.load(output/"revealed_final_truth"/f"{eid}.npz") as truth:center=np.asarray(pred["U3_posterior_median_temperature_k"]);sd=np.asarray(pred["U3_predictive_standard_deviation_k"]);actual=np.asarray(truth["actual_plant_temperature_k"]);nom=np.asarray(pred["nominal_temperature_k"])
        q=chosen["conformal_quantile"];lo=center-q*sd;hi=center+q*sd;fig,ax=plt.subplots(figsize=(7,4));h=np.arange(1,241);ax.fill_between(h,lo,hi,alpha=.25,label="B2 95% trajectory band");ax.plot(h,nom,"--",label="nominal DT");ax.plot(h,center,label="frozen U3 median");ax.plot(h,actual,color="black",label="revealed plant");ax.set(xlabel="Horizon",ylabel="Temperature (K)",title=f"{category}: {eid}");ax.legend();save(fig,f"figure_TRAJECTORY_CONFORMAL_{number}_{category}_trajectory")
    fig,axes=plt.subplots(1,2,figsize=(10,4));
    for n in SIZE_VALUES:
        rows=[r for r in size_rows if r["calibration_size"]==n];axes[0].errorbar(n,np.median([r["q_0_95"] for r in rows]),yerr=np.std([r["q_0_95"] for r in rows]),fmt="o") ;axes[1].errorbar(n,np.mean([r["diagnostic_simultaneous_95_coverage"] for r in rows]),yerr=np.std([r["diagnostic_simultaneous_95_coverage"] for r in rows]),fmt="o")
    axes[0].set(xlabel="n_cal",ylabel="q_0.95 median ± SD");axes[1].axhline(.95,color="black",ls="--");axes[1].set(xlabel="n_cal",ylabel="Diagnostic 95% coverage");save(fig,"figure_TRAJECTORY_CONFORMAL_11_calibration_size")
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3);pooled_b2=[next(r for r in coverage if r["method"]=="B2" and r["action_category"]==c and np.isclose(r["nominal_level"],.95))["simultaneous_empirical_coverage"] for c in CATEGORIES];mond=[next(r for r in coverage if r["method"]=="B2M" and r["action_category"]==c and np.isclose(r["nominal_level"],.95))["simultaneous_empirical_coverage"] for c in CATEGORIES];ax.bar(x-.18,pooled_b2,.36,label="pooled B2");ax.bar(x+.18,mond,.36,label="Mondrian B2M");ax.axhline(.95,color="black",ls="--");ax.set_xticks(x,CATEGORIES);ax.set(ylabel="95% simultaneous coverage");ax.legend();save(fig,"figure_TRAJECTORY_CONFORMAL_12_pooled_vs_Mondrian")


def create_reports(output,coverage,sharpness,quantiles,size_rows,runtime,decision_name,metrics)->None:
    b0=next(r for r in coverage if r["method"]=="B0" and r["action_category"]=="pooled" and np.isclose(r["nominal_level"],.95));b2=next(r for r in coverage if r["method"]=="B2" and r["action_category"]=="pooled" and np.isclose(r["nominal_level"],.95));q=next(r for r in quantiles if r["method"]=="B2" and r["calibration_scope"]=="pooled" and np.isclose(r["nominal_level"],.95))["conformal_quantile"];online=next(r for r in runtime if r["runtime_type"]=="online_final" and r["action_category"]=="pooled");stability=[]
    for n in SIZE_VALUES:
        rows=[r for r in size_rows if r["calibration_size"]==n];stability.append((n,float(np.std([r["q_0_95"] for r in rows])),float(np.mean([r["diagnostic_simultaneous_95_coverage"] for r in rows]))))
    stable_n=next((n for n,sd,cov in stability if sd/max(q,1e-12)<=.05 and abs(cov-.95)<=.05),500)
    readme=f"""# trajectory calibration — conformal trajectory calibration

## Decision

**{decision_name}**

The bootstrap ensemble U3 predictor is frozen exactly at J=10 and M=2000. The posterior median is the predeclared centre and the predictive-scale floor is {EPSILON_SCALE_K:g} K. Primary B2 uses one max-standardized score per complete 240-step calibration trajectory. For level 1-alpha, the finite-sample rank is `ceil((n_cal+1)*(1-alpha))` and the corresponding ordered episode score is used.

Five hundred new calibration episodes were used to freeze all scores, the 31-point smoothed horizon ablation, pooled/Mondrian quantiles, and method definitions. A separate 150-episode diagnostic population was used only for n_cal sensitivity. Five hundred subsequently generated final episodes remained untouched until each prediction snapshot was written and hashed.

## Findings

Raw B0 95% simultaneous coverage is {b0['simultaneous_empirical_coverage']:.3f}; pooled B2 coverage is {b2['simultaneous_empirical_coverage']:.3f} with Wilson CI [{b2['Wilson_CI_lower']:.3f}, {b2['Wilson_CI_upper']:.3f}]. The primary 95% inflation factor is **q={q:.3f}**. B2/B0 95% mean-width ratio is {metrics['B2_to_B0_95_width_ratio']:.2f}, and B2 mean width is {metrics['B2_mean_95_width_k']:.3f} K.

Pooled B2 MACE across 50/80/90/95/99% is {metrics['B2_pooled_MACE']:.3f}. Minimum regime-specific 95% coverage is {metrics['B2_min_category_95_coverage']:.3f}; the Mondrian minimum is {metrics['B2M_min_category_95_coverage']:.3f}. These are empirical regime diagnostics, not conditional conformal guarantees. The primary guarantee remains marginal over the defined episode population.

Calibration-size diagnostics indicate approximately n_cal={stable_n} episodes for stable q and diagnostic coverage under the frozen rule. Median U3 latency is {online['median_U3_prediction_seconds']:.4f} s, conformal construction adds {online['median_conformal_band_seconds']:.6f} s, and total latency is {online['median_total_online_seconds']:.4f} s.

## Answers

1. The frozen mean can be calibrated without modification: {'yes' if decision_name!='STRUCTURAL GENERALIZATION FAILURE' else 'not reliably on the new population'}.
2. Episode-level B2 changes 95% simultaneous coverage from {b0['simultaneous_empirical_coverage']:.3f} to {b2['simultaneous_empirical_coverage']:.3f}.
3. Required 95% inflation is q={q:.3f}; q well above one quantifies U3 scale underestimation.
4. Sharpness is quantified by the {metrics['B2_to_B0_95_width_ratio']:.2f} width ratio and absolute width above.
5. Regime generalization is reported in `coverage_summary.csv`; no category-specific guarantee is claimed.
6. Calibration data efficiency is summarized above and in `calibration_size_sensitivity.csv`.
7. Online conformal overhead is {online['median_conformal_band_seconds']:.6f} s.
8. Action-specific safety certification readiness: {'yes for the next decision-layer evaluation, not as a safety guarantee by itself' if decision_name=='TRAJECTORY CALIBRATION SUFFICIENT' else 'no'}.
"""
    (output/"README.md").write_text(readme,encoding="utf-8");(output/"decision_summary.md").write_text(f"# trajectory calibration decision\n\n**{decision_name}**\n\nB2 pooled 95% simultaneous coverage is {b2['simultaneous_empirical_coverage']:.3f} (Wilson 95% CI {b2['Wilson_CI_lower']:.3f}–{b2['Wilson_CI_upper']:.3f}), q_0.95={q:.3f}, and the B2/B0 width ratio is {metrics['B2_to_B0_95_width_ratio']:.2f}.\n",encoding="utf-8")


def run(config:TrajectoryConformalConfig)->str:
    config.validate();b11dir=Path(config.bootstrap_dir);output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True);models=[discrepancy_freeze._parameter_from_payload(x) for x in json.loads((b11dir/"bootstrap_models.json").read_text(encoding="utf-8"))[:config.ensemble_size]];b5cfg=episode_design.EpisodeDesignConfig();template=episode_design.build_template(b5cfg);ncal=config.limit_episodes or config.calibration_count;ndiag=config.limit_episodes or config.diagnostic_count;ntest=config.limit_episodes or config.final_count
    calibration,cal_manifest=generate_population(b5cfg,template,output,ncal,1_330_000,"cstr_conformal_calibration","revealed_calibration_truth");cal_rows,cal_arrays,_=calibration_records(calibration,models,config.trajectories,output,"calibration");horizon_scale=smooth_horizon_scale(np.vstack([x["standardized_error"] for x in cal_arrays]));offline_start=time.perf_counter();quantile_rows,quantiles=build_quantiles(cal_rows,cal_arrays,horizon_scale);offline_seconds=time.perf_counter()-offline_start;np.save(output/"horizon_scale.npy",horizon_scale)
    diagnostic,diag_manifest=generate_population(b5cfg,template,output,ndiag,1_430_000,"cstr_conformal_diagnostic","revealed_diagnostic_truth");diag_rows,diag_arrays,_=calibration_records(diagnostic,models,config.trajectories,output,"diagnostic");size_rows=size_sensitivity(cal_rows,cal_arrays,diag_rows,diag_arrays)
    for row in cal_manifest:row["population_role"]="primary_conformal_calibration"
    for row in diag_manifest:row["population_role"]="calibration_size_diagnostic"
    _write_csv(output/"calibration_population_manifest.csv",cal_manifest+diag_manifest);_write_csv(output/"calibration_scores.csv",cal_rows);_write_csv(output/"conformal_quantiles.csv",quantile_rows);_write_csv(output/"calibration_size_sensitivity.csv",size_rows)
    # Quantiles and every optional method are frozen before final population creation.
    del calibration,diagnostic
    final,final_manifest=generate_population(b5cfg,template,output,ntest,1_530_000,"cstr_conformal_test","revealed_final_truth");_write_csv(output/"final_test_population_manifest.csv",final_manifest);episode_rows,horizon_raw,runtime_rows=evaluate_final(final,models,config.trajectories,output,quantiles,horizon_scale);coverage,sharpness,horizon=summarize(episode_rows,horizon_raw);runtime=runtime_summary(runtime_rows,offline_seconds);decision_name,metrics=decide(coverage,sharpness)
    _write_csv(output/"final_episode_results.csv",episode_rows);_write_csv(output/"coverage_summary.csv",coverage);_write_csv(output/"sharpness_summary.csv",sharpness);_write_csv(output/"horizon_diagnostics.csv",horizon);_write_csv(output/"runtime_summary.csv",runtime)
    metadata={"configuration":asdict(config),"frozen_predictor":"bootstrap ensemble U3, first 10 accepted bootstrap models, M=2000","centre":"posterior median","epsilon_scale_k":EPSILON_SCALE_K,"primary_score":"max_h |Y_h-m_h|/max(s_h,epsilon)","horizon_ablation":"31-point moving average of calibration median standardized absolute error, normalized to median one","quantile_convention":"k=ceil((n_cal+1)*(1-alpha)); kth ordered episode score","primary_guarantee":"marginal whole-trajectory coverage over the defined episode population; no conditional guarantee","decision_rule":{"sufficient":"B2 pooled MACE <=.05, pooled 95% >=.90, every category 95% >=.88, width <=3x B0 and <=5 K","regime":"pooled adequate, a category <.88, and Mondrian improves minimum by >=.05","too_conservative":"coverage adequate but width rule fails"},"decision_metrics":metrics,"decision":decision_name};(output/"configuration.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    create_figures(output,coverage,sharpness,horizon,cal_rows,quantile_rows,size_rows,episode_rows);create_reports(output,coverage,sharpness,quantile_rows,size_rows,runtime,decision_name,metrics);return decision_name


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--bootstrap-dir",default="artifacts/cstr/bootstrap_ensemble");p.add_argument("--output-dir",default="artifacts/cstr/conformal_calibration");p.add_argument("--limit-episodes",type=int);p.add_argument("--trajectories",type=int,default=2000);p.add_argument("--bootstrap-samples",type=int,default=2000);return p


if __name__=="__main__":
    result=run(TrajectoryConformalConfig(**vars(parser().parse_args())));print(f"trajectory calibration complete: {result}")

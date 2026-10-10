#!/usr/bin/env python3
"""Persistent-offset conformal baseline for the flash process.

Uses the saved calibration design and fixed engineering scales with the
frozen test predictions and truths. Does not regenerate test futures.
Uses twinguard.baselines.flash_conformal_controls.
"""
from __future__ import annotations
import argparse
import os
import csv
from pathlib import Path
import numpy as np
from twinguard.baselines import flash_conformal_controls as base


def offset_center(nominal,observed_residual,window,mode):
    res=np.asarray(observed_residual,float)
    if res.ndim != 2 or res.shape[1]!=3 or len(res)<window:
        raise ValueError(f'Expected at least {window} observations of 3 outputs; got {res.shape}')
    bias=np.mean(res[-window:],axis=0) if mode=='mean' else res[-1]
    return np.asarray(nominal,float)+bias[None,:],bias


def calibration(root,truths,window,mode):
    frozen=base.load_frozen_ordinary_inputs(root)
    result=[]
    for i,eid in enumerate(frozen['episode_id']):
        eid=str(eid)
        future=np.asarray(frozen['nominal_future'][i],float)
        actual=np.asarray(truths[eid],float)
        center,bias=offset_center(future,frozen['observed_residual'][i],window,mode)
        if len(actual)!=120 or len(center)!=120: raise RuntimeError(f'Incomplete frozen calibration episode {eid}')
        scale=base.fixed_scale(len(center))
        score,direction,horizon=base.safety_direction_score(actual,center,scale)
        result.append({'episode_id':eid,'score':float(score),'max_direction':direction,
                       'max_horizon':horizon,'offset_pressure':bias[0],
                       'offset_level':bias[1],'offset_temperature':bias[2]})
    if len(result)!=447: raise RuntimeError(f'Expected 447 calibration complete calibration episodes, got {len(result)}')
    return result


def evaluate(root,q,window,mode):
    safety=base.SafetySpecification();rows=[]
    for eid,p_path,t_path in base.final_episode_paths(root):
        pred=base.load_npz(p_path);truth=base.load_npz(t_path)
        actual=np.asarray(truth['actual_horizon_outputs'],float)
        complete=base.scalar_str(pred['prediction_status'])=='COMPLETE'
        truth_complete=len(actual)==120
        cert=False;covered=None
        bias=[float('nan')]*3
        if complete:
            center,bias=offset_center(pred['nominal_horizon_outputs'],pred['observed_residual'],window,mode)
            scale=base.fixed_scale(len(center))
            cert=base.safety_decision(center,scale,q,safety=safety)=='SAFE'
            if truth_complete:
                covered=base.directional_coverage(actual,center,scale,q)
        safe=base.actual_safe(actual,safety)
        rows.append({'episode_id':eid,'prediction_complete':int(complete),'truth_complete':int(truth_complete),
                     'certified':int(cert),'physically_safe':int(safe),'false_safe':int(cert and not safe),
                     'covered': '' if covered is None else int(covered),
                     'offset_pressure':bias[0],'offset_level':bias[1],'offset_temperature':bias[2]})
    return rows


TRUTH_CACHE='calibration_truths_verified.npz'


def load_or_regenerate_truths(root,out,workers):
    frozen_ids=[str(x) for x in base.load_frozen_ordinary_inputs(root)['episode_id']]
    cache=out/TRUTH_CACHE
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            if [str(x) for x in saved['episode_id']]==frozen_ids and bool(saved['all_inputs_matched_frozen_calibration_stage']):
                print(f'Reading cached, verified calibration futures from {cache}',flush=True)
                return {eid:np.asarray(f,float) for eid,f in zip(frozen_ids,saved['actual_future'])}
        print(f'{cache} does not match the frozen calibration inputs; regenerating',flush=True)
    print(f'Regenerating {len(frozen_ids)} calibration ordinary calibration futures with {workers} workers ...',flush=True)
    checks,truths=base.regenerate_calibration_truth(root,workers)
    if any(int(c['frozen_input_match'])!=1 for c in checks): raise RuntimeError('Frozen calibration check failed')
    base.write_csv(out/'calibration_regeneration_checks.csv',checks)
    np.savez_compressed(cache,episode_id=np.asarray(frozen_ids),
                        actual_future=np.asarray([truths[e] for e in frozen_ids]),
                        all_inputs_matched_frozen_calibration_stage=np.asarray(True))
    return truths


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment-dir',default='artifacts/flash_study')
    p.add_argument('--output-dir',default='runs/flash_persistent_offset')
    p.add_argument('--workers',type=int,default=max(1,min(16,(os.cpu_count() or 2)-2)))
    p.add_argument('--window',type=int,default=10)
    p.add_argument('--mode',choices=('mean','last'),default='mean')
    p.add_argument('--alpha',type=float,default=0.05)
    a=p.parse_args()
    if not 0<a.alpha<1 or a.window<=0: raise ValueError('Invalid alpha/window')
    root=Path(a.experiment_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    # Verify frozen layout and ensure primary evaluation provides historical residuals before any calibration work.
    required=[root/'calibration_data'/'calibration_ordinary_operational_inputs.npz',
              root/'primary_evaluation_episodes'/'predictions', root/'primary_evaluation_episodes'/'truths']
    missing=[str(x) for x in required if not x.exists()]
    if missing: raise FileNotFoundError('Missing flash artifacts: '+', '.join(missing))
    e5files=sorted((root/'primary_evaluation_episodes'/'predictions').glob('*.npz'))
    if len(e5files)!=1200: raise RuntimeError(f'Expected 1200 primary evaluation predictions, found {len(e5files)}')
    seen_complete=False
    for sample in e5files:
        with np.load(sample,allow_pickle=False) as rec:
            if base.scalar_str(rec['prediction_status'])=='COMPLETE':
                seen_complete=True
                if 'observed_residual' not in rec.files:
                    raise KeyError(f'{sample}: primary evaluation pre-decision residuals absent')
                if np.asarray(rec['observed_residual']).ndim!=2:
                    raise ValueError(f'{sample}: invalid observed residual shape')
                break
    if not seen_complete: raise RuntimeError('No complete primary evaluation prediction exists')
    # calibration stored calibration inputs and scores but not calibration futures. Regenerate them once
    # (verified against the frozen calibration inputs) and cache them; later runs read the cache.
    truths=load_or_regenerate_truths(root,out,a.workers)
    cal=calibration(root,truths,a.window,a.mode)
    q,_=base.corrected_quantile(np.array([r['score'] for r in cal]),a.alpha)
    test=evaluate(root,q,a.window,a.mode)
    n=len(test);cert=sum(r['certified'] for r in test);fs=sum(r['false_safe'] for r in test)
    cov=[int(r['covered']) for r in test if r['covered']!='']
    summ={'method':f'Offset-C_{a.mode}_L{a.window}','alpha':a.alpha,'n_cal_complete':len(cal),
          'n_test_reserved':n,'n_coverage_pairs':len(cov),'quantile':q,'certified_count':cert,
          'availability':cert/n,'false_safe_count':fs,'joint_false_safe':fs/n,
          'conditional_false_safe':fs/cert if cert else float('nan'),
          'coverage_complete_pairs':sum(cov)/len(cov) if cov else float('nan'),
          'certified_domain_exits':sum(int(r['certified']) and not int(r['truth_complete']) for r in test)}
    base.write_csv(out/'offset_calibration_scores.csv',cal)
    base.write_csv(out/'offset_test_decisions.csv',test)
    base.write_csv(out/'offset_summary.csv',[summ])
    print('FLASH Offset-C:',summ)

if __name__=='__main__':main()

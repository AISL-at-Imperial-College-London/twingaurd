#!/usr/bin/env python3
"""Persistent-offset conformal baseline for the primary CSTR evaluation.

Requires the frozen CSTR primary-evaluation artifacts (per-episode prediction and truth files of the primary evaluation).
Reconstructs only PRE-DECISION residual histories from frozen seeds; it does
does not simulate future trajectories, refit the model, or alter primary evaluation results.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import math
import os
import numpy as np
from twinguard.cstr import safety_certification as certification
from twinguard.cstr import episode_design as episode_design

# Pre-decision residual histories are not stored in the frozen primary certification artifacts, so they are
# rebuilt once from the frozen seeds (observation period only, no future rollout) and cached.
HISTORY_CACHE = 'offset_residual_histories.npz'
_CFG = None
_TEMPLATE = None


def _init_worker():
    global _CFG, _TEMPLATE
    _CFG = episode_design.EpisodeDesignConfig()
    _TEMPLATE = episode_design.build_template(_CFG)


def _history(seed):
    _, _, residual, _, _, _ = certification._reconstruct_decision_state(int(seed), _CFG, _TEMPLATE)
    return np.asarray(residual, float)


def load_or_build_histories(folder, out, workers):
    """Return {episode_id: observed residual history} for every calibration and test episode."""
    episodes = []
    for name in ('safety_calibration_manifest.csv', 'final_safety_population_manifest.csv'):
        episodes += [(r['episode_id'], int(float(r['random_seed']))) for r in rows(folder / name)]
    cache = out / HISTORY_CACHE
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            ids, seeds = [str(x) for x in saved['episode_id']], saved['random_seed'].tolist()
            if ids == [e for e, _ in episodes] and seeds == [s for _, s in episodes]:
                print(f'Reading cached residual histories from {cache}', flush=True)
                return dict(zip(ids, saved['residual']))
        print(f'{cache} does not match the frozen manifests; rebuilding', flush=True)
    print(f'Rebuilding {len(episodes)} pre-decision residual histories with {workers} workers ...', flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        histories = list(pool.map(_history, [s for _, s in episodes], chunksize=8))
    np.savez_compressed(cache, episode_id=np.asarray([e for e, _ in episodes]),
                        random_seed=np.asarray([s for _, s in episodes]), residual=np.asarray(histories))
    return {e: h for (e, _), h in zip(episodes, histories)}


def rows(path):
    with path.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))


def qsplit(scores, alpha):
    values=np.sort(np.asarray(scores,float))
    assert values.size and np.all(np.isfinite(values))
    k=math.ceil((len(values)+1)*(1-alpha))
    return float(values[k-1]) if k <= len(values) else float('inf')


def get_history_offset(residual, window, mode):
    if len(residual)<window: raise ValueError(f'History too short: {len(residual)} < {window}')
    tail=np.asarray(residual[-window:],float)
    return float(np.mean(tail) if mode=='mean' else tail[-1])


def evaluate_population(folder, role, histories, window, mode, q=None):
    cal=role=='cal'
    manifest=rows(folder/('safety_calibration_manifest.csv' if cal else 'final_safety_population_manifest.csv'))
    pred_dir=folder/('calibration_predictions' if cal else 'predictions')
    truth_dir=folder/('revealed_calibration_truth' if cal else 'revealed_final_truth')
    expected=500 if cal else 1000
    if len(manifest)!=expected: raise RuntimeError(f'Expected {expected} {role} episodes; got {len(manifest)}')
    out=[]
    for j,r in enumerate(manifest):
        eid=r['episode_id']
        seed=int(float(r['random_seed']))
        if not (pred_dir/f'{eid}.npz').exists() or not (truth_dir/f'{eid}.npz').exists():
            raise FileNotFoundError(f'Missing frozen pair for {eid}')
        with np.load(pred_dir/f'{eid}.npz',allow_pickle=False) as p, np.load(truth_dir/f'{eid}.npz',allow_pickle=False) as t:
            nominal=np.asarray(p['nominal_temperature_k'],float)
            actual=np.asarray(t['actual_physical_temperature_k'],float)
        b=get_history_offset(histories[eid],window,mode)
        center=nominal+b
        if center.shape!=actual.shape: raise RuntimeError(f'Prediction/physical shape mismatch: {eid}')
        score=float(np.max(actual-center))
        if cal:
            out.append({'episode_id':eid,'random_seed':seed,'offset_K':b,'score_K':score})
        else:
            if q is None: raise ValueError('Provide calibrated q for test')
            cert=bool(np.max(center+q)<=certification.T_MAX_K)
            physical_safe=bool(np.max(actual)<=certification.T_MAX_K)
            covered=bool(np.all(actual<=center+q))
            out.append({'episode_id':eid,'random_seed':seed,'offset_K':b,'certified':int(cert),
                        'physical_safe':int(physical_safe),'false_safe':int(cert and not physical_safe),
                        'covered':int(covered)})
        if (j+1)%200==0: print(role,j+1,'/',expected,flush=True)
    return out


def write(path,rr):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--primary-dir',default='artifacts/cstr/primary_evaluation')
    p.add_argument('--output-dir',default='runs/cstr_persistent_offset')
    p.add_argument('--window',type=int,default=10,help='Predeclared last L observed residual samples')
    p.add_argument('--mode',choices=('mean','last'),default='mean')
    p.add_argument('--alpha',type=float,default=0.05)
    p.add_argument('--workers',type=int,default=max(1,min(16,(os.cpu_count() or 2)-2)))
    a=p.parse_args()
    assert a.window>0 and 0<a.alpha<1
    root=Path(a.primary_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    # Verify the immutable primary certification directory layout before reconstructing observation histories.
    required = [root / name for name in ('safety_calibration_manifest.csv', 'final_safety_population_manifest.csv',
                'calibration_predictions', 'revealed_calibration_truth', 'predictions', 'revealed_final_truth')]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError('Missing frozen CSTR artifacts: '+', '.join(missing))
    histories=load_or_build_histories(root,out,a.workers)
    cal=evaluate_population(root,'cal',histories,a.window,a.mode)
    q=qsplit([r['score_K'] for r in cal],a.alpha)
    test=evaluate_population(root,'test',histories,a.window,a.mode,q)
    n=len(test);cert=sum(r['certified'] for r in test);fs=sum(r['false_safe'] for r in test)
    covered=sum(r['covered'] for r in test)
    summary={'method':f'Offset-C_{a.mode}_L{a.window}','n_cal':len(cal),'n_test':n,'alpha':a.alpha,
             'q_K':q,'certified_count':cert,'availability':cert/n,'false_safe_count':fs,
             'joint_false_safe':fs/n,'conditional_false_safe':fs/cert if cert else float('nan'),
             'upper_trajectory_coverage':covered/n,'safe_certified':sum(r['certified'] and r['physical_safe'] for r in test)}
    write(out/'offset_calibration_scores.csv',cal);write(out/'offset_test_decisions.csv',test)
    write(out/'offset_summary.csv',[summary]);print('CSTR Offset-C:',summary)

if __name__=='__main__':main()

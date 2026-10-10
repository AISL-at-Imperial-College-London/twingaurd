from __future__ import annotations

import numpy as np
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from twinguard.cstr.safety_certification import one_sided_scores,reveal_physical_temperature,safety_decision,summarize


def test_one_sided_scores_preserve_asymmetry_and_use_trajectory_maximum() -> None:
    actual=np.array([2.,-4.,1.]);center=np.zeros(3);scale=np.array([1.,2.,.5])
    plus,minus=one_sided_scores(actual,center,scale)
    assert plus==2. and minus==2.


def test_three_way_safety_rule_does_not_force_boundary_overlap() -> None:
    assert safety_decision(np.array([309.,310.]),np.array([311.,312.]))=="SAFE"
    assert safety_decision(np.array([312.,314.]),np.array([314.,316.]))=="UNSAFE"
    assert safety_decision(np.array([311.,312.]),np.array([312.,314.]))=="UNRESOLVED"


def test_truth_value_cannot_change_precomputed_safety_decision() -> None:
    lower=np.array([310.,311.]);upper=np.array([311.,312.])
    before=safety_decision(lower,upper)
    for hidden_truth in (np.array([309.,310.]),np.array([315.,316.])):
        assert hidden_truth.shape==upper.shape
        assert safety_decision(lower,upper)==before


def test_physical_safety_truth_removes_only_the_frozen_measurement_draw() -> None:
    root=Path.cwd()/f".test_certification_physical_truth_{uuid4().hex}";root.mkdir()
    truth=root/"truth.npz";prediction=root/"prediction.npz"
    try:
        seed=1234;physical=np.array([310.,311.,312.]);noise=np.random.default_rng(seed).normal(0,.1,3);measured=physical+noise
        np.savez_compressed(truth,actual_plant_temperature_k=measured);np.savez_compressed(prediction,center=np.zeros(3))
        item=SimpleNamespace(truth_path=truth,hidden_measurement_seed=seed)
        np.testing.assert_allclose(reveal_physical_temperature(item,prediction),physical)
        with np.load(truth) as saved:
            np.testing.assert_allclose(saved["measured_plant_temperature_k"],measured)
            np.testing.assert_allclose(saved["actual_physical_temperature_k"],physical)
    finally:
        for path in (truth,prediction):
            if path.exists():path.unlink()
        root.rmdir()


def test_summary_distinguishes_joint_and_conditional_false_safe() -> None:
    rows=[]
    for i,(decision,actual) in enumerate((("SAFE",0),("SAFE",1),("UNRESOLVED",0),("UNSAFE",0))):
        for method in ("S0_nominal_DT","S1_raw_bootstrap","S2_symmetric_conformal","S3_one_sided"):
            for level in (.9,.95,.99):
                rows.append({"episode_id":str(i),"method":method,"nominal_level":level,"action_category":"interpolation","decision":decision,"actual_safe":actual,"false_safe_indicator":int(decision=="SAFE" and not actual),"false_unsafe_indicator":int(decision=="UNSAFE" and actual),"upper_trajectory_covered":1,"lower_trajectory_covered":1,"mean_envelope_width_k":2.,"upper_inflation_max_k":1.,"total_latency":.1})
    pooled=next(r for r in summarize(rows) if r["method"]=="S3_one_sided" and r["action_category"]=="pooled" and np.isclose(r["nominal_level"],.95))
    assert pooled["joint_false_safe_probability"]==.25
    assert pooled["conditional_false_safe_rate"]==.5
    assert pooled["SAFE_count"]==2

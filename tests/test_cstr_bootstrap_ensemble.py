from __future__ import annotations

import numpy as np

from twinguard.discrepancy.cstr_state_space import StateSpaceParameters
from twinguard.cstr.bootstrap_ensemble import choose_j,parameter_summary


def _parameter(a:float)->StateSpaceParameters:
    return StateSpaceParameters(A=np.array([[a]]),B=np.zeros((1,4)),C=np.ones((1,1)),D=np.zeros((1,4)),Q=np.array([[.1]]),R=.2,initial_mean=np.zeros(1),initial_covariance=np.eye(1),z_mean=np.zeros(4),z_scale=np.ones(4),log_likelihood=1.,iterations=100,converged=True,initialization_seed=91005,fit_seconds=.1,final_parameter_change=1e-5)


def test_parameter_summary_propagates_slow_mode_by_horizon() -> None:
    rows=parameter_summary([_parameter(.98),_parameter(.99),_parameter(.97)]);lookup={r["parameter"]:r for r in rows}
    assert lookup["A_power_240"]["mean"]<lookup["A_power_1"]["mean"]
    assert lookup["spectral_radius"]["maximum"]<1


def test_operational_ensemble_uses_smallest_calibration_equivalent_j() -> None:
    rows=[]
    for j,point,sim,rmse in ((10,.80,.75,.61),(20,.82,.77,.60),(50,.82,.78,.60)):
        rows.append({"experiment_type":"ensemble_size","ensemble_size_J":j,"pointwise_95_coverage":point,"simultaneous_95_coverage":sim,"median_recursive_RMSE_k":rmse})
    assert choose_j(rows)==10

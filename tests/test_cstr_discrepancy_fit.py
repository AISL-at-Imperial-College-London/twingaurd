from __future__ import annotations

from twinguard.cstr.fit_discrepancy_model import (
    FEATURE_NAMES,
    _selection_roles,
    select_dimension,
)


def test_latent_exogenous_features_are_observable_only() -> None:
    assert FEATURE_NAMES == ("u_t","delta_u_t","y_dt_t","elapsed_since_detection_s")
    assert not any(token in name for name in FEATURE_NAMES for token in ("fault","ua","magnitude","concentration","future_true"))


def test_frozen_selection_split_counts() -> None:
    rows=[{"episode_id":f"{c}_{i}","action_category":c} for c in ("interpolation","boundary","extrapolation") for i in range(100)];split=_selection_roles(rows)
    assert sum(r["split_role"]=="model_selection" for r in split)==60
    assert sum(r["split_role"]=="final_assessment" for r in split)==240


def test_dimension_selection_prefers_smallest_near_best() -> None:
    rows=[]
    for k,score,acf in ((1,1.03,.2),(2,1.0,.19),(3,.99,.18),(4,.985,.18)):
        rows.append({"latent_dimension":k,"action_category":"pooled","median_recursive_RMSE_k":score,"maximum_absolute_ACF_lags_1_10":acf})
    selected,_=select_dimension(rows);assert selected==1

from __future__ import annotations

import numpy as np

from twinguard.cstr.stable_arx_order_study import (
    _acf_within_episodes,
    _selection_roles,
    select_order,
)


def test_selection_split_is_predeclared_and_disjoint() -> None:
    rows = [
        {"episode_id": f"e{i}", "action_category": category}
        for category in ("interpolation", "boundary", "extrapolation")
        for i in range(100)
    ]
    split = _selection_roles(rows)
    for category in ("interpolation", "boundary", "extrapolation"):
        category_rows = [row for row in split if row["action_category"] == category]
        assert sum(row["split_role"] == "model_selection" for row in category_rows) == 20
        assert sum(row["split_role"] == "final_assessment" for row in category_rows) == 80


def test_selection_prefers_smallest_order_within_five_percent_of_best() -> None:
    scores = [1.02, 1.00, 0.99, 0.985, 0.98]
    rows = []
    for order, score in enumerate(scores, start=1):
        rows.append({
            "order": order, "split_role": "model_selection", "action_category": "pooled",
            "median_posterior_mean_trajectory_RMSE_k": score,
            "median_stable_draw_acceptance_rate": 0.5,
        })
    selected, _ = select_order(rows)
    assert selected == 1


def test_acf_has_no_cross_episode_boundary_pairs() -> None:
    acf = _acf_within_episodes([np.ones(3), -np.ones(3)], max_lag=1)
    assert np.isclose(acf[1], 2 / 3)

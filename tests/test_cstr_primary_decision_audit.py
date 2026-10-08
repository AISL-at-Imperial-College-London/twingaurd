from __future__ import annotations

import numpy as np

from twinguard.cstr.primary_decision_audit import critical_upper_quantile,decision_from_envelope


def test_critical_q_is_largest_inflation_that_stays_under_limit() -> None:
    center=np.array([310.,312.]);scale=np.array([1.,.25]);q=critical_upper_quantile(center,scale,313.)
    assert q==3.
    assert np.all(center+q*scale<=313.)
    assert np.any(center+(q+1e-6)*scale>313.)


def test_frozen_three_way_decision_reconstruction() -> None:
    assert decision_from_envelope(np.array([310.,311.]),np.array([311.,312.]))=="SAFE"
    assert decision_from_envelope(np.array([312.,313.1]),np.array([314.,315.]))=="UNSAFE"
    assert decision_from_envelope(np.array([311.,312.]),np.array([312.,314.]))=="UNRESOLVED"

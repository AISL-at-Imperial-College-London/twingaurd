from __future__ import annotations

import numpy as np

from twinguard.discrepancy.cstr_state_space import (
    fit_diagonal_state_space_em_checkpoints,
    fit_scalar_state_space_em_checkpoints,
    fit_state_space_em,
    kalman_filter,
    predictive_rollout,
)


def _episodes():
    rng=np.random.default_rng(9);episodes=[]
    for _ in range(8):
        z=rng.normal(size=(40,4));state=0.;y=[]
        for t in range(40):state=.75*state+.2*z[t,0]+rng.normal(0,.08);y.append(state+.1*z[t,2]+rng.normal(0,.05))
        episodes.append((np.asarray(y),z))
    return episodes


def test_em_uses_stable_parameterization_and_episode_resets() -> None:
    parameters=fit_state_space_em(_episodes(),1,13,max_iterations=8)
    assert parameters.spectral_radius<1
    first=kalman_filter(parameters,*_episodes()[0]);second=kalman_filter(parameters,*_episodes()[1])
    assert first.filtered_mean.shape==second.filtered_mean.shape==(40,1)
    assert np.isfinite(first.log_likelihood)


def test_coherent_rollout_matches_analytical_mean() -> None:
    episodes=_episodes();parameters=fit_state_space_em(episodes,1,14,max_iterations=8);filtered=kalman_filter(parameters,*episodes[0]);z=np.zeros((20,4));result=predictive_rollout(parameters,filtered.filtered_mean[-1],filtered.filtered_covariance[-1],z,10000,np.random.default_rng(15))
    assert result["residual_samples"].shape==(10000,20)
    assert result["sample_mean_max_abs_difference"]<.08


def test_batched_scalar_em_matches_episode_loop_formulation() -> None:
    episodes=_episodes()[:5];ordinary=fit_state_space_em(episodes,1,21,max_iterations=4);y=np.stack([x[0] for x in episodes]);z=np.stack([x[1] for x in episodes]);batched=fit_scalar_state_space_em_checkpoints(y,z,21,(4,))[4]
    np.testing.assert_allclose(batched.A,ordinary.A,rtol=2e-5,atol=2e-7)
    np.testing.assert_allclose(batched.B,ordinary.B,rtol=2e-5,atol=2e-7)
    assert abs(batched.log_likelihood-ordinary.log_likelihood)<1e-4
    assert np.isfinite(batched.final_parameter_change)


def test_batched_diagonal_em_matches_episode_loop_for_k2() -> None:
    episodes = _episodes()[:5]
    ordinary = fit_state_space_em(episodes, 2, 22, max_iterations=3)
    y = np.stack([item[0] for item in episodes])
    z = np.stack([item[1] for item in episodes])
    batched = fit_diagonal_state_space_em_checkpoints(y, z, 2, 22, (3,))[3]
    np.testing.assert_allclose(batched.A, ordinary.A, rtol=3e-5, atol=3e-7)
    np.testing.assert_allclose(batched.B, ordinary.B, rtol=3e-5, atol=3e-7)
    np.testing.assert_allclose(batched.C, ordinary.C, rtol=3e-5, atol=3e-7)
    assert abs(batched.log_likelihood - ordinary.log_likelihood) < 1e-4
    assert batched.spectral_radius < 1.0

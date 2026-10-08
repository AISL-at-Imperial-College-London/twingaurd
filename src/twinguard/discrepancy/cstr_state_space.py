"""Stable linear-Gaussian latent discrepancy state-space identification.

The latent coordinates are abstract statistical memory variables.  Dynamics are
parameterized as independent stable modes, a_i = 0.995 tanh(theta_i), which
guarantees stability throughout optimization rather than clipping a fitted A.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


STABILITY_LIMIT = 0.995


@dataclass(frozen=True)
class StateSpaceParameters:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: float
    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    z_mean: np.ndarray
    z_scale: np.ndarray
    log_likelihood: float
    iterations: int
    converged: bool
    initialization_seed: int
    fit_seconds: float
    final_parameter_change: float = float("nan")

    @property
    def latent_dimension(self) -> int:
        return int(self.A.shape[0])

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.A))))

    def standardize(self, z: np.ndarray) -> np.ndarray:
        values=np.asarray(z,dtype=float)
        if values.shape[-1] != self.z_mean.size:
            raise ValueError("exogenous feature dimension mismatch")
        return (values-self.z_mean)/self.z_scale


@dataclass(frozen=True)
class FilterResult:
    filtered_mean: np.ndarray
    filtered_covariance: np.ndarray
    predicted_mean: np.ndarray
    predicted_covariance: np.ndarray
    innovations: np.ndarray
    innovation_variance: np.ndarray
    standardized_innovations: np.ndarray
    log_likelihood: float


@dataclass(frozen=True)
class SmootherResult:
    mean: np.ndarray
    covariance: np.ndarray
    lag_covariance: np.ndarray


def _symmetrize(matrix: np.ndarray, floor: float=1e-10) -> np.ndarray:
    value=0.5*(matrix+matrix.T)
    eigenvalues,eigenvectors=np.linalg.eigh(value)
    return (eigenvectors*np.maximum(eigenvalues,floor))@eigenvectors.T


def kalman_filter(parameters:StateSpaceParameters,residual:np.ndarray,z_raw:np.ndarray)->FilterResult:
    y=np.asarray(residual,dtype=float);z=parameters.standardize(np.asarray(z_raw,dtype=float))
    if y.ndim!=1 or z.ndim!=2 or z.shape[0]!=y.size:
        raise ValueError("residual and exogenous sequences must align")
    n=y.size;k=parameters.latent_dimension
    mf=np.empty((n,k));pf=np.empty((n,k,k));mp=np.empty((n,k));pp=np.empty((n,k,k));innov=np.empty(n);svar=np.empty(n);loglik=0.0
    for t in range(n):
        if t==0:
            prior_mean=parameters.initial_mean;prior_cov=parameters.initial_covariance
        else:
            prior_mean=parameters.A@mf[t-1]+parameters.B@z[t]
            prior_cov=_symmetrize(parameters.A@pf[t-1]@parameters.A.T+parameters.Q)
        expected=float((parameters.C@prior_mean+parameters.D@z[t]).item());variance=float((parameters.C@prior_cov@parameters.C.T).item()+parameters.R);variance=max(variance,1e-10)
        innovation=y[t]-expected;gain=(prior_cov@parameters.C.T).reshape(k)/variance
        posterior_mean=prior_mean+gain*innovation
        posterior_cov=_symmetrize(prior_cov-np.outer(gain,parameters.C@prior_cov))
        mp[t]=prior_mean;pp[t]=prior_cov;mf[t]=posterior_mean;pf[t]=posterior_cov;innov[t]=innovation;svar[t]=variance
        loglik+=-0.5*(np.log(2*np.pi*variance)+innovation*innovation/variance)
    return FilterResult(mf,pf,mp,pp,innov,svar,innov/np.sqrt(svar),float(loglik))


def rts_smoother(parameters:StateSpaceParameters,filtered:FilterResult)->SmootherResult:
    mean=filtered.filtered_mean.copy();cov=filtered.filtered_covariance.copy();n,k=mean.shape;lag=np.zeros((n,k,k))
    for t in range(n-2,-1,-1):
        gain=filtered.filtered_covariance[t]@parameters.A.T@np.linalg.inv(filtered.predicted_covariance[t+1])
        mean[t]=filtered.filtered_mean[t]+gain@(mean[t+1]-filtered.predicted_mean[t+1])
        cov[t]=_symmetrize(filtered.filtered_covariance[t]+gain@(cov[t+1]-filtered.predicted_covariance[t+1])@gain.T)
        lag[t+1]=cov[t+1]@gain.T
    return SmootherResult(mean,cov,lag)


def _initial_parameters(episodes:Sequence[tuple[np.ndarray,np.ndarray]],k:int,seed:int,z_mean:np.ndarray,z_scale:np.ndarray)->StateSpaceParameters:
    rng=np.random.default_rng(seed);all_y=np.concatenate([e[0] for e in episodes]);all_z=np.vstack([(e[1]-z_mean)/z_scale for e in episodes]);design=np.column_stack((np.ones(all_z.shape[0]),all_z));ols=np.linalg.lstsq(design,all_y,rcond=None)[0];residual=all_y-design@ols;variance=max(float(np.var(residual)),1e-3)
    modes=np.linspace(.25,.85,k)+rng.normal(0,.03,k);modes=np.clip(modes,-.95,.95)
    return StateSpaceParameters(A=np.diag(modes),B=rng.normal(0,.02,(k,all_z.shape[1])),C=np.full((1,k),1/max(k,1)),D=ols[1:].reshape(1,-1),Q=np.eye(k)*variance*.08,R=variance*.5,initial_mean=np.zeros(k),initial_covariance=np.eye(k)*variance,z_mean=z_mean,z_scale=z_scale,log_likelihood=-np.inf,iterations=0,converged=False,initialization_seed=seed,fit_seconds=0.0)


def _stable_dynamic_row(sxx:np.ndarray,syx:np.ndarray,syy:float,start_a:float,start_b:np.ndarray)->tuple[float,np.ndarray,float]:
    theta0=np.arctanh(np.clip(start_a/STABILITY_LIMIT,-.999,.999));initial=np.concatenate(([theta0],start_b))
    def objective(params:np.ndarray)->float:
        coefficients=np.concatenate(([STABILITY_LIMIT*np.tanh(params[0])],params[1:]));return float(syy-2*coefficients@syx+coefficients@sxx@coefficients)
    result=minimize(objective,initial,method="L-BFGS-B",options={"maxiter":100,"ftol":1e-12})
    a=float(STABILITY_LIMIT*np.tanh(result.x[0]));b=np.asarray(result.x[1:]);return a,b,max(objective(result.x),1e-10)


def _em_update(parameters:StateSpaceParameters,episodes:Sequence[tuple[np.ndarray,np.ndarray]])->tuple[StateSpaceParameters,float]:
    k=parameters.latent_dimension;p=parameters.z_mean.size
    dynamic_sxx=[np.zeros((p+1,p+1)) for _ in range(k)];dynamic_syx=[np.zeros(p+1) for _ in range(k)];dynamic_syy=np.zeros(k);transition_count=0
    observation_sxx=np.zeros((k+p,k+p));observation_syx=np.zeros(k+p);observation_syy=0.0;observation_count=0
    initial_second=np.zeros((k,k));initial_sum=np.zeros(k);episode_count=0;total_loglik=0.0
    for y,z_raw in episodes:
        z=parameters.standardize(z_raw);filtered=kalman_filter(parameters,y,z_raw);smooth=rts_smoother(parameters,filtered);total_loglik+=filtered.log_likelihood;episode_count+=1
        initial_sum+=smooth.mean[0];initial_second+=smooth.covariance[0]+np.outer(smooth.mean[0],smooth.mean[0])
        for t in range(y.size):
            ex=np.concatenate((smooth.mean[t],z[t]));exx=np.block([[smooth.covariance[t]+np.outer(smooth.mean[t],smooth.mean[t]),np.outer(smooth.mean[t],z[t])],[np.outer(z[t],smooth.mean[t]),np.outer(z[t],z[t])]])
            observation_sxx+=exx;observation_syx+=float(y[t])*ex;observation_syy+=float(y[t]*y[t]);observation_count+=1
            if t==0:continue
            transition_count+=1
            for index in range(k):
                qmean=np.concatenate(([smooth.mean[t-1,index]],z[t]));qxx=np.zeros((p+1,p+1));qxx[0,0]=smooth.covariance[t-1,index,index]+smooth.mean[t-1,index]**2;qxx[0,1:]=smooth.mean[t-1,index]*z[t];qxx[1:,0]=qxx[0,1:];qxx[1:,1:]=np.outer(z[t],z[t])
                cross=float(smooth.lag_covariance[t,index,index]+smooth.mean[t,index]*smooth.mean[t-1,index]);yx=np.concatenate(([cross],smooth.mean[t,index]*z[t]));yy=float(smooth.covariance[t,index,index]+smooth.mean[t,index]**2)
                dynamic_sxx[index]+=qxx;dynamic_syx[index]+=yx;dynamic_syy[index]+=yy
    A=np.zeros((k,k));B=np.zeros((k,p));qdiag=np.zeros(k)
    for index in range(k):
        a,b,sse=_stable_dynamic_row(dynamic_sxx[index],dynamic_syx[index],dynamic_syy[index],parameters.A[index,index],parameters.B[index]);A[index,index]=a;B[index]=b;qdiag[index]=max(sse/transition_count,1e-7)
    H=observation_syx@np.linalg.pinv(observation_sxx);C=H[:k].reshape(1,k);D=H[k:].reshape(1,p);R=max(float((observation_syy-H@observation_syx)/observation_count),1e-7)
    m0=initial_sum/episode_count;P0=_symmetrize(initial_second/episode_count-np.outer(m0,m0),1e-7)
    updated=StateSpaceParameters(A=A,B=B,C=C,D=D,Q=np.diag(qdiag),R=R,initial_mean=m0,initial_covariance=P0,z_mean=parameters.z_mean,z_scale=parameters.z_scale,log_likelihood=total_loglik,iterations=parameters.iterations+1,converged=False,initialization_seed=parameters.initialization_seed,fit_seconds=0.0)
    return updated,total_loglik


def fit_state_space_em(episodes:Sequence[tuple[np.ndarray,np.ndarray]],latent_dimension:int,initialization_seed:int,max_iterations:int=30,relative_tolerance:float=1e-4)->StateSpaceParameters:
    if latent_dimension not in (1,2,3,4):raise ValueError("latent dimension must be one of 1,2,3,4")
    if not episodes:raise ValueError("at least one training episode is required")
    z_all=np.vstack([np.asarray(z,dtype=float) for _,z in episodes]);z_mean=np.mean(z_all,axis=0);z_scale=np.std(z_all,axis=0,ddof=0);z_scale=np.where(z_scale>1e-12,z_scale,1.0)
    normalized=[(np.asarray(y,dtype=float),np.asarray(z,dtype=float)) for y,z in episodes];parameters=_initial_parameters(normalized,latent_dimension,initialization_seed,z_mean,z_scale);start=time.perf_counter();previous=-np.inf;converged=False
    final_parameter_change=float("nan")
    for _ in range(max_iterations):
        before=parameters
        parameters,likelihood=_em_update(parameters,normalized)
        old=np.concatenate((before.A.ravel(),before.B.ravel(),before.C.ravel(),before.D.ravel(),before.Q.ravel(),[before.R],before.initial_mean.ravel(),before.initial_covariance.ravel()))
        new=np.concatenate((parameters.A.ravel(),parameters.B.ravel(),parameters.C.ravel(),parameters.D.ravel(),parameters.Q.ravel(),[parameters.R],parameters.initial_mean.ravel(),parameters.initial_covariance.ravel()))
        final_parameter_change=float(np.linalg.norm(new-old)/max(1.0,np.linalg.norm(old)))
        if np.isfinite(previous) and abs(likelihood-previous)<=relative_tolerance*(1+abs(previous)):
            converged=True;break
        previous=likelihood
    # Evaluate the likelihood under the final updated parameters.
    final_likelihood=float(sum(kalman_filter(parameters,y,z).log_likelihood for y,z in normalized));elapsed=time.perf_counter()-start
    order=np.argsort(np.diag(parameters.A));A=parameters.A[np.ix_(order,order)];B=parameters.B[order];C=parameters.C[:,order];Q=parameters.Q[np.ix_(order,order)];m0=parameters.initial_mean[order];P0=parameters.initial_covariance[np.ix_(order,order)]
    result=StateSpaceParameters(A=A,B=B,C=C,D=parameters.D,Q=Q,R=parameters.R,initial_mean=m0,initial_covariance=P0,z_mean=z_mean,z_scale=z_scale,log_likelihood=final_likelihood,iterations=parameters.iterations,converged=converged,initialization_seed=initialization_seed,fit_seconds=elapsed,final_parameter_change=final_parameter_change)
    if result.spectral_radius>=1:raise RuntimeError("stable parameterization invariant failed")
    return result


def fit_scalar_state_space_em_checkpoints(
    residual: np.ndarray,
    z_raw: np.ndarray,
    initialization_seed: int,
    checkpoints: Sequence[int] = (15, 30, 60, 100),
    relative_tolerance: float = 1e-4,
) -> dict[int, StateSpaceParameters]:
    """Fit the exact k=1 EM formulation and retain requested iteration checkpoints.

    Episodes must have a common length.  The E step is batched across independent
    episode resets; this is algebraically equivalent to :func:`_em_update` for
    ``k=1`` but avoids Python loops over hundreds of training episodes.
    """
    y=np.asarray(residual,dtype=float);z0=np.asarray(z_raw,dtype=float)
    if y.ndim!=2 or z0.ndim!=3 or z0.shape[:2]!=y.shape:
        raise ValueError("batched residual and exogenous arrays must align")
    requested=tuple(sorted(set(int(x) for x in checkpoints)))
    if not requested or requested[0]<=0:raise ValueError("checkpoints must be positive")
    e,t=y.shape;p=z0.shape[2];z_mean=np.mean(z0,axis=(0,1));z_scale=np.std(z0,axis=(0,1));z_scale=np.where(z_scale>1e-12,z_scale,1.0);z=(z0-z_mean)/z_scale
    episodes=[(y[i],z0[i]) for i in range(e)]
    current=_initial_parameters(episodes,1,initialization_seed,z_mean,z_scale)
    started=time.perf_counter();previous=-np.inf;converged=False;outputs={};last_change=float("nan")
    for iteration in range(1,requested[-1]+1):
        a=float(current.A[0,0]);b=current.B[0];c=float(current.C[0,0]);d=current.D[0];q=float(current.Q[0,0]);r=float(current.R)
        mf=np.empty((e,t));mp=np.empty((e,t));innov=np.empty((e,t));pf=np.empty(t);pp=np.empty(t);svar=np.empty(t);loglik=0.0
        for j in range(t):
            if j==0:pm=np.full(e,float(current.initial_mean[0]));pv=float(current.initial_covariance[0,0])
            else:pm=a*mf[:,j-1]+z[:,j]@b;pv=a*a*pf[j-1]+q
            pred=c*pm+z[:,j]@d;s=c*c*pv+r;nu=y[:,j]-pred;gain=pv*c/s
            mp[:,j]=pm;pp[j]=pv;mf[:,j]=pm+gain*nu;pf[j]=max(pv-gain*c*pv,1e-10);innov[:,j]=nu
            loglik+=float(np.sum(-.5*(np.log(2*np.pi*s)+nu*nu/s)));svar[j]=s
        sm=mf.copy();sp=pf.copy();lag=np.zeros(t)
        for j in range(t-2,-1,-1):
            gain=pf[j]*a/pp[j+1];sm[:,j]=mf[:,j]+gain*(sm[:,j+1]-mp[:,j+1]);sp[j]=max(pf[j]+gain*gain*(sp[j+1]-pp[j+1]),1e-10);lag[j+1]=sp[j+1]*gain
        # Observation sufficient statistics for x=[state,z].
        obs_xx=np.zeros((1+p,1+p));obs_yx=np.zeros(1+p)
        obs_xx[0,0]=float(e*np.sum(sp)+np.sum(sm*sm));obs_xx[0,1:]=np.einsum('et,etp->p',sm,z);obs_xx[1:,0]=obs_xx[0,1:];obs_xx[1:,1:]=np.einsum('etp,etq->pq',z,z)
        obs_yx[0]=float(np.sum(y*sm));obs_yx[1:]=np.einsum('et,etp->p',y,z);obs_yy=float(np.sum(y*y));count=e*t
        h=obs_yx@np.linalg.pinv(obs_xx);new_c=float(h[0]);new_d=h[1:];new_r=max(float((obs_yy-h@obs_yx)/count),1e-7)
        # Transition sufficient statistics for x=[previous state,z_t].
        zp=z[:,1:];prev=sm[:,:-1];curr=sm[:,1:];dyn_xx=np.zeros((1+p,1+p));dyn_yx=np.zeros(1+p)
        dyn_xx[0,0]=float(e*np.sum(sp[:-1])+np.sum(prev*prev));dyn_xx[0,1:]=np.einsum('et,etp->p',prev,zp);dyn_xx[1:,0]=dyn_xx[0,1:];dyn_xx[1:,1:]=np.einsum('etp,etq->pq',zp,zp)
        dyn_yx[0]=float(e*np.sum(lag[1:])+np.sum(curr*prev));dyn_yx[1:]=np.einsum('et,etp->p',curr,zp);dyn_yy=float(e*np.sum(sp[1:])+np.sum(curr*curr))
        new_a,new_b,sse=_stable_dynamic_row(dyn_xx,dyn_yx,dyn_yy,a,b);new_q=max(sse/(e*(t-1)),1e-7)
        new_m0=np.array([float(np.mean(sm[:,0]))]);new_p0=np.array([[max(float(sp[0]+np.mean(sm[:,0]**2)-new_m0[0]**2),1e-7)]])
        updated=StateSpaceParameters(A=np.array([[new_a]]),B=new_b[None,:],C=np.array([[new_c]]),D=new_d[None,:],Q=np.array([[new_q]]),R=new_r,initial_mean=new_m0,initial_covariance=new_p0,z_mean=z_mean,z_scale=z_scale,log_likelihood=loglik,iterations=iteration,converged=False,initialization_seed=initialization_seed,fit_seconds=time.perf_counter()-started)
        old_vector=np.concatenate((current.A.ravel(),current.B.ravel(),current.C.ravel(),current.D.ravel(),current.Q.ravel(),[current.R],current.initial_mean,current.initial_covariance.ravel()))
        new_vector=np.concatenate((updated.A.ravel(),updated.B.ravel(),updated.C.ravel(),updated.D.ravel(),updated.Q.ravel(),[updated.R],updated.initial_mean,updated.initial_covariance.ravel()))
        last_change=float(np.linalg.norm(new_vector-old_vector)/max(1.0,np.linalg.norm(old_vector)))
        if np.isfinite(previous) and abs(loglik-previous)<=relative_tolerance*(1+abs(previous)):converged=True
        previous=loglik;current=updated
        if iteration in requested:
            final_ll=float(sum(kalman_filter(current,y[i],z0[i]).log_likelihood for i in range(e)))
            outputs[iteration]=StateSpaceParameters(**{**current.__dict__,"log_likelihood":final_ll,"converged":converged,"fit_seconds":time.perf_counter()-started,"final_parameter_change":last_change})
        # Continue after first convergence so every requested limit has an
        # explicit reproducibility record; convergence flags remain sticky.
    return outputs


def fit_diagonal_state_space_em_checkpoints(
    residual: np.ndarray,
    z_raw: np.ndarray,
    latent_dimension: int,
    initialization_seed: int,
    checkpoints: Sequence[int] = (30, 60, 100),
    relative_tolerance: float = 1e-4,
) -> dict[int, StateSpaceParameters]:
    """Batched EM checkpoints for a small diagonal latent-state model.

    This extends the optimized scalar routine to the same diagonal stable-mode
    parameterization used by :func:`fit_state_space_em`. Episode lengths must be
    common so filtering covariances can be shared across the batch.
    """

    y = np.asarray(residual, dtype=float)
    z0 = np.asarray(z_raw, dtype=float)
    if y.ndim != 2 or z0.ndim != 3 or z0.shape[:2] != y.shape:
        raise ValueError("batched residual and exogenous arrays must align")
    if latent_dimension < 1:
        raise ValueError("latent_dimension must be positive")
    requested = tuple(sorted(set(int(value) for value in checkpoints)))
    if not requested or requested[0] <= 0:
        raise ValueError("checkpoints must be positive")
    episode_count, steps = y.shape
    feature_count = z0.shape[2]
    z_mean = np.mean(z0, axis=(0, 1))
    z_scale = np.std(z0, axis=(0, 1))
    z_scale = np.where(z_scale > 1e-12, z_scale, 1.0)
    z = (z0 - z_mean) / z_scale
    episodes = [(y[index], z0[index]) for index in range(episode_count)]
    current = _initial_parameters(
        episodes, latent_dimension, initialization_seed, z_mean, z_scale
    )
    started = time.perf_counter()
    previous = -np.inf
    converged = False
    outputs: dict[int, StateSpaceParameters] = {}
    last_change = float("nan")

    for iteration in range(1, requested[-1] + 1):
        A, B, C, D, Q, R = current.A, current.B, current.C, current.D, current.Q, current.R
        filtered_mean = np.empty((episode_count, steps, latent_dimension))
        predicted_mean = np.empty_like(filtered_mean)
        filtered_covariance = np.empty((steps, latent_dimension, latent_dimension))
        predicted_covariance = np.empty_like(filtered_covariance)
        innovations = np.empty((episode_count, steps))
        log_likelihood = 0.0
        for step in range(steps):
            if step == 0:
                prior_mean = np.broadcast_to(current.initial_mean, (episode_count, latent_dimension))
                prior_covariance = current.initial_covariance
            else:
                prior_mean = filtered_mean[:, step - 1] @ A.T + z[:, step] @ B.T
                prior_covariance = _symmetrize(
                    A @ filtered_covariance[step - 1] @ A.T + Q
                )
            expected = (prior_mean @ C.T).ravel() + (z[:, step] @ D.T).ravel()
            variance = max(float((C @ prior_covariance @ C.T).item() + R), 1e-10)
            innovation = y[:, step] - expected
            gain = (prior_covariance @ C.T).reshape(latent_dimension) / variance
            posterior_mean = prior_mean + innovation[:, None] * gain[None, :]
            posterior_covariance = _symmetrize(
                prior_covariance - np.outer(gain, C @ prior_covariance)
            )
            predicted_mean[:, step] = prior_mean
            predicted_covariance[step] = prior_covariance
            filtered_mean[:, step] = posterior_mean
            filtered_covariance[step] = posterior_covariance
            innovations[:, step] = innovation
            log_likelihood += float(
                np.sum(-0.5 * (np.log(2.0 * np.pi * variance) + innovation**2 / variance))
            )

        smoothed_mean = filtered_mean.copy()
        smoothed_covariance = filtered_covariance.copy()
        lag_covariance = np.zeros_like(filtered_covariance)
        for step in range(steps - 2, -1, -1):
            gain = (
                filtered_covariance[step]
                @ A.T
                @ np.linalg.inv(predicted_covariance[step + 1])
            )
            smoothed_mean[:, step] = filtered_mean[:, step] + (
                smoothed_mean[:, step + 1] - predicted_mean[:, step + 1]
            ) @ gain.T
            smoothed_covariance[step] = _symmetrize(
                filtered_covariance[step]
                + gain
                @ (smoothed_covariance[step + 1] - predicted_covariance[step + 1])
                @ gain.T
            )
            lag_covariance[step + 1] = smoothed_covariance[step + 1] @ gain.T

        total_dimension = latent_dimension + feature_count
        observation_xx = np.zeros((total_dimension, total_dimension))
        observation_yx = np.zeros(total_dimension)
        observation_xx[:latent_dimension, :latent_dimension] = (
            episode_count * np.sum(smoothed_covariance, axis=0)
            + np.einsum("etk,etl->kl", smoothed_mean, smoothed_mean)
        )
        state_feature = np.einsum("etk,etp->kp", smoothed_mean, z)
        observation_xx[:latent_dimension, latent_dimension:] = state_feature
        observation_xx[latent_dimension:, :latent_dimension] = state_feature.T
        observation_xx[latent_dimension:, latent_dimension:] = np.einsum(
            "etp,etq->pq", z, z
        )
        observation_yx[:latent_dimension] = np.einsum("et,etk->k", y, smoothed_mean)
        observation_yx[latent_dimension:] = np.einsum("et,etp->p", y, z)
        observation_yy = float(np.sum(y * y))
        observation_count = episode_count * steps
        observation_coefficients = observation_yx @ np.linalg.pinv(observation_xx)
        new_C = observation_coefficients[:latent_dimension].reshape(1, latent_dimension)
        new_D = observation_coefficients[latent_dimension:].reshape(1, feature_count)
        new_R = max(
            float(
                (observation_yy - observation_coefficients @ observation_yx)
                / observation_count
            ),
            1e-7,
        )

        new_A = np.zeros((latent_dimension, latent_dimension))
        new_B = np.zeros((latent_dimension, feature_count))
        new_q = np.zeros(latent_dimension)
        future_z = z[:, 1:]
        transition_count = episode_count * (steps - 1)
        for state_index in range(latent_dimension):
            previous_state = smoothed_mean[:, :-1, state_index]
            current_state = smoothed_mean[:, 1:, state_index]
            dynamic_xx = np.zeros((1 + feature_count, 1 + feature_count))
            dynamic_xx[0, 0] = float(
                episode_count * np.sum(smoothed_covariance[:-1, state_index, state_index])
                + np.sum(previous_state**2)
            )
            dynamic_xx[0, 1:] = np.einsum("et,etp->p", previous_state, future_z)
            dynamic_xx[1:, 0] = dynamic_xx[0, 1:]
            dynamic_xx[1:, 1:] = np.einsum("etp,etq->pq", future_z, future_z)
            dynamic_yx = np.empty(1 + feature_count)
            dynamic_yx[0] = float(
                episode_count * np.sum(lag_covariance[1:, state_index, state_index])
                + np.sum(current_state * previous_state)
            )
            dynamic_yx[1:] = np.einsum("et,etp->p", current_state, future_z)
            dynamic_yy = float(
                episode_count * np.sum(smoothed_covariance[1:, state_index, state_index])
                + np.sum(current_state**2)
            )
            a_value, b_value, sse = _stable_dynamic_row(
                dynamic_xx,
                dynamic_yx,
                dynamic_yy,
                current.A[state_index, state_index],
                current.B[state_index],
            )
            new_A[state_index, state_index] = a_value
            new_B[state_index] = b_value
            new_q[state_index] = max(sse / transition_count, 1e-7)

        new_initial_mean = np.mean(smoothed_mean[:, 0], axis=0)
        centered_initial = smoothed_mean[:, 0] - new_initial_mean
        new_initial_covariance = _symmetrize(
            smoothed_covariance[0]
            + centered_initial.T @ centered_initial / episode_count,
            1e-7,
        )
        updated = StateSpaceParameters(
            A=new_A,
            B=new_B,
            C=new_C,
            D=new_D,
            Q=np.diag(new_q),
            R=new_R,
            initial_mean=new_initial_mean,
            initial_covariance=new_initial_covariance,
            z_mean=z_mean,
            z_scale=z_scale,
            log_likelihood=log_likelihood,
            iterations=iteration,
            converged=False,
            initialization_seed=initialization_seed,
            fit_seconds=time.perf_counter() - started,
        )
        old_vector = np.concatenate(
            (
                current.A.ravel(), current.B.ravel(), current.C.ravel(), current.D.ravel(),
                current.Q.ravel(), [current.R], current.initial_mean, current.initial_covariance.ravel(),
            )
        )
        new_vector = np.concatenate(
            (
                updated.A.ravel(), updated.B.ravel(), updated.C.ravel(), updated.D.ravel(),
                updated.Q.ravel(), [updated.R], updated.initial_mean, updated.initial_covariance.ravel(),
            )
        )
        last_change = float(
            np.linalg.norm(new_vector - old_vector) / max(1.0, np.linalg.norm(old_vector))
        )
        if np.isfinite(previous) and abs(log_likelihood - previous) <= relative_tolerance * (
            1.0 + abs(previous)
        ):
            converged = True
        previous = log_likelihood
        current = updated
        if iteration in requested:
            final_likelihood = float(
                sum(kalman_filter(current, y[index], z0[index]).log_likelihood for index in range(episode_count))
            )
            outputs[iteration] = StateSpaceParameters(
                **{
                    **current.__dict__,
                    "log_likelihood": final_likelihood,
                    "converged": converged,
                    "fit_seconds": time.perf_counter() - started,
                    "final_parameter_change": last_change,
                }
            )
    return outputs


def fit_multiple_initializations(episodes:Sequence[tuple[np.ndarray,np.ndarray]],latent_dimension:int,seeds:Sequence[int],max_iterations:int=30)->tuple[StateSpaceParameters,list[StateSpaceParameters]]:
    candidates=[fit_state_space_em(episodes,latent_dimension,seed,max_iterations=max_iterations) for seed in seeds]
    return max(candidates,key=lambda item:item.log_likelihood),candidates


def predictive_rollout(parameters:StateSpaceParameters,filtered_mean:np.ndarray,filtered_covariance:np.ndarray,z_future_raw:np.ndarray,sample_count:int,rng:np.random.Generator)->dict[str,np.ndarray|float]:
    z=parameters.standardize(np.asarray(z_future_raw,dtype=float));horizon=z.shape[0];k=parameters.latent_dimension
    mean_state=np.asarray(filtered_mean,dtype=float).copy();cov_state=np.asarray(filtered_covariance,dtype=float).copy();analytical_mean=np.empty(horizon);analytical_variance=np.empty(horizon)
    for h in range(horizon):
        mean_state=parameters.A@mean_state+parameters.B@z[h];cov_state=_symmetrize(parameters.A@cov_state@parameters.A.T+parameters.Q);analytical_mean[h]=float((parameters.C@mean_state+parameters.D@z[h]).item());analytical_variance[h]=float((parameters.C@cov_state@parameters.C.T).item()+parameters.R)
    start=time.perf_counter();state=rng.multivariate_normal(filtered_mean,filtered_covariance,size=sample_count);paths=np.empty((sample_count,horizon));q_factor=np.linalg.cholesky(_symmetrize(parameters.Q));r_sd=np.sqrt(parameters.R)
    for h in range(horizon):
        state=state@parameters.A.T+z[h]@parameters.B.T+rng.normal(size=(sample_count,k))@q_factor.T;paths[:,h]=state@parameters.C.reshape(k)+float((parameters.D@z[h]).item())+rng.normal(size=sample_count)*r_sd
    monte_carlo_seconds=time.perf_counter()-start
    return {"residual_samples":paths,"analytical_mean":analytical_mean,"analytical_variance":analytical_variance,"monte_carlo_seconds":monte_carlo_seconds,"sample_mean_max_abs_difference":float(np.max(np.abs(np.mean(paths,axis=0)-analytical_mean)))}

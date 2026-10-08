"""Minimal conjugate Gaussian model for scalar model--plant residuals.

The physical fault and CSTR state are intentionally absent from this module.  The
model learns only from a one-dimensional array of observed residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import invgamma


@dataclass(frozen=True)
class GaussianPosterior:
    """Normal--inverse-gamma posterior for a Gaussian mean and variance."""

    n_observations: int
    mean: float
    kappa: float
    alpha: float
    beta: float

    @property
    def expected_residual_variance(self) -> float:
        """Posterior expectation of the conditional residual variance."""

        if self.alpha <= 1.0:
            return float("nan")
        return float(self.beta / (self.alpha - 1.0))

    @property
    def expected_residual_std(self) -> float:
        return float(np.sqrt(self.expected_residual_variance))

    @property
    def posterior_predictive_variance(self) -> float:
        """Variance of one future residual after parameter uncertainty."""

        return float(
            self.expected_residual_variance * (self.kappa + 1.0) / self.kappa
        )


@dataclass(frozen=True)
class BayesianGaussianResidualModel:
    """Scalar Gaussian likelihood with a normal--inverse-gamma prior.

    Prior parameterization::

        sigma^2 ~ InvGamma(alpha0, beta0)
        mu | sigma^2 ~ Normal(mu0, sigma^2 / kappa0)

    The defaults are weak on the mean while retaining a proper variance prior.
    """

    prior_mean: float = 0.0
    prior_kappa: float = 1.0e-3
    prior_alpha: float = 2.0
    prior_beta: float = 0.25

    def fit(self, residuals: np.ndarray) -> GaussianPosterior:
        values = np.asarray(residuals, dtype=float)
        if values.ndim != 1:
            raise ValueError("residuals must be a one-dimensional array")
        if values.size < 2:
            raise ValueError("at least two residual observations are required")
        if not np.all(np.isfinite(values)):
            raise ValueError("residuals must contain only finite values")
        if self.prior_kappa <= 0.0 or self.prior_alpha <= 1.0:
            raise ValueError("prior_kappa must be positive and prior_alpha must exceed 1")
        if self.prior_beta <= 0.0:
            raise ValueError("prior_beta must be positive")

        n = int(values.size)
        sample_mean = float(np.mean(values))
        centered_sum_squares = float(np.sum((values - sample_mean) ** 2))
        kappa_n = self.prior_kappa + n
        mean_n = (
            self.prior_kappa * self.prior_mean + n * sample_mean
        ) / kappa_n
        alpha_n = self.prior_alpha + 0.5 * n
        beta_n = (
            self.prior_beta
            + 0.5 * centered_sum_squares
            + 0.5
            * (self.prior_kappa * n / kappa_n)
            * (sample_mean - self.prior_mean) ** 2
        )
        return GaussianPosterior(
            n_observations=n,
            mean=float(mean_n),
            kappa=float(kappa_n),
            alpha=float(alpha_n),
            beta=float(beta_n),
        )

    @staticmethod
    def sample_constant_biases(
        posterior: GaussianPosterior,
        n_rollouts: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw one residual bias per rollout.

        Each returned scalar is held constant for its entire trajectory.  This is
        deliberately the simplest rollout assumption; no temporal dynamics are
        learned here.
        """

        if n_rollouts <= 0:
            raise ValueError("n_rollouts must be positive")
        variances = invgamma.rvs(
            posterior.alpha,
            scale=posterior.beta,
            size=int(n_rollouts),
            random_state=rng,
        )
        means = rng.normal(
            loc=posterior.mean,
            scale=np.sqrt(variances / posterior.kappa),
        )
        return np.asarray(
            rng.normal(loc=means, scale=np.sqrt(variances)), dtype=float
        )

    @classmethod
    def corrected_rollouts(
        cls,
        nominal_trajectory: np.ndarray,
        posterior: GaussianPosterior,
        n_rollouts: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        nominal = np.asarray(nominal_trajectory, dtype=float)
        if nominal.ndim != 1:
            raise ValueError("nominal_trajectory must be one-dimensional")
        biases = cls.sample_constant_biases(posterior, n_rollouts, rng)
        return nominal[None, :] + biases[:, None]

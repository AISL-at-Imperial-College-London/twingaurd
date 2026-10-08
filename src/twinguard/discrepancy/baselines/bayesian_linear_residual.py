"""Transparent conjugate Bayesian linear models for residual dynamics.

The learner is deliberately restricted to online-observable quantities.  Hidden
fault metadata is rejected by name before a model can be constructed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import invgamma, t as student_t


ALLOWED_FEATURES = frozenset(
    {"r_t", "delta_r_t", "u_t", "delta_u_t", "y_dt_next"}
)
MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "M2_residual_ar": ("r_t", "delta_r_t"),
    "M3_residual_arx": ("r_t", "delta_r_t", "u_t", "delta_u_t"),
    "M4_full_bayesian": (
        "r_t",
        "delta_r_t",
        "u_t",
        "delta_u_t",
        "y_dt_next",
    ),
}
FORBIDDEN_FEATURE_TOKENS = (
    "fault",
    "ua",
    "fouling",
    "hidden",
    "parameter",
    "magnitude",
    "plant_state",
)


def assert_online_feature_names(feature_names: tuple[str, ...]) -> None:
    """Reject any feature outside the explicit online-observable allowlist."""

    unknown = set(feature_names).difference(ALLOWED_FEATURES)
    suspicious = {
        name
        for name in feature_names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    }
    if unknown or suspicious:
        rejected = sorted(unknown.union(suspicious))
        raise ValueError(f"non-online or hidden feature(s) rejected: {rejected}")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names must be unique")


@dataclass(frozen=True)
class BayesianLinearPosterior:
    feature_names: tuple[str, ...]
    n_observations: int
    standardization_mean: np.ndarray
    standardization_scale: np.ndarray
    standardized_coefficient_mean: np.ndarray
    standardized_precision_inverse: np.ndarray
    alpha: float
    beta: float

    @property
    def expected_noise_variance(self) -> float:
        if self.alpha <= 1.0:
            return float("nan")
        return float(self.beta / (self.alpha - 1.0))

    @property
    def coefficient_mean(self) -> np.ndarray:
        """Posterior coefficient mean on the original, unscaled features."""

        transform = self._standardized_design_transform()
        return transform.T @ self.standardized_coefficient_mean

    @property
    def coefficient_covariance(self) -> np.ndarray:
        """Posterior coefficient covariance on original feature units."""

        transform = self._standardized_design_transform()
        covariance_standard = (
            self.expected_noise_variance * self.standardized_precision_inverse
        )
        return transform.T @ covariance_standard @ transform

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return ("intercept",) + self.feature_names

    def _standardized_design_transform(self) -> np.ndarray:
        """Return B where standardized_augmented_x = B @ raw_augmented_x."""

        p = len(self.feature_names)
        transform = np.zeros((p + 1, p + 1), dtype=float)
        transform[0, 0] = 1.0
        transform[1:, 0] = -self.standardization_mean / self.standardization_scale
        transform[1:, 1:] = np.diag(1.0 / self.standardization_scale)
        return transform

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected feature array with {len(self.feature_names)} columns"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        standardized = (
            values - self.standardization_mean[None, :]
        ) / self.standardization_scale[None, :]
        return np.column_stack((np.ones(values.shape[0]), standardized))


@dataclass(frozen=True)
class PredictiveDistribution:
    mean: np.ndarray
    variance: np.ndarray
    degrees_of_freedom: float
    scale: np.ndarray

    def interval(self, coverage: float) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie in (0, 1)")
        quantile = float(
            student_t.ppf(0.5 + 0.5 * coverage, self.degrees_of_freedom)
        )
        return self.mean - quantile * self.scale, self.mean + quantile * self.scale

    def negative_log_density(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=float)
        return -student_t.logpdf(
            values,
            df=self.degrees_of_freedom,
            loc=self.mean,
            scale=self.scale,
        )


@dataclass(frozen=True)
class BayesianResidualDynamicsModel:
    """Normal--inverse-gamma Bayesian linear regression.

    Prior, after standardizing non-intercept features::

        sigma^2 ~ InvGamma(prior_alpha, prior_beta)
        beta | sigma^2 ~ Normal(0, sigma^2 / prior_precision * I)

    Feature standardization is fitted only from the supplied training rows.
    """

    feature_names: tuple[str, ...]
    prior_precision: float = 0.01
    prior_alpha: float = 2.0
    prior_beta: float = 0.25

    def __post_init__(self) -> None:
        assert_online_feature_names(self.feature_names)
        if self.prior_precision <= 0.0:
            raise ValueError("prior_precision must be positive")
        if self.prior_alpha <= 1.0 or self.prior_beta <= 0.0:
            raise ValueError("the inverse-gamma prior must have finite variance mean")

    def fit(
        self, features: np.ndarray, targets: np.ndarray
    ) -> BayesianLinearPosterior:
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features must have shape (n, {len(self.feature_names)})"
            )
        if y.ndim != 1 or y.size != x.shape[0]:
            raise ValueError("targets must be one-dimensional and aligned with features")
        if x.shape[0] < 2:
            raise ValueError("at least two training observations are required")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("training data must be finite")

        feature_mean = np.mean(x, axis=0)
        feature_scale = np.std(x, axis=0, ddof=0)
        feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
        standardized = (x - feature_mean[None, :]) / feature_scale[None, :]
        design = np.column_stack((np.ones(x.shape[0]), standardized))

        dimension = design.shape[1]
        prior_precision_matrix = self.prior_precision * np.eye(dimension)
        posterior_precision = prior_precision_matrix + design.T @ design
        factor = cho_factor(posterior_precision, lower=True, check_finite=True)
        posterior_precision_inverse = cho_solve(
            factor, np.eye(dimension), check_finite=True
        )
        posterior_mean = cho_solve(factor, design.T @ y, check_finite=True)
        alpha = self.prior_alpha + 0.5 * x.shape[0]
        beta = self.prior_beta + 0.5 * (
            float(y @ y) - float(posterior_mean @ posterior_precision @ posterior_mean)
        )
        beta = max(float(beta), np.finfo(float).tiny)
        return BayesianLinearPosterior(
            feature_names=self.feature_names,
            n_observations=int(x.shape[0]),
            standardization_mean=np.asarray(feature_mean, dtype=float),
            standardization_scale=np.asarray(feature_scale, dtype=float),
            standardized_coefficient_mean=np.asarray(posterior_mean, dtype=float),
            standardized_precision_inverse=np.asarray(
                posterior_precision_inverse, dtype=float
            ),
            alpha=float(alpha),
            beta=beta,
        )

    @staticmethod
    def predict(
        posterior: BayesianLinearPosterior, features: np.ndarray
    ) -> PredictiveDistribution:
        design = posterior.transform_features(features)
        mean = design @ posterior.standardized_coefficient_mean
        leverage = np.einsum(
            "ij,jk,ik->i",
            design,
            posterior.standardized_precision_inverse,
            design,
        )
        degrees_of_freedom = 2.0 * posterior.alpha
        scale_squared = (posterior.beta / posterior.alpha) * (1.0 + leverage)
        variance = (
            scale_squared
            * degrees_of_freedom
            / (degrees_of_freedom - 2.0)
        )
        return PredictiveDistribution(
            mean=np.asarray(mean, dtype=float),
            variance=np.asarray(variance, dtype=float),
            degrees_of_freedom=degrees_of_freedom,
            scale=np.sqrt(scale_squared),
        )

    @staticmethod
    def sample_posterior(
        posterior: BayesianLinearPosterior,
        n_samples: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample one coefficient vector and noise variance per trajectory."""

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        variances = np.asarray(
            invgamma.rvs(
                posterior.alpha,
                scale=posterior.beta,
                size=n_samples,
                random_state=rng,
            ),
            dtype=float,
        )
        covariance_factor = np.linalg.cholesky(
            posterior.standardized_precision_inverse
        )
        standard_normal = rng.normal(
            size=(n_samples, posterior.standardized_coefficient_mean.size)
        )
        coefficient_offsets = standard_normal @ covariance_factor.T
        coefficients = (
            posterior.standardized_coefficient_mean[None, :]
            + np.sqrt(variances)[:, None] * coefficient_offsets
        )
        return np.asarray(coefficients, dtype=float), variances

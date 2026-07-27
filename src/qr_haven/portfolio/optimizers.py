"""Portfolio optimizer implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizerConstraints:
    """Core v0 portfolio constraints."""

    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_sum: float = 1.0
    risk_aversion: float = 1.0
    ridge: float = 1e-8
    max_iterations: int = 2_000
    tolerance: float = 1e-10

    @classmethod
    def from_mapping(cls, constraints: Mapping[str, Any] | None) -> OptimizerConstraints:
        """Build constraints from a plain mapping used by the shared protocol."""

        if constraints is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        values = {key: value for key, value in constraints.items() if key in allowed}
        return cls(**values)

    def bounds(self) -> tuple[float, float]:
        """Return effective lower and upper bounds."""

        lower = max(0.0, self.min_weight) if self.long_only else self.min_weight
        upper = self.max_weight
        if lower > upper:
            raise ValueError("min_weight cannot be greater than max_weight.")
        return lower, upper


def estimate_expected_returns(
    returns: pd.DataFrame,
    annualization: float | None = None,
) -> pd.Series:
    """Estimate expected returns from historical return observations."""

    expected = returns.mean(axis=0, skipna=True)
    if annualization is not None:
        expected = expected * annualization
    return expected.astype(float)


def estimate_return_covariance(
    returns: pd.DataFrame,
    annualization: float | None = None,
) -> pd.DataFrame:
    """Estimate return covariance from historical return observations."""

    covariance = returns.cov()
    if annualization is not None:
        covariance = covariance * annualization
    return covariance.astype(float)


class EqualWeightOptimizer:
    """Allocate equally across assets that have expected-return estimates."""

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: Mapping[str, Any] | None = None,
    ) -> pd.Series:
        """Return equal portfolio weights respecting simple max-weight limits."""

        del covariance
        optimizer_constraints = OptimizerConstraints.from_mapping(constraints)
        assets = list(expected_returns.dropna().index)
        if not assets:
            raise ValueError("At least one asset is required.")

        lower, upper = optimizer_constraints.bounds()
        raw = np.full(len(assets), optimizer_constraints.target_sum / len(assets))
        weights = _project_to_bounded_simplex(raw, optimizer_constraints.target_sum, lower, upper)
        return pd.Series(weights, index=assets, name="weight")


class MeanVarianceOptimizer:
    """Mean-variance optimizer with bounded simplex constraints."""

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: Mapping[str, Any] | None = None,
    ) -> pd.Series:
        """Maximize expected return penalized by portfolio variance."""

        optimizer_constraints = OptimizerConstraints.from_mapping(constraints)
        assets = list(expected_returns.dropna().index)
        if not assets:
            raise ValueError("At least one asset is required.")

        covariance = covariance.reindex(index=assets, columns=assets).fillna(0.0)
        means = expected_returns.reindex(assets).astype(float).to_numpy()
        covariance_matrix = covariance.astype(float).to_numpy()
        covariance_matrix = covariance_matrix + np.eye(len(assets)) * optimizer_constraints.ridge

        lower, upper = optimizer_constraints.bounds()
        weights = _solve_projected_mean_variance(
            expected_returns=means,
            covariance=covariance_matrix,
            constraints=optimizer_constraints,
            lower=lower,
            upper=upper,
        )
        return pd.Series(weights, index=assets, name="weight")


def _solve_projected_mean_variance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    constraints: OptimizerConstraints,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Solve the v0 bounded mean-variance problem with projected gradient ascent."""

    weights = _project_to_bounded_simplex(
        np.full(len(expected_returns), constraints.target_sum / len(expected_returns)),
        constraints.target_sum,
        lower,
        upper,
    )
    largest_eigenvalue = float(np.linalg.eigvalsh(covariance).max())
    lipschitz = max(2.0 * constraints.risk_aversion * largest_eigenvalue, constraints.ridge)
    step_size = 1.0 / lipschitz

    for _ in range(constraints.max_iterations):
        gradient = expected_returns - 2.0 * constraints.risk_aversion * covariance @ weights
        candidate = _project_to_bounded_simplex(
            weights + step_size * gradient,
            constraints.target_sum,
            lower,
            upper,
        )
        if float(np.linalg.norm(candidate - weights, ord=1)) < constraints.tolerance:
            return candidate
        weights = candidate
    return weights


def _project_to_bounded_simplex(
    values: np.ndarray,
    target_sum: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Project values onto sum(weights)=target_sum with lower/upper bounds."""

    size = values.size
    if size == 0:
        raise ValueError("Cannot project an empty vector.")
    if target_sum < lower * size or target_sum > upper * size:
        raise ValueError("Constraints cannot satisfy target_sum.")

    lo = float(values.min() - upper)
    hi = float(values.max() - lower)
    for _ in range(100):
        midpoint = (lo + hi) / 2.0
        projected = np.clip(values - midpoint, lower, upper)
        if projected.sum() > target_sum:
            lo = midpoint
        else:
            hi = midpoint

    projected = np.clip(values - hi, lower, upper)
    residual = target_sum - float(projected.sum())
    if abs(residual) > 1e-10:
        free = (projected > lower + 1e-12) & (projected < upper - 1e-12)
        if free.any():
            projected[free] += residual / float(free.sum())
    return projected

"""Portfolio optimizer implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizerConstraints:
    """Core portfolio constraints for v0/v1 optimizers."""

    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_sum: float = 1.0
    min_net_exposure: float | None = None
    max_net_exposure: float | None = None
    max_gross_exposure: float | None = None
    max_turnover: float | None = None
    group_max_exposure: Mapping[str, float] = field(default_factory=dict)
    asset_groups: Mapping[str, str] = field(default_factory=dict)
    risk_aversion: float = 1.0
    ridge: float = 1e-8
    max_iterations: int = 2_000
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.target_sum < 0.0:
            raise ValueError("target_sum cannot be negative.")
        if self.max_weight < self.min_weight:
            raise ValueError("max_weight cannot be less than min_weight.")
        if self.max_gross_exposure is not None and self.max_gross_exposure < 0.0:
            raise ValueError("max_gross_exposure cannot be negative.")
        if self.max_turnover is not None and self.max_turnover < 0.0:
            raise ValueError("max_turnover cannot be negative.")
        if self.risk_aversion < 0.0:
            raise ValueError("risk_aversion cannot be negative.")

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


@dataclass(frozen=True)
class OptimizerDiagnostics:
    """Diagnostics explaining optimizer output and constraint pressure."""

    asset_count: int
    expected_portfolio_return: float
    ex_ante_volatility: float
    variance: float
    objective_value: float
    gross_exposure: float
    net_exposure: float
    concentration: float
    effective_holdings: float
    min_weight: float
    max_weight: float
    names_at_min_weight: int
    names_at_max_weight: int
    min_weight_binding: int
    max_weight_binding: int
    gross_exposure_binding: int
    net_exposure_binding: int
    turnover: float
    max_turnover_binding: int
    max_group_exposure: float
    group_exposure_binding: int
    constraint_pressure: float

    def to_dict(self) -> dict[str, float | int]:
        """Return serializable optimizer diagnostics."""

        return asdict(self)


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


def calculate_optimizer_diagnostics(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: Mapping[str, Any] | OptimizerConstraints | None = None,
    previous_weights: pd.Series | None = None,
) -> OptimizerDiagnostics:
    """Calculate diagnostics for a portfolio optimizer output."""

    optimizer_constraints = (
        constraints
        if isinstance(constraints, OptimizerConstraints)
        else OptimizerConstraints.from_mapping(constraints)
    )
    weights = weights.dropna().astype(float)
    expected_returns = expected_returns.reindex(weights.index).fillna(0.0).astype(float)
    covariance = covariance.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    covariance_matrix = covariance.astype(float).to_numpy()
    weight_vector = weights.to_numpy()
    expected_vector = expected_returns.to_numpy()

    variance = float(weight_vector.T @ covariance_matrix @ weight_vector)
    ex_ante_volatility = float(np.sqrt(max(variance, 0.0)))
    expected_portfolio_return = float(weight_vector @ expected_vector)
    objective_value = (
        expected_portfolio_return - optimizer_constraints.risk_aversion * variance
    )
    gross_exposure = float(weights.abs().sum())
    net_exposure = float(weights.sum())
    concentration = float((weights**2).sum())
    effective_holdings = _safe_inverse(concentration)
    lower, upper = optimizer_constraints.bounds()
    tolerance = max(optimizer_constraints.tolerance, 1e-9)
    names_at_min_weight = int((weights <= lower + tolerance).sum())
    names_at_max_weight = int((weights >= upper - tolerance).sum())
    previous = (
        previous_weights.reindex(weights.index).fillna(0.0).astype(float)
        if previous_weights is not None
        else pd.Series(0.0, index=weights.index, dtype=float)
    )
    turnover = float((weights - previous).abs().sum())
    max_group_exposure, group_exposure_binding = _group_exposure_diagnostics(
        weights,
        optimizer_constraints,
        tolerance,
    )

    pressure_components = [
        _ratio_pressure(weights.max(), upper) if upper > 0.0 else 0.0,
        _ratio_pressure(gross_exposure, optimizer_constraints.max_gross_exposure),
        _range_pressure(
            net_exposure,
            optimizer_constraints.min_net_exposure,
            optimizer_constraints.max_net_exposure,
        ),
        _ratio_pressure(turnover, optimizer_constraints.max_turnover),
        _ratio_pressure(max_group_exposure, _tightest_group_limit(optimizer_constraints)),
    ]
    constraint_pressure = float(max(pressure_components))

    return OptimizerDiagnostics(
        asset_count=int(len(weights)),
        expected_portfolio_return=expected_portfolio_return,
        ex_ante_volatility=ex_ante_volatility,
        variance=variance,
        objective_value=float(objective_value),
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        concentration=concentration,
        effective_holdings=effective_holdings,
        min_weight=float(weights.min()) if not weights.empty else 0.0,
        max_weight=float(weights.max()) if not weights.empty else 0.0,
        names_at_min_weight=names_at_min_weight,
        names_at_max_weight=names_at_max_weight,
        min_weight_binding=int(names_at_min_weight > 0),
        max_weight_binding=int(names_at_max_weight > 0),
        gross_exposure_binding=int(
            _is_above_limit(gross_exposure, optimizer_constraints.max_gross_exposure, tolerance)
        ),
        net_exposure_binding=int(
            _is_outside_range(
                net_exposure,
                optimizer_constraints.min_net_exposure,
                optimizer_constraints.max_net_exposure,
                tolerance,
            )
        ),
        turnover=turnover,
        max_turnover_binding=int(
            _is_above_limit(turnover, optimizer_constraints.max_turnover, tolerance)
        ),
        max_group_exposure=max_group_exposure,
        group_exposure_binding=group_exposure_binding,
        constraint_pressure=constraint_pressure,
    )


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

    def optimize_with_diagnostics(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: Mapping[str, Any] | None = None,
        previous_weights: pd.Series | None = None,
    ) -> tuple[pd.Series, OptimizerDiagnostics]:
        """Return equal weights and diagnostics for the allocation."""

        weights = self.optimize(expected_returns, covariance, constraints)
        diagnostics = calculate_optimizer_diagnostics(
            weights,
            expected_returns,
            covariance,
            constraints,
            previous_weights,
        )
        return weights, diagnostics


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

    def optimize_with_diagnostics(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: Mapping[str, Any] | None = None,
        previous_weights: pd.Series | None = None,
    ) -> tuple[pd.Series, OptimizerDiagnostics]:
        """Return mean-variance weights and diagnostics for the allocation."""

        weights = self.optimize(expected_returns, covariance, constraints)
        diagnostics = calculate_optimizer_diagnostics(
            weights,
            expected_returns,
            covariance,
            constraints,
            previous_weights,
        )
        return weights, diagnostics


def _solve_projected_mean_variance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    constraints: OptimizerConstraints,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Solve the bounded mean-variance QP.

    Uses HiGHS (highspy) when available — exact interior-point QP.
    Falls back to projected gradient ascent if highspy is not installed.
    """
    try:
        return _solve_mean_variance_highs(expected_returns, covariance, constraints, lower, upper)
    except ImportError:
        return _solve_mean_variance_gradient(expected_returns, covariance, constraints, lower, upper)


def _solve_mean_variance_highs(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    constraints: OptimizerConstraints,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Exact mean-variance QP via HiGHS interior-point solver.

    Problem (minimisation form):
        min  -μᵀw + λ · wᵀΣw
        s.t. Σᵢ wᵢ = target_sum
             lower ≤ wᵢ ≤ upper

    HiGHS convention: objective = cᵀx + 0.5·xᵀHx, so H = 2λΣ.
    """
    import highspy
    from scipy.sparse import triu as _triu

    N = len(expected_returns)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    h.addVars(N, np.full(N, lower), np.full(N, upper))
    h.changeColsCostByRange(0, N - 1, (-expected_returns).tolist())

    H = _triu(2.0 * constraints.risk_aversion * (covariance + np.eye(N) * constraints.ridge),
              format="csr")
    h.passHessian(N, H.nnz, 1, H.indptr.tolist(), H.indices.tolist(), H.data.tolist())

    h.addRow(constraints.target_sum, constraints.target_sum, N, list(range(N)), [1.0] * N)

    h.run()

    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return _solve_mean_variance_gradient(expected_returns, covariance, constraints, lower, upper)

    return np.clip(np.array(h.getSolution().col_value), lower, upper)


def _solve_mean_variance_gradient(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    constraints: OptimizerConstraints,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Fallback: projected gradient ascent for the bounded mean-variance problem."""

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


def _safe_inverse(value: float) -> float:
    if value == 0.0:
        return 0.0
    return 1.0 / value


def _ratio_pressure(value: float, limit: float | None) -> float:
    if limit is None or limit == 0.0:
        return 0.0
    return float(value / limit)


def _range_pressure(value: float, lower: float | None, upper: float | None) -> float:
    lower_pressure = float(lower / value) if lower is not None and value != 0.0 else 0.0
    upper_pressure = _ratio_pressure(value, upper)
    return max(lower_pressure, upper_pressure)


def _is_above_limit(value: float, limit: float | None, tolerance: float) -> bool:
    return limit is not None and value >= limit - tolerance


def _is_outside_range(
    value: float,
    lower: float | None,
    upper: float | None,
    tolerance: float,
) -> bool:
    below = lower is not None and value <= lower + tolerance
    above = upper is not None and value >= upper - tolerance
    return below or above


def _tightest_group_limit(constraints: OptimizerConstraints) -> float | None:
    if not constraints.group_max_exposure:
        return None
    return min(constraints.group_max_exposure.values())


def _group_exposure_diagnostics(
    weights: pd.Series,
    constraints: OptimizerConstraints,
    tolerance: float,
) -> tuple[float, int]:
    if not constraints.asset_groups or not constraints.group_max_exposure:
        return 0.0, 0

    group_exposures: dict[str, float] = {}
    for symbol, weight in weights.abs().items():
        group = constraints.asset_groups.get(str(symbol))
        if group is None:
            continue
        group_exposures[group] = group_exposures.get(group, 0.0) + float(weight)

    if not group_exposures:
        return 0.0, 0

    max_group_exposure = max(group_exposures.values())
    binding = any(
        exposure >= constraints.group_max_exposure.get(group, float("inf")) - tolerance
        for group, exposure in group_exposures.items()
    )
    return float(max_group_exposure), int(binding)

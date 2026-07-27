"""Risk analytics for portfolio research."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskMetrics:
    """Core v0 risk metrics for a weighted portfolio."""

    observations: int
    mean_return: float
    volatility: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    value_at_risk: float
    conditional_value_at_risk: float
    gross_exposure: float
    net_exposure: float
    concentration: float
    largest_weight: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a serializable risk payload."""

        return asdict(self)


class SimpleRiskEngine:
    """Compute single-period portfolio risk diagnostics."""

    def __init__(
        self,
        annualization: float = 252.0,
        risk_free_rate: float = 0.0,
        var_confidence: float = 0.95,
    ) -> None:
        if not 0.0 < var_confidence < 1.0:
            raise ValueError("var_confidence must be between 0 and 1.")
        self.annualization = annualization
        self.risk_free_rate = risk_free_rate
        self.var_confidence = var_confidence

    def evaluate(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, float | int]:
        """Return risk metrics for a weighted return matrix."""

        del context
        portfolio = portfolio_returns(weights, returns)
        metrics = self.evaluate_portfolio_returns(weights, portfolio)
        return metrics.to_dict()

    def evaluate_portfolio_returns(
        self,
        weights: pd.Series,
        returns: pd.Series,
    ) -> RiskMetrics:
        """Return risk metrics from already-computed portfolio returns."""

        clean_returns = returns.dropna().astype(float)
        if clean_returns.empty:
            raise ValueError("At least one portfolio return observation is required.")

        mean_return = float(clean_returns.mean())
        volatility = float(clean_returns.std(ddof=1)) if len(clean_returns) > 1 else 0.0
        annualized_return = mean_return * self.annualization
        annualized_volatility = volatility * np.sqrt(self.annualization)
        excess_annualized_return = annualized_return - self.risk_free_rate
        sharpe_ratio = _safe_divide(excess_annualized_return, annualized_volatility)
        max_drawdown = _max_drawdown(clean_returns)
        value_at_risk = _value_at_risk(clean_returns, self.var_confidence)
        conditional_value_at_risk = _conditional_value_at_risk(
            clean_returns,
            value_at_risk,
        )

        aligned_weights = weights.dropna().astype(float)
        gross_exposure = float(aligned_weights.abs().sum())
        net_exposure = float(aligned_weights.sum())
        concentration = float((aligned_weights**2).sum())
        largest_weight = float(aligned_weights.abs().max()) if not aligned_weights.empty else 0.0

        return RiskMetrics(
            observations=int(len(clean_returns)),
            mean_return=mean_return,
            volatility=volatility,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            value_at_risk=value_at_risk,
            conditional_value_at_risk=conditional_value_at_risk,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            concentration=concentration,
            largest_weight=largest_weight,
        )


def portfolio_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Calculate weighted portfolio returns from asset returns."""

    aligned_returns = returns.reindex(columns=weights.index).astype(float)
    aligned_weights = weights.astype(float).reindex(aligned_returns.columns).fillna(0.0)
    portfolio = aligned_returns.fillna(0.0).dot(aligned_weights)
    portfolio.name = "portfolio_return"
    return portfolio


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def _value_at_risk(returns: pd.Series, confidence: float) -> float:
    return float(-returns.quantile(1.0 - confidence))


def _conditional_value_at_risk(returns: pd.Series, value_at_risk: float) -> float:
    tail_returns = returns[returns <= -value_at_risk]
    if tail_returns.empty:
        return value_at_risk
    return float(-tail_returns.mean())


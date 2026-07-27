"""Performance attribution for portfolio backtests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qr_haven.backtesting.engine import BacktestResult


@dataclass(frozen=True)
class PerformanceAttribution:
    """Serializable attribution bundle for a portfolio backtest."""

    performance_summary: dict[str, float | int]
    turnover_cost_summary: dict[str, float | int]
    optimizer_diagnostics: dict[str, float | int]
    asset_contributions: pd.DataFrame


def attribute_backtest(
    result: BacktestResult,
    annualization: float = 252.0,
) -> PerformanceAttribution:
    """Calculate v0 performance, cost, optimizer, and asset attribution."""

    return PerformanceAttribution(
        performance_summary=summarize_performance(result, annualization=annualization),
        turnover_cost_summary=summarize_turnover_and_costs(result),
        optimizer_diagnostics=summarize_optimizer_diagnostics(result),
        asset_contributions=summarize_asset_contributions(result),
    )


def summarize_performance(
    result: BacktestResult,
    annualization: float = 252.0,
) -> dict[str, float | int]:
    """Summarize return-path performance."""

    returns = result.portfolio_returns.dropna().astype(float)
    gross_returns = result.gross_portfolio_returns.reindex(returns.index).dropna().astype(float)
    observations = len(returns)
    total_return = float((1.0 + returns).prod() - 1.0)
    gross_total_return = float((1.0 + gross_returns).prod() - 1.0)
    years = observations / annualization
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    return {
        "observations": int(observations),
        "total_return": total_return,
        "gross_total_return": gross_total_return,
        "transaction_cost_drag": gross_total_return - total_return,
        "cagr": cagr,
        "hit_rate": float((returns > 0.0).mean()) if observations else 0.0,
        "best_period": float(returns.max()) if observations else 0.0,
        "worst_period": float(returns.min()) if observations else 0.0,
        "ending_equity": float(result.equity_curve.iloc[-1]),
    }


def summarize_turnover_and_costs(result: BacktestResult) -> dict[str, float | int]:
    """Summarize turnover and transaction-cost drag."""

    turnover = result.turnover.dropna().astype(float)
    transaction_costs = result.transaction_costs.dropna().astype(float)
    return {
        "rebalance_count": int(len(turnover)),
        "average_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
        "max_turnover": float(turnover.max()) if not turnover.empty else 0.0,
        "total_transaction_cost": float(transaction_costs.sum()),
        "average_transaction_cost": (
            float(transaction_costs.mean()) if not transaction_costs.empty else 0.0
        ),
        "max_transaction_cost": (
            float(transaction_costs.max()) if not transaction_costs.empty else 0.0
        ),
    }


def summarize_optimizer_diagnostics(result: BacktestResult) -> dict[str, float | int]:
    """Summarize optimizer behavior across rebalance dates."""

    diagnostics = result.diagnostics.astype(float)
    if diagnostics.empty:
        return {
            "rebalance_count": 0,
            "average_concentration": 0.0,
            "average_effective_holdings": 0.0,
            "average_gross_exposure": 0.0,
            "average_max_weight": 0.0,
            "max_weight": 0.0,
            "average_ex_ante_volatility": 0.0,
            "average_expected_portfolio_return": 0.0,
            "average_objective_value": 0.0,
            "average_constraint_pressure": 0.0,
            "max_constraint_pressure": 0.0,
            "max_weight_binding_rate": 0.0,
            "max_turnover_binding_rate": 0.0,
            "group_exposure_binding_rate": 0.0,
        }

    return {
        "rebalance_count": int(len(diagnostics)),
        "average_concentration": _column_mean(diagnostics, "concentration"),
        "average_effective_holdings": _column_mean(diagnostics, "effective_holdings"),
        "average_gross_exposure": _column_mean(diagnostics, "gross_exposure"),
        "average_max_weight": _column_mean(diagnostics, "max_weight"),
        "max_weight": _column_max(diagnostics, "max_weight"),
        "average_ex_ante_volatility": _column_mean(diagnostics, "ex_ante_volatility"),
        "average_expected_portfolio_return": _column_mean(
            diagnostics,
            "expected_portfolio_return",
        ),
        "average_objective_value": _column_mean(diagnostics, "objective_value"),
        "average_constraint_pressure": _column_mean(diagnostics, "constraint_pressure"),
        "max_constraint_pressure": _column_max(diagnostics, "constraint_pressure"),
        "max_weight_binding_rate": _column_mean(diagnostics, "max_weight_binding"),
        "max_turnover_binding_rate": _column_mean(diagnostics, "max_turnover_binding"),
        "group_exposure_binding_rate": _column_mean(diagnostics, "group_exposure_binding"),
    }


def summarize_asset_contributions(result: BacktestResult) -> pd.DataFrame:
    """Summarize asset-level return contribution and portfolio usage."""

    total_contribution = result.asset_contributions.sum(axis=0)
    average_weight = result.weights.mean(axis=0)
    ending_weight = result.weights.iloc[-1]
    contribution = pd.DataFrame(
        {
            "total_contribution": total_contribution,
            "average_weight": average_weight,
            "ending_weight": ending_weight,
        }
    )
    contribution.index.name = "symbol"
    return contribution.sort_values("total_contribution", ascending=False)


def _column_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    return float(frame[column].mean())


def _column_max(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    return float(frame[column].max())

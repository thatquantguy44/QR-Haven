"""Walk-forward portfolio backtesting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from qr_haven.interfaces import PortfolioOptimizer
from qr_haven.portfolio import (
    calculate_optimizer_diagnostics,
    estimate_expected_returns,
    estimate_return_covariance,
)
from qr_haven.risk import SimpleRiskEngine


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a v0 walk-forward backtest."""

    lookback_periods: int = 60
    rebalance_periods: int = 21
    transaction_cost_bps: float = 0.0
    annualization: float = 252.0
    optimizer_constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lookback_periods < 2:
            raise ValueError("lookback_periods must be at least 2.")
        if self.rebalance_periods < 1:
            raise ValueError("rebalance_periods must be at least 1.")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative.")


@dataclass(frozen=True)
class BacktestResult:
    """Outputs from a walk-forward portfolio backtest."""

    portfolio_returns: pd.Series
    gross_portfolio_returns: pd.Series
    equity_curve: pd.Series
    asset_contributions: pd.DataFrame
    weights: pd.DataFrame
    diagnostics: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    risk_metrics: Mapping[str, float | int]

    def summary(self) -> dict[str, float | int]:
        """Return a compact serializable backtest summary."""

        total_return = float(self.equity_curve.iloc[-1] - 1.0)
        gross_total_return = float((1.0 + self.gross_portfolio_returns).prod() - 1.0)
        average_turnover = float(self.turnover.mean()) if not self.turnover.empty else 0.0
        total_transaction_cost = float(self.transaction_costs.sum())
        return {
            "observations": int(len(self.portfolio_returns)),
            "total_return": total_return,
            "gross_total_return": gross_total_return,
            "ending_equity": float(self.equity_curve.iloc[-1]),
            "average_turnover": average_turnover,
            "total_transaction_cost": total_transaction_cost,
            **self.risk_metrics,
        }


class WalkForwardBacktester:
    """Run a rolling optimization and hold weights between rebalance dates."""

    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        config: BacktestConfig | None = None,
        risk_engine: SimpleRiskEngine | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config or BacktestConfig()
        self.risk_engine = risk_engine or SimpleRiskEngine(
            annualization=self.config.annualization,
        )

    def run(self, returns: pd.DataFrame) -> BacktestResult:
        """Run a walk-forward backtest on a wide asset-return matrix."""

        clean_returns = returns.sort_index().astype(float)
        if len(clean_returns) <= self.config.lookback_periods:
            raise ValueError("Returns must contain more rows than the lookback window.")

        rebalance_positions = list(
            range(
                self.config.lookback_periods,
                len(clean_returns),
                self.config.rebalance_periods,
            )
        )
        if not rebalance_positions:
            raise ValueError("No rebalance dates could be generated.")

        portfolio_returns: list[pd.Series] = []
        gross_portfolio_returns: list[pd.Series] = []
        asset_contributions: list[pd.DataFrame] = []
        weight_rows: list[pd.Series] = []
        diagnostic_rows: list[pd.Series] = []
        turnover_values: list[float] = []
        transaction_cost_values: list[float] = []
        transaction_cost_index: list[pd.Timestamp] = []
        previous_weights = pd.Series(0.0, index=clean_returns.columns, dtype=float)

        for rebalance_position in rebalance_positions:
            rebalance_date = clean_returns.index[rebalance_position]
            window = clean_returns.iloc[
                rebalance_position - self.config.lookback_periods : rebalance_position
            ]
            expected_returns = estimate_expected_returns(window)
            covariance = estimate_return_covariance(window)
            target_weights = self.optimizer.optimize(
                expected_returns,
                covariance,
                self.config.optimizer_constraints,
            )
            target_weights = target_weights.reindex(clean_returns.columns).fillna(0.0)
            diagnostics = calculate_optimizer_diagnostics(
                target_weights,
                expected_returns,
                covariance,
                self.config.optimizer_constraints,
                previous_weights,
            )
            turnover = float((target_weights - previous_weights).abs().sum())
            transaction_cost = turnover * self.config.transaction_cost_bps / 10_000.0

            period_end = min(
                rebalance_position + self.config.rebalance_periods,
                len(clean_returns),
            )
            period_asset_contributions = clean_returns.iloc[
                rebalance_position:period_end
            ].multiply(target_weights, axis="columns")
            period_gross_returns = period_asset_contributions.sum(axis=1)
            period_gross_returns.name = "gross_portfolio_return"
            period_returns = period_gross_returns.copy()
            period_returns.name = "portfolio_return"
            if not period_returns.empty:
                period_returns.iloc[0] = period_returns.iloc[0] - transaction_cost

            portfolio_returns.append(period_returns)
            gross_portfolio_returns.append(period_gross_returns)
            asset_contributions.append(period_asset_contributions)
            weight_rows.append(target_weights.rename(rebalance_date))
            diagnostic_rows.append(pd.Series(diagnostics.to_dict(), name=rebalance_date))
            turnover_values.append(turnover)
            transaction_cost_values.append(transaction_cost)
            transaction_cost_index.append(rebalance_date)
            previous_weights = target_weights

        combined_returns = pd.concat(portfolio_returns).sort_index()
        combined_returns.name = "portfolio_return"
        combined_gross_returns = pd.concat(gross_portfolio_returns).sort_index()
        combined_gross_returns.name = "gross_portfolio_return"
        combined_asset_contributions = pd.concat(asset_contributions).sort_index()
        equity_curve = (1.0 + combined_returns).cumprod()
        equity_curve.name = "equity"
        weights = pd.DataFrame(weight_rows).sort_index()
        diagnostics = pd.DataFrame(diagnostic_rows).sort_index()
        turnover = pd.Series(turnover_values, index=transaction_cost_index, name="turnover")
        transaction_costs = pd.Series(
            transaction_cost_values,
            index=transaction_cost_index,
            name="transaction_cost",
        )
        risk_metrics = self.risk_engine.evaluate_portfolio_returns(
            previous_weights,
            combined_returns,
        ).to_dict()

        return BacktestResult(
            portfolio_returns=combined_returns,
            gross_portfolio_returns=combined_gross_returns,
            equity_curve=equity_curve,
            asset_contributions=combined_asset_contributions,
            weights=weights,
            diagnostics=diagnostics,
            turnover=turnover,
            transaction_costs=transaction_costs,
            risk_metrics=risk_metrics,
        )

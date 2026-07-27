import numpy as np
import pandas as pd

from qr_haven.backtesting import BacktestConfig, WalkForwardBacktester
from qr_haven.integrations.market_terminal import (
    backtest_diagnostics_panel,
    backtest_equity_curve_panel,
    backtest_summary_panel,
    constraint_pressure_panel,
)
from qr_haven.portfolio import EqualWeightOptimizer, MeanVarianceOptimizer


def _sample_returns() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=12, tz="UTC")
    return pd.DataFrame(
        {
            "AAPL": [
                0.010,
                0.005,
                -0.002,
                0.006,
                0.003,
                0.004,
                0.008,
                -0.001,
                0.002,
                0.004,
                0.006,
                0.003,
            ],
            "MSFT": [
                0.004,
                0.006,
                0.003,
                -0.001,
                0.005,
                0.002,
                0.001,
                0.007,
                0.004,
                0.002,
                0.003,
                0.005,
            ],
        },
        index=index,
    )


def test_walk_forward_backtester_runs_equal_weight_backtest() -> None:
    config = BacktestConfig(
        lookback_periods=4,
        rebalance_periods=3,
        transaction_cost_bps=10.0,
    )
    result = WalkForwardBacktester(EqualWeightOptimizer(), config).run(_sample_returns())

    assert len(result.portfolio_returns) == 8
    assert list(result.weights.columns) == ["AAPL", "MSFT"]
    assert not result.diagnostics.empty
    assert "effective_holdings" in result.diagnostics.columns
    assert "constraint_pressure" in result.diagnostics.columns
    assert np.isclose(result.weights.iloc[0].sum(), 1.0)
    assert result.turnover.iloc[0] == 1.0
    assert np.isclose(result.transaction_costs.iloc[0], 0.001)
    assert result.equity_curve.iloc[-1] > 1.0
    assert result.summary()["observations"] == 8
    assert "sharpe_ratio" in result.summary()


def test_walk_forward_backtester_runs_mean_variance_with_constraints() -> None:
    config = BacktestConfig(
        lookback_periods=4,
        rebalance_periods=4,
        optimizer_constraints={"max_weight": 0.75, "risk_aversion": 1.0},
    )
    result = WalkForwardBacktester(MeanVarianceOptimizer(), config).run(_sample_returns())

    assert not result.weights.empty
    assert result.weights.max(axis=None) <= 0.75 + 1e-9
    assert np.allclose(result.weights.sum(axis=1), 1.0)


def test_backtest_terminal_panels_are_serializable_payloads() -> None:
    config = BacktestConfig(lookback_periods=4, rebalance_periods=3)
    result = WalkForwardBacktester(EqualWeightOptimizer(), config).run(_sample_returns())

    summary_panel = backtest_summary_panel(result)
    equity_panel = backtest_equity_curve_panel(result)
    diagnostics_panel = backtest_diagnostics_panel(result)
    pressure_panel = constraint_pressure_panel(result)

    assert summary_panel.panel_id == "backtest.summary"
    assert summary_panel.kind == "table"
    ending_equity_row = {
        "metric": "ending_equity",
        "value": result.summary()["ending_equity"],
    }
    assert ending_equity_row in summary_panel.payload["rows"]
    assert equity_panel.panel_id == "backtest.equity_curve"
    assert equity_panel.kind == "line"
    assert len(equity_panel.payload["rows"]) == len(result.equity_curve)
    assert diagnostics_panel.panel_id == "backtest.optimizer_diagnostics_history"
    assert diagnostics_panel.kind == "table"
    assert pressure_panel.panel_id == "backtest.constraint_pressure"
    assert pressure_panel.kind == "line"

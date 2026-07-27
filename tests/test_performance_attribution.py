import numpy as np
import pandas as pd

from qr_haven.backtesting import (
    BacktestConfig,
    WalkForwardBacktester,
    attribute_backtest,
)
from qr_haven.integrations.market_terminal import (
    asset_contribution_panel,
    optimizer_diagnostics_panel,
    performance_summary_panel,
    turnover_cost_panel,
)
from qr_haven.portfolio import EqualWeightOptimizer


def _sample_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAPL": [0.010, 0.005, -0.002, 0.006, 0.003, 0.004, 0.008, -0.001],
            "MSFT": [0.004, 0.006, 0.003, -0.001, 0.005, 0.002, 0.001, 0.007],
        },
        index=pd.date_range("2024-01-01", periods=8, tz="UTC"),
    )


def _backtest_result():
    config = BacktestConfig(
        lookback_periods=3,
        rebalance_periods=2,
        transaction_cost_bps=10.0,
    )
    return WalkForwardBacktester(EqualWeightOptimizer(), config).run(_sample_returns())


def test_attribute_backtest_summarizes_performance_and_cost_drag() -> None:
    result = _backtest_result()
    attribution = attribute_backtest(result, annualization=252)

    assert attribution.performance_summary["observations"] == len(result.portfolio_returns)
    assert attribution.performance_summary["gross_total_return"] > attribution.performance_summary[
        "total_return"
    ]
    assert attribution.performance_summary["transaction_cost_drag"] > 0.0
    assert attribution.turnover_cost_summary["rebalance_count"] == len(result.turnover)
    assert np.isclose(attribution.turnover_cost_summary["total_transaction_cost"], 0.001)


def test_attribute_backtest_summarizes_asset_contributions_and_optimizer_diagnostics() -> None:
    result = _backtest_result()
    attribution = attribute_backtest(result)

    assert list(attribution.asset_contributions.columns) == [
        "total_contribution",
        "average_weight",
        "ending_weight",
    ]
    assert set(attribution.asset_contributions.index) == {"AAPL", "MSFT"}
    assert attribution.optimizer_diagnostics["rebalance_count"] == len(result.weights)
    assert np.isclose(attribution.optimizer_diagnostics["average_gross_exposure"], 1.0)
    assert np.isclose(attribution.optimizer_diagnostics["max_weight"], 0.5)
    assert np.isclose(attribution.optimizer_diagnostics["average_effective_holdings"], 2.0)
    assert "average_constraint_pressure" in attribution.optimizer_diagnostics


def test_performance_attribution_terminal_panels_are_serializable() -> None:
    attribution = attribute_backtest(_backtest_result())

    performance_panel = performance_summary_panel(attribution)
    asset_panel = asset_contribution_panel(attribution)
    turnover_panel = turnover_cost_panel(attribution)
    optimizer_panel = optimizer_diagnostics_panel(attribution)

    assert performance_panel.panel_id == "performance.summary"
    assert asset_panel.panel_id == "performance.asset_contribution"
    assert turnover_panel.panel_id == "performance.turnover_cost"
    assert optimizer_panel.panel_id == "performance.optimizer_diagnostics"
    assert performance_panel.payload["rows"]
    assert asset_panel.payload["rows"]
    assert turnover_panel.payload["rows"]
    assert optimizer_panel.payload["rows"]

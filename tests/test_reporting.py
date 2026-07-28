import numpy as np
import pandas as pd

from qr_haven.backtesting import BacktestConfig, WalkForwardBacktester
from qr_haven.integrations.market_terminal import backtest_terminal_panels
from qr_haven.portfolio import EqualWeightOptimizer
from qr_haven.reporting import (
    build_backtest_report_bundle,
    calculate_cumulative_return,
    calculate_drawdown,
    calculate_rolling_sharpe,
    calculate_rolling_turnover,
    calculate_rolling_volatility,
)


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


def test_portfolio_analytics_calculates_core_series() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.03, 0.01],
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
        name="portfolio_return",
    )
    turnover = pd.Series(
        [1.0, 0.2, 0.4],
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
        name="turnover",
    )

    cumulative_return = calculate_cumulative_return(returns)
    drawdown = calculate_drawdown(returns)
    rolling_volatility = calculate_rolling_volatility(returns, window=2, annualization=252)
    rolling_sharpe = calculate_rolling_sharpe(returns, window=2, annualization=252)
    rolling_turnover = calculate_rolling_turnover(turnover, window=2)

    assert np.isclose(cumulative_return.iloc[-1], (1.01 * 0.98 * 1.03 * 1.01) - 1.0)
    assert drawdown.min() < 0.0
    assert rolling_volatility.iloc[-1] > 0.0
    assert rolling_sharpe.iloc[-1] != 0.0
    assert np.isclose(rolling_turnover.iloc[-1], 0.3)


def test_backtest_report_bundle_groups_result_attribution_and_analytics() -> None:
    result = _backtest_result()
    bundle = build_backtest_report_bundle(
        result,
        annualization=252,
        return_window=2,
        turnover_window=2,
    )

    assert bundle.result is result
    assert bundle.attribution.performance_summary["observations"] == len(result.portfolio_returns)
    assert len(bundle.analytics.drawdown) == len(result.portfolio_returns)
    assert len(bundle.analytics.rolling_volatility) == len(result.portfolio_returns)


def test_backtest_terminal_panels_returns_complete_panel_package() -> None:
    bundle = build_backtest_report_bundle(
        _backtest_result(),
        return_window=2,
        turnover_window=2,
    )

    panels = backtest_terminal_panels(bundle)
    panel_ids = {panel.panel_id for panel in panels}

    assert "backtest.summary" in panel_ids
    assert "performance.summary" in panel_ids
    assert "backtest.equity_curve" in panel_ids
    assert "backtest.drawdown" in panel_ids
    assert "backtest.rolling_risk" in panel_ids
    assert "backtest.weights" in panel_ids
    assert "backtest.optimizer_diagnostics_history" in panel_ids
    assert "backtest.constraint_pressure" in panel_ids
    assert "performance.asset_contribution" in panel_ids
    assert "performance.turnover_cost" in panel_ids
    assert "performance.optimizer_diagnostics" in panel_ids
    assert all(panel.payload["rows"] for panel in panels)


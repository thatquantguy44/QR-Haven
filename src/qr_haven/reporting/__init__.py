"""Research reports, dashboards, metrics, and terminal-ready outputs."""

from qr_haven.reporting.analytics import (
    PortfolioAnalytics,
    calculate_cumulative_return,
    calculate_drawdown,
    calculate_rolling_sharpe,
    calculate_rolling_turnover,
    calculate_rolling_volatility,
)
from qr_haven.reporting.backtest_report import BacktestReportBundle, build_backtest_report_bundle
from qr_haven.reporting.pipeline_report import PipelineReportBundle, build_pipeline_report_bundle

__all__ = [
    "BacktestReportBundle",
    "PipelineReportBundle",
    "PortfolioAnalytics",
    "build_backtest_report_bundle",
    "build_pipeline_report_bundle",
    "calculate_cumulative_return",
    "calculate_drawdown",
    "calculate_rolling_sharpe",
    "calculate_rolling_turnover",
    "calculate_rolling_volatility",
]


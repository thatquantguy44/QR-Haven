"""Backtest report bundles for terminal and dashboard handoff."""

from dataclasses import dataclass

from qr_haven.backtesting import BacktestResult, PerformanceAttribution, attribute_backtest
from qr_haven.reporting.analytics import PortfolioAnalytics, calculate_portfolio_analytics


@dataclass(frozen=True)
class BacktestReportBundle:
    """Grouped backtest result, attribution, and analytics views."""

    result: BacktestResult
    attribution: PerformanceAttribution
    analytics: PortfolioAnalytics


def build_backtest_report_bundle(
    result: BacktestResult,
    annualization: float = 252.0,
    return_window: int = 21,
    turnover_window: int = 3,
) -> BacktestReportBundle:
    """Build a complete report bundle for a backtest result."""

    attribution = attribute_backtest(result, annualization=annualization)
    analytics = calculate_portfolio_analytics(
        result,
        return_window=return_window,
        turnover_window=turnover_window,
        annualization=annualization,
    )
    return BacktestReportBundle(
        result=result,
        attribution=attribution,
        analytics=analytics,
    )


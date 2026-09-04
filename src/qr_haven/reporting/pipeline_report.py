"""Report bundles for ``ResearchPipeline`` outputs (terminal and dashboard handoff)."""

from dataclasses import dataclass

from qr_haven.reporting.analytics import PortfolioAnalytics, calculate_portfolio_analytics
from qr_haven.research.pipeline import PipelineResult


@dataclass(frozen=True)
class PipelineReportBundle:
    """Grouped cost-aware pipeline result and derived analytics views."""

    result: PipelineResult
    analytics: PortfolioAnalytics


def build_pipeline_report_bundle(
    result: PipelineResult,
    return_window: int = 21,
    turnover_window: int = 3,
    annualization: float = 252.0,
) -> PipelineReportBundle:
    """Build a complete report bundle for a cost-aware research pipeline result."""

    analytics = calculate_portfolio_analytics(
        result,
        return_window=return_window,
        turnover_window=turnover_window,
        annualization=annualization,
    )
    return PipelineReportBundle(result=result, analytics=analytics)

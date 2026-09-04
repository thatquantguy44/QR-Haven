"""Cost-aware research pipeline panel adapters for market_terminal."""

from qr_haven.integrations.market_terminal.contracts import TerminalPanel
from qr_haven.reporting import PipelineReportBundle
from qr_haven.research import PipelineResult


def pipeline_summary_panel(
    result: PipelineResult,
    panel_id: str = "pipeline.summary",
    title: str = "Research Pipeline Summary",
) -> TerminalPanel:
    """Convert a pipeline result summary into a terminal table panel payload."""

    rows = [{"metric": key, "value": value} for key, value in result.summary().items()]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def pipeline_equity_curve_panel(
    result: PipelineResult,
    panel_id: str = "pipeline.equity_curve",
    title: str = "Net Equity Curve",
) -> TerminalPanel:
    """Convert the pipeline's net equity curve into a terminal line-chart payload."""

    rows = [
        {"timestamp": timestamp.isoformat(), "equity": float(equity)}
        for timestamp, equity in result.equity_curve.items()
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="line",
        payload={"rows": rows, "x": "timestamp", "y": "equity"},
    )


def pipeline_cumulative_return_panel(
    bundle: PipelineReportBundle,
    panel_id: str = "pipeline.cumulative_return",
    title: str = "Cumulative Net Return",
) -> TerminalPanel:
    """Convert cumulative return analytics into a terminal line-chart payload."""

    rows = [
        {"timestamp": timestamp.isoformat(), "cumulative_return": float(value)}
        for timestamp, value in bundle.analytics.cumulative_return.items()
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="line",
        payload={"rows": rows, "x": "timestamp", "y": "cumulative_return"},
    )


def pipeline_drawdown_panel(
    bundle: PipelineReportBundle,
    panel_id: str = "pipeline.drawdown",
    title: str = "Drawdown",
) -> TerminalPanel:
    """Convert drawdown analytics into a terminal line-chart payload."""

    rows = [
        {"timestamp": timestamp.isoformat(), "drawdown": float(drawdown)}
        for timestamp, drawdown in bundle.analytics.drawdown.items()
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="line",
        payload={"rows": rows, "x": "timestamp", "y": "drawdown"},
    )


def pipeline_rolling_risk_panel(
    bundle: PipelineReportBundle,
    panel_id: str = "pipeline.rolling_risk",
    title: str = "Rolling Risk",
) -> TerminalPanel:
    """Convert rolling volatility and Sharpe analytics into a terminal table payload."""

    rows = []
    rolling_volatility = bundle.analytics.rolling_volatility
    rolling_sharpe = bundle.analytics.rolling_sharpe
    for timestamp in rolling_volatility.index.union(rolling_sharpe.index).sort_values():
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "rolling_volatility": float(rolling_volatility.get(timestamp, 0.0)),
                "rolling_sharpe": float(rolling_sharpe.get(timestamp, 0.0)),
            }
        )
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def pipeline_rolling_turnover_panel(
    bundle: PipelineReportBundle,
    panel_id: str = "pipeline.rolling_turnover",
    title: str = "Rolling Turnover",
) -> TerminalPanel:
    """Convert rolling turnover analytics into a terminal line-chart payload."""

    rows = [
        {"timestamp": timestamp.isoformat(), "rolling_turnover": float(value)}
        for timestamp, value in bundle.analytics.rolling_turnover.dropna().items()
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="line",
        payload={"rows": rows, "x": "timestamp", "y": "rolling_turnover"},
    )


def pipeline_weights_panel(
    result: PipelineResult,
    panel_id: str = "pipeline.weights",
    title: str = "Weights History",
) -> TerminalPanel:
    """Convert rebalance weights into a terminal table panel payload."""

    rows = result.weights.reset_index(names="timestamp").to_dict(orient="records")
    for row in rows:
        row["timestamp"] = row["timestamp"].isoformat()
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def pipeline_diagnostics_panel(
    result: PipelineResult,
    panel_id: str = "pipeline.optimizer_diagnostics_history",
    title: str = "Optimizer Diagnostics History",
) -> TerminalPanel:
    """Convert rebalance diagnostics into a terminal table panel payload."""

    rows = result.diagnostics.reset_index(names="timestamp").to_dict(orient="records")
    for row in rows:
        row["timestamp"] = row["timestamp"].isoformat()
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def pipeline_cost_breakdown_panel(
    result: PipelineResult,
    panel_id: str = "pipeline.cost_breakdown",
    title: str = "Cost Attribution By Rebalance",
) -> TerminalPanel:
    """Convert per-rebalance cost attribution into a terminal table panel payload."""

    costs = result.costs
    rows = [
        {
            "timestamp": timestamp.isoformat(),
            "market_impact_usd": float(costs.market_impact.get(timestamp, 0.0)),
            "borrow_cost_usd": float(costs.borrow_cost.get(timestamp, 0.0)),
            "financing_cost_usd": float(costs.financing_cost.get(timestamp, 0.0)),
            "total_cost_usd": float(costs.total.get(timestamp, 0.0)),
        }
        for timestamp in costs.total.index
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def pipeline_cost_summary_panel(
    result: PipelineResult,
    panel_id: str = "pipeline.cost_summary",
    title: str = "Cost Attribution Summary",
) -> TerminalPanel:
    """Convert aggregate cost totals into a terminal table panel payload."""

    rows = [{"metric": key, "value": value} for key, value in result.costs.summary().items()]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def pipeline_alpha_scores_panel(
    result: PipelineResult,
    panel_id: str = "pipeline.alpha_scores",
    title: str = "Alpha Scores Used",
) -> TerminalPanel | None:
    """Convert the alpha scores used by the optimizer into a terminal table payload.

    Returns ``None`` when the pipeline ran on historical-mean expected returns instead
    of an explicit alpha model, since there is nothing to show.
    """

    if result.alpha_scores_used is None:
        return None
    rows = result.alpha_scores_used.reset_index(names="timestamp").to_dict(orient="records")
    for row in rows:
        row["timestamp"] = row["timestamp"].isoformat()
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def pipeline_terminal_panels(bundle: PipelineReportBundle) -> list[TerminalPanel]:
    """Return the complete terminal panel package for a pipeline report bundle."""

    panels = [
        pipeline_summary_panel(bundle.result),
        pipeline_equity_curve_panel(bundle.result),
        pipeline_cumulative_return_panel(bundle),
        pipeline_drawdown_panel(bundle),
        pipeline_rolling_risk_panel(bundle),
        pipeline_rolling_turnover_panel(bundle),
        pipeline_weights_panel(bundle.result),
        pipeline_diagnostics_panel(bundle.result),
        pipeline_cost_breakdown_panel(bundle.result),
        pipeline_cost_summary_panel(bundle.result),
    ]
    alpha_panel = pipeline_alpha_scores_panel(bundle.result)
    if alpha_panel is not None:
        panels.append(alpha_panel)
    return panels

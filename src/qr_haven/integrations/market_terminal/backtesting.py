"""Backtest-panel adapters for market_terminal."""

from qr_haven.backtesting import BacktestResult, PerformanceAttribution
from qr_haven.integrations.market_terminal.contracts import TerminalPanel


def backtest_summary_panel(
    result: BacktestResult,
    panel_id: str = "backtest.summary",
    title: str = "Backtest Summary",
) -> TerminalPanel:
    """Convert a backtest result into a terminal table panel payload."""

    rows = [{"metric": key, "value": value} for key, value in result.summary().items()]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def backtest_equity_curve_panel(
    result: BacktestResult,
    panel_id: str = "backtest.equity_curve",
    title: str = "Equity Curve",
) -> TerminalPanel:
    """Convert a backtest equity curve into a terminal line-chart payload."""

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


def backtest_diagnostics_panel(
    result: BacktestResult,
    panel_id: str = "backtest.optimizer_diagnostics_history",
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


def constraint_pressure_panel(
    result: BacktestResult,
    panel_id: str = "backtest.constraint_pressure",
    title: str = "Constraint Pressure",
) -> TerminalPanel:
    """Convert constraint pressure history into a terminal line-chart payload."""

    rows = [
        {"timestamp": timestamp.isoformat(), "constraint_pressure": float(value)}
        for timestamp, value in result.diagnostics["constraint_pressure"].items()
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="line",
        payload={"rows": rows, "x": "timestamp", "y": "constraint_pressure"},
    )


def performance_summary_panel(
    attribution: PerformanceAttribution,
    panel_id: str = "performance.summary",
    title: str = "Performance Summary",
) -> TerminalPanel:
    """Convert performance attribution into a terminal table panel payload."""

    rows = [
        {"metric": key, "value": value}
        for key, value in attribution.performance_summary.items()
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def asset_contribution_panel(
    attribution: PerformanceAttribution,
    panel_id: str = "performance.asset_contribution",
    title: str = "Asset Contribution",
) -> TerminalPanel:
    """Convert asset attribution into a terminal table panel payload."""

    rows = attribution.asset_contributions.reset_index().to_dict(orient="records")
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def turnover_cost_panel(
    attribution: PerformanceAttribution,
    panel_id: str = "performance.turnover_cost",
    title: str = "Turnover and Cost",
) -> TerminalPanel:
    """Convert turnover and cost attribution into a terminal table panel payload."""

    rows = [
        {"metric": key, "value": value}
        for key, value in attribution.turnover_cost_summary.items()
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def optimizer_diagnostics_panel(
    attribution: PerformanceAttribution,
    panel_id: str = "performance.optimizer_diagnostics",
    title: str = "Optimizer Diagnostics",
) -> TerminalPanel:
    """Convert optimizer diagnostics into a terminal table panel payload."""

    rows = [
        {"metric": key, "value": value}
        for key, value in attribution.optimizer_diagnostics.items()
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )

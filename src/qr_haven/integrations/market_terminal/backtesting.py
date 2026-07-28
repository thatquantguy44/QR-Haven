"""Backtest-panel adapters for market_terminal."""

from qr_haven.backtesting import BacktestResult, PerformanceAttribution
from qr_haven.integrations.market_terminal.contracts import TerminalPanel
from qr_haven.reporting import BacktestReportBundle


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


def backtest_drawdown_panel(
    bundle: BacktestReportBundle,
    panel_id: str = "backtest.drawdown",
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


def backtest_rolling_risk_panel(
    bundle: BacktestReportBundle,
    panel_id: str = "backtest.rolling_risk",
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


def backtest_weights_panel(
    result: BacktestResult,
    panel_id: str = "backtest.weights",
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


def backtest_terminal_panels(bundle: BacktestReportBundle) -> list[TerminalPanel]:
    """Return the complete terminal panel package for a backtest report bundle."""

    return [
        backtest_summary_panel(bundle.result),
        performance_summary_panel(bundle.attribution),
        backtest_equity_curve_panel(bundle.result),
        backtest_drawdown_panel(bundle),
        backtest_rolling_risk_panel(bundle),
        backtest_weights_panel(bundle.result),
        backtest_diagnostics_panel(bundle.result),
        constraint_pressure_panel(bundle.result),
        asset_contribution_panel(bundle.attribution),
        turnover_cost_panel(bundle.attribution),
        optimizer_diagnostics_panel(bundle.attribution),
    ]

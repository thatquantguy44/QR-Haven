"""Risk-panel adapters for market_terminal."""

from collections.abc import Mapping

from qr_haven.integrations.market_terminal.contracts import TerminalPanel
from qr_haven.risk import RiskMetrics


def risk_metrics_panel(
    metrics: Mapping[str, float | int],
    panel_id: str = "risk.metrics",
    title: str = "Risk Metrics",
) -> TerminalPanel:
    """Convert a generic risk-metrics mapping into a terminal table panel payload."""

    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def risk_return_panel(
    metrics: RiskMetrics,
    panel_id: str = "risk.return",
    title: str = "Return & Risk Summary",
) -> TerminalPanel:
    """Convert return and volatility metrics into a terminal table panel payload."""

    rows = [
        {"metric": "annualized_return", "value": metrics.annualized_return},
        {"metric": "annualized_volatility", "value": metrics.annualized_volatility},
        {"metric": "sharpe_ratio", "value": metrics.sharpe_ratio},
        {"metric": "max_drawdown", "value": metrics.max_drawdown},
        {"metric": "mean_return", "value": metrics.mean_return},
        {"metric": "volatility", "value": metrics.volatility},
        {"metric": "observations", "value": metrics.observations},
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def risk_var_panel(
    metrics: RiskMetrics,
    panel_id: str = "risk.tail",
    title: str = "Tail Risk",
) -> TerminalPanel:
    """Convert VaR and CVaR metrics into a terminal table panel payload."""

    rows = [
        {"metric": "value_at_risk", "value": metrics.value_at_risk},
        {"metric": "conditional_value_at_risk", "value": metrics.conditional_value_at_risk},
        {"metric": "max_drawdown", "value": metrics.max_drawdown},
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


def risk_exposure_panel(
    metrics: RiskMetrics,
    panel_id: str = "risk.exposure",
    title: str = "Portfolio Exposure",
) -> TerminalPanel:
    """Convert exposure and concentration metrics into a terminal table panel payload."""

    rows = [
        {"metric": "gross_exposure", "value": metrics.gross_exposure},
        {"metric": "net_exposure", "value": metrics.net_exposure},
        {"metric": "concentration", "value": metrics.concentration},
        {"metric": "largest_weight", "value": metrics.largest_weight},
    ]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


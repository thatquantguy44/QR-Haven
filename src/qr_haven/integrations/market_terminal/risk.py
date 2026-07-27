"""Risk-panel adapters for market_terminal."""

from collections.abc import Mapping

from qr_haven.integrations.market_terminal.contracts import TerminalPanel


def risk_metrics_panel(
    metrics: Mapping[str, float | int],
    panel_id: str = "risk.metrics",
    title: str = "Risk Metrics",
) -> TerminalPanel:
    """Convert risk metrics into a terminal table panel payload."""

    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    return TerminalPanel(
        panel_id=panel_id,
        title=title,
        kind="table",
        payload={"rows": rows},
    )


"""Concrete TerminalPlugin implementations for backtest and risk outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qr_haven.integrations.market_terminal.backtesting import backtest_terminal_panels
from qr_haven.integrations.market_terminal.contracts import TerminalPanel
from qr_haven.integrations.market_terminal.risk import (
    risk_exposure_panel,
    risk_return_panel,
    risk_var_panel,
)
from qr_haven.reporting import BacktestReportBundle
from qr_haven.risk import RiskMetrics


@dataclass
class BacktestPlugin:
    """Terminal plugin that exposes all panels for a completed backtest."""

    plugin_id: str
    bundle: BacktestReportBundle

    def panels(self) -> Sequence[TerminalPanel]:
        """Return the complete panel set for the backtest bundle."""

        return backtest_terminal_panels(self.bundle)


@dataclass
class RiskPlugin:
    """Terminal plugin that exposes grouped risk panels for a RiskMetrics snapshot."""

    plugin_id: str
    metrics: RiskMetrics

    def panels(self) -> Sequence[TerminalPanel]:
        """Return return-summary, tail-risk, and exposure panels."""

        return [
            risk_return_panel(self.metrics),
            risk_var_panel(self.metrics),
            risk_exposure_panel(self.metrics),
        ]

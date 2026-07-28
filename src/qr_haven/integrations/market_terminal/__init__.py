"""Adapters for exposing QR-Haven research outputs to market_terminal."""

from qr_haven.integrations.market_terminal.backtesting import (
    asset_contribution_panel,
    backtest_diagnostics_panel,
    backtest_drawdown_panel,
    backtest_equity_curve_panel,
    backtest_rolling_risk_panel,
    backtest_summary_panel,
    backtest_terminal_panels,
    backtest_weights_panel,
    constraint_pressure_panel,
    optimizer_diagnostics_panel,
    performance_summary_panel,
    turnover_cost_panel,
)
from qr_haven.integrations.market_terminal.contracts import TerminalPanel, TerminalPlugin
from qr_haven.integrations.market_terminal.risk import risk_metrics_panel

__all__ = [
    "TerminalPanel",
    "TerminalPlugin",
    "asset_contribution_panel",
    "backtest_diagnostics_panel",
    "backtest_drawdown_panel",
    "backtest_equity_curve_panel",
    "backtest_rolling_risk_panel",
    "backtest_summary_panel",
    "backtest_terminal_panels",
    "backtest_weights_panel",
    "constraint_pressure_panel",
    "optimizer_diagnostics_panel",
    "performance_summary_panel",
    "risk_metrics_panel",
    "turnover_cost_panel",
]

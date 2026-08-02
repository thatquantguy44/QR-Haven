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
    cumulative_return_panel,
    optimizer_diagnostics_panel,
    performance_summary_panel,
    rolling_turnover_panel,
    turnover_cost_panel,
)
from qr_haven.integrations.market_terminal.contracts import TerminalPanel, TerminalPlugin
from qr_haven.integrations.market_terminal.plugins import BacktestPlugin, RiskPlugin
from qr_haven.integrations.market_terminal.registry import TerminalPluginRegistry, default_registry
from qr_haven.integrations.market_terminal.risk import (
    risk_exposure_panel,
    risk_metrics_panel,
    risk_return_panel,
    risk_var_panel,
)

__all__ = [
    "BacktestPlugin",
    "RiskPlugin",
    "TerminalPanel",
    "TerminalPlugin",
    "TerminalPluginRegistry",
    "asset_contribution_panel",
    "backtest_diagnostics_panel",
    "backtest_drawdown_panel",
    "backtest_equity_curve_panel",
    "backtest_rolling_risk_panel",
    "backtest_summary_panel",
    "backtest_terminal_panels",
    "backtest_weights_panel",
    "constraint_pressure_panel",
    "cumulative_return_panel",
    "default_registry",
    "optimizer_diagnostics_panel",
    "performance_summary_panel",
    "risk_exposure_panel",
    "risk_metrics_panel",
    "risk_return_panel",
    "risk_var_panel",
    "rolling_turnover_panel",
    "turnover_cost_panel",
]

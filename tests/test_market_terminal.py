import pandas as pd
import pytest

from qr_haven.backtesting import BacktestConfig, WalkForwardBacktester
from qr_haven.integrations.market_terminal import (
    BacktestPlugin,
    RiskPlugin,
    TerminalPanel,
    TerminalPluginRegistry,
    backtest_terminal_panels,
    cumulative_return_panel,
    default_registry,
    risk_exposure_panel,
    risk_return_panel,
    risk_var_panel,
    rolling_turnover_panel,
)
from qr_haven.portfolio import EqualWeightOptimizer
from qr_haven.reporting import build_backtest_report_bundle
from qr_haven.risk import RiskMetrics, SimpleRiskEngine


def _sample_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAPL": [0.010, 0.005, -0.002, 0.006, 0.003, 0.004, 0.008, -0.001],
            "MSFT": [0.004, 0.006, 0.003, -0.001, 0.005, 0.002, 0.001, 0.007],
        },
        index=pd.date_range("2024-01-01", periods=8, tz="UTC"),
    )


def _bundle():
    config = BacktestConfig(lookback_periods=3, rebalance_periods=2, transaction_cost_bps=10.0)
    result = WalkForwardBacktester(EqualWeightOptimizer(), config).run(_sample_returns())
    return build_backtest_report_bundle(result, return_window=2, turnover_window=2)


def _sample_risk_metrics() -> RiskMetrics:
    returns = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03, 0.005], "B": [0.005, 0.01, -0.01, 0.02]},
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})
    return SimpleRiskEngine().evaluate_portfolio_returns(
        weights,
        returns.dot(weights),
    )


# ---------------------------------------------------------------------------
# TerminalPanel serialization
# ---------------------------------------------------------------------------


def test_terminal_panel_to_dict_round_trips() -> None:
    panel = TerminalPanel(
        panel_id="test.panel",
        title="Test Panel",
        kind="table",
        payload={"rows": [{"metric": "sharpe", "value": 1.2}]},
    )
    data = panel.to_dict()
    assert data["panel_id"] == "test.panel"
    assert data["kind"] == "table"
    restored = TerminalPanel.from_dict(data)
    assert restored == panel


def test_terminal_panel_from_dict_handles_missing_payload() -> None:
    data = {"panel_id": "x", "title": "X", "kind": "line"}
    panel = TerminalPanel.from_dict(data)
    assert panel.payload == {}


# ---------------------------------------------------------------------------
# Analytics panels
# ---------------------------------------------------------------------------


def test_cumulative_return_panel_is_line_chart() -> None:
    bundle = _bundle()
    panel = cumulative_return_panel(bundle)
    assert panel.panel_id == "backtest.cumulative_return"
    assert panel.kind == "line"
    assert panel.payload["x"] == "timestamp"
    assert panel.payload["y"] == "cumulative_return"
    assert len(panel.payload["rows"]) > 0


def test_rolling_turnover_panel_is_line_chart() -> None:
    bundle = _bundle()
    panel = rolling_turnover_panel(bundle)
    assert panel.panel_id == "backtest.rolling_turnover"
    assert panel.kind == "line"
    assert panel.payload["x"] == "timestamp"
    assert panel.payload["y"] == "rolling_turnover"


def test_backtest_terminal_panels_includes_new_analytics_panels() -> None:
    bundle = _bundle()
    panels = backtest_terminal_panels(bundle)
    panel_ids = {p.panel_id for p in panels}
    assert "backtest.cumulative_return" in panel_ids
    assert "backtest.rolling_turnover" in panel_ids
    assert len(panels) == 13


# ---------------------------------------------------------------------------
# Risk panels
# ---------------------------------------------------------------------------


def test_risk_return_panel_contains_expected_metrics() -> None:
    metrics = _sample_risk_metrics()
    panel = risk_return_panel(metrics)
    assert panel.panel_id == "risk.return"
    assert panel.kind == "table"
    metric_names = {row["metric"] for row in panel.payload["rows"]}
    assert "annualized_return" in metric_names
    assert "sharpe_ratio" in metric_names
    assert "max_drawdown" in metric_names


def test_risk_var_panel_contains_var_and_cvar() -> None:
    metrics = _sample_risk_metrics()
    panel = risk_var_panel(metrics)
    assert panel.panel_id == "risk.tail"
    metric_names = {row["metric"] for row in panel.payload["rows"]}
    assert "value_at_risk" in metric_names
    assert "conditional_value_at_risk" in metric_names


def test_risk_exposure_panel_contains_exposure_fields() -> None:
    metrics = _sample_risk_metrics()
    panel = risk_exposure_panel(metrics)
    assert panel.panel_id == "risk.exposure"
    metric_names = {row["metric"] for row in panel.payload["rows"]}
    assert "gross_exposure" in metric_names
    assert "net_exposure" in metric_names
    assert "concentration" in metric_names
    assert "largest_weight" in metric_names


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------


def test_backtest_plugin_panels_matches_backtest_terminal_panels() -> None:
    bundle = _bundle()
    plugin = BacktestPlugin(plugin_id="backtest.v0", bundle=bundle)
    assert plugin.plugin_id == "backtest.v0"
    panels = list(plugin.panels())
    expected = backtest_terminal_panels(bundle)
    assert [p.panel_id for p in panels] == [p.panel_id for p in expected]


def test_risk_plugin_returns_three_panels() -> None:
    metrics = _sample_risk_metrics()
    plugin = RiskPlugin(plugin_id="risk.live", metrics=metrics)
    panels = list(plugin.panels())
    assert len(panels) == 3
    panel_ids = {p.panel_id for p in panels}
    assert "risk.return" in panel_ids
    assert "risk.tail" in panel_ids
    assert "risk.exposure" in panel_ids


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_default_registry_is_terminal_plugin_registry() -> None:
    from qr_haven.integrations.market_terminal.registry import TerminalPluginRegistry

    assert isinstance(default_registry, TerminalPluginRegistry)


def test_registry_register_and_retrieve_plugin() -> None:
    registry = TerminalPluginRegistry()
    metrics = _sample_risk_metrics()
    plugin = RiskPlugin(plugin_id="risk.test", metrics=metrics)
    registry.register(plugin)
    retrieved = registry.get("risk.test")
    assert retrieved.plugin_id == "risk.test"
    assert len(list(retrieved.panels())) == 3


def test_registry_all_returns_plugins_in_sorted_order() -> None:
    registry = TerminalPluginRegistry()
    metrics = _sample_risk_metrics()
    registry.register(RiskPlugin(plugin_id="b.plugin", metrics=metrics))
    registry.register(RiskPlugin(plugin_id="a.plugin", metrics=metrics))
    ids = [p.plugin_id for p in registry.all()]
    assert ids == ["a.plugin", "b.plugin"]


def test_registry_register_replaces_existing_plugin() -> None:
    registry = TerminalPluginRegistry()
    metrics = _sample_risk_metrics()
    registry.register(RiskPlugin(plugin_id="risk.slot", metrics=metrics))
    registry.register(RiskPlugin(plugin_id="risk.slot", metrics=metrics))
    assert len(registry.all()) == 1

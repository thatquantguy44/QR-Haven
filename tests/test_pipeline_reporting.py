"""Tests for PipelineResult reporting and market_terminal panel adapters."""

import numpy as np
import pandas as pd

from qr_haven.costs.borrow import BorrowCostSchedule
from qr_haven.costs.financing import FinancingRates
from qr_haven.costs.market_impact import SquareRootImpactModel
from qr_haven.integrations.market_terminal import (
    ResearchPipelinePlugin,
    pipeline_alpha_scores_panel,
    pipeline_cost_breakdown_panel,
    pipeline_cost_summary_panel,
    pipeline_cumulative_return_panel,
    pipeline_diagnostics_panel,
    pipeline_drawdown_panel,
    pipeline_equity_curve_panel,
    pipeline_rolling_risk_panel,
    pipeline_rolling_turnover_panel,
    pipeline_summary_panel,
    pipeline_terminal_panels,
    pipeline_weights_panel,
)
from qr_haven.portfolio import EqualWeightOptimizer
from qr_haven.reporting import PipelineReportBundle, build_pipeline_report_bundle
from qr_haven.research import PipelineConfig, ResearchPipeline

ASSETS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
N_PERIODS = 200


def _returns(n: int = N_PERIODS, assets: list[str] = ASSETS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0.0005, 0.015, size=(n, len(assets)))
    return pd.DataFrame(data, index=dates, columns=assets)


def _adv_usd(assets: list[str] = ASSETS) -> pd.Series:
    return pd.Series({a: 500_000_000.0 for a in assets})


def _pipeline_result(alpha_scores: pd.DataFrame | None = None):
    config = PipelineConfig(lookback_periods=40, rebalance_periods=20, nav=10_000_000.0)
    pipeline = ResearchPipeline(
        optimizer=EqualWeightOptimizer(),
        borrow_schedule=BorrowCostSchedule.gc_schedule(ASSETS),
        financing_rates=FinancingRates(debit_rate=0.055, credit_rate=0.04),
        impact_model=SquareRootImpactModel(eta=0.1),
        config=config,
    )
    return pipeline.run(_returns(), alpha_scores=alpha_scores, adv_usd=_adv_usd())


def _bundle(alpha_scores: pd.DataFrame | None = None) -> PipelineReportBundle:
    result = _pipeline_result(alpha_scores=alpha_scores)
    return build_pipeline_report_bundle(result, return_window=20, turnover_window=2)


def _alpha_panel(returns: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=returns.shape)
    return pd.DataFrame(data, index=returns.index, columns=returns.columns)


# ---------------------------------------------------------------------------
# Report bundle
# ---------------------------------------------------------------------------


def test_build_pipeline_report_bundle_groups_result_and_analytics() -> None:
    bundle = _bundle()
    assert bundle.result is not None
    assert len(bundle.analytics.drawdown) == len(bundle.result.portfolio_returns)
    assert len(bundle.analytics.rolling_volatility) == len(bundle.result.portfolio_returns)


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------


def test_pipeline_summary_panel_is_table() -> None:
    bundle = _bundle()
    panel = pipeline_summary_panel(bundle.result)
    assert panel.panel_id == "pipeline.summary"
    assert panel.kind == "table"
    metric_names = {row["metric"] for row in panel.payload["rows"]}
    assert "total_return" in metric_names
    assert "cost_drag" in metric_names


def test_pipeline_equity_curve_panel_is_line_chart() -> None:
    bundle = _bundle()
    panel = pipeline_equity_curve_panel(bundle.result)
    assert panel.kind == "line"
    assert panel.payload["x"] == "timestamp"
    assert panel.payload["y"] == "equity"
    assert len(panel.payload["rows"]) == len(bundle.result.equity_curve)


def test_pipeline_cumulative_return_panel_is_line_chart() -> None:
    bundle = _bundle()
    panel = pipeline_cumulative_return_panel(bundle)
    assert panel.kind == "line"
    assert panel.payload["y"] == "cumulative_return"
    assert len(panel.payload["rows"]) > 0


def test_pipeline_drawdown_panel_is_line_chart() -> None:
    bundle = _bundle()
    panel = pipeline_drawdown_panel(bundle)
    assert panel.kind == "line"
    assert panel.payload["y"] == "drawdown"


def test_pipeline_rolling_risk_panel_has_volatility_and_sharpe() -> None:
    bundle = _bundle()
    panel = pipeline_rolling_risk_panel(bundle)
    assert panel.kind == "table"
    assert all(
        "rolling_volatility" in row and "rolling_sharpe" in row for row in panel.payload["rows"]
    )


def test_pipeline_rolling_turnover_panel_is_line_chart() -> None:
    bundle = _bundle()
    panel = pipeline_rolling_turnover_panel(bundle)
    assert panel.kind == "line"
    assert panel.payload["y"] == "rolling_turnover"


def test_pipeline_weights_panel_has_all_assets() -> None:
    bundle = _bundle()
    panel = pipeline_weights_panel(bundle.result)
    assert panel.kind == "table"
    row = panel.payload["rows"][0]
    assert all(asset in row for asset in ASSETS)


def test_pipeline_diagnostics_panel_matches_rebalance_count() -> None:
    bundle = _bundle()
    panel = pipeline_diagnostics_panel(bundle.result)
    assert len(panel.payload["rows"]) == len(bundle.result.diagnostics)


def test_pipeline_cost_breakdown_panel_has_all_cost_components() -> None:
    bundle = _bundle()
    panel = pipeline_cost_breakdown_panel(bundle.result)
    assert len(panel.payload["rows"]) == len(bundle.result.costs.total)
    row = panel.payload["rows"][0]
    expected_keys = {"market_impact_usd", "borrow_cost_usd", "financing_cost_usd", "total_cost_usd"}
    assert expected_keys <= row.keys()


def test_pipeline_cost_breakdown_totals_match_cost_breakdown() -> None:
    bundle = _bundle()
    panel = pipeline_cost_breakdown_panel(bundle.result)
    computed_total = sum(row["total_cost_usd"] for row in panel.payload["rows"])
    assert np.isclose(computed_total, float(bundle.result.costs.total.sum()))


def test_pipeline_cost_summary_panel_matches_costs_summary() -> None:
    bundle = _bundle()
    panel = pipeline_cost_summary_panel(bundle.result)
    metric_names = {row["metric"] for row in panel.payload["rows"]}
    assert metric_names == set(bundle.result.costs.summary().keys())


def test_pipeline_alpha_scores_panel_none_without_alpha_model() -> None:
    bundle = _bundle()
    assert pipeline_alpha_scores_panel(bundle.result) is None


def test_pipeline_alpha_scores_panel_present_with_alpha_model() -> None:
    returns = _returns()
    bundle = _bundle(alpha_scores=_alpha_panel(returns))
    panel = pipeline_alpha_scores_panel(bundle.result)
    assert panel is not None
    assert panel.kind == "table"
    row = panel.payload["rows"][0]
    assert all(asset in row for asset in ASSETS)


# ---------------------------------------------------------------------------
# Full panel package
# ---------------------------------------------------------------------------


def test_pipeline_terminal_panels_without_alpha_excludes_alpha_panel() -> None:
    bundle = _bundle()
    panels = pipeline_terminal_panels(bundle)
    panel_ids = {p.panel_id for p in panels}
    assert "pipeline.alpha_scores" not in panel_ids
    assert len(panels) == 10


def test_pipeline_terminal_panels_with_alpha_includes_alpha_panel() -> None:
    returns = _returns()
    bundle = _bundle(alpha_scores=_alpha_panel(returns))
    panels = pipeline_terminal_panels(bundle)
    panel_ids = {p.panel_id for p in panels}
    assert "pipeline.alpha_scores" in panel_ids
    assert len(panels) == 11


def test_pipeline_terminal_panels_includes_expected_ids() -> None:
    bundle = _bundle()
    panel_ids = {p.panel_id for p in pipeline_terminal_panels(bundle)}
    assert "pipeline.summary" in panel_ids
    assert "pipeline.equity_curve" in panel_ids
    assert "pipeline.cumulative_return" in panel_ids
    assert "pipeline.drawdown" in panel_ids
    assert "pipeline.rolling_risk" in panel_ids
    assert "pipeline.rolling_turnover" in panel_ids
    assert "pipeline.weights" in panel_ids
    assert "pipeline.optimizer_diagnostics_history" in panel_ids
    assert "pipeline.cost_breakdown" in panel_ids
    assert "pipeline.cost_summary" in panel_ids


def test_pipeline_terminal_panels_rows_non_empty() -> None:
    bundle = _bundle()
    assert all(panel.payload["rows"] for panel in pipeline_terminal_panels(bundle))


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


def test_research_pipeline_plugin_panels_matches_pipeline_terminal_panels() -> None:
    bundle = _bundle()
    plugin = ResearchPipelinePlugin(plugin_id="pipeline.v0", bundle=bundle)
    assert plugin.plugin_id == "pipeline.v0"
    panels = list(plugin.panels())
    expected = pipeline_terminal_panels(bundle)
    assert [p.panel_id for p in panels] == [p.panel_id for p in expected]

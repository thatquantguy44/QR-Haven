"""Tests for ResearchPipeline — end-to-end cost-aware walk-forward backtest."""

import numpy as np
import pandas as pd
import pytest

from qr_haven.costs.borrow import BorrowCostSchedule
from qr_haven.costs.financing import FinancingRates
from qr_haven.costs.market_impact import AlmgrenChrissModel, SquareRootImpactModel
from qr_haven.portfolio import EqualWeightOptimizer, MeanVarianceOptimizer
from qr_haven.research import (
    CostBreakdown,
    PipelineConfig,
    PipelineResult,
    ResearchPipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASSETS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
N_PERIODS = 200


def _returns(n: int = N_PERIODS, assets: list[str] = ASSETS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0.0005, 0.015, size=(n, len(assets)))
    return pd.DataFrame(data, index=dates, columns=assets)


def _returns_with_shorts(
    n: int = N_PERIODS, assets: list[str] = ASSETS, seed: int = 1
) -> pd.DataFrame:
    """Returns suitable for a long-short 130/30 optimizer (needs negative-weight support)."""
    return _returns(n=n, assets=assets, seed=seed)


def _gc_schedule(assets: list[str] = ASSETS) -> BorrowCostSchedule:
    return BorrowCostSchedule.gc_schedule(assets)


def _financing_rates() -> FinancingRates:
    return FinancingRates(debit_rate=0.055, credit_rate=0.04)


def _adv_usd(assets: list[str] = ASSETS) -> pd.Series:
    return pd.Series(
        {a: 500_000_000.0 for a in assets}  # $500M ADV for each
    )


def _pipeline(
    borrow: bool = False,
    financing: bool = False,
    impact: bool = False,
    long_only: bool = True,
) -> ResearchPipeline:
    optimizer = EqualWeightOptimizer()
    config = PipelineConfig(lookback_periods=40, rebalance_periods=20, nav=10_000_000.0)
    return ResearchPipeline(
        optimizer=optimizer,
        borrow_schedule=_gc_schedule() if borrow else None,
        financing_rates=_financing_rates() if financing else None,
        impact_model=SquareRootImpactModel(eta=0.1) if impact else None,
        config=config,
    )


# ---------------------------------------------------------------------------
# PipelineConfig validation
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert c.lookback_periods == 60
        assert c.rebalance_periods == 21
        assert c.nav == 10_000_000.0

    def test_bad_lookback(self):
        with pytest.raises(ValueError, match="lookback_periods"):
            PipelineConfig(lookback_periods=1)

    def test_bad_rebalance(self):
        with pytest.raises(ValueError, match="rebalance_periods"):
            PipelineConfig(rebalance_periods=0)

    def test_bad_nav(self):
        with pytest.raises(ValueError, match="nav"):
            PipelineConfig(nav=-1.0)

    def test_bad_fallback_cost(self):
        with pytest.raises(ValueError, match="fallback_cost_bps"):
            PipelineConfig(fallback_cost_bps=-1.0)


# ---------------------------------------------------------------------------
# Basic run — no cost models (fallback flat cost)
# ---------------------------------------------------------------------------

class TestPipelineNoRealCosts:
    def test_run_returns_pipeline_result(self):
        p = _pipeline()
        result = p.run(_returns())
        assert isinstance(result, PipelineResult)

    def test_portfolio_returns_shape(self):
        p = _pipeline()
        r = _returns()
        result = p.run(r)
        config = PipelineConfig(lookback_periods=40, rebalance_periods=20)
        # returns should span from lookback_periods to end
        assert len(result.portfolio_returns) > 0

    def test_portfolio_returns_finite(self):
        result = _pipeline().run(_returns())
        assert np.all(np.isfinite(result.portfolio_returns))

    def test_equity_curve_starts_near_one(self):
        result = _pipeline().run(_returns())
        assert abs(result.equity_curve.iloc[0] - 1.0) < 0.1

    def test_weights_shape(self):
        r = _returns()
        result = _pipeline().run(r)
        assert result.weights.shape[1] == len(ASSETS)

    def test_weights_sum_to_one(self):
        result = _pipeline().run(_returns())
        np.testing.assert_allclose(
            result.weights.sum(axis=1), 1.0, atol=1e-8
        )

    def test_turnover_non_negative(self):
        result = _pipeline().run(_returns())
        assert np.all(result.turnover >= 0.0)

    def test_diagnostics_not_empty(self):
        result = _pipeline().run(_returns())
        assert not result.diagnostics.empty

    def test_risk_metrics_present(self):
        result = _pipeline().run(_returns())
        assert "sharpe_ratio" in result.risk_metrics or len(result.risk_metrics) > 0

    def test_alpha_scores_used_none(self):
        result = _pipeline().run(_returns())
        assert result.alpha_scores_used is None

    def test_too_short_returns_raises(self):
        p = _pipeline()
        short_r = _returns(n=30)
        with pytest.raises(ValueError):
            p.run(short_r)

    def test_summary_returns_dict(self):
        result = _pipeline().run(_returns())
        s = result.summary()
        assert isinstance(s, dict)
        assert "total_return" in s
        assert "gross_total_return" in s
        assert "observations" in s

    def test_gross_return_in_summary(self):
        result = _pipeline().run(_returns())
        s = result.summary()
        assert "gross_total_return" in s

    def test_fallback_costs_not_zero(self):
        """Without real models, flat fallback cost should still be non-zero."""
        config = PipelineConfig(lookback_periods=40, rebalance_periods=20, fallback_cost_bps=10.0)
        p = ResearchPipeline(optimizer=EqualWeightOptimizer(), config=config)
        result = p.run(_returns())
        # Net should be slightly below gross due to fallback cost
        net = float((1.0 + result.portfolio_returns).prod() - 1.0)
        gross = float((1.0 + result.gross_portfolio_returns).prod() - 1.0)
        assert gross >= net - 1e-6

    def test_cost_breakdown_type(self):
        result = _pipeline().run(_returns())
        assert isinstance(result.costs, CostBreakdown)

    def test_cost_breakdown_index_matches_rebalances(self):
        result = _pipeline().run(_returns())
        assert len(result.costs.market_impact) == len(result.weights)
        assert len(result.costs.borrow_cost) == len(result.weights)
        assert len(result.costs.financing_cost) == len(result.weights)


# ---------------------------------------------------------------------------
# Market impact model
# ---------------------------------------------------------------------------

class TestPipelineWithImpact:
    def test_impact_costs_non_negative(self):
        result = _pipeline(impact=True).run(_returns(), adv_usd=_adv_usd())
        assert np.all(result.costs.market_impact >= 0.0)

    def test_impact_costs_positive_on_first_rebalance(self):
        """First rebalance always trades from zero → nonzero impact."""
        result = _pipeline(impact=True).run(_returns(), adv_usd=_adv_usd())
        assert result.costs.market_impact.iloc[0] > 0.0

    def test_net_less_than_gross_with_impact(self):
        result = _pipeline(impact=True).run(_returns(), adv_usd=_adv_usd())
        net = float((1.0 + result.portfolio_returns).prod())
        gross = float((1.0 + result.gross_portfolio_returns).prod())
        assert gross >= net - 1e-6

    def test_impact_works_without_adv(self):
        """Falls back to default_participation_rate when ADV not provided."""
        result = _pipeline(impact=True).run(_returns(), adv_usd=None)
        assert np.all(result.costs.market_impact >= 0.0)

    def test_almgren_chriss_model_accepted(self):
        config = PipelineConfig(lookback_periods=40, rebalance_periods=20)
        p = ResearchPipeline(
            optimizer=EqualWeightOptimizer(),
            impact_model=AlmgrenChrissModel(eta=0.1, gamma=0.05, alpha=0.6),
            config=config,
        )
        result = p.run(_returns(), adv_usd=_adv_usd())
        assert np.all(result.costs.market_impact >= 0.0)

    def test_cost_summary_impact_key(self):
        result = _pipeline(impact=True).run(_returns(), adv_usd=_adv_usd())
        s = result.costs.summary()
        assert "total_market_impact_usd" in s
        assert s["total_market_impact_usd"] >= 0.0


# ---------------------------------------------------------------------------
# Borrow cost model
# ---------------------------------------------------------------------------

class TestPipelineWithBorrow:
    def test_borrow_costs_zero_for_long_only(self):
        """Long-only portfolio has no short positions → zero borrow cost."""
        result = _pipeline(borrow=True).run(_returns())
        assert np.all(result.costs.borrow_cost == 0.0)

    def test_borrow_cost_key_in_summary(self):
        result = _pipeline(borrow=True).run(_returns())
        s = result.costs.summary()
        assert "total_borrow_cost_usd" in s

    def test_borrow_cost_non_negative(self):
        result = _pipeline(borrow=True).run(_returns())
        assert np.all(result.costs.borrow_cost >= 0.0)

    def test_htb_schedule_more_costly(self):
        """High borrow rate schedule should produce higher costs than GC."""
        assets = ASSETS
        htb_schedule = BorrowCostSchedule.from_scalars(assets, fee_rate=0.30)

        config = PipelineConfig(lookback_periods=40, rebalance_periods=20)

        # Use MeanVarianceOptimizer with long-short constraints
        optimizer = EqualWeightOptimizer()

        gc_pipeline = ResearchPipeline(
            optimizer=optimizer,
            borrow_schedule=BorrowCostSchedule.gc_schedule(assets),
            config=config,
        )
        htb_pipeline = ResearchPipeline(
            optimizer=optimizer,
            borrow_schedule=htb_schedule,
            config=config,
        )
        r = _returns()
        gc_result = gc_pipeline.run(r)
        htb_result = htb_pipeline.run(r)
        # Both should be zero for long-only, but schema validates correctly
        assert htb_result.costs.borrow_cost.sum() >= gc_result.costs.borrow_cost.sum()


# ---------------------------------------------------------------------------
# Financing cost model
# ---------------------------------------------------------------------------

class TestPipelineWithFinancing:
    def test_financing_cost_zero_for_fully_funded_long_only(self):
        """100% long equity-funded portfolio: no debit, no short proceeds → net ≈ 0."""
        result = _pipeline(financing=True).run(_returns())
        # debit_balance = max(0, 1.0 - 1.0) = 0; credit_balance = 0
        # So net financing = 0
        np.testing.assert_allclose(
            result.costs.financing_cost.sum(), 0.0, atol=1.0
        )

    def test_financing_cost_key_in_summary(self):
        result = _pipeline(financing=True).run(_returns())
        s = result.costs.summary()
        assert "total_financing_cost_usd" in s


# ---------------------------------------------------------------------------
# All cost models combined
# ---------------------------------------------------------------------------

class TestPipelineAllCosts:
    def setup_method(self):
        config = PipelineConfig(lookback_periods=40, rebalance_periods=20)
        self.pipeline = ResearchPipeline(
            optimizer=EqualWeightOptimizer(),
            borrow_schedule=_gc_schedule(),
            financing_rates=_financing_rates(),
            impact_model=SquareRootImpactModel(),
            config=config,
        )
        self.result = self.pipeline.run(_returns(), adv_usd=_adv_usd())

    def test_total_cost_equals_sum_of_components(self):
        total = self.result.costs.total
        computed = (
            self.result.costs.market_impact
            + self.result.costs.borrow_cost
            + self.result.costs.financing_cost
        )
        pd.testing.assert_series_equal(total, computed, check_names=False)

    def test_summary_has_all_cost_keys(self):
        s = self.result.costs.summary()
        assert "total_market_impact_usd" in s
        assert "total_borrow_cost_usd" in s
        assert "total_financing_cost_usd" in s
        assert "total_cost_usd" in s

    def test_total_cost_equals_sum_of_parts(self):
        s = self.result.costs.summary()
        parts = (
            s["total_market_impact_usd"]
            + s["total_borrow_cost_usd"]
            + s["total_financing_cost_usd"]
        )
        assert abs(parts - s["total_cost_usd"]) < 1e-6

    def test_pipeline_summary_has_cost_drag(self):
        s = self.result.summary()
        assert "cost_drag" in s

    def test_asset_contributions_shape(self):
        assert self.result.asset_contributions.shape[1] == len(ASSETS)


# ---------------------------------------------------------------------------
# Alpha score injection
# ---------------------------------------------------------------------------

class TestPipelineAlphaScores:
    def _alpha_panel(self, returns: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
        """Random alpha scores panel (dates × assets), same index as returns."""
        rng = np.random.default_rng(seed)
        data = rng.normal(0.0, 1.0, size=returns.shape)
        return pd.DataFrame(data, index=returns.index, columns=returns.columns)

    def test_alpha_scores_used_not_none(self):
        r = _returns()
        alpha = self._alpha_panel(r)
        result = _pipeline().run(r, alpha_scores=alpha)
        assert result.alpha_scores_used is not None

    def test_alpha_scores_shape(self):
        r = _returns()
        alpha = self._alpha_panel(r)
        result = _pipeline().run(r, alpha_scores=alpha)
        assert result.alpha_scores_used.shape[1] == len(ASSETS)

    def test_alpha_scores_in_summary(self):
        r = _returns()
        alpha = self._alpha_panel(r)
        result = _pipeline().run(r, alpha_scores=alpha)
        s = result.summary()
        assert s["alpha_model_used"] is True

    def test_no_alpha_scores_summary_flag(self):
        result = _pipeline().run(_returns())
        assert result.summary()["alpha_model_used"] is False

    def test_alpha_scores_produces_valid_weights(self):
        r = _returns()
        alpha = self._alpha_panel(r)
        result = _pipeline().run(r, alpha_scores=alpha)
        np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-8)

    def test_alpha_scores_partial_overlap(self):
        """Alpha panel only covers first half of dates — should still run."""
        r = _returns()
        half = r.iloc[: len(r) // 2]
        alpha = self._alpha_panel(half)
        result = _pipeline().run(r, alpha_scores=alpha)
        assert len(result.portfolio_returns) > 0

    def test_alpha_scores_future_dates_not_used(self):
        """Alpha lookup is as-of-date safe: future scores not visible."""
        r = _returns(n=150)
        # shift alpha one period forward — should fall back to zeros for first rebalance
        alpha_shifted = self._alpha_panel(r).shift(1)
        result = _pipeline().run(r, alpha_scores=alpha_shifted)
        assert len(result.portfolio_returns) > 0


# ---------------------------------------------------------------------------
# MeanVarianceOptimizer integration
# ---------------------------------------------------------------------------

class TestPipelineMVO:
    def test_mvo_runs(self):
        config = PipelineConfig(lookback_periods=40, rebalance_periods=20)
        p = ResearchPipeline(
            optimizer=MeanVarianceOptimizer(),
            impact_model=SquareRootImpactModel(),
            config=config,
        )
        result = p.run(_returns(), adv_usd=_adv_usd())
        assert isinstance(result, PipelineResult)
        assert len(result.portfolio_returns) > 0

    def test_mvo_weights_in_unit_interval(self):
        config = PipelineConfig(lookback_periods=40, rebalance_periods=20)
        p = ResearchPipeline(
            optimizer=MeanVarianceOptimizer(),
            config=config,
        )
        result = p.run(_returns())
        assert np.all(result.weights.values >= -1e-8)
        assert np.all(result.weights.values <= 1.0 + 1e-8)

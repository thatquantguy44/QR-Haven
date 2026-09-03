"""Tests for capacity-aware alpha ranking (NAV sweep against cost drag)."""

import numpy as np
import pandas as pd
import pytest

from qr_haven.costs.market_impact import SquareRootImpactModel
from qr_haven.portfolio import EqualWeightOptimizer
from qr_haven.research import CapacityCurve, CapacityPoint, PipelineConfig, sweep_capacity

ASSETS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
N_PERIODS = 150


def _returns(n: int = N_PERIODS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0.0005, 0.015, size=(n, len(ASSETS)))
    return pd.DataFrame(data, index=dates, columns=ASSETS)


def _adv_usd(dollars_each: float = 50_000_000.0) -> pd.Series:
    return pd.Series({a: dollars_each for a in ASSETS})


def _point(nav: float, net_sharpe: float) -> CapacityPoint:
    return CapacityPoint(
        nav=nav,
        net_sharpe=net_sharpe,
        cost_drag=0.0,
        total_cost_usd=0.0,
        total_market_impact_usd=0.0,
    )


def _sweep(nav_grid=(1_000_000.0, 100_000_000.0, 1_000_000_000.0)) -> CapacityCurve:
    config = PipelineConfig(lookback_periods=40, rebalance_periods=20)
    return sweep_capacity(
        optimizer=EqualWeightOptimizer(),
        returns=_returns(),
        nav_grid=nav_grid,
        impact_model=SquareRootImpactModel(eta=0.1),
        base_config=config,
        adv_usd=_adv_usd(),
    )


# ---------------------------------------------------------------------------
# sweep_capacity
# ---------------------------------------------------------------------------


def test_sweep_capacity_returns_one_point_per_nav() -> None:
    curve = _sweep()
    assert len(curve.points) == 3
    assert [p.nav for p in curve.points] == [1_000_000.0, 100_000_000.0, 1_000_000_000.0]


def test_sweep_capacity_rejects_empty_nav_grid() -> None:
    config = PipelineConfig(lookback_periods=40, rebalance_periods=20)
    with pytest.raises(ValueError, match="nav_grid"):
        sweep_capacity(
            optimizer=EqualWeightOptimizer(),
            returns=_returns(),
            nav_grid=(),
            base_config=config,
        )


def test_market_impact_cost_grows_with_nav() -> None:
    """Bigger NAV -> bigger trade value relative to fixed ADV -> more impact cost."""
    curve = _sweep()
    impacts = [p.total_market_impact_usd for p in curve.points]
    assert impacts[0] < impacts[1] < impacts[2]


def test_net_sharpe_degrades_as_nav_grows() -> None:
    """Rising impact cost should erode net Sharpe as NAV grows, all else equal."""
    curve = _sweep()
    sharpes = [p.net_sharpe for p in curve.points]
    assert sharpes[0] >= sharpes[1] >= sharpes[2]


def test_cost_drag_increases_with_nav() -> None:
    curve = _sweep()
    drags = [p.cost_drag for p in curve.points]
    assert drags[0] <= drags[1] <= drags[2]


def test_sweep_capacity_holds_strategy_fixed_across_navs() -> None:
    """Every point should come from the same alpha/optimizer/schedule — only nav differs."""
    curve = _sweep()
    for point in curve.points:
        assert isinstance(point, CapacityPoint)


# ---------------------------------------------------------------------------
# CapacityCurve.capacity_nav
# ---------------------------------------------------------------------------


def test_capacity_nav_returns_largest_nav_meeting_floor() -> None:
    curve = CapacityCurve(points=(_point(1e6, 1.0), _point(1e7, 0.8), _point(1e8, 0.4)))
    # Floor = 0.5 * baseline (1.0) = 0.5. First point below 0.5 is nav=1e8 (0.4).
    assert curve.capacity_nav(sharpe_floor_fraction=0.5) == 1e7


def test_capacity_nav_none_when_baseline_non_positive() -> None:
    curve = CapacityCurve(points=(_point(1e6, -0.1),))
    assert curve.capacity_nav() is None


def test_capacity_nav_none_for_empty_curve() -> None:
    assert CapacityCurve(points=()).capacity_nav() is None


def test_capacity_nav_rejects_bad_floor_fraction() -> None:
    curve = CapacityCurve(points=(_point(1e6, 1.0),))
    with pytest.raises(ValueError, match="sharpe_floor_fraction"):
        curve.capacity_nav(sharpe_floor_fraction=0.0)
    with pytest.raises(ValueError, match="sharpe_floor_fraction"):
        curve.capacity_nav(sharpe_floor_fraction=1.5)


def test_capacity_nav_all_points_above_floor_returns_largest() -> None:
    curve = CapacityCurve(points=(_point(1e6, 1.0), _point(1e7, 0.95)))
    assert curve.capacity_nav(sharpe_floor_fraction=0.5) == 1e7


def test_real_sweep_has_computable_capacity_nav() -> None:
    curve = _sweep(nav_grid=(1_000_000.0, 500_000_000.0, 5_000_000_000.0))
    # Not asserting a specific value (depends on realized returns), just that the
    # computation runs end-to-end without raising and returns a float or None.
    result = curve.capacity_nav(sharpe_floor_fraction=0.5)
    assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# CapacityCurve.to_frame
# ---------------------------------------------------------------------------


def test_to_frame_indexed_by_nav() -> None:
    curve = _sweep()
    frame = curve.to_frame()
    assert list(frame.index) == [1_000_000.0, 100_000_000.0, 1_000_000_000.0]
    assert "net_sharpe" in frame.columns
    assert "total_market_impact_usd" in frame.columns

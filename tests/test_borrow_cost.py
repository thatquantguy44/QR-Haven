import math

import pandas as pd
import pytest

from qr_haven.costs import (
    BorrowCostModel,
    BorrowCostResult,
    BorrowCostSchedule,
)


# ---------------------------------------------------------------------------
# BorrowCostSchedule
# ---------------------------------------------------------------------------


def test_schedule_rejects_negative_fee_rate() -> None:
    idx = pd.Index(["AAPL"], name="symbol")
    with pytest.raises(ValueError, match="fee_rates"):
        BorrowCostSchedule(fee_rates=pd.Series([-0.01], index=idx))


def test_schedule_rejects_non_positive_days_in_year() -> None:
    idx = pd.Index(["AAPL"], name="symbol")
    with pytest.raises(ValueError, match="days_in_year"):
        BorrowCostSchedule(
            fee_rates=pd.Series([0.01], index=idx),
            days_in_year=0.0,
        )


def test_from_scalars_builds_uniform_schedule() -> None:
    symbols = ["AAPL", "MSFT", "GOOGL"]
    sched = BorrowCostSchedule.from_scalars(symbols, fee_rate=0.05)
    assert list(sched.fee_rates.index) == symbols
    assert (sched.fee_rates == 0.05).all()
    assert sched.days_in_year == 360.0


def test_gc_schedule_defaults_to_20_bps() -> None:
    sched = BorrowCostSchedule.gc_schedule(["AAPL", "MSFT"])
    assert math.isclose(sched.fee_rates["AAPL"], 0.002)
    assert math.isclose(sched.fee_rates["MSFT"], 0.002)


def test_gc_schedule_custom_rate() -> None:
    sched = BorrowCostSchedule.gc_schedule(["XYZ"], gc_rate=0.01)
    assert math.isclose(sched.fee_rates["XYZ"], 0.01)


# ---------------------------------------------------------------------------
# BorrowCostModel — known-value tests
# ---------------------------------------------------------------------------


def _simple_schedule(symbols, fee_rate=0.10):
    return BorrowCostSchedule.from_scalars(symbols, fee_rate=fee_rate)


def test_estimate_no_short_positions_gives_zero_cost() -> None:
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    sched = _simple_schedule(["AAPL", "MSFT"])
    result = BorrowCostModel().estimate(weights, nav=1_000_000, schedule=sched)
    assert result.total_borrow_cost == 0.0
    assert result.cost_on_nav == 0.0


def test_estimate_known_value_single_short() -> None:
    # short_weight=0.30, fee=0.10 ann, holding=1 day, days_in_year=360
    # cost = 0.30 × 1_000_000 × 0.10 × (1/360) = 83.333...
    weights = pd.Series({"AAPL": -0.30})
    sched = BorrowCostSchedule.from_scalars(["AAPL"], fee_rate=0.10)
    result = BorrowCostModel().estimate(weights, nav=1_000_000, schedule=sched)
    expected = 0.30 * 1_000_000 * 0.10 * (1 / 360)
    assert math.isclose(result.total_borrow_cost, expected, rel_tol=1e-9)


def test_estimate_cost_on_nav_is_annualized() -> None:
    # With holding_period_days=30, cost_on_nav should equal total/nav × (360/30)
    weights = pd.Series({"AAPL": -0.30})
    sched = BorrowCostSchedule.from_scalars(["AAPL"], fee_rate=0.10)
    result = BorrowCostModel().estimate(
        weights, nav=1_000_000, schedule=sched, holding_period_days=30
    )
    expected_annualized = (result.total_borrow_cost / 1_000_000) * (360 / 30)
    assert math.isclose(result.cost_on_nav, expected_annualized, rel_tol=1e-9)


def test_estimate_long_weights_ignored() -> None:
    weights = pd.Series({"AAPL": 0.60, "MSFT": -0.30})
    sched = BorrowCostSchedule.from_scalars(["AAPL", "MSFT"], fee_rate=0.10)
    result = BorrowCostModel().estimate(weights, nav=1_000_000, schedule=sched)
    # Only MSFT (short) contributes
    expected = 0.30 * 1_000_000 * 0.10 / 360
    assert math.isclose(result.total_borrow_cost, expected, rel_tol=1e-9)
    assert result.position_costs["AAPL"] == 0.0


def test_estimate_multiple_shorts_sum_correctly() -> None:
    weights = pd.Series({"AAPL": -0.20, "MSFT": -0.10})
    sched = BorrowCostSchedule.from_scalars(["AAPL", "MSFT"], fee_rate=0.10)
    result = BorrowCostModel().estimate(weights, nav=1_000_000, schedule=sched)
    expected = (0.20 + 0.10) * 1_000_000 * 0.10 / 360
    assert math.isclose(result.total_borrow_cost, expected, rel_tol=1e-9)


def test_estimate_symbols_missing_from_schedule_ignored() -> None:
    weights = pd.Series({"AAPL": -0.30, "UNKNOWN": -0.10})
    sched = BorrowCostSchedule.from_scalars(["AAPL"], fee_rate=0.10)
    result = BorrowCostModel().estimate(weights, nav=1_000_000, schedule=sched)
    expected = 0.30 * 1_000_000 * 0.10 / 360
    assert math.isclose(result.total_borrow_cost, expected, rel_tol=1e-9)


def test_estimate_per_symbol_heterogeneous_rates() -> None:
    weights = pd.Series({"EASY": -0.20, "HTB": -0.10})
    idx = pd.Index(["EASY", "HTB"], name="symbol")
    fee_rates = pd.Series([0.002, 0.30], index=idx, name="fee_rate")
    sched = BorrowCostSchedule(fee_rates=fee_rates)
    result = BorrowCostModel().estimate(weights, nav=1_000_000, schedule=sched)
    easy_cost = 0.20 * 1_000_000 * 0.002 / 360
    htb_cost = 0.10 * 1_000_000 * 0.30 / 360
    assert math.isclose(result.position_costs["EASY"], easy_cost, rel_tol=1e-9)
    assert math.isclose(result.position_costs["HTB"], htb_cost, rel_tol=1e-9)


def test_estimate_rejects_non_positive_nav() -> None:
    weights = pd.Series({"AAPL": -0.30})
    sched = _simple_schedule(["AAPL"])
    with pytest.raises(ValueError, match="nav"):
        BorrowCostModel().estimate(weights, nav=0.0, schedule=sched)


def test_estimate_rejects_non_positive_holding_period() -> None:
    weights = pd.Series({"AAPL": -0.30})
    sched = _simple_schedule(["AAPL"])
    with pytest.raises(ValueError, match="holding_period_days"):
        BorrowCostModel().estimate(weights, nav=1_000_000, schedule=sched, holding_period_days=0)


# ---------------------------------------------------------------------------
# BorrowCostResult.summary
# ---------------------------------------------------------------------------


def test_result_summary_contains_expected_keys() -> None:
    weights = pd.Series({"AAPL": -0.30})
    sched = _simple_schedule(["AAPL"])
    result = BorrowCostModel().estimate(weights, nav=1_000_000, schedule=sched)
    s = result.summary()
    assert set(s) == {"total_borrow_cost", "cost_on_nav"}
    assert s["total_borrow_cost"] == result.total_borrow_cost

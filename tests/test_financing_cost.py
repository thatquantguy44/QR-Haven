import math

import pandas as pd
import pytest

from qr_haven.costs import (
    FinancingCostModel,
    FinancingCostResult,
    FinancingRates,
)


# ---------------------------------------------------------------------------
# FinancingRates
# ---------------------------------------------------------------------------


def test_rates_rejects_negative_debit_rate() -> None:
    with pytest.raises(ValueError, match="debit_rate"):
        FinancingRates(debit_rate=-0.01, credit_rate=0.0)


def test_rates_rejects_negative_credit_rate() -> None:
    with pytest.raises(ValueError, match="credit_rate"):
        FinancingRates(debit_rate=0.05, credit_rate=-0.01)


def test_rates_rejects_non_positive_days_in_year() -> None:
    with pytest.raises(ValueError, match="days_in_year"):
        FinancingRates(debit_rate=0.05, credit_rate=0.0, days_in_year=0.0)


def test_rates_valid_construction() -> None:
    r = FinancingRates(debit_rate=0.055, credit_rate=0.02)
    assert math.isclose(r.debit_rate, 0.055)
    assert r.days_in_year == 360.0


# ---------------------------------------------------------------------------
# FinancingCostModel — structural tests
# ---------------------------------------------------------------------------


def _rates(debit=0.055, credit=0.02):
    return FinancingRates(debit_rate=debit, credit_rate=credit)


def test_long_only_no_leverage_zero_debit_balance() -> None:
    # 100% long, fully equity-funded → no margin debit
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})
    result = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates())
    assert result.debit_balance == 0.0
    assert result.credit_balance == 0.0
    assert result.net_financing_cost == 0.0


def test_130_30_zero_debit_balance() -> None:
    # 130/30: long_gross=1.30, short_gross=0.30
    # debit = max(0, 1.30 − 1.0 − 0.30) × NAV ≈ 0 (IEEE 754 rounding)
    weights = pd.Series({"A": 1.30, "B": -0.30})
    result = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates())
    assert math.isclose(result.debit_balance, 0.0, abs_tol=1e-6)
    assert math.isclose(result.credit_balance, 0.30 * 1_000_000, rel_tol=1e-9)


def test_leveraged_long_creates_debit_balance() -> None:
    # 150/30: long_gross=1.50, short_gross=0.30
    # debit = max(0, 1.50 − 1.0 − 0.30) × NAV = 0.20 × NAV
    weights = pd.Series({"A": 1.50, "B": -0.30})
    result = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates())
    assert math.isclose(result.debit_balance, 0.20 * 1_000_000, rel_tol=1e-9)


def test_market_neutral_zero_debit_balance() -> None:
    # 100/100: long_gross=1.0, short_gross=1.0
    # debit = max(0, 1.0 − 1.0 − 1.0) = 0; credit = 1.0 × NAV
    weights = pd.Series({"A": 1.0, "B": -1.0})
    result = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates())
    assert result.debit_balance == 0.0
    assert math.isclose(result.credit_balance, 1_000_000, rel_tol=1e-9)


def test_debit_cost_known_value() -> None:
    # debit=0.20 × 1_000_000, debit_rate=0.055, 1 day, 360 days/yr
    # debit_cost = 200_000 × 0.055 / 360 = 30.555...
    weights = pd.Series({"A": 1.50, "B": -0.30})
    result = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates(debit=0.055))
    expected = 0.20 * 1_000_000 * 0.055 / 360
    assert math.isclose(result.debit_cost, expected, rel_tol=1e-9)


def test_credit_income_known_value() -> None:
    # credit=0.30 × 1_000_000, credit_rate=0.02, 1 day
    # credit_income = 300_000 × 0.02 / 360 = 16.666...
    weights = pd.Series({"A": 1.30, "B": -0.30})
    result = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates(credit=0.02))
    expected = 0.30 * 1_000_000 * 0.02 / 360
    assert math.isclose(result.credit_income, expected, rel_tol=1e-9)


def test_net_financing_cost_is_debit_minus_credit() -> None:
    weights = pd.Series({"A": 1.50, "B": -0.30})
    result = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates())
    assert math.isclose(
        result.net_financing_cost, result.debit_cost - result.credit_income, rel_tol=1e-12
    )


def test_cost_on_nav_is_annualized() -> None:
    # With holding_period_days=30, cost_on_nav = (net/nav) × (360/30)
    weights = pd.Series({"A": 1.50, "B": -0.30})
    result = FinancingCostModel().estimate(
        weights, nav=1_000_000, rates=_rates(), holding_period_days=30
    )
    expected = (result.net_financing_cost / 1_000_000) * (360 / 30)
    assert math.isclose(result.cost_on_nav, expected, rel_tol=1e-9)


def test_market_neutral_earns_net_credit_income() -> None:
    # Zero debit balance → net_financing_cost is negative (net income)
    weights = pd.Series({"A": 0.50, "B": -0.50})
    result = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates(credit=0.02))
    assert result.debit_balance == 0.0
    assert result.net_financing_cost < 0.0  # income to the fund


def test_zero_credit_rate_no_credit_income() -> None:
    weights = pd.Series({"A": 1.30, "B": -0.30})
    result = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates(credit=0.0))
    assert result.credit_income == 0.0


def test_holding_period_scales_costs_linearly() -> None:
    weights = pd.Series({"A": 1.50, "B": -0.30})
    r1 = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates(), holding_period_days=1)
    r30 = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates(), holding_period_days=30)
    assert math.isclose(r30.debit_cost, 30 * r1.debit_cost, rel_tol=1e-9)
    assert math.isclose(r30.credit_income, 30 * r1.credit_income, rel_tol=1e-9)


def test_estimate_rejects_non_positive_nav() -> None:
    with pytest.raises(ValueError, match="nav"):
        FinancingCostModel().estimate(pd.Series({"A": 1.0}), nav=0.0, rates=_rates())


def test_estimate_rejects_non_positive_holding_period() -> None:
    with pytest.raises(ValueError, match="holding_period_days"):
        FinancingCostModel().estimate(
            pd.Series({"A": 1.0}), nav=1_000_000, rates=_rates(), holding_period_days=0
        )


# ---------------------------------------------------------------------------
# FinancingCostResult.summary
# ---------------------------------------------------------------------------


def test_result_summary_contains_expected_keys() -> None:
    weights = pd.Series({"A": 1.50, "B": -0.30})
    result = FinancingCostModel().estimate(weights, nav=1_000_000, rates=_rates())
    s = result.summary()
    expected_keys = {
        "debit_balance",
        "credit_balance",
        "debit_cost",
        "credit_income",
        "net_financing_cost",
        "cost_on_nav",
    }
    assert set(s) == expected_keys
    assert s["net_financing_cost"] == result.net_financing_cost

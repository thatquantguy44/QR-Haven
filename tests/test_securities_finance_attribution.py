import math

import pandas as pd
import pytest

from qr_haven.costs import (
    BorrowCostSchedule,
    FinancingRates,
    LendingFeeSchedule,
    SecuritiesFinanceAttribution,
    SecuritiesFinanceAttributionModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _weights_130_30():
    return pd.Series({"AAPL": 0.80, "MSFT": 0.50, "TSLA": -0.30})


def _weights_market_neutral():
    return pd.Series({"AAPL": 0.50, "MSFT": -0.50})


def _lending_schedule(symbols):
    return LendingFeeSchedule.from_scalars(
        symbols,
        fee_rate=0.002,
        on_loan_fraction=0.20,
        reinvestment_spread=0.001,
    )


def _borrow_schedule(symbols):
    return BorrowCostSchedule.from_scalars(symbols, fee_rate=0.10)


def _financing_rates():
    return FinancingRates(debit_rate=0.055, credit_rate=0.02)


# ---------------------------------------------------------------------------
# SecuritiesFinanceAttributionModel — structural
# ---------------------------------------------------------------------------


def test_attribute_returns_attribution_instance() -> None:
    w = _weights_130_30()
    symbols = list(w.index)
    model = SecuritiesFinanceAttributionModel()
    result = model.attribute(
        weights=w,
        nav=1_000_000,
        lending_schedule=_lending_schedule(symbols),
        borrow_schedule=_borrow_schedule(symbols),
        financing_rates=_financing_rates(),
    )
    assert isinstance(result, SecuritiesFinanceAttribution)


def test_net_securities_finance_is_correct_sign_combination() -> None:
    w = _weights_130_30()
    symbols = list(w.index)
    model = SecuritiesFinanceAttributionModel()
    result = model.attribute(
        weights=w,
        nav=1_000_000,
        lending_schedule=_lending_schedule(symbols),
        borrow_schedule=_borrow_schedule(symbols),
        financing_rates=_financing_rates(),
    )
    expected_net = (
        result.lending_revenue.total_lending_revenue
        - result.borrow_cost.total_borrow_cost
        - result.financing_cost.net_financing_cost
    )
    assert math.isclose(result.net_securities_finance, expected_net, rel_tol=1e-12)


def test_net_on_nav_is_annualized() -> None:
    w = _weights_130_30()
    symbols = list(w.index)
    ls = _lending_schedule(symbols)
    model = SecuritiesFinanceAttributionModel()
    result = model.attribute(
        weights=w,
        nav=1_000_000,
        lending_schedule=ls,
        borrow_schedule=_borrow_schedule(symbols),
        financing_rates=_financing_rates(),
        holding_period_days=30,
    )
    expected = (result.net_securities_finance / 1_000_000) * (ls.days_in_year / 30)
    assert math.isclose(result.net_on_nav, expected, rel_tol=1e-9)


def test_long_only_portfolio_no_borrow_cost() -> None:
    w = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
    symbols = list(w.index)
    model = SecuritiesFinanceAttributionModel()
    result = model.attribute(
        weights=w,
        nav=1_000_000,
        lending_schedule=_lending_schedule(symbols),
        borrow_schedule=_borrow_schedule(symbols),
        financing_rates=_financing_rates(),
    )
    assert result.borrow_cost.total_borrow_cost == 0.0
    assert result.financing_cost.debit_balance == 0.0


def test_short_only_portfolio_no_lending_revenue() -> None:
    # Short-only: no long positions to lend
    w = pd.Series({"TSLA": -0.50, "GME": -0.50})
    symbols = list(w.index)
    model = SecuritiesFinanceAttributionModel()
    result = model.attribute(
        weights=w,
        nav=1_000_000,
        lending_schedule=_lending_schedule(symbols),
        borrow_schedule=_borrow_schedule(symbols),
        financing_rates=_financing_rates(),
    )
    assert result.lending_revenue.total_lending_revenue == 0.0
    assert result.borrow_cost.total_borrow_cost > 0.0


def test_market_neutral_earns_net_income() -> None:
    # Equal long and short, low borrow rate, high credit rate → net positive
    w = _weights_market_neutral()
    symbols = list(w.index)
    model = SecuritiesFinanceAttributionModel()
    result = model.attribute(
        weights=w,
        nav=1_000_000,
        lending_schedule=LendingFeeSchedule.from_scalars(
            symbols, fee_rate=0.002, on_loan_fraction=0.30, reinvestment_spread=0.002
        ),
        borrow_schedule=BorrowCostSchedule.from_scalars(symbols, fee_rate=0.002),
        financing_rates=FinancingRates(debit_rate=0.0, credit_rate=0.02),
    )
    # zero borrow drag, credit income > borrow cost → net positive
    assert result.net_securities_finance > 0.0


# ---------------------------------------------------------------------------
# SecuritiesFinanceAttribution.summary
# ---------------------------------------------------------------------------


def test_summary_contains_all_keys() -> None:
    w = _weights_130_30()
    symbols = list(w.index)
    model = SecuritiesFinanceAttributionModel()
    result = model.attribute(
        weights=w,
        nav=1_000_000,
        lending_schedule=_lending_schedule(symbols),
        borrow_schedule=_borrow_schedule(symbols),
        financing_rates=_financing_rates(),
    )
    s = result.summary()
    expected_keys = {
        "total_lending_revenue",
        "lending_revenue_on_nav",
        "total_borrow_cost",
        "borrow_cost_on_nav",
        "debit_cost",
        "credit_income",
        "net_financing_cost",
        "financing_cost_on_nav",
        "net_securities_finance",
        "net_on_nav",
    }
    assert set(s) == expected_keys
    assert s["net_securities_finance"] == result.net_securities_finance


# ---------------------------------------------------------------------------
# SecuritiesFinanceAttribution.waterfall
# ---------------------------------------------------------------------------


def test_waterfall_returns_series_with_correct_index() -> None:
    w = _weights_130_30()
    symbols = list(w.index)
    model = SecuritiesFinanceAttributionModel()
    result = model.attribute(
        weights=w,
        nav=1_000_000,
        lending_schedule=_lending_schedule(symbols),
        borrow_schedule=_borrow_schedule(symbols),
        financing_rates=_financing_rates(),
    )
    wf = result.waterfall()
    assert isinstance(wf, pd.Series)
    assert "net_securities_finance" in wf.index
    assert "lending_fee_income" in wf.index
    assert "borrow_cost" in wf.index


def test_waterfall_net_matches_attribution() -> None:
    w = _weights_130_30()
    symbols = list(w.index)
    model = SecuritiesFinanceAttributionModel()
    result = model.attribute(
        weights=w,
        nav=1_000_000,
        lending_schedule=_lending_schedule(symbols),
        borrow_schedule=_borrow_schedule(symbols),
        financing_rates=_financing_rates(),
    )
    wf = result.waterfall()
    assert math.isclose(
        wf["net_securities_finance"], result.net_securities_finance, rel_tol=1e-12
    )


def test_waterfall_borrow_cost_is_negative() -> None:
    # Borrow cost is a drag → appears as negative in the waterfall
    w = _weights_130_30()
    symbols = list(w.index)
    model = SecuritiesFinanceAttributionModel()
    result = model.attribute(
        weights=w,
        nav=1_000_000,
        lending_schedule=_lending_schedule(symbols),
        borrow_schedule=_borrow_schedule(symbols),
        financing_rates=_financing_rates(),
    )
    wf = result.waterfall()
    assert wf["borrow_cost"] <= 0.0

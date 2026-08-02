import numpy as np
import pandas as pd
import pytest

from qr_haven.costs import (
    LendingFeeSchedule,
    SecuritiesLendingRevenueModel,
)
from qr_haven.features import (
    FeatureRegistry,
    borrow_fee_rate,
    compute_features,
    days_to_cover,
    securities_lending_feature_registry,
    short_interest_change,
    short_interest_ratio,
    utilization_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weights(symbols: list[str], values: list[float]) -> pd.Series:
    return pd.Series(dict(zip(symbols, values)), name="weight")


def _sl_frame(
    n: int,
    symbol: str = "AAPL",
    *,
    short_interest: float = 1_000_000,
    float_shares: float = 10_000_000,
    volume: float = 500_000,
    on_loan: float = 800_000,
    lendable_supply: float = 1_000_000,
    fee_rate: float = 0.01,
) -> pd.DataFrame:
    """Build a canonical MultiIndex frame with securities-lending columns."""
    timestamps = pd.date_range("2024-01-01", periods=n, tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [timestamps, [symbol] * n], names=["timestamp", "symbol"]
    )
    return pd.DataFrame(
        {
            "short_interest": [short_interest] * n,
            "float_shares": [float_shares] * n,
            "volume": [volume] * n,
            "on_loan": [on_loan] * n,
            "lendable_supply": [lendable_supply] * n,
            "fee_rate": [fee_rate] * n,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# LendingFeeSchedule
# ---------------------------------------------------------------------------


def test_schedule_from_scalars_builds_uniform_series() -> None:
    schedule = LendingFeeSchedule.from_scalars(
        ["AAPL", "MSFT"], fee_rate=0.01, on_loan_fraction=0.5
    )
    assert list(schedule.fee_rates.index) == ["AAPL", "MSFT"]
    assert (schedule.fee_rates == 0.01).all()
    assert (schedule.on_loan_fractions == 0.5).all()
    assert (schedule.reinvestment_spread == 0.0).all()


def test_schedule_rejects_invalid_on_loan_fraction() -> None:
    with pytest.raises(ValueError, match="on_loan_fractions"):
        LendingFeeSchedule.from_scalars(["AAPL"], fee_rate=0.01, on_loan_fraction=1.5)


def test_schedule_rejects_negative_fee_rate() -> None:
    with pytest.raises(ValueError, match="fee_rates"):
        LendingFeeSchedule.from_scalars(["AAPL"], fee_rate=-0.01, on_loan_fraction=0.5)


def test_schedule_rejects_collateral_haircut_below_one() -> None:
    with pytest.raises(ValueError, match="collateral_haircut"):
        LendingFeeSchedule.from_scalars(
            ["AAPL"], fee_rate=0.01, on_loan_fraction=0.5, collateral_haircut=0.99
        )


# ---------------------------------------------------------------------------
# SecuritiesLendingRevenueModel — fee income only
# ---------------------------------------------------------------------------


def test_revenue_model_fee_income_non_cash_collateral() -> None:
    symbols = ["AAPL", "MSFT"]
    weights = _weights(symbols, [0.5, 0.5])
    nav = 1_000_000.0
    schedule = LendingFeeSchedule.from_scalars(
        symbols,
        fee_rate=0.01,
        on_loan_fraction=0.5,
        reinvestment_spread=0.0,
    )
    result = SecuritiesLendingRevenueModel().estimate(
        weights, nav, schedule, holding_period_days=360.0
    )

    # Each position: 500k market value, 50% on loan → 250k on loan
    # Fee income = 250k × 0.01 × (360/360) = 2,500 per symbol
    assert np.isclose(result.total_fee_income, 5_000.0)
    assert np.isclose(result.total_reinvestment_income, 0.0)
    assert np.isclose(result.total_lending_revenue, 5_000.0)
    assert np.isclose(result.revenue_on_nav, 0.005)


def test_revenue_model_reinvestment_income_cash_collateral() -> None:
    symbols = ["AAPL"]
    weights = _weights(symbols, [1.0])
    nav = 1_000_000.0
    schedule = LendingFeeSchedule.from_scalars(
        symbols,
        fee_rate=0.01,
        on_loan_fraction=1.0,
        reinvestment_spread=0.002,
        collateral_haircut=1.02,
    )
    result = SecuritiesLendingRevenueModel().estimate(
        weights, nav, schedule, holding_period_days=360.0
    )

    # 1M on loan; collateral = 1.02M; reinvest income = 1.02M × 0.002 = 2,040
    expected_reinvest = 1_000_000 * 1.02 * 0.002
    assert np.isclose(result.total_reinvestment_income, expected_reinvest)
    assert np.isclose(result.total_lending_revenue, 10_000.0 + expected_reinvest)


def test_revenue_model_prorates_by_holding_period() -> None:
    symbols = ["AAPL"]
    weights = _weights(symbols, [1.0])
    nav = 1_000_000.0
    schedule = LendingFeeSchedule.from_scalars(
        symbols, fee_rate=0.01, on_loan_fraction=1.0
    )
    one_day = SecuritiesLendingRevenueModel().estimate(
        weights, nav, schedule, holding_period_days=1.0
    )
    full_year = SecuritiesLendingRevenueModel().estimate(
        weights, nav, schedule, holding_period_days=360.0
    )

    assert np.isclose(one_day.total_lending_revenue * 360, full_year.total_lending_revenue)
    assert np.isclose(one_day.revenue_on_nav, full_year.revenue_on_nav)


def test_revenue_model_ignores_short_weights() -> None:
    symbols = ["AAPL", "MSFT"]
    weights = _weights(symbols, [1.5, -0.5])
    nav = 1_000_000.0
    schedule = LendingFeeSchedule.from_scalars(
        symbols, fee_rate=0.01, on_loan_fraction=1.0
    )
    result = SecuritiesLendingRevenueModel().estimate(
        weights, nav, schedule, holding_period_days=360.0
    )
    assert result.fee_income["MSFT"] == 0.0
    assert result.fee_income["AAPL"] > 0.0


def test_revenue_model_summary_keys() -> None:
    symbols = ["AAPL"]
    weights = _weights(symbols, [1.0])
    schedule = LendingFeeSchedule.from_scalars(symbols, fee_rate=0.005, on_loan_fraction=0.3)
    result = SecuritiesLendingRevenueModel().estimate(weights, 500_000.0, schedule)
    summary = result.summary()
    assert set(summary) == {
        "total_fee_income",
        "total_reinvestment_income",
        "total_lending_revenue",
        "revenue_on_nav",
    }


def test_revenue_model_rejects_non_positive_nav() -> None:
    symbols = ["AAPL"]
    schedule = LendingFeeSchedule.from_scalars(symbols, fee_rate=0.01, on_loan_fraction=0.5)
    with pytest.raises(ValueError, match="nav"):
        SecuritiesLendingRevenueModel().estimate(_weights(symbols, [1.0]), 0.0, schedule)


def test_revenue_model_symbols_not_in_schedule_are_excluded() -> None:
    weights = _weights(["AAPL", "UNKNOWN"], [0.6, 0.4])
    schedule = LendingFeeSchedule.from_scalars(["AAPL"], fee_rate=0.01, on_loan_fraction=1.0)
    result = SecuritiesLendingRevenueModel().estimate(weights, 1_000_000.0, schedule)
    assert "UNKNOWN" not in result.position_revenue.index


# ---------------------------------------------------------------------------
# Securities-lending features
# ---------------------------------------------------------------------------


def test_short_interest_ratio_computes_correctly() -> None:
    frame = _sl_frame(3, short_interest=2_000_000, float_shares=10_000_000)
    result = compute_features(frame, [short_interest_ratio()])
    values = result[result["feature_name"] == "short_interest_ratio"]["value"].to_numpy()
    np.testing.assert_allclose(values, [0.2, 0.2, 0.2])


def test_days_to_cover_requires_warmup() -> None:
    frame = _sl_frame(5, short_interest=1_000_000, volume=500_000)
    result = compute_features(frame, [days_to_cover(volume_window=3)])
    values = result[result["feature_name"] == "days_to_cover"]["value"].to_numpy()
    assert np.isnan(values[0])
    assert np.isnan(values[1])
    np.testing.assert_allclose(values[2:], [2.0, 2.0, 2.0])


def test_utilization_rate_computes_correctly() -> None:
    frame = _sl_frame(2, on_loan=800_000, lendable_supply=1_000_000)
    result = compute_features(frame, [utilization_rate()])
    values = result[result["feature_name"] == "utilization_rate"]["value"].to_numpy()
    np.testing.assert_allclose(values, [0.8, 0.8])


def test_borrow_fee_rate_passes_through_column() -> None:
    frame = _sl_frame(3, fee_rate=0.025)
    result = compute_features(frame, [borrow_fee_rate()])
    values = result[result["feature_name"] == "borrow_fee_rate"]["value"].to_numpy()
    np.testing.assert_allclose(values, [0.025, 0.025, 0.025])


def test_short_interest_change_computes_percentage_change() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [timestamps, ["AAPL"] * 4], names=["timestamp", "symbol"]
    )
    frame = pd.DataFrame(
        {"short_interest": [1_000_000, 1_100_000, 1_210_000, 1_100_000]},
        index=index,
    )
    result = compute_features(frame, [short_interest_change(window=1)])
    values = result[result["feature_name"] == "short_interest_change"]["value"].to_numpy()
    assert np.isnan(values[0])
    np.testing.assert_allclose(values[1], 0.1, rtol=1e-6)
    np.testing.assert_allclose(values[2], 0.1, rtol=1e-6)
    np.testing.assert_allclose(values[3], -1 / 11, rtol=1e-6)


def test_securities_lending_registry_returns_five_definitions() -> None:
    defs = securities_lending_feature_registry()
    names = {d.name for d in defs}
    assert names == {
        "short_interest_ratio",
        "days_to_cover",
        "utilization_rate",
        "borrow_fee_rate",
        "short_interest_change",
    }


def test_securities_lending_features_register_without_collision() -> None:
    registry = FeatureRegistry(securities_lending_feature_registry())
    assert len(registry.definitions()) == 5


def test_days_to_cover_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="volume_window"):
        days_to_cover(volume_window=0)


def test_short_interest_change_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window"):
        short_interest_change(window=0)

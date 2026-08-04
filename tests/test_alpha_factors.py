"""Tests for factor-based feature definitions."""

import math

import numpy as np
import pandas as pd
import pytest

from qr_haven.features import (
    book_to_market,
    compute_features,
    earnings_yield,
    factor_feature_registry,
    low_volatility_score,
    price_momentum,
    short_term_reversal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_prices(values: list[float], symbol: str = "AAPL") -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=len(values), tz="UTC")
    idx = pd.MultiIndex.from_arrays(
        [timestamps, [symbol] * len(values)], names=["timestamp", "symbol"]
    )
    return pd.DataFrame({"adjusted_close": values}, index=idx)


def make_fundamental_prices(
    prices: list[float],
    eps_values: list[float],
    bvps_values: list[float],
    symbol: str = "AAPL",
) -> pd.DataFrame:
    n = len(prices)
    timestamps = pd.date_range("2020-01-01", periods=n, tz="UTC")
    idx = pd.MultiIndex.from_arrays(
        [timestamps, [symbol] * n], names=["timestamp", "symbol"]
    )
    return pd.DataFrame(
        {
            "adjusted_close": prices,
            "earnings_per_share": eps_values,
            "book_value_per_share": bvps_values,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# price_momentum
# ---------------------------------------------------------------------------


def test_price_momentum_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="formation_window"):
        price_momentum(formation_window=1)
    with pytest.raises(ValueError, match="skip_window"):
        price_momentum(skip_window=-1)
    with pytest.raises(ValueError, match="skip_window must be less"):
        price_momentum(formation_window=5, skip_window=5)


def test_price_momentum_known_value() -> None:
    # 6-period formation, 1-period skip
    # prices: 100 → 120 (then 118 at t[-1], 120 at t[-skip-1=t[-2]])
    # momentum = 120 / 100 - 1 = 0.20
    n = 7  # need formation_window + 1 = 7 rows
    prices = [100.0, 105.0, 110.0, 115.0, 118.0, 120.0, 122.0]
    feat = price_momentum(formation_window=6, skip_window=1)
    result = compute_features(make_prices(prices), [feat])
    last = result[result["feature_name"] == "price_momentum"].iloc[-1]["value"]
    # skip_price = prices[-2] = 120.0, form_price = prices[0] = 100.0
    assert math.isclose(last, 120.0 / 100.0 - 1.0, rel_tol=1e-9)


def test_price_momentum_nan_before_formation_window() -> None:
    # With formation_window=5, first 5 rows should be NaN
    feat = price_momentum(formation_window=5, skip_window=1)
    prices = list(range(1, 8))
    result = compute_features(make_prices(prices), [feat])
    mom = result[result["feature_name"] == "price_momentum"]["value"].tolist()
    assert all(math.isnan(v) for v in mom[:5])
    assert not math.isnan(mom[-1])


def test_price_momentum_version_field() -> None:
    feat = price_momentum(version="2")
    assert feat.version == "2"
    assert feat.name == "price_momentum"


# ---------------------------------------------------------------------------
# short_term_reversal
# ---------------------------------------------------------------------------


def test_short_term_reversal_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window"):
        short_term_reversal(window=0)


def test_short_term_reversal_known_value() -> None:
    # 3-day reversal: prices 100 → 80 → 90 → 95 → 110
    # at t[-1]: start=prices[0]=100 (window=4), end=110 → raw_ret = 0.10 → reversal = -0.10
    feat = short_term_reversal(window=4)
    prices = [100.0, 80.0, 90.0, 95.0, 110.0]
    result = compute_features(make_prices(prices), [feat])
    rev = result[result["feature_name"] == "short_term_reversal"]["value"].tolist()
    assert math.isclose(rev[-1], -0.10, rel_tol=1e-9)


def test_short_term_reversal_negative_sign_for_rising_prices() -> None:
    # Prices uniformly rising → reversal score should be negative
    feat = short_term_reversal(window=3)
    prices = [100.0, 105.0, 110.0, 115.0]
    result = compute_features(make_prices(prices), [feat])
    last = result[result["feature_name"] == "short_term_reversal"].iloc[-1]["value"]
    assert last < 0.0


def test_short_term_reversal_positive_for_falling_prices() -> None:
    feat = short_term_reversal(window=3)
    prices = [100.0, 95.0, 90.0, 85.0]
    result = compute_features(make_prices(prices), [feat])
    last = result[result["feature_name"] == "short_term_reversal"].iloc[-1]["value"]
    assert last > 0.0


def test_short_term_reversal_nan_before_window() -> None:
    feat = short_term_reversal(window=3)
    prices = [100.0, 95.0, 90.0, 85.0]
    result = compute_features(make_prices(prices), [feat])
    vals = result[result["feature_name"] == "short_term_reversal"]["value"].tolist()
    assert all(math.isnan(v) for v in vals[:3])
    assert not math.isnan(vals[-1])


# ---------------------------------------------------------------------------
# low_volatility_score
# ---------------------------------------------------------------------------


def test_low_volatility_score_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window"):
        low_volatility_score(window=1)


def test_low_volatility_score_is_negative_of_volatility() -> None:
    # Constant prices → vol = 0 → score = 0
    feat = low_volatility_score(window=3)
    prices = [100.0] * 10
    result = compute_features(make_prices(prices), [feat])
    last = result[result["feature_name"] == "low_volatility_score"].iloc[-1]["value"]
    assert math.isclose(last, 0.0, abs_tol=1e-12)


def test_low_volatility_score_lower_for_more_volatile() -> None:
    # Two symbols: low-vol and high-vol prices
    n = 20
    ts = pd.date_range("2020-01-01", periods=n, tz="UTC")
    idx_lv = pd.MultiIndex.from_arrays([ts, ["LV"] * n], names=["timestamp", "symbol"])
    idx_hv = pd.MultiIndex.from_arrays([ts, ["HV"] * n], names=["timestamp", "symbol"])
    np.random.seed(42)
    lv_prices = pd.DataFrame({"adjusted_close": 100 + np.random.normal(0, 0.1, n).cumsum()}, index=idx_lv)
    hv_prices = pd.DataFrame({"adjusted_close": 100 + np.random.normal(0, 2.0, n).cumsum()}, index=idx_hv)
    prices = pd.concat([lv_prices, hv_prices]).sort_index()
    feat = low_volatility_score(window=10)
    result = compute_features(prices, [feat])
    at_last = result[result["timestamp"] == ts[-1]][["symbol", "value"]].set_index("symbol")["value"]
    assert at_last["LV"] > at_last["HV"]  # low-vol gets higher score


def test_low_volatility_score_nan_before_window() -> None:
    feat = low_volatility_score(window=5)
    prices = [100.0 + i for i in range(10)]
    result = compute_features(make_prices(prices), [feat])
    vals = result[result["feature_name"] == "low_volatility_score"]["value"].tolist()
    assert math.isnan(vals[0])
    assert not math.isnan(vals[-1])


# ---------------------------------------------------------------------------
# earnings_yield
# ---------------------------------------------------------------------------


def test_earnings_yield_known_value() -> None:
    # eps=5, price=100 → E/P = 0.05
    prices = make_fundamental_prices([100.0], [5.0], [30.0])
    feat = earnings_yield()
    result = compute_features(prices, [feat])
    val = result[result["feature_name"] == "earnings_yield"].iloc[-1]["value"]
    assert math.isclose(val, 0.05, rel_tol=1e-9)


def test_earnings_yield_zero_price_gives_nan() -> None:
    prices = make_fundamental_prices([0.0], [5.0], [30.0])
    feat = earnings_yield()
    result = compute_features(prices, [feat])
    val = result[result["feature_name"] == "earnings_yield"].iloc[-1]["value"]
    assert math.isnan(val)


def test_earnings_yield_negative_eps_allowed() -> None:
    # Negative earnings are valid (losses)
    prices = make_fundamental_prices([100.0], [-3.0], [30.0])
    feat = earnings_yield()
    result = compute_features(prices, [feat])
    val = result[result["feature_name"] == "earnings_yield"].iloc[-1]["value"]
    assert math.isclose(val, -0.03, rel_tol=1e-9)


def test_earnings_yield_requires_correct_columns() -> None:
    prices = make_prices([100.0])
    with pytest.raises(ValueError, match="missing input columns"):
        compute_features(prices, [earnings_yield()])


# ---------------------------------------------------------------------------
# book_to_market
# ---------------------------------------------------------------------------


def test_book_to_market_known_value() -> None:
    # bvps=40, price=100 → B/M = 0.40
    prices = make_fundamental_prices([100.0], [5.0], [40.0])
    feat = book_to_market()
    result = compute_features(prices, [feat])
    val = result[result["feature_name"] == "book_to_market"].iloc[-1]["value"]
    assert math.isclose(val, 0.40, rel_tol=1e-9)


def test_book_to_market_zero_price_gives_nan() -> None:
    prices = make_fundamental_prices([0.0], [5.0], [40.0])
    feat = book_to_market()
    result = compute_features(prices, [feat])
    val = result[result["feature_name"] == "book_to_market"].iloc[-1]["value"]
    assert math.isnan(val)


# ---------------------------------------------------------------------------
# factor_feature_registry
# ---------------------------------------------------------------------------


def test_factor_feature_registry_returns_three_price_only_factors() -> None:
    registry = factor_feature_registry()
    names = {f.name for f in registry}
    assert names == {"price_momentum", "short_term_reversal", "low_volatility_score"}
    # All require only adjusted_close
    for feat in registry:
        assert feat.required_columns == ("adjusted_close",)


def test_factor_feature_registry_plugs_into_compute_features() -> None:
    prices = make_prices([float(100 + i) for i in range(300)])
    result = compute_features(prices, list(factor_feature_registry()))
    assert set(result["feature_name"]) == {
        "price_momentum",
        "short_term_reversal",
        "low_volatility_score",
    }

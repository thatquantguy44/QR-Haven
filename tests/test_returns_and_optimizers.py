import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from qr_haven.data import calculate_returns
from qr_haven.portfolio import (
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
    estimate_expected_returns,
    estimate_return_covariance,
)

PRICE_ROWS = pd.DataFrame(
    [
        {"timestamp": "2024-01-01", "symbol": "AAPL", "adjusted_close": 100.0},
        {"timestamp": "2024-01-01", "symbol": "MSFT", "adjusted_close": 200.0},
        {"timestamp": "2024-01-02", "symbol": "AAPL", "adjusted_close": 101.0},
        {"timestamp": "2024-01-02", "symbol": "MSFT", "adjusted_close": 204.0},
        {"timestamp": "2024-01-03", "symbol": "AAPL", "adjusted_close": 103.0},
        {"timestamp": "2024-01-03", "symbol": "MSFT", "adjusted_close": 202.0},
    ]
)


def _canonical_prices() -> pd.DataFrame:
    prices = PRICE_ROWS.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    return prices.set_index(["timestamp", "symbol"]).sort_index()


def test_calculate_returns_creates_wide_asset_matrix() -> None:
    returns = calculate_returns(_canonical_prices())

    assert list(returns.columns) == ["AAPL", "MSFT"]
    assert len(returns) == 2
    expected = pd.Series(
        {"AAPL": 0.01, "MSFT": 0.02},
        index=pd.Index(["AAPL", "MSFT"], name="symbol"),
        name=pd.Timestamp("2024-01-02", tz="UTC"),
    )
    assert_series_equal(returns.loc[pd.Timestamp("2024-01-02", tz="UTC")], expected)


def test_equal_weight_optimizer_allocates_evenly() -> None:
    returns = calculate_returns(_canonical_prices())
    expected_returns = estimate_expected_returns(returns)
    covariance = estimate_return_covariance(returns)

    weights = EqualWeightOptimizer().optimize(expected_returns, covariance)

    assert weights.to_dict() == {"AAPL": 0.5, "MSFT": 0.5}
    assert np.isclose(weights.sum(), 1.0)


def test_mean_variance_optimizer_prefers_better_risk_adjusted_asset() -> None:
    expected_returns = pd.Series({"AAPL": 0.10, "MSFT": 0.05})
    covariance = pd.DataFrame(
        [[0.04, 0.00], [0.00, 0.04]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    weights = MeanVarianceOptimizer().optimize(
        expected_returns,
        covariance,
        constraints={"max_weight": 0.8, "risk_aversion": 1.0},
    )

    assert np.isclose(weights.sum(), 1.0)
    assert weights["AAPL"] > weights["MSFT"]
    assert weights["AAPL"] <= 0.8 + 1e-9

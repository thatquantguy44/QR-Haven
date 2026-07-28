import numpy as np
import pandas as pd
import pytest

from qr_haven.features import (
    FEATURE_OUTPUT_COLUMNS,
    FeatureDefinition,
    FeatureRegistry,
    compute_features,
    lagged_return,
    rolling_momentum,
    rolling_volatility,
)


def price_frame(values: list[float], symbol: str = "AAPL") -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(values), tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [timestamps, [symbol] * len(values)], names=["timestamp", "symbol"]
    )
    return pd.DataFrame({"adjusted_close": values}, index=index)


def test_feature_definition_validates_contract() -> None:
    with pytest.raises(ValueError, match="name cannot be empty"):
        FeatureDefinition("", "1", ("close",), "float64", 1, lambda frame: frame["close"])
    with pytest.raises(ValueError, match="lookback"):
        FeatureDefinition("return", "1", ("close",), "float64", -1, lambda frame: frame["close"])
    with pytest.raises(ValueError, match="Invalid output dtype"):
        FeatureDefinition("return", "1", ("close",), "not-a-dtype", 1, lambda frame: frame["close"])


def test_registry_rejects_duplicate_name_and_version() -> None:
    definition = lagged_return()
    registry = FeatureRegistry([definition])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(lagged_return())

    registry.register(lagged_return(version="2"))
    assert registry.get("lagged_return", "2").version == "2"


def test_reference_features_are_deterministic_and_use_canonical_schema() -> None:
    prices = price_frame([100.0, 110.0, 121.0, 108.9])
    definitions = [lagged_return(), rolling_volatility(window=2), rolling_momentum(window=2)]

    first = compute_features(prices, definitions)
    second = compute_features(prices, definitions)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == FEATURE_OUTPUT_COLUMNS
    returns = first[first["feature_name"] == "lagged_return"]["value"].to_numpy()
    np.testing.assert_allclose(returns, [np.nan, 0.1, 0.1, -0.1], equal_nan=True)
    momentum = first[first["feature_name"] == "rolling_momentum"]["value"].to_numpy()
    np.testing.assert_allclose(momentum, [np.nan, np.nan, 0.21, -0.01], equal_nan=True)
    volatility = first[first["feature_name"] == "rolling_volatility"]["value"].to_numpy()
    np.testing.assert_allclose(
        volatility,
        [np.nan, np.nan, 0.0, np.std([0.1, -0.1], ddof=1)],
        atol=1e-12,
        equal_nan=True,
    )


def test_engine_prevents_transform_from_reading_future_values() -> None:
    observed_windows: list[pd.DatetimeIndex] = []

    def curious_transform(frame: pd.DataFrame) -> pd.Series:
        observed_windows.append(frame.index)
        return pd.Series(frame["adjusted_close"].iloc[-1], index=[frame.index[-1]])

    definition = FeatureDefinition(
        "current_price", "1", ("adjusted_close",), "float64", 1, curious_transform
    )
    prices = price_frame([100.0, 200.0, 9999.0])

    result = compute_features(prices, [definition])

    assert [len(window) for window in observed_windows] == [1, 2, 2]
    assert observed_windows[0].max() < observed_windows[1].max() < observed_windows[2].max()
    assert result["value"].tolist() == [100.0, 200.0, 9999.0]


def test_feature_values_before_cutoff_do_not_change_when_future_is_appended() -> None:
    original = price_frame([100.0, 110.0, 121.0])
    extended = price_frame([100.0, 110.0, 121.0, 10_000.0])
    definitions = [lagged_return(), rolling_volatility(window=2), rolling_momentum(window=2)]

    original_features = compute_features(original, definitions)
    extended_features = compute_features(extended, definitions)
    before_cutoff = extended_features[
        extended_features["timestamp"] <= original.index.get_level_values("timestamp").max()
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(original_features, before_cutoff)


def test_batch_computation_validates_canonical_input_and_columns() -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        compute_features(pd.DataFrame({"adjusted_close": [1.0]}), [lagged_return()])
    with pytest.raises(ValueError, match="missing input columns"):
        invalid = price_frame([1.0]).rename(columns={"adjusted_close": "close"})
        compute_features(invalid, [lagged_return()])

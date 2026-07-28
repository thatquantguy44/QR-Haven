"""Reference price features for Feature Store v0."""

from __future__ import annotations

import pandas as pd

from qr_haven.features.store import FeatureDefinition


def lagged_return(lag: int = 1, *, version: str = "1") -> FeatureDefinition:
    """Create a simple return over ``lag`` prior observations."""

    if lag < 1:
        raise ValueError("lag must be at least one.")

    def transform(frame: pd.DataFrame) -> pd.Series:
        return frame["adjusted_close"].pct_change(periods=lag, fill_method=None)

    return FeatureDefinition(
        name="lagged_return",
        version=version,
        required_columns=("adjusted_close",),
        output_dtype="float64",
        lookback=lag,
        transform=transform,
    )


def rolling_volatility(window: int = 20, *, version: str = "1") -> FeatureDefinition:
    """Create sample volatility of simple returns over a trailing window."""

    if window < 2:
        raise ValueError("window must be at least two.")

    def transform(frame: pd.DataFrame) -> pd.Series:
        returns = frame["adjusted_close"].pct_change(fill_method=None)
        return returns.rolling(window=window, min_periods=window).std()

    return FeatureDefinition(
        name="rolling_volatility",
        version=version,
        required_columns=("adjusted_close",),
        output_dtype="float64",
        lookback=window,
        transform=transform,
    )


def rolling_momentum(window: int = 20, *, version: str = "1") -> FeatureDefinition:
    """Create trailing price momentum relative to ``window`` observations ago."""

    if window < 1:
        raise ValueError("window must be at least one.")

    def transform(frame: pd.DataFrame) -> pd.Series:
        return frame["adjusted_close"].pct_change(periods=window, fill_method=None)

    return FeatureDefinition(
        name="rolling_momentum",
        version=version,
        required_columns=("adjusted_close",),
        output_dtype="float64",
        lookback=window,
        transform=transform,
    )


def reference_feature_registry() -> tuple[FeatureDefinition, ...]:
    """Return the default v0 reference feature definitions."""

    return (lagged_return(), rolling_volatility(), rolling_momentum())


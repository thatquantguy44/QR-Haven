"""Factor-based feature definitions for cross-sectional equity alpha research.

Each factory function returns a ``FeatureDefinition`` that plugs into the
point-in-time-safe ``compute_features`` engine.  Factors follow standard
academic / practitioner conventions:

- **Price momentum** — Jegadeesh & Titman (1993) 12-1 month return.
- **Short-term reversal** — Lo & MacKinlay (1990) 1-week mean reversion.
- **Low-volatility score** — negative of rolling realised vol; high score ↔ low vol.
- **Earnings yield** — E/P (fundamental value proxy); needs ``earnings_per_share``.
- **Book-to-market** — B/P (Fama-French HML driver); needs ``book_value_per_share``.
"""

from __future__ import annotations

import math

import pandas as pd

from qr_haven.features.store import FeatureDefinition


def price_momentum(
    formation_window: int = 252,
    skip_window: int = 21,
    *,
    version: str = "1",
) -> FeatureDefinition:
    """12-1 month price momentum (skip most-recent month to avoid reversal).

    Parameters
    ----------
    formation_window:
        Number of trading days defining the full lookback. Default 252 (≈ 1 year).
    skip_window:
        Days to exclude from the trailing end of the formation window.
        Default 21 (≈ 1 month). Must be < formation_window.
    """

    if formation_window < 2:
        raise ValueError("formation_window must be at least 2.")
    if skip_window < 0:
        raise ValueError("skip_window cannot be negative.")
    if skip_window >= formation_window:
        raise ValueError("skip_window must be less than formation_window.")

    def transform(frame: pd.DataFrame) -> pd.Series:
        prices = frame["adjusted_close"]
        result = pd.Series(float("nan"), index=prices.index)
        n = len(prices)
        if n >= formation_window + 1:
            skip_price = float(prices.iloc[-(skip_window + 1)])
            form_price = float(prices.iloc[0])
            if form_price != 0.0 and not (math.isnan(skip_price) or math.isnan(form_price)):
                result.iloc[-1] = skip_price / form_price - 1.0
        return result

    return FeatureDefinition(
        name="price_momentum",
        version=version,
        required_columns=("adjusted_close",),
        output_dtype="float64",
        lookback=formation_window,
        transform=transform,
    )


def short_term_reversal(
    window: int = 5,
    *,
    version: str = "1",
) -> FeatureDefinition:
    """Short-term price reversal (negative of recent return).

    A negative sign converts the raw return into a reversal score so that
    positive values predict positive forward returns.

    Parameters
    ----------
    window:
        Number of trading days for the reversal window. Default 5 (≈ 1 week).
    """

    if window < 1:
        raise ValueError("window must be at least 1.")

    def transform(frame: pd.DataFrame) -> pd.Series:
        prices = frame["adjusted_close"]
        result = pd.Series(float("nan"), index=prices.index)
        n = len(prices)
        if n >= window + 1:
            start_price = float(prices.iloc[0])
            end_price = float(prices.iloc[-1])
            if start_price != 0.0 and not (math.isnan(start_price) or math.isnan(end_price)):
                result.iloc[-1] = -(end_price / start_price - 1.0)
        return result

    return FeatureDefinition(
        name="short_term_reversal",
        version=version,
        required_columns=("adjusted_close",),
        output_dtype="float64",
        lookback=window,
        transform=transform,
    )


def low_volatility_score(
    window: int = 60,
    *,
    version: str = "1",
) -> FeatureDefinition:
    """Negative of trailing realised volatility (low-vol anomaly).

    High score ↔ low volatility, consistent with sign convention where larger
    values predict more positive returns.

    Parameters
    ----------
    window:
        Number of daily returns used to estimate volatility. Default 60.
    """

    if window < 2:
        raise ValueError("window must be at least 2.")

    def transform(frame: pd.DataFrame) -> pd.Series:
        prices = frame["adjusted_close"]
        result = pd.Series(float("nan"), index=prices.index)
        returns = prices.pct_change(fill_method=None)
        if len(returns) >= window:
            vol = float(returns.iloc[-window:].std())
            if not math.isnan(vol) and vol >= 0.0:
                result.iloc[-1] = -vol
        return result

    return FeatureDefinition(
        name="low_volatility_score",
        version=version,
        required_columns=("adjusted_close",),
        output_dtype="float64",
        lookback=window + 1,
        transform=transform,
    )


def earnings_yield(*, version: str = "1") -> FeatureDefinition:
    """Earnings yield: earnings_per_share / adjusted_close.

    A higher value signals cheapness relative to earnings (value factor, E/P).
    Requires the input data to carry an ``earnings_per_share`` column.
    """

    def transform(frame: pd.DataFrame) -> pd.Series:
        result = pd.Series(float("nan"), index=frame.index)
        price = float(frame["adjusted_close"].iloc[-1])
        eps = float(frame["earnings_per_share"].iloc[-1])
        if price > 0.0 and not (math.isnan(price) or math.isnan(eps)):
            result.iloc[-1] = eps / price
        return result

    return FeatureDefinition(
        name="earnings_yield",
        version=version,
        required_columns=("adjusted_close", "earnings_per_share"),
        output_dtype="float64",
        lookback=0,
        transform=transform,
    )


def book_to_market(*, version: str = "1") -> FeatureDefinition:
    """Book-to-market ratio: book_value_per_share / adjusted_close.

    Classic HML value driver (Fama & French 1993).  Requires the input data
    to carry a ``book_value_per_share`` column.
    """

    def transform(frame: pd.DataFrame) -> pd.Series:
        result = pd.Series(float("nan"), index=frame.index)
        price = float(frame["adjusted_close"].iloc[-1])
        bvps = float(frame["book_value_per_share"].iloc[-1])
        if price > 0.0 and not (math.isnan(price) or math.isnan(bvps)):
            result.iloc[-1] = bvps / price
        return result

    return FeatureDefinition(
        name="book_to_market",
        version=version,
        required_columns=("adjusted_close", "book_value_per_share"),
        output_dtype="float64",
        lookback=0,
        transform=transform,
    )


def factor_feature_registry() -> tuple[FeatureDefinition, ...]:
    """Return the default price-only factor definitions.

    Includes price momentum (252/21), short-term reversal (5 days), and
    low-volatility score (60 days). Fundamental factors (earnings yield,
    book-to-market) are excluded because they require non-price columns.
    """

    return (
        price_momentum(),
        short_term_reversal(),
        low_volatility_score(),
    )

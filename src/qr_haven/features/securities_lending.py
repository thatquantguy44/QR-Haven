"""Securities-lending alpha features for the Feature Store.

Required canonical price columns beyond adjusted_close:

- short_interest      : number of shares currently sold short
- float_shares        : shares available for public trading
- volume              : shares traded (daily)
- on_loan             : shares currently on loan
- lendable_supply     : shares available to lend from custodied inventory
- fee_rate            : annualized borrow fee as a decimal fraction
"""

from __future__ import annotations

import pandas as pd

from qr_haven.features.store import FeatureDefinition


def short_interest_ratio(*, version: str = "1") -> FeatureDefinition:
    """Short interest as a fraction of float shares outstanding.

    A high SIR indicates crowded short positioning relative to the tradeable
    supply. Values above 0.20 are typically considered elevated; values above
    0.50 are extreme and often signal either conviction or a short-squeeze setup.
    """

    def transform(frame: pd.DataFrame) -> pd.Series:
        return (frame["short_interest"] / frame["float_shares"]).astype(float)

    return FeatureDefinition(
        name="short_interest_ratio",
        version=version,
        required_columns=("short_interest", "float_shares"),
        output_dtype="float64",
        lookback=0,
        transform=transform,
    )


def days_to_cover(volume_window: int = 20, *, version: str = "1") -> FeatureDefinition:
    """Short interest relative to rolling average daily volume.

    Measures how many days of trading it would take all short sellers to cover
    their positions at current volume. A high value signals that a short squeeze
    could persist; a declining value suggests orderly short reduction.

    Parameters
    ----------
    volume_window:
        Number of prior observations used to compute the average daily volume.
    """

    if volume_window < 1:
        raise ValueError("volume_window must be at least 1.")

    def transform(frame: pd.DataFrame) -> pd.Series:
        avg_volume = (
            frame["volume"]
            .rolling(window=volume_window, min_periods=volume_window)
            .mean()
        )
        return (frame["short_interest"] / avg_volume).astype(float)

    return FeatureDefinition(
        name="days_to_cover",
        version=version,
        required_columns=("short_interest", "volume"),
        output_dtype="float64",
        lookback=volume_window,
        transform=transform,
    )


def utilization_rate(*, version: str = "1") -> FeatureDefinition:
    """Fraction of the lendable supply currently on loan.

    Utilization is the most direct measure of borrow demand relative to available
    inventory. High utilization (>80%) often precedes fee rate increases and can
    signal upcoming restrictions on new short positions.
    """

    def transform(frame: pd.DataFrame) -> pd.Series:
        return (frame["on_loan"] / frame["lendable_supply"]).astype(float)

    return FeatureDefinition(
        name="utilization_rate",
        version=version,
        required_columns=("on_loan", "lendable_supply"),
        output_dtype="float64",
        lookback=0,
        transform=transform,
    )


def borrow_fee_rate(*, version: str = "1") -> FeatureDefinition:
    """Annualized borrow fee rate as a direct signal input.

    The cost to borrow encodes aggregate market conviction about a security's
    future price direction. General collateral (GC) names trade at 10-25 bps;
    specials trade at 100 bps to 30%+. Hard-to-borrow names carry a positive
    signal decay from the fee cost itself in addition to any directional content.
    """

    def transform(frame: pd.DataFrame) -> pd.Series:
        return frame["fee_rate"].astype(float)

    return FeatureDefinition(
        name="borrow_fee_rate",
        version=version,
        required_columns=("fee_rate",),
        output_dtype="float64",
        lookback=0,
        transform=transform,
    )


def short_interest_change(window: int = 5, *, version: str = "1") -> FeatureDefinition:
    """Percentage change in short interest over a trailing window.

    Rising short interest indicates growing bearish conviction and additional
    borrow demand. Declining short interest may signal short covering and
    near-term price support as forced buyers enter the market.

    Parameters
    ----------
    window:
        Number of prior observations over which to compute the percentage change.
    """

    if window < 1:
        raise ValueError("window must be at least 1.")

    def transform(frame: pd.DataFrame) -> pd.Series:
        return frame["short_interest"].pct_change(periods=window, fill_method=None).astype(float)

    return FeatureDefinition(
        name="short_interest_change",
        version=version,
        required_columns=("short_interest",),
        output_dtype="float64",
        lookback=window,
        transform=transform,
    )


def securities_lending_feature_registry() -> tuple[FeatureDefinition, ...]:
    """Return the default v0 securities-lending feature definitions."""

    return (
        short_interest_ratio(),
        days_to_cover(),
        utilization_rate(),
        borrow_fee_rate(),
        short_interest_change(),
    )

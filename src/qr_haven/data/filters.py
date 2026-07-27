"""Shared filtering helpers for ingested price data."""

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from qr_haven.data.errors import DataValidationError
from qr_haven.data.schemas import PriceDataSchema, normalize_price_frame


def _to_utc_timestamp(value: datetime) -> pd.Timestamp:
    """Convert naive or timezone-aware datetimes into UTC pandas timestamps."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def normalize_filter_prices(
    frame: pd.DataFrame,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    frequency: str,
    schema: PriceDataSchema,
) -> pd.DataFrame:
    """Normalize raw price rows and apply common symbol/date/frequency filters."""

    if not symbols:
        raise DataValidationError("At least one symbol is required.")

    normalized = normalize_price_frame(frame, schema).reset_index()
    start_timestamp = _to_utc_timestamp(start)
    end_timestamp = _to_utc_timestamp(end)

    symbol_set = {str(symbol) for symbol in symbols}
    mask = (
        normalized["symbol"].isin(symbol_set)
        & (normalized["timestamp"] >= start_timestamp)
        & (normalized["timestamp"] <= end_timestamp)
    )
    if "frequency" in normalized.columns:
        mask &= normalized["frequency"].astype(str) == frequency

    filtered = normalized.loc[mask].copy()
    if filtered.empty:
        raise DataValidationError(
            "No price rows matched the requested symbols, dates, and frequency."
        )

    return filtered.set_index(["timestamp", "symbol"]).sort_index()

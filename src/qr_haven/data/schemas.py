"""Schemas and normalization helpers for market data ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from qr_haven.data.errors import DataValidationError


@dataclass(frozen=True)
class PriceDataSchema:
    """Column mapping for long-form price data."""

    timestamp: str = "timestamp"
    symbol: str = "symbol"
    open: str | None = "open"
    high: str | None = "high"
    low: str | None = "low"
    close: str | None = "close"
    adjusted_close: str | None = "adjusted_close"
    volume: str | None = "volume"
    frequency: str | None = "frequency"

    def required_columns(self) -> set[str]:
        """Return source columns that must always exist."""

        if self.adjusted_close is None and self.close is None:
            raise DataValidationError("Price schema must define adjusted_close or close.")
        return {self.timestamp, self.symbol}

    def price_columns(self) -> list[str]:
        """Return source price columns in preferred order."""

        return [column for column in [self.adjusted_close, self.close] if column is not None]

    def canonical_columns(self) -> Mapping[str, str]:
        """Map source columns to canonical QR-Haven column names."""

        candidates = {
            self.timestamp: "timestamp",
            self.symbol: "symbol",
            self.open: "open",
            self.high: "high",
            self.low: "low",
            self.close: "close",
            self.adjusted_close: "adjusted_close",
            self.volume: "volume",
            self.frequency: "frequency",
        }
        return {source: target for source, target in candidates.items() if source is not None}


def normalize_price_frame(frame: pd.DataFrame, schema: PriceDataSchema) -> pd.DataFrame:
    """Normalize source price rows into QR-Haven's canonical indexed format."""

    missing = schema.required_columns() - set(frame.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise DataValidationError(f"Price data is missing required columns: {missing_columns}")

    if not any(column in frame.columns for column in schema.price_columns()):
        price_columns = ", ".join(schema.price_columns())
        raise DataValidationError(
            f"Price data must include at least one price column: {price_columns}"
        )

    normalized = frame.rename(columns=schema.canonical_columns()).copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized["symbol"] = normalized["symbol"].astype(str)

    if "adjusted_close" not in normalized.columns and "close" in normalized.columns:
        normalized["adjusted_close"] = normalized["close"]

    ordered_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "frequency",
    ]
    available_columns = [column for column in ordered_columns if column in normalized.columns]
    return normalized.set_index(["timestamp", "symbol"]).sort_index()[available_columns]

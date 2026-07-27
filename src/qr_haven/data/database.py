"""Database-backed data portals."""

import re
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from qr_haven.data.filters import normalize_filter_prices
from qr_haven.data.schemas import PriceDataSchema

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(identifier: str, label: str) -> None:
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Unsafe {label}: {identifier!r}")


class DatabasePriceDataPortal:
    """Load long-form price data from a SQLite database table."""

    def __init__(
        self,
        database_path: str | Path,
        table: str = "prices",
        schema: PriceDataSchema | None = None,
    ) -> None:
        schema = schema or PriceDataSchema()
        _validate_identifier(table, "table name")
        _validate_identifier(schema.timestamp, "timestamp column")
        _validate_identifier(schema.symbol, "symbol column")
        if schema.frequency is not None:
            _validate_identifier(schema.frequency, "frequency column")
        self.database_path = Path(database_path)
        self.table = table
        self.schema = schema

    def load_prices(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        frequency: str,
    ) -> pd.DataFrame:
        """Return canonical price data indexed by timestamp and symbol."""

        if not symbols:
            return normalize_filter_prices(
                pd.DataFrame(),
                symbols,
                start,
                end,
                frequency,
                self.schema,
            )

        placeholders = ", ".join("?" for _ in symbols)
        frequency_filter = ""
        params: list[object] = [*map(str, symbols), start.isoformat(), end.isoformat()]
        if self.schema.frequency is not None:
            frequency_filter = f" AND {self.schema.frequency} = ?"
            params.append(frequency)

        query = (
            f"SELECT * FROM {self.table} "
            f"WHERE {self.schema.symbol} IN ({placeholders}) "
            f"AND {self.schema.timestamp} >= ? "
            f"AND {self.schema.timestamp} <= ?"
            f"{frequency_filter}"
        )

        with sqlite3.connect(self.database_path) as connection:
            frame = pd.read_sql_query(query, connection, params=params)

        return normalize_filter_prices(frame, symbols, start, end, frequency, self.schema)

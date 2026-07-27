"""CSV-backed data portals."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from qr_haven.data.filters import normalize_filter_prices
from qr_haven.data.schemas import PriceDataSchema


class CSVPriceDataPortal:
    """Load long-form price data from a local CSV file."""

    def __init__(self, path: str | Path, schema: PriceDataSchema | None = None) -> None:
        self.path = Path(path)
        self.schema = schema or PriceDataSchema()

    def load_prices(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        frequency: str,
    ) -> pd.DataFrame:
        """Return canonical price data indexed by timestamp and symbol."""

        frame = pd.read_csv(self.path)
        return normalize_filter_prices(frame, symbols, start, end, frequency, self.schema)


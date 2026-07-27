"""Factory helpers for constructing data portals from configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qr_haven.data.csv import CSVPriceDataPortal
from qr_haven.data.database import DatabasePriceDataPortal
from qr_haven.data.schemas import PriceDataSchema
from qr_haven.interfaces import DataPortal


@dataclass(frozen=True)
class DataPortalConfig:
    """Configuration for constructing a v0 price data portal."""

    source_type: Literal["csv", "database"]
    path: Path
    table: str = "prices"
    schema: PriceDataSchema = PriceDataSchema()


def create_price_data_portal(config: DataPortalConfig) -> DataPortal:
    """Create a price data portal from a typed config object."""

    if config.source_type == "csv":
        return CSVPriceDataPortal(config.path, schema=config.schema)
    return DatabasePriceDataPortal(config.path, table=config.table, schema=config.schema)

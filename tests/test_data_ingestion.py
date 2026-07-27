import sqlite3
from datetime import UTC, datetime

import pandas as pd

from qr_haven.data import CSVPriceDataPortal, DataPortalConfig, create_price_data_portal
from qr_haven.data.database import DatabasePriceDataPortal
from qr_haven.data.errors import DataValidationError

SAMPLE_PRICES = pd.DataFrame(
    [
        {
            "timestamp": "2024-01-02T00:00:00+00:00",
            "symbol": "AAPL",
            "close": 185.64,
            "adjusted_close": 184.12,
            "volume": 1000,
            "frequency": "daily",
        },
        {
            "timestamp": "2024-01-02T00:00:00+00:00",
            "symbol": "MSFT",
            "close": 370.60,
            "adjusted_close": 369.01,
            "volume": 900,
            "frequency": "daily",
        },
        {
            "timestamp": "2024-01-03T00:00:00+00:00",
            "symbol": "AAPL",
            "close": 184.25,
            "adjusted_close": 182.78,
            "volume": 1100,
            "frequency": "daily",
        },
    ]
)


def test_csv_price_portal_loads_canonical_prices(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    SAMPLE_PRICES.to_csv(csv_path, index=False)

    portal = CSVPriceDataPortal(csv_path)
    prices = portal.load_prices(
        symbols=["AAPL"],
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 31, tzinfo=UTC),
        frequency="daily",
    )

    assert prices.index.names == ["timestamp", "symbol"]
    assert list(prices.index.get_level_values("symbol").unique()) == ["AAPL"]
    assert list(prices.columns) == ["close", "adjusted_close", "volume", "frequency"]
    assert prices.loc[(pd.Timestamp("2024-01-02", tz="UTC"), "AAPL"), "adjusted_close"] == 184.12


def test_database_price_portal_loads_canonical_prices(tmp_path) -> None:
    database_path = tmp_path / "market_data.sqlite"
    with sqlite3.connect(database_path) as connection:
        SAMPLE_PRICES.to_sql("prices", connection, index=False)

    portal = DatabasePriceDataPortal(database_path)
    prices = portal.load_prices(
        symbols=["AAPL", "MSFT"],
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 1, 2, tzinfo=UTC),
        frequency="daily",
    )

    assert len(prices) == 2
    assert set(prices.index.get_level_values("symbol")) == {"AAPL", "MSFT"}


def test_factory_creates_csv_data_portal(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    SAMPLE_PRICES.to_csv(csv_path, index=False)
    config = DataPortalConfig(source_type="csv", path=csv_path)

    portal = create_price_data_portal(config)

    assert isinstance(portal, CSVPriceDataPortal)


def test_price_portal_rejects_missing_required_columns(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame([{"timestamp": "2024-01-02", "close": 185.64}]).to_csv(csv_path, index=False)

    portal = CSVPriceDataPortal(csv_path)

    try:
        portal.load_prices(
            symbols=["AAPL"],
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 31, tzinfo=UTC),
            frequency="daily",
        )
    except DataValidationError as exc:
        assert "symbol" in str(exc)
    else:
        raise AssertionError("Expected missing symbol column to raise DataValidationError.")


def test_price_portal_uses_close_when_adjusted_close_is_absent(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    SAMPLE_PRICES.drop(columns=["adjusted_close"]).to_csv(csv_path, index=False)

    portal = CSVPriceDataPortal(csv_path)
    prices = portal.load_prices(
        symbols=["AAPL"],
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 31, tzinfo=UTC),
        frequency="daily",
    )

    assert prices.loc[(pd.Timestamp("2024-01-02", tz="UTC"), "AAPL"), "adjusted_close"] == 185.64

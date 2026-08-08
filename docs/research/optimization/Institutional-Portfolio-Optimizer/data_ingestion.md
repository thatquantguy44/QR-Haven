# Data Ingestion

QR-Haven v0 supports price ingestion from either CSV files or SQLite database tables through the
same `DataPortal` interface.

## Canonical Price Format

Source data should be long-form:

```text
timestamp,symbol,close,adjusted_close,volume,frequency
2024-01-02T00:00:00+00:00,AAPL,185.64,184.12,1000,daily
2024-01-02T00:00:00+00:00,MSFT,370.60,369.01,900,daily
```

The normalized output is indexed by `timestamp` and `symbol`.

## CSV

```python
from datetime import UTC, datetime

from qr_haven.data import CSVPriceDataPortal

portal = CSVPriceDataPortal("data/raw/prices.csv")
prices = portal.load_prices(
    symbols=["AAPL", "MSFT"],
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 1, 31, tzinfo=UTC),
    frequency="daily",
)
```

## SQLite Database

```python
from datetime import UTC, datetime

from qr_haven.data import DatabasePriceDataPortal

portal = DatabasePriceDataPortal("data/raw/market_data.sqlite", table="prices")
prices = portal.load_prices(
    symbols=["AAPL", "MSFT"],
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 1, 31, tzinfo=UTC),
    frequency="daily",
)
```

## Factory

```python
from pathlib import Path

from qr_haven.data import DataPortalConfig, create_price_data_portal

config = DataPortalConfig(source_type="csv", path=Path("data/raw/prices.csv"))
portal = create_price_data_portal(config)
```


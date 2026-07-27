"""Price-to-return transformations."""

from typing import Literal

import numpy as np
import pandas as pd

from qr_haven.data.errors import DataValidationError

ReturnMethod = Literal["simple", "log"]


def prices_to_matrix(
    prices: pd.DataFrame,
    price_column: str = "adjusted_close",
) -> pd.DataFrame:
    """Convert canonical long-form prices into a timestamp-by-symbol price matrix."""

    if not isinstance(prices.index, pd.MultiIndex):
        raise DataValidationError("Prices must be indexed by timestamp and symbol.")
    if prices.index.names != ["timestamp", "symbol"]:
        raise DataValidationError("Price index names must be ['timestamp', 'symbol'].")
    if price_column not in prices.columns:
        raise DataValidationError(f"Price column is missing: {price_column}")

    matrix = prices[price_column].unstack("symbol").sort_index()
    return matrix.astype(float)


def calculate_returns(
    prices: pd.DataFrame,
    price_column: str = "adjusted_close",
    method: ReturnMethod = "simple",
    dropna: bool = True,
) -> pd.DataFrame:
    """Calculate asset returns from canonical prices."""

    price_matrix = prices_to_matrix(prices, price_column=price_column)
    if method == "simple":
        returns = price_matrix.pct_change(fill_method=None)
    elif method == "log":
        returns = np.log(price_matrix / price_matrix.shift(1))
    else:
        raise ValueError(f"Unsupported return method: {method}")

    if dropna:
        return returns.dropna(how="all")
    return returns


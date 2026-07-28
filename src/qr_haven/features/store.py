"""Typed, point-in-time-safe batch feature computation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import pandas_dtype

FeatureTransform = Callable[[pd.DataFrame], pd.Series]

FEATURE_OUTPUT_COLUMNS = [
    "timestamp",
    "symbol",
    "feature_name",
    "feature_version",
    "value",
]


@dataclass(frozen=True)
class FeatureDefinition:
    """Versioned contract for a transformation over one symbol's price history.

    ``lookback`` is the number of observations before the calculation timestamp that
    the transformation may inspect. The current row is supplied in addition to it.
    """

    name: str
    version: str
    required_columns: tuple[str, ...]
    output_dtype: str
    lookback: int
    transform: FeatureTransform

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Feature name cannot be empty.")
        if not self.version.strip():
            raise ValueError("Feature version cannot be empty.")
        if not self.required_columns:
            raise ValueError("A feature must require at least one input column.")
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("Feature required columns must be unique.")
        if any(not column.strip() for column in self.required_columns):
            raise ValueError("Feature input column names cannot be empty.")
        if self.lookback < 0:
            raise ValueError("Feature lookback cannot be negative.")
        if not callable(self.transform):
            raise TypeError("Feature transform must be callable.")
        try:
            pandas_dtype(self.output_dtype)
        except TypeError as exc:
            raise ValueError(f"Invalid output dtype: {self.output_dtype}") from exc

    @property
    def key(self) -> tuple[str, str]:
        """Return the registry identity for this definition."""

        return self.name, self.version


class FeatureRegistry:
    """In-memory registry keyed by immutable feature name and version."""

    def __init__(self, definitions: Iterable[FeatureDefinition] = ()) -> None:
        self._definitions: dict[tuple[str, str], FeatureDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: FeatureDefinition) -> None:
        """Register a definition, rejecting an already-used name/version pair."""

        if definition.key in self._definitions:
            name, version = definition.key
            raise ValueError(f"Feature '{name}' version '{version}' is already registered.")
        self._definitions[definition.key] = definition

    def get(self, name: str, version: str) -> FeatureDefinition:
        """Retrieve a registered definition."""

        try:
            return self._definitions[(name, version)]
        except KeyError as exc:
            raise KeyError(f"Unknown feature '{name}' version '{version}'.") from exc

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        """Return definitions in deterministic registration order."""

        return tuple(self._definitions.values())


def compute_features(
    prices: pd.DataFrame,
    definitions: Iterable[FeatureDefinition] | FeatureRegistry,
) -> pd.DataFrame:
    """Compute features from canonical prices without exposing future rows.

    Each transformation is invoked once per symbol/timestamp with only the current
    row and the declared amount of preceding history. This makes temporal safety a
    property of the execution engine rather than a convention for transformations.
    """

    _validate_canonical_prices(prices)
    selected = (
        definitions.definitions()
        if isinstance(definitions, FeatureRegistry)
        else tuple(definitions)
    )
    if not selected:
        return pd.DataFrame(columns=FEATURE_OUTPUT_COLUMNS)

    available = set(prices.columns)
    for definition in selected:
        missing = set(definition.required_columns) - available
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"Feature '{definition.name}' is missing input columns: {columns}")

    rows: list[dict[str, object]] = []
    ordered = prices.sort_index()
    for symbol, symbol_frame in ordered.groupby(level="symbol", sort=True):
        history = symbol_frame.droplevel("symbol")
        for definition in selected:
            inputs = history.loc[:, list(definition.required_columns)]
            for position, timestamp in enumerate(inputs.index):
                start = max(0, position - definition.lookback)
                safe_window = inputs.iloc[start : position + 1].copy()
                transformed = definition.transform(safe_window)
                if not isinstance(transformed, pd.Series):
                    raise TypeError(
                        f"Feature '{definition.name}' transform must return a pandas Series."
                    )
                if timestamp not in transformed.index:
                    raise ValueError(
                        f"Feature '{definition.name}' transform did not return the "
                        "current timestamp."
                    )
                value = pd.Series(
                    [transformed.loc[timestamp]], dtype=definition.output_dtype
                ).iloc[0]
                rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": str(symbol),
                        "feature_name": definition.name,
                        "feature_version": definition.version,
                        "value": value,
                    }
                )

    result = pd.DataFrame(rows, columns=FEATURE_OUTPUT_COLUMNS)
    return result.sort_values(
        ["timestamp", "symbol", "feature_name", "feature_version"], ignore_index=True
    )


def _validate_canonical_prices(prices: pd.DataFrame) -> None:
    if not isinstance(prices.index, pd.MultiIndex) or prices.index.names != ["timestamp", "symbol"]:
        raise ValueError("Canonical prices must use a ['timestamp', 'symbol'] MultiIndex.")
    if prices.index.has_duplicates:
        raise ValueError("Canonical prices cannot contain duplicate timestamp/symbol rows.")
    timestamps = prices.index.get_level_values("timestamp")
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise ValueError("Canonical price timestamps must be datetime values.")

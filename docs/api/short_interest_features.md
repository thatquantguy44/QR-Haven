# API: Short Interest and Utilization Features

All securities-lending features return a `FeatureDefinition` and are registered through
the standard `FeatureRegistry` / `compute_features` interface.

```python
from qr_haven.features import (
    short_interest_ratio,
    days_to_cover,
    utilization_rate,
    borrow_fee_rate,
    short_interest_change,
    securities_lending_feature_registry,
)
```

---

## `short_interest_ratio`

```python
short_interest_ratio(*, version: str = "1") -> FeatureDefinition
```

| Property | Value |
|----------|-------|
| Name | `short_interest_ratio` |
| Required columns | `short_interest`, `float_shares` |
| Lookback | 0 |
| Output dtype | `float64` |

Formula: `short_interest / float_shares`

---

## `days_to_cover`

```python
days_to_cover(volume_window: int = 20, *, version: str = "1") -> FeatureDefinition
```

| Property | Value |
|----------|-------|
| Name | `days_to_cover` |
| Required columns | `short_interest`, `volume` |
| Lookback | `volume_window` |
| Output dtype | `float64` |

Formula: `short_interest / rolling_mean(volume, volume_window)`

Raises `ValueError` if `volume_window < 1`.

---

## `utilization_rate`

```python
utilization_rate(*, version: str = "1") -> FeatureDefinition
```

| Property | Value |
|----------|-------|
| Name | `utilization_rate` |
| Required columns | `on_loan`, `lendable_supply` |
| Lookback | 0 |
| Output dtype | `float64` |

Formula: `on_loan / lendable_supply`

---

## `borrow_fee_rate`

```python
borrow_fee_rate(*, version: str = "1") -> FeatureDefinition
```

| Property | Value |
|----------|-------|
| Name | `borrow_fee_rate` |
| Required columns | `fee_rate` |
| Lookback | 0 |
| Output dtype | `float64` |

Passes the `fee_rate` column through as a feature value.

---

## `short_interest_change`

```python
short_interest_change(window: int = 5, *, version: str = "1") -> FeatureDefinition
```

| Property | Value |
|----------|-------|
| Name | `short_interest_change` |
| Required columns | `short_interest` |
| Lookback | `window` |
| Output dtype | `float64` |

Formula: `pct_change(short_interest, periods=window)`

Raises `ValueError` if `window < 1`.

---

## `securities_lending_feature_registry`

```python
securities_lending_feature_registry() -> tuple[FeatureDefinition, ...]
```

Returns all five feature definitions with default parameters, ready to pass to
`FeatureRegistry`.

```python
from qr_haven.features import FeatureRegistry, compute_features, securities_lending_feature_registry

registry = FeatureRegistry(securities_lending_feature_registry())
features = compute_features(prices, registry)
```

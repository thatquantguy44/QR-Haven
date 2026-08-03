# API: Borrow Cost Model

```python
from qr_haven.costs import (
    BorrowCostSchedule,
    BorrowCostResult,
    BorrowCostModel,
)
```

---

## `BorrowCostSchedule`

```python
BorrowCostSchedule(
    fee_rates: pd.Series,    # annualized fee rates by symbol, must be >= 0
    days_in_year: float = 360.0,
)
```

Raises `ValueError` if any `fee_rates` are negative or `days_in_year <= 0`.

### `from_scalars`

```python
BorrowCostSchedule.from_scalars(
    symbols: Sequence[str],
    fee_rate: float,
    days_in_year: float = 360.0,
) -> BorrowCostSchedule
```

Build a uniform schedule with the same fee rate for all symbols.

### `gc_schedule`

```python
BorrowCostSchedule.gc_schedule(
    symbols: Sequence[str],
    gc_rate: float = 0.002,   # 20 bps default
    days_in_year: float = 360.0,
) -> BorrowCostSchedule
```

Convenience constructor for general-collateral (GC) names at 20 bps.

---

## `BorrowCostResult`

| Field | Type | Description |
|-------|------|-------------|
| `position_costs` | `pd.Series` | Per-symbol borrow cost for the holding period |
| `total_borrow_cost` | `float` | Total dollar borrow cost |
| `cost_on_nav` | `float` | Annualized borrow cost as fraction of NAV |

### `summary`

```python
result.summary() -> dict[str, float]
# {"total_borrow_cost": ..., "cost_on_nav": ...}
```

---

## `BorrowCostModel`

### `estimate`

```python
model.estimate(
    weights: pd.Series,                  # signed portfolio weights
    nav: float,                          # portfolio NAV in dollars, must be > 0
    schedule: BorrowCostSchedule,
    holding_period_days: float = 1.0,    # must be > 0
) -> BorrowCostResult
```

Only short (negative) weights contribute to borrow cost. Long positions and symbols absent
from `schedule` are ignored.

**Formula:**

```text
borrow_cost_i = |short_weight_i| × NAV × fee_rate_i × (holding_days / days_in_year)
cost_on_nav   = (total_borrow_cost / NAV) × (days_in_year / holding_days)
```

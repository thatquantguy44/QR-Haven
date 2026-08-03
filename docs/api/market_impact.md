# API: Market Impact Models

```python
from qr_haven.costs import (
    ImpactEstimate,
    SquareRootImpactModel,
    AlmgrenChrissModel,
    estimate_portfolio_impact,
    portfolio_total_impact_cost,
)
```

---

## `ImpactEstimate`

Frozen dataclass returned by both model `estimate` methods.

| Field | Type | Description |
|-------|------|-------------|
| `temporary_impact_bps` | `float` | Temporary price concession in basis points |
| `permanent_impact_bps` | `float` | Permanent equilibrium shift in basis points |
| `total_impact_bps` | `float` | `temporary + 0.5 × permanent` |

### `to_dict`

```python
est.to_dict() -> dict[str, float]
```

---

## `SquareRootImpactModel`

```python
SquareRootImpactModel(eta: float = 0.1)
```

Raises `ValueError` if `eta <= 0`.

### `estimate`

```python
model.estimate(
    participation_rate: float,   # |trade_value| / ADV, non-negative
    daily_volatility: float,     # daily return vol as decimal, non-negative
) -> ImpactEstimate
```

Formula: `impact_bps = eta × daily_volatility × 10_000 × sqrt(participation_rate)`

All impact is classified as temporary; permanent is always 0.

---

## `AlmgrenChrissModel`

```python
AlmgrenChrissModel(
    eta: float = 0.1,    # temporary impact coefficient, must be > 0
    gamma: float = 0.1,  # permanent impact coefficient, must be >= 0
    alpha: float = 0.6,  # temporary impact exponent, must be in (0, 1]
)
```

### `estimate`

```python
model.estimate(
    participation_rate: float,
    daily_volatility: float,
) -> ImpactEstimate
```

Formulas:

```text
sigma_bps     = daily_volatility × 10_000
temporary_bps = eta × sigma_bps × participation_rate^alpha
permanent_bps = gamma × sigma_bps × participation_rate
total_bps     = temporary_bps + 0.5 × permanent_bps
```

---

## `estimate_portfolio_impact`

```python
estimate_portfolio_impact(
    weight_changes: pd.Series,       # |w_new - w_old| per symbol
    daily_volatilities: pd.Series,   # daily vol per symbol (decimal)
    participation_rates: pd.Series,  # |trade_value| / ADV per symbol
    model: SquareRootImpactModel | AlmgrenChrissModel,
) -> pd.DataFrame
```

Returns a DataFrame indexed by symbol with columns:

| Column | Description |
|--------|-------------|
| `weight_change` | Absolute weight change |
| `participation_rate` | Input participation rate |
| `daily_volatility` | Input daily volatility |
| `temporary_impact_bps` | Temporary impact in bps |
| `permanent_impact_bps` | Permanent impact in bps |
| `total_impact_bps` | Total impact in bps |
| `weighted_cost` | `total_impact_bps × weight_change / 10_000` |

Symbols absent from `daily_volatilities` or `participation_rates` are filled with 0.

---

## `portfolio_total_impact_cost`

```python
portfolio_total_impact_cost(impact_frame: pd.DataFrame) -> float
```

Sums `weighted_cost` across all assets. Result is the total portfolio impact cost
as a fraction of NAV, directly subtractable from a gross return.

# API: Securities Lending Revenue

## `LendingFeeSchedule`

```python
from qr_haven.costs import LendingFeeSchedule
```

Frozen dataclass holding per-symbol lending parameters.

| Field | Type | Description |
|-------|------|-------------|
| `fee_rates` | `pd.Series` | Annualized borrow fee by symbol (e.g. 0.005 = 50 bps) |
| `on_loan_fractions` | `pd.Series` | Fraction of each position expected on loan, in [0, 1] |
| `reinvestment_spread` | `pd.Series` | Cash-collateral reinvestment spread by symbol; 0.0 for non-cash |
| `collateral_haircut` | `float` | Over-collateralization ratio (default 1.02) |
| `days_in_year` | `float` | Day-count convention for fee accrual (default 360.0) |

### `from_scalars`

```python
LendingFeeSchedule.from_scalars(
    symbols: Sequence[str],
    fee_rate: float,
    on_loan_fraction: float,
    reinvestment_spread: float = 0.0,
    collateral_haircut: float = 1.02,
    days_in_year: float = 360.0,
) -> LendingFeeSchedule
```

Convenience constructor that applies the same parameters to all symbols.

---

## `LendingRevenueResult`

```python
from qr_haven.costs import LendingRevenueResult
```

Frozen dataclass returned by `SecuritiesLendingRevenueModel.estimate`.

| Field | Type | Description |
|-------|------|-------------|
| `fee_income` | `pd.Series` | Per-symbol lending fee income over the holding period |
| `reinvestment_income` | `pd.Series` | Per-symbol cash-collateral reinvestment income |
| `position_revenue` | `pd.Series` | Total per-symbol revenue (fee + reinvestment) |
| `total_fee_income` | `float` | Sum of fee income across all symbols |
| `total_reinvestment_income` | `float` | Sum of reinvestment income across all symbols |
| `total_lending_revenue` | `float` | Total lending revenue over the holding period |
| `revenue_on_nav` | `float` | Annualized revenue as fraction of NAV |

### `summary`

```python
result.summary() -> dict[str, float]
```

Returns a compact serializable dict with the four aggregate fields.

---

## `SecuritiesLendingRevenueModel`

```python
from qr_haven.costs import SecuritiesLendingRevenueModel
```

### `estimate`

```python
SecuritiesLendingRevenueModel().estimate(
    weights: pd.Series,
    nav: float,
    schedule: LendingFeeSchedule,
    holding_period_days: float = 1.0,
) -> LendingRevenueResult
```

Estimates lending revenue for a single portfolio snapshot.

**Parameters**

- `weights` — Portfolio weights by symbol. Only long (positive) weights contribute;
  short weights are clamped to zero.
- `nav` — Portfolio net asset value in dollars. Must be positive.
- `schedule` — Per-symbol fee rates and reinvestment parameters. Symbols in `weights`
  not present in `schedule` are silently excluded from revenue.
- `holding_period_days` — Calendar days over which to estimate revenue. Revenue is
  linearly prorated via the schedule's day-count convention.

**Revenue formulas**

```text
on_loan_value       = weight × nav × on_loan_fraction
fee_income          = on_loan_value × fee_rate × (holding_period_days / days_in_year)
collateral_value    = on_loan_value × collateral_haircut
reinvestment_income = collateral_value × reinvestment_spread × (holding_period_days / days_in_year)
position_revenue    = fee_income + reinvestment_income
revenue_on_nav      = (total_revenue / nav) × (days_in_year / holding_period_days)
```

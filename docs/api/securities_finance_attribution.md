# API: Securities Finance Attribution

```python
from qr_haven.costs import (
    SecuritiesFinanceAttribution,
    SecuritiesFinanceAttributionModel,
)
```

---

## `SecuritiesFinanceAttributionModel`

### `attribute`

```python
model.attribute(
    weights: pd.Series,                      # signed portfolio weights
    nav: float,                              # portfolio NAV in dollars, must be > 0
    lending_schedule: LendingFeeSchedule,    # long-book lending parameters
    borrow_schedule: BorrowCostSchedule,     # short-book borrow rates
    financing_rates: FinancingRates,         # prime broker debit/credit rates
    holding_period_days: float = 1.0,
) -> SecuritiesFinanceAttribution
```

Runs all three sub-models and returns a unified attribution object.

---

## `SecuritiesFinanceAttribution`

| Field | Type | Description |
|-------|------|-------------|
| `lending_revenue` | `LendingRevenueResult` | Full lending revenue breakdown |
| `borrow_cost` | `BorrowCostResult` | Full borrow cost breakdown |
| `financing_cost` | `FinancingCostResult` | Full financing cost breakdown |
| `net_securities_finance` | `float` | Net dollar P&L (lending − borrow − financing) |
| `net_on_nav` | `float` | Annualized net P&L as fraction of NAV |

### `summary`

```python
result.summary() -> dict[str, float]
```

Returns 10 scalar fields:

| Key | Description |
|-----|-------------|
| `total_lending_revenue` | |
| `lending_revenue_on_nav` | |
| `total_borrow_cost` | |
| `borrow_cost_on_nav` | |
| `debit_cost` | |
| `credit_income` | |
| `net_financing_cost` | |
| `financing_cost_on_nav` | |
| `net_securities_finance` | |
| `net_on_nav` | |

### `waterfall`

```python
result.waterfall() -> pd.Series
```

Returns a `pd.Series` named `"securities_finance_waterfall"` with signed dollar values:

| Label | Sign | Source |
|-------|------|--------|
| `lending_fee_income` | + | Lending fee revenue |
| `reinvestment_income` | + | Cash collateral reinvestment |
| `borrow_cost` | − | Short-book borrow fee (stored as negative) |
| `debit_interest` | − | Margin debit interest (stored as negative) |
| `credit_income` | + | Short-proceeds credit |
| `net_securities_finance` | ± | Signed net total |

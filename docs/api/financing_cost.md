# API: Financing Cost Model

```python
from qr_haven.costs import (
    FinancingRates,
    FinancingCostResult,
    FinancingCostModel,
)
```

---

## `FinancingRates`

```python
FinancingRates(
    debit_rate: float,           # annualized margin debit rate, must be >= 0
    credit_rate: float,          # annualized credit rate on short proceeds, must be >= 0
    days_in_year: float = 360.0,
)
```

Raises `ValueError` if `debit_rate < 0`, `credit_rate < 0`, or `days_in_year <= 0`.

---

## `FinancingCostResult`

| Field | Type | Description |
|-------|------|-------------|
| `debit_balance` | `float` | Dollar margin debit (`max(0, long_gross − 1 − short_gross) × NAV`) |
| `credit_balance` | `float` | Dollar short-sale proceeds at broker (`short_gross × NAV`) |
| `debit_cost` | `float` | Dollar debit interest for the holding period |
| `credit_income` | `float` | Dollar credit income on short proceeds |
| `net_financing_cost` | `float` | `debit_cost − credit_income` |
| `cost_on_nav` | `float` | Annualized net financing cost as fraction of NAV |

### `summary`

```python
result.summary() -> dict[str, float]
# {"debit_balance", "credit_balance", "debit_cost",
#  "credit_income", "net_financing_cost", "cost_on_nav"}
```

---

## `FinancingCostModel`

### `estimate`

```python
model.estimate(
    weights: pd.Series,                  # signed portfolio weights
    nav: float,                          # portfolio NAV in dollars, must be > 0
    rates: FinancingRates,
    holding_period_days: float = 1.0,    # must be > 0
) -> FinancingCostResult
```

**Formulas:**

```text
long_gross     = sum(w_i for w_i > 0)
short_gross    = sum(|w_i| for w_i < 0)

debit_balance  = max(0, long_gross − 1.0 − short_gross) × NAV
credit_balance = short_gross × NAV

day_fraction   = holding_period_days / days_in_year
debit_cost     = debit_balance × debit_rate × day_fraction
credit_income  = credit_balance × credit_rate × day_fraction

net_financing_cost = debit_cost − credit_income
cost_on_nav        = (net_financing_cost / NAV) × (days_in_year / holding_period_days)
```

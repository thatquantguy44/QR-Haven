# Financing Cost Model

## Overview

When a fund holds gross long exposure beyond its equity (NAV), it must borrow cash from the
prime broker to fund the excess. This margin loan accrues **debit interest**. Simultaneously,
the proceeds from short sales are held at the broker; the fund earns a **credit rate** on
these proceeds (typically a haircut to the overnight rate).

```text
debit_balance  = max(0, long_gross − 1.0 − short_gross) × NAV
credit_balance = short_gross × NAV

debit_cost     = debit_balance × debit_rate × day_fraction
credit_income  = credit_balance × credit_rate × day_fraction

net_financing_cost = debit_cost − credit_income
```

A **positive** net financing cost is a drag on P&L; a **negative** value means the short
proceeds credit more than offsets any margin interest (net income).

## Portfolio Structure Examples

| Strategy | long_gross | short_gross | debit_balance | Notes |
|----------|-----------|-------------|---------------|-------|
| Long-only 100% | 1.00 | 0.00 | 0 | No leverage |
| 130/30 | 1.30 | 0.30 | 0 | Short proceeds fund excess longs exactly |
| 150/30 | 1.50 | 0.30 | 0.20 × NAV | Requires margin borrowing |
| Market-neutral 100/100 | 1.00 | 1.00 | 0 | Large credit balance |

## Usage

```python
from qr_haven.costs import FinancingCostModel, FinancingRates
import pandas as pd

# 130/30 portfolio — no margin debit needed
weights = pd.Series({"AAPL": 0.80, "MSFT": 0.50, "TSLA": -0.30})

rates = FinancingRates(
    debit_rate=0.055,   # SOFR + spread (~5.5%)
    credit_rate=0.020,  # short proceeds credit (prime − spread)
)

model = FinancingCostModel()
result = model.estimate(weights, nav=10_000_000, rates=rates)

print(result.debit_balance)        # dollars borrowed on margin
print(result.credit_balance)       # short-sale proceeds at broker
print(result.net_financing_cost)   # net dollar cost (positive = drag)
print(result.cost_on_nav)          # annualized net cost as fraction of NAV
```

## Typical Rates

| Rate | Typical Level | Notes |
|------|--------------|-------|
| Debit rate | Fed Funds / SOFR + 50–150 bps | Higher spread for smaller accounts |
| Credit rate on short proceeds | Fed Funds − 25–75 bps | Large institutional accounts get tighter spread |

## Annualization

`cost_on_nav` is always annualized regardless of `holding_period_days`:

```text
cost_on_nav = (net_financing_cost / NAV) × (days_in_year / holding_period_days)
```

This allows direct comparison across different estimation windows.

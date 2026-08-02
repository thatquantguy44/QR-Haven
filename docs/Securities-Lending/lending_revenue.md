# Securities Lending Revenue Model

## Overview

`SecuritiesLendingRevenueModel` estimates the income a lending agent or beneficial owner
earns from putting a long portfolio's inventory on loan to short sellers.

This model is designed for:

- **Agency lenders** estimating revenue from custodied client assets
- **Long-only funds** modeling the lending income offset to management fees and operational costs
- **Long-short funds** building a complete P&L attribution that nets lending income against
  borrow costs on the short book

## Revenue Components

### 1. Lending Fee Income

When a borrower draws securities from the lending pool, they pay an annualized fee
(the borrow rate). The lender earns:

```text
fee_income = on_loan_value × fee_rate × (holding_period_days / days_in_year)
```

where:

- `on_loan_value = weight × NAV × on_loan_fraction`
- `fee_rate` is the annualized borrow fee (e.g., 0.005 = 50 bps for GC, 0.30+ for hard-to-borrow)
- `on_loan_fraction` is the expected fraction of the position placed on loan

### 2. Reinvestment Income (Cash Collateral)

When borrowers post cash collateral — common in the US equity lending market — the agent
invests those proceeds and earns a spread over the rebate rate paid back to borrowers:

```text
collateral_value = on_loan_value × collateral_haircut   (typically 1.02 for equities)
reinvestment_income = collateral_value × reinvestment_spread × day_fraction
```

For non-cash collateral (securities, government bonds), this component is zero.

### Total Revenue

```text
position_revenue = fee_income + reinvestment_income
revenue_on_nav   = (total_revenue / NAV) × (days_in_year / holding_period_days)
```

`revenue_on_nav` is always expressed as an annualized rate for comparability.

## Typical Market Parameters

| Category | Fee Rate | On-Loan Fraction |
|----------|----------|-----------------|
| General collateral (GC) | 10–25 bps | 20–40% |
| Warm names | 25–100 bps | 30–60% |
| Hard-to-borrow (specials) | 100 bps – 30%+ | 50–95% |

Reinvestment spread for cash collateral varies with the rate environment; typical
ranges are 5–30 bps over the overnight rate.

## Usage

### Uniform schedule

```python
from qr_haven.costs import LendingFeeSchedule, SecuritiesLendingRevenueModel
import pandas as pd

symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
weights = pd.Series([0.30, 0.25, 0.25, 0.20], index=symbols)
nav = 50_000_000.0

schedule = LendingFeeSchedule.from_scalars(
    symbols,
    fee_rate=0.005,          # 50 bps — typical GC basket
    on_loan_fraction=0.35,   # 35% of each position on loan
    reinvestment_spread=0.0015,  # 15 bps cash reinvestment spread
    collateral_haircut=1.02,
)

result = SecuritiesLendingRevenueModel().estimate(
    weights, nav, schedule, holding_period_days=252.0
)
print(result.summary())
# {
#   "total_fee_income": ...,
#   "total_reinvestment_income": ...,
#   "total_lending_revenue": ...,
#   "revenue_on_nav": 0.00535,   # ~53.5 bps annualized
# }
```

### Per-symbol schedule

```python
from qr_haven.costs import LendingFeeSchedule
import pandas as pd

fee_rates = pd.Series(
    {"AAPL": 0.003, "GME": 0.85, "AMC": 1.20},
    name="fee_rate",
)
on_loan_fractions = pd.Series(
    {"AAPL": 0.30, "GME": 0.90, "AMC": 0.85},
    name="on_loan_fraction",
)
reinvestment_spread = pd.Series(
    {"AAPL": 0.001, "GME": 0.001, "AMC": 0.001},
    name="reinvestment_spread",
)

schedule = LendingFeeSchedule(
    fee_rates=fee_rates,
    on_loan_fractions=on_loan_fractions,
    reinvestment_spread=reinvestment_spread,
)
```

## Integration with Performance Attribution

The `summary()` output is designed to slot into the QR-Haven P&L waterfall alongside
other attribution components:

```python
from qr_haven.backtesting import attribute_backtest

attribution = attribute_backtest(result)
lending_summary = lending_result.summary()

# Combine into a unified attribution table:
# gross_return + lending_revenue - borrow_costs - financing_costs = net_return
```

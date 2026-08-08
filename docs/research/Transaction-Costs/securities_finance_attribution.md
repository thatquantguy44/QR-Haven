# Securities Finance Attribution

## Overview

Securities finance attribution decomposes the total funding P&L of a portfolio into three
signed components:

```text
net_securities_finance = lending_revenue − borrow_cost − financing_cost
```

| Component | Direction | Model |
|-----------|-----------|-------|
| Lending revenue | Income (+) | `SecuritiesLendingRevenueModel` |
| Borrow cost | Drag (−) | `BorrowCostModel` |
| Net financing cost | Usually drag (−) | `FinancingCostModel` |

`SecuritiesFinanceAttributionModel` wraps all three sub-models and aggregates them in a
single call.

## P&L Waterfall

```text
+ Lending fee income          (long positions × on-loan fraction × fee rate)
+ Reinvestment income         (cash collateral × reinvestment spread)
─ Borrow cost                 (short positions × borrow fee rate)
─ Debit interest              (margin debit balance × debit rate)
+ Credit income               (short proceeds × credit rate)
═══════════════════════════════
= Net securities finance P&L
```

## Usage

```python
from qr_haven.costs import (
    SecuritiesFinanceAttributionModel,
    LendingFeeSchedule,
    BorrowCostSchedule,
    FinancingRates,
)
import pandas as pd

weights = pd.Series({"AAPL": 0.80, "MSFT": 0.50, "TSLA": -0.30})
symbols = list(weights.index)

# Long-book lending schedule
lending_schedule = LendingFeeSchedule.from_scalars(
    symbols,
    fee_rate=0.002,          # 20 bps GC fee
    on_loan_fraction=0.20,   # 20% of long positions on loan
    reinvestment_spread=0.001,
)

# Short-book borrow schedule (TSLA is HTB)
from pandas import Index, Series
idx = Index(symbols, name="symbol")
borrow_schedule = BorrowCostSchedule(
    fee_rates=Series([0.002, 0.002, 0.15], index=idx, name="fee_rate")
)

# Prime broker financing rates
financing_rates = FinancingRates(debit_rate=0.055, credit_rate=0.020)

model = SecuritiesFinanceAttributionModel()
result = model.attribute(
    weights=weights,
    nav=10_000_000,
    lending_schedule=lending_schedule,
    borrow_schedule=borrow_schedule,
    financing_rates=financing_rates,
    holding_period_days=1,
)

# Compact summary
print(result.summary())

# Full P&L waterfall (dollar terms)
print(result.waterfall())

# Net annualized contribution to return
print(f"Net securities finance: {result.net_on_nav * 10_000:.1f} bps/yr")
```

## Output Fields

### `SecuritiesFinanceAttribution.summary()`

| Key | Description |
|-----|-------------|
| `total_lending_revenue` | Dollar lending revenue (fee + reinvestment) |
| `lending_revenue_on_nav` | Annualized lending revenue / NAV |
| `total_borrow_cost` | Dollar borrow fee for the holding period |
| `borrow_cost_on_nav` | Annualized borrow cost / NAV |
| `debit_cost` | Dollar debit interest for the holding period |
| `credit_income` | Dollar credit income on short proceeds |
| `net_financing_cost` | `debit_cost − credit_income` |
| `financing_cost_on_nav` | Annualized net financing cost / NAV |
| `net_securities_finance` | Net dollar P&L from all three components |
| `net_on_nav` | Annualized net / NAV |

### `SecuritiesFinanceAttribution.waterfall()`

Returns a labelled `pd.Series` with items:

```
lending_fee_income      +
reinvestment_income     +
borrow_cost             -  (negative = drag)
debit_interest          -  (negative = drag)
credit_income           +
net_securities_finance  =  (signed sum)
```

## Strategy Profiles

| Strategy | Dominant effect |
|----------|----------------|
| Long-only passive | Small lending income; zero borrow and financing cost |
| 130/30 | Lending income partially offset by HTB borrow cost; no margin debit |
| Market-neutral | Large credit income from shorts; borrow cost is key drag |
| Levered long | Debit interest is primary cost; lending provides partial offset |

# Securities Lending Research

This documentation folder covers QR-Haven's securities lending models and alpha features.

Securities lending sits at the intersection of portfolio construction, risk, and revenue
generation. For a long-only portfolio it is an income source. For a long-short portfolio
it is a cost center on the short side and a revenue source on the long side. Understanding
both faces is essential for accurate P&L attribution and realistic backtesting.

## Subsystems

### Revenue Modeling

`SecuritiesLendingRevenueModel` estimates the income earned by a lending agent or custodian
from putting a long portfolio's inventory on loan. Revenue has two components:

```text
long portfolio inventory
  -> on-loan fraction (demand)
  -> fee income   = on_loan_value × fee_rate × day_fraction
  -> cash collateral reinvestment income
       = collateral_value × reinvestment_spread × day_fraction
  -> total lending revenue
  -> revenue as fraction of NAV (annualized)
```

### Short Interest and Utilization Features

`features/securities_lending.py` registers five point-in-time-safe alpha features
derived from the securities lending market:

| Feature | Signal | Required columns |
|---------|--------|-----------------|
| `short_interest_ratio` | Crowding relative to float | `short_interest`, `float_shares` |
| `days_to_cover` | Short squeeze persistence | `short_interest`, `volume` |
| `utilization_rate` | Borrow demand vs. supply | `on_loan`, `lendable_supply` |
| `borrow_fee_rate` | Cost-to-borrow as signal | `fee_rate` |
| `short_interest_change` | Momentum in short conviction | `short_interest` |

All features are computed through `compute_features` and inherit the Feature Store's
temporal safety guarantee — no future data leaks into any calculation.

## P&L Waterfall

These models are designed to slot into the QR-Haven attribution waterfall:

```text
gross return
  → execution costs (market impact, slippage)
  → borrow costs (short side)
  → financing costs (leverage)
  → lending revenue (long side)
  → net return
```

## Documentation Map

- [Lending Revenue Model](lending_revenue.md)
- [Short Interest and Utilization Features](short_interest_features.md)
- [API: Lending Revenue](../api/lending_revenue.md)
- [API: Short Interest Features](../api/short_interest_features.md)

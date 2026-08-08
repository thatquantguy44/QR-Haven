# Transaction Cost Models

Research home for Almgren-Chriss, square-root impact, borrow costs, financing costs, slippage, and
opportunity cost models.

Accurate transaction cost modeling is essential for realistic backtesting. A strategy
that looks profitable gross of costs can be unprofitable net of impact, especially at
higher frequencies or with lower-liquidity names. The models here are designed to slot
into the QR-Haven P&L attribution waterfall:

```text
gross return
  → market impact costs   ← this module
  → borrow costs (short side)
  → financing costs
  → lending revenue (long side)
  → net return
```

### Square-Root Impact Model

The industry-standard workhorse. Impact scales with the square root of the participation
rate (trade size relative to average daily volume):

```text
impact_bps = eta × sigma_daily_bps × sqrt(participation_rate)
```

### Almgren-Chriss Model

Separates temporary and permanent impact components. Temporary impact is the price
concession paid to attract liquidity; permanent impact is the lasting shift in
equilibrium price from revealed order flow:

```text
temporary_bps = eta × sigma_daily_bps × participation_rate^alpha
permanent_bps = gamma × sigma_daily_bps × participation_rate
total_bps     = temporary_bps + 0.5 × permanent_bps
```


## Implemented (src/qr_haven/costs)

- `SquareRootImpactModel` — square-root market impact; eta × sigma × sqrt(participation_rate)
- `AlmgrenChrissModel` — temporary + permanent impact; parameterized by eta, gamma, alpha
- `estimate_portfolio_impact` — per-asset impact frame for a full rebalance
- `portfolio_total_impact_cost` — aggregate cost as fraction of NAV
- `SecuritiesLendingRevenueModel` — lending fee and reinvestment income for long portfolios
- `BorrowCostModel` — daily fee accrual on short positions; GC and HTB schedules
- `FinancingCostModel` — prime broker debit/credit balance; handles 130/30, MN, levered longs
- `SecuritiesFinanceAttributionModel` — unified P&L waterfall across all three models

## Planned

- Slippage model (bid-ask spread component)
- Opportunity cost (risk of not trading)

## Documentation

- [Transaction Cost Models](../../docs/Transaction-Costs/README.md)
- [Market Impact](../../docs/Transaction-Costs/market_impact.md)
- [Borrow Cost](../../docs/Transaction-Costs/borrow_cost.md)
- [Financing Cost](../../docs/Transaction-Costs/financing_cost.md)
- [Securities Finance Attribution](../../docs/Transaction-Costs/securities_finance_attribution.md)
- [Securities Lending Revenue](../../docs/Securities-Lending/lending_revenue.md)


# Transaction Cost Models

Research home for Almgren-Chriss, square-root impact, borrow costs, financing costs, slippage, and
opportunity cost models.

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


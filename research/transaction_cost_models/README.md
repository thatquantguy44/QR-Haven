# Transaction Cost Models

Research home for Almgren-Chriss, square-root impact, borrow costs, financing costs, slippage, and
opportunity cost models.

## Implemented (src/qr_haven/costs)

- `SquareRootImpactModel` — square-root market impact; eta × sigma × sqrt(participation_rate)
- `AlmgrenChrissModel` — temporary + permanent impact; parameterized by eta, gamma, alpha
- `estimate_portfolio_impact` — per-asset impact frame for a full rebalance
- `portfolio_total_impact_cost` — aggregate cost as fraction of NAV
- `SecuritiesLendingRevenueModel` — lending fee and reinvestment income for long portfolios

## Planned

- Slippage model (bid-ask spread component)
- Financing cost model (debit/credit balance at prime broker)
- Borrow cost model (short-side fee accrual)
- Opportunity cost (risk of not trading)

## Documentation

- [Transaction Cost Models](../../docs/Transaction-Costs/README.md)
- [Market Impact](../../docs/Transaction-Costs/market_impact.md)
- [Securities Lending Revenue](../../docs/Securities-Lending/lending_revenue.md)


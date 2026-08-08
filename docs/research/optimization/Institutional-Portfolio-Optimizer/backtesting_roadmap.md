# Backtesting Roadmap

The next portfolio optimizer milestone is a small, repeatable backtest harness.

## v0 Workflow

```text
CSV / SQLite prices
  -> daily return matrix
  -> rolling rebalance dates
  -> estimate returns and covariance from lookback window
  -> optimize target weights
  -> calculate forward portfolio returns
  -> apply transaction cost placeholder
  -> compute performance and risk summary
  -> emit terminal panel payload
```

## Core Parameters

- Rebalance frequency: monthly or every N periods
- Lookback window: default 60 trading days
- Optimizer: equal weight or mean variance
- Constraints: long-only, max weight, fully invested
- Transaction cost: flat basis-point cost multiplied by turnover

## Outputs

- Portfolio return series
- Equity curve
- Target weights by rebalance date
- Turnover by rebalance date
- Risk metrics from `SimpleRiskEngine`
- Terminal-ready backtest summary panel

## Non-goals for v0

- Intraday simulation
- Borrow and financing costs
- Market impact
- Corporate action handling beyond the adjusted prices supplied by the data source
- Attribution decomposition


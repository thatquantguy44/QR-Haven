# Institutional Portfolio Optimizer

This documentation folder captures the design, implementation notes, and research workflow for
QR-Haven's institutional portfolio optimizer.

The optimizer is not just a standalone allocation function. It is the first vertical slice of the
QR-Haven research platform:

```text
CSV / SQLite prices
  -> canonical price data
  -> returns
  -> expected return and covariance estimates
  -> optimizer weights
  -> risk metrics
  -> market_terminal panels
```

## Current v0 Scope

- CSV and SQLite price ingestion
- Canonical long-form price schema
- Simple and log return calculation
- Equal-weight optimizer
- Mean-variance optimizer with bounded long-only constraints
- Portfolio risk metrics
- Terminal-ready risk panel payload
- Walk-forward backtest harness
- Terminal-ready backtest summary and equity curve panels
- Performance attribution for returns, costs, assets, and optimizer diagnostics
- Optimizer diagnostics history and constraint-pressure panels
- Portfolio analytics and complete terminal panel bundle

## Documentation Map

- [Data Ingestion](data_ingestion.md)
- [Feature Store](../api/feature_store.md)
- [Returns and Optimization](returns_and_optimization.md)
- [Optimizer Diagnostics](optimizer_diagnostics.md)
- [Risk Engine](risk_engine.md)
- [Backtesting](backtesting.md)
- [Performance Attribution](performance_attribution.md)
- [Terminal Integration](terminal_integration.md)
- [Backtesting Roadmap](backtesting_roadmap.md)

## Promotion Rule

Exploratory research belongs in `research/portfolio_optimizer`. Reusable platform code belongs in
`src/qr_haven` with tests and typed interfaces.

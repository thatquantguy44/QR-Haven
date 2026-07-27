# Backtesting

QR-Haven v0 provides a walk-forward backtest harness for portfolio optimizer research.

## Workflow

```text
returns
  -> rolling lookback window
  -> expected return and covariance estimates
  -> optimizer target weights
  -> hold weights until next rebalance
  -> apply flat transaction cost
  -> portfolio returns and equity curve
  -> risk summary
```

## Example

```python
from qr_haven.backtesting import BacktestConfig, WalkForwardBacktester
from qr_haven.portfolio import MeanVarianceOptimizer

config = BacktestConfig(
    lookback_periods=60,
    rebalance_periods=21,
    transaction_cost_bps=10.0,
    optimizer_constraints={"long_only": True, "max_weight": 0.25},
)

result = WalkForwardBacktester(MeanVarianceOptimizer(), config).run(returns)
summary = result.summary()
```

## Outputs

`BacktestResult` contains:

- `portfolio_returns`
- `equity_curve`
- `weights`
- `turnover`
- `transaction_costs`
- `risk_metrics`

Return-path risk metrics such as volatility, Sharpe, drawdown, VaR, and CVaR are calculated from
the full backtested portfolio return stream. Exposure and concentration metrics use the final
rebalance weights in v0.

## Transaction Costs

The v0 transaction cost model is intentionally simple:

```text
cost = turnover * transaction_cost_bps / 10,000
```

The cost is subtracted from the first portfolio return in each holding period.

## Terminal Panels

```python
from qr_haven.integrations.market_terminal import (
    backtest_equity_curve_panel,
    backtest_summary_panel,
)

summary_panel = backtest_summary_panel(result)
equity_panel = backtest_equity_curve_panel(result)
```

## Non-goals for v0

- Intraday execution simulation
- Borrow and financing costs
- Market impact
- Corporate-action modeling beyond source adjusted prices
- Performance attribution

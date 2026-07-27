# Performance Attribution

QR-Haven v0 explains a portfolio backtest across four practical lenses:

- Return-path performance
- Turnover and transaction-cost drag
- Asset contribution
- Optimizer diagnostics

## Example

```python
from qr_haven.backtesting import attribute_backtest

attribution = attribute_backtest(result, annualization=252)
```

## Performance Summary

The performance summary includes:

- Observations
- Total return
- Gross total return before transaction costs
- Transaction cost drag
- CAGR
- Hit rate
- Best period
- Worst period
- Ending equity

## Turnover and Costs

Turnover and cost attribution includes:

- Rebalance count
- Average turnover
- Max turnover
- Total transaction cost
- Average transaction cost
- Max transaction cost

## Asset Contribution

Asset contribution is calculated from per-period asset returns multiplied by the portfolio weights
held during that period.

The asset contribution table includes:

- Total contribution
- Average weight
- Ending weight

## Optimizer Diagnostics

Optimizer diagnostics summarize rebalance-time portfolio construction behavior:

- Rebalance count
- Average concentration
- Average effective holdings
- Average gross exposure
- Average max weight
- Max weight
- Average ex-ante volatility
- Average expected portfolio return
- Average objective value
- Constraint pressure
- Max-weight binding rate
- Max-turnover binding rate
- Group exposure binding rate

## Terminal Panels

```python
from qr_haven.integrations.market_terminal import (
    asset_contribution_panel,
    optimizer_diagnostics_panel,
    performance_summary_panel,
    turnover_cost_panel,
)

performance_panel = performance_summary_panel(attribution)
asset_panel = asset_contribution_panel(attribution)
turnover_panel = turnover_cost_panel(attribution)
optimizer_panel = optimizer_diagnostics_panel(attribution)
```

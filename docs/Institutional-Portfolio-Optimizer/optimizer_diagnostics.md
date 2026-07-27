# Optimizer Diagnostics

QR-Haven v1 diagnostics explain whether an optimizer's output is reasonable and which constraints
appear to be shaping the portfolio.

## Constraint Model

`OptimizerConstraints` supports:

- Long-only allocations
- Min and max asset weights
- Target net allocation
- Min and max net exposure diagnostics
- Max gross exposure diagnostics
- Max turnover diagnostics
- Group max exposure diagnostics through asset-to-group mappings

The v1 optimizer directly enforces the bounded simplex constraints: long-only, min weight, max
weight, and target sum. Gross, net, turnover, and group limits are exposed as diagnostic pressure
metrics so they can be hardened into optimizer constraints later without changing the public shape.

## Single Optimization Diagnostics

```python
from qr_haven.portfolio import calculate_optimizer_diagnostics

diagnostics = calculate_optimizer_diagnostics(
    weights,
    expected_returns,
    covariance,
    constraints,
    previous_weights=previous_weights,
)
```

Diagnostics include:

- Asset count
- Expected portfolio return
- Ex-ante volatility
- Variance
- Objective value
- Gross exposure
- Net exposure
- Concentration
- Effective holdings
- Min and max weight
- Number of names at min and max weight
- Max-weight binding flag
- Turnover and max-turnover binding flag
- Max group exposure and group exposure binding flag
- Overall constraint pressure

## Backtest Diagnostics

`WalkForwardBacktester` stores optimizer diagnostics at each rebalance:

```python
result = backtester.run(returns)
diagnostics = result.diagnostics
```

The diagnostics dataframe is indexed by rebalance timestamp. It gives a historical view of whether
the optimizer was diversified, whether max-weight constraints were binding, and whether turnover or
group exposure pressure was elevated.

## Terminal Panels

```python
from qr_haven.integrations.market_terminal import (
    backtest_diagnostics_panel,
    constraint_pressure_panel,
    optimizer_diagnostics_panel,
)

history_panel = backtest_diagnostics_panel(result)
pressure_panel = constraint_pressure_panel(result)
summary_panel = optimizer_diagnostics_panel(attribution)
```


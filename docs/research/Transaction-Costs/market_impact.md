# Market Impact Models

## Overview

Market impact is the adverse price movement caused by a trade. It has two sources:

- **Temporary impact** — the price concession paid to attract counterparty liquidity.
  It reverts after execution completes and does not shift the long-run equilibrium price.

- **Permanent impact** — the lasting price shift from order flow revealing private
  information (or the market updating its beliefs about supply/demand balance).

QR-Haven provides two models with a common interface: both accept a participation rate
and daily volatility and return an `ImpactEstimate` with the breakdown.

## Participation Rate

The key input to both models is the **participation rate**:

```text
participation_rate = |trade_value| / ADV
```

where ADV is the average daily value traded. A participation rate of 0.01 means the
order is 1% of one day's trading activity.

In a portfolio backtest, the participation rate for each asset is:

```text
participation_rate_i = |Δw_i × NAV| / ADV_i
```

where `Δw_i` is the weight change for asset `i`.

## Square-Root Impact Model

```python
from qr_haven.costs import SquareRootImpactModel

model = SquareRootImpactModel(eta=0.1)
est = model.estimate(participation_rate=0.01, daily_volatility=0.02)
# ImpactEstimate(temporary=2.0 bps, permanent=0.0 bps, total=2.0 bps)
```

### Formula

```text
impact_bps = eta × sigma_daily × 10_000 × sqrt(participation_rate)
```

### Parameter guidance

| Parameter | Default | Typical range | Notes |
|-----------|---------|---------------|-------|
| `eta` | 0.10 | 0.05–0.30 | Lower for large-cap liquid; higher for small-cap/illiquid |

### Calibration

`eta` can be estimated by regressing realized implementation shortfall against
`sigma × sqrt(participation_rate)` across a historical trade sample.

## Almgren-Chriss Model

```python
from qr_haven.costs import AlmgrenChrissModel

model = AlmgrenChrissModel(eta=0.1, gamma=0.1, alpha=0.6)
est = model.estimate(participation_rate=0.01, daily_volatility=0.02)
# ImpactEstimate(temporary=..., permanent=0.2 bps, total=...)
```

### Formula

```text
temporary_bps = eta × sigma_daily_bps × participation_rate^alpha
permanent_bps = gamma × sigma_daily_bps × participation_rate
total_bps     = temporary_bps + 0.5 × permanent_bps
```

### Parameter guidance

| Parameter | Default | Typical range | Notes |
|-----------|---------|---------------|-------|
| `eta` | 0.10 | 0.05–0.30 | Temporary impact scale |
| `gamma` | 0.10 | 0.05–0.20 | Permanent impact scale |
| `alpha` | 0.60 | 0.50–0.70 | Temporary impact exponent; 0.5 recovers sqrt model |

### Relationship to the square-root model

Setting `gamma=0` and `alpha=0.5` gives temporary impact equal to the `SquareRootImpactModel`
and permanent impact of zero. Almgren-Chriss is strictly a generalization.

## Portfolio-Level Cost Estimation

```python
from qr_haven.costs import (
    AlmgrenChrissModel,
    estimate_portfolio_impact,
    portfolio_total_impact_cost,
)
import pandas as pd

symbols = ["AAPL", "MSFT", "GOOGL"]

weight_changes = pd.Series([0.10, 0.05, 0.08], index=symbols)   # |Δw| per asset
daily_vols     = pd.Series([0.015, 0.018, 0.020], index=symbols)
participation  = pd.Series([0.005, 0.010, 0.008], index=symbols) # |ΔW × NAV| / ADV

model = AlmgrenChrissModel(eta=0.1, gamma=0.1, alpha=0.6)
frame = estimate_portfolio_impact(weight_changes, daily_vols, participation, model)

# Per-asset breakdown
print(frame[["total_impact_bps", "weighted_cost"]])

# Total portfolio impact cost as fraction of NAV
total_cost = portfolio_total_impact_cost(frame)
# Equivalent to: sum(weight_change × total_impact_bps / 10_000)
```

The `weighted_cost` column is `total_impact_bps × weight_change / 10_000` — the
contribution of each asset to the portfolio-level cost expressed as a fraction of NAV.
Subtract `portfolio_total_impact_cost` from the gross portfolio return to get the
impact-adjusted return.

## Fitting into the Backtester

The current `BacktestConfig.transaction_cost_bps` is a flat rate applied to turnover.
Market impact models can replace this by pre-computing asset-level participation rates
from NAV and ADV estimates, then calling `portfolio_total_impact_cost` to produce
a dynamic cost figure at each rebalance date.

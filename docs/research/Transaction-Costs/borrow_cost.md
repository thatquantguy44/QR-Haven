# Borrow Cost Model

## Overview

When a fund sells shares short it must first borrow them from a lender. The prime broker
charges a daily fee — the **borrow rate** — expressed as an annualized percentage of the
market value of the borrowed position. This fee is the exact mirror of the lending revenue
earned by the long-side lender:

```text
borrow_cost = |short_weight| × NAV × fee_rate × (holding_days / days_in_year)
```

Hard-to-borrow (HTB) names can cost 100 bps to 30%+ per annum and represent a meaningful
drag on short-book alpha. General-collateral (GC) names — liquid, easy-to-borrow equities —
typically cost 10–25 bps driven by the overnight rate environment.

## Usage

```python
from qr_haven.costs import BorrowCostModel, BorrowCostSchedule
import pandas as pd

symbols = ["AAPL", "MSFT", "TSLA"]
weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40, "TSLA": -0.30})

# Option 1: uniform schedule (e.g., all GC names)
schedule = BorrowCostSchedule.gc_schedule(symbols)  # 20 bps default

# Option 2: per-symbol rates (mix of GC and HTB)
from pandas import Index, Series
idx = Index(symbols, name="symbol")
schedule = BorrowCostSchedule(
    fee_rates=Series([0.002, 0.002, 0.15], index=idx, name="fee_rate")
)

model = BorrowCostModel()
result = model.estimate(weights, nav=10_000_000, schedule=schedule)

print(result.total_borrow_cost)   # dollar cost for 1 day
print(result.cost_on_nav)         # annualized borrow drag as fraction of NAV
print(result.position_costs)      # per-symbol breakdown
```

## Fee Schedule Construction

| Method | Description |
|--------|-------------|
| `BorrowCostSchedule(fee_rates)` | Direct construction; pass a `pd.Series` keyed by symbol |
| `BorrowCostSchedule.from_scalars(symbols, fee_rate)` | Uniform rate for all symbols |
| `BorrowCostSchedule.gc_schedule(symbols, gc_rate=0.002)` | General-collateral convenience; default 20 bps |

## Typical Borrow Rates

| Category | Typical Rate | Examples |
|----------|-------------|---------|
| General Collateral (GC) | 10–25 bps | Large-cap S&P 500 names |
| Warm | 50–150 bps | Mid-cap, recent IPOs |
| Hard-to-Borrow (HTB) | 100 bps–10% | Small-cap, meme stocks, distressed |
| Specials | 10%–30%+ | Short squeezes, concentrated borrow demand |

## Relationship to Lending Revenue

The borrow cost paid by the short-side fund is the revenue received by the long-side lender
(minus broker spread). In a 130/30 portfolio:

- Long book earns lending fees on loaned shares → `SecuritiesLendingRevenueModel`
- Short book pays borrow fees on borrowed shares → `BorrowCostModel`

Use `SecuritiesFinanceAttributionModel` to compute the net P&L from both sides together.

## Integration with Attribution

See [Securities Finance Attribution](./securities_finance_attribution.md) for the unified
model that combines lending revenue, borrow cost, and financing cost into a single P&L
waterfall.

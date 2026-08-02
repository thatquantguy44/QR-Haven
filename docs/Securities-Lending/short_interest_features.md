# Short Interest and Utilization Features

## Overview

`features/securities_lending.py` provides five point-in-time-safe alpha features derived
from the securities lending market. These features are registered through the same
`FeatureDefinition` contract as all other QR-Haven features and flow directly into the
Alpha Research Factory's IC evaluation and signal combination framework.

Securities lending data exposes real-money positioning in a way that most public data
sources do not. Utilization and fee rates reflect what sophisticated investors are actually
paying to maintain short exposure — a signal embedded directly in the cost of the trade.

## Required Canonical Columns

Beyond `adjusted_close`, the securities lending features expect these columns in the
canonical price frame:

| Column | Description | Typical source |
|--------|-------------|---------------|
| `short_interest` | Shares currently sold short | FINRA, SFT data |
| `float_shares` | Shares available for public trading | Reference data |
| `volume` | Daily share volume | Exchange data |
| `on_loan` | Shares currently out on loan | Prime broker, DataLend, S3 |
| `lendable_supply` | Shares available to lend | Prime broker, custodian |
| `fee_rate` | Annualized borrow fee (decimal) | DataLend, Markit |

## Features

### `short_interest_ratio`

```text
SIR = short_interest / float_shares
```

What fraction of the freely tradeable supply is currently sold short. A high SIR
indicates crowded positioning. Values above 20% are notable; above 50% are extreme
and often signal either strong conviction or a setup for a short squeeze.

**Lookback**: 0 (uses current observation only)

---

### `days_to_cover`

```text
DTC = short_interest / rolling_avg_daily_volume(volume_window)
```

How many days of average trading volume it would take all short sellers to cover. A
high DTC amplifies squeeze risk because even moderate buying pressure takes a long time
to clear. A declining DTC suggests orderly short reduction.

**Lookback**: `volume_window` (default 20)

---

### `utilization_rate`

```text
utilization = on_loan / lendable_supply
```

The fraction of the available lending inventory currently out on loan. This is the most
direct measure of borrow demand relative to supply. High utilization (>80%) precedes fee
rate increases and can signal upcoming locate restrictions.

**Lookback**: 0

---

### `borrow_fee_rate`

The annualized fee rate charged to borrowers, passed through directly as a signal.

General collateral names trade at 10–25 bps; warm names at 25–100 bps; hard-to-borrow
specials at 1%–30%+. The fee encodes aggregate conviction: every basis point above GC
represents real dollars that bearish investors are paying to maintain their view.

**Lookback**: 0

---

### `short_interest_change`

```text
SI_change = pct_change(short_interest, periods=window)
```

The percentage change in short interest over a trailing window. Rising short interest
indicates growing bearish conviction and additional borrow demand; declining short interest
may signal short covering and near-term price support as forced buyers re-enter.

**Lookback**: `window` (default 5)

## Usage

```python
from qr_haven.features import (
    FeatureRegistry,
    compute_features,
    short_interest_ratio,
    days_to_cover,
    utilization_rate,
    borrow_fee_rate,
    short_interest_change,
    securities_lending_feature_registry,
)

# Individual features with custom parameters
registry = FeatureRegistry([
    short_interest_ratio(),
    days_to_cover(volume_window=20),
    utilization_rate(),
    borrow_fee_rate(),
    short_interest_change(window=5),
])

features = compute_features(prices, registry)
```

Or use the default registry:

```python
registry = FeatureRegistry(securities_lending_feature_registry())
features = compute_features(prices, registry)
```

## Signal Interpretation

| Feature | High value | Low value |
|---------|------------|-----------|
| `short_interest_ratio` | Crowded short; squeeze candidate | Low short interest; less controversy |
| `days_to_cover` | Hard to unwind shorts quickly | Short position can be exited readily |
| `utilization_rate` | Borrow scarce; fees likely rising | Plentiful supply; borrow available |
| `borrow_fee_rate` | Strong conviction from shorts | GC / no meaningful short interest signal |
| `short_interest_change` | Shorts building; growing pessimism | Covering; bullish pressure possible |

## Alpha Research Integration

Securities lending features can be consumed alongside price-based features for multi-factor
IC analysis:

```python
as_of = features[features["timestamp"] <= rebalance_date]
latest = (
    as_of
    .groupby(["symbol", "feature_name"])
    .tail(1)
    .pivot(index="symbol", columns="feature_name", values="value")
)

# latest now has one row per symbol, one column per feature
# Pass to AlphaModel.fit() or directly into the optimizer
```

Point-in-time safety is guaranteed: the feature store never exposes a future row to a
transform, so there is no lookahead bias in any of these features regardless of their
implementation.

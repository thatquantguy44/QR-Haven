# Feature Store v0

The feature store turns canonical, long-form price data into versioned and reproducible feature
observations. Its intentionally small scope establishes feature contracts and temporal correctness
before QR-Haven adds persistence, distributed execution, or an online serving layer.

## Definitions and registry

Each `FeatureDefinition` declares a name and version, required canonical price columns, output
dtype, the number of prior observations it needs, and a transformation callable. A name/version
pair is immutable within a `FeatureRegistry`; changing feature semantics requires a new version.

```python
from qr_haven.features import FeatureRegistry, lagged_return, rolling_momentum

registry = FeatureRegistry([
    lagged_return(lag=1, version="1"),
    rolling_momentum(window=20, version="1"),
])
```

## Point-in-time-safe batch computation

`compute_features` accepts the canonical price frame returned by the data portals. For every
symbol and calculation timestamp, it gives a transformation only the current row and its declared
trailing lookback. A callable therefore cannot accidentally inspect later prices, even if its own
implementation attempts to use the last row it receives.

```python
from qr_haven.features import (
    FeatureRegistry,
    compute_features,
    lagged_return,
    rolling_momentum,
    rolling_volatility,
)

registry = FeatureRegistry([
    lagged_return(),
    rolling_volatility(window=20),
    rolling_momentum(window=20),
])
features = compute_features(prices, registry)
```

The output is sorted long-form data with this stable schema:

```text
timestamp | symbol | feature_name | feature_version | value
```

Warm-up observations remain present with null values. This makes feature availability explicit and
prevents downstream code from silently shortening or shifting a research sample.

## Feeding portfolio optimization

At a rebalance timestamp, select only observations known by that timestamp, filter the desired
feature name/version, and pivot symbols into an optimizer input. For example, trailing momentum can
serve as an expected-return score while historical returns continue to estimate covariance:

```python
as_of = features[features["timestamp"] <= rebalance_timestamp]
latest = as_of[as_of["feature_name"] == "rolling_momentum"].groupby("symbol").tail(1)
expected_returns = latest.set_index("symbol")["value"].dropna()
weights = optimizer.optimize(expected_returns, covariance.loc[expected_returns.index, expected_returns.index])
```

Production workflows should record the exact feature version alongside optimizer results. This
keeps a backtest reproducible after a transformation evolves.

## Path to the Alpha Research Factory

The Alpha Research Factory can consume the same long-form table to assemble point-in-time design
matrices, compare feature versions, calculate information coefficients, and train models without
reimplementing feature calculations. Later storage and online-serving implementations should
preserve this contract and temporal boundary. Regime and machine-learning features can register as
additional definitions while sharing the same validation and leakage-prevention behavior.


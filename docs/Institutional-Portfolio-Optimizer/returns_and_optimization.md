# Returns and Optimization

QR-Haven v0 converts canonical price data into a wide return matrix, then feeds expected returns and
covariance estimates into portfolio optimizers.

## Returns

```python
from qr_haven.data import calculate_returns

returns = calculate_returns(prices, price_column="adjusted_close", method="simple")
```

`returns` is indexed by timestamp with one column per symbol.

## Expected Returns and Covariance

```python
from qr_haven.portfolio import estimate_expected_returns, estimate_return_covariance

expected_returns = estimate_expected_returns(returns)
covariance = estimate_return_covariance(returns)
```

## Equal Weight

```python
from qr_haven.portfolio import EqualWeightOptimizer

weights = EqualWeightOptimizer().optimize(expected_returns, covariance)
```

## Mean Variance

```python
from qr_haven.portfolio import MeanVarianceOptimizer

weights = MeanVarianceOptimizer().optimize(
    expected_returns,
    covariance,
    constraints={"long_only": True, "max_weight": 0.25, "risk_aversion": 2.0},
)
```

The v0 mean-variance optimizer supports bounded long-only allocations without making an external
solver mandatory. A later version can add an optional `cvxpy` backend for richer constraints.


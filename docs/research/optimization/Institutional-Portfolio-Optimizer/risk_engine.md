# Risk Engine

QR-Haven v0 computes single-period portfolio risk from optimizer weights and a wide asset-return
matrix.

```python
from qr_haven.risk import SimpleRiskEngine

metrics = SimpleRiskEngine(var_confidence=0.95).evaluate(weights, returns)
```

Returned metrics include:

- Observations
- Mean return
- Volatility
- Annualized return
- Annualized volatility
- Sharpe ratio
- Max drawdown
- VaR
- CVaR
- Gross exposure
- Net exposure
- Weight concentration
- Largest absolute weight

The expected workflow is:

```text
prices -> returns -> optimizer weights -> risk metrics
```

## Terminal Panel

Risk metrics can be converted into a terminal-ready table payload:

```python
from qr_haven.integrations.market_terminal import risk_metrics_panel

panel = risk_metrics_panel(metrics)
```


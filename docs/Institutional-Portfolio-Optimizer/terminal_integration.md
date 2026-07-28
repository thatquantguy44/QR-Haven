# Terminal Integration

QR-Haven exposes a complete terminal-ready portfolio optimizer report package for the separate
`market_terminal` project.

## Report Bundle

```python
from qr_haven.reporting import build_backtest_report_bundle

bundle = build_backtest_report_bundle(
    result,
    annualization=252,
    return_window=21,
    turnover_window=3,
)
```

The bundle groups:

- Raw `BacktestResult`
- Performance attribution
- Portfolio analytics

## Portfolio Analytics

Portfolio analytics include:

- Cumulative return
- Drawdown
- Rolling volatility
- Rolling Sharpe
- Rolling turnover

## Complete Terminal Panel Package

```python
from qr_haven.integrations.market_terminal import backtest_terminal_panels

panels = backtest_terminal_panels(bundle)
```

The panel package includes:

- `backtest.summary`
- `performance.summary`
- `backtest.equity_curve`
- `backtest.drawdown`
- `backtest.rolling_risk`
- `backtest.weights`
- `backtest.optimizer_diagnostics_history`
- `backtest.constraint_pressure`
- `performance.asset_contribution`
- `performance.turnover_cost`
- `performance.optimizer_diagnostics`

This is the first clean handoff contract for mounting QR-Haven portfolio research inside
`market_terminal`.


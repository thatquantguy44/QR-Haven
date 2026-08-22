# Spec001: Cost-Aware Research Backtest DAG

## Summary

Create a simple DAG workflow in `workflows/cost_aware_research_backtest.py` that turns cleaned
market data into a cost-aware backtest and terminal-ready research output.

This workflow should be the first concrete orchestration layer for QR Haven. It should demonstrate
how the existing models fit together without introducing heavyweight scheduler infrastructure.

## Why This Workflow

QR Haven already has strong reusable components for:

- Price and return preparation
- Securities lending features
- Alpha scoring
- Regime detection
- Portfolio optimization
- Borrow, financing, and market-impact costs
- Walk-forward backtesting
- Risk metrics
- Reporting and terminal panel contracts

The missing layer is a readable workflow that shows how those pieces compose into one institutional
research process.

## Target File

```text
workflows/cost_aware_research_backtest.py
```

## DAG Overview

```text
load_prices
  -> compute_returns
  -> build_securities_lending_features
  -> generate_alpha_scores
  -> detect_market_regime
  -> run_cost_aware_research_pipeline
  -> calculate_risk_metrics
  -> build_reporting_bundle
  -> emit_market_terminal_panels
```

## Node Specifications

| Node | Existing subsystem | Input | Output |
| --- | --- | --- | --- |
| `load_prices` | `qr_haven.data` | CSV path, SQLite config, or in-memory frame | Canonical price panel |
| `compute_returns` | `qr_haven.data.returns` | Price panel | Wide return matrix |
| `build_securities_lending_features` | `qr_haven.features.securities_lending` | Point-in-time securities lending frame | Feature frame |
| `generate_alpha_scores` | `qr_haven.alpha` | Feature frame and optional forward returns | Alpha score panel |
| `detect_market_regime` | `qr_haven.regimes` | Return matrix or derived regime features | Regime labels and probabilities |
| `run_cost_aware_research_pipeline` | `qr_haven.research.ResearchPipeline` | Returns, alpha scores, ADV, optimizer, cost models | `PipelineResult` |
| `calculate_risk_metrics` | `qr_haven.risk` | Portfolio returns and final weights | Risk metric dictionary |
| `build_reporting_bundle` | `qr_haven.reporting` | Backtest-compatible result or adapter output | Analytics/report payload |
| `emit_market_terminal_panels` | `qr_haven.integrations.market_terminal` | Report bundle or panel adapter output | Terminal panel payloads |

## Recommended V0 Design

Use a small local DAG runner instead of a framework:

```python
@dataclass(frozen=True)
class WorkflowContext:
    prices: pd.DataFrame | None = None
    returns: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    alpha_scores: pd.DataFrame | None = None
    regimes: pd.DataFrame | None = None
    pipeline_result: PipelineResult | None = None
    risk_metrics: dict[str, float] | None = None
    terminal_panels: list[TerminalPanel] | None = None
```

Each node should accept a `WorkflowContext` and return a new `WorkflowContext`. This keeps the DAG
easy to test and avoids hidden mutable state.

## Configuration

Create one workflow config dataclass:

```python
@dataclass(frozen=True)
class CostAwareResearchBacktestConfig:
    price_source: str | None = None
    lookback_periods: int = 60
    rebalance_periods: int = 21
    nav: float = 10_000_000.0
    fallback_cost_bps: float = 5.0
    optimizer_name: str = "mean_variance"
    impact_model_name: str = "square_root"
    use_regimes: bool = True
    emit_terminal_panels: bool = True
```

The implementation should prefer explicit defaults over environment variables.

## Default Model Choices

| Concern | V0 default |
| --- | --- |
| Optimizer | `MeanVarianceOptimizer` |
| Fallback optimizer | `EqualWeightOptimizer` |
| Market impact | `SquareRootImpactModel` |
| Borrow schedule | `BorrowCostSchedule.gc_schedule(symbols)` |
| Financing rates | `FinancingRates(debit_rate=0.055, credit_rate=0.04)` |
| Alpha model | `RankAlphaModel` or `CompositeAlphaModel` over securities lending features |
| Regime model | GMM first, HMM optional |
| Risk engine | `SimpleRiskEngine` |

## Inputs

V0 should support:

- In-memory price and feature frames for tests.
- Optional local CSV path for manual runs.
- Optional ADV series for market-impact estimation.
- Optional securities lending feature frame.

V0 should not require:

- Live data feeds.
- Broker connections.
- Secrets.
- External databases.
- Cloud scheduler configuration.

## Outputs

The final workflow result should include:

- Return matrix
- Feature frame
- Alpha scores
- Regime labels or probabilities, if enabled
- `PipelineResult`
- Risk metrics
- Cost attribution summary
- Terminal panel payloads, if enabled

## Error Handling

Each node should raise a clear `ValueError` when required upstream data is missing.

Examples:

- `compute_returns` should fail if `prices` is absent.
- `generate_alpha_scores` should fail if `features` is absent.
- `run_cost_aware_research_pipeline` should fail if `returns` is absent.

## Testing Plan

Add focused tests around:

- Node order and dependency validation.
- Successful end-to-end run on deterministic synthetic data.
- Cost attribution is present in the pipeline result.
- Terminal panel emission returns serializable payloads when enabled.
- Workflow runs without regimes when `use_regimes=False`.

## Future Extensions

- Add `PipelineResult` native terminal panels.
- Add regime-conditioned optimizer constraints.
- Add execution simulator nodes for VWAP, TWAP, POV, and implementation shortfall.
- Add capacity analysis by sweeping NAV and participation rates.
- Add persisted run metadata under `experiments/`.
- Add optional scheduler adapters after the pure Python workflow is stable.

## Acceptance Criteria

The implementation is complete when:

- `workflows/cost_aware_research_backtest.py` exists.
- The workflow can run from a single Python entry point.
- The workflow uses existing QR Haven model classes.
- Tests cover a deterministic end-to-end run.
- The result includes costs, returns, weights, risk metrics, and terminal payloads.
- No external scheduler or network access is required for tests.


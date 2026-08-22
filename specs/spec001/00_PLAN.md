# Spec001 Plan: Cost-Aware Research Backtest DAG

## Objective

Define a first workflow specification for a dependency-light DAG in `workflows/` that connects the
existing QR Haven models into one readable research pipeline.

The target workflow is:

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

## Scope

This spec covers the design for a pure Python DAG runner. It does not require Airflow, Prefect,
Dagster, Celery, or any external orchestration service.

The workflow should use existing source modules wherever possible:

- `qr_haven.data`
- `qr_haven.features`
- `qr_haven.alpha`
- `qr_haven.regimes`
- `qr_haven.portfolio`
- `qr_haven.costs`
- `qr_haven.research`
- `qr_haven.risk`
- `qr_haven.reporting`
- `qr_haven.integrations.market_terminal`

## Deliverables

1. Write the spec document for the proposed workflow.
2. Define the DAG nodes, inputs, outputs, and success criteria.
3. Identify which existing models each node should call.
4. Keep v0 implementation dependency-light and testable.
5. Leave implementation for a follow-up change unless explicitly requested.

## Proposed Files

| File | Purpose |
| --- | --- |
| `specs/spec001/00_PLAN.md` | Planning document and implementation sequence. |
| `specs/spec001/01_SPEC.md` | Formal workflow specification. |
| `workflows/cost_aware_research_backtest.py` | Future implementation target. |
| `tests/test_cost_aware_research_backtest_workflow.py` | Future workflow-level tests. |

## Implementation Sequence

1. Create a small DAG abstraction inside the workflow file.
2. Add typed node functions for each workflow stage.
3. Use deterministic sample or fixture data for tests.
4. Wire the default optimizer and cost models through configuration.
5. Return a structured workflow result with intermediate artifacts.
6. Add serialization helpers for terminal panel payloads.
7. Add tests for node ordering, node outputs, and end-to-end success.

## V0 Constraints

- No network dependency.
- No external scheduler dependency.
- No secrets or broker connections.
- No persisted model artifacts required.
- No intraday execution simulation.
- No live market data pull.

## Acceptance Criteria

- The spec clearly describes each DAG node.
- Each node maps to an existing QR Haven subsystem or a small adapter.
- Inputs and outputs are explicit enough for direct implementation.
- The planned workflow can run locally from a single Python entry point.
- The design supports future export to `market_terminal` panels.


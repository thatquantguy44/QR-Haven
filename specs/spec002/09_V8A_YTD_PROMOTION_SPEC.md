# Spec002: V8A provisional 2026 YTD promotion gate

Protocol version: `volatility-continuous-ytd-v1`.

Status: implementation and frozen candidate complete; untouched extension acquisition pending.

Frozen: 2026-10-07 America/New_York.

Candidate fitted: 2026-10-07 America/New_York.

## Purpose and relationship to V8

Run an earlier, provisional promotion test for the V8-selected continuous-volatility candidate.
This is a new protocol; it does not replace or rewrite the complete-calendar-2026 gate in V8.
Passing V8A permits provisional shadow or limited operational promotion. Final research promotion
still requires the complete V8 evaluation.

The V8 candidate, its development ranking, and the exposed 2024–2025 classifier results are already
known. V8A therefore fixes its population, model, alert threshold, uncertainty method, and gates
before acquiring any 2026 data. No 2024–2025 forward-volatility outcome may fit, tune, calibrate,
select, or score the V8A candidate.

## Immutable market-data extension

Acquire one Tiingo SPY adjusted-OHLC snapshot from 2025-10-01 through 2026-10-02 inclusive and
store it under:

```text
data/raw/market_data/tiingo/spy-2025-10-01-2026-10-02-v1/
```

The overlap from 2025-10-01 through 2025-12-31 must match the frozen
`spy-2005-2025-v1` canonical timestamp, adjusted OHLC, symbol, volume, and frequency fields exactly.
This proves that the appended 2026 returns share the selected model’s adjustment basis. Reject the
extension if any overlap value differs. Overlap rows supply price history only; do not save or use a
2025 forward outcome.

## Frozen model and alert rule

Fit `hist_gradient_boosting_regression` exactly as defined by V8 using the twelve-calendar-year
membership preceding 2024-01-01 from the original verified development artifact. This excludes all
2024–2025 outcomes. Blend 75% model forecast variance with 25% current persistence variance and
take the square root.

The alert threshold is fixed at `0.197901781191757`, the linear 75th percentile of the 748 original
development outcomes with origins in 2021–2023. The threshold is committed here before 2026
acquisition. An actual or forecast value equal to the threshold is normal.

## Frozen evaluation population

Evaluate every XNYS origin from 2026-01-02 through 2026-09-25 inclusive whose five-session outcome
ends no later than 2026-10-02. Features use information through the origin. The source overlap may
provide trailing windows, but only 2026 origins and their five-session 2026 outcomes enter metrics.

Preparation writes evaluation features and sealed continuous outcomes separately without outcome
summaries. Training reads only the original through-2023 development artifact. Before evaluation
opens sealed outcomes, reserve the immutable evaluation directory and durably append an exposure
ledger entry keyed by extension hash and exact sample membership. Another fresh evaluation of the
same population is rejected; repeated commands only verify saved evidence.

## Metrics, uncertainty, and provisional gate

Report candidate and persistence QLIKE, absolute log-volatility error, volatility MAE/RMSE, and
fixed-threshold alert confusion metrics. Alert cost is
`(false positives + 5 * false negatives) / origins`.

Use a paired circular moving-block bootstrap over the row-level difference
`persistence QLIKE - candidate QLIKE`, with 20-origin blocks, 2,000 valid replicates, seed 5413,
and a percentile 95% interval. Both alert classes must be present. V8A passes only if all conditions
hold:

1. candidate mean QLIKE is lower than persistence;
2. candidate absolute log-volatility error is lower than persistence;
3. the bootstrap interval for QLIKE improvement is strictly above zero;
4. candidate high-volatility recall is at least persistence recall;
5. candidate five-to-one alert cost is no greater than persistence cost.

No condition may be dropped, weakened, or reweighted after outcome exposure. A failed gate is a
completed negative result, and the full-year V8 gate remains pending.

## Artifacts

Write generated evidence under:

```text
artifacts/classification/volatility/tiingo-spy-v1/continuous_ytd/
├── dataset-v1/
├── candidate-v1/
└── evaluations/candidate-v1/2026-ytd-v1/
```

Save source/input/code/environment identity, exact population and training membership, fit audit,
model bundle, individual forecasts and row losses, metrics, bootstrap summary and samples, report,
exposure ledger, and hashes. All three stages are immutable and independently verifiable.

## Acceptance checks

- The protocol commit predates the extension snapshot and every 2026 outcome exposure.
- Extension overlap matches the frozen source exactly.
- Only 2026 origins through September 25 enter the evaluation.
- Training membership ends in 2023 and is purged before 2024.
- Saved-model forecasts reproduce evaluation forecasts.
- Ledger persistence occurs before sealed outcomes are opened.
- All five gates use unrounded values and logical AND.
- Artifact hashes detect any mutation and existing IDs cannot be overwritten.

## Implementation status

The immutable candidate bundle is fitted from 3,013 rows ending 2023-12-21 and verifies against the
original development and V8 selection manifests. The prepare, train, evaluate, and independent
verification commands are implemented. Tests cover exact overlap rejection, deterministic QLIKE
bootstrap behavior, ledger-before-outcome ordering, all-gate conjunction, repeat verification, and
hash damage.

No extension snapshot or V8A outcome has been opened. Acquisition of the frozen
2025-10-01–2026-10-02 snapshot is the remaining prerequisite. Preparation currently fails before
creating an artifact because that snapshot is absent. The full repository suite passes 893 tests;
focused Ruff and strict mypy checks pass for the V8A modules.

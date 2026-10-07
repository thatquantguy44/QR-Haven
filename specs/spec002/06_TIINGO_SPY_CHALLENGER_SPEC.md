# Spec002: Frozen Tiingo SPY challenger

Protocol version: `volatility-challenger-tiingo-v1`.

Status: complete; frozen before acquisition and evaluated once; research gate not met.

Frozen: 2026-10-07 America/New_York.

Evaluated: 2026-10-07 America/New_York.

## Purpose

Run the completed V5 challenger design on a separately versioned, dividend-adjusted SPY series
with an untouched 2024–2025 evaluation. The local SPX results through 2020 motivated this study,
so they are background evidence only. No SPX observation participates in SPY fitting, calibration,
selection, threshold estimation, or evaluation.

## Source and price basis

Use the existing `tiingo-spy-v1` profile and an immutable Tiingo End-of-Day snapshot for SPY from
2005-01-01 through 2025-12-31. Returns use `adjClose`. Range features use `adjHigh` and `adjLow`,
and canonical adjusted OHLC fields must therefore share the same split/dividend adjustment basis.
The snapshot manifest records provider hashes, retrieval time, revision policy, calendar, and
license. Raw/provider files remain Git-ignored and may not be redistributed.

The provider supplies a current revised adjusted history rather than a vintage point-in-time
archive. The immutable downloaded bytes define this experiment's data version. This limitation is
reported but does not permit changing the data after results are seen.

## Frozen pipeline

Reuse the V5 definitions without adding, removing, or tuning a feature or candidate:

- original nine volatility features plus the eleven downside, upside, jump, sign, vol-of-vol, and
  Parkinson features defined in `05_FROZEN_CHALLENGER_SPEC.md`;
- purged five-calendar-year training windows and training-side 75th-percentile truth thresholds;
- `hist_gradient_boosting_01`, HAR-RV, and EWMA decays 0.90, 0.94, and 0.97;
- chronological Platt calibration from the immediately preceding three out-of-fold years;
- calibrated-model weights `{0, 0.25, 0.50, 0.75, 1}` against the persistence state;
- prediction cutoff strictly greater than 0.5 and the same deterministic ranking rules.

This is an independent application of the fixed candidate grid. It does not force the SPX-selected
75/25 blend onto SPY; SPY development selects among the already-frozen 25 candidates.

## Development and final fit

Outer development folds are 2013–2023 inclusive. Candidate ranking uses unrounded mean yearly
balanced accuracy, then mean yearly macro F1, base-candidate order, and ensemble-weight order.
Every candidate shares validation dates and fold truth.

After selection, generate final calibration examples for 2021, 2022, and 2023 using each year's
prior five-year history. Fit the chosen base forecast on the purged five-year window immediately
before the first 2024 origin, freeze its 75th-percentile threshold, and serialize the complete
pipeline before opening evaluation outcomes.

## One-time 2024–2025 evaluation

The evaluation population is every eligible 2024–2025 origin whose five-session label is complete
by 2025-12-31. Features are stored separately from sealed outcomes. Before reading outcomes, reserve
the evaluation directory and sync an exposure-ledger entry keyed by source hash and exact sample
membership. Repeated verification may use saved outputs; another fresh evaluation is rejected.

Report the same daily and every-fifth metrics and baselines as V5. Compare the challenger with
persistence using a paired circular moving-block bootstrap with 20-origin blocks, 2,000 valid
replicates, seed 5411, and a percentile 95% interval. The gate passes only when:

1. balanced accuracy exceeds both majority and persistence;
2. high-volatility recall is at least persistence recall;
3. the paired balanced-accuracy improvement interval lies strictly above zero.

No part of the pipeline may change in response to the evaluation. A failed gate completes the
experiment as a negative result.

## Artifact location and execution

Write immutable generated evidence under:

```text
artifacts/classification/volatility/tiingo-spy-v1/challengers/
├── dataset-v1/
├── challenger-v1/
└── evaluations/challenger-v1/2024-2025-v1/
```

The workflow must support the same prepare, train, evaluate, and verify stages as the SPX challenger,
selected by explicit profile. Tests must cover adjusted OHLC alignment, the two-year evaluation
membership, no sealed-outcome access during preparation/training, outer-fold chronology through
2023, final-fit membership, deterministic selection, ledger-before-outcome ordering, and immutable
verification.

## Recorded result

The immutable Tiingo snapshot contains 5,283 verified XNYS sessions from 2005-01-03 through
2025-12-31. Its raw price SHA-256 is
`0add3f3561a06559ebb24ab4a15eaf0da7943850943e645e744caf1c4a852ccf`.

Development selection chose `hist_gradient_boosting_01` with 75% calibrated-model probability and
25% persistence. The single 497-origin evaluation ran from 2024-01-02 through 2025-12-23. The
challenger scored 0.662825 balanced accuracy and 0.442623 high-volatility recall, versus 0.654290
and 0.393443 for persistence. The paired balanced-accuracy improvement was +0.008535; its 95%
moving-block bootstrap interval was [-0.0627, +0.0895]. The first two gates passed, but the interval
crossed zero, so the overall research gate was not met. No post-evaluation tuning was performed.

See the [measured result](../../docs/research/classification/spy_volatility_challenger.md) for the
complete metrics and interpretation.

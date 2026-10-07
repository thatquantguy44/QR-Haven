# SPX Next-Five-Session Volatility Classification: Final Result

Status: local SPX experiment complete; predeclared research gate not met.

Measured: 2026-10-06 America/New_York

## Result

The frozen `hist_gradient_boosting_01` model was evaluated once on 498 SPX forecast origins from
2018-01-02 through 2019-12-23. Each truth label indicates whether realized volatility over the
following five XNYS sessions exceeded the fixed pre-2018 threshold of 0.176685395725789 annualized
volatility. The holdout contained 109 high-volatility observations and 389 normal observations, a
21.89% high-volatility prevalence.

| Model | Accuracy | Balanced accuracy | High precision | High recall | High F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen histogram gradient boosting | 0.744980 | 0.701387 | 0.441558 | 0.623853 | 0.517110 | 0.777151 |
| Persistence | 0.807229 | 0.718120 | 0.559633 | 0.559633 | 0.559633 | 0.818636 |
| Majority / always normal | 0.781124 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 0.500000 |

The selected model correctly classified 371 of 498 observations. Its confusion matrix, with rows
as true `[normal, high]` and columns as predicted `[normal, high]`, was `[[303, 86], [41, 68]]`.
It detected 68 of the 109 high-volatility periods, while persistence detected 61.

The every-fifth-origin secondary check used 100 less-overlapping observations. The model scored
0.756917 balanced accuracy and persistence scored 0.745906. This secondary result was favorable,
but the predeclared research decision uses the daily result and bootstrap gate.

## Research gate

| Requirement | Result | Passed |
| --- | --- | --- |
| Balanced accuracy exceeds majority | 0.701387 versus 0.500000 | Yes |
| Balanced accuracy exceeds persistence | 0.701387 versus 0.718120 | No |
| High-volatility recall is at least persistence | 0.623853 versus 0.559633 | Yes |
| Bootstrap interval lies strictly above zero | [-0.107333, 0.075386] | No |

The point difference in balanced accuracy versus persistence was -0.016733. The paired circular
moving-block bootstrap used 20-session blocks, 2,000 valid replicates, seed 5403, and no discarded
replicates. Because the model did not beat persistence and the interval crossed zero, the overall
research status is **target not met**.

This is still a useful negative result. The model was more sensitive to high-volatility periods,
but it generated 86 false high-volatility predictions compared with persistence's 48. That loss of
normal-regime specificity outweighed the gain in high-regime recall under balanced accuracy.

## Integrity and reproducibility

The evaluation loaded the V3 model without fitting or changing its features, hyperparameters,
class weights, probability cutoff, or threshold. Before reading outcomes, it reserved the immutable
`holdout-v1` directory and recorded the exposure in the profile-level ledger. The manifest binds
the result to the exact V2 dataset manifest, V3 model bytes, evaluation source, and Python package
versions.

Local generated evidence is stored at:

```text
artifacts/classification/volatility/spx-local-v1/evaluations/spx-vol-v2/holdout-v1/
├── manifest.json
├── metrics.json
├── holdout_predictions.csv
├── bootstrap.json
└── report.md
```

Verify it without reopening or recomputing the holdout:

```bash
python -m qr_haven.ml.volatility verify-evaluation \
  --output-dir artifacts/classification/volatility/spx-local-v1/evaluations/spx-vol-v2/holdout-v1
```

The source series is the supplied SPX price index. Its `Adj Close` equals `Close`, so it does not
represent dividend-reinvested total return. That limitation is acceptable for this volatility
classification experiment but prevents interpreting the input as an investable portfolio return.
The raw source's upstream provenance and redistribution terms remain unverified, and generated
data and artifacts remain local.

## Decision

The local SPX Spec002 experiment is closed without promoting this model as superior to the simple
volatility-persistence rule. The exposed 2018–2019 period must not be reused for confirmatory model
selection. Further work can use it only as exploratory evidence or begin a separately frozen
experiment, such as the Tiingo SPY profile with its own untouched holdout.

# SPX Next-Five-Session Volatility Classification: Development Result

Status: V3 development selection complete. The subsequent
[V4 holdout result](spx_volatility_final.md) did not meet the research gate.

Measured: 2026-10-06

## Question and protocol

The experiment predicts whether annualized realized volatility over the next five XNYS sessions
exceeds the 75th percentile estimated from eligible training labels. It uses nine features known
after the current close. The supplied SPX price-index data define the `spx-local-v1` profile.

Run `spx-vol-v2` compared the predeclared 20 learned candidates and three baselines on expanding,
purged calendar-year folds from 2010 through 2017. Every fold estimated its target threshold and
inverse-frequency weights from its training history. Selection used the unweighted mean of yearly
balanced accuracy, then macro F1 and the frozen grid order. The 2018–2019 outcome file was not read.

## Development result

`hist_gradient_boosting_01` ranked first with `learning_rate=0.03`, `max_leaf_nodes=7`, and
`l2_regularization=1.0`.

| Model | Mean yearly balanced accuracy | Mean macro F1 | Mean accuracy | Mean high recall |
| --- | ---: | ---: | ---: | ---: |
| Selected histogram gradient boosting | 0.708885 | 0.626525 | 0.876417 | 0.404289 |
| Persistence | 0.662901 | 0.600305 | 0.886889 | 0.271075 |
| Majority / always normal | 0.562500 | 0.466784 | 0.881937 | 0.000000 |

Across all 2,008 out-of-fold predictions, the selected model's pooled balanced accuracy was
0.807831 and persistence scored 0.731938. These pooled figures are descriptive because folds use
different training-only thresholds; they did not determine selection. The predeclared
every-fifth-origin check produced mean yearly balanced accuracy of 0.748350 for the selected model.

The selected model traded some ordinary accuracy for materially higher high-volatility recall.
That is consistent with inverse-frequency weighting and the strong class imbalance. The 2017 fold
had no high-volatility observations under its fold threshold, so its balanced accuracy equals
normal-class recall and its ROC-AUC and average precision are undefined. It remains in the yearly
mean under the convention recorded before training.

The selected estimator was refit on 3,207 pre-2018 development origins. Its frozen high-volatility
threshold is 0.176685395725789 annualized volatility. Artifact hashes, the resolved grid,
environment versions, source hashes, predictions, and an exact serialized-model replay are stored
under the ignored local run directory.

## Reproduce and inspect

```bash
python -m qr_haven.ml.volatility train --profile spx-local-v1 --run-id spx-vol-v2
python -m qr_haven.ml.volatility verify-run \
  --run-dir artifacts/classification/volatility/spx-local-v1/spx-vol-v2
```

Run IDs are immutable, so choose a new ID when reproducing an existing run. The complete protocol
and model grid are in the [volatility specification](../../../specs/spec002/03_VOLATILITY_SPEC.md).

## What this result supports

The development evidence supported freezing histogram gradient boosting for the local SPX V4
evaluation. It did not establish final out-of-sample performance, statistical significance, a
tradable strategy, or transfer to the separate Tiingo SPY profile. V4 subsequently loaded this
model without refitting and opened the 498 sealed labels once. Its failed research gate is retained
as the final result; the grid and threshold were not changed after holdout exposure.

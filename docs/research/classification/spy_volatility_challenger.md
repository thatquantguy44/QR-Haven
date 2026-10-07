# Tiingo SPY five-year volatility challenger: frozen 2024–2025 result

Status: complete; predeclared research gate not met.

Measured: 2026-10-07 America/New_York.

## Data and protocol

The protocol was frozen before downloading Tiingo data or inspecting a 2024–2025 outcome. The
immutable snapshot contains 5,283 adjusted SPY observations from 2005-01-03 through 2025-12-31 and
matches the XNYS session calendar exactly. Returns use `adjClose`; range-volatility features use
`adjHigh` and `adjLow`. The raw price SHA-256 is
`0add3f3561a06559ebb24ab4a15eaf0da7943850943e645e744caf1c4a852ccf`. Provider files remain local
and Git-ignored under the Tiingo license.

This independent replication applied the already-frozen 25-candidate grid: the histogram model,
HAR-RV, and three EWMA decays, each combined with persistence at five fixed weights. Every fitted
model used a purged five-calendar-year window. Chronological Platt calibration used only earlier
out-of-fold predictions. Selection used yearly folds from 2013 through 2023; calibration for the
final fit used 2021–2023. The 2024–2025 outcomes remained sealed until the pipeline was serialized
and the exposure ledger was written.

## Development selection

Selection chose `hist_gradient_boosting_01` with 75% calibrated-model probability and 25%
persistence. Its mean yearly development balanced accuracy was **0.6919**. Pure persistence under
the same outer folds scored **0.6617**. Development results chose the pipeline; they are not an
unbiased performance estimate.

## Frozen evaluation

The evaluation contains 497 eligible origins from 2024-01-02 through 2025-12-23: 436 normal and 61
high-volatility outcomes. The frozen threshold was **0.1981748589512833**.

| Model | Accuracy | Balanced accuracy | High precision | High recall | High F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen challenger | 0.828974 | 0.662825 | 0.346154 | 0.442623 | 0.388489 | 0.779177 |
| Persistence | 0.851107 | 0.654290 | 0.393443 | 0.393443 | 0.393443 | 0.709693 |
| Training majority / always normal | 0.877264 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 0.500000 |

The challenger confusion matrix is `[[385, 51], [34, 27]]`; persistence is
`[[399, 37], [37, 24]]`. Relative to persistence, the challenger found three additional high-
volatility outcomes and produced fourteen additional false alarms. This raised balanced accuracy
while lowering ordinary accuracy and high-class F1.

The every-fifth-origin diagnostic used 100 less-overlapping forecasts. Balanced accuracy was
0.717507 for the challenger and 0.690539 for persistence. High-volatility recall was 0.538462 for
the challenger and 0.461538 for persistence.

## Research gate and interpretation

| Requirement | Result | Passed |
| --- | --- | --- |
| Balanced accuracy above majority and persistence | 0.662825 versus 0.500000 and 0.654290 | Yes |
| High-volatility recall at least persistence | 0.442623 versus 0.393443 | Yes |
| Paired improvement interval strictly above zero | [-0.0627, +0.0895] | No |

The paired balanced-accuracy difference was +0.008535. The circular moving-block bootstrap used
20-origin blocks, 2,000 valid replicates, and seed 5411. Its interval crosses zero, so the observed
gain is too uncertain to support promotion of the challenger.

The 87.73% majority accuracy reflects class imbalance and is not the decision metric. It predicts
every observation as normal and has zero high-volatility recall. Balanced accuracy gives equal
weight to normal and high-volatility recall, making the challenger-versus-persistence comparison
more informative for this objective.

## Integrity and outputs

Generated evidence is stored locally under:

```text
artifacts/classification/volatility/tiingo-spy-v1/challengers/
├── dataset-v1/
├── challenger-v1/
└── evaluations/challenger-v1/2024-2025-v1/
```

The evaluation directory contains individual predictions, daily and every-fifth metrics, 2,000
bootstrap differences, a report, and a hash-bound manifest. The original generated report contains
one inherited sentence describing the period as a 2020 evaluation; its dates, metrics, manifest,
and predictions correctly identify 2024–2025. The immutable evidence was left unchanged, and the
report generator was corrected for future profiles.

The [market-data guide](../../api/volatility_market_data.md#continue-on-untouched-tiingo-spy-data)
contains verification commands. The
[frozen specification](../../../specs/spec002/06_TIINGO_SPY_CHALLENGER_SPEC.md) contains the complete
methodology. The one-time outcomes are now exposed and cannot serve as a fresh holdout for another
tuned variant.

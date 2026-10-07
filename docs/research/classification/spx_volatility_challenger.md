# SPX five-year volatility challenger: frozen 2020 result

Status: complete; predeclared research gate not met.

Measured: 2026-10-07 America/New_York.

## What changed

The challenger implemented the six requested improvements under a protocol committed before model
comparison or 2020 outcome access:

1. every fitted model uses a purged five-calendar-year training window;
2. candidates include HAR-RV, EWMA decays 0.90/0.94/0.97, and the existing histogram model;
3. eleven point-in-time features add downside/upside volatility, maximum absolute returns,
   negative-return share, volatility-of-volatility, and Parkinson range volatility;
4. each outer score is Platt-calibrated using only earlier out-of-fold predictions;
5. five fixed model/persistence weights are selected on nested 2013–2017 development evidence;
6. the selected pipeline is evaluated once on the previously quarantined 2020 observations.

The source remains the supplied SPX price index, not a dividend-reinvested total-return series.
This does not prevent volatility forecasting, but results do not represent a portfolio return.

## Development selection

The cross-product contained 25 candidates. Candidate selection used five outer years, 2013–2017.
For each outer year, the Platt calibrator used raw out-of-fold forecasts from the preceding three
years, and every fitted forecast used its own preceding five-year history.

Selection chose `hist_gradient_boosting_01` with 75% calibrated model probability and 25%
persistence state. Its mean yearly balanced accuracy was **0.668734**, versus **0.667432** for pure
persistence under the same outer folds. The 0.13 percentage-point difference was small. After
selection, the final histogram model trained on 1,253 purged 2015–2019 origins and froze a
high-volatility threshold of **0.14056353864178892**.

## Frozen 2020 evaluation

The evaluation contains 209 eligible origins from 2020-01-02 through 2020-10-28. It contains 144
high-volatility and 65 normal origins, so high volatility is the majority state in this unusual
period. The training-majority and always-normal baselines both predict normal because their rule is
fixed from the training population.

| Model | Accuracy | Balanced accuracy | High precision | High recall | High F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen challenger | 0.688995 | 0.770085 | 0.987654 | 0.555556 | 0.711111 | 0.859081 |
| Persistence | 0.765550 | 0.737019 | 0.841727 | 0.812500 | 0.826855 | 0.820513 |
| Training majority / always normal | 0.311005 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 0.500000 |

The challenger confusion matrix is `[[64, 1], [64, 80]]`; persistence is
`[[43, 22], [27, 117]]`. The challenger nearly eliminated false high-volatility alarms, but missed
64 high-volatility origins. Persistence generated 22 false alarms and missed 27 high origins.

The every-fifth-origin diagnostic used 42 less-overlapping forecasts. Balanced accuracy was
0.793103 for the challenger and 0.759947 for persistence. This supports the positive point estimate
but is a secondary, smaller comparison.

## Research gate and interpretation

| Requirement | Result | Passed |
| --- | --- | --- |
| Balanced accuracy above majority and persistence | 0.770085 versus 0.500000 and 0.737019 | Yes |
| High-volatility recall at least persistence | 0.555556 versus 0.812500 | No |
| Paired improvement interval strictly above zero | [-0.094833, +0.212605] | No |

The paired balanced-accuracy difference was +0.033066. The circular moving-block bootstrap used
20-origin blocks, 2,000 valid replicates, seed 5410, and no discarded replicates. Its wide interval
crosses zero. The challenger therefore cannot be promoted as a reliable improvement.

Calibration and the persistence blend changed the error tradeoff substantially: specificity rose
to 98.46%, while sensitivity fell to 55.56%. That behavior is unattractive for the predeclared
goal because missing a high-volatility period was explicitly constrained. It also explains why
plain accuracy is lower even though balanced accuracy is higher: 2020 contains many more high than
normal outcomes.

## Integrity and outputs

The protocol was committed as `b92130d` before feature construction. The implementation was
committed as `9a0538b` before dataset preparation, training, or outcome exposure. The training run
records that 2018–2019 were excluded from selection and included only as historical observations in
the final fit. Before the sealed outcomes were read, the evaluation directory was reserved and one
entry was synced to the profile-level exposure ledger.

Generated evidence is stored locally under:

```text
artifacts/classification/volatility/spx-local-v1/challengers/
├── dataset-v1/
├── challenger-v1/
└── evaluations/challenger-v1/2020-v1/
```

The evaluation includes individual predictions, daily and every-fifth metrics, a bootstrap summary,
all 2,000 bootstrap differences, and a hash-bound report. Model and original V3/V4 hashes were
unchanged after evaluation. The [market-data guide](../../api/volatility_market_data.md#frozen-five-year-challenger)
contains verification commands, and the
[frozen specification](../../../specs/spec002/05_FROZEN_CHALLENGER_SPEC.md) contains the complete
methodology.

Verification completed with **878 repository tests passed**. Focused Ruff and volatility mypy
checks pass. Repository-wide Ruff retains 184 existing findings, and repository-wide mypy retains
65 errors in 17 existing files; none are in the challenger modules.

This closes the local 2020 experiment as a useful negative result. The next confirmatory test needs
another untouched period or separately versioned data source. The exposed 2018–2020 outcomes cannot
be reused as fresh evidence for another tuned variant.

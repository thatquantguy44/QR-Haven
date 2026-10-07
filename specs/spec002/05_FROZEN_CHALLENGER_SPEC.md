# Spec002: Frozen five-year volatility challenger

Protocol version: `volatility-challenger-v1`.

Status: completed; the frozen 2020 research gate was not met.

Frozen: 2026-10-06 America/New_York.

## Purpose and evidence boundary

This phase turns the post-V4 exploratory findings into one separately frozen experiment. It uses
five-year rolling training, volatility-specific features, HAR-RV, EWMA, the existing histogram
gradient boosting configuration, chronological score calibration, and a model-plus-persistence
ensemble. Candidate selection uses development evidence ending in 2017. After selection and final
fitting are complete, the chosen pipeline is evaluated once on the quarantined 2020 SPX origins.

The 2018–2019 V4 holdout is already exposed. It is not used to rank candidates or report new
selection scores. Once the pipeline is selected, 2018–2019 outcomes may enter the final 2015–2019
training and calibration history because they precede every 2020 forecast. This distinction is
recorded in all artifacts. No 2020 outcome may be read before the chosen pipeline is frozen and an
exposure-ledger entry is durable.

The source remains the immutable local SPX price-index snapshot ending 2020-11-04. It is not a
dividend-reinvested total-return series. This experiment forecasts volatility, not investment
returns. No download is required.

## Point-in-time observations and new features

Retain the original nine V2 features and five-session forward realized-volatility outcome. Add
the following features, calculated only from information available through the forecast origin:

1. `downside_vol_5`, `downside_vol_20`, `downside_vol_60`: annualized square root of mean squared
   negative log returns, with positive returns contributing zero;
2. `upside_vol_5`, `upside_vol_20`: the corresponding positive-return measures;
3. `max_abs_return_5`, `max_abs_return_20`: maximum absolute log return in the trailing window;
4. `negative_return_share_20`: fraction of the last 20 log returns below zero;
5. `vol_of_vol_20`: annualized sample standard deviation of the last 20 absolute log returns;
6. `parkinson_vol_5`, `parkinson_vol_20`: annualized Parkinson range volatility using the local
   daily high and low, `sqrt(252 * mean(log(high / low)^2) / (4 * log(2)))`.

All rolling windows include the current session and exclude future sessions. High and low must be
finite, positive, and satisfy `high >= low`. Feature construction writes exact window endpoints
and separates 2020 features from sealed 2020 outcomes. Development loaders must not open the
sealed file.

## Five-year training and truth

For any validation year `Y`, training origins start on or after the first validation origin minus
five calendar years and must have `label_end < first_validation_origin`. The training-side 75th
percentile of continuous `forward_vol_5`, using linear interpolation, is the fold truth threshold.
The threshold is applied unchanged to that fold's validation outcomes. Equality is normal.

The outer selection folds are calendar years 2013–2017. All candidates use identical outer dates,
training membership, and truth. The final 2020 pipeline trains on eligible origins from 2015–2019,
with boundary-crossing labels purged, and freezes their 75th-percentile threshold.

## Candidate forecasts

The base candidate order is fixed:

1. `hist_gradient_boosting_01`: the existing V3 settings (`learning_rate=0.03`,
   `max_leaf_nodes=7`, `l2_regularization=1`, `max_iter=300`, no early stopping, seed 5402), fitted
   with inverse-frequency class weights and all original plus new features;
2. `har_rv`: ordinary least squares with an intercept, forecasting
   `log(forward_vol_5^2 + 1e-12)` from `log(trailing_vol_5^2 + 1e-12)`,
   `log(trailing_vol_20^2 + 1e-12)`, and `log(trailing_vol_60^2 + 1e-12)`; transform forecasts
   back to annualized volatility and clip only for finite numerical safety;
3. `ewma_090`, `ewma_094`, `ewma_097`: causal variance recursions with the stated decay. Initialize
   from the first available `trailing_vol_60^2 / 252`, update after each observed return, and emit
   annualized volatility. State can advance through observed validation returns but never uses a
   forward outcome.

Histogram raw scores are native class-one scores. HAR-RV and EWMA raw scores are annualized
volatility forecasts. No candidate hyperparameter is changed after comparison begins.

## Nested chronological calibration and ensemble

Each outer fold obtains calibration examples from the three immediately preceding calendar years.
For each calibration year, refit the base candidate on the five calendar years preceding that
year, with the same label purge and a training-only threshold, then predict that year. Concatenate
these genuinely out-of-fold raw scores in chronological order.

Fit Platt calibration as a one-feature pipeline of `StandardScaler` and unweighted
`LogisticRegression(C=1, solver="lbfgs", max_iter=2000)`. Both classes must occur in calibration
truth. Fit the outer base model on the outer five-year training side, apply the calibrator to its
raw scores, and never refit calibration on the outer validation year.

Combine the calibrated class-one probability `p_model` with the persistence state
`p_persist = 1[trailing_vol_5 > outer_threshold]`:

```text
p_ensemble = w * p_model + (1 - w) * p_persist
```

The fixed model weights are `{0, 0.25, 0.50, 0.75, 1}` in ascending order. Predict high exactly
when `p_ensemble > 0.5`; equality is normal. The cross-product contains 25 candidates.

## Development selection and final fit

Evaluate every fixed candidate on every 2013–2017 outer fold. Rank candidates using unrounded
mean yearly balanced accuracy descending, then mean yearly macro F1 descending, then base-candidate
order, then ensemble-weight order. Report accuracy, precision/recall/F1 by class, confusion counts,
ROC-AUC, average precision, class supports, daily results, and every-fifth-origin diagnostics.
Pooled metrics are descriptive and do not select the candidate.

After selection, save the complete chosen pipeline before opening 2020 outcomes. For its final
calibrator, generate out-of-fold raw scores for 2017, 2018, and 2019 using only each year's prior
five-year history. Then fit the chosen base candidate on purged 2015–2019 observations and bind the
model, calibrator, ensemble weight, feature order, threshold, memberships, source hash, protocol,
code identity, and environment versions in an immutable training run.

## Single frozen 2020 evaluation

The evaluation population is every eligible 2020 origin in the source snapshot whose five-session
label is complete by 2020-11-04. Before reading its sealed outcome file, reserve an immutable
evaluation ID and append a durable profile-level ledger entry keyed by source hash and sorted sample
IDs. Repeated verification may read saved evaluation outputs; a second fresh evaluation of the same
population is rejected.

Compare the chosen ensemble with majority, always-normal, and persistence baselines. The primary
metric is daily balanced accuracy. Also report accuracy, macro F1, high-class precision/recall/F1,
ROC-AUC, average precision, confusion matrices, prevalence, every-fifth-origin metrics, and a paired
circular moving-block bootstrap versus persistence using 20-origin blocks, 2,000 valid replicates,
seed 5410, percentile 95% interval, and single-class replicate rejection.

The research gate passes only if all are true:

1. chosen-ensemble balanced accuracy is above majority and persistence;
2. high-volatility recall is at least persistence recall;
3. the paired bootstrap interval for balanced-accuracy improvement over persistence lies strictly
   above zero.

Missing the gate is a valid completed negative result. No candidate, threshold, calibration, feature,
or ensemble weight may be changed in response to 2020 outcomes.

## Immutable artifacts and verification

Write local generated artifacts beneath:

```text
artifacts/classification/volatility/spx-local-v1/challengers/
├── dataset-v1/
├── challenger-v1/
└── evaluations/challenger-v1/2020-v1/
```

The dataset stores development observations, evaluation features, sealed evaluation outcomes,
membership, feature-window audit, configuration, hashes, and manifest. The training run stores
fold metrics and predictions, calibration predictions, ranking, exact fit memberships, chosen
pipeline, replay/configuration evidence, report, and manifest. The evaluation stores predictions,
metrics, bootstrap samples/summary, report, and manifest. Every stage refuses overwrite, verifies
hashes, records failures, and leaves the original V2–V4 and exploratory artifacts unchanged.

Tests must demonstrate point-in-time features, rolling/purge membership, causal EWMA state, HAR-RV
training isolation, out-of-fold calibration, identical outer truth, deterministic ranking, ensemble
arithmetic, sealed-outcome isolation, model persistence/replay, exposure-ledger ordering, immutable
repeat behavior, corruption detection, and no changes to the existing V3/V4 evidence.

## Measured result added after the frozen evaluation

The implementation preserved the protocol above. Nested 2013–2017 selection chose
`hist_gradient_boosting_01` with a 0.75 calibrated-model weight and 0.25 persistence weight. Its
mean yearly development balanced accuracy was 0.668734, compared with 0.667432 for pure
persistence under this nested design. The final threshold from the purged 2015–2019 training
population was 0.14056353864178892.

The single 2020 evaluation opened 209 sealed outcomes dated 2020-01-02 through 2020-10-28. The
challenger scored 0.770085 balanced accuracy versus 0.737019 for persistence. Its high-volatility
recall was 0.555556 versus 0.812500 for persistence. The paired balanced-accuracy improvement was
+0.033066, with a 95% circular-block-bootstrap interval of [-0.094833, +0.212605]. It therefore
failed the recall and strictly-positive-interval gates. The overall status is `target_not_met`;
no post-evaluation changes were made to the selected pipeline or generated evidence.

See the [measured challenger report](../../docs/research/classification/spx_volatility_challenger.md)
for interpretation, commands, and artifact locations.

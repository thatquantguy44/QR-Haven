# Spec002: V8 continuous-volatility promotion candidate

Protocol version: `volatility-continuous-v1`.

Status: frozen before implementation or development comparison.

Frozen: 2026-10-07 America/New_York.

## Purpose and evidence boundary

Replace the unstable binary fitting objective with a direct forecast of annualized realized
volatility over the next five sessions. Select a forecast using Tiingo SPY observations through
2023 only, then reserve calendar 2026 for a separately acquired, one-time prospective promotion
test. The exposed 2024–2025 outcomes may not be read, fitted, diagnosed, or scored by V8.

V8 has two stages. This specification implements and runs development selection now. Final
evaluation remains pending until the complete 2026 extension is available and must use the frozen
selection and gates below without retuning.

## Development population and target

Use the verified `tiingo-spy-v1` challenger development observations and existing 20 point-in-time
features. The continuous target is the existing positive annualized `forward_vol_5`. Outer folds
are calendar years 2013–2023. All fitted models use observations from the preceding twelve calendar
years, limited by source availability, with `label_end < validation_start`.

The alert diagnostic uses the V7 point-in-time target: forward volatility above the linear 75th
percentile of at least 500 outcomes from the preceding three calendar years whose label ends before
the origin. Every forecast is assessed against identical continuous outcomes, adaptive thresholds,
and alert truth.

## Forecasts and blends

Compare these base forecasts in declared order:

1. `hist_gradient_boosting_regression`: histogram gradient boosting on log variance, with the V5
   learning rate, leaf count, regularization, iteration count, seed, and no early stopping;
2. `har_rv`: linear regression of log forward variance on log trailing 5-, 20-, and 60-session
   variances;
3. `ewma_090`;
4. `ewma_094`;
5. `ewma_097`.

Floor variances at `1e-12` before logarithms and inverse transforms. Fit the histogram and HAR
models with uniform weights and one thread. EWMA follows the causal recursion already frozen for
V5 and uses no future return.

For each base forecast, blend forecast variance with persistence variance using model weights
`{0.25, 0.50, 0.75, 1.00}`. Take the square root to return to annualized volatility. Add pure
persistence, `trailing_vol_5`, as a separate baseline. There are 21 ranked candidates.

## Development metrics and selection

For actual variance `a` and forecast variance `f`, define row QLIKE as
`a / f - log(a / f) - 1`. Also report mean absolute log-volatility error, volatility MAE, volatility
RMSE, and alert metrics. Daily metrics are primary; every-fifth-origin metrics are diagnostic.

Rank candidates by unrounded mean yearly daily QLIKE ascending, then mean yearly absolute
log-volatility error ascending, base order, and blend-weight order. Persistence participates in the
ranking. If persistence ranks first, V8 selects no fitted replacement.

The alert rule predicts high when forecast volatility is strictly greater than the point-in-time
adaptive threshold. Report false alarms, misses, high recall, balanced accuracy, and asymmetric
alert cost `(false positives + 5 * false negatives) / origins`. The five-to-one miss cost is a
predeclared diagnostic and does not override continuous-forecast selection.

## Development outputs

Write an immutable run to:

```text
artifacts/classification/volatility/tiingo-spy-v1/continuous_experiments/v8-continuous-v1/
```

Save configuration before fitting, adaptive thresholds, exact training membership, fit audits,
individual forecasts, fold and pooled metrics, ranking, selection, report, environment/code/input
identity, and hashes. No final model is fitted in development. Failure reserves the run ID.

## Frozen 2026 promotion evaluation

After 2026 is complete, acquire an immutable Tiingo SPY extension containing all 2026 XNYS
sessions. Combine it only with the frozen history needed for feature construction. Eligible test
origins are 2026 sessions whose five-session outcome is complete within the extension. Fit the
selected pipeline using data available before the first test origin. Do not use 2024–2025 outcomes.

If a fitted candidate beats persistence in development, promotion requires all of these on the
single 2026 evaluation:

1. mean QLIKE is lower than persistence;
2. mean absolute log-volatility error is lower than persistence;
3. a paired 20-origin circular moving-block bootstrap with 2,000 valid replicates and seed 5412 has
   a 95% interval strictly above zero for `persistence QLIKE - candidate QLIKE`;
4. adaptive-alert high recall is at least persistence recall;
5. five-to-one alert cost is no greater than persistence cost.

If persistence wins development selection, no fitted candidate advances and the promotion test is
not run. Shadow publication of forecasts remains allowed, but research promotion is not.

## Acceptance checks

- V8 loads only development observations through 2023.
- Training labels end before each validation boundary and obey the twelve-year lower bound.
- EWMA forecasts at an origin are invariant to later returns.
- All forecasts are finite and strictly positive.
- Every candidate forecasts identical dates and outcomes.
- QLIKE and secondary losses match their row-level definitions.
- Adaptive alert thresholds are causal and shared by every candidate.
- Existing runs cannot be overwritten and every saved output hash verifies.

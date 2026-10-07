# Spec002: V7 adaptive-threshold training-history experiment

Protocol version: `volatility-history-adaptation-v1`.

Status: complete; development selection favored adaptive persistence.

Frozen: 2026-10-07 America/New_York.

Run: 2026-10-07 America/New_York.

## Purpose and evidence boundary

Determine whether more training history improves the Tiingo SPY histogram classifier when “high
volatility” is defined relative to the recent market regime. This is a development-only selection
experiment. It reads only the verified `tiingo-spy-v1` challenger development observations through
2023. It must not open the sealed-outcome file or load the exposed 2024–2025 evaluation outputs.

The 2024–2025 result motivated the adaptive target, so that period is not fresh evidence and may
not be used to rank, tune, fit, or evaluate a V7 candidate. V7 produces a recommended design for a
separately frozen future evaluation; it does not fit a deployable final model.

## Point-in-time adaptive target

For every origin, calculate a threshold from `forward_vol_5` values whose `label_end` is strictly
earlier than the origin and whose `as_of` is at least the origin minus three calendar years. The
threshold is the linear 75th percentile. Require at least 500 eligible historical outcomes and both
target classes in every fitted and calibration sample. Rows without 500 prior outcomes are excluded
from all V7 membership.

The target at origin `t` is high exactly when `forward_vol_5_t > threshold_t`; equality is normal.
The threshold therefore changes through time but uses only information available before the
origin. All candidates share the same per-origin thresholds and truth labels. Persistence is high
exactly when `trailing_vol_5_t > threshold_t`.

## Fixed model and training-history designs

Use only the existing `hist_gradient_boosting_01` estimator and the existing 20 challenger
features. Compare these five designs in declared order:

1. `rolling_5y`: preceding five calendar years;
2. `rolling_8y`: preceding eight calendar years;
3. `rolling_12y`: preceding twelve calendar years;
4. `expanding`: every eligible prior observation;
5. `expanding_decay_5y`: expanding membership with a five-calendar-year exponential half-life.

Every membership ends before the first origin in its validation year and requires
`label_end < validation_start`. A rolling lower bound is inclusive. Ordinary designs use the
existing inverse-frequency class weights normalized to mean one. The decayed design multiplies
those class weights by `0.5 ** (age_days / (365.2425 * 5))` as of the validation boundary and
renormalizes the result to mean one.

## Calibration, ensemble, and selection

Outer folds are calendar years 2013–2023. For every history design and outer year, fit a Platt
calibrator on that design’s raw out-of-fold predictions from the immediately preceding three
calendar years. Calibration targets use their own point-in-time adaptive thresholds.

Combine calibrated model probability with the point-in-time persistence state using model weights
`{0, 0.25, 0.50, 0.75, 1}`. Predict high only when the combined score is strictly greater than
0.5. Rank the 25 candidates by unrounded mean yearly daily balanced accuracy, then mean yearly
macro F1, history-design order, and ensemble-weight order. Daily metrics are primary; every-fifth
origin metrics are diagnostic.

## Outputs and interpretation

Write an immutable run to:

```text
artifacts/classification/volatility/tiingo-spy-v1/history_experiments/v7-history-v1/
```

Save the protocol configuration before fitting. Save adaptive thresholds, exact training
membership and weights, fold audits, calibration rows, predictions, fold metrics, rankings,
pooled metrics, selection, a Markdown report, input/code/environment identity, and hashes for all
artifacts. Failure reserves the run ID.

The report must compare every training-history design with pure persistence under the identical
adaptive truth, quantify class prevalence by year, and state that all scores are development
selection evidence. No bootstrap significance test or success gate applies. A future evaluation
requires a new specification and outcomes that were untouched when that specification was frozen.

## Acceptance checks

- Each adaptive threshold uses at least 500 rows, all with `label_end < as_of` and within the
  trailing three-calendar-year interval.
- Training membership is purged and follows its frozen rolling or expanding rule.
- Exponential and class weights are finite, positive, deterministic, and normalized to mean one.
- Every candidate predicts the same outer-fold dates against identical truth labels.
- Calibration years strictly precede their outer year.
- The workflow never opens 2024–2025 outcomes or evaluation artifacts.
- Repeated verification checks every saved hash, and an existing run cannot be overwritten.

## Recorded result

The workflow created 4,211 point-in-time eligible rows and compared all 25 candidates across the
2013–2023 folds without opening evaluation outcomes. Pure adaptive persistence ranked first with
mean yearly balanced accuracy 0.716707 and macro F1 0.625748. Because its model weight is zero,
the displayed `rolling_5y` history is only the deterministic tie-break label; no fitted model or
training-history rule is active in the selected predictions.

Among decision-changing 75% model / 25% persistence blends, expanding history ranked highest at
0.7138 balanced accuracy, only 0.0002 above the five-year blend. Among model-only candidates,
12-year rolling history was best at 0.694039, versus 0.686054 for five years. More observations
therefore helped the standalone model modestly but did not beat adaptive persistence.

The outer-fold population contained 693 high outcomes among 2,763 origins, or 25.08% overall.
Yearly prevalence still ranged from zero in 2017 and 2023 to 53.78% in 2022. The adaptive target
fixed aggregate prevalence drift but did not make yearly regimes stable. V7 does not justify a new
final model or another evaluation. See the
[measured result](../../docs/research/classification/spy_volatility_history_v7.md).

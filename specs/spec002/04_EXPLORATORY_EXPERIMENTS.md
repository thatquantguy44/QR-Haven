# Spec002: Four exploratory volatility experiments

Protocol version: `volatility-experiments-v1`, defined before running this comparison.

The V4 2018–2019 holdout has already been exposed. These experiments are motivated by that
result and are exploratory. Model comparisons reuse the original development folds only.
Saved V4 predictions may be diagnosed, but are never used to fit or rank new candidates.
The V3 model, V4 evidence, exposure ledger, and quarantined 2020 observations are preserved.

## 1. Error diagnosis

Inspect the saved selected-model and persistence predictions on development folds and the
completed V4 evaluation. Save every observation with its outcome (TN, FP, FN, TP), by-year and
by-regime counts, error rates with denominators, and contiguous runs of false alarms/misses.
Runs count overlapping forecast origins, not independent volatility events.

Regime descriptors use only current trailing volatility versus the row's training threshold:
`calm` (5- and 20-session volatility at/below threshold), `rising` (5-session above, 20-session
at/below), `cooling` (5-session at/below, 20-session above), and `elevated` (both above).
These are deterministic descriptors, not independently verified market regimes. No causal claim
is made from error clustering.

## 2. Class weights and decision cutoffs

Keep the V3-selected model family and hyperparameters. For each fold use sample weights
proportional to `class_count ** (-power)`, normalized to mean one, with power in `{0, 0.5, 1}`.
Zero means uniform weights, one reproduces V3 inverse-frequency weights. Reuse each fitted score
for cutoffs `{0.5, 0.6, 0.7}`; predict high exactly when its class-1 score exceeds the cutoff.
The power-one/cutoff-0.5 case is the V3 control. These are native weighted scores, not calibrated
probability estimates. All nine variants use expanding training windows and the same truth labels.

## 3. Simpler volatility forecasts

Compare three causal exponentially weighted variance forecasts with decay `{0.90, 0.94, 0.97}`.
At the first development origin, initialize variance to `trailing_vol_60 ** 2 / 252`. Subsequent
origins update `variance_t = decay * variance_(t-1) + (1-decay) * log_return_1(t) ** 2`.
Return `sqrt(252 * variance_t)` as the annualized next-five-session RMS forecast, assuming constant
conditional variance over that horizon. Sequential updates may use observed validation returns,
but never forward outcomes. The state is not restarted at fold boundaries. Initialization and all
updates use the development artifact only; no raw-data download or holdout input is needed.

Also fit ordinary least-squares regression with an intercept using `trailing_vol_5`,
`trailing_vol_20`, and `trailing_vol_60` to predict continuous `forward_vol_5`, training side only.
Clip negative forecasts to zero. Both forecast types classify high iff predicted volatility
exceeds the original fold threshold. Scores retain volatility units and are not probabilities.

## 4. Rolling training windows

Compare 3- and 5-calendar-year histories against the expanding V3 control. The lower bound is
the first validation origin minus the specified years (inclusive). Intersect with original purged
training membership and require `label_end < first_validation_origin`. Keep the selected family,
hyperparameters, inverse-frequency weighting, and cutoff 0.5. Recompute weights on the shorter
training side. Keep the original expanding-history truth threshold in each fold; this isolates
training-window length and permits an identical-target comparison. Feature lookbacks may precede
the rolling training start because they were already available at each observation's origin.

## Reporting and artifacts

There are 15 forecast variants (including the V3 control) and three reference baselines. Compare
daily and every-fifth-origin metrics on identical dates; select no final fitted replacement model.
Rank within each experiment by unrounded mean yearly balanced accuracy, mean macro F1, then
declared variant order. Also provide an overall exploratory ranking and pooled descriptive metrics.
Retain the V3 single-class-year convention and report class support. Comparisons used for ranking
are selection evidence and do not establish an unbiased performance improvement.

Write an immutable run to
`artifacts/classification/volatility/<profile>/experiments/<model-run>/<experiment-id>/` with
`config.json` saved before fitting, an input/code/environment manifest, fold and pooled metrics,
rankings, individual development predictions, fit/window audits, exact training membership,
diagnostic tables, and `report.md`. Failure reserves the ID with state `failed`.
Copies of raw prices and new holdout forecasts are not outputs of this workflow.

After reviewing these experiments, any proposed replacement needs a separately frozen protocol
and an untouched test period. That future evaluation is outside this experiment's scope.

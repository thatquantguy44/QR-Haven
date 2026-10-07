# Follow-on: Next-Period High-Volatility Classification

Status: V1–V4 complete for the local SPX profile; the final research gate was not met. See the
[full implementation specification](03_VOLATILITY_SPEC.md),
[development result](../../docs/research/classification/spx_volatility_development.md), and
[final holdout result](../../docs/research/classification/spx_volatility_final.md).

Created: 2026-10-06

Dependency: close B1–B5 in the [implementation plan](00_PLAN.md), including the actual benchmark
outcome and limitations from the [banknote specification](01_SPEC.md).

## Objective

Predict whether volatility over the **next five trading sessions** will be high or normal, using
only information available after the current session's close. Five sessions is the explicit v0
meaning of "next period"; changing that horizon creates a different experiment.

Start with one liquid equity-market proxy, SPY. The output is a risk forecast that can eventually
inform exposure limits or risk budgets. It is not a forecast of return direction and is not
automatically an alpha score. Do not promise 90% accuracy: success requires useful performance
against majority and volatility-persistence baselines on future periods.

## First milestone: market-data contract

Before fitting models, record a source that permits research use, its license, available history,
adjustment methodology, timestamp convention, session calendar, retrieval date, and data hash.
Use the existing `qr_haven.data.CSVPriceDataPortal` through a small adapter for a local, validated
daily-price snapshot. Reuse ingestion infrastructure without assuming its generic `close` field
already contains the appropriate adjusted price.

The default research population is SPY from 2005-01-01 through 2025-12-31, subject to verified
source availability. The final two full calendar years, 2024–2025, are reserved before model
exploration. If that coverage cannot be obtained, explicitly version the date policy before
examining model results rather than quietly using a shorter favorable sample.

Require a documented split/dividend-adjusted close series, chronological unique sessions, and
positive finite prices. Audit corporate actions and missing sessions. Do not fill missing trading
days with zero returns; distinguish exchange closures from missing records. Keep an immutable
snapshot and record revision limitations when historical adjusted values are not true as-of data.
Delay macro, VIX, intraday, and cross-asset inputs until their own availability policies are defined.

Two provider profiles are frozen. The recommended primary profile is Tiingo SPY adjusted close for
2005–2025 with the 2024–2025 holdout. A validated fallback uses the locally supplied SPX price-index
CSV from 2005 onward, with 2018–2019 held out and 2020 quarantined. The profiles are separate
experiments and their results are not pooled. Acquisition and validation commands are documented in
the [market-data guide](../../docs/api/volatility_market_data.md).

## Label definition

Let `P_t` be the validated adjusted close and `r_t = log(P_t / P_(t-1))`. Define the v0 realized
volatility proxy using root mean squared returns, with 252 sessions per year:

```text
forward_vol_5(t)  = sqrt((252 / 5) * sum(r_(t+i)^2 for i = 1..5))
trailing_vol_5(t) = sqrt((252 / 5) * sum(r_(t-i)^2 for i = 0..4))
```

This definition intentionally does not subtract the sample mean. Apply it consistently to labels
and persistence features. Each observation records `as_of=t`, `label_start=t+1`, and
`label_end=t+5`, using exchange sessions rather than calendar days.

For each training fold, compute `q_train`, the 75th percentile of forward five-session volatility
among eligible training observations whose full label windows have already completed. Fix the
quantile convention to linear interpolation and record the numeric threshold.

```text
target(t) = 1 (high)   if forward_vol_5(t) > q_train
target(t) = 0 (normal) otherwise
```

Use that training threshold for both training and the corresponding validation block. The last
five unlabeled origins in the price snapshot are excluded. Never define thresholds from the
full history or force the test period to have a particular class balance. At the final fit, estimate
one threshold from eligible pre-2024 development labels and freeze it for all of 2024–2025.

Since thresholds can change across development folds, every prediction must carry its threshold
and fold ID. Report metrics within each fold before combining predictions; disclose that the
pooled validation task uses fold-specific definitions of high volatility.

## Initial features and models

All features are available at or before `as_of`, after the close:

- Trailing realized volatility over 5, 10, 20, and 60 sessions, using the same RMS definition.
- One-session and five-session log returns, plus the absolute one-session return.
- The ratio of five-session to twenty-session volatility, with explicit zero-denominator handling.
- Current drawdown from the trailing sixty-session adjusted-close maximum.

Drop origins without complete feature history; do not backfill from the future. Record feature
availability, label windows, and the price-source revision caveat. If portfolio use is later added,
the first eligible trade is after the forecast's availability, not at the already observed close.

Start with logistic regression, random forest, and `HistGradientBoostingClassifier`. Use small
predeclared grids and development-only preprocessing. Reuse banknote model/metric infrastructure
where appropriate but introduce a separate volatility configuration and target schema.

Required baselines:

1. Majority class fitted on the eligible training labels; also report the always-normal rule.
2. Persistence: predict high exactly when `trailing_vol_5(t) > q_train`. The trailing volatility
   itself provides a continuous ranking score for ROC-AUC and average precision.

## Chronological validation

Use expanding-window validation on pre-2024 history. Start with the five calendar years of price
history from 2005–2009, removing feature warmup and incomplete-label rows, and evaluate successive
calendar-year validation blocks from 2010 through 2023 where coverage allows. Candidate models
and baselines share identical eligible origins, thresholds, and boundaries.

For a block whose first forecast origin is `b`, require every training observation's `label_end`
to be **strictly earlier than `b`**. This removes at least the last five daily forecast origins
before the boundary. Apply the same rule before the final 2024–2025 holdout and before every
inner tuning/calibration boundary if those operations are introduced.

Development validation labels must also finish strictly before the first 2024 holdout session.
Trim the last development block accordingly: a December 2023 forecast whose target needs a
January 2024 return cannot be used for tuning. Later validation blocks may use earlier observed
returns as feature history, but no evaluation outcome may cross into the reserved final holdout
during model selection.

Construct and persist explicit origin/label intervals and audit the purge. A generic time splitter
does not inspect forward-label availability; its `gap` option alone is not evidence that all
interval constraints hold. The [scikit-learn TimeSeriesSplit reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
describes chronological splitting and its sample-based gap parameter.

No random train/test shuffle is allowed. No later observations enter a fold's training set, so a
separate post-validation embargo is unnecessary in this strictly forward design. If future work
allows training on both sides of a validation block, specify and test an embargo separately.

Daily forecast targets overlap within evaluation blocks. Report block-level results, a secondary
evaluation on every fifth eligible session anchored to the first origin of each block, and a
paired moving-block bootstrap for differences against persistence. Predeclare a 20-session
block length, 2,000 replicates, and a fixed bootstrap seed in the implementation spec. Do not use
an IID Wilson interval to imply independence across overlapping daily labels.

Select the model by mean development-fold balanced accuracy, with macro F1 as tie-breaker and
a fixed model/grid order thereafter. Freeze features, model settings, the final training threshold,
and native prediction rule before evaluating 2024–2025. In v0, fit once on eligible pre-2024 data
and keep model/threshold fixed throughout the final holdout; scheduled retraining is a later protocol.
Exclude any origin whose five-session label window crosses the available data end.

## Evaluation and success

Report accuracy, balanced accuracy, high-volatility precision/recall/F1, macro F1, ROC-AUC,
average precision, both class supports, confusion matrices, and the same metrics for baselines.
Compare performance by year and volatility environment. Report actual holdout prevalence;
the training quantile does not guarantee 25% high labels in unseen years.

The provisional research gate is final holdout balanced accuracy above both majority and
persistence baselines, with high-volatility recall no worse than persistence. Report the paired
block-bootstrap interval for the balanced-accuracy improvement over persistence. Call improvement
statistically supported only if its 95% interval excludes zero on the positive side. A mixed or
negative result remains a completed research outcome and must be retained.

Probability calibration, Brier score, and economic evaluation are later additions if calibrated
risk probabilities or position sizing are required. Keep confidence scores distinct from
validated probabilities. Classification performance alone is not evidence of profitable trading.

## Repository handoff and implementation sequence

1. Finalize the provider, snapshot, calendar, and adjustment contract and write the full volatility
   implementation spec before looking at predictive results.
2. Add point-in-time feature/label builders and tests that inspect individual forward windows.
3. Add a dedicated purged walk-forward split policy and train-only label-threshold estimation.
4. Implement majority/persistence baselines, bounded model selection, and locked final evaluation.
5. Reuse classification manifests, persistence, and reports, adding interval, threshold, and source
   availability fields; document dependence-aware uncertainty.
6. Publish the research writeup under `docs/research/market_regime_detection/` and then assess a
   separate, cost-aware risk-budget integration with `ResearchPipeline`.

Existing HMM/GMM components are context, not ground-truth labels. Do not use full-sample HMM
smoothed posteriors or a full-sequence Viterbi path as historical predictive features; they can use
future observations. Any later HMM feature must demonstrate filtering using data available at
the forecast origin and fitting restricted to the eligible training history.

Completion requires reproducible real-data forecasts, verified timing/purging, baseline comparisons,
an honest holdout report, and relevant tests. Achieving a particular headline accuracy is not the
completion gate for this follow-on project.

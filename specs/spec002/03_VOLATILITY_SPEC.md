# Spec002 V1: Next-Five-Session Volatility Classification

Status: V1–V4 complete for the local SPX profile; the final research gate was not met.

Frozen: 2026-10-06

This document turns the [volatility brief](02_VOLATILITY_FOLLOW_ON.md) into an executable contract.
No predictive result was inspected before freezing these choices.

## Research question

After session `t` closes, predict whether annualized realized volatility over sessions `t+1`
through `t+5` will exceed a threshold estimated only from the eligible training history. The output
is a risk-regime classification. It is not a return-direction forecast or a trading strategy.

Each observation stores `as_of`, `available_at`, `label_start`, and `label_end`. Features use data no
later than `as_of`. A forecast becomes available after the official close, so the earliest eligible
portfolio action is the next session.

## Frozen data profiles

The implementation accepts one explicit profile per run. Models, thresholds, selection results,
and reports are never pooled across profiles.

| Profile | Series | Study data | Development validation | Locked final holdout | Status |
| --- | --- | --- | --- | --- | --- |
| `tiingo-spy-v1` | Tiingo SPY `adjClose` | 2005-01-01–2025-12-31 | calendar years 2010–2023 | 2024–2025 | Recommended; snapshot awaits a local token |
| `spx-local-v1` | Supplied SPX `Adj Close` (= `Close`) | 2005-01-01–2020-11-04 | calendar years 2010–2017 | 2018–2019 | Available and validated locally |

For `spx-local-v1`, all 2020 observations are quarantined from v1. They are not used for fitting,
selection, threshold estimation, or the final evaluation. A later spec may define them as a
separate stress period. The archive's 1927–2004 rows are also excluded because upstream provenance
and historical session-calendar quality are unverified.

`tiingo-spy-v1` is the primary research profile. Its adjusted close incorporates splits and cash
dividends under Tiingo's documented CRSP-style methodology. The immutable downloaded bytes define
the data version because adjusted histories may later be revised. The token must be supplied through
`TIINGO_API_TOKEN`, is sent in the authorization header, and is never stored.

`spx-local-v1` is a price-index experiment. Its adjusted-close column equals close in every row and
does not include reinvested dividends. This is acceptable for a separately labeled volatility
study: the objective needs a consistent price-return series, not portfolio total return. Its source
and redistribution terms remain unverified, so the raw file stays local and ignored by Git.

Both profiles use the XNYS calendar from `exchange-calendars`, UTC session labels, and the existing
`CSVPriceDataPortal` canonical schema. Every snapshot must have chronological unique sessions,
positive finite prices, valid OHLC bounds, nonnegative whole-number raw volume, exact expected
sessions, an immutable manifest, and SHA-256 hashes. Missing exchange sessions are errors and are
never filled with zero returns.

## Features and target

Let `P_t` be the selected profile's declared price field and
`r_t = log(P_t / P_(t-1))`. Use root mean squared log returns without subtracting the mean:

```text
forward_vol_5(t)  = sqrt((252 / 5) * sum(r_(t+i)^2, i=1..5))
trailing_vol_n(t) = sqrt((252 / n) * sum(r_(t-i)^2, i=0..n-1))
```

Construct these nine features after session `t`:

1. `trailing_vol_5`, `trailing_vol_10`, `trailing_vol_20`, `trailing_vol_60`;
2. `log_return_1 = r_t` and `log_return_5 = log(P_t / P_(t-5))`;
3. `abs_log_return_1 = abs(r_t)`;
4. `vol_ratio_5_20 = trailing_vol_5 / trailing_vol_20`, set to zero only when the denominator is zero;
5. `drawdown_60 = P_t / max(P_(t-59)..P_t) - 1`.

Drop an origin unless all nine features and the full five-session forward label window exist. Never
backfill a feature. Store the input session identifiers used for every feature and label in the
audit table so unit tests can inspect individual windows.

Within each fold, estimate `q_train` as the 75th percentile of eligible training
`forward_vol_5` values using linear interpolation. Define `target=1` only when
`forward_vol_5 > q_train`; equality is normal (`0`). Apply that same threshold to the fold's
training and validation observations. For the final fit, estimate one threshold using eligible
pre-holdout labels and freeze it for the entire holdout.

## Chronological split and purge

Use expanding calendar-year validation. The initial training period is 2005–2009 after feature
warmup. For each validation year, training may include only earlier origins whose `label_end` is
strictly earlier than the first validation origin. Apply the same interval rule at the final
holdout boundary. No random split, future preprocessing, or full-history target threshold is
allowed.

The Tiingo profile validates 2010 through 2023 and then fits on eligible pre-2024 observations. The
local SPX profile validates 2010 through 2017 and then fits on eligible pre-2018 observations. A
late-December origin whose label enters the next boundary is purged. Persist membership by stable
`profile/symbol/as_of` IDs and assert that every training label ends before its validation block.

## Baselines and learned candidates

Every candidate uses identical eligible origins and fold boundaries. The required baselines are:

- majority class estimated from the fold's training labels;
- always normal;
- persistence: high when `trailing_vol_5 > q_train`, using trailing volatility as its ranking score.

Fit exactly 20 learned candidates in the order below. Use inverse-frequency weights computed from
each training fold. Fit the logistic scaler on training features only. Reject nonfinite features;
do not impute.

| Order | Family | Frozen settings | Grid |
| --- | --- | --- | --- |
| 1 | Logistic regression | `StandardScaler`, `solver=lbfgs`, `max_iter=2000` | `C ∈ {0.1, 1, 10}` |
| 2 | Random forest | `n_estimators=400`, `max_features=sqrt`, `random_state=5401`, `n_jobs=1` | `max_depth ∈ {5, 10, None}` × `min_samples_leaf ∈ {1, 5, 20}` |
| 3 | Histogram gradient boosting | `max_iter=300`, `early_stopping=False`, `random_state=5402` | `learning_rate ∈ {0.03, 0.1}` × `max_leaf_nodes ∈ {7, 15}` × `l2_regularization ∈ {0, 1}` |

Use each classifier's native class prediction and positive-class score. Select the candidate with
highest unweighted mean validation-year balanced accuracy, then mean macro F1, then the fixed table
and grid order. Do not tune a probability cutoff. Freeze the selected feature order, preprocessing,
hyperparameters, threshold procedure, and prediction rule before opening the final holdout.

## Metrics and dependence-aware comparison

Report by fold, pooled out-of-fold development, and holdout:

- accuracy and balanced accuracy;
- high-volatility precision, recall, and F1;
- macro F1, ROC-AUC, and average precision;
- both class supports and a confusion matrix;
- prevalence, the numeric training threshold, and all baseline results.

V3 metric convention, recorded before model comparison: retain single-class validation years.
Balanced accuracy averages recall over observed classes (and therefore equals that class's recall
when only one class occurs). Macro F1 uses both labels, with undefined precision/recall/F1 set to
zero. ROC-AUC and average precision are null for a single-class block. Report class support and
flag these blocks; a high score in a quiet year is not evidence of high-volatility detection.
All validation years have equal weight in candidate selection, including these blocks.

Daily five-session labels overlap. Also report every fifth eligible origin, anchored to the first
origin in each block. For the holdout comparison with persistence, use a paired circular
moving-block bootstrap over chronological daily predictions: 20-session blocks, 2,000 valid
replicates, seed `5403`, and a percentile 95% interval for the balanced-accuracy difference. Draw
the same sampled blocks for the model and persistence predictions. Discard a replicate if either
class is absent and continue until 2,000 valid replicates or 20,000 total attempts; record discarded
and attempted counts and fail the interval calculation if 2,000 valid replicates are not obtained.

The research gate passes only when final holdout balanced accuracy exceeds both majority and
persistence, high-volatility recall is at least persistence recall, and the bootstrap interval for
balanced-accuracy improvement over persistence lies strictly above zero. A failed gate remains a
valid completed result and must be retained without changing this contract.

## Artifacts and interfaces

Data preparation is currently available as:

```bash
python scripts/prepare_spx_local.py
python scripts/prepare_spx_local.py --verify-only
python scripts/fetch_spy_tiingo.py
python scripts/fetch_spy_tiingo.py --verify-only
```

V2 dataset preparation and integrity verification are implemented as:

```bash
python -m qr_haven.ml.volatility prepare --profile spx-local-v1
python -m qr_haven.ml.volatility verify \
  --output-dir artifacts/classification/volatility/spx-local-v1/dataset-v1
```

V2 writes development observations, holdout features, sealed holdout outcomes, exact input-window
audits, split membership, hashes, and the frozen split plan. Development and holdout outcomes are in
separate files so model-selection code does not need access to final outcomes.

V3 development-only model comparison is implemented as:

```bash
python -m qr_haven.ml.volatility train --profile spx-local-v1 --run-id spx-vol-v2
python -m qr_haven.ml.volatility verify-run \
  --run-dir artifacts/classification/volatility/spx-local-v1/spx-vol-v2
```

Replace the profile with `tiingo-spy-v1` after its snapshot exists. V3 writes an immutable run
directory containing the resolved configuration, environment, code hashes, source manifest, split,
fold thresholds, candidate metrics, out-of-fold predictions, selected pipeline, final-development
fit replay, and a Markdown report. It does not read or write holdout outcomes. V4 adds the single
final evaluation, holdout predictions, bootstrap summary, machine-readable results, and final
report without refitting or changing the V3 selection.

V4 outputs use a separate immutable evaluation directory so opening the holdout never modifies the
V3 training run:

```text
artifacts/classification/volatility/<profile>/
├── <model-run>/
│   ├── model.pkl
│   ├── selection.json
│   └── development_report.md
└── evaluations/<model-run>/<evaluation-id>/
    ├── manifest.json
    ├── metrics.json
    ├── holdout_predictions.csv
    ├── bootstrap.json
    └── report.md
```

The default local evaluation path is
`artifacts/classification/volatility/spx-local-v1/evaluations/spx-vol-v2/holdout-v1/`.
The manifest binds the evaluation to the V2 data hashes and exact V3 model hash. Evaluation IDs are
single safe path components, and an existing evaluation directory is never overwritten. Generated
artifacts remain ignored by Git; after review, the durable result is summarized under
`docs/research/classification/`.

The completed local evaluation can be returned or verified without reopening outcomes:

```bash
python -m qr_haven.ml.volatility evaluate \
  --run-dir artifacts/classification/volatility/spx-local-v1/spx-vol-v2
python -m qr_haven.ml.volatility verify-evaluation \
  --output-dir artifacts/classification/volatility/spx-local-v1/evaluations/spx-vol-v2/holdout-v1
```

## V2 measured evidence

The local SPX build produced 3,924 eligible origins after feature warmup and the five-session label
tail. Final training contains 3,207 origins. Eight validation folds cover 2010–2017, each with five
training origins purged at its boundary. The locked 2018–2019 holdout contains 498 origins, with five
additional origins purged because their labels cross into 2020. The remaining 209 eligible 2020
origins are quarantined. Artifact hashes verify, and no holdout outcome distribution or metric was
summarized during preparation.

## V3 measured evidence

Run `spx-vol-v2` evaluated the frozen 20 learned candidates and three baselines on eight purged
yearly folds. `hist_gradient_boosting_01` won with `learning_rate=0.03`, `max_leaf_nodes=7`, and
`l2_regularization=1.0`. Its unweighted mean yearly balanced accuracy was **0.708885**, compared
with **0.662901** for persistence and **0.562500** for majority/always-normal. Mean macro F1 was
**0.626525**. Pooled out-of-fold balanced accuracy was **0.807831** over 2,008 predictions, but that
pooled value was not used for selection. The final model was fit to 3,207 development origins using
the frozen threshold **0.176685395725789**.

The selected model's mean ordinary accuracy, **0.876417**, was below persistence's **0.886889**.
This is consistent with the strong and variable class imbalance and is why the predeclared ranking
metric is balanced accuracy. The 2017 validation block contained no high-volatility labels under
its training-only threshold; its ROC-AUC and average precision are undefined and it remains in the
equal-weight yearly mean under the frozen convention. The 2018–2019 outcomes remained sealed
through V3 and were opened only by the V4 evaluation below.

## V4 measured evidence

The frozen `hist_gradient_boosting_01` model was evaluated once on 498 holdout origins from
2018-01-02 through 2019-12-23. The fixed threshold produced 109 high-volatility observations
(21.89%) and 389 normal observations. The model achieved **0.744980 accuracy**, **0.701387 balanced
accuracy**, and **0.623853 high-volatility recall**. Persistence achieved **0.807229 accuracy**,
**0.718120 balanced accuracy**, and **0.559633 high-volatility recall**. Majority and always-normal
both achieved 0.781124 accuracy and 0.500000 balanced accuracy.

The model beat majority on balanced accuracy and exceeded persistence on high-volatility recall,
but it did not beat persistence on balanced accuracy. Its paired balanced-accuracy difference was
**-0.016733**, with a 95% circular moving-block bootstrap interval of **[-0.107333, 0.075386]**.
The interval was not strictly above zero, so the predeclared research gate was not met. The
every-fifth-origin secondary comparison favored the model on balanced accuracy, 0.756917 versus
0.745906, but that secondary result does not replace the daily gate. The model, threshold, grid,
and holdout policy remain unchanged after exposure.

## Acceptance checklist

- [x] Both data profiles prepare through a common canonical price schema and enforce their manifests.
- [x] Tests prove individual feature and target windows contain no future observations.
- [x] Every fold and final boundary passes the strict `label_end < next_origin` purge assertion.
- [x] Thresholds, scalers, weights, and models are fit only on the eligible training side.
- [x] All baselines and learned candidates use identical validation origins.
- [x] The selected model is determined from development folds only and evaluated once per profile.
- [x] Reports disclose price basis, source limitations, holdout prevalence, overlap, and bootstrap method.
- [x] Relevant tests, focused Ruff, and focused mypy checks pass; repository-wide pre-existing findings
  are reported separately.

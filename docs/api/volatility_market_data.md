# Volatility market data

The volatility experiment accepts two versioned data profiles. **Tiingo SPY is the recommended
final-research profile** because it covers the frozen 2024–2025 holdout and documents its split and
dividend adjustment. The supplied SPX CSV is ready for immediate local development and a separately
dated benchmark. Results from the two profiles are distinct experiments.

## Use the supplied local SPX file

The source file at `data/input/spx/SPX.csv` is ignored by Git. The reviewed file contains 23,323
rows from 1927-12-30 through 2020-11-04 and has SHA-256
`54aa877d5d275b660bc8171642dc3a2eece16b9709e1d35f13a3b4e7972cc35c`.

Prepare the default 2005–2020 study slice:

```bash
python scripts/prepare_spx_local.py
python scripts/prepare_spx_local.py --verify-only
```

This writes an immutable, canonical snapshot under the ignored directory
`data/processed/market_data/spx-local-2005-2020-v1/`. It checks schema, order, uniqueness, finite
positive prices, OHLC consistency, and exact XNYS sessions. It refuses to overwrite an existing
snapshot.

The file's `Adj Close` equals `Close` in all rows. It is therefore treated as an S&P 500 price-index
series, not as a dividend-reinvested total-return series. Total return is not required to classify
future volatility, but the price basis must stay fixed within an experiment. This profile uses
2018–2019 as its locked final holdout and leaves 2020 unused in v1. Data before 2005 remain in the
local archive but are outside the default study because their provenance and historical calendar
quality have not been established.

## Fetch the recommended Tiingo SPY profile

Create a Tiingo account and obtain an API token. Enter it without placing the value in shell
history:

```zsh
read -s "TIINGO_API_TOKEN?Tiingo token: "
echo
export TIINGO_API_TOKEN
python scripts/fetch_spy_tiingo.py
unset TIINGO_API_TOKEN
python scripts/fetch_spy_tiingo.py --verify-only
```

The fetcher requests SPY from 2005-01-01 through 2025-12-31, sends the token only in the HTTPS
authorization header, validates the returned metadata and exact XNYS sessions, and stores raw and
canonical files under the ignored `data/raw/` tree. It never writes the token. The snapshot uses
Tiingo `adjClose`, which Tiingo documents as split and dividend adjusted using CRSP methodology.

The snapshot manifest records the raw hashes, retrieval time, endpoints, calendar version, price
adjustment, revision policy, and license restriction. Tiingo's standard terms permit internal use
and prohibit redistribution unless a separate license allows it, so do not commit the downloaded
files. Commit only a reviewed dataset record derived from the generated manifest.

## Choosing a profile

Use `spx-local-v1` now to implement and test feature construction, purging, baselines, and model
selection without another download. Use `tiingo-spy-v1` for the primary final report once a token is
available. Freeze a model independently for each profile and never tune one profile after examining
the other's holdout results.

The complete timing, model-grid, and evaluation contract is in
[`specs/spec002/03_VOLATILITY_SPEC.md`](../../specs/spec002/03_VOLATILITY_SPEC.md).

## Build the point-in-time dataset

After preparing either source snapshot, build V2 features, continuous forward-volatility outcomes,
and purged expanding-year membership:

```bash
python -m qr_haven.ml.volatility prepare --profile spx-local-v1
python -m qr_haven.ml.volatility verify \
  --output-dir artifacts/classification/volatility/spx-local-v1/dataset-v1
```

Use `--profile tiingo-spy-v1` after its snapshot is available. Preparation refuses to overwrite an
existing V2 directory. The artifacts separate development observations from holdout features and
sealed holdout outcomes. V3 model selection reads only `development_observations.csv` and split
metadata.

Run and verify the frozen development comparison:

```bash
python -m qr_haven.ml.volatility train --profile spx-local-v1 --run-id spx-vol-v2
python -m qr_haven.ml.volatility verify-run \
  --run-dir artifacts/classification/volatility/spx-local-v1/spx-vol-v2
```

The local SPX build contains 3,924 eligible forecast origins after 60-return warmup and the
five-session label tail. Its final training population has 3,207 origins, eight development folds
cover 2010–2017, and the untouched 2018–2019 holdout has 498 origins. Five origins are purged at
each chronological boundary. The 209 eligible 2020 origins remain quarantined.

The local `spx-vol-v2` run selected histogram gradient boosting with mean yearly balanced accuracy
0.708885, versus 0.662901 for the persistence baseline. See the
[development result](../research/classification/spx_volatility_development.md).

## V4 output location

Final validation outputs will be stored separately from the fitted V3 run:

```text
artifacts/classification/volatility/spx-local-v1/
├── spx-vol-v2/                         # immutable V3 model-selection run
└── evaluations/spx-vol-v2/holdout-v1/ # immutable V4 holdout evaluation
    ├── manifest.json
    ├── metrics.json
    ├── holdout_predictions.csv
    ├── bootstrap.json
    └── report.md
```

This directory is local and ignored by Git because it contains generated predictions and model
evidence. The reviewed conclusion will be copied into a tracked research note under
`docs/research/classification/`. The evaluation implementation will refuse to overwrite an existing
evaluation ID.

Run or verify the completed local evaluation:

```bash
python -m qr_haven.ml.volatility evaluate \
  --run-dir artifacts/classification/volatility/spx-local-v1/spx-vol-v2
python -m qr_haven.ml.volatility verify-evaluation \
  --output-dir artifacts/classification/volatility/spx-local-v1/evaluations/spx-vol-v2/holdout-v1
```

The 2018–2019 evaluation is now recorded. The model scored 0.701387 balanced accuracy versus
0.718120 for persistence; its paired improvement interval was [-0.107333, 0.075386]. The research
gate was not met. Repeating the command verifies and returns the immutable existing evaluation
without reopening the outcome file. See the
[final research result](../research/classification/spx_volatility_final.md).

## Four exploratory experiments after V4

The `experiment` command diagnoses saved errors, compares class weights and cutoffs, evaluates
EWMA/linear volatility forecasts, and tests rolling training windows. It uses the existing local
V2/V3 artifacts; no download is needed. All new models are fitted and compared on the original
development folds. Supplying `--evaluation-dir` adds diagnosis of already-saved V4 predictions;
it does not refit on, forecast, or evaluate new variants against the holdout.

The completed local run is `improvements-v1`. To create that run on a fresh checkout with the
input artifacts available:

```bash
python -m qr_haven.ml.volatility experiment \
  --run-dir artifacts/classification/volatility/spx-local-v1/spx-vol-v2 \
  --evaluation-dir artifacts/classification/volatility/spx-local-v1/evaluations/spx-vol-v2/holdout-v1 \
  --experiment-id improvements-v1
```

An existing experiment ID cannot be overwritten. Verify or open the completed run instead:

```bash
python -m qr_haven.ml.volatility verify-experiment \
  --output-dir artifacts/classification/volatility/spx-local-v1/experiments/spx-vol-v2/improvements-v1
open artifacts/classification/volatility/spx-local-v1/experiments/spx-vol-v2/improvements-v1/report.md
```

Use a new `--experiment-id` only when another development run is intended. Omit `--evaluation-dir`
to diagnose development errors alone; `--v2-dir` supports a nondefault frozen dataset location.
The default V2 directory is the model run's sibling `dataset-v1`.

Each run saves these local, Git-ignored files under
`artifacts/classification/volatility/<profile>/experiments/<model-run>/<experiment-id>/`:

| Files | Purpose |
| --- | --- |
| `report.md`, `ranking.csv` | All four experiment summaries; mean yearly rankings, including every-fifth-origin results. |
| `cv_results.csv`, `cv_predictions.csv`, `pooled_metrics.csv` | Fold scores, individual development predictions, and descriptive pooled scores. |
| `diagnostic_observations.csv`, `error_summary.csv`, `error_runs.csv`, `paired_errors.csv` | Saved errors by date, year, and current-volatility regime, with denominators and comparisons to persistence. |
| `fit_audit.csv`, `training_membership.csv` | Fit weights/coefficients, purged training windows, and exact sample membership. |
| `config.json`, `manifest.json` | Protocol saved before fitting, source/environment/input identity, state, and output hashes. |

The original V3 predictions must replay before the comparison can complete. Failed runs retain
their ID and failure reason. New variants do not produce a replacement final model. The
[experiment protocol](../../specs/spec002/04_EXPLORATORY_EXPERIMENTS.md) and
[measured results](../research/classification/spx_volatility_experiments.md) explain the choices
and the limits of selecting candidates on reused development folds.

## V7 adaptive-target history experiment

V7 uses only the Tiingo challenger development file through 2023. It defines high volatility at
each origin against the trailing three-calendar-year 75th percentile of outcomes already
observable at that time, then compares five training-history rules. It does not open 2024–2025
outcomes and does not fit a final model.

Run or verify the immutable output:

```bash
python -m qr_haven.ml.volatility history-experiment
python -m qr_haven.ml.volatility history-verify \
  --output-dir artifacts/classification/volatility/tiingo-spy-v1/history_experiments/v7-history-v1
```

The completed run selected pure adaptive persistence with 0.716707 mean yearly balanced accuracy.
The best model-only design used twelve rolling years and scored 0.694039, compared with 0.686054
for five years. Generated thresholds, weights, memberships, fold predictions, rankings, and the
report are under the ignored `history_experiments/v7-history-v1/` directory. See the
[V7 protocol](../../specs/spec002/07_V7_HISTORY_ADAPTATION_SPEC.md) and
[measured result](../research/classification/spy_volatility_history_v7.md).

## V8 continuous volatility experiment

V8 forecasts continuous next-five-session annualized volatility and ranks candidates by mean
yearly QLIKE. It compares histogram regression, HAR-RV, three EWMA forecasts, variance blends with
persistence, and pure persistence. Development uses only observations through 2023.

Run or verify the immutable development output:

```bash
python -m qr_haven.ml.volatility continuous-experiment
python -m qr_haven.ml.volatility continuous-verify \
  --output-dir artifacts/classification/volatility/tiingo-spy-v1/continuous_experiments/v8-continuous-v1
```

The completed run selected `hist_gradient_boosting_regression_w075`: 75% histogram forecast
variance and 25% persistence variance. Mean yearly QLIKE was 0.510667, versus 1.132493 for
persistence. The candidate advances to a separately frozen one-time 2026 evaluation after the
complete extension is available. This development result is not a passed research gate.

Generated forecasts, losses, alert diagnostics, membership, audits, ranking, selection, and report
are stored under the ignored `continuous_experiments/v8-continuous-v1/` directory. See the
[V8 protocol](../../specs/spec002/08_V8_CONTINUOUS_VOLATILITY_SPEC.md) and
[development result](../research/classification/spy_continuous_volatility_v8.md).

## Frozen five-year challenger

The V5 challenger implements the next protocol with extended realized-volatility and downside-risk
features, HAR-RV, three EWMA decays, the existing histogram configuration, chronological Platt
calibration, and model-plus-persistence ensembles. Its selection folds stop in 2017. The final
pipeline can use 2018–2019 as past training observations after selection, and 2020 is opened once.

The completed commands were:

```bash
python -m qr_haven.ml.volatility challenger-prepare
python -m qr_haven.ml.volatility challenger-train
python -m qr_haven.ml.volatility challenger-evaluate \
  --run-dir artifacts/classification/volatility/spx-local-v1/challengers/challenger-v1
```

These commands are immutable: preparation and training refuse an existing output, while repeating
the same evaluation command verifies and returns the completed result without reopening outcomes.
Verify each stage directly with:

```bash
python -m qr_haven.ml.volatility challenger-verify-data \
  --output-dir artifacts/classification/volatility/spx-local-v1/challengers/dataset-v1
python -m qr_haven.ml.volatility challenger-verify-run \
  --run-dir artifacts/classification/volatility/spx-local-v1/challengers/challenger-v1
python -m qr_haven.ml.volatility challenger-verify-evaluation \
  --output-dir artifacts/classification/volatility/spx-local-v1/challengers/evaluations/challenger-v1/2020-v1
```

The local generated output contains the extended dataset and sealed outcomes, all development
scores and calibration examples, the fitted pipeline, exact memberships, the one-time evaluation
predictions, 2,000 bootstrap samples, metrics, reports, and manifests. The profile-level
`challenger_evaluation_history.jsonl` ledger contains the single 2020 exposure.

The 2020 gate was not met. See the
[frozen protocol](../../specs/spec002/05_FROZEN_CHALLENGER_SPEC.md) and
[measured result](../research/classification/spx_volatility_challenger.md).

## Continue on untouched Tiingo SPY data

The challenger commands also support the separately frozen `tiingo-spy-v1` design. That design
uses adjusted SPY OHLC consistently: `adjClose` drives returns and adjusted high/low drive
Parkinson volatility. It selects candidates on 2013–2023, fits final calibration on 2021–2023,
and reserves 2024–2025 for one evaluation.

No Tiingo snapshot is currently present. Acquire it with a user-owned token entered without shell
history, then verify it offline:

```zsh
read -s "TIINGO_API_TOKEN?Tiingo token: "
echo
export TIINGO_API_TOKEN
python scripts/fetch_spy_tiingo.py
unset TIINGO_API_TOKEN
python scripts/fetch_spy_tiingo.py --verify-only
```

After the snapshot verifies, build and select the challenger without opening evaluation outcomes:

```bash
python -m qr_haven.ml.volatility challenger-prepare --profile tiingo-spy-v1
python -m qr_haven.ml.volatility challenger-train --profile tiingo-spy-v1
```

Review and verify the development report before the one-time evaluation:

```bash
python -m qr_haven.ml.volatility challenger-verify-data \
  --output-dir artifacts/classification/volatility/tiingo-spy-v1/challengers/dataset-v1
python -m qr_haven.ml.volatility challenger-verify-run \
  --run-dir artifacts/classification/volatility/tiingo-spy-v1/challengers/challenger-v1
```

The final command opens the sealed 2024–2025 outcomes once and defaults to evaluation ID
`2024-2025-v1`:

```bash
python -m qr_haven.ml.volatility challenger-evaluate \
  --run-dir artifacts/classification/volatility/tiingo-spy-v1/challengers/challenger-v1
```

The [Tiingo SPY challenger specification](../../specs/spec002/06_TIINGO_SPY_CHALLENGER_SPEC.md)
was committed before acquisition. Do not run the evaluation until the selected development report
has been reviewed and the training manifest verifies.

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
[development result](../research/classification/spx_volatility_development.md). The 2018–2019
holdout outcomes remain sealed for V4.

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

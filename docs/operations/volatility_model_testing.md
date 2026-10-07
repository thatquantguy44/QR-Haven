# V8A volatility model testing deployment

The V8A candidate can be deployed for **shadow testing**. Its continuous forecast beat persistence
on the frozen 2026 YTD population, but the provisional promotion gate did not pass because alert
recall was lower. Every operational surface therefore reports `shadow_only` and
`target_not_met`; it must not be presented as a promoted trading or risk model.

## Generate the visual report and Power BI datasets

Run the export from the verified immutable evaluation:

```bash
python -m qr_haven.ml.volatility continuous-ytd-export
```

This creates the ignored local directory:

```text
artifacts/classification/volatility/tiingo-spy-v1/continuous_ytd/deployment/v8a-shadow-v1/
├── dashboard/index.html
├── power_bi/
│   ├── DimDate.csv
│   ├── DimModel.csv
│   ├── FactForecast.csv
│   ├── FactModelMetric.csv
│   ├── FactGate.csv
│   ├── FactBootstrap.csv
│   ├── data_dictionary.csv
│   └── model_schema.json
└── manifest.json
```

Open `dashboard/index.html` in a browser for the review page. It shows the forecast time series,
benchmark comparison, confidence interval, frozen gate results, and alert errors. It uses no CDN or
remote runtime assets.

In Power BI Desktop, choose **Get data → Text/CSV** and load the six `Dim*` and `Fact*` files.
Create the two relationships listed in `model_schema.json`:

- `DimDate[DateKey]` to `FactForecast[DateKey]`, one-to-many;
- `DimModel[ModelKey]` to `FactModelMetric[ModelKey]`, one-to-many.

The schema file also contains suggested DAX measures. `FactForecast` has one row per forecast
origin, `FactModelMetric` has one row per model and metric, and `FactBootstrap` has one row per
bootstrap replicate. The data dictionary defines the derived fields.

## Run the test API locally

Install the deployment dependencies and start the API:

```bash
pip install -e ".[research,deployment]"
uvicorn qr_haven.ml.volatility.deployment_api:app --host 127.0.0.1 --port 8000
```

The service exposes:

- `GET /health` for readiness and research/deployment status;
- `GET /model` for the feature contract, threshold, and training population;
- `POST /predict` for one or more point-in-time feature records;
- `/docs` for the generated OpenAPI test page.

`POST /predict` accepts `records`, where each record contains all 20 fields returned by
`GET /model`. Optional `sample_id`, `as_of`, and `available_at` values are copied to the response.
The response includes the model-only forecast, persistence forecast, blended candidate forecast,
fixed threshold, and alert flag. This testing API consumes already constructed point-in-time
features; raw OHLC ingestion and scheduled feature production remain upstream responsibilities.

Batch scoring is also available without running a service:

```bash
python -m qr_haven.ml.volatility continuous-ytd-score \
  --input-csv path/to/point_in_time_features.csv \
  --output-csv artifacts/shadow_forecasts.csv
```

## Run the container

The Compose service mounts the verified model bundle read-only instead of copying it into an image:

```bash
docker compose -f infrastructure/volatility-api/compose.yaml up --build
curl http://127.0.0.1:8000/health
```

The container runs with a read-only filesystem and serves port 8000. Model artifacts and Tiingo
data remain ignored by Git and outside the image.

## Testing path to a promotion decision

Use the API in shadow mode and retain forecasts before their five-session outcomes exist. Store an
append-only table with `sample_id`, feature availability time, forecast creation time, model hash,
forecast, threshold, and later realized outcome. Operational monitoring should track forecast
coverage, feature failures, QLIKE, log error, high-volatility recall, false alerts, and the
five-to-one alert cost against persistence.

The existing V8 complete-calendar-2026 protocol remains the next promotion test. Shadow monitoring
is operational evidence and must not change that frozen gate or turn the exposed V8A population
into a new holdout.

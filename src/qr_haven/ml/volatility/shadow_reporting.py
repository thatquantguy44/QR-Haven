"""Atomic reporting generations and Power BI data from the shadow ledger."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, sha256, utc_now
from qr_haven.ml.classification.artifacts import exclusive_lock
from qr_haven.ml.volatility.shadow_features import latest_completed_session, session_calendar

FIELD_TYPES = {
    "ForecastKey": "text",
    "SampleId": "text",
    "DateKey": "Int64.Type",
    "ModelKey": "text",
    "AsOfDate": "date",
    "HorizonEnd": "date",
    "Mode": "text",
    "OutcomeStatus": "text",
    "ModelHash": "text",
    "SourceHash": "text",
    "OutcomeSourceHash": "text",
    "CreatedAtUtc": "datetimezone",
    "ObservedAtUtc": "datetimezone",
    "AvailableAt": "datetimezone",
    "ForecastVolatility": "number",
    "AlertThreshold": "number",
    "PredictedHigh": "Int64.Type",
    "ActualVolatility": "number",
    "ActualHigh": "Int64.Type",
    "QLIKE": "number",
    "LogAbsoluteError": "number",
    "AbsoluteError": "number",
    "SquaredError": "number",
    "TruePositive": "Int64.Type",
    "FalsePositive": "Int64.Type",
    "FalseNegative": "Int64.Type",
    "TrueNegative": "Int64.Type",
    "AlertCost": "number",
    "Scored": "Int64.Type",
}


def ledger_frames(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    path = root / "ledger.sqlite"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        settings = db.execute("SELECT payload FROM settings WHERE id=1").fetchone()
        if settings is None:
            raise ValueError("Shadow ledger has no bound model")
        config = json.loads(settings[0])
        records = db.execute(
            "SELECT f.payload,o.payload FROM forecasts f LEFT JOIN outcomes o USING(sample_id) "
            "ORDER BY f.as_of"
        ).fetchall()
        runs = db.execute("SELECT id,created_at,payload FROM runs ORDER BY id").fetchall()
    rows = []
    for raw_forecast, raw_outcome in records:
        forecast = json.loads(raw_forecast)
        outcome = json.loads(raw_outcome) if raw_outcome else {}
        scored = outcome.get("outcome_status") == "scored"
        actual_high = outcome.get("actual_high") if scored else None
        for model in ("candidate", "persistence"):
            high = forecast[f"{model}_high"]
            tp = int(high == 1 and actual_high == 1) if scored else None
            fp = int(high == 1 and actual_high == 0) if scored else None
            fn = int(high == 0 and actual_high == 1) if scored else None
            tn = int(high == 0 and actual_high == 0) if scored else None
            error = outcome.get(f"{model}_absolute_error") if scored else None
            rows.append(
                {
                    "ForecastKey": f"{forecast['sample_id']}/{model}",
                    "SampleId": forecast["sample_id"],
                    "DateKey": int(forecast["as_of"].replace("-", "")),
                    "AsOfDate": forecast["as_of"],
                    "HorizonEnd": forecast["label_end"],
                    "ModelKey": model,
                    "Mode": forecast["mode"],
                    "OutcomeStatus": outcome.get("outcome_status", "pending"),
                    "ModelHash": forecast["model_sha256"],
                    "SourceHash": forecast["source_sha256"],
                    "OutcomeSourceHash": outcome.get("outcome_source_sha256"),
                    "CreatedAtUtc": forecast["created_at_utc"],
                    "ObservedAtUtc": forecast["observed_at_utc"],
                    "AvailableAt": forecast["available_at"],
                    "ForecastVolatility": forecast[f"{model}_forecast"],
                    "AlertThreshold": forecast["alert_threshold"],
                    "PredictedHigh": high,
                    "ActualVolatility": outcome.get("actual_volatility"),
                    "ActualHigh": actual_high,
                    "QLIKE": outcome.get(f"{model}_qlike"),
                    "LogAbsoluteError": outcome.get(f"{model}_log_error"),
                    "AbsoluteError": error,
                    "SquaredError": error**2 if error is not None else None,
                    "TruePositive": tp,
                    "FalsePositive": fp,
                    "FalseNegative": fn,
                    "TrueNegative": tn,
                    "AlertCost": fp + 5 * fn if fp is not None and fn is not None else None,
                    "Scored": int(scored),
                }
            )
    fact = pd.DataFrame(rows, columns=list(FIELD_TYPES))
    dates = (
        pd.date_range(fact["AsOfDate"].min(), fact["HorizonEnd"].max())
        if rows
        else pd.DatetimeIndex([])
    )
    dim_date = pd.DataFrame(
        {
            "DateKey": dates.strftime("%Y%m%d").astype("int64"),
            "Date": dates.strftime("%Y-%m-%d"),
            "Year": dates.year,
            "MonthNumber": dates.month,
            "YearMonth": dates.strftime("%Y-%m"),
        }
    )
    dim_model = pd.DataFrame(
        [
            {"ModelKey": "candidate", "Model": "75% histogram / 25% persistence"},
            {"ModelKey": "persistence", "Model": "Persistence (trailing five sessions)"},
        ]
    )
    run_records = [
        {"RunId": rid, "CreatedAtUtc": created, **json.loads(payload)}
        for rid, created, payload in runs
    ]
    run_frame = pd.DataFrame(
        run_records,
        columns=[
            "RunId",
            "CreatedAtUtc",
            "state",
            "mode",
            "source_last_session",
            "expected_latest_session",
            "data_fresh",
            "new_forecasts",
            "new_outcomes",
            "error",
        ],
    )
    latest = run_records[-1] if run_records else {}
    successes = [r for r in run_records if r.get("state") == "complete"]
    last_success = successes[-1] if successes else {}
    expected = latest_completed_session(pd.Timestamp.now(tz="UTC")).date().isoformat()
    source_last = last_success.get("source_last_session")
    candidate = fact.loc[fact["ModelKey"] == "candidate"]
    coverage_end = expected if config["mode"] == "live" else source_last
    coverage_dates = (
        session_calendar(
            pd.Timestamp(candidate["AsOfDate"].min()) - pd.Timedelta(days=1),
            pd.Timestamp(coverage_end) + pd.Timedelta(days=1),
        ).sessions_in_range(pd.Timestamp(candidate["AsOfDate"].min()), pd.Timestamp(coverage_end))
        if len(candidate) and coverage_end
        else pd.DatetimeIndex([])
    )
    recorded = set(candidate["AsOfDate"])
    coverage = pd.DataFrame(
        {
            "DateKey": coverage_dates.strftime("%Y%m%d").astype("int64"),
            "Date": coverage_dates.strftime("%Y-%m-%d"),
            "ForecastRecorded": [int(d.date().isoformat() in recorded) for d in coverage_dates],
        }
    )
    if len(coverage_dates) and (dates.empty or coverage_dates[-1] > dates[-1]):
        dates = pd.date_range(dates[0], coverage_dates[-1])
        dim_date = pd.DataFrame(
            {
                "DateKey": dates.strftime("%Y%m%d").astype("int64"),
                "Date": dates.strftime("%Y-%m-%d"),
                "Year": dates.year,
                "MonthNumber": dates.month,
                "YearMonth": dates.strftime("%Y-%m"),
            }
        )
    summary = {
        "generated_at_utc": utc_now(),
        "mode": config["mode"],
        "model_sha256": config["model_sha256"],
        "deployment_status": "shadow_only",
        "research_status": "target_not_met",
        "origins": len(candidate),
        "scored_origins": int((candidate["OutcomeStatus"] == "scored").sum()),
        "pending_origins": int((candidate["OutcomeStatus"] == "pending").sum()),
        "unscorable_origins": int((candidate["OutcomeStatus"] == "unscorable").sum()),
        "source_last_session": source_last,
        "expected_latest_session": expected,
        "data_fresh": source_last == expected,
        "latest_run_state": latest.get("state", "none"),
        "failed_runs": sum(r.get("state") == "failed" for r in run_records),
        "latest_error": latest.get("error"),
        "missing_origins": int((coverage["ForecastRecorded"] == 0).sum()),
    }
    return {
        "DimDate": dim_date,
        "DimModel": dim_model,
        "FactForecast": fact,
        "FactRun": run_frame,
        "FactStatus": pd.DataFrame([summary]),
        "FactCoverage": coverage,
    }, summary


def _power_query(name: str, frame: pd.DataFrame) -> str:
    types = []
    for field in frame.columns:
        kind = FIELD_TYPES.get(field, "text")
        if field == "Date":
            kind = "date"
        if field in {"Year", "MonthNumber", "RunId", "ForecastRecorded", "missing_origins"}:
            kind = "Int64.Type"
        if field in {
            "origins",
            "scored_origins",
            "pending_origins",
            "unscorable_origins",
            "failed_runs",
            "new_forecasts",
            "new_outcomes",
        }:
            kind = "Int64.Type"
        if field == "data_fresh":
            kind = "logical"
        if field in {"source_last_session", "expected_latest_session"}:
            kind = "date"
        if field == "generated_at_utc":
            kind = "datetimezone"
        type_value = kind if kind == "Int64.Type" else f"type nullable {kind}"
        types.append('{"' + field + '", ' + type_value + "}")
    return (
        "// Define ShadowRoot as a text parameter: directory containing ledger.sqlite.\n"
        "let\n"
        '    Manifest = Json.Document(File.Contents(ShadowRoot & "/reports/current.json")),\n'
        "    Generation = Manifest[generation],\n"
        '    Folder = ShadowRoot & "/reports/generations/" & Generation & "/power_bi/",\n'
        f'    Source = Csv.Document(File.Contents(Folder & "{name}.csv"), '
        '[Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),\n'
        "    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),\n"
        '    Blanks = Table.ReplaceValue(Headers, "", null, Replacer.ReplaceValue, '
        "Table.ColumnNames(Headers)),\n"
        "    Typed = Table.TransformColumnTypes(Blanks, {" + ", ".join(types) + '}, "en-US")\n'
        "in Typed\n"
    )


DAX = """// Create each measure in Power BI. Filter DimModel to compare the two forecasts.
Forecast origins = DISTINCTCOUNT(FactForecast[SampleId])
Scored origins = CALCULATE(DISTINCTCOUNT(FactForecast[SampleId]), FactForecast[Scored] = 1)
Pending origins =
    CALCULATE(DISTINCTCOUNT(FactForecast[SampleId]), FactForecast[OutcomeStatus] = "pending")
Mean QLIKE = CALCULATE(AVERAGE(FactForecast[QLIKE]), FactForecast[Scored] = 1)
Log MAE = CALCULATE(AVERAGE(FactForecast[LogAbsoluteError]), FactForecast[Scored] = 1)
Volatility MAE = CALCULATE(AVERAGE(FactForecast[AbsoluteError]), FactForecast[Scored] = 1)
Volatility RMSE = SQRT(CALCULATE(AVERAGE(FactForecast[SquaredError]), FactForecast[Scored] = 1))
High recall = DIVIDE(SUM(FactForecast[TruePositive]),
    SUM(FactForecast[TruePositive]) + SUM(FactForecast[FalseNegative]))
Alert cost = DIVIDE(SUM(FactForecast[AlertCost]), SUM(FactForecast[Scored]))
Candidate QLIKE = CALCULATE([Mean QLIKE], REMOVEFILTERS(DimModel), DimModel[ModelKey] = "candidate")
Persistence QLIKE =
    CALCULATE([Mean QLIKE], REMOVEFILTERS(DimModel), DimModel[ModelKey] = "persistence")
QLIKE improvement = [Persistence QLIKE] - [Candidate QLIKE]
// Never SUM volatility or average subgroup recall. A BLANK recall means no high outcomes.
"""


def export_shadow(root: Path) -> dict[str, Any]:
    root = root.resolve()
    with exclusive_lock(root / ".report.lock"):
        frames, summary = ledger_frames(root)
        generation = uuid.uuid4().hex
        base = root / "reports"
        output = base / "generations" / generation
        pbi = output / "power_bi"
        pbi.mkdir(parents=True)
        for name, frame in frames.items():
            atomic_write(pbi / f"{name}.csv", frame.to_csv(index=False).encode())
            atomic_write(pbi / f"{name}.m", _power_query(name, frame).encode())
        schema = {
            "grain": "FactForecast: one origin × model; two rows per origin",
            "relationships": [
                "DimDate[DateKey] 1:* FactForecast[DateKey] (single direction)",
                "DimModel[ModelKey] 1:* FactForecast[ModelKey] (single direction)",
                "DimDate[DateKey] 1:* FactCoverage[DateKey] (single direction)",
            ],
            "disconnected_tables": ["FactRun", "FactStatus"],
            "null_policy": "Pending truth, losses and counts stay null; never fill with zero",
            "fact_forecast_types": FIELD_TYPES,
            "generation": generation,
        }
        atomic_write(pbi / "model_schema.json", json_bytes(schema))
        atomic_write(pbi / "measures.dax", DAX.encode())
        definitions = {
            "ForecastKey": "Unique origin and model key; do not count this as unique origins",
            "SampleId": "Unique forecast origin; shared by candidate and persistence",
            "Mode": "live or replay; replay never constitutes prospective evidence",
            "OutcomeStatus": "pending, scored, or unscorable",
            "ObservedAtUtc": "Operational clock; simulated only in replay",
            "CreatedAtUtc": "Actual UTC time the forecast was committed",
            "ForecastVolatility": "Annualized next-five-session volatility forecast, fraction",
            "ActualVolatility": "sqrt(252/5 × sum of next five squared log returns)",
            "QLIKE": "actual variance / forecast variance - log(ratio) - 1; lower is better",
            "AlertCost": "FalsePositive + 5 × FalseNegative per scored model-origin",
            "Scored": "1 only for complete valid positive realized volatility",
        }
        dictionary = pd.DataFrame(
            [
                {
                    "Field": field,
                    "Type": kind,
                    "Definition": definitions.get(field, field),
                    "NullWhenPending": field
                    in {
                        "ActualVolatility",
                        "ActualHigh",
                        "QLIKE",
                        "LogAbsoluteError",
                        "AbsoluteError",
                        "SquaredError",
                        "TruePositive",
                        "FalsePositive",
                        "FalseNegative",
                        "TrueNegative",
                        "AlertCost",
                        "OutcomeSourceHash",
                    },
                }
                for field, kind in FIELD_TYPES.items()
            ]
        )
        atomic_write(pbi / "data_dictionary.csv", dictionary.to_csv(index=False).encode())
        atomic_write(output / "summary.json", json_bytes(summary))
        # Pandas converts NaN to JSON null. Escape HTML delimiters in inline evidence.
        payload = (
            json.dumps(
                {
                    "summary": summary,
                    "rows": json.loads(
                        frames["FactForecast"].to_json(orient="records", double_precision=15)
                    ),
                },
                allow_nan=False,
            )
            .replace("<", "\\u003c")
            .replace("&", "\\u0026")
        )
        template = Path(__file__).with_name("shadow_dashboard.html").read_text()
        atomic_write(
            output / "dashboard.html", template.replace("__SHADOW_DATA__", payload).encode()
        )
        files = {
            str(path.relative_to(output)): sha256(path.read_bytes())
            for path in output.rglob("*")
            if path.is_file()
        }
        manifest = {
            "generation": generation,
            "model_sha256": summary["model_sha256"],
            "created_at_utc": utc_now(),
            "files": files,
            "mode": summary["mode"],
        }
        atomic_write(output / "manifest.json", json_bytes(manifest))
        # Immutable complete generation first; only then move the stable reader pointers.
        temporary = base / f".current-{generation}"
        temporary.symlink_to(Path("generations") / generation, target_is_directory=True)
        os.replace(temporary, base / "current")
        atomic_write(base / "current.json", json_bytes(manifest), replace=True)
        return {
            "dashboard": str(base / "current" / "dashboard.html"),
            "power_bi_dir": str(base / "current" / "power_bi"),
            "summary": summary,
        }

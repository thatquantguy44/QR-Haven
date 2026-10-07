"""Shadow-deployment scoring and reporting for the V8A volatility candidate."""

# ruff: noqa: E501 - embedded self-contained HTML is kept readable as markup

from __future__ import annotations

import html
import json
from importlib.metadata import version
from pathlib import Path
from threading import Lock
from typing import Any, cast

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import sha256, utc_now
from qr_haven.ml.volatility.challenger_data import CHALLENGER_FEATURES
from qr_haven.ml.volatility.continuous_experiments import (
    _inverse_log_variance,
    blend_variance,
)
from qr_haven.ml.volatility.continuous_ytd import (
    ALERT_THRESHOLD,
    MODEL_WEIGHT,
    _load_ytd_model,
    verify_ytd_candidate,
    verify_ytd_evaluation,
)

DEFAULT_CANDIDATE_DIR = Path(
    "artifacts/classification/volatility/tiingo-spy-v1/continuous_ytd/candidate-v1"
)
DEFAULT_EVALUATION_DIR = Path(
    "artifacts/classification/volatility/tiingo-spy-v1/continuous_ytd/evaluations/"
    "candidate-v1/2026-ytd-v1"
)
DEFAULT_EXPORT_DIR = Path(
    "artifacts/classification/volatility/tiingo-spy-v1/continuous_ytd/"
    "deployment/v8a-shadow-v1"
)
_PREDICTION_LOCK = Lock()


def load_shadow_model(candidate_dir: Path) -> dict[str, Any]:
    """Fail before unpickling when the trusted bundle's runtime does not match training."""
    manifest = verify_ytd_candidate(candidate_dir)
    for package in ("scikit-learn", "numpy", "scipy", "pandas"):
        if version(package) != manifest["environment"][package]:
            raise ValueError(f"Model runtime mismatch for {package}; use the pinned deployment runtime")
    bundle = _load_ytd_model(candidate_dir)
    if bundle["model_weight"] != MODEL_WEIGHT or bundle["alert_threshold"] != ALERT_THRESHOLD:
        raise ValueError("Bundle does not match the frozen V8A serving rule")
    return bundle


def score_feature_rows(bundle: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    """Score point-in-time feature rows using the verified V8A bundle."""
    missing = sorted(set(CHALLENGER_FEATURES) - set(rows.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    features = rows.loc[:, list(CHALLENGER_FEATURES)].apply(pd.to_numeric, errors="raise")
    if features.empty or rows.columns.duplicated().any():
        raise ValueError("Feature rows must be nonempty with unique columns")
    values = features.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Feature rows must contain only finite values")
    persistence = features["trailing_vol_5"].astype("float64")
    if (persistence <= 0).any():
        raise ValueError("trailing_vol_5 must be positive")
    from threadpoolctl import threadpool_limits

    with _PREDICTION_LOCK, threadpool_limits(limits=1):
        predicted = bundle["estimator"].predict(features)
    base = pd.Series(
        _inverse_log_variance(np.asarray(predicted, dtype=float)),
        index=rows.index,
        name="base_forecast",
    )
    candidate = blend_variance(base, persistence, MODEL_WEIGHT)
    result = pd.DataFrame(
        {
            "base_forecast": base,
            "persistence_forecast": persistence,
            "candidate_forecast": candidate,
            "alert_threshold": ALERT_THRESHOLD,
            "candidate_high": (candidate > ALERT_THRESHOLD).astype("int64"),
        },
        index=rows.index,
    )
    for name in ("sample_id", "as_of", "available_at"):
        if name in rows.columns:
            result.insert(len(result.columns), name, rows[name].to_numpy())
    return result


def score_feature_csv(candidate_dir: Path, input_csv: Path, output_csv: Path) -> dict[str, Any]:
    """Score a CSV at the documented feature contract and write a flat result."""
    bundle = load_shadow_model(Path(candidate_dir))
    rows = pd.read_csv(input_csv, float_precision="round_trip")
    result = score_feature_rows(bundle, rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return {
        "rows": len(result),
        "output_csv": str(output_csv),
        "candidate_id": bundle["candidate_id"],
        "alert_threshold": ALERT_THRESHOLD,
        "deployment_status": "shadow_only",
    }


def _date_dimension(predictions: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(predictions["as_of"])
    return pd.DataFrame(
        {
            "DateKey": dates.dt.strftime("%Y%m%d").astype("int64"),
            "Date": dates.dt.strftime("%Y-%m-%d"),
            "Year": dates.dt.year,
            "Quarter": "Q" + dates.dt.quarter.astype(str),
            "MonthNumber": dates.dt.month,
            "Month": dates.dt.strftime("%b"),
            "YearMonth": dates.dt.strftime("%Y-%m"),
            "Weekday": dates.dt.strftime("%a"),
        }
    ).drop_duplicates("DateKey")


def _model_metrics(metrics: dict[str, Any]) -> pd.DataFrame:
    mapping = {
        "mean_qlike": ("QLIKE", "loss"),
        "log_mae": ("Log MAE", "loss"),
        "volatility_mae": ("Volatility MAE", "annualized volatility"),
        "volatility_rmse": ("Volatility RMSE", "annualized volatility"),
        "alert_high_recall": ("High-volatility recall", "rate"),
        "alert_high_precision": ("High-volatility precision", "rate"),
        "alert_balanced_accuracy": ("Alert balanced accuracy", "rate"),
        "alert_cost": ("Five-to-one alert cost", "cost per origin"),
    }
    records = []
    for model_key in ("candidate", "persistence"):
        for source, (label, unit) in mapping.items():
            records.append(
                {
                    "ModelKey": model_key,
                    "MetricKey": source,
                    "Metric": label,
                    "Value": metrics[model_key][source],
                    "Unit": unit,
                }
            )
    return pd.DataFrame(records)


def build_power_bi_frames(
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    bootstrap_samples: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build stable Power BI import tables from verified evaluation evidence."""
    forecast = predictions.copy()
    dates = pd.to_datetime(forecast["as_of"])
    forecast.insert(1, "DateKey", dates.dt.strftime("%Y%m%d").astype("int64"))
    forecast = forecast.rename(
        columns={
            "sample_id": "SampleId",
            "as_of": "AsOfDate",
            "actual_volatility": "ActualVolatility",
            "alert_threshold": "AlertThreshold",
            "true_high": "ActualHigh",
            "base_forecast": "BaseForecast",
            "candidate_forecast": "CandidateForecast",
            "persistence_forecast": "PersistenceForecast",
            "candidate_high": "CandidateHigh",
            "persistence_high": "PersistenceHigh",
            "candidate_qlike": "CandidateQLIKE",
            "persistence_qlike": "PersistenceQLIKE",
            "qlike_improvement": "QLIKEImprovement",
        }
    )
    forecast["CandidateAbsoluteError"] = (
        forecast["ActualVolatility"] - forecast["CandidateForecast"]
    ).abs()
    forecast["PersistenceAbsoluteError"] = (
        forecast["ActualVolatility"] - forecast["PersistenceForecast"]
    ).abs()
    forecast["CandidateCorrectAlert"] = (
        forecast["ActualHigh"] == forecast["CandidateHigh"]
    ).astype("int64")
    forecast["PersistenceCorrectAlert"] = (
        forecast["ActualHigh"] == forecast["PersistenceHigh"]
    ).astype("int64")
    gates = pd.DataFrame(
        [
            {
                "GateKey": name,
                "Gate": name.replace("_", " ").title(),
                "Passed": bool(passed),
            }
            for name, passed in metrics["gates"].items()
        ]
    )
    models = pd.DataFrame(
        [
            {
                "ModelKey": "candidate",
                "Model": "75% histogram / 25% persistence",
                "Role": "V8A candidate",
            },
            {
                "ModelKey": "persistence",
                "Model": "Persistence",
                "Role": "Benchmark",
            },
        ]
    )
    bootstrap = bootstrap_samples.rename(
        columns={"replicate": "Replicate", "qlike_improvement": "QLIKEImprovement"}
    )
    return {
        "DimDate": _date_dimension(predictions),
        "DimModel": models,
        "FactForecast": forecast,
        "FactModelMetric": _model_metrics(metrics),
        "FactGate": gates,
        "FactBootstrap": bootstrap,
    }


def _svg_points(values: pd.Series, width: int, height: int, maximum: float) -> str:
    if len(values) == 1:
        return f"0,{height / 2:.1f}"
    return " ".join(
        f"{index * width / (len(values) - 1):.1f},{height - value * height / maximum:.1f}"
        for index, value in enumerate(values.astype(float))
    )


def _dashboard_html(
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    bootstrap: dict[str, Any],
) -> str:
    width, height = 1040, 300
    maximum = float(
        max(
            predictions["actual_volatility"].max(),
            predictions["candidate_forecast"].max(),
            predictions["persistence_forecast"].max(),
            predictions["alert_threshold"].max(),
        )
        * 1.08
    )
    actual = _svg_points(predictions["actual_volatility"], width, height, maximum)
    candidate = _svg_points(predictions["candidate_forecast"], width, height, maximum)
    persistence = _svg_points(predictions["persistence_forecast"], width, height, maximum)
    threshold_y = height - ALERT_THRESHOLD * height / maximum
    candidate_metrics = metrics["candidate"]
    persistence_metrics = metrics["persistence"]
    reduction = 1 - candidate_metrics["mean_qlike"] / persistence_metrics["mean_qlike"]
    gates = "".join(
        "<tr><td>" + html.escape(name.replace("_", " ").title()) + "</td><td class='" +
        ("pass'>Passed" if passed else "fail'>Failed") + "</td></tr>"
        for name, passed in metrics["gates"].items()
    )
    first = html.escape(str(predictions["as_of"].iloc[0]))
    last = html.escape(str(predictions["as_of"].iloc[-1]))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPY five-session volatility shadow test</title>
<style>
:root{{--ink:#152238;--muted:#607086;--panel:#fff;--bg:#f4f7fb;--blue:#1f6feb;--orange:#d97706;--red:#c2415d;--green:#16835b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px}} h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:0 0 14px}} p{{color:var(--muted)}}
.status{{display:inline-block;background:#fff1f2;color:#9f1239;border:1px solid #fecdd3;padding:5px 10px;border-radius:999px;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}} .card,.panel{{background:var(--panel);border:1px solid #dce4ef;border-radius:12px;box-shadow:0 3px 12px #1020400a}}
.card{{padding:17px}} .label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}} .value{{font-size:26px;font-weight:750;margin-top:5px}}
.panel{{padding:20px;margin:14px 0;overflow:hidden}} svg{{width:100%;height:auto;display:block;background:linear-gradient(#fff,#fbfcfe)}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-bottom:10px}} .dot{{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:10px;border-bottom:1px solid #e7edf4;text-align:left}} th{{font-size:12px;color:var(--muted);text-transform:uppercase}} .pass{{color:var(--green);font-weight:700}} .fail{{color:var(--red);font-weight:700}}
.two{{display:grid;grid-template-columns:1.25fr .75fr;gap:14px}} .note{{border-left:4px solid var(--orange);padding-left:12px}} footer{{color:var(--muted);font-size:12px;margin-top:18px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}.two{{grid-template-columns:1fr}}}} @media(max-width:480px){{.grid{{grid-template-columns:1fr}}main{{padding:16px}}}}
</style></head><body><main>
<span class="status">Shadow testing only · promotion target not met</span>
<h1>SPY five-session volatility forecast</h1><p>{first} through {last} · {len(predictions)} forecast origins · fixed threshold {ALERT_THRESHOLD:.1%}</p>
<section class="grid">
<div class="card"><div class="label">Candidate QLIKE</div><div class="value">{candidate_metrics['mean_qlike']:.3f}</div><div>{reduction:.1%} below persistence</div></div>
<div class="card"><div class="label">QLIKE improvement</div><div class="value">+{bootstrap['point_difference']:.3f}</div><div>95% CI +{bootstrap['interval'][0]:.3f} to +{bootstrap['interval'][1]:.3f}</div></div>
<div class="card"><div class="label">High-vol recall</div><div class="value">{candidate_metrics['true_positive']}/{metrics['high_support']}</div><div>Persistence: {persistence_metrics['true_positive']}/{metrics['high_support']}</div></div>
<div class="card"><div class="label">Alert cost</div><div class="value">{candidate_metrics['alert_cost']:.3f}</div><div>Persistence: {persistence_metrics['alert_cost']:.3f}</div></div>
</section>
<section class="panel"><h2>Forecasts and realized volatility</h2><div class="legend"><span><i class="dot" style="background:#152238"></i>Actual</span><span><i class="dot" style="background:#1f6feb"></i>Candidate</span><span><i class="dot" style="background:#d97706"></i>Persistence</span><span><i class="dot" style="background:#c2415d"></i>Alert threshold</span></div>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Actual and forecast annualized volatility time series"><line x1="0" y1="{threshold_y:.1f}" x2="{width}" y2="{threshold_y:.1f}" stroke="#c2415d" stroke-width="2" stroke-dasharray="8 6"/><polyline points="{persistence}" fill="none" stroke="#d97706" stroke-width="1.6" opacity=".8"/><polyline points="{candidate}" fill="none" stroke="#1f6feb" stroke-width="2.3"/><polyline points="{actual}" fill="none" stroke="#152238" stroke-width="2"/></svg></section>
<section class="two"><div class="panel"><h2>Frozen promotion requirements</h2><table><thead><tr><th>Requirement</th><th>Result</th></tr></thead><tbody>{gates}</tbody></table></div>
<div class="panel"><h2>Interpretation</h2><p class="note">Continuous forecast accuracy improved materially, but the fixed alert rule caught one fewer high-volatility outcome than persistence. The model remains eligible for shadow testing only.</p><table><tr><th></th><th>Candidate</th><th>Persistence</th></tr><tr><td>False positives</td><td>{candidate_metrics['false_positive']}</td><td>{persistence_metrics['false_positive']}</td></tr><tr><td>False negatives</td><td>{candidate_metrics['false_negative']}</td><td>{persistence_metrics['false_negative']}</td></tr></table></div></section>
<footer>Source: verified V8A evaluation artifacts. Forecast horizon: next five sessions. Values are annualized volatility. Generated {html.escape(utc_now())}.</footer>
</main></body></html>"""


def _schema_document() -> dict[str, Any]:
    return {
        "format": "Power BI import star schema",
        "tables": {
            "DimDate": {"key": "DateKey"},
            "DimModel": {"key": "ModelKey"},
            "FactForecast": {"key": "SampleId", "date_key": "DateKey"},
            "FactModelMetric": {"model_key": "ModelKey"},
            "FactGate": {"key": "GateKey"},
            "FactBootstrap": {"key": "Replicate"},
        },
        "relationships": [
            "DimDate[DateKey] 1:* FactForecast[DateKey]",
            "DimModel[ModelKey] 1:* FactModelMetric[ModelKey]",
        ],
        "suggested_measures": {
            "Candidate QLIKE": "AVERAGE(FactForecast[CandidateQLIKE])",
            "Persistence QLIKE": "AVERAGE(FactForecast[PersistenceQLIKE])",
            "QLIKE Improvement": "[Persistence QLIKE] - [Candidate QLIKE]",
            "Candidate High Recall": "DIVIDE(SUMX(FactForecast, FactForecast[CandidateHigh] * FactForecast[ActualHigh]), SUM(FactForecast[ActualHigh]))",
            "Candidate Alert Rate": "AVERAGE(FactForecast[CandidateHigh])",
            "Actual High Rate": "AVERAGE(FactForecast[ActualHigh])",
        },
    }


def export_ytd_deployment(evaluation_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Export verified V8A evidence to a visual report and Power BI import tables."""
    evaluation, output = Path(evaluation_dir), Path(output_dir)
    verified = verify_ytd_evaluation(evaluation)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite deployment export: {output}")
    metrics = cast(dict[str, Any], json.loads((evaluation / "metrics.json").read_text()))
    bootstrap = cast(dict[str, Any], json.loads((evaluation / "bootstrap.json").read_text()))
    predictions = pd.read_csv(
        evaluation / "evaluation_predictions.csv", float_precision="round_trip"
    )
    samples = pd.read_csv(evaluation / "bootstrap_samples.csv", float_precision="round_trip")
    frames = build_power_bi_frames(predictions, metrics, samples)
    pbi_dir = output / "power_bi"
    dashboard_dir = output / "dashboard"
    pbi_dir.mkdir(parents=True)
    dashboard_dir.mkdir(parents=True)
    files: dict[str, str] = {}
    for name, frame in frames.items():
        path = pbi_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        files[str(path.relative_to(output))] = sha256(path.read_bytes())
    schema_path = pbi_dir / "model_schema.json"
    schema_path.write_text(json.dumps(_schema_document(), indent=2, sort_keys=True) + "\n")
    files[str(schema_path.relative_to(output))] = sha256(schema_path.read_bytes())
    dictionary = pd.DataFrame(
        [
            ("FactForecast", "ActualVolatility", "Realized next-five-session annualized volatility"),
            ("FactForecast", "CandidateForecast", "75% model / 25% persistence variance blend"),
            ("FactForecast", "PersistenceForecast", "Trailing five-session annualized volatility"),
            ("FactForecast", "QLIKEImprovement", "Persistence QLIKE minus candidate QLIKE; positive favors candidate"),
            ("FactForecast", "ActualHigh", "1 when realized volatility is strictly above the fixed threshold"),
            ("FactGate", "Passed", "Frozen V8A requirement result; all requirements must pass"),
            ("FactBootstrap", "QLIKEImprovement", "Paired circular block-bootstrap replicate"),
        ],
        columns=["Table", "Field", "Definition"],
    )
    dictionary_path = pbi_dir / "data_dictionary.csv"
    dictionary.to_csv(dictionary_path, index=False)
    files[str(dictionary_path.relative_to(output))] = sha256(dictionary_path.read_bytes())
    dashboard_path = dashboard_dir / "index.html"
    dashboard_path.write_text(_dashboard_html(predictions, metrics, bootstrap))
    files[str(dashboard_path.relative_to(output))] = sha256(dashboard_path.read_bytes())
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "source_evaluation_manifest_sha256": sha256(
            (evaluation / "manifest.json").read_bytes()
        ),
        "source_research_status": verified["research_status"],
        "deployment_status": "shadow_only",
        "rows": len(predictions),
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

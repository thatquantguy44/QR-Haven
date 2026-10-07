"""V8 development-only continuous volatility forecast selection."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, sha256, utc_now
from qr_haven.ml.classification.evaluation import classification_metrics
from qr_haven.ml.volatility.challenger_data import (
    CHALLENGER_FEATURES,
    TIINGO_PROTOCOL,
    load_challenger_development,
)
from qr_haven.ml.volatility.challenger_models import HAR_FEATURES, ewma_scores
from qr_haven.ml.volatility.history_experiments import adaptive_thresholds
from qr_haven.ml.volatility.persistence import environment_versions, source_identity

PROTOCOL = "volatility-continuous-v1"
OUTER_YEARS = tuple(range(2013, 2024))
TRAINING_WINDOW_YEARS = 12
EPSILON = 1e-12
BLEND_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
ALERT_MISS_COST = 5.0


@dataclass(frozen=True)
class ContinuousBase:
    """One fixed continuous forecast family."""

    base_id: str
    kind: str
    order: int
    decay: float | None = None


def continuous_bases() -> tuple[ContinuousBase, ...]:
    return (
        ContinuousBase("hist_gradient_boosting_regression", "histogram", 0),
        ContinuousBase("har_rv", "har", 1),
        ContinuousBase("ewma_090", "ewma", 2, 0.90),
        ContinuousBase("ewma_094", "ewma", 3, 0.94),
        ContinuousBase("ewma_097", "ewma", 4, 0.97),
    )


def continuous_training_ids(
    observations: pd.DataFrame, validation_year: int
) -> tuple[str, ...]:
    validation = observations.loc[observations["as_of"].dt.year.eq(validation_year)]
    if validation.empty:
        raise ValueError(f"No V8 validation origins for {validation_year}")
    boundary = pd.Timestamp(validation["as_of"].iloc[0])
    rows = observations.loc[
        (observations["as_of"] >= boundary - pd.DateOffset(years=TRAINING_WINDOW_YEARS))
        & (observations["as_of"] < boundary)
        & (observations["label_end"] < boundary)
    ]
    if rows.empty or not (rows["label_end"] < boundary).all():
        raise ValueError("V8 training membership is empty or crosses validation")
    return tuple(rows.index.astype(str))


def _inverse_log_variance(values: np.ndarray) -> npt.NDArray[np.float64]:
    variance = np.maximum(np.exp(values) - EPSILON, EPSILON)
    return np.asarray(np.sqrt(variance), dtype=np.float64)


def _fit_forecast(
    observations: pd.DataFrame,
    base: ContinuousBase,
    validation_year: int,
    ewma: pd.Series | None,
) -> tuple[pd.Series, dict[str, Any], pd.DataFrame]:
    validation = observations.loc[observations["as_of"].dt.year.eq(validation_year)]
    boundary = pd.Timestamp(validation["as_of"].iloc[0])
    training_ids = continuous_training_ids(observations, validation_year)
    training = observations.loc[list(training_ids)]
    if base.kind == "ewma":
        if ewma is None:
            raise ValueError("V8 EWMA requires a causal precomputed forecast")
        forecast = ewma.loc[validation.index].astype("float64").rename("base_forecast")
        fitted = False
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from threadpoolctl import threadpool_limits

        target = np.log(np.square(training["forward_vol_5"].to_numpy(float)) + EPSILON)
        if base.kind == "histogram":
            estimator: Any = HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_leaf_nodes=7,
                l2_regularization=1.0,
                max_iter=300,
                early_stopping=False,
                random_state=5402,
            )
            train_features = training.loc[:, list(CHALLENGER_FEATURES)]
            validation_features = validation.loc[:, list(CHALLENGER_FEATURES)]
        elif base.kind == "har":
            estimator = LinearRegression()
            train_features = np.log(
                np.square(training.loc[:, list(HAR_FEATURES)].to_numpy(float)) + EPSILON
            )
            validation_features = np.log(
                np.square(validation.loc[:, list(HAR_FEATURES)].to_numpy(float)) + EPSILON
            )
        else:
            raise ValueError(f"Unknown V8 base kind: {base.kind}")
        with threadpool_limits(limits=1):
            estimator.fit(train_features, target)
            predicted = estimator.predict(validation_features)
        forecast = pd.Series(
            _inverse_log_variance(np.asarray(predicted, dtype=float)),
            index=validation.index,
            name="base_forecast",
        )
        fitted = True
    if not np.isfinite(forecast.to_numpy()).all() or (forecast <= 0).any():
        raise ValueError("V8 forecasts must be finite and strictly positive")
    audit = {
        "base_id": base.base_id,
        "validation_year": validation_year,
        "fitted_model": fitted,
        "training_rows": len(training_ids) if fitted else 0,
        "training_start": training["as_of"].iloc[0] if fitted else None,
        "training_end": training["as_of"].iloc[-1] if fitted else None,
        "max_training_label_end": training["label_end"].max() if fitted else None,
        "validation_start": boundary,
    }
    membership = pd.DataFrame(
        {"sample_id": training.index if fitted else pd.Index([], dtype=str)}
    )
    membership.insert(0, "validation_year", validation_year)
    membership.insert(0, "base_id", base.base_id)
    return forecast, audit, membership


def blend_variance(
    base_forecast: pd.Series, persistence: pd.Series, model_weight: float
) -> pd.Series:
    if (
        model_weight not in BLEND_WEIGHTS
        or not base_forecast.index.equals(persistence.index)
        or (base_forecast <= 0).any()
        or (persistence <= 0).any()
    ):
        raise ValueError("V8 variance-blend inputs are invalid")
    variance = model_weight * np.square(base_forecast) + (1 - model_weight) * np.square(persistence)
    return pd.Series(np.sqrt(variance), index=base_forecast.index, name="forecast_volatility")


def qlike_rows(actual_volatility: pd.Series, forecast_volatility: pd.Series) -> pd.Series:
    if (
        not actual_volatility.index.equals(forecast_volatility.index)
        or (actual_volatility <= 0).any()
        or (forecast_volatility <= 0).any()
    ):
        raise ValueError("QLIKE requires aligned positive volatility values")
    ratio = np.square(actual_volatility) / np.maximum(np.square(forecast_volatility), EPSILON)
    values = ratio - np.log(ratio) - 1
    if not np.isfinite(values.to_numpy()).all() or (values < -1e-12).any():
        raise ValueError("QLIKE rows must be finite and nonnegative")
    return values.clip(lower=0).rename("qlike")


def _forecast_metrics(
    actual: pd.Series,
    forecast: pd.Series,
    threshold: pd.Series,
) -> dict[str, Any]:
    if not actual.index.equals(forecast.index) or not actual.index.equals(threshold.index):
        raise ValueError("V8 metrics require aligned inputs")
    qlike = qlike_rows(actual, forecast)
    log_error = np.log(forecast) - np.log(actual)
    error = forecast - actual
    alert_target = (actual > threshold).astype("int64")
    alert_prediction = pd.DataFrame(
        {
            "predicted_class": (forecast > threshold).astype("int64"),
            "score_class_1": forecast / threshold,
            "score_kind": "forecast_to_adaptive_threshold_ratio",
        },
        index=actual.index,
    )
    alert = classification_metrics(alert_target, alert_prediction)
    tn, fp = alert["confusion_matrix"][0]
    fn, tp = alert["confusion_matrix"][1]
    return {
        "rows": len(actual),
        "mean_qlike": float(qlike.mean()),
        "log_mae": float(np.abs(log_error).mean()),
        "volatility_mae": float(np.abs(error).mean()),
        "volatility_rmse": float(np.sqrt(np.square(error).mean())),
        "alert_accuracy": alert["accuracy"],
        "alert_balanced_accuracy": alert["balanced_accuracy"],
        "alert_high_precision": alert["per_class"]["1"]["precision"],
        "alert_high_recall": alert["per_class"]["1"]["recall"],
        "alert_high_f1": alert["per_class"]["1"]["f1"],
        "alert_cost": float((fp + ALERT_MISS_COST * fn) / len(actual)),
        "normal_support": tn + fp,
        "high_support": tp + fn,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "true_positive": tp,
        "diagnostics": json.dumps(alert["diagnostics"]),
    }


def _candidate_id(base_id: str, model_weight: float) -> str:
    return f"{base_id}_w{int(round(model_weight * 100)):03d}"


def _rank(results: pd.DataFrame) -> pd.DataFrame:
    expected = set(OUTER_YEARS)
    orders = {base.base_id: base.order for base in continuous_bases()}
    orders["persistence"] = len(orders)
    records: list[dict[str, Any]] = []
    for (candidate_id, base_id, weight, frequency), rows in results.groupby(
        ["candidate_id", "base_id", "model_weight", "frequency"], sort=False
    ):
        if len(rows) != len(expected) or set(rows["validation_year"]) != expected:
            raise ValueError("V8 ranking requires every frozen outer fold")
        records.append(
            {
                "candidate_id": candidate_id,
                "base_id": base_id,
                "model_weight": weight,
                "frequency": frequency,
                "base_order": orders[str(base_id)],
                "weight_order": (
                    BLEND_WEIGHTS.index(float(weight)) if base_id != "persistence" else 0
                ),
                **{
                    f"mean_yearly_{name}": float(rows[name].mean())
                    for name in (
                        "mean_qlike",
                        "log_mae",
                        "volatility_mae",
                        "volatility_rmse",
                        "alert_balanced_accuracy",
                        "alert_high_recall",
                        "alert_cost",
                    )
                },
                "false_positive": int(rows["false_positive"].sum()),
                "false_negative": int(rows["false_negative"].sum()),
            }
        )
    ranking = pd.DataFrame(records)
    persistence = ranking.loc[ranking["candidate_id"].eq("persistence")].set_index("frequency")
    ranking["qlike_improvement_vs_persistence"] = [
        float(persistence.loc[frequency, "mean_yearly_mean_qlike"]) - loss
        for frequency, loss in zip(
            ranking["frequency"], ranking["mean_yearly_mean_qlike"], strict=True
        )
    ]
    ranking = ranking.sort_values(
        [
            "frequency",
            "mean_yearly_mean_qlike",
            "mean_yearly_log_mae",
            "base_order",
            "weight_order",
        ],
        ascending=[True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking["rank"] = ranking.groupby("frequency").cumcount() + 1
    return ranking


def _pooled(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for candidate_id, rows in predictions.groupby("candidate_id", sort=False):
        rows = rows.sort_values(["validation_year", "as_of"]).set_index("sample_id")
        for frequency, subset in (
            ("daily", rows),
            ("every_fifth", rows.loc[rows["every_fifth"]]),
        ):
            records.append(
                {
                    "candidate_id": candidate_id,
                    "base_id": subset["base_id"].iloc[0],
                    "model_weight": subset["model_weight"].iloc[0],
                    "frequency": frequency,
                    **_forecast_metrics(
                        subset["actual_volatility"],
                        subset["forecast_volatility"],
                        subset["adaptive_threshold"],
                    ),
                }
            )
    return pd.DataFrame(records)


def _report(ranking: pd.DataFrame, selection: dict[str, Any]) -> bytes:
    daily = ranking.loc[ranking["frequency"].eq("daily")]
    status = (
        "fitted candidate advances" if selection["advances"] else "no fitted candidate advances"
    )
    lines = [
        "# V8 continuous volatility development selection",
        "",
        f"Status: **{status}**.",
        "",
        "The comparison used only Tiingo SPY development observations through 2023. It did not "
        "load 2024–2025 outcomes or evaluation artifacts, and it did not fit a final model.",
        "",
        f"Selected candidate: `{selection['candidate_id']}`.",
        "",
        "| Rank | Candidate | Mean yearly QLIKE | Improvement vs persistence | Log MAE | Vol MAE | "
        "Alert recall | Alert cost |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in daily.to_dict("records"):
        lines.append(
            f"| {row['rank']} | {row['candidate_id']} | "
            f"{row['mean_yearly_mean_qlike']:.6f} | "
            f"{row['qlike_improvement_vs_persistence']:+.6f} | "
            f"{row['mean_yearly_log_mae']:.6f} | "
            f"{row['mean_yearly_volatility_mae']:.6f} | "
            f"{row['mean_yearly_alert_high_recall']:.4f} | "
            f"{row['mean_yearly_alert_cost']:.4f} |"
        )
    lines.extend(
        [
            "",
            "QLIKE and log MAE are computed on continuous five-session realized volatility. "
            "Alerts compare each forecast with the same causal trailing-three-year threshold; "
            "their cost assigns five units to a miss and one to a false alarm.",
            "",
            "These results are development selection evidence. A fitted winner must remain frozen "
            "and pass every predeclared gate on untouched 2026 data before research promotion.",
            "",
        ]
    )
    return "\n".join(lines).encode()


def run_continuous_experiment(
    dataset_dir: Path,
    output_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    dataset, output = Path(dataset_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable V8 experiment: {output}")
    observations, dataset_manifest = load_challenger_development(dataset)
    if (
        dataset_manifest.get("protocol") != TIINGO_PROTOCOL
        or dataset_manifest.get("profile_id") != "tiingo-spy-v1"
    ):
        raise ValueError("V8 requires the verified tiingo-spy-v1 challenger development dataset")
    config = {
        "protocol": PROTOCOL,
        "status": "development-only selection",
        "profile_id": "tiingo-spy-v1",
        "outer_years": list(OUTER_YEARS),
        "training_window_years": TRAINING_WINDOW_YEARS,
        "bases": [asdict(base) for base in continuous_bases()],
        "blend_weights": list(BLEND_WEIGHTS),
        "target": "annualized forward_vol_5",
        "primary_metric": "mean yearly QLIKE ascending",
        "secondary_metric": "mean yearly absolute log-volatility error ascending",
        "alert_miss_cost": ALERT_MISS_COST,
        "feature_order": list(CHALLENGER_FEATURES),
        "evaluation_years_loaded": [],
        "final_model_fitted": False,
        "future_evaluation_year": 2026,
        "future_bootstrap_seed": 5412,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "stage": "continuous volatility development selection",
        "state": "running",
        "profile_id": "tiingo-spy-v1",
        "created_at_utc": utc_now(),
        "dataset_manifest_sha256": sha256((dataset / "manifest.json").read_bytes()),
        "source_sha256": dataset_manifest["source_sha256"],
        "environment": environment_versions(),
        "code": source_identity(),
        "evaluation_outcomes_opened": False,
        "evaluation_artifacts_loaded": False,
        "final_model_fitted": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_write(output / "config.json", json_bytes(config))
    atomic_write(output / "manifest.json", json_bytes(manifest))
    started = time.perf_counter()
    try:
        threshold_audit = adaptive_thresholds(observations)
        thresholds = threshold_audit["adaptive_threshold"]
        ewmas = {
            base.base_id: ewma_scores(observations, float(base.decay))
            for base in continuous_bases()
            if base.decay is not None
        }
        result_rows: list[dict[str, Any]] = []
        prediction_rows: list[pd.DataFrame] = []
        audits: list[dict[str, Any]] = []
        memberships: list[pd.DataFrame] = []
        for base in continuous_bases():
            for year in OUTER_YEARS:
                if progress:
                    progress(f"V8 {base.base_id}: chronological fold {year}")
                forecast, audit, membership = _fit_forecast(
                    observations, base, year, ewmas.get(base.base_id)
                )
                audits.append(audit)
                memberships.append(membership)
                validation = observations.loc[forecast.index]
                actual = validation["forward_vol_5"].astype("float64")
                threshold = thresholds.loc[forecast.index].astype("float64")
                persistence = validation["trailing_vol_5"].astype("float64")
                for weight in BLEND_WEIGHTS:
                    blended = blend_variance(forecast, persistence, weight)
                    candidate_id = _candidate_id(base.base_id, weight)
                    common = {
                        "candidate_id": candidate_id,
                        "base_id": base.base_id,
                        "model_weight": weight,
                        "validation_year": year,
                    }
                    for frequency, positions in (
                        ("daily", np.arange(len(actual))),
                        ("every_fifth", np.arange(0, len(actual), 5)),
                    ):
                        result_rows.append(
                            {
                                **common,
                                "frequency": frequency,
                                **_forecast_metrics(
                                    actual.iloc[positions],
                                    blended.iloc[positions],
                                    threshold.iloc[positions],
                                ),
                            }
                        )
                    recorded = pd.DataFrame(
                        {
                            "actual_volatility": actual,
                            "base_forecast": forecast,
                            "persistence_forecast": persistence,
                            "forecast_volatility": blended,
                            "adaptive_threshold": threshold,
                            "true_high": (actual > threshold).astype("int64"),
                            "predicted_high": (blended > threshold).astype("int64"),
                            "qlike": qlike_rows(actual, blended),
                            "as_of": validation["as_of"],
                            "every_fifth": np.arange(len(actual)) % 5 == 0,
                            **{name: value for name, value in common.items()},
                        }
                    )
                    prediction_rows.append(recorded.reset_index(names="sample_id"))
        for year in OUTER_YEARS:
            validation = observations.loc[observations["as_of"].dt.year.eq(year)]
            actual = validation["forward_vol_5"].astype("float64")
            persistence = validation["trailing_vol_5"].astype("float64")
            threshold = thresholds.loc[validation.index].astype("float64")
            common = {
                "candidate_id": "persistence",
                "base_id": "persistence",
                "model_weight": 0.0,
                "validation_year": year,
            }
            for frequency, positions in (
                ("daily", np.arange(len(actual))),
                ("every_fifth", np.arange(0, len(actual), 5)),
            ):
                result_rows.append(
                    {
                        **common,
                        "frequency": frequency,
                        **_forecast_metrics(
                            actual.iloc[positions],
                            persistence.iloc[positions],
                            threshold.iloc[positions],
                        ),
                    }
                )
            recorded = pd.DataFrame(
                {
                    "actual_volatility": actual,
                    "base_forecast": persistence,
                    "persistence_forecast": persistence,
                    "forecast_volatility": persistence,
                    "adaptive_threshold": threshold,
                    "true_high": (actual > threshold).astype("int64"),
                    "predicted_high": (persistence > threshold).astype("int64"),
                    "qlike": qlike_rows(actual, persistence),
                    "as_of": validation["as_of"],
                    "every_fifth": np.arange(len(actual)) % 5 == 0,
                    **{name: value for name, value in common.items()},
                }
            )
            prediction_rows.append(recorded.reset_index(names="sample_id"))
        results = pd.DataFrame(result_rows)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        ranking = _rank(results)
        pooled = _pooled(predictions)
        selected = ranking.loc[(ranking["frequency"] == "daily") & (ranking["rank"] == 1)].iloc[0]
        advances = bool(
            selected["candidate_id"] != "persistence"
            and selected["qlike_improvement_vs_persistence"] > 0
        )
        selection = {
            "candidate_id": str(selected["candidate_id"]),
            "base_id": str(selected["base_id"]),
            "model_weight": float(selected["model_weight"]),
            "mean_yearly_qlike": float(selected["mean_yearly_mean_qlike"]),
            "qlike_improvement_vs_persistence": float(
                selected["qlike_improvement_vs_persistence"]
            ),
            "advances": advances,
        }
        payloads = {
            "adaptive_thresholds.csv": threshold_audit.reset_index().to_csv(index=False).encode(),
            "cv_results.csv": results.to_csv(index=False).encode(),
            "cv_predictions.csv": predictions.to_csv(index=False).encode(),
            "ranking.csv": ranking.to_csv(index=False).encode(),
            "pooled_metrics.csv": pooled.to_csv(index=False).encode(),
            "fit_audit.csv": pd.DataFrame(audits).to_csv(index=False).encode(),
            "training_membership.csv": pd.concat(memberships, ignore_index=True)
            .to_csv(index=False)
            .encode(),
            "selection.json": json_bytes(selection),
            "report.md": _report(ranking, selection),
        }
        for name, payload in payloads.items():
            atomic_write(output / name, payload)
        payloads["config.json"] = (output / "config.json").read_bytes()
        manifest.update(
            {
                "state": "complete",
                "research_status": "candidate_selected" if advances else "no_candidate_advances",
                "selected_candidate": selection["candidate_id"],
                "selected_base": selection["base_id"],
                "selected_model_weight": selection["model_weight"],
                "candidate_advances": advances,
                "evaluation_outcomes_opened": False,
                "evaluation_artifacts_loaded": False,
                "final_model_fitted": False,
                "seconds": time.perf_counter() - started,
                "files": {name: sha256(payload) for name, payload in payloads.items()},
            }
        )
        atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
    except Exception as exc:
        manifest.update({"state": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
        raise
    return manifest


def verify_continuous_experiment(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    value = json.loads((root / "manifest.json").read_text())
    if (
        not isinstance(value, dict)
        or value.get("protocol") != PROTOCOL
        or value.get("state") != "complete"
    ):
        raise ValueError("Unsupported or incomplete V8 continuous experiment")
    required = {
        "config.json",
        "adaptive_thresholds.csv",
        "cv_results.csv",
        "cv_predictions.csv",
        "ranking.csv",
        "pooled_metrics.csv",
        "fit_audit.csv",
        "training_membership.csv",
        "selection.json",
        "report.md",
    }
    hashes = value.get("files")
    if not isinstance(hashes, dict) or not required <= hashes.keys():
        raise ValueError("V8 manifest lacks required file hashes")
    for name, digest in hashes.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid V8 artifact path")
        if sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"V8 artifact hash mismatch: {name}")
    return cast(dict[str, Any], value)

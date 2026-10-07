"""V7 development-only adaptive-target and training-history experiment."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, sha256, utc_now
from qr_haven.ml.classification.evaluation import classification_metrics
from qr_haven.ml.volatility.challenger_data import (
    CHALLENGER_FEATURES,
    TIINGO_PROTOCOL,
    load_challenger_development,
)
from qr_haven.ml.volatility.challenger_models import ENSEMBLE_WEIGHTS, calibrated_scores, fit_platt
from qr_haven.ml.volatility.models import (
    VolatilityCandidate,
    fit_estimator,
    inverse_frequency_weights,
    make_estimator,
)
from qr_haven.ml.volatility.persistence import environment_versions, source_identity

PROTOCOL = "volatility-history-adaptation-v1"
OUTER_YEARS = tuple(range(2013, 2024))
CALIBRATION_YEARS = 3
THRESHOLD_LOOKBACK_YEARS = 3
THRESHOLD_QUANTILE = 0.75
MIN_THRESHOLD_HISTORY = 500
DECAY_HALF_LIFE_YEARS = 5.0


@dataclass(frozen=True)
class HistoryDesign:
    """One fixed training-membership and weighting rule."""

    design_id: str
    order: int
    window_years: int | None
    half_life_years: float | None = None


def history_designs() -> tuple[HistoryDesign, ...]:
    return (
        HistoryDesign("rolling_5y", 0, 5),
        HistoryDesign("rolling_8y", 1, 8),
        HistoryDesign("rolling_12y", 2, 12),
        HistoryDesign("expanding", 3, None),
        HistoryDesign("expanding_decay_5y", 4, None, DECAY_HALF_LIFE_YEARS),
    )


def _histogram_candidate() -> VolatilityCandidate:
    return VolatilityCandidate(
        candidate_id="hist_gradient_boosting_01",
        family="hist_gradient_boosting",
        family_order=2,
        grid_order=1,
        parameters={
            "learning_rate": 0.03,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
        },
    )


def adaptive_thresholds(observations: pd.DataFrame) -> pd.DataFrame:
    """Calculate causal trailing-three-year thresholds and their exact audit bounds."""
    required = {"as_of", "label_end", "forward_vol_5"}
    if not required <= set(observations) or observations.empty or not observations.index.is_unique:
        raise ValueError("Adaptive thresholds require unique, nonempty volatility observations")
    if not observations["as_of"].is_monotonic_increasing:
        raise ValueError("Adaptive threshold observations must be chronological")
    values = observations["forward_vol_5"].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Adaptive threshold outcomes must be finite and nonnegative")
    records: list[dict[str, Any]] = []
    for sample_id, row in observations.iterrows():
        origin = pd.Timestamp(row["as_of"])
        start = origin - pd.DateOffset(years=THRESHOLD_LOOKBACK_YEARS)
        history = observations.loc[
            (observations["as_of"] >= start) & (observations["label_end"] < origin)
        ]
        if len(history) < MIN_THRESHOLD_HISTORY:
            continue
        threshold = float(
            np.quantile(
                history["forward_vol_5"].to_numpy(float),
                THRESHOLD_QUANTILE,
                method="linear",
            )
        )
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Adaptive threshold must be finite and nonnegative")
        records.append(
            {
                "sample_id": str(sample_id),
                "as_of": origin,
                "adaptive_threshold": threshold,
                "threshold_history_rows": len(history),
                "threshold_history_start": history["as_of"].iloc[0],
                "threshold_history_end": history["as_of"].iloc[-1],
                "max_history_label_end": history["label_end"].max(),
            }
        )
    if not records:
        raise ValueError("No observation has enough history for an adaptive threshold")
    result = pd.DataFrame(records).set_index("sample_id")
    result.index.name = observations.index.name or "sample_id"
    if not (result["max_history_label_end"] < result["as_of"]).all():
        raise ValueError("Adaptive threshold history crosses an origin")
    return result


def _eligible_observations(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = adaptive_thresholds(observations)
    rows = observations.loc[thresholds.index].copy()
    rows["adaptive_threshold"] = thresholds["adaptive_threshold"]
    rows["adaptive_target"] = (
        rows["forward_vol_5"] > rows["adaptive_threshold"]
    ).astype("int64")
    return rows, thresholds


def history_training_ids(
    observations: pd.DataFrame, validation_year: int, design: HistoryDesign
) -> tuple[str, ...]:
    validation = observations.loc[observations["as_of"].dt.year.eq(validation_year)]
    if validation.empty:
        raise ValueError(f"No V7 validation origins for {validation_year}")
    boundary = pd.Timestamp(validation["as_of"].iloc[0])
    mask = (observations["as_of"] < boundary) & (observations["label_end"] < boundary)
    if design.window_years is not None:
        mask &= observations["as_of"] >= boundary - pd.DateOffset(years=design.window_years)
    rows = observations.loc[mask]
    if rows.empty or set(rows["adaptive_target"]) != {0, 1}:
        raise ValueError(f"V7 training membership is empty or single-class for {design.design_id}")
    return tuple(rows.index.astype(str))


def history_weights(
    observations: pd.DataFrame,
    training_ids: tuple[str, ...],
    boundary: pd.Timestamp,
    design: HistoryDesign,
) -> pd.Series:
    rows = observations.loc[list(training_ids)]
    target = rows["adaptive_target"].astype("int64")
    weights = inverse_frequency_weights(target)
    if design.half_life_years is not None:
        age_days = (boundary - rows["as_of"]).dt.total_seconds() / 86_400
        decay = np.power(0.5, age_days / (365.2425 * design.half_life_years))
        weights = weights * decay
        weights = weights / float(weights.mean())
    weights = weights.astype("float64").rename("sample_weight")
    if (
        not weights.index.equals(rows.index)
        or not np.isfinite(weights.to_numpy()).all()
        or (weights <= 0).any()
        or not math.isclose(float(weights.mean()), 1.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("V7 sample weights must be aligned, positive, finite, and mean one")
    return weights


def _raw_fold(
    observations: pd.DataFrame, design: HistoryDesign, validation_year: int
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    validation = observations.loc[observations["as_of"].dt.year.eq(validation_year)]
    boundary = pd.Timestamp(validation["as_of"].iloc[0])
    training_ids = history_training_ids(observations, validation_year, design)
    training = observations.loc[list(training_ids)]
    weights = history_weights(observations, training_ids, boundary, design)
    estimator = fit_estimator(
        make_estimator(_histogram_candidate()),
        training.loc[:, list(CHALLENGER_FEATURES)],
        training["adaptive_target"].astype("int64"),
        weights,
    )
    classes = list(estimator.classes_)
    raw = estimator.predict_proba(validation.loc[:, list(CHALLENGER_FEATURES)])[
        :, classes.index(1)
    ]
    frame = pd.DataFrame(
        {
            "raw_score": raw,
            "true_class": validation["adaptive_target"].to_numpy(),
            "trailing_vol_5": validation["trailing_vol_5"].to_numpy(),
            "adaptive_threshold": validation["adaptive_threshold"].to_numpy(),
            "as_of": validation["as_of"].to_numpy(),
            "validation_year": validation_year,
        },
        index=validation.index,
    )
    audit = {
        "design_id": design.design_id,
        "validation_year": validation_year,
        "training_rows": len(training),
        "training_start": training["as_of"].iloc[0],
        "training_end": training["as_of"].iloc[-1],
        "max_training_label_end": training["label_end"].max(),
        "validation_start": boundary,
        "normal_rows": int((training["adaptive_target"] == 0).sum()),
        "high_rows": int((training["adaptive_target"] == 1).sum()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    }
    membership = pd.DataFrame(
        {
            "design_id": design.design_id,
            "validation_year": validation_year,
            "sample_id": training.index,
            "as_of": training["as_of"].to_numpy(),
            "label_end": training["label_end"].to_numpy(),
            "adaptive_target": training["adaptive_target"].to_numpy(),
            "sample_weight": weights.to_numpy(),
        }
    )
    return frame, audit, membership


def _ensemble(probability: pd.Series, rows: pd.DataFrame, weight: float) -> pd.DataFrame:
    if weight not in ENSEMBLE_WEIGHTS or not probability.index.equals(rows.index):
        raise ValueError("V7 ensemble inputs are invalid")
    persistence = (rows["trailing_vol_5"] > rows["adaptive_threshold"]).astype("float64")
    score = weight * probability + (1 - weight) * persistence
    return pd.DataFrame(
        {
            "predicted_class": (score > 0.5).astype("int64"),
            "score_class_1": score,
            "score_kind": "calibrated_model_plus_adaptive_persistence",
        },
        index=rows.index,
    )


def _metric_row(target: pd.Series, prediction: pd.DataFrame) -> dict[str, Any]:
    value = classification_metrics(target, prediction)
    tn, fp = value["confusion_matrix"][0]
    fn, tp = value["confusion_matrix"][1]
    return {
        "rows": len(target),
        "accuracy": value["accuracy"],
        "balanced_accuracy": value["balanced_accuracy"],
        "macro_f1": value["f1_macro"],
        "high_precision": value["per_class"]["1"]["precision"],
        "high_recall": value["per_class"]["1"]["recall"],
        "high_f1": value["per_class"]["1"]["f1"],
        "roc_auc": value["roc_auc"],
        "average_precision": value["average_precision"],
        "normal_support": tn + fp,
        "high_support": tp + fn,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "diagnostics": json.dumps(value["diagnostics"]),
    }


def _rank(results: pd.DataFrame) -> pd.DataFrame:
    expected = set(OUTER_YEARS)
    design_order = {design.design_id: design.order for design in history_designs()}
    records: list[dict[str, Any]] = []
    for (design_id, weight, frequency), rows in results.groupby(
        ["design_id", "model_weight", "frequency"], sort=False
    ):
        if len(rows) != len(expected) or set(rows["validation_year"]) != expected:
            raise ValueError("V7 ranking requires every frozen outer fold")
        records.append(
            {
                "design_id": design_id,
                "model_weight": weight,
                "frequency": frequency,
                "design_order": design_order[str(design_id)],
                "weight_order": ENSEMBLE_WEIGHTS.index(float(weight)),
                **{
                    f"mean_{name}": float(rows[name].mean())
                    for name in (
                        "balanced_accuracy",
                        "macro_f1",
                        "accuracy",
                        "high_precision",
                        "high_recall",
                    )
                },
                "false_positive": int(rows["false_positive"].sum()),
                "false_negative": int(rows["false_negative"].sum()),
            }
        )
    ranking = pd.DataFrame(records).sort_values(
        [
            "frequency",
            "mean_balanced_accuracy",
            "mean_macro_f1",
            "design_order",
            "weight_order",
        ],
        ascending=[True, False, False, True, True],
        kind="stable",
    )
    ranking["rank"] = ranking.groupby("frequency").cumcount() + 1
    return ranking.reset_index(drop=True)


def _pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (design_id, weight), rows in predictions.groupby(
        ["design_id", "model_weight"], sort=False
    ):
        rows = rows.sort_values(["validation_year", "as_of"]).set_index("sample_id")
        for frequency, subset in (
            ("daily", rows),
            ("every_fifth", rows.loc[rows["every_fifth"]]),
        ):
            records.append(
                {
                    "design_id": design_id,
                    "model_weight": weight,
                    "frequency": frequency,
                    **_metric_row(subset["true_class"], subset),
                }
            )
    return pd.DataFrame(records)


def _report(
    ranking: pd.DataFrame, selection: dict[str, Any], prevalence: pd.DataFrame
) -> bytes:
    daily = ranking.loc[ranking["frequency"] == "daily"]
    lines = [
        "# V7 adaptive-target training-history experiment",
        "",
        "Status: **development selection complete**.",
        "",
        "The experiment used only verified Tiingo SPY development observations through 2023. "
        "It did not load 2024–2025 outcomes or evaluation artifacts.",
        "",
        "High volatility is defined point in time as forward five-session volatility above the "
        "trailing three-calendar-year 75th percentile of outcomes already observable at each "
        "origin.",
        "",
        f"Selected history design: `{selection['design_id']}`",
        "",
        f"Selected model weight: `{selection['model_weight']:.2f}`; adaptive-persistence weight: "
        f"`{1 - selection['model_weight']:.2f}`.",
        "",
        "| Rank | Training history | Model weight | Balanced accuracy | Macro F1 | Accuracy | "
        "High recall | False alarms | Misses |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in daily.to_dict("records"):
        lines.append(
            f"| {row['rank']} | {row['design_id']} | {row['model_weight']:.2f} | "
            f"{row['mean_balanced_accuracy']:.4f} | {row['mean_macro_f1']:.4f} | "
            f"{row['mean_accuracy']:.4f} | {row['mean_high_recall']:.4f} | "
            f"{row['false_positive']} | {row['false_negative']} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive target prevalence",
            "",
            "| Year | Origins | High origins | High prevalence | Mean threshold |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in prevalence.to_dict("records"):
        lines.append(
            f"| {row['validation_year']} | {row['origins']} | {row['high_origins']} | "
            f"{row['high_prevalence']:.4f} | {row['mean_threshold']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Every outer score used a Platt calibrator fitted on the same history design's raw "
            "out-of-fold predictions from the three preceding years. Training membership and "
            "threshold histories were purged at each origin.",
            "",
            "These rankings are reused development evidence. They select a design for a future "
            "frozen protocol and are not an unbiased estimate of performance on new data.",
            "",
        ]
    )
    return "\n".join(lines).encode()


def run_history_experiment(
    dataset_dir: Path,
    output_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    dataset, output = Path(dataset_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable V7 experiment: {output}")
    observations, dataset_manifest = load_challenger_development(dataset)
    if (
        dataset_manifest.get("protocol") != TIINGO_PROTOCOL
        or dataset_manifest.get("profile_id") != "tiingo-spy-v1"
    ):
        raise ValueError("V7 requires the verified tiingo-spy-v1 challenger development dataset")
    config = {
        "protocol": PROTOCOL,
        "status": "development-only selection",
        "profile_id": "tiingo-spy-v1",
        "outer_years": list(OUTER_YEARS),
        "calibration_years_per_outer_fold": CALIBRATION_YEARS,
        "adaptive_threshold": {
            "lookback_calendar_years": THRESHOLD_LOOKBACK_YEARS,
            "quantile": THRESHOLD_QUANTILE,
            "method": "linear",
            "minimum_observable_outcomes": MIN_THRESHOLD_HISTORY,
            "availability_rule": "label_end strictly before origin",
        },
        "history_designs": [asdict(design) for design in history_designs()],
        "base_candidate": asdict(_histogram_candidate()),
        "ensemble_weights": list(ENSEMBLE_WEIGHTS),
        "feature_order": list(CHALLENGER_FEATURES),
        "ranking": "mean yearly balanced accuracy, macro F1, design order, weight order",
        "evaluation_years_loaded": [],
        "final_model_fitted": False,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "stage": "adaptive-target training-history development selection",
        "state": "running",
        "research_status": "exploratory",
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
        eligible, thresholds = _eligible_observations(observations)
        raw_cache: dict[tuple[str, int], pd.DataFrame] = {}
        audits: list[dict[str, Any]] = []
        memberships: list[pd.DataFrame] = []
        first_year = OUTER_YEARS[0] - CALIBRATION_YEARS
        for design in history_designs():
            for year in range(first_year, OUTER_YEARS[-1] + 1):
                if progress:
                    progress(f"V7 {design.design_id}: chronological fold {year}")
                raw, audit, membership = _raw_fold(eligible, design, year)
                raw_cache[(design.design_id, year)] = raw
                audits.append(audit)
                memberships.append(membership)
        result_rows: list[dict[str, Any]] = []
        prediction_rows: list[pd.DataFrame] = []
        calibration_rows: list[pd.DataFrame] = []
        for design in history_designs():
            for outer_year in OUTER_YEARS:
                calibration = pd.concat(
                    [
                        raw_cache[(design.design_id, year)]
                        for year in range(outer_year - CALIBRATION_YEARS, outer_year)
                    ]
                )
                calibrator = fit_platt(calibration["raw_score"], calibration["true_class"])
                recorded_calibration = calibration.copy()
                recorded_calibration["design_id"] = design.design_id
                recorded_calibration["outer_year"] = outer_year
                calibration_rows.append(recorded_calibration.reset_index(names="sample_id"))
                outer = raw_cache[(design.design_id, outer_year)]
                probability = calibrated_scores(calibrator, outer["raw_score"])
                for weight in ENSEMBLE_WEIGHTS:
                    predicted = _ensemble(probability, outer, weight)
                    common = {
                        "design_id": design.design_id,
                        "model_weight": weight,
                        "validation_year": outer_year,
                    }
                    for frequency, positions in (
                        ("daily", np.arange(len(outer))),
                        ("every_fifth", np.arange(0, len(outer), 5)),
                    ):
                        result_rows.append(
                            {
                                **common,
                                "frequency": frequency,
                                **_metric_row(
                                    outer["true_class"].iloc[positions],
                                    predicted.iloc[positions],
                                ),
                            }
                        )
                    recorded = predicted.copy()
                    recorded["raw_score"] = outer["raw_score"]
                    recorded["calibrated_probability"] = probability
                    recorded["true_class"] = outer["true_class"]
                    recorded["adaptive_threshold"] = outer["adaptive_threshold"]
                    recorded["as_of"] = outer["as_of"]
                    recorded["every_fifth"] = np.arange(len(outer)) % 5 == 0
                    for name, value in common.items():
                        recorded[name] = value
                    prediction_rows.append(recorded.reset_index(names="sample_id"))
        results = pd.DataFrame(result_rows)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        ranking = _rank(results)
        pooled = _pooled_metrics(predictions)
        selected = ranking.loc[(ranking["frequency"] == "daily") & (ranking["rank"] == 1)].iloc[0]
        selection = {
            "design_id": str(selected["design_id"]),
            "model_weight": float(selected["model_weight"]),
            "mean_balanced_accuracy": float(selected["mean_balanced_accuracy"]),
            "mean_macro_f1": float(selected["mean_macro_f1"]),
        }
        outer = eligible.loc[eligible["as_of"].dt.year.isin(OUTER_YEARS)]
        prevalence = (
            outer.assign(validation_year=outer["as_of"].dt.year)
            .groupby("validation_year", as_index=False)
            .agg(
                origins=("adaptive_target", "size"),
                high_origins=("adaptive_target", "sum"),
                high_prevalence=("adaptive_target", "mean"),
                mean_threshold=("adaptive_threshold", "mean"),
            )
        )
        tables = {
            "adaptive_thresholds.csv": thresholds.reset_index(),
            "cv_results.csv": results,
            "cv_predictions.csv": predictions,
            "calibration_predictions.csv": pd.concat(calibration_rows, ignore_index=True),
            "ranking.csv": ranking,
            "pooled_metrics.csv": pooled,
            "fit_audit.csv": pd.DataFrame(audits),
            "training_membership.csv": pd.concat(memberships, ignore_index=True),
            "target_prevalence.csv": prevalence,
        }
        payloads = {name: frame.to_csv(index=False).encode() for name, frame in tables.items()}
        payloads["selection.json"] = json_bytes(selection)
        payloads["report.md"] = _report(ranking, selection, prevalence)
        for name, payload in payloads.items():
            atomic_write(output / name, payload)
        payloads["config.json"] = (output / "config.json").read_bytes()
        manifest.update(
            {
                "state": "complete",
                "eligible_rows": len(eligible),
                "selected_design": selection["design_id"],
                "selected_model_weight": selection["model_weight"],
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


def verify_history_experiment(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    value = json.loads((root / "manifest.json").read_text())
    if (
        not isinstance(value, dict)
        or value.get("protocol") != PROTOCOL
        or value.get("state") != "complete"
    ):
        raise ValueError("Unsupported or incomplete V7 history experiment")
    required = {
        "config.json",
        "adaptive_thresholds.csv",
        "cv_results.csv",
        "cv_predictions.csv",
        "calibration_predictions.csv",
        "ranking.csv",
        "pooled_metrics.csv",
        "fit_audit.csv",
        "training_membership.csv",
        "target_prevalence.csv",
        "selection.json",
        "report.md",
    }
    hashes = value.get("files")
    if not isinstance(hashes, dict) or not required <= hashes.keys():
        raise ValueError("V7 manifest lacks required file hashes")
    for name, digest in hashes.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid V7 artifact path")
        if sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"V7 artifact hash mismatch: {name}")
    return cast(dict[str, Any], value)

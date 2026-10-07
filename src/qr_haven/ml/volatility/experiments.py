"""Reproducible post-V4 experiments with development-only fitting and saved-error diagnosis."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, sha256, utc_now
from qr_haven.ml.classification.evaluation import classification_metrics
from qr_haven.ml.volatility.artifacts import load_v2_development
from qr_haven.ml.volatility.contracts import FEATURES, WalkForwardPlan
from qr_haven.ml.volatility.evaluation import verify_v4_evaluation
from qr_haven.ml.volatility.experiment_diagnostics import error_tables
from qr_haven.ml.volatility.experiment_models import (
    CONTROL_ID,
    classifier_scores,
    ewma_forecast,
    experiment_variants,
    linear_forecast,
    scored_predictions,
    training_ids,
)
from qr_haven.ml.volatility.models import VolatilityCandidate, baseline_predictions, make_estimator
from qr_haven.ml.volatility.persistence import environment_versions, source_identity
from qr_haven.ml.volatility.training import verify_v3_run

PROTOCOL = "volatility-experiments-v1"


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


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


def _compare(
    observations: pd.DataFrame,
    plan: WalkForwardPlan,
    candidate: VolatilityCandidate,
    progress: Callable[[str], None] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variants = experiment_variants()
    ewmas = {
        variant.variant_id: ewma_forecast(observations, variant.decay)
        for variant in variants
        if variant.decay is not None
    }
    records: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    membership: list[dict[str, Any]] = []
    for number, fold in enumerate(plan.folds, 1):
        if progress:
            progress(f"Experiment fold {number}/{len(plan.folds)}: {fold.validation_year}")
        valid = observations.loc[list(fold.validation_ids)]
        target = (valid["forward_vol_5"] > fold.threshold).astype("int64")
        cache: dict[tuple[int | None, float], tuple[pd.Series, dict[str, Any]]] = {}
        predictions: list[tuple[str, str, pd.DataFrame]] = []
        for variant in variants:
            ids = training_ids(observations, fold, variant.window_years)
            fit_id = f"{fold.fold_id}/classifier/{variant.window_years}/{variant.weight_power}"
            if variant.kind == "classifier":
                key = (variant.window_years, variant.weight_power)
                if key not in cache:
                    cache[key] = classifier_scores(
                        candidate,
                        observations,
                        ids,
                        fold.validation_ids,
                        fold.threshold,
                        variant.weight_power,
                    )
                scores, details = cache[key]
                predicted = scored_predictions(scores, variant.cutoff, "weighted_class_1_score")
            elif variant.kind == "linear":
                scores, details = linear_forecast(observations, ids, fold.validation_ids)
                predicted = scored_predictions(scores, fold.threshold, "forecast_annualized_vol")
                fit_id = f"{fold.fold_id}/linear"
            else:
                scores = ewmas[variant.variant_id].loc[list(fold.validation_ids)]
                details = {"decay": variant.decay, "fitted": False}
                predicted = scored_predictions(scores, fold.threshold, "forecast_annualized_vol")
                fit_id = f"{fold.fold_id}/{variant.variant_id}"
            train = observations.loc[list(ids)]
            fits.append(
                {
                    "fold_id": fold.fold_id,
                    "variant_id": variant.variant_id,
                    "fit_id": fit_id,
                    "training_window": "expanding"
                    if variant.window_years is None
                    else f"{variant.window_years} years",
                    "training_rows": len(ids) if variant.kind != "ewma" else 0,
                    "training_start": train["as_of"].iloc[0] if variant.kind != "ewma" else None,
                    "training_end": train["as_of"].iloc[-1] if variant.kind != "ewma" else None,
                    "max_training_label_end": train["label_end"].max()
                    if variant.kind != "ewma"
                    else None,
                    "validation_start": valid["as_of"].iloc[0],
                    "truth_threshold": fold.threshold,
                    "decision_cutoff": variant.cutoff if variant.kind == "classifier" else None,
                    "fit_details": json.dumps(details, sort_keys=True),
                }
            )
            if variant.kind != "ewma":
                membership.extend(
                    {"fold_id": fold.fold_id, "variant_id": variant.variant_id, "sample_id": key}
                    for key in ids
                )
            predictions.append((variant.variant_id, variant.experiment, predicted))
        training_target = (
            observations.loc[list(fold.training_ids), "forward_vol_5"] > fold.threshold
        ).astype("int64")
        for name in ("majority", "always_normal", "persistence"):
            predictions.append(
                (
                    f"baseline_{name}",
                    "baseline",
                    baseline_predictions(
                        name, training_target, valid.loc[:, list(FEATURES)], fold.threshold
                    ),
                )
            )
        for variant_id, experiment, predicted in predictions:
            predicted.index.name = "sample_id"
            if tuple(predicted.index) != fold.validation_ids:
                raise ValueError("Variant predictions differ from frozen validation membership")
            common = {
                "variant_id": variant_id,
                "experiment": experiment,
                "fold_id": fold.fold_id,
                "validation_year": fold.validation_year,
                "threshold": fold.threshold,
            }
            for frequency, positions in (
                ("daily", np.arange(len(target))),
                ("every_fifth", np.arange(0, len(target), 5)),
            ):
                metrics.append(
                    {
                        **common,
                        "frequency": frequency,
                        **_metric_row(target.iloc[positions], predicted.iloc[positions]),
                    }
                )
            recorded = predicted.copy()
            for name, value in common.items():
                recorded[name] = value
            recorded["as_of"] = valid["as_of"].to_numpy()
            recorded["true_class"] = target.to_numpy()
            recorded["every_fifth"] = np.arange(len(valid)) % 5 == 0
            records.append(recorded.reset_index())
    return (
        pd.DataFrame(metrics),
        pd.concat(records, ignore_index=True),
        pd.DataFrame(fits),
        pd.DataFrame(membership),
    )


def _rank_and_pool(
    results: pd.DataFrame, predictions: pd.DataFrame, expected_folds: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary: list[dict[str, Any]] = []
    pooled: list[dict[str, Any]] = []
    order = {v.variant_id: index for index, v in enumerate(experiment_variants())}
    metric_names = ["balanced_accuracy", "macro_f1", "accuracy", "high_recall", "high_precision"]
    for (variant_id, frequency), rows in results.groupby(["variant_id", "frequency"], sort=False):
        if len(rows) != len(expected_folds) or set(rows["fold_id"]) != expected_folds:
            raise ValueError("Experiment ranking requires one result per frozen fold")
        if not np.isfinite(rows[metric_names].to_numpy(dtype=float)).all():
            raise ValueError("Selection metrics must be finite")
        summary.append(
            {
                "variant_id": variant_id,
                "experiment": rows["experiment"].iloc[0],
                "frequency": frequency,
                "variant_order": order.get(variant_id, 100),
                **{f"mean_{name}": float(rows[name].mean()) for name in metric_names},
                "false_positive": int(rows["false_positive"].sum()),
                "false_negative": int(rows["false_negative"].sum()),
            }
        )
    ranked = pd.DataFrame(summary)
    references = ranked.set_index(["variant_id", "frequency"])
    for name in ("control", "persistence"):
        reference_id = CONTROL_ID if name == "control" else "baseline_persistence"
        ranked[f"delta_vs_{name}"] = [
            value - references.loc[(reference_id, frequency), "mean_balanced_accuracy"]
            for value, frequency in zip(
                ranked["mean_balanced_accuracy"], ranked["frequency"], strict=True
            )
        ]
    ranked = ranked.sort_values(
        ["frequency", "mean_balanced_accuracy", "mean_macro_f1", "variant_order"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked["overall_rank"] = ranked.groupby("frequency").cumcount() + 1
    ranked["experiment_rank"] = ranked.groupby(["frequency", "experiment"]).cumcount() + 1
    for variant_id, rows in predictions.groupby("variant_id", sort=False):
        rows = rows.sort_values(["validation_year", "as_of"]).set_index("sample_id")
        for frequency, frame in (("daily", rows), ("every_fifth", rows.loc[rows["every_fifth"]])):
            pooled.append(
                {
                    "variant_id": variant_id,
                    "frequency": frequency,
                    **_metric_row(frame["true_class"], frame),
                }
            )
    return ranked, pd.DataFrame(pooled)


def _saved_diagnostics(
    run: Path,
    v2: Path,
    evaluation: Path | None,
    observations: pd.DataFrame,
    plan: WalkForwardPlan,
    v3: dict[str, Any],
    candidate_id: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    saved = _csv(run / "cv_predictions.csv")
    saved = saved.loc[saved["model_id"].isin([candidate_id, "baseline_persistence"])].copy()
    saved["as_of"] = pd.to_datetime(saved["as_of"])
    saved["partition"] = "development"
    feature_columns = ["trailing_vol_5", "trailing_vol_20"]
    saved = saved.merge(
        observations[feature_columns],
        left_on="sample_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    context: dict[str, Any] = {"holdout_diagnosed": False}
    frames = [saved]
    if evaluation is not None:
        evaluated = verify_v4_evaluation(evaluation)
        if (
            evaluated["profile_id"] != plan.profile_id
            or evaluated["model_sha256"] != v3["files"]["model.pkl"]
            or evaluated["v2_manifest_sha256"] != v3["v2_manifest_sha256"]
            or evaluated["v3_manifest_sha256"] != sha256((run / "manifest.json").read_bytes())
        ):
            raise ValueError("Saved evaluation is not bound to this V3 run and V2 dataset")
        v2_manifest = json.loads((v2 / "manifest.json").read_text())
        payload = (v2 / "holdout_features.csv").read_bytes()
        if sha256(payload) != v2_manifest["files"]["holdout_features.csv"]:
            raise ValueError("Holdout diagnostic feature hash mismatch")
        features = pd.read_csv(
            BytesIO(payload), index_col="sample_id", float_precision="round_trip"
        )
        holdout = _csv(evaluation / "holdout_predictions.csv")
        if (
            tuple(holdout["sample_id"]) != plan.holdout_ids
            or tuple(features.index) != plan.holdout_ids
        ):
            raise ValueError("Saved diagnostic holdout membership does not match V2")
        if not (holdout["threshold"] == plan.final_threshold).all():
            raise ValueError("Saved diagnostic threshold differs from the frozen truth threshold")
        for model_id in (candidate_id, "baseline_persistence"):
            frame = holdout[["sample_id", "as_of", "true_class", "threshold"]].copy()
            frame["predicted_class"] = holdout[f"{model_id}_predicted_class"]
            frame["score_class_1"] = holdout[f"{model_id}_score_class_1"]
            frame["model_id"] = model_id
            frame["fold_id"] = "final_holdout"
            frame["partition"] = "exposed_holdout_diagnosis_only"
            frame["as_of"] = pd.to_datetime(frame["as_of"])
            frames.append(
                frame.merge(
                    features[feature_columns],
                    left_on="sample_id",
                    right_index=True,
                    how="left",
                    validate="one_to_one",
                )
            )
        context = {
            "holdout_diagnosed": True,
            "v4_manifest_sha256": sha256((evaluation / "manifest.json").read_bytes()),
            "saved_holdout_predictions_sha256": evaluated["files"]["holdout_predictions.csv"],
        }
    normalized = pd.concat(frames, ignore_index=True)
    if normalized[feature_columns].isna().any().any():
        raise ValueError("Missing feature observations in diagnostic join")
    return error_tables(normalized), context


def _report(ranking: pd.DataFrame, diagnostics: dict[str, pd.DataFrame]) -> bytes:
    daily = ranking.loc[ranking["frequency"] == "daily"]
    lines = [
        "# Four exploratory volatility experiments",
        "",
        "Status: exploratory. All new forecasts and rankings use development folds only. "
        "The original V4 result remains unchanged. No replacement model or fresh holdout "
        "performance is claimed.",
        "",
        "## 1. Saved-error diagnosis",
        "",
        "Regime descriptors compare currently observed five- and twenty-session volatility "
        "with each row's training threshold. Error runs count overlapping forecast origins, "
        "not independent market events.",
        "",
        "| Partition | Model | False alarms | Missed high periods | High support | Rows |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    totals = diagnostics["error_summary.csv"]
    for row in totals.loc[totals["grouping"] == "overall"].to_dict("records"):
        lines.append(
            f"| {row['partition']} | {row['model_id']} | {row['false_positive']} | "
            f"{row['false_negative']} | {row['high_support']} | {row['rows']} |"
        )
    for experiment, heading in (
        ("weights_cutoffs", "2. Class weights and decision cutoffs"),
        ("simple_forecasts", "3. Simpler volatility forecasts"),
        ("rolling_windows", "4. Rolling training windows"),
    ):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| Variant | Mean yearly balanced accuracy | Change vs V3 (pp) | "
                "Change vs persistence (pp) | Mean accuracy | Mean high recall | "
                "False alarms | Misses |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        selected = daily.loc[
            daily["experiment"].eq(experiment)
            | daily["variant_id"].isin([CONTROL_ID, "baseline_persistence"])
        ]
        for row in selected.to_dict("records"):
            lines.append(
                f"| {row['variant_id']} | {row['mean_balanced_accuracy']:.4f} | "
                f"{100 * row['delta_vs_control']:+.2f} | "
                f"{100 * row['delta_vs_persistence']:+.2f} | "
                f"{row['mean_accuracy']:.4f} | {row['mean_high_recall']:.4f} | "
                f"{row['false_positive']} | {row['false_negative']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation and evidence",
            "",
            "The truth threshold is identical for every variant within a fold, including rolling "
            "training windows. A score cutoff changes the prediction, not truth. "
            "EWMA and linear forecasts are in annualized-volatility units.",
            "",
            "Years have equal weight. A single-class year contributes that class's recall "
            "to balanced accuracy and has undefined ROC-AUC/average precision. Fold supports and "
            "diagnostics are in [cv_results.csv](cv_results.csv). Every-fifth-origin comparisons "
            "are included in [ranking.csv](ranking.csv); pooled descriptive scores are in "
            "[pooled_metrics.csv](pooled_metrics.csv).",
            "",
            "See [error_summary.csv](error_summary.csv) for year/regime denominators, "
            "[paired_errors.csv](paired_errors.csv) for gains and losses versus persistence, "
            "[error_runs.csv](error_runs.csv) for consecutive errors, and "
            "[diagnostic_observations.csv](diagnostic_observations.csv) for individual dates.",
            "",
            "The best development score was selected using these same folds. These experiments "
            "were proposed after viewing V4, so a separately frozen experiment and an untouched "
            "future period are required to assess generalization. No new forecast was made on "
            "the exposed holdout or quarantined data.",
            "",
        ]
    )
    return "\n".join(lines).encode()


def run_volatility_experiments(
    run_dir: Path,
    v2_dir: Path,
    output_dir: Path,
    *,
    evaluation_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run a fixed exploratory protocol; diagnoses are downstream of comparison results."""
    run, v2, output = Path(run_dir), Path(v2_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable experiments: {output}")
    for source in (run, v2, evaluation_dir):
        if source is not None and output.resolve().is_relative_to(Path(source).resolve()):
            raise ValueError("Experiment output must be outside frozen input directories")
    v3 = verify_v3_run(run)
    observations, plan, v2_manifest = load_v2_development(v2)
    if (
        v3["v2_manifest_sha256"] != sha256((v2 / "manifest.json").read_bytes())
        or v3["profile_id"] != plan.profile_id
        or v3["source_sha256"] != plan.source_sha256
    ):
        raise ValueError("Experiment inputs do not match the frozen V3/V2 protocol")
    selection = json.loads((run / "selection.json").read_text())
    candidate = VolatilityCandidate(**selection["chosen"])
    if candidate.candidate_id != v3["selected_candidate"]:
        raise ValueError("V3 selection disagrees with its manifest")
    config = {
        "protocol": PROTOCOL,
        "status": "exploratory",
        "base_candidate": asdict(candidate),
        "variants": [asdict(item) for item in experiment_variants()],
        "feature_order": list(FEATURES),
        "validation_years": v2_manifest["profile"]["validation_years"],
        "truth_threshold_policy": "original expanding-history threshold shared by all variants",
        "ranking_rule": "mean yearly balanced accuracy DESC, macro F1 DESC, variant order",
        "ewma_initialization": "first development row trailing_vol_60 squared / 252",
        "linear_forecast": "OLS with intercept on trailing_vol_5/20/60; clip negative forecasts",
        "new_holdout_predictions": False,
        "class_weight_normalization": "mean one",
        "fit_thread_limit": 1,
        "estimator_parameters": {
            key: value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
            for key, value in make_estimator(candidate).get_params(deep=True).items()
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "state": "running",
        "research_status": "exploratory",
        "profile_id": plan.profile_id,
        "source_sha256": plan.source_sha256,
        "v3_manifest_sha256": sha256((run / "manifest.json").read_bytes()),
        "v2_manifest_sha256": v3["v2_manifest_sha256"],
        "environment": environment_versions(),
        "code": source_identity(),
        "created_at_utc": utc_now(),
        "new_holdout_predictions": False,
        "sealed_outcomes_opened": False,
        "quarantined_observations_used": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_write(output / "config.json", json_bytes(config))
    atomic_write(output / "manifest.json", json_bytes(manifest))
    started = time.perf_counter()
    try:
        results, predictions, fits, membership = _compare(observations, plan, candidate, progress)
        ranking, pooled = _rank_and_pool(
            results, predictions, {fold.fold_id for fold in plan.folds}
        )
        original = _csv(run / "cv_predictions.csv")
        original = original.loc[original["model_id"] == candidate.candidate_id].set_index(
            "sample_id"
        )
        replay = predictions.loc[predictions["variant_id"] == CONTROL_ID].set_index("sample_id")
        if (
            not original.index.equals(replay.index)
            or not np.array_equal(original["predicted_class"], replay["predicted_class"])
            or not np.array_equal(original["true_class"], replay["true_class"])
            or not np.allclose(
                original["score_class_1"], replay["score_class_1"], rtol=0, atol=1e-10
            )
        ):
            raise ValueError("V3 control replay differs from saved development predictions")
        if progress:
            progress("Development comparison complete; diagnosing saved V3/V4 errors")
        diagnostics, context = _saved_diagnostics(
            run, v2, evaluation_dir, observations, plan, v3, candidate.candidate_id
        )
        tables = {
            "cv_results.csv": results,
            "cv_predictions.csv": predictions,
            "ranking.csv": ranking,
            "pooled_metrics.csv": pooled,
            "fit_audit.csv": fits,
            "training_membership.csv": membership,
            **diagnostics,
        }
        payloads = {name: frame.to_csv(index=False).encode() for name, frame in tables.items()}
        payloads["report.md"] = _report(ranking, diagnostics)
        for name, payload in payloads.items():
            atomic_write(output / name, payload)
        payloads["config.json"] = (output / "config.json").read_bytes()
        manifest.update(
            {
                **context,
                "state": "complete",
                "v3_control_replayed": True,
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


def verify_volatility_experiments(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    value = json.loads((root / "manifest.json").read_text())
    if (
        not isinstance(value, dict)
        or value.get("protocol") != PROTOCOL
        or value.get("state") != "complete"
    ):
        raise ValueError("Unsupported or incomplete exploratory experiment")
    required = {
        "config.json",
        "cv_results.csv",
        "cv_predictions.csv",
        "ranking.csv",
        "pooled_metrics.csv",
        "fit_audit.csv",
        "training_membership.csv",
        "diagnostic_observations.csv",
        "error_summary.csv",
        "error_runs.csv",
        "paired_errors.csv",
        "report.md",
    }
    hashes = value.get("files")
    if not isinstance(hashes, dict) or not required <= hashes.keys():
        raise ValueError("Experiment manifest lacks required file hashes")
    for name, digest in hashes.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid experiment artifact path")
        if sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"Experiment artifact hash mismatch: {name}")
    return value

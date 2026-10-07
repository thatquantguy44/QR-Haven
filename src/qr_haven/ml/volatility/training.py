"""V3 development-only model comparison and immutable selection artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qr_haven.ml.classification.evaluation import classification_metrics
from qr_haven.ml.volatility.artifacts import load_v2_development
from qr_haven.ml.volatility.contracts import FEATURES
from qr_haven.ml.volatility.models import (
    VolatilityCandidate,
    baseline_predictions,
    candidates,
    fit_estimator,
    inverse_frequency_weights,
    make_estimator,
    predict_estimator,
)
from qr_haven.ml.volatility.persistence import environment_versions, source_identity

BASELINES = ("majority", "always_normal", "persistence")


@dataclass(frozen=True)
class DevelopmentTrainingResult:
    """Paths and selected candidate from a completed V3 comparison."""

    output_dir: Path
    selected: VolatilityCandidate
    ranking: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _metrics_record(target: pd.Series, predictions: pd.DataFrame) -> dict[str, Any]:
    metrics = classification_metrics(target, predictions)
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "high_precision": metrics["per_class"]["1"]["precision"],
        "high_recall": metrics["per_class"]["1"]["recall"],
        "high_f1": metrics["per_class"]["1"]["f1"],
        "macro_f1": metrics["f1_macro"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "normal_support": metrics["per_class"]["0"]["support"],
        "high_support": metrics["per_class"]["1"]["support"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"], separators=(",", ":")),
        "diagnostics": json.dumps(metrics["diagnostics"], separators=(",", ":")),
    }


def _result_rows(
    *,
    model_id: str,
    model_kind: str,
    family: str,
    fold_id: str,
    validation_year: int,
    threshold: float,
    target: pd.Series,
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for frequency, positions in (
        ("daily", np.arange(len(target))),
        ("every_fifth", np.arange(0, len(target), 5)),
    ):
        selected_target = target.iloc[positions]
        selected_predictions = predictions.iloc[positions]
        rows.append(
            {
                "model_id": model_id,
                "model_kind": model_kind,
                "family": family,
                "fold_id": fold_id,
                "validation_year": validation_year,
                "evaluation_frequency": frequency,
                "threshold": threshold,
                "rows": len(selected_target),
                **_metrics_record(selected_target, selected_predictions),
            }
        )
    return rows


def rank_candidates(
    results: pd.DataFrame, candidate_set: Sequence[VolatilityCandidate]
) -> tuple[dict[str, Any], ...]:
    """Rank complete learned candidates using the frozen unrounded rule."""
    all_daily = results[results["evaluation_frequency"] == "daily"]
    daily = all_daily[all_daily["model_kind"] == "learned"]
    by_id = {candidate.candidate_id: candidate for candidate in candidate_set}
    records: list[dict[str, Any]] = []
    expected_folds = set(all_daily["fold_id"])
    if not expected_folds:
        raise ValueError("No learned daily fold results are available")
    for candidate_id, candidate in by_id.items():
        rows = daily[daily["model_id"] == candidate_id]
        if len(rows) != len(expected_folds) or set(rows["fold_id"]) != expected_folds:
            raise ValueError(f"Candidate {candidate_id} lacks one result per development fold")
        metrics = rows[["balanced_accuracy", "macro_f1", "accuracy"]].to_numpy(dtype=float)
        if not np.isfinite(metrics).all():
            raise ValueError(f"Candidate {candidate_id} has nonfinite selection metrics")
        records.append(
            {
                "candidate_id": candidate_id,
                "family": candidate.family,
                "family_order": candidate.family_order,
                "grid_order": candidate.grid_order,
                "parameters": candidate.parameters,
                "mean_balanced_accuracy": float(rows["balanced_accuracy"].mean()),
                "mean_macro_f1": float(rows["macro_f1"].mean()),
                "mean_accuracy": float(rows["accuracy"].mean()),
            }
        )
    records.sort(
        key=lambda row: (
            -row["mean_balanced_accuracy"],
            -row["mean_macro_f1"],
            row["family_order"],
            row["grid_order"],
        )
    )
    return tuple(records)


def _pooled_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, model_predictions in predictions.groupby("model_id", sort=False):
        ordered = model_predictions.sort_values(["validation_year", "as_of"])
        for frequency, selected in (
            ("daily", ordered),
            ("every_fifth", ordered[ordered["secondary_every_fifth"]]),
        ):
            target = selected.set_index("sample_id")["true_class"]
            predicted = selected.set_index("sample_id")[
                ["predicted_class", "score_class_1", "score_kind"]
            ]
            rows.append(
                {
                    "model_id": model_id,
                    "model_kind": selected["model_kind"].iloc[0],
                    "family": selected["family"].iloc[0],
                    "evaluation_frequency": frequency,
                    "rows": len(selected),
                    **_metrics_record(target, predicted),
                }
            )
    return pd.DataFrame(rows)


def _development_report(
    ranking: tuple[dict[str, Any], ...], results: pd.DataFrame, pooled: pd.DataFrame
) -> bytes:
    winner = ranking[0]
    baseline_daily = results[
        (results["model_kind"] == "baseline")
        & (results["evaluation_frequency"] == "daily")
    ]
    lines = [
        "# Volatility V3 development model comparison",
        "",
        "The 2018–2019 final holdout was not evaluated. Model selection used only the frozen "
        "2010–2017 purged yearly folds.",
        "",
        f"Selected `{winner['candidate_id']}` with `{winner['parameters']}`. Mean yearly balanced "
        f"accuracy: **{winner['mean_balanced_accuracy']:.6f}**; mean macro F1: "
        f"**{winner['mean_macro_f1']:.6f}**.",
        "",
        "## Ranking",
        "",
        "| Rank | Candidate | Family | Mean balanced accuracy | Mean macro F1 |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {rank} | {row['candidate_id']} | {row['family']} | "
        f"{row['mean_balanced_accuracy']:.6f} | {row['mean_macro_f1']:.6f} |"
        for rank, row in enumerate(ranking, 1)
    )
    lines.extend(
        [
            "",
            "## Baseline mean yearly results",
            "",
            "| Baseline | Balanced accuracy | Macro F1 | Accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model_id, rows in baseline_daily.groupby("model_id", sort=False):
        lines.append(
            f"| {model_id} | {rows['balanced_accuracy'].mean():.6f} | "
            f"{rows['macro_f1'].mean():.6f} | {rows['accuracy'].mean():.6f} |"
        )
    winner_pooled = pooled[
        (pooled["model_id"] == winner["candidate_id"])
        & (pooled["evaluation_frequency"] == "daily")
    ].iloc[0]
    lines.extend(
        [
            "",
            "## Selected model versus persistence by year",
            "",
            "| Year | High support | Selected balanced accuracy | Selected high recall | "
            "Persistence balanced accuracy | Persistence high recall |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    daily = results[results["evaluation_frequency"] == "daily"]
    for year in sorted(daily["validation_year"].unique()):
        winner_year = daily[
            (daily["model_id"] == winner["candidate_id"])
            & (daily["validation_year"] == year)
        ].iloc[0]
        persistence_year = daily[
            (daily["model_id"] == "baseline_persistence")
            & (daily["validation_year"] == year)
        ].iloc[0]
        lines.append(
            f"| {int(year)} | {int(winner_year['high_support'])}/{int(winner_year['rows'])} | "
            f"{winner_year['balanced_accuracy']:.6f} | {winner_year['high_recall']:.6f} | "
            f"{persistence_year['balanced_accuracy']:.6f} | "
            f"{persistence_year['high_recall']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Pooled development context",
            "",
            f"The selected model's pooled out-of-fold balanced accuracy is "
            f"**{winner_pooled['balanced_accuracy']:.6f}** over "
            f"{int(winner_pooled['rows'])} predictions. Selection used the unweighted mean of "
            "yearly "
            "scores above, not this pooled number.",
            "",
            "Daily five-session targets overlap. `cv_results.csv` and "
            "`pooled_development_metrics.csv` also contain the predeclared every-fifth-session "
            "secondary evaluation. No probability calibration claim is made.",
            "A validation year with one observed class has undefined ROC-AUC and average "
            "precision; "
            "its balanced accuracy is the recall of the observed class. Diagnostics are retained "
            "in `cv_results.csv`.",
            "",
        ]
    )
    return "\n".join(lines).encode()


def train_volatility_development(
    v2_dir: Path,
    output_dir: Path,
    *,
    candidate_set: Sequence[VolatilityCandidate] | None = None,
    expected_profile: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> DevelopmentTrainingResult:
    """Compare candidates on development folds and freeze the selected final-training fit."""
    started = time.perf_counter()
    observations, plan, v2_manifest = load_v2_development(v2_dir)
    if expected_profile is not None and plan.profile_id != expected_profile:
        raise ValueError(
            f"V2 profile {plan.profile_id!r} does not match requested profile "
            f"{expected_profile!r}"
        )
    chosen_candidates = tuple(candidate_set) if candidate_set is not None else candidates()
    if not chosen_candidates or len({item.candidate_id for item in chosen_candidates}) != len(
        chosen_candidates
    ):
        raise ValueError("Candidate set must be nonempty with unique IDs")
    features = observations.loc[:, list(FEATURES)].astype("float64")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Development features must be finite")
    if not np.isfinite(observations["forward_vol_5"].to_numpy(dtype=float)).all():
        raise ValueError("Development outcomes must be finite")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold_number, fold in enumerate(plan.folds, 1):
        if progress is not None:
            progress(
                f"V3 fold {fold_number}/{len(plan.folds)}: validating {fold.validation_year} "
                f"with {len(chosen_candidates)} learned candidates"
            )
        training_ids = list(fold.training_ids)
        validation_ids = list(fold.validation_ids)
        training_target = (
            observations.loc[training_ids, "forward_vol_5"] > fold.threshold
        ).astype("int64")
        validation_target = (
            observations.loc[validation_ids, "forward_vol_5"] > fold.threshold
        ).astype("int64")
        training_features = features.loc[training_ids]
        validation_features = features.loc[validation_ids]
        weights = inverse_frequency_weights(training_target)
        fold_predictions: list[tuple[str, str, str, pd.DataFrame]] = []
        for baseline in BASELINES:
            fold_predictions.append(
                (
                    f"baseline_{baseline}",
                    "baseline",
                    baseline,
                    baseline_predictions(
                        baseline, training_target, validation_features, fold.threshold
                    ),
                )
            )
        for candidate in chosen_candidates:
            estimator = fit_estimator(
                make_estimator(candidate), training_features, training_target, weights
            )
            fold_predictions.append(
                (
                    candidate.candidate_id,
                    "learned",
                    candidate.family,
                    predict_estimator(estimator, validation_features),
                )
            )
        for model_id, model_kind, family, predictions in fold_predictions:
            metric_rows.extend(
                _result_rows(
                    model_id=model_id,
                    model_kind=model_kind,
                    family=family,
                    fold_id=fold.fold_id,
                    validation_year=fold.validation_year,
                    threshold=fold.threshold,
                    target=validation_target,
                    predictions=predictions,
                )
            )
            recorded = predictions.copy()
            recorded.insert(0, "sample_id", recorded.index)
            recorded.insert(1, "as_of", observations.loc[validation_ids, "as_of"].to_numpy())
            recorded.insert(2, "fold_id", fold.fold_id)
            recorded.insert(3, "validation_year", fold.validation_year)
            recorded.insert(4, "model_id", model_id)
            recorded.insert(5, "model_kind", model_kind)
            recorded.insert(6, "family", family)
            recorded.insert(7, "threshold", fold.threshold)
            recorded.insert(8, "true_class", validation_target.to_numpy())
            recorded.insert(9, "secondary_every_fifth", np.arange(len(recorded)) % 5 == 0)
            prediction_frames.append(recorded.reset_index(drop=True))

    results = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    ranking = rank_candidates(results, chosen_candidates)
    selected = {candidate.candidate_id: candidate for candidate in chosen_candidates}[
        ranking[0]["candidate_id"]
    ]
    pooled = _pooled_results(predictions)

    final_ids = list(plan.final_training_ids)
    final_target = (observations.loc[final_ids, "forward_vol_5"] > plan.final_threshold).astype(
        "int64"
    )
    final_features = features.loc[final_ids]
    final_estimator = fit_estimator(
        make_estimator(selected),
        final_features,
        final_target,
        inverse_frequency_weights(final_target),
    )
    final_predictions = predict_estimator(final_estimator, final_features)
    final_predictions.insert(0, "sample_id", final_predictions.index)
    final_predictions.insert(1, "true_class", final_target.to_numpy())
    final_predictions.insert(2, "threshold", plan.final_threshold)
    final_predictions = final_predictions.reset_index(drop=True)
    if progress is not None:
        progress(f"V3 selected {selected.candidate_id}; freezing final development fit")
    bundle = {
        "schema_version": 1,
        "profile_id": plan.profile_id,
        "source_sha256": plan.source_sha256,
        "candidate": asdict(selected),
        "feature_order": FEATURES,
        "threshold": plan.final_threshold,
        "estimator": final_estimator,
    }
    selection = {
        "schema_version": 1,
        "chosen": asdict(selected),
        "ranking": ranking,
        "ranking_rule": (
            "mean yearly balanced_accuracy DESC, mean yearly macro_f1 DESC, "
            "fixed family order, fixed grid order"
        ),
        "holdout_evaluated": False,
    }
    configuration = {
        "schema_version": 1,
        "profile_id": plan.profile_id,
        "feature_order": list(FEATURES),
        "label": "forward_vol_5 > training-only 75th percentile",
        "threshold_quantile": 0.75,
        "threshold_interpolation": "linear",
        "sample_weight": "inverse class frequency with equal class totals",
        "prediction_rule": "native class prediction; no probability cutoff tuning",
        "baselines": list(BASELINES),
        "candidates": [asdict(candidate) for candidate in chosen_candidates],
        "fixed_family_settings": {
            "logistic_regression": {
                "preprocessing": "StandardScaler",
                "solver": "lbfgs",
                "max_iter": 2000,
            },
            "random_forest": {
                "n_estimators": 400,
                "max_features": "sqrt",
                "random_state": 5401,
                "n_jobs": 1,
            },
            "hist_gradient_boosting": {
                "max_iter": 300,
                "early_stopping": False,
                "random_state": 5402,
            },
        },
        "selection_rule": selection["ranking_rule"],
        "holdout_evaluated": False,
    }
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable V3 run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        v2_root = Path(v2_dir)
        payloads = {
            "config.json": _json_bytes(configuration),
            "cv_results.csv": results.to_csv(index=False).encode(),
            "cv_predictions.csv": predictions.to_csv(index=False).encode(),
            "final_training_predictions.csv": final_predictions.to_csv(index=False).encode(),
            "pooled_development_metrics.csv": pooled.to_csv(index=False).encode(),
            "selection.json": _json_bytes(selection),
            "model.pkl": pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL),
            "development_report.md": _development_report(ranking, results, pooled),
            "split.json": (v2_root / "split.json").read_bytes(),
            "v2_manifest.json": (v2_root / "manifest.json").read_bytes(),
        }
        restored = pickle.loads(payloads["model.pkl"])
        replay = predict_estimator(restored["estimator"], final_features)
        expected = final_predictions.set_index("sample_id")[
            ["predicted_class", "score_class_1", "score_kind"]
        ]
        if not replay.equals(expected):
            raise ValueError("Serialized model does not exactly replay final-training predictions")
        manifest = {
            "schema_version": 1,
            "stage": "V3 development model comparison",
            "profile_id": plan.profile_id,
            "source_sha256": plan.source_sha256,
            "v2_manifest_sha256": _sha256((Path(v2_dir) / "manifest.json").read_bytes()),
            "development_folds": len(plan.folds),
            "learned_candidates": len(chosen_candidates),
            "baselines": list(BASELINES),
            "selected_candidate": selected.candidate_id,
            "feature_order": list(FEATURES),
            "final_threshold": plan.final_threshold,
            "final_training_rows": len(final_ids),
            "holdout_evaluated": False,
            "sealed_holdout_outcomes_opened": False,
            "training_seconds": time.perf_counter() - started,
            "environment": environment_versions(),
            "code": source_identity(),
            "files": {name: _sha256(payload) for name, payload in payloads.items()},
        }
        payloads["manifest.json"] = _json_bytes(manifest)
        for name, payload in payloads.items():
            with (temporary / name).open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return DevelopmentTrainingResult(output, selected, ranking, manifest)


def verify_v3_run(output_dir: Path) -> dict[str, Any]:
    """Verify the immutable V3 run without evaluating or loading holdout outcomes."""
    root = Path(output_dir)
    try:
        value = json.loads((root / "manifest.json").read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"V3 manifest is unavailable or invalid: {root}") from exc
    if not isinstance(value, dict):
        raise ValueError("V3 manifest must be a JSON object")
    if value.get("schema_version") != 1 or value.get("stage") != "V3 development model comparison":
        raise ValueError("Unsupported V3 manifest schema or stage")
    if value.get("holdout_evaluated") is not False:
        raise ValueError("V3 run must remain development-only")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("V3 manifest contains no file hashes")
    required = {
        "config.json",
        "cv_predictions.csv",
        "cv_results.csv",
        "development_report.md",
        "final_training_predictions.csv",
        "model.pkl",
        "pooled_development_metrics.csv",
        "selection.json",
        "split.json",
        "v2_manifest.json",
    }
    if not required <= set(files):
        raise ValueError("V3 manifest is missing required artifact hashes")
    for name, expected in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("V3 manifest contains an invalid artifact path")
        try:
            payload = (root / name).read_bytes()
        except OSError as exc:
            raise ValueError(f"V3 artifact is unavailable: {name}") from exc
        if _sha256(payload) != expected:
            raise ValueError(f"V3 artifact hash mismatch: {name}")
    return value

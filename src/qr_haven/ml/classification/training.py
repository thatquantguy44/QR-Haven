"""Development-only candidate comparison and frozen final development fits."""

from __future__ import annotations

import json
import pickle
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import (
    BanknoteDataset,
    atomic_write,
    load_banknote_dataset,
    sha256,
    utc_now,
)
from qr_haven.ml.classification.artifacts import (
    prepare_run,
    read_json,
    seal,
    write_csv,
    write_json,
)
from qr_haven.ml.classification.contracts import (
    ClassificationConfig,
    SplitManifest,
    TrainingResult,
    require_sklearn,
)
from qr_haven.ml.classification.models import (
    Candidate,
    ClassificationModel,
    candidates,
    make_estimator,
    predict_frame,
)
from qr_haven.ml.classification.splits import prepare_banknote_split, validate_split


def rank_candidates(rows: list[dict[str, Any]], expected_folds: int) -> list[dict[str, Any]]:
    """Use unrounded metrics, complete successful folds, family order, then grid order."""
    family_order = {"logistic_regression": 0, "svm": 1, "random_forest": 2}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["family"] != "baseline":
            grouped.setdefault(row["candidate_id"], []).append(row)
    summaries = []
    for candidate_id, folds in grouped.items():
        eligible = (
            len(folds) == expected_folds
            and {row["fold"] for row in folds} == set(range(expected_folds))
            and all(row["status"] == "ok" for row in folds)
        )
        if eligible:
            summaries.append(
                {
                    "candidate_id": candidate_id,
                    "family": folds[0]["family"],
                    "grid_order": folds[0]["grid_order"],
                    "mean_balanced_accuracy": float(
                        np.mean([r["balanced_accuracy"] for r in folds])
                    ),
                    "mean_f1_macro": float(np.mean([r["f1_macro"] for r in folds])),
                    "mean_accuracy": float(np.mean([r["accuracy"] for r in folds])),
                }
            )
    return sorted(
        summaries,
        key=lambda row: (
            -row["mean_balanced_accuracy"],
            -row["mean_f1_macro"],
            family_order[row["family"]],
            row["grid_order"],
        ),
    )


def _fit(estimator: Any, features: pd.DataFrame, target: pd.Series) -> list[str]:
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(features, target)
    messages = [f"{item.category.__name__}: {item.message}" for item in caught]
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise ValueError("Nonconverged fit: " + "; ".join(messages))
    return messages


def train_banknote_classifier(
    dataset: BanknoteDataset,
    split: SplitManifest,
    config: ClassificationConfig,
    output_dir: Path,
) -> TrainingResult:
    require_sklearn()
    from qr_haven.ml.classification.evaluation import classification_metrics

    validate_split(dataset, split, config.split.cv_folds)
    if split.metadata["config"] != config.split.model_dump():
        raise ValueError("Split configuration differs from the frozen experiment")
    original = load_banknote_dataset(
        Path(dataset.source_manifest["path"]),
        reference_sha256=dataset.source_manifest["raw_sha256"],
        canonical=dataset.source_manifest["canonical"],
        expected_raw_rows=config.dataset.expected_raw_rows,
    )
    # Reconstruct the predeclared split from source, never from modified test labels.
    if split != prepare_banknote_split(original, config):
        raise ValueError("Split membership differs from the predeclared source-data protocol")
    development = list(split.development_ids)
    if not original.features.loc[development].equals(
        dataset.features.loc[development]
    ) or not original.target.loc[development].equals(dataset.target.loc[development]):
        raise ValueError("Development data differ from the frozen source")
    if dataset.source_manifest["canonical"]:
        reference = read_json(config.dataset.reference_manifest)
        if reference["raw_sha256"] != dataset.source_manifest["raw_sha256"]:
            raise ValueError("Dataset differs from the reviewed reference manifest")
    output_dir = Path(output_dir)
    manifest = prepare_run(dataset, split, config, output_dir)
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    oof: list[pd.DataFrame] = []
    learned = candidates(config)
    dummy = Candidate("baseline", "baseline", 0, {})
    try:
        for candidate in [dummy, *learned]:
            for fold in range(config.split.cv_folds):
                train_ids = [
                    key for key in split.development_ids if split.validation_folds[key] != fold
                ]
                valid_ids = [
                    key for key in split.development_ids if split.validation_folds[key] == fold
                ]
                row: dict[str, Any] = {
                    **asdict(candidate),
                    "parameters": json.dumps(candidate.parameters, sort_keys=True),
                    "fold": fold,
                    "training_rows": len(train_ids),
                    "validation_rows": len(valid_ids),
                    "accuracy": None,
                    "balanced_accuracy": None,
                    "f1_macro": None,
                    "status": "failed",
                    "warnings": "",
                    "failure": "",
                    "fit_seconds": None,
                }
                fit_start = time.perf_counter()
                try:
                    estimator = make_estimator(candidate, config)
                    row["warnings"] = " | ".join(
                        _fit(
                            estimator,
                            dataset.features.loc[train_ids],
                            dataset.target.loc[train_ids],
                        )
                    )
                    row["fit_seconds"] = time.perf_counter() - fit_start
                    predictions = predict_frame(estimator, dataset.features.loc[valid_ids])
                    metrics = classification_metrics(dataset.target.loc[valid_ids], predictions)
                    row.update(
                        {key: metrics[key] for key in ("accuracy", "balanced_accuracy", "f1_macro")}
                    )
                    row["status"] = "ok"
                    predictions["true_class"] = dataset.target.loc[valid_ids]
                    predictions["candidate_id"] = candidate.candidate_id
                    predictions["fold"] = fold
                    oof.append(predictions.reset_index())
                except Exception as exc:
                    row["failure"] = f"{type(exc).__name__}: {exc}"
                row["fold_seconds"] = time.perf_counter() - fit_start
                if row["fit_seconds"] is None:
                    row["fit_seconds"] = row["fold_seconds"]
                rows.append(row)
        cv_results = pd.DataFrame(rows)
        write_csv(output_dir / "cv_results.csv", cv_results)
        write_csv(
            output_dir / "cv_predictions.csv",
            pd.concat(oof, ignore_index=True)
            if oof
            else pd.DataFrame(columns=["sample_id", "candidate_id", "fold"]),
        )
        ranking = rank_candidates(rows, config.split.cv_folds)
        if not ranking:
            raise ValueError("Every learned candidate failed/nonconverged; inspect cv_results.csv")
        if any(row["status"] != "ok" for row in rows if row["family"] == "baseline"):
            raise ValueError("Majority baseline failed; inspect cv_results.csv")
        chosen = next(
            candidate
            for candidate in learned
            if candidate.candidate_id == ranking[0]["candidate_id"]
        )
        final_warnings = {}
        for filename, candidate in (("model.pkl", chosen), ("baseline.pkl", dummy)):
            estimator = make_estimator(candidate, config)
            final_warnings[filename] = _fit(
                estimator,
                dataset.features.loc[list(split.development_ids)],
                dataset.target.loc[list(split.development_ids)],
            )
            bundle = ClassificationModel(estimator, dataset.audit["feature_ranges"])
            atomic_write(
                output_dir / filename, pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
            )
        selection = {
            "chosen": asdict(chosen),
            "development_scores": ranking[0],
            "ranking": ranking,
            "ranking_rule": "mean balanced_accuracy DESC, mean f1_macro DESC, family, grid order",
            "ineligible_candidates": sorted(
                {row["candidate_id"] for row in rows if row["status"] != "ok"}
            ),
            "frozen_at_utc": utc_now(),
            "final_fit_warnings": final_warnings,
            "development_only": True,
            "test_evaluated": False,
            "estimator_parameters": ClassificationModel.__name__,
        }
        # Save the complete resolved sklearn settings, including library defaults.
        selection["estimator_parameters"] = {
            key: value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
            for key, value in make_estimator(chosen, config).get_params(deep=True).items()
        }
        write_json(output_dir / "selection.json", selection)
        manifest["training_seconds"] = time.perf_counter() - start
        manifest["thread_limit"] = 1
        manifest["frozen_training_sha256"] = {
            name: sha256((output_dir / name).read_bytes())
            for name in ("model.pkl", "baseline.pkl", "selection.json")
        }
        seal(output_dir, manifest, "trained")
        return TrainingResult(
            cv_results, selection, output_dir / "model.pkl", output_dir / "baseline.pkl", manifest
        )
    except Exception as exc:
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        seal(output_dir, manifest, "invalid")
        raise

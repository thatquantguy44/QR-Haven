"""Metric definitions, exact gates and the single final-evaluation boundary."""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import BanknoteDataset, load_banknote_dataset, utc_now
from qr_haven.ml.classification.artifacts import (
    exclusive_lock,
    load_bundle,
    read_json,
    read_split,
    record_exposure,
    seal,
    verify_run,
    write_csv,
    write_json,
)
from qr_haven.ml.classification.contracts import (
    ClassificationConfig,
    EvaluationConfig,
    EvaluationResult,
    require_sklearn,
)
from qr_haven.ml.classification.splits import validate_split


def wilson_interval(correct: int, total: int) -> list[float]:
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("Wilson interval requires 0 <= correct <= total and total > 0")
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def classification_metrics(target: pd.Series, predictions: pd.DataFrame) -> dict[str, Any]:
    require_sklearn()
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    if target.empty or not target.index.equals(predictions.index):
        raise ValueError("Metrics require nonempty, index-aligned targets and predictions")
    actual = target.to_numpy()
    predicted = predictions["predicted_class"].to_numpy()
    scores = predictions["score_class_1"].to_numpy(dtype=float)
    if not set(actual) <= {0, 1} or not set(predicted) <= {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Metrics require binary labels and finite label-1 scores")
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[0, 1],
        zero_division=0,
    )
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros((2, 2), dtype=float),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )
    correct = int((actual == predicted).sum())
    both = len(set(actual)) == 2
    diagnostics = [
        f"class_{label}: zero predicted support; precision/F1 set to zero"
        for label in (0, 1)
        if not (predicted == label).any()
    ]
    if not both:
        diagnostics.append(
            "Only one true class: ROC-AUC/average precision undefined for this benchmark"
        )
    return {
        "correct": correct,
        "total": len(actual),
        "accuracy": correct / len(actual),
        "balanced_accuracy": float(np.mean(recall[support > 0])),
        "f1_macro": float(np.mean(f1)),
        "per_class": {
            str(i): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in (0, 1)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
        "matrix_order": [0, 1],
        "matrix_orientation": "rows=true, columns=predicted",
        "roc_auc": float(roc_auc_score(actual, scores)) if both else None,
        "average_precision": float(average_precision_score(actual, scores)) if both else None,
        "accuracy_wilson_95": wilson_interval(correct, len(actual)),
        "diagnostics": diagnostics,
    }


def accuracy_gain(winner: dict[str, Any], baseline: dict[str, Any]) -> float:
    """Subtract integer rates before rounding, preserving exact gate boundaries."""
    if "correct" in winner and "correct" in baseline:
        return float(
            (winner["correct"] * baseline["total"] - baseline["correct"] * winner["total"])
            / (winner["total"] * baseline["total"])
        )
    return float(Decimal(str(winner["accuracy"])) - Decimal(str(baseline["accuracy"])))


def numerical_gates(
    winner: dict[str, Any],
    baseline: dict[str, Any],
    config: EvaluationConfig,
) -> dict[str, bool]:
    return {
        "accuracy": winner["accuracy"] >= config.minimum_accuracy,
        "balanced_accuracy": winner["balanced_accuracy"] >= config.minimum_balanced_accuracy,
        "class_0_recall": winner["per_class"]["0"]["recall"] >= config.minimum_class_recall,
        "class_1_recall": winner["per_class"]["1"]["recall"] >= config.minimum_class_recall,
        "accuracy_gain": accuracy_gain(winner, baseline) >= config.minimum_accuracy_gain,
    }


def _result(run_dir: Path) -> EvaluationResult:
    metrics = read_json(run_dir / "metrics.json")
    return EvaluationResult(
        metrics,
        pd.read_csv(run_dir / "test_predictions.csv"),
        metrics["gates"],
        run_dir / "report.md",
    )


def evaluate_banknote_run(run_dir: Path, dataset: BanknoteDataset) -> EvaluationResult:
    from qr_haven.ml.classification.reporting import write_report

    run_dir = Path(run_dir).resolve()
    # A root-wide lock also serializes different runs sharing the same holdout history.
    with exclusive_lock(run_dir.parent / ".evaluation.lock"):
        manifest = verify_run(run_dir)
        if dataset.source_manifest["raw_sha256"] != manifest["source"]["raw_sha256"]:
            raise ValueError("Evaluation dataset hash mismatch")
        original = load_banknote_dataset(
            Path(manifest["source"]["path"]),
            reference_sha256=manifest["source"]["raw_sha256"],
            canonical=manifest["source"]["canonical"],
            expected_raw_rows=manifest["protocol"]["config"]["dataset"]["expected_raw_rows"],
        )
        if (
            not original.features.equals(dataset.features)
            or not original.target.equals(dataset.target)
            or not original.lineage.equals(dataset.lineage)
        ):
            raise ValueError("Evaluation data were modified after parsing the frozen source")
        if manifest["state"] == "evaluated":
            return _result(run_dir)
        config = ClassificationConfig.from_yaml(run_dir / "config.resolved.yaml")
        split = read_split(run_dir)
        validate_split(dataset, split, config.split.cv_folds)
        model, baseline = load_bundle(run_dir), load_bundle(run_dir, "baseline.pkl")
        prior = record_exposure(run_dir, manifest, split)
        try:
            ids = list(split.test_ids)
            target = dataset.target.loc[ids]
            if set(target) != {0, 1}:
                raise ValueError("Final benchmark requires both classes in the test partition")
            predictions = model.predict(dataset.features.loc[ids])
            baseline_predictions = baseline.predict(dataset.features.loc[ids])
            winner_metrics = classification_metrics(target, predictions)
            baseline_metrics = classification_metrics(target, baseline_predictions)
            gates = numerical_gates(winner_metrics, baseline_metrics, config.evaluation)
            reasons = []
            if not manifest["canonical_protocol"]:
                reasons.append("Noncanonical data/configuration: exploratory result only")
            if prior:
                reasons.append(
                    "Overlapping holdout labels were previously exposed under this artifact root"
                )
            if config.known_external_holdout_exposure:
                reasons.append("External/manual holdout exposure was reported")
            fresh = not reasons
            status = (
                ("passed" if all(gates.values()) else "target_not_met") if fresh else "exploratory"
            )
            metrics = {
                "winner": winner_metrics,
                "baseline": baseline_metrics,
                "gates": gates,
                "accuracy_gain_percentage_points": 100
                * accuracy_gain(winner_metrics, baseline_metrics),
                "benchmark_status": status,
                "numerical_gates_passed": all(gates.values()),
                "fresh_confirmatory_evaluation": fresh,
                "reasons": reasons,
                "prior_exposures": prior,
                "evaluated_at_utc": utc_now(),
                "engineering_status": "workflow_complete; see research writeup for checks",
                "audit_checks": {
                    "artifact_integrity": True,
                    "data_integrity": True,
                    "disjoint_complete_split": True,
                    "frozen_development_fit": True,
                },
            }
            predictions["true_class"] = target
            predictions["baseline_predicted_class"] = baseline_predictions["predicted_class"]
            predictions["baseline_score_class_1"] = baseline_predictions["score_class_1"]
            predictions["baseline_score_kind"] = baseline_predictions["score_kind"]
            line_groups = dataset.lineage.groupby("sample_id")["source_line"].agg(list)
            predictions["source_lines"] = [";".join(map(str, line_groups[key])) for key in ids]
            write_json(run_dir / "metrics.json", metrics)
            write_csv(run_dir / "test_predictions.csv", predictions, index=True)
            write_report(run_dir, manifest, dataset, split, metrics, predictions)
            manifest["evaluation"] = {
                "fresh": fresh,
                "prior_exposure_count": len(prior),
                "status": status,
                "evaluated_at_utc": metrics["evaluated_at_utc"],
            }
            seal(run_dir, manifest, "evaluated")
            return _result(run_dir)
        except Exception as exc:
            manifest["failure"] = f"Evaluation failed after exposure: {type(exc).__name__}: {exc}"
            seal(run_dir, manifest, "invalid")
            raise

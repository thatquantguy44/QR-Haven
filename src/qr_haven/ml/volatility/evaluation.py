"""Single-use V4 holdout evaluation for a frozen volatility model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, sha256, utc_now
from qr_haven.ml.classification.artifacts import exclusive_lock
from qr_haven.ml.classification.evaluation import classification_metrics
from qr_haven.ml.volatility.artifacts import load_v2_development
from qr_haven.ml.volatility.contracts import FEATURES, WalkForwardPlan
from qr_haven.ml.volatility.models import baseline_predictions, predict_estimator
from qr_haven.ml.volatility.persistence import (
    environment_versions,
    load_volatility_model,
    source_identity,
)
from qr_haven.ml.volatility.training import verify_v3_run

BOOTSTRAP_BLOCK_LENGTH = 20
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_MAX_ATTEMPTS = 20_000
BOOTSTRAP_SEED = 5403


@dataclass(frozen=True)
class VolatilityEvaluationResult:
    """Completed V4 metrics and immutable artifact locations."""

    output_dir: Path
    metrics: dict[str, Any]
    manifest: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON artifact is unavailable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _verified_bytes(root: Path, manifest: dict[str, Any], name: str) -> bytes:
    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or not isinstance(hashes.get(name), str):
        raise ValueError(f"V2 manifest has no valid hash for {name}")
    try:
        payload = (root / name).read_bytes()
    except OSError as exc:
        raise ValueError(f"V2 artifact is unavailable: {name}") from exc
    if sha256(payload) != hashes[name]:
        raise ValueError(f"V2 artifact hash mismatch: {name}")
    return payload


def _load_holdout_features(
    v2_dir: Path, manifest: dict[str, Any], plan: WalkForwardPlan
) -> pd.DataFrame:
    payload = _verified_bytes(v2_dir, manifest, "holdout_features.csv")
    frame = pd.read_csv(
        BytesIO(payload),
        index_col="sample_id",
        parse_dates=["as_of", "available_at", "label_start", "label_end"],
        float_precision="round_trip",
    )
    if not frame.index.is_unique or tuple(frame.index.astype(str)) != plan.holdout_ids:
        raise ValueError("Holdout features do not match frozen holdout membership")
    if not frame["as_of"].is_monotonic_increasing:
        raise ValueError("Holdout features must be chronological")
    if set(FEATURES) - set(frame.columns):
        raise ValueError("Holdout feature artifact lacks frozen model features")
    values = frame.loc[:, list(FEATURES)].to_numpy(dtype=float)
    if frame.empty or not np.isfinite(values).all():
        raise ValueError("Holdout features must be nonempty and finite")
    return frame


def _load_sealed_outcomes(
    v2_dir: Path, manifest: dict[str, Any], plan: WalkForwardPlan
) -> pd.Series:
    payload = _verified_bytes(v2_dir, manifest, "sealed_holdout_outcomes.csv")
    frame = pd.read_csv(
        BytesIO(payload), index_col="sample_id", float_precision="round_trip"
    )
    if list(frame.columns) != ["forward_vol_5"]:
        raise ValueError("Sealed holdout outcome schema is invalid")
    if not frame.index.is_unique or tuple(frame.index.astype(str)) != plan.holdout_ids:
        raise ValueError("Sealed outcomes do not match frozen holdout membership")
    values = frame["forward_vol_5"].astype("float64")
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any():
        raise ValueError("Sealed holdout outcomes must be finite and nonnegative")
    return values


def _frequency_metrics(target: pd.Series, predictions: pd.DataFrame) -> dict[str, Any]:
    positions = np.arange(0, len(target), 5)
    return {
        "daily": classification_metrics(target, predictions),
        "every_fifth": classification_metrics(
            target.iloc[positions], predictions.iloc[positions]
        ),
    }


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    normal = actual == 0
    high = actual == 1
    return float(
        ((predicted[normal] == 0).mean() + (predicted[high] == 1).mean()) / 2
    )


def paired_block_bootstrap(
    target: pd.Series,
    selected: pd.Series,
    persistence: pd.Series,
    *,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    replicates: int = BOOTSTRAP_REPLICATES,
    max_attempts: int = BOOTSTRAP_MAX_ATTEMPTS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap the paired balanced-accuracy difference with circular blocks."""
    if not target.index.equals(selected.index) or not target.index.equals(persistence.index):
        raise ValueError("Bootstrap inputs must have identical indexes")
    if block_length <= 0 or replicates <= 0 or max_attempts < replicates:
        raise ValueError("Bootstrap configuration is invalid")
    actual = target.to_numpy(dtype=np.int64)
    chosen = selected.to_numpy(dtype=np.int64)
    baseline = persistence.to_numpy(dtype=np.int64)
    if len(actual) < 2 or set(actual) != {0, 1}:
        raise ValueError("Bootstrap requires a two-class holdout")
    rng = np.random.default_rng(seed)
    block_offsets = np.arange(block_length)
    block_count = int(np.ceil(len(actual) / block_length))
    differences: list[float] = []
    attempts = 0
    while len(differences) < replicates and attempts < max_attempts:
        attempts += 1
        starts = rng.integers(0, len(actual), size=block_count)
        indices = ((starts[:, None] + block_offsets) % len(actual)).ravel()[: len(actual)]
        sampled_actual = actual[indices]
        if set(sampled_actual) != {0, 1}:
            continue
        differences.append(
            _balanced_accuracy(sampled_actual, chosen[indices])
            - _balanced_accuracy(sampled_actual, baseline[indices])
        )
    if len(differences) != replicates:
        raise ValueError(
            f"Bootstrap obtained {len(differences)} valid replicates in {attempts} attempts"
        )
    values = np.asarray(differences, dtype=float)
    interval = np.quantile(values, [0.025, 0.975], method="linear")
    return {
        "method": "paired circular moving-block bootstrap",
        "metric": "selected balanced_accuracy - persistence balanced_accuracy",
        "block_length": block_length,
        "requested_replicates": replicates,
        "valid_replicates": len(differences),
        "max_attempts": max_attempts,
        "attempted_replicates": attempts,
        "discarded_replicates": attempts - len(differences),
        "seed": seed,
        "point_difference": _balanced_accuracy(actual, chosen)
        - _balanced_accuracy(actual, baseline),
        "mean_bootstrap_difference": float(values.mean()),
        "confidence_level": 0.95,
        "interval": [float(interval[0]), float(interval[1])],
    }


def _history_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        values = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Holdout exposure history is invalid") from exc
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("Holdout exposure history must contain JSON objects")
    return cast(list[dict[str, Any]], values)


def _reserve_exposure(
    history_path: Path,
    *,
    source_sha256: str,
    holdout_ids: tuple[str, ...],
    model_run_id: str,
    evaluation_id: str,
) -> dict[str, Any]:
    entries = _history_entries(history_path)
    holdout_key = sha256(json_bytes([source_sha256, sorted(holdout_ids)]))
    if any(entry.get("holdout_key") == holdout_key for entry in entries):
        raise ValueError("This frozen holdout was already exposed by a prior evaluation")
    event = {
        "event": "holdout_evaluation_started",
        "exposed_at_utc": utc_now(),
        "source_sha256": source_sha256,
        "holdout_key": holdout_key,
        "holdout_rows": len(holdout_ids),
        "model_run_id": model_run_id,
        "evaluation_id": evaluation_id,
    }
    payload = b"".join(
        json.dumps(item, sort_keys=True, allow_nan=False).encode() + b"\n"
        for item in [*entries, event]
    )
    atomic_write(history_path, payload, replace=history_path.exists())
    return event


def _report(metrics: dict[str, Any], bootstrap: dict[str, Any]) -> bytes:
    selected_id = metrics["selected_candidate"]
    gates = metrics["gates"]
    lines = [
        "# Volatility V4 final holdout evaluation",
        "",
        f"Status: **{metrics['research_status']}**.",
        "",
        f"The frozen `{selected_id}` model was evaluated once on {metrics['holdout_rows']} "
        f"observations from {metrics['holdout_start']} through {metrics['holdout_end']}. The "
        f"high-volatility threshold remained {metrics['threshold']:.15f}.",
        "",
        f"The holdout contained {metrics['high_support']} high-volatility and "
        f"{metrics['normal_support']} normal observations (high prevalence "
        f"{metrics['high_prevalence']:.2%}).",
        "",
    ]
    model_order = (
        selected_id,
        "baseline_majority",
        "baseline_always_normal",
        "baseline_persistence",
    )
    for frequency, heading in (("daily", "Daily"), ("every_fifth", "Every-fifth-origin")):
        lines.extend(
            [
                f"## {heading} results",
                "",
                "| Model | Accuracy | Balanced accuracy | High precision | High recall | "
                "High F1 | Macro F1 | ROC-AUC | Average precision |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name in model_order:
            result = metrics["models"][name][frequency]
            roc_auc = "undefined" if result["roc_auc"] is None else f"{result['roc_auc']:.6f}"
            average_precision = (
                "undefined"
                if result["average_precision"] is None
                else f"{result['average_precision']:.6f}"
            )
            lines.append(
                f"| {name} | {result['accuracy']:.6f} | "
                f"{result['balanced_accuracy']:.6f} | "
                f"{result['per_class']['1']['precision']:.6f} | "
                f"{result['per_class']['1']['recall']:.6f} | "
                f"{result['per_class']['1']['f1']:.6f} | {result['f1_macro']:.6f} | "
                f"{roc_auc} | {average_precision} |"
            )
        lines.append("")
    selected_daily = metrics["models"][selected_id]["daily"]
    lines.extend(
        [
            "## Selected-model confusion matrix",
            "",
            "Rows are true classes and columns are predicted classes in `[normal, high]` order: "
            f"`{selected_daily['confusion_matrix']}`.",
            "",
            "## Research gate",
            "",
            f"- Balanced accuracy exceeds majority: "
            f"**{gates['beats_majority_balanced_accuracy']}**",
            f"- Balanced accuracy exceeds persistence: "
            f"**{gates['beats_persistence_balanced_accuracy']}**",
            f"- High-volatility recall is at least persistence: "
            f"**{gates['high_recall_at_least_persistence']}**",
            f"- Bootstrap interval is strictly above zero: "
            f"**{gates['bootstrap_interval_above_zero']}**",
            "",
            f"The paired 95% moving-block bootstrap interval for balanced-accuracy improvement "
            f"over persistence was [{bootstrap['interval'][0]:.6f}, "
            f"{bootstrap['interval'][1]:.6f}], using {bootstrap['valid_replicates']} valid "
            f"replicates in {bootstrap['attempted_replicates']} attempts.",
            "",
            "Daily five-session targets overlap. The machine-readable metrics also report the "
            "predeclared every-fifth-origin evaluation. This is a risk-regime classification "
            "result, not a return forecast or trading-strategy backtest.",
            "",
        ]
    )
    return "\n".join(lines).encode()


def verify_v4_evaluation(output_dir: Path) -> dict[str, Any]:
    """Verify a completed immutable V4 evaluation and all recorded file hashes."""
    root = Path(output_dir)
    manifest = _read_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != "V4 final holdout evaluation"
        or manifest.get("state") != "evaluated"
    ):
        raise ValueError("V4 evaluation is incomplete or has an unsupported manifest")
    files = manifest.get("files")
    required = {
        "bootstrap.json",
        "holdout_predictions.csv",
        "metrics.json",
        "report.md",
    }
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("V4 manifest is missing required artifact hashes")
    for name, expected in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("V4 manifest contains an invalid artifact path")
        try:
            payload = (root / name).read_bytes()
        except OSError as exc:
            raise ValueError(f"V4 artifact is unavailable: {name}") from exc
        if sha256(payload) != expected:
            raise ValueError(f"V4 artifact hash mismatch: {name}")
    return manifest


def evaluate_volatility_run(
    run_dir: Path,
    v2_dir: Path,
    output_dir: Path,
    *,
    evaluation_id: str = "holdout-v1",
) -> VolatilityEvaluationResult:
    """Open the sealed holdout once and evaluate the frozen V3 model without refitting."""
    run_root = Path(run_dir)
    v2_root = Path(v2_dir)
    output = Path(output_dir)
    if output.name != evaluation_id:
        raise ValueError("Evaluation output directory name must equal evaluation_id")
    if output.exists():
        existing_manifest = verify_v4_evaluation(output)
        return VolatilityEvaluationResult(
            output, _read_json(output / "metrics.json"), existing_manifest
        )

    v3_manifest = verify_v3_run(run_root)
    development, plan, v2_manifest = load_v2_development(v2_root)
    if (
        v3_manifest["profile_id"] != plan.profile_id
        or v3_manifest["source_sha256"] != plan.source_sha256
        or v3_manifest["v2_manifest_sha256"]
        != sha256((v2_root / "manifest.json").read_bytes())
        or v3_manifest["final_threshold"] != plan.final_threshold
    ):
        raise ValueError("V2 data and V3 model run do not share the frozen protocol")
    features = _load_holdout_features(v2_root, v2_manifest, plan)
    model = load_volatility_model(run_root)
    feature_values = features.loc[:, list(FEATURES)]
    selected_predictions = predict_estimator(model["estimator"], feature_values)
    development_target = (
        development["forward_vol_5"] > plan.final_threshold
    ).astype("int64")
    prediction_map = {
        v3_manifest["selected_candidate"]: selected_predictions,
        **{
            f"baseline_{name}": baseline_predictions(
                name, development_target, feature_values, plan.final_threshold
            )
            for name in ("majority", "always_normal", "persistence")
        },
    }

    history_path = v2_root.parent / "holdout_evaluation_history.jsonl"
    lock_path = v2_root.parent / ".holdout_evaluation.lock"
    manifest: dict[str, Any] = {}
    with exclusive_lock(lock_path):
        if output.exists():
            existing = verify_v4_evaluation(output)
            return VolatilityEvaluationResult(
                output, _read_json(output / "metrics.json"), existing
            )
        if _history_entries(history_path):
            holdout_key = sha256(json_bytes([plan.source_sha256, sorted(plan.holdout_ids)]))
            if any(
                entry.get("holdout_key") == holdout_key
                for entry in _history_entries(history_path)
            ):
                raise ValueError("This frozen holdout was already exposed by a prior evaluation")
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "stage": "V4 final holdout evaluation",
            "state": "exposure_reserved",
            "profile_id": plan.profile_id,
            "source_sha256": plan.source_sha256,
            "model_run_id": run_root.name,
            "evaluation_id": evaluation_id,
            "model_sha256": v3_manifest["files"]["model.pkl"],
            "v3_manifest_sha256": sha256((run_root / "manifest.json").read_bytes()),
            "v2_manifest_sha256": sha256((v2_root / "manifest.json").read_bytes()),
            "holdout_rows": len(plan.holdout_ids),
            "holdout_evaluated": False,
            "environment": environment_versions(),
            "evaluation_code": source_identity(),
            "created_at_utc": utc_now(),
        }
        atomic_write(output / "manifest.json", json_bytes(manifest))
        try:
            exposure = _reserve_exposure(
                history_path,
                source_sha256=plan.source_sha256,
                holdout_ids=plan.holdout_ids,
                model_run_id=run_root.name,
                evaluation_id=evaluation_id,
            )
            outcomes = _load_sealed_outcomes(v2_root, v2_manifest, plan)
            target = (outcomes > plan.final_threshold).astype("int64")
            if set(target) != {0, 1}:
                raise ValueError("Final holdout must contain both classes")
            model_metrics = {
                name: _frequency_metrics(target, predictions)
                for name, predictions in prediction_map.items()
            }
            selected_id = v3_manifest["selected_candidate"]
            selected_daily = model_metrics[selected_id]["daily"]
            majority_daily = model_metrics["baseline_majority"]["daily"]
            persistence_daily = model_metrics["baseline_persistence"]["daily"]
            bootstrap = paired_block_bootstrap(
                target,
                prediction_map[selected_id]["predicted_class"],
                prediction_map["baseline_persistence"]["predicted_class"],
            )
            gates = {
                "beats_majority_balanced_accuracy": selected_daily["balanced_accuracy"]
                > majority_daily["balanced_accuracy"],
                "beats_persistence_balanced_accuracy": selected_daily["balanced_accuracy"]
                > persistence_daily["balanced_accuracy"],
                "high_recall_at_least_persistence": selected_daily["per_class"]["1"]["recall"]
                >= persistence_daily["per_class"]["1"]["recall"],
                "bootstrap_interval_above_zero": bootstrap["interval"][0] > 0,
            }
            metrics = {
                "schema_version": 1,
                "profile_id": plan.profile_id,
                "model_run_id": run_root.name,
                "evaluation_id": evaluation_id,
                "selected_candidate": selected_id,
                "threshold": plan.final_threshold,
                "holdout_rows": len(target),
                "holdout_start": features["as_of"].iloc[0].date().isoformat(),
                "holdout_end": features["as_of"].iloc[-1].date().isoformat(),
                "high_support": int(target.sum()),
                "normal_support": int((target == 0).sum()),
                "high_prevalence": float(target.mean()),
                "models": model_metrics,
                "gates": gates,
                "research_status": "passed" if all(gates.values()) else "target_not_met",
                "evaluated_at_utc": utc_now(),
            }
            predictions = pd.DataFrame(
                {
                    "sample_id": target.index,
                    "as_of": features["as_of"].to_numpy(),
                    "label_start": features["label_start"].to_numpy(),
                    "label_end": features["label_end"].to_numpy(),
                    "forward_vol_5": outcomes.to_numpy(),
                    "threshold": plan.final_threshold,
                    "true_class": target.to_numpy(),
                    "secondary_every_fifth": np.arange(len(target)) % 5 == 0,
                }
            )
            for name, values in prediction_map.items():
                predictions[f"{name}_predicted_class"] = values["predicted_class"].to_numpy()
                predictions[f"{name}_score_class_1"] = values["score_class_1"].to_numpy()
            payloads = {
                "metrics.json": json_bytes(metrics),
                "holdout_predictions.csv": predictions.to_csv(index=False).encode(),
                "bootstrap.json": json_bytes(bootstrap),
                "report.md": _report(metrics, bootstrap),
            }
            for name, payload in payloads.items():
                atomic_write(output / name, payload)
            manifest.update(
                {
                    "state": "evaluated",
                    "holdout_evaluated": True,
                    "research_status": metrics["research_status"],
                    "selected_candidate": selected_id,
                    "threshold": plan.final_threshold,
                    "exposure": exposure,
                    "files": {name: sha256(payload) for name, payload in payloads.items()},
                    "completed_at_utc": utc_now(),
                }
            )
            atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
        except Exception as exc:
            manifest.update(
                {
                    "state": "invalid",
                    "holdout_evaluated": True,
                    "failure": f"{type(exc).__name__}: {exc}",
                    "failed_at_utc": utc_now(),
                }
            )
            atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
            raise
    return VolatilityEvaluationResult(output, metrics, manifest)

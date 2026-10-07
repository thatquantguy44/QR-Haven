"""Single-use final evaluation for a frozen Spec002 volatility challenger."""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, sha256, utc_now
from qr_haven.ml.classification.artifacts import exclusive_lock
from qr_haven.ml.classification.evaluation import classification_metrics
from qr_haven.ml.volatility.challenger_data import (
    CHALLENGER_FEATURES,
    PROTOCOL,
    TIINGO_PROTOCOL,
    get_challenger_design,
    load_challenger_development,
    load_challenger_evaluation_features,
    verify_challenger_dataset,
)
from qr_haven.ml.volatility.challenger_models import (
    ChallengerCandidate,
    calibrated_scores,
    ensemble_predictions,
    raw_scores,
)
from qr_haven.ml.volatility.challenger_training import (
    load_challenger_model,
    verify_challenger_run,
)
from qr_haven.ml.volatility.models import baseline_predictions
from qr_haven.ml.volatility.persistence import environment_versions, source_identity

BLOCK_LENGTH = 20
REPLICATES = 2_000
MAX_ATTEMPTS = 20_000


def _verified_outcomes(dataset_dir: Path, manifest: dict[str, Any]) -> pd.Series:
    payload = (dataset_dir / "sealed_evaluation_outcomes.csv").read_bytes()
    if manifest["files"].get("sealed_evaluation_outcomes.csv") != sha256(payload):
        raise ValueError("Sealed challenger outcomes hash mismatch")
    frame = pd.read_csv(BytesIO(payload), index_col="sample_id", float_precision="round_trip")
    if list(frame.columns) != ["forward_vol_5"] or not frame.index.is_unique:
        raise ValueError("Sealed challenger outcomes have an invalid schema")
    values = frame["forward_vol_5"].astype("float64")
    if values.empty or not np.isfinite(values.to_numpy()).all() or (values < 0).any():
        raise ValueError("Sealed challenger outcomes must be finite and nonnegative")
    return values


def _predict_raw(bundle: dict[str, Any], features: pd.DataFrame) -> pd.Series:
    candidate = ChallengerCandidate(**bundle["candidate"])
    if candidate.kind != "ewma":
        return raw_scores(
            candidate,
            bundle["estimator"],
            features,
            tuple(features.index.astype(str)),
        )
    variance = bundle.get("ewma_variance_after_boundary")
    if not isinstance(variance, float) or not np.isfinite(variance) or variance < 0:
        raise ValueError("Frozen EWMA challenger lacks a valid boundary state")
    if candidate.decay is None:
        raise ValueError("Frozen EWMA challenger lacks its decay")
    decay = candidate.decay
    values: list[float] = []
    for observed_return in features["log_return_1"].to_numpy(float):
        variance = decay * variance + (1 - decay) * observed_return**2
        values.append(float(np.sqrt(252 * variance)))
    return pd.Series(values, index=features.index, name="raw_score")


def _frequency_metrics(target: pd.Series, prediction: pd.DataFrame) -> dict[str, Any]:
    every_fifth = np.arange(0, len(target), 5)
    return {
        "daily": classification_metrics(target, prediction),
        "every_fifth": classification_metrics(
            target.iloc[every_fifth], prediction.iloc[every_fifth]
        ),
    }


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    normal, high = actual == 0, actual == 1
    return float(((predicted[normal] == 0).mean() + (predicted[high] == 1).mean()) / 2)


def _bootstrap(
    target: pd.Series, chosen: pd.Series, persistence: pd.Series, *, seed: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    actual = target.to_numpy(np.int64)
    selected = chosen.to_numpy(np.int64)
    baseline = persistence.to_numpy(np.int64)
    if set(actual) != {0, 1}:
        raise ValueError("Challenger bootstrap requires both evaluation classes")
    rng = np.random.default_rng(seed)
    offsets = np.arange(BLOCK_LENGTH)
    block_count = int(np.ceil(len(actual) / BLOCK_LENGTH))
    differences: list[float] = []
    attempts = 0
    while len(differences) < REPLICATES and attempts < MAX_ATTEMPTS:
        attempts += 1
        starts = rng.integers(0, len(actual), size=block_count)
        indices = ((starts[:, None] + offsets) % len(actual)).ravel()[: len(actual)]
        sample = actual[indices]
        if set(sample) != {0, 1}:
            continue
        differences.append(
            _balanced_accuracy(sample, selected[indices])
            - _balanced_accuracy(sample, baseline[indices])
        )
    if len(differences) != REPLICATES:
        raise ValueError("Challenger bootstrap could not obtain enough two-class replicates")
    values = np.asarray(differences)
    interval = np.quantile(values, [0.025, 0.975], method="linear")
    summary = {
        "method": "paired circular moving-block bootstrap",
        "metric": "challenger balanced_accuracy - persistence balanced_accuracy",
        "block_length": BLOCK_LENGTH,
        "valid_replicates": REPLICATES,
        "attempted_replicates": attempts,
        "discarded_replicates": attempts - REPLICATES,
        "seed": seed,
        "point_difference": _balanced_accuracy(actual, selected)
        - _balanced_accuracy(actual, baseline),
        "mean_bootstrap_difference": float(values.mean()),
        "confidence_level": 0.95,
        "interval": [float(interval[0]), float(interval[1])],
    }
    samples = pd.DataFrame(
        {"replicate": np.arange(1, REPLICATES + 1), "balanced_accuracy_difference": values}
    )
    return summary, samples


def _append_ledger(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _report(metrics: dict[str, Any], bootstrap: dict[str, Any], selected: str) -> bytes:
    daily = metrics["models"]
    lines = [
        "# Frozen volatility challenger: final evaluation",
        "",
        f"Status: **{metrics['research_status'].replace('_', ' ')}**.",
        "",
        f"The frozen `{selected}` pipeline was evaluated once on {metrics['evaluation_rows']} "
        f"origins from {metrics['evaluation_start']} through {metrics['evaluation_end']}.",
        "",
        "| Model | Accuracy | Balanced accuracy | High precision | High recall | "
        "High F1 | ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_id, values in daily.items():
        lines.append(
            f"| {model_id} | {values['accuracy']:.4f} | {values['balanced_accuracy']:.4f} | "
            f"{values['per_class']['1']['precision']:.4f} | "
            f"{values['per_class']['1']['recall']:.4f} | "
            f"{values['per_class']['1']['f1']:.4f} | {values['roc_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Research gate",
            "",
            f"Paired balanced-accuracy difference versus persistence: "
            f"{bootstrap['point_difference']:+.4f}; 95% block-bootstrap interval "
            f"[{bootstrap['interval'][0]:+.4f}, {bootstrap['interval'][1]:+.4f}].",
            "",
            f"- Above majority and persistence: {metrics['gates']['above_both_baselines']}",
            f"- High-volatility recall at least persistence: "
            f"{metrics['gates']['recall_at_least_persistence']}",
            f"- Bootstrap interval strictly above zero: {metrics['gates']['bootstrap_above_zero']}",
            "",
            "The candidate, feature set, calibration, ensemble weight, training membership, and "
            "threshold were frozen before the exposure ledger was written and outcomes opened. "
            "The evaluation period was fixed by the profile-specific protocol before outcomes "
            "were opened and should be interpreted in its market context.",
            "",
        ]
    )
    return "\n".join(lines).encode()


def evaluate_challenger(
    run_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    *,
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    run, dataset, output = Path(run_dir), Path(dataset_dir), Path(output_dir)
    dataset_header = json.loads((dataset / "manifest.json").read_text())
    design = get_challenger_design(str(dataset_header["profile_id"]))
    evaluation_id = evaluation_id or design.evaluation_id
    if output.name != evaluation_id:
        raise ValueError("Challenger evaluation directory name must equal evaluation_id")
    if output.exists():
        return verify_challenger_evaluation(output)
    run_manifest = verify_challenger_run(run)
    dataset_manifest = verify_challenger_dataset(dataset, include_sealed_outcomes=False)
    if (
        run_manifest["protocol"] != design.protocol
        or dataset_manifest["protocol"] != design.protocol
    ):
        raise ValueError("Challenger run and dataset do not match the frozen profile design")
    if run_manifest["dataset_manifest_sha256"] != sha256((dataset / "manifest.json").read_bytes()):
        raise ValueError("Challenger run is not bound to this dataset")
    bundle = load_challenger_model(run)
    features = load_challenger_evaluation_features(dataset, dataset_manifest)
    raw = _predict_raw(bundle, features)
    probability = calibrated_scores(bundle["calibrator"], raw)
    selected = ensemble_predictions(
        probability,
        features["trailing_vol_5"],
        float(bundle["threshold"]),
        float(bundle["model_weight"]),
    )
    development, _ = load_challenger_development(dataset)
    final_target = (
        development.loc[list(bundle["final_training_ids"]), "forward_vol_5"]
        > float(bundle["threshold"])
    ).astype("int64")
    baselines = {
        name: baseline_predictions(
            name,
            final_target,
            features.loc[:, list(CHALLENGER_FEATURES)],
            float(bundle["threshold"]),
        )
        for name in ("majority", "always_normal", "persistence")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": design.protocol,
        "stage": "single frozen final evaluation",
        "state": "reserved",
        "evaluation_id": evaluation_id,
        "created_at_utc": utc_now(),
        "run_manifest_sha256": sha256((run / "manifest.json").read_bytes()),
        "dataset_manifest_sha256": sha256((dataset / "manifest.json").read_bytes()),
        "source_sha256": dataset_manifest["source_sha256"],
        "model_sha256": run_manifest["files"]["model.pkl"],
        "evaluation_outcomes_opened": False,
        "environment": environment_versions(),
        "code": source_identity(),
    }
    atomic_write(output / "manifest.json", json_bytes(manifest))
    history = dataset.parent / "challenger_evaluation_history.jsonl"
    lock = dataset.parent / ".challenger_evaluation.lock"
    population_key = sha256(
        json_bytes([dataset_manifest["source_sha256"], sorted(features.index.astype(str))])
    )
    try:
        with exclusive_lock(lock):
            entries = []
            if history.exists():
                entries = [json.loads(line) for line in history.read_text().splitlines() if line]
            if any(entry.get("population_key") == population_key for entry in entries):
                raise ValueError("This frozen challenger evaluation population was already exposed")
            _append_ledger(
                history,
                {
                    "event": "challenger_evaluation_started",
                    "at_utc": utc_now(),
                    "evaluation_id": evaluation_id,
                    "population_key": population_key,
                    "evaluation_rows": len(features),
                    "run_manifest_sha256": manifest["run_manifest_sha256"],
                },
            )
            outcomes = _verified_outcomes(dataset, dataset_manifest)
            if not outcomes.index.equals(features.index):
                raise ValueError("Challenger features and sealed outcomes do not align")
            target = (outcomes > float(bundle["threshold"])).astype("int64")
            models = {"challenger": selected, **{f"baseline_{k}": v for k, v in baselines.items()}}
            all_metrics = {
                name: _frequency_metrics(target, frame) for name, frame in models.items()
            }
            bootstrap, samples = _bootstrap(
                target,
                selected["predicted_class"],
                baselines["persistence"]["predicted_class"],
                seed=design.bootstrap_seed,
            )
            daily = {name: values["daily"] for name, values in all_metrics.items()}
            gates = {
                "above_both_baselines": daily["challenger"]["balanced_accuracy"]
                > max(
                    daily["baseline_majority"]["balanced_accuracy"],
                    daily["baseline_persistence"]["balanced_accuracy"],
                ),
                "recall_at_least_persistence": daily["challenger"]["per_class"]["1"]["recall"]
                >= daily["baseline_persistence"]["per_class"]["1"]["recall"],
                "bootstrap_above_zero": bootstrap["interval"][0] > 0,
            }
            metrics = {
                "protocol": design.protocol,
                "evaluation_rows": len(target),
                "evaluation_start": features["as_of"].iloc[0].date().isoformat(),
                "evaluation_end": features["as_of"].iloc[-1].date().isoformat(),
                "threshold": float(bundle["threshold"]),
                "high_support": int((target == 1).sum()),
                "normal_support": int((target == 0).sum()),
                "models": daily,
                "every_fifth": {
                    name: values["every_fifth"] for name, values in all_metrics.items()
                },
                "gates": gates,
                "research_status": "passed" if all(gates.values()) else "target_not_met",
            }
            predictions = pd.DataFrame(
                {
                    "sample_id": features.index,
                    "as_of": features["as_of"].to_numpy(),
                    "forward_vol_5": outcomes.to_numpy(),
                    "threshold": float(bundle["threshold"]),
                    "true_class": target.to_numpy(),
                    "raw_score": raw.to_numpy(),
                    "calibrated_probability": probability.to_numpy(),
                    "challenger_score_class_1": selected["score_class_1"].to_numpy(),
                    "challenger_predicted_class": selected["predicted_class"].to_numpy(),
                    **{
                        f"baseline_{name}_predicted_class": frame["predicted_class"].to_numpy()
                        for name, frame in baselines.items()
                    },
                }
            )
            payloads = {
                "metrics.json": json_bytes(metrics),
                "bootstrap.json": json_bytes(bootstrap),
                "bootstrap_samples.csv": samples.to_csv(index=False).encode(),
                "evaluation_predictions.csv": predictions.to_csv(index=False).encode(),
                "report.md": _report(metrics, bootstrap, run_manifest["selected_candidate"]),
            }
            for name, payload in payloads.items():
                atomic_write(output / name, payload)
            manifest.update(
                {
                    "state": "complete",
                    "evaluation_outcomes_opened": True,
                    "research_status": metrics["research_status"],
                    "population_key": population_key,
                    "files": {name: sha256(payload) for name, payload in payloads.items()},
                }
            )
            atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
    except Exception as exc:
        manifest.update({"state": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
        raise
    return manifest


def verify_challenger_evaluation(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    value = json.loads((root / "manifest.json").read_text())
    if value.get("protocol") not in {PROTOCOL, TIINGO_PROTOCOL} or value.get("state") != "complete":
        raise ValueError("Unsupported or incomplete challenger evaluation")
    required = {
        "metrics.json",
        "bootstrap.json",
        "bootstrap_samples.csv",
        "evaluation_predictions.csv",
        "report.md",
    }
    hashes = value.get("files", {})
    if not required <= hashes.keys():
        raise ValueError("Challenger evaluation manifest lacks required files")
    for name, digest in hashes.items():
        if Path(name).name != name or sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"Challenger evaluation artifact hash mismatch: {name}")
    return cast(dict[str, Any], value)

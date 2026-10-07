"""V8A immutable 2026 YTD provisional promotion workflow."""

from __future__ import annotations

import json
import os
import pickle
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, sha256, utc_now
from qr_haven.data.tiingo import load_tiingo_spy_snapshot
from qr_haven.ml.classification.artifacts import exclusive_lock
from qr_haven.ml.volatility.challenger_data import (
    CHALLENGER_FEATURES,
    TIINGO_PROTOCOL,
    extend_challenger_features,
    load_challenger_development,
)
from qr_haven.ml.volatility.continuous_experiments import (
    EPSILON,
    _forecast_metrics,
    _inverse_log_variance,
    blend_variance,
    qlike_rows,
    verify_continuous_experiment,
)
from qr_haven.ml.volatility.contracts import VolatilityProfile
from qr_haven.ml.volatility.dataset import build_volatility_dataset
from qr_haven.ml.volatility.persistence import environment_versions, source_identity

PROTOCOL = "volatility-continuous-ytd-v1"
EXTENSION_START = date(2025, 10, 1)
EXTENSION_END = date(2026, 10, 2)
EVALUATION_START = pd.Timestamp("2026-01-02")
EVALUATION_END = pd.Timestamp("2026-09-25")
ALERT_THRESHOLD = 0.197901781191757
MODEL_WEIGHT = 0.75
BLOCK_LENGTH = 20
REPLICATES = 2_000
BOOTSTRAP_SEED = 5413
BASE_SNAPSHOT_DIR = Path("data/raw/market_data/tiingo/spy-2005-2025-v1")
EXTENSION_SNAPSHOT_DIR = Path(
    "data/raw/market_data/tiingo/spy-2025-10-01-2026-10-02-v1"
)
DEVELOPMENT_DATASET_DIR = Path(
    "artifacts/classification/volatility/tiingo-spy-v1/challengers/dataset-v1"
)
V8_RUN_DIR = Path(
    "artifacts/classification/volatility/tiingo-spy-v1/continuous_experiments/v8-continuous-v1"
)


def _extension_profile(snapshot_dir: Path) -> VolatilityProfile:
    return VolatilityProfile(
        profile_id="tiingo-spy-ytd-v1",
        symbol="SPY",
        snapshot_dir=snapshot_dir,
        canonical_filename="spy_daily_canonical.csv",
        study_start=EXTENSION_START,
        study_end=EXTENSION_END,
        validation_years=(),
        holdout_start=date(2026, 1, 2),
        holdout_end=date(2026, 9, 25),
    )


def _verify_overlap(base: pd.DataFrame, extension: pd.DataFrame) -> int:
    columns = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "frequency",
    ]
    left = base.copy()
    right = extension.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)
    overlap_end = pd.Timestamp("2025-12-31", tz="UTC")
    overlap_start = pd.Timestamp(EXTENSION_START, tz="UTC")
    left = left.loc[left["timestamp"].between(overlap_start, overlap_end), columns].reset_index(
        drop=True
    )
    right = right.loc[right["timestamp"] <= overlap_end, columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=True)
    except AssertionError as exc:
        raise ValueError(
            "Tiingo extension overlap differs from the frozen 2005–2025 source"
        ) from exc
    if left.empty:
        raise ValueError("Tiingo extension has no overlap with the frozen source")
    return len(left)


def prepare_ytd_dataset(
    output_dir: Path,
    *,
    extension_dir: Path = EXTENSION_SNAPSHOT_DIR,
    base_dir: Path = BASE_SNAPSHOT_DIR,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable V8A dataset: {output}")
    base = load_tiingo_spy_snapshot(base_dir)
    extension = load_tiingo_spy_snapshot(extension_dir)
    if (
        extension.manifest.get("requested_start") != EXTENSION_START.isoformat()
        or extension.manifest.get("requested_end") != EXTENSION_END.isoformat()
    ):
        raise ValueError("Tiingo extension dates do not match the frozen V8A protocol")
    overlap_rows = _verify_overlap(base.canonical, extension.canonical)
    profile = _extension_profile(extension_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "stage": "V8A point-in-time evaluation dataset",
        "state": "running",
        "created_at_utc": utc_now(),
        "base_source_sha256": base.manifest["raw_price_sha256"],
        "extension_source_sha256": extension.manifest["raw_price_sha256"],
        "overlap_rows": overlap_rows,
        "evaluation_outcomes_summarized": False,
        "environment": environment_versions(),
        "code": source_identity(),
    }
    atomic_write(output / "manifest.json", json_bytes(manifest))
    try:
        source = {
            "source_sha256": extension.manifest["raw_price_sha256"],
            "provider": extension.manifest["provider"],
        }
        dataset = build_volatility_dataset(extension.canonical, source, profile)
        observations, audit = extend_challenger_features(dataset, extension.canonical)
        mask = (
            observations["as_of"].between(EVALUATION_START, EVALUATION_END)
            & (observations["label_end"] <= pd.Timestamp(EXTENSION_END))
        )
        evaluation = observations.loc[mask]
        evaluation_audit = audit.loc[evaluation.index]
        if (
            evaluation.empty
            or evaluation["as_of"].iloc[0] != EVALUATION_START
            or evaluation["as_of"].iloc[-1] != EVALUATION_END
            or not (evaluation["label_end"] <= pd.Timestamp(EXTENSION_END)).all()
        ):
            raise ValueError("V8A evaluation membership does not match the frozen population")
        files = {
            "evaluation_features.csv": evaluation.drop(columns="forward_vol_5")
            .to_csv(index=True)
            .encode(),
            "sealed_evaluation_outcomes.csv": evaluation.loc[:, ["forward_vol_5"]]
            .to_csv(index=True)
            .encode(),
            "feature_window_audit.csv": evaluation_audit.to_csv(index=True).encode(),
            "membership.csv": pd.DataFrame(
                {"sample_id": evaluation.index, "partition": "evaluation_2026_ytd"}
            )
            .to_csv(index=False)
            .encode(),
            "config.json": json_bytes(
                {
                    "protocol": PROTOCOL,
                    "feature_order": list(CHALLENGER_FEATURES),
                    "evaluation_start": EVALUATION_START.date().isoformat(),
                    "evaluation_end": EVALUATION_END.date().isoformat(),
                    "label_completion_end": EXTENSION_END.isoformat(),
                    "alert_threshold": ALERT_THRESHOLD,
                }
            ),
        }
        for name, payload in files.items():
            atomic_write(output / name, payload)
        manifest.update(
            {
                "state": "complete",
                "evaluation_rows": len(evaluation),
                "evaluation_start": evaluation["as_of"].iloc[0].date().isoformat(),
                "evaluation_end": evaluation["as_of"].iloc[-1].date().isoformat(),
                "evaluation_outcomes_summarized": False,
                "files": {name: sha256(payload) for name, payload in files.items()},
            }
        )
        atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
    except Exception as exc:
        manifest.update({"state": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
        raise
    return manifest


def verify_ytd_dataset(
    output_dir: Path, *, include_sealed_outcomes: bool = True
) -> dict[str, Any]:
    root = Path(output_dir)
    value = json.loads((root / "manifest.json").read_text())
    if value.get("protocol") != PROTOCOL or value.get("state") != "complete":
        raise ValueError("Unsupported or incomplete V8A dataset")
    required = {
        "evaluation_features.csv",
        "sealed_evaluation_outcomes.csv",
        "feature_window_audit.csv",
        "membership.csv",
        "config.json",
    }
    hashes = value.get("files", {})
    if not required <= hashes.keys():
        raise ValueError("V8A dataset manifest lacks required files")
    for name, digest in hashes.items():
        if name == "sealed_evaluation_outcomes.csv" and not include_sealed_outcomes:
            continue
        if Path(name).name != name or sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"V8A dataset artifact hash mismatch: {name}")
    return cast(dict[str, Any], value)


def _load_ytd_features(output_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    root = Path(output_dir)
    payload = (root / "evaluation_features.csv").read_bytes()
    if manifest["files"].get("evaluation_features.csv") != sha256(payload):
        raise ValueError("V8A evaluation feature hash mismatch")
    frame = pd.read_csv(
        BytesIO(payload),
        index_col="sample_id",
        parse_dates=["as_of", "available_at", "label_start", "label_end"],
        float_precision="round_trip",
    )
    if (
        frame.empty
        or not frame.index.is_unique
        or frame["as_of"].iloc[0] != EVALUATION_START
        or frame["as_of"].iloc[-1] != EVALUATION_END
    ):
        raise ValueError("V8A evaluation features have invalid membership")
    return frame


def train_ytd_candidate(
    development_dir: Path,
    v8_run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    development, v8_run, output = Path(development_dir), Path(v8_run_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable V8A candidate: {output}")
    observations, dataset_manifest = load_challenger_development(development)
    selected = verify_continuous_experiment(v8_run)
    selection = json.loads((v8_run / "selection.json").read_text())
    if (
        dataset_manifest.get("protocol") != TIINGO_PROTOCOL
        or selection.get("candidate_id") != "hist_gradient_boosting_regression_w075"
        or selection.get("model_weight") != MODEL_WEIGHT
        or selected.get("candidate_advances") is not True
    ):
        raise ValueError("V8A inputs do not match the frozen advancing candidate")
    boundary = pd.Timestamp("2024-01-01")
    training_rows = observations.loc[
        (observations["as_of"] >= boundary - pd.DateOffset(years=12))
        & (observations["as_of"] < boundary)
        & (observations["label_end"] < boundary)
    ]
    if training_rows.empty:
        raise ValueError("V8A through-2023 training membership is empty")
    training_ids = tuple(training_rows.index.astype(str))
    training = observations.loc[list(training_ids)]
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "stage": "V8A frozen candidate fit",
        "state": "running",
        "created_at_utc": utc_now(),
        "development_manifest_sha256": sha256((development / "manifest.json").read_bytes()),
        "v8_run_manifest_sha256": sha256((v8_run / "manifest.json").read_bytes()),
        "source_sha256": dataset_manifest["source_sha256"],
        "evaluation_outcomes_opened": False,
        "environment": environment_versions(),
        "code": source_identity(),
    }
    atomic_write(output / "manifest.json", json_bytes(manifest))
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from threadpoolctl import threadpool_limits

        estimator = HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            max_iter=300,
            early_stopping=False,
            random_state=5402,
        )
        target = np.log(np.square(training["forward_vol_5"].to_numpy(float)) + EPSILON)
        with threadpool_limits(limits=1):
            estimator.fit(training.loc[:, list(CHALLENGER_FEATURES)], target)
        bundle = {
            "protocol": PROTOCOL,
            "candidate_id": "hist_gradient_boosting_regression_w075",
            "base_id": "hist_gradient_boosting_regression",
            "model_weight": MODEL_WEIGHT,
            "alert_threshold": ALERT_THRESHOLD,
            "feature_order": CHALLENGER_FEATURES,
            "training_ids": training_ids,
            "estimator": estimator,
            "source_sha256": dataset_manifest["source_sha256"],
        }
        payloads = {
            "model.pkl": pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL),
            "training_membership.csv": pd.DataFrame({"sample_id": training_ids})
            .to_csv(index=False)
            .encode(),
            "fit_audit.json": json_bytes(
                {
                    "training_rows": len(training),
                    "training_start": training["as_of"].iloc[0].date().isoformat(),
                    "training_end": training["as_of"].iloc[-1].date().isoformat(),
                    "max_training_label_end": training["label_end"].max().date().isoformat(),
                    "validation_boundary": "2024-01-01",
                    "uses_2024_2025_outcomes": False,
                }
            ),
            "config.json": json_bytes(
                {
                    "protocol": PROTOCOL,
                    "candidate_id": bundle["candidate_id"],
                    "model_weight": MODEL_WEIGHT,
                    "alert_threshold": ALERT_THRESHOLD,
                    "feature_order": list(CHALLENGER_FEATURES),
                }
            ),
        }
        for name, payload in payloads.items():
            atomic_write(output / name, payload)
        manifest.update(
            {
                "state": "complete",
                "selected_candidate": bundle["candidate_id"],
                "selected_model_weight": MODEL_WEIGHT,
                "alert_threshold": ALERT_THRESHOLD,
                "training_rows": len(training),
                "training_end": training["as_of"].iloc[-1].date().isoformat(),
                "evaluation_outcomes_opened": False,
                "files": {name: sha256(payload) for name, payload in payloads.items()},
            }
        )
        atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
    except Exception as exc:
        manifest.update({"state": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
        raise
    return manifest


def verify_ytd_candidate(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    value = json.loads((root / "manifest.json").read_text())
    if value.get("protocol") != PROTOCOL or value.get("state") != "complete":
        raise ValueError("Unsupported or incomplete V8A candidate")
    required = {"model.pkl", "training_membership.csv", "fit_audit.json", "config.json"}
    hashes = value.get("files", {})
    if not required <= hashes.keys():
        raise ValueError("V8A candidate manifest lacks required files")
    for name, digest in hashes.items():
        if Path(name).name != name or sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"V8A candidate artifact hash mismatch: {name}")
    return cast(dict[str, Any], value)


def _load_ytd_model(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest = verify_ytd_candidate(root)
    value = pickle.loads((root / "model.pkl").read_bytes())  # noqa: S301 - verified local artifact
    if (
        not isinstance(value, dict)
        or value.get("protocol") != PROTOCOL
        or value.get("candidate_id") != manifest["selected_candidate"]
        or tuple(value.get("feature_order", ())) != CHALLENGER_FEATURES
    ):
        raise ValueError("V8A model bundle disagrees with its manifest")
    return cast(dict[str, Any], value)


def _append_ledger(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _bootstrap_qlike(difference: pd.Series) -> tuple[dict[str, Any], pd.DataFrame]:
    values = difference.to_numpy(float)
    if not np.isfinite(values).all() or len(values) < BLOCK_LENGTH:
        raise ValueError("V8A bootstrap requires enough finite paired losses")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    offsets = np.arange(BLOCK_LENGTH)
    block_count = int(np.ceil(len(values) / BLOCK_LENGTH))
    samples = np.empty(REPLICATES, dtype=float)
    for index in range(REPLICATES):
        starts = rng.integers(0, len(values), size=block_count)
        positions = ((starts[:, None] + offsets) % len(values)).ravel()[: len(values)]
        samples[index] = float(values[positions].mean())
    interval = np.quantile(samples, [0.025, 0.975], method="linear")
    summary = {
        "method": "paired circular moving-block bootstrap",
        "metric": "persistence QLIKE - candidate QLIKE",
        "block_length": BLOCK_LENGTH,
        "valid_replicates": REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "point_difference": float(values.mean()),
        "mean_bootstrap_difference": float(samples.mean()),
        "confidence_level": 0.95,
        "interval": [float(interval[0]), float(interval[1])],
    }
    frame = pd.DataFrame(
        {"replicate": np.arange(1, REPLICATES + 1), "qlike_improvement": samples}
    )
    return summary, frame


def _verified_ytd_outcomes(dataset_dir: Path, manifest: dict[str, Any]) -> pd.Series:
    payload = (dataset_dir / "sealed_evaluation_outcomes.csv").read_bytes()
    if manifest["files"].get("sealed_evaluation_outcomes.csv") != sha256(payload):
        raise ValueError("V8A sealed outcome hash mismatch")
    frame = pd.read_csv(BytesIO(payload), index_col="sample_id", float_precision="round_trip")
    values = frame["forward_vol_5"].astype("float64")
    if frame.columns.tolist() != ["forward_vol_5"] or (values <= 0).any():
        raise ValueError("V8A sealed outcomes are invalid")
    return values


def _report(metrics: dict[str, Any], bootstrap: dict[str, Any]) -> bytes:
    candidate = metrics["candidate"]
    persistence = metrics["persistence"]
    lines = [
        "# V8A provisional 2026 YTD promotion evaluation",
        "",
        f"Status: **{metrics['research_status'].replace('_', ' ')}**.",
        "",
        f"The frozen V8 candidate was evaluated once on {metrics['evaluation_rows']} origins "
        f"from {metrics['evaluation_start']} through {metrics['evaluation_end']}.",
        "",
        "| Forecast | QLIKE | Log MAE | Volatility MAE | Volatility RMSE | Alert recall | "
        "Alert cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| candidate | {candidate['mean_qlike']:.6f} | {candidate['log_mae']:.6f} | "
        f"{candidate['volatility_mae']:.6f} | {candidate['volatility_rmse']:.6f} | "
        f"{candidate['alert_high_recall']:.6f} | {candidate['alert_cost']:.6f} |",
        f"| persistence | {persistence['mean_qlike']:.6f} | {persistence['log_mae']:.6f} | "
        f"{persistence['volatility_mae']:.6f} | {persistence['volatility_rmse']:.6f} | "
        f"{persistence['alert_high_recall']:.6f} | {persistence['alert_cost']:.6f} |",
        "",
        f"Paired QLIKE improvement: {bootstrap['point_difference']:+.6f}; 95% interval "
        f"[{bootstrap['interval'][0]:+.6f}, {bootstrap['interval'][1]:+.6f}].",
        "",
        "## Provisional gate",
        "",
        *[f"- {name}: {passed}" for name, passed in metrics["gates"].items()],
        "",
        "This YTD gate was frozen before 2026 acquisition. Passing permits provisional promotion; "
        "the complete-calendar-2026 V8 gate remains pending.",
        "",
    ]
    return "\n".join(lines).encode()


def evaluate_ytd_candidate(
    candidate_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    candidate, dataset, output = Path(candidate_dir), Path(dataset_dir), Path(output_dir)
    if output.exists():
        return verify_ytd_evaluation(output)
    candidate_manifest = verify_ytd_candidate(candidate)
    dataset_manifest = verify_ytd_dataset(dataset, include_sealed_outcomes=False)
    bundle = _load_ytd_model(candidate)
    features = _load_ytd_features(dataset, dataset_manifest)
    predicted_log_variance = bundle["estimator"].predict(
        features.loc[:, list(CHALLENGER_FEATURES)]
    )
    base = pd.Series(
        _inverse_log_variance(np.asarray(predicted_log_variance, dtype=float)),
        index=features.index,
        name="base_forecast",
    )
    persistence = features["trailing_vol_5"].astype("float64")
    forecast = blend_variance(base, persistence, MODEL_WEIGHT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "stage": "single frozen V8A evaluation",
        "state": "reserved",
        "created_at_utc": utc_now(),
        "candidate_manifest_sha256": sha256((candidate / "manifest.json").read_bytes()),
        "dataset_manifest_sha256": sha256((dataset / "manifest.json").read_bytes()),
        "extension_source_sha256": dataset_manifest["extension_source_sha256"],
        "model_sha256": candidate_manifest["files"]["model.pkl"],
        "evaluation_outcomes_opened": False,
        "environment": environment_versions(),
        "code": source_identity(),
    }
    atomic_write(output / "manifest.json", json_bytes(manifest))
    history = dataset.parent / "evaluation_history.jsonl"
    lock = dataset.parent / ".evaluation.lock"
    population_key = sha256(
        json_bytes(
            [dataset_manifest["extension_source_sha256"], sorted(features.index.astype(str))]
        )
    )
    try:
        with exclusive_lock(lock):
            entries = []
            if history.exists():
                entries = [json.loads(line) for line in history.read_text().splitlines() if line]
            if any(entry.get("population_key") == population_key for entry in entries):
                raise ValueError("This V8A evaluation population was already exposed")
            _append_ledger(
                history,
                {
                    "event": "v8a_evaluation_started",
                    "at_utc": utc_now(),
                    "population_key": population_key,
                    "evaluation_rows": len(features),
                    "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
                },
            )
            outcomes = _verified_ytd_outcomes(dataset, dataset_manifest)
            if not outcomes.index.equals(features.index):
                raise ValueError("V8A features and sealed outcomes do not align")
            threshold = pd.Series(ALERT_THRESHOLD, index=features.index, name="threshold")
            candidate_metrics = _forecast_metrics(outcomes, forecast, threshold)
            persistence_metrics = _forecast_metrics(outcomes, persistence, threshold)
            difference = qlike_rows(outcomes, persistence) - qlike_rows(outcomes, forecast)
            bootstrap, samples = _bootstrap_qlike(difference)
            target = (outcomes > ALERT_THRESHOLD).astype("int64")
            both_classes = set(target) == {0, 1}
            gates = {
                "candidate_qlike_lower": candidate_metrics["mean_qlike"]
                < persistence_metrics["mean_qlike"],
                "candidate_log_mae_lower": candidate_metrics["log_mae"]
                < persistence_metrics["log_mae"],
                "bootstrap_interval_above_zero": bootstrap["interval"][0] > 0,
                "candidate_recall_at_least_persistence": both_classes
                and candidate_metrics["alert_high_recall"]
                >= persistence_metrics["alert_high_recall"],
                "candidate_alert_cost_no_greater": both_classes
                and candidate_metrics["alert_cost"] <= persistence_metrics["alert_cost"],
            }
            metrics = {
                "protocol": PROTOCOL,
                "evaluation_rows": len(outcomes),
                "evaluation_start": features["as_of"].iloc[0].date().isoformat(),
                "evaluation_end": features["as_of"].iloc[-1].date().isoformat(),
                "alert_threshold": ALERT_THRESHOLD,
                "high_support": int((target == 1).sum()),
                "normal_support": int((target == 0).sum()),
                "candidate": candidate_metrics,
                "persistence": persistence_metrics,
                "gates": gates,
                "research_status": "passed" if all(gates.values()) else "target_not_met",
            }
            predictions = pd.DataFrame(
                {
                    "sample_id": features.index,
                    "as_of": features["as_of"].to_numpy(),
                    "actual_volatility": outcomes.to_numpy(),
                    "alert_threshold": ALERT_THRESHOLD,
                    "true_high": target.to_numpy(),
                    "base_forecast": base.to_numpy(),
                    "candidate_forecast": forecast.to_numpy(),
                    "persistence_forecast": persistence.to_numpy(),
                    "candidate_high": (forecast > ALERT_THRESHOLD).astype("int64").to_numpy(),
                    "persistence_high": (persistence > ALERT_THRESHOLD)
                    .astype("int64")
                    .to_numpy(),
                    "candidate_qlike": qlike_rows(outcomes, forecast).to_numpy(),
                    "persistence_qlike": qlike_rows(outcomes, persistence).to_numpy(),
                    "qlike_improvement": difference.to_numpy(),
                }
            )
            payloads = {
                "metrics.json": json_bytes(metrics),
                "bootstrap.json": json_bytes(bootstrap),
                "bootstrap_samples.csv": samples.to_csv(index=False).encode(),
                "evaluation_predictions.csv": predictions.to_csv(index=False).encode(),
                "report.md": _report(metrics, bootstrap),
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


def verify_ytd_evaluation(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    value = json.loads((root / "manifest.json").read_text())
    if value.get("protocol") != PROTOCOL or value.get("state") != "complete":
        raise ValueError("Unsupported or incomplete V8A evaluation")
    required = {
        "metrics.json",
        "bootstrap.json",
        "bootstrap_samples.csv",
        "evaluation_predictions.csv",
        "report.md",
    }
    hashes = value.get("files", {})
    if not required <= hashes.keys():
        raise ValueError("V8A evaluation manifest lacks required files")
    for name, digest in hashes.items():
        if Path(name).name != name or sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"V8A evaluation artifact hash mismatch: {name}")
    return cast(dict[str, Any], value)

"""Atomic run artifacts, integrity checks, environment binding and exposure ledger."""

from __future__ import annotations

import fcntl
import importlib.metadata
import io
import json
import os
import pickle
import platform
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from qr_haven.data.banknotes import (
    FEATURES,
    REFERENCE_SHA256,
    BanknoteDataset,
    atomic_write,
    json_bytes,
    sha256,
    utc_now,
)
from qr_haven.ml.classification.contracts import (
    ClassificationConfig,
    SplitManifest,
    require_sklearn,
)
from qr_haven.ml.classification.models import ClassificationModel


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    atomic_write(path, json_bytes(value), replace=replace)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_csv(path: Path, frame: pd.DataFrame, *, index: bool = False) -> None:
    atomic_write(path, frame.to_csv(index=index).encode())


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "qr-haven"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "0.1.0+source" if name == "qr-haven" else "not_installed"
    return result


def code_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[2]
    repo = package.parents[1]
    paths = sorted((package / "ml" / "classification").glob("*.py"))
    paths += [package / "data" / "banknotes.py"]
    hashes = {str(path.relative_to(package)): sha256(path.read_bytes()) for path in paths}

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    return {
        "revision": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain")),
        "source_hashes": hashes,
        "source_sha256": sha256(json_bytes(hashes)),
    }


def protocol_id(manifest: dict[str, Any]) -> str:
    return sha256(json_bytes(manifest["protocol"]))


def seal(run_dir: Path, manifest: dict[str, Any], state: str) -> None:
    manifest["artifact_sha256"] = {
        path.name: sha256(path.read_bytes())
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "manifest.json" and not path.name.startswith(".")
    }
    manifest["state"] = state
    manifest["updated_at_utc"] = utc_now()
    write_json(run_dir / "manifest.json", manifest, replace=True)


def verify_run(
    run_dir: Path, *, states: tuple[str, ...] = ("trained", "evaluated")
) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported artifact schema version")
    if manifest.get("state") not in states:
        raise ValueError(f"Invalid experiment state: {manifest.get('state')}; expected {states}")
    if protocol_id(manifest) != manifest["protocol_id"]:
        raise ValueError("Protocol hash mismatch")
    if manifest["feature_order"] != list(FEATURES):
        raise ValueError("Saved feature schema mismatch")
    required = {
        "config.resolved.yaml",
        "environment.txt",
        "data_audit.json",
        "lineage.csv",
        "split.csv",
        "split.json",
        "selection.json",
        "cv_results.csv",
        "model.pkl",
        "baseline.pkl",
        "cv_predictions.csv",
    }
    if manifest["state"] == "evaluated":
        required |= {"metrics.json", "test_predictions.csv", "report.md"}
    if not required <= set(manifest["artifact_sha256"]):
        raise ValueError("Missing required artifact checksums")
    for filename, digest in manifest["artifact_sha256"].items():
        if Path(filename).name != filename:
            raise ValueError("Invalid artifact path")
        path = run_dir / filename
        if not path.is_file() or sha256(path.read_bytes()) != digest:
            raise ValueError(f"Artifact hash mismatch: {filename}")
    if manifest["protocol"]["config"] != ClassificationConfig.from_yaml(
        run_dir / "config.resolved.yaml"
    ).model_dump(mode="json"):
        raise ValueError("Frozen configuration mismatch")
    if manifest["protocol"]["split"] != read_json(run_dir / "split.json"):
        raise ValueError("Frozen split membership mismatch")
    for name in ("model.pkl", "baseline.pkl", "selection.json"):
        if manifest["frozen_training_sha256"][name] != manifest["artifact_sha256"][name]:
            raise ValueError(f"Frozen training hash mismatch: {name}")
    return manifest


def prepare_run(
    dataset: BanknoteDataset,
    split: SplitManifest,
    config: ClassificationConfig,
    run_dir: Path,
) -> dict[str, Any]:
    # mkdir is the run-ID reservation: retraining an existing run is always an error.
    run_dir.mkdir(parents=True, exist_ok=False)
    config_dict = config.model_dump(mode="json")
    code = code_identity()
    protocol = {
        "source_sha256": split.source_sha256,
        "schema_version": 1,
        "duplicate_policy": dataset.audit["duplicate_policy"],
        "split": asdict(split),
        "config": config_dict,
        "code": code,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "protocol": protocol,
        "feature_order": list(FEATURES),
        "source": dataset.source_manifest,
        "environment": versions(),
        "code": code,
        "created_at_utc": utc_now(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "executable": sys.executable,
        },
        "canonical_protocol": (
            config.canonical_protocol()
            and dataset.source_manifest["canonical"]
            and dataset.source_manifest["raw_sha256"] == REFERENCE_SHA256
        ),
        "known_external_holdout_exposure": config.known_external_holdout_exposure,
        "exposure_notes": config.exposure_notes,
    }
    manifest["protocol_id"] = protocol_id(manifest)
    atomic_write(
        run_dir / "config.resolved.yaml", yaml.safe_dump(config_dict, sort_keys=False).encode()
    )
    snapshot = io.BytesIO()
    package = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for filename in code["source_hashes"]:
            zipped.writestr(filename, (package / filename).read_bytes())
    atomic_write(run_dir / "source_snapshot.zip", snapshot.getvalue())
    environment = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    )
    atomic_write(run_dir / "environment.txt", ("\n".join(environment) + "\n").encode())
    write_json(run_dir / "data_audit.json", dataset.audit)
    write_csv(run_dir / "lineage.csv", dataset.lineage)
    write_json(run_dir / "split.json", asdict(split))
    rows = [
        {"sample_id": key, "role": "development", "validation_fold": split.validation_folds[key]}
        for key in split.development_ids
    ]
    rows += [{"sample_id": key, "role": "test", "validation_fold": None} for key in split.test_ids]
    write_csv(run_dir / "split.csv", pd.DataFrame(rows))
    seal(run_dir, manifest, "prepared")
    return manifest


def read_split(run_dir: Path) -> SplitManifest:
    data = read_json(run_dir / "split.json")
    return SplitManifest(
        tuple(data["development_ids"]),
        tuple(data["test_ids"]),
        data["validation_folds"],
        data["source_sha256"],
        data["metadata"],
    )


def load_bundle(run_dir: Path, filename: str = "model.pkl") -> ClassificationModel:
    require_sklearn()
    manifest = verify_run(run_dir)
    if versions() != manifest["environment"]:
        raise ValueError("Serialization environment/version mismatch; restore environment.txt")
    if manifest["code"]["source_sha256"] != code_identity()["source_sha256"]:
        raise ValueError(
            "Classification source-version mismatch; restore the recorded source revision"
        )
    # Only use trusted local run directories. Hashes detect corruption, not malicious pickles.
    bundle = pickle.loads((run_dir / filename).read_bytes())
    if not isinstance(bundle, ClassificationModel) or bundle.feature_order != FEATURES:
        raise ValueError("Invalid fitted model bundle/schema")
    return bundle


def load_classifier(run_dir: Path) -> ClassificationModel:
    return load_bundle(Path(run_dir))


def record_exposure(
    run_dir: Path, manifest: dict[str, Any], split: SplitManifest
) -> list[dict[str, Any]]:
    """Caller holds the root lock; reserve exposure BEFORE reading test labels/predictions."""
    history = run_dir.parent / "evaluation_history.jsonl"
    entries = (
        [json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
    )
    test_ids = set(split.test_ids)
    prior = [
        entry
        for entry in entries
        if entry["source_sha256"] == split.source_sha256 and test_ids & set(entry["test_ids"])
    ]
    entry = {
        "run_id": run_dir.name,
        "protocol_id": manifest["protocol_id"],
        "source_sha256": split.source_sha256,
        "test_ids": sorted(test_ids),
        "holdout_key": sha256(json_bytes([split.source_sha256, sorted(test_ids)])),
        "exposed_at_utc": utc_now(),
        "event": "evaluation_started",
    }
    lines = b"".join(
        json.dumps(item, sort_keys=True).encode() + b"\n" for item in [*entries, entry]
    )
    atomic_write(history, lines, replace=True)
    return prior

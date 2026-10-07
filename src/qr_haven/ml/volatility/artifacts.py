"""Immutable V2 dataset and split artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

from qr_haven.ml.volatility.contracts import FEATURES, VolatilityDataset, WalkForwardPlan
from qr_haven.ml.volatility.splits import membership_frame, plan_dict


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_v2_artifacts(
    output_dir: Path, dataset: VolatilityDataset, plan: WalkForwardPlan
) -> dict[str, Any]:
    """Atomically write the V2 observations, intervals, membership, and manifest."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable V2 artifacts: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    observations = dataset.observations
    development = observations.loc[list(plan.final_training_ids)]
    holdout = observations.loc[list(plan.holdout_ids)]
    development_bytes = development.to_csv(index=True).encode()
    holdout_features = holdout.drop(columns="forward_vol_5").to_csv(index=True).encode()
    sealed_holdout_outcomes = holdout.loc[:, ["forward_vol_5"]].to_csv(index=True).encode()
    intervals = dataset.interval_audit.to_csv(index=True).encode()
    split = _json_bytes(plan_dict(plan))
    membership = membership_frame(plan).to_csv(index=False).encode()
    files = {
        "development_observations.csv": development_bytes,
        "holdout_features.csv": holdout_features,
        "sealed_holdout_outcomes.csv": sealed_holdout_outcomes,
        "interval_audit.csv": intervals,
        "split.json": split,
        "split_membership.csv": membership,
    }
    manifest = {
        "schema_version": 1,
        "stage": "V2 point-in-time dataset",
        "profile_id": dataset.profile.profile_id,
        "source_sha256": dataset.source_sha256,
        "observation_rows": len(dataset.observations),
        "development_rows": len(development),
        "development_folds": len(plan.folds),
        "holdout_rows": len(plan.holdout_ids),
        "feature_order": list(FEATURES),
        "profile": {
            "symbol": dataset.profile.symbol,
            "calendar": dataset.profile.calendar,
            "study_start": dataset.profile.study_start.isoformat(),
            "study_end": dataset.profile.study_end.isoformat(),
            "validation_years": list(dataset.profile.validation_years),
            "holdout_start": dataset.profile.holdout_start.isoformat(),
            "holdout_end": dataset.profile.holdout_end.isoformat(),
        },
        "holdout_outcomes_summarized": False,
        "development_access_to_holdout_outcomes": False,
        "files": {name: _sha256(payload) for name, payload in files.items()},
    }
    files["manifest.json"] = _json_bytes(manifest)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name, payload in files.items():
            with (temporary / name).open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def verify_v2_artifacts(output_dir: Path) -> dict[str, Any]:
    """Verify every artifact hash without loading holdout outcomes into model code."""
    root = Path(output_dir)
    try:
        value = json.loads((root / "manifest.json").read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"V2 manifest is unavailable or invalid: {root}") from exc
    if not isinstance(value, dict):
        raise ValueError("V2 manifest must be a JSON object")
    manifest = cast(dict[str, Any], value)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("V2 manifest contains no file hashes")
    for name, expected in files.items():
        try:
            payload = (root / name).read_bytes()
        except OSError as exc:
            raise ValueError(f"V2 artifact is unavailable: {name}") from exc
        if _sha256(payload) != expected:
            raise ValueError(f"V2 artifact hash mismatch: {name}")
    return manifest

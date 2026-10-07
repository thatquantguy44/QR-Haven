"""V3 environment, source provenance, and trusted fitted-model loading."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
import platform
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from qr_haven.ml.volatility.contracts import FEATURES

_MODEL_SOURCE_PATHS = {
    "ml/volatility/contracts.py",
    "ml/volatility/models.py",
}


def environment_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    packages = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "threadpoolctl",
        "exchange-calendars",
    )
    for name in packages:
        result[name] = importlib.metadata.version(name)
    return result


def source_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[2]
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths.append(package / "ml/classification/evaluation.py")
    hashes = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    result = subprocess.run(
        ["git", "-C", str(package.parents[1]), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    return {
        "revision": result.stdout.strip() if result.returncode == 0 else "unavailable",
        "source_hashes": hashes,
        "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }


def _candidate_id(value: Any) -> Any:
    return value.get("candidate_id") if isinstance(value, dict) else None


def _verify_model_source(recorded: Any) -> None:
    if not isinstance(recorded, dict) or not isinstance(recorded.get("source_hashes"), dict):
        raise ValueError("Model manifest has no valid source identity")
    current_hashes = source_identity()["source_hashes"]
    recorded_hashes = recorded["source_hashes"]
    if any(recorded_hashes.get(path) != current_hashes.get(path) for path in _MODEL_SOURCE_PATHS):
        raise ValueError("Volatility model source mismatch; restore the recorded model source")


def load_volatility_model(run_dir: Path) -> dict[str, Any]:
    """Verify and load a trusted local model, including its frozen threshold."""
    from qr_haven.ml.volatility.training import verify_v3_run

    root = Path(run_dir)
    manifest = verify_v3_run(root)
    if manifest["environment"] != environment_versions():
        raise ValueError("Model environment/version mismatch; restore the recorded environment")
    _verify_model_source(manifest.get("code"))
    # Pickle is for trusted local artifacts; checksums detect corruption, not malicious code.
    value = pickle.loads((root / "model.pkl").read_bytes())
    if not isinstance(value, dict) or value.get("feature_order") != FEATURES:
        raise ValueError("Invalid volatility model bundle or feature order")
    bundle = cast(dict[str, Any], value)
    if (
        bundle.get("profile_id") != manifest["profile_id"]
        or bundle.get("source_sha256") != manifest["source_sha256"]
        or bundle.get("threshold") != manifest["final_threshold"]
        or _candidate_id(bundle.get("candidate")) != manifest["selected_candidate"]
    ):
        raise ValueError("Frozen model metadata disagrees with the run manifest")
    return bundle


def predict_volatility(run_dir: Path, features: pd.DataFrame) -> pd.DataFrame:
    """Predict named feature rows using the frozen model, with no fitting or evaluation."""
    from qr_haven.ml.volatility.models import predict_estimator

    if not features.columns.is_unique or set(features.columns) != set(FEATURES):
        raise ValueError("Prediction requires exactly the nine frozen feature columns")
    ordered = features.loc[:, list(FEATURES)].astype("float64")
    if ordered.empty or not np.isfinite(ordered.to_numpy()).all():
        raise ValueError("Prediction features must be nonempty and finite")
    return predict_estimator(load_volatility_model(run_dir)["estimator"], ordered)

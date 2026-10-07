"""Canonical local artifact paths for volatility datasets, fits, and evaluations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ARTIFACT_ROOT = Path("artifacts/classification/volatility")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _safe_name(value: str, field: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{field} must be a single safe name, with no path separators")
    return value


@dataclass(frozen=True)
class VolatilityEvaluationPaths:
    """Files produced by one immutable V4 holdout evaluation."""

    output_dir: Path
    manifest: Path
    metrics: Path
    predictions: Path
    bootstrap: Path
    report: Path


def evaluation_paths(
    profile_id: str,
    model_run_id: str,
    evaluation_id: str = "holdout-v1",
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> VolatilityEvaluationPaths:
    """Return a traversal-safe V4 location separate from the immutable V3 model run."""
    profile = _safe_name(profile_id, "profile_id")
    model_run = _safe_name(model_run_id, "model_run_id")
    evaluation = _safe_name(evaluation_id, "evaluation_id")
    output = Path(artifact_root) / profile / "evaluations" / model_run / evaluation
    return VolatilityEvaluationPaths(
        output_dir=output,
        manifest=output / "manifest.json",
        metrics=output / "metrics.json",
        predictions=output / "holdout_predictions.csv",
        bootstrap=output / "bootstrap.json",
        report=output / "report.md",
    )

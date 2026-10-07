"""Canonical and traversal-safe volatility evaluation output paths."""

from pathlib import Path

import pytest

from qr_haven.ml.volatility.paths import evaluation_paths


def test_evaluation_paths_separate_v4_outputs_from_v3_run():
    paths = evaluation_paths(
        "spx-local-v1",
        "spx-vol-v2",
        artifact_root=Path("artifacts/classification/volatility"),
    )
    expected = Path(
        "artifacts/classification/volatility/spx-local-v1/"
        "evaluations/spx-vol-v2/holdout-v1"
    )
    assert paths.output_dir == expected
    assert paths.manifest == expected / "manifest.json"
    assert paths.metrics == expected / "metrics.json"
    assert paths.predictions == expected / "holdout_predictions.csv"
    assert paths.bootstrap == expected / "bootstrap.json"
    assert paths.report == expected / "report.md"


@pytest.mark.parametrize(
    ("profile", "run", "evaluation"),
    [
        ("../profile", "run", "holdout-v1"),
        ("profile", "runs/model", "holdout-v1"),
        ("profile", "run", "../holdout"),
    ],
)
def test_evaluation_paths_reject_path_traversal(profile, run, evaluation):
    with pytest.raises(ValueError, match="single safe name"):
        evaluation_paths(profile, run, evaluation)

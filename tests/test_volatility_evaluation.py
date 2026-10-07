"""Single-use V4 evaluation, bootstrap, gates, and exposure-ledger tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="Install .[research] for volatility evaluation tests")

from qr_haven.ml.volatility import (
    VolatilityProfile,
    build_volatility_dataset,
    evaluate_volatility_run,
    prepare_walk_forward_plan,
    verify_v4_evaluation,
    write_v2_artifacts,
)
from qr_haven.ml.volatility.models import VolatilityCandidate
from qr_haven.ml.volatility.training import train_volatility_development


def test_paired_block_bootstrap_is_deterministic_and_paired():
    from qr_haven.ml.volatility.evaluation import paired_block_bootstrap

    index = pd.Index([f"row-{value}" for value in range(80)])
    target = pd.Series(np.tile([0, 0, 0, 1], 20), index=index)
    selected = target.copy()
    persistence = pd.Series(np.zeros(80, dtype=int), index=index)
    first = paired_block_bootstrap(
        target, selected, persistence, block_length=8, replicates=100, max_attempts=500, seed=9
    )
    second = paired_block_bootstrap(
        target, selected, persistence, block_length=8, replicates=100, max_attempts=500, seed=9
    )
    assert first == second
    assert first["valid_replicates"] == 100
    assert first["point_difference"] == pytest.approx(0.5)
    assert first["interval"][0] > 0


@pytest.fixture
def frozen_run(tmp_path):
    profile = VolatilityProfile(
        profile_id="evaluation-fixture-v1",
        symbol="FIX",
        snapshot_dir=Path("fixture"),
        canonical_filename="fixture.csv",
        study_start=date(2008, 1, 1),
        study_end=date(2012, 12, 31),
        validation_years=(2010,),
        holdout_start=date(2011, 1, 1),
        holdout_end=date(2012, 12, 31),
    )
    sessions = xcals.get_calendar(
        "XNYS", start=profile.study_start, end=profile.study_end
    ).sessions
    rng = np.random.default_rng(923)
    scales = np.where(np.arange(len(sessions)) % 90 < 20, 0.025, 0.006)
    returns = rng.normal(0.0001, scales)
    frame = pd.DataFrame(
        {
            "timestamp": sessions.tz_localize("UTC"),
            "symbol": profile.symbol,
            "adjusted_close": 100 * np.exp(np.cumsum(returns)),
        }
    )
    dataset = build_volatility_dataset(frame, {"source_sha256": "d" * 64}, profile)
    v2_dir = tmp_path / "dataset-v1"
    write_v2_artifacts(v2_dir, dataset, prepare_walk_forward_plan(dataset))
    run_dir = tmp_path / "model-v1"
    train_volatility_development(
        v2_dir,
        run_dir,
        candidate_set=(
            VolatilityCandidate(
                "logistic_regression_00", "logistic_regression", 0, 0, {"C": 0.1}
            ),
        ),
    )
    return tmp_path, v2_dir, run_dir


def test_evaluation_opens_outcomes_after_ledger_and_is_immutable(
    frozen_run, monkeypatch
):
    root, v2_dir, run_dir = frozen_run
    output = root / "evaluations" / run_dir.name / "holdout-v1"
    history = root / "holdout_evaluation_history.jsonl"
    model_hash_before = (run_dir / "model.pkl").read_bytes()
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path.name == "sealed_holdout_outcomes.csv":
            assert history.exists(), "outcome opened before exposure was recorded"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = evaluate_volatility_run(run_dir, v2_dir, output)
    assert result.metrics["holdout_rows"] > 0
    assert result.metrics["research_status"] in {"passed", "target_not_met"}
    assert result.manifest["holdout_evaluated"] is True
    assert verify_v4_evaluation(output) == result.manifest
    assert len(pd.read_csv(output / "holdout_predictions.csv")) == result.metrics["holdout_rows"]
    report = (output / "report.md").read_text()
    assert "Every-fifth-origin results" in report
    assert "baseline_always_normal" in report
    assert "Selected-model confusion matrix" in report
    assert (run_dir / "model.pkl").read_bytes() == model_hash_before

    repeated = evaluate_volatility_run(run_dir, v2_dir, output)
    assert repeated.metrics == result.metrics
    with pytest.raises(ValueError, match="already exposed"):
        evaluate_volatility_run(
            run_dir, v2_dir, root / "evaluations" / run_dir.name / "holdout-v2",
            evaluation_id="holdout-v2",
        )

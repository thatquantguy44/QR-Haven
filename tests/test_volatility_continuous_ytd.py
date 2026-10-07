"""Frozen-source and single-exposure checks for the V8A promotion gate."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="Install .[research] for V8A evaluation")

from qr_haven.ml.volatility.challenger_data import CHALLENGER_FEATURES
from qr_haven.ml.volatility.continuous_ytd import (
    ALERT_THRESHOLD,
    _bootstrap_qlike,
    _verify_overlap,
    evaluate_ytd_candidate,
    verify_ytd_evaluation,
)


def _canonical(dates: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.arange(len(dates), dtype=float) + 100
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": values,
            "high": values + 1,
            "low": values - 1,
            "close": values + 0.5,
            "adjusted_close": values + 0.5,
            "volume": np.arange(len(dates), dtype="int64") + 1_000,
            "frequency": "daily",
        }
    )


def test_extension_overlap_must_match_frozen_canonical_exactly():
    overlap_dates = pd.bdate_range("2025-10-01", "2025-12-31", tz="UTC")
    base = _canonical(overlap_dates)
    extension = pd.concat(
        [base.copy(), _canonical(pd.bdate_range("2026-01-02", periods=5, tz="UTC"))],
        ignore_index=True,
    )
    assert _verify_overlap(base, extension) == len(base)
    damaged = extension.copy()
    damaged.loc[0, "adjusted_close"] += 0.01
    with pytest.raises(ValueError, match="overlap differs"):
        _verify_overlap(base, damaged)


def test_qlike_bootstrap_is_deterministic_and_preserves_positive_gain():
    difference = pd.Series(np.tile([0.1, 0.2, 0.3, 0.4], 50))
    first, samples = _bootstrap_qlike(difference)
    second, replay = _bootstrap_qlike(difference)
    assert first == second
    pd.testing.assert_frame_equal(samples, replay)
    assert first["point_difference"] == pytest.approx(0.25)
    assert first["interval"][0] > 0
    assert len(samples) == 2_000


def test_evaluation_records_exposure_before_outcomes_and_is_immutable(tmp_path, monkeypatch):
    import qr_haven.ml.volatility.continuous_ytd as module

    candidate = tmp_path / "candidate-v1"
    dataset = tmp_path / "dataset-v1"
    output = tmp_path / "evaluations" / "candidate-v1" / "2026-ytd-v1"
    candidate.mkdir()
    dataset.mkdir()
    (candidate / "manifest.json").write_text("{}")
    (dataset / "manifest.json").write_text("{}")
    dates = pd.bdate_range("2026-01-02", periods=100)
    index = pd.Index([f"sample-{i}" for i in range(len(dates))], name="sample_id")
    actual = pd.Series(np.where(np.arange(len(dates)) % 2, 0.3, 0.1), index=index)
    features = pd.DataFrame(index=index)
    features["as_of"] = dates
    features["trailing_vol_5"] = np.where(np.arange(len(dates)) % 2, 0.27, 0.13)
    for feature in CHALLENGER_FEATURES:
        if feature not in features:
            features[feature] = actual.to_numpy()
    candidate_manifest = {
        "selected_candidate": "hist_gradient_boosting_regression_w075",
        "files": {"model.pkl": "c" * 64},
    }
    dataset_manifest = {"extension_source_sha256": "d" * 64}

    class ExactEstimator:
        def predict(self, rows):
            return np.log(np.square(rows.iloc[:, 0].to_numpy(float)) + module.EPSILON)

    bundle = {"estimator": ExactEstimator()}
    monkeypatch.setattr(module, "verify_ytd_candidate", lambda _path: candidate_manifest)
    monkeypatch.setattr(
        module,
        "verify_ytd_dataset",
        lambda _path, include_sealed_outcomes=False: dataset_manifest,
    )
    monkeypatch.setattr(module, "_load_ytd_model", lambda _path: bundle)
    monkeypatch.setattr(module, "_load_ytd_features", lambda _path, _manifest: features)

    def guarded_outcomes(_path, _manifest):
        history = dataset.parent / "evaluation_history.jsonl"
        assert history.exists(), "V8A outcomes opened before durable exposure ledger"
        return actual

    monkeypatch.setattr(module, "_verified_ytd_outcomes", guarded_outcomes)
    result = evaluate_ytd_candidate(candidate, dataset, output)
    assert result["evaluation_outcomes_opened"] is True
    assert result["research_status"] == "passed"
    assert verify_ytd_evaluation(output) == result
    metrics = json.loads((output / "metrics.json").read_text())
    assert all(metrics["gates"].values())
    assert metrics["alert_threshold"] == ALERT_THRESHOLD
    assert evaluate_ytd_candidate(candidate, dataset, output) == result
    damaged = tmp_path / "damaged"
    shutil.copytree(output, damaged)
    (damaged / "metrics.json").write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch: metrics.json"):
        verify_ytd_evaluation(damaged)

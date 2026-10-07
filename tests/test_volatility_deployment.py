from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qr_haven.ml.volatility.challenger_data import CHALLENGER_FEATURES
from qr_haven.ml.volatility.deployment import build_power_bi_frames, score_feature_rows


class ConstantEstimator:
    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.log(np.full(len(rows), 0.2**2))


def test_score_feature_rows_applies_variance_blend_and_alert() -> None:
    rows = pd.DataFrame({name: [0.1] for name in CHALLENGER_FEATURES})
    result = score_feature_rows({"estimator": ConstantEstimator()}, rows)
    expected = np.sqrt(0.75 * 0.2**2 + 0.25 * 0.1**2)
    assert result.loc[0, "candidate_forecast"] == pytest.approx(expected)
    assert result.loc[0, "candidate_high"] == 0


def test_score_feature_rows_rejects_missing_features() -> None:
    with pytest.raises(ValueError, match="Missing required feature columns"):
        score_feature_rows({"estimator": ConstantEstimator()}, pd.DataFrame({"x": [1.0]}))


def test_power_bi_frames_have_stable_keys_and_grain() -> None:
    predictions = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "as_of": ["2026-01-02", "2026-01-05"],
            "actual_volatility": [0.1, 0.3],
            "alert_threshold": [0.2, 0.2],
            "true_high": [0, 1],
            "base_forecast": [0.11, 0.25],
            "candidate_forecast": [0.10, 0.22],
            "persistence_forecast": [0.09, 0.18],
            "candidate_high": [0, 1],
            "persistence_high": [0, 0],
            "candidate_qlike": [0.1, 0.2],
            "persistence_qlike": [0.2, 0.4],
            "qlike_improvement": [0.1, 0.2],
        }
    )
    model_metrics = {
        "mean_qlike": 0.1,
        "log_mae": 0.2,
        "volatility_mae": 0.03,
        "volatility_rmse": 0.04,
        "alert_high_recall": 0.5,
        "alert_high_precision": 0.5,
        "alert_balanced_accuracy": 0.6,
        "alert_cost": 0.3,
    }
    metrics = {
        "candidate": model_metrics,
        "persistence": {**model_metrics, "mean_qlike": 0.2},
        "gates": {"candidate_qlike_lower": True},
    }
    frames = build_power_bi_frames(
        predictions,
        metrics,
        pd.DataFrame({"replicate": [1], "qlike_improvement": [0.1]}),
    )
    assert frames["FactForecast"]["SampleId"].is_unique
    assert frames["DimDate"]["DateKey"].is_unique
    assert set(frames) == {
        "DimDate",
        "DimModel",
        "FactForecast",
        "FactModelMetric",
        "FactGate",
        "FactBootstrap",
    }

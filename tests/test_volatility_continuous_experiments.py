"""Loss, chronology, and immutability checks for V8 continuous forecasts."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="Install .[research] for V8 continuous experiments")

from qr_haven.ml.volatility.challenger_data import CHALLENGER_FEATURES, TIINGO_PROTOCOL
from qr_haven.ml.volatility.continuous_experiments import (
    BLEND_WEIGHTS,
    OUTER_YEARS,
    _rank,
    blend_variance,
    continuous_bases,
    continuous_training_ids,
    qlike_rows,
    run_continuous_experiment,
    verify_continuous_experiment,
)


def test_qlike_is_zero_for_exact_forecast_and_positive_for_errors():
    actual = pd.Series([0.1, 0.2, 0.3], index=list("abc"))
    exact = qlike_rows(actual, actual.copy())
    assert np.allclose(exact, 0, atol=1e-15)
    forecast = pd.Series([0.2, 0.1, 0.6], index=actual.index)
    assert (qlike_rows(actual, forecast) > 0).all()
    with pytest.raises(ValueError, match="positive"):
        qlike_rows(actual, forecast * 0)


def test_variance_blend_uses_variance_space_and_validates_alignment():
    base = pd.Series([0.2], index=["a"])
    persistence = pd.Series([0.1], index=["a"])
    blended = blend_variance(base, persistence, 0.5)
    assert blended.iloc[0] == pytest.approx(np.sqrt(0.5 * 0.2**2 + 0.5 * 0.1**2))
    with pytest.raises(ValueError, match="invalid"):
        blend_variance(base, persistence.rename(index={"a": "b"}), 0.5)


def test_twelve_year_training_membership_is_purged():
    dates = pd.bdate_range("2000-01-03", "2014-12-31")
    observations = pd.DataFrame(
        {"as_of": dates, "label_end": dates + pd.offsets.BDay(5)},
        index=[f"id-{i}" for i in range(len(dates))],
    )
    ids = continuous_training_ids(observations, 2013)
    boundary = observations.loc[observations["as_of"].dt.year.eq(2013), "as_of"].iloc[0]
    training = observations.loc[list(ids)]
    assert (training["as_of"] >= boundary - pd.DateOffset(years=12)).all()
    assert (training["label_end"] < boundary).all()


def test_ranking_prefers_lower_yearly_qlike_then_log_mae():
    records = []
    candidates = [("better", "har_rv", 1.0, 0.2), ("persistence", "persistence", 0.0, 0.3)]
    for candidate_id, base_id, weight, loss in candidates:
        for frequency in ("daily", "every_fifth"):
            for year in OUTER_YEARS:
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "base_id": base_id,
                        "model_weight": weight,
                        "frequency": frequency,
                        "validation_year": year,
                        "mean_qlike": loss,
                        "log_mae": loss,
                        "volatility_mae": loss,
                        "volatility_rmse": loss,
                        "alert_balanced_accuracy": 0.5,
                        "alert_high_recall": 0.5,
                        "alert_cost": 1.0,
                        "false_positive": 1,
                        "false_negative": 1,
                    }
                )
    ranking = _rank(pd.DataFrame(records))
    winner = ranking.loc[(ranking["frequency"] == "daily") & (ranking["rank"] == 1)].iloc[0]
    assert winner["candidate_id"] == "better"
    assert winner["qlike_improvement_vs_persistence"] == pytest.approx(0.1)


def test_immutable_run_saves_config_before_forecasting_and_excludes_evaluation(
    tmp_path, monkeypatch
):
    import qr_haven.ml.volatility.continuous_experiments as module

    dataset = tmp_path / "dataset-v1"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    output = tmp_path / "continuous" / "v8-continuous-v1"
    frames = []
    for year in OUTER_YEARS:
        dates = pd.bdate_range(f"{year}-01-02", periods=10)
        actual = np.where(np.arange(10) % 2, 0.3, 0.1)
        frame = pd.DataFrame(index=[f"id-{year}-{i}" for i in range(10)])
        frame.index.name = "sample_id"
        frame["as_of"] = dates
        frame["label_end"] = dates + pd.offsets.BDay(5)
        frame["forward_vol_5"] = actual
        frame["trailing_vol_5"] = np.where(np.arange(10) % 2, 0.25, 0.15)
        for feature in CHALLENGER_FEATURES:
            if feature not in frame:
                frame[feature] = 0.1
        frames.append(frame)
    observations = pd.concat(frames)
    threshold_audit = pd.DataFrame(
        {"adaptive_threshold": 0.2, "as_of": observations["as_of"]},
        index=observations.index,
    )
    manifest = {
        "protocol": TIINGO_PROTOCOL,
        "profile_id": "tiingo-spy-v1",
        "source_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        module, "load_challenger_development", lambda _path: (observations, manifest)
    )
    monkeypatch.setattr(module, "adaptive_thresholds", lambda _rows: threshold_audit)
    monkeypatch.setattr(
        module,
        "ewma_scores",
        lambda rows, _decay: pd.Series(0.2, index=rows.index, name="raw_score"),
    )

    def fake_forecast(rows, base, year, _ewma):
        assert (output / "config.json").is_file(), "V8 config must exist before forecasting"
        validation = rows.loc[rows["as_of"].dt.year.eq(year)]
        forecast = validation["forward_vol_5"].copy().rename("base_forecast")
        audit = {"base_id": base.base_id, "validation_year": year, "training_rows": 2}
        membership = pd.DataFrame(
            {"base_id": [base.base_id], "validation_year": [year], "sample_id": ["train"]}
        )
        return forecast, audit, membership

    monkeypatch.setattr(module, "_fit_forecast", fake_forecast)
    result = run_continuous_experiment(dataset, output)
    assert result["candidate_advances"] is True
    assert result["evaluation_outcomes_opened"] is False
    assert result["evaluation_artifacts_loaded"] is False
    assert result["final_model_fitted"] is False
    assert verify_continuous_experiment(output) == result
    assert json.loads((output / "config.json").read_text())["evaluation_years_loaded"] == []
    with pytest.raises(FileExistsError, match="immutable"):
        run_continuous_experiment(dataset, output)
    damaged = tmp_path / "damaged"
    shutil.copytree(output, damaged)
    (damaged / "ranking.csv").write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch: ranking.csv"):
        verify_continuous_experiment(damaged)


def test_candidate_grid_has_twenty_blends_plus_persistence():
    assert len(continuous_bases()) * len(BLEND_WEIGHTS) + 1 == 21

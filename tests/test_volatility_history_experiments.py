"""Causality and weighting checks for the V7 training-history experiment."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="Install .[research] for V7 history experiments")

from qr_haven.ml.volatility.challenger_data import CHALLENGER_FEATURES, TIINGO_PROTOCOL
from qr_haven.ml.volatility.challenger_models import ENSEMBLE_WEIGHTS
from qr_haven.ml.volatility.history_experiments import (
    MIN_THRESHOLD_HISTORY,
    OUTER_YEARS,
    _rank,
    adaptive_thresholds,
    history_designs,
    history_training_ids,
    history_weights,
    run_history_experiment,
    verify_history_experiment,
)


def _observations(start: str, periods: int) -> pd.DataFrame:
    origins = pd.bdate_range(start, periods=periods)
    phase = np.arange(periods)
    frame = pd.DataFrame(
        {
            "as_of": origins,
            "label_end": origins + pd.offsets.BDay(5),
            "forward_vol_5": 0.1 + 0.03 * np.sin(phase / 17) + (phase % 13) / 1000,
        },
        index=[f"sample-{value.date()}" for value in origins],
    )
    frame.index.name = "sample_id"
    return frame


def test_adaptive_thresholds_use_only_observable_trailing_history():
    observations = _observations("2007-01-02", 1_300)
    thresholds = adaptive_thresholds(observations)
    assert thresholds["threshold_history_rows"].min() >= MIN_THRESHOLD_HISTORY
    assert (thresholds["max_history_label_end"] < thresholds["as_of"]).all()
    assert (
        thresholds["threshold_history_start"]
        >= thresholds["as_of"] - pd.DateOffset(years=3)
    ).all()

    cutoff = thresholds.index[50]
    shocked = observations.copy()
    shocked.loc[shocked["as_of"] >= thresholds.loc[cutoff, "as_of"], "forward_vol_5"] = 100
    replay = adaptive_thresholds(shocked)
    assert replay.loc[cutoff, "adaptive_threshold"] == pytest.approx(
        thresholds.loc[cutoff, "adaptive_threshold"]
    )


def test_history_membership_and_decay_weights_are_purged_and_normalized():
    observations = _observations("2005-01-03", 2_600)
    observations["adaptive_target"] = (np.arange(len(observations)) % 4 == 0).astype("int64")
    five, eight, twelve, expanding, decayed = history_designs()
    five_ids = history_training_ids(observations, 2013, five)
    eight_ids = history_training_ids(observations, 2013, eight)
    twelve_ids = history_training_ids(observations, 2013, twelve)
    expanding_ids = history_training_ids(observations, 2013, expanding)
    assert set(five_ids) < set(eight_ids) <= set(twelve_ids) == set(expanding_ids)
    boundary = observations.loc[observations["as_of"].dt.year.eq(2013), "as_of"].iloc[0]
    assert (observations.loc[list(five_ids), "label_end"] < boundary).all()
    assert (
        observations.loc[list(five_ids), "as_of"] >= boundary - pd.DateOffset(years=5)
    ).all()

    weights = history_weights(observations, expanding_ids, boundary, decayed)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert weights.mean() == pytest.approx(1)
    normal = observations.loc[list(expanding_ids), "adaptive_target"].eq(0)
    assert weights.loc[normal].iloc[-1] > weights.loc[normal].iloc[0]


def test_ranking_uses_yearly_means_and_frozen_tie_order():
    records = []
    for design in history_designs():
        for weight in ENSEMBLE_WEIGHTS:
            for frequency in ("daily", "every_fifth"):
                for year in OUTER_YEARS:
                    score = 0.6 + (0.02 if design.design_id == "rolling_8y" else 0)
                    records.append(
                        {
                            "design_id": design.design_id,
                            "model_weight": weight,
                            "frequency": frequency,
                            "validation_year": year,
                            "balanced_accuracy": score,
                            "macro_f1": score,
                            "accuracy": score,
                            "high_precision": score,
                            "high_recall": score,
                            "false_positive": 1,
                            "false_negative": 1,
                        }
                    )
    ranking = _rank(pd.DataFrame(records))
    winner = ranking.loc[(ranking["frequency"] == "daily") & (ranking["rank"] == 1)].iloc[0]
    assert winner["design_id"] == "rolling_8y"
    assert winner["model_weight"] == 0


def test_immutable_run_saves_protocol_before_fits_and_never_requests_evaluation(
    tmp_path, monkeypatch
):
    import qr_haven.ml.volatility.history_experiments as module

    dataset = tmp_path / "dataset-v1"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    output = tmp_path / "history" / "v7-history-v1"
    frames = []
    for year in range(2010, 2024):
        dates = pd.bdate_range(f"{year}-01-02", periods=10)
        frame = pd.DataFrame(index=[f"id-{year}-{i}" for i in range(10)])
        frame.index.name = "sample_id"
        frame["as_of"] = dates
        frame["label_end"] = dates + pd.offsets.BDay(5)
        frame["forward_vol_5"] = np.where(np.arange(10) % 2, 0.3, 0.1)
        frame["adaptive_threshold"] = 0.2
        frame["adaptive_target"] = np.arange(10) % 2
        frame["trailing_vol_5"] = np.where(np.arange(10) % 2, 0.25, 0.15)
        for feature in CHALLENGER_FEATURES:
            if feature not in frame:
                frame[feature] = 0.1
        frames.append(frame)
    eligible = pd.concat(frames)
    thresholds = pd.DataFrame(
        {
            "as_of": eligible["as_of"],
            "adaptive_threshold": 0.2,
            "threshold_history_rows": 500,
            "threshold_history_start": eligible["as_of"] - pd.DateOffset(years=3),
            "threshold_history_end": eligible["as_of"] - pd.offsets.BDay(6),
            "max_history_label_end": eligible["as_of"] - pd.offsets.BDay(1),
        },
        index=eligible.index,
    )
    manifest = {
        "protocol": TIINGO_PROTOCOL,
        "profile_id": "tiingo-spy-v1",
        "source_sha256": "a" * 64,
    }
    monkeypatch.setattr(module, "load_challenger_development", lambda _path: (eligible, manifest))
    monkeypatch.setattr(module, "_eligible_observations", lambda _rows: (eligible, thresholds))

    def fake_raw(rows, design, year):
        assert (output / "config.json").is_file(), "V7 config must exist before the first fit"
        validation = rows.loc[rows["as_of"].dt.year.eq(year)]
        raw = pd.DataFrame(
            {
                "raw_score": np.where(validation["adaptive_target"], 0.8, 0.2),
                "true_class": validation["adaptive_target"],
                "trailing_vol_5": validation["trailing_vol_5"],
                "adaptive_threshold": validation["adaptive_threshold"],
                "as_of": validation["as_of"],
                "validation_year": year,
            },
            index=validation.index,
        )
        audit = {
            "design_id": design.design_id,
            "validation_year": year,
            "training_rows": 2,
        }
        membership = pd.DataFrame(
            {
                "design_id": [design.design_id],
                "validation_year": [year],
                "sample_id": ["training-row"],
            }
        )
        return raw, audit, membership

    monkeypatch.setattr(module, "_raw_fold", fake_raw)
    result = run_history_experiment(dataset, output)
    assert result["evaluation_outcomes_opened"] is False
    assert result["evaluation_artifacts_loaded"] is False
    assert result["final_model_fitted"] is False
    assert verify_history_experiment(output) == result
    config = json.loads((output / "config.json").read_text())
    assert config["evaluation_years_loaded"] == []
    with pytest.raises(FileExistsError, match="immutable"):
        run_history_experiment(dataset, output)
    damaged = tmp_path / "damaged"
    shutil.copytree(output, damaged)
    (damaged / "ranking.csv").write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch: ranking.csv"):
        verify_history_experiment(damaged)

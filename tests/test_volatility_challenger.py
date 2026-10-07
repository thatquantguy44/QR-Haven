"""Point-in-time, nested calibration, ensemble, and single-use challenger tests."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="Install .[research] for challenger tests")

from qr_haven.ml.volatility.challenger_data import (
    CHALLENGER_FEATURES,
    TIINGO_PROTOCOL,
    extend_challenger_features,
    get_challenger_design,
    prepare_challenger_dataset,
    verify_challenger_dataset,
)
from qr_haven.ml.volatility.challenger_evaluation import (
    evaluate_challenger,
    verify_challenger_evaluation,
)
from qr_haven.ml.volatility.challenger_models import (
    ChallengerCandidate,
    calibrated_scores,
    ensemble_predictions,
    ewma_scores,
    fit_base_model,
    fit_platt,
    five_year_training_ids,
    raw_scores,
    threshold_for,
)
from qr_haven.ml.volatility.challenger_training import (
    train_challenger,
    verify_challenger_run,
)
from qr_haven.ml.volatility.contracts import VolatilityProfile
from qr_haven.ml.volatility.dataset import build_volatility_dataset


@pytest.fixture(scope="module")
def challenger_observations():
    profile = VolatilityProfile(
        profile_id="challenger-fixture-v1",
        symbol="FIX",
        snapshot_dir=Path("fixture"),
        canonical_filename="fixture.csv",
        study_start=date(2005, 1, 1),
        study_end=date(2020, 11, 4),
        validation_years=tuple(range(2010, 2018)),
        holdout_start=date(2018, 1, 1),
        holdout_end=date(2019, 12, 31),
    )
    sessions = xcals.get_calendar("XNYS", start=profile.study_start, end=profile.study_end).sessions
    rng = np.random.default_rng(1907)
    positions = np.arange(len(sessions))
    scale = np.where(positions % 100 < 25, 0.025, 0.006)
    returns = rng.normal(0.0001, scale)
    close = 100 * np.exp(np.cumsum(returns))
    canonical = pd.DataFrame(
        {
            "timestamp": sessions.tz_localize("UTC"),
            "symbol": profile.symbol,
            "open": close,
            "high": close * (1.003 + np.abs(returns)),
            "low": close * (0.997 - np.minimum(np.abs(returns), 0.002)),
            "close": close,
            "adjusted_close": close,
            "volume": 1_000_000,
            "frequency": "daily",
        }
    )
    source = {
        "source_sha256": "f" * 64,
        "dataset_id": "challenger-fixture",
    }
    base = build_volatility_dataset(canonical, source, profile)
    observations, audit = extend_challenger_features(base, canonical)
    return observations, audit, source


def test_extended_features_are_point_in_time_and_audited(challenger_observations):
    observations, audit, _ = challenger_observations
    assert set(CHALLENGER_FEATURES) <= set(observations)
    assert np.isfinite(observations.loc[:, list(CHALLENGER_FEATURES)].to_numpy()).all()
    sample = observations.index[100]
    assert observations.loc[sample, "downside_vol_20"] >= 0
    assert 0 <= observations.loc[sample, "negative_return_share_20"] <= 1
    windows = json.loads(audit.loc[sample, "downside_vol_60_price_sessions"])
    assert len(windows) == 61
    assert windows[-1] == observations.loc[sample, "as_of"].date().isoformat()
    assert json.loads(audit.loc[sample, "parkinson_vol_20_price_sessions"])[-1] == windows[-1]


def test_tiingo_design_reserves_two_untouched_years():
    design = get_challenger_design("tiingo-spy-v1")
    assert design.protocol == TIINGO_PROTOCOL
    assert design.outer_years == tuple(range(2013, 2024))
    assert design.final_calibration_years == (2021, 2022, 2023)
    assert design.evaluation_years == (2024, 2025)
    assert design.evaluation_id == "2024-2025-v1"
    assert design.bootstrap_seed == 5411


def test_tiingo_dataset_keeps_2024_2025_outcomes_sealed(
    tmp_path, challenger_observations, monkeypatch
):
    import qr_haven.ml.volatility.challenger_data as data_module

    observations, audit, source = challenger_observations
    source_rows = observations.loc[observations["as_of"].dt.year.eq(2019)]
    source_audit = audit.loc[source_rows.index]
    observation_frames = [observations]
    audit_frames = [audit]
    for year in range(2021, 2026):
        offset = pd.DateOffset(years=year - 2019)
        copied = source_rows.copy()
        for column in ("as_of", "available_at", "label_start", "label_end"):
            copied[column] = pd.to_datetime(copied[column]) + offset
        copied.index = [f"tiingo-spy-v1/SPY/{value.date()}" for value in copied["as_of"]]
        copied.index.name = "sample_id"
        copied_audit = source_audit.copy()
        copied_audit.index = copied.index
        observation_frames.append(copied)
        audit_frames.append(copied_audit)
    expanded = pd.concat(observation_frames).sort_values("as_of")
    expanded_audit = pd.concat(audit_frames).loc[expanded.index]
    monkeypatch.setattr(
        data_module,
        "build_challenger_observations",
        lambda _profile_id: (expanded.copy(), expanded_audit.copy(), source.copy()),
    )
    snapshot_manifest = (
        get_challenger_design("tiingo-spy-v1").profile.snapshot_dir / "manifest.json"
    )
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: True if path == snapshot_manifest else original_is_file(path),
    )
    output = tmp_path / "tiingo" / "dataset-v1"
    manifest = prepare_challenger_dataset(output, profile_id="tiingo-spy-v1")
    assert manifest["protocol"] == TIINGO_PROTOCOL
    assert manifest["purged_at_evaluation_rows"] == 5
    features = pd.read_csv(output / "evaluation_features.csv", parse_dates=["as_of"])
    assert set(features["as_of"].dt.year) == {2024, 2025}
    development = pd.read_csv(output / "development_observations.csv", parse_dates=["as_of"])
    assert development["as_of"].dt.year.max() == 2023
    assert "forward_vol_5" not in features
    assert (output / "sealed_evaluation_outcomes.csv").exists()


def test_five_year_membership_threshold_and_har_do_not_use_validation_outcomes(
    challenger_observations,
):
    observations, _, _ = challenger_observations
    ids = five_year_training_ids(observations, 2017)
    validation = observations.loc[observations["as_of"].dt.year.eq(2017)]
    boundary = validation["as_of"].iloc[0]
    assert observations.loc[list(ids), "as_of"].min() >= boundary - pd.DateOffset(years=5)
    assert observations.loc[list(ids), "label_end"].max() < boundary
    threshold = threshold_for(observations, ids)
    candidate = ChallengerCandidate("har_rv", "har", 1)
    model = fit_base_model(candidate, observations, ids, threshold)
    valid_ids = tuple(validation.index.astype(str))
    expected = raw_scores(candidate, model, observations, valid_ids)
    changed = observations.copy()
    changed.loc[list(valid_ids), "forward_vol_5"] = 10_000
    replay = raw_scores(candidate, model, changed, valid_ids)
    pd.testing.assert_series_equal(expected, replay)


def test_ewma_calibration_and_ensemble_are_causal_and_exact(challenger_observations):
    observations, _, _ = challenger_observations
    subset = observations.iloc[:100].copy()
    expected = ewma_scores(subset, 0.94)
    changed = subset.copy()
    changed.iloc[-1, changed.columns.get_loc("log_return_1")] = 50
    pd.testing.assert_series_equal(expected.iloc[:-1], ewma_scores(changed, 0.94).iloc[:-1])
    target = pd.Series([0, 0, 1, 1], index=list("abcd"))
    raw = pd.Series([-2.0, -1.0, 1.0, 2.0], index=target.index)
    probability = calibrated_scores(fit_platt(raw, target), raw)
    trailing = pd.Series([0.1, 0.3, 0.1, 0.3], index=target.index)
    model_only = ensemble_predictions(probability, trailing, 0.2, 1.0)
    persistence = ensemble_predictions(probability, trailing, 0.2, 0.0)
    assert np.allclose(model_only["score_class_1"], probability)
    assert persistence["predicted_class"].tolist() == [0, 1, 0, 1]
    halfway = ensemble_predictions(probability, trailing, 0.2, 0.5)
    assert np.allclose(halfway["score_class_1"], 0.5 * probability + 0.5 * (trailing > 0.2))


@pytest.fixture
def challenger_dataset(tmp_path, challenger_observations, monkeypatch):
    import qr_haven.ml.volatility.challenger_data as data_module

    observations, audit, source = challenger_observations
    monkeypatch.setattr(
        data_module,
        "build_challenger_observations",
        lambda _profile_id: (observations.copy(), audit.copy(), source.copy()),
    )
    output = tmp_path / "challengers" / "dataset-v1"
    manifest = prepare_challenger_dataset(output)
    assert manifest["evaluation_rows"] > 0
    assert verify_challenger_dataset(output) == manifest
    return output


def test_nested_training_and_single_use_evaluation_are_immutable(challenger_dataset, monkeypatch):
    import qr_haven.ml.volatility.challenger_training as training_module

    har = ChallengerCandidate("har_rv", "har", 1)
    monkeypatch.setattr(training_module, "challenger_candidates", lambda: (har,))
    run = challenger_dataset.parent / "challenger-v1"
    trained = train_challenger(challenger_dataset, run)
    assert trained["selected_candidate"] == "har_rv"
    assert trained["selection_uses_2018_2019"] is False
    assert trained["final_training_uses_2018_2019"] is True
    assert trained["evaluation_outcomes_opened"] is False
    assert verify_challenger_run(run) == trained
    results = pd.read_csv(run / "cv_results.csv")
    assert set(results.validation_year) == set(range(2013, 2018))
    assert results.model_weight.nunique() == 5
    predictions = pd.read_csv(run / "cv_predictions.csv")
    assert (
        predictions.groupby(["validation_year", "sample_id"])["true_class"].nunique() == 1
    ).all()
    calibration = pd.read_csv(run / "calibration_predictions.csv")
    assert (calibration.validation_year < calibration.outer_year).all()

    output = challenger_dataset.parent / "evaluations" / run.name / "2020-v1"
    history = challenger_dataset.parent / "challenger_evaluation_history.jsonl"
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path.name == "sealed_evaluation_outcomes.csv":
            assert history.exists(), "sealed outcomes opened before durable exposure ledger"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    evaluated = evaluate_challenger(run, challenger_dataset, output)
    assert evaluated["evaluation_outcomes_opened"] is True
    assert evaluated["research_status"] in {"passed", "target_not_met"}
    assert verify_challenger_evaluation(output) == evaluated
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["evaluation_rows"] > 0
    assert len(pd.read_csv(output / "bootstrap_samples.csv")) == 2_000
    repeated = evaluate_challenger(run, challenger_dataset, output)
    assert repeated == evaluated
    damaged = output.parent / "damaged"
    shutil.copytree(output, damaged)
    (damaged / "metrics.json").write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch: metrics.json"):
        verify_challenger_evaluation(damaged)
    with pytest.raises(ValueError, match="already exposed"):
        evaluate_challenger(
            run,
            challenger_dataset,
            challenger_dataset.parent / "evaluations" / run.name / "2020-v2",
            evaluation_id="2020-v2",
        )
    with pytest.raises(FileExistsError, match="immutable"):
        train_challenger(challenger_dataset, run)

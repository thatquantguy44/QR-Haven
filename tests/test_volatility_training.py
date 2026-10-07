"""Frozen V3 grid, baselines, ranking, holdout isolation, and artifact tests."""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="Install .[research] for volatility model tests")

from qr_haven.ml.volatility import (
    FEATURES,
    VolatilityProfile,
    build_volatility_dataset,
    prepare_walk_forward_plan,
    write_v2_artifacts,
)
from qr_haven.ml.volatility.artifacts import load_v2_development
from qr_haven.ml.volatility.models import (
    VolatilityCandidate,
    baseline_predictions,
    candidates,
    inverse_frequency_weights,
    make_estimator,
)
from qr_haven.ml.volatility.persistence import load_volatility_model, predict_volatility
from qr_haven.ml.volatility.training import (
    rank_candidates,
    train_volatility_development,
    verify_v3_run,
)


@pytest.fixture(scope="module")
def development_dataset():
    profile = VolatilityProfile(
        profile_id="training-fixture-v1",
        symbol="FIX",
        snapshot_dir=Path("fixture"),
        canonical_filename="fixture.csv",
        study_start=date(2005, 1, 1),
        study_end=date(2014, 12, 31),
        validation_years=(2010, 2011),
        holdout_start=date(2012, 1, 1),
        holdout_end=date(2013, 12, 31),
    )
    sessions = xcals.get_calendar(
        "XNYS", start=profile.study_start, end=profile.study_end
    ).sessions
    rng = np.random.default_rng(91)
    returns = rng.normal(0.0001, 0.006 + 0.006 * (sessions.year % 2))
    canonical = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(sessions).tz_localize("UTC"),
            "symbol": profile.symbol,
            "adjusted_close": 100 * np.exp(np.cumsum(returns)),
        }
    )
    return build_volatility_dataset(canonical, {"source_sha256": "b" * 64}, profile)


@pytest.fixture
def v2_dir(tmp_path, development_dataset):
    output = tmp_path / "v2"
    write_v2_artifacts(
        output, development_dataset, prepare_walk_forward_plan(development_dataset)
    )
    return output


def test_frozen_candidate_grid_has_exact_family_counts_and_settings():
    grid = candidates()
    assert len(grid) == 20
    assert [sum(item.family == family for item in grid) for family in (
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    )] == [3, 9, 8]
    assert [item.family_order for item in grid[:3]] == [0, 0, 0]
    forest = make_estimator(next(item for item in grid if item.family == "random_forest"))
    assert forest.n_estimators == 400
    assert forest.random_state == 5401
    histogram = make_estimator(
        next(item for item in grid if item.family == "hist_gradient_boosting")
    )
    assert histogram.max_iter == 300
    assert histogram.early_stopping is False
    assert histogram.random_state == 5402


def test_inverse_frequency_weights_balance_training_classes():
    target = pd.Series([0, 0, 0, 1], index=list("abcd"))
    weights = inverse_frequency_weights(target)
    assert weights[target == 0].sum() == pytest.approx(2.0)
    assert weights[target == 1].sum() == pytest.approx(2.0)
    assert weights.index.equals(target.index)
    with pytest.raises(ValueError, match="both training classes"):
        inverse_frequency_weights(pd.Series([0, 0]))


def test_baselines_use_only_training_target_features_and_threshold():
    target = pd.Series([0, 0, 1])
    features = pd.DataFrame({"trailing_vol_5": [0.1, 0.3]}, index=["x", "y"])
    majority = baseline_predictions("majority", target, features, 0.2)
    normal = baseline_predictions("always_normal", target, features, 0.2)
    persistence = baseline_predictions("persistence", target, features, 0.2)
    assert majority["predicted_class"].tolist() == [0, 0]
    assert majority["score_class_1"].tolist() == pytest.approx([1 / 3, 1 / 3])
    assert normal["predicted_class"].tolist() == [0, 0]
    assert persistence["predicted_class"].tolist() == [0, 1]
    assert persistence["score_class_1"].tolist() == [0.1, 0.3]


def test_ranking_uses_unrounded_metrics_then_fixed_order():
    first = VolatilityCandidate("first", "logistic_regression", 0, 0, {"C": 1.0})
    second = VolatilityCandidate("second", "random_forest", 1, 0, {})
    rows = []
    for fold in ("year-1", "year-2"):
        rows.extend(
            [
                {
                    "model_id": "first",
                    "model_kind": "learned",
                    "evaluation_frequency": "daily",
                    "fold_id": fold,
                    "balanced_accuracy": 0.8,
                    "macro_f1": 0.7,
                    "accuracy": 0.75,
                },
                {
                    "model_id": "second",
                    "model_kind": "learned",
                    "evaluation_frequency": "daily",
                    "fold_id": fold,
                    "balanced_accuracy": 0.8,
                    "macro_f1": 0.7,
                    "accuracy": 0.9,
                },
            ]
        )
    ranked = rank_candidates(pd.DataFrame(rows), (first, second))
    assert [row["candidate_id"] for row in ranked] == ["first", "second"]
    rows[-1]["balanced_accuracy"] = 0.8000000000001
    assert rank_candidates(pd.DataFrame(rows), (first, second))[0]["candidate_id"] == "second"


def test_development_loader_never_opens_sealed_holdout(v2_dir, monkeypatch):
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path.name == "sealed_holdout_outcomes.csv":
            pytest.fail("Development loader opened sealed holdout outcomes")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    observations, plan, manifest = load_v2_development(v2_dir)
    assert tuple(observations.index) == plan.final_training_ids
    assert set(observations.index).isdisjoint(plan.holdout_ids)
    assert manifest["development_access_to_holdout_outcomes"] is False


def test_development_training_freezes_selection_without_holdout_access(
    v2_dir, tmp_path, monkeypatch
):
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path.name == "sealed_holdout_outcomes.csv":
            pytest.fail("V3 training opened sealed holdout outcomes")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    subset = (
        VolatilityCandidate("logistic_00", "logistic_regression", 0, 0, {"C": 0.1}),
        VolatilityCandidate("logistic_01", "logistic_regression", 0, 1, {"C": 1.0}),
    )
    output = tmp_path / "run"
    trained = train_volatility_development(v2_dir, output, candidate_set=subset)
    assert trained.selected in subset
    assert trained.manifest["holdout_evaluated"] is False
    assert trained.manifest["sealed_holdout_outcomes_opened"] is False
    assert verify_v3_run(output) == trained.manifest
    results = pd.read_csv(output / "cv_results.csv")
    assert len(results) == (len(subset) + 3) * 2 * 2
    assert set(results["evaluation_frequency"]) == {"daily", "every_fifth"}
    predictions = pd.read_csv(output / "cv_predictions.csv")
    _, plan, _ = load_v2_development(v2_dir)
    assert set(predictions["sample_id"]).isdisjoint(plan.holdout_ids)
    assert set(predictions["model_id"]) == {
        "baseline_majority",
        "baseline_always_normal",
        "baseline_persistence",
        "logistic_00",
        "logistic_01",
    }
    with (output / "model.pkl").open("rb") as stream:
        bundle = pickle.load(stream)  # noqa: S301 - trusted test artifact created above
    assert bundle["candidate"]["candidate_id"] == trained.selected.candidate_id
    assert bundle["threshold"] == plan.final_threshold
    loaded = load_volatility_model(output)
    assert loaded["threshold"] == plan.final_threshold
    development, _, _ = load_v2_development(v2_dir)
    predicted = predict_volatility(
        output, development.loc[development.index[:3], list(FEATURES)]
    )
    assert len(predicted) == 3
    assert set(predicted.columns) == {"predicted_class", "score_class_1", "score_kind"}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_volatility_development(v2_dir, output, candidate_set=subset)


def test_training_rejects_profile_mismatch(v2_dir, tmp_path):
    with pytest.raises(ValueError, match="does not match requested profile"):
        train_volatility_development(
            v2_dir,
            tmp_path / "run",
            candidate_set=(
                VolatilityCandidate(
                    "logistic_00", "logistic_regression", 0, 0, {"C": 0.1}
                ),
            ),
            expected_profile="another-profile",
        )

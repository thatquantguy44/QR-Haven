"""Causality, comparable truth, control replay, and immutable experiment evidence."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="Install .[research] for volatility experiments")

from qr_haven.ml.volatility import (
    VolatilityProfile,
    build_volatility_dataset,
    evaluate_volatility_run,
    prepare_walk_forward_plan,
    write_v2_artifacts,
)
from qr_haven.ml.volatility.__main__ import main
from qr_haven.ml.volatility.artifacts import load_v2_development
from qr_haven.ml.volatility.experiment_diagnostics import error_tables
from qr_haven.ml.volatility.experiment_models import (
    CONTROL_ID,
    class_weights,
    ewma_forecast,
    experiment_variants,
    linear_forecast,
    scored_predictions,
    training_ids,
)
from qr_haven.ml.volatility.experiments import (
    _rank_and_pool,
    run_volatility_experiments,
    verify_volatility_experiments,
)
from qr_haven.ml.volatility.models import candidates, inverse_frequency_weights
from qr_haven.ml.volatility.training import train_volatility_development


@pytest.fixture(scope="module")
def frozen_experiment_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments") / "experiment-fixture-v1"
    profile = VolatilityProfile(
        profile_id=root.name,
        symbol="FIX",
        snapshot_dir=Path("fixture"),
        canonical_filename="fixture.csv",
        study_start=date(2005, 1, 1),
        study_end=date(2014, 12, 31),
        validation_years=(2010, 2011),
        holdout_start=date(2012, 1, 1),
        holdout_end=date(2013, 12, 31),
    )
    sessions = xcals.get_calendar("XNYS", start=profile.study_start, end=profile.study_end).sessions
    rng = np.random.default_rng(918)
    scales = np.where(np.arange(len(sessions)) % 90 < 20, 0.025, 0.006)
    returns = rng.normal(0.0001, scales)
    frame = pd.DataFrame(
        {
            "timestamp": sessions.tz_localize("UTC"),
            "symbol": profile.symbol,
            "adjusted_close": 100 * np.exp(np.cumsum(returns)),
        }
    )
    dataset = build_volatility_dataset(frame, {"source_sha256": "e" * 64}, profile)
    v2 = root / "dataset-v1"
    write_v2_artifacts(v2, dataset, prepare_walk_forward_plan(dataset))
    run = root / "model-v1"
    candidate = next(c for c in candidates() if c.candidate_id == "hist_gradient_boosting_01")
    train_volatility_development(v2, run, candidate_set=(candidate,))
    evaluation = root / "evaluations" / run.name / "holdout-v1"
    evaluate_volatility_run(run, v2, evaluation)
    return root, v2, run, evaluation


def test_weights_preserve_control_and_cutoff_ties_are_normal():
    target = pd.Series([0, 0, 0, 1])
    pd.testing.assert_series_equal(class_weights(target, 1), inverse_frequency_weights(target))
    assert class_weights(target, 0).tolist() == [1, 1, 1, 1]
    halfway = class_weights(target, 0.5)
    assert halfway.mean() == pytest.approx(1)
    assert halfway.iloc[-1] / halfway.iloc[0] == pytest.approx(np.sqrt(3))
    for power in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="power"):
            class_weights(target, power)
    scores = pd.Series([0.5, 0.6, 0.60001, 0.9])
    assert scored_predictions(scores, 0.6, "test")["predicted_class"].tolist() == [0, 0, 1, 1]


def test_ewma_recursion_and_future_independence():
    observations = pd.DataFrame(
        {
            "as_of": pd.date_range("2010-01-01", periods=4),
            "log_return_1": [0.5, 0.01, -0.02, 0.03],
            "trailing_vol_60": [0.2] * 4,
        }
    )
    forecast = ewma_forecast(observations, 0.9)
    assert forecast.iloc[0] == pytest.approx(0.2)
    assert forecast.iloc[1] ** 2 == pytest.approx(0.9 * 0.2**2 + 0.1 * 252 * 0.01**2)
    shocked = observations.copy()
    shocked.loc[3, "log_return_1"] = 100
    pd.testing.assert_series_equal(forecast.iloc[:3], ewma_forecast(shocked, 0.9).iloc[:3])
    pd.testing.assert_series_equal(forecast.iloc[:3], ewma_forecast(observations.iloc[:3], 0.9))
    with pytest.raises(ValueError, match="chronological"):
        ewma_forecast(observations.iloc[::-1], 0.9)


def test_linear_forecast_never_uses_validation_outcomes():
    rng = np.random.default_rng(22)
    rows = pd.DataFrame(
        rng.uniform(0.01, 0.5, size=(50, 3)),
        columns=["trailing_vol_5", "trailing_vol_20", "trailing_vol_60"],
        index=[f"id-{i}" for i in range(50)],
    )
    rows["forward_vol_5"] = 0.01 + rows["trailing_vol_5"] * 0.8
    fit, valid = tuple(rows.index[:40]), tuple(rows.index[40:])
    forecast, coefficients = linear_forecast(rows, fit, valid)
    assert np.allclose(forecast, rows.loc[list(valid), "forward_vol_5"])
    rows.loc[list(valid), "forward_vol_5"] = 1000
    replay, _ = linear_forecast(rows, fit, valid)
    pd.testing.assert_series_equal(forecast, replay)
    assert coefficients["coefficients"]["trailing_vol_5"] == pytest.approx(0.8)


def test_rolling_windows_are_subsets_of_purged_training(frozen_experiment_inputs):
    _, v2, _, _ = frozen_experiment_inputs
    observations, plan, _ = load_v2_development(v2)
    for fold in plan.folds:
        three = training_ids(observations, fold, 3)
        five = training_ids(observations, fold, 5)
        assert set(three) < set(five) <= set(fold.training_ids)
        boundary = observations.loc[fold.validation_ids[0], "as_of"]
        assert (observations.loc[list(three), "as_of"] >= boundary - pd.DateOffset(years=3)).all()
        assert (observations.loc[list(five), "label_end"] < boundary).all()


def diagnostic_fixture():
    common = pd.DataFrame(
        {
            "sample_id": list("abcdef"),
            "as_of": pd.date_range("2019-01-01", periods=6),
            "partition": "saved",
            "fold_id": "final",
            "true_class": [0, 0, 1, 1, 0, 1],
            "threshold": 0.2,
            "trailing_vol_5": [0.1, 0.3, 0.1, 0.3, 0.1, 0.3],
            "trailing_vol_20": [0.1, 0.1, 0.3, 0.3, 0.1, 0.3],
        }
    )
    model = common.assign(model_id="model", predicted_class=[1, 1, 0, 1, 0, 0])
    baseline = common.assign(model_id="baseline_persistence", predicted_class=[0, 1, 0, 1, 0, 1])
    return pd.concat([model, baseline], ignore_index=True)


def test_saved_errors_have_denominators_runs_and_paired_comparisons():
    tables = error_tables(diagnostic_fixture())
    annotated = tables["diagnostic_observations.csv"]
    model = annotated.loc[annotated["model_id"] == "model"]
    assert model["regime"].tolist()[:4] == ["calm", "rising", "cooling", "elevated"]
    assert model["outcome"].tolist() == ["FP", "FP", "FN", "TP", "TN", "FN"]
    summary = tables["error_summary.csv"]
    total = summary.loc[(summary.model_id == "model") & (summary.grouping == "overall")].iloc[0]
    assert total["normal_support"] == total["high_support"] == 3
    assert total["false_positive_rate"] == total["miss_rate"] == pytest.approx(2 / 3)
    runs = tables["error_runs.csv"]
    assert runs.loc[runs.model_id == "model", "origins"].tolist() == [2, 1, 1]
    paired = tables["paired_errors.csv"].iloc[0]
    assert paired["extra_false_alarms"] == 1
    assert paired["lost_high_detections"] == 1
    assert paired["extra_high_detections"] == paired["avoided_false_alarms"] == 0
    # A calm group has no high outcomes: an absent denominator is not reported as 0% misses.
    calm = summary.loc[
        (summary.model_id == "model") & (summary.grouping == "regime") & (summary.regime == "calm")
    ].iloc[0]
    assert pd.isna(calm["miss_rate"])
    bad = diagnostic_fixture()
    bad.loc[0, "true_class"] = 1
    with pytest.raises(ValueError, match="same truth"):
        error_tables(bad)


def test_cli_runs_all_four_experiments_without_new_holdout_access(
    frozen_experiment_inputs, monkeypatch, capsys, tmp_path
):
    import qr_haven.ml.volatility.experiments as experiments

    root, v2, run, evaluation = frozen_experiment_inputs
    output = root / "experiments" / run.name / "improvements-v1"
    observations, plan, _ = load_v2_development(v2)
    frozen_files = [
        p for directory in (v2, run, evaluation) for p in directory.iterdir() if p.is_file()
    ]
    frozen_files.append(root / "holdout_evaluation_history.jsonl")
    before = {p: p.read_bytes() for p in frozen_files}
    original_open = Path.open
    original_scores = experiments.classifier_scores
    fit_calls = []

    def guarded_open(path, *args, **kwargs):
        if path.name == "sealed_holdout_outcomes.csv":
            pytest.fail("Exploratory workflow opened sealed outcomes")
        return original_open(path, *args, **kwargs)

    def guarded_scores(candidate, rows, fit_ids, valid_ids, threshold, power):
        assert (output / "config.json").exists(), "protocol must be saved before fitting"
        assert set(rows.index).isdisjoint(plan.holdout_ids + plan.quarantined_ids)
        assert set(fit_ids).isdisjoint(valid_ids)
        assert rows.loc[list(fit_ids), "label_end"].max() < rows.loc[valid_ids[0], "as_of"]
        fit_calls.append((fit_ids, valid_ids, power))
        return original_scores(candidate, rows, fit_ids, valid_ids, threshold, power)

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "open", guarded_open)
        guarded.setattr(experiments, "classifier_scores", guarded_scores)
        assert main(["experiment", "--run-dir", str(run), "--evaluation-dir", str(evaluation)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["output_dir"] == str(output)
    assert status["holdout_diagnosed"] is True
    assert status["new_holdout_predictions"] is False
    assert len(fit_calls) == 5 * len(plan.folds), "cutoff variants must reuse fitted scores"
    assert {p: p.read_bytes() for p in frozen_files} == before
    manifest = verify_volatility_experiments(output)
    assert manifest["v3_control_replayed"] is True
    assert manifest["quarantined_observations_used"] is False
    predictions = pd.read_csv(output / "cv_predictions.csv")
    assert predictions.variant_id.nunique() == len(experiment_variants()) + 3 == 18
    assert set(predictions.sample_id).isdisjoint(plan.holdout_ids + plan.quarantined_ids)
    assert (predictions.groupby(["fold_id", "sample_id"])["true_class"].nunique() == 1).all()
    assert (predictions.groupby("fold_id")["threshold"].nunique() == 1).all()
    for _variant, rows in predictions.groupby("variant_id"):
        assert set(rows.sample_id) == {key for fold in plan.folds for key in fold.validation_ids}
    ranking = pd.read_csv(output / "ranking.csv")
    results = pd.read_csv(output / "cv_results.csv")
    means = results.groupby(["variant_id", "frequency"])["balanced_accuracy"].mean()
    actual = ranking.set_index(["variant_id", "frequency"])["mean_balanced_accuracy"]
    assert np.allclose(actual.sort_index(), means.sort_index())
    assert ranking.loc[ranking.variant_id == CONTROL_ID, "delta_vs_control"].eq(0).all()
    membership = pd.read_csv(output / "training_membership.csv")
    for fold in plan.folds:
        rows = membership.loc[
            (membership.fold_id == fold.fold_id) & (membership.variant_id == "rolling_3y")
        ]
        assert tuple(rows.sample_id) == training_ids(observations, fold, 3)
    assert "## 4. Rolling training windows" in (output / "report.md").read_text()
    assert main(["verify-experiment", "--output-dir", str(output)]) == 0
    with pytest.raises(FileExistsError, match="immutable"):
        run_volatility_experiments(run, v2, output)
    damaged = tmp_path / "damaged"
    shutil.copytree(output, damaged)
    (damaged / "ranking.csv").write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch: ranking.csv"):
        verify_volatility_experiments(damaged)
    with pytest.raises(ValueError, match="one result per frozen fold"):
        _rank_and_pool(results.iloc[1:], predictions, {f.fold_id for f in plan.folds})


def test_failure_reserves_experiment_id(frozen_experiment_inputs, monkeypatch):
    import qr_haven.ml.volatility.experiments as experiments

    root, v2, run, _ = frozen_experiment_inputs
    output = root / "experiments" / run.name / "failed-run"

    def fail(*args, **kwargs):
        raise ValueError("simulated fitting failure")

    monkeypatch.setattr(experiments, "_compare", fail)
    with pytest.raises(ValueError, match="simulated fitting failure"):
        run_volatility_experiments(run, v2, output)
    assert json.loads((output / "manifest.json").read_text())["state"] == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        verify_volatility_experiments(output)
    with pytest.raises(FileExistsError, match="immutable"):
        run_volatility_experiments(run, v2, output)


def test_output_cannot_be_written_into_frozen_input(frozen_experiment_inputs):
    _, v2, run, _ = frozen_experiment_inputs
    with pytest.raises(ValueError, match="outside frozen input"):
        run_volatility_experiments(run, v2, run / "accidental-output")

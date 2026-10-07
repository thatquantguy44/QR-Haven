"""Point-in-time feature, forward-label, purge, and V2 artifact tests."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from qr_haven.ml.volatility import (
    FEATURES,
    VolatilityProfile,
    build_volatility_dataset,
    fold_targets,
    prepare_walk_forward_plan,
    verify_v2_artifacts,
    write_v2_artifacts,
)


@pytest.fixture(scope="module")
def profile() -> VolatilityProfile:
    return VolatilityProfile(
        profile_id="fixture-v1",
        symbol="FIX",
        snapshot_dir=Path("fixture"),
        canonical_filename="fixture.csv",
        study_start=date(2005, 1, 1),
        study_end=date(2014, 12, 31),
        validation_years=(2010, 2011),
        holdout_start=date(2012, 1, 1),
        holdout_end=date(2013, 12, 31),
    )


@pytest.fixture(scope="module")
def canonical(profile: VolatilityProfile) -> pd.DataFrame:
    sessions = xcals.get_calendar(
        "XNYS", start=profile.study_start, end=profile.study_end
    ).sessions
    rng = np.random.default_rng(72)
    scale = np.where((sessions.year % 3) == 0, 0.018, 0.007)
    returns = rng.normal(0.0002, scale)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(sessions).tz_localize("UTC"),
            "symbol": profile.symbol,
            "adjusted_close": prices,
        }
    )


@pytest.fixture(scope="module")
def dataset(profile: VolatilityProfile, canonical: pd.DataFrame):
    return build_volatility_dataset(canonical, {"source_sha256": "a" * 64}, profile)


def test_feature_and_label_values_use_exact_point_in_time_windows(dataset, canonical):
    row = dataset.observations.iloc[100]
    sample_id = str(dataset.observations.index[100])
    origin = canonical.index[
        pd.to_datetime(canonical["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
        == row["as_of"]
    ][0]
    prices = canonical["adjusted_close"].to_numpy(dtype=float)
    returns = np.full(len(prices), np.nan)
    returns[1:] = np.log(prices[1:] / prices[:-1])
    for window in (5, 10, 20, 60):
        expected = math.sqrt(
            (252 / window) * np.square(returns[origin - window + 1 : origin + 1]).sum()
        )
        assert row[f"trailing_vol_{window}"] == pytest.approx(expected)
    assert row["log_return_1"] == pytest.approx(returns[origin])
    assert row["log_return_5"] == pytest.approx(math.log(prices[origin] / prices[origin - 5]))
    assert row["abs_log_return_1"] == pytest.approx(abs(returns[origin]))
    assert row["vol_ratio_5_20"] == pytest.approx(
        row["trailing_vol_5"] / row["trailing_vol_20"]
    )
    assert row["drawdown_60"] == pytest.approx(
        prices[origin] / prices[origin - 59 : origin + 1].max() - 1
    )
    expected_forward = math.sqrt(
        (252 / 5) * np.square(returns[origin + 1 : origin + 6]).sum()
    )
    assert row["forward_vol_5"] == pytest.approx(expected_forward)

    audit = dataset.interval_audit.loc[sample_id]
    sessions = pd.to_datetime(canonical["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    assert json.loads(audit["trailing_vol_60_price_sessions"]) == sessions.iloc[
        origin - 60 : origin + 1
    ].tolist()
    assert json.loads(audit["drawdown_60_price_sessions"]) == sessions.iloc[
        origin - 59 : origin + 1
    ].tolist()
    assert json.loads(audit["forward_vol_5_price_sessions"]) == sessions.iloc[
        origin : origin + 6
    ].tolist()
    calendar = xcals.get_calendar(
        "XNYS", start=dataset.profile.study_start, end=dataset.profile.study_end
    )
    assert row["available_at"] == calendar.schedule.loc[row["as_of"], "close"]


def test_every_audited_feature_window_ends_by_as_of_and_label_starts_after(dataset):
    for location in (0, len(dataset.interval_audit) // 2, len(dataset.interval_audit) - 1):
        row = dataset.interval_audit.iloc[location]
        as_of = row["as_of"].strftime("%Y-%m-%d")
        for name in FEATURES:
            sessions = json.loads(row[f"{name}_price_sessions"])
            assert sessions[-1] <= as_of
        label_sessions = json.loads(row["forward_vol_5_price_sessions"])
        assert label_sessions[0] == as_of
        assert label_sessions[1] > as_of
        assert label_sessions[-1] == row["label_end"].strftime("%Y-%m-%d")


def test_feature_warmup_and_forward_tail_are_excluded(dataset, canonical):
    sessions = pd.to_datetime(canonical["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    assert dataset.observations["as_of"].iloc[0] == sessions.iloc[60]
    assert dataset.observations["as_of"].iloc[-1] == sessions.iloc[-6]
    assert len(dataset.observations) == len(canonical) - 65
    assert tuple(dataset.observations.loc[:, list(FEATURES)].columns) == FEATURES


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(index=100), "sessions do not match"),
        (lambda frame: pd.concat([frame, frame.iloc[[10]]]).sort_values("timestamp"), "duplicate"),
        (lambda frame: frame.iloc[::-1], "chronological"),
        (lambda frame: frame.assign(symbol="OTHER"), "only symbol"),
        (lambda frame: frame.assign(adjusted_close=np.nan), "positive and finite"),
        (lambda frame: frame.assign(adjusted_close=0.0), "positive and finite"),
        (lambda frame: frame.drop(columns="adjusted_close"), "missing columns"),
    ],
)
def test_invalid_canonical_prices_are_rejected(profile, canonical, mutation, message):
    with pytest.raises(ValueError, match=message):
        build_volatility_dataset(mutation(canonical), {"source_sha256": "a" * 64}, profile)


def test_walk_forward_folds_use_train_only_thresholds_and_strict_purge(dataset):
    plan = prepare_walk_forward_plan(dataset)
    assert [fold.validation_year for fold in plan.folds] == [2010, 2011]
    for fold in plan.folds:
        training = dataset.observations.loc[list(fold.training_ids)]
        validation = dataset.observations.loc[list(fold.validation_ids)]
        assert training["label_end"].max() < validation["as_of"].min()
        assert set(validation["as_of"].dt.year) == {fold.validation_year}
        assert len(fold.purged_boundary_ids) == 5
        expected = np.quantile(training["forward_vol_5"], 0.75, method="linear")
        assert fold.threshold == expected
        assert set(fold_targets(dataset, fold.training_ids, fold.threshold)) == {0, 1}
        assert set(fold_targets(dataset, fold.validation_ids, fold.threshold)) <= {0, 1}

    final_training = dataset.observations.loc[list(plan.final_training_ids)]
    holdout = dataset.observations.loc[list(plan.holdout_ids)]
    assert final_training["label_end"].max() < holdout["as_of"].min()
    assert len(plan.final_purged_boundary_ids) == 5
    assert len(plan.holdout_end_purged_ids) == 5
    assert holdout["as_of"].dt.year.min() == 2012
    assert holdout["as_of"].dt.year.max() == 2013
    assert holdout["label_end"].max().date() <= date(2013, 12, 31)
    quarantined = dataset.observations.loc[list(plan.quarantined_ids)]
    assert quarantined["as_of"].dt.year.eq(2014).all()
    assert plan.metadata["holdout_labels_inspected_during_preparation"] is False


def test_holdout_outcome_changes_cannot_change_development_or_final_threshold(dataset):
    plan = prepare_walk_forward_plan(dataset)
    changed_observations = dataset.observations.copy()
    changed_observations.loc[list(plan.holdout_ids), "forward_vol_5"] = 1_000_000.0
    changed = replace(dataset, observations=changed_observations)
    assert prepare_walk_forward_plan(changed) == plan


def test_v2_artifacts_are_immutable_and_hash_verified(dataset, tmp_path):
    plan = prepare_walk_forward_plan(dataset)
    output = tmp_path / "dataset-v1"
    manifest = write_v2_artifacts(output, dataset, plan)
    assert manifest["holdout_outcomes_summarized"] is False
    assert manifest["development_access_to_holdout_outcomes"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "development_observations.csv",
        "holdout_features.csv",
        "interval_audit.csv",
        "manifest.json",
        "sealed_holdout_outcomes.csv",
        "split.json",
        "split_membership.csv",
    ]
    development = pd.read_csv(output / "development_observations.csv")
    holdout_features = pd.read_csv(output / "holdout_features.csv")
    sealed_outcomes = pd.read_csv(output / "sealed_holdout_outcomes.csv")
    assert len(development) == len(plan.final_training_ids)
    assert "forward_vol_5" not in holdout_features
    assert list(sealed_outcomes.columns) == ["sample_id", "forward_vol_5"]
    assert verify_v2_artifacts(output) == manifest
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_v2_artifacts(output, dataset, plan)
    with (output / "split.json").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_v2_artifacts(output)

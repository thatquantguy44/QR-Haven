"""Reject malformed development membership before any model can fit."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from qr_haven.ml.volatility import (
    VolatilityProfile,
    build_volatility_dataset,
    prepare_walk_forward_plan,
    write_v2_artifacts,
)
from qr_haven.ml.volatility.artifacts import load_v2_development
from qr_haven.ml.volatility.validation import validate_development_inputs


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("boundary")
    profile = VolatilityProfile(
        "boundary-fixture", "FIX", Path("fixture"), "fixture.csv",
        date(2008, 1, 1), date(2012, 12, 31), (2010,),
        date(2011, 1, 1), date(2012, 12, 31),
    )
    sessions = xcals.get_calendar("XNYS", start=profile.study_start, end=profile.study_end).sessions
    rng = np.random.default_rng(45)
    frame = pd.DataFrame({
        "timestamp": sessions.tz_localize("UTC"), "symbol": "FIX",
        "adjusted_close": np.exp(rng.normal(0, 0.01, len(sessions)).cumsum()) * 100,
    })
    dataset = build_volatility_dataset(frame, {"source_sha256": "e" * 64}, profile)
    plan = prepare_walk_forward_plan(dataset)
    write_v2_artifacts(root / "v2", dataset, plan)
    return load_v2_development(root / "v2")


def test_round_trip_revalidates_thresholds_and_boundaries(inputs):
    observations, plan, manifest = inputs
    validate_development_inputs(observations, plan, manifest)


@pytest.mark.parametrize("change", ["leakage", "duplicate", "omission", "threshold", "purge"])
def test_invalid_fold_rejected_before_fit(inputs, change):
    observations, plan, manifest = inputs
    fold = plan.folds[0]
    if change == "leakage":
        fold = replace(fold, training_ids=(*fold.training_ids, fold.validation_ids[0]))
    elif change == "duplicate":
        fold = replace(fold, training_ids=(*fold.training_ids, fold.training_ids[-1]))
    elif change == "omission":
        fold = replace(fold, validation_ids=fold.validation_ids[:-1])
    elif change == "threshold":
        fold = replace(fold, threshold=fold.threshold + 0.01)
    else:
        fold = replace(fold, purged_boundary_ids=())
    with pytest.raises(ValueError):
        validate_development_inputs(observations, replace(plan, folds=(fold,)), manifest)


def test_final_threshold_and_dates_cannot_be_changed(inputs):
    observations, plan, manifest = inputs
    with pytest.raises(ValueError, match="75th percentile"):
        validate_development_inputs(
            observations, replace(plan, final_threshold=0.0), manifest
        )
    altered = observations.copy()
    altered.loc[altered.index[-1], "label_end"] = pd.Timestamp("2011-01-03")
    with pytest.raises(ValueError, match="cross the holdout"):
        validate_development_inputs(altered, plan, manifest)

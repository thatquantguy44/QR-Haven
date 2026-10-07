"""Recheck V2 development inputs without reading holdout files."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qr_haven.ml.volatility.contracts import FEATURES, PROFILES, WalkForwardPlan


def validate_development_inputs(
    observations: pd.DataFrame, plan: WalkForwardPlan, manifest: dict[str, Any]
) -> None:
    """Prove frozen membership, dates and thresholds using development rows alone."""
    if manifest.get("schema_version") != 1 or manifest.get("feature_order") != list(FEATURES):
        raise ValueError("Unsupported V2 schema or feature order")
    profile = manifest["profile"]
    years = tuple(profile["validation_years"])
    holdout_start = pd.Timestamp(profile["holdout_start"])
    if plan.profile_id in PROFILES:
        frozen = PROFILES[plan.profile_id]
        for key, expected in (
            ("symbol", frozen.symbol),
            ("calendar", frozen.calendar),
            ("study_start", frozen.study_start.isoformat()),
            ("study_end", frozen.study_end.isoformat()),
            ("holdout_start", frozen.holdout_start.isoformat()),
            ("holdout_end", frozen.holdout_end.isoformat()),
            ("validation_years", list(frozen.validation_years)),
        ):
            if profile.get(key) != expected:
                raise ValueError(f"V2 profile disagrees with frozen {key}")
    if not years or tuple(f.validation_year for f in plan.folds) != years:
        raise ValueError("V2 validation years disagree with the profile")
    if len({f.fold_id for f in plan.folds}) != len(years):
        raise ValueError("V2 fold identifiers must be unique")
    if observations.empty or not observations["as_of"].is_monotonic_increasing:
        raise ValueError("Development observations must be nonempty and chronological")
    if observations["as_of"].duplicated().any():
        raise ValueError("Development origins must be unique")
    if not np.isfinite(observations[[*FEATURES, "forward_vol_5"]].to_numpy(dtype=float)).all():
        raise ValueError("Development features and outcomes must be finite")
    if (observations["forward_vol_5"] < 0).any():
        raise ValueError("Development outcomes must be nonnegative")
    if not (
        (observations["as_of"] < observations["label_start"])
        & (observations["label_start"] <= observations["label_end"])
        & (observations["label_end"] < holdout_start)
    ).all():
        raise ValueError("Development label intervals cross the holdout or have invalid timing")
    expected_ids = tuple(
        f"{plan.profile_id}/{profile['symbol']}/{value.date().isoformat()}"
        for value in observations["as_of"]
    )
    if expected_ids != plan.final_training_ids:
        raise ValueError("Development IDs disagree with their forecast origins")
    excluded = (
        *plan.holdout_ids,
        *plan.final_purged_boundary_ids,
        *plan.holdout_end_purged_ids,
        *plan.quarantined_ids,
    )
    if set(excluded) & set(expected_ids):
        raise ValueError("Excluded/holdout IDs cannot enter development")

    def check_threshold(ids: tuple[str, ...], threshold: float) -> None:
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Training IDs must be nonempty and unique")
        values = observations.loc[list(ids), "forward_vol_5"].to_numpy(dtype=float)
        expected = float(np.quantile(values, 0.75, method="linear"))
        if not np.isfinite(threshold) or not np.isclose(threshold, expected, rtol=0, atol=1e-14):
            raise ValueError("Threshold differs from the training-only 75th percentile")

    check_threshold(plan.final_training_ids, plan.final_threshold)
    for fold in plan.folds:
        expected_validation = tuple(
            observations.index[observations["as_of"].dt.year == fold.validation_year]
        )
        if not expected_validation or fold.validation_ids != expected_validation:
            raise ValueError("Validation membership must cover its complete eligible year")
        boundary = observations.loc[fold.validation_ids[0], "as_of"]
        expected_training = tuple(observations.index[observations["label_end"] < boundary])
        if fold.training_ids != expected_training:
            raise ValueError("Training membership violates the expanding-window purge")
        expected_purged = tuple(
            observations.index[
                (observations["as_of"] < boundary) & (observations["label_end"] >= boundary)
            ]
        )
        if fold.purged_boundary_ids != expected_purged:
            raise ValueError("Purged membership disagrees with the label intervals")
        check_threshold(fold.training_ids, fold.threshold)

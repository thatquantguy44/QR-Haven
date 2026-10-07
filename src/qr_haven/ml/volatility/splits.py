"""Purged expanding-year split construction with train-only label thresholds."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

import numpy as np
import pandas as pd

from qr_haven.ml.volatility.contracts import (
    VolatilityDataset,
    WalkForwardFold,
    WalkForwardPlan,
)
from qr_haven.ml.volatility.dataset import fold_targets

THRESHOLD_QUANTILE = 0.75
THRESHOLD_INTERPOLATION: Literal["linear"] = "linear"


def _ids(mask: pd.Series, observations: pd.DataFrame) -> tuple[str, ...]:
    return tuple(observations.index[mask].astype(str))


def _threshold(dataset: VolatilityDataset, sample_ids: tuple[str, ...]) -> float:
    if not sample_ids:
        raise ValueError("Cannot estimate a threshold from an empty training partition")
    values = dataset.observations.loc[list(sample_ids), "forward_vol_5"].to_numpy(dtype=float)
    return float(np.quantile(values, THRESHOLD_QUANTILE, method=THRESHOLD_INTERPOLATION))


def prepare_walk_forward_plan(dataset: VolatilityDataset) -> WalkForwardPlan:
    """Create chronological folds without reading final-holdout outcome values."""
    observations = dataset.observations
    profile = dataset.profile
    as_of = pd.to_datetime(observations["as_of"])
    label_end = pd.to_datetime(observations["label_end"])
    holdout_candidates = observations.loc[
        (as_of.dt.date >= profile.holdout_start) & (as_of.dt.date <= profile.holdout_end)
    ]
    if holdout_candidates.empty:
        raise ValueError("Frozen holdout contains no eligible forecast origins")
    first_holdout_origin = pd.Timestamp(holdout_candidates["as_of"].min())
    folds: list[WalkForwardFold] = []
    for validation_year in profile.validation_years:
        validation_mask = as_of.dt.year.eq(validation_year) & (label_end < first_holdout_origin)
        validation_ids = _ids(validation_mask, observations)
        if not validation_ids:
            raise ValueError(f"Validation year {validation_year} has no eligible observations")
        first_validation_origin = pd.Timestamp(
            observations.loc[list(validation_ids), "as_of"].min()
        )
        prior_mask = as_of < first_validation_origin
        training_mask = prior_mask & (label_end < first_validation_origin)
        purged_mask = prior_mask & (label_end >= first_validation_origin)
        training_ids = _ids(training_mask, observations)
        fold = WalkForwardFold(
            fold_id=f"year-{validation_year}",
            validation_year=validation_year,
            training_ids=training_ids,
            validation_ids=validation_ids,
            purged_boundary_ids=_ids(purged_mask, observations),
            threshold=_threshold(dataset, training_ids),
        )
        folds.append(fold)

    prior_holdout_mask = as_of < first_holdout_origin
    final_training_mask = prior_holdout_mask & (label_end < first_holdout_origin)
    final_purged_mask = prior_holdout_mask & (label_end >= first_holdout_origin)
    final_training_ids = _ids(final_training_mask, observations)
    holdout_mask = (
        (as_of >= first_holdout_origin)
        & (as_of.dt.date <= profile.holdout_end)
        & (label_end.dt.date <= profile.holdout_end)
    )
    holdout_end_purged_mask = (
        (as_of >= first_holdout_origin)
        & (as_of.dt.date <= profile.holdout_end)
        & (label_end.dt.date > profile.holdout_end)
    )
    quarantined_mask = as_of.dt.date > profile.holdout_end
    plan = WalkForwardPlan(
        profile_id=profile.profile_id,
        source_sha256=dataset.source_sha256,
        folds=tuple(folds),
        final_training_ids=final_training_ids,
        final_purged_boundary_ids=_ids(final_purged_mask, observations),
        final_threshold=_threshold(dataset, final_training_ids),
        holdout_ids=_ids(holdout_mask, observations),
        holdout_end_purged_ids=_ids(holdout_end_purged_mask, observations),
        quarantined_ids=_ids(quarantined_mask, observations),
        metadata={
            "schema_version": 1,
            "threshold_quantile": THRESHOLD_QUANTILE,
            "threshold_interpolation": THRESHOLD_INTERPOLATION,
            "validation_years": list(profile.validation_years),
            "holdout_start": profile.holdout_start.isoformat(),
            "holdout_end": profile.holdout_end.isoformat(),
            "first_holdout_origin": first_holdout_origin.date().isoformat(),
            "holdout_labels_inspected_during_preparation": False,
        },
    )
    validate_walk_forward_plan(dataset, plan)
    return plan


def validate_walk_forward_plan(dataset: VolatilityDataset, plan: WalkForwardPlan) -> None:
    """Recompute thresholds and prove every fold's strict interval purge."""
    if plan.profile_id != dataset.profile.profile_id or plan.source_sha256 != dataset.source_sha256:
        raise ValueError("Walk-forward plan does not match its dataset profile and source")
    if tuple(fold.validation_year for fold in plan.folds) != dataset.profile.validation_years:
        raise ValueError("Walk-forward validation years do not match the frozen profile")
    observations = dataset.observations
    all_ids = set(dataset.sample_ids)
    for fold in plan.folds:
        training, validation, purged = (
            set(fold.training_ids),
            set(fold.validation_ids),
            set(fold.purged_boundary_ids),
        )
        if not training or not validation or (training | validation | purged) - all_ids:
            raise ValueError(f"Fold {fold.fold_id} has empty or unknown membership")
        if training & validation or training & purged or validation & purged:
            raise ValueError(f"Fold {fold.fold_id} membership must be disjoint")
        first_validation = pd.Timestamp(observations.loc[list(validation), "as_of"].min())
        training_end = pd.to_datetime(observations.loc[list(training), "label_end"])
        if not (training_end < first_validation).all():
            raise ValueError(f"Fold {fold.fold_id} has a training label crossing validation")
        validation_years = set(
            pd.to_datetime(observations.loc[list(validation), "as_of"]).dt.year
        )
        if validation_years != {fold.validation_year}:
            raise ValueError(f"Fold {fold.fold_id} validation membership has the wrong year")
        if not np.isclose(fold.threshold, _threshold(dataset, fold.training_ids), rtol=0, atol=0):
            raise ValueError(f"Fold {fold.fold_id} threshold is not the exact training quantile")
        if set(fold_targets(dataset, fold.training_ids, fold.threshold)) != {0, 1}:
            raise ValueError(f"Fold {fold.fold_id} training target must contain both classes")

    final_training, holdout = set(plan.final_training_ids), set(plan.holdout_ids)
    if not final_training or not holdout or final_training & holdout:
        raise ValueError("Final training and holdout membership must be nonempty and disjoint")
    first_holdout = pd.Timestamp(observations.loc[list(holdout), "as_of"].min())
    final_label_end = pd.to_datetime(observations.loc[list(final_training), "label_end"])
    if not (final_label_end < first_holdout).all():
        raise ValueError("A final-training label crosses into the holdout")
    if not np.isclose(
        plan.final_threshold, _threshold(dataset, plan.final_training_ids), rtol=0, atol=0
    ):
        raise ValueError("Final threshold is not the exact final-training quantile")
    if set(fold_targets(dataset, plan.final_training_ids, plan.final_threshold)) != {0, 1}:
        raise ValueError("Final training target must contain both classes")
    holdout_rows = observations.loc[list(holdout)]
    if not (
        (pd.to_datetime(holdout_rows["as_of"]).dt.date >= dataset.profile.holdout_start)
        & (pd.to_datetime(holdout_rows["label_end"]).dt.date <= dataset.profile.holdout_end)
    ).all():
        raise ValueError("Holdout origins or labels fall outside the frozen boundary")


def plan_dict(plan: WalkForwardPlan) -> dict[str, Any]:
    """Return a JSON-serializable plan representation."""
    return asdict(plan)


def membership_frame(plan: WalkForwardPlan) -> pd.DataFrame:
    """Flatten fold and final membership into an inspectable audit table."""
    rows: list[dict[str, Any]] = []
    for fold in plan.folds:
        for role, sample_ids in (
            ("training", fold.training_ids),
            ("validation", fold.validation_ids),
            ("purged_at_boundary", fold.purged_boundary_ids),
        ):
            rows.extend(
                {
                    "stage": "development",
                    "fold_id": fold.fold_id,
                    "role": role,
                    "sample_id": sample_id,
                    "threshold": fold.threshold,
                }
                for sample_id in sample_ids
            )
    for role, sample_ids in (
        ("training", plan.final_training_ids),
        ("purged_at_start", plan.final_purged_boundary_ids),
        ("holdout", plan.holdout_ids),
        ("purged_at_end", plan.holdout_end_purged_ids),
        ("quarantined", plan.quarantined_ids),
    ):
        rows.extend(
            {
                "stage": "final",
                "fold_id": "final",
                "role": role,
                "sample_id": sample_id,
                "threshold": plan.final_threshold if role == "training" else np.nan,
            }
            for sample_id in sample_ids
        )
    return pd.DataFrame(rows)

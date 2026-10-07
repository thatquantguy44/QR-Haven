"""Persistable sample membership; preprocessing is deliberately absent."""

from __future__ import annotations

from typing import Any

from qr_haven.data.banknotes import DUPLICATE_POLICY, BanknoteDataset
from qr_haven.ml.classification.contracts import (
    ClassificationConfig,
    SplitManifest,
    require_sklearn,
)


def validate_split(dataset: BanknoteDataset, split: SplitManifest, folds: int) -> None:
    development, test = set(split.development_ids), set(split.test_ids)
    if (
        len(development) != len(split.development_ids)
        or len(test) != len(split.test_ids)
        or development & test
        or development | test != set(dataset.sample_ids)
    ):
        raise ValueError("Split must have disjoint, unique IDs and complete population coverage")
    if not development or not test or set(split.validation_folds) != development:
        raise ValueError("Split needs both partitions and one fold assignment per development row")
    if split.source_sha256 != dataset.source_manifest["raw_sha256"]:
        raise ValueError("Split source hash mismatch")
    if set(split.validation_folds.values()) != set(range(folds)):
        raise ValueError("Invalid validation-fold assignments")
    # Test labels are inspected only at preparation/evaluation, never during model selection.
    for fold in range(folds):
        for validation in (True, False):
            ids = [
                key
                for key in split.development_ids
                if (split.validation_folds[key] == fold) == validation
            ]
            if set(dataset.target.loc[ids]) != {0, 1}:
                raise ValueError("Every CV training/validation side must contain both classes")


def prepare_banknote_split(
    dataset: BanknoteDataset,
    config: ClassificationConfig,
) -> SplitManifest:
    require_sklearn()
    from sklearn.model_selection import StratifiedKFold, train_test_split

    ids = sorted(dataset.sample_ids)
    development, test = train_test_split(
        ids,
        test_size=config.split.test_size,
        random_state=config.split.seed,
        stratify=dataset.target.loc[ids],
    )
    counts = dataset.target.loc[development].value_counts()
    if len(counts) != 2 or int(counts.min()) < config.split.cv_folds:
        raise ValueError("Too few examples per development class for the requested CV folds")
    if set(dataset.target.loc[test]) != {0, 1}:
        raise ValueError("The test partition must contain both classes")
    cv = StratifiedKFold(
        n_splits=config.split.cv_folds, shuffle=True, random_state=config.split.cv_seed
    )
    assignments = {
        development[int(i)]: fold
        for fold, (_, validation) in enumerate(
            cv.split(dataset.features.loc[development], dataset.target.loc[development])
        )
        for i in validation
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "duplicate_policy": DUPLICATE_POLICY,
        "config": config.split.model_dump(),
        "counts": {"development": len(development), "test": len(test)},
        "class_counts": {
            role: {str(k): int(v) for k, v in dataset.target.loc[part].value_counts().items()}
            for role, part in (("development", development), ("test", test))
        },
        "class_proportions": {
            role: {
                str(k): float(v)
                for k, v in dataset.target.loc[part].value_counts(normalize=True).items()
            }
            for role, part in (("development", development), ("test", test))
        },
    }
    result = SplitManifest(
        tuple(development),
        tuple(test),
        assignments,
        dataset.source_manifest["raw_sha256"],
        metadata,
    )
    validate_split(dataset, result, config.split.cv_folds)
    return result

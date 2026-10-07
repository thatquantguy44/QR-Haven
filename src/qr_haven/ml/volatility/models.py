"""Frozen V3 model grid, weighted fitting, and development baselines."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from qr_haven.ml.classification.contracts import require_sklearn


@dataclass(frozen=True)
class VolatilityCandidate:
    """One learned candidate in the fixed deterministic ranking order."""

    candidate_id: str
    family: str
    family_order: int
    grid_order: int
    parameters: dict[str, Any]


def candidates() -> tuple[VolatilityCandidate, ...]:
    """Return the predeclared three logistic, nine forest, and eight HGB candidates."""
    grids: tuple[tuple[str, list[dict[str, Any]]], ...] = (
        ("logistic_regression", [{"C": value} for value in (0.1, 1.0, 10.0)]),
        (
            "random_forest",
            [
                {"max_depth": depth, "min_samples_leaf": leaf}
                for depth, leaf in product((5, 10, None), (1, 5, 20))
            ],
        ),
        (
            "hist_gradient_boosting",
            [
                {
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": leaves,
                    "l2_regularization": regularization,
                }
                for learning_rate, leaves, regularization in product(
                    (0.03, 0.1), (7, 15), (0.0, 1.0)
                )
            ],
        ),
    )
    return tuple(
        VolatilityCandidate(
            candidate_id=f"{family}_{grid_order:02d}",
            family=family,
            family_order=family_order,
            grid_order=grid_order,
            parameters=parameters,
        )
        for family_order, (family, grid) in enumerate(grids)
        for grid_order, parameters in enumerate(grid)
    )


def make_estimator(candidate: VolatilityCandidate) -> Any:
    """Create an unfitted estimator with only the frozen settings."""
    require_sklearn()
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if candidate.family == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=candidate.parameters["C"], solver="lbfgs", max_iter=2000
                    ),
                ),
            ]
        )
    if candidate.family == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            random_state=5401,
            n_jobs=1,
            **candidate.parameters,
        )
    if candidate.family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300,
            early_stopping=False,
            random_state=5402,
            **candidate.parameters,
        )
    raise ValueError(f"Unknown volatility model family: {candidate.family}")


def inverse_frequency_weights(target: pd.Series) -> pd.Series:
    """Assign equal total weight to each class, with mean sample weight one."""
    counts = target.value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("Weighted fitting requires both training classes")
    weights = {label: len(target) / (2 * int(count)) for label, count in counts.items()}
    return target.map(weights).astype("float64").rename("sample_weight")


def fit_estimator(
    estimator: Any, features: pd.DataFrame, target: pd.Series, weights: pd.Series
) -> Any:
    """Fit an estimator while routing weights through a scaler pipeline when needed."""
    from threadpoolctl import threadpool_limits

    if not features.index.equals(target.index) or not target.index.equals(weights.index):
        raise ValueError("Features, targets, and weights must have identical indexes")
    with threadpool_limits(limits=1):
        if hasattr(estimator, "named_steps"):
            estimator.fit(features, target, classifier__sample_weight=weights)
        else:
            estimator.fit(features, target, sample_weight=weights)
    return estimator


def predict_estimator(estimator: Any, features: pd.DataFrame) -> pd.DataFrame:
    """Return native binary predictions and positive-class probabilities."""
    classes = list(estimator.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Fitted volatility classifier must contain classes 0 and 1")
    scores = estimator.predict_proba(features)[:, classes.index(1)]
    return pd.DataFrame(
        {
            "predicted_class": estimator.predict(features).astype("int64"),
            "score_class_1": np.asarray(scores, dtype=float),
            "score_kind": "native_probability_class_1",
        },
        index=features.index,
    )


def baseline_predictions(
    name: str,
    training_target: pd.Series,
    validation_features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Predict one frozen baseline without accessing validation outcomes."""
    values: npt.NDArray[np.int64]
    scores: npt.NDArray[np.float64]
    if name == "majority":
        counts = training_target.value_counts().reindex([0, 1], fill_value=0)
        predicted = int(counts.loc[1] > counts.loc[0])
        values = np.full(len(validation_features), predicted, dtype=np.int64)
        scores = np.full(len(validation_features), float(training_target.mean()))
        kind = "training_prevalence"
    elif name == "always_normal":
        values = np.zeros(len(validation_features), dtype=np.int64)
        scores = np.zeros(len(validation_features), dtype=float)
        kind = "constant_zero"
    elif name == "persistence":
        trailing = validation_features["trailing_vol_5"].to_numpy(dtype=float)
        values = (trailing > threshold).astype(np.int64)
        scores = trailing
        kind = "trailing_vol_5"
    else:
        raise ValueError(f"Unknown volatility baseline: {name}")
    return pd.DataFrame(
        {"predicted_class": values, "score_class_1": scores, "score_kind": kind},
        index=validation_features.index,
    )

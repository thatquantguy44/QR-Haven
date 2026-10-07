"""Stable bounded grids, estimator factories and label-oriented inference."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import FEATURES
from qr_haven.ml.classification.contracts import ClassificationConfig, require_sklearn


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    grid_order: int
    parameters: dict[str, Any]


def candidates(config: ClassificationConfig) -> list[Candidate]:
    result = []
    grids: list[tuple[str, list[dict[str, Any]]]] = [
        ("logistic_regression", [{"C": c} for c in config.models.logistic_regression.C]),
        (
            "svm",
            [
                {"C": c, "gamma": gamma}
                for c, gamma in product(config.models.svm.C, config.models.svm.gamma)
            ],
        ),
        (
            "random_forest",
            [
                {"max_depth": depth, "min_samples_leaf": leaf}
                for depth, leaf in product(
                    config.models.random_forest.max_depth,
                    config.models.random_forest.min_samples_leaf,
                )
            ],
        ),
    ]
    for family, grid in grids:
        for i, params in enumerate(grid):
            result.append(Candidate(f"{family}_{i:02d}", family, i, params))
    return result


def make_estimator(candidate: Candidate, config: ClassificationConfig) -> Any:
    require_sklearn()
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if candidate.family == "baseline":
        return DummyClassifier(strategy=config.models.baseline_strategy)
    if candidate.family == "random_forest":
        settings = config.models.random_forest.model_dump(exclude={"max_depth", "min_samples_leaf"})
        return RandomForestClassifier(
            **settings,
            **candidate.parameters,
            random_state=config.training.estimator_seed,
            n_jobs=config.training.n_jobs,
        )
    if candidate.family == "logistic_regression":
        settings = config.models.logistic_regression.model_dump(exclude={"C", "scaler"})
        estimator = LogisticRegression(**settings, **candidate.parameters)
    elif candidate.family == "svm":
        settings = config.models.svm.model_dump(exclude={"C", "gamma", "scaler"})
        estimator = SVC(**settings, **candidate.parameters)
    else:
        raise ValueError(f"Unknown model family: {candidate.family}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def predict_frame(estimator: Any, features: pd.DataFrame) -> pd.DataFrame:
    classes = list(estimator.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Fitted classifier must contain numeric classes 0 and 1")
    # LogisticRegression also has decision_function; prefer its actual probabilities.
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)[:, classes.index(1)]
        kind = "probability_class_1"
    else:
        scores = estimator.decision_function(features)
        if classes[1] != 1:
            scores = -scores
        kind = "decision_margin"
    return pd.DataFrame(
        {
            "predicted_class": estimator.predict(features),
            "score_class_1": scores,
            "score_kind": kind,
        },
        index=features.index,
    )


@dataclass
class ClassificationModel:
    estimator: Any
    feature_ranges: dict[str, dict[str, float]]
    feature_order: tuple[str, ...] = FEATURES

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise ValueError("Inference requires a DataFrame with named feature columns")
        if not features.columns.is_unique or set(features.columns) != set(self.feature_order):
            raise ValueError(f"Supply exactly these feature columns: {list(self.feature_order)}")
        try:
            frame = features.loc[:, list(self.feature_order)].astype("float64")
        except (TypeError, ValueError) as exc:
            raise ValueError("Inference features must be numeric") from exc
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Inference features must be finite")
        diagnostics = {
            name: int(((frame[name] < bounds["min"]) | (frame[name] > bounds["max"])).sum())
            for name, bounds in self.feature_ranges.items()
        }
        if frame.empty:
            result = pd.DataFrame(
                index=frame.index, columns=["predicted_class", "score_class_1", "score_kind"]
            )
        else:
            result = predict_frame(self.estimator, frame)
        result.attrs["out_of_source_range_counts"] = diagnostics
        return result

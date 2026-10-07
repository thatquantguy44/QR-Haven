"""Fixed forecasts, Platt calibration, and ensemble logic for the V5 challenger."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from qr_haven.ml.volatility.challenger_data import CHALLENGER_FEATURES
from qr_haven.ml.volatility.models import (
    VolatilityCandidate,
    fit_estimator,
    inverse_frequency_weights,
    make_estimator,
)

EPSILON = 1e-12
HAR_FEATURES = ("trailing_vol_5", "trailing_vol_20", "trailing_vol_60")
ENSEMBLE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class ChallengerCandidate:
    candidate_id: str
    kind: str
    order: int
    decay: float | None = None


def challenger_candidates() -> tuple[ChallengerCandidate, ...]:
    return (
        ChallengerCandidate("hist_gradient_boosting_01", "classifier", 0),
        ChallengerCandidate("har_rv", "har", 1),
        ChallengerCandidate("ewma_090", "ewma", 2, 0.90),
        ChallengerCandidate("ewma_094", "ewma", 3, 0.94),
        ChallengerCandidate("ewma_097", "ewma", 4, 0.97),
    )


def five_year_training_ids(
    observations: pd.DataFrame,
    validation_year: int,
    *,
    validation_boundary: pd.Timestamp | None = None,
) -> tuple[str, ...]:
    valid = observations.loc[observations["as_of"].dt.year.eq(validation_year)]
    if valid.empty and validation_boundary is None:
        raise ValueError(f"No challenger validation origins for {validation_year}")
    boundary = (
        pd.Timestamp(valid["as_of"].iloc[0])
        if validation_boundary is None
        else pd.Timestamp(validation_boundary)
    )
    start = boundary - pd.DateOffset(years=5)
    rows = observations.loc[
        (observations["as_of"] >= start)
        & (observations["as_of"] < boundary)
        & (observations["label_end"] < boundary)
    ]
    if rows.empty or not (rows["label_end"] < boundary).all():
        raise ValueError("Five-year training membership is empty or crosses validation")
    return tuple(rows.index.astype(str))


def threshold_for(observations: pd.DataFrame, training_ids: tuple[str, ...]) -> float:
    values = observations.loc[list(training_ids), "forward_vol_5"].to_numpy(float)
    threshold = float(np.quantile(values, 0.75, method="linear"))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Challenger threshold must be finite and nonnegative")
    return threshold


def _histogram_candidate() -> VolatilityCandidate:
    return VolatilityCandidate(
        candidate_id="hist_gradient_boosting_01",
        family="hist_gradient_boosting",
        family_order=2,
        grid_order=1,
        parameters={
            "learning_rate": 0.03,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
        },
    )


def fit_base_model(
    candidate: ChallengerCandidate,
    observations: pd.DataFrame,
    training_ids: tuple[str, ...],
    threshold: float,
) -> Any:
    if candidate.kind == "ewma":
        return None
    rows = observations.loc[list(training_ids)]
    if candidate.kind == "classifier":
        target = (rows["forward_vol_5"] > threshold).astype("int64")
        return fit_estimator(
            make_estimator(_histogram_candidate()),
            rows.loc[:, list(CHALLENGER_FEATURES)],
            target,
            inverse_frequency_weights(target),
        )
    if candidate.kind == "har":
        from sklearn.linear_model import LinearRegression
        from threadpoolctl import threadpool_limits

        estimator = LinearRegression()
        features = np.log(np.square(rows.loc[:, list(HAR_FEATURES)]) + EPSILON)
        target = np.log(np.square(rows["forward_vol_5"]) + EPSILON)
        with threadpool_limits(limits=1):
            estimator.fit(features, target)
        return estimator
    raise ValueError(f"Unknown challenger candidate kind: {candidate.kind}")


def ewma_scores(observations: pd.DataFrame, decay: float) -> pd.Series:
    if not 0 < decay < 1 or observations.empty or not observations["as_of"].is_monotonic_increasing:
        raise ValueError("EWMA requires chronological observations and a valid decay")
    returns = observations["log_return_1"].to_numpy(float)
    variance: npt.NDArray[np.float64] = np.empty(len(returns), dtype=np.float64)
    variance[0] = float(observations["trailing_vol_60"].iloc[0]) ** 2 / 252
    for index in range(1, len(returns)):
        variance[index] = decay * variance[index - 1] + (1 - decay) * returns[index] ** 2
    return pd.Series(np.sqrt(252 * variance), index=observations.index, name="raw_score")


def raw_scores(
    candidate: ChallengerCandidate,
    model: Any,
    observations: pd.DataFrame,
    validation_ids: tuple[str, ...],
    *,
    ewma: pd.Series | None = None,
) -> pd.Series:
    rows = observations.loc[list(validation_ids)]
    if candidate.kind == "classifier":
        classes = list(model.classes_)
        values = model.predict_proba(rows.loc[:, list(CHALLENGER_FEATURES)])[:, classes.index(1)]
    elif candidate.kind == "har":
        logged = model.predict(np.log(np.square(rows.loc[:, list(HAR_FEATURES)]) + EPSILON))
        values = np.sqrt(np.maximum(np.exp(logged) - EPSILON, 0.0))
    elif candidate.kind == "ewma" and ewma is not None:
        values = ewma.loc[list(validation_ids)].to_numpy(float)
    else:
        raise ValueError("EWMA raw scores require a precomputed causal series")
    result = pd.Series(values, index=list(validation_ids), name="raw_score", dtype="float64")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Challenger raw scores must be finite")
    return result


def fit_platt(raw: pd.Series, target: pd.Series) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    if not raw.index.equals(target.index) or set(target) != {0, 1}:
        raise ValueError("Platt calibration requires aligned scores with both classes")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
        ]
    )
    with threadpool_limits(limits=1):
        pipeline.fit(raw.to_numpy(float).reshape(-1, 1), target)
    return pipeline


def calibrated_scores(calibrator: Any, raw: pd.Series) -> pd.Series:
    classes = list(calibrator.classes_)
    values = calibrator.predict_proba(raw.to_numpy(float).reshape(-1, 1))[:, classes.index(1)]
    return pd.Series(values, index=raw.index, name="calibrated_probability")


def ensemble_predictions(
    probability: pd.Series,
    trailing_volatility: pd.Series,
    threshold: float,
    weight: float,
) -> pd.DataFrame:
    if not probability.index.equals(trailing_volatility.index) or weight not in ENSEMBLE_WEIGHTS:
        raise ValueError("Ensemble inputs or fixed model weight are invalid")
    persistence = (trailing_volatility > threshold).astype("float64")
    score = weight * probability + (1 - weight) * persistence
    return pd.DataFrame(
        {
            "predicted_class": (score > 0.5).astype("int64"),
            "score_class_1": score,
            "score_kind": "calibrated_model_plus_persistence",
        },
        index=probability.index,
    )

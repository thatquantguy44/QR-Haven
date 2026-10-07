"""Causal forecasts and fixed variants for post-V4 exploratory comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from qr_haven.ml.volatility.contracts import FEATURES, WalkForwardFold
from qr_haven.ml.volatility.models import (
    VolatilityCandidate,
    fit_estimator,
    inverse_frequency_weights,
    make_estimator,
)

CONTROL_ID = "weight_1_cutoff_0.5"
REGRESSION_FEATURES = ("trailing_vol_5", "trailing_vol_20", "trailing_vol_60")


@dataclass(frozen=True)
class ExperimentVariant:
    variant_id: str
    experiment: str
    kind: str
    weight_power: float = 1.0
    cutoff: float = 0.5
    window_years: int | None = None
    decay: float | None = None


def experiment_variants() -> tuple[ExperimentVariant, ...]:
    return (
        *(
            ExperimentVariant(
                f"weight_{power:g}_cutoff_{cutoff:g}",
                "weights_cutoffs",
                "classifier",
                weight_power=power,
                cutoff=cutoff,
            )
            for power in (0.0, 0.5, 1.0)
            for cutoff in (0.5, 0.6, 0.7)
        ),
        *(
            ExperimentVariant(f"ewma_{decay:g}", "simple_forecasts", "ewma", decay=decay)
            for decay in (0.90, 0.94, 0.97)
        ),
        ExperimentVariant("linear_vol_5_20_60", "simple_forecasts", "linear"),
        *(
            ExperimentVariant(
                f"rolling_{years}y", "rolling_windows", "classifier", window_years=years
            )
            for years in (3, 5)
        ),
    )


def class_weights(target: pd.Series, power: float) -> pd.Series:
    """Temper inverse-frequency weights while preserving mean sample weight one."""
    if not np.isfinite(power) or not 0 <= power <= 1:
        raise ValueError("Class-weight power must be between zero and one")
    counts = target.value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("Experiment training requires both classes")
    if power == 1:
        return inverse_frequency_weights(target)
    weights = target.map({key: float(count) ** (-power) for key, count in counts.items()})
    return (weights / weights.mean()).astype("float64").rename("sample_weight")


def training_ids(
    observations: pd.DataFrame, fold: WalkForwardFold, years: int | None
) -> tuple[str, ...]:
    """Restrict an already-purged fold, without recomputing its truth threshold."""
    boundary = pd.Timestamp(observations.loc[fold.validation_ids[0], "as_of"])
    rows = observations.loc[list(fold.training_ids)]
    if years is not None:
        if years <= 0:
            raise ValueError("Rolling window years must be positive")
        rows = rows.loc[rows["as_of"] >= boundary - pd.DateOffset(years=years)]
    if rows.empty or not (rows["label_end"] < boundary).all():
        raise ValueError("Experiment training window is empty or crosses validation")
    return tuple(rows.index.astype(str))


def ewma_forecast(observations: pd.DataFrame, decay: float) -> pd.Series:
    """Causal variance recursion, initialized from the first origin's trailing 60 returns."""
    if not np.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("EWMA decay must be between zero and one")
    if observations.empty or not observations["as_of"].is_monotonic_increasing:
        raise ValueError("EWMA observations must be nonempty and chronological")
    returns = observations["log_return_1"].to_numpy(dtype=float)
    initial = float(observations["trailing_vol_60"].iloc[0])
    if not np.isfinite(returns).all() or not np.isfinite(initial) or initial < 0:
        raise ValueError("EWMA inputs must be finite with a nonnegative initial volatility")
    variance: npt.NDArray[np.float64] = np.empty(len(returns), dtype=np.float64)
    variance[0] = initial**2 / 252
    for index in range(1, len(returns)):
        variance[index] = decay * variance[index - 1] + (1 - decay) * returns[index] ** 2
    return pd.Series(np.sqrt(252 * variance), index=observations.index, name="forecast_vol")


def classifier_scores(
    candidate: VolatilityCandidate,
    observations: pd.DataFrame,
    fit_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    threshold: float,
    power: float,
) -> tuple[pd.Series, dict[str, Any]]:
    target = (observations.loc[list(fit_ids), "forward_vol_5"] > threshold).astype("int64")
    weights = class_weights(target, power)
    estimator = fit_estimator(
        make_estimator(candidate),
        observations.loc[list(fit_ids), list(FEATURES)],
        target,
        weights,
    )
    scores = estimator.predict_proba(observations.loc[list(validation_ids), list(FEATURES)])
    result = pd.Series(scores[:, list(estimator.classes_).index(1)], index=list(validation_ids))
    return result, {
        "normal_training_rows": int((target == 0).sum()),
        "high_training_rows": int((target == 1).sum()),
        "normal_weight": float(weights[target == 0].iloc[0]),
        "high_weight": float(weights[target == 1].iloc[0]),
    }


def linear_forecast(
    observations: pd.DataFrame, fit_ids: tuple[str, ...], validation_ids: tuple[str, ...]
) -> tuple[pd.Series, dict[str, Any]]:
    from sklearn.linear_model import LinearRegression
    from threadpoolctl import threadpool_limits

    estimator = LinearRegression()
    with threadpool_limits(limits=1):
        estimator.fit(
            observations.loc[list(fit_ids), list(REGRESSION_FEATURES)],
            observations.loc[list(fit_ids), "forward_vol_5"],
        )
        forecasts = estimator.predict(
            observations.loc[list(validation_ids), list(REGRESSION_FEATURES)]
        )
    return pd.Series(np.maximum(forecasts, 0), index=list(validation_ids)), {
        "intercept": float(estimator.intercept_),
        "coefficients": dict(zip(REGRESSION_FEATURES, estimator.coef_.tolist(), strict=True)),
    }


def scored_predictions(scores: pd.Series, cutoff: float, kind: str) -> pd.DataFrame:
    if scores.empty or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Experiment scores must be nonempty and finite")
    return pd.DataFrame(
        {
            "predicted_class": (scores > cutoff).astype("int64"),
            "score_class_1": scores,
            "score_kind": kind,
        }
    )

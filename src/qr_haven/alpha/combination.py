"""Cross-sectional alpha score utilities: z-scoring, winsorization, and combination."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def cross_sectional_zscore(scores: pd.Series) -> pd.Series:
    """Standardize scores across the cross-section (zero mean, unit variance).

    Returns NaN for the whole series if the cross-sectional standard deviation
    is zero (all values identical).  NaN inputs are excluded from the mean/std
    calculation and preserved as NaN in the output.

    Parameters
    ----------
    scores:
        Raw factor values indexed by asset.
    """

    mean = scores.mean(skipna=True)
    std = scores.std(skipna=True)
    if pd.isna(std) or std == 0.0:
        return pd.Series(float("nan"), index=scores.index, name=scores.name)
    return ((scores - mean) / std).rename(scores.name)


def winsorize(scores: pd.Series, z_limit: float = 3.0) -> pd.Series:
    """Clip extreme z-scores to ``±z_limit``.

    Applies after :func:`cross_sectional_zscore` to limit the influence of
    extreme outliers on the composite signal.

    Parameters
    ----------
    scores:
        Z-scored factor values indexed by asset.
    z_limit:
        Symmetric clip boundary. Must be positive.
    """

    if z_limit <= 0.0:
        raise ValueError("z_limit must be positive.")
    return scores.clip(lower=-z_limit, upper=z_limit)


def combine_scores(
    scores: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    """Weighted combination of factor scores into a single composite alpha.

    NaN factor values are treated as neutral (zero) so an asset with a missing
    factor is not excluded from the composite — its score is simply diluted by
    the absent factor's weight.  Weights are normalized by their absolute sum
    so the composite remains on a z-score scale.

    Parameters
    ----------
    scores:
        Mapping of factor name → asset scores (indexed by asset symbol).
        Typically already z-scored and winsorized.
    weights:
        Signed importance of each factor. Positive means positive relationship
        between factor score and expected return.
    """

    if not scores:
        return pd.Series(dtype=float, name="alpha_score")

    frame = pd.DataFrame({k: v for k, v in scores.items()}).fillna(0.0)
    w = pd.Series({k: weights.get(k, 0.0) for k in frame.columns})
    total_abs_weight = float(w.abs().sum())
    if total_abs_weight == 0.0:
        return pd.Series(0.0, index=frame.index, dtype=float, name="alpha_score")

    composite = frame.multiply(w, axis="columns").sum(axis=1) / total_abs_weight
    return composite.rename("alpha_score")


def information_coefficient(predicted: pd.Series, realized: pd.Series) -> float:
    """Spearman rank correlation (IC) between predicted scores and realized returns.

    A standard measure of factor predictive power.  Returns NaN when fewer than
    two pairwise observations are available after dropping NaN values.

    Parameters
    ----------
    predicted:
        Factor scores (or model predictions) at the start of the period.
    realized:
        Forward returns over the holding period.
    """

    paired = pd.concat([predicted, realized], axis=1).dropna()
    if len(paired) < 2:
        return float("nan")
    rank_x = paired.iloc[:, 0].rank()
    rank_y = paired.iloc[:, 1].rank()
    return float(rank_x.corr(rank_y))


def pivot_feature_store(
    feature_frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert a feature-store long-format frame to a wide per-asset cross-section.

    Slices ``feature_frame`` at ``timestamp`` and pivots to a DataFrame with
    assets as rows and feature names as columns.  Typically called before
    passing features into an :class:`~qr_haven.alpha.models.CompositeAlphaModel`
    or :class:`~qr_haven.alpha.models.RankAlphaModel`.

    Parameters
    ----------
    feature_frame:
        Long-format output of :func:`~qr_haven.features.compute_features`.
    timestamp:
        The exact timestamp to slice.
    feature_names:
        Optional subset of feature names to include.  If None, all features at
        the timestamp are included.
    """

    at = feature_frame[feature_frame["timestamp"] == timestamp].copy()
    if feature_names is not None:
        at = at[at["feature_name"].isin(feature_names)]
    if at.empty:
        return pd.DataFrame()
    pivoted = at.pivot_table(
        index="symbol",
        columns="feature_name",
        values="value",
        aggfunc="first",
    )
    pivoted.columns.name = None
    return pivoted

"""Alpha model implementations satisfying the AlphaModel protocol.

Both models accept a wide cross-sectional feature matrix (assets × features)
and return a pd.Series of scores indexed by asset.  Use
:func:`~qr_haven.alpha.combination.pivot_feature_store` to convert the
feature-store long format to this shape before calling ``predict``.
"""

from __future__ import annotations

import pandas as pd

from qr_haven.alpha.combination import (
    combine_scores,
    cross_sectional_zscore,
    information_coefficient,
    winsorize,
)


class CompositeAlphaModel:
    """Weighted composite of cross-sectionally z-scored factor signals.

    Satisfies the :class:`~qr_haven.interfaces.AlphaModel` protocol.

    Each factor column is standardized (z-score) and winsorized at ±``winsorize_z``
    before being combined with the supplied weights.

    After calling :meth:`fit`, the model optionally replaces the static
    ``feature_weights`` with information-coefficient (IC) based weights derived
    from the training data.  The IC for a factor is its Spearman rank correlation
    with realized returns in the training sample.

    Parameters
    ----------
    name:
        Human-readable model name.
    feature_weights:
        Mapping of factor/feature name → signed weight.  Positive weights mean
        higher factor values predict higher forward returns.  These weights are
        used when the model has not been fitted, or as a fallback.
    winsorize_z:
        Symmetric z-score clip applied to each factor before combination.
    use_ic_weights:
        If True, :meth:`fit` overwrites the working weights with IC-derived
        values.  If False, :meth:`fit` is a no-op and fixed weights are always
        used.
    """

    def __init__(
        self,
        name: str,
        feature_weights: dict[str, float],
        winsorize_z: float = 3.0,
        use_ic_weights: bool = False,
    ) -> None:
        if not feature_weights:
            raise ValueError("feature_weights cannot be empty.")
        if winsorize_z <= 0.0:
            raise ValueError("winsorize_z must be positive.")
        self.name = name
        self.feature_weights: dict[str, float] = dict(feature_weights)
        self.winsorize_z = winsorize_z
        self.use_ic_weights = use_ic_weights
        self._ic_weights: dict[str, float] | None = None

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Calibrate IC-based weights from a historical cross-section.

        Parameters
        ----------
        features:
            Wide DataFrame (assets × features) for a historical period.
        targets:
            Realized forward returns indexed by asset.
        """

        if not self.use_ic_weights:
            return
        ic_weights: dict[str, float] = {}
        for col in features.columns:
            if col in self.feature_weights:
                ic = information_coefficient(features[col], targets)
                ic_weights[col] = 0.0 if pd.isna(ic) else ic
        self._ic_weights = ic_weights

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return composite alpha scores for one cross-section.

        Parameters
        ----------
        features:
            Wide DataFrame (assets × feature names) for a single date.
            Missing factor columns are skipped.
        """

        active_weights = (
            self._ic_weights
            if (self.use_ic_weights and self._ic_weights is not None)
            else self.feature_weights
        )
        factor_scores: dict[str, pd.Series] = {}
        for col in features.columns:
            w = active_weights.get(col, 0.0)
            if w == 0.0:
                continue
            z = cross_sectional_zscore(features[col])
            factor_scores[col] = winsorize(z, self.winsorize_z)
        return combine_scores(factor_scores, active_weights)

    @property
    def fitted(self) -> bool:
        """True after :meth:`fit` has been called with ``use_ic_weights=True``."""

        return self._ic_weights is not None


class RankAlphaModel:
    """Cross-sectional rank-based composite alpha model.

    Satisfies the :class:`~qr_haven.interfaces.AlphaModel` protocol.

    Each factor is ranked within the cross-section and normalized to the
    interval ``[-1, 1]``.  This approach is robust to outliers and monotone
    transformations of the raw factor values.

    Parameters
    ----------
    name:
        Human-readable model name.
    feature_weights:
        Mapping of factor name → signed weight.
    """

    def __init__(
        self,
        name: str,
        feature_weights: dict[str, float],
    ) -> None:
        if not feature_weights:
            raise ValueError("feature_weights cannot be empty.")
        self.name = name
        self.feature_weights: dict[str, float] = dict(feature_weights)

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """No-op — rank model is parameter-free."""

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return rank-based composite alpha scores for one cross-section.

        Parameters
        ----------
        features:
            Wide DataFrame (assets × feature names) for a single date.
        """

        factor_scores: dict[str, pd.Series] = {}
        for col in features.columns:
            w = self.feature_weights.get(col, 0.0)
            if w == 0.0:
                continue
            ranked = features[col].rank(method="average", na_option="keep")
            n_valid = ranked.notna().sum()
            if n_valid < 2:
                factor_scores[col] = pd.Series(float("nan"), index=features.index)
                continue
            min_r = ranked.min()
            max_r = ranked.max()
            if max_r == min_r:
                factor_scores[col] = pd.Series(0.0, index=features.index)
            else:
                # normalize to [-1, 1]
                normalized = 2.0 * (ranked - min_r) / (max_r - min_r) - 1.0
                factor_scores[col] = normalized
        return combine_scores(factor_scores, self.feature_weights)

"""Short squeeze risk scoring model.

Combines securities-lending-derived signals — borrow rate, short interest
ratio, days-to-cover, borrow utilisation — with recent price momentum to
produce a composite squeeze risk score for each security in a cross-section.

A high score flags elevated conditions that historically coincide with squeeze
episodes, giving risk managers a ranked watch-list to monitor.  It does not
forecast a squeeze with certainty; it surfaces names where the prerequisite
conditions (crowded shorts, illiquid borrow, upward price pressure) are
simultaneously elevated.

Usage
-----
>>> model = ShortSqueezeModel()
>>> result = model.score(universe_df)
>>> result.squeeze_score    # 0–1 per security
>>> result.squeeze_tier     # "LOW" / "MODERATE" / "HIGH"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_REQUIRED_COLS: frozenset[str] = frozenset(
    ["short_interest_ratio", "days_to_cover", "borrow_rate"]
)
_OPTIONAL_COLS: frozenset[str] = frozenset(
    ["borrow_utilization", "price_return_20d"]
)

_DEFAULT_WEIGHTS: dict[str, float] = {
    "short_interest_ratio": 0.25,   # higher % float short = more squeeze risk
    "days_to_cover": 0.25,          # longer to unwind = more risk
    "borrow_rate": 0.20,            # expensive borrow = crowded short
    "borrow_utilization": 0.15,     # near supply ceiling = tighter squeeze
    "price_return_20d": 0.15,       # shorts already losing = forced covering
}


@dataclass
class SqueezeScores:
    """Output of :meth:`ShortSqueezeModel.score`."""

    squeeze_score: pd.Series
    """Composite squeeze risk score in [0, 1]. Higher = greater squeeze risk."""

    squeeze_tier: pd.Series
    """String tier label: ``'LOW'``, ``'MODERATE'``, or ``'HIGH'``."""

    component_zscores: dict[str, pd.Series] = field(default_factory=dict)
    """Winsorised z-score for each factor that contributed to the composite."""


class ShortSqueezeModel:
    """Cross-sectional short squeeze risk scorer.

    Parameters
    ----------
    weights:
        Per-factor contribution weights.  Must cover all three required factors
        (``short_interest_ratio``, ``days_to_cover``, ``borrow_rate``).
        Optional factors (``borrow_utilization``, ``price_return_20d``) are
        included only when present in the input DataFrame; weights are
        renormalised over whichever factors are available.
    tier_thresholds:
        ``(low_cutoff, high_cutoff)`` in (0, 1).  Scores below *low_cutoff*
        are labelled ``'LOW'``; scores at or above *high_cutoff* are
        ``'HIGH'``; the rest are ``'MODERATE'``.
    winsorize_z:
        Clip per-factor z-scores at ±*winsorize_z* before combining.
    min_universe:
        Minimum number of securities needed for meaningful cross-sectional
        z-scores.  :meth:`score` raises ``ValueError`` if the DataFrame has
        fewer rows.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        tier_thresholds: tuple[float, float] = (0.33, 0.67),
        winsorize_z: float = 3.0,
        min_universe: int = 5,
    ) -> None:
        raw = weights if weights is not None else _DEFAULT_WEIGHTS.copy()
        missing = _REQUIRED_COLS - raw.keys()
        if missing:
            raise ValueError(
                f"weights must include required factors: {sorted(missing)}"
            )
        if any(v < 0.0 for v in raw.values()):
            raise ValueError("All factor weights must be non-negative.")
        low, high = tier_thresholds
        if not (0.0 < low < high < 1.0):
            raise ValueError("tier_thresholds must satisfy 0 < low < high < 1.")
        if winsorize_z <= 0.0:
            raise ValueError("winsorize_z must be positive.")
        if min_universe < 2:
            raise ValueError("min_universe must be at least 2.")

        self.weights = raw
        self.tier_thresholds = tier_thresholds
        self.winsorize_z = winsorize_z
        self.min_universe = min_universe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> SqueezeScores:
        """Score squeeze risk for a cross-section of securities.

        Parameters
        ----------
        df:
            DataFrame with one row per security.  Required columns:
            ``short_interest_ratio``, ``days_to_cover``, ``borrow_rate``.
            Optional columns: ``borrow_utilization``, ``price_return_20d``.

        Returns
        -------
        :class:`SqueezeScores` with composite score, tier labels, and
        per-factor z-scores.
        """
        df = pd.DataFrame(df)
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if len(df) < self.min_universe:
            raise ValueError(
                f"Universe has {len(df)} securities; "
                f"need at least min_universe={self.min_universe}."
            )

        active = [
            f for f in self.weights
            if f in df.columns and self.weights[f] > 0.0
        ]
        total_w = sum(self.weights[f] for f in active)

        component_zscores: dict[str, pd.Series] = {}
        composite = pd.Series(0.0, index=df.index)

        for factor in active:
            z = self._zscore(df[factor].astype(float))
            component_zscores[factor] = z
            composite += z * (self.weights[factor] / total_w)

        raw_score = pd.Series(
            self._sigmoid(composite.values), index=df.index, name="squeeze_score"
        )
        tier = raw_score.map(self._assign_tier)
        tier.name = "squeeze_tier"

        return SqueezeScores(
            squeeze_score=raw_score,
            squeeze_tier=tier,
            component_zscores=component_zscores,
        )

    def score_to_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return input DataFrame augmented with squeeze score columns.

        Adds ``squeeze_score``, ``squeeze_tier``, and ``{factor}_z`` columns.
        """
        result = self.score(df)
        out = df.copy()
        out["squeeze_score"] = result.squeeze_score
        out["squeeze_tier"] = result.squeeze_tier
        for factor, z in result.component_zscores.items():
            out[f"{factor}_z"] = z
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zscore(self, series: pd.Series) -> pd.Series:
        mu = series.mean()
        sigma = series.std()
        if sigma < 1e-10:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return ((series - mu) / sigma).clip(-self.winsorize_z, self.winsorize_z)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def _assign_tier(self, score: float) -> str:
        low, high = self.tier_thresholds
        if score < low:
            return "LOW"
        if score < high:
            return "MODERATE"
        return "HIGH"

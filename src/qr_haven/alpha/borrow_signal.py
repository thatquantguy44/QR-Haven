"""Borrow rate alpha signal.

High borrow rates are a proxy for crowded short positioning.  Two
empirically grounded interpretations exist:

**momentum** (Cohen, Diether & Malloy 2007; Saffi & Sigurdsson 2011):
    Follow the shorts.  Securities expensive to borrow underperform because
    the short sellers are informed.  High borrow → negative alpha score.

**contrarian**:
    Anticipate short cover.  When borrow becomes extremely expensive, the
    short base is stretched; any positive catalyst triggers forced covering
    and a price spike.  High borrow → positive alpha score.

Additionally, the *direction of change* in borrow rates carries incremental
information beyond the level: a rapidly rising borrow rate signals increasing
short-seller conviction (bearish), while a falling rate signals shorts
closing out (potentially bullish in the contrarian frame).

Usage
-----
>>> sig = BorrowRateAlphaSignal(direction="momentum")
>>> scores = sig.compute(cross_section_df)          # cross-section
>>> panel_scores = sig.compute_panel(rate_panel_df) # time × ticker panel
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


class BorrowRateAlphaSignal:
    """Cross-sectional alpha signal derived from securities lending borrow rates.

    Parameters
    ----------
    direction:
        ``'momentum'`` — high borrow rate implies negative alpha (follow shorts).
        ``'contrarian'`` — high borrow rate implies positive alpha (anticipate cover).
    include_utilization:
        When ``True`` and ``borrow_utilization`` is present in the input,
        incorporate utilisation as a secondary component.  Utilisation near
        100 % signals near-exhausted borrow supply — a leading indicator of
        rate spikes and forced cover.
    utilization_weight:
        Fractional weight on utilisation in [0, 1).  Rate receives the
        remaining ``1 - utilization_weight``.
    winsorize_z:
        Clip per-factor z-scores at ±*winsorize_z* before combining.
    """

    def __init__(
        self,
        direction: Literal["momentum", "contrarian"] = "momentum",
        include_utilization: bool = True,
        utilization_weight: float = 0.30,
        winsorize_z: float = 3.0,
    ) -> None:
        if direction not in ("momentum", "contrarian"):
            raise ValueError("direction must be 'momentum' or 'contrarian'.")
        if not (0.0 <= utilization_weight < 1.0):
            raise ValueError("utilization_weight must be in [0, 1).")
        if winsorize_z <= 0.0:
            raise ValueError("winsorize_z must be positive.")
        self.direction = direction
        self.include_utilization = include_utilization
        self.utilization_weight = utilization_weight
        self.winsorize_z = winsorize_z

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute cross-sectional borrow rate alpha scores for a single date.

        Parameters
        ----------
        df:
            DataFrame with one row per security.  Required: ``borrow_rate``
            (annualised, e.g. 0.05 = 5 %).  Optional: ``borrow_utilization``
            (fraction in [0, 1]).

        Returns
        -------
        Series of alpha scores indexed like *df*.  Higher = more positive
        expected alpha.
        """
        df = pd.DataFrame(df)
        if "borrow_rate" not in df.columns:
            raise ValueError("df must contain 'borrow_rate' column.")

        rate_z = self._zscore(df["borrow_rate"].astype(float))

        has_util = (
            self.include_utilization and "borrow_utilization" in df.columns
        )
        if has_util:
            util_z = self._zscore(df["borrow_utilization"].astype(float))
            rate_w = 1.0 - self.utilization_weight
            composite_z = rate_w * rate_z + self.utilization_weight * util_z
        else:
            composite_z = rate_z

        # momentum: flip sign so high borrow → negative alpha
        sign = -1.0 if self.direction == "momentum" else 1.0
        return pd.Series(
            sign * composite_z, index=df.index, name="borrow_alpha"
        )

    def compute_panel(
        self,
        borrow_rate_panel: pd.DataFrame,
        change_window: int = 20,
        include_level: bool = True,
        include_change: bool = True,
    ) -> pd.DataFrame:
        """Compute borrow rate signals from a time × ticker panel.

        Parameters
        ----------
        borrow_rate_panel:
            DataFrame with dates as index, tickers as columns, values as
            annualised borrow rates.  NaNs are handled gracefully.
        change_window:
            Number of periods over which to compute the rate-change signal.
        include_level:
            Include the cross-sectional level signal (equivalent to
            :meth:`compute` applied row-wise).
        include_change:
            Include the rate-change signal.  A rising borrow rate always
            predicts further short pressure (bearish), so the change signal
            is negated regardless of ``direction``.

        Returns
        -------
        DataFrame of alpha scores with the same shape as *borrow_rate_panel*.
        """
        if not (include_level or include_change):
            raise ValueError(
                "At least one of include_level or include_change must be True."
            )

        sign = -1.0 if self.direction == "momentum" else 1.0
        parts: list[pd.DataFrame] = []

        if include_level:
            level = borrow_rate_panel.apply(self._row_zscore, axis=1)
            parts.append(sign * level)

        if include_change:
            delta = borrow_rate_panel.diff(change_window)
            change = delta.apply(self._row_zscore, axis=1)
            # rising borrow is bearish in both modes
            parts.append(-1.0 * change)

        result: pd.DataFrame = parts[0] if len(parts) == 1 else sum(parts) / len(parts)  # type: ignore[arg-type]
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zscore(self, series: pd.Series) -> pd.Series:
        """Cross-sectional z-score over non-NaN values, winsorised."""
        mu = series.mean()
        sigma = series.std()
        if sigma < 1e-10:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return ((series - mu) / sigma).clip(-self.winsorize_z, self.winsorize_z)

    def _row_zscore(self, row: pd.Series) -> pd.Series:
        """Apply cross-sectional z-score to a single row (one date)."""
        valid = row.dropna()
        if len(valid) < 2:
            return pd.Series(0.0, index=row.index)
        mu = valid.mean()
        sigma = valid.std()
        if sigma < 1e-10:
            return pd.Series(0.0, index=row.index)
        z = (row - mu) / sigma
        return z.clip(-self.winsorize_z, self.winsorize_z).fillna(0.0)

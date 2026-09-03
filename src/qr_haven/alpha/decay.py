"""Signal decay and IC stability diagnostics for alpha research.

Answers two standing research questions from the alpha factory backlog:

* **Signal decay** — how does an alpha's predictive power (IC and rank IC) change
  as the holding period grows?  A signal with a short half-life needs frequent
  rebalancing, which pushes turnover and market-impact cost higher.
* **IC stability** — is the signal's predictive power consistent over time, or
  driven by a handful of lucky periods?  A high mean IC with a low IC information
  ratio is a much weaker basis for sizing a strategy than a modest but stable one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qr_haven.alpha.combination import information_coefficient


@dataclass(frozen=True)
class SignalDecayProfile:
    """IC and rank IC of an alpha signal across a set of holding-period horizons."""

    horizons: tuple[int, ...]
    """Holding-period horizons evaluated, in periods (e.g. trading days)."""

    ic_by_horizon: pd.Series
    """Mean Pearson IC (predicted score vs. realized forward return) per horizon."""

    rank_ic_by_horizon: pd.Series
    """Mean Spearman rank IC per horizon."""

    observations_by_horizon: pd.Series
    """Number of cross-sections (dates) contributing to each horizon's mean."""

    def half_life(self) -> float | None:
        """Return the interpolated horizon at which rank IC first decays to half.

        Uses the shortest evaluated horizon's rank IC as the reference level. Returns
        ``None`` when there are fewer than two horizons, the reference IC is zero or
        NaN, or rank IC never decays to half of the reference within the evaluated
        horizons (the signal's half-life exceeds what was tested).
        """

        if len(self.horizons) < 2:
            return None
        reference = float(self.rank_ic_by_horizon.iloc[0])
        if not np.isfinite(reference) or reference == 0.0:
            return None
        target = abs(reference) / 2.0
        magnitudes = self.rank_ic_by_horizon.abs()

        for i in range(1, len(self.horizons)):
            previous_h, current_h = self.horizons[i - 1], self.horizons[i]
            previous_mag = float(magnitudes.iloc[i - 1])
            current_mag = float(magnitudes.iloc[i])
            if not (np.isfinite(previous_mag) and np.isfinite(current_mag)):
                continue
            if previous_mag >= target >= current_mag and previous_mag != current_mag:
                # Linear interpolation between the two bracketing horizons.
                frac = (previous_mag - target) / (previous_mag - current_mag)
                return float(previous_h + frac * (current_h - previous_h))
        return None


@dataclass(frozen=True)
class ICStability:
    """Time-series stability diagnostics for a single-horizon IC series."""

    ic_series: pd.Series
    """Per-date rank IC observations, indexed by date."""

    mean_ic: float
    """Mean rank IC across all dates."""

    ic_volatility: float
    """Standard deviation of the rank IC series."""

    ic_information_ratio: float
    """``mean_ic / ic_volatility``. Zero when volatility is zero."""

    hit_rate: float
    """Fraction of dates whose IC sign matches the sign of ``mean_ic``."""


def calculate_forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compounded forward return over ``horizon`` periods, aligned to the signal date.

    The value at date ``t`` is the compounded return realized over periods
    ``t + 1`` through ``t + horizon`` — i.e. what an alpha score observed as of ``t``
    would be forecasting. The current period's own return (already known at ``t``) is
    excluded to avoid look-ahead. The final ``horizon`` rows are ``NaN`` because no
    full forward window exists yet.

    Parameters
    ----------
    returns:
        Wide DataFrame of asset returns (DatetimeIndex x assets).
    horizon:
        Number of forward periods to compound over. Must be at least 1.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    log_returns = np.log1p(returns.astype(float))
    cumulative_log = log_returns.rolling(window=horizon).sum()
    forward = np.expm1(cumulative_log.shift(-horizon))
    return forward


def calculate_ic_by_horizon(
    alpha_scores: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10, 21, 42, 63),
) -> SignalDecayProfile:
    """Calculate mean cross-sectional IC and rank IC at each holding-period horizon.

    Parameters
    ----------
    alpha_scores:
        Panel of alpha scores (DatetimeIndex x assets), one cross-section per
        signal date.
    returns:
        Wide DataFrame of asset returns used to compute forward returns.
    horizons:
        Holding-period horizons to evaluate, in periods. Must be non-empty and
        strictly increasing.
    """

    if not horizons:
        raise ValueError("horizons must not be empty.")
    if list(horizons) != sorted(set(horizons)):
        raise ValueError("horizons must be strictly increasing with no duplicates.")

    ic_values: dict[int, float] = {}
    rank_ic_values: dict[int, float] = {}
    observation_counts: dict[int, int] = {}

    for horizon in horizons:
        forward = calculate_forward_returns(returns, horizon)
        common_dates = alpha_scores.index.intersection(forward.index)

        pearson_ics: list[float] = []
        rank_ics: list[float] = []
        for date in common_dates:
            predicted = alpha_scores.loc[date]
            realized = forward.loc[date]
            paired = pd.concat([predicted, realized], axis=1).dropna()
            if len(paired) < 2:
                continue
            pearson = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
            if np.isfinite(pearson):
                pearson_ics.append(pearson)
            rank_ic = information_coefficient(predicted, realized)
            if np.isfinite(rank_ic):
                rank_ics.append(rank_ic)

        ic_values[horizon] = float(np.mean(pearson_ics)) if pearson_ics else float("nan")
        rank_ic_values[horizon] = float(np.mean(rank_ics)) if rank_ics else float("nan")
        observation_counts[horizon] = len(rank_ics)

    horizon_tuple = tuple(horizons)
    return SignalDecayProfile(
        horizons=horizon_tuple,
        ic_by_horizon=pd.Series(ic_values, name="ic").reindex(horizon_tuple),
        rank_ic_by_horizon=pd.Series(rank_ic_values, name="rank_ic").reindex(horizon_tuple),
        observations_by_horizon=pd.Series(
            observation_counts, name="observations", dtype=int
        ).reindex(horizon_tuple),
    )


def calculate_rolling_ic(
    alpha_scores: pd.DataFrame,
    returns: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """Return the per-date rank IC time series for a single holding-period horizon."""

    forward = calculate_forward_returns(returns, horizon)
    common_dates = alpha_scores.index.intersection(forward.index)
    values: dict[pd.Timestamp, float] = {}
    for date in common_dates:
        rank_ic = information_coefficient(alpha_scores.loc[date], forward.loc[date])
        if np.isfinite(rank_ic):
            values[date] = rank_ic
    return pd.Series(values, name="rank_ic").sort_index()


def calculate_ic_stability(
    alpha_scores: pd.DataFrame,
    returns: pd.DataFrame,
    horizon: int = 1,
) -> ICStability:
    """Summarize the stability of an alpha's rank IC over time at one horizon."""

    ic_series = calculate_rolling_ic(alpha_scores, returns, horizon)
    if ic_series.empty:
        return ICStability(
            ic_series=ic_series,
            mean_ic=float("nan"),
            ic_volatility=float("nan"),
            ic_information_ratio=float("nan"),
            hit_rate=float("nan"),
        )

    mean_ic = float(ic_series.mean())
    ic_volatility = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    ic_information_ratio = mean_ic / ic_volatility if ic_volatility > 0.0 else 0.0
    reference_sign = np.sign(mean_ic)
    if reference_sign != 0.0:
        hit_rate = float((np.sign(ic_series) == reference_sign).mean())
    else:
        hit_rate = float("nan")

    return ICStability(
        ic_series=ic_series,
        mean_ic=mean_ic,
        ic_volatility=ic_volatility,
        ic_information_ratio=ic_information_ratio,
        hit_rate=hit_rate,
    )

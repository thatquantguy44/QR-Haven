"""Tests for signal decay and IC stability diagnostics."""

import numpy as np
import pandas as pd
import pytest

from qr_haven.alpha import (
    ICStability,
    SignalDecayProfile,
    calculate_forward_returns,
    calculate_ic_by_horizon,
    calculate_ic_stability,
    calculate_rolling_ic,
)

ASSETS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
N_PERIODS = 150


def _iid_returns(n: int = N_PERIODS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0.0, 0.01, size=(n, len(ASSETS)))
    return pd.DataFrame(data, index=dates, columns=ASSETS)


def _perfect_one_day_alpha(returns: pd.DataFrame) -> pd.DataFrame:
    """A signal that exactly equals next period's return: a perfect 1-day predictor."""
    return returns.shift(-1)


def _random_alpha(returns: pd.DataFrame, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=returns.shape)
    return pd.DataFrame(data, index=returns.index, columns=returns.columns)


# ---------------------------------------------------------------------------
# calculate_forward_returns
# ---------------------------------------------------------------------------


def test_forward_returns_horizon_one_matches_next_period() -> None:
    returns = _iid_returns()
    forward = calculate_forward_returns(returns, horizon=1)
    pd.testing.assert_frame_equal(
        forward.iloc[:-1], returns.shift(-1).iloc[:-1], check_exact=False, atol=1e-10
    )


def test_forward_returns_tail_is_nan() -> None:
    returns = _iid_returns()
    forward = calculate_forward_returns(returns, horizon=5)
    assert forward.iloc[-5:].isna().all().all()


def test_forward_returns_compounds_correctly() -> None:
    """forward_t covers periods (t+1)..(t+horizon) — the return an alpha at t would forecast."""
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01, 0.03]}, index=pd.date_range("2024-01-01", periods=4)
    )
    forward = calculate_forward_returns(returns, horizon=2)
    expected_first = (1.02 * 0.99) - 1.0
    assert np.isclose(forward["A"].iloc[0], expected_first)


def test_forward_returns_rejects_non_positive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        calculate_forward_returns(_iid_returns(), horizon=0)


# ---------------------------------------------------------------------------
# calculate_ic_by_horizon — signal decay
# ---------------------------------------------------------------------------


def test_ic_by_horizon_returns_profile() -> None:
    returns = _iid_returns()
    alpha = _perfect_one_day_alpha(returns)
    profile = calculate_ic_by_horizon(alpha, returns, horizons=(1, 5, 10))
    assert isinstance(profile, SignalDecayProfile)
    assert profile.horizons == (1, 5, 10)


def test_perfect_one_day_predictor_has_near_unit_ic_at_horizon_one() -> None:
    returns = _iid_returns()
    alpha = _perfect_one_day_alpha(returns)
    profile = calculate_ic_by_horizon(alpha, returns, horizons=(1,))
    assert profile.rank_ic_by_horizon.loc[1] > 0.99
    assert profile.ic_by_horizon.loc[1] > 0.99


def test_ic_decays_as_horizon_grows_for_iid_returns() -> None:
    """A 1-day-ahead predictor should correlate less with longer, noisier forward returns."""
    returns = _iid_returns()
    alpha = _perfect_one_day_alpha(returns)
    profile = calculate_ic_by_horizon(alpha, returns, horizons=(1, 5, 10, 21))
    ic = profile.rank_ic_by_horizon
    assert ic.loc[1] > ic.loc[5] > ic.loc[10] > ic.loc[21]
    assert ic.loc[21] > 0.0


def test_ic_by_horizon_observation_counts_decrease_with_horizon() -> None:
    returns = _iid_returns()
    alpha = _perfect_one_day_alpha(returns)
    profile = calculate_ic_by_horizon(alpha, returns, horizons=(1, 21, 63))
    obs = profile.observations_by_horizon
    assert obs.loc[1] > obs.loc[21] > obs.loc[63]


def test_ic_by_horizon_rejects_empty_horizons() -> None:
    with pytest.raises(ValueError, match="horizons"):
        calculate_ic_by_horizon(_iid_returns(), _iid_returns(), horizons=())


def test_ic_by_horizon_rejects_unsorted_horizons() -> None:
    with pytest.raises(ValueError, match="horizons"):
        calculate_ic_by_horizon(_iid_returns(), _iid_returns(), horizons=(5, 1))


def test_random_alpha_has_low_ic() -> None:
    returns = _iid_returns()
    alpha = _random_alpha(returns)
    profile = calculate_ic_by_horizon(alpha, returns, horizons=(1, 21))
    assert abs(profile.rank_ic_by_horizon.loc[1]) < 0.2
    assert abs(profile.rank_ic_by_horizon.loc[21]) < 0.2


# ---------------------------------------------------------------------------
# half_life
# ---------------------------------------------------------------------------


def test_half_life_is_finite_for_decaying_signal() -> None:
    returns = _iid_returns()
    alpha = _perfect_one_day_alpha(returns)
    profile = calculate_ic_by_horizon(alpha, returns, horizons=(1, 5, 10, 21, 42, 63))
    half_life = profile.half_life()
    assert half_life is not None
    assert 1.0 < half_life < 63.0


def test_half_life_none_with_single_horizon() -> None:
    returns = _iid_returns()
    alpha = _perfect_one_day_alpha(returns)
    profile = calculate_ic_by_horizon(alpha, returns, horizons=(1,))
    assert profile.half_life() is None


def test_half_life_none_when_reference_ic_is_zero() -> None:
    profile = SignalDecayProfile(
        horizons=(1, 5),
        ic_by_horizon=pd.Series({1: 0.0, 5: 0.0}),
        rank_ic_by_horizon=pd.Series({1: 0.0, 5: 0.0}),
        observations_by_horizon=pd.Series({1: 10, 5: 10}),
    )
    assert profile.half_life() is None


# ---------------------------------------------------------------------------
# calculate_rolling_ic / calculate_ic_stability
# ---------------------------------------------------------------------------


def test_rolling_ic_returns_series_indexed_by_date() -> None:
    returns = _iid_returns()
    alpha = _perfect_one_day_alpha(returns)
    ic_series = calculate_rolling_ic(alpha, returns, horizon=1)
    assert isinstance(ic_series, pd.Series)
    assert len(ic_series) > 0
    assert ic_series.index.is_monotonic_increasing


def test_ic_stability_perfect_predictor_has_high_mean_and_hit_rate() -> None:
    returns = _iid_returns()
    alpha = _perfect_one_day_alpha(returns)
    stability = calculate_ic_stability(alpha, returns, horizon=1)
    assert isinstance(stability, ICStability)
    assert stability.mean_ic > 0.99
    assert stability.hit_rate == 1.0
    assert stability.ic_information_ratio > 0.0


def test_ic_stability_empty_series_returns_nan_fields() -> None:
    empty = pd.DataFrame(columns=ASSETS, dtype=float)
    stability = calculate_ic_stability(empty, empty, horizon=1)
    assert stability.ic_series.empty
    assert np.isnan(stability.mean_ic)
    assert np.isnan(stability.hit_rate)


def test_ic_stability_random_alpha_has_low_information_ratio() -> None:
    returns = _iid_returns(n=60)
    alpha = _random_alpha(returns)
    stability = calculate_ic_stability(alpha, returns, horizon=1)
    assert abs(stability.ic_information_ratio) < 2.0

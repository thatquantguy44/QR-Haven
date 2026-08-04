"""Tests for alpha combination utilities and model implementations."""

import math

import numpy as np
import pandas as pd
import pytest

from qr_haven.alpha import (
    CompositeAlphaModel,
    RankAlphaModel,
    combine_scores,
    cross_sectional_zscore,
    information_coefficient,
    pivot_feature_store,
    winsorize,
)
from qr_haven.features import compute_features, factor_feature_registry


# ---------------------------------------------------------------------------
# cross_sectional_zscore
# ---------------------------------------------------------------------------


def test_zscore_mean_zero_std_one() -> None:
    scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    z = cross_sectional_zscore(scores)
    assert math.isclose(z.mean(), 0.0, abs_tol=1e-10)
    assert math.isclose(z.std(), 1.0, rel_tol=1e-9)


def test_zscore_identical_values_returns_nan() -> None:
    scores = pd.Series({"A": 5.0, "B": 5.0, "C": 5.0})
    z = cross_sectional_zscore(scores)
    assert z.isna().all()


def test_zscore_preserves_nan_inputs() -> None:
    scores = pd.Series({"A": 1.0, "B": float("nan"), "C": 3.0})
    z = cross_sectional_zscore(scores)
    assert math.isnan(z["B"])
    assert not math.isnan(z["A"])


def test_zscore_two_asset_symmetry() -> None:
    scores = pd.Series({"A": 0.0, "B": 10.0})
    z = cross_sectional_zscore(scores)
    assert math.isclose(z["A"], -z["B"], rel_tol=1e-9)


# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------


def test_winsorize_clips_at_limit() -> None:
    scores = pd.Series({"A": -5.0, "B": 0.0, "C": 5.0})
    w = winsorize(scores, z_limit=3.0)
    assert w["A"] == -3.0
    assert w["B"] == 0.0
    assert w["C"] == 3.0


def test_winsorize_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="z_limit"):
        winsorize(pd.Series([1.0, 2.0]), z_limit=0.0)


def test_winsorize_does_not_change_values_within_limits() -> None:
    scores = pd.Series({"A": -2.0, "B": 1.5, "C": 2.9})
    w = winsorize(scores, z_limit=3.0)
    pd.testing.assert_series_equal(w, scores)


# ---------------------------------------------------------------------------
# combine_scores
# ---------------------------------------------------------------------------


def test_combine_scores_empty_returns_empty_series() -> None:
    result = combine_scores({}, {})
    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_combine_scores_single_factor() -> None:
    scores = {"mom": pd.Series({"A": 1.0, "B": -1.0, "C": 0.0})}
    result = combine_scores(scores, {"mom": 1.0})
    pd.testing.assert_series_equal(result, scores["mom"].rename("alpha_score"))


def test_combine_scores_equal_weights_average() -> None:
    f1 = pd.Series({"A": 2.0, "B": -2.0})
    f2 = pd.Series({"A": -2.0, "B": 2.0})
    result = combine_scores({"f1": f1, "f2": f2}, {"f1": 1.0, "f2": 1.0})
    assert math.isclose(result["A"], 0.0, abs_tol=1e-12)
    assert math.isclose(result["B"], 0.0, abs_tol=1e-12)


def test_combine_scores_missing_factor_weight_treated_as_zero() -> None:
    scores = {"mom": pd.Series({"A": 1.0}), "rev": pd.Series({"A": -1.0})}
    result = combine_scores(scores, {"mom": 1.0})  # rev weight = 0
    assert math.isclose(result["A"], 1.0, rel_tol=1e-9)


def test_combine_scores_all_zero_weights_returns_zeros() -> None:
    scores = {"f1": pd.Series({"A": 1.0, "B": 2.0})}
    result = combine_scores(scores, {"f1": 0.0})
    assert (result == 0.0).all()


def test_combine_scores_nan_treated_as_neutral() -> None:
    # NaN factor value → treated as 0 (neutral), not dropped
    f1 = pd.Series({"A": 1.0, "B": float("nan")})
    result = combine_scores({"f1": f1}, {"f1": 1.0})
    assert math.isclose(result["A"], 1.0, rel_tol=1e-9)
    assert math.isclose(result["B"], 0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# information_coefficient
# ---------------------------------------------------------------------------


def test_ic_perfect_rank_correlation_is_one() -> None:
    predicted = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    realized = pd.Series({"A": 0.01, "B": 0.05, "C": 0.10})
    ic = information_coefficient(predicted, realized)
    assert math.isclose(ic, 1.0, rel_tol=1e-9)


def test_ic_perfect_negative_rank_correlation_is_minus_one() -> None:
    predicted = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    realized = pd.Series({"A": 0.01, "B": 0.05, "C": 0.10})
    ic = information_coefficient(predicted, realized)
    assert math.isclose(ic, -1.0, rel_tol=1e-9)


def test_ic_fewer_than_two_observations_returns_nan() -> None:
    ic = information_coefficient(pd.Series({"A": 1.0}), pd.Series({"A": 0.05}))
    assert math.isnan(ic)


def test_ic_nan_values_dropped_pairwise() -> None:
    predicted = pd.Series({"A": 1.0, "B": float("nan"), "C": 3.0})
    realized = pd.Series({"A": 0.01, "B": 0.05, "C": 0.10})
    ic = information_coefficient(predicted, realized)
    assert not math.isnan(ic)


# ---------------------------------------------------------------------------
# pivot_feature_store
# ---------------------------------------------------------------------------


def _make_feature_store_frame() -> tuple[pd.DataFrame, pd.Timestamp]:
    n = 5
    ts = pd.date_range("2024-01-01", periods=n, tz="UTC")
    prices = []
    for i, t in enumerate(ts):
        for sym, base in [("AAPL", 150.0), ("MSFT", 400.0)]:
            prices.append({"timestamp": t, "symbol": sym, "feature_name": "price_momentum",
                           "feature_version": "1", "value": 0.1 * i * (1 if sym == "AAPL" else -1)})
            prices.append({"timestamp": t, "symbol": sym, "feature_name": "short_term_reversal",
                           "feature_version": "1", "value": -0.05 * i * (1 if sym == "AAPL" else -1)})
    return pd.DataFrame(prices), ts[-1]


def test_pivot_feature_store_returns_wide_dataframe() -> None:
    frame, ts = _make_feature_store_frame()
    wide = pivot_feature_store(frame, ts)
    assert "AAPL" in wide.index
    assert "MSFT" in wide.index
    assert "price_momentum" in wide.columns
    assert "short_term_reversal" in wide.columns


def test_pivot_feature_store_filters_feature_names() -> None:
    frame, ts = _make_feature_store_frame()
    wide = pivot_feature_store(frame, ts, feature_names=["price_momentum"])
    assert list(wide.columns) == ["price_momentum"]


def test_pivot_feature_store_unknown_timestamp_returns_empty() -> None:
    frame, _ = _make_feature_store_frame()
    future = pd.Timestamp("2099-01-01", tz="UTC")
    result = pivot_feature_store(frame, future)
    assert result.empty


# ---------------------------------------------------------------------------
# CompositeAlphaModel
# ---------------------------------------------------------------------------


def _cross_section() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_momentum": [0.5, -0.5, 1.0, -1.0, 0.0],
            "short_term_reversal": [-0.3, 0.3, -0.6, 0.6, 0.0],
        },
        index=["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"],
    )


def test_composite_model_predict_returns_series_indexed_by_asset() -> None:
    model = CompositeAlphaModel("test", {"price_momentum": 1.0})
    scores = model.predict(_cross_section())
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == {"AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"}


def test_composite_model_predict_zero_net_for_symmetric_cross_section() -> None:
    # Symmetric z-scores sum to zero
    model = CompositeAlphaModel("test", {"price_momentum": 1.0, "short_term_reversal": 1.0})
    scores = model.predict(_cross_section())
    assert math.isclose(scores.sum(), 0.0, abs_tol=1e-10)


def test_composite_model_rejects_empty_weights() -> None:
    with pytest.raises(ValueError, match="feature_weights"):
        CompositeAlphaModel("test", {})


def test_composite_model_rejects_non_positive_winsorize_z() -> None:
    with pytest.raises(ValueError, match="winsorize_z"):
        CompositeAlphaModel("test", {"f": 1.0}, winsorize_z=0.0)


def test_composite_model_negative_weight_inverts_signal() -> None:
    model_pos = CompositeAlphaModel("pos", {"price_momentum": 1.0})
    model_neg = CompositeAlphaModel("neg", {"price_momentum": -1.0})
    cs = _cross_section()
    pos_scores = model_pos.predict(cs)
    neg_scores = model_neg.predict(cs)
    pd.testing.assert_series_equal(pos_scores, (-neg_scores).rename("alpha_score"))


def test_composite_model_fit_sets_ic_weights() -> None:
    model = CompositeAlphaModel(
        "test", {"price_momentum": 1.0, "short_term_reversal": 1.0}, use_ic_weights=True
    )
    cs = _cross_section()
    targets = pd.Series(
        {"AAPL": 0.05, "MSFT": -0.05, "GOOGL": 0.10, "TSLA": -0.10, "AMZN": 0.01}
    )
    assert not model.fitted
    model.fit(cs, targets)
    assert model.fitted


def test_composite_model_fit_noop_when_use_ic_weights_false() -> None:
    model = CompositeAlphaModel("test", {"price_momentum": 1.0}, use_ic_weights=False)
    model.fit(_cross_section(), pd.Series({"AAPL": 0.01}))
    assert not model.fitted


def test_composite_model_name_attribute() -> None:
    model = CompositeAlphaModel("momentum_composite", {"price_momentum": 1.0})
    assert model.name == "momentum_composite"


# ---------------------------------------------------------------------------
# RankAlphaModel
# ---------------------------------------------------------------------------


def test_rank_model_predict_returns_series() -> None:
    model = RankAlphaModel("rank_test", {"price_momentum": 1.0})
    scores = model.predict(_cross_section())
    assert isinstance(scores, pd.Series)


def test_rank_model_scores_bounded() -> None:
    model = RankAlphaModel("rank_test", {"price_momentum": 1.0, "short_term_reversal": 1.0})
    scores = model.predict(_cross_section())
    assert (scores >= -1.0).all()
    assert (scores <= 1.0).all()


def test_rank_model_top_rank_gets_score_one() -> None:
    # Single factor: verify that the highest-ranked asset gets score +1
    cs = pd.DataFrame(
        {"price_momentum": [3.0, 1.0, 2.0]},
        index=["A", "B", "C"],
    )
    model = RankAlphaModel("rank_test", {"price_momentum": 1.0})
    scores = model.predict(cs)
    assert math.isclose(scores["A"], 1.0, rel_tol=1e-9)
    assert math.isclose(scores["B"], -1.0, rel_tol=1e-9)


def test_rank_model_fit_is_noop() -> None:
    model = RankAlphaModel("rank_test", {"price_momentum": 1.0})
    cs = _cross_section()
    targets = pd.Series({"AAPL": 0.05, "MSFT": -0.05})
    model.fit(cs, targets)  # must not raise


def test_rank_model_rejects_empty_weights() -> None:
    with pytest.raises(ValueError, match="feature_weights"):
        RankAlphaModel("test", {})


def test_rank_model_name_attribute() -> None:
    model = RankAlphaModel("my_rank_model", {"price_momentum": 1.0})
    assert model.name == "my_rank_model"


# ---------------------------------------------------------------------------
# End-to-end: factor features → pivot → alpha model
# ---------------------------------------------------------------------------


def test_end_to_end_factor_to_alpha_score() -> None:
    np.random.seed(0)
    n = 300  # need formation_window(252) + some buffer
    symbols = ["AAPL", "MSFT", "GOOGL"]
    ts = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")

    frames = []
    for sym in symbols:
        prices = 100.0 + np.random.randn(n).cumsum()
        prices = np.abs(prices) + 10.0
        idx = pd.MultiIndex.from_arrays([ts, [sym] * n], names=["timestamp", "symbol"])
        frames.append(pd.DataFrame({"adjusted_close": prices}, index=idx))
    prices_df = pd.concat(frames).sort_index()

    features = compute_features(prices_df, list(factor_feature_registry()))
    last_ts = ts[-1]
    wide = pivot_feature_store(features, last_ts)
    assert set(wide.index) == set(symbols)

    model = CompositeAlphaModel(
        "factor_composite",
        {"price_momentum": 0.6, "short_term_reversal": 0.2, "low_volatility_score": 0.2},
    )
    scores = model.predict(wide)
    assert set(scores.index) == set(symbols)
    assert scores.notna().any()

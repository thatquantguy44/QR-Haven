"""Tests for LSTMReturnForecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qr_haven.ml.lstm import (
    ForecasterFitResult,
    LSTMReturnForecaster,
    _Params,
    _lstm_backward,
    _lstm_forward,
    _sigmoid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASSETS = ["AAPL", "MSFT", "GOOG", "AMZN"]
N_FEATURES = 4
HIDDEN = 8
LOOKBACK = 10
N_DATES = 120


def _factor_panel(
    n_dates: int = N_DATES,
    n_features: int = N_FEATURES,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    data = rng.normal(0, 1, (n_dates, n_features))
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(data, index=dates, columns=cols)


def _returns(n_dates: int = N_DATES, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    return pd.Series(rng.normal(0, 0.01, n_dates), index=dates, name="ret")


def _make_sequences(
    n: int = 120,
    lookback: int = LOOKBACK,
    n_features: int = N_FEATURES,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N, lookback, n_features) X and (N,) y."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, lookback, n_features))
    y = rng.normal(0, 0.01, n)
    return X, y


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_zero_input(self):
        assert abs(_sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10

    def test_large_positive(self):
        assert abs(_sigmoid(np.array([100.0]))[0] - 1.0) < 1e-6

    def test_large_negative(self):
        assert abs(_sigmoid(np.array([-100.0]))[0]) < 1e-6

    def test_shape_preserved(self):
        x = np.ones((3, 4))
        assert _sigmoid(x).shape == (3, 4)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


class TestParams:
    def test_shapes(self):
        p = _Params(N_FEATURES, HIDDEN, seed=0)
        assert p.Wx.shape == (4 * HIDDEN, N_FEATURES)
        assert p.Wh.shape == (4 * HIDDEN, HIDDEN)
        assert p.b.shape == (4 * HIDDEN,)
        assert p.Wy.shape == (1, HIDDEN)
        assert p.by.shape == (1,)

    def test_forget_gate_bias_initialized_to_one(self):
        p = _Params(N_FEATURES, HIDDEN, seed=0)
        # Forget gate is second block: [HIDDEN : 2*HIDDEN]
        fb = p.b[HIDDEN : 2 * HIDDEN]
        np.testing.assert_allclose(fb, np.ones(HIDDEN))

    def test_no_nan(self):
        p = _Params(N_FEATURES, HIDDEN, seed=42)
        for name in ("Wx", "Wh", "b", "Wy", "by"):
            assert not np.any(np.isnan(getattr(p, name))), f"NaN in {name}"

    def test_different_seeds_differ(self):
        p1 = _Params(N_FEATURES, HIDDEN, seed=0)
        p2 = _Params(N_FEATURES, HIDDEN, seed=99)
        assert not np.allclose(p1.Wx, p2.Wx)


# ---------------------------------------------------------------------------
# LSTM forward pass
# ---------------------------------------------------------------------------


class TestLSTMForward:
    def test_output_shape(self):
        params = _Params(N_FEATURES, HIDDEN)
        X = np.random.default_rng(0).normal(0, 1, (LOOKBACK, N_FEATURES))
        h, caches = _lstm_forward(X, params)
        assert h.shape == (HIDDEN,)
        assert len(caches) == LOOKBACK

    def test_all_zero_input_stable(self):
        params = _Params(N_FEATURES, HIDDEN)
        X = np.zeros((LOOKBACK, N_FEATURES))
        h, _ = _lstm_forward(X, params)
        assert np.all(np.isfinite(h))

    def test_cache_keys(self):
        params = _Params(N_FEATURES, HIDDEN)
        X = np.zeros((LOOKBACK, N_FEATURES))
        _, caches = _lstm_forward(X, params)
        for key in ("x", "h_prev", "c_prev", "i", "f", "g", "o", "c", "h"):
            assert key in caches[0], f"Missing cache key: {key}"

    def test_gate_values_in_unit_interval(self):
        params = _Params(N_FEATURES, HIDDEN)
        X = np.random.default_rng(5).normal(0, 2, (LOOKBACK, N_FEATURES))
        _, caches = _lstm_forward(X, params)
        for cache in caches:
            for gate in ("i", "f", "o"):
                assert np.all(cache[gate] >= 0.0 - 1e-8)
                assert np.all(cache[gate] <= 1.0 + 1e-8)
            assert np.all(cache["g"] >= -1.0 - 1e-8)
            assert np.all(cache["g"] <= 1.0 + 1e-8)

    def test_hidden_state_finite_for_large_input(self):
        params = _Params(N_FEATURES, HIDDEN)
        X = np.full((LOOKBACK, N_FEATURES), 50.0)
        h, _ = _lstm_forward(X, params)
        assert np.all(np.isfinite(h))


# ---------------------------------------------------------------------------
# LSTM backward pass (sanity checks, not full numerical gradient check)
# ---------------------------------------------------------------------------


class TestLSTMBackward:
    def test_grad_shapes(self):
        params = _Params(N_FEATURES, HIDDEN)
        X = np.random.default_rng(3).normal(0, 0.1, (LOOKBACK, N_FEATURES))
        h, caches = _lstm_forward(X, params)
        dh = np.ones(HIDDEN)
        dWx, dWh, db = _lstm_backward(dh, caches, params)
        assert dWx.shape == params.Wx.shape
        assert dWh.shape == params.Wh.shape
        assert db.shape == params.b.shape

    def test_grads_finite(self):
        params = _Params(N_FEATURES, HIDDEN)
        X = np.random.default_rng(4).normal(0, 0.1, (LOOKBACK, N_FEATURES))
        _, caches = _lstm_forward(X, params)
        dWx, dWh, db = _lstm_backward(np.ones(HIDDEN), caches, params)
        assert np.all(np.isfinite(dWx))
        assert np.all(np.isfinite(dWh))
        assert np.all(np.isfinite(db))

    def test_zero_dh_gives_zero_grads(self):
        params = _Params(N_FEATURES, HIDDEN)
        X = np.random.default_rng(5).normal(0, 0.1, (LOOKBACK, N_FEATURES))
        _, caches = _lstm_forward(X, params)
        dWx, dWh, db = _lstm_backward(np.zeros(HIDDEN), caches, params)
        np.testing.assert_allclose(dWx, 0.0, atol=1e-10)
        np.testing.assert_allclose(dWh, 0.0, atol=1e-10)
        np.testing.assert_allclose(db, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Forecaster init validation
# ---------------------------------------------------------------------------


class TestForecasterInit:
    def test_bad_hidden_size(self):
        with pytest.raises(ValueError, match="hidden_size"):
            LSTMReturnForecaster(hidden_size=0)

    def test_bad_lookback(self):
        with pytest.raises(ValueError, match="lookback"):
            LSTMReturnForecaster(lookback=0)

    def test_bad_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            LSTMReturnForecaster(learning_rate=0.0)

    def test_bad_n_epochs(self):
        with pytest.raises(ValueError, match="n_epochs"):
            LSTMReturnForecaster(n_epochs=0)

    def test_bad_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            LSTMReturnForecaster(batch_size=0)

    def test_defaults(self):
        f = LSTMReturnForecaster()
        assert f.hidden_size == 32
        assert f.lookback == 20
        assert not f.is_fitted


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_self(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=2, seed=0)
        X, y = _make_sequences(n=60, lookback=LOOKBACK)
        result = f.fit(X, y)
        assert result is f

    def test_fit_sets_is_fitted(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=2)
        X, y = _make_sequences(n=60, lookback=LOOKBACK)
        f.fit(X, y)
        assert f.is_fitted

    def test_fit_result_type(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=3)
        X, y = _make_sequences(n=60, lookback=LOOKBACK)
        f.fit(X, y)
        assert isinstance(f.fit_result, ForecasterFitResult)

    def test_fit_result_fields(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=4)
        X, y = _make_sequences(n=60, lookback=LOOKBACK)
        f.fit(X, y)
        r = f.fit_result
        assert r is not None
        assert r.n_samples == 60
        assert r.n_features == N_FEATURES
        assert r.n_epochs_run == 4
        assert len(r.train_loss_history) == 4

    def test_too_few_samples_raises(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, min_train_samples=50)
        X, y = _make_sequences(n=20, lookback=LOOKBACK)
        with pytest.raises(ValueError, match="samples"):
            f.fit(X, y)

    def test_wrong_lookback_raises(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK)
        X, y = _make_sequences(n=60, lookback=LOOKBACK + 5)
        with pytest.raises(ValueError, match="lookback"):
            f.fit(X, y)

    def test_bad_X_ndim_raises(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK)
        with pytest.raises(ValueError, match="3-D"):
            f.fit(np.ones((60, LOOKBACK)), np.ones(60))

    def test_no_nan_in_weights_after_fit(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=5)
        X, y = _make_sequences(n=60, lookback=LOOKBACK)
        f.fit(X, y)
        assert f._params is not None
        for name in ("Wx", "Wh", "b", "Wy", "by"):
            arr = getattr(f._params, name)
            assert not np.any(np.isnan(arr)), f"NaN in {name} after fit"


# ---------------------------------------------------------------------------
# predict() / predict_zscore()
# ---------------------------------------------------------------------------


class TestPredict:
    def _fitted(self) -> LSTMReturnForecaster:
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=3, seed=7)
        X, y = _make_sequences(n=80, lookback=LOOKBACK)
        f.fit(X, y)
        return f

    def test_predict_before_fit_raises(self):
        f = LSTMReturnForecaster()
        with pytest.raises(RuntimeError):
            f.predict(np.ones((5, LOOKBACK, N_FEATURES)))

    def test_predict_output_shape(self):
        f = self._fitted()
        X, _ = _make_sequences(n=10, lookback=LOOKBACK)
        preds = f.predict(X)
        assert preds.shape == (10,)

    def test_predict_finite(self):
        f = self._fitted()
        X, _ = _make_sequences(n=10, lookback=LOOKBACK)
        preds = f.predict(X)
        assert np.all(np.isfinite(preds))

    def test_predict_zscore_mean_near_zero(self):
        f = self._fitted()
        X, _ = _make_sequences(n=20, lookback=LOOKBACK)
        z = f.predict_zscore(X)
        assert abs(float(z.mean())) < 1e-10

    def test_predict_zscore_std_near_one(self):
        f = self._fitted()
        X, _ = _make_sequences(n=20, lookback=LOOKBACK)
        z = f.predict_zscore(X)
        assert abs(float(z.std()) - 1.0) < 1e-10

    def test_reproducibility(self):
        X_train, y_train = _make_sequences(n=80, lookback=LOOKBACK, seed=0)
        X_test, _ = _make_sequences(n=10, lookback=LOOKBACK, seed=99)

        f1 = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=3, seed=42)
        f2 = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=3, seed=42)
        f1.fit(X_train, y_train)
        f2.fit(X_train, y_train)
        np.testing.assert_array_equal(f1.predict(X_test), f2.predict(X_test))

    def test_different_seeds_differ(self):
        X_train, y_train = _make_sequences(n=80, lookback=LOOKBACK, seed=0)
        X_test, _ = _make_sequences(n=10, lookback=LOOKBACK, seed=99)

        f1 = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=3, seed=1)
        f2 = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=3, seed=2)
        f1.fit(X_train, y_train)
        f2.fit(X_train, y_train)
        assert not np.allclose(f1.predict(X_test), f2.predict(X_test))

    def test_predict_bad_ndim_raises(self):
        f = self._fitted()
        with pytest.raises(ValueError, match="3-D"):
            f.predict(np.ones((10, LOOKBACK)))


# ---------------------------------------------------------------------------
# Loss decreases on memorizable data
# ---------------------------------------------------------------------------


class TestTrainingConvergence:
    def test_loss_decreases_on_overfit_data(self):
        """On a tiny dataset, the LSTM should memorize (loss falls)."""
        rng = np.random.default_rng(0)
        # Fixed linear signal for easy memorization
        X = rng.normal(0, 1, (40, LOOKBACK, N_FEATURES))
        y = X[:, -1, 0]  # target = last step's first feature (linear signal)
        y = (y - y.mean()) / y.std() * 0.01

        f = LSTMReturnForecaster(
            hidden_size=HIDDEN,
            lookback=LOOKBACK,
            n_epochs=30,
            learning_rate=1e-3,
            batch_size=10,
            seed=0,
        )
        f.fit(X, y)
        r = f.fit_result
        assert r is not None
        # Loss at final epoch should be below initial
        assert r.final_train_loss < r.initial_train_loss


# ---------------------------------------------------------------------------
# build_sequences()
# ---------------------------------------------------------------------------


class TestBuildSequences:
    def test_shape(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK)
        fp = _factor_panel(n_dates=N_DATES)
        ret = _returns(n_dates=N_DATES)
        X, y = f.build_sequences(fp, ret)
        expected_n = N_DATES - LOOKBACK
        assert X.shape == (expected_n, LOOKBACK, N_FEATURES)
        assert y.shape == (expected_n,)

    def test_empty_when_too_short(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK)
        fp = _factor_panel(n_dates=5)  # less than lookback
        ret = _returns(n_dates=5)
        X, y = f.build_sequences(fp, ret)
        assert X.shape[0] == 0

    def test_window_is_last_lookback_rows(self):
        """X[0] should be fp[0:lookback] (the first window)."""
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK)
        fp = _factor_panel(n_dates=N_DATES)
        ret = _returns(n_dates=N_DATES)
        X, _ = f.build_sequences(fp, ret)
        expected = fp.iloc[:LOOKBACK].to_numpy()
        np.testing.assert_allclose(X[0], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# generate_alpha_panel()
# ---------------------------------------------------------------------------


class TestGenerateAlphaPanel:
    def _panels(self, seed: int = 0) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        dates = pd.date_range("2018-01-01", periods=N_DATES * 2, freq="B")
        rng = np.random.default_rng(seed)
        factor_panels = {}
        returns_data = {}
        for asset in ASSETS:
            fp_data = rng.normal(0, 1, (len(dates), N_FEATURES))
            factor_panels[asset] = pd.DataFrame(
                fp_data, index=dates, columns=[f"f{i}" for i in range(N_FEATURES)]
            )
            returns_data[asset] = rng.normal(0, 0.01, len(dates))
        returns = pd.DataFrame(returns_data, index=dates)
        return factor_panels, returns

    def test_returns_dataframe(self):
        f = LSTMReturnForecaster(
            hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=2, min_train_samples=20
        )
        fp, ret = self._panels()
        alpha = f.generate_alpha_panel(fp, ret, min_train_periods=60, retrain_every=21)
        assert isinstance(alpha, pd.DataFrame)

    def test_columns_match_assets(self):
        f = LSTMReturnForecaster(
            hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=2, min_train_samples=20
        )
        fp, ret = self._panels()
        alpha = f.generate_alpha_panel(fp, ret, min_train_periods=60, retrain_every=21)
        assert list(alpha.columns) == ASSETS

    def test_alpha_values_finite(self):
        f = LSTMReturnForecaster(
            hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=2, min_train_samples=20
        )
        fp, ret = self._panels()
        alpha = f.generate_alpha_panel(fp, ret, min_train_periods=60, retrain_every=21)
        if len(alpha) > 0:
            assert np.all(np.isfinite(alpha.values))

    def test_alpha_panel_not_empty(self):
        f = LSTMReturnForecaster(
            hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=2, min_train_samples=20
        )
        fp, ret = self._panels()
        alpha = f.generate_alpha_panel(fp, ret, min_train_periods=60, retrain_every=21)
        assert len(alpha) > 0

    def test_alpha_z_scores_bounded(self):
        """Cross-sectional z-scores should be reasonable in magnitude."""
        f = LSTMReturnForecaster(
            hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=2, min_train_samples=20, seed=0
        )
        fp, ret = self._panels()
        alpha = f.generate_alpha_panel(fp, ret, min_train_periods=60, retrain_every=21)
        if len(alpha) > 0:
            # z-scores should be bounded within ±10σ
            assert float(alpha.abs().max().max()) < 10.0

    def test_empty_panels_returns_empty_df(self):
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=2)
        fp, ret = self._panels()
        # Pass very high min_train_periods so no date qualifies
        alpha = f.generate_alpha_panel(fp, ret, min_train_periods=999_999)
        assert len(alpha) == 0


# ---------------------------------------------------------------------------
# ForecasterFitResult
# ---------------------------------------------------------------------------


class TestForecasterFitResult:
    def _result(self) -> ForecasterFitResult:
        f = LSTMReturnForecaster(hidden_size=HIDDEN, lookback=LOOKBACK, n_epochs=5)
        X, y = _make_sequences(n=60, lookback=LOOKBACK)
        f.fit(X, y)
        return f.fit_result  # type: ignore[return-value]

    def test_final_train_loss_property(self):
        r = self._result()
        assert r.final_train_loss == r.train_loss_history[-1]

    def test_initial_train_loss_property(self):
        r = self._result()
        assert r.initial_train_loss == r.train_loss_history[0]

    def test_loss_history_length(self):
        r = self._result()
        assert len(r.train_loss_history) == 5

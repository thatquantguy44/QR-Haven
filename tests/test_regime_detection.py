"""Tests for GMMRegimeDetector and GaussianHMMDetector."""

import numpy as np
import pytest

from qr_haven.regimes import GMMRegimeDetector, GaussianHMMDetector
from qr_haven.regimes._math import (
    log_multivariate_normal,
    logsumexp,
    logsumexp_cols,
    logsumexp_rows,
    simple_kmeans,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_state_data(n: int = 300, seed: int = 0) -> np.ndarray:
    """Two clearly separated regimes: low-vol bull and high-vol bear."""
    rng = np.random.default_rng(seed)
    half = n // 2
    bull = rng.normal([0.05, 0.10], [0.02, 0.01], size=(half, 2))
    bear = rng.normal([-0.05, 0.30], [0.03, 0.02], size=(n - half, 2))
    return np.vstack([bull, bear])


def _single_feature_data(n: int = 200, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(n,))


# ---------------------------------------------------------------------------
# _math utilities
# ---------------------------------------------------------------------------

class TestMathUtils:
    def test_logsumexp_simple(self):
        a = np.array([0.0, 0.0, 0.0])
        assert abs(logsumexp(a) - np.log(3)) < 1e-10

    def test_logsumexp_single(self):
        assert abs(logsumexp(np.array([5.0])) - 5.0) < 1e-10

    def test_logsumexp_negative_inf(self):
        a = np.array([-np.inf, -np.inf])
        assert np.isinf(logsumexp(a))

    def test_logsumexp_rows(self):
        a = np.zeros((3, 4))
        result = logsumexp_rows(a)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, np.log(4), atol=1e-10)

    def test_logsumexp_cols(self):
        a = np.zeros((3, 4))
        result = logsumexp_cols(a)
        assert result.shape == (4,)
        np.testing.assert_allclose(result, np.log(3), atol=1e-10)

    def test_log_multivariate_normal_1d(self):
        X = np.array([[0.0], [1.0], [-1.0]])
        mean = np.array([0.0])
        cov = np.eye(1)
        log_p = log_multivariate_normal(X, mean, cov)
        assert log_p.shape == (3,)
        # mode should be at x=0
        assert log_p[0] > log_p[1]
        assert log_p[0] > log_p[2]

    def test_log_multivariate_normal_2d(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 2))
        mean = np.zeros(2)
        cov = np.eye(2)
        log_p = log_multivariate_normal(X, mean, cov)
        assert log_p.shape == (50,)
        assert np.all(np.isfinite(log_p))

    def test_simple_kmeans_returns_correct_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        labels, centers = simple_kmeans(X, k=4, rng=rng)
        assert labels.shape == (100,)
        assert centers.shape == (4, 3)
        assert set(labels).issubset(set(range(4)))

    def test_simple_kmeans_separable(self):
        rng = np.random.default_rng(7)
        a = np.zeros((50, 2))
        b = np.full((50, 2), 100.0)
        X = np.vstack([a, b])
        labels, centers = simple_kmeans(X, k=2, rng=rng)
        # first 50 and last 50 should be in distinct clusters
        assert len(set(labels[:50])) == 1
        assert len(set(labels[50:])) == 1
        assert labels[0] != labels[50]


# ---------------------------------------------------------------------------
# GMMRegimeDetector
# ---------------------------------------------------------------------------

class TestGMMRegimeDetector:
    def test_init_defaults(self):
        g = GMMRegimeDetector()
        assert g.n_regimes == 4
        assert g.n_iter == 200
        assert g.tol == 1e-4

    def test_init_bad_n_regimes(self):
        with pytest.raises(ValueError):
            GMMRegimeDetector(n_regimes=0)

    def test_init_bad_n_iter(self):
        with pytest.raises(ValueError):
            GMMRegimeDetector(n_iter=0)

    def test_init_bad_tol(self):
        with pytest.raises(ValueError):
            GMMRegimeDetector(tol=0.0)

    def test_init_bad_reg_covar(self):
        with pytest.raises(ValueError):
            GMMRegimeDetector(reg_covar=-1.0)

    def test_not_fitted(self):
        g = GMMRegimeDetector()
        assert not g.fitted

    def test_predict_before_fit_raises(self):
        g = GMMRegimeDetector()
        with pytest.raises(RuntimeError):
            g.predict(np.zeros((10, 2)))

    def test_predict_proba_before_fit_raises(self):
        g = GMMRegimeDetector()
        with pytest.raises(RuntimeError):
            g.predict_proba(np.zeros((10, 2)))

    def test_fit_returns_self(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=2, random_state=0)
        result = g.fit(X)
        assert result is g

    def test_fitted_after_fit(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=2, random_state=0).fit(X)
        assert g.fitted

    def test_fit_1d_input(self):
        X = _single_feature_data()
        g = GMMRegimeDetector(n_regimes=2, random_state=0).fit(X)
        assert g.means_.shape == (2, 1)

    def test_fit_requires_enough_data(self):
        # 1 sample is still allowed (no T>=2 check in GMM), just verify no crash
        X = np.array([[1.0, 2.0]])
        GMMRegimeDetector(n_regimes=1, random_state=0).fit(X)

    def test_predict_shape(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=2, random_state=0).fit(X)
        labels = g.predict(X)
        assert labels.shape == (len(X),)

    def test_predict_labels_in_range(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=3, random_state=0).fit(X)
        labels = g.predict(X)
        assert set(labels).issubset(set(range(3)))

    def test_predict_proba_shape(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=2, random_state=0).fit(X)
        proba = g.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=2, random_state=0).fit(X)
        proba = g.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_non_negative(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=2, random_state=0).fit(X)
        assert np.all(g.predict_proba(X) >= 0)

    def test_log_likelihoods_increases(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=2, random_state=0).fit(X)
        lls = g.log_likelihoods_
        assert len(lls) >= 2
        # allow tiny numerical noise but overall trend should be non-decreasing
        for i in range(1, len(lls)):
            assert lls[i] >= lls[i - 1] - 1e-6

    def test_separable_regimes_detected(self):
        """Two clearly separated blobs should map to two distinct regimes."""
        n = 200
        rng = np.random.default_rng(42)
        X_low = rng.normal([0.0, 0.1], 0.01, size=(n, 2))
        X_high = rng.normal([1.0, 0.9], 0.01, size=(n, 2))
        X = np.vstack([X_low, X_high])
        g = GMMRegimeDetector(n_regimes=2, random_state=42).fit(X)
        labels = g.predict(X)
        # first half and second half should be in different regimes
        assert len(set(labels[:n])) == 1
        assert len(set(labels[n:])) == 1
        assert labels[0] != labels[n]

    def test_weights_sum_to_one(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=3, random_state=0).fit(X)
        np.testing.assert_allclose(g.weights_.sum(), 1.0, atol=1e-10)

    def test_means_shape(self):
        X = _two_state_data()  # (T, 2)
        g = GMMRegimeDetector(n_regimes=3, random_state=0).fit(X)
        assert g.means_.shape == (3, 2)

    def test_covs_positive_definite(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=2, random_state=0).fit(X)
        for k in range(2):
            eigvals = np.linalg.eigvalsh(g.covs_[k])
            assert np.all(eigvals > 0)

    def test_predict_1d_input(self):
        X = _single_feature_data()
        g = GMMRegimeDetector(n_regimes=2, random_state=0).fit(X)
        labels = g.predict(X)
        assert labels.shape == (len(X),)

    def test_n_regimes_1(self):
        X = _two_state_data()
        g = GMMRegimeDetector(n_regimes=1, random_state=0).fit(X)
        labels = g.predict(X)
        assert np.all(labels == 0)

    def test_reproducible_with_seed(self):
        X = _two_state_data()
        g1 = GMMRegimeDetector(n_regimes=2, random_state=7).fit(X)
        g2 = GMMRegimeDetector(n_regimes=2, random_state=7).fit(X)
        np.testing.assert_array_equal(g1.predict(X), g2.predict(X))


# ---------------------------------------------------------------------------
# GaussianHMMDetector
# ---------------------------------------------------------------------------

class TestGaussianHMMDetector:
    def test_init_defaults(self):
        h = GaussianHMMDetector()
        assert h.n_states == 2
        assert h.n_iter == 100
        assert h.tol == 1e-4

    def test_init_bad_n_states(self):
        with pytest.raises(ValueError):
            GaussianHMMDetector(n_states=0)

    def test_init_bad_n_iter(self):
        with pytest.raises(ValueError):
            GaussianHMMDetector(n_iter=0)

    def test_init_bad_tol(self):
        with pytest.raises(ValueError):
            GaussianHMMDetector(tol=-0.1)

    def test_init_bad_reg_covar(self):
        with pytest.raises(ValueError):
            GaussianHMMDetector(reg_covar=-1e-7)

    def test_not_fitted(self):
        h = GaussianHMMDetector()
        assert not h.fitted

    def test_predict_before_fit_raises(self):
        h = GaussianHMMDetector()
        with pytest.raises(RuntimeError):
            h.predict(np.zeros((10, 2)))

    def test_predict_proba_before_fit_raises(self):
        h = GaussianHMMDetector()
        with pytest.raises(RuntimeError):
            h.predict_proba(np.zeros((10, 2)))

    def test_fit_requires_t_ge_2(self):
        h = GaussianHMMDetector()
        with pytest.raises(ValueError):
            h.fit(np.array([[1.0, 2.0]]))

    def test_fit_returns_self(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0)
        assert h.fit(X) is h

    def test_fitted_after_fit(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        assert h.fitted

    def test_fit_1d_input(self):
        X = _single_feature_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        assert h.means_.shape == (2, 1)

    def test_predict_shape(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        labels = h.predict(X)
        assert labels.shape == (len(X),)

    def test_predict_labels_in_range(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=3, random_state=0).fit(X)
        labels = h.predict(X)
        assert set(labels).issubset(set(range(3)))

    def test_predict_proba_shape(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        proba = h.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        proba = h.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_predict_proba_non_negative(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        assert np.all(h.predict_proba(X) >= -1e-12)

    def test_log_likelihoods_non_decreasing(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        lls = h.log_likelihoods_
        assert len(lls) >= 1
        for i in range(1, len(lls)):
            assert lls[i] >= lls[i - 1] - 1e-4

    def test_startprob_normalised(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        np.testing.assert_allclose(
            np.exp(h.log_startprob_).sum(), 1.0, atol=1e-10
        )

    def test_transmat_rows_sum_to_one(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        row_sums = np.exp(h.log_transmat_).sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_separable_regimes_detected(self):
        """Two clearly separated blobs — HMM should assign them to distinct states."""
        n = 150
        rng = np.random.default_rng(42)
        X_low = rng.normal([0.0, 0.1], 0.01, size=(n, 2))
        X_high = rng.normal([1.0, 0.9], 0.01, size=(n, 2))
        X = np.vstack([X_low, X_high])
        h = GaussianHMMDetector(n_states=2, random_state=42).fit(X)
        labels = h.predict(X)
        assert len(set(labels[:n])) == 1
        assert len(set(labels[n:])) == 1
        assert labels[0] != labels[n]

    def test_means_shape(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=3, random_state=0).fit(X)
        assert h.means_.shape == (3, 2)

    def test_covs_positive_definite(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        for k in range(2):
            eigvals = np.linalg.eigvalsh(h.covs_[k])
            assert np.all(eigvals > 0)

    def test_predict_1d_input(self):
        X = _single_feature_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        labels = h.predict(X)
        assert labels.shape == (len(X),)

    def test_n_states_1(self):
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=1, random_state=0).fit(X)
        labels = h.predict(X)
        assert np.all(labels == 0)

    def test_reproducible_with_seed(self):
        X = _two_state_data()
        h1 = GaussianHMMDetector(n_states=2, random_state=5).fit(X)
        h2 = GaussianHMMDetector(n_states=2, random_state=5).fit(X)
        np.testing.assert_array_equal(h1.predict(X), h2.predict(X))

    def test_viterbi_vs_smoothed_agreement(self):
        """Viterbi hard labels should mostly agree with argmax of smoothed posteriors."""
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        viterbi = h.predict(X)
        smoothed = h.predict_proba(X).argmax(axis=1)
        agreement = (viterbi == smoothed).mean()
        assert agreement > 0.85

    def test_high_persistence_transition(self):
        """After fitting, diagonal of transition matrix should dominate (> 0.5)."""
        X = _two_state_data()
        h = GaussianHMMDetector(n_states=2, random_state=0).fit(X)
        A = np.exp(h.log_transmat_)
        assert np.all(np.diag(A) > 0.5)

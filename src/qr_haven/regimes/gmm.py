"""Gaussian Mixture Model for market regime detection.

Fits K Gaussian components via the EM algorithm.  Each observation is
soft-assigned to all K regimes with a probability proportional to the
component's likelihood — no hard state boundary.

Typical use: cluster (volatility, return, correlation-change) feature
vectors into bull / bear / low-vol / crisis regimes.
"""

from __future__ import annotations

import numpy as np

from qr_haven.regimes._math import (
    log_multivariate_normal,
    logsumexp_cols,
    simple_kmeans,
)


class GMMRegimeDetector:
    """Gaussian Mixture Model regime detector (EM algorithm, numpy-only).

    Parameters
    ----------
    n_regimes:
        Number of Gaussian components (regimes). Default 4.
    n_iter:
        Maximum EM iterations. Default 200.
    tol:
        Convergence threshold on log-likelihood improvement. Default 1e-4.
    reg_covar:
        Regularisation added to each component covariance diagonal to prevent
        degenerate fits. Default 1e-6.
    random_state:
        Seed for reproducible initialisation.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        n_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        if n_regimes < 1:
            raise ValueError("n_regimes must be at least 1.")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")
        if reg_covar < 0.0:
            raise ValueError("reg_covar cannot be negative.")
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covs_: np.ndarray | None = None
        self.log_likelihoods_: list[float] = []

    @property
    def fitted(self) -> bool:
        return self.weights_ is not None

    def fit(self, X: np.ndarray) -> "GMMRegimeDetector":
        """Fit the GMM via EM.

        Parameters
        ----------
        X:
            (T, d) feature matrix — e.g. rolling volatility and return columns.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        T, d = X.shape
        K = self.n_regimes
        rng = np.random.default_rng(self.random_state)

        _, centers = simple_kmeans(X, K, rng=rng)
        self.means_ = centers.copy()
        self.weights_ = np.full(K, 1.0 / K)
        base_var = float(np.var(X, axis=0).mean()) + self.reg_covar
        self.covs_ = np.stack([np.eye(d) * base_var for _ in range(K)])
        self.log_likelihoods_ = []

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            log_resp = self._compute_log_responsibilities(X)  # (T, K)
            ll = float(self._log_likelihood_from_log_resp(log_resp))
            self.log_likelihoods_.append(ll)
            if ll - prev_ll < self.tol:
                break
            prev_ll = ll

            resp = self._normalize_log_resp(log_resp)
            Nk = resp.sum(axis=0)  # (K,)

            self.weights_ = Nk / T
            self.means_ = (resp.T @ X) / np.maximum(Nk[:, None], 1e-300)
            for k in range(K):
                diff = X - self.means_[k]
                self.covs_[k] = (resp[:, k : k + 1] * diff).T @ diff / max(Nk[k], 1e-300)
                self.covs_[k] += self.reg_covar * np.eye(d)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most likely regime index for each observation.

        Parameters
        ----------
        X:
            (T, d) feature matrix.

        Returns
        -------
        (T,) integer array of regime labels in [0, n_regimes).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return self._compute_log_responsibilities(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return soft regime membership probabilities.

        Parameters
        ----------
        X:
            (T, d) feature matrix.

        Returns
        -------
        (T, n_regimes) array of probabilities summing to 1 along axis 1.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return self._normalize_log_resp(self._compute_log_responsibilities(X))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_log_responsibilities(self, X: np.ndarray) -> np.ndarray:
        K = self.n_regimes
        log_resp = np.column_stack([
            np.log(max(self.weights_[k], 1e-300))
            + log_multivariate_normal(X, self.means_[k], self.covs_[k])
            for k in range(K)
        ])
        return log_resp

    @staticmethod
    def _normalize_log_resp(log_resp: np.ndarray) -> np.ndarray:
        max_lr = log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp - max_lr)
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    @staticmethod
    def _log_likelihood_from_log_resp(log_resp: np.ndarray) -> float:
        max_lr = log_resp.max(axis=1, keepdims=True)
        log_sum = (max_lr[:, 0] + np.log(np.exp(log_resp - max_lr).sum(axis=1))).mean()
        return float(log_sum)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("GMMRegimeDetector must be fitted before calling predict.")

"""Gaussian Hidden Markov Model for market regime detection.

Implements the complete Baum-Welch (EM) training algorithm and Viterbi
decoding in numpy.  All computations are performed in log-space for
numerical stability over long return series.

Compared to the GMM, the HMM explicitly models temporal persistence:
regimes cannot jump instantaneously — they transition with probability
governed by the learned transition matrix.  This makes the HMM better
at capturing sustained market environments (e.g. a bear market lasting
several months) rather than day-to-day volatility clusters.
"""

from __future__ import annotations

import numpy as np

from qr_haven.regimes._math import (
    log_multivariate_normal,
    logsumexp,
    logsumexp_cols,
    logsumexp_rows,
    simple_kmeans,
)

_LOG_EPSILON = np.log(1e-300)


class GaussianHMMDetector:
    """Gaussian HMM regime detector (Baum-Welch + Viterbi, numpy-only).

    Parameters
    ----------
    n_states:
        Number of hidden states (regimes). Default 2.
    n_iter:
        Maximum Baum-Welch iterations. Default 100.
    tol:
        Convergence threshold on log-likelihood improvement. Default 1e-4.
    reg_covar:
        Regularisation added to each emission covariance diagonal. Default 1e-6.
    random_state:
        Seed for k-means initialisation.
    """

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be at least 1.")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")
        if reg_covar < 0.0:
            raise ValueError("reg_covar cannot be negative.")
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.log_startprob_: np.ndarray | None = None
        self.log_transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covs_: np.ndarray | None = None
        self.log_likelihoods_: list[float] = []

    @property
    def fitted(self) -> bool:
        return self.log_startprob_ is not None

    def fit(self, X: np.ndarray) -> "GaussianHMMDetector":
        """Fit HMM parameters via the Baum-Welch algorithm.

        Parameters
        ----------
        X:
            (T, d) observation matrix (e.g. returns and rolling volatility).
            Must have T >= 2.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        T, d = X.shape
        K = self.n_states
        if T < 2:
            raise ValueError("X must have at least 2 observations.")

        rng = np.random.default_rng(self.random_state)
        labels, centers = simple_kmeans(X, K, rng=rng)

        self.means_ = centers.copy()
        base_var = float(np.var(X, axis=0).mean()) + self.reg_covar
        self.covs_ = np.stack([np.eye(d) * base_var for _ in range(K)])
        self.log_startprob_ = np.log(np.full(K, 1.0 / K))
        A = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(A, 0.9)
        self.log_transmat_ = np.log(A)

        self.log_likelihoods_ = []
        prev_ll = -np.inf

        for _ in range(self.n_iter):
            log_b = self._compute_log_emissions(X)          # (T, K)
            log_alpha = self._forward_log(log_b)             # (T, K)
            log_beta = self._backward_log(log_b)             # (T, K)
            log_gamma = self._compute_log_gamma(log_alpha, log_beta)   # (T, K)
            log_xi = self._compute_log_xi(log_alpha, log_beta, log_b)  # (T-1, K, K)

            ll = float(logsumexp(log_alpha[-1]))
            self.log_likelihoods_.append(ll)
            if ll - prev_ll < self.tol:
                break
            prev_ll = ll

            gamma = np.exp(log_gamma)
            xi = np.exp(log_xi)

            # Update start probs
            self.log_startprob_ = log_gamma[0] - logsumexp(log_gamma[0])

            # Update transition matrix
            xi_sum = xi.sum(axis=0)                                # (K, K)
            gamma_sum = gamma[:-1].sum(axis=0)                     # (K,)
            self.log_transmat_ = (
                np.log(np.maximum(xi_sum, 1e-300))
                - np.log(np.maximum(gamma_sum[:, None], 1e-300))
            )

            # Update emission parameters
            for k in range(K):
                gk = gamma[:, k]
                Nk = max(float(gk.sum()), 1e-300)
                self.means_[k] = (gk[:, None] * X).sum(axis=0) / Nk
                diff = X - self.means_[k]
                self.covs_[k] = (gk[:, None] * diff).T @ diff / Nk
                self.covs_[k] += self.reg_covar * np.eye(d)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the Viterbi (most likely) state sequence.

        Parameters
        ----------
        X:
            (T, d) observation matrix.

        Returns
        -------
        (T,) integer array of state labels in [0, n_states).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return self._viterbi(self._compute_log_emissions(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return smoothed state posterior probabilities (forward-backward).

        Parameters
        ----------
        X:
            (T, d) observation matrix.

        Returns
        -------
        (T, n_states) array of probabilities summing to 1 along axis 1.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        log_b = self._compute_log_emissions(X)
        log_alpha = self._forward_log(log_b)
        log_beta = self._backward_log(log_b)
        log_gamma = self._compute_log_gamma(log_alpha, log_beta)
        return np.exp(log_gamma)

    # ------------------------------------------------------------------
    # Log-domain algorithms
    # ------------------------------------------------------------------

    def _compute_log_emissions(self, X: np.ndarray) -> np.ndarray:
        """Return (T, K) log emission probabilities."""
        return np.column_stack([
            log_multivariate_normal(X, self.means_[k], self.covs_[k])
            for k in range(self.n_states)
        ])

    def _forward_log(self, log_b: np.ndarray) -> np.ndarray:
        """Log-domain forward algorithm.

        log_alpha[t, k] = log P(x_0..x_t, z_t=k)
        """
        T, K = log_b.shape
        log_alpha = np.full((T, K), _LOG_EPSILON)
        log_alpha[0] = self.log_startprob_ + log_b[0]
        for t in range(1, T):
            # scores[i, j] = log_alpha[t-1, i] + log_transmat_[i, j]
            scores = log_alpha[t - 1, :, None] + self.log_transmat_   # (K, K)
            log_alpha[t] = logsumexp_cols(scores) + log_b[t]
        return log_alpha

    def _backward_log(self, log_b: np.ndarray) -> np.ndarray:
        """Log-domain backward algorithm.

        log_beta[t, k] = log P(x_{t+1}..x_{T-1} | z_t=k)
        """
        T, K = log_b.shape
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            # scores[i, j] = log_transmat_[i,j] + log_b[t+1,j] + log_beta[t+1,j]
            scores = (
                self.log_transmat_
                + (log_b[t + 1] + log_beta[t + 1])[None, :]   # (K, K)
            )
            log_beta[t] = logsumexp_rows(scores)
        return log_beta

    def _compute_log_gamma(
        self, log_alpha: np.ndarray, log_beta: np.ndarray
    ) -> np.ndarray:
        """Compute log P(z_t=k | X), normalised per timestep."""
        log_gamma = log_alpha + log_beta             # (T, K)
        log_norm = logsumexp_rows(log_gamma)[:, None]
        return log_gamma - log_norm

    def _compute_log_xi(
        self,
        log_alpha: np.ndarray,
        log_beta: np.ndarray,
        log_b: np.ndarray,
    ) -> np.ndarray:
        """Compute log P(z_t=i, z_{t+1}=j | X), normalised per t."""
        T, K = log_b.shape
        log_xi = np.full((T - 1, K, K), _LOG_EPSILON)
        for t in range(T - 1):
            # log_xi[t, i, j] = log_alpha[t,i] + log_transmat[i,j]
            #                    + log_b[t+1,j] + log_beta[t+1,j]
            log_xi[t] = (
                log_alpha[t, :, None]
                + self.log_transmat_
                + (log_b[t + 1] + log_beta[t + 1])[None, :]
            )
            log_xi[t] -= logsumexp(log_xi[t].ravel())
        return log_xi

    def _viterbi(self, log_b: np.ndarray) -> np.ndarray:
        """Viterbi algorithm: most likely state sequence."""
        T, K = log_b.shape
        log_delta = np.full((T, K), _LOG_EPSILON)
        psi = np.zeros((T, K), dtype=int)

        log_delta[0] = self.log_startprob_ + log_b[0]
        for t in range(1, T):
            scores = log_delta[t - 1, :, None] + self.log_transmat_   # (K, K)
            psi[t] = scores.argmax(axis=0)
            log_delta[t] = scores.max(axis=0) + log_b[t]

        states = np.zeros(T, dtype=int)
        states[-1] = log_delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("GaussianHMMDetector must be fitted before calling predict.")

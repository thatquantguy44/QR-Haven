"""Numerically stable log-domain math utilities for regime models."""

from __future__ import annotations

import numpy as np


def logsumexp(a: np.ndarray) -> float:
    """Log-sum-exp for a 1-D array."""
    a_max = float(a.max())
    if np.isinf(a_max):
        return a_max
    return a_max + float(np.log(np.sum(np.exp(a - a_max))))


def logsumexp_rows(a: np.ndarray) -> np.ndarray:
    """Log-sum-exp along axis=1 (sum over columns for each row)."""
    a_max = a.max(axis=1, keepdims=True)
    a_max = np.where(np.isinf(a_max), 0.0, a_max)
    return np.log(np.sum(np.exp(a - a_max), axis=1)) + a_max[:, 0]


def logsumexp_cols(a: np.ndarray) -> np.ndarray:
    """Log-sum-exp along axis=0 (sum over rows for each column)."""
    a_max = a.max(axis=0, keepdims=True)
    a_max = np.where(np.isinf(a_max), 0.0, a_max)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max[0]


def log_multivariate_normal(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log PDF of a multivariate Gaussian evaluated at each row of X.

    Parameters
    ----------
    X:
        (T, d) observation matrix.
    mean:
        (d,) component mean.
    cov:
        (d, d) covariance matrix.

    Returns
    -------
    (T,) array of log probabilities.
    """
    d = mean.shape[0]
    diff = X - mean  # (T, d)
    try:
        L = np.linalg.cholesky(cov)
        log_det = 2.0 * np.sum(np.log(np.maximum(np.diag(L), 1e-300)))
        solved = np.linalg.solve(L, diff.T).T  # (T, d)
        maha = np.sum(solved ** 2, axis=1)
    except np.linalg.LinAlgError:
        var = np.maximum(np.diag(cov), 1e-300)
        log_det = np.sum(np.log(var))
        maha = np.sum(diff ** 2 / var, axis=1)
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)


def simple_kmeans(
    X: np.ndarray,
    k: int,
    n_iter: int = 50,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """k-means clustering for initializing mixture model means.

    Returns
    -------
    labels: (T,) integer cluster assignments.
    centers: (k, d) cluster centroids.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    T, d = X.shape
    centers = X[rng.choice(T, k, replace=False)].copy()
    labels = np.zeros(T, dtype=int)
    for _ in range(n_iter):
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.array(
            [X[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
             for j in range(k)]
        )
        if np.allclose(centers, new_centers, atol=1e-8):
            break
        centers = new_centers
    return labels, centers

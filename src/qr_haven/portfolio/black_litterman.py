"""Black-Litterman optimizer for multi-asset class portfolios."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from qr_haven.portfolio.optimizers import (
    MeanVarianceOptimizer,
    OptimizerConstraints,
    OptimizerDiagnostics,
    calculate_optimizer_diagnostics,
)


# ---------------------------------------------------------------------------
# View specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class View:
    """A single investor view for the Black-Litterman model.

    Absolute view — "AAPL returns 10% annualized":
        View(assets=["AAPL"], weights=[1.0], expected_return=0.10)

    Relative view — "AAPL outperforms MSFT by 3%":
        View(assets=["AAPL", "MSFT"], weights=[1.0, -1.0], expected_return=0.03)

    confidence: 0 < c ≤ 1.  Under the Idzorek method this maps exactly to a
    (1-c)·prior + c·view blend for a single absolute view.
    """

    assets: list[str]
    weights: list[float]
    expected_return: float
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if len(self.assets) == 0:
            raise ValueError("View must reference at least one asset.")
        if len(self.assets) != len(self.weights):
            raise ValueError(
                f"assets has {len(self.assets)} elements but weights has {len(self.weights)}."
            )
        if not (0.0 < self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in (0, 1], got {self.confidence}."
            )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlackLittermanResult:
    """Output of a Black-Litterman posterior estimation and optimization."""

    posterior_returns: pd.Series
    posterior_covariance: pd.DataFrame
    prior_returns: pd.Series
    weights: pd.Series
    views_used: int
    tau: float
    risk_aversion: float

    def summary(self) -> dict[str, float | int]:
        """Compact summary of the BL estimation."""
        hhi = float((self.weights**2).sum())
        return {
            "views_used": self.views_used,
            "tau": self.tau,
            "risk_aversion": self.risk_aversion,
            "prior_return_mean": float(self.prior_returns.mean()),
            "posterior_return_mean": float(self.posterior_returns.mean()),
            "portfolio_weight_sum": float(self.weights.sum()),
            "effective_holdings": float(1.0 / hhi) if hhi > 0.0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class BlackLittermanOptimizer:
    """Black-Litterman optimizer for multi-asset class portfolios.

    Combines a market-equilibrium prior (reverse-optimized from market-cap
    weights) with investor views to produce posterior expected returns, then
    optimizes weights on the posterior mean–variance frontier.

    Supports:
    - Absolute views ("asset A returns 8% annually")
    - Relative views ("asset A outperforms asset B by 3%")
    - Multi-asset class portfolios: equity, fixed income, commodities, FX, alts
    - Idzorek confidence scaling or proportional (He–Litterman) Omega
    - Drop-in replacement for EqualWeightOptimizer / MeanVarianceOptimizer

    References:
    - Black & Litterman (1992), "Global Portfolio Optimization"
    - He & Litterman (1999), "The Intuition Behind Black-Litterman"
    - Idzorek (2005), "A Step-by-Step Guide to the Black-Litterman Model"
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        omega_method: str = "proportional",
        views: Sequence[View] | None = None,
        market_weights: pd.Series | None = None,
    ) -> None:
        """
        Args:
            risk_aversion: Market risk aversion δ. Typical range 2–4. Controls
                how large the equilibrium return implied by market weights is.
            tau: Prior uncertainty scalar τ ∈ (0, ∞). Typical range 0.01–0.10.
                Smaller τ → equilibrium prior is more certain → views have less impact.
            omega_method: "proportional" sets Ω = τ·P·Σ·P' (He–Litterman default).
                "idzorek" scales each row by (1-c)/c so that confidence c maps to
                an exact (1-c)·prior + c·view blend.
            views: Pre-specified views incorporated on every optimize() call.
            market_weights: Market-cap or equilibrium portfolio weights. Falls back
                to equal weights when not provided.
        """
        if risk_aversion <= 0.0:
            raise ValueError(f"risk_aversion must be positive, got {risk_aversion}.")
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}.")
        if omega_method not in ("proportional", "idzorek"):
            raise ValueError(
                f"omega_method must be 'proportional' or 'idzorek', got '{omega_method}'."
            )
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.omega_method = omega_method
        self.views: list[View] = list(views) if views is not None else []
        self.market_weights = market_weights

    # ------------------------------------------------------------------
    # Drop-in optimizer interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: Mapping[str, Any] | None = None,
    ) -> pd.Series:
        """Return BL-optimal weights.

        expected_returns is used as the equilibrium prior when market_weights is
        not set on the optimizer; if market_weights is set, it is ignored and the
        prior is derived by reverse-optimization from covariance + market_weights.
        """
        result = self._run(
            prior_override=None if self.market_weights is not None else expected_returns,
            market_weights=self.market_weights,
            covariance=covariance,
            views=self.views,
            tau=self.tau,
            constraints=constraints,
        )
        return result.weights

    def optimize_with_views(
        self,
        covariance: pd.DataFrame,
        views: Sequence[View] | None = None,
        market_weights: pd.Series | None = None,
        tau: float | None = None,
        constraints: Mapping[str, Any] | None = None,
    ) -> BlackLittermanResult:
        """Full BL workflow: compute posterior, then optimize.

        Args:
            covariance: N×N annualized return covariance.
            views: Views to blend with the prior (merged with instance-level views).
            market_weights: Market-cap weights for the equilibrium prior. Overrides
                the instance-level market_weights for this call.
            tau: Per-call override for prior uncertainty scalar.
            constraints: Portfolio constraints forwarded to the inner MVO.

        Returns:
            BlackLittermanResult with posterior returns, posterior covariance, and
            optimal weights.
        """
        all_views = self.views + (list(views) if views is not None else [])
        mkt_w = market_weights if market_weights is not None else self.market_weights
        return self._run(
            prior_override=None,
            market_weights=mkt_w,
            covariance=covariance,
            views=all_views,
            tau=tau if tau is not None else self.tau,
            constraints=constraints,
        )

    def optimize_with_diagnostics(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: Mapping[str, Any] | None = None,
        previous_weights: pd.Series | None = None,
    ) -> tuple[pd.Series, OptimizerDiagnostics]:
        """Return BL weights and diagnostics (compatible with other optimizers)."""
        result = self._run(
            prior_override=None if self.market_weights is not None else expected_returns,
            market_weights=self.market_weights,
            covariance=covariance,
            views=self.views,
            tau=self.tau,
            constraints=constraints,
        )
        diagnostics = calculate_optimizer_diagnostics(
            result.weights,
            result.posterior_returns,
            result.posterior_covariance,
            constraints,
            previous_weights,
        )
        return result.weights, diagnostics

    # ------------------------------------------------------------------
    # Core estimation
    # ------------------------------------------------------------------

    def _run(
        self,
        prior_override: pd.Series | None,
        market_weights: pd.Series | None,
        covariance: pd.DataFrame,
        views: Sequence[View],
        tau: float,
        constraints: Mapping[str, Any] | None,
    ) -> BlackLittermanResult:
        assets = list(covariance.columns)
        n = len(assets)
        sigma = (
            covariance.reindex(index=assets, columns=assets)
            .fillna(0.0)
            .to_numpy()
            .astype(float)
        )

        # -- Equilibrium prior π ------------------------------------------
        if prior_override is not None:
            pi = prior_override.reindex(assets).fillna(0.0).to_numpy().astype(float)
        else:
            if market_weights is not None:
                w_mkt = market_weights.reindex(assets).fillna(0.0).to_numpy().astype(float)
            else:
                w_mkt = np.ones(n) / n
            w_sum = float(w_mkt.sum())
            if w_sum > 1e-10:
                w_mkt = w_mkt / w_sum
            pi = self.risk_aversion * sigma @ w_mkt

        prior_series = pd.Series(pi, index=assets)

        # -- BL posterior --------------------------------------------------
        if len(views) == 0:
            posterior_mu = pi.copy()
            posterior_sigma = sigma.copy()
        else:
            P, Q, confidences = _build_pick_matrix(views, assets)
            omega = _build_omega(P, sigma, tau, confidences, self.omega_method)
            posterior_mu, posterior_sigma = _bl_posterior(pi, sigma, P, Q, omega, tau)

        posterior_series = pd.Series(posterior_mu, index=assets)
        posterior_cov = pd.DataFrame(posterior_sigma, index=assets, columns=assets)

        # -- Optimize on posterior -----------------------------------------
        inner = MeanVarianceOptimizer()
        weights = inner.optimize(posterior_series, posterior_cov, constraints)

        return BlackLittermanResult(
            posterior_returns=posterior_series,
            posterior_covariance=posterior_cov,
            prior_returns=prior_series,
            weights=weights,
            views_used=len(views),
            tau=tau,
            risk_aversion=self.risk_aversion,
        )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _build_pick_matrix(
    views: Sequence[View],
    assets: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return P (K×N), Q (K,), confidences (K,)."""
    asset_idx = {a: i for i, a in enumerate(assets)}
    n = len(assets)
    k = len(views)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    confidences = np.zeros(k)

    for row, view in enumerate(views):
        for asset, weight in zip(view.assets, view.weights):
            j = asset_idx.get(asset)
            if j is None:
                raise ValueError(
                    f"View references asset '{asset}' not found in covariance columns."
                )
            P[row, j] = float(weight)
        Q[row] = view.expected_return
        confidences[row] = view.confidence

    return P, Q, confidences


def _build_omega(
    P: np.ndarray,
    sigma: np.ndarray,
    tau: float,
    confidences: np.ndarray,
    method: str,
) -> np.ndarray:
    """Build view uncertainty matrix Ω (K×K, diagonal).

    proportional: Ω_kk = τ · (PΣP')_kk   [He–Litterman default]
    idzorek:      Ω_kk = τ · (PΣP')_kk · (1-c_k)/c_k
                  → posterior = (1-c)·prior + c·view for a single absolute view
    """
    base_diag = np.diag(tau * P @ sigma @ P.T)  # (K,)

    if method == "proportional":
        omega_diag = base_diag
    else:
        # Idzorek scaling: (1-c)/c maps c=1 → zero uncertainty (certainty)
        scales = np.where(
            confidences >= 1.0 - 1e-9,
            1e-12,  # near-certainty
            (1.0 - confidences) / np.maximum(confidences, 1e-12),
        )
        omega_diag = base_diag * scales

    # Regularize for numerical stability
    return np.diag(omega_diag) + np.eye(len(confidences)) * 1e-12


def _bl_posterior(
    pi: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Black-Litterman posterior mean and covariance.

    μ* = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹π + P'Ω⁻¹Q]
    Σ* = Σ + [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
    """
    n = len(pi)
    eye_n = np.eye(n)
    reg = 1e-10

    tau_sigma = tau * sigma + eye_n * reg
    tau_sigma_inv = np.linalg.solve(tau_sigma, eye_n)

    omega_reg = omega + np.eye(len(Q)) * 1e-12
    omega_inv = np.linalg.solve(omega_reg, np.eye(len(Q)))

    # Posterior precision M = (τΣ)⁻¹ + P'Ω⁻¹P
    M = tau_sigma_inv + P.T @ omega_inv @ P
    M_inv = np.linalg.solve(M + eye_n * reg, eye_n)

    # Posterior mean
    rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
    mu_bl = M_inv @ rhs

    # Posterior covariance (includes estimation uncertainty in μ)
    sigma_bl = sigma + M_inv

    return mu_bl, sigma_bl


# ---------------------------------------------------------------------------
# Convenience: equilibrium returns from market portfolio
# ---------------------------------------------------------------------------


def compute_equilibrium_returns(
    market_weights: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Compute implied equilibrium excess returns via reverse optimization.

    π = δ · Σ · w_mkt

    This is the expected return vector that makes the market portfolio mean-variance
    optimal under risk aversion δ.
    """
    assets = list(covariance.columns)
    sigma = covariance.reindex(index=assets, columns=assets).fillna(0.0).to_numpy()
    w = market_weights.reindex(assets).fillna(0.0).to_numpy()
    w_sum = float(w.sum())
    if w_sum > 1e-10:
        w = w / w_sum
    pi = risk_aversion * sigma @ w
    return pd.Series(pi, index=assets, name="equilibrium_return")

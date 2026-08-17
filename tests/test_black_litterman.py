"""Tests for the Black-Litterman optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qr_haven.portfolio.black_litterman import (
    BlackLittermanOptimizer,
    BlackLittermanResult,
    View,
    _bl_posterior,
    _build_omega,
    _build_pick_matrix,
    compute_equilibrium_returns,
)
from qr_haven.portfolio.optimizers import OptimizerConstraints

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ASSETS_EQ = ["AAPL", "MSFT", "GOOG"]
ASSETS_MAC = ["SPY", "TLT", "GLD", "UUP", "REET"]  # equity, bonds, gold, FX, RE

RNG = np.random.default_rng(7)


def _cov(assets: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(assets)
    A = rng.normal(0, 0.01, (n, n))
    S = A @ A.T + np.eye(n) * 0.04
    return pd.DataFrame(S, index=assets, columns=assets)


def _mkt_weights(assets: list[str]) -> pd.Series:
    n = len(assets)
    w = np.arange(1, n + 1, dtype=float)
    return pd.Series(w / w.sum(), index=assets)


# ---------------------------------------------------------------------------
# View dataclass
# ---------------------------------------------------------------------------


class TestView:
    def test_valid_absolute_view(self):
        v = View(assets=["AAPL"], weights=[1.0], expected_return=0.10)
        assert v.confidence == 0.5

    def test_valid_relative_view(self):
        v = View(assets=["AAPL", "MSFT"], weights=[1.0, -1.0], expected_return=0.03)
        assert len(v.assets) == 2

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="assets has"):
            View(assets=["AAPL"], weights=[1.0, -1.0], expected_return=0.05)

    def test_empty_assets_raises(self):
        with pytest.raises(ValueError, match="at least one asset"):
            View(assets=[], weights=[], expected_return=0.05)

    def test_zero_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            View(assets=["AAPL"], weights=[1.0], expected_return=0.05, confidence=0.0)

    def test_negative_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            View(assets=["AAPL"], weights=[1.0], expected_return=0.05, confidence=-0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            View(assets=["AAPL"], weights=[1.0], expected_return=0.05, confidence=1.01)

    def test_confidence_one_valid(self):
        v = View(assets=["AAPL"], weights=[1.0], expected_return=0.05, confidence=1.0)
        assert v.confidence == 1.0


# ---------------------------------------------------------------------------
# Optimizer init validation
# ---------------------------------------------------------------------------


class TestOptimizerInit:
    def test_bad_risk_aversion_zero(self):
        with pytest.raises(ValueError, match="risk_aversion"):
            BlackLittermanOptimizer(risk_aversion=0.0)

    def test_bad_risk_aversion_negative(self):
        with pytest.raises(ValueError, match="risk_aversion"):
            BlackLittermanOptimizer(risk_aversion=-1.0)

    def test_bad_tau_zero(self):
        with pytest.raises(ValueError, match="tau"):
            BlackLittermanOptimizer(tau=0.0)

    def test_bad_omega_method(self):
        with pytest.raises(ValueError, match="omega_method"):
            BlackLittermanOptimizer(omega_method="unknown")

    def test_defaults(self):
        opt = BlackLittermanOptimizer()
        assert opt.risk_aversion == 2.5
        assert opt.tau == 0.05
        assert opt.omega_method == "proportional"
        assert opt.views == []
        assert opt.market_weights is None


# ---------------------------------------------------------------------------
# Equilibrium returns
# ---------------------------------------------------------------------------


class TestEquilibriumReturns:
    def test_single_asset_formula(self):
        """π = δ × σ² × 1 for a single asset with full weight."""
        delta = 3.0
        sigma_sq = 0.04  # 20% vol
        cov = pd.DataFrame([[sigma_sq]], index=["A"], columns=["A"])
        w = pd.Series({"A": 1.0})
        pi = compute_equilibrium_returns(w, cov, risk_aversion=delta)
        assert abs(float(pi["A"]) - delta * sigma_sq) < 1e-10

    def test_equal_weights_proportional_to_variance(self):
        """With equal market weights, π_i ∝ sum of row i of Σ."""
        cov = _cov(ASSETS_EQ)
        w = pd.Series({a: 1.0 / 3 for a in ASSETS_EQ})
        pi = compute_equilibrium_returns(w, cov, risk_aversion=1.0)
        expected = cov.to_numpy() @ w.to_numpy()
        np.testing.assert_allclose(pi.to_numpy(), expected, atol=1e-10)

    def test_pi_scales_with_risk_aversion(self):
        cov = _cov(ASSETS_EQ)
        w = _mkt_weights(ASSETS_EQ)
        pi1 = compute_equilibrium_returns(w, cov, risk_aversion=1.0)
        pi3 = compute_equilibrium_returns(w, cov, risk_aversion=3.0)
        np.testing.assert_allclose(pi3.to_numpy(), 3.0 * pi1.to_numpy(), atol=1e-10)

    def test_returns_series_with_correct_index(self):
        cov = _cov(ASSETS_EQ)
        w = _mkt_weights(ASSETS_EQ)
        pi = compute_equilibrium_returns(w, cov)
        assert list(pi.index) == ASSETS_EQ


# ---------------------------------------------------------------------------
# No views → posterior equals prior
# ---------------------------------------------------------------------------


class TestNoViews:
    def test_posterior_equals_prior_no_views(self):
        cov = _cov(ASSETS_EQ)
        w = _mkt_weights(ASSETS_EQ)
        opt = BlackLittermanOptimizer(market_weights=w)
        result = opt.optimize_with_views(cov, views=[])
        np.testing.assert_allclose(
            result.posterior_returns.to_numpy(),
            result.prior_returns.to_numpy(),
            atol=1e-10,
        )

    def test_views_used_zero(self):
        cov = _cov(ASSETS_EQ)
        opt = BlackLittermanOptimizer()
        result = opt.optimize_with_views(cov, views=[])
        assert result.views_used == 0

    def test_weights_sum_to_one_no_views(self):
        cov = _cov(ASSETS_EQ)
        opt = BlackLittermanOptimizer()
        result = opt.optimize_with_views(cov, views=[])
        assert abs(float(result.weights.sum()) - 1.0) < 1e-7


# ---------------------------------------------------------------------------
# Idzorek confidence blending (algebraically exact for single absolute view)
# ---------------------------------------------------------------------------


class TestIdzorekBlending:
    """For a single absolute view on a single asset with Idzorek omega:
    μ_BL = (1-c) × π + c × Q"""

    def _setup_single(self, confidence: float, delta: float = 2.5) -> tuple:
        sigma_sq = 0.04
        cov = pd.DataFrame([[sigma_sq]], index=["A"], columns=["A"])
        w = pd.Series({"A": 1.0})
        pi = delta * sigma_sq  # scalar
        view = View(assets=["A"], weights=[1.0], expected_return=0.12, confidence=confidence)
        return cov, w, pi, view

    def test_confidence_half_is_midpoint(self):
        cov, w, pi, view = self._setup_single(confidence=0.5)
        opt = BlackLittermanOptimizer(
            risk_aversion=2.5, tau=0.05, omega_method="idzorek", market_weights=w
        )
        result = opt.optimize_with_views(cov, views=[view])
        expected = 0.5 * pi + 0.5 * 0.12
        assert abs(float(result.posterior_returns["A"]) - expected) < 1e-5

    def test_high_confidence_near_view(self):
        cov, w, pi, view = self._setup_single(confidence=0.99)
        opt = BlackLittermanOptimizer(
            risk_aversion=2.5, tau=0.05, omega_method="idzorek", market_weights=w
        )
        result = opt.optimize_with_views(cov, views=[view])
        # Posterior should be very close to view (0.12)
        assert abs(float(result.posterior_returns["A"]) - 0.12) < 0.01

    def test_low_confidence_near_prior(self):
        cov, w, pi, view = self._setup_single(confidence=0.01)
        opt = BlackLittermanOptimizer(
            risk_aversion=2.5, tau=0.05, omega_method="idzorek", market_weights=w
        )
        result = opt.optimize_with_views(cov, views=[view])
        # Posterior should be close to prior
        assert abs(float(result.posterior_returns["A"]) - pi) < 0.02

    def test_higher_confidence_closer_to_view(self):
        """Posterior should move monotonically toward the view as confidence increases."""
        cov, w, pi, _ = self._setup_single(confidence=0.5)
        view_ret = 0.12
        opt = BlackLittermanOptimizer(
            risk_aversion=2.5, tau=0.05, omega_method="idzorek", market_weights=w
        )
        results = []
        for c in (0.1, 0.3, 0.5, 0.7, 0.9):
            view = View(assets=["A"], weights=[1.0], expected_return=view_ret, confidence=c)
            r = opt.optimize_with_views(cov, views=[view])
            results.append(float(r.posterior_returns["A"]))
        # Should increase monotonically toward view_ret
        for a, b in zip(results, results[1:]):
            assert b > a - 1e-6


# ---------------------------------------------------------------------------
# Relative view
# ---------------------------------------------------------------------------


class TestRelativeView:
    def test_relative_view_shifts_spread(self):
        """A relative view (A outperforms B) should increase A's posterior return
        relative to B's, compared to the prior."""
        assets = ["A", "B"]
        cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.04]], index=assets, columns=assets
        )
        w = pd.Series({"A": 0.5, "B": 0.5})
        view = View(assets=["A", "B"], weights=[1.0, -1.0], expected_return=0.05)
        opt = BlackLittermanOptimizer(
            market_weights=w, tau=0.05, omega_method="proportional"
        )
        result = opt.optimize_with_views(cov, views=[view])
        prior_spread = float(result.prior_returns["A"] - result.prior_returns["B"])
        post_spread = float(result.posterior_returns["A"] - result.posterior_returns["B"])
        # Posterior spread should be >= prior spread (pulled toward view of 5%)
        assert post_spread >= prior_spread - 1e-8

    def test_views_used_count(self):
        assets = ASSETS_EQ
        cov = _cov(assets)
        views = [
            View(assets=["AAPL", "MSFT"], weights=[1.0, -1.0], expected_return=0.02),
            View(assets=["GOOG"], weights=[1.0], expected_return=0.08),
        ]
        opt = BlackLittermanOptimizer()
        result = opt.optimize_with_views(cov, views=views)
        assert result.views_used == 2


# ---------------------------------------------------------------------------
# Proportional vs Idzorek Omega
# ---------------------------------------------------------------------------


class TestOmegaMethods:
    def test_both_methods_produce_valid_weights(self):
        cov = _cov(ASSETS_EQ)
        w = _mkt_weights(ASSETS_EQ)
        views = [View(assets=["AAPL"], weights=[1.0], expected_return=0.15)]
        for method in ("proportional", "idzorek"):
            opt = BlackLittermanOptimizer(
                market_weights=w, omega_method=method
            )
            result = opt.optimize_with_views(cov, views=views)
            assert abs(float(result.weights.sum()) - 1.0) < 1e-7

    def test_idzorek_sensitivity_greater_than_proportional(self):
        """At confidence=0.9, Idzorek should pull the posterior closer to the view
        than proportional (which ignores confidence)."""
        cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        w = pd.Series({"A": 1.0})
        view = View(assets=["A"], weights=[1.0], expected_return=0.20, confidence=0.9)
        pi = float(compute_equilibrium_returns(w, cov)["A"])

        prop_opt = BlackLittermanOptimizer(
            risk_aversion=2.5, tau=0.05, omega_method="proportional", market_weights=w
        )
        idz_opt = BlackLittermanOptimizer(
            risk_aversion=2.5, tau=0.05, omega_method="idzorek", market_weights=w
        )
        prop_post = float(prop_opt.optimize_with_views(cov, views=[view]).posterior_returns["A"])
        idz_post = float(idz_opt.optimize_with_views(cov, views=[view]).posterior_returns["A"])
        # Idzorek with high confidence should be closer to the view (0.20)
        assert abs(idz_post - 0.20) < abs(prop_post - 0.20) + 1e-6


# ---------------------------------------------------------------------------
# Drop-in optimize() API
# ---------------------------------------------------------------------------


class TestOptimizeAPI:
    def test_optimize_returns_series(self):
        cov = _cov(ASSETS_EQ)
        er = pd.Series({a: 0.08 for a in ASSETS_EQ})
        opt = BlackLittermanOptimizer()
        w = opt.optimize(er, cov)
        assert isinstance(w, pd.Series)

    def test_optimize_weights_sum_to_one(self):
        cov = _cov(ASSETS_EQ)
        er = pd.Series({a: 0.08 for a in ASSETS_EQ})
        opt = BlackLittermanOptimizer()
        w = opt.optimize(er, cov)
        assert abs(float(w.sum()) - 1.0) < 1e-7

    def test_optimize_weights_non_negative_long_only(self):
        cov = _cov(ASSETS_EQ)
        er = pd.Series({a: 0.08 for a in ASSETS_EQ})
        opt = BlackLittermanOptimizer()
        w = opt.optimize(er, cov)
        assert np.all(w.values >= -1e-8)

    def test_optimize_index_matches_covariance(self):
        cov = _cov(ASSETS_EQ)
        er = pd.Series({a: 0.08 for a in ASSETS_EQ})
        opt = BlackLittermanOptimizer()
        w = opt.optimize(er, cov)
        assert list(w.index) == ASSETS_EQ

    def test_optimize_with_diagnostics_shape(self):
        from qr_haven.portfolio.optimizers import OptimizerDiagnostics
        cov = _cov(ASSETS_EQ)
        er = pd.Series({a: 0.08 for a in ASSETS_EQ})
        opt = BlackLittermanOptimizer()
        weights, diag = opt.optimize_with_diagnostics(er, cov)
        assert isinstance(diag, OptimizerDiagnostics)
        assert abs(float(weights.sum()) - 1.0) < 1e-7

    def test_instance_views_used_in_optimize(self):
        """Views set at construction time must be reflected in optimize()."""
        cov = _cov(ASSETS_EQ)
        er = pd.Series({a: 0.08 for a in ASSETS_EQ})
        views = [View(assets=["AAPL"], weights=[1.0], expected_return=0.20)]
        opt_no_view = BlackLittermanOptimizer()
        opt_with_view = BlackLittermanOptimizer(views=views)
        w_no = opt_no_view.optimize(er, cov)
        w_yes = opt_with_view.optimize(er, cov)
        # With a bullish AAPL view, AAPL weight should be at least as high
        assert float(w_yes["AAPL"]) >= float(w_no["AAPL"]) - 1e-6


# ---------------------------------------------------------------------------
# Multi-asset class portfolio
# ---------------------------------------------------------------------------


class TestMultiAssetClass:
    def test_five_asset_class_portfolio(self):
        """Equity, bonds, gold, FX, real estate — a realistic MAB setup."""
        cov = _cov(ASSETS_MAC, seed=99)
        mkt_w = _mkt_weights(ASSETS_MAC)

        views = [
            # Equities outperform bonds by 4%
            View(assets=["SPY", "TLT"], weights=[1.0, -1.0], expected_return=0.04),
            # Gold absolute return 6%
            View(assets=["GLD"], weights=[1.0], expected_return=0.06, confidence=0.6),
        ]
        opt = BlackLittermanOptimizer(
            risk_aversion=3.0,
            tau=0.05,
            omega_method="idzorek",
            market_weights=mkt_w,
        )
        result = opt.optimize_with_views(cov, views=views)
        assert result.views_used == 2
        assert abs(float(result.weights.sum()) - 1.0) < 1e-7
        assert all(result.weights.index == ASSETS_MAC)

    def test_cross_asset_views_do_not_crash(self):
        cov = _cov(ASSETS_MAC)
        views = [
            View(assets=["SPY", "GLD"], weights=[1.0, -1.0], expected_return=0.03),
            View(assets=["TLT", "UUP"], weights=[0.6, -0.4], expected_return=-0.01),
        ]
        opt = BlackLittermanOptimizer()
        result = opt.optimize_with_views(cov, views=views)
        assert not result.weights.isna().any()


# ---------------------------------------------------------------------------
# BlackLittermanResult
# ---------------------------------------------------------------------------


class TestBlackLittermanResult:
    def _result(self) -> BlackLittermanResult:
        cov = _cov(ASSETS_EQ)
        opt = BlackLittermanOptimizer(market_weights=_mkt_weights(ASSETS_EQ))
        return opt.optimize_with_views(
            cov,
            views=[View(assets=["AAPL"], weights=[1.0], expected_return=0.10)],
        )

    def test_summary_keys(self):
        r = self._result()
        s = r.summary()
        for key in (
            "views_used",
            "tau",
            "risk_aversion",
            "prior_return_mean",
            "posterior_return_mean",
            "portfolio_weight_sum",
            "effective_holdings",
        ):
            assert key in s, f"Missing key: {key}"

    def test_posterior_covariance_shape(self):
        r = self._result()
        n = len(ASSETS_EQ)
        assert r.posterior_covariance.shape == (n, n)

    def test_posterior_covariance_symmetric(self):
        r = self._result()
        sigma = r.posterior_covariance.to_numpy()
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)

    def test_posterior_covariance_positive_definite(self):
        r = self._result()
        eigvals = np.linalg.eigvalsh(r.posterior_covariance.to_numpy())
        assert np.all(eigvals > -1e-8)

    def test_weights_index_matches_assets(self):
        r = self._result()
        assert list(r.weights.index) == ASSETS_EQ


# ---------------------------------------------------------------------------
# Error: unknown view asset
# ---------------------------------------------------------------------------


class TestViewErrors:
    def test_unknown_asset_in_view_raises(self):
        cov = _cov(ASSETS_EQ)
        bad_view = View(assets=["NVDA"], weights=[1.0], expected_return=0.15)
        opt = BlackLittermanOptimizer()
        with pytest.raises(ValueError, match="not found"):
            opt.optimize_with_views(cov, views=[bad_view])

    def test_partial_unknown_asset_raises(self):
        cov = _cov(ASSETS_EQ)
        bad_view = View(assets=["AAPL", "NVDA"], weights=[1.0, -1.0], expected_return=0.05)
        opt = BlackLittermanOptimizer()
        with pytest.raises(ValueError, match="not found"):
            opt.optimize_with_views(cov, views=[bad_view])


# ---------------------------------------------------------------------------
# Math internals
# ---------------------------------------------------------------------------


class TestBLPosteriorMath:
    def test_single_asset_proportional_formula(self):
        """Verify posterior mean formula algebraically for 1 asset, 1 absolute view."""
        sigma_sq = 0.04
        delta = 2.5
        tau = 0.05
        q = 0.12

        sigma = np.array([[sigma_sq]])
        pi = np.array([delta * sigma_sq])
        P = np.array([[1.0]])
        Q = np.array([q])

        # proportional omega: ω = τ × σ²
        omega_diag = np.array([tau * sigma_sq])
        omega = np.diag(omega_diag) + np.eye(1) * 1e-12

        mu, _ = _bl_posterior(pi, sigma, P, Q, omega, tau)

        # Analytic formula:
        # M = 1/(τσ²) + 1/ω,  μ* = M⁻¹ × (π/(τσ²) + q/ω)
        M_inv = 1.0 / (1.0 / (tau * sigma_sq) + 1.0 / float(omega_diag[0]))
        expected = M_inv * (float(pi[0]) / (tau * sigma_sq) + q / float(omega_diag[0]))
        assert abs(float(mu[0]) - expected) < 1e-8

    def test_posterior_covariance_adds_uncertainty(self):
        """Σ_BL = Σ + M⁻¹ should be larger than Σ in each diagonal element."""
        sigma = np.diag([0.04, 0.04, 0.04])
        pi = np.array([0.05, 0.07, 0.06])
        P = np.array([[1.0, -1.0, 0.0]])
        Q = np.array([0.03])
        confidences = np.array([0.5])
        omega = _build_omega(P, sigma, 0.05, confidences, "proportional")

        _, sigma_bl = _bl_posterior(pi, sigma, P, Q, omega, 0.05)
        assert np.all(np.diag(sigma_bl) >= np.diag(sigma) - 1e-10)

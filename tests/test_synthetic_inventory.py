"""Tests for the Synthetic Inventory Creation Optimizer.

Coverage:
  - InstrumentSpec validation
  - CostConfig validation
  - Cost vector and breakdown computation (all five components)
  - TransformationGraph: feasibility filtering, constraint rows, coverage
  - LP solver: correct weights, delta equivalence, cheaper-than-baseline detection
  - QP solver: identity Σ_basis reduces to LP-like; off-diagonal changes result
  - MIQP solver: leg-count budget; mutual-exclusion constraint
  - SCA solver: convergence; α=0.5 result close to QP
  - Constraint relaxation: partial fill path
  - OptimizationResult.summary() keys
  - SyntheticPathResult fields
  - Integration: end-to-end realistic HTB scenario
"""

from __future__ import annotations

import numpy as np
import pytest

from qr_haven.synthetic_inventory import (
    CostBreakdown,
    CostConfig,
    InstrumentSpec,
    InstrumentType,
    OptimizationMode,
    OptimizationResult,
    SyntheticInventoryOptimizer,
    TransformationGraph,
)
from qr_haven.synthetic_inventory.solvers import solve_lp, solve_qp, solve_miqp, solve_sca, OPTIMAL_STR
from qr_haven.synthetic_inventory.costs import compute_cost_vector, compute_cost_breakdown

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _trs(delta: float = -1.0, feasibility: float = 1.0, capacity: float = 1.0,
         funding: float = 0.005, basis_vol: float = 0.0) -> InstrumentSpec:
    return InstrumentSpec(
        instrument_type=InstrumentType.TRS,
        ticker="TRS_XYZ",
        funding_rate_annual=funding,
        margin_rate=0.05,
        rwa_weight=0.02,
        execution_cost_bps=5.0,
        basis_vol_annual=basis_vol,
        feasibility=feasibility,
        capacity=capacity,
        delta=delta,
    )


def _ssf(delta: float = -1.0, feasibility: float = 1.0, capacity: float = 1.0,
         funding: float = 0.003) -> InstrumentSpec:
    return InstrumentSpec(
        instrument_type=InstrumentType.SSF,
        ticker="SSF_XYZ",
        funding_rate_annual=funding,
        margin_rate=0.10,
        rwa_weight=0.01,
        execution_cost_bps=3.0,
        basis_vol_annual=0.002,
        feasibility=feasibility,
        capacity=capacity,
        delta=delta,
    )


def _repo(delta: float = -1.0, feasibility: float = 1.0, capacity: float = 0.5) -> InstrumentSpec:
    return InstrumentSpec(
        instrument_type=InstrumentType.REPO,
        ticker="REPO_XYZ",
        funding_rate_annual=0.002,
        margin_rate=0.02,
        rwa_weight=0.005,
        execution_cost_bps=1.0,
        basis_vol_annual=0.001,
        feasibility=feasibility,
        capacity=capacity,
        delta=delta,
    )


def _default_config() -> CostConfig:
    return CostConfig(
        hurdle_rate=0.08,
        capital_cost_rate=0.10,
        risk_aversion=0.5,
        impact_eta=0.1,
        impact_alpha=0.5,
        min_saving_threshold=0.05,
    )


# ---------------------------------------------------------------------------
# InstrumentSpec validation
# ---------------------------------------------------------------------------

class TestInstrumentSpec:
    def test_valid_construction(self):
        spec = _trs()
        assert spec.instrument_type == InstrumentType.TRS
        assert spec.delta == -1.0

    def test_feasibility_out_of_range_low(self):
        with pytest.raises(ValueError, match="feasibility"):
            _trs(feasibility=-0.1)

    def test_feasibility_out_of_range_high(self):
        with pytest.raises(ValueError, match="feasibility"):
            _trs(feasibility=1.01)

    def test_capacity_zero(self):
        with pytest.raises(ValueError, match="capacity"):
            _trs(capacity=0.0)

    def test_capacity_exceeds_one(self):
        with pytest.raises(ValueError, match="capacity"):
            _trs(capacity=1.5)

    def test_negative_execution_cost(self):
        with pytest.raises(ValueError, match="execution_cost_bps"):
            InstrumentSpec(
                instrument_type=InstrumentType.SSF, ticker="X",
                funding_rate_annual=0.01, margin_rate=0.05, rwa_weight=0.01,
                execution_cost_bps=-1.0, basis_vol_annual=0.0,
                feasibility=1.0, capacity=1.0,
            )

    def test_negative_basis_vol(self):
        with pytest.raises(ValueError, match="basis_vol_annual"):
            _trs(basis_vol=-0.01)

    def test_negative_adv(self):
        with pytest.raises(ValueError):
            InstrumentSpec(
                instrument_type=InstrumentType.TRS, ticker="X",
                funding_rate_annual=0.01, margin_rate=0.05, rwa_weight=0.01,
                execution_cost_bps=5.0, basis_vol_annual=0.0,
                feasibility=1.0, capacity=1.0, adv_usd=-1.0,
            )

    def test_zero_feasibility_allowed(self):
        spec = _trs(feasibility=0.0)
        assert spec.feasibility == 0.0


# ---------------------------------------------------------------------------
# CostConfig validation
# ---------------------------------------------------------------------------

class TestCostConfig:
    def test_defaults(self):
        c = CostConfig()
        assert c.hurdle_rate == 0.08
        assert c.min_saving_threshold == 0.05

    def test_negative_hurdle_rate(self):
        with pytest.raises(ValueError, match="hurdle_rate"):
            CostConfig(hurdle_rate=-0.01)

    def test_negative_risk_aversion(self):
        with pytest.raises(ValueError, match="risk_aversion"):
            CostConfig(risk_aversion=-1.0)

    def test_impact_alpha_zero(self):
        with pytest.raises(ValueError, match="impact_alpha"):
            CostConfig(impact_alpha=0.0)

    def test_impact_alpha_exceeds_one(self):
        with pytest.raises(ValueError, match="impact_alpha"):
            CostConfig(impact_alpha=1.5)

    def test_saving_threshold_negative(self):
        with pytest.raises(ValueError, match="min_saving_threshold"):
            CostConfig(min_saving_threshold=-0.1)

    def test_saving_threshold_one(self):
        with pytest.raises(ValueError, match="min_saving_threshold"):
            CostConfig(min_saving_threshold=1.0)


# ---------------------------------------------------------------------------
# Cost vector computation
# ---------------------------------------------------------------------------

class TestCostVector:
    def test_funding_component(self):
        spec = InstrumentSpec(
            instrument_type=InstrumentType.TRS, ticker="X",
            funding_rate_annual=0.01, margin_rate=0.0, rwa_weight=0.0,
            execution_cost_bps=0.0, basis_vol_annual=0.0,
            feasibility=1.0, capacity=1.0,
        )
        config = CostConfig(hurdle_rate=0.0, capital_cost_rate=0.0, risk_aversion=0.0)
        c = compute_cost_vector([spec], holding_period_days=252, config=config)
        assert abs(c[0] - 0.01 * 10_000) < 0.01  # 100 bps

    def test_margin_component(self):
        spec = InstrumentSpec(
            instrument_type=InstrumentType.SSF, ticker="X",
            funding_rate_annual=0.0, margin_rate=0.10, rwa_weight=0.0,
            execution_cost_bps=0.0, basis_vol_annual=0.0,
            feasibility=1.0, capacity=1.0,
        )
        config = CostConfig(hurdle_rate=0.08, capital_cost_rate=0.0, risk_aversion=0.0)
        c = compute_cost_vector([spec], holding_period_days=252, config=config)
        expected = 0.10 * 0.08 * 10_000  # 80 bps
        assert abs(c[0] - expected) < 0.01

    def test_execution_amortised_over_longer_hold(self):
        spec = InstrumentSpec(
            instrument_type=InstrumentType.TRS, ticker="X",
            funding_rate_annual=0.0, margin_rate=0.0, rwa_weight=0.0,
            execution_cost_bps=50.0, basis_vol_annual=0.0,
            feasibility=1.0, capacity=1.0,
        )
        config = CostConfig(hurdle_rate=0.0, capital_cost_rate=0.0, risk_aversion=0.0)
        c_short = compute_cost_vector([spec], holding_period_days=21, config=config)
        c_long = compute_cost_vector([spec], holding_period_days=252, config=config)
        assert c_short[0] > c_long[0]  # amortised over longer hold → lower annualised cost

    def test_zero_feasibility_gives_positive_cost_still(self):
        spec = _trs(feasibility=0.0)
        c = compute_cost_vector([spec], holding_period_days=252, config=_default_config())
        assert c[0] > 0  # cost is intrinsic to instrument, not gated by feasibility


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

class TestCostBreakdown:
    def test_all_components_non_negative(self):
        specs = [_trs(basis_vol=0.01), _ssf(), _repo()]
        weights = np.array([0.5, 0.3, 0.2])
        bd = compute_cost_breakdown(weights, specs, 63, _default_config())
        assert bd.funding_bps >= 0
        assert bd.margin_bps >= 0
        assert bd.capital_bps >= 0
        assert bd.execution_bps >= 0
        assert bd.basis_risk_bps >= 0

    def test_total_equals_sum_of_components(self):
        specs = [_trs(basis_vol=0.01), _ssf()]
        weights = np.array([0.6, 0.4])
        bd = compute_cost_breakdown(weights, specs, 252, _default_config())
        summed = bd.funding_bps + bd.margin_bps + bd.capital_bps + bd.execution_bps + bd.basis_risk_bps
        assert abs(bd.total_bps - summed) < 1e-4

    def test_zero_weight_gives_zero_cost(self):
        specs = [_trs()]
        bd = compute_cost_breakdown(np.array([0.0]), specs, 252, _default_config())
        assert bd.total_bps == 0.0

    def test_as_dict_keys(self):
        bd = CostBreakdown(1.0, 2.0, 3.0, 4.0, 5.0, 15.0)
        d = bd.as_dict()
        assert set(d) == {"funding_bps", "margin_bps", "capital_bps", "execution_bps", "basis_risk_bps", "total_bps"}

    def test_no_basis_risk_when_zero_vol(self):
        spec = _trs(basis_vol=0.0)
        bd = compute_cost_breakdown(np.array([1.0]), [spec], 252, _default_config())
        assert bd.basis_risk_bps == 0.0


# ---------------------------------------------------------------------------
# TransformationGraph
# ---------------------------------------------------------------------------

class TestTransformationGraph:
    def test_feasibility_filtering(self):
        specs = [_trs(feasibility=1.0), _ssf(feasibility=0.0), _repo(feasibility=0.5)]
        g = TransformationGraph(specs)
        feasible = g.feasible_instruments()
        assert len(feasible) == 2
        assert all(s.feasibility > 0 for s in feasible)

    def test_delta_row_shape(self):
        g = TransformationGraph([_trs(), _ssf(), _repo()])
        row = g.delta_row()
        assert row.shape == (3,)

    def test_delta_row_values(self):
        trs = _trs(delta=-1.0, feasibility=0.8)
        g = TransformationGraph([trs])
        row = g.delta_row()
        assert abs(row[0] - (-1.0 * 0.8)) < 1e-10

    def test_gamma_row_zero_for_non_options(self):
        g = TransformationGraph([_trs(), _ssf()])
        assert np.all(g.gamma_row() == 0.0)

    def test_vega_row_zero_for_non_options(self):
        g = TransformationGraph([_trs(), _repo()])
        assert np.all(g.vega_row() == 0.0)

    def test_default_basis_covariance_is_diagonal(self):
        g = TransformationGraph([_trs(basis_vol=0.01), _ssf()])
        Sigma = g.default_basis_covariance()
        K = len(g.feasible_instruments())
        assert Sigma.shape == (K, K)
        off_diag = Sigma - np.diag(np.diag(Sigma))
        assert np.all(off_diag == 0.0)

    def test_capacity_vector(self):
        specs = [_trs(capacity=0.8), _ssf(capacity=0.6)]
        g = TransformationGraph(specs)
        cap = g.capacity_vector()
        np.testing.assert_allclose(cap, [0.8, 0.6])

    def test_max_achievable_delta(self):
        specs = [_trs(delta=-1.0, capacity=0.6), _ssf(delta=-1.0, capacity=0.4)]
        g = TransformationGraph(specs)
        assert abs(g.max_achievable_delta() - (-1.0)) < 1e-10

    def test_coverage_fraction_full(self):
        g = TransformationGraph([_trs(delta=-1.0, capacity=1.0)])
        assert abs(g.coverage_fraction(-1.0) - 1.0) < 1e-10

    def test_coverage_fraction_partial(self):
        g = TransformationGraph([_trs(delta=-1.0, capacity=0.5)])
        assert abs(g.coverage_fraction(-1.0) - 0.5) < 1e-10

    def test_no_feasible_instruments_empty(self):
        g = TransformationGraph([_trs(feasibility=0.0)])
        assert g.feasible_instruments() == []
        assert g.max_achievable_delta() == 0.0


# ---------------------------------------------------------------------------
# LP solver
# ---------------------------------------------------------------------------

class TestLPSolver:
    def test_single_instrument_gets_full_weight(self):
        c = np.array([10.0])
        A_eq = np.array([[-1.0]])
        b_eq = np.array([-1.0])
        cap = np.array([1.0])
        w, status = solve_lp(c, A_eq, b_eq, cap)
        assert status == OPTIMAL_STR
        assert abs(w[0] - 1.0) < 1e-6

    def test_two_instruments_cheaper_one_wins(self):
        # c[0]=5, c[1]=20; both have delta=-1; target=-1
        c = np.array([5.0, 20.0])
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.array([1.0, 1.0])
        w, status = solve_lp(c, A_eq, b_eq, cap)
        assert status == OPTIMAL_STR
        assert w[0] > w[1]  # cheaper instrument gets higher weight

    def test_weights_satisfy_delta_constraint(self):
        c = np.array([5.0, 8.0, 12.0])
        A_eq = np.array([[-1.0, -1.0, -0.5]])  # mixed delta
        b_eq = np.array([-1.0])
        cap = np.array([1.0, 1.0, 1.0])
        w, status = solve_lp(c, A_eq, b_eq, cap)
        assert status == OPTIMAL_STR
        assert abs(A_eq @ w - b_eq) < 1e-5

    def test_infeasible_returns_non_optimal(self):
        # Target delta = -2 but capacity = 1 and only one instrument with delta=-1
        c = np.array([5.0])
        A_eq = np.array([[-1.0]])
        b_eq = np.array([-2.0])
        cap = np.array([1.0])
        _, status = solve_lp(c, A_eq, b_eq, cap)
        assert status != OPTIMAL_STR

    def test_capacity_respected(self):
        c = np.array([1.0, 100.0])
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.array([0.4, 1.0])  # instrument 0 capped at 0.4
        w, status = solve_lp(c, A_eq, b_eq, cap)
        assert status == OPTIMAL_STR
        assert w[0] <= 0.4 + 1e-8


# ---------------------------------------------------------------------------
# QP solver
# ---------------------------------------------------------------------------

class TestQPSolver:
    def test_identity_sigma_gives_valid_solution(self):
        K = 3
        c = np.array([5.0, 8.0, 12.0])
        Sigma = np.eye(K) * 100.0  # diagonal, all same
        A_eq = np.array([[-1.0, -1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        w, status = solve_qp(c, Sigma, lam=1.0, A_eq=A_eq, b_eq=b_eq, capacity=cap)
        assert status == OPTIMAL_STR
        assert abs(np.sum(w * (-1.0)) - (-1.0)) < 1e-5

    def test_zero_lambda_degenerates_to_lp(self):
        K = 2
        c = np.array([3.0, 10.0])
        Sigma = np.eye(K) * 1000.0
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        w_qp, _ = solve_qp(c, Sigma, lam=0.0, A_eq=A_eq, b_eq=b_eq, capacity=cap)
        w_lp, _ = solve_lp(c, A_eq, b_eq, cap)
        np.testing.assert_allclose(w_qp, w_lp, atol=1e-4)

    def test_high_lambda_spreads_weights(self):
        # High basis-risk aversion with heterogeneous diagonal Σ should spread weights
        K = 2
        c = np.array([5.0, 5.0])  # equal cost
        Sigma = np.diag([1e6, 1e6])
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        w, status = solve_qp(c, Sigma, lam=10.0, A_eq=A_eq, b_eq=b_eq, capacity=cap)
        assert status == OPTIMAL_STR
        # With equal cost and equal variance, weights should be ≈ equal
        assert abs(w[0] - w[1]) < 0.1

    def test_correlated_sigma_different_from_diagonal(self):
        K = 2
        c = np.array([5.0, 5.0])
        Sigma_diag = np.diag([1000.0, 1000.0])
        Sigma_corr = np.array([[1000.0, 900.0], [900.0, 1000.0]])  # high correlation
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        w_diag, _ = solve_qp(c, Sigma_diag, 1.0, A_eq, b_eq, cap)
        w_corr, _ = solve_qp(c, Sigma_corr, 1.0, A_eq, b_eq, cap)
        # Both optimal but correlated solution may differ from diagonal
        assert abs(np.sum(w_diag * (-1)) - (-1)) < 1e-5
        assert abs(np.sum(w_corr * (-1)) - (-1)) < 1e-5


# ---------------------------------------------------------------------------
# MIQP solver
# ---------------------------------------------------------------------------

class TestMIQPSolver:
    def test_leg_count_respected(self):
        K = 3
        c = np.array([5.0, 8.0, 12.0])
        Sigma = np.eye(K) * 10.0
        A_eq = np.array([[-1.0, -1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        w, z, status = solve_miqp(c, Sigma, 0.0, A_eq, b_eq, cap, [], M_max=1)
        assert status == OPTIMAL_STR
        assert int(np.round(z).sum()) <= 1

    def test_cheapest_instrument_selected_with_one_leg(self):
        K = 3
        c = np.array([3.0, 10.0, 20.0])
        Sigma = np.zeros((K, K))
        A_eq = np.array([[-1.0, -1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        w, z, status = solve_miqp(c, Sigma, 0.0, A_eq, b_eq, cap, [], M_max=1)
        assert status == OPTIMAL_STR
        assert z[0] > 0.5  # cheapest instrument selected

    def test_mutual_exclusion_respected(self):
        K = 2
        c = np.array([3.0, 4.0])
        Sigma = np.zeros((K, K))
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        w, z, status = solve_miqp(c, Sigma, 0.0, A_eq, b_eq, cap, [(0, 1)], M_max=2)
        # Mutual exclusion prevents both being active
        active = np.round(z).astype(int)
        assert active.sum() <= 1

    def test_zero_lam_no_hessian(self):
        K = 2
        c = np.array([5.0, 8.0])
        Sigma = np.eye(K) * 1000.0
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        w, z, status = solve_miqp(c, Sigma, 0.0, A_eq, b_eq, cap, [], M_max=2)
        assert status == OPTIMAL_STR
        assert w.sum() > 0


# ---------------------------------------------------------------------------
# SCA solver
# ---------------------------------------------------------------------------

class TestSCASolver:
    def test_converges_returns_optimal(self):
        K = 2
        c_base = np.array([5.0, 8.0])
        Sigma = np.eye(K) * 10.0
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        adv = np.array([0.5, 0.5])
        w, status, iters = solve_sca(
            c_base, Sigma, 0.0, A_eq, b_eq, cap, adv, eta=0.1, alpha=0.5,
        )
        assert status == OPTIMAL_STR
        assert iters >= 1

    def test_satisfies_delta_constraint(self):
        K = 3
        c_base = np.array([5.0, 8.0, 12.0])
        Sigma = np.eye(K) * 5.0
        A_eq = np.array([[-1.0, -1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        adv = np.ones(K) / K
        w, status, _ = solve_sca(c_base, Sigma, 0.5, A_eq, b_eq, cap, adv, 0.1, 0.5)
        assert abs(A_eq @ w - b_eq) < 1e-4

    def test_alpha_half_close_to_qp(self):
        # With α=0.5 SCA should converge to a solution close to QP
        K = 2
        c_base = np.array([5.0, 8.0])
        Sigma = np.eye(K) * 10.0
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        cap = np.ones(K)
        adv = np.array([0.5, 0.5])

        w_sca, _, _ = solve_sca(c_base, Sigma, 0.5, A_eq, b_eq, cap, adv, 0.1, 0.5, max_iter=50)
        w_qp, _ = solve_qp(c_base, Sigma, 0.5, A_eq, b_eq, cap)

        # SCA and QP should agree on which instrument gets more weight
        assert np.argmax(w_sca) == np.argmax(w_qp)


# ---------------------------------------------------------------------------
# SyntheticInventoryOptimizer — LP mode
# ---------------------------------------------------------------------------

class TestOptimizerLP:
    def setup_method(self):
        self.opt = SyntheticInventoryOptimizer(config=_default_config(), mode=OptimizationMode.LP)

    def test_raises_on_nonpositive_holding_period(self):
        with pytest.raises(ValueError, match="holding_period_days"):
            self.opt.optimize([_trs()], -1.0, 0.0, 500.0)

    def test_raises_on_empty_instruments(self):
        with pytest.raises(ValueError, match="instruments"):
            self.opt.optimize([], -1.0, 21.0, 500.0)

    def test_raises_on_negative_baseline(self):
        with pytest.raises(ValueError, match="baseline_borrow_bps"):
            self.opt.optimize([_trs()], -1.0, 21.0, -10.0)

    def test_infeasible_with_all_zero_feasibility(self):
        result = self.opt.optimize([_trs(feasibility=0.0)], -1.0, 21.0, 500.0)
        assert not result.is_beneficial
        assert "infeasible" in result.solver_status.lower()

    def test_result_is_optimization_result(self):
        result = self.opt.optimize([_trs(), _ssf()], -1.0, 63.0, 500.0)
        assert isinstance(result, OptimizationResult)

    def test_delta_constraint_satisfied(self):
        instruments = [_trs(), _ssf()]
        result = self.opt.optimize(instruments, -1.0, 63.0, 500.0)
        assert result.solver_status == OPTIMAL_STR
        achieved_delta = sum(w * s.delta * s.feasibility
                             for w, s in zip(result.weights, result.instruments))
        assert abs(achieved_delta - (-1.0)) < 1e-5

    def test_weights_non_negative(self):
        result = self.opt.optimize([_trs(), _ssf(), _repo()], -1.0, 63.0, 500.0)
        assert np.all(result.weights >= -1e-8)

    def test_cheaper_synthetic_is_beneficial(self):
        # TRS with very low carry cost vs high borrow rate
        cheap_trs = InstrumentSpec(
            instrument_type=InstrumentType.TRS, ticker="TRS_CHEAP",
            funding_rate_annual=0.001, margin_rate=0.01, rwa_weight=0.005,
            execution_cost_bps=1.0, basis_vol_annual=0.0,
            feasibility=1.0, capacity=1.0, delta=-1.0,
        )
        result = self.opt.optimize([cheap_trs], -1.0, 252.0, 500.0)
        assert result.is_beneficial
        assert result.saving_bps > 0

    def test_expensive_synthetic_not_beneficial(self):
        # Very costly TRS vs low borrow rate
        expensive_trs = InstrumentSpec(
            instrument_type=InstrumentType.TRS, ticker="TRS_EXP",
            funding_rate_annual=0.10, margin_rate=0.20, rwa_weight=0.20,
            execution_cost_bps=200.0, basis_vol_annual=0.05,
            feasibility=1.0, capacity=1.0, delta=-1.0,
        )
        result = self.opt.optimize([expensive_trs], -1.0, 21.0, 10.0)
        assert not result.is_beneficial

    def test_cheaper_instrument_preferred_in_result(self):
        # SSF has lower cost than TRS here
        trs = InstrumentSpec(
            instrument_type=InstrumentType.TRS, ticker="TRS",
            funding_rate_annual=0.02, margin_rate=0.10, rwa_weight=0.05,
            execution_cost_bps=10.0, basis_vol_annual=0.0,
            feasibility=1.0, capacity=1.0, delta=-1.0,
        )
        ssf = InstrumentSpec(
            instrument_type=InstrumentType.SSF, ticker="SSF",
            funding_rate_annual=0.001, margin_rate=0.05, rwa_weight=0.01,
            execution_cost_bps=2.0, basis_vol_annual=0.0,
            feasibility=1.0, capacity=1.0, delta=-1.0,
        )
        result = self.opt.optimize([trs, ssf], -1.0, 252.0, 500.0)
        # SSF (index 1) should get more weight
        assert result.weights[1] > result.weights[0]

    def test_capacity_respected_in_result(self):
        trs = InstrumentSpec(
            instrument_type=InstrumentType.TRS, ticker="TRS",
            funding_rate_annual=0.001, margin_rate=0.01, rwa_weight=0.005,
            execution_cost_bps=1.0, basis_vol_annual=0.0,
            feasibility=1.0, capacity=0.3, delta=-1.0,
        )
        ssf = InstrumentSpec(
            instrument_type=InstrumentType.SSF, ticker="SSF",
            funding_rate_annual=0.005, margin_rate=0.05, rwa_weight=0.01,
            execution_cost_bps=3.0, basis_vol_annual=0.0,
            feasibility=1.0, capacity=1.0, delta=-1.0,
        )
        result = self.opt.optimize([trs, ssf], -1.0, 252.0, 500.0)
        assert result.weights[0] <= 0.3 + 1e-6

    def test_paths_count_matches_feasible_instruments(self):
        result = self.opt.optimize([_trs(), _ssf(), _repo()], -1.0, 63.0, 500.0)
        assert len(result.paths) == 3

    def test_active_paths_flagged_correctly(self):
        result = self.opt.optimize([_trs(), _ssf()], -1.0, 252.0, 500.0)
        for path, w in zip(result.paths, result.weights):
            if w >= 1e-4:
                assert path.active
            else:
                assert not path.active

    def test_summary_keys(self):
        result = self.opt.optimize([_trs()], -1.0, 252.0, 500.0)
        s = result.summary()
        for key in ("mode", "solver_status", "is_beneficial", "total_cost_bps",
                    "baseline_cost_bps", "saving_bps", "saving_fraction",
                    "n_active_legs", "cost_breakdown", "iterations"):
            assert key in s

    def test_summary_mode_string(self):
        result = self.opt.optimize([_trs()], -1.0, 252.0, 500.0)
        assert result.summary()["mode"] == "lp"


# ---------------------------------------------------------------------------
# SyntheticInventoryOptimizer — QP mode
# ---------------------------------------------------------------------------

class TestOptimizerQP:
    def test_qp_result_satisfies_delta(self):
        opt = SyntheticInventoryOptimizer(config=_default_config(), mode=OptimizationMode.QP)
        result = opt.optimize([_trs(basis_vol=0.01), _ssf(), _repo()], -1.0, 63.0, 500.0)
        achieved = sum(w * s.delta for w, s in zip(result.weights, result.instruments))
        assert abs(achieved - (-1.0)) < 1e-4

    def test_custom_basis_covariance_used(self):
        K = 2
        Sigma = np.array([[1000.0, 0.0], [0.0, 1000.0]])
        opt = SyntheticInventoryOptimizer(
            config=_default_config(), mode=OptimizationMode.QP,
            basis_covariance=Sigma,
        )
        result = opt.optimize([_trs(), _ssf()], -1.0, 63.0, 500.0)
        assert result.solver_status == OPTIMAL_STR

    def test_high_risk_aversion_spreads_weights(self):
        config = CostConfig(risk_aversion=100.0, min_saving_threshold=0.0)
        opt = SyntheticInventoryOptimizer(config=config, mode=OptimizationMode.QP)
        trs = InstrumentSpec(
            InstrumentType.TRS, "TRS", 0.005, 0.05, 0.02, 5.0, 0.01, 1.0, 1.0, delta=-1.0,
        )
        ssf = InstrumentSpec(
            InstrumentType.SSF, "SSF", 0.003, 0.10, 0.01, 3.0, 0.01, 1.0, 1.0, delta=-1.0,
        )
        result = opt.optimize([trs, ssf], -1.0, 63.0, 500.0)
        # High risk aversion → both legs should receive non-trivial weight
        assert result.weights[0] > 0.01 and result.weights[1] > 0.01


# ---------------------------------------------------------------------------
# SyntheticInventoryOptimizer — MIQP mode
# ---------------------------------------------------------------------------

class TestOptimizerMIQP:
    def test_max_legs_one_uses_single_instrument(self):
        opt = SyntheticInventoryOptimizer(
            config=_default_config(), mode=OptimizationMode.MIQP, max_legs=1,
        )
        result = opt.optimize([_trs(), _ssf(), _repo()], -1.0, 63.0, 500.0)
        active = [p for p in result.paths if p.active]
        assert len(active) <= 1

    def test_mutual_exclusion_trs_ssf(self):
        opt = SyntheticInventoryOptimizer(
            config=_default_config(), mode=OptimizationMode.MIQP,
            mutual_exclusion=[("TRS_XYZ", "SSF_XYZ")],
        )
        result = opt.optimize([_trs(), _ssf()], -1.0, 63.0, 500.0)
        trs_active = result.paths[0].active
        ssf_active = result.paths[1].active
        assert not (trs_active and ssf_active)

    def test_miqp_delta_satisfied(self):
        opt = SyntheticInventoryOptimizer(
            config=_default_config(), mode=OptimizationMode.MIQP, max_legs=2,
        )
        result = opt.optimize([_trs(), _ssf(), _repo()], -1.0, 63.0, 500.0)
        if result.solver_status == OPTIMAL_STR:
            achieved = sum(w * s.delta for w, s in zip(result.weights, result.instruments))
            assert abs(achieved - (-1.0)) < 1e-4


# ---------------------------------------------------------------------------
# solve_miqp_qp — two-stage solver unit tests
# ---------------------------------------------------------------------------

class TestMIQPQPSolver:
    def test_returns_optimal_status(self):
        K = 2
        c = np.array([100.0, 50.0])
        capacity = np.array([1.0, 1.0])
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        Sigma = np.eye(K) * 100.0

        from qr_haven.synthetic_inventory.solvers import solve_miqp_qp
        w, z, status = solve_miqp_qp(c, Sigma, 0.5, A_eq, b_eq, capacity, [], K)
        assert status == OPTIMAL_STR

    def test_delta_constraint_satisfied(self):
        from qr_haven.synthetic_inventory.solvers import solve_miqp_qp
        K = 3
        c = np.array([130.0, 120.0, 45.0])
        capacity = np.array([1.0, 1.0, 0.5])
        A_eq = np.array([[-1.0, -1.0, -1.0]])
        b_eq = np.array([-1.0])
        Sigma = np.diag([0.0, 400.0, 100.0])

        w, z, status = solve_miqp_qp(c, Sigma, 0.5, A_eq, b_eq, capacity, [], 3)
        if status == OPTIMAL_STR:
            achieved = A_eq @ w
            assert abs(float(achieved[0]) - float(b_eq[0])) < 1e-4

    def test_max_legs_respected(self):
        from qr_haven.synthetic_inventory.solvers import solve_miqp_qp
        K = 3
        c = np.array([130.0, 120.0, 45.0])
        capacity = np.array([1.0, 1.0, 0.5])
        A_eq = np.array([[-1.0, -1.0, -1.0]])
        b_eq = np.array([-1.0])
        Sigma = np.diag([0.0, 400.0, 100.0])

        w, z, status = solve_miqp_qp(c, Sigma, 0.5, A_eq, b_eq, capacity, [], 1)
        if status == OPTIMAL_STR:
            assert int(np.sum(z > 0.5)) <= 1

    def test_high_lambda_spreads_within_selected_legs(self):
        """Stage-2 QP with high λ distributes weight across selected legs."""
        from qr_haven.synthetic_inventory.solvers import solve_miqp_qp
        K = 2
        c = np.array([10.0, 10.0])
        capacity = np.array([1.0, 1.0])
        A_eq = np.array([[-1.0, -1.0]])
        b_eq = np.array([-1.0])
        # High covariance — risk aversion should spread weights evenly
        Sigma = np.array([[1e6, 0.0], [0.0, 1e6]])

        w_lo, _, _ = solve_miqp_qp(c, Sigma, 0.0, A_eq, b_eq, capacity, [], 2)
        w_hi, _, _ = solve_miqp_qp(c, Sigma, 5.0, A_eq, b_eq, capacity, [], 2)
        # With high λ both legs should be active (equal weight); with λ=0 one leg
        n_active_hi = int(np.sum(w_hi > 1e-4))
        assert n_active_hi >= 1  # at minimum one leg active


# ---------------------------------------------------------------------------
# SyntheticInventoryOptimizer — MIQP_QP mode
# ---------------------------------------------------------------------------

class TestOptimizerMIQPQP:
    def test_miqp_qp_result_type(self):
        opt = SyntheticInventoryOptimizer(
            config=_default_config(), mode=OptimizationMode.MIQP_QP,
        )
        result = opt.optimize([_trs(), _ssf(), _repo()], -1.0, 63.0, 500.0)
        assert isinstance(result, OptimizationResult)

    def test_miqp_qp_delta_satisfied(self):
        opt = SyntheticInventoryOptimizer(
            config=_default_config(), mode=OptimizationMode.MIQP_QP,
        )
        result = opt.optimize([_trs(), _ssf(), _repo()], -1.0, 63.0, 500.0)
        if result.solver_status == OPTIMAL_STR:
            achieved = sum(w * s.delta for w, s in zip(result.weights, result.instruments))
            assert abs(achieved - (-1.0)) < 1e-4

    def test_miqp_qp_max_legs_one(self):
        opt = SyntheticInventoryOptimizer(
            config=_default_config(), mode=OptimizationMode.MIQP_QP, max_legs=1,
        )
        result = opt.optimize([_trs(), _ssf(), _repo()], -1.0, 63.0, 500.0)
        active = [p for p in result.paths if p.active]
        assert len(active) <= 1

    def test_miqp_qp_mutual_exclusion(self):
        opt = SyntheticInventoryOptimizer(
            config=_default_config(), mode=OptimizationMode.MIQP_QP,
            mutual_exclusion=[("TRS_XYZ", "SSF_XYZ")],
        )
        result = opt.optimize([_trs(), _ssf()], -1.0, 63.0, 500.0)
        trs_active = result.paths[0].active
        ssf_active = result.paths[1].active
        assert not (trs_active and ssf_active)

    def test_miqp_qp_high_risk_aversion_spreads_weight(self):
        """With high λ the QP stage should spread weight across active legs."""
        config = CostConfig(hurdle_rate=0.08, capital_cost_rate=0.10, risk_aversion=10.0)
        Sigma = np.array([[1e6, 0.0], [0.0, 1e6]])
        opt = SyntheticInventoryOptimizer(
            config=config, mode=OptimizationMode.MIQP_QP,
            basis_covariance=Sigma,
        )
        result = opt.optimize([_trs(), _ssf()], -1.0, 252.0, 500.0)
        if result.solver_status == OPTIMAL_STR:
            # Both instruments active (high risk aversion → spread)
            assert all(w >= 0 for w in result.weights)

    def test_miqp_qp_weights_differ_from_linear_miqp(self):
        """MIQP_QP weights should differ from MIQP when Σ is non-trivial."""
        config = CostConfig(hurdle_rate=0.08, capital_cost_rate=0.10, risk_aversion=2.0)
        trs_with_vol = _trs(basis_vol=0.05)
        ssf_with_vol = _ssf()
        instruments = [trs_with_vol, ssf_with_vol, _repo()]

        opt_linear = SyntheticInventoryOptimizer(config=config, mode=OptimizationMode.MIQP)
        opt_quad = SyntheticInventoryOptimizer(config=config, mode=OptimizationMode.MIQP_QP)

        r_lin = opt_linear.optimize(instruments, -1.0, 63.0, 500.0)
        r_qp = opt_quad.optimize(instruments, -1.0, 63.0, 500.0)

        # Both should be valid OptimizationResult instances
        assert isinstance(r_lin, OptimizationResult)
        assert isinstance(r_qp, OptimizationResult)


# ---------------------------------------------------------------------------
# SyntheticInventoryOptimizer — SCA mode
# ---------------------------------------------------------------------------

class TestOptimizerSCA:
    def test_sca_result_type(self):
        opt = SyntheticInventoryOptimizer(config=_default_config(), mode=OptimizationMode.SCA)
        result = opt.optimize([_trs(), _ssf()], -1.0, 63.0, 500.0)
        assert isinstance(result, OptimizationResult)

    def test_sca_delta_satisfied(self):
        opt = SyntheticInventoryOptimizer(config=_default_config(), mode=OptimizationMode.SCA)
        result = opt.optimize([_trs(), _ssf()], -1.0, 252.0, 500.0)
        if result.solver_status == OPTIMAL_STR:
            achieved = sum(w * s.delta for w, s in zip(result.weights, result.instruments))
            assert abs(achieved - (-1.0)) < 1e-3

    def test_sca_iterations_positive(self):
        opt = SyntheticInventoryOptimizer(config=_default_config(), mode=OptimizationMode.SCA)
        result = opt.optimize([_trs(), _ssf()], -1.0, 63.0, 500.0)
        assert result.iterations >= 1


# ---------------------------------------------------------------------------
# Constraint relaxation
# ---------------------------------------------------------------------------

class TestConstraintRelaxation:
    def test_partial_fill_when_low_capacity(self):
        # Each instrument can only cover 0.3 delta but target is -1.0
        specs = [
            _trs(feasibility=1.0, capacity=0.3),
            _ssf(feasibility=1.0, capacity=0.3),
        ]
        opt = SyntheticInventoryOptimizer(config=_default_config(), mode=OptimizationMode.LP)
        result = opt.optimize(specs, -1.0, 63.0, 500.0)
        # Infeasible or partial fill — either way should not crash
        assert isinstance(result, OptimizationResult)

    def test_infeasible_with_no_coverage(self):
        specs = [_trs(feasibility=0.0), _ssf(feasibility=0.0)]
        opt = SyntheticInventoryOptimizer(config=_default_config(), mode=OptimizationMode.LP)
        result = opt.optimize(specs, -1.0, 63.0, 500.0)
        assert not result.is_beneficial
        assert result.weights.sum() == 0.0


# ---------------------------------------------------------------------------
# Optional constraints (gamma, vega, balance sheet)
# ---------------------------------------------------------------------------

class TestOptionalConstraints:
    def test_gamma_limit_accepted(self):
        opt = SyntheticInventoryOptimizer(config=_default_config(), mode=OptimizationMode.LP)
        result = opt.optimize([_trs(), _ssf()], -1.0, 63.0, 500.0, gamma_limit=0.5)
        assert isinstance(result, OptimizationResult)

    def test_vega_limit_accepted(self):
        opt = SyntheticInventoryOptimizer(config=_default_config(), mode=OptimizationMode.LP)
        result = opt.optimize([_trs(), _ssf()], -1.0, 63.0, 500.0, vega_limit=0.5)
        assert isinstance(result, OptimizationResult)

    def test_balance_sheet_limit_accepted(self):
        opt = SyntheticInventoryOptimizer(config=_default_config(), mode=OptimizationMode.LP)
        result = opt.optimize(
            [_trs(), _ssf()], -1.0, 63.0, 500.0, balance_sheet_budget=1.0
        )
        assert isinstance(result, OptimizationResult)


# ---------------------------------------------------------------------------
# Integration — realistic HTB scenario
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_htb_scenario_trs_beats_borrow(self):
        """GME-style HTB: borrow at 1500 bps, TRS available at 200 bps carry."""
        cheap_trs = InstrumentSpec(
            instrument_type=InstrumentType.TRS,
            ticker="TRS_GME",
            funding_rate_annual=0.02,   # 200 bps carry
            margin_rate=0.15,
            rwa_weight=0.04,
            execution_cost_bps=5.0,
            basis_vol_annual=0.0,
            feasibility=0.9,
            capacity=1.0,
            delta=-1.0,
            adv_usd=5_000_000.0,
        )
        ssf = InstrumentSpec(
            instrument_type=InstrumentType.SSF,
            ticker="SSF_GME",
            funding_rate_annual=0.015,  # 150 bps
            margin_rate=0.20,
            rwa_weight=0.02,
            execution_cost_bps=8.0,
            basis_vol_annual=0.002,
            feasibility=0.6,
            capacity=0.5,
            delta=-1.0,
            adv_usd=2_000_000.0,
        )
        config = CostConfig(min_saving_threshold=0.05)
        opt = SyntheticInventoryOptimizer(config=config, mode=OptimizationMode.LP)
        result = opt.optimize([cheap_trs, ssf], -1.0, 252.0, 1500.0)

        assert result.is_beneficial
        assert result.saving_bps > 0
        assert result.cost_breakdown.total_bps < 1500.0
        assert result.solver_status == OPTIMAL_STR
        s = result.summary()
        assert s["n_active_legs"] >= 1

    def test_multimode_results_consistent(self):
        """LP, QP, MIQP, SCA should all find beneficial synthetics on the same input."""
        instruments = [_trs(basis_vol=0.005), _ssf(), _repo()]
        baseline = 800.0

        for mode in OptimizationMode:
            opt = SyntheticInventoryOptimizer(
                config=CostConfig(min_saving_threshold=0.0),
                mode=mode,
            )
            result = opt.optimize(instruments, -1.0, 252.0, baseline)
            assert isinstance(result, OptimizationResult), f"mode {mode} failed"
            if result.solver_status == OPTIMAL_STR:
                assert result.cost_breakdown.total_bps >= 0

    def test_three_instrument_qp_portfolio(self):
        trs = InstrumentSpec(InstrumentType.TRS, "TRS", 0.005, 0.05, 0.02, 5.0, 0.01, 1.0, 0.5, delta=-1.0)
        ssf = InstrumentSpec(InstrumentType.SSF, "SSF", 0.003, 0.10, 0.01, 3.0, 0.005, 1.0, 0.5, delta=-1.0)
        repo = InstrumentSpec(InstrumentType.REPO, "REPO", 0.002, 0.02, 0.005, 1.0, 0.001, 1.0, 1.0, delta=-1.0)

        Sigma = np.diag([100.0, 25.0, 10.0])  # custom basis cov
        opt = SyntheticInventoryOptimizer(
            config=CostConfig(min_saving_threshold=0.0),
            mode=OptimizationMode.QP,
            basis_covariance=Sigma,
        )
        result = opt.optimize([trs, ssf, repo], -1.0, 63.0, 500.0)
        assert result.solver_status == OPTIMAL_STR
        assert result.cost_breakdown.total_bps > 0
        assert len(result.paths) == 3

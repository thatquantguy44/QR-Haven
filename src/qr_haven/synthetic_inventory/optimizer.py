"""Synthetic inventory creation optimizer — public API class."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from qr_haven.synthetic_inventory.costs import (
    CostBreakdown,
    CostConfig,
    compute_cost_breakdown,
    compute_cost_vector,
    _BPS,
)
from qr_haven.synthetic_inventory.graph import TransformationGraph
from qr_haven.synthetic_inventory.instruments import InstrumentSpec
from qr_haven.synthetic_inventory import solvers

_ACTIVE_THRESHOLD = 1e-4


class OptimizationMode(str, Enum):
    LP = "lp"      # Phase 1: linear costs, independent basis risks
    QP = "qp"      # Phase 1b: correlated Σ_basis
    MIQP = "miqp"  # Phase 1b: binary instrument selection
    SCA = "sca"    # Phase 1b: power-law impact via sequential convex approximation


@dataclass(frozen=True)
class SyntheticPathResult:
    """One active leg in the optimised synthetic replication portfolio."""

    instrument_type: str
    ticker: str
    notional_fraction: float
    annualised_cost_bps: float
    basis_vol_bps: float
    active: bool


@dataclass(frozen=True)
class OptimizationResult:
    """Complete output of a single SyntheticInventoryOptimizer.optimize() call."""

    weights: np.ndarray
    instruments: list[InstrumentSpec]
    cost_breakdown: CostBreakdown
    baseline_cost_bps: float
    saving_bps: float
    saving_fraction: float
    is_beneficial: bool
    mode: OptimizationMode
    solver_status: str
    paths: list[SyntheticPathResult]
    iterations: int

    def summary(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "solver_status": self.solver_status,
            "is_beneficial": self.is_beneficial,
            "total_cost_bps": self.cost_breakdown.total_bps,
            "baseline_cost_bps": round(self.baseline_cost_bps, 4),
            "saving_bps": round(self.saving_bps, 4),
            "saving_fraction": round(self.saving_fraction, 6),
            "n_active_legs": sum(1 for p in self.paths if p.active),
            "cost_breakdown": self.cost_breakdown.as_dict(),
            "iterations": self.iterations,
        }


class SyntheticInventoryOptimizer:
    """
    Optimise a synthetic replication portfolio subject to exposure equivalence.

    The optimizer selects instrument weights minimising total carrying cost:

        C(w) = Funding + Margin + Capital + Execution + BasisRisk

    subject to:
        Σ delta_i · w_i = target_delta           (exposure equivalence)
        0 ≤ w_i ≤ capacity_i                     (instrument limits)
        [optional]  Σ gamma_i · w_i ≤ gamma_limit
        [optional]  Σ vega_i  · w_i ≤ vega_limit
        [optional]  Σ rwa_i   · w_i ≤ bs_budget

    The result is compared against the baseline (external HTB borrow) and flagged
    as beneficial when the saving exceeds ``config.min_saving_threshold``.

    Parameters
    ----------
    config : CostConfig, optional
        Cost model parameters.  Defaults to CostConfig().
    mode : OptimizationMode
        Solver formulation.  LP is the fastest; QP / MIQP / SCA add precision.
    basis_covariance : np.ndarray, optional
        Full Σ_basis matrix (K×K) for QP / MIQP / SCA modes.  When omitted, a
        diagonal matrix of per-instrument basis variances is used.
    max_legs : int, optional
        Maximum number of active legs (MIQP only).
    mutual_exclusion : list of (str, str)
        Pairs of tickers that cannot both be active (MIQP only).
    sca_max_iter : int
        Maximum SCA outer iterations.
    sca_tol : float
        L1 convergence tolerance for SCA.
    """

    def __init__(
        self,
        config: CostConfig | None = None,
        mode: OptimizationMode = OptimizationMode.LP,
        basis_covariance: np.ndarray | None = None,
        max_legs: int | None = None,
        mutual_exclusion: list[tuple[str, str]] | None = None,
        sca_max_iter: int = 20,
        sca_tol: float = 1e-6,
    ) -> None:
        self.config = config or CostConfig()
        self.mode = mode
        self.basis_covariance = basis_covariance
        self.max_legs = max_legs
        self.mutual_exclusion: list[tuple[str, str]] = mutual_exclusion or []
        self.sca_max_iter = sca_max_iter
        self.sca_tol = sca_tol

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        instruments: list[InstrumentSpec],
        target_delta: float,
        holding_period_days: float,
        baseline_borrow_bps: float,
        gamma_limit: float | None = None,
        vega_limit: float | None = None,
        balance_sheet_budget: float | None = None,
    ) -> OptimizationResult:
        """Find the minimum-cost synthetic portfolio matching target_delta.

        Parameters
        ----------
        instruments : list[InstrumentSpec]
            All candidate legs (feasibility-gated internally).
        target_delta : float
            Required first-order exposure (e.g. -1.0 for a full synthetic short).
        holding_period_days : float
            Expected holding period in calendar days (> 0).
        baseline_borrow_bps : float
            Current HTB rate in bps per annum — the do-nothing baseline.
        gamma_limit : float, optional
            Upper bound on total portfolio gamma.
        vega_limit : float, optional
            Upper bound on total portfolio vega.
        balance_sheet_budget : float, optional
            Cap on Σ rwa_i · capital_rate · w_i (balance-sheet usage).
        """
        if holding_period_days <= 0:
            raise ValueError("holding_period_days must be positive")
        if not instruments:
            raise ValueError("instruments list cannot be empty")
        if baseline_borrow_bps < 0:
            raise ValueError("baseline_borrow_bps cannot be negative")

        graph = TransformationGraph(instruments)
        feasible = graph.feasible_instruments()

        if not feasible:
            return self._infeasible_result(
                instruments, baseline_borrow_bps, "no feasible instruments"
            )

        A_eq, b_eq, capacity, A_ineq, b_lo, b_hi = self._build_constraints(
            graph, target_delta, gamma_limit, vega_limit, balance_sheet_budget
        )

        c = compute_cost_vector(feasible, holding_period_days, self.config)
        weights, status, iters = self._dispatch(
            feasible, c, A_eq, b_eq, capacity, A_ineq, b_lo, b_hi, graph
        )

        if status != solvers.OPTIMAL_STR:
            return self._try_relaxations(
                instruments, target_delta, holding_period_days, baseline_borrow_bps, status
            )

        return self._build_result(weights, feasible, holding_period_days, baseline_borrow_bps, status, iters)

    # ------------------------------------------------------------------
    # Constraint builder
    # ------------------------------------------------------------------

    def _build_constraints(
        self,
        graph: TransformationGraph,
        target_delta: float,
        gamma_limit: float | None,
        vega_limit: float | None,
        balance_sheet_budget: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return A_eq, b_eq, capacity, A_ineq, b_lo, b_hi."""
        capacity = graph.capacity_vector()

        # Equality: delta equivalence
        A_eq = graph.delta_row().reshape(1, -1)
        b_eq = np.array([target_delta])

        # Inequality rows (upper bounds)
        ineq_rows: list[np.ndarray] = []
        lo_vals: list[float] = []
        hi_vals: list[float] = []

        if gamma_limit is not None:
            ineq_rows.append(graph.gamma_row())
            lo_vals.append(-solvers._INF)
            hi_vals.append(gamma_limit)

        if vega_limit is not None:
            ineq_rows.append(graph.vega_row())
            lo_vals.append(-solvers._INF)
            hi_vals.append(vega_limit)

        if balance_sheet_budget is not None:
            ineq_rows.append(graph.capital_row(self.config))
            lo_vals.append(-solvers._INF)
            hi_vals.append(balance_sheet_budget)

        if ineq_rows:
            A_ineq = np.vstack(ineq_rows)
            b_lo = np.array(lo_vals)
            b_hi = np.array(hi_vals)
        else:
            K = len(capacity)
            A_ineq = np.zeros((0, K))
            b_lo = np.array([])
            b_hi = np.array([])

        return A_eq, b_eq, capacity, A_ineq, b_lo, b_hi

    # ------------------------------------------------------------------
    # Solver dispatch
    # ------------------------------------------------------------------

    def _augmented_eq(
        self, A_eq: np.ndarray, b_eq: np.ndarray,
        A_ineq: np.ndarray, b_lo: np.ndarray, b_hi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge strict equality rows with any finite inequality rows for LP/QP calls."""
        if A_ineq.shape[0] == 0:
            return A_eq, b_eq

        # For LP/QP we pass inequality rows directly to addRow via _add_ineq_row;
        # here we just pass the equality block to _add_eq_rows and will separately
        # inject the inequality rows before h.run().  To keep the solver helpers
        # simple (they only accept A_eq / b_eq), we stack finite-both-sided rows
        # as equalities (midpoint) when both bounds are finite, otherwise skip.
        # In practice our inequality rows are one-sided (gamma/vega/BS upper bounds).
        return A_eq, b_eq  # inequality rows handled in _dispatch_with_ineq

    def _dispatch(
        self,
        feasible: list[InstrumentSpec],
        c: np.ndarray,
        A_eq: np.ndarray,
        b_eq: np.ndarray,
        capacity: np.ndarray,
        A_ineq: np.ndarray,
        b_lo: np.ndarray,
        b_hi: np.ndarray,
        graph: TransformationGraph,
    ) -> tuple[np.ndarray, str, int]:
        K = len(feasible)
        lam = self.config.risk_aversion
        Sigma = self._covariance(graph, K)

        if A_ineq.shape[0] > 0:
            # Run via raw HiGHS model with both equality and inequality rows
            return self._solve_with_ineq(c, Sigma, A_eq, b_eq, A_ineq, b_lo, b_hi, capacity, K, lam)

        if self.mode == OptimizationMode.LP:
            w, status = solvers.solve_lp(c, A_eq, b_eq, capacity)
            return w, status, 1

        if self.mode == OptimizationMode.QP:
            w, status = solvers.solve_qp(c, Sigma, lam, A_eq, b_eq, capacity)
            return w, status, 1

        if self.mode == OptimizationMode.MIQP:
            me = self._me_indices(feasible)
            M = self.max_legs if self.max_legs is not None else K
            w, _, status = solvers.solve_miqp(c, Sigma, lam, A_eq, b_eq, capacity, me, M)
            return w, status, 1

        if self.mode == OptimizationMode.SCA:
            adv = np.array([s.adv_usd for s in feasible], dtype=float)
            adv_frac = adv / adv.sum()
            w, status, iters = solvers.solve_sca(
                c, Sigma, lam, A_eq, b_eq, capacity,
                adv_frac, self.config.impact_eta, self.config.impact_alpha,
                self.sca_max_iter, self.sca_tol,
            )
            return w, status, iters

        raise ValueError(f"Unknown mode: {self.mode}")

    def _solve_with_ineq(
        self,
        c: np.ndarray,
        Sigma: np.ndarray,
        A_eq: np.ndarray,
        b_eq: np.ndarray,
        A_ineq: np.ndarray,
        b_lo: np.ndarray,
        b_hi: np.ndarray,
        capacity: np.ndarray,
        K: int,
        lam: float,
    ) -> tuple[np.ndarray, str, int]:
        """Build a HiGHS model with both equality and inequality constraints."""
        import highspy
        h = solvers._new_model(K, np.zeros(K), capacity, c)
        if self.mode != OptimizationMode.LP:
            solvers._pass_hessian(h, K, lam, Sigma)
        solvers._add_eq_rows(h, A_eq, b_eq)
        for i in range(A_ineq.shape[0]):
            solvers._add_ineq_row(h, A_ineq[i], float(b_lo[i]), float(b_hi[i]))
        h.run()
        w, status = solvers._extract(h, K)
        return w, status, 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _covariance(self, graph: TransformationGraph, K: int) -> np.ndarray:
        if self.basis_covariance is not None and self.basis_covariance.shape == (K, K):
            return self.basis_covariance
        return graph.default_basis_covariance()

    def _me_indices(self, feasible: list[InstrumentSpec]) -> list[tuple[int, int]]:
        idx = {s.ticker: i for i, s in enumerate(feasible)}
        return [
            (idx[a], idx[b])
            for a, b in self.mutual_exclusion
            if a in idx and b in idx
        ]

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(
        self,
        weights: np.ndarray,
        feasible: list[InstrumentSpec],
        holding_period_days: float,
        baseline_borrow_bps: float,
        status: str,
        iters: int,
    ) -> OptimizationResult:
        cost = compute_cost_breakdown(weights, feasible, holding_period_days, self.config)
        saving = baseline_borrow_bps - cost.total_bps
        frac = saving / baseline_borrow_bps if baseline_borrow_bps > 0 else 0.0

        paths = [
            SyntheticPathResult(
                instrument_type=s.instrument_type.value,
                ticker=s.ticker,
                notional_fraction=float(w),
                annualised_cost_bps=float(
                    w * (s.funding_rate_annual + s.margin_rate * self.config.hurdle_rate
                         + s.rwa_weight * self.config.capital_cost_rate) * _BPS
                ),
                basis_vol_bps=float(s.basis_vol_annual * _BPS),
                active=float(w) >= _ACTIVE_THRESHOLD,
            )
            for w, s in zip(weights, feasible)
        ]

        return OptimizationResult(
            weights=weights,
            instruments=feasible,
            cost_breakdown=cost,
            baseline_cost_bps=baseline_borrow_bps,
            saving_bps=saving,
            saving_fraction=frac,
            is_beneficial=frac >= self.config.min_saving_threshold,
            mode=self.mode,
            solver_status=status,
            paths=paths,
            iterations=iters,
        )

    def _infeasible_result(
        self,
        instruments: list[InstrumentSpec],
        baseline_borrow_bps: float,
        reason: str,
    ) -> OptimizationResult:
        empty = CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return OptimizationResult(
            weights=np.zeros(len(instruments)),
            instruments=instruments,
            cost_breakdown=empty,
            baseline_cost_bps=baseline_borrow_bps,
            saving_bps=0.0,
            saving_fraction=0.0,
            is_beneficial=False,
            mode=self.mode,
            solver_status=f"infeasible:{reason}",
            paths=[],
            iterations=0,
        )

    def _try_relaxations(
        self,
        instruments: list[InstrumentSpec],
        target_delta: float,
        holding_period_days: float,
        baseline_borrow_bps: float,
        original_status: str,
    ) -> OptimizationResult:
        """Relaxation 1: partial fill — scale target_delta to what is achievable."""
        graph = TransformationGraph(instruments)
        feasible = graph.feasible_instruments()
        if not feasible:
            return self._infeasible_result(
                instruments, baseline_borrow_bps,
                f"no feasible instruments (original:{original_status})"
            )

        coverage = graph.coverage_fraction(target_delta)
        if coverage <= 0:
            return self._infeasible_result(
                instruments, baseline_borrow_bps,
                f"zero coverage (original:{original_status})"
            )

        relaxed_delta = target_delta * coverage * 0.99
        A_eq = graph.delta_row().reshape(1, -1)
        b_eq = np.array([relaxed_delta])
        capacity = graph.capacity_vector()
        c = compute_cost_vector(feasible, holding_period_days, self.config)

        K = len(feasible)
        Sigma = self._covariance(graph, K)
        lam = self.config.risk_aversion

        if self.mode in (OptimizationMode.LP,):
            w, status = solvers.solve_lp(c, A_eq, b_eq, capacity)
        else:
            w, status = solvers.solve_qp(c, Sigma, lam, A_eq, b_eq, capacity)

        if status == solvers.OPTIMAL_STR:
            return self._build_result(
                w, feasible, holding_period_days, baseline_borrow_bps,
                f"partial_fill:{status}", 1,
            )

        return self._infeasible_result(
            instruments, baseline_borrow_bps,
            f"infeasible after relaxation (original:{original_status})"
        )

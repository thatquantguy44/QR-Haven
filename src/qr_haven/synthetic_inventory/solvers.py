"""HiGHS-backed LP, QP, MIQP, and SCA solvers for synthetic inventory.

All functions return (weights, status_str) or (weights, selection_flags, status_str).
A status of str(HighsModelStatus.kOptimal) signals a feasible optimal solution.
"""

from __future__ import annotations

import numpy as np

import highspy
from scipy.sparse import triu as _triu

_INF = 1e30
_OPTIMAL = highspy.HighsModelStatus.kOptimal
OPTIMAL_STR = str(_OPTIMAL)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _new_model(K: int, lb: np.ndarray, ub: np.ndarray, c: np.ndarray) -> highspy.Highs:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.addVars(K, lb.astype(np.float64), ub.astype(np.float64))
    h.changeColsCost(K, np.arange(K, dtype=np.int32), c.astype(np.float64))
    return h


def _add_eq_rows(h: highspy.Highs, A: np.ndarray, b: np.ndarray) -> None:
    for i in range(A.shape[0]):
        nz = np.flatnonzero(A[i])
        if len(nz) == 0:
            continue
        h.addRow(float(b[i]), float(b[i]), len(nz), nz.tolist(), A[i, nz].tolist())


def _add_ineq_row(
    h: highspy.Highs,
    row: np.ndarray,
    lo: float = -_INF,
    hi: float = _INF,
) -> None:
    nz = np.flatnonzero(row)
    if len(nz) == 0:
        return
    h.addRow(lo, hi, len(nz), nz.tolist(), row[nz].tolist())


def _pass_hessian(h: highspy.Highs, K: int, lam: float, Sigma: np.ndarray) -> None:
    """Pass H = 2λΣ (upper triangular, CSR) to HiGHS (objective = 0.5·wᵀHw)."""
    if lam <= 0:
        return
    H = _triu(2.0 * lam * Sigma, format="csr")
    h.passHessian(K, H.nnz, 1, H.indptr.tolist(), H.indices.tolist(), H.data.tolist())


def _extract(h: highspy.Highs, K: int) -> tuple[np.ndarray, str]:
    status = h.getModelStatus()
    if status == _OPTIMAL:
        return np.array(h.getSolution().col_value[:K]), OPTIMAL_STR
    return np.zeros(K), str(status)


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------

def solve_lp(
    c: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    capacity: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Phase 1 minimum-cost LP via HiGHS simplex.

    min  cᵀw
    s.t. A_eq w = b_eq
         0 ≤ w_i ≤ capacity_i
    """
    K = len(c)
    h = _new_model(K, np.zeros(K), capacity, c)
    _add_eq_rows(h, A_eq, b_eq)
    h.run()
    return _extract(h, K)


def solve_qp(
    c: np.ndarray,
    Sigma_basis: np.ndarray,
    lam: float,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    capacity: np.ndarray,
) -> tuple[np.ndarray, str]:
    """QP with correlated basis-risk portfolio via HiGHS interior-point QP.

    min  cᵀw + λ·wᵀΣ_basis·w
    s.t. A_eq w = b_eq
         0 ≤ w_i ≤ capacity_i

    HiGHS objective convention: cᵀx + 0.5·xᵀHx  →  H = 2λΣ_basis.
    """
    K = len(c)
    h = _new_model(K, np.zeros(K), capacity, c)
    _pass_hessian(h, K, lam, Sigma_basis)
    _add_eq_rows(h, A_eq, b_eq)
    h.run()
    return _extract(h, K)


def solve_miqp(
    c: np.ndarray,
    Sigma_basis: np.ndarray,
    lam: float,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    capacity: np.ndarray,
    mutual_exclusion: list[tuple[int, int]],
    M_max: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Linear MIP with binary instrument-selection variables via HiGHS branch-and-bound.

    HiGHS 1.x does not support mixed-integer QP (MIQP), so the objective is the
    linear cost term only.  Sigma_basis and lam are accepted for API compatibility
    but are not used.

    Variables: [w_0 … w_{K-1}, z_0 … z_{K-1}]
      w_i ∈ [0, capacity_i]  continuous notional fractions
      z_i ∈ {0, 1}           active-leg indicators

    min  cᵀw
    s.t. A_eq w = b_eq
         w_i ≤ z_i · capacity_i        (big-M link)
         Σ z_i ≤ M_max                 (leg-count budget)
         z_i + z_j ≤ 1  ∀ (i,j) ∈ ME  (mutual exclusion)
         z_i ∈ {0, 1}
    """
    K = len(c)
    h = _new_model(K, np.zeros(K), capacity, c)

    # Binary selection variables
    h.addVars(K, np.zeros(K), np.ones(K))
    h.changeColsIntegrality(
        K,
        np.arange(K, 2 * K, dtype=np.int32),
        [highspy.HighsVarType.kInteger] * K,
    )

    _add_eq_rows(h, A_eq, b_eq)

    # Big-M: w_i ≤ capacity_i · z_i
    for i in range(K):
        h.addRow(-_INF, 0.0, 2, [i, K + i], [1.0, -float(capacity[i])])

    # Leg-count budget
    h.addRow(-_INF, float(M_max), K, list(range(K, 2 * K)), [1.0] * K)

    # Mutual exclusion
    for i, j in mutual_exclusion:
        h.addRow(-_INF, 1.0, 2, [K + i, K + j], [1.0, 1.0])

    h.run()
    status = h.getModelStatus()
    if status == _OPTIMAL:
        sol = h.getSolution().col_value
        return np.array(sol[:K]), np.array(sol[K:2 * K]), OPTIMAL_STR
    return np.zeros(K), np.zeros(K), str(status)


def solve_sca(
    c_base: np.ndarray,
    Sigma_basis: np.ndarray,
    lam: float,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    capacity: np.ndarray,
    adv_fractions: np.ndarray,
    eta: float,
    alpha: float,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> tuple[np.ndarray, str, int]:
    """Sequential Convex Approximation for power-law impact (α ≠ 0.5).

    Linearises  impact_i(w_i) = η · (w_i · adv_fraction_i)^α
    around the current iterate w^k:

        impact_i(w_i) ≈ g_i · w_i + const
        g_i = α · η · (w^k_i · adv_fraction_i)^{α - 1} · adv_fraction_i

    Each outer iteration solves a HiGHS QP sub-problem.
    Converges to a KKT point of the original NLP in 5–20 iterations.
    """
    K = len(c_base)
    EPS = 1e-8

    # Warm start: equal weight, clipped to capacity
    w = np.minimum(np.full(K, 1.0 / K), capacity)
    if w.sum() > 0:
        w = w / w.sum() * min(np.sum(capacity), 1.0)

    final_status = OPTIMAL_STR
    for iteration in range(max_iter):
        # Linearised impact gradient at current w
        w_adv = np.maximum(w * adv_fractions, EPS)
        impact_grad = alpha * eta * w_adv ** (alpha - 1.0) * adv_fractions
        c = c_base + impact_grad

        w_new, status = solve_qp(c, Sigma_basis, lam, A_eq, b_eq, capacity)
        final_status = status

        if status != OPTIMAL_STR:
            break

        delta = float(np.linalg.norm(w_new - w, ord=1))
        w = w_new
        if delta < tol:
            return w, OPTIMAL_STR, iteration + 1

    return w, final_status, max_iter

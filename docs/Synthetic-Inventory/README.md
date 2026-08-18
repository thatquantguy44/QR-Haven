# Synthetic Inventory Creation Optimizer

**Priority Score: 90 / 100**

> Manufacture economically equivalent positions when physical inventory is scarce or
> prohibitively expensive to borrow. A network-flow optimizer traverses a transformation
> graph of cash equities, ETFs, futures, options, swaps, and repo to find the
> cheapest replication path subject to exact exposure equivalence.

---

## Problem Statement

When a prime brokerage client needs short exposure to a name that is hard-to-borrow
(HTB), the desk has two choices: locate physical shares (at a potentially punishing borrow
rate) or construct a *synthetic* position that replicates the same economic exposure at
lower total cost. The same problem arises on the long side when regulatory or operational
constraints block direct ownership.

The **Synthetic Inventory Creation Optimizer** answers: *given a target exposure P_target,
find the portfolio of derivative and repo instruments P* that is economically equivalent
and minimises total carrying cost.*

---

## Transformation Graph

The feasible replication space is modelled as a directed weighted graph where:

- **Nodes** are instrument types that can carry the target exposure:
  - Cash equity (long / short)
  - ETF (and its constituent basket)
  - Single-stock futures (SSF)
  - Listed options (synthetic forward via put–call parity)
  - Total-return swaps (TRS)
  - Repo / reverse repo

- **Edges** are transformations between instrument types with associated cost and feasibility.  
  For example: `CashEquity → TRS` (borrow the economic exposure via swap),
  `ETF → ConstituentBasket` (create/redeem via authorised participant),
  `CashEquity → SSF` (replace physical with futures).

Each edge carries:

| Attribute | Description |
|---|---|
| `transformation_cost_bps` | Execution cost to switch from node A to node B |
| `carrying_cost_bps` | Per-day holding cost along the edge |
| `basis_risk_vol` | Annualised residual basis volatility (0 for exact replication) |
| `margin_rate` | Required margin as fraction of notional |
| `capital_charge` | RWA / balance-sheet cost per unit of notional |
| `feasibility` | Binary or fractional (0–1 liquidity score) |

---

## Cost Function

For a candidate synthetic portfolio P (a weight vector over edges/instruments):

```text
C(P) = Funding(P) + Margin(P) + Capital(P) + Execution(P) + BasisRisk(P)
```

### Component Definitions

**Funding cost** — cost to finance the position over the holding period H:

```text
Funding(P) = Σ_i  notional_i × funding_rate_i × H
```

where `funding_rate_i` is the instrument-specific financing spread over risk-free.

**Margin cost** — opportunity cost of posted margin / collateral:

```text
Margin(P) = Σ_i  notional_i × margin_rate_i × hurdle_rate × H
```

**Capital charge** — balance-sheet cost (RWA) priced at the firm's ROE hurdle:

```text
Capital(P) = Σ_i  notional_i × RWA_weight_i × capital_cost_rate
```

**Execution cost** — market impact and spread to enter and exit the position:

```text
Execution(P) = Σ_i  notional_i × execution_cost_bps_i / 10_000
```

Execution is amortised over holding period H; for short-dated positions it dominates.

**Basis risk penalty** — expected cost of residual tracking error priced at a risk
aversion coefficient λ:

```text
BasisRisk(P) = λ × Σ_i  notional_i² × basis_variance_i
```

This term penalises imperfect replication and ensures the solver balances cost against
exposure fidelity.

---

## Optimization Problem

```text
P* = argmin_{P}   C(P)

subject to:
    Exposure equivalence:   Δ(P) = Δ_target          (delta match)
                            Γ(P) ≤ Γ_limit            (gamma budget)
                            Vega(P) ≤ Vega_limit      (vega budget)
    Instrument limits:      0 ≤ w_i ≤ capacity_i      (position limits)
    Feasibility:            feasibility_i × w_i = w_i  (liquidity gating)
    Balance sheet:          Σ_i Capital_i(w_i) ≤ BS_budget
    Notional conservation:  Σ_i w_i = 1               (fully invested)
```

The primary constraint is **exposure equivalence** — the synthetic must deliver
the same first-order market exposure as the physical position. Higher-order constraints
(gamma, vega) are optional guardrails that prevent the solver from inadvertently
introducing material options risk.

### Constraint Relaxation

When the strict feasibility set is empty (e.g., no liquid SSF exists and TRS liquidity
is zero), the solver applies a hierarchy of relaxations:

1. Relax gamma / vega budget constraints
2. Allow partial physical fill (hybrid synthetic + physical)
3. Accept basis risk up to a configurable tolerance
4. Return infeasible signal and fall back to external borrow

---

## Baseline

The cost baseline is **external borrow at the prevailing HTB rate**:

```text
C_baseline = notional × borrow_rate_annualised × H
```

The optimizer is only called when at least one synthetic path satisfies:

```text
C(P*) < C_baseline × (1 - min_saving_threshold)
```

where `min_saving_threshold` defaults to 0.05 (5 % saving required to warrant
the operational complexity of a synthetic).

---

## Suggested Implementation

### Phase 1 — Constrained Network-Flow Optimizer

Model the transformation graph as a minimum-cost flow problem:

- Source node S emits one unit of target exposure.
- Sink node T absorbs the replicated exposure.
- Intermediate nodes represent instrument types.
- Edge costs encode `C(P)` components.
- Flow conservation at each node enforces the notional constraint.

Standard LP solvers (scipy.optimize.linprog, CVXPY) can handle graphs with
~10² instrument nodes efficiently. The LP relaxation of the integer feasibility
constraints is tight in practice because position limits are continuous.

```python
# Pseudocode — Phase 1 skeleton
from dataclasses import dataclass
import numpy as np

@dataclass
class SyntheticPath:
    instrument_type: str        # "TRS" | "SSF" | "ETF_basket" | "options_forward"
    notional_fraction: float    # fraction of target exposure replicated by this leg
    total_cost_bps: float       # annualised cost in bps
    basis_vol_bps: float        # residual tracking error vol

class SyntheticInventoryOptimizer:
    def optimize(
        self,
        target_exposure: float,         # USD notional
        target_delta: float,            # e.g. -1.0 for full short
        available_instruments: list,    # InstrumentSpec objects
        holding_period_days: int,
        cost_of_borrow_bps: float,      # HTB rate — the baseline
    ) -> list[SyntheticPath]:
        ...
```

### Phase 1b — Quadratic and Non-Linear Extensions

The Phase 1 LP makes two simplifying assumptions that can be lifted:
(a) execution costs are linear in trade size, and
(b) basis risks across synthetic legs are independent.
Lifting either assumption introduces non-linearity and improves cost accuracy
substantially for large or complex synthetics.

---

#### QP: Correlated Basis Risk Portfolio

When the synthetic uses multiple legs (e.g., ETF basket + repo + SSF), residual
tracking errors across legs are correlated. The scalar basis risk penalty in the
LP objective becomes a portfolio variance term:

```text
BasisRisk(w) = λ · wᵀ Σ_basis w
```

where `Σ_basis` (K×K) is the covariance matrix of the per-leg residual basis.
The full objective is then:

```text
min_{w}   cᵀw  +  λ · wᵀ Σ_basis w
s.t.      Aw = b,  w ≥ 0
```

This is a **convex QP** (since `Σ_basis ⪰ 0`), solvable with CVXPY + OSQP or
`scipy.optimize.minimize` with method `trust-constr`. The off-diagonal terms of
`Σ_basis` capture co-movement of, e.g., ETF premium/discount and SSF roll: when
both compress simultaneously the total basis loss is larger than independent models
predict.

```python
import cvxpy as cp

w = cp.Variable(K)
cost_linear  = c @ w
cost_quadratic = lam * cp.quad_form(w, Sigma_basis)
objective = cp.Minimize(cost_linear + cost_quadratic)
constraints = [A @ w == b, w >= 0, w <= capacity]
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.OSQP)
```

---

#### SOCP: Square-Root Market Impact

The Almgren-Chriss square-root impact model is non-linear:

```text
execution_cost_i(w_i) = η · σ_i · √(w_i / ADV_i) · notional_i
```

Substituting this into the LP objective breaks linearity. However, the function
`t_i ≥ η · σ_i · √(w_i)` can be rewritten exactly as a second-order cone constraint:

```text
‖ [2·w_i^{0.5}, t_i − η²·σ_i²] ‖₂ ≤ t_i + η²·σ_i²
```

which after substituting variables is a standard rotated cone:

```text
t_i² ≥ η² · σ_i² · w_i    →    (t_i, 1, η·σ_i·w_i^{0.5}) ∈ RSOC
```

The full problem becomes a **Second-Order Cone Program (SOCP)**, still solvable
in polynomial time via interior-point methods (MOSEK, ECOS):

```python
t = cp.Variable(K, nonneg=True)          # epigraph vars for impact terms
w = cp.Variable(K, nonneg=True)

impact_cost = cp.sum(t)
carry_cost   = c_carry @ w
basis_cost   = lam * cp.quad_form(w, Sigma_basis)

objective = cp.Minimize(carry_cost + impact_cost + basis_cost)
constraints = [
    A @ w == b,
    w <= capacity,
    *[cp.SOC(t[i] + eta2_sigma2[i], cp.vstack([2 * w[i], t[i] - eta2_sigma2[i]]))
      for i in range(K)],
]
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK)
```

The SOCP formulation is strictly better than linearising impact (e.g., fixing
participation rate) because it captures the concavity of cost in trade size —
large trades get penalised appropriately without needing to enumerate size buckets.

---

#### NLP: Power-Law Impact (α ≠ 0.5)

Almgren-Chriss temporary impact with exponent α:

```text
temp_impact_i = η · σ_i · (w_i / ADV_i)^α
```

For α ∈ (0, 1) this is concave in `w_i`; for α > 1 it is convex. Neither maps to
a cone constraint in the general case, making this a **Non-Linear Program (NLP)**.
Two practical approaches:

**Sequential Convex Approximation (SCA)** — at iteration k, linearise the
non-linear terms around the current iterate `w^k` and solve the resulting LP/QP:

```text
impact_i(w_i) ≈ impact_i(w_i^k) + α·η·σ_i·(w_i^k / ADV_i)^{α-1} · (w_i − w_i^k)
```

Converges to a KKT point in 5–20 iterations for typical inventory sizes; each
iteration is an LP or QP and takes milliseconds.

**Interior-point NLP (IPOPT / scipy SLSQP)** — pass the true non-linear objective
and gradient directly. Warranted when α is calibrated from intraday data and
departs materially from 0.5.

---

#### MIQP: Instrument Selection with Binary Variables

In practice the desk may want to activate at most `M_max` instrument types
(operational complexity constraint) or must choose between mutually exclusive
legs (e.g., TRS *or* SSF but not both on the same name). Introducing binary
selection variables `z_i ∈ {0, 1}` yields a **Mixed-Integer QP (MIQP)**:

```text
min_{w, z}   cᵀw + λ · wᵀ Σ_basis w
s.t.         Aw = b
             0 ≤ w_i ≤ z_i · capacity_i      (big-M link)
             z_i ∈ {0, 1}
             Σ_i z_i ≤ M_max                  (leg-count budget)
             z_TRS + z_SSF ≤ 1               (mutual exclusion example)
```

MIQP is NP-hard in general but tractable for K ≤ 20 instrument nodes (the typical
inventory graph is small). Branch-and-bound solvers (CVXPY + GUROBI/HiGHS) handle
this comfortably. For K > 50, relax to QP and round with a feasibility repair step.

---

#### Robust QP: Uncertainty in Borrow Rates and Basis Spreads

When borrow rates `r` and basis spreads `b` are uncertain, replace point estimates
with an ellipsoidal uncertainty set:

```text
u ∈ U = { u : (u − û)ᵀ Σ_u⁻¹ (u − û) ≤ κ² }
```

The robust counterpart of the carry cost minimisation problem:

```text
min_{w}  max_{u ∈ U}  uᵀw + λ · wᵀ Σ_basis w
```

Applying the standard epigraph reformulation (Ben-Tal & Nemirovski), the inner
`max` has a closed form and the robust problem becomes:

```text
min_{w}   ûᵀw + κ · ‖Σ_u^{0.5} w‖₂  +  λ · wᵀ Σ_basis w
```

which is a **Robust QP** (a QP with an added SOCP term), solvable directly by
MOSEK. The parameter κ controls conservatism: κ = 0 recovers the nominal QP;
κ = 2 covers ~95 % of the uncertainty set under Gaussian assumptions.

This formulation is particularly valuable for HTB names with squeeze risk, where
borrow rates can gap by 50–200 bps overnight — a scenario the nominal LP/QP
systematically underweights.

---

#### Problem Hierarchy Summary

| Formulation | When to use | Solver |
|---|---|---|
| LP (Phase 1) | Linear carry costs, independent basis risks, no impact non-linearity | `linprog`, GLPK |
| QP | Correlated basis risk across legs | OSQP, CVXPY |
| SOCP | Square-root (α = 0.5) execution impact included | ECOS, MOSEK |
| NLP (SCA) | Power-law impact α ≠ 0.5, 5–20 outer iterations | IPOPT, SLSQP |
| MIQP | Leg-count limit or mutual-exclusion constraints | Gurobi, HiGHS |
| Robust QP | Uncertain borrow rates / basis spreads (squeeze risk) | MOSEK |
| Stochastic LP (Phase 2) | Multi-period hold, scenario-tree CVaR constraint | Benders + LP |

In practice the QP + SOCP combination (correlated basis risk + square-root impact)
is the right default upgrade from the Phase 1 LP: it captures the two largest
sources of model error with solvers that are reliable and fast at inventory-graph
scale.

---

### Phase 2 — Multi-Period Stochastic Replication Optimizer

For exposures that need to be held over uncertain horizons, extend Phase 1 to a
multi-period stochastic program:

- States of the world are discretised borrow-rate and basis-spread scenarios.
- The optimizer selects an instrument mix that minimises *expected* cost subject to
  a CVaR constraint on worst-case carry.
- Roll and transformation costs between periods are modelled explicitly.
- Solved via scenario-tree LP or robust optimisation (Benders decomposition for
  large scenario sets).

This phase is warranted when:
- The target holding period is > 30 days (roll risk is material).
- Borrow rates are volatile (HTB names with squeeze risk).
- The desk wants to commit to a synthetic path that is stable across market regimes.

---

## Required Data Inputs

| Input | Source | Notes |
|---|---|---|
| Lending fee time series | Securities lending desk / data vendor | Per-CUSIP daily rate |
| SSF / futures prices and open interest | Exchange | Liquidity filter |
| Listed options chain | Market data feed | For put–call parity synthetics |
| TRS funding spreads | Prime brokerage | Dealer-quoted, updated daily |
| Repo / reverse repo rates | Tri-party platform | GC + specials |
| Margin requirements | Prime brokerage / exchange | SPAN or TIMS |
| Capital / RWA weights | Internal risk model | SA-CCR for derivatives |
| Execution cost estimates | Market impact model | Plug into existing `AlmgrenChrissModel` |
| Basis risk estimates | Historical analysis | Residual vol of hedge ratio |
| Basis risk covariance matrix | Historical regression residuals | Required for QP / Robust QP formulations |
| ADV per instrument | Market data | Required for SOCP / NLP impact models |
| Impact exponent α | Empirical calibration from intraday data | Defaults to 0.5; calibrate for large-cap vs small-cap |
| Borrow rate uncertainty (Σ_u) | Historical borrow rate vol, CUSP | Required for Robust QP; proxied by 30-day rolling vol |

---

## Business Value

- **Synthetic supply**: the desk can short names that are otherwise unborrowable,
  capturing client flow that would otherwise be turned away.
- **Hard ceiling on borrow fees**: any HTB rate above the synthetic carry cost is
  captured as P&L by replacing the physical borrow with a synthetic.
- **Balance-sheet efficiency**: derivatives (TRS, SSF) carry lower capital charges
  than on-balance-sheet short positions.
- **Client pricing**: the optimizer output directly informs the rate at which the
  desk can offer synthetic shorts to clients, replacing ad-hoc trader intuition.

---

## Integration Points

- **`costs/borrow.py`** — `BorrowCostSchedule` provides the baseline HTB rate input.
- **`costs/market_impact.py`** — `AlmgrenChrissModel` estimates execution costs
  for entering / exiting each leg of the synthetic.
- **`costs/squeeze.py`** — `ShortSqueezeModel` scores basis risk and flags names
  where physical-synthetic basis could gap adversely.
- **`portfolio/optimizers.py`** — `OptimizerConstraints` can enforce exposure
  equivalence constraints within the existing MeanVarianceOptimizer framework.
- **`research/pipeline.py`** — The optimizer slots in as a pre-rebalance
  inventory step: before `ResearchPipeline` assigns weights, it checks whether
  any target short is HTB and routes through the synthetic optimizer.

---

## Metrics

| Metric | Definition |
|---|---|
| **Synthetic vs physical cost** | `C_baseline − C(P*)` in bps per annum |
| **Executable opportunities** | Count of HTB names where `C(P*) < C_baseline` |
| **Realised basis P&L** | Actual P&L of the residual hedge ratio over holding period |
| **Fill rate** | Fraction of target exposure achieved via feasible synthetic paths |
| **Roll slippage** | Cost of rolling futures / TRS legs versus model estimate |

---

## Complexity and Feasibility

| Dimension | Assessment |
|---|---|
| **Difficulty** | Very high — multi-instrument pricing, regulatory capital modelling; QP/SOCP/MIQP extensions add solver complexity |
| **ROI** | Potentially very high — direct P&L from fee arbitrage and captured client flow |
| **Publication value** | 5 / 5 — network-flow formulation of synthetic inventory is novel in public literature |
| **Production feasibility** | Medium — requires reliable derivatives pricing and margin data feeds |

---

## References

- Almgren & Chriss (2001) — *Optimal execution of portfolio transactions*
  (foundation for the execution cost model plugged into Phase 1)
- He & Litterman (1999) — *The intuition behind Black-Litterman*
  (exposure equivalence constraint mirrors BL posterior blending)
- King (2002) — *Duality and martingales: A stochastic programming perspective
  on contingent claims* (theoretical basis for Phase 2 stochastic replication)
- ISDA SIMM documentation — standard margin model used for Phase 1 capital constraints
- Ben-Tal & Nemirovski (1998) — *Robust convex optimization* (theoretical basis for the
  Robust QP ellipsoidal uncertainty reformulation)
- Lobo, Vandenberghe, Boyd & Lebret (1998) — *Applications of second-order cone programming*
  (SOCP reformulation of square-root impact; see §4 on portfolio optimisation)
- Boyd & Vandenberghe (2004) — *Convex Optimization*, Ch. 11 (interior-point methods
  underpinning MOSEK/ECOS solvers used in SOCP and Robust QP)
- Yuille & Rangarajan (2003) — *The concave-convex procedure* (SCA convergence for
  power-law NLP when α < 1 makes temporary impact concave)

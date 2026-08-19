"""Cost function components for synthetic inventory positions.

C(P) = Funding(P) + Margin(P) + Capital(P) + Execution(P) + BasisRisk(P)

All costs are expressed in annualised basis points per unit of notional weight.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from qr_haven.synthetic_inventory.instruments import InstrumentSpec

_BPS = 10_000.0
_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class CostConfig:
    """Parameters controlling the cost function and solver behaviour.

    Attributes
    ----------
    hurdle_rate : float
        Opportunity cost of posted margin capital, per annum.
    capital_cost_rate : float
        ROE hurdle applied to RWA / balance-sheet usage, per annum.
    risk_aversion : float
        Lambda (λ) weighting the basis-risk penalty in the objective.
    impact_eta : float
        Almgren-Chriss η scaling the market-impact component.
    impact_alpha : float
        Impact exponent (0.5 = square-root; use SCA mode for α ≠ 0.5).
    min_saving_threshold : float
        Minimum fractional saving over the baseline borrow cost required
        before the synthetic is deemed beneficial (default 5 %).
    """

    hurdle_rate: float = 0.08
    capital_cost_rate: float = 0.10
    risk_aversion: float = 0.5
    impact_eta: float = 0.1
    impact_alpha: float = 0.5
    min_saving_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.hurdle_rate < 0:
            raise ValueError("hurdle_rate cannot be negative")
        if self.capital_cost_rate < 0:
            raise ValueError("capital_cost_rate cannot be negative")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion cannot be negative")
        if not 0 < self.impact_alpha <= 1:
            raise ValueError("impact_alpha must be in (0, 1]")
        if not 0 <= self.min_saving_threshold < 1:
            raise ValueError("min_saving_threshold must be in [0, 1)")


@dataclass(frozen=True)
class CostBreakdown:
    """Annualised cost decomposition in basis points for a given weight vector."""

    funding_bps: float
    margin_bps: float
    capital_bps: float
    execution_bps: float
    basis_risk_bps: float
    total_bps: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_cost_vector(
    instruments: list[InstrumentSpec],
    holding_period_days: float,
    config: CostConfig,
) -> np.ndarray:
    """Build the linear cost vector c (K,) for the LP / QP objective.

    The vector encodes *annualised* cost in bps per unit of notional weight for
    each instrument.  Execution cost is a one-off paid at entry, amortised over
    the holding period so it is expressed on the same per-annum scale.

    For the QP formulation, the diagonal basis-risk component is replaced by the
    full Σ_basis matrix passed separately; here it appears only as an independent
    per-instrument penalty suitable for the LP.
    """
    K = len(instruments)
    H = max(holding_period_days / _DAYS_PER_YEAR, 1.0 / _DAYS_PER_YEAR)
    c = np.empty(K)

    for i, spec in enumerate(instruments):
        funding = spec.funding_rate_annual * _BPS
        margin = spec.margin_rate * config.hurdle_rate * _BPS
        capital = spec.rwa_weight * config.capital_cost_rate * _BPS
        # One-off execution cost amortised over holding period → bps p.a.
        execution = spec.execution_cost_bps / H
        # Diagonal basis-risk penalty (independent-legs assumption for LP mode)
        basis = config.risk_aversion * (spec.basis_vol_annual * _BPS) ** 2 / _BPS
        c[i] = funding + margin + capital + execution + basis

    return c


def compute_cost_breakdown(
    weights: np.ndarray,
    instruments: list[InstrumentSpec],
    holding_period_days: float,
    config: CostConfig,
) -> CostBreakdown:
    """Decompose the five cost components for a given weight vector.

    Each component is summed across all legs using the instrument-level rates
    and the provided weight vector, then expressed in annualised bps.
    """
    H = max(holding_period_days / _DAYS_PER_YEAR, 1.0 / _DAYS_PER_YEAR)
    funding = margin = capital = execution = basis = 0.0

    for w, spec in zip(weights, instruments):
        w = float(w)
        funding += w * spec.funding_rate_annual * _BPS
        margin += w * spec.margin_rate * config.hurdle_rate * _BPS
        capital += w * spec.rwa_weight * config.capital_cost_rate * _BPS
        execution += w * spec.execution_cost_bps / H
        basis += config.risk_aversion * (w ** 2) * (spec.basis_vol_annual * _BPS) ** 2 / _BPS

    total = funding + margin + capital + execution + basis
    return CostBreakdown(
        funding_bps=round(funding, 6),
        margin_bps=round(margin, 6),
        capital_bps=round(capital, 6),
        execution_bps=round(execution, 6),
        basis_risk_bps=round(basis, 6),
        total_bps=round(total, 6),
    )

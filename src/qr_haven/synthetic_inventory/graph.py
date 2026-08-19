"""Transformation graph over synthetic replication instruments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qr_haven.synthetic_inventory.costs import CostConfig, _BPS
from qr_haven.synthetic_inventory.instruments import InstrumentSpec, InstrumentType


@dataclass
class TransformationGraph:
    """
    Directed graph whose nodes are InstrumentSpec objects.

    The graph encodes all feasible paths from the target exposure (source node)
    to a set of synthetic replication instruments (sink nodes).  Feasibility-
    gated instruments (feasibility = 0) are excluded from optimization but
    retained in the full graph for diagnostics.

    In practice only the feasible subset is passed to the solver; the graph
    class provides the constraint-matrix rows and per-instrument vectors
    consumed by the solver layer.
    """

    instruments: list[InstrumentSpec]

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def feasible_instruments(self) -> list[InstrumentSpec]:
        """Return instruments with positive feasibility and capacity."""
        return [s for s in self.instruments if s.feasibility > 0 and s.capacity > 0]

    def instruments_by_type(self, instrument_type: InstrumentType) -> list[InstrumentSpec]:
        return [s for s in self.instruments if s.instrument_type == instrument_type]

    # ------------------------------------------------------------------
    # Constraint rows (returned as flat numpy arrays over feasible instruments)
    # ------------------------------------------------------------------

    def capacity_vector(self) -> np.ndarray:
        """Upper bounds on weights: capacity_i per feasible instrument."""
        return np.array([s.capacity for s in self.feasible_instruments()])

    def delta_row(self) -> np.ndarray:
        """Constraint coefficients for exposure-equivalence: Σ delta_i · w_i = target."""
        return np.array([s.delta * s.feasibility for s in self.feasible_instruments()])

    def gamma_row(self) -> np.ndarray:
        """Constraint coefficients for gamma budget: Σ gamma_i · w_i ≤ limit."""
        return np.array([s.gamma * s.feasibility for s in self.feasible_instruments()])

    def vega_row(self) -> np.ndarray:
        """Constraint coefficients for vega budget: Σ vega_i · w_i ≤ limit."""
        return np.array([s.vega * s.feasibility for s in self.feasible_instruments()])

    def capital_row(self, config: CostConfig) -> np.ndarray:
        """Capital-charge coefficients: Σ rwa_i · capital_rate · w_i ≤ BS_budget."""
        return np.array(
            [s.rwa_weight * config.capital_cost_rate for s in self.feasible_instruments()]
        )

    # ------------------------------------------------------------------
    # Basis-risk covariance
    # ------------------------------------------------------------------

    def basis_variance_vector(self) -> np.ndarray:
        """Per-instrument annualised basis variance (diagonal of Σ_basis in bps²)."""
        return np.array(
            [(s.basis_vol_annual * _BPS) ** 2 for s in self.feasible_instruments()]
        )

    def default_basis_covariance(self) -> np.ndarray:
        """Diagonal Σ_basis assuming independent leg residuals."""
        return np.diag(self.basis_variance_vector())

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def max_achievable_delta(self) -> float:
        """Maximum signed delta achievable across all feasible instruments."""
        feasible = self.feasible_instruments()
        if not feasible:
            return 0.0
        return sum(s.delta * s.capacity * s.feasibility for s in feasible)

    def coverage_fraction(self, target_delta: float) -> float:
        """Fraction of target_delta achievable with feasible instruments."""
        if target_delta == 0:
            return 1.0
        achievable = abs(self.max_achievable_delta())
        return min(achievable / abs(target_delta), 1.0)

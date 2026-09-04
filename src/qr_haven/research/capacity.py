"""Capacity-aware alpha ranking: how strategy size erodes net Sharpe via cost drag.

Holding the alpha, optimizer, and rebalance schedule fixed, this sweeps portfolio
NAV upward and re-runs :class:`~qr_haven.research.pipeline.ResearchPipeline` at each
size. Market-impact cost grows with NAV because trade value relative to ADV (the
participation rate) grows with it; borrow and financing costs do not, since they are
already expressed as a rate on NAV. The resulting curve of net Sharpe vs. NAV is the
standard capacity diagnostic: it identifies the size at which cost drag has eaten a
chosen fraction of the strategy's edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from qr_haven.costs.borrow import BorrowCostSchedule
from qr_haven.costs.financing import FinancingRates
from qr_haven.interfaces import PortfolioOptimizer
from qr_haven.research.pipeline import PipelineConfig, ResearchPipeline
from qr_haven.risk import SimpleRiskEngine


@dataclass(frozen=True)
class CapacityPoint:
    """Pipeline outcome at one NAV level in a capacity sweep."""

    nav: float
    net_sharpe: float
    cost_drag: float
    total_cost_usd: float
    total_market_impact_usd: float


@dataclass(frozen=True)
class CapacityCurve:
    """Ordered set of :class:`CapacityPoint` values across a NAV grid."""

    points: tuple[CapacityPoint, ...]

    def capacity_nav(self, sharpe_floor_fraction: float = 0.5) -> float | None:
        """Return the largest swept NAV whose net Sharpe still meets the floor.

        The floor is ``sharpe_floor_fraction`` of the smallest-NAV point's net Sharpe
        (the least cost-constrained point in the sweep). Returns ``None`` when there
        are no points, the baseline Sharpe is non-positive (no floor is meaningful),
        or every point in the sweep already breaches the floor.

        Parameters
        ----------
        sharpe_floor_fraction:
            Fraction of baseline Sharpe treated as the capacity constraint. Must be
            in ``(0, 1]``.
        """

        if not 0.0 < sharpe_floor_fraction <= 1.0:
            raise ValueError("sharpe_floor_fraction must be in (0, 1].")
        if not self.points:
            return None
        baseline = self.points[0].net_sharpe
        if baseline <= 0.0:
            return None
        threshold = baseline * sharpe_floor_fraction

        capacity: float | None = None
        for point in self.points:
            if point.net_sharpe < threshold:
                break
            capacity = point.nav
        return capacity

    def to_frame(self) -> pd.DataFrame:
        """Return the sweep as a DataFrame indexed by NAV, for reporting or plotting."""

        return pd.DataFrame([vars(point) for point in self.points]).set_index("nav")


def sweep_capacity(
    optimizer: PortfolioOptimizer,
    returns: pd.DataFrame,
    nav_grid: Sequence[float],
    borrow_schedule: BorrowCostSchedule | None = None,
    financing_rates: FinancingRates | None = None,
    impact_model: Any | None = None,
    base_config: PipelineConfig | None = None,
    risk_engine: SimpleRiskEngine | None = None,
    alpha_scores: pd.DataFrame | None = None,
    adv_usd: pd.Series | None = None,
) -> CapacityCurve:
    """Run a cost-aware research pipeline across a NAV grid and return the capacity curve.

    Parameters
    ----------
    optimizer, borrow_schedule, financing_rates, impact_model, risk_engine:
        Passed straight through to :class:`~qr_haven.research.pipeline.ResearchPipeline`
        for every NAV in the grid, so the only thing that changes between points is size.
    returns, alpha_scores, adv_usd:
        Passed straight through to :meth:`ResearchPipeline.run` for every NAV in the grid.
    nav_grid:
        Portfolio NAVs to evaluate, in ascending order. Must be non-empty.
    base_config:
        Template :class:`PipelineConfig`. Its ``nav`` field is overridden per grid point;
        all other fields (lookback, rebalance cadence, fallback cost, etc.) are held fixed.
        Defaults to ``PipelineConfig()``.
    """

    if not nav_grid:
        raise ValueError("nav_grid must not be empty.")

    template = base_config or PipelineConfig()
    points: list[CapacityPoint] = []
    for nav in nav_grid:
        config = replace(template, nav=float(nav))
        pipeline = ResearchPipeline(
            optimizer=optimizer,
            borrow_schedule=borrow_schedule,
            financing_rates=financing_rates,
            impact_model=impact_model,
            config=config,
            risk_engine=risk_engine,
        )
        result = pipeline.run(returns, alpha_scores=alpha_scores, adv_usd=adv_usd)
        summary = result.summary()
        cost_summary = result.costs.summary()
        points.append(
            CapacityPoint(
                nav=float(nav),
                net_sharpe=float(summary.get("sharpe_ratio", 0.0)),
                cost_drag=float(summary["cost_drag"]),
                total_cost_usd=float(cost_summary["total_cost_usd"]),
                total_market_impact_usd=float(cost_summary["total_market_impact_usd"]),
            )
        )

    return CapacityCurve(points=tuple(points))

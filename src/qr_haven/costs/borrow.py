"""Borrow cost model for short positions at a prime broker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BorrowCostSchedule:
    """Per-symbol borrow fee rates for short positions.

    Attributes
    ----------
    fee_rates:
        Annualized borrow fee as a decimal fraction by symbol
        (e.g. 0.002 = 20 bps for GC; 0.30+ for hard-to-borrow specials).
    days_in_year:
        Day-count convention for fee accrual (typically 360 for equity lending).
    """

    fee_rates: pd.Series
    days_in_year: float = 360.0

    def __post_init__(self) -> None:
        if (self.fee_rates < 0).any():
            raise ValueError("fee_rates cannot be negative.")
        if self.days_in_year <= 0:
            raise ValueError("days_in_year must be positive.")

    @classmethod
    def from_scalars(
        cls,
        symbols: Sequence[str],
        fee_rate: float,
        days_in_year: float = 360.0,
    ) -> "BorrowCostSchedule":
        """Build a uniform schedule with the same fee rate for all symbols."""

        index = pd.Index(list(symbols), name="symbol")
        return cls(
            fee_rates=pd.Series(fee_rate, index=index, name="fee_rate", dtype=float),
            days_in_year=days_in_year,
        )

    @classmethod
    def gc_schedule(
        cls,
        symbols: Sequence[str],
        gc_rate: float = 0.002,
        days_in_year: float = 360.0,
    ) -> "BorrowCostSchedule":
        """Build a general-collateral schedule at a standard GC borrow rate.

        GC names are easy-to-borrow equities where the rate is driven by the
        overnight rate environment rather than security-specific demand. Typical
        GC rates range from 10 to 25 bps above the overnight risk-free rate.
        The default of 20 bps is a reasonable mid-cycle estimate.
        """

        return cls.from_scalars(symbols, fee_rate=gc_rate, days_in_year=days_in_year)


@dataclass(frozen=True)
class BorrowCostResult:
    """Outputs of a borrow cost estimate for short positions."""

    position_costs: pd.Series
    total_borrow_cost: float
    cost_on_nav: float

    def summary(self) -> dict[str, float]:
        """Return a compact serializable borrow cost summary."""

        return {
            "total_borrow_cost": self.total_borrow_cost,
            "cost_on_nav": self.cost_on_nav,
        }


class BorrowCostModel:
    """Estimate the daily borrow fee accrued on short positions.

    When a fund sells shares short, it must borrow them first and pay the
    lender a daily fee. This fee is the mirror of the lending revenue earned
    by the long-side lender — the same trade from the other direction.

    The cost accrues on the absolute value of short positions only:

        borrow_cost = |short_weight| × NAV × fee_rate × day_fraction

    Hard-to-borrow (HTB) names can cost 100 bps to 30%+ per annum and
    represent a significant drag on short-book alpha.
    """

    def estimate(
        self,
        weights: pd.Series,
        nav: float,
        schedule: BorrowCostSchedule,
        holding_period_days: float = 1.0,
    ) -> BorrowCostResult:
        """Estimate borrow costs for a single portfolio snapshot.

        Parameters
        ----------
        weights:
            Portfolio weights by symbol (signed). Only short (negative) weights
            contribute to borrow cost; long weights are ignored.
        nav:
            Portfolio net asset value in dollars. Must be positive.
        schedule:
            Per-symbol annualized borrow fee rates.
        holding_period_days:
            Number of calendar days over which to estimate cost. Cost is
            linearly prorated using the schedule's day-count convention.
        """

        if nav <= 0:
            raise ValueError("nav must be positive.")
        if holding_period_days <= 0:
            raise ValueError("holding_period_days must be positive.")

        symbols = weights.index.intersection(schedule.fee_rates.index)
        short_weights = weights.reindex(symbols).clip(upper=0.0).abs().astype(float)
        short_values = short_weights * nav
        day_fraction = holding_period_days / schedule.days_in_year

        position_costs = (
            short_values * schedule.fee_rates.reindex(symbols).fillna(0.0) * day_fraction
        )
        total_cost = float(position_costs.sum())
        cost_on_nav = (total_cost / nav) * (schedule.days_in_year / holding_period_days)

        return BorrowCostResult(
            position_costs=position_costs.rename("borrow_cost"),
            total_borrow_cost=total_cost,
            cost_on_nav=float(cost_on_nav),
        )

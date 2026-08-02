"""Securities lending revenue model for long portfolio income estimation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LendingFeeSchedule:
    """Per-symbol lending parameters for a portfolio snapshot.

    Attributes
    ----------
    fee_rates:
        Annualized borrow fee as a decimal fraction by symbol (e.g. 0.005 = 50 bps).
    on_loan_fractions:
        Fraction of each position expected to be on loan, in [0, 1].
    reinvestment_spread:
        Annualized cash-collateral reinvestment spread per symbol — the difference
        between the reinvestment rate earned and the rebate rate paid back to
        borrowers. Set to 0.0 for non-cash collateral positions.
    collateral_haircut:
        Over-collateralization ratio. 1.02 is the standard 102% margin for equities.
    days_in_year:
        Day-count convention used for fee accrual (typically 360 for securities lending).
    """

    fee_rates: pd.Series
    on_loan_fractions: pd.Series
    reinvestment_spread: pd.Series
    collateral_haircut: float = 1.02
    days_in_year: float = 360.0

    def __post_init__(self) -> None:
        if self.collateral_haircut < 1.0:
            raise ValueError("collateral_haircut must be at least 1.0.")
        if self.days_in_year <= 0:
            raise ValueError("days_in_year must be positive.")
        if (self.on_loan_fractions < 0).any() or (self.on_loan_fractions > 1).any():
            raise ValueError("on_loan_fractions must be in [0, 1].")
        if (self.fee_rates < 0).any():
            raise ValueError("fee_rates cannot be negative.")
        if (self.reinvestment_spread < 0).any():
            raise ValueError("reinvestment_spread cannot be negative.")

    @classmethod
    def from_scalars(
        cls,
        symbols: Sequence[str],
        fee_rate: float,
        on_loan_fraction: float,
        reinvestment_spread: float = 0.0,
        collateral_haircut: float = 1.02,
        days_in_year: float = 360.0,
    ) -> "LendingFeeSchedule":
        """Build a uniform schedule with the same parameters applied to all symbols."""

        index = pd.Index(list(symbols), name="symbol")
        return cls(
            fee_rates=pd.Series(fee_rate, index=index, name="fee_rate", dtype=float),
            on_loan_fractions=pd.Series(
                on_loan_fraction, index=index, name="on_loan_fraction", dtype=float
            ),
            reinvestment_spread=pd.Series(
                reinvestment_spread, index=index, name="reinvestment_spread", dtype=float
            ),
            collateral_haircut=collateral_haircut,
            days_in_year=days_in_year,
        )


@dataclass(frozen=True)
class LendingRevenueResult:
    """Outputs of a securities lending revenue estimate."""

    fee_income: pd.Series
    reinvestment_income: pd.Series
    position_revenue: pd.Series
    total_fee_income: float
    total_reinvestment_income: float
    total_lending_revenue: float
    revenue_on_nav: float

    def summary(self) -> dict[str, float]:
        """Return a compact serializable revenue summary."""

        return {
            "total_fee_income": self.total_fee_income,
            "total_reinvestment_income": self.total_reinvestment_income,
            "total_lending_revenue": self.total_lending_revenue,
            "revenue_on_nav": self.revenue_on_nav,
        }


class SecuritiesLendingRevenueModel:
    """Estimate lending revenue earned by a long portfolio over a holding period.

    Revenue has two components:

    1. **Lending fee income** — the annualized fee charged to borrowers for using
       the securities, prorated by the fraction on loan and the holding period.

    2. **Reinvestment income** — when borrowers post cash collateral, the lending
       agent invests the proceeds and earns a spread over the rebate rate paid back
       to borrowers. For non-cash collateral this component is zero.

    The model operates on end-of-day market values (weight × NAV) and produces
    revenue estimates suitable for inclusion in a P&L attribution waterfall:

        gross return → execution costs → borrow costs → financing costs
          → lending revenue → net return
    """

    def estimate(
        self,
        weights: pd.Series,
        nav: float,
        schedule: LendingFeeSchedule,
        holding_period_days: float = 1.0,
    ) -> LendingRevenueResult:
        """Estimate lending revenue for a single portfolio snapshot.

        Parameters
        ----------
        weights:
            Portfolio weights by symbol. Only long (positive) weights contribute
            to lending revenue; short weights are ignored.
        nav:
            Portfolio net asset value in dollars.
        schedule:
            Per-symbol fee rates, on-loan fractions, and reinvestment parameters.
        holding_period_days:
            Number of calendar days over which to estimate revenue. Revenue is
            linearly prorated using the schedule's day-count convention.

        Returns
        -------
        LendingRevenueResult
            Per-symbol and aggregate revenue broken down into fee income and
            reinvestment income, plus annualized revenue on NAV.
        """

        if nav <= 0:
            raise ValueError("nav must be positive.")
        if holding_period_days <= 0:
            raise ValueError("holding_period_days must be positive.")

        symbols = weights.index.intersection(schedule.fee_rates.index)
        long_weights = weights.reindex(symbols).clip(lower=0.0).astype(float)
        market_values = long_weights * nav
        on_loan_values = (
            market_values * schedule.on_loan_fractions.reindex(symbols).fillna(0.0)
        )
        day_fraction = holding_period_days / schedule.days_in_year

        fee_income = (
            on_loan_values
            * schedule.fee_rates.reindex(symbols).fillna(0.0)
            * day_fraction
        )
        collateral_value = on_loan_values * schedule.collateral_haircut
        reinvestment_income = (
            collateral_value
            * schedule.reinvestment_spread.reindex(symbols).fillna(0.0)
            * day_fraction
        )
        position_revenue = fee_income + reinvestment_income

        total_fee = float(fee_income.sum())
        total_reinvest = float(reinvestment_income.sum())
        total_revenue = total_fee + total_reinvest
        revenue_on_nav = (total_revenue / nav) * (schedule.days_in_year / holding_period_days)

        return LendingRevenueResult(
            fee_income=fee_income.rename("fee_income"),
            reinvestment_income=reinvestment_income.rename("reinvestment_income"),
            position_revenue=position_revenue.rename("position_revenue"),
            total_fee_income=total_fee,
            total_reinvestment_income=total_reinvest,
            total_lending_revenue=total_revenue,
            revenue_on_nav=float(revenue_on_nav),
        )

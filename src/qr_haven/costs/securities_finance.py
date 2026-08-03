"""Securities finance P&L attribution: unified lending revenue, borrow cost, and financing cost."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qr_haven.costs.borrow import BorrowCostModel, BorrowCostResult, BorrowCostSchedule
from qr_haven.costs.financing import FinancingCostModel, FinancingCostResult, FinancingRates
from qr_haven.costs.lending import (
    LendingFeeSchedule,
    LendingRevenueResult,
    SecuritiesLendingRevenueModel,
)


@dataclass(frozen=True)
class SecuritiesFinanceAttribution:
    """Decomposed securities-finance P&L for a single portfolio snapshot.

    Attributes
    ----------
    lending_revenue:
        Revenue from lending out long positions (fee income + reinvestment income).
    borrow_cost:
        Cost of borrowing shares to cover short positions.
    financing_cost:
        Net prime-broker financing cost (debit interest − short-proceeds credit).
    net_securities_finance:
        ``lending_revenue.total_lending_revenue
          − borrow_cost.total_borrow_cost
          − financing_cost.net_financing_cost``
        Positive means securities finance is a net contributor to P&L.
    net_on_nav:
        ``net_securities_finance / nav``, annualized.
    """

    lending_revenue: LendingRevenueResult
    borrow_cost: BorrowCostResult
    financing_cost: FinancingCostResult
    net_securities_finance: float
    net_on_nav: float

    def summary(self) -> dict[str, float]:
        """Return a compact serializable attribution summary."""

        return {
            "total_lending_revenue": self.lending_revenue.total_lending_revenue,
            "lending_revenue_on_nav": self.lending_revenue.revenue_on_nav,
            "total_borrow_cost": self.borrow_cost.total_borrow_cost,
            "borrow_cost_on_nav": self.borrow_cost.cost_on_nav,
            "debit_cost": self.financing_cost.debit_cost,
            "credit_income": self.financing_cost.credit_income,
            "net_financing_cost": self.financing_cost.net_financing_cost,
            "financing_cost_on_nav": self.financing_cost.cost_on_nav,
            "net_securities_finance": self.net_securities_finance,
            "net_on_nav": self.net_on_nav,
        }

    def waterfall(self) -> pd.Series:
        """P&L waterfall as a labelled Series (absolute dollar terms)."""

        items = {
            "lending_fee_income": self.lending_revenue.total_fee_income,
            "reinvestment_income": self.lending_revenue.total_reinvestment_income,
            "borrow_cost": -self.borrow_cost.total_borrow_cost,
            "debit_interest": -self.financing_cost.debit_cost,
            "credit_income": self.financing_cost.credit_income,
            "net_securities_finance": self.net_securities_finance,
        }
        return pd.Series(items, name="securities_finance_waterfall")


class SecuritiesFinanceAttributionModel:
    """Attribute total securities-finance P&L across lending, borrow, and financing.

    This model is the central entry point for prime-broker cost/revenue accounting.
    It wraps the three sub-models:

    - :class:`~qr_haven.costs.lending.SecuritiesLendingRevenueModel` — long-book income
    - :class:`~qr_haven.costs.borrow.BorrowCostModel` — short-book fee accrual
    - :class:`~qr_haven.costs.financing.FinancingCostModel` — margin debit/credit

    and aggregates them into a signed net P&L and a P&L waterfall.
    """

    def __init__(self) -> None:
        self._lending_model = SecuritiesLendingRevenueModel()
        self._borrow_model = BorrowCostModel()
        self._financing_model = FinancingCostModel()

    def attribute(
        self,
        weights: pd.Series,
        nav: float,
        lending_schedule: LendingFeeSchedule,
        borrow_schedule: BorrowCostSchedule,
        financing_rates: FinancingRates,
        holding_period_days: float = 1.0,
    ) -> SecuritiesFinanceAttribution:
        """Run all three sub-models and return a unified attribution.

        Parameters
        ----------
        weights:
            Portfolio weights by symbol (signed). Long positions positive, short negative.
        nav:
            Portfolio NAV in dollars. Must be positive.
        lending_schedule:
            Fee rates, on-loan fractions, and reinvestment spreads for the long book.
        borrow_schedule:
            Per-symbol borrow fee rates for the short book.
        financing_rates:
            Prime broker debit and credit rates.
        holding_period_days:
            Number of calendar days for the estimate.
        """

        lending = self._lending_model.estimate(
            weights, nav, lending_schedule, holding_period_days
        )
        borrow = self._borrow_model.estimate(
            weights, nav, borrow_schedule, holding_period_days
        )
        financing = self._financing_model.estimate(
            weights, nav, financing_rates, holding_period_days
        )

        net = (
            lending.total_lending_revenue
            - borrow.total_borrow_cost
            - financing.net_financing_cost
        )
        annualization = lending_schedule.days_in_year / holding_period_days
        net_on_nav = (net / nav) * annualization

        return SecuritiesFinanceAttribution(
            lending_revenue=lending,
            borrow_cost=borrow,
            financing_cost=financing,
            net_securities_finance=float(net),
            net_on_nav=float(net_on_nav),
        )

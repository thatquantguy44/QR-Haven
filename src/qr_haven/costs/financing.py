"""Financing cost model for leveraged portfolios at a prime broker."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FinancingRates:
    """Prime broker debit and credit rates for a financing cost estimate.

    Attributes
    ----------
    debit_rate:
        Annualized interest rate charged on margin debit balances (borrowed cash).
        Typically overnight risk-free rate + spread (e.g. SOFR + 50–150 bps).
    credit_rate:
        Annualized interest rate earned on short-sale proceeds held at the broker.
        Typically overnight rate − spread; often zero for retail, slightly positive
        for institutional accounts with favorable prime arrangements.
    days_in_year:
        Day-count convention (typically 360 for money-market instruments).
    """

    debit_rate: float
    credit_rate: float
    days_in_year: float = 360.0

    def __post_init__(self) -> None:
        if self.debit_rate < 0:
            raise ValueError("debit_rate cannot be negative.")
        if self.credit_rate < 0:
            raise ValueError("credit_rate cannot be negative.")
        if self.days_in_year <= 0:
            raise ValueError("days_in_year must be positive.")


@dataclass(frozen=True)
class FinancingCostResult:
    """Outputs of a prime-broker financing cost estimate.

    Attributes
    ----------
    debit_balance:
        Dollar amount borrowed from the broker (cash needed beyond equity + short
        proceeds). Zero for market-neutral or underleveraged portfolios.
    credit_balance:
        Dollar value of short-sale proceeds held at the broker.
    debit_cost:
        Interest paid on the debit balance for the holding period.
    credit_income:
        Interest earned on the credit balance for the holding period.
    net_financing_cost:
        ``debit_cost − credit_income``. Positive means net outflow.
    cost_on_nav:
        ``net_financing_cost / nav``, annualized regardless of holding period.
    """

    debit_balance: float
    credit_balance: float
    debit_cost: float
    credit_income: float
    net_financing_cost: float
    cost_on_nav: float

    def summary(self) -> dict[str, float]:
        """Return a compact serializable financing cost summary."""

        return {
            "debit_balance": self.debit_balance,
            "credit_balance": self.credit_balance,
            "debit_cost": self.debit_cost,
            "credit_income": self.credit_income,
            "net_financing_cost": self.net_financing_cost,
            "cost_on_nav": self.cost_on_nav,
        }


class FinancingCostModel:
    """Estimate the prime-broker financing cost for a leveraged portfolio.

    The model separates two financing flows:

    1. **Debit balance** — cash the fund must borrow from the prime broker
       when gross long exposure exceeds equity (NAV) plus short proceeds:

           debit_balance = max(0, long_gross − 1.0 − short_gross) × NAV

       Examples:
       - 100/0 long-only → debit = 0 (fully equity-funded)
       - 130/30 long-short → debit = max(0, 1.30 − 1.0 − 0.30) × NAV = 0
         (short proceeds exactly cover the 30% excess longs)
       - 150/30 long-short → debit = max(0, 1.50 − 1.0 − 0.30) × NAV = 0.20 × NAV
       - Market-neutral (1.0/1.0) → debit = 0; short proceeds produce a credit

    2. **Credit balance** — short-sale proceeds held at the broker:

           credit_balance = short_gross × NAV

       The fund earns (a portion of) the overnight rate on these proceeds,
       offset by the prime broker's spread.

    Net financing cost:

        net_financing_cost = debit_cost − credit_income
                           = debit_balance × debit_rate × day_fraction
                             − credit_balance × credit_rate × day_fraction
    """

    def estimate(
        self,
        weights: pd.Series,
        nav: float,
        rates: FinancingRates,
        holding_period_days: float = 1.0,
    ) -> FinancingCostResult:
        """Estimate financing costs for a single portfolio snapshot.

        Parameters
        ----------
        weights:
            Portfolio weights by symbol (signed). Long positions are positive,
            short positions negative.
        nav:
            Portfolio net asset value in dollars. Must be positive.
        rates:
            Annualized debit and credit rates for the prime broker relationship.
        holding_period_days:
            Number of calendar days over which to estimate cost.
        """

        if nav <= 0:
            raise ValueError("nav must be positive.")
        if holding_period_days <= 0:
            raise ValueError("holding_period_days must be positive.")

        w = weights.astype(float)
        long_gross = float(w.clip(lower=0.0).sum())
        short_gross = float(w.clip(upper=0.0).abs().sum())

        debit_balance = max(0.0, long_gross - 1.0 - short_gross) * nav
        credit_balance = short_gross * nav

        day_fraction = holding_period_days / rates.days_in_year
        debit_cost = debit_balance * rates.debit_rate * day_fraction
        credit_income = credit_balance * rates.credit_rate * day_fraction
        net_financing_cost = debit_cost - credit_income

        annualization = rates.days_in_year / holding_period_days
        cost_on_nav = (net_financing_cost / nav) * annualization

        return FinancingCostResult(
            debit_balance=debit_balance,
            credit_balance=credit_balance,
            debit_cost=debit_cost,
            credit_income=credit_income,
            net_financing_cost=net_financing_cost,
            cost_on_nav=float(cost_on_nav),
        )

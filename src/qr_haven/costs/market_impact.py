"""Market impact models for portfolio execution cost estimation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ImpactEstimate:
    """Breakdown of estimated market impact for a single order."""

    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float

    def to_dict(self) -> dict[str, float]:
        """Return a serializable impact breakdown."""

        return asdict(self)


class SquareRootImpactModel:
    """Market impact model based on the square root of participation rate.

    The square-root law is the industry standard for equity market impact
    estimation. It captures the empirical observation that impact scales with
    the square root of trade size relative to average daily volume (ADV):

        impact_bps = eta × sigma_daily_bps × sqrt(participation_rate)

    where participation_rate = |trade_value| / ADV and sigma_daily_bps is the
    daily volatility expressed in basis points.

    The square root arises from the statistical nature of order flow: larger
    trades require more time to complete, exposing the trader to price movement
    proportional to sqrt(time). This model treats all impact as temporary —
    it does not model a lasting shift in equilibrium price.

    Parameters
    ----------
    eta:
        Impact coefficient. Typical range 0.05–0.30. Lower values apply to
        large-cap liquid stocks; higher values apply to small-cap or illiquid names.
    """

    def __init__(self, eta: float = 0.1) -> None:
        if eta <= 0:
            raise ValueError("eta must be positive.")
        self.eta = eta

    def estimate(
        self,
        participation_rate: float,
        daily_volatility: float,
    ) -> ImpactEstimate:
        """Estimate market impact for a single order.

        Parameters
        ----------
        participation_rate:
            |trade_value| / ADV. Must be non-negative. A value of 0.01 means the
            order is 1% of average daily volume.
        daily_volatility:
            Daily return volatility as a decimal fraction (e.g. 0.02 = 2%).
        """

        if participation_rate < 0:
            raise ValueError("participation_rate cannot be negative.")
        if daily_volatility < 0:
            raise ValueError("daily_volatility cannot be negative.")

        impact_bps = self.eta * daily_volatility * 10_000 * math.sqrt(participation_rate)
        return ImpactEstimate(
            temporary_impact_bps=impact_bps,
            permanent_impact_bps=0.0,
            total_impact_bps=impact_bps,
        )


class AlmgrenChrissModel:
    """Almgren-Chriss market impact model with temporary and permanent components.

    The Almgren-Chriss framework (2001) decomposes market impact into two terms:

    **Temporary impact** — the instantaneous price concession paid to attract
    liquidity. It decays after trade completion and does not shift the long-run
    equilibrium price:

        temporary_bps = eta × sigma_daily_bps × participation_rate^alpha

    **Permanent impact** — the lasting shift in equilibrium price caused by
    information revealed through the order flow:

        permanent_bps = gamma × sigma_daily_bps × participation_rate

    The total one-way cost includes the full temporary impact and half the
    permanent impact, since the permanent component shifts the midpoint that
    subsequent trades are measured against:

        total_bps = temporary_bps + 0.5 × permanent_bps

    The square-root model is the special case alpha=0.5 with gamma=0.

    Parameters
    ----------
    eta:
        Temporary impact coefficient (default 0.1).
    gamma:
        Permanent impact coefficient (default 0.1).
    alpha:
        Temporary impact exponent (default 0.6). Empirical estimates typically
        fall in the range [0.5, 0.7].
    """

    def __init__(
        self,
        eta: float = 0.1,
        gamma: float = 0.1,
        alpha: float = 0.6,
    ) -> None:
        if eta <= 0:
            raise ValueError("eta must be positive.")
        if gamma < 0:
            raise ValueError("gamma cannot be negative.")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        self.eta = eta
        self.gamma = gamma
        self.alpha = alpha

    def estimate(
        self,
        participation_rate: float,
        daily_volatility: float,
    ) -> ImpactEstimate:
        """Estimate market impact for a single order.

        Parameters
        ----------
        participation_rate:
            |trade_value| / ADV. Must be non-negative.
        daily_volatility:
            Daily return volatility as a decimal fraction (e.g. 0.02 = 2%).
        """

        if participation_rate < 0:
            raise ValueError("participation_rate cannot be negative.")
        if daily_volatility < 0:
            raise ValueError("daily_volatility cannot be negative.")

        sigma_bps = daily_volatility * 10_000
        temporary = self.eta * sigma_bps * (participation_rate ** self.alpha)
        permanent = self.gamma * sigma_bps * participation_rate
        return ImpactEstimate(
            temporary_impact_bps=float(temporary),
            permanent_impact_bps=float(permanent),
            total_impact_bps=float(temporary + 0.5 * permanent),
        )


def estimate_portfolio_impact(
    weight_changes: pd.Series,
    daily_volatilities: pd.Series,
    participation_rates: pd.Series,
    model: SquareRootImpactModel | AlmgrenChrissModel,
) -> pd.DataFrame:
    """Estimate per-asset market impact costs for a portfolio rebalance.

    Parameters
    ----------
    weight_changes:
        Absolute weight changes per asset (|w_new - w_old|).
    daily_volatilities:
        Daily return volatility per asset as decimal fractions.
    participation_rates:
        Trade participation rate per asset (|trade_value| / ADV).
    model:
        A calibrated impact model.

    Returns
    -------
    pd.DataFrame
        Per-asset impact breakdown indexed by symbol with columns:
        weight_change, participation_rate, daily_volatility,
        temporary_impact_bps, permanent_impact_bps, total_impact_bps,
        weighted_cost (impact_bps × weight_change / 10_000).
    """

    symbols = weight_changes.index
    rows = []
    for symbol in symbols:
        weight_change = float(weight_changes.get(symbol, 0.0))
        prate = float(participation_rates.reindex([symbol]).fillna(0.0).iloc[0])
        dvol = float(daily_volatilities.reindex([symbol]).fillna(0.0).iloc[0])
        est = model.estimate(abs(prate), dvol)
        rows.append(
            {
                "weight_change": weight_change,
                "participation_rate": prate,
                "daily_volatility": dvol,
                "temporary_impact_bps": est.temporary_impact_bps,
                "permanent_impact_bps": est.permanent_impact_bps,
                "total_impact_bps": est.total_impact_bps,
                "weighted_cost": est.total_impact_bps * weight_change / 10_000,
            }
        )

    result = pd.DataFrame(rows, index=symbols)
    result.index.name = "symbol"
    return result


def portfolio_total_impact_cost(impact_frame: pd.DataFrame) -> float:
    """Return the total portfolio impact cost as a fraction of NAV.

    Sums ``weighted_cost`` across all assets. The result is directly comparable
    to a return: subtract it from gross portfolio return to get net-of-impact return.
    """

    return float(impact_frame["weighted_cost"].sum())

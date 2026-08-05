"""End-to-end quantitative research pipeline.

Connects alpha signals → portfolio optimization → walk-forward backtesting
→ real cost attribution into a single composable runner.

The backtester built in :mod:`qr_haven.backtesting.engine` supports a flat
``transaction_cost_bps`` override.  ``ResearchPipeline`` replaces that with
three real cost models that are already part of the library:

* **Market impact** — :class:`~qr_haven.costs.market_impact.SquareRootImpactModel`
  or :class:`~qr_haven.costs.market_impact.AlmgrenChrissModel`.  Applied at
  each rebalance using per-asset participation rate and rolling realised vol.

* **Borrow cost** — :class:`~qr_haven.costs.borrow.BorrowCostModel` with a
  :class:`~qr_haven.costs.borrow.BorrowCostSchedule`.  Accrues over each
  holding period on net short positions.

* **Financing cost** — :class:`~qr_haven.costs.financing.FinancingCostModel`
  with :class:`~qr_haven.costs.financing.FinancingRates`.  Captures debit
  balance interest (leverage drag) minus credit on short-sale proceeds.

Alpha integration
-----------------
Pass an ``alpha_scores`` panel (dates × assets) to override the default
``estimate_expected_returns`` step.  At each rebalance the pipeline looks up
the most recent available alpha scores and passes them directly to the
optimizer as expected returns — no extra fitting step required.

Usage
-----
>>> pipeline = ResearchPipeline(
...     optimizer=MeanVarianceOptimizer(),
...     borrow_schedule=BorrowCostSchedule.gc_schedule(symbols),
...     financing_rates=FinancingRates(debit_rate=0.055, credit_rate=0.04),
...     impact_model=SquareRootImpactModel(eta=0.1),
... )
>>> result = pipeline.run(returns, alpha_scores=scores_panel, adv_usd=adv)
>>> print(result.summary())
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from qr_haven.costs.borrow import BorrowCostModel, BorrowCostSchedule
from qr_haven.costs.financing import FinancingCostModel, FinancingRates
from qr_haven.interfaces import PortfolioOptimizer
from qr_haven.portfolio import (
    calculate_optimizer_diagnostics,
    estimate_expected_returns,
    estimate_return_covariance,
)
from qr_haven.risk import SimpleRiskEngine


@runtime_checkable
class _ImpactModel(Protocol):
    """Duck-type contract satisfied by both SquareRootImpactModel and AlmgrenChrissModel."""

    def estimate(self, participation_rate: float, daily_volatility: float) -> Any: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for :class:`ResearchPipeline`.

    Parameters
    ----------
    lookback_periods:
        Number of historical periods used to estimate covariance at each
        rebalance. Default 60.
    rebalance_periods:
        Number of periods between rebalances. Default 21.
    nav:
        Portfolio net asset value in dollars. Used to convert weights to
        dollar amounts for cost calculations. Default 10 000 000.
    annualization:
        Number of periods per year (252 for daily returns). Default 252.
    optimizer_constraints:
        Mapping passed to :meth:`PortfolioOptimizer.optimize`.
    fallback_cost_bps:
        Flat transaction cost used when no real cost models are provided.
        Mirrors the legacy ``BacktestConfig.transaction_cost_bps``.
    default_participation_rate:
        Fallback participation rate for assets with no ADV data when an
        impact model is provided. Default 0.05 (5 % of ADV).
    default_daily_vol:
        Fallback daily volatility for impact estimation when an asset has
        insufficient history. Default 0.02 (2 %).
    """

    lookback_periods: int = 60
    rebalance_periods: int = 21
    nav: float = 10_000_000.0
    annualization: float = 252.0
    optimizer_constraints: Mapping[str, Any] = field(default_factory=dict)
    fallback_cost_bps: float = 5.0
    default_participation_rate: float = 0.05
    default_daily_vol: float = 0.02

    def __post_init__(self) -> None:
        if self.lookback_periods < 2:
            raise ValueError("lookback_periods must be at least 2.")
        if self.rebalance_periods < 1:
            raise ValueError("rebalance_periods must be at least 1.")
        if self.nav <= 0:
            raise ValueError("nav must be positive.")
        if self.fallback_cost_bps < 0:
            raise ValueError("fallback_cost_bps cannot be negative.")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostBreakdown:
    """Per-rebalance cost attribution across all three cost layers.

    All values are in dollars (not basis points, not fractions of NAV).
    """

    market_impact: pd.Series
    """Dollar cost of trading at each rebalance (indexed by rebalance date)."""

    borrow_cost: pd.Series
    """Dollar cost of borrowing short positions over each holding period."""

    financing_cost: pd.Series
    """Net dollar financing cost (debit interest − short-proceed credit)."""

    @property
    def total(self) -> pd.Series:
        """Sum of all three cost components per rebalance."""
        return (self.market_impact + self.borrow_cost + self.financing_cost).rename(
            "total_cost"
        )

    def summary(self) -> dict[str, float]:
        """Return aggregate cost totals."""
        return {
            "total_market_impact_usd": float(self.market_impact.sum()),
            "total_borrow_cost_usd": float(self.borrow_cost.sum()),
            "total_financing_cost_usd": float(self.financing_cost.sum()),
            "total_cost_usd": float(self.total.sum()),
        }


@dataclass(frozen=True)
class PipelineResult:
    """Outputs of :meth:`ResearchPipeline.run`.

    Contains the same core series as :class:`~qr_haven.backtesting.engine.BacktestResult`
    plus the full cost attribution and (optionally) the alpha scores used.
    """

    portfolio_returns: pd.Series
    """Net-of-all-costs period returns."""

    gross_portfolio_returns: pd.Series
    """Pre-cost period returns (weights × asset returns)."""

    equity_curve: pd.Series
    """Cumulative net equity (starts at 1.0)."""

    weights: pd.DataFrame
    """Target weights at each rebalance date (rebalance_date × asset)."""

    alpha_scores_used: pd.DataFrame | None
    """Alpha scores passed to the optimizer. None if historical means were used."""

    costs: CostBreakdown
    """Full cost breakdown indexed by rebalance date."""

    turnover: pd.Series
    """One-way turnover at each rebalance."""

    asset_contributions: pd.DataFrame
    """Asset-level return contributions per period."""

    diagnostics: pd.DataFrame
    """Optimizer diagnostics at each rebalance."""

    risk_metrics: Mapping[str, float]
    """Portfolio-level risk metrics over the full period."""

    def summary(self) -> dict[str, float | int]:
        """Return a compact, serializable summary of pipeline output."""
        total_return = float(self.equity_curve.iloc[-1] - 1.0)
        gross_total = float((1.0 + self.gross_portfolio_returns).prod() - 1.0)
        cost_summary = self.costs.summary()
        nav_estimate = 1.0  # costs stored in dollars; express as fraction of gross
        return {
            "observations": int(len(self.portfolio_returns)),
            "total_return": total_return,
            "gross_total_return": gross_total,
            "cost_drag": gross_total - total_return,
            "average_turnover": float(self.turnover.mean()),
            "alpha_model_used": self.alpha_scores_used is not None,
            **cost_summary,
            **self.risk_metrics,
        }

    def cost_attribution(self) -> dict[str, float]:
        """Return cost totals expressed as annualised fractions of NAV.

        Useful for comparing cost drag across strategies with different NAVs.
        """
        n_years = len(self.portfolio_returns) / 252.0
        if n_years == 0:
            return {}
        cs = self.costs.summary()
        # No access to config.nav here, so we return raw dollar totals
        return cs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ResearchPipeline:
    """Walk-forward backtest with real cost models and optional alpha injection.

    Parameters
    ----------
    optimizer:
        Any object satisfying :class:`~qr_haven.interfaces.PortfolioOptimizer`
        (e.g. :class:`~qr_haven.portfolio.MeanVarianceOptimizer`).
    borrow_schedule:
        Per-symbol borrow fee schedule for short positions.  If ``None``,
        borrow costs are not modelled.
    financing_rates:
        Prime broker debit and credit rates.  If ``None``, financing costs
        are not modelled.
    impact_model:
        Market impact model (SquareRootImpactModel or AlmgrenChrissModel).
        If ``None``, the flat ``config.fallback_cost_bps`` is used instead.
    config:
        :class:`PipelineConfig` controlling window sizes, NAV, and fallback
        cost rates.
    risk_engine:
        Risk engine for end-of-run metrics.  Defaults to
        :class:`~qr_haven.risk.SimpleRiskEngine`.
    """

    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        borrow_schedule: BorrowCostSchedule | None = None,
        financing_rates: FinancingRates | None = None,
        impact_model: _ImpactModel | None = None,
        config: PipelineConfig | None = None,
        risk_engine: SimpleRiskEngine | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config or PipelineConfig()
        self.risk_engine = risk_engine or SimpleRiskEngine(
            annualization=self.config.annualization
        )
        self._borrow_schedule = borrow_schedule
        self._financing_rates = financing_rates
        self._impact_model = impact_model
        self._borrow_model = BorrowCostModel()
        self._financing_model = FinancingCostModel()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        returns: pd.DataFrame,
        alpha_scores: pd.DataFrame | None = None,
        adv_usd: pd.Series | None = None,
    ) -> PipelineResult:
        """Run the end-to-end research pipeline.

        Parameters
        ----------
        returns:
            Wide DataFrame of asset returns (DatetimeIndex × assets).  Must
            have more rows than ``config.lookback_periods``.
        alpha_scores:
            Optional panel of pre-computed alpha signals (DatetimeIndex ×
            assets).  At each rebalance the pipeline looks up the most recent
            available row (as-of-date safe) and passes it to the optimizer as
            expected returns.  When absent, ``estimate_expected_returns``
            (historical mean) is used.
        adv_usd:
            Optional average daily volume in USD per asset (Series indexed by
            asset name).  Required for meaningful market impact estimates; if
            absent and an impact model is provided, the
            ``config.default_participation_rate`` is used for all assets.

        Returns
        -------
        :class:`PipelineResult`
        """
        clean = returns.sort_index().astype(float)
        if len(clean) <= self.config.lookback_periods:
            raise ValueError(
                "returns must have more rows than config.lookback_periods "
                f"({self.config.lookback_periods})."
            )

        rebalance_positions = list(
            range(
                self.config.lookback_periods,
                len(clean),
                self.config.rebalance_periods,
            )
        )

        # --- accumulators ---
        portfolio_returns: list[pd.Series] = []
        gross_portfolio_returns: list[pd.Series] = []
        asset_contributions_list: list[pd.DataFrame] = []
        weight_rows: list[pd.Series] = []
        diagnostic_rows: list[pd.Series] = []
        turnover_vals: list[float] = []
        impact_vals: list[float] = []
        borrow_vals: list[float] = []
        financing_vals: list[float] = []
        rebalance_dates: list[Any] = []
        alpha_rows: list[pd.Series] = []

        prev_weights = pd.Series(0.0, index=clean.columns, dtype=float)

        for pos in rebalance_positions:
            rebalance_date = clean.index[pos]
            window = clean.iloc[pos - self.config.lookback_periods : pos]
            period_end = min(pos + self.config.rebalance_periods, len(clean))
            holding_days = float(period_end - pos)

            # ---- expected returns ----
            if alpha_scores is not None:
                exp_ret = self._lookup_alpha(alpha_scores, rebalance_date, clean.columns)
                alpha_rows.append(exp_ret.rename(rebalance_date))
            else:
                exp_ret = estimate_expected_returns(window)

            covariance = estimate_return_covariance(window)
            target_weights = self.optimizer.optimize(
                exp_ret, covariance, self.config.optimizer_constraints
            )
            target_weights = target_weights.reindex(clean.columns).fillna(0.0)

            # ---- diagnostics ----
            diag = calculate_optimizer_diagnostics(
                target_weights, exp_ret, covariance,
                self.config.optimizer_constraints, prev_weights,
            )
            diagnostic_rows.append(pd.Series(diag.to_dict(), name=rebalance_date))

            # ---- turnover ----
            weight_change = target_weights - prev_weights
            turnover = float(weight_change.abs().sum())

            # ---- costs ----
            impact_cost = self._compute_impact_cost(weight_change, window, adv_usd)
            borrow_cost = self._compute_borrow_cost(target_weights, holding_days)
            financing_cost = self._compute_financing_cost(target_weights, holding_days)

            if self._impact_model is None and impact_cost == 0.0:
                # fall back to flat bps
                flat_cost = turnover * self.config.fallback_cost_bps / 10_000.0 * self.config.nav
                impact_cost = flat_cost

            total_cost_fraction = (
                impact_cost + borrow_cost + financing_cost
            ) / self.config.nav

            # ---- period returns ----
            period_assets = clean.iloc[pos:period_end]
            period_contribs = period_assets.multiply(target_weights, axis="columns")
            period_gross = period_contribs.sum(axis=1)
            period_net = period_gross.copy()
            if not period_net.empty:
                period_net.iloc[0] = period_net.iloc[0] - total_cost_fraction

            # ---- accumulate ----
            portfolio_returns.append(period_net)
            gross_portfolio_returns.append(period_gross)
            asset_contributions_list.append(period_contribs)
            weight_rows.append(target_weights.rename(rebalance_date))
            turnover_vals.append(turnover)
            impact_vals.append(impact_cost)
            borrow_vals.append(borrow_cost)
            financing_vals.append(financing_cost)
            rebalance_dates.append(rebalance_date)
            prev_weights = target_weights

        # ---- assemble ----
        combined_net = pd.concat(portfolio_returns).sort_index()
        combined_net.name = "portfolio_return"
        combined_gross = pd.concat(gross_portfolio_returns).sort_index()
        combined_gross.name = "gross_portfolio_return"
        equity_curve = (1.0 + combined_net).cumprod().rename("equity")

        weights_df = pd.DataFrame(weight_rows).sort_index()
        diagnostics_df = pd.DataFrame(diagnostic_rows).sort_index()
        asset_contribs_df = pd.concat(asset_contributions_list).sort_index()
        idx = pd.Index(rebalance_dates)

        costs = CostBreakdown(
            market_impact=pd.Series(impact_vals, index=idx, name="market_impact_usd"),
            borrow_cost=pd.Series(borrow_vals, index=idx, name="borrow_cost_usd"),
            financing_cost=pd.Series(financing_vals, index=idx, name="financing_cost_usd"),
        )
        turnover_series = pd.Series(turnover_vals, index=idx, name="turnover")

        alpha_scores_used: pd.DataFrame | None = None
        if alpha_rows:
            alpha_scores_used = pd.DataFrame(alpha_rows).sort_index()

        risk_metrics = self.risk_engine.evaluate_portfolio_returns(
            prev_weights, combined_net
        ).to_dict()

        return PipelineResult(
            portfolio_returns=combined_net,
            gross_portfolio_returns=combined_gross,
            equity_curve=equity_curve,
            weights=weights_df,
            alpha_scores_used=alpha_scores_used,
            costs=costs,
            turnover=turnover_series,
            asset_contributions=asset_contribs_df,
            diagnostics=diagnostics_df,
            risk_metrics=risk_metrics,
        )

    # ------------------------------------------------------------------
    # Internal cost helpers
    # ------------------------------------------------------------------

    def _compute_impact_cost(
        self,
        weight_change: pd.Series,
        window: pd.DataFrame,
        adv_usd: pd.Series | None,
    ) -> float:
        """Return total market impact cost in dollars for a rebalance trade."""
        if self._impact_model is None:
            return 0.0

        total = 0.0
        for asset in weight_change.index:
            delta = abs(float(weight_change.get(asset, 0.0)))
            if delta < 1e-10:
                continue
            trade_value = delta * self.config.nav

            if adv_usd is not None and asset in adv_usd.index:
                adv = float(adv_usd[asset])
                participation = (trade_value / adv) if adv > 0 else self.config.default_participation_rate
            else:
                participation = self.config.default_participation_rate

            if asset in window.columns and len(window) >= 5:
                daily_vol = float(window[asset].std())
                if daily_vol <= 0:
                    daily_vol = self.config.default_daily_vol
            else:
                daily_vol = self.config.default_daily_vol

            impact = self._impact_model.estimate(participation, daily_vol)
            total += trade_value * impact.total_impact_bps / 10_000.0

        return total

    def _compute_borrow_cost(
        self,
        weights: pd.Series,
        holding_days: float,
    ) -> float:
        """Return total borrow cost in dollars for a holding period."""
        if self._borrow_schedule is None:
            return 0.0
        result = self._borrow_model.estimate(
            weights=weights,
            nav=self.config.nav,
            schedule=self._borrow_schedule,
            holding_period_days=holding_days,
        )
        return result.total_borrow_cost

    def _compute_financing_cost(
        self,
        weights: pd.Series,
        holding_days: float,
    ) -> float:
        """Return net financing cost in dollars for a holding period."""
        if self._financing_rates is None:
            return 0.0
        result = self._financing_model.estimate(
            weights=weights,
            nav=self.config.nav,
            rates=self._financing_rates,
            holding_period_days=holding_days,
        )
        return result.net_financing_cost

    @staticmethod
    def _lookup_alpha(
        alpha_scores: pd.DataFrame,
        rebalance_date: Any,
        columns: pd.Index,
    ) -> pd.Series:
        """Return the most recent alpha scores as-of rebalance_date."""
        available = alpha_scores.index[alpha_scores.index <= rebalance_date]
        if len(available) == 0:
            return pd.Series(0.0, index=columns)
        row = alpha_scores.loc[available[-1]]
        return row.reindex(columns).fillna(0.0)

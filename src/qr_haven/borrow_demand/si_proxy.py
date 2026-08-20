"""Real-time short interest proxy from daily loan transaction flows.

FINRA SI is published bi-monthly at T+14 lag; Markit is T-1.  Daily loan
transaction data is real-time.  Net loan flow (shares lent out minus shares
returned) is a proxy for the daily change in short interest:

    ΔSI_proxy(cusip, t) = new_loan_shares(t) − returned_loan_shares(t)
    SI_proxy(cusip, t)  = anchor_shares_short + Σ_{s=anchor+1}^{t} ΔSI_proxy(s)

Anchored to the most recent Markit reading, this produces an estimate that
leads Markit by up to one day and FINRA by up to fourteen.

Loan flow understates true SI because OTC-swap and synthetic-short positions
are not captured in the loan book.  A hedge-ratio correction (default 0.75,
calibratable against Markit actuals) adjusts for this systematic gap.

Reference: spec § Extension Model 2 — Real-Time Short Interest Proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class SIAnchor:
    """Most recent validated short interest reading for a single CUSIP.

    Typically sourced from IHS Markit (T-1) or FINRA (bi-monthly, T-14).
    """

    cusip: str
    as_of_date: date
    shares_short: float     # absolute share count at anchor date
    shares_float: float     # public float used for SI ratio denominator
    si_ratio: float         # = shares_short / shares_float at anchor date
    data_source: str = "markit"  # "markit" | "finra" | "estimated"


@dataclass
class SIProxyResult:
    """Real-time SI proxy estimate for one CUSIP on a specific as-of date."""

    cusip: str
    as_of_date: date
    shares_short_proxy: float       # hedge-ratio-corrected proxy share count
    si_ratio_proxy: float           # shares_short_proxy / shares_float
    anchor_date: date               # date of the Markit/FINRA anchor used
    anchor_age_trading_days: int    # trading days elapsed since anchor
    si_stale: bool                  # True when anchor_age > staleness_threshold
    hedge_ratio: float              # correction factor applied


class RealTimeSIProxy:
    """Build intraday-current short interest estimates from daily loan flows.

    Parameters
    ----------
    hedge_ratio : float
        Correction for OTC-swap / synthetic-short exposure not captured in the
        loan book.  Default 0.75 (loan flow captures ~75% of SI changes on
        average).  Calibrate per-universe with calibrate_hedge_ratio().
    staleness_threshold_days : int
        Trading days after which an anchor is considered stale and
        SIProxyResult.si_stale is set True.  Default 5 (one trading week).

    Usage
    -----
    proxy = RealTimeSIProxy(hedge_ratio=0.75)
    results = proxy.compute(loan_df, anchors, as_of_date=date.today())
    patched = proxy.patch_raw_features(raw_batch, results)
    """

    def __init__(
        self,
        hedge_ratio: float = 0.75,
        staleness_threshold_days: int = 5,
    ) -> None:
        if not (0.0 < hedge_ratio <= 2.0):
            raise ValueError("hedge_ratio must be in (0, 2]")
        if staleness_threshold_days < 1:
            raise ValueError("staleness_threshold_days must be ≥ 1")
        self.hedge_ratio = hedge_ratio
        self.staleness_threshold_days = staleness_threshold_days

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _net_flow(self, loan_df: pd.DataFrame) -> pd.DataFrame:
        """Compute net daily loan flow per (cusip, trade_date).

        loan_df must contain: cusip (str), trade_date (date/datetime),
        direction ('lend' | 'borrow'), quantity_shares (numeric).

        'lend'   = desk lends shares to a short seller  → SI increases
        'borrow' = shares returned to desk (close/recall) → SI decreases

        Returns DataFrame with columns [cusip, trade_date, net_flow].
        """
        df = loan_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

        lend = (
            df[df["direction"] == "lend"]
            .groupby(["cusip", "trade_date"])["quantity_shares"]
            .sum()
            .rename("lend_shares")
        )
        borrow = (
            df[df["direction"] == "borrow"]
            .groupby(["cusip", "trade_date"])["quantity_shares"]
            .sum()
            .rename("borrow_shares")
        )
        net = lend.to_frame().join(borrow, how="outer").fillna(0.0)
        net["net_flow"] = net["lend_shares"] - net["borrow_shares"]
        return net[["net_flow"]].reset_index()

    def _count_trading_days(
        self,
        start: date,
        end: date,
        trading_calendar: list[date] | None,
    ) -> int:
        """Count trading days strictly after start and up to end inclusive."""
        if trading_calendar is not None:
            return sum(1 for d in trading_calendar if start < d <= end)
        # Fallback: approximate with business days
        return int(np.busday_count(start, end))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        loan_df: pd.DataFrame,
        anchors: dict[str, SIAnchor],
        as_of_date: date,
        trading_calendar: list[date] | None = None,
    ) -> dict[str, SIProxyResult]:
        """Build SI proxy for all anchored CUSIPs as of *as_of_date*.

        Parameters
        ----------
        loan_df : DataFrame
            Loan transaction log (Dataset 2 subset).  Required columns:
            ``cusip``, ``trade_date``, ``direction`` ('lend'/'borrow'),
            ``quantity_shares``.
        anchors : dict[str, SIAnchor]
            Mapping from CUSIP to most recent validated SI reading.  Only
            CUSIPs present in this dict will have a proxy computed.
        as_of_date : date
            The date for which the proxy is being computed.
        trading_calendar : list[date], optional
            Ordered list of valid trading dates used to compute anchor age.
            If omitted, np.busday_count is used as an approximation.

        Returns
        -------
        dict mapping CUSIP → SIProxyResult for every CUSIP in *anchors*
        that has valid shares_float > 0.
        """
        flows = self._net_flow(loan_df)

        results: dict[str, SIProxyResult] = {}

        for cusip, anchor in anchors.items():
            if anchor.shares_float <= 0:
                continue

            # Cumulate loan flows between anchor date and as_of_date
            mask = (
                (flows["cusip"] == cusip)
                & (flows["trade_date"] > anchor.as_of_date)
                & (flows["trade_date"] <= as_of_date)
            )
            cumulative_flow = float(flows.loc[mask, "net_flow"].sum())

            # Apply hedge-ratio correction then add to anchor
            proxy_shares = anchor.shares_short + self.hedge_ratio * cumulative_flow
            proxy_shares = max(proxy_shares, 0.0)  # short interest cannot go negative

            si_ratio_proxy = proxy_shares / anchor.shares_float

            anchor_age = self._count_trading_days(
                anchor.as_of_date, as_of_date, trading_calendar
            )
            si_stale = anchor_age > self.staleness_threshold_days

            results[cusip] = SIProxyResult(
                cusip=cusip,
                as_of_date=as_of_date,
                shares_short_proxy=proxy_shares,
                si_ratio_proxy=si_ratio_proxy,
                anchor_date=anchor.as_of_date,
                anchor_age_trading_days=anchor_age,
                si_stale=si_stale,
                hedge_ratio=self.hedge_ratio,
            )

        return results

    def calibrate_hedge_ratio(
        self,
        loan_df: pd.DataFrame,
        markit_df: pd.DataFrame,
        lookback_days: int = 60,
    ) -> float:
        """Calibrate hedge_ratio against Markit actuals on a rolling window.

        For each pair of consecutive Markit readings (t1, t2) per CUSIP within
        the lookback window, the implied SI change is compared to observed loan
        net flow over that interval:

            implied_ΔSI  = markit_shares_short(t2) − markit_shares_short(t1)
            observed_flow = Σ net_flow(t1 < date ≤ t2)

        The calibrated hedge_ratio is the OLS slope (no intercept):

            hedge_ratio = Σ(observed_flow × implied_ΔSI) / Σ(observed_flow²)

        Parameters
        ----------
        loan_df : DataFrame
            Loan transaction log.  Same schema as compute().
        markit_df : DataFrame
            Historical Markit SI data.  Required columns: ``cusip``,
            ``as_of_date``, ``shares_short``.
        lookback_days : int
            Calendar days of history to use for calibration.

        Returns
        -------
        float — calibrated hedge_ratio, clipped to [0.3, 1.5].
        """
        markit = markit_df.copy()
        markit["as_of_date"] = pd.to_datetime(markit["as_of_date"]).dt.date

        cutoff = markit["as_of_date"].max() - pd.Timedelta(days=lookback_days)
        recent = markit[markit["as_of_date"] >= cutoff].sort_values("as_of_date")

        flows = self._net_flow(loan_df)
        flow_idx = flows.set_index(["cusip", "trade_date"])

        numerator = 0.0
        denominator = 0.0

        for cusip, grp in recent.groupby("cusip"):
            dates = sorted(grp["as_of_date"].tolist())
            si_map = dict(zip(grp["as_of_date"], grp["shares_short"]))

            for t1, t2 in zip(dates[:-1], dates[1:]):
                implied_delta = float(si_map[t2] - si_map[t1])

                try:
                    cusip_flows = flows[
                        (flows["cusip"] == cusip)
                        & (flows["trade_date"] > t1)
                        & (flows["trade_date"] <= t2)
                    ]
                    obs_flow = float(cusip_flows["net_flow"].sum())
                except (KeyError, TypeError):
                    obs_flow = 0.0

                if abs(obs_flow) < 1.0:
                    continue  # skip periods with no observable loan flow

                numerator += obs_flow * implied_delta
                denominator += obs_flow ** 2

        if denominator == 0.0:
            return self.hedge_ratio  # no data — return current value

        calibrated = numerator / denominator
        return float(np.clip(calibrated, 0.3, 1.5))

    def patch_raw_features(
        self,
        raw_batch: list,
        proxy_results: dict[str, SIProxyResult],
    ) -> list:
        """Replace si_ratio in RawFeatures with the proxy where available.

        For each RawFeature whose CUSIP has a proxy result, replaces
        ``si_ratio`` with ``si_ratio_proxy`` and sets ``si_stale``
        accordingly.  RawFeatures with no proxy are returned unchanged.

        Parameters
        ----------
        raw_batch : list[RawFeatures]
            Same-date cross-section to be patched.
        proxy_results : dict[str, SIProxyResult]
            Output of compute().

        Returns
        -------
        New list of RawFeatures with patched SI values.
        """
        from qr_haven.borrow_demand.features import RawFeatures

        patched = []
        for rf in raw_batch:
            result = proxy_results.get(rf.cusip)
            if result is None:
                patched.append(rf)
                continue

            import dataclasses
            patched.append(
                dataclasses.replace(
                    rf,
                    si_ratio=result.si_ratio_proxy,
                    si_stale=result.si_stale,
                )
            )
        return patched

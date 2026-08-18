"""Instrument types and specifications for synthetic inventory construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InstrumentType(str, Enum):
    CASH_EQUITY = "cash_equity"
    ETF = "etf"
    SSF = "ssf"                    # single-stock future
    OPTIONS_FORWARD = "options_forward"  # put-call parity synthetic forward
    TRS = "trs"                    # total-return swap
    REPO = "repo"                  # repo / reverse repo


@dataclass
class InstrumentSpec:
    """Full specification for one leg of a synthetic replication portfolio.

    Attributes
    ----------
    instrument_type : InstrumentType
    ticker : str
        Unique identifier used for mutual-exclusion lookups.
    funding_rate_annual : float
        Financing spread over risk-free, per annum (decimal; 0.015 = 150 bps).
    margin_rate : float
        Required margin as a fraction of notional (e.g. 0.10 for 10 %).
    rwa_weight : float
        Basel risk-weighted-asset weight (SA-CCR for derivatives).
    execution_cost_bps : float
        One-way market impact and spread in bps.
    basis_vol_annual : float
        Annualised residual basis volatility against the target exposure (0 = exact replica).
    feasibility : float
        Liquidity score in [0, 1].  0 = instrument unavailable.
    capacity : float
        Maximum fraction of the target notional this instrument can absorb, in (0, 1].
    delta : float
        First-order price sensitivity per unit of notional (negative for short-type instruments).
    gamma : float
        Second-order price sensitivity (non-zero for options legs).
    vega : float
        Volatility sensitivity (non-zero for options legs).
    adv_usd : float
        Average daily volume in USD, used for impact scaling in SOCP / SCA modes.
    """

    instrument_type: InstrumentType
    ticker: str
    funding_rate_annual: float
    margin_rate: float
    rwa_weight: float
    execution_cost_bps: float
    basis_vol_annual: float
    feasibility: float
    capacity: float
    delta: float = 1.0
    gamma: float = 0.0
    vega: float = 0.0
    adv_usd: float = 1_000_000.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.feasibility <= 1.0:
            raise ValueError(f"feasibility must be in [0, 1], got {self.feasibility}")
        if not 0.0 < self.capacity <= 1.0:
            raise ValueError(f"capacity must be in (0, 1], got {self.capacity}")
        if self.adv_usd <= 0:
            raise ValueError(f"adv_usd must be positive, got {self.adv_usd}")
        if self.execution_cost_bps < 0:
            raise ValueError(f"execution_cost_bps cannot be negative, got {self.execution_cost_bps}")
        if self.basis_vol_annual < 0:
            raise ValueError(f"basis_vol_annual cannot be negative, got {self.basis_vol_annual}")
        if self.margin_rate < 0:
            raise ValueError(f"margin_rate cannot be negative, got {self.margin_rate}")
        if self.rwa_weight < 0:
            raise ValueError(f"rwa_weight cannot be negative, got {self.rwa_weight}")

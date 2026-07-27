"""Stable contracts between data, research, portfolio, risk, execution, and UI layers."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol

import pandas as pd


class DataPortal(Protocol):
    """Read market, reference, and alternative data at a requested grain."""

    def load_prices(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        frequency: str,
    ) -> pd.DataFrame:
        """Return adjusted price history indexed by timestamp and symbol."""


class FeatureStore(Protocol):
    """Store and retrieve reusable research features."""

    def get_features(
        self,
        feature_names: Sequence[str],
        universe: Sequence[str],
        as_of: datetime,
    ) -> pd.DataFrame:
        """Return point-in-time features for the requested universe."""


class AlphaModel(Protocol):
    """Generate forecast, ranking, or position-intent signals."""

    name: str

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train or calibrate the model."""

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return model scores indexed by asset."""


class PortfolioOptimizer(Protocol):
    """Convert expected returns, risk, and constraints into target weights."""

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: Mapping[str, Any],
    ) -> pd.Series:
        """Return target portfolio weights indexed by asset."""


class RiskEngine(Protocol):
    """Compute portfolio and strategy risk diagnostics."""

    def evaluate(self, weights: pd.Series, returns: pd.DataFrame) -> Mapping[str, float]:
        """Return risk metrics such as volatility, VaR, CVaR, and drawdown."""


class ExecutionSimulator(Protocol):
    """Estimate fill quality and transaction costs for target portfolio changes."""

    def simulate(self, orders: pd.DataFrame, market_state: pd.DataFrame) -> pd.DataFrame:
        """Return simulated fills, slippage, and implementation shortfall."""


class TerminalRenderable(Protocol):
    """Expose QR-Haven outputs in a format a terminal UI can consume."""

    def to_terminal_payload(self) -> Mapping[str, Any]:
        """Return serializable data for market_terminal panels, tables, and charts."""


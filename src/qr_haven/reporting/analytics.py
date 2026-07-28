"""Portfolio analytics used by reports and terminal views."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qr_haven.backtesting import BacktestResult


@dataclass(frozen=True)
class PortfolioAnalytics:
    """Derived analytics for a backtest return stream."""

    cumulative_return: pd.Series
    drawdown: pd.Series
    rolling_volatility: pd.Series
    rolling_sharpe: pd.Series
    rolling_turnover: pd.Series


def calculate_cumulative_return(returns: pd.Series) -> pd.Series:
    """Calculate cumulative return from a return series."""

    cumulative = (1.0 + returns.dropna().astype(float)).cumprod() - 1.0
    cumulative.name = "cumulative_return"
    return cumulative


def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """Calculate drawdown from a return series."""

    wealth = (1.0 + returns.dropna().astype(float)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    drawdown.name = "drawdown"
    return drawdown


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    annualization: float = 252.0,
) -> pd.Series:
    """Calculate annualized rolling volatility."""

    volatility = returns.dropna().astype(float).rolling(window=window).std(ddof=1)
    volatility = volatility * np.sqrt(annualization)
    volatility.name = "rolling_volatility"
    return volatility


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 21,
    annualization: float = 252.0,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Calculate annualized rolling Sharpe ratio."""

    clean_returns = returns.dropna().astype(float)
    rolling_mean = clean_returns.rolling(window=window).mean() * annualization
    rolling_volatility = calculate_rolling_volatility(
        clean_returns,
        window=window,
        annualization=annualization,
    )
    sharpe = (rolling_mean - risk_free_rate) / rolling_volatility.replace(0.0, np.nan)
    sharpe = sharpe.fillna(0.0)
    sharpe.name = "rolling_sharpe"
    return sharpe


def calculate_rolling_turnover(turnover: pd.Series, window: int = 3) -> pd.Series:
    """Calculate rolling average turnover across rebalance dates."""

    rolling = turnover.dropna().astype(float).rolling(window=window).mean()
    rolling.name = "rolling_turnover"
    return rolling


def calculate_portfolio_analytics(
    result: BacktestResult,
    return_window: int = 21,
    turnover_window: int = 3,
    annualization: float = 252.0,
) -> PortfolioAnalytics:
    """Calculate standard analytics for a backtest result."""

    return PortfolioAnalytics(
        cumulative_return=calculate_cumulative_return(result.portfolio_returns),
        drawdown=calculate_drawdown(result.portfolio_returns),
        rolling_volatility=calculate_rolling_volatility(
            result.portfolio_returns,
            window=return_window,
            annualization=annualization,
        ),
        rolling_sharpe=calculate_rolling_sharpe(
            result.portfolio_returns,
            window=return_window,
            annualization=annualization,
        ),
        rolling_turnover=calculate_rolling_turnover(result.turnover, window=turnover_window),
    )


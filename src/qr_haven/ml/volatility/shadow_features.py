"""Causal V8A features, including the final session with no forward outcome."""

from __future__ import annotations

import math
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from qr_haven.ml.volatility.challenger_data import CHALLENGER_FEATURES


def session_calendar(start: pd.Timestamp, end: pd.Timestamp) -> Any:
    return xcals.get_calendar("XNYS", start=start.date(), end=end.date())


def latest_completed_session(now: pd.Timestamp) -> pd.Timestamp:
    if now.tzinfo is None:
        raise ValueError("Execution time must include a timezone")
    day = now.tz_convert("UTC").tz_localize(None).normalize()
    calendar = session_calendar(day - pd.Timedelta(days=14), day + pd.Timedelta(days=14))
    return pd.Timestamp(calendar.schedule.loc[calendar.schedule["close"] <= now].index[-1])


def validate_shadow_prices(canonical: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "symbol", "open", "high", "low", "close", "adjusted_close"}
    if required - set(canonical.columns) or len(canonical) < 61:
        raise ValueError("Shadow scoring requires adjusted SPY OHLC and at least 61 sessions")
    frame = canonical.copy()
    times = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    sessions = pd.DatetimeIndex(times.dt.tz_localize(None).dt.normalize())
    if not sessions.is_monotonic_increasing or not sessions.is_unique:
        raise ValueError("Price sessions must be chronological and unique")
    if set(frame["symbol"]) != {"SPY"}:
        raise ValueError("The frozen model accepts SPY only")
    calendar = session_calendar(sessions[0], sessions[-1])
    if not sessions.equals(calendar.sessions):
        raise ValueError("Prices must cover every XNYS session without gaps")
    columns = ["open", "high", "low", "close", "adjusted_close"]
    values = frame[columns].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Adjusted OHLC must be finite and positive")
    if (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1) - 1e-8).any()
        or (frame["low"] > frame[["open", "close", "high"]].min(axis=1) + 1e-8).any()
        or not np.allclose(frame["close"], frame["adjusted_close"], rtol=1e-12, atol=0)
    ):
        raise ValueError("OHLC must use a consistent adjusted price basis")
    frame.index = sessions
    frame.index.name = "session"
    return frame


def causal_features(prices: pd.DataFrame, origin: pd.Timestamp) -> dict[str, float]:
    """Use exactly 61 prices ending at origin; no future prices or labels are read."""
    history = prices.loc[:origin].tail(61)
    if len(history) != 61 or history.index[-1] != origin:
        raise ValueError("Origin needs 60 preceding sessions and its own completed price")
    close = history["adjusted_close"].to_numpy(float)
    # Match the research ratio-before-log implementation, including numerical precision.
    returns = np.log(close[1:] / close[:-1])
    ranges = np.log(history["high"].to_numpy(float) / history["low"].to_numpy(float))
    features = {
        f"trailing_vol_{window}": math.sqrt(
            252 / window * float(np.square(returns[-window:]).sum())
        )
        for window in (5, 10, 20, 60)
    }
    features.update(
        {
            "log_return_1": float(returns[-1]),
            "log_return_5": math.log(close[-1] / close[-6]),
            "abs_log_return_1": float(abs(returns[-1])),
            "vol_ratio_5_20": (
                features["trailing_vol_5"] / features["trailing_vol_20"]
                if features["trailing_vol_20"]
                else 0.0
            ),
            "drawdown_60": float(close[-1] / close[-60:].max() - 1),
            "negative_return_share_20": float((returns[-20:] < 0).mean()),
            "vol_of_vol_20": float(math.sqrt(252) * np.abs(returns[-20:]).std(ddof=1)),
        }
    )
    for window in (5, 20, 60):
        values = returns[-window:]
        features[f"downside_vol_{window}"] = math.sqrt(
            252 / window * float(np.square(np.minimum(values, 0)).sum())
        )
        if window != 60:
            features[f"upside_vol_{window}"] = math.sqrt(
                252 / window * float(np.square(np.maximum(values, 0)).sum())
            )
            features[f"max_abs_return_{window}"] = float(np.abs(values).max())
            features[f"parkinson_vol_{window}"] = math.sqrt(
                252 * float(np.square(ranges[-window:]).mean()) / (4 * math.log(2))
            )
    return {name: features[name] for name in CHALLENGER_FEATURES}


def forecast_timing(origin: pd.Timestamp) -> dict[str, str]:
    calendar = session_calendar(origin, origin + pd.Timedelta(days=30))
    schedule = calendar.schedule
    return {
        "available_at": schedule.iloc[0]["close"].isoformat(),
        "next_open": schedule.iloc[1]["open"].isoformat(),
        "label_start": schedule.index[1].date().isoformat(),
        "label_end": schedule.index[5].date().isoformat(),
        "outcome_available_at": schedule.iloc[5]["close"].isoformat(),
    }

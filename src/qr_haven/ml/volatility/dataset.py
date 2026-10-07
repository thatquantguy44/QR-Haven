"""Construct audited point-in-time volatility features and forward outcomes."""

from __future__ import annotations

import json
import math
from typing import Any

import exchange_calendars as xcals
import numpy as np
import numpy.typing as npt
import pandas as pd

from qr_haven.ml.volatility.contracts import FEATURES, VolatilityDataset, VolatilityProfile

ANNUALIZATION = 252
FORWARD_SESSIONS = 5
MAX_TRAILING_RETURNS = 60


def _session_strings(sessions: pd.DatetimeIndex) -> str:
    return json.dumps(sessions.strftime("%Y-%m-%d").tolist(), separators=(",", ":"))


def _source_sha256(manifest: dict[str, Any]) -> str:
    for key in ("source_sha256", "raw_price_sha256"):
        value = manifest.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value
    raise ValueError("Source manifest lacks a valid source SHA-256")


def _validate_canonical_prices(
    canonical: pd.DataFrame, profile: VolatilityProfile
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    required = {"timestamp", "symbol", "adjusted_close"}
    missing = required - set(canonical.columns)
    if missing:
        raise ValueError(f"Canonical prices are missing columns: {sorted(missing)}")
    frame = canonical.copy()
    try:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame["adjusted_close"] = frame["adjusted_close"].astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Canonical timestamps and adjusted_close must be valid") from exc
    if frame.empty:
        raise ValueError("Canonical prices contain no rows")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Canonical prices contain duplicate sessions")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Canonical prices must be chronological")
    if set(frame["symbol"].astype(str)) != {profile.symbol}:
        raise ValueError(f"Canonical prices must contain only symbol {profile.symbol}")
    prices = frame["adjusted_close"].to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Canonical adjusted_close values must be positive and finite")
    sessions = pd.DatetimeIndex(frame["timestamp"].dt.tz_convert(None).dt.normalize())
    calendar = xcals.get_calendar(
        profile.calendar, start=profile.study_start, end=profile.study_end
    )
    expected = calendar.sessions
    if not sessions.equals(expected):
        missing_sessions = expected.difference(sessions)
        extra_sessions = sessions.difference(expected)
        raise ValueError(
            "Canonical sessions do not match the frozen profile calendar: "
            f"missing={missing_sessions.strftime('%Y-%m-%d').tolist()}, "
            f"extra={extra_sessions.strftime('%Y-%m-%d').tolist()}"
        )
    frame = frame.reset_index(drop=True)
    return frame, sessions


def build_volatility_dataset(
    canonical: pd.DataFrame,
    source_manifest: dict[str, Any],
    profile: VolatilityProfile,
) -> VolatilityDataset:
    """Build all eligible origins without applying a fold-specific class threshold."""
    frame, sessions = _validate_canonical_prices(canonical, profile)
    if len(frame) <= MAX_TRAILING_RETURNS + FORWARD_SESSIONS:
        raise ValueError("Too few sessions for 60-session features and five-session labels")
    prices = frame["adjusted_close"].to_numpy(dtype=float)
    returns: npt.NDArray[np.float64] = np.full(len(prices), np.nan, dtype=float)
    returns[1:] = np.log(prices[1:] / prices[:-1])
    calendar = xcals.get_calendar(
        profile.calendar, start=profile.study_start, end=profile.study_end
    )
    closes = calendar.schedule["close"]

    observation_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    last_origin = len(prices) - FORWARD_SESSIONS
    for origin in range(MAX_TRAILING_RETURNS, last_origin):
        as_of = sessions[origin]
        sample_id = f"{profile.profile_id}/{profile.symbol}/{as_of.date().isoformat()}"
        trailing: dict[int, float] = {}
        for window in (5, 10, 20, 60):
            window_returns = returns[origin - window + 1 : origin + 1]
            trailing[window] = float(
                math.sqrt((ANNUALIZATION / window) * float(np.square(window_returns).sum()))
            )
        forward_returns = returns[origin + 1 : origin + FORWARD_SESSIONS + 1]
        forward_volatility = float(
            math.sqrt(
                (ANNUALIZATION / FORWARD_SESSIONS)
                * float(np.square(forward_returns).sum())
            )
        )
        one_return = float(returns[origin])
        five_return = float(math.log(prices[origin] / prices[origin - 5]))
        ratio = trailing[5] / trailing[20] if trailing[20] != 0 else 0.0
        drawdown = float(prices[origin] / prices[origin - 59 : origin + 1].max() - 1)
        label_start = sessions[origin + 1]
        label_end = sessions[origin + FORWARD_SESSIONS]
        observation_rows.append(
            {
                "sample_id": sample_id,
                "as_of": as_of,
                "available_at": closes.loc[as_of],
                "label_start": label_start,
                "label_end": label_end,
                "trailing_vol_5": trailing[5],
                "trailing_vol_10": trailing[10],
                "trailing_vol_20": trailing[20],
                "trailing_vol_60": trailing[60],
                "log_return_1": one_return,
                "log_return_5": five_return,
                "abs_log_return_1": abs(one_return),
                "vol_ratio_5_20": ratio,
                "drawdown_60": drawdown,
                "forward_vol_5": forward_volatility,
            }
        )
        windows = {
            "trailing_vol_5_price_sessions": sessions[origin - 5 : origin + 1],
            "trailing_vol_10_price_sessions": sessions[origin - 10 : origin + 1],
            "trailing_vol_20_price_sessions": sessions[origin - 20 : origin + 1],
            "trailing_vol_60_price_sessions": sessions[origin - 60 : origin + 1],
            "log_return_1_price_sessions": sessions[[origin - 1, origin]],
            "log_return_5_price_sessions": sessions[[origin - 5, origin]],
            "abs_log_return_1_price_sessions": sessions[[origin - 1, origin]],
            "vol_ratio_5_20_price_sessions": sessions[origin - 20 : origin + 1],
            "drawdown_60_price_sessions": sessions[origin - 59 : origin + 1],
            "forward_vol_5_price_sessions": sessions[origin : origin + 6],
        }
        audit_rows.append(
            {
                "sample_id": sample_id,
                "as_of": as_of,
                "label_start": label_start,
                "label_end": label_end,
                **{name: _session_strings(values) for name, values in windows.items()},
            }
        )

    observations = pd.DataFrame(observation_rows).set_index("sample_id")
    interval_audit = pd.DataFrame(audit_rows).set_index("sample_id")
    result = VolatilityDataset(
        observations=observations,
        interval_audit=interval_audit,
        profile=profile,
        source_manifest=source_manifest,
        source_sha256=_source_sha256(source_manifest),
    )
    validate_volatility_dataset(result)
    return result


def load_profile_dataset(profile: VolatilityProfile) -> VolatilityDataset:
    """Load a verified prepared snapshot and construct its V2 observation table."""
    if profile.profile_id == "spx-local-v1":
        from qr_haven.data.spx import load_prepared_local_spx_snapshot

        snapshot = load_prepared_local_spx_snapshot(profile.snapshot_dir)
        return build_volatility_dataset(snapshot.canonical, snapshot.manifest, profile)
    elif profile.profile_id == "tiingo-spy-v1":
        from qr_haven.data.tiingo import load_tiingo_spy_snapshot

        tiingo_snapshot = load_tiingo_spy_snapshot(profile.snapshot_dir)
        return build_volatility_dataset(
            tiingo_snapshot.canonical, tiingo_snapshot.manifest, profile
        )
    else:
        raise ValueError(f"No snapshot loader for volatility profile: {profile.profile_id}")


def validate_volatility_dataset(dataset: VolatilityDataset) -> None:
    """Validate shape, timing, and finite-value invariants for a built dataset."""
    observations = dataset.observations
    audit = dataset.interval_audit
    required = {"as_of", "available_at", "label_start", "label_end", "forward_vol_5", *FEATURES}
    if required - set(observations.columns):
        raise ValueError("Volatility observations lack required columns")
    if observations.empty or not observations.index.is_unique:
        raise ValueError("Volatility observations require unique, nonempty sample IDs")
    if not observations.index.equals(audit.index):
        raise ValueError("Observation and interval-audit sample IDs must align")
    numeric = observations.loc[:, [*FEATURES, "forward_vol_5"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (observations["forward_vol_5"] < 0).any():
        raise ValueError("Volatility features and outcomes must be finite and outcomes nonnegative")
    if not observations["as_of"].is_monotonic_increasing:
        raise ValueError("Volatility observations must be chronological")
    if not (observations["label_start"] > observations["as_of"]).all():
        raise ValueError("Every label must begin after its forecast origin")
    if not (observations["label_end"] >= observations["label_start"]).all():
        raise ValueError("Every label end must be on or after its label start")
    available_dates = (
        pd.to_datetime(observations["available_at"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    if not np.array_equal(available_dates.to_numpy(), observations["as_of"].to_numpy()):
        raise ValueError("Forecast availability must be after the matching as-of session")


def fold_targets(
    dataset: VolatilityDataset, sample_ids: tuple[str, ...], threshold: float
) -> pd.Series:
    """Apply a frozen training threshold to specified observation IDs."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Volatility threshold must be finite and nonnegative")
    try:
        outcomes = dataset.observations.loc[list(sample_ids), "forward_vol_5"]
    except KeyError as exc:
        raise ValueError("Unknown sample ID in target request") from exc
    return (outcomes > threshold).astype("int64").rename("target")

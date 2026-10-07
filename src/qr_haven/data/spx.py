"""Validation and preparation for the locally supplied S&P 500 index CSV."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from qr_haven.data.csv import CSVPriceDataPortal

DEFAULT_INPUT = Path("data/input/spx/SPX.csv")
DEFAULT_OUTPUT = Path("data/processed/market_data/spx-local-2005-2020-v1")
DEFAULT_STUDY_START = date(2005, 1, 1)
DEFAULT_STUDY_END = date(2020, 11, 4)
DEFAULT_CALENDAR = "XNYS"
SYMBOL = "SPX"
REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume")


@dataclass(frozen=True)
class LocalSPXSnapshot:
    """Validated local price-index data restricted to a declared study window."""

    canonical: pd.DataFrame
    manifest: dict[str, Any]
    audit: dict[str, Any]

    def price_portal(self, canonical_path: Path) -> CSVPriceDataPortal:
        """Return the existing CSV portal over a persisted canonical CSV."""
        return CSVPriceDataPortal(canonical_path)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def validate_local_spx_bytes(
    payload: bytes,
    *,
    study_start: date = DEFAULT_STUDY_START,
    study_end: date = DEFAULT_STUDY_END,
    calendar_name: str = DEFAULT_CALENDAR,
) -> LocalSPXSnapshot:
    """Validate a Yahoo-style local SPX CSV and create a canonical study frame."""
    if study_start > study_end:
        raise ValueError("study_start must be on or before study_end")
    try:
        header = payload.splitlines()[0].decode("utf-8").split(",")
    except (IndexError, UnicodeDecodeError) as exc:
        raise ValueError("Local SPX input is not a UTF-8 CSV") from exc
    if len(header) != len(set(header)):
        raise ValueError("Local SPX CSV contains duplicate column names")
    missing_columns = set(REQUIRED_COLUMNS) - set(header)
    if missing_columns:
        raise ValueError(f"Local SPX CSV is missing columns: {sorted(missing_columns)}")
    try:
        raw = pd.read_csv(io.BytesIO(payload), usecols=list(REQUIRED_COLUMNS))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Local SPX input is not a valid CSV: {exc}") from exc
    if raw.empty:
        raise ValueError("Local SPX CSV contains no rows")

    try:
        timestamps = pd.to_datetime(raw["Date"], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Local SPX CSV contains invalid dates") from exc
    session_dates = timestamps.dt.tz_convert(None).dt.normalize()
    if session_dates.duplicated().any():
        raise ValueError("Local SPX CSV contains duplicate sessions")
    if not session_dates.is_monotonic_increasing:
        raise ValueError("Local SPX CSV must be chronological")

    numeric_columns = list(REQUIRED_COLUMNS[1:])
    parsed = raw.copy()
    try:
        parsed[numeric_columns] = parsed[numeric_columns].astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Local SPX CSV contains nonnumeric values") from exc
    values = parsed[numeric_columns].to_numpy().ravel()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Local SPX CSV contains missing or nonfinite values")
    price_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    if (parsed[price_columns] <= 0).any().any():
        raise ValueError("Local SPX prices must be positive")
    if (parsed["Volume"] < 0).any():
        raise ValueError("Local SPX volume must be nonnegative")
    if (parsed["Volume"] % 1 != 0).any():
        raise ValueError("Local SPX volume must contain whole numbers")
    if (parsed["High"] < parsed[["Open", "Low", "Close"]].max(axis=1)).any():
        raise ValueError("Local SPX high is below another OHLC value")
    if (parsed["Low"] > parsed[["Open", "High", "Close"]].min(axis=1)).any():
        raise ValueError("Local SPX low is above another OHLC value")

    archive_start = session_dates.iloc[0].date()
    archive_end = session_dates.iloc[-1].date()
    if archive_start > study_start or archive_end < study_end:
        raise ValueError(
            f"Local SPX coverage {archive_start}–{archive_end} does not cover "
            f"{study_start}–{study_end}"
        )
    study_mask = (session_dates >= pd.Timestamp(study_start)) & (
        session_dates <= pd.Timestamp(study_end)
    )
    study = parsed.loc[study_mask].reset_index(drop=True)
    study_timestamps = timestamps.loc[study_mask].reset_index(drop=True)
    if study.empty:
        raise ValueError("Local SPX study window contains no sessions")
    observed = pd.DatetimeIndex(study_timestamps.dt.tz_convert(None).dt.normalize())
    calendar = xcals.get_calendar(calendar_name, start=study_start, end=study_end)
    expected = calendar.sessions
    missing_sessions = expected.difference(observed)
    extra_sessions = observed.difference(expected)
    if len(missing_sessions) or len(extra_sessions):
        raise ValueError(
            "Local SPX study sessions do not match the exchange calendar: "
            f"missing={missing_sessions.strftime('%Y-%m-%d').tolist()}, "
            f"extra={extra_sessions.strftime('%Y-%m-%d').tolist()}"
        )

    canonical = pd.DataFrame(
        {
            "timestamp": study_timestamps,
            "symbol": SYMBOL,
            "open": study["Open"],
            "high": study["High"],
            "low": study["Low"],
            "close": study["Close"],
            "adjusted_close": study["Adj Close"],
            "volume": study["Volume"].astype("int64"),
            "frequency": "daily",
        }
    )
    audit = {
        "schema_version": 1,
        "symbol": SYMBOL,
        "calendar": calendar_name,
        "calendar_library": "exchange-calendars",
        "calendar_library_version": importlib.metadata.version("exchange-calendars"),
        "archive_first_date": archive_start.isoformat(),
        "archive_last_date": archive_end.isoformat(),
        "archive_rows": len(raw),
        "study_start": study_start.isoformat(),
        "study_end": study_end.isoformat(),
        "study_first_session": observed[0].date().isoformat(),
        "study_last_session": observed[-1].date().isoformat(),
        "study_rows": len(study),
        "expected_sessions": len(expected),
        "missing_sessions": [],
        "extra_sessions": [],
        "duplicate_sessions": 0,
        "adjusted_close_differs_from_close": int((study["Adj Close"] != study["Close"]).sum()),
        "adjusted_close_positive_finite": True,
    }
    manifest = {
        "schema_version": 1,
        "dataset_id": (
            f"spx_local_price_index_{study_start.isoformat()}_{study_end.isoformat()}_v1"
        ),
        "provider": "locally supplied CSV; upstream provider unverified",
        "symbol": SYMBOL,
        "source_filename": DEFAULT_INPUT.name,
        "source_sha256": _sha256(payload),
        "license": "unverified; keep local and do not redistribute",
        "redistribution_permitted": False,
        "review_status": "usable_for_local_research; source_and_license_unverified",
        "price_basis": (
            "S&P 500 price-index close; the supplied Adj Close equals Close and does not "
            "represent a dividend-reinvested total-return index"
        ),
        "revision_policy": "Unknown; the immutable local bytes define this experiment's version.",
        "timestamp_convention": (
            "Date is treated as the XNYS session label and available only after that session "
            "closes."
        ),
        "calendar": calendar_name,
        "study_start": study_start.isoformat(),
        "study_end": study_end.isoformat(),
        "raw_bytes": len(payload),
        "audit": audit,
    }
    return LocalSPXSnapshot(canonical=canonical, manifest=manifest, audit=audit)


def prepare_local_spx_snapshot(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    study_start: date = DEFAULT_STUDY_START,
    study_end: date = DEFAULT_STUDY_END,
    calendar_name: str = DEFAULT_CALENDAR,
) -> LocalSPXSnapshot:
    """Validate local input and atomically publish an immutable canonical snapshot."""
    payload = Path(input_path).read_bytes()
    snapshot = validate_local_spx_bytes(
        payload,
        study_start=study_start,
        study_end=study_end,
        calendar_name=calendar_name,
    )
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        canonical_bytes = snapshot.canonical.to_csv(index=False).encode()
        manifest = {
            **snapshot.manifest,
            "files": {"spx_daily_canonical.csv": _sha256(canonical_bytes)},
        }
        for path, data in (
            (temporary / "spx_daily_canonical.csv", canonical_bytes),
            (temporary / "manifest.json", _json_bytes(manifest)),
        ):
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return LocalSPXSnapshot(snapshot.canonical, manifest, snapshot.audit)


def load_prepared_local_spx_snapshot(output_dir: Path) -> LocalSPXSnapshot:
    """Load a prepared local SPX snapshot and verify its canonical-file hash."""
    root = Path(output_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Prepared local SPX manifest is unavailable or invalid: {root}") from exc
    canonical_path = root / "spx_daily_canonical.csv"
    try:
        payload = canonical_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Prepared local SPX canonical CSV is unavailable: {root}") from exc
    if manifest.get("files", {}).get(canonical_path.name) != _sha256(payload):
        raise ValueError("Snapshot hash mismatch: spx_daily_canonical.csv")
    canonical = pd.read_csv(canonical_path)
    audit = manifest.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("Prepared local SPX manifest lacks an audit object")
    return LocalSPXSnapshot(canonical=canonical, manifest=manifest, audit=audit)

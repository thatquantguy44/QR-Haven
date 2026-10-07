"""Licensed Tiingo EOD acquisition and validation for the SPY volatility study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from qr_haven.data.csv import CSVPriceDataPortal

TIINGO_PROVIDER = "Tiingo End-of-Day Composite Prices"
TIINGO_DOCUMENTATION_URL = "https://www.tiingo.com/documentation/end-of-day"
TIINGO_PRICING_URL = "https://www.tiingo.com/about/pricing"
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"
TIINGO_LICENSE = "Internal use only; redistribution prohibited without a separate license"
DEFAULT_SYMBOL = "SPY"
DEFAULT_START = date(2005, 1, 1)
DEFAULT_END = date(2025, 12, 31)
DEFAULT_CALENDAR = "XNYS"
MAX_RESPONSE_BYTES = 10 * 1024 * 1024
REQUIRED_COLUMNS = (
    "date",
    "close",
    "high",
    "low",
    "open",
    "volume",
    "adjClose",
    "adjHigh",
    "adjLow",
    "adjOpen",
    "adjVolume",
    "divCash",
    "splitFactor",
)


@dataclass(frozen=True)
class TiingoSnapshot:
    """Validated raw provider rows and a QR Haven-compatible canonical frame."""

    raw: pd.DataFrame
    canonical: pd.DataFrame
    metadata: dict[str, Any]
    manifest: dict[str, Any]
    audit: dict[str, Any]

    def price_portal(self, canonical_path: Path) -> CSVPriceDataPortal:
        """Return the existing CSV portal over this snapshot's persisted canonical CSV."""
        return CSVPriceDataPortal(canonical_path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _request(url: str, token: str, accept: str) -> bytes:
    """Bounded HTTPS GET with two retries for transient provider/network failures."""
    if not token.strip():
        raise ValueError("TIINGO_API_TOKEN is empty; create a Tiingo account and export the token")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "Authorization": f"Token {token}",
            "User-Agent": "QR-Haven/0.1 market-data research",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = bytes(response.read(MAX_RESPONSE_BYTES + 1))
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ValueError("Tiingo response exceeded the 10 MiB safety limit")
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise ValueError("Tiingo rejected the API token or plan entitlement") from exc
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise ValueError(f"Tiingo request failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 2:
                raise ValueError(f"Tiingo request failed after three attempts: {exc}") from exc
        time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def _parse_metadata(payload: bytes, symbol: str, start: date, end: date) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Tiingo metadata response was not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Tiingo metadata response must be a JSON object")
    if str(value.get("ticker", "")).upper() != symbol:
        raise ValueError(f"Tiingo metadata ticker mismatch: expected {symbol}")
    try:
        available_start = date.fromisoformat(str(value["startDate"])[:10])
        available_end = date.fromisoformat(str(value["endDate"])[:10])
    except (KeyError, ValueError) as exc:
        raise ValueError("Tiingo metadata lacks valid startDate/endDate coverage") from exc
    if available_start > start or available_end < end:
        raise ValueError(
            "Tiingo metadata coverage "
            f"{available_start}–{available_end} does not cover {start}–{end}"
        )
    return value


def _parse_prices(payload: bytes) -> pd.DataFrame:
    try:
        header = payload.splitlines()[0].decode("utf-8").split(",")
    except (IndexError, UnicodeDecodeError) as exc:
        raise ValueError("Tiingo price response is not a UTF-8 CSV") from exc
    if len(header) != len(set(header)):
        raise ValueError("Tiingo price CSV contains duplicate column names")
    missing = set(REQUIRED_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"Tiingo price CSV is missing columns: {sorted(missing)}")
    try:
        frame = pd.read_csv(io.BytesIO(payload), usecols=list(REQUIRED_COLUMNS))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Tiingo price response is not a valid CSV: {exc}") from exc
    return frame


def _validate_prices(
    frame: pd.DataFrame,
    *,
    symbol: str,
    start: date,
    end: date,
    calendar_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        raise ValueError("Tiingo price CSV contains no rows")
    parsed = frame.copy()
    try:
        timestamps = pd.to_datetime(parsed["date"], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Tiingo price CSV contains invalid timestamps") from exc
    if timestamps.dt.normalize().duplicated().any():
        raise ValueError("Tiingo price CSV contains duplicate sessions")
    session_dates = timestamps.dt.tz_convert(None).dt.normalize()
    if not session_dates.is_monotonic_increasing:
        raise ValueError("Tiingo price CSV must be chronological")

    numeric_columns = [column for column in REQUIRED_COLUMNS if column != "date"]
    try:
        parsed[numeric_columns] = parsed[numeric_columns].astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Tiingo price CSV contains nonnumeric values") from exc
    if not all(math.isfinite(value) for value in parsed[numeric_columns].to_numpy().ravel()):
        raise ValueError("Tiingo price CSV contains missing or nonfinite values")
    price_columns = ["close", "high", "low", "open", "adjClose", "adjHigh", "adjLow", "adjOpen"]
    if (parsed[price_columns] <= 0).any().any():
        raise ValueError("Tiingo raw and adjusted prices must be positive")
    if (parsed[["volume", "adjVolume", "divCash"]] < 0).any().any():
        raise ValueError("Tiingo volumes and cash dividends must be nonnegative")
    if (parsed["volume"] % 1 != 0).any():
        raise ValueError("Tiingo raw volume must contain whole numbers")
    if (parsed["splitFactor"] <= 0).any():
        raise ValueError("Tiingo split factors must be positive")

    calendar = xcals.get_calendar(calendar_name, start=start, end=end)
    expected = calendar.sessions
    observed = pd.DatetimeIndex(session_dates)
    missing_sessions = expected.difference(observed)
    extra_sessions = observed.difference(expected)
    if len(missing_sessions) or len(extra_sessions):
        raise ValueError(
            "Tiingo sessions do not match the exchange calendar: "
            f"missing={missing_sessions.strftime('%Y-%m-%d').tolist()}, "
            f"extra={extra_sessions.strftime('%Y-%m-%d').tolist()}"
        )

    canonical = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": parsed["adjOpen"],
            "high": parsed["adjHigh"],
            "low": parsed["adjLow"],
            "close": parsed["adjClose"],
            "adjusted_close": parsed["adjClose"],
            "volume": parsed["volume"].astype("int64"),
            "frequency": "daily",
        }
    )
    audit = {
        "schema_version": 1,
        "symbol": symbol,
        "calendar": calendar_name,
        "calendar_library": "exchange-calendars",
        "calendar_library_version": importlib.metadata.version("exchange-calendars"),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "first_session": observed[0].date().isoformat(),
        "last_session": observed[-1].date().isoformat(),
        "rows": len(parsed),
        "expected_sessions": len(expected),
        "missing_sessions": [],
        "extra_sessions": [],
        "duplicate_sessions": 0,
        "cash_dividend_sessions": int((parsed["divCash"] != 0).sum()),
        "split_sessions": int((parsed["splitFactor"] != 1).sum()),
        "adjusted_close_min": float(parsed["adjClose"].min()),
        "adjusted_close_max": float(parsed["adjClose"].max()),
        "adjusted_close_positive_finite": True,
        "canonical_ohlc_basis": "Tiingo adjusted OHLC",
    }
    return canonical, audit


def validate_tiingo_snapshot_bytes(
    price_bytes: bytes,
    metadata_bytes: bytes,
    *,
    symbol: str = DEFAULT_SYMBOL,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    calendar_name: str = DEFAULT_CALENDAR,
) -> TiingoSnapshot:
    """Validate provider bytes without performing network or filesystem operations."""
    symbol = symbol.upper()
    metadata = _parse_metadata(metadata_bytes, symbol, start, end)
    raw = _parse_prices(price_bytes)
    canonical, audit = _validate_prices(
        raw, symbol=symbol, start=start, end=end, calendar_name=calendar_name
    )
    manifest = {
        "schema_version": 1,
        "provider": TIINGO_PROVIDER,
        "symbol": symbol,
        "source": "Tiingo EOD historical prices API",
        "documentation_url": TIINGO_DOCUMENTATION_URL,
        "pricing_and_license_url": TIINGO_PRICING_URL,
        "license": TIINGO_LICENSE,
        "redistribution_permitted": False,
        "adjustment": "Tiingo adjClose; split- and dividend-adjusted using CRSP methodology",
        "revision_policy": (
            "Adjusted history is a current revised view and is not a point-in-time archive; "
            "the immutable downloaded bytes define this experiment's data version."
        ),
        "timestamp_convention": (
            "Provider date normalized to a UTC session label; data are treated as available only "
            "after that session's official close."
        ),
        "calendar": calendar_name,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "raw_price_sha256": _sha256(price_bytes),
        "metadata_sha256": _sha256(metadata_bytes),
        "raw_price_bytes": len(price_bytes),
        "metadata_bytes": len(metadata_bytes),
        "audit": audit,
    }
    return TiingoSnapshot(raw, canonical, metadata, manifest, audit)


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def fetch_tiingo_spy_snapshot(
    output_dir: Path,
    *,
    token: str,
    symbol: str = DEFAULT_SYMBOL,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    calendar_name: str = DEFAULT_CALENDAR,
) -> TiingoSnapshot:
    """Explicitly acquire, validate and atomically publish an immutable licensed snapshot."""
    symbol = symbol.upper()
    query = urllib.parse.urlencode(
        {"startDate": start.isoformat(), "endDate": end.isoformat(), "format": "csv"}
    )
    price_url = f"{TIINGO_BASE_URL}/{symbol}/prices?{query}"
    metadata_url = f"{TIINGO_BASE_URL}/{symbol}"
    price_bytes = _request(price_url, token, "text/csv")
    metadata_bytes = _request(metadata_url, token, "application/json")
    snapshot = validate_tiingo_snapshot_bytes(
        price_bytes,
        metadata_bytes,
        symbol=symbol,
        start=start,
        end=end,
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
            "retrieved_at_utc": _utc_now(),
            "endpoint": price_url,
            "metadata_endpoint": metadata_url,
            "files": {
                "spy_daily.csv": _sha256(price_bytes),
                "spy_metadata.json": _sha256(metadata_bytes),
                "spy_daily_canonical.csv": _sha256(canonical_bytes),
            },
            "environment": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "exchange_calendars": snapshot.audit["calendar_library_version"],
            },
        }
        _write(temporary / "spy_daily.csv", price_bytes)
        _write(temporary / "spy_metadata.json", metadata_bytes)
        _write(temporary / "spy_daily_canonical.csv", canonical_bytes)
        _write(temporary / "manifest.json", _json_bytes(manifest))
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return TiingoSnapshot(
        snapshot.raw, snapshot.canonical, snapshot.metadata, manifest, snapshot.audit
    )


def load_tiingo_spy_snapshot(output_dir: Path) -> TiingoSnapshot:
    """Load and verify a previously acquired snapshot without network access."""
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Snapshot manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Snapshot manifest is not valid JSON") from exc
    price_bytes = (root / "spy_daily.csv").read_bytes()
    metadata_bytes = (root / "spy_metadata.json").read_bytes()
    canonical_bytes_saved = (root / "spy_daily_canonical.csv").read_bytes()
    for filename, payload in (
        ("spy_daily.csv", price_bytes),
        ("spy_metadata.json", metadata_bytes),
    ):
        if manifest.get("files", {}).get(filename) != _sha256(payload):
            raise ValueError(f"Snapshot hash mismatch: {filename}")
    snapshot = validate_tiingo_snapshot_bytes(
        price_bytes,
        metadata_bytes,
        symbol=manifest["symbol"],
        start=date.fromisoformat(manifest["requested_start"]),
        end=date.fromisoformat(manifest["requested_end"]),
        calendar_name=manifest["calendar"],
    )
    canonical_bytes = snapshot.canonical.to_csv(index=False).encode()
    if manifest["files"].get("spy_daily_canonical.csv") != _sha256(canonical_bytes_saved):
        raise ValueError("Snapshot hash mismatch: spy_daily_canonical.csv")
    if canonical_bytes_saved != canonical_bytes:
        raise ValueError("Snapshot canonical transformation hash mismatch")
    if snapshot.manifest["raw_price_sha256"] != manifest["raw_price_sha256"]:
        raise ValueError("Snapshot manifest raw hash mismatch")
    return TiingoSnapshot(
        snapshot.raw, snapshot.canonical, snapshot.metadata, manifest, snapshot.audit
    )

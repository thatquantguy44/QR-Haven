"""Offline Tiingo snapshot acquisition, calendar and integrity tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from qr_haven.data import tiingo

START = date(2024, 1, 2)
END = date(2024, 1, 10)


def metadata_bytes(start: str = "1993-01-29", end: str = "2026-10-05") -> bytes:
    return json.dumps(
        {
            "ticker": "SPY",
            "name": "SPDR S&P 500 ETF Trust",
            "exchangeCode": "ARCA",
            "startDate": start,
            "endDate": end,
        }
    ).encode()


def price_frame(start: date = START, end: date = END) -> pd.DataFrame:
    sessions = xcals.get_calendar("XNYS", start=start, end=end).sessions
    count = len(sessions)
    close = 470.0 + np.arange(count)
    return pd.DataFrame(
        {
            "date": [f"{value.date().isoformat()}T00:00:00.000Z" for value in sessions],
            "close": close,
            "high": close + 2,
            "low": close - 2,
            "open": close - 1,
            "volume": np.arange(count) + 1_000_000,
            "adjClose": close - 1.5,
            "adjHigh": close + 0.5,
            "adjLow": close - 3.5,
            "adjOpen": close - 2.5,
            "adjVolume": np.arange(count) + 1_010_000,
            "divCash": [0.0] * count,
            "splitFactor": [1.0] * count,
        }
    )


def price_bytes(frame: pd.DataFrame | None = None) -> bytes:
    return (frame if frame is not None else price_frame()).to_csv(index=False).encode()


def validate(frame: pd.DataFrame | None = None):
    return tiingo.validate_tiingo_snapshot_bytes(
        price_bytes(frame), metadata_bytes(), start=START, end=END
    )


def test_valid_snapshot_uses_adjusted_close_and_exact_xnys_sessions(tmp_path):
    snapshot = validate()
    assert snapshot.audit["rows"] == 7
    assert snapshot.audit["expected_sessions"] == 7
    assert snapshot.audit["missing_sessions"] == []
    assert snapshot.audit["calendar"] == "XNYS"
    assert snapshot.manifest["redistribution_permitted"] is False
    assert "split- and dividend-adjusted" in snapshot.manifest["adjustment"]
    assert snapshot.canonical["adjusted_close"].tolist() == price_frame()["adjClose"].tolist()
    assert snapshot.canonical["open"].tolist() == price_frame()["adjOpen"].tolist()
    assert snapshot.canonical["high"].tolist() == price_frame()["adjHigh"].tolist()
    assert snapshot.canonical["low"].tolist() == price_frame()["adjLow"].tolist()
    assert snapshot.canonical["close"].tolist() == price_frame()["adjClose"].tolist()
    assert snapshot.audit["canonical_ohlc_basis"] == "Tiingo adjusted OHLC"
    assert snapshot.canonical["symbol"].eq("SPY").all()
    assert snapshot.canonical["frequency"].eq("daily").all()

    canonical_path = tmp_path / "canonical.csv"
    snapshot.canonical.to_csv(canonical_path, index=False)
    prices = snapshot.price_portal(canonical_path).load_prices(
        ["SPY"],
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 31, tzinfo=UTC),
        "daily",
    )
    assert len(prices) == 7
    assert prices["adjusted_close"].iloc[0] == price_frame()["adjClose"].iloc[0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(index=2), "missing="),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate sessions"),
        (lambda frame: frame.iloc[::-1], "chronological"),
        (lambda frame: frame.assign(adjClose=np.nan), "missing or nonfinite"),
        (lambda frame: frame.assign(adjClose=0.0), "prices must be positive"),
        (lambda frame: frame.assign(volume=-1), "must be nonnegative"),
        (lambda frame: frame.assign(splitFactor=0.0), "split factors must be positive"),
        (lambda frame: frame.drop(columns="adjClose"), "missing columns"),
    ],
)
def test_invalid_price_snapshots_are_rejected(mutation, message):
    with pytest.raises(ValueError, match=message):
        validate(mutation(price_frame()))


def test_extra_non_session_is_distinguished_from_exchange_closure():
    frame = price_frame()
    weekend = frame.iloc[[0]].copy()
    weekend["date"] = "2024-01-06T00:00:00.000Z"
    frame = pd.concat([frame, weekend]).sort_values("date")
    with pytest.raises(ValueError, match="extra=.+2024-01-06"):
        validate(frame)


def test_metadata_ticker_and_coverage_are_enforced():
    wrong_ticker = metadata_bytes().replace(b'"SPY"', b'"QQQ"')
    with pytest.raises(ValueError, match="ticker mismatch"):
        tiingo.validate_tiingo_snapshot_bytes(price_bytes(), wrong_ticker, start=START, end=END)
    with pytest.raises(ValueError, match="does not cover"):
        tiingo.validate_tiingo_snapshot_bytes(
            price_bytes(), metadata_bytes(start="2024-01-03"), start=START, end=END
        )


def test_fetch_is_atomic_immutable_and_never_persists_token(tmp_path, monkeypatch):
    secret = "super-secret-fixture-token"

    def response(url: str, token: str, accept: str) -> bytes:
        assert token == secret
        assert secret not in url
        return price_bytes() if accept == "text/csv" else metadata_bytes()

    monkeypatch.setattr(tiingo, "_request", response)
    destination = tmp_path / "snapshot-v1"
    snapshot = tiingo.fetch_tiingo_spy_snapshot(destination, token=secret, start=START, end=END)
    assert snapshot.audit["rows"] == 7
    assert sorted(path.name for path in destination.iterdir()) == [
        "manifest.json",
        "spy_daily.csv",
        "spy_daily_canonical.csv",
        "spy_metadata.json",
    ]
    assert all(secret.encode() not in path.read_bytes() for path in destination.iterdir())
    loaded = tiingo.load_tiingo_spy_snapshot(destination)
    pd.testing.assert_frame_equal(snapshot.canonical, loaded.canonical)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        tiingo.fetch_tiingo_spy_snapshot(destination, token=secret, start=START, end=END)


def test_existing_snapshot_hash_and_canonical_transform_are_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tiingo,
        "_request",
        lambda url, token, accept: price_bytes() if accept == "text/csv" else metadata_bytes(),
    )
    destination = tmp_path / "snapshot-v1"
    tiingo.fetch_tiingo_spy_snapshot(destination, token="fixture", start=START, end=END)
    with (destination / "spy_daily_canonical.csv").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="canonical.csv"):
        tiingo.load_tiingo_spy_snapshot(destination)


def test_empty_token_fails_before_network(monkeypatch, tmp_path):
    def no_network(*args):
        pytest.fail("Empty token must fail before network access")

    monkeypatch.setattr(tiingo.urllib.request, "urlopen", no_network)
    with pytest.raises(ValueError, match="TIINGO_API_TOKEN is empty"):
        tiingo.fetch_tiingo_spy_snapshot(tmp_path / "snapshot", token="", start=START, end=END)


def test_calendar_is_constructed_for_early_requested_history():
    start = date(2005, 1, 1)
    end = date(2005, 1, 5)
    snapshot = tiingo.validate_tiingo_snapshot_bytes(
        price_bytes(price_frame(start, end)), metadata_bytes(), start=start, end=end
    )
    assert snapshot.audit["first_session"] == "2005-01-03"
    assert snapshot.audit["last_session"] == "2005-01-05"

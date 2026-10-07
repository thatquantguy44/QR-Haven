"""Tests for preparing the locally supplied SPX price-index data."""

from __future__ import annotations

from datetime import UTC, date, datetime

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from qr_haven.data import spx

START = date(2024, 1, 2)
END = date(2024, 1, 10)


def local_frame() -> pd.DataFrame:
    sessions = xcals.get_calendar("XNYS", start=START, end=END).sessions
    close = 4700.0 + np.arange(len(sessions))
    return pd.DataFrame(
        {
            "Date": sessions.strftime("%Y-%m-%d"),
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Adj Close": close,
            "Volume": np.arange(len(sessions)) + 1_000_000,
        }
    )


def local_bytes(frame: pd.DataFrame | None = None) -> bytes:
    return (frame if frame is not None else local_frame()).to_csv(index=False).encode()


def validate(frame: pd.DataFrame | None = None):
    return spx.validate_local_spx_bytes(local_bytes(frame), study_start=START, study_end=END)


def test_local_spx_is_validated_as_a_price_index(tmp_path):
    snapshot = validate()
    assert snapshot.audit["study_rows"] == 7
    assert snapshot.audit["expected_sessions"] == 7
    assert snapshot.audit["adjusted_close_differs_from_close"] == 0
    assert "does not represent" in snapshot.manifest["price_basis"]
    assert snapshot.manifest["review_status"].endswith("source_and_license_unverified")
    assert snapshot.canonical["symbol"].eq("SPX").all()

    path = tmp_path / "canonical.csv"
    snapshot.canonical.to_csv(path, index=False)
    prices = snapshot.price_portal(path).load_prices(
        ["SPX"],
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 31, tzinfo=UTC),
        "daily",
    )
    assert len(prices) == 7


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(index=2), "missing="),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate sessions"),
        (lambda frame: frame.iloc[::-1], "chronological"),
        (lambda frame: frame.assign(**{"Adj Close": np.nan}), "missing or nonfinite"),
        (lambda frame: frame.assign(Close=0.0), "prices must be positive"),
        (lambda frame: frame.assign(Volume=-1), "volume must be nonnegative"),
        (lambda frame: frame.assign(Volume=1.5), "whole numbers"),
        (lambda frame: frame.assign(High=1.0), "high is below"),
        (lambda frame: frame.drop(columns="Adj Close"), "missing columns"),
    ],
)
def test_invalid_local_spx_inputs_are_rejected(mutation, message):
    with pytest.raises(ValueError, match=message):
        validate(mutation(local_frame()))


def test_prepare_is_immutable_and_load_verifies_hash(tmp_path):
    input_path = tmp_path / "SPX.csv"
    input_path.write_bytes(local_bytes())
    destination = tmp_path / "prepared"
    snapshot = spx.prepare_local_spx_snapshot(
        input_path, destination, study_start=START, study_end=END
    )
    assert snapshot.audit["study_rows"] == 7
    assert sorted(path.name for path in destination.iterdir()) == [
        "manifest.json",
        "spx_daily_canonical.csv",
    ]
    loaded = spx.load_prepared_local_spx_snapshot(destination)
    assert loaded.manifest["source_sha256"] == snapshot.manifest["source_sha256"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        spx.prepare_local_spx_snapshot(
            input_path, destination, study_start=START, study_end=END
        )

    with (destination / "spx_daily_canonical.csv").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        spx.load_prepared_local_spx_snapshot(destination)

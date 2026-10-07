"""Temporal, operational, and reporting tests for the shadow deployment."""

from __future__ import annotations

import json
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from importlib.metadata import version

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from qr_haven.data import tiingo
from qr_haven.data.banknotes import json_bytes, sha256
from qr_haven.ml.volatility.challenger_data import CHALLENGER_FEATURES, extend_challenger_features
from qr_haven.ml.volatility.continuous_ytd import ALERT_THRESHOLD, MODEL_WEIGHT, PROTOCOL
from qr_haven.ml.volatility.contracts import VolatilityProfile
from qr_haven.ml.volatility.dataset import build_volatility_dataset
from qr_haven.ml.volatility.shadow import bind_ledger, mature_outcomes, open_ledger, run_shadow
from qr_haven.ml.volatility.shadow_features import (
    causal_features,
    forecast_timing,
    latest_completed_session,
    validate_shadow_prices,
)
from qr_haven.ml.volatility.shadow_reporting import ledger_frames


class FixtureEstimator:
    def predict(self, rows):
        return np.log(np.full(len(rows), 0.12**2))


@pytest.fixture
def candidate(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    bundle = {
        "protocol": PROTOCOL,
        "candidate_id": "hist_gradient_boosting_regression_w075",
        "feature_order": CHALLENGER_FEATURES,
        "model_weight": MODEL_WEIGHT,
        "alert_threshold": ALERT_THRESHOLD,
        "estimator": FixtureEstimator(),
        "training_ids": ("training-row",),
    }
    files = {
        "model.pkl": pickle.dumps(bundle),
        "training_membership.csv": b"sample_id\ntraining-row\n",
        "fit_audit.json": b"{}",
        "config.json": b"{}",
    }
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    (root / "manifest.json").write_bytes(
        json_bytes(
            {
                "protocol": PROTOCOL,
                "state": "complete",
                "selected_candidate": bundle["candidate_id"],
                "files": {name: sha256(payload) for name, payload in files.items()},
                "environment": {
                    p: version(p) for p in ("scikit-learn", "numpy", "scipy", "pandas")
                },
            }
        )
    )
    return root


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    sessions = xcals.get_calendar("XNYS", start="2026-01-02", end="2026-06-30").sessions
    rng = np.random.default_rng(512)
    close = 450 * np.exp(np.cumsum(rng.normal(0, 0.01, len(sessions))))
    raw = pd.DataFrame(
        {
            "date": sessions.strftime("%Y-%m-%dT00:00:00Z"),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adjOpen": close * 0.999,
            "adjHigh": close * 1.01,
            "adjLow": close * 0.99,
            "adjClose": close,
            "volume": 1_000_000,
            "adjVolume": 1_000_000,
            "divCash": 0,
            "splitFactor": 1,
        }
    )
    metadata = json_bytes({"ticker": "SPY", "startDate": "1993-01-29", "endDate": "2026-06-30"})
    monkeypatch.setattr(
        tiingo,
        "_request",
        lambda _u, _t, accept: (
            raw.to_csv(index=False).encode() if accept == "text/csv" else metadata
        ),
    )
    root = tmp_path / "source"
    tiingo.fetch_tiingo_spy_snapshot(
        root, token="fixture", start=date(2026, 1, 2), end=date(2026, 6, 30)
    )
    return root


def test_causal_features_match_research_and_ignore_future_prices(snapshot):
    source = tiingo.load_tiingo_spy_snapshot(snapshot)
    canonical = source.canonical
    prices = validate_shadow_prices(canonical)
    profile = VolatilityProfile(
        "test",
        "SPY",
        snapshot,
        "unused",
        date(2026, 1, 2),
        date(2026, 6, 30),
        (),
        date(2026, 1, 2),
        date(2026, 6, 30),
    )
    data = build_volatility_dataset(canonical, source.manifest, profile)
    research, _ = extend_challenger_features(data, canonical)
    for _, row in research.iterrows():
        actual = causal_features(prices, row.as_of)
        np.testing.assert_allclose(
            list(actual.values()), row[list(CHALLENGER_FEATURES)].to_numpy(float), rtol=1e-13
        )
    origin = prices.index[65]
    mutated = prices.copy()
    mutated.loc[mutated.index > origin, "adjusted_close"] *= 3
    assert causal_features(mutated, origin) == causal_features(prices, origin)
    assert causal_features(prices, prices.index[-1])  # Latest session has no forward label.


def test_calendar_handles_holiday_and_early_close():
    assert latest_completed_session(pd.Timestamp("2026-07-03T20:00Z")) == pd.Timestamp("2026-07-02")
    timing = forecast_timing(pd.Timestamp("2026-11-27"))
    assert pd.Timestamp(timing["available_at"]).hour == 18  # Black Friday early close, UTC.
    assert timing["label_start"] == "2026-11-30"


def test_replay_matures_only_after_five_sessions_and_is_idempotent(snapshot, candidate, tmp_path):
    prices = validate_shadow_prices(tiingo.load_tiingo_spy_snapshot(snapshot).canonical)
    origin, before, maturity = (prices.index[i].date().isoformat() for i in (60, 64, 65))
    root = tmp_path / "replay"
    first = run_shadow(
        root=root,
        snapshot_dir=snapshot,
        candidate_dir=candidate,
        mode="replay",
        start=origin,
        through=origin,
    )
    assert first["new_forecasts"] == 1 and first["new_outcomes"] == 0
    frames, _ = ledger_frames(root)
    assert frames["FactForecast"]["ActualVolatility"].isna().all()
    assert frames["FactForecast"]["QLIKE"].isna().all()
    run_shadow(
        root=root,
        snapshot_dir=snapshot,
        candidate_dir=candidate,
        mode="replay",
        start=origin,
        through=before,
    )
    frames, _ = ledger_frames(root)
    assert frames["FactForecast"]["Scored"].sum() == 0
    result = run_shadow(
        root=root,
        snapshot_dir=snapshot,
        candidate_dir=candidate,
        mode="replay",
        start=origin,
        through=maturity,
    )
    assert result["new_outcomes"] == 1
    frames, summary = ledger_frames(root)
    assert summary["origins"] == 6 and summary["pending_origins"] == 5
    scored = frames["FactForecast"].query("Scored == 1")
    closes = prices.iloc[60:66].adjusted_close.to_numpy()
    expected = np.sqrt(252 / 5 * np.sum(np.log(closes[1:] / closes[:-1]) ** 2))
    np.testing.assert_allclose(scored.ActualVolatility, expected)
    ratio = expected**2 / scored.ForecastVolatility.to_numpy() ** 2
    np.testing.assert_allclose(scored.QLIKE, ratio - np.log(ratio) - 1)
    repeated = run_shadow(
        root=root,
        snapshot_dir=snapshot,
        candidate_dir=candidate,
        mode="replay",
        start=origin,
        through=maturity,
    )
    assert repeated["new_forecasts"] == repeated["new_outcomes"] == 0
    newer, _ = ledger_frames(root)
    pd.testing.assert_frame_equal(frames["FactForecast"], newer["FactForecast"])
    current = json.loads((root / "reports/current.json").read_text())
    output = root / "reports/generations" / current["generation"]
    for name, digest in current["files"].items():
        assert sha256((output / name).read_bytes()) == digest
    assert frames["FactForecast"].ForecastKey.is_unique
    assert len(frames["FactForecast"]) == 2 * summary["origins"]


def test_live_rejects_old_snapshot_and_writes_failure_status(snapshot, candidate, tmp_path):
    root = tmp_path / "live"
    with pytest.raises(ValueError, match="Stale or unfinished"):
        run_shadow(root=root, snapshot_dir=snapshot, candidate_dir=candidate)
    frames, summary = ledger_frames(root)
    assert frames["FactForecast"].empty
    assert summary["latest_run_state"] == "failed"
    assert summary["failed_runs"] == 1
    assert (root / "reports/current/dashboard.html").exists()


def test_modes_are_bound_and_backdated_live_is_rejected(snapshot, candidate, tmp_path):
    root = tmp_path / "ledger"
    run_shadow(root=root, snapshot_dir=snapshot, candidate_dir=candidate, mode="replay")
    with pytest.raises(ValueError, match="Ledger mode"):
        run_shadow(root=root, snapshot_dir=snapshot, candidate_dir=candidate, mode="live")
    with pytest.raises(ValueError, match="only permitted in replay"):
        run_shadow(root=root, snapshot_dir=snapshot, candidate_dir=candidate, through="2026-06-30")


def test_ledger_rejects_mutation_and_different_model(tmp_path):
    db = open_ledger(tmp_path)
    bind_ledger(db, {"mode": "replay", "model": "first"})
    with pytest.raises(ValueError, match="Ledger mode"):
        bind_ledger(db, {"mode": "replay", "model": "changed"})
    with pytest.raises(sqlite3.IntegrityError, match="Append-only"):
        db.execute("DELETE FROM settings")
    db.close()


def test_invalid_market_data_is_rejected(snapshot):
    raw = tiingo.load_tiingo_spy_snapshot(snapshot).canonical
    with pytest.raises(ValueError, match="gaps"):
        validate_shadow_prices(raw.drop(index=70))
    with pytest.raises(ValueError, match="SPY only"):
        validate_shadow_prices(raw.assign(symbol="SPX"))
    with pytest.raises(ValueError, match="consistent adjusted"):
        validate_shadow_prices(raw.assign(close=raw.close * 2))


def test_api_predict_validates_inputs_and_report_routes(candidate, snapshot, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from qr_haven.ml.volatility.deployment_api import create_app

    root = tmp_path / "report"
    run_shadow(root=root, snapshot_dir=snapshot, candidate_dir=candidate, mode="replay")
    prices = validate_shadow_prices(tiingo.load_tiingo_spy_snapshot(snapshot).canonical)
    features = causal_features(prices, prices.index[-1])
    with TestClient(create_app(candidate, root)) as client:
        assert client.get("/health").json()["research_status"] == "target_not_met"
        assert len(client.get("/model").json()["feature_order"]) == 20
        response = client.post("/predict", json={"records": [features]})
        assert response.status_code == 200
        assert response.json()["predictions"][0]["candidate_forecast"] > 0
        assert client.post("/predict", json={"records": [{}]}).status_code == 422
        assert client.post("/predict", json={"records": []}).status_code == 422
        bad = {**features, "trailing_vol_5": -1}
        assert client.post("/predict", json={"records": [bad]}).status_code == 422
        assert client.get("/dashboard").status_code == 200
        assert client.get("/exports/FactForecast.csv").status_code == 200
        assert client.get("/exports/ledger.sqlite").status_code == 404
        assert client.get("/status").json()["mode"] == "replay"


def test_runtime_mismatch_fails_before_model_load(candidate, monkeypatch):
    import qr_haven.ml.volatility.deployment as module

    monkeypatch.setattr(module, "version", lambda _package: "wrong-version")
    with pytest.raises(ValueError, match="runtime mismatch"):
        module.load_shadow_model(candidate)


def test_concurrent_replay_calls_do_not_duplicate(snapshot, candidate, tmp_path):
    root = tmp_path / "concurrent"
    prices = validate_shadow_prices(tiingo.load_tiingo_spy_snapshot(snapshot).canonical)
    through = prices.index[65].date().isoformat()

    def run():
        return run_shadow(
            root=root,
            snapshot_dir=snapshot,
            candidate_dir=candidate,
            mode="replay",
            through=through,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert sorted(r["new_forecasts"] for r in results) == [0, 6]
    _, summary = ledger_frames(root)
    assert summary["origins"] == 6 and summary["scored_origins"] == 1


def test_zero_outcomes_are_explicitly_unscorable(snapshot, candidate, tmp_path):
    prices = validate_shadow_prices(tiingo.load_tiingo_spy_snapshot(snapshot).canonical)
    origin = prices.index[60]
    root = tmp_path / "zeros"
    run_shadow(
        root=root,
        snapshot_dir=snapshot,
        candidate_dir=candidate,
        mode="replay",
        through=origin.date().isoformat(),
    )
    modified = prices.copy()
    modified.loc[modified.index[60:66], "adjusted_close"] = prices.iloc[60].adjusted_close
    with open_ledger(root) as db:
        mature_outcomes(
            db,
            modified,
            pd.Timestamp(forecast_timing(origin)["outcome_available_at"]),
            "fixture-revision",
        )
    frames, summary = ledger_frames(root)
    assert summary["unscorable_origins"] == 1
    assert frames["FactForecast"]["QLIKE"].isna().all()
    assert frames["FactForecast"]["Scored"].sum() == 0


def test_live_forecast_window_and_durable_failure(snapshot, candidate, tmp_path, monkeypatch):
    import qr_haven.ml.volatility.shadow as module

    root = tmp_path / "live-window"
    # Tests control the execution clock; the public live CLI exposes no clock override.
    monkeypatch.setattr(module, "utc_now", lambda: "2026-06-30T20:30:00+00:00")
    result = run_shadow(root=root, snapshot_dir=snapshot, candidate_dir=candidate)
    assert result["new_forecasts"] == 1 and result["new_outcomes"] == 0
    frames, _ = ledger_frames(root)
    assert frames["FactForecast"].ObservedAtUtc.eq("2026-06-30T20:30:00+00:00").all()
    with pytest.raises(ValueError, match="before the next open"):
        monkeypatch.setattr(module, "utc_now", lambda: "2026-07-01T14:00:00+00:00")
        run_shadow(root=tmp_path / "too-late", snapshot_dir=snapshot, candidate_dir=candidate)


def test_source_failure_preserves_committed_forecasts(snapshot, candidate, tmp_path, monkeypatch):
    import qr_haven.ml.volatility.shadow as module

    root = tmp_path / "interrupted"
    original = module.mature_outcomes

    def fail(*args, **kwargs):
        raise ValueError("Simulated outcome failure")

    monkeypatch.setattr(module, "mature_outcomes", fail)
    with pytest.raises(ValueError, match="Simulated"):
        run_shadow(root=root, snapshot_dir=snapshot, candidate_dir=candidate, mode="replay")
    before, summary = ledger_frames(root)
    assert summary["origins"] > 0 and summary["scored_origins"] == 0
    monkeypatch.setattr(module, "mature_outcomes", original)
    run_shadow(root=root, snapshot_dir=snapshot, candidate_dir=candidate, mode="replay")
    after, summary = ledger_frames(root)
    assert summary["scored_origins"] > 0
    assert before["FactForecast"].ForecastVolatility.equals(
        after["FactForecast"].ForecastVolatility
    )

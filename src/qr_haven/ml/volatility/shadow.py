"""Daily causal shadow forecasts, immutable SQLite evidence, and replay tooling."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, sha256, utc_now
from qr_haven.data.tiingo import fetch_tiingo_spy_snapshot, load_tiingo_spy_snapshot
from qr_haven.ml.classification.artifacts import exclusive_lock
from qr_haven.ml.volatility.continuous_ytd import verify_ytd_candidate
from qr_haven.ml.volatility.deployment import (
    DEFAULT_CANDIDATE_DIR,
    load_shadow_model,
    score_feature_rows,
)
from qr_haven.ml.volatility.shadow_features import (
    causal_features,
    forecast_timing,
    latest_completed_session,
    validate_shadow_prices,
)

PROTOCOL = "volatility-shadow-v1"
DEFAULT_ROOT = Path("artifacts/classification/volatility/tiingo-spy-v1/shadow/live-v1")


def open_ledger(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "ledger.sqlite", timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT);
        CREATE TABLE IF NOT EXISTS forecasts (
            sample_id TEXT PRIMARY KEY, as_of TEXT NOT NULL UNIQUE, payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS outcomes (
            sample_id TEXT PRIMARY KEY REFERENCES forecasts(sample_id), payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL
        );
    """)
    for table in ("settings", "forecasts", "outcomes", "runs"):
        for action in ("UPDATE", "DELETE"):
            connection.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()} BEFORE {action} "
                f"ON {table} BEGIN SELECT RAISE(ABORT, 'Append-only ledger'); END"
            )
    connection.commit()
    return connection


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def bind_ledger(db: sqlite3.Connection, config: dict[str, Any]) -> None:
    row = db.execute("SELECT payload FROM settings WHERE id=1").fetchone()
    if row is not None and json.loads(row[0]) != config:
        raise ValueError("Ledger mode, model, or scoring code differs; use a new shadow root")
    if row is None:
        db.execute("INSERT INTO settings VALUES (1, ?)", (_json(config),))
        db.commit()


def record_forecast(
    db: sqlite3.Connection,
    prices: pd.DataFrame,
    origin: pd.Timestamp,
    bundle: dict[str, Any],
    config: dict[str, Any],
    source_hash: str,
    observed_at: pd.Timestamp,
    source_received_at: str,
) -> bool:
    """Commit a forecast first. Duplicate origins preserve their original bytes."""
    day = origin.date().isoformat()
    if db.execute("SELECT 1 FROM forecasts WHERE as_of=?", (day,)).fetchone():
        return False
    timing = forecast_timing(origin)
    if not pd.Timestamp(timing["available_at"]) <= observed_at < pd.Timestamp(timing["next_open"]):
        raise ValueError("Forecast must be created after the close and before the next open")
    features = causal_features(prices, origin)
    prediction = score_feature_rows(bundle, pd.DataFrame([features])).iloc[0]
    created = utc_now()
    if config["mode"] == "live":
        observed_at = pd.Timestamp(created)
        if observed_at >= pd.Timestamp(timing["next_open"]):
            raise ValueError("Scoring completed after the next session opened")
    sample_id = f"{PROTOCOL}/SPY/{day}"
    record = {
        "sample_id": sample_id,
        "as_of": day,
        **timing,
        "created_at_utc": created,
        "observed_at_utc": observed_at.isoformat(),
        "source_received_at_utc": source_received_at,
        "mode": config["mode"],
        "model_sha256": config["model_sha256"],
        "source_sha256": source_hash,
        "features": features,
        "feature_sha256": sha256(json_bytes(features)),
        "candidate_forecast": float(prediction["candidate_forecast"]),
        "persistence_forecast": float(prediction["persistence_forecast"]),
        "alert_threshold": float(prediction["alert_threshold"]),
        "candidate_high": int(prediction["candidate_high"]),
        "persistence_high": int(prediction["persistence_forecast"] > prediction["alert_threshold"]),
    }
    with db:
        db.execute("INSERT INTO forecasts VALUES (?, ?, ?)", (sample_id, day, _json(record)))
    return True


def mature_outcomes(
    db: sqlite3.Connection, prices: pd.DataFrame, observed_at: pd.Timestamp, source_hash: str
) -> int:
    rows = db.execute(
        "SELECT f.payload FROM forecasts f LEFT JOIN outcomes o USING(sample_id) "
        "WHERE o.sample_id IS NULL ORDER BY f.as_of"
    ).fetchall()
    added = 0
    for row in rows:
        forecast = json.loads(row[0])
        if pd.Timestamp(forecast["outcome_available_at"]) > observed_at:
            continue
        window = prices.loc[forecast["as_of"] : forecast["label_end"], "adjusted_close"]
        if (
            len(window) != 6
            or window.index[0].date().isoformat() != forecast["as_of"]
            or window.index[-1].date().isoformat() != forecast["label_end"]
        ):
            raise ValueError(
                "Snapshot does not cover a pending outcome's complete six-price window"
            )
        values = window.to_numpy(float)
        actual = math.sqrt(252 / 5 * float(np.square(np.log(values[1:] / values[:-1])).sum()))
        result: dict[str, Any] = {
            "sample_id": forecast["sample_id"],
            "actual_volatility": actual,
            "actual_high": int(actual > forecast["alert_threshold"]),
            "scored_at_utc": utc_now(),
            "outcome_observed_at_utc": observed_at.isoformat(),
            "outcome_source_sha256": source_hash,
            "outcome_status": "scored" if actual > 0 else "unscorable",
            "reason": None if actual > 0 else "Zero realized volatility; log losses undefined",
        }
        for model in ("candidate", "persistence"):
            predicted = forecast[f"{model}_forecast"]
            ratio = actual**2 / max(predicted**2, 1e-12)
            result[f"{model}_qlike"] = max(0, ratio - math.log(ratio) - 1) if actual > 0 else None
            result[f"{model}_log_error"] = abs(math.log(predicted / actual)) if actual > 0 else None
            result[f"{model}_absolute_error"] = abs(predicted - actual)
        with db:
            db.execute("INSERT INTO outcomes VALUES (?, ?)", (forecast["sample_id"], _json(result)))
        added += 1
    return added


def _archive_source(root: Path, canonical: pd.DataFrame, manifest: dict[str, Any]) -> str:
    payload = canonical.to_csv(index=False).encode()
    digest = sha256(payload)
    source_dir = root / "sources"
    source_dir.mkdir(exist_ok=True)
    path = source_dir / f"{digest}.csv"
    if path.exists():
        if sha256(path.read_bytes()) != digest:
            raise ValueError("Archived source has been modified")
    else:
        atomic_write(path, payload)
        # Keep provenance free of endpoint query strings or credentials.
        metadata = {
            k: manifest.get(k)
            for k in (
                "provider",
                "symbol",
                "raw_price_sha256",
                "retrieved_at_utc",
                "revision_policy",
            )
        }
        atomic_write(source_dir / f"{digest}.json", json_bytes(metadata))
    return digest


def run_shadow(
    *,
    root: Path,
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR,
    snapshot_dir: Path | None = None,
    fetch: bool = False,
    mode: str = "live",
    start: str | None = None,
    through: str | None = None,
) -> dict[str, Any]:
    """One operational cycle. Only replay accepts backdated clock boundaries."""
    if mode not in {"live", "replay"} or (mode == "live" and (start or through)):
        raise ValueError("Historical start/through dates are only permitted in replay mode")
    if (snapshot_dir is None) == (not fetch) or (fetch and mode != "live"):
        raise ValueError("Choose a saved snapshot or a live Tiingo fetch")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".run.lock"):
        db = open_ledger(root)
        try:
            now = pd.Timestamp(utc_now())
            manifest = verify_ytd_candidate(candidate_dir)
            bundle = load_shadow_model(candidate_dir)
            config = {
                "protocol": PROTOCOL,
                "mode": mode,
                "model_sha256": manifest["files"]["model.pkl"],
                "candidate_id": bundle["candidate_id"],
                "feature_code_sha256": sha256(
                    Path(__file__).with_name("shadow_features.py").read_bytes()
                ),
                "scoring_code_sha256": sha256(
                    Path(__file__).with_name("deployment.py").read_bytes()
                ),
            }
            bind_ledger(db, config)
            expected = latest_completed_session(now)
            if fetch:
                token = os.environ.get("TIINGO_API_TOKEN", "")
                if not token:
                    raise ValueError("Set TIINGO_API_TOKEN in the job environment for a live fetch")
                snapshot_dir = root / "downloads" / expected.date().isoformat()
                if not snapshot_dir.exists():
                    first = expected - pd.Timedelta(days=400)
                    pending = db.execute(
                        "SELECT MIN(as_of) FROM forecasts LEFT JOIN outcomes USING(sample_id) "
                        "WHERE outcomes.sample_id IS NULL"
                    ).fetchone()[0]
                    if pending:
                        first = min(first, pd.Timestamp(pending) - pd.Timedelta(days=100))
                    fetch_tiingo_spy_snapshot(
                        snapshot_dir, token=token, start=first.date(), end=expected.date()
                    )
            assert snapshot_dir is not None
            snapshot = load_tiingo_spy_snapshot(snapshot_dir)
            prices = validate_shadow_prices(snapshot.canonical)
            source_hash = _archive_source(root, snapshot.canonical, snapshot.manifest)
            received = utc_now()
            # Re-read wall clock after acquisition: a slow fetch must not backdate forecasts.
            now = pd.Timestamp(utc_now())
            expected = latest_completed_session(now)
            cutoff = pd.Timestamp(through) if through else prices.index[-1]
            if cutoff not in prices.index:
                raise ValueError("Replay cutoff must be a session present in the snapshot")
            if mode == "live" and prices.index[-1] != expected:
                raise ValueError(
                    f"Stale or unfinished source; expected completed session {expected.date()}"
                )
            if pd.Timestamp(forecast_timing(cutoff)["available_at"]) > now:
                raise ValueError("Cannot process an unfinished session, including in replay")
            previous = db.execute("SELECT MAX(as_of) FROM forecasts").fetchone()[0]
            if previous and cutoff < pd.Timestamp(previous):
                raise ValueError("A shadow ledger cannot move its observation clock backwards")
            visible = prices.loc[:cutoff]
            origins = prices.index[60:][prices.index[60:] <= cutoff]
            if mode == "live":
                origins = pd.DatetimeIndex([cutoff])
            elif start:
                origins = origins[origins >= pd.Timestamp(start)]
            if origins.empty:
                raise ValueError("No eligible origins after the 60-return warmup")
            added = 0
            for origin in origins:
                observed = (
                    now
                    if mode == "live"
                    else pd.Timestamp(forecast_timing(origin)["available_at"])
                    + pd.Timedelta(minutes=30)
                )
                added += record_forecast(
                    db, visible, origin, bundle, config, source_hash, observed, received
                )
            observed = (
                now
                if mode == "live"
                else pd.Timestamp(forecast_timing(cutoff)["available_at"])
                + pd.Timedelta(minutes=30)
            )
            matured = mature_outcomes(db, visible, observed, source_hash)
            result = {
                "state": "complete",
                "mode": mode,
                "deployment_status": "shadow_only",
                "source_last_session": cutoff.date().isoformat(),
                "expected_latest_session": expected.date().isoformat(),
                "data_fresh": cutoff == expected,
                "new_forecasts": added,
                "new_outcomes": matured,
                "source_sha256": source_hash,
            }
            with db:
                db.execute(
                    "INSERT INTO runs(created_at,payload) VALUES (?,?)", (utc_now(), _json(result))
                )
        except Exception as exc:
            with db:
                db.execute(
                    "INSERT INTO runs(created_at,payload) VALUES (?,?)",
                    (utc_now(), _json({"state": "failed", "error": str(exc), "mode": mode})),
                )
            # Failure status is visible even when an earlier report generation already exists.
            from qr_haven.ml.volatility.shadow_reporting import export_shadow

            if db.execute("SELECT 1 FROM settings").fetchone():
                try:
                    export_shadow(root)
                except (OSError, ValueError, sqlite3.Error):
                    pass  # Preserve the original actionable data/model failure.
            raise
        finally:
            db.close()
        from qr_haven.ml.volatility.shadow_reporting import export_shadow

        report = export_shadow(root)
        return {**result, **report}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--snapshot-dir", type=Path)
    source.add_argument("--fetch-tiingo", action="store_true")
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    parser.add_argument("--start")
    parser.add_argument("--through")
    parser.add_argument(
        "--watch", action="store_true", help="Run live after close; retry every 15 minutes"
    )
    args = parser.parse_args(argv)
    if args.watch and (args.mode != "live" or not args.fetch_tiingo or args.command != "run"):
        parser.error("--watch requires run --mode live --fetch-tiingo")
    last_success: str | None = None
    while True:
        try:
            if args.command == "report":
                from qr_haven.ml.volatility.shadow_reporting import export_shadow

                result = export_shadow(args.root)
            else:
                if args.watch:
                    now = pd.Timestamp.now(tz="UTC")
                    expected = latest_completed_session(now)
                    timing = forecast_timing(expected)
                    due = pd.Timestamp(timing["available_at"]) + pd.Timedelta(minutes=30)
                    if (
                        expected.isoformat() == last_success
                        or now < due
                        or now >= pd.Timestamp(timing["next_open"])
                    ):
                        time.sleep(30)
                        continue
                result = run_shadow(
                    root=args.root,
                    candidate_dir=args.candidate_dir,
                    snapshot_dir=args.snapshot_dir,
                    fetch=args.fetch_tiingo,
                    mode=args.mode,
                    start=args.start,
                    through=args.through,
                )
                if args.watch:
                    last_success = expected.isoformat()
            print(_json(result), flush=True)
            if not args.watch:
                return 0
        except (ValueError, OSError, sqlite3.Error) as exc:
            print(_json({"state": "failed", "error": str(exc)}), file=sys.stderr, flush=True)
            if not args.watch:
                return 1
            for _ in range(15):
                time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())

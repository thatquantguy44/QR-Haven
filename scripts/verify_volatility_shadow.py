"""Verify causal features and HTTP predictions against saved, exposed V8A evidence."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, utc_now
from qr_haven.data.tiingo import load_tiingo_spy_snapshot
from qr_haven.ml.volatility.challenger_data import CHALLENGER_FEATURES
from qr_haven.ml.volatility.continuous_ytd import (
    EXTENSION_SNAPSHOT_DIR,
    verify_ytd_dataset,
    verify_ytd_evaluation,
)
from qr_haven.ml.volatility.deployment import DEFAULT_EVALUATION_DIR
from qr_haven.ml.volatility.shadow_features import causal_features, validate_shadow_prices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Exercise the ASGI application without starting a server",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("artifacts/classification/volatility/tiingo-spy-v1/shadow/replay-v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset_dir = DEFAULT_EVALUATION_DIR.parents[2] / "dataset-v1"
    verify_ytd_dataset(dataset_dir, include_sealed_outcomes=False)
    verify_ytd_evaluation(DEFAULT_EVALUATION_DIR)
    features = pd.read_csv(dataset_dir / "evaluation_features.csv", float_precision="round_trip")
    saved = pd.read_csv(
        DEFAULT_EVALUATION_DIR / "evaluation_predictions.csv", float_precision="round_trip"
    )
    assert features.sample_id.equals(saved.sample_id)
    prices = validate_shadow_prices(load_tiingo_spy_snapshot(EXTENSION_SNAPSHOT_DIR).canonical)
    rebuilt = pd.DataFrame([causal_features(prices, pd.Timestamp(day)) for day in features.as_of])
    np.testing.assert_allclose(
        rebuilt[list(CHALLENGER_FEATURES)].to_numpy(float),
        features[list(CHALLENGER_FEATURES)].to_numpy(float),
        rtol=1e-12,
    )

    client = None
    if args.in_process:
        from fastapi.testclient import TestClient

        from qr_haven.ml.volatility.deployment_api import create_app

        client = TestClient(create_app(report_root=args.report_root))

    def request(path: str, payload=None):
        if client is not None:
            response = client.get(path) if payload is None else client.post(path, json=payload)
            if response.status_code >= 400:
                raise urllib.error.HTTPError(path, response.status_code, response.text, None, None)
            return response.status_code, response.content
        data = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        req = urllib.request.Request(
            args.api_url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read()

    status, health_bytes = request("/health")
    assert status == 200
    health = json.loads(health_bytes)
    assert health["research_status"] == "target_not_met"
    assert health["deployment_status"] == "shadow_only"
    _, body = request("/predict", {"records": rebuilt.to_dict(orient="records")})
    predictions = pd.DataFrame(json.loads(body)["predictions"])
    for name in ("candidate_forecast", "persistence_forecast", "base_forecast"):
        np.testing.assert_allclose(predictions[name], saved[name], rtol=1e-12, atol=1e-14)
    assert predictions.candidate_high.equals(saved.candidate_high)
    for payload in ({"records": []}, {"records": [{}]}):
        try:
            request("/predict", payload)
        except urllib.error.HTTPError as exc:
            assert exc.code == 422
        else:
            raise AssertionError("Malformed prediction request accepted")
    assert request("/dashboard")[0] == 200
    assert request("/exports/FactForecast.csv")[0] == 200
    _, operational = request("/status")
    status_payload = json.loads(operational)
    result = {
        "checked_at_utc": utc_now(),
        "api_url": args.api_url if client is None else None,
        "execution": "in_process_asgi" if client is not None else "http_server",
        "rows_matched": len(saved),
        "model_sha256": health["model_sha256"],
        "feature_parity": "passed",
        "prediction_parity": "passed",
        "invalid_inputs": "rejected",
        "report_routes": "passed",
        "report_summary": status_payload["summary"],
        "browser_rendered_review": "not_performed_by_this_script",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, json_bytes(result), replace=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

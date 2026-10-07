"""FastAPI application for shadow testing of the V8A volatility candidate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qr_haven.ml.volatility.continuous_ytd import _load_ytd_model
from qr_haven.ml.volatility.deployment import DEFAULT_CANDIDATE_DIR, score_feature_rows


class PredictionRequest(BaseModel):
    """Batch of point-in-time feature records."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)


def create_app(candidate_dir: Path | None = None) -> FastAPI:
    root = candidate_dir or Path(
        os.environ.get("QR_HAVEN_VOLATILITY_CANDIDATE_DIR", str(DEFAULT_CANDIDATE_DIR))
    )
    bundle = _load_ytd_model(root)
    app = FastAPI(title="QR Haven V8A shadow volatility API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "candidate_id": bundle["candidate_id"],
            "protocol": bundle["protocol"],
            "deployment_status": "shadow_only",
            "research_status": "target_not_met",
        }

    @app.get("/model")
    def model() -> dict[str, Any]:
        return {
            "candidate_id": bundle["candidate_id"],
            "feature_order": list(bundle["feature_order"]),
            "model_weight": bundle["model_weight"],
            "alert_threshold": bundle["alert_threshold"],
            "training_rows": len(bundle["training_ids"]),
            "deployment_status": "shadow_only",
        }

    @app.post("/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        try:
            rows = pd.DataFrame.from_records(request.records)
            forecasts = score_feature_rows(bundle, rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "candidate_id": bundle["candidate_id"],
            "deployment_status": "shadow_only",
            "predictions": forecasts.to_dict(orient="records"),
        }

    return app


app = create_app()

"""FastAPI application for shadow testing of the V8A volatility candidate."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from qr_haven.ml.volatility.continuous_ytd import verify_ytd_candidate
from qr_haven.ml.volatility.deployment import (
    DEFAULT_CANDIDATE_DIR,
    load_shadow_model,
    score_feature_rows,
)


class PredictionRequest(BaseModel):
    """Batch of point-in-time feature records."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)


def create_app(candidate_dir: Path | None = None, report_root: Path | None = None) -> FastAPI:
    root = candidate_dir or Path(
        os.environ.get("QR_HAVEN_VOLATILITY_CANDIDATE_DIR", str(DEFAULT_CANDIDATE_DIR))
    )
    bundle = load_shadow_model(root)
    model_hash = verify_ytd_candidate(root)["files"]["model.pkl"]
    configured = os.environ.get("QR_HAVEN_SHADOW_ROOT")
    reports = report_root or (Path(configured) if configured else None)
    app = FastAPI(title="QR Haven V8A shadow volatility API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "candidate_id": bundle["candidate_id"],
            "protocol": bundle["protocol"],
            "deployment_status": "shadow_only",
            "research_status": "target_not_met",
            "model_sha256": model_hash,
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
            "model_sha256": model_hash,
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
            "model_sha256": model_hash,
        }

    def current_report() -> tuple[Path, dict[str, Any]]:
        if reports is None or not (reports / "reports" / "current.json").is_file():
            raise HTTPException(
                status_code=503, detail="Run the daily job to create reporting outputs"
            )
        manifest = json.loads((reports / "reports" / "current.json").read_text())
        generation = str(manifest["generation"])
        if Path(generation).name != generation:
            raise HTTPException(status_code=503, detail="Invalid report generation")
        if manifest["model_sha256"] != model_hash:
            raise HTTPException(
                status_code=503, detail="Report model does not match the serving model"
            )
        return reports / "reports" / "generations" / generation, manifest

    @app.get("/status")
    def status() -> dict[str, Any]:
        directory, manifest = current_report()
        return {**manifest, "summary": json.loads((directory / "summary.json").read_text())}

    @app.get("/dashboard", response_class=FileResponse)
    def dashboard() -> FileResponse:
        directory, _ = current_report()
        return FileResponse(
            directory / "dashboard.html",
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/exports/{filename}", response_class=FileResponse)
    def export(filename: str) -> FileResponse:
        directory, manifest = current_report()
        name = f"power_bi/{filename}"
        if Path(filename).name != filename or name not in manifest["files"]:
            raise HTTPException(status_code=404, detail="Unknown reporting table")
        return FileResponse(
            directory / name, filename=filename, headers={"Cache-Control": "no-store"}
        )

    return app

"""Immutable point-in-time inputs for the frozen Spec002 volatility challenger."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, sha256, utc_now
from qr_haven.data.spx import load_prepared_local_spx_snapshot
from qr_haven.ml.volatility.contracts import (
    FEATURES,
    SPX_LOCAL_PROFILE,
    TIINGO_SPY_PROFILE,
    VolatilityDataset,
    VolatilityProfile,
)
from qr_haven.ml.volatility.dataset import build_volatility_dataset
from qr_haven.ml.volatility.persistence import environment_versions, source_identity

PROTOCOL = "volatility-challenger-v1"
TIINGO_PROTOCOL = "volatility-challenger-tiingo-v1"
NEW_FEATURES = (
    "downside_vol_5",
    "downside_vol_20",
    "downside_vol_60",
    "upside_vol_5",
    "upside_vol_20",
    "max_abs_return_5",
    "max_abs_return_20",
    "negative_return_share_20",
    "vol_of_vol_20",
    "parkinson_vol_5",
    "parkinson_vol_20",
)
CHALLENGER_FEATURES = (*FEATURES, *NEW_FEATURES)


@dataclass(frozen=True)
class ChallengerDesign:
    protocol: str
    profile: VolatilityProfile
    outer_years: tuple[int, ...]
    final_calibration_years: tuple[int, ...]
    evaluation_years: tuple[int, ...]
    evaluation_id: str
    bootstrap_seed: int


SPX_DESIGN = ChallengerDesign(
    PROTOCOL,
    SPX_LOCAL_PROFILE,
    tuple(range(2013, 2018)),
    (2017, 2018, 2019),
    (2020,),
    "2020-v1",
    5410,
)
TIINGO_DESIGN = ChallengerDesign(
    TIINGO_PROTOCOL,
    TIINGO_SPY_PROFILE,
    tuple(range(2013, 2024)),
    (2021, 2022, 2023),
    (2024, 2025),
    "2024-2025-v1",
    5411,
)


def get_challenger_design(profile_id: str) -> ChallengerDesign:
    for design in (SPX_DESIGN, TIINGO_DESIGN):
        if design.profile.profile_id == profile_id:
            return design
    raise ValueError(f"No challenger design for profile: {profile_id}")


def challenger_design_from_manifest(manifest: dict[str, Any]) -> ChallengerDesign:
    profile_id = manifest.get("profile_id")
    if isinstance(profile_id, str):
        return get_challenger_design(profile_id)
    protocol = manifest.get("protocol")
    for design in (SPX_DESIGN, TIINGO_DESIGN):
        if design.protocol == protocol:
            return design
    raise ValueError("Manifest does not identify a frozen challenger design")


def _window_json(sessions: pd.DatetimeIndex) -> str:
    return json.dumps(sessions.strftime("%Y-%m-%d").tolist(), separators=(",", ":"))


def extend_challenger_features(
    dataset: VolatilityDataset, canonical: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the frozen point-in-time feature set to an already validated base dataset."""
    sessions = pd.DatetimeIndex(
        pd.to_datetime(canonical["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    )
    session_positions = {value: index for index, value in enumerate(sessions)}
    prices = canonical["adjusted_close"].to_numpy(dtype=float)
    highs = canonical["high"].to_numpy(dtype=float)
    lows = canonical["low"].to_numpy(dtype=float)
    if (
        not np.isfinite(highs).all()
        or not np.isfinite(lows).all()
        or (highs <= 0).any()
        or (lows <= 0).any()
        or (highs < lows).any()
    ):
        raise ValueError("Challenger range features require valid positive high/low prices")
    returns: npt.NDArray[np.float64] = np.full(len(prices), np.nan, dtype=np.float64)
    returns[1:] = np.log(prices[1:] / prices[:-1])
    log_ranges = np.log(highs / lows)
    observations = dataset.observations.copy()
    audit = dataset.interval_audit.copy()
    rows: list[dict[str, float]] = []
    audit_values: list[dict[str, str]] = []
    for as_of in pd.to_datetime(observations["as_of"]):
        origin = session_positions[pd.Timestamp(as_of)]
        features: dict[str, float] = {}
        windows: dict[str, str] = {}
        for window in (5, 20, 60):
            values = returns[origin - window + 1 : origin + 1]
            downside = np.minimum(values, 0.0)
            features[f"downside_vol_{window}"] = float(
                math.sqrt(252 / window * float(np.square(downside).sum()))
            )
            windows[f"downside_vol_{window}_price_sessions"] = _window_json(
                sessions[origin - window : origin + 1]
            )
            if window in (5, 20):
                upside = np.maximum(values, 0.0)
                features[f"upside_vol_{window}"] = float(
                    math.sqrt(252 / window * float(np.square(upside).sum()))
                )
                features[f"max_abs_return_{window}"] = float(np.abs(values).max())
                features[f"parkinson_vol_{window}"] = float(
                    math.sqrt(
                        252
                        * float(np.square(log_ranges[origin - window + 1 : origin + 1]).mean())
                        / (4 * math.log(2))
                    )
                )
                for name in ("upside_vol", "max_abs_return", "parkinson_vol"):
                    windows[f"{name}_{window}_price_sessions"] = _window_json(
                        sessions[origin - window : origin + 1]
                    )
        recent = returns[origin - 19 : origin + 1]
        features["negative_return_share_20"] = float((recent < 0).mean())
        features["vol_of_vol_20"] = float(math.sqrt(252) * np.abs(recent).std(ddof=1))
        shared_window = _window_json(sessions[origin - 20 : origin + 1])
        windows["negative_return_share_20_price_sessions"] = shared_window
        windows["vol_of_vol_20_price_sessions"] = shared_window
        rows.append(features)
        audit_values.append(windows)
    extended = pd.DataFrame(rows, index=observations.index)
    extended_audit = pd.DataFrame(audit_values, index=observations.index)
    observations = pd.concat([observations, extended], axis=1)
    audit = pd.concat([audit, extended_audit], axis=1)
    numeric = observations.loc[:, [*CHALLENGER_FEATURES, "forward_vol_5"]].to_numpy(float)
    if observations.empty or not np.isfinite(numeric).all():
        raise ValueError("Challenger observations must be nonempty and finite")
    return observations, audit


def build_challenger_observations(
    profile_id: str = SPX_LOCAL_PROFILE.profile_id,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build extended features without summarizing the profile's evaluation outcomes."""
    design = get_challenger_design(profile_id)
    if profile_id == SPX_LOCAL_PROFILE.profile_id:
        spx_snapshot = load_prepared_local_spx_snapshot(design.profile.snapshot_dir)
        canonical = spx_snapshot.canonical
        source_manifest = spx_snapshot.manifest
    else:
        from qr_haven.data.tiingo import load_tiingo_spy_snapshot

        tiingo_snapshot = load_tiingo_spy_snapshot(design.profile.snapshot_dir)
        canonical = tiingo_snapshot.canonical
        source_manifest = tiingo_snapshot.manifest
    dataset = build_volatility_dataset(canonical, source_manifest, design.profile)
    observations, audit = extend_challenger_features(dataset, canonical)
    return observations, audit, source_manifest


def prepare_challenger_dataset(
    output_dir: Path, *, profile_id: str = SPX_LOCAL_PROFILE.profile_id
) -> dict[str, Any]:
    design = get_challenger_design(profile_id)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable challenger dataset: {destination}")
    if not (design.profile.snapshot_dir / "manifest.json").is_file():
        raise ValueError(
            f"Verified source snapshot is missing: {design.profile.snapshot_dir}. "
            "Acquire and verify the profile data before challenger preparation."
        )
    destination.mkdir(parents=True, exist_ok=False)
    running: dict[str, Any] = {
        "schema_version": 1,
        "protocol": design.protocol,
        "stage": "challenger point-in-time dataset",
        "state": "running",
        "created_at_utc": utc_now(),
        "profile_id": profile_id,
        "evaluation_years": list(design.evaluation_years),
        "evaluation_outcomes_summarized": False,
        "development_access_to_evaluation_outcomes": False,
        "environment": environment_versions(),
        "code": source_identity(),
    }
    atomic_write(destination / "manifest.json", json_bytes(running))
    try:
        observations, audit, source = build_challenger_observations(profile_id)
        source_sha = source.get("source_sha256", source.get("raw_price_sha256"))
        if not isinstance(source_sha, str) or len(source_sha) != 64:
            raise ValueError("Challenger source manifest lacks a valid source hash")
        first_evaluation = pd.Timestamp(f"{design.evaluation_years[0]}-01-01")
        development_mask = (observations["as_of"] < first_evaluation) & (
            observations["label_end"] < first_evaluation
        )
        purged_mask = (observations["as_of"] < first_evaluation) & (
            observations["label_end"] >= first_evaluation
        )
        evaluation_mask = observations["as_of"].dt.year.isin(design.evaluation_years)
        development = observations.loc[development_mask]
        evaluation = observations.loc[evaluation_mask]
        if development.empty or evaluation.empty or purged_mask.sum() != 5:
            raise ValueError("Challenger development/evaluation boundary is invalid")
        membership = pd.concat(
            [
                pd.DataFrame({"sample_id": development.index, "partition": "development"}),
                pd.DataFrame(
                    {"sample_id": observations.index[purged_mask], "partition": "purged_at_2020"}
                ),
                pd.DataFrame({"sample_id": evaluation.index, "partition": "evaluation_2020"}),
            ],
            ignore_index=True,
        )
        files = {
            "development_observations.csv": development.to_csv(index=True).encode(),
            "boundary_features.csv": observations.loc[purged_mask]
            .drop(columns="forward_vol_5")
            .to_csv(index=True)
            .encode(),
            "evaluation_features.csv": evaluation.drop(columns="forward_vol_5")
            .to_csv(index=True)
            .encode(),
            "sealed_evaluation_outcomes.csv": evaluation.loc[:, ["forward_vol_5"]]
            .to_csv(index=True)
            .encode(),
            "feature_window_audit.csv": audit.to_csv(index=True).encode(),
            "membership.csv": membership.to_csv(index=False).encode(),
            "config.json": json_bytes(
                {
                    "protocol": design.protocol,
                    "feature_order": list(CHALLENGER_FEATURES),
                    "evaluation_years": list(design.evaluation_years),
                    "source_sha256": source_sha,
                    "source_dataset_id": source.get("dataset_id", source.get("provider")),
                }
            ),
        }
        for name, payload in files.items():
            atomic_write(destination / name, payload)
        running.update(
            {
                "state": "complete",
                "profile_id": profile_id,
                "source_sha256": source_sha,
                "feature_order": list(CHALLENGER_FEATURES),
                "observation_rows": len(observations),
                "development_rows": len(development),
                "purged_at_evaluation_rows": int(purged_mask.sum()),
                **(
                    {"purged_at_2020_rows": int(purged_mask.sum())}
                    if profile_id == SPX_LOCAL_PROFILE.profile_id
                    else {}
                ),
                "evaluation_rows": len(evaluation),
                "files": {name: sha256(payload) for name, payload in files.items()},
            }
        )
        atomic_write(destination / "manifest.json", json_bytes(running), replace=True)
    except Exception as exc:
        running.update({"state": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        atomic_write(destination / "manifest.json", json_bytes(running), replace=True)
        raise
    return running


def verify_challenger_dataset(
    output_dir: Path, *, include_sealed_outcomes: bool = True
) -> dict[str, Any]:
    root = Path(output_dir)
    value = json.loads((root / "manifest.json").read_text())
    if value.get("protocol") not in {PROTOCOL, TIINGO_PROTOCOL} or value.get("state") != "complete":
        raise ValueError("Unsupported or incomplete challenger dataset")
    required = {
        "development_observations.csv",
        "boundary_features.csv",
        "evaluation_features.csv",
        "sealed_evaluation_outcomes.csv",
        "feature_window_audit.csv",
        "membership.csv",
        "config.json",
    }
    hashes = value.get("files", {})
    if not required <= hashes.keys():
        raise ValueError("Challenger dataset manifest lacks required files")
    for name, digest in hashes.items():
        if name == "sealed_evaluation_outcomes.csv" and not include_sealed_outcomes:
            continue
        if Path(name).name != name or sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"Challenger dataset artifact hash mismatch: {name}")
    return cast(dict[str, Any], value)


def _verified_payload(root: Path, manifest: dict[str, Any], name: str) -> bytes:
    payload = (root / name).read_bytes()
    if manifest["files"].get(name) != sha256(payload):
        raise ValueError(f"Challenger dataset artifact hash mismatch: {name}")
    return payload


def load_challenger_development(
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load development only; deliberately never opens sealed evaluation outcomes."""
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("protocol") not in {PROTOCOL, TIINGO_PROTOCOL}
        or manifest.get("state") != "complete"
    ):
        raise ValueError("Unsupported or incomplete challenger dataset")
    payload = _verified_payload(root, manifest, "development_observations.csv")
    frame = pd.read_csv(
        BytesIO(payload),
        index_col="sample_id",
        parse_dates=["as_of", "available_at", "label_start", "label_end"],
        float_precision="round_trip",
    )
    if tuple(manifest.get("feature_order", ())) != CHALLENGER_FEATURES:
        raise ValueError("Challenger feature order does not match protocol")
    design = get_challenger_design(str(manifest["profile_id"]))
    if (
        frame.empty
        or not frame.index.is_unique
        or frame["as_of"].dt.year.max() != design.evaluation_years[0] - 1
    ):
        raise ValueError("Challenger development observations are invalid")
    values = frame.loc[:, [*CHALLENGER_FEATURES, "forward_vol_5"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Challenger development values must be finite")
    return frame, manifest


def load_challenger_evaluation_features(output_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    root = Path(output_dir)
    payload = _verified_payload(root, manifest, "evaluation_features.csv")
    frame = pd.read_csv(
        BytesIO(payload),
        index_col="sample_id",
        parse_dates=["as_of", "available_at", "label_start", "label_end"],
        float_precision="round_trip",
    )
    design = get_challenger_design(str(manifest["profile_id"]))
    if (
        frame.empty
        or not frame.index.is_unique
        or not frame["as_of"].dt.year.isin(design.evaluation_years).all()
    ):
        raise ValueError("Challenger evaluation features are invalid")
    if not np.isfinite(frame.loc[:, list(CHALLENGER_FEATURES)].to_numpy(float)).all():
        raise ValueError("Challenger evaluation features must be finite")
    return frame


def load_challenger_boundary_features(output_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    root = Path(output_dir)
    payload = _verified_payload(root, manifest, "boundary_features.csv")
    frame = pd.read_csv(
        BytesIO(payload),
        index_col="sample_id",
        parse_dates=["as_of", "available_at", "label_start", "label_end"],
        float_precision="round_trip",
    )
    design = get_challenger_design(str(manifest["profile_id"]))
    if (
        len(frame) != 5
        or not frame.index.is_unique
        or frame["as_of"].dt.year.max() != design.evaluation_years[0] - 1
    ):
        raise ValueError("Challenger boundary features are invalid")
    return frame

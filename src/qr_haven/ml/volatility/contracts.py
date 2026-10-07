"""Frozen contracts for point-in-time volatility observations and splits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

FEATURES = (
    "trailing_vol_5",
    "trailing_vol_10",
    "trailing_vol_20",
    "trailing_vol_60",
    "log_return_1",
    "log_return_5",
    "abs_log_return_1",
    "vol_ratio_5_20",
    "drawdown_60",
)


@dataclass(frozen=True)
class VolatilityProfile:
    """Versioned source and chronological experiment boundary."""

    profile_id: str
    symbol: str
    snapshot_dir: Path
    canonical_filename: str
    study_start: date
    study_end: date
    validation_years: tuple[int, ...]
    holdout_start: date
    holdout_end: date
    calendar: str = "XNYS"


SPX_LOCAL_PROFILE = VolatilityProfile(
    profile_id="spx-local-v1",
    symbol="SPX",
    snapshot_dir=Path("data/processed/market_data/spx-local-2005-2020-v1"),
    canonical_filename="spx_daily_canonical.csv",
    study_start=date(2005, 1, 1),
    study_end=date(2020, 11, 4),
    validation_years=tuple(range(2010, 2018)),
    holdout_start=date(2018, 1, 1),
    holdout_end=date(2019, 12, 31),
)

TIINGO_SPY_PROFILE = VolatilityProfile(
    profile_id="tiingo-spy-v1",
    symbol="SPY",
    snapshot_dir=Path("data/raw/market_data/tiingo/spy-2005-2025-v1"),
    canonical_filename="spy_daily_canonical.csv",
    study_start=date(2005, 1, 1),
    study_end=date(2025, 12, 31),
    validation_years=tuple(range(2010, 2024)),
    holdout_start=date(2024, 1, 1),
    holdout_end=date(2025, 12, 31),
)

PROFILES = {
    profile.profile_id: profile for profile in (SPX_LOCAL_PROFILE, TIINGO_SPY_PROFILE)
}


def get_profile(profile_id: str) -> VolatilityProfile:
    """Return a frozen profile by its public identifier."""
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown volatility profile: {profile_id}") from exc


@dataclass(frozen=True)
class VolatilityDataset:
    """Point-in-time features, continuous outcomes, and exact input-window audit."""

    observations: pd.DataFrame
    interval_audit: pd.DataFrame
    profile: VolatilityProfile
    source_manifest: dict[str, Any]
    source_sha256: str

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self.observations.index.astype(str))


@dataclass(frozen=True)
class WalkForwardFold:
    """One purged expanding-year validation fold and its train-only threshold."""

    fold_id: str
    validation_year: int
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    purged_boundary_ids: tuple[str, ...]
    threshold: float


@dataclass(frozen=True)
class WalkForwardPlan:
    """Development folds plus the still-unevaluated final holdout membership."""

    profile_id: str
    source_sha256: str
    folds: tuple[WalkForwardFold, ...]
    final_training_ids: tuple[str, ...]
    final_purged_boundary_ids: tuple[str, ...]
    final_threshold: float
    holdout_ids: tuple[str, ...]
    holdout_end_purged_ids: tuple[str, ...]
    quarantined_ids: tuple[str, ...]
    metadata: dict[str, Any]

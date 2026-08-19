"""Feature construction for the Dynamic Borrow Demand Surface.

Raw market data → cross-sectional percentile-rank features consumed by the GP model.
All features live in [0, 1] via percentile ranking within the daily cross-section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

N_FEATURES = 5
FEATURE_NAMES = ("sigma_pct", "adv_pct", "si_pct", "etf_pct", "rq_pct")


@dataclass
class RawFeatures:
    """Pre-normalisation observations for one (cusip, date) pair.

    Fields map directly to Dataset columns defined in the spec.
    Use float('nan') for genuinely missing values — the pipeline imputes them.
    """

    cusip: str
    as_of_date: date
    realized_vol: float     # annualised decimal; NaN if < 15 non-zero returns
    adv_usd: float          # 20-day median dollar ADV; NaN if < 15 trading days
    si_ratio: float         # shares_short / float; NaN if no SI data
    etf_ownership_pct: float  # fraction of float held in in-kind ETFs; NaN → 0
    rq_5d_shares: float     # 5-day rolling sum of locate-request shares
    demand_shares: float = 0.0   # training target
    fee_bps: float = float("nan")  # realised borrow fee for calibration
    si_stale: bool = False  # True when SI observation > 5 trading days old


@dataclass
class SurfaceFeatures:
    """Normalised GP input vector for one security on one date.

    All components are cross-sectional percentile ranks in [0, 1].
    """

    sigma_pct: float  # percentile rank of realised/implied vol
    adv_pct: float    # percentile rank of 20-day median dollar ADV
    si_pct: float     # percentile rank of short interest ratio
    etf_pct: float    # percentile rank of in-kind ETF ownership fraction
    rq_pct: float     # percentile rank of 5-day rolling locate request volume

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.sigma_pct, self.adv_pct, self.si_pct, self.etf_pct, self.rq_pct],
            dtype=np.float32,
        )

    def to_tensor(self):  # type: ignore[return]
        import torch
        return torch.tensor(self.to_array())


def _pct_rank(arr: np.ndarray, nan_fill: float) -> np.ndarray:
    """Percentile rank within arr, with nan_fill for NaN entries."""
    from scipy.stats import rankdata

    valid = ~np.isnan(arr)
    result = np.full(len(arr), nan_fill, dtype=np.float64)
    n_valid = int(valid.sum())
    if n_valid > 0:
        ranks = rankdata(arr[valid], method="average")
        denom = max(n_valid - 1, 1)
        result[valid] = (ranks - 1.0) / denom
    return result


class FeaturePipeline:
    """Cross-sectional percentile-ranking pipeline.

    Call ``transform(raw_batch)`` with a list of RawFeatures all from the same
    date.  Each call is independent — no state is stored across dates.
    """

    def transform(self, raw_batch: list[RawFeatures]) -> list[SurfaceFeatures]:
        """Normalise a same-date cross-section to SurfaceFeatures.

        Missing-value imputation rules (from spec):
          sigma  NaN → 0.5  (cross-sectional median rank)
          adv    NaN → 0.5
          si     NaN → 0.5; stale → raw × 0.95 before ranking
          etf    NaN → 0.0  (no ETF pressure assumed)
          rq     NaN / 0 → 0.0  (no recent demand observed)
        """
        n = len(raw_batch)
        if n == 0:
            return []

        sigma = np.array([r.realized_vol for r in raw_batch], dtype=np.float64)
        adv = np.array([r.adv_usd for r in raw_batch], dtype=np.float64)
        si = np.array([r.si_ratio for r in raw_batch], dtype=np.float64)
        etf = np.array([r.etf_ownership_pct for r in raw_batch], dtype=np.float64)
        rq = np.array([r.rq_5d_shares for r in raw_batch], dtype=np.float64)

        # Apply SI staleness decay before ranking
        for i, r in enumerate(raw_batch):
            if r.si_stale and not np.isnan(si[i]):
                si[i] *= 0.95

        sigma_pct = _pct_rank(sigma, nan_fill=0.5)
        adv_pct = _pct_rank(adv, nan_fill=0.5)
        si_pct = _pct_rank(si, nan_fill=0.5)
        etf_pct = _pct_rank(etf, nan_fill=0.0)
        rq_pct = _pct_rank(rq, nan_fill=0.0)

        return [
            SurfaceFeatures(
                sigma_pct=float(sigma_pct[i]),
                adv_pct=float(adv_pct[i]),
                si_pct=float(si_pct[i]),
                etf_pct=float(etf_pct[i]),
                rq_pct=float(rq_pct[i]),
            )
            for i in range(n)
        ]

    def build_matrix(
        self, raw_batch: list[RawFeatures]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays ready for training.

        X shape: (n, N_FEATURES), dtype float32
        y shape: (n,), dtype float32  — demand_shares (training target)
        """
        features = self.transform(raw_batch)
        X = np.stack([f.to_array() for f in features])
        y = np.array([r.demand_shares for r in raw_batch], dtype=np.float32)
        return X, y

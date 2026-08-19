"""Monotone spline calibration: demand → borrow fee in bps.

Maps posterior mean demand estimates from the GP surface to borrow-rate
recommendations using an isotonic regression fit on historical (demand, fee)
pairs.  Isotonic regression guarantees that higher demand always maps to an
equal or higher fee (monotone non-decreasing).

Reference: spec § Phase 3 — Downstream Applications / Rate Recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    n_observations: int
    demand_range: tuple[float, float]   # (min, max) of training demand
    fee_range: tuple[float, float]      # (min, max) of fitted fee
    n_knots: int                        # number of isotonic regression knots


class DemandRateCalibrator:
    """Monotone isotonic-regression spline from demand to borrow fee.

    Usage
    -----
    calibrator = DemandRateCalibrator()
    calibrator.fit(demand_array, fee_bps_array, sample_weight=notional_array)
    fee = calibrator.fee_for_demand(demand_estimate)
    """

    def __init__(self) -> None:
        self._demand_knots: np.ndarray | None = None
        self._fee_knots: np.ndarray | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        demand: np.ndarray,
        fee_bps: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> CalibrationResult:
        """Fit isotonic regression of fee_bps on demand.

        Parameters
        ----------
        demand : array (n,)
            Historical demand observations (daily locate request shares per CUSIP).
        fee_bps : array (n,)
            Corresponding realised borrow fee in annualised bps.
        sample_weight : array (n,), optional
            Per-observation weight, e.g. total_notional_usd.  Larger loans
            dominate the fit.

        Returns
        -------
        CalibrationResult with summary statistics.
        """
        from sklearn.isotonic import IsotonicRegression

        demand = np.asarray(demand, dtype=np.float64)
        fee_bps = np.asarray(fee_bps, dtype=np.float64)

        valid = ~np.isnan(demand) & ~np.isnan(fee_bps) & (fee_bps >= 0)
        if valid.sum() < 10:
            raise ValueError(
                f"Calibration requires ≥ 10 valid observations; got {int(valid.sum())}"
            )

        w = sample_weight[valid].astype(np.float64) if sample_weight is not None else None
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        ir.fit(demand[valid], fee_bps[valid], sample_weight=w)

        self._demand_knots = ir.X_thresholds_
        self._fee_knots = ir.y_thresholds_
        self._fitted = True

        return CalibrationResult(
            n_observations=int(valid.sum()),
            demand_range=(float(demand[valid].min()), float(demand[valid].max())),
            fee_range=(float(self._fee_knots.min()), float(self._fee_knots.max())),
            n_knots=len(self._demand_knots),
        )

    def predict(self, demand: np.ndarray) -> np.ndarray:
        """Return fee_bps estimates for an array of demand values."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        demand = np.asarray(demand, dtype=np.float64)
        return np.interp(demand, self._demand_knots, self._fee_knots)

    def fee_for_demand(self, demand_shares: float) -> float:
        """Single-value convenience wrapper around predict()."""
        return float(self.predict(np.array([demand_shares]))[0])

    def fee_with_uncertainty(
        self,
        demand_mean: float,
        demand_std: float,
        n_samples: int = 500,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Propagate GP posterior uncertainty through the calibration spline.

        Samples demand from N(demand_mean, demand_std²), maps each sample
        through the spline, and returns (mean_fee, std_fee).

        Parameters
        ----------
        demand_mean : float
            Posterior predictive mean from the GP surface.
        demand_std : float
            Posterior predictive std from the GP surface.
        n_samples : int
            Monte Carlo samples (default 500 — sufficient for ±2 bps accuracy).

        Returns
        -------
        (mean_fee_bps, std_fee_bps)
        """
        rng = np.random.default_rng(seed)
        samples = rng.normal(demand_mean, demand_std, size=n_samples)
        samples = np.maximum(samples, 0.0)  # demand is non-negative
        fees = self.predict(samples)
        return float(fees.mean()), float(fees.std())

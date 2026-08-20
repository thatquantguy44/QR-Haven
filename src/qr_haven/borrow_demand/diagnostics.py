"""Diagnostic metrics for the borrow demand surface.

Three evaluation dimensions:
  1. Surface accuracy — RMSE / MAE of GP demand predictions vs actuals
  2. Calibration reliability — reliability diagram of demand→fee spline
  3. Shortage recall — how often shortage events are flagged in advance

Reference: spec § Phase 4 — Evaluation & Monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SurfaceMetrics:
    """Pointwise accuracy metrics for the GP demand surface."""

    n_observations: int
    rmse: float           # root mean squared error (demand units)
    mae: float            # mean absolute error (demand units)
    coverage_90: float    # fraction of actuals within ±1.645σ predictive interval
    coverage_95: float    # fraction of actuals within ±1.960σ predictive interval
    mean_sharpness: float  # mean predictive std (lower is sharper)


@dataclass
class CalibrationDiagnostic:
    """Reliability-diagram data for the demand→fee calibration spline."""

    n_bins: int
    bin_midpoints: np.ndarray   # shape (n_bins,) — demand percentile bin centres
    mean_predicted_fee: np.ndarray   # shape (n_bins,)
    mean_actual_fee: np.ndarray      # shape (n_bins,)
    bin_counts: np.ndarray           # shape (n_bins,) — observations per bin
    mean_absolute_error: float       # weighted MAE across bins


@dataclass
class ShortageRecallMetrics:
    """Precision / recall of shortage event detection."""

    threshold: float        # demand threshold used to define predicted shortage
    n_actual_shortages: int
    n_predicted_shortages: int
    true_positives: int
    precision: float
    recall: float
    f1: float


def surface_rmse(
    actual_demand: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
) -> SurfaceMetrics:
    """Compute pointwise accuracy and calibration metrics for the GP surface.

    Parameters
    ----------
    actual_demand : array (n,)
        Held-out ground-truth demand observations.
    predicted_mean : array (n,)
        GP posterior predictive means.
    predicted_std : array (n,)
        GP posterior predictive standard deviations.

    Returns
    -------
    SurfaceMetrics with RMSE, MAE, and interval coverage statistics.
    """
    actual = np.asarray(actual_demand, dtype=np.float64)
    pred_mean = np.asarray(predicted_mean, dtype=np.float64)
    pred_std = np.asarray(predicted_std, dtype=np.float64)

    if actual.shape != pred_mean.shape or actual.shape != pred_std.shape:
        raise ValueError("actual_demand, predicted_mean, and predicted_std must have the same shape")

    n = len(actual)
    if n == 0:
        raise ValueError("Cannot compute metrics on empty arrays")

    residuals = actual - pred_mean
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    z_scores = np.abs(residuals) / np.maximum(pred_std, 1e-12)
    coverage_90 = float(np.mean(z_scores <= 1.645))
    coverage_95 = float(np.mean(z_scores <= 1.960))
    mean_sharpness = float(np.mean(pred_std))

    return SurfaceMetrics(
        n_observations=n,
        rmse=rmse,
        mae=mae,
        coverage_90=coverage_90,
        coverage_95=coverage_95,
        mean_sharpness=mean_sharpness,
    )


def calibration_reliability(
    demand: np.ndarray,
    predicted_fee: np.ndarray,
    actual_fee: np.ndarray,
    n_bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> CalibrationDiagnostic:
    """Build a reliability diagram for the demand→fee calibration spline.

    Bins demand into *n_bins* equal-frequency buckets.  Within each bin,
    compares mean predicted fee to mean realised fee.

    Parameters
    ----------
    demand : array (n,)
        Demand values used to generate fee predictions.
    predicted_fee : array (n,)
        Fee predictions from DemandRateCalibrator.predict(demand).
    actual_fee : array (n,)
        Corresponding realised borrow fees in bps.
    n_bins : int
        Number of equal-frequency bins (default 10).
    sample_weight : array (n,), optional
        Per-observation notional weights for weighted averaging.

    Returns
    -------
    CalibrationDiagnostic with per-bin averages and overall weighted MAE.
    """
    demand = np.asarray(demand, dtype=np.float64)
    predicted_fee = np.asarray(predicted_fee, dtype=np.float64)
    actual_fee = np.asarray(actual_fee, dtype=np.float64)

    valid = ~np.isnan(demand) & ~np.isnan(predicted_fee) & ~np.isnan(actual_fee)
    demand = demand[valid]
    predicted_fee = predicted_fee[valid]
    actual_fee = actual_fee[valid]
    w = sample_weight[valid] if sample_weight is not None else np.ones(len(demand))

    n = len(demand)
    if n < n_bins:
        raise ValueError(f"Need at least {n_bins} valid observations; got {n}")

    # Equal-frequency bin edges via percentile on demand
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(demand, percentiles)
    bin_edges[-1] += 1e-12  # include max in last bin

    bin_idx = np.digitize(demand, bin_edges[1:])  # 0-indexed bucket

    bin_midpoints = np.zeros(n_bins)
    mean_pred = np.zeros(n_bins)
    mean_act = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] == 0:
            continue
        w_b = w[mask]
        w_sum = w_b.sum()
        if w_sum <= 0:
            w_b = np.ones(counts[b])
            w_sum = float(counts[b])
        bin_midpoints[b] = float(np.average(demand[mask], weights=w_b))
        mean_pred[b] = float(np.average(predicted_fee[mask], weights=w_b))
        mean_act[b] = float(np.average(actual_fee[mask], weights=w_b))

    filled = counts > 0
    total_count = counts[filled].sum()
    weighted_mae = float(
        np.sum(np.abs(mean_pred[filled] - mean_act[filled]) * counts[filled]) / total_count
        if total_count > 0
        else 0.0
    )

    return CalibrationDiagnostic(
        n_bins=n_bins,
        bin_midpoints=bin_midpoints,
        mean_predicted_fee=mean_pred,
        mean_actual_fee=mean_act,
        bin_counts=counts,
        mean_absolute_error=weighted_mae,
    )


def shortage_recall(
    actual_shortage: np.ndarray,
    predicted_demand: np.ndarray,
    demand_threshold: float,
) -> ShortageRecallMetrics:
    """Evaluate shortage event detection precision / recall.

    A shortage is *predicted* when predicted_demand > demand_threshold.
    A shortage is *actual* when actual_shortage is True (or 1).

    Parameters
    ----------
    actual_shortage : bool/int array (n,)
        Ground-truth shortage indicator (1 = shortage occurred).
    predicted_demand : array (n,)
        GP posterior mean demand predictions.
    demand_threshold : float
        Demand level above which a shortage is predicted.

    Returns
    -------
    ShortageRecallMetrics with precision, recall, and F1.
    """
    actual = np.asarray(actual_shortage, dtype=bool)
    pred_demand = np.asarray(predicted_demand, dtype=np.float64)

    predicted = pred_demand > demand_threshold

    n_actual = int(actual.sum())
    n_predicted = int(predicted.sum())
    tp = int((actual & predicted).sum())

    precision = tp / n_predicted if n_predicted > 0 else 0.0
    recall = tp / n_actual if n_actual > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return ShortageRecallMetrics(
        threshold=demand_threshold,
        n_actual_shortages=n_actual,
        n_predicted_shortages=n_predicted,
        true_positives=tp,
        precision=precision,
        recall=recall,
        f1=f1,
    )

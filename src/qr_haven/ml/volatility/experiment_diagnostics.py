"""Descriptive error tables from saved predictions; no model fitting or new forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def error_tables(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Annotate errors and group by year and features known at the forecast origin."""
    rows = predictions.sort_values(["partition", "model_id", "as_of"]).copy()
    if rows.empty or rows.duplicated(["partition", "model_id", "sample_id"]).any():
        raise ValueError("Error diagnosis requires nonempty, unique saved predictions")
    if not set(rows["true_class"]) <= {0, 1} or not set(rows["predicted_class"]) <= {0, 1}:
        raise ValueError("Error diagnosis requires binary labels")
    if not np.isfinite(rows[["threshold", "trailing_vol_5", "trailing_vol_20"]]).all().all():
        raise ValueError("Error diagnosis requires finite thresholds and regime features")
    shared = rows.groupby(["partition", "sample_id"])[["true_class", "threshold"]].nunique()
    if not (shared == 1).all().all():
        raise ValueError("Saved models must share the same truth and threshold")
    rows["year"] = pd.to_datetime(rows["as_of"]).dt.year
    short_high = rows["trailing_vol_5"] > rows["threshold"]
    medium_high = rows["trailing_vol_20"] > rows["threshold"]
    rows["regime"] = np.select(
        [short_high & medium_high, short_high & ~medium_high, ~short_high & medium_high],
        ["elevated", "rising", "cooling"],
        default="calm",
    )
    high = rows["true_class"].eq(1)
    predicted_high = rows["predicted_class"].eq(1)
    rows["outcome"] = np.select(
        [high & predicted_high, ~high & predicted_high, high & ~predicted_high],
        ["TP", "FP", "FN"],
        default="TN",
    )
    rows["error"] = rows["true_class"] != rows["predicted_class"]
    summaries: list[dict[str, Any]] = []
    for grouping, dimensions in (
        ("overall", []),
        ("year", ["year"]),
        ("regime", ["regime"]),
        ("year_regime", ["year", "regime"]),
    ):
        columns = ["partition", "model_id", *dimensions]
        for keys, group in rows.groupby(columns, sort=True):
            counts = group["outcome"].value_counts()
            tp, fp, fn, tn = (int(counts.get(key, 0)) for key in ("TP", "FP", "FN", "TN"))
            normal, high_count = tn + fp, tp + fn
            summaries.append(
                {
                    **dict(zip(columns, keys, strict=True)),
                    "grouping": grouping,
                    "rows": len(group),
                    "normal_support": normal,
                    "high_support": high_count,
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_positive": tp,
                    "false_positive_rate": fp / normal if normal else None,
                    "miss_rate": fn / high_count if high_count else None,
                    "high_recall": tp / high_count if high_count else None,
                    "high_precision": tp / (tp + fp) if tp + fp else None,
                }
            )

    runs: list[dict[str, Any]] = []
    for keys, group in rows.groupby(["partition", "model_id", "fold_id"], sort=True):
        ordered = group.sort_values("as_of")
        segments = ordered["outcome"].ne(ordered["outcome"].shift()).cumsum()
        for _, segment in ordered.groupby(segments, sort=False):
            outcome = segment["outcome"].iloc[0]
            if outcome not in {"FP", "FN"}:
                continue
            runs.append(
                {
                    "partition": keys[0],
                    "model_id": keys[1],
                    "fold_id": keys[2],
                    "outcome": outcome,
                    "start": segment["as_of"].iloc[0],
                    "end": segment["as_of"].iloc[-1],
                    "origins": len(segment),
                    "first_sample_id": segment["sample_id"].iloc[0],
                    "last_sample_id": segment["sample_id"].iloc[-1],
                }
            )
    run_columns = [
        "partition",
        "model_id",
        "fold_id",
        "outcome",
        "start",
        "end",
        "origins",
        "first_sample_id",
        "last_sample_id",
    ]
    paired: list[dict[str, Any]] = []
    for partition, group in rows.groupby("partition", sort=True):
        matrix = group.pivot(index="sample_id", columns="model_id", values="predicted_class")
        if "baseline_persistence" not in matrix or matrix.isna().any().any():
            raise ValueError("Diagnosis requires persistence on every saved forecast origin")
        truths = group.drop_duplicates("sample_id").set_index("sample_id")["true_class"]
        truths = truths.loc[matrix.index]
        baseline = matrix["baseline_persistence"]
        for model_id in matrix.columns:
            if model_id == "baseline_persistence":
                continue
            model = matrix[model_id]
            paired.append(
                {
                    "partition": partition,
                    "model_id": model_id,
                    "rows": len(matrix),
                    "extra_high_detections": int(
                        ((truths == 1) & (model == 1) & (baseline == 0)).sum()
                    ),
                    "lost_high_detections": int(
                        ((truths == 1) & (model == 0) & (baseline == 1)).sum()
                    ),
                    "extra_false_alarms": int(
                        ((truths == 0) & (model == 1) & (baseline == 0)).sum()
                    ),
                    "avoided_false_alarms": int(
                        ((truths == 0) & (model == 0) & (baseline == 1)).sum()
                    ),
                }
            )
    return {
        "diagnostic_observations.csv": rows.reset_index(drop=True),
        "error_summary.csv": pd.DataFrame(summaries),
        "error_runs.csv": pd.DataFrame(runs, columns=run_columns),
        "paired_errors.csv": pd.DataFrame(paired),
    }

"""Nested chronological selection and frozen final fit for the V5 challenger."""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from qr_haven.data.banknotes import atomic_write, json_bytes, sha256, utc_now
from qr_haven.ml.classification.evaluation import classification_metrics
from qr_haven.ml.volatility.challenger_data import (
    CHALLENGER_FEATURES,
    PROTOCOL,
    TIINGO_PROTOCOL,
    ChallengerDesign,
    get_challenger_design,
    load_challenger_boundary_features,
    load_challenger_development,
    load_challenger_evaluation_features,
)
from qr_haven.ml.volatility.challenger_models import (
    ENSEMBLE_WEIGHTS,
    ChallengerCandidate,
    calibrated_scores,
    challenger_candidates,
    ensemble_predictions,
    ewma_scores,
    fit_base_model,
    fit_platt,
    five_year_training_ids,
    raw_scores,
    threshold_for,
)
from qr_haven.ml.volatility.persistence import environment_versions, source_identity


def _metric_row(target: pd.Series, prediction: pd.DataFrame) -> dict[str, Any]:
    value = classification_metrics(target, prediction)
    tn, fp = value["confusion_matrix"][0]
    fn, tp = value["confusion_matrix"][1]
    return {
        "rows": len(target),
        "accuracy": value["accuracy"],
        "balanced_accuracy": value["balanced_accuracy"],
        "macro_f1": value["f1_macro"],
        "high_precision": value["per_class"]["1"]["precision"],
        "high_recall": value["per_class"]["1"]["recall"],
        "high_f1": value["per_class"]["1"]["f1"],
        "roc_auc": value["roc_auc"],
        "average_precision": value["average_precision"],
        "normal_support": tn + fp,
        "high_support": tp + fn,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "diagnostics": json.dumps(value["diagnostics"]),
    }


def _fold_raw_prediction(
    observations: pd.DataFrame,
    candidate: ChallengerCandidate,
    year: int,
    ewma: pd.Series | None,
) -> tuple[pd.DataFrame, dict[str, Any], Any]:
    training_ids = five_year_training_ids(observations, year)
    threshold = threshold_for(observations, training_ids)
    validation = observations.loc[observations["as_of"].dt.year.eq(year)]
    validation_ids = tuple(validation.index.astype(str))
    model = fit_base_model(candidate, observations, training_ids, threshold)
    score = raw_scores(candidate, model, observations, validation_ids, ewma=ewma)
    target = (validation["forward_vol_5"] > threshold).astype("int64")
    frame = pd.DataFrame(
        {
            "raw_score": score,
            "true_class": target,
            "trailing_vol_5": validation["trailing_vol_5"],
            "as_of": validation["as_of"],
            "threshold": threshold,
            "validation_year": year,
        }
    )
    audit = {
        "candidate_id": candidate.candidate_id,
        "validation_year": year,
        "training_rows": len(training_ids),
        "training_start": observations.loc[training_ids[0], "as_of"],
        "training_end": observations.loc[training_ids[-1], "as_of"],
        "max_training_label_end": observations.loc[list(training_ids), "label_end"].max(),
        "validation_start": validation["as_of"].iloc[0],
        "threshold": threshold,
        "fitted_model": candidate.kind != "ewma",
    }
    return frame, audit, model


def _rank(results: pd.DataFrame, outer_years: tuple[int, ...]) -> pd.DataFrame:
    expected = set(outer_years)
    rows: list[dict[str, Any]] = []
    order = {candidate.candidate_id: candidate.order for candidate in challenger_candidates()}
    for (candidate_id, weight, frequency), group in results.groupby(
        ["candidate_id", "model_weight", "frequency"], sort=False
    ):
        if len(group) != len(expected) or set(group["validation_year"]) != expected:
            raise ValueError("Challenger ranking requires every frozen outer fold")
        rows.append(
            {
                "candidate_id": candidate_id,
                "model_weight": weight,
                "frequency": frequency,
                "candidate_order": order[candidate_id],
                "weight_order": ENSEMBLE_WEIGHTS.index(float(weight)),
                **{
                    f"mean_{name}": float(group[name].mean())
                    for name in (
                        "balanced_accuracy",
                        "macro_f1",
                        "accuracy",
                        "high_precision",
                        "high_recall",
                    )
                },
                "false_positive": int(group["false_positive"].sum()),
                "false_negative": int(group["false_negative"].sum()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        [
            "frequency",
            "mean_balanced_accuracy",
            "mean_macro_f1",
            "candidate_order",
            "weight_order",
        ],
        ascending=[True, False, False, True, True],
        kind="stable",
    )
    ranking["rank"] = ranking.groupby("frequency").cumcount() + 1
    return ranking.reset_index(drop=True)


def _report(ranking: pd.DataFrame, selection: dict[str, Any], design: ChallengerDesign) -> bytes:
    daily = ranking.loc[ranking["frequency"] == "daily"]
    lines = [
        "# Frozen five-year volatility challenger: development selection",
        "",
        f"Candidate selection used outer development folds from {design.outer_years[0]} through "
        f"{design.outer_years[-1]}. Evaluation outcomes for "
        f"{design.evaluation_years[0]}–{design.evaluation_years[-1]} remained sealed.",
        "",
        f"Selected base forecast: `{selection['candidate_id']}`",
        "",
        f"Selected calibrated-model weight: `{selection['model_weight']:.2f}`; persistence weight: "
        f"`{1 - selection['model_weight']:.2f}`.",
        "",
        "| Rank | Base forecast | Model weight | Balanced accuracy | Macro F1 | Accuracy | "
        "High recall | False alarms | Misses |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in daily.to_dict("records"):
        lines.append(
            f"| {row['rank']} | {row['candidate_id']} | {row['model_weight']:.2f} | "
            f"{row['mean_balanced_accuracy']:.4f} | {row['mean_macro_f1']:.4f} | "
            f"{row['mean_accuracy']:.4f} | {row['mean_high_recall']:.4f} | "
            f"{row['false_positive']} | {row['false_negative']} |"
        )
    lines.extend(
        [
            "",
            "Every outer score used a Platt calibrator fitted on out-of-fold predictions from "
            "the three preceding years. Each underlying forecast was trained on its own preceding "
            "five-year window with boundary labels purged. The final model and calibrator use only "
            "observations preceding the evaluation boundary.",
            "",
            "The selected pipeline is frozen before the single evaluation outcome exposure. "
            "Development ranking is selection evidence and is not an unbiased final performance "
            "estimate.",
            "",
        ]
    )
    return "\n".join(lines).encode()


def train_challenger(
    dataset_dir: Path,
    output_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    root, output = Path(dataset_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable challenger run: {output}")
    observations, dataset_manifest = load_challenger_development(root)
    design = get_challenger_design(str(dataset_manifest["profile_id"]))
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "protocol": design.protocol,
        "profile_id": design.profile.profile_id,
        "outer_years": list(design.outer_years),
        "calibration_years_per_outer_fold": 3,
        "training_window_years": 5,
        "threshold_quantile": 0.75,
        "candidates": [candidate.__dict__ for candidate in challenger_candidates()],
        "ensemble_weights": list(ENSEMBLE_WEIGHTS),
        "feature_order": list(CHALLENGER_FEATURES),
        "ranking": "mean yearly balanced accuracy, macro F1, candidate order, weight order",
        "final_calibration_years": list(design.final_calibration_years),
        "final_training_years": list(
            range(design.evaluation_years[0] - 5, design.evaluation_years[0])
        ),
        "evaluation_years": list(design.evaluation_years),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": design.protocol,
        "stage": "nested development selection and final fit",
        "state": "running",
        "profile_id": design.profile.profile_id,
        "created_at_utc": utc_now(),
        "dataset_manifest_sha256": sha256((root / "manifest.json").read_bytes()),
        "source_sha256": dataset_manifest["source_sha256"],
        "environment": environment_versions(),
        "code": source_identity(),
        "selection_uses_evaluation_outcomes": False,
        "selection_uses_2018_2019": bool({2018, 2019} & set(design.outer_years)),
        "evaluation_outcomes_opened": False,
    }
    atomic_write(output / "config.json", json_bytes(config))
    atomic_write(output / "manifest.json", json_bytes(manifest))
    started = time.perf_counter()
    try:
        ewmas = {
            candidate.candidate_id: ewma_scores(observations, float(candidate.decay))
            for candidate in challenger_candidates()
            if candidate.decay is not None
        }
        raw_cache: dict[tuple[str, int], pd.DataFrame] = {}
        audits: list[dict[str, Any]] = []
        memberships: list[dict[str, Any]] = []
        for candidate in challenger_candidates():
            first_required_year = design.outer_years[0] - 3
            last_required_year = max(design.outer_years[-1], design.final_calibration_years[-1])
            for year in range(first_required_year, last_required_year + 1):
                if progress:
                    progress(f"Preparing {candidate.candidate_id} chronological fold {year}")
                frame, audit, _model = _fold_raw_prediction(
                    observations, candidate, year, ewmas.get(candidate.candidate_id)
                )
                raw_cache[(candidate.candidate_id, year)] = frame
                audits.append(audit)
                if candidate.kind != "ewma":
                    memberships.extend(
                        {
                            "candidate_id": candidate.candidate_id,
                            "validation_year": year,
                            "sample_id": sample_id,
                        }
                        for sample_id in five_year_training_ids(observations, year)
                    )
        result_rows: list[dict[str, Any]] = []
        prediction_rows: list[pd.DataFrame] = []
        calibration_rows: list[pd.DataFrame] = []
        for candidate in challenger_candidates():
            for outer_year in design.outer_years:
                calibration_years = range(outer_year - 3, outer_year)
                calibration = pd.concat(
                    [raw_cache[(candidate.candidate_id, year)] for year in calibration_years]
                )
                calibrator = fit_platt(calibration["raw_score"], calibration["true_class"])
                calibration_record = calibration.copy()
                calibration_record["candidate_id"] = candidate.candidate_id
                calibration_record["outer_year"] = outer_year
                calibration_rows.append(calibration_record.reset_index(names="sample_id"))
                outer = raw_cache[(candidate.candidate_id, outer_year)]
                probability = calibrated_scores(calibrator, outer["raw_score"])
                for weight in ENSEMBLE_WEIGHTS:
                    predicted = ensemble_predictions(
                        probability,
                        outer["trailing_vol_5"],
                        float(outer["threshold"].iloc[0]),
                        weight,
                    )
                    common = {
                        "candidate_id": candidate.candidate_id,
                        "model_weight": weight,
                        "validation_year": outer_year,
                        "threshold": float(outer["threshold"].iloc[0]),
                    }
                    for frequency, positions in (
                        ("daily", np.arange(len(outer))),
                        ("every_fifth", np.arange(0, len(outer), 5)),
                    ):
                        result_rows.append(
                            {
                                **common,
                                "frequency": frequency,
                                **_metric_row(
                                    outer["true_class"].iloc[positions],
                                    predicted.iloc[positions],
                                ),
                            }
                        )
                    recorded = predicted.copy()
                    recorded["raw_score"] = outer["raw_score"]
                    recorded["calibrated_probability"] = probability
                    recorded["true_class"] = outer["true_class"]
                    recorded["as_of"] = outer["as_of"]
                    recorded["every_fifth"] = np.arange(len(outer)) % 5 == 0
                    for name, value in common.items():
                        recorded[name] = value
                    prediction_rows.append(recorded.reset_index(names="sample_id"))
        results = pd.DataFrame(result_rows)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        ranking = _rank(results, design.outer_years)
        chosen = ranking.loc[(ranking["frequency"] == "daily") & (ranking["rank"] == 1)].iloc[0]
        selection = {
            "candidate_id": str(chosen["candidate_id"]),
            "model_weight": float(chosen["model_weight"]),
            "mean_balanced_accuracy": float(chosen["mean_balanced_accuracy"]),
            "mean_macro_f1": float(chosen["mean_macro_f1"]),
        }
        candidate = next(
            item
            for item in challenger_candidates()
            if item.candidate_id == selection["candidate_id"]
        )
        final_calibration = pd.concat(
            [raw_cache[(candidate.candidate_id, year)] for year in design.final_calibration_years]
        )
        calibrator = fit_platt(final_calibration["raw_score"], final_calibration["true_class"])
        evaluation_features = load_challenger_evaluation_features(root, dataset_manifest)
        final_ids = five_year_training_ids(
            observations,
            design.evaluation_years[0],
            validation_boundary=pd.Timestamp(evaluation_features["as_of"].iloc[0]),
        )
        final_threshold = threshold_for(observations, final_ids)
        estimator = fit_base_model(candidate, observations, final_ids, final_threshold)
        ewma_variance = None
        if candidate.kind == "ewma":
            if candidate.decay is None:
                raise ValueError("EWMA candidate lacks its frozen decay")
            boundary = load_challenger_boundary_features(root, dataset_manifest)
            bridge = pd.concat([observations, boundary]).sort_values("as_of")
            final_ewma = ewma_scores(bridge, candidate.decay)
            ewma_variance = float(final_ewma.iloc[-1] ** 2 / 252)
        bundle = {
            "protocol": design.protocol,
            "candidate": candidate.__dict__,
            "model_weight": selection["model_weight"],
            "threshold": final_threshold,
            "feature_order": CHALLENGER_FEATURES,
            "estimator": estimator,
            "calibrator": calibrator,
            "ewma_variance_after_boundary": ewma_variance,
            "final_training_ids": final_ids,
            "calibration_years": design.final_calibration_years,
            "source_sha256": dataset_manifest["source_sha256"],
        }
        payloads = {
            "cv_results.csv": results.to_csv(index=False).encode(),
            "cv_predictions.csv": predictions.to_csv(index=False).encode(),
            "calibration_predictions.csv": pd.concat(calibration_rows, ignore_index=True)
            .to_csv(index=False)
            .encode(),
            "ranking.csv": ranking.to_csv(index=False).encode(),
            "fit_audit.csv": pd.DataFrame(audits).to_csv(index=False).encode(),
            "training_membership.csv": pd.DataFrame(memberships).to_csv(index=False).encode(),
            "selection.json": json_bytes(selection),
            "model.pkl": pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL),
            "development_report.md": _report(ranking, selection, design),
        }
        for name, payload in payloads.items():
            atomic_write(output / name, payload)
        payloads["config.json"] = (output / "config.json").read_bytes()
        manifest.update(
            {
                "state": "complete",
                "selected_candidate": selection["candidate_id"],
                "selected_model_weight": selection["model_weight"],
                "final_threshold": final_threshold,
                "final_training_rows": len(final_ids),
                "final_training_uses_2018_2019": design.evaluation_years[0] > 2019,
                "selection_uses_2018_2019": bool({2018, 2019} & set(design.outer_years)),
                "selection_uses_evaluation_outcomes": False,
                "evaluation_outcomes_opened": False,
                "seconds": time.perf_counter() - started,
                "files": {name: sha256(payload) for name, payload in payloads.items()},
            }
        )
        atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
    except Exception as exc:
        manifest.update({"state": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        atomic_write(output / "manifest.json", json_bytes(manifest), replace=True)
        raise
    return manifest


def verify_challenger_run(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    value = json.loads((root / "manifest.json").read_text())
    if value.get("protocol") not in {PROTOCOL, TIINGO_PROTOCOL} or value.get("state") != "complete":
        raise ValueError("Unsupported or incomplete challenger run")
    required = {
        "config.json",
        "cv_results.csv",
        "cv_predictions.csv",
        "calibration_predictions.csv",
        "ranking.csv",
        "fit_audit.csv",
        "training_membership.csv",
        "selection.json",
        "model.pkl",
        "development_report.md",
    }
    hashes = value.get("files", {})
    if not required <= hashes.keys():
        raise ValueError("Challenger run manifest lacks required files")
    for name, digest in hashes.items():
        if Path(name).name != name or sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"Challenger run artifact hash mismatch: {name}")
    return cast(dict[str, Any], value)


def load_challenger_model(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest = verify_challenger_run(root)
    value = pickle.loads((root / "model.pkl").read_bytes())  # noqa: S301 - verified local artifact
    if (
        not isinstance(value, dict)
        or value.get("protocol") != manifest["protocol"]
        or value.get("threshold") != manifest["final_threshold"]
        or value.get("candidate", {}).get("candidate_id") != manifest["selected_candidate"]
        or tuple(value.get("feature_order", ())) != CHALLENGER_FEATURES
    ):
        raise ValueError("Challenger model bundle disagrees with its manifest")
    return cast(dict[str, Any], value)

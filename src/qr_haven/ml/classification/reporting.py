"""Source-backed Markdown model card assembled from frozen run evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from qr_haven.data.banknotes import DATASET_URL, BanknoteDataset, atomic_write
from qr_haven.ml.classification.artifacts import read_json
from qr_haven.ml.classification.contracts import SplitManifest


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
        ]
    )


def write_report(
    run_dir: Path,
    manifest: dict[str, Any],
    dataset: BanknoteDataset,
    split: SplitManifest,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
) -> None:
    selection = read_json(run_dir / "selection.json")
    winner, baseline = metrics["winner"], metrics["baseline"]
    sections = [
        "# Banknote authentication model card",
        f"Predict numeric banknote class from four supplied wavelet/image statistics. "
        f"**Benchmark status: {metrics['benchmark_status']}**. "
        f"Winner accuracy: **{winner['accuracy']:.4%} ({winner['correct']}/{winner['total']})**. "
        "Engineering: workflow complete; see the research writeup for repository checks.",
        "## Data and class definitions",
        f"[UCI Banknote Authentication]({DATASET_URL}). {dataset.source_manifest['citation']} "
        f"License: {dataset.source_manifest['license']}. "
        f"Raw SHA-256: `{dataset.source_manifest['raw_sha256']}`. "
        f"Acquisition: `{dataset.source_manifest['acquisition']}`. "
        f"Retrieved at UTC: {dataset.source_manifest['retrieved_at_utc']}. "
        "Class mapping is **unverified**: class_0 and class_1 retain source codes. "
        "Label 1 is the statistical positive class. Semantic class names remain unverified.",
        "## Validation and frozen partitions",
        f"{dataset.audit['raw_rows']} raw rows → {dataset.audit['retained_rows']} unique rows; "
        f"{dataset.audit['removed_duplicates']} repeated rows removed in "
        f"{len(dataset.audit['duplicate_groups'])} duplicate groups. "
        "Exact parsed feature equality, signed-zero normalization and earliest-line retention were "
        "applied before splitting. Every source line is preserved in [lineage.csv](lineage.csv). "
        "Full-population ranges are provenance diagnostics and did not select models.",
        _table(
            ["Partition", "Rows", "class_0", "class_1"],
            [
                [
                    role,
                    split.metadata["counts"][role],
                    split.metadata["class_counts"][role]["0"],
                    split.metadata["class_counts"][role]["1"],
                ]
                for role in ("development", "test")
            ],
        ),
        "Stratified 80/20 split, seed 42; development CV: five shuffled stratified folds, seed 43 "
        "in the canonical protocol. The actual settings and membership are frozen in "
        "[config.resolved.yaml](config.resolved.yaml), [split.json](split.json) and "
        "[split.csv](split.csv). Scalers fit separately on each training fold. "
        "Final fits use development only.",
        "## Development model comparison",
        f"Selected `{selection['chosen']['candidate_id']}` with parameters "
        f"`{selection['chosen']['parameters']}`. "
        "Ranking uses unrounded balanced accuracy and macro F1, "
        "then logistic regression/SVM/forest order and declared grid order. "
        "Selection CV is development evidence, not an unbiased generalization estimate.",
        _table(
            ["Candidate", "Mean accuracy", "Mean balanced accuracy", "Mean macro F1"],
            [
                [
                    r["candidate_id"],
                    f"{r['mean_accuracy']:.6f}",
                    f"{r['mean_balanced_accuracy']:.6f}",
                    f"{r['mean_f1_macro']:.6f}",
                ]
                for r in selection["ranking"]
            ],
        ),
        f"Ineligible candidates: {selection['ineligible_candidates'] or 'none'}. "
        "[Every candidate/fold, failures and warnings](cv_results.csv); "
        "[out-of-fold predictions](cv_predictions.csv); [frozen selection](selection.json).",
        "## Final winner and majority baseline",
        _table(
            ["Metric", "Winner", "Majority baseline"],
            [
                [name, winner[name], baseline[name]]
                for name in (
                    "correct",
                    "total",
                    "accuracy",
                    "balanced_accuracy",
                    "f1_macro",
                    "roc_auc",
                    "average_precision",
                    "accuracy_wilson_95",
                )
            ],
        ),
        f"Accuracy gain: {metrics['accuracy_gain_percentage_points']:.6f} percentage points. "
        "The 95% Wilson interval assumes independent binomial outcomes. Its lower bound is not "
        "an extra gate. The interval does not address dataset shift or unknown grouping.",
        _table(
            ["Model", "Class", "Precision", "Recall", "F1", "Support"],
            [
                [
                    name,
                    f"class_{label}",
                    *(values[key] for key in ("precision", "recall", "f1", "support")),
                ]
                for name, record in (("winner", winner), ("baseline", baseline))
                for label, values in record["per_class"].items()
            ],
        ),
    ]
    for name, record in (("Winner", winner), ("Baseline", baseline)):
        sections += [
            f"### {name} confusion matrices (rows true, columns predicted)",
            _table(
                ["True", "Predicted 0", "Predicted 1"],
                [[label, *record["confusion_matrix"][label]] for label in (0, 1)],
            ),
            "Row normalized:",
            _table(
                ["True", "Predicted 0", "Predicted 1"],
                [[label, *record["confusion_matrix_normalized"][label]] for label in (0, 1)],
            ),
            f"Diagnostics: {record['diagnostics'] or 'none'}.",
        ]
    config = manifest["protocol"]["config"]["evaluation"]
    requirements = {
        "accuracy": config["minimum_accuracy"],
        "balanced_accuracy": config["minimum_balanced_accuracy"],
        "class_0_recall": config["minimum_class_recall"],
        "class_1_recall": config["minimum_class_recall"],
        "accuracy_gain": config["minimum_accuracy_gain"],
    }
    errors = predictions[predictions["true_class"] != predictions["predicted_class"]]
    sections += [
        "### Acceptance gates",
        _table(
            ["Gate", "Minimum (unrounded)", "Passed"],
            [[gate, requirements[gate], passed] for gate, passed in metrics["gates"].items()],
        ),
        f"Fresh confirmatory evaluation: {metrics['fresh_confirmatory_evaluation']}. "
        f"Status reasons: {metrics['reasons'] or 'none'}. Audit checks: {metrics['audit_checks']}.",
        "## Individual test errors",
        _table(
            ["Sample ID", "Source lines", "True", "Predicted", "Score"],
            [
                [
                    key,
                    row["source_lines"],
                    row["true_class"],
                    row["predicted_class"],
                    row["score_class_1"],
                ]
                for key, row in errors.iterrows()
            ],
        )
        if len(errors)
        else "No test classification errors.",
        "These outcomes are retained for diagnosis; they have not been used to retune this run. "
        "All individual predictions: [test_predictions.csv](test_predictions.csv).",
        "## Reproduction and limitations",
        f"Protocol: `{manifest['protocol_id']}`. Revision: `{manifest['code']['revision']}`; "
        f"dirty tree: {manifest['code']['worktree_dirty']}; relevant source hash: "
        f"`{manifest['code']['source_sha256']}`. "
        f"Training time: {manifest['training_seconds']:.3f} seconds. "
        f"Environment: `{manifest['environment']}`. Machine: `{manifest['machine']}`.",
        "The full [manifest](manifest.json) binds source, configuration, split, model and artifact "
        "hashes. Restore [environment.txt](environment.txt) and recorded source for replay. "
        "Inference uses the saved development-fitted model; no all-data refit is performed. "
        "Floating scores may be compared at absolute tolerance 1e-12 in the pinned environment; "
        "timings and serialized bytes are not reproducibility criteria. "
        "Only load trusted local pickle bundles. Cross-version loading is rejected.",
        "Deduplication cannot establish independence of near-identical observations. Specimen, "
        "camera, session and time identifiers are unavailable. Results describe this benchmark "
        "population; they do not generalize to new cameras, counterfeits or investment returns. "
        "SVM scores are decision margins, never probabilities; no calibration quality is claimed. "
        "The local ledger cannot detect history in other directories or manual/external use. "
        f"External/manual disclosure: {manifest['exposure_notes']}",
        "## Next project",
        "[Queued next-period high/normal-volatility brief]"
        "(../../../../specs/spec002/02_VOLATILITY_FOLLOW_ON.md). "
        "Use purged chronological partitions and realistic persistence baselines for that project.",
    ]
    atomic_write(run_dir / "report.md", ("\n\n".join(sections) + "\n").encode())

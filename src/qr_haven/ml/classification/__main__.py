"""CLI orchestration; training, evaluation and prediction use the public API."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

from qr_haven.data.banknotes import atomic_write
from qr_haven.ml.classification import (
    ClassificationConfig,
    evaluate_banknote_run,
    fetch_banknote_dataset,
    load_banknote_dataset,
    load_classifier,
    prepare_banknote_split,
    train_banknote_classifier,
)
from qr_haven.ml.classification.artifacts import read_json, verify_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible UCI banknote classification")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser(
        "fetch-banknotes", help="Explicitly download or verify the official cache"
    )
    fetch.add_argument("--cache-dir", type=Path, default=Path("data/raw/banknote_authentication"))
    fetch.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path("configs/datasets/banknote_authentication.json"),
    )
    train = sub.add_parser("train", help="Freeze the split and select using development data only")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--data", type=Path, help="Use an already downloaded local raw file")
    evaluate = sub.add_parser("evaluate", help="Evaluate only the frozen winner and baseline")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    predict = sub.add_parser("predict", help="Predict named four-feature CSV rows with no fitting")
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-banknotes":
            reference = read_json(args.reference_manifest)
            print(fetch_banknote_dataset(args.cache_dir, reference_sha256=reference["raw_sha256"]))
        elif args.command == "train":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
                raise ValueError("run-id must be a single safe name, with no path separators")
            config = ClassificationConfig.from_yaml(args.config)
            if args.data:
                config = config.model_copy(
                    update={"dataset": config.dataset.model_copy(update={"path": args.data})}
                )
            reference = read_json(config.dataset.reference_manifest)
            dataset = load_banknote_dataset(
                config.dataset.path,
                reference_sha256=reference["raw_sha256"],
                expected_raw_rows=config.dataset.expected_raw_rows,
            )
            split = prepare_banknote_split(dataset, config)
            run_dir = config.artifacts.root / args.run_id
            trained = train_banknote_classifier(dataset, split, config, run_dir)
            chosen = trained.selection["chosen"]["candidate_id"]
            print(f"Trained {chosen} using development data only.")
            print(f"Next: python -m qr_haven.ml.classification evaluate --run-dir {run_dir}")
        elif args.command == "evaluate":
            manifest = verify_run(args.run_dir)
            dataset = load_banknote_dataset(
                Path(manifest["source"]["path"]),
                reference_sha256=manifest["source"]["raw_sha256"],
                canonical=manifest["source"]["canonical"],
                expected_raw_rows=manifest["protocol"]["config"]["dataset"]["expected_raw_rows"],
            )
            result = evaluate_banknote_run(args.run_dir, dataset)
            print(
                json.dumps(
                    {
                        "benchmark_status": result.metrics["benchmark_status"],
                        "accuracy": result.metrics["winner"]["accuracy"],
                        "reasons": result.metrics["reasons"],
                        "report": str(result.report_path),
                    }
                )
            )
            return 0 if result.metrics["benchmark_status"] == "passed" else 2
        else:
            if args.output.exists():
                raise ValueError(f"Refusing to overwrite predictions: {args.output}")
            # pandas mangles duplicate headers; reject them before it can hide the error.
            with args.input.open(newline="") as stream:
                header = next(csv.reader(stream), [])
            if len(header) != len(set(header)):
                raise ValueError("Duplicate feature columns in input CSV")
            predictions = load_classifier(args.run_dir).predict(pd.read_csv(args.input))
            atomic_write(args.output, predictions.to_csv(index=False).encode())
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "rows": len(predictions),
                        "diagnostics": predictions.attrs,
                    }
                )
            )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

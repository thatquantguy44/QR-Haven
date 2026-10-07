"""CLI for immutable V2 volatility observations and chronological split artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from qr_haven.ml.volatility import (
    PROFILES,
    evaluate_volatility_run,
    evaluation_paths,
    get_profile,
    load_profile_dataset,
    prepare_walk_forward_plan,
    train_volatility_development,
    verify_v2_artifacts,
    verify_v3_run,
    verify_v4_evaluation,
    write_v2_artifacts,
)
from qr_haven.ml.volatility.challenger_data import (
    challenger_design_from_manifest,
    prepare_challenger_dataset,
    verify_challenger_dataset,
)
from qr_haven.ml.volatility.challenger_evaluation import (
    evaluate_challenger,
    verify_challenger_evaluation,
)
from qr_haven.ml.volatility.challenger_training import (
    train_challenger,
    verify_challenger_run,
)
from qr_haven.ml.volatility.experiments import (
    run_volatility_experiments,
    verify_volatility_experiments,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="Build and freeze V2 point-in-time data")
    prepare.add_argument("--profile", choices=sorted(PROFILES), required=True)
    prepare.add_argument("--output-dir", type=Path)
    verify = subcommands.add_parser("verify", help="Verify frozen V2 artifact hashes")
    verify.add_argument("--output-dir", type=Path, required=True)
    train = subcommands.add_parser("train", help="Run V3 development-only model comparison")
    train.add_argument("--profile", choices=sorted(PROFILES), required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--v2-dir", type=Path)
    verify_run = subcommands.add_parser("verify-run", help="Verify frozen V3 artifact hashes")
    verify_run.add_argument("--run-dir", type=Path, required=True)
    evaluate = subcommands.add_parser("evaluate", help="Run the single V4 holdout evaluation")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--v2-dir", type=Path)
    evaluate.add_argument("--evaluation-id", default="holdout-v1")
    verify_evaluation = subcommands.add_parser(
        "verify-evaluation", help="Verify frozen V4 evaluation hashes"
    )
    verify_evaluation.add_argument("--output-dir", type=Path, required=True)
    experiment = subcommands.add_parser(
        "experiment", help="Run four exploratory experiments using development folds"
    )
    experiment.add_argument("--run-dir", type=Path, required=True)
    experiment.add_argument("--v2-dir", type=Path)
    experiment.add_argument(
        "--evaluation-dir", type=Path, help="Completed V4 evaluation for saved-error diagnosis only"
    )
    experiment.add_argument("--experiment-id", default="improvements-v1")
    verify_experiment = subcommands.add_parser(
        "verify-experiment", help="Verify immutable exploratory output hashes"
    )
    verify_experiment.add_argument("--output-dir", type=Path, required=True)
    challenger_prepare = subcommands.add_parser(
        "challenger-prepare", help="Build the frozen V5 extended-feature dataset"
    )
    challenger_prepare.add_argument("--output-dir", type=Path)
    challenger_prepare.add_argument(
        "--profile", choices=("spx-local-v1", "tiingo-spy-v1"), default="spx-local-v1"
    )
    challenger_verify_data = subcommands.add_parser(
        "challenger-verify-data", help="Verify frozen V5 dataset hashes"
    )
    challenger_verify_data.add_argument("--output-dir", type=Path, required=True)
    challenger_train = subcommands.add_parser(
        "challenger-train", help="Run nested V5 development selection and final fit"
    )
    challenger_train.add_argument("--dataset-dir", type=Path)
    challenger_train.add_argument("--run-id", default="challenger-v1")
    challenger_train.add_argument(
        "--profile", choices=("spx-local-v1", "tiingo-spy-v1"), default="spx-local-v1"
    )
    challenger_verify_run = subcommands.add_parser(
        "challenger-verify-run", help="Verify frozen V5 training artifacts"
    )
    challenger_verify_run.add_argument("--run-dir", type=Path, required=True)
    challenger_evaluate = subcommands.add_parser(
        "challenger-evaluate", help="Run a single frozen challenger evaluation"
    )
    challenger_evaluate.add_argument("--run-dir", type=Path, required=True)
    challenger_evaluate.add_argument("--dataset-dir", type=Path)
    challenger_evaluate.add_argument("--evaluation-id")
    challenger_verify_evaluation = subcommands.add_parser(
        "challenger-verify-evaluation", help="Verify frozen V5 evaluation artifacts"
    )
    challenger_verify_evaluation.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_v2_artifacts(args.output_dir)
        elif args.command == "verify-run":
            result = verify_v3_run(args.run_dir)
        elif args.command == "verify-evaluation":
            result = verify_v4_evaluation(args.output_dir)
        elif args.command == "verify-experiment":
            result = verify_volatility_experiments(args.output_dir)
        elif args.command == "challenger-verify-data":
            result = verify_challenger_dataset(args.output_dir)
        elif args.command == "challenger-verify-run":
            result = verify_challenger_run(args.run_dir)
        elif args.command == "challenger-verify-evaluation":
            result = verify_challenger_evaluation(args.output_dir)
        elif args.command == "challenger-prepare":
            root = Path("artifacts/classification/volatility", args.profile, "challengers")
            output = args.output_dir or root / "dataset-v1"
            manifest = prepare_challenger_dataset(output, profile_id=args.profile)
            result = {
                "output_dir": str(output),
                "development_rows": manifest["development_rows"],
                "evaluation_rows": manifest["evaluation_rows"],
                "evaluation_outcomes_summarized": False,
            }
        elif args.command == "challenger-train":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
                raise ValueError("run-id must be a single safe name")
            root = Path("artifacts/classification/volatility", args.profile, "challengers")
            dataset = args.dataset_dir or root / "dataset-v1"
            output = root / args.run_id
            manifest = train_challenger(
                dataset,
                output,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            result = {
                "run_dir": str(output),
                "selected_candidate": manifest["selected_candidate"],
                "selected_model_weight": manifest["selected_model_weight"],
                "evaluation_outcomes_opened": False,
                "report": str(output / "development_report.md"),
            }
        elif args.command == "challenger-evaluate":
            run = args.run_dir.resolve()
            root = run.parent
            frozen = verify_challenger_run(run)
            design = challenger_design_from_manifest(frozen)
            dataset = args.dataset_dir or root / "dataset-v1"
            evaluation_id = args.evaluation_id or design.evaluation_id
            output = root / "evaluations" / run.name / evaluation_id
            manifest = evaluate_challenger(run, dataset, output, evaluation_id=evaluation_id)
            result = {
                "output_dir": str(output),
                "research_status": manifest["research_status"],
                "evaluation_outcomes_opened": True,
                "report": str(output / "report.md"),
            }
        elif args.command == "experiment":
            run_dir = args.run_dir.resolve()
            frozen = verify_v3_run(run_dir)
            if run_dir.parent.name != frozen["profile_id"]:
                raise ValueError("V3 run must be under its recorded profile directory")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.experiment_id):
                raise ValueError("experiment-id must be a single safe name")
            output = run_dir.parent / "experiments" / run_dir.name / args.experiment_id
            manifest = run_volatility_experiments(
                run_dir,
                args.v2_dir or run_dir.parent / "dataset-v1",
                output,
                evaluation_dir=args.evaluation_dir,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            result = {
                "output_dir": str(output),
                "report": str(output / "report.md"),
                "research_status": manifest["research_status"],
                "holdout_diagnosed": manifest["holdout_diagnosed"],
                "new_holdout_predictions": False,
            }
        elif args.command == "evaluate":
            run_dir = args.run_dir
            run_manifest = verify_v3_run(run_dir)
            profile_id = str(run_manifest["profile_id"])
            if run_dir.parent.name != profile_id:
                raise ValueError("V3 run directory is not under its recorded profile directory")
            v2_dir = args.v2_dir or run_dir.parent / "dataset-v1"
            paths = evaluation_paths(
                profile_id,
                run_dir.name,
                args.evaluation_id,
                artifact_root=run_dir.parent.parent,
            )
            evaluated = evaluate_volatility_run(
                run_dir,
                v2_dir,
                paths.output_dir,
                evaluation_id=args.evaluation_id,
            )
            result = {
                "research_status": evaluated.metrics["research_status"],
                "output_dir": str(evaluated.output_dir),
                "report": str(paths.report),
                "holdout_rows": evaluated.metrics["holdout_rows"],
                "holdout_evaluated": True,
            }
        elif args.command == "train":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
                raise ValueError("run-id must be a single safe name, with no path separators")
            profile = get_profile(args.profile)
            v2_dir = args.v2_dir or Path(
                "artifacts/classification/volatility", profile.profile_id, "dataset-v1"
            )
            run_dir = Path("artifacts/classification/volatility", profile.profile_id, args.run_id)
            trained = train_volatility_development(
                v2_dir,
                run_dir,
                expected_profile=profile.profile_id,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            result = {
                "run_dir": str(run_dir),
                "selected_candidate": trained.selected.candidate_id,
                "holdout_evaluated": False,
                "report": str(run_dir / "development_report.md"),
            }
        else:
            profile = get_profile(args.profile)
            output_dir = args.output_dir or Path(
                "artifacts/classification/volatility", profile.profile_id, "dataset-v1"
            )
            dataset = load_profile_dataset(profile)
            plan = prepare_walk_forward_plan(dataset)
            result = write_v2_artifacts(output_dir, dataset, plan)
            result = {**result, "output_dir": str(output_dir)}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

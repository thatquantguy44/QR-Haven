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
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_v2_artifacts(args.output_dir)
        elif args.command == "verify-run":
            result = verify_v3_run(args.run_dir)
        elif args.command == "verify-evaluation":
            result = verify_v4_evaluation(args.output_dir)
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
            run_dir = Path(
                "artifacts/classification/volatility", profile.profile_id, args.run_id
            )
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

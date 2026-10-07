"""CLI for immutable V2 volatility observations and chronological split artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qr_haven.ml.volatility import (
    PROFILES,
    get_profile,
    load_profile_dataset,
    prepare_walk_forward_plan,
    verify_v2_artifacts,
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
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_v2_artifacts(args.output_dir)
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

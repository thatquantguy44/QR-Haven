"""Validate the supplied SPX CSV and prepare an ignored canonical snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from qr_haven.data.spx import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_STUDY_END,
    DEFAULT_STUDY_START,
    load_prepared_local_spx_snapshot,
    prepare_local_spx_snapshot,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_STUDY_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_STUDY_END)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing prepared snapshot without reading the source CSV.",
    )
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            snapshot = load_prepared_local_spx_snapshot(args.output_dir)
        else:
            snapshot = prepare_local_spx_snapshot(
                args.input,
                args.output_dir,
                study_start=args.start,
                study_end=args.end,
            )
        print(
            json.dumps(
                {
                    "snapshot": str(args.output_dir.resolve()),
                    "study_rows": snapshot.audit["study_rows"],
                    "first_session": snapshot.audit["study_first_session"],
                    "last_session": snapshot.audit["study_last_session"],
                    "source_sha256": snapshot.manifest["source_sha256"],
                    "price_basis": snapshot.manifest["price_basis"],
                }
            )
        )
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

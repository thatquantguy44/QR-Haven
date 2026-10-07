"""Explicitly fetch the licensed SPY 2005–2025 snapshot from Tiingo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from qr_haven.data.tiingo import fetch_tiingo_spy_snapshot, load_tiingo_spy_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/market_data/tiingo/spy-2005-2025-v1"),
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2005, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing snapshot without a token or network access.",
    )
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            snapshot = load_tiingo_spy_snapshot(args.output_dir)
        else:
            token = os.environ.get("TIINGO_API_TOKEN", "")
            snapshot = fetch_tiingo_spy_snapshot(
                args.output_dir, token=token, start=args.start, end=args.end
            )
        print(
            json.dumps(
                {
                    "snapshot": str(args.output_dir.resolve()),
                    "rows": snapshot.audit["rows"],
                    "first_session": snapshot.audit["first_session"],
                    "last_session": snapshot.audit["last_session"],
                    "raw_sha256": snapshot.manifest["raw_price_sha256"],
                    "calendar": snapshot.manifest["calendar"],
                    "redistribution_permitted": snapshot.manifest["redistribution_permitted"],
                }
            )
        )
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

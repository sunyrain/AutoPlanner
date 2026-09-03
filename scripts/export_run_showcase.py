"""Export one saved retrosynthesis run as a self-contained playback page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.web.v4_showcase_export import (  # noqa: E402
    export_run_showcase,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile model-io and Host replay events into one offline HTML page."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Saved run directory containing target-only-solve-report.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .html file.",
    )
    parser.add_argument(
        "--kind",
        choices=("interaction", "route", "graph"),
        default="interaction",
        help="Export model playback, one static Strategy route, or the complete graph.",
    )
    parser.add_argument(
        "--branch",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Strategy branch used by --kind route.",
    )
    args = parser.parse_args()
    receipt = export_run_showcase(
        run_dir=args.run_dir,
        output_path=args.output,
        export_kind=args.kind,
        branch_index=args.branch,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

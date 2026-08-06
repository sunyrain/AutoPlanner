#!/usr/bin/env python3
"""Run the frozen combined V3/V4 Web application explicitly."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.legacy.web import serve_combined_web


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the frozen combined V3/V4 compatibility UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--server",
        choices=("auto", "waitress", "flask"),
        default="auto",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    serve_combined_web(
        host=args.host,
        port=args.port,
        server=args.server,
        threads=args.threads,
        debug=args.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

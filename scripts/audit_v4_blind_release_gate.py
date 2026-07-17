#!/usr/bin/env python3
"""Compile the P9 release gate from one baseline and optional ablation panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.eval.v4_blind_release_gate import (  # noqa: E402
    compile_v4_blind_release_gate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ablation", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = compile_v4_blind_release_gate(
        args.baseline,
        ablation_status_paths=args.ablation,
        repository_root=ROOT,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "target_count": report["target_count"],
                "output_dir": str(Path(args.output_dir).resolve()),
                "failed_gates": [
                    key
                    for key, value in dict(report.get("gates") or {}).items()
                    if not dict(value).get("passed")
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

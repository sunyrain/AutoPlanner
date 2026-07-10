#!/usr/bin/env python3
"""Migrate a saved pre-outbox Codex campaign to fenced V2 expansion commits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.orchestration.codex_retrosynthesis import (  # noqa: E402
    migrate_legacy_campaign_commits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--target-smiles", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = migrate_legacy_campaign_commits(
        case_id=args.case_id,
        target_smiles=args.target_smiles,
        run_dir=args.run_dir,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("accepted") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())

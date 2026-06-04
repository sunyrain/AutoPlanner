#!/usr/bin/env python3
"""Run the P0 SMILES-first literature strategic workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.agent.smiles_first import (
    SmilesFirstWorkflowConfig,
    run_smiles_first_workflow,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-smiles", required=True)
    parser.add_argument("--target-name", default="")
    parser.add_argument("--family-hint", default="")
    parser.add_argument("--objective", default="route")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frontier-smiles", default="")
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--evidence-jsonl", default=None)
    parser.add_argument("--db", action="append", default=None, help="Strategic disconnection DB JSON. May be repeated.")
    parser.add_argument("--query-budget", type=int, default=12)
    args = parser.parse_args()

    result = run_smiles_first_workflow(
        SmilesFirstWorkflowConfig(
            target_smiles=args.target_smiles,
            target_name=args.target_name,
            family_hint=args.family_hint,
            objective=args.objective,
            output_dir=Path(args.output_dir),
            frontier_smiles=args.frontier_smiles,
            baseline_json=args.baseline_json,
            evidence_jsonl=args.evidence_jsonl,
            db_paths=args.db,
            query_budget=args.query_budget,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

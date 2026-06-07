#!/usr/bin/env python3
"""Run the AutoPlanner Codex-entry route controller harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.runner import run_codex_entry_controller
from cascade_planner.harness.tools import HarnessBudget


DEFAULT_KEY_PATH = ROOT / "key.txt"
DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-smiles", required=True)
    parser.add_argument("--family-hint", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--chem-enzy-timeout-s", type=float, default=None)
    parser.add_argument("--guided-chemenzy-timeout-s", type=float, default=None)
    parser.add_argument("--open-research-timeout-s", type=float, default=None)
    parser.add_argument("--smiles-first-timeout-s", type=float, default=None)
    parser.add_argument("--max-route-expansion-subgoal-runs", type=int, default=None)
    parser.add_argument("--key-path", default=str(DEFAULT_KEY_PATH))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--offline-planner",
        action="store_true",
        help="Use the deterministic local planner instead of live Codex. Intended for CI/debug only.",
    )
    args = parser.parse_args()
    budget = HarnessBudget(timeout_s=float(args.timeout_s))
    if args.chem_enzy_timeout_s is not None:
        budget.chem_enzy_timeout_s = float(args.chem_enzy_timeout_s)
    if args.guided_chemenzy_timeout_s is not None:
        budget.guided_chemenzy_timeout_s = float(args.guided_chemenzy_timeout_s)
    if args.open_research_timeout_s is not None:
        budget.open_research_timeout_s = float(args.open_research_timeout_s)
    if args.smiles_first_timeout_s is not None:
        budget.smiles_first_timeout_s = float(args.smiles_first_timeout_s)
    if args.max_route_expansion_subgoal_runs is not None:
        budget.max_route_expansion_subgoal_runs = int(args.max_route_expansion_subgoal_runs)

    result = run_codex_entry_controller(
        target_name=args.target_name,
        target_smiles=args.target_smiles,
        family_hint=args.family_hint,
        output_dir=args.output_dir,
        timeout_s=float(args.timeout_s),
        key_path=args.key_path,
        base_url=args.base_url,
        model=args.model,
        use_live_planner=not bool(args.offline_planner),
        budget=budget,
    )
    print(json.dumps({
        "schema_version": "codex_entry_controller_cli_result.v1",
        "run_dir": result["run_dir"],
        "final_verdict": result["final_verdict"],
        "artifacts": result["artifacts"],
    }, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

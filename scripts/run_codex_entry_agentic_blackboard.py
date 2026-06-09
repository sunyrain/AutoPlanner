#!/usr/bin/env python3
"""Run the agentic blackboard Codex-entry controller."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.agent.target_profile import build_target_profile
from cascade_planner.harness.agentic_blackboard_controller import run_agentic_blackboard_controller
from cascade_planner.harness.tools import HarnessBudget


DEFAULT_OUTPUT_ROOT = ROOT / "results" / "shared"
DEFAULT_KEY_PATH = ROOT / "key.txt"
DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", nargs="*", help="One or more target SMILES.")
    parser.add_argument("--target-name", default="")
    parser.add_argument("--target-smiles", default="")
    parser.add_argument("--family-hint", default="")
    parser.add_argument("--literature-pdf-path", default="")
    parser.add_argument("--literature-pdf-source-ref", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-prefix", default="agentic_blackboard")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--guided-chemenzy-timeout-s", type=float, default=None)
    parser.add_argument("--max-route-expansion-subgoal-runs", type=int, default=None)
    parser.add_argument("--key-path", default=str(DEFAULT_KEY_PATH))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    targets = _resolve_targets(args)
    results = [_run_one(target, args) for target in targets]
    print(json.dumps(_cli_result(results), indent=2, ensure_ascii=False, sort_keys=True))


def _resolve_targets(args: argparse.Namespace) -> list[dict[str, str | Path]]:
    smiles_values = [str(item).strip() for item in args.smiles if str(item).strip()]
    if str(args.target_smiles or "").strip():
        smiles_values.append(str(args.target_smiles).strip())
    if not smiles_values:
        raise SystemExit("Provide at least one SMILES, either positional or with --target-smiles.")
    if str(args.target_name or "").strip() and len(smiles_values) != 1:
        raise SystemExit("--target-name can only be used with exactly one target SMILES.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser() if str(args.output_dir or "").strip() else None
    output_root = Path(args.output_root).expanduser()
    rows: list[dict[str, str | Path]] = []
    for idx, smiles in enumerate(smiles_values, start=1):
        name = str(args.target_name or "").strip() or _case_slug(smiles, idx=idx, total=len(smiles_values))
        slug = _safe_path_part(name)
        run_dir = output_dir if output_dir is not None and len(smiles_values) == 1 else (
            (output_dir or output_root) / f"{_safe_path_part(args.run_prefix) or 'agentic_blackboard'}_{slug}_{timestamp}"
        )
        rows.append({"target_name": name, "target_smiles": smiles, "output_dir": run_dir})
    return rows


def _run_one(target: dict[str, str | Path], args: argparse.Namespace) -> dict:
    return run_agentic_blackboard_controller(
        target_name=str(target["target_name"]),
        target_smiles=str(target["target_smiles"]),
        family_hint=str(args.family_hint or ""),
        output_dir=Path(target["output_dir"]),
        literature_pdf_path=args.literature_pdf_path,
        literature_pdf_source_ref=args.literature_pdf_source_ref,
        timeout_s=float(args.timeout_s),
        key_path=args.key_path,
        base_url=args.base_url,
        model=args.model,
        max_rounds=int(args.max_rounds or 3),
        budget=_budget_from_args(args),
    )


def _budget_from_args(args: argparse.Namespace) -> HarnessBudget:
    budget = HarnessBudget(timeout_s=float(args.timeout_s))
    if args.guided_chemenzy_timeout_s is not None:
        budget.guided_chemenzy_timeout_s = float(args.guided_chemenzy_timeout_s)
    if args.max_route_expansion_subgoal_runs is not None:
        budget.max_route_expansion_subgoal_runs = int(args.max_route_expansion_subgoal_runs)
    return budget


def _case_slug(smiles: str, *, idx: int, total: int) -> str:
    profile = build_target_profile(smiles)
    slug = profile.case_id or f"target_{idx}"
    return slug if total == 1 else f"{slug}_{idx:02d}"


def _safe_path_part(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    return "_".join(part for part in safe.split("_") if part)


def _cli_result(results: list[dict]) -> dict:
    if len(results) == 1:
        result = results[0]
        return {
            "schema_version": "agentic_blackboard_controller_cli_result.v1",
            "run_dir": result["run_dir"],
            "final_verdict": result["final_verdict"],
            "artifacts": result["artifacts"],
        }
    return {
        "schema_version": "agentic_blackboard_controller_cli_batch_result.v1",
        "run_count": len(results),
        "runs": [
            {
                "run_dir": result["run_dir"],
                "target_name": result["target_input"]["target_name"],
                "final_verdict": result["final_verdict"],
                "artifacts": result["artifacts"],
            }
            for result in results
        ],
    }


if __name__ == "__main__":
    main()

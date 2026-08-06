#!/usr/bin/env python3
"""Run the AutoPlanner Codex-entry route controller harness."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.agent.target_profile import build_target_profile
from cascade_planner.legacy.harness_runtime.runner import run_codex_entry_controller
from cascade_planner.legacy.harness_runtime.tools import HarnessBudget


DEFAULT_KEY_PATH = ROOT / "key.txt"
DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "shared"


@dataclass(frozen=True)
class CliTarget:
    target_name: str
    target_smiles: str
    output_dir: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "smiles",
        nargs="*",
        help="One or more target SMILES. When supplied, each target gets its own run_dir.",
    )
    parser.add_argument("--target-name", default="")
    parser.add_argument("--target-smiles", default="")
    parser.add_argument("--family-hint", default="")
    parser.add_argument(
        "--literature-pdf-path",
        default="",
        help="Local literature PDF to render and pass into PDF/visual source-detail extraction tools.",
    )
    parser.add_argument(
        "--literature-pdf-source-ref",
        default="",
        help="Source reference for --literature-pdf-path, for example doi:10.3184/0308234054213519.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Run directory for one target. For multiple positional SMILES, this is treated "
            "as the parent directory for per-target run dirs."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Parent directory for automatically named run dirs when --output-dir is omitted.",
    )
    parser.add_argument(
        "--run-prefix",
        default="codex_entry",
        help="Prefix used for automatically named run dirs.",
    )
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
    targets = _resolve_cli_targets(args)
    results = [_run_one(target, args) for target in targets]
    print(json.dumps(_cli_result(results), indent=2, ensure_ascii=False, sort_keys=True))


def _resolve_cli_targets(args: argparse.Namespace) -> list[CliTarget]:
    smiles_values = [str(item).strip() for item in args.smiles if str(item).strip()]
    legacy_smiles = str(args.target_smiles or "").strip()
    if legacy_smiles:
        smiles_values.append(legacy_smiles)
    if not smiles_values:
        raise SystemExit("Provide at least one SMILES, either positional or with --target-smiles.")
    if str(args.target_name or "").strip() and len(smiles_values) != 1:
        raise SystemExit("--target-name can only be used with exactly one target SMILES.")

    timestamp = _timestamp_utc()
    output_dir = Path(args.output_dir).expanduser() if str(args.output_dir or "").strip() else None
    output_root = Path(args.output_root).expanduser()
    duplicate_counts: dict[str, int] = {}
    targets: list[CliTarget] = []
    for index, smiles in enumerate(smiles_values, start=1):
        target_name = _target_name_for_smiles(
            smiles=smiles,
            explicit_name=str(args.target_name or "").strip(),
            index=index,
            total=len(smiles_values),
        )
        slug = _case_slug(target_name=target_name, smiles=smiles)
        duplicate_counts[slug] = duplicate_counts.get(slug, 0) + 1
        if duplicate_counts[slug] > 1:
            slug = f"{slug}_{duplicate_counts[slug]}"
        targets.append(
            CliTarget(
                target_name=target_name,
                target_smiles=smiles,
                output_dir=_output_dir_for_target(
                    output_dir=output_dir,
                    output_root=output_root,
                    run_prefix=str(args.run_prefix or "codex_entry"),
                    slug=slug,
                    timestamp=timestamp,
                    total=len(smiles_values),
                ),
            )
        )
    return targets


def _target_name_for_smiles(*, smiles: str, explicit_name: str, index: int, total: int) -> str:
    if explicit_name:
        return explicit_name
    slug = _case_slug(target_name="", smiles=smiles)
    if total == 1:
        return slug
    return f"{slug}_{index:02d}"


def _case_slug(*, target_name: str, smiles: str) -> str:
    profile = build_target_profile(smiles, target_name=target_name)
    return profile.case_id or "target"


def _output_dir_for_target(
    *,
    output_dir: Path | None,
    output_root: Path,
    run_prefix: str,
    slug: str,
    timestamp: str,
    total: int,
) -> Path:
    if output_dir is not None and total == 1:
        return output_dir
    parent = output_dir if output_dir is not None else output_root
    prefix = _safe_path_part(run_prefix) or "codex_entry"
    return parent / f"{prefix}_{slug}_{timestamp}"


def _safe_path_part(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    return "_".join(part for part in safe.split("_") if part)


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _budget_from_args(args: argparse.Namespace) -> HarnessBudget:
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
    return budget


def _run_one(target: CliTarget, args: argparse.Namespace) -> dict:
    result = run_codex_entry_controller(
        target_name=target.target_name,
        target_smiles=target.target_smiles,
        family_hint=args.family_hint,
        output_dir=target.output_dir,
        literature_pdf_path=args.literature_pdf_path,
        literature_pdf_source_ref=args.literature_pdf_source_ref,
        timeout_s=float(args.timeout_s),
        key_path=args.key_path,
        base_url=args.base_url,
        model=args.model,
        use_live_planner=not bool(args.offline_planner),
        budget=_budget_from_args(args),
    )
    return result


def _cli_result(results: list[dict]) -> dict:
    if len(results) == 1:
        result = results[0]
        return {
            "schema_version": "codex_entry_controller_cli_result.v1",
            "run_dir": result["run_dir"],
            "final_verdict": result["final_verdict"],
            "artifacts": result["artifacts"],
        }
    return {
        "schema_version": "codex_entry_controller_cli_batch_result.v1",
        "run_count": len(results),
        "runs": [
            {
                "run_dir": result["run_dir"],
                "final_verdict": result["final_verdict"],
                "artifacts": result["artifacts"],
            }
            for result in results
        ],
    }


if __name__ == "__main__":
    main()

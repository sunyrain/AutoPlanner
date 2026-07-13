"""CLI surface for genuine target-only retrosynthesis campaigns."""
from __future__ import annotations

import argparse
from typing import Any

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.interfaces.target_solver import TargetSolveConfig


TARGET_COMMANDS = frozenset({"solve-target", "import-evidence"})


def add_target_commands(sub: argparse._SubParsersAction) -> None:
    solve = sub.add_parser(
        "solve-target",
        help="run a fresh bounded Codex campaign from only an arbitrary target SMILES",
    )
    solve.add_argument("--target-smiles", required=True)
    solve.add_argument("--target-name", default="blind target")
    solve.add_argument("--run-id")
    solve.add_argument("--run-dir")
    solve.add_argument("--manifest", help="target-only manifest allowed by blind preflight")
    solve.add_argument("--resume", action="store_true")
    solve.add_argument(
        "--model",
        default="gpt-5.5",
        help="explicit Codex model; defaults to the strongest current CLI-compatible tier",
    )
    solve.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
    )
    solve.add_argument("--single-agent", action="store_true")
    solve.add_argument("--no-web-search", action="store_true")
    solve.add_argument("--no-replan", action="store_true")
    solve.add_argument("--no-live-benchmark-stock", action="store_true")
    solve.add_argument("--minimum-complete-routes", type=int, default=2)
    solve.add_argument("--minimum-edge-proof-level", type=int, default=2)
    solve.add_argument("--minimum-source-groups", type=int, default=2)
    solve.add_argument(
        "--stock-boundary",
        choices=("benchmark_search", "procurement", "in_house"),
        default="benchmark_search",
    )
    solve.add_argument("--max-model-invocations", type=int, default=2)
    solve.add_argument("--max-input-tokens", type=int, default=50_000)
    solve.add_argument("--max-output-tokens", type=int, default=14_000)
    solve.add_argument("--max-model-wall-time-s", type=float, default=720.0)
    solve.add_argument("--max-accepted-expansions", type=int, default=32)
    solve.add_argument("--max-attempt-runs", type=int, default=72)
    solve.add_argument("--max-map-reactions", type=int, default=48)
    solve.add_argument("--max-stock-molecules", type=int, default=24)

    evidence = sub.add_parser(
        "import-evidence",
        help="import trusted structured exact-source rows into an unresolved run",
    )
    evidence.add_argument("run_id")
    evidence.add_argument("--run-dir")
    evidence.add_argument("--input", required=True)


def dispatch_target_command(gateway: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "import-evidence":
        return gateway.import_evidence(
            run_id=args.run_id,
            run_dir=args.run_dir,
            import_path=args.input,
        )
    if args.command != "solve-target":
        raise ValueError(f"unsupported_target_command:{args.command}")
    return gateway.solve_target(
        target_name=args.target_name,
        target_smiles=args.target_smiles,
        run_id=args.run_id,
        run_dir=args.run_dir,
        manifest_path=args.manifest,
        resume=args.resume,
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=args.minimum_complete_routes,
            minimum_edge_proof_level=args.minimum_edge_proof_level,
            minimum_independent_source_groups=args.minimum_source_groups,
            stock_boundary=args.stock_boundary,
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=args.max_model_invocations,
            max_total_input_tokens=args.max_input_tokens,
            max_total_output_tokens=args.max_output_tokens,
            max_total_wall_time_s=args.max_model_wall_time_s,
            max_visual_invocations=0,
            max_accepted_expansions=args.max_accepted_expansions,
            max_attempt_runs=args.max_attempt_runs,
            max_prompt_context_bytes=96_000,
        ),
        config=TargetSolveConfig(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            use_coordinator=not args.single_agent,
            enable_web_search=not args.no_web_search,
            enable_replan=not args.no_replan,
            enable_live_benchmark_stock=not args.no_live_benchmark_stock,
            max_atom_mapping_reactions=args.max_map_reactions,
            max_live_stock_molecules=args.max_stock_molecules,
        ),
    )


__all__ = ["TARGET_COMMANDS", "add_target_commands", "dispatch_target_command"]

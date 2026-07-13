"""CLI surface for deterministic scientific case compilation and replay."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cascade_planner.runtime.paths import RuntimePaths


CASE_COMMANDS = frozenset(
    {"compile-case", "replay-case", "solve-case", "replay-dossier"}
)


def add_case_commands(sub: argparse._SubParsersAction) -> None:
    replay = sub.add_parser(
        "replay-case",
        help="rebuild a compact, model-free scientific acceptance pack",
    )
    replay.add_argument("--pack", type=Path, required=True)
    replay.add_argument("--run-id")
    replay.add_argument("--run-dir")
    replay.add_argument(
        "--stop-after",
        choices=("plan", "materialization", "evidence", "validation", "stock"),
        default="",
        help="pause after a stage to exercise deterministic recovery",
    )

    compile_case = sub.add_parser(
        "compile-case",
        help="compile a concise exact-source dossier into a replay pack",
    )
    compile_case.add_argument("--dossier", type=Path, required=True)
    compile_case.add_argument("--output", type=Path, required=True)
    compile_case.add_argument(
        "--map-missing",
        action="store_true",
        help="use the installed local RXNMapper for dossier steps without maps",
    )

    solve = sub.add_parser(
        "solve-case",
        help="compatibility alias: replay a supplied exact-source dossier (not blind solve)",
    )
    replay_dossier = sub.add_parser(
        "replay-dossier",
        help="compile, replay, close, and export one supplied exact-source dossier",
    )
    for parser in (solve, replay_dossier):
        parser.add_argument("--dossier", type=Path, required=True)
        parser.add_argument("--run-id")
        parser.add_argument("--run-dir")
        parser.add_argument("--output-dir")
        parser.add_argument(
            "--map-missing",
            action="store_true",
            help="use the installed local RXNMapper for dossier steps without maps",
        )


def dispatch_case_command(
    args: argparse.Namespace,
    *,
    paths: RuntimePaths,
) -> dict[str, Any]:
    if args.command in {"solve-case", "replay-dossier"}:
        from .case_runner import run_case_dossier

        return run_case_dossier(
            args.dossier,
            paths=paths,
            run_id=args.run_id,
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            map_missing=args.map_missing,
        )
    if args.command == "compile-case":
        from .case_dossier import write_compiled_replay_pack

        return write_compiled_replay_pack(
            args.dossier,
            args.output,
            map_missing=args.map_missing,
        )
    if args.command == "replay-case":
        from .replay_pack import run_replay_pack

        return run_replay_pack(
            args.pack,
            paths=paths,
            run_id=args.run_id,
            run_dir=args.run_dir,
            stop_after=args.stop_after,
        )
    raise ValueError(f"unsupported_case_command:{args.command}")


__all__ = ["CASE_COMMANDS", "add_case_commands", "dispatch_case_command"]

"""CLI surface for the non-authoritative TransformationProgram migration."""

from __future__ import annotations

import argparse
from typing import Any

from cascade_planner.interfaces.experimental_claim_cli import (
    EXPERIMENTAL_CLAIM_COMMANDS,
    add_experimental_claim_commands,
    dispatch_experimental_claim_command,
)
from cascade_planner.interfaces.program_innovation_cli import (
    PROGRAM_INNOVATION_COMMANDS,
    add_program_innovation_commands,
    dispatch_program_innovation_command,
)
from cascade_planner.interfaces.experiment_dispatch_cli import (
    EXPERIMENT_DISPATCH_COMMANDS,
    add_experiment_dispatch_commands,
    dispatch_experiment_command,
)

_BASE_PROGRAM_COMMANDS = frozenset(
    {"programs", "program-routes", "program-store", "admit-programs", "audit-programs"}
)
PROGRAM_COMMANDS = (
    _BASE_PROGRAM_COMMANDS
    | PROGRAM_INNOVATION_COMMANDS
    | EXPERIMENTAL_CLAIM_COMMANDS
    | EXPERIMENT_DISPATCH_COMMANDS
)


def add_program_commands(sub: argparse._SubParsersAction) -> None:
    projection = sub.add_parser(
        "programs", help="project the canonical graph to read-only Programs"
    )
    _add_run_reference(projection)
    routes = sub.add_parser(
        "program-routes", help="verify the Workbench edge/Program dual-read overlay"
    )
    _add_run_reference(routes)
    store = sub.add_parser("program-store", help="replay and compare the shadow Program store")
    _add_run_reference(store)
    admit = sub.add_parser("admit-programs", help="append the exact current Program projection")
    _add_run_reference(admit)
    admit.add_argument(
        "--enable-program-admission",
        action="store_true",
        required=True,
        help="explicitly enable this shadow-only append operation",
    )
    audit = sub.add_parser(
        "audit-programs", help="read-only Program migration audit across indexed runs"
    )
    audit.add_argument("--run-id", action="append", default=[])
    audit.add_argument("--limit", type=int, default=100)
    add_program_innovation_commands(sub)
    add_experimental_claim_commands(sub)
    add_experiment_dispatch_commands(sub)


def dispatch_program_command(gateway: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.command in PROGRAM_INNOVATION_COMMANDS:
        return dispatch_program_innovation_command(gateway, args)
    if args.command in EXPERIMENTAL_CLAIM_COMMANDS:
        return dispatch_experimental_claim_command(gateway, args)
    if args.command in EXPERIMENT_DISPATCH_COMMANDS:
        return dispatch_experiment_command(gateway, args)
    if args.command == "programs":
        return gateway.program_projection(args.run_id, run_dir=args.run_dir)
    if args.command == "program-routes":
        return gateway.route_program_dual_read(args.run_id, run_dir=args.run_dir)
    if args.command == "program-store":
        return gateway.program_store(args.run_id, run_dir=args.run_dir)
    if args.command == "admit-programs":
        return gateway.admit_programs(
            args.run_id,
            run_dir=args.run_dir,
            enable_program_admission=args.enable_program_admission,
        )
    if args.command == "audit-programs":
        return gateway.audit_programs(
            run_ids=tuple(args.run_id),
            limit=args.limit,
        )
    raise ValueError(f"unsupported_program_command:{args.command}")


def _add_run_reference(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id")
    parser.add_argument("--run-dir")


__all__ = ["PROGRAM_COMMANDS", "add_program_commands", "dispatch_program_command"]

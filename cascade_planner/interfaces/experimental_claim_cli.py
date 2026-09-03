"""CLI commands for admitting and replaying exact-boundary Claims."""

from __future__ import annotations

import argparse
from typing import Any

from cascade_planner.interfaces.program_innovation_cli import (
    _add_material_arguments,
    _add_run_reference,
    _read_json,
)


EXPERIMENTAL_CLAIM_COMMANDS = frozenset(
    {"admit-experimental-claims", "experimental-claim-store"}
)


def add_experimental_claim_commands(sub: argparse._SubParsersAction) -> None:
    admit = sub.add_parser(
        "admit-experimental-claims",
        help="append exact-boundary positive, negative, and inconclusive observations",
    )
    _add_material_arguments(admit)
    admit.add_argument(
        "--enable-experimental-claim-admission",
        action="store_true",
        required=True,
        help="explicitly enable this non-authoritative append operation",
    )
    store = sub.add_parser(
        "experimental-claim-store",
        help="replay admitted exact-boundary experimental Claims",
    )
    _add_run_reference(store)


def dispatch_experimental_claim_command(
    gateway: Any, args: argparse.Namespace
) -> dict[str, Any]:
    if args.command == "experimental-claim-store":
        return gateway.experimental_claim_store(args.run_id, run_dir=args.run_dir)
    if args.command == "admit-experimental-claims":
        return gateway.admit_route_experimental_claims(
            args.run_id,
            route_id=args.route_id,
            capabilities=_read_json(args.capabilities_json),
            mechanism_proposals=_read_json(args.mechanism_proposals_json, default=[]),
            validations=_read_json(args.validations_json, default=[]),
            run_dir=args.run_dir,
            enable_experimental_claim_admission=(
                args.enable_experimental_claim_admission
            ),
        )
    raise ValueError(f"unsupported_experimental_claim_command:{args.command}")


__all__ = [
    "EXPERIMENTAL_CLAIM_COMMANDS",
    "add_experimental_claim_commands",
    "dispatch_experimental_claim_command",
]

"""CLI commands for reviewing and admitting biocatalytic Program alternatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROGRAM_INNOVATION_COMMANDS = frozenset(
    {
        "program-innovations",
        "admit-program-innovation",
        "program-innovation-store",
        "audit-experiment-result",
    }
)


def add_program_innovation_commands(sub: argparse._SubParsersAction) -> None:
    review = sub.add_parser(
        "program-innovations", help="review route enzyme windows as Program drafts"
    )
    _add_material_arguments(review)
    review.add_argument("--reported-candidates-json", type=Path)
    admit = sub.add_parser(
        "admit-program-innovation", help="append validated enzyme Program alternatives"
    )
    _add_material_arguments(admit)
    admit.add_argument(
        "--enable-biocatalytic-program-admission",
        action="store_true",
        required=True,
        help="explicitly enable this shadow-only append operation",
    )
    store = sub.add_parser(
        "program-innovation-store", help="replay admitted enzyme Program alternatives"
    )
    _add_run_reference(store)
    result_audit = sub.add_parser(
        "audit-experiment-result",
        help="audit an executor result against the current experiment frontier",
    )
    _add_material_arguments(result_audit)
    result_audit.add_argument("--result-json", type=Path, required=True)


def dispatch_program_innovation_command(
    gateway: Any, args: argparse.Namespace
) -> dict[str, Any]:
    if args.command == "program-innovation-store":
        return gateway.biocatalytic_program_store(args.run_id, run_dir=args.run_dir)
    kwargs: dict[str, Any] = {
        "route_id": args.route_id,
        "capabilities": _read_json(args.capabilities_json),
        "mechanism_proposals": _read_json(args.mechanism_proposals_json, default=[]),
        "validations": _read_json(args.validations_json, default=[]),
        "run_dir": args.run_dir,
    }
    if args.command == "program-innovations":
        kwargs["reported_candidate_packs"] = _read_json(
            args.reported_candidates_json,
            default=[],
        )
        return gateway.route_program_innovations(args.run_id, **kwargs)
    if args.command == "audit-experiment-result":
        return gateway.audit_route_experiment_result(
            args.run_id,
            **kwargs,
            result=_read_json(args.result_json),
        )
    if args.command == "admit-program-innovation":
        return gateway.admit_route_program_innovations(
            args.run_id,
            **kwargs,
            enable_biocatalytic_program_admission=(
                args.enable_biocatalytic_program_admission
            ),
        )
    raise ValueError(f"unsupported_program_innovation_command:{args.command}")


def _add_material_arguments(parser: argparse.ArgumentParser) -> None:
    _add_run_reference(parser)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--capabilities-json", type=Path, required=True)
    parser.add_argument("--mechanism-proposals-json", type=Path)
    parser.add_argument("--validations-json", type=Path)


def _add_run_reference(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id")
    parser.add_argument("--run-dir")


def _read_json(path: Path | None, *, default: Any = None) -> Any:
    if path is None:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"program_innovation_json_invalid:{path.name}") from exc


__all__ = [
    "PROGRAM_INNOVATION_COMMANDS",
    "add_program_innovation_commands",
    "dispatch_program_innovation_command",
]

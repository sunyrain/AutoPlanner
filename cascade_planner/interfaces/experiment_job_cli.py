"""CLI commands for external experiment job observations and cancellation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPERIMENT_JOB_COMMANDS = frozenset({
    "record-experiment-job", "cancel-experiment-job"
})


def add_experiment_job_commands(sub: argparse._SubParsersAction) -> None:
    receipt = sub.add_parser(
        "record-experiment-job", help="record an external experiment job receipt"
    )
    _add_material_arguments(receipt)
    receipt.add_argument("--job-receipt-json", type=Path, required=True)
    receipt.add_argument(
        "--enable-experiment-job-receipt", action="store_true", required=True
    )
    cancellation = sub.add_parser(
        "cancel-experiment-job", help="request cancellation of an external job"
    )
    _add_material_arguments(cancellation)
    cancellation.add_argument("--cancellation-request-json", type=Path, required=True)
    cancellation.add_argument(
        "--enable-experiment-cancellation", action="store_true", required=True
    )


def dispatch_experiment_job_command(
    gateway: Any, args: argparse.Namespace
) -> dict[str, Any]:
    kwargs = {
        "route_id": args.route_id,
        "capabilities": _read_json(args.capabilities_json),
        "mechanism_proposals": _read_json(args.mechanism_proposals_json, default=[]),
        "validations": _read_json(args.validations_json, default=[]),
        "run_dir": args.run_dir,
        "dispatch_id": args.dispatch_id,
    }
    if args.command == "record-experiment-job":
        return gateway.record_route_experiment_job_receipt(
            args.run_id, **kwargs,
            job_receipt=_read_json(args.job_receipt_json),
            enable_experiment_job_receipt=args.enable_experiment_job_receipt,
        )
    if args.command == "cancel-experiment-job":
        return gateway.request_route_experiment_cancellation(
            args.run_id, **kwargs,
            cancellation_request=_read_json(args.cancellation_request_json),
            enable_experiment_cancellation=args.enable_experiment_cancellation,
        )
    raise ValueError(f"unsupported_experiment_job_command:{args.command}")


def _add_material_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id")
    parser.add_argument("--run-dir")
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--capabilities-json", type=Path, required=True)
    parser.add_argument("--mechanism-proposals-json", type=Path)
    parser.add_argument("--validations-json", type=Path)
    parser.add_argument("--dispatch-id", required=True)


def _read_json(path: Path | None, *, default: Any = None) -> Any:
    if path is None:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"experiment_job_json_invalid:{path.name}") from exc


__all__ = [
    "EXPERIMENT_JOB_COMMANDS",
    "add_experiment_job_commands",
    "dispatch_experiment_job_command",
]

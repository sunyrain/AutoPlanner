"""CLI commands for explicit experiment submit, poll, and cancel transport."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


EXPERIMENT_TRANSPORT_COMMANDS = frozenset({
    "submit-experiment-job",
    "poll-experiment-job",
    "transmit-experiment-cancel",
})


def add_experiment_transport_commands(sub: argparse._SubParsersAction) -> None:
    for command, help_text in (
        ("submit-experiment-job", "submit the reserved experiment to its provider"),
        ("poll-experiment-job", "poll the configured external experiment provider"),
        ("transmit-experiment-cancel", "send a recorded cancellation to the provider"),
    ):
        parser = sub.add_parser(command, help=help_text)
        _add_material_arguments(parser)


def dispatch_experiment_transport_command(
    gateway: Any, args: argparse.Namespace
) -> dict[str, Any]:
    kwargs = {
        "route_id": args.route_id,
        "capabilities": _read_json(args.capabilities_json),
        "mechanism_proposals": _read_json(args.mechanism_proposals_json, default=[]),
        "validations": _read_json(args.validations_json, default=[]),
        "dispatch_id": args.dispatch_id,
        "timeout_s": args.timeout_s,
        "enable_experiment_transport": args.enable_experiment_transport,
        "run_dir": args.run_dir,
    }
    method = {
        "submit-experiment-job": gateway.submit_route_experiment_job,
        "poll-experiment-job": gateway.poll_route_experiment_job,
        "transmit-experiment-cancel": gateway.transmit_route_experiment_cancellation,
    }.get(args.command)
    if method is None:
        raise ValueError(f"unsupported_experiment_transport_command:{args.command}")
    return method(args.run_id, **kwargs)


def _add_material_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id")
    parser.add_argument("--run-dir")
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--capabilities-json", type=Path, required=True)
    parser.add_argument("--mechanism-proposals-json", type=Path)
    parser.add_argument("--validations-json", type=Path)
    parser.add_argument("--dispatch-id", required=True)
    parser.add_argument("--timeout-s", type=float, default=0.0)
    parser.add_argument(
        "--enable-experiment-transport", action="store_true", required=True
    )


def _read_json(path: Path | None, *, default: Any = None) -> Any:
    if path is None:
        return default
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"experiment_transport_json_invalid:{path.name}") from exc


__all__ = [
    "EXPERIMENT_TRANSPORT_COMMANDS",
    "add_experiment_transport_commands",
    "dispatch_experiment_transport_command",
]

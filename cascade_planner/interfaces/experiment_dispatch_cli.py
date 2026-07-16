"""CLI commands for bounded experiment dispatch and settlement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPERIMENT_DISPATCH_COMMANDS = frozenset({
    "dispatch-experiment",
    "recover-experiment-dispatch",
    "settle-experiment-dispatch",
    "stage-experiment-artifact",
})


def add_experiment_dispatch_commands(sub: argparse._SubParsersAction) -> None:
    dispatch = sub.add_parser(
        "dispatch-experiment", help="reserve and materialize an experiment handoff"
    )
    _add_material_arguments(dispatch)
    dispatch.add_argument("--request-id", required=True)
    dispatch.add_argument("--provider-policy-json", type=Path, required=True)
    dispatch.add_argument("--enable-experiment-dispatch", action="store_true", required=True)

    recovery = sub.add_parser(
        "recover-experiment-dispatch", help="recover a missing idempotent handoff"
    )
    _add_material_arguments(recovery)
    recovery.add_argument("--dispatch-id", required=True)
    recovery.add_argument(
        "--enable-experiment-dispatch-recovery", action="store_true", required=True
    )

    settlement = sub.add_parser(
        "settle-experiment-dispatch", help="audit and settle an experiment result"
    )
    _add_material_arguments(settlement)
    settlement.add_argument("--dispatch-id", required=True)
    settlement.add_argument("--result-json", type=Path, required=True)
    settlement.add_argument("--enable-experiment-settlement", action="store_true", required=True)

    stage = sub.add_parser(
        "stage-experiment-artifact", help="stage untrusted experiment JSON in the CAS"
    )
    _add_run_reference(stage)
    stage.add_argument("--artifact-json", type=Path, required=True)
    stage.add_argument("--logical-name", required=True)
    stage.add_argument(
        "--enable-experiment-artifact-staging", action="store_true", required=True
    )


def dispatch_experiment_command(gateway: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "stage-experiment-artifact":
        return gateway.stage_experiment_json_artifact(
            args.run_id,
            run_dir=args.run_dir,
            artifact=_read_json(args.artifact_json),
            logical_name=args.logical_name,
            enable_experiment_artifact_staging=args.enable_experiment_artifact_staging,
        )
    kwargs = {
        "route_id": args.route_id,
        "capabilities": _read_json(args.capabilities_json),
        "mechanism_proposals": _read_json(args.mechanism_proposals_json, default=[]),
        "validations": _read_json(args.validations_json, default=[]),
        "run_dir": args.run_dir,
    }
    if args.command == "dispatch-experiment":
        return gateway.dispatch_route_experiment(
            args.run_id, **kwargs, request_id=args.request_id,
            policy=_read_json(args.provider_policy_json),
            enable_experiment_dispatch=args.enable_experiment_dispatch,
        )
    if args.command == "recover-experiment-dispatch":
        return gateway.recover_route_experiment_dispatch(
            args.run_id, **kwargs, dispatch_id=args.dispatch_id,
            enable_experiment_dispatch_recovery=args.enable_experiment_dispatch_recovery,
        )
    if args.command == "settle-experiment-dispatch":
        return gateway.settle_route_experiment_dispatch(
            args.run_id, **kwargs, dispatch_id=args.dispatch_id,
            result=_read_json(args.result_json),
            enable_experiment_settlement=args.enable_experiment_settlement,
        )
    raise ValueError(f"unsupported_experiment_dispatch_command:{args.command}")


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
        raise ValueError(f"experiment_dispatch_json_invalid:{path.name}") from exc


__all__ = [
    "EXPERIMENT_DISPATCH_COMMANDS",
    "add_experiment_dispatch_commands",
    "dispatch_experiment_command",
]

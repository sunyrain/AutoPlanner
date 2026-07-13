"""Canonical AutoPlanner command-line interface.

All campaign commands call the same gateway used by the V4 HTTP adapter.  No
command invokes a model or network service implicitly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.interfaces.campaign_gateway import (
    CampaignGateway,
    CampaignGatewayError,
)
from cascade_planner.interfaces.case_cli import (
    CASE_COMMANDS,
    add_case_commands,
    dispatch_case_command,
)
from cascade_planner.runtime.paths import RuntimePaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoplanner",
        description="Bounded, resumable V4 retrosynthesis campaigns",
    )
    parser.add_argument("--repository-root")
    parser.add_argument("--runtime-root")
    parser.add_argument("--runs-root")
    parser.add_argument("--artifact-store-root")
    parser.add_argument("--run-index-path")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="create or idempotently reopen one V4 run")
    run.add_argument("--target-name", required=True)
    run.add_argument("--target-smiles", required=True)
    run.add_argument("--run-id")
    run.add_argument("--run-dir")
    run.add_argument("--plan", type=Path, help="validated global campaign plan JSON")
    run.add_argument("--materialize", action="store_true")
    run.add_argument("--closeout", action="store_true")
    run.add_argument("--minimum-complete-routes", type=int, default=2)
    run.add_argument("--minimum-edge-proof-level", type=int, default=3)
    run.add_argument("--minimum-source-groups", type=int, default=2)
    run.add_argument(
        "--stock-boundary",
        choices=("benchmark_search", "procurement", "in_house"),
        default="procurement",
    )
    run.add_argument("--max-accepted-expansions", type=int, default=8)
    run.add_argument("--max-attempt-runs", type=int, default=12)

    resume = sub.add_parser("resume", help="resume deterministic frontier work")
    _add_run_reference(resume)
    resume.add_argument("--materialize", action="store_true")
    resume.add_argument("--closeout", action="store_true")

    for name, help_text in (
        ("status", "show canonical run, graph, frontier, and proof status"),
        ("validate", "validate event replay, graph oracle, and UI bindings"),
        ("replay", "rebuild projections and verify deterministic equality"),
    ):
        command = sub.add_parser(name, help=help_text)
        _add_run_reference(command)

    benchmark = sub.add_parser(
        "benchmark", help="measure model-free status/oracle/projection work"
    )
    _add_run_reference(benchmark)
    benchmark.add_argument("--iterations", type=int, default=3)

    export = sub.add_parser("export", help="export bounded JSON and offline HTML")
    _add_run_reference(export)
    export.add_argument("--output-dir")

    gc = sub.add_parser("gc", help="plan immutable-artifact garbage collection")
    gc.add_argument(
        "--dry-run",
        action="store_true",
        help="explicitly confirm that no objects may be deleted",
    )
    gc.add_argument("--minimum-age-hours", type=float, default=24.0)

    listing = sub.add_parser("list", help="list indexed V4 runs")
    listing.add_argument("--limit", type=int, default=100)

    audit = sub.add_parser("audit", help="audit the tracked current repository tree")
    audit.add_argument("--large-blob-bytes", type=int, default=1_000_000)
    audit.add_argument("--output", type=Path)

    add_case_commands(sub)

    serve = sub.add_parser("serve", help="serve the Web UI and V4 API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--server", choices=("waitress", "flask"), default="waitress")
    serve.add_argument("--threads", type=int, default=2)
    serve.add_argument("--debug", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            _serve(args)
            return 0
        if args.command == "audit":
            result = _audit(args)
        else:
            result = _dispatch(CampaignGateway(_paths(args)), args)
    except (CampaignGatewayError, FileNotFoundError, ValueError) as exc:
        _emit(
            {
                "schema_version": "autoplanner_cli_error.v1",
                "error": type(exc).__name__,
                "reason": str(exc),
            },
            stream=sys.stderr,
        )
        return 2
    _emit(result)
    return 0


def _dispatch(gateway: CampaignGateway, args: argparse.Namespace) -> dict[str, Any]:
    if args.command in CASE_COMMANDS:
        return dispatch_case_command(args, paths=gateway.paths)
    if args.command == "run":
        plan = _read_object(args.plan) if args.plan else None
        return gateway.create_run(
            target_name=args.target_name,
            target_smiles=args.target_smiles,
            run_id=args.run_id,
            run_dir=args.run_dir,
            acceptance=RetrosynthesisAcceptanceSpec(
                minimum_complete_routes=args.minimum_complete_routes,
                minimum_edge_proof_level=args.minimum_edge_proof_level,
                minimum_independent_source_groups=args.minimum_source_groups,
                stock_boundary=args.stock_boundary,
            ),
            budget=RetrosynthesisRunBudget(
                max_model_invocations=0,
                max_visual_invocations=0,
                max_accepted_expansions=args.max_accepted_expansions,
                max_attempt_runs=args.max_attempt_runs,
            ),
            global_plan=plan,
            materialize=args.materialize,
            closeout=args.closeout,
        )
    if args.command == "resume":
        return gateway.resume(
            args.run_id,
            run_dir=args.run_dir,
            materialize=args.materialize,
            closeout=args.closeout,
        )
    if args.command == "status":
        return gateway.status(args.run_id, run_dir=args.run_dir)
    if args.command == "validate":
        return gateway.validate(args.run_id, run_dir=args.run_dir)
    if args.command == "replay":
        return gateway.replay(args.run_id, run_dir=args.run_dir)
    if args.command == "benchmark":
        return gateway.benchmark(
            args.run_id,
            run_dir=args.run_dir,
            iterations=args.iterations,
        )
    if args.command == "export":
        return gateway.export(
            args.run_id,
            run_dir=args.run_dir,
            output_dir=args.output_dir,
        )
    if args.command == "gc":
        if not args.dry_run:
            raise ValueError("gc_requires_explicit_--dry-run")
        return gateway.gc_plan(minimum_age_s=args.minimum_age_hours * 3_600.0)
    if args.command == "list":
        return gateway.list_runs(limit=args.limit)
    raise ValueError(f"unsupported_command:{args.command}")


def _paths(args: argparse.Namespace) -> RuntimePaths:
    env = dict(os.environ)
    for attribute, variable in (
        ("runtime_root", "AUTOPLANNER_RUNTIME_ROOT"),
        ("runs_root", "AUTOPLANNER_RUNS_ROOT"),
        ("artifact_store_root", "AUTOPLANNER_ARTIFACT_STORE_ROOT"),
        ("run_index_path", "AUTOPLANNER_RUN_INDEX_PATH"),
    ):
        value = getattr(args, attribute, None)
        if value:
            env[variable] = str(value)
    return RuntimePaths.discover(
        repository_root=args.repository_root,
        environ=env,
    )


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    from cascade_planner.runtime.repository_audit import audit_repository

    result = audit_repository(
        args.repository_root or Path(__file__).resolve().parents[1],
        large_blob_bytes=args.large_blob_bytes,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _serve(args: argparse.Namespace) -> None:
    for variable, attribute in (
        ("AUTOPLANNER_RUNTIME_ROOT", "runtime_root"),
        ("AUTOPLANNER_RUNS_ROOT", "runs_root"),
        ("AUTOPLANNER_ARTIFACT_STORE_ROOT", "artifact_store_root"),
        ("AUTOPLANNER_RUN_INDEX_PATH", "run_index_path"),
    ):
        value = getattr(args, attribute, None)
        if value:
            os.environ[variable] = str(value)
    from cascade_planner.web.app import create_app

    app = create_app()
    if args.server == "waitress":
        try:
            from waitress import serve
        except ImportError as exc:
            raise ValueError(
                "waitress_not_installed; install requirements or use --server flask"
            ) from exc
        serve(
            app,
            host=args.host,
            port=args.port,
            threads=max(1, min(32, int(args.threads))),
            channel_timeout=30,
        )
    else:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
        )


def _add_run_reference(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id")
    parser.add_argument("--run-dir")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_input_must_be_an_object")
    return value


def _emit(value: Any, *, stream: Any = None) -> None:
    stream = sys.stdout if stream is None else stream
    json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

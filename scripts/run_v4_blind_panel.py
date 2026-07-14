#!/usr/bin/env python3
"""Run a target-only V4 manifest in one fresh, isolated external workspace."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, get_ident
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.blind_benchmark_contract import (  # noqa: E402
    BlindCase,
    load_blind_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", choices=("low", "medium"), default="low")
    parser.add_argument(
        "--execution-profile",
        choices=("fast", "standard", "proof"),
        default="standard",
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args(argv)

    manifest = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    cases = list(load_blind_manifest(manifest))
    selected = {str(value).casefold() for value in args.only}
    if selected:
        cases = [case for case in cases if case.target_name.casefold() in selected]
    if not cases:
        raise SystemExit("No benchmark cases selected")

    for name in ("audit", "runtime", "runs", "artifacts", "external", "logs"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    status_path = output_root / "panel-status.json"
    state_lock = Lock()
    state: dict[str, Any] = {
        "schema_version": "v4_blind_panel_status.v1",
        "manifest_path": str(manifest),
        "output_root": str(output_root),
        "started_at": _utc_now(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "execution_profile": args.execution_profile,
        "worker_count": args.workers,
        "target_count": len(cases),
        "targets": {
            case.target_name: {"status": "queued", "case_id": case.case_id}
            for case in cases
        },
        "semantics": {
            "target_name_and_smiles_only": True,
            "no_local_pdf_doi_patent_or_route_seed": True,
            "isolated_runtime_and_external_evidence_root": True,
            "one_global_codex_call_per_target": True,
        },
    }
    _write_json(status_path, state)

    def run(case: BlindCase) -> tuple[str, dict[str, Any]]:
        with state_lock:
            state["targets"][case.target_name] = {
                "status": "running",
                "case_id": case.case_id,
                "started_at": _utc_now(),
            }
            _write_json(status_path, state)
        return case.target_name, _run_case(
            case,
            manifest=manifest,
            output_root=output_root,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            execution_profile=args.execution_profile,
            resume=args.resume,
            visual=args.visual,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                name, result = future.result()
            except Exception as exc:  # bounded batch supervisor boundary
                name = case.target_name
                result = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:4000],
                    "finished_at": _utc_now(),
                }
            with state_lock:
                state["targets"][name] = result
                _write_json(status_path, state)
    state["finished_at"] = _utc_now()
    state["complete"] = True
    state["completed_count"] = sum(
        row.get("status") == "completed" for row in state["targets"].values()
    )
    _write_json(status_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["completed_count"] == len(cases) else 2


def _run_case(
    case: BlindCase,
    *,
    manifest: Path,
    output_root: Path,
    model: str,
    reasoning_effort: str,
    execution_profile: str,
    resume: bool,
    visual: bool,
) -> dict[str, Any]:
    run_id = f"statin-{case.target_name}-v4-blind-20260715"
    run_dir = output_root / "runs" / case.target_name
    report_path = run_dir / "target-only-solve-report.json"
    can_resume = resume and (run_dir / ".autoplanner" / "kernel" / "run_spec.json").is_file()
    if run_dir.exists() and any(run_dir.iterdir()) and not can_resume:
        if report_path.is_file():
            return _summarize_report(report_path, elapsed_s=0.0, reused=True)
        raise RuntimeError("non_fresh_run_dir_requires_resume")
    command = [
        sys.executable,
        "-m",
        "cascade_planner.cli",
        "solve-target",
        "--target-name",
        case.target_name,
        "--target-smiles",
        case.target_smiles,
        "--run-id",
        run_id,
        "--run-dir",
        str(run_dir),
        "--manifest",
        str(manifest),
        "--blind-audit-root",
        str(output_root / "audit"),
        "--model",
        model,
        "--reasoning-effort",
        reasoning_effort,
        "--execution-profile",
        execution_profile,
        "--no-replan",
        "--minimum-complete-routes",
        "2",
        "--minimum-edge-proof-level",
        "2",
        "--minimum-source-groups",
        "2",
        "--max-model-invocations",
        "1",
        "--max-input-tokens",
        "55000",
        "--max-output-tokens",
        "10000",
        "--max-model-wall-time-s",
        "420",
        "--max-accepted-expansions",
        "40",
        "--max-attempt-runs",
        "96",
        "--max-map-reactions",
        "64",
        "--max-stock-molecules",
        "32",
        "--max-patent-sources",
        "2",
        "--max-literature-sources",
        "3",
        "--guided-chemenzy-frontiers",
        "1",
        "--guided-chemenzy-iterations",
        "4",
        "--guided-chemenzy-timeout-s",
        "60",
        "--max-visual-invocations",
        "1" if visual else "0",
        "--max-visual-pages",
        "2",
    ]
    if can_resume:
        command.append("--resume")
    environment = dict(os.environ)
    environment.update(
        {
            "AUTOPLANNER_RUNTIME_ROOT": str(output_root / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(output_root / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(output_root / "artifacts"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(output_root / "runtime" / "run_index.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(output_root / "external"),
            "AUTOPLANNER_VENDOR_ROOT": str(ROOT / "vendor"),
        }
    )
    log_path = output_root / "logs" / f"{case.target_name}.log"
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed_s = round(time.monotonic() - started, 3)
    if not report_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "elapsed_s": elapsed_s,
            "log_path": str(log_path),
            "error": tail,
            "finished_at": _utc_now(),
        }
    return _summarize_report(report_path, elapsed_s=elapsed_s, reused=False)


def _summarize_report(
    path: Path,
    *,
    elapsed_s: float,
    reused: bool,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    gate_report = dict(report.get("gates") or {})
    gates = dict(gate_report.get("gates") or {})
    counts = dict(gate_report.get("counts") or {})
    claim = dict(report.get("claim") or {})
    model = dict(report.get("model_cost") or {})
    stages = list(report.get("stages") or [])
    chemenzy_stages = [
        dict(row.get("detail") or {})
        for row in stages
        if isinstance(row, Mapping)
        and row.get("stage")
        in {"chemenzy_guided_frontier", "chemenzy_stock_recovery"}
    ]
    guided = chemenzy_stages[0] if chemenzy_stages else {}
    return {
        "status": "completed",
        "claim": str(claim.get("achieved_profile") or "unresolved"),
        "accepted_under_configured_policy": (
            claim.get("accepted_under_configured_policy") is True
        ),
        "elapsed_s": elapsed_s,
        "reused": reused,
        "run_id": str(report.get("run_id") or ""),
        "run_dir": str(report.get("run_dir") or path.parent),
        "report_path": str(path),
        "model_cost": model,
        "attempt_count": int(report.get("attempt_count") or 0),
        "accepted_expansion_count": int(report.get("accepted_expansion_count") or 0),
        "gate_summary": {
            key.split("_", 1)[0]: bool(value)
            for key, value in gates.items()
            if key[:2] in {"B0", "B1", "B2", "B3", "B4", "B5"}
        },
        "route_counts": counts,
        "chemenzy": {
            "status": (
                "completed"
                if any(row.get("status") == "completed" for row in chemenzy_stages)
                else str(guided.get("status") or "")
            ),
            "frontier_count": sum(
                int(row.get("frontier_count") or 0) for row in chemenzy_stages
            ),
            "provider_invocation_count": sum(
                int(
                    row.get("provider_invocation_count")
                    or row.get("executed_frontier_count")
                    or 0
                )
                for row in chemenzy_stages
            ),
            "proposal_count": sum(
                int(row.get("proposal_count") or 0) for row in chemenzy_stages
            ),
            "initial_delegation_status": str(guided.get("status") or ""),
            "stock_recovery_used": any(
                row.get("status") == "completed" for row in chemenzy_stages[1:]
            ),
        },
        "stage_timings": {
            str(row.get("stage") or ""): float(row.get("elapsed_s") or 0.0)
            for row in stages
            if isinstance(row, Mapping) and row.get("stage")
        },
        "workbench_url": (
            f"/api/v4/runs/{report.get('run_id')}/workbench.html"
        ),
        "finished_at": _utc_now(),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

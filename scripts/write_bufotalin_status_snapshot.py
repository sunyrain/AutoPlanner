"""Write a compact status snapshot for the active bufotalin long run."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_bufotalin_12h_completion import audit_completion
from scripts.summarize_bufotalin_iteration import summarize_iteration_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Write bufotalin status snapshot JSON.")
    parser.add_argument("root")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "status_snapshot.json"
    snapshot = build_status_snapshot(root)
    output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), **snapshot["headline"]}, indent=2, ensure_ascii=False))


def build_status_snapshot(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    summary = summarize_iteration_root(root)
    audit = audit_completion(root)
    processes = _matching_processes()
    failed_checks = [
        check["name"]
        for check in audit.get("checks") or []
        if not check.get("passed")
    ]
    final_candidates = _read_json(root / "final_candidates" / "final_candidates.json")
    proposal_gate = _read_json(root / "cycle_proposal_gate_retrofit_summary.json")
    early_stop = _early_stop_readiness(summary, audit, final_candidates, processes)
    return {
        "schema_version": "bufotalin_status_snapshot.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "headline": {
            "complete": bool(audit.get("complete")),
            "early_stop_review_ready": early_stop["review_ready"],
            "status_label": "complete" if audit.get("complete") else early_stop["status_label"],
            "failed_checks": failed_checks,
            "completed_payload_count": summary.get("completed_payload_count"),
            "running_cycle_count": summary.get("running_cycle_count"),
            "native_success_payloads": summary.get("native_success_payloads"),
            "high_confidence_final_routes": final_candidates.get("high_confidence_count"),
            "selected_final_routes": final_candidates.get("selected_count"),
            "excluded_final_candidates": final_candidates.get("excluded_route_count"),
            "proposal_gate_dropped_routes": proposal_gate.get("dropped_routes"),
        },
        "processes": processes,
        "summary": summary,
        "audit": audit,
        "early_stop": early_stop,
        "final_candidates": {
            key: final_candidates.get(key)
            for key in [
                "generated_at",
                "high_confidence_count",
                "stitched_review_only_count",
                "native_review_only_count",
                "selected_count",
                "excluded_route_count",
            ]
        },
        "proposal_gate": _proposal_gate_summary(proposal_gate),
    }


def _early_stop_readiness(
    summary: dict[str, Any],
    audit: dict[str, Any],
    final_candidates: dict[str, Any],
    processes: list[dict[str, str]],
) -> dict[str, Any]:
    failed_checks = {
        check["name"]
        for check in audit.get("checks") or []
        if not check.get("passed")
    }
    only_time_gate_failed = failed_checks.issubset({"finished", "ran_min_hours"})
    no_active_workers = not processes and int(summary.get("running_cycle_count") or 0) == 0
    has_high_confidence = int(final_candidates.get("high_confidence_count") or 0) > 0
    has_selected = int(final_candidates.get("selected_count") or 0) > 0
    review_ready = bool(
        not audit.get("complete")
        and only_time_gate_failed
        and no_active_workers
        and has_high_confidence
        and has_selected
    )
    if audit.get("complete"):
        status_label = "complete"
    elif review_ready:
        status_label = "early_stop_review_ready"
    elif no_active_workers:
        status_label = "stopped_incomplete"
    else:
        status_label = "running"
    return {
        "review_ready": review_ready,
        "status_label": status_label,
        "only_time_gate_failed": only_time_gate_failed,
        "no_active_workers": no_active_workers,
        "failed_checks": sorted(failed_checks),
        "rationale": (
            "Original 12h completion gate is not satisfied, but the stopped run has a "
            "high-confidence final candidate package and no active workers."
            if review_ready
            else ""
        ),
    }


def _matching_processes() -> list[dict[str, str]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid,ppid,sid,stat,etime,%cpu,%mem,cmd"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if not _is_relevant_runtime_process(line):
            continue
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        rows.append(
            {
                "pid": parts[0],
                "ppid": parts[1],
                "sid": parts[2],
                "stat": parts[3],
                "etime": parts[4],
                "cpu": parts[5],
                "mem": parts[6],
                "cmd": parts[7],
            }
        )
    return rows


def _is_relevant_runtime_process(line: str) -> bool:
    runtime_markers = (
        "python scripts/run_bufotalin_12h_iteration.py",
        "python -u scripts/watch_bufotalin_run_completion.py",
        "/scripts/run_bufotalin_12h_iteration.py",
        "/scripts/watch_bufotalin_run_completion.py",
        "render_linear_route.py",
    )
    return any(marker in line for marker in runtime_markers)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _proposal_gate_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {"available": False}
    return {
        "available": True,
        "mode": report.get("mode"),
        "payload_count": report.get("payload_count"),
        "input_routes": report.get("input_routes"),
        "kept_routes": report.get("kept_routes"),
        "dropped_routes": report.get("dropped_routes"),
        "reason_counts": report.get("reason_counts") or {},
    }


if __name__ == "__main__":
    main()

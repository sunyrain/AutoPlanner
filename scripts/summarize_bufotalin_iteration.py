"""Summarize a bufotalin 12h iteration directory."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


_TQDM_PROGRESS_RE = re.compile(r"(\d+)%\|[^\r\n]*?\|\s*(\d+)/(\d+)\s*\[")


def summarize_iteration_root(root: Path | str) -> dict:
    root = Path(root)
    cycle_dirs = sorted({path.parent for path in root.glob("*/cycle_config.json")})
    payload_paths = {path.parent: path for path in root.glob("*/web_payload.json")}
    active_process_commands = _active_process_commands()
    rows = []
    for payload_path in sorted(payload_paths.values()):
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        routes = payload.get("routes") or []
        figures = _figure_summary(payload_path.parent)
        probe = (payload.get("route_set_metrics") or {}).get("template_relevance_top_level_probe") or {}
        condition_prediction = (payload.get("route_set_metrics") or {}).get("condition_prediction") or {}
        rows.append(
            {
                "cycle": payload_path.parent.name,
                "complete": True,
                "status": (payload.get("search_status") or {}).get("status"),
                "ok": bool(payload.get("ok")),
                "n_results": int(payload.get("n_results") or 0),
                "native_raw_n_routes": (payload.get("search_status") or {}).get("native_raw_n_routes"),
                "semisynthesis_rescue_n_routes": (payload.get("search_status") or {}).get("semisynthesis_rescue_n_routes"),
                "semisynthesis_anchor_routes": sum(
                    1 for route in routes if (route.get("metrics") or {}).get("semisynthesis_anchor")
                ),
                "route_solved": sum(1 for route in routes if (route.get("metrics") or {}).get("route_solved")),
                "upstream_stitched": sum(
                    1
                    for route in routes
                    if (route.get("raw_backend_metadata") or {}).get("route_class_hint") == "stitched_semisynthesis_upstream"
                ),
                "cascade_verifier_feasible": sum(
                    1
                    for route in routes
                    if ((route.get("metrics") or {}).get("cascade_verifier") or {}).get("feasible")
                ),
                "source_supported_semisynthesis": sum(
                    1 for route in routes if (route.get("metrics") or {}).get("source_supported_semisynthesis")
                ),
                "condition_complete_routes": sum(1 for route in routes if _condition_coverage(route) >= 1.0),
                "native_condition_complete_routes": sum(
                    1
                    for route in routes
                    if not (route.get("metrics") or {}).get("source_supported_semisynthesis")
                    and _condition_coverage(route) >= 1.0
                ),
                "renderable_conditioned_routes": sum(1 for route in routes if _renderable_conditioned(route)),
                "condition_prediction_enabled": bool(condition_prediction.get("enabled")),
                "template_relevance_probe_hit": bool(probe.get("hit_expected_precursor")),
                "template_relevance_probe_returned": int(probe.get("returned") or 0),
                "figures_svg": figures["svg"],
                "figures_pdf": figures["pdf"],
                "failures": [
                    row.get("category")
                    for row in payload.get("backend_failures") or []
                    if isinstance(row, dict)
                ],
                "path": str(payload_path),
            }
        )
    completed_dirs = set(payload_paths)
    for cycle_dir in cycle_dirs:
        if cycle_dir in completed_dirs:
            continue
        progress = _worker_progress(cycle_dir / "worker.log")
        if progress and _cycle_has_active_process(cycle_dir, active_process_commands):
            status = "running"
        elif progress:
            status = "stopped"
        else:
            status = "pending"
        figures = _figure_summary(cycle_dir)
        rows.append(
            {
                "cycle": cycle_dir.name,
                "complete": False,
                "status": status,
                "ok": False,
                "n_results": 0,
                "native_raw_n_routes": None,
                "semisynthesis_rescue_n_routes": None,
                "semisynthesis_anchor_routes": 0,
                "route_solved": 0,
                "upstream_stitched": 0,
                "cascade_verifier_feasible": 0,
                "source_supported_semisynthesis": 0,
                "condition_complete_routes": 0,
                "native_condition_complete_routes": 0,
                "renderable_conditioned_routes": 0,
                "condition_prediction_enabled": False,
                "template_relevance_probe_hit": False,
                "template_relevance_probe_returned": 0,
                "figures_svg": figures["svg"],
                "figures_pdf": figures["pdf"],
                "worker_progress": progress,
                "failures": [],
                "path": str(cycle_dir),
            }
        )
    rows.sort(key=lambda row: (row["cycle"] != "anchor", row["cycle"]))
    summary = {
        "root": str(root),
        "payload_count": len(rows),
        "completed_payload_count": sum(1 for row in rows if row.get("complete")),
        "running_cycle_count": sum(1 for row in rows if row.get("status") == "running"),
        "native_success_payloads": sum(1 for row in rows if int(row.get("native_raw_n_routes") or 0) > 0),
        "route_solved_payloads": sum(1 for row in rows if int(row.get("route_solved") or 0) > 0),
        "semisynthesis_anchor_payloads": sum(1 for row in rows if int(row.get("semisynthesis_anchor_routes") or 0) > 0),
        "upstream_stitched_payloads": sum(1 for row in rows if int(row.get("upstream_stitched") or 0) > 0),
        "cascade_verifier_feasible_payloads": sum(1 for row in rows if int(row.get("cascade_verifier_feasible") or 0) > 0),
        "source_supported_semisynthesis_payloads": sum(
            1 for row in rows if int(row.get("source_supported_semisynthesis") or 0) > 0
        ),
        "condition_complete_payloads": sum(1 for row in rows if int(row.get("condition_complete_routes") or 0) > 0),
        "native_condition_complete_payloads": sum(
            1 for row in rows if int(row.get("native_condition_complete_routes") or 0) > 0
        ),
        "renderable_conditioned_payloads": sum(1 for row in rows if int(row.get("renderable_conditioned_routes") or 0) > 0),
        "condition_prediction_enabled_payloads": sum(1 for row in rows if row.get("condition_prediction_enabled")),
        "template_relevance_probe_hit_payloads": sum(1 for row in rows if row.get("template_relevance_probe_hit")),
        "figure_svg_count": sum(int(row.get("figures_svg") or 0) for row in rows),
        "figure_pdf_count": sum(int(row.get("figures_pdf") or 0) for row in rows),
        "rows": rows,
    }
    return summary


def _active_process_commands() -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "cmd="],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.splitlines()


def _cycle_has_active_process(cycle_dir: Path, commands: list[str]) -> bool:
    cycle_dir_text = str(cycle_dir)
    return any(
        "run_bufotalin_12h_iteration.py" in command and cycle_dir_text in command
        for command in commands
    )


def _figure_summary(cycle_dir: Path) -> dict[str, int]:
    figure_dir = cycle_dir / "figures"
    manifest_path = figure_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            figures = manifest.get("figures") or []
            return {
                "svg": sum(1 for item in figures if item.get("svg")),
                "pdf": sum(1 for item in figures if item.get("pdf")),
            }
        except Exception:
            pass
    return {
        "svg": len(list(figure_dir.glob("scheme_route_*.svg"))),
        "pdf": len(list(figure_dir.glob("scheme_route_*.pdf"))),
    }


def _worker_progress(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    text = _tail_text(log_path)
    matches = list(_TQDM_PROGRESS_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    percent, current, total = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return {
        "percent": percent,
        "current_iteration": current,
        "total_iterations": total,
    }


def _condition_coverage(route: dict) -> float:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    if not steps:
        return 0.0
    return sum(1 for step in steps if step.get("condition_predictions")) / max(1, len(steps))


def _renderable_conditioned(route: dict) -> bool:
    metrics = route.get("metrics") or {}
    verifier = metrics.get("cascade_verifier") or {}
    if metrics.get("source_supported_semisynthesis"):
        return _condition_coverage(route) >= 1.0
    return bool(verifier.get("feasible") and _condition_coverage(route) >= 1.0)


def _tail_text(path: Path, max_bytes: int = 1_000_000) -> str:
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        return fh.read().decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize bufotalin iteration outputs.")
    parser.add_argument("root")
    args = parser.parse_args()
    summary = summarize_iteration_root(args.root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

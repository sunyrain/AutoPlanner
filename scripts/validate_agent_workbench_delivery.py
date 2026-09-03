"""Validate the unified V4 workspace and representative route outputs."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


REPRESENTATIVE_RUNS = {
    "bufotalin_solved_mixed": {
        "run_dir": "results/shared/bufotalin_full_exact_stitch_rerun_20260622_073847",
        "min_steps": 20,
        "min_branches": 4,
        "required_branch_kinds": {"integrated_solution", "exact_literature", "subgoal_verified_route"},
        "min_agent_steps": 0,
    },
    "atorvastatin_mixed": {
        "run_dir": "results/shared/ui_agent_runs/ui_agent_complex_atorvastatin_web_direct_chemenzy_20260707_174922_0ad499",
        "min_steps": 20,
        "min_branches": 8,
        "required_branch_kinds": {"recommended_strategy", "direct_verified_route"},
        "min_agent_steps": 0,
    },
    "paclitaxel_delivery_smoke": {
        "run_dir": "results/shared/ui_agent_runs/ui_agent_delivery_smoke_paclitaxel_delivery_smoke_20260707_182713_9f9839",
        "min_steps": 4,
        "min_branches": 1,
        "required_branch_kinds": {"recommended_strategy"},
        "min_agent_steps": 5,
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860", help="Running workspace URL")
    parser.add_argument("--skip-server", action="store_true", help="Only validate local files")
    args = parser.parse_args(argv)

    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    _check_file("workspace html", ROOT / "cascade_planner/web/static/workspace.html", failures, checks)

    if not args.skip_server:
        _check_http(args.base_url.rstrip("/") + "/api/v4/workspace", failures, checks, expect_json=True)
        _check_http(args.base_url.rstrip("/") + "/v4", failures, checks)

    for name, spec in REPRESENTATIVE_RUNS.items():
        _validate_run(name, spec, base_url=args.base_url.rstrip("/"), skip_server=args.skip_server, failures=failures, checks=checks)

    report = {
        "schema_version": "unified_workspace_delivery_validation.v2",
        "ok": not failures,
        "checks": checks,
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def _check_file(label: str, path: Path, failures: list[str], checks: list[dict[str, Any]]) -> None:
    ok = path.is_file() and path.stat().st_size > 0
    checks.append({"label": label, "path": str(path.relative_to(ROOT)), "ok": ok})
    if not ok:
        failures.append(f"{label} missing or empty: {path}")


def _check_http(url: str, failures: list[str], checks: list[dict[str, Any]], *, expect_json: bool = False) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read()
            ok = 200 <= response.status < 300
            data = json.loads(body.decode("utf-8")) if expect_json else body.decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        checks.append({"label": "http", "url": url, "ok": False, "error": str(exc)})
        failures.append(f"http check failed for {url}: {exc}")
        return None
    checks.append({"label": "http", "url": url, "ok": ok})
    if not ok:
        failures.append(f"http check returned non-2xx for {url}")
    return data


def _validate_run(
    name: str,
    spec: dict[str, Any],
    *,
    base_url: str,
    skip_server: bool,
    failures: list[str],
    checks: list[dict[str, Any]],
) -> None:
    run_dir = ROOT / str(spec["run_dir"])
    forest_path = run_dir / "explored_route_forest.json"
    html_path = run_dir / "route_forest.html"
    _check_file(f"{name} forest json", forest_path, failures, checks)
    _check_file(f"{name} route html", html_path, failures, checks)
    if not forest_path.is_file():
        return

    forest = json.loads(forest_path.read_text(encoding="utf-8"))
    counts = dict(forest.get("counts") or {})
    branch_kinds = {str(row.get("kind") or "") for row in forest.get("branches") or []}
    html = html_path.read_text(encoding="utf-8", errors="replace") if html_path.is_file() else ""
    agent_step_count = _agent_step_count(run_dir)

    assertions = {
        "min_steps": int(counts.get("steps") or 0) >= int(spec["min_steps"]),
        "min_branches": int(counts.get("branches") or 0) >= int(spec["min_branches"]),
        "required_branch_kinds": set(spec["required_branch_kinds"]).issubset(branch_kinds),
        "delivery_html_markers": all(marker in html for marker in ["routeFlowSvg", "activeReplacement", "备选", "检查器"]),
        "agent_steps": agent_step_count >= int(spec["min_agent_steps"]),
    }
    checks.append(
        {
            "label": name,
            "ok": all(assertions.values()),
            "counts": counts,
            "branch_kinds": sorted(branch_kinds),
            "agent_steps": agent_step_count,
            "assertions": assertions,
        }
    )
    for key, ok in assertions.items():
        if not ok:
            failures.append(f"{name} failed {key}")

    if not skip_server:
        rel = str(html_path.relative_to(ROOT)).replace("\\", "/")
        url = f"{base_url}/api/v4/result-file?path={urllib.parse.quote(rel)}"
        body = _check_http(url, failures, checks)
        if isinstance(body, str) and "routeFlowSvg" not in body:
            failures.append(f"{name} served route html missing routeFlowSvg")


def _agent_step_count(run_dir: Path) -> int:
    summary = run_dir / "blackboard_steps" / "summary.jsonl"
    if not summary.is_file():
        return 0
    return sum(1 for line in summary.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


if __name__ == "__main__":
    raise SystemExit(main())

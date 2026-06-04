"""Write a concise gap report for the bufotalin strict 12h goal."""
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
from scripts.write_bufotalin_package_manifest import build_package_manifest


def build_gap_report(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    package = build_package_manifest(root)
    strict = audit_completion(root)
    failed = [check for check in strict.get("checks") or [] if not check.get("passed")]
    passed = [check for check in strict.get("checks") or [] if check.get("passed")]
    return {
        "schema_version": "bufotalin_completion_gap_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "strict_12h_complete": bool(strict.get("complete")),
        "early_stop_review_ready": bool(package.get("status", {}).get("early_stop_review_ready")),
        "elapsed_hours": strict.get("elapsed_hours"),
        "failed_checks": failed,
        "passed_check_count": len(passed),
        "failed_check_count": len(failed),
        "status": package.get("status") or {},
        "conclusion": package.get("conclusion") or {},
        "proposal_gate": package.get("proposal_gate") or {},
        "proposal_frontiers": package.get("proposal_frontiers") or {},
        "frontier_proposal_probe": package.get("frontier_proposal_probe") or {},
        "next_steps": _next_steps(failed),
    }


def write_gap_report(root: Path | str, output: Path | str | None = None) -> dict[str, Any]:
    root = Path(root)
    output = Path(output) if output else root / "completion_gap_report.md"
    report = build_gap_report(root)
    output.write_text(_render_markdown(report), encoding="utf-8")
    return {"output": str(output), **report}


def _next_steps(failed_checks: list[dict[str, Any]]) -> list[str]:
    names = {check.get("name") for check in failed_checks}
    steps: list[str] = []
    if "finished" in names or "ran_min_hours" in names:
        steps.append(
            "Resume or rerun the 12h optimization if strict completion evidence is required; "
            "the current package is only early-stop review-ready."
        )
    if any(name not in {"finished", "ran_min_hours"} for name in names):
        steps.append("Resolve non-time-gate audit failures before claiming a stable final package.")
    if not steps:
        steps.append("No completion gaps remain.")
    return steps


def _render_markdown(report: dict[str, Any]) -> str:
    status = report.get("status") or {}
    conclusion = report.get("conclusion") or {}
    proposal_gate = report.get("proposal_gate") or {}
    proposal_frontiers = report.get("proposal_frontiers") or {}
    frontier_probe = report.get("frontier_proposal_probe") or {}
    failed = report.get("failed_checks") or []
    next_steps = report.get("next_steps") or []
    elapsed = report.get("elapsed_hours")
    elapsed_text = f"{elapsed:.3f} h" if isinstance(elapsed, (int, float)) else "unavailable"
    lines = [
        "# Bufotalin Completion Gap Report",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Result root: `{report.get('root')}`",
        "",
        "## Status",
        "",
        f"- Strict 12h complete: `{str(report.get('strict_12h_complete')).lower()}`",
        f"- Early-stop review-ready: `{str(report.get('early_stop_review_ready')).lower()}`",
        f"- Elapsed hours used for strict audit: `{elapsed_text}`",
        f"- Stop reason: `{status.get('stop_reason')}`",
        f"- Completed payloads: {status.get('completed_payload_count')}",
        f"- Native-search payloads: {status.get('native_success_payloads')}",
        f"- Running cycles: {status.get('running_cycle_count')}",
        "",
        "## Current Route Conclusion",
        "",
        f"- High-confidence route count: {conclusion.get('high_confidence_route_count')}",
        f"- Stitched semisynthesis upstream review-only route count: {conclusion.get('stitched_review_only_route_count')}",
        f"- Review-only native route count: {conclusion.get('review_only_route_count')}",
        f"- Excluded route count: {conclusion.get('excluded_route_count')}",
        f"- Main route: {conclusion.get('main_route_summary')}",
        "",
        "## Proposal Gate Filtering",
        "",
        f"- Available: `{str(bool(proposal_gate.get('available'))).lower()}`",
        f"- Input routes: {proposal_gate.get('input_routes')}",
        f"- Kept routes: {proposal_gate.get('kept_routes')}",
        f"- Dropped routes: {proposal_gate.get('dropped_routes')}",
        f"- Source-supported frontier repairs: {proposal_gate.get('repaired_routes')}",
        f"- Repair reasons: {_format_reason_counts(proposal_gate.get('repair_reason_counts') or {})}",
        f"- Top rejection reasons: {_format_reason_counts(proposal_gate.get('reason_counts') or {})}",
        "",
        "## Proposal Frontier Analysis",
        "",
        f"- Available: `{str(bool(proposal_frontiers.get('available'))).lower()}`",
        f"- Dropped rows with frontier: {proposal_frontiers.get('dropped_rows_with_frontier')}",
        f"- Unique frontiers: {proposal_frontiers.get('unique_frontiers')}",
        f"- Complex-core-like frontiers: {proposal_frontiers.get('complex_core_frontier_count')}",
        f"- Unsupported prenyl frontiers: {proposal_frontiers.get('unsupported_prenyl_frontier_count')}",
        f"- Top frontiers: {_format_top_frontiers(proposal_frontiers.get('top_frontiers') or [])}",
        "",
        "## Frontier Proposal Probe",
        "",
        f"- Available: `{str(bool(frontier_probe.get('available'))).lower()}`",
        f"- Frontier count: {frontier_probe.get('frontier_count')}",
        f"- Proposal count: {frontier_probe.get('proposal_count')}",
        f"- Gate-pass proposals: {frontier_probe.get('gate_keep_count')}",
        f"- Gate-reject proposals: {frontier_probe.get('gate_reject_count')}",
        "",
        "## Failed Strict-Completion Checks",
        "",
    ]
    if failed:
        for check in failed:
            lines.append(f"- `{check.get('name')}`: {check.get('evidence')}")
    else:
        lines.append("- None")
    lines.extend(["", "## Next Steps", ""])
    for step in next_steps:
        lines.append(f"- {step}")
    lines.extend(
        [
            "",
            "## Revalidation",
            "",
            "```bash",
            f"python scripts/audit_bufotalin_12h_completion.py {report.get('root')} || true",
            f"python scripts/audit_bufotalin_early_stop_review.py {report.get('root')}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _format_reason_counts(reason_counts: dict[str, Any], *, limit: int = 8) -> str:
    if not reason_counts:
        return "none"
    rows = sorted(reason_counts.items(), key=lambda item: (-int(item[1]), item[0]))[: int(limit)]
    return "; ".join(f"{key}={value}" for key, value in rows)


def _format_top_frontiers(frontiers: list[dict[str, Any]], *, limit: int = 3) -> str:
    if not frontiers:
        return "none"
    rows = []
    for row in frontiers[: int(limit)]:
        profile = row.get("profile") or {}
        formula = profile.get("formula") or "unknown"
        rows.append(f"count={row.get('count')} formula={formula} smiles={row.get('smiles')}")
    return " | ".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a bufotalin strict-completion gap report.")
    parser.add_argument("root")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = write_gap_report(args.root, output=args.output or None)
    print(
        json.dumps(
            {
                "output": result["output"],
                "strict_12h_complete": result["strict_12h_complete"],
                "early_stop_review_ready": result["early_stop_review_ready"],
                "failed_check_count": result["failed_check_count"],
                "elapsed_hours": result["elapsed_hours"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

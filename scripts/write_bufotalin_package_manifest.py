"""Write a machine-readable manifest for a bufotalin result package."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_bufotalin_final_candidate_quality import audit_final_candidate_quality
from scripts.audit_bufotalin_early_stop_review import audit_early_stop_review
from scripts.write_bufotalin_status_snapshot import build_status_snapshot


CORE_FILES = [
    "README.md",
    "completion_gap_report.md",
    "early_stop_result_report.md",
    "status_snapshot.json",
    "early_stop_review_audit.json",
    "final_candidate_quality_audit.json",
    "cycle_proposal_gate_retrofit_summary.json",
    "proposal_frontier_analysis.json",
    "frontier_proposal_probe.json",
    "manifest.json",
    "runner_events.jsonl",
    "final_candidates/final_candidates.md",
    "final_candidates/final_candidates.json",
    "final_candidates/final_candidates_payload.json",
    "final_candidates/figures/index.html",
    "final_candidates/figures/manifest.json",
]


def build_package_manifest(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    snapshot = build_status_snapshot(root)
    audit = audit_early_stop_review(root)
    quality = audit_final_candidate_quality(root)
    final_candidates = _read_json(root / "final_candidates" / "final_candidates.json")
    figure_manifest = _read_json(root / "final_candidates" / "figures" / "manifest.json")
    manifest = _read_json(root / "manifest.json")
    proposal_gate = _read_json(root / "cycle_proposal_gate_retrofit_summary.json")
    proposal_frontiers = _read_json(root / "proposal_frontier_analysis.json")
    frontier_probe = _read_json(root / "frontier_proposal_probe.json")

    files = []
    for rel in CORE_FILES:
        path = root / rel
        files.append(
            {
                "path": rel,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    selected = final_candidates.get("selected") or []
    high_confidence = [
        row for row in selected
        if isinstance(row, dict) and row.get("confidence_tier") == "high_confidence_source_supported"
    ]
    stitched_review = [
        row for row in selected
        if isinstance(row, dict) and row.get("confidence_tier") == "stitched_semisynthesis_upstream_review_only"
    ]
    review_only = [
        row for row in selected
        if isinstance(row, dict) and row.get("confidence_tier") == "native_model_candidate_review_only"
    ]
    return {
        "schema_version": "bufotalin_package_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "target": manifest.get("target"),
        "status": {
            "strict_12h_complete": bool(snapshot.get("headline", {}).get("complete")),
            "early_stop_review_ready": bool(audit.get("review_ready")),
            "status_label": snapshot.get("headline", {}).get("status_label"),
            "stop_reason": manifest.get("stop_reason"),
            "stopped_at": manifest.get("stopped_at"),
            "completed_payload_count": snapshot.get("headline", {}).get("completed_payload_count"),
            "running_cycle_count": snapshot.get("headline", {}).get("running_cycle_count"),
            "native_success_payloads": snapshot.get("headline", {}).get("native_success_payloads"),
        },
        "conclusion": {
            "high_confidence_route_count": len(high_confidence),
            "stitched_review_only_route_count": len(stitched_review),
            "review_only_route_count": len(review_only),
            "excluded_route_count": final_candidates.get("excluded_route_count"),
            "main_route_summary": (
                "Deacetylbufotalin / Bufogenin B + acetic anhydride -> Bufotalin; "
                "Ac2O, catalytic DMAP, CH2Cl2, 0-25 C"
                if high_confidence
                else ""
            ),
            "native_routes_position": "review_only",
        },
        "proposal_gate": _proposal_gate_summary(proposal_gate),
        "proposal_frontiers": _proposal_frontier_summary(proposal_frontiers),
        "frontier_proposal_probe": _frontier_probe_summary(frontier_probe),
        "figures": {
            "figure_count": len(figure_manifest.get("figures") or []),
            "index": "final_candidates/figures/index.html",
        },
        "core_files": files,
        "all_core_files_present": all(item["exists"] for item in files),
        "audits": {
            "early_stop_review": {
                "review_ready": bool(audit.get("review_ready")),
                "path": "early_stop_review_audit.json",
            },
            "strict_12h_completion": {
                "complete": bool(snapshot.get("headline", {}).get("complete")),
                "failed_checks": snapshot.get("headline", {}).get("failed_checks") or [],
            },
            "final_candidate_quality": {
                "passed": bool(quality.get("passed")),
                "path": "final_candidate_quality_audit.json",
            },
        },
    }


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
        "repaired_routes": report.get("repaired_routes"),
        "repair_reason_counts": report.get("repair_reason_counts") or {},
        "reason_counts": report.get("reason_counts") or {},
    }


def _proposal_frontier_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {"available": False}
    summary = report.get("summary") or {}
    return {
        "available": True,
        "dropped_rows_with_frontier": summary.get("dropped_rows_with_frontier"),
        "unique_frontiers": summary.get("unique_frontiers"),
        "complex_core_frontier_count": summary.get("complex_core_frontier_count"),
        "unsupported_prenyl_frontier_count": summary.get("unsupported_prenyl_frontier_count"),
        "top_frontiers": (report.get("top_frontiers") or [])[:5],
    }


def _frontier_probe_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {"available": False}
    summary = report.get("summary") or {}
    return {
        "available": True,
        "frontier_count": summary.get("frontier_count"),
        "proposal_count": summary.get("proposal_count"),
        "gate_keep_count": summary.get("gate_keep_count"),
        "gate_reject_count": summary.get("gate_reject_count"),
        "elapsed_s": summary.get("elapsed_s"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write bufotalin package_manifest.json.")
    parser.add_argument("root")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "package_manifest.json"
    manifest = build_package_manifest(root)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "early_stop_review_ready": manifest["status"]["early_stop_review_ready"],
                "strict_12h_complete": manifest["status"]["strict_12h_complete"],
                "all_core_files_present": manifest["all_core_files_present"],
                "figure_count": manifest["figures"]["figure_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

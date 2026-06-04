"""Audit whether a stopped bufotalin run has a review-ready result package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_bufotalin_final_candidate_quality import audit_final_candidate_quality
from scripts.write_bufotalin_status_snapshot import build_status_snapshot


def audit_early_stop_review(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    snapshot = build_status_snapshot(root)
    headline = snapshot.get("headline") or {}
    early_stop = snapshot.get("early_stop") or {}
    final_candidates = snapshot.get("final_candidates") or {}
    manifest = _read_json(root / "manifest.json")
    quality = audit_final_candidate_quality(root)

    checks = [
        _check("root_exists", root.exists(), f"{root} exists"),
        _check(
            "strict_12h_goal_not_claimed",
            not bool(headline.get("complete")),
            "strict 12h completion remains false for an early-stop review package",
        ),
        _check(
            "only_time_gate_failed",
            bool(early_stop.get("only_time_gate_failed")),
            "strict completion failed only on finished/ran_min_hours gates",
        ),
        _check(
            "no_active_workers",
            bool(early_stop.get("no_active_workers")),
            "no bufotalin worker/watch processes and no running cycles remain",
        ),
        _check(
            "manifest_records_user_stop",
            manifest.get("running") is False
            and manifest.get("stopped") is True
            and manifest.get("stop_reason") == "user_cancelled",
            "manifest records a user-cancelled stopped run",
        ),
        _check(
            "has_completed_payloads",
            int(headline.get("completed_payload_count") or 0) > 0,
            "at least one completed payload is available",
        ),
        _check(
            "has_native_search_evidence",
            int(headline.get("native_success_payloads") or 0) > 0,
            "at least one native ChemEnzy search payload returned routes",
        ),
        _check(
            "has_high_confidence_final_route",
            int(final_candidates.get("high_confidence_count") or 0) > 0,
            "final_candidates contains at least one high-confidence route",
        ),
        _check(
            "has_selected_final_routes",
            int(final_candidates.get("selected_count") or 0) > 0,
            "final_candidates contains selected routes for review",
        ),
        _check(
            "final_candidate_quality_gate",
            bool(quality.get("passed")),
            "final_candidates satisfy review-quality gates",
        ),
        _check(
            "status_snapshot_review_ready",
            bool(headline.get("early_stop_review_ready"))
            and headline.get("status_label") == "early_stop_review_ready",
            "status_snapshot marks this stopped run as early_stop_review_ready",
        ),
    ]
    review_ready = all(check["passed"] for check in checks)
    return {
        "root": str(root),
        "review_ready": review_ready,
        "checks": checks,
        "headline": headline,
        "early_stop": early_stop,
        "final_candidates": final_candidates,
        "final_candidate_quality": quality,
        "manifest_stop": {
            "running": manifest.get("running"),
            "stopped": manifest.get("stopped"),
            "stop_reason": manifest.get("stop_reason"),
            "stopped_at": manifest.get("stopped_at"),
        },
    }


def _check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bufotalin early-stop review readiness.")
    parser.add_argument("root")
    args = parser.parse_args()
    report = audit_early_stop_review(args.root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["review_ready"] else 1)


if __name__ == "__main__":
    main()

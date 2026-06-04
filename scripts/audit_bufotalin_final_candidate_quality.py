"""Audit final-candidate quality gates for the bufotalin result package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BUFOTALIN_REVIEW_TIERS = {
    "stitched_semisynthesis_upstream_review_only",
    "native_model_candidate_review_only",
}
BUFOTALIN_ALLOWED_REVIEW_WARNINGS = {
    "rcr_condition_prediction_only",
    "low_condition_prediction_score",
}
BUFOTALIN_HIGH_RISK_WARNINGS = {
    "non_mild_predicted_temperature",
    "strong_hydride_reagent_predicted",
    "circular_or_target_as_terminal_stock",
    "unsupported_biosynthetic_prenyl_terminal",
}


def audit_final_candidate_quality(root: Path | str, *, min_review_steps: int = 3) -> dict[str, Any]:
    root = Path(root)
    summary = _read_json(root / "final_candidates" / "final_candidates.json")
    payload = _read_json(root / "final_candidates" / "final_candidates_payload.json")
    figure_manifest = _read_json(root / "final_candidates" / "figures" / "manifest.json")
    routes = [route for route in payload.get("routes") or [] if isinstance(route, dict)]
    selected = [row for row in summary.get("selected") or [] if isinstance(row, dict)]
    route_finals = [(route.get("final_candidate") or {}) for route in routes]
    review_routes = [
        (route, final)
        for route, final in zip(routes, route_finals)
        if str(final.get("confidence_tier") or "") in BUFOTALIN_REVIEW_TIERS
    ]
    high_confidence = [
        final for final in route_finals
        if final.get("confidence_tier") == "high_confidence_source_supported"
    ]
    high_risk_rows = [
        {
            "source_cycle": final.get("source_cycle"),
            "source_route_index": final.get("source_route_index"),
            "confidence_tier": final.get("confidence_tier"),
            "warnings": sorted(set(final.get("warnings") or []) & BUFOTALIN_HIGH_RISK_WARNINGS),
        }
        for final in route_finals
        if set(final.get("warnings") or []) & BUFOTALIN_HIGH_RISK_WARNINGS
    ]
    short_review_rows = [
        {
            "source_cycle": final.get("source_cycle"),
            "source_route_index": final.get("source_route_index"),
            "confidence_tier": final.get("confidence_tier"),
            "n_steps": _route_step_count(route),
        }
        for route, final in review_routes
        if _route_step_count(route) < int(min_review_steps)
    ]
    disallowed_warning_rows = [
        {
            "source_cycle": final.get("source_cycle"),
            "source_route_index": final.get("source_route_index"),
            "confidence_tier": final.get("confidence_tier"),
            "warnings": sorted(set(final.get("warnings") or []) - BUFOTALIN_ALLOWED_REVIEW_WARNINGS),
        }
        for _route, final in review_routes
        if set(final.get("warnings") or []) - BUFOTALIN_ALLOWED_REVIEW_WARNINGS
    ]
    target_terminal_rows = [
        {
            "source_cycle": final.get("source_cycle"),
            "source_route_index": final.get("source_route_index"),
            "confidence_tier": final.get("confidence_tier"),
        }
        for final in route_finals
        if final.get("target_terminal")
        or "terminal_reactants_include_target" in (final.get("exclusion_reasons") or [])
    ]

    checks = [
        _check("root_exists", root.exists(), f"{root} exists"),
        _check("final_summary_exists", bool(summary), "final_candidates.json is readable"),
        _check("final_payload_exists", bool(payload), "final_candidates_payload.json is readable"),
        _check("selected_count_matches_payload", int(summary.get("selected_count") or 0) == len(routes), "summary selected_count equals payload route count"),
        _check("selected_summary_matches_payload", len(selected) == len(routes), "summary selected rows equal payload route count"),
        _check("figures_match_selected", len(figure_manifest.get("figures") or []) == len(routes) and len(routes) > 0, "rendered figure count equals selected route count"),
        _check("has_one_high_confidence_route", len(high_confidence) == 1, "exactly one high-confidence source-supported route is selected"),
        _check(
            "high_confidence_is_source_supported_only",
            all(final.get("source_supported_semisynthesis") for final in high_confidence),
            "high-confidence routes are source-supported semisynthesis only",
        ),
        _check("has_stitched_review_routes", int(summary.get("stitched_review_only_count") or 0) > 0, "stitched semisynthesis upstream review routes are present"),
        _check("has_native_review_routes", int(summary.get("native_review_only_count") or 0) > 0, "native review-only routes are present"),
        _check("review_routes_min_steps", not short_review_rows, f"review-only routes have at least {int(min_review_steps)} steps"),
        _check("no_target_terminal_selected", not target_terminal_rows, "selected routes do not use the target as a terminal reactant"),
        _check("no_high_risk_warnings_selected", not high_risk_rows, "selected routes do not contain high-risk condition warnings"),
        _check("review_warning_set_allowed", not disallowed_warning_rows, "review-only warnings are limited to allowed RCR confidence warnings"),
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": "bufotalin_final_candidate_quality_audit.v1",
        "root": str(root),
        "passed": passed,
        "checks": checks,
        "metrics": {
            "selected_count": len(routes),
            "high_confidence_count": len(high_confidence),
            "stitched_review_only_count": int(summary.get("stitched_review_only_count") or 0),
            "native_review_only_count": int(summary.get("native_review_only_count") or 0),
            "figure_count": len(figure_manifest.get("figures") or []),
            "min_review_steps": int(min_review_steps),
        },
        "violations": {
            "short_review_routes": short_review_rows,
            "target_terminal_routes": target_terminal_rows,
            "high_risk_warning_routes": high_risk_rows,
            "disallowed_warning_routes": disallowed_warning_rows,
        },
    }


def _route_step_count(route: dict[str, Any]) -> int:
    return int(route.get("n_steps") or len(route.get("steps") or []))


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
    parser = argparse.ArgumentParser(description="Audit bufotalin final-candidate quality gates.")
    parser.add_argument("root")
    parser.add_argument("--min-review-steps", type=int, default=3)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = audit_final_candidate_quality(args.root, min_review_steps=args.min_review_steps)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

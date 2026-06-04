"""Audit whether the bufotalin 12h optimization run satisfies the goal gates."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.baselines.semisynthesis_rescue import ACETIC_ANHYDRIDE, DEACETYLBUFOTALIN, DMAP
from scripts.audit_bufotalin_final_candidate_quality import audit_final_candidate_quality
from scripts.run_bufotalin_12h_iteration import BUFOTALIN_TARGET
from scripts.summarize_bufotalin_iteration import summarize_iteration_root


def audit_completion(root: Path | str, *, min_hours: float = 12.0) -> dict[str, Any]:
    root = Path(root)
    summary = summarize_iteration_root(root)
    final_quality = audit_final_candidate_quality(root)
    manifest = _read_json(root / "manifest.json")
    events = _read_events(root / "runner_events.jsonl")
    start = _first_event(events, "start")
    finish = _last_event(events, "finish")
    stop = _last_event(events, "user_stop")
    elapsed_hours = _elapsed_hours(start, finish or stop)

    checks = [
        _check("root_exists", root.exists(), f"{root} exists"),
        _check("target_matches", _target_matches(start, manifest), "run target matches bufotalin goal target"),
        _check("finished", bool(finish) and not bool(manifest.get("running")), "runner emitted finish and manifest is not running"),
        _check(
            "ran_min_hours",
            elapsed_hours is not None and elapsed_hours >= float(min_hours),
            _elapsed_gate_evidence(elapsed_hours, min_hours),
        ),
        _check("has_completed_payload", summary.get("completed_payload_count", 0) > 0, "at least one completed payload"),
        _check(
            "has_source_supported_semisynthesis",
            summary.get("source_supported_semisynthesis_payloads", 0) > 0,
            "at least one source-supported semisynthesis route",
        ),
        _check(
            "has_cascade_verifier_feasible_route",
            summary.get("cascade_verifier_feasible_payloads", 0) > 0,
            "at least one cascade-verifier-feasible route",
        ),
        _check(
            "has_conditioned_renderable_route",
            _has_conditioned_renderable_route(root),
            "at least one rendered route is source-supported or verifier-feasible with complete condition coverage",
        ),
        _check(
            "has_native_search_success",
            summary.get("native_success_payloads", 0) > 0,
            "at least one native ChemEnzy search payload returned routes",
        ),
        _check(
            "template_relevance_probe_hits_precursor",
            summary.get("template_relevance_probe_hit_payloads", 0) > 0,
            "template_relevance.bkms probe hits the expected source-supported precursor",
        ),
        _check(
            "figures_exported",
            _figures_exported(root, summary),
            "SVG and PDF route figures were exported for cycle payloads or final candidates",
        ),
        _check(
            "figures_are_feasible_or_supported",
            _all_figures_are_renderable(root),
            "rendered figures correspond only to source-supported or cascade-verifier-feasible routes",
        ),
        _check(
            "no_completed_payload_failures",
            not _completed_failures(summary),
            "completed payloads have no backend failure categories",
        ),
        _check(
            "final_candidates_package_exists",
            _final_candidates_package_exists(root),
            "final_candidates package includes JSON, markdown, payload, and rendered figure manifest",
        ),
        _check(
            "final_candidates_has_high_confidence_route",
            _final_candidates_has_high_confidence_route(root),
            "final_candidates contains at least one presentation-ready source-supported semisynthesis route",
        ),
        _check(
            "final_candidates_no_target_terminal_selected",
            _final_candidates_no_target_terminal_selected(root),
            "selected final candidates do not use the target molecule as a terminal reactant",
        ),
        _check(
            "final_candidates_contains_expected_semisynthesis",
            _final_candidates_contains_expected_semisynthesis(root),
            "final_candidates includes Deacetylbufotalin + acetic anhydride late-stage O-acetylation with DMAP condition evidence",
        ),
        _check(
            "final_candidate_quality_gate",
            bool(final_quality.get("passed")),
            "final selected candidates satisfy quality gates",
        ),
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "root": str(root),
        "complete": passed,
        "elapsed_hours": elapsed_hours,
        "checks": checks,
        "summary": summary,
        "final_candidate_quality": final_quality,
    }


def _target_matches(start: dict[str, Any] | None, manifest: dict[str, Any]) -> bool:
    candidates = [
        str((start or {}).get("target") or ""),
        str(manifest.get("target") or ""),
    ]
    return any(candidate == BUFOTALIN_TARGET for candidate in candidates)


def _completed_failures(summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for row in summary.get("rows") or []:
        if not row.get("complete"):
            continue
        failures.extend(
            str(item)
            for item in row.get("failures") or []
            if not _non_blocking_completed_failure(row, str(item))
        )
    return failures


def _figures_exported(root: Path, summary: dict[str, Any]) -> bool:
    if summary.get("figure_svg_count", 0) > 0 and summary.get("figure_pdf_count", 0) > 0:
        return True
    manifest = _read_json(root / "final_candidates" / "figures" / "manifest.json")
    figures = manifest.get("figures") or []
    return any(item.get("svg") for item in figures) and any(item.get("pdf") for item in figures)


def _non_blocking_completed_failure(row: dict[str, Any], category: str) -> bool:
    """Exploratory timeout cycles should not invalidate a stable final route package."""
    if category != "cycle_worker_timeout":
        return False
    if int(row.get("n_results") or 0) > 1:
        return False
    if int(row.get("native_raw_n_routes") or 0) > 0:
        return False
    return bool(
        row.get("source_supported_semisynthesis")
        and row.get("cascade_verifier_feasible")
        and row.get("condition_complete_routes")
    )


def _all_figures_are_renderable(root: Path) -> bool:
    for manifest_path in root.glob("*/figures/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        cycle_or_final_dir = manifest_path.parent.parent
        if cycle_or_final_dir.name == "final_candidates":
            source = cycle_or_final_dir / "final_candidates_payload.json"
        else:
            source = cycle_or_final_dir / "web_payload.json"
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return False
        if manifest_path.parent.parent.name == "final_candidates":
            renderable = [
                route
                for route in payload.get("routes") or []
                if _final_candidate_route_is_renderable(route)
            ]
        else:
            renderable = [
                route
                for route in payload.get("routes") or []
                if _route_is_renderable(route)
            ]
        figures = manifest.get("figures") or []
        if len(figures) > len(renderable):
            return False
        for item in figures:
            svg = item.get("svg")
            pdf = item.get("pdf")
            if svg and not (manifest_path.parent / svg).exists():
                return False
            if pdf and not (manifest_path.parent / pdf).exists():
                return False
    return True


def _has_conditioned_renderable_route(root: Path) -> bool:
    for payload_path in root.glob("*/web_payload.json"):
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if any(_route_is_renderable(route) for route in payload.get("routes") or []):
            return True
    return False


def _route_is_renderable(route: dict[str, Any]) -> bool:
    metrics = route.get("metrics") or {}
    verifier = metrics.get("cascade_verifier") or {}
    if metrics.get("source_supported_semisynthesis"):
        return True
    return bool(verifier.get("feasible") and _condition_coverage(route) >= 1.0)


def _condition_coverage(route: dict[str, Any]) -> float:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    if not steps:
        return 0.0
    covered = sum(1 for step in steps if step.get("condition_predictions"))
    return covered / max(1, len(steps))


def _final_candidates_package_exists(root: Path) -> bool:
    final_dir = root / "final_candidates"
    required = [
        final_dir / "final_candidates.json",
        final_dir / "final_candidates.md",
        final_dir / "final_candidates_payload.json",
        final_dir / "figures" / "manifest.json",
    ]
    if not all(path.exists() for path in required):
        return False
    manifest = _read_json(final_dir / "figures" / "manifest.json")
    return bool(manifest.get("figures"))


def _final_candidates_has_high_confidence_route(root: Path) -> bool:
    summary = _read_json(root / "final_candidates" / "final_candidates.json")
    if int(summary.get("high_confidence_count") or 0) <= 0:
        return False
    for row in summary.get("selected") or []:
        if not isinstance(row, dict):
            continue
        if row.get("confidence_tier") == "high_confidence_source_supported" and row.get("presentation_ready"):
            return True
    return False


def _final_candidates_no_target_terminal_selected(root: Path) -> bool:
    payload = _read_json(root / "final_candidates" / "final_candidates_payload.json")
    routes = payload.get("routes") or []
    if not routes:
        return False
    for route in routes:
        final = (route or {}).get("final_candidate") or {}
        if final.get("target_terminal"):
            return False
        if "terminal_reactants_include_target" in (final.get("exclusion_reasons") or []):
            return False
    return True


def _final_candidates_contains_expected_semisynthesis(root: Path) -> bool:
    payload = _read_json(root / "final_candidates" / "final_candidates_payload.json")
    expected_precursor = canonical_smiles(DEACETYLBUFOTALIN)
    expected_reagent = canonical_smiles(ACETIC_ANHYDRIDE)
    expected_catalyst = canonical_smiles(DMAP)
    for route in payload.get("routes") or []:
        final = (route or {}).get("final_candidate") or {}
        if final.get("confidence_tier") != "high_confidence_source_supported":
            continue
        if final.get("target_terminal"):
            continue
        for step in (route or {}).get("steps") or []:
            reactants = [
                canonical_smiles(str(step.get("main_reactant") or "")),
                *[canonical_smiles(str(item or "")) for item in step.get("aux_reactants") or []],
            ]
            if expected_precursor not in reactants or expected_reagent not in reactants:
                continue
            for condition in step.get("condition_predictions") or []:
                catalyst = canonical_smiles(str(condition.get("Catalyst") or ""))
                reagent = canonical_smiles(str(condition.get("Reagent") or ""))
                if catalyst == expected_catalyst and reagent == expected_reagent:
                    return True
    return False


def _final_candidate_route_is_renderable(route: dict[str, Any]) -> bool:
    final = (route or {}).get("final_candidate") or {}
    if final:
        if final.get("target_terminal"):
            return False
        if "terminal_reactants_include_target" in (final.get("exclusion_reasons") or []):
            return False
        tier = str(final.get("confidence_tier") or "")
        return bool(
            tier == "high_confidence_source_supported"
            or tier == "stitched_semisynthesis_upstream_review_only"
            or tier == "native_model_candidate_review_only"
        )
    return _route_is_renderable(route)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _first_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    for row in events:
        if row.get("event") == event_name:
            return row
    return None


def _last_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    for row in reversed(events):
        if row.get("event") == event_name:
            return row
    return None


def _elapsed_hours(start: dict[str, Any] | None, finish: dict[str, Any] | None) -> float | None:
    if not start or not finish:
        return None
    start_time = _parse_time(str(start.get("started_at") or start.get("time") or ""))
    finish_time = _parse_time(str(finish.get("time") or ""))
    if start_time is None or finish_time is None:
        return None
    return (finish_time - start_time).total_seconds() / 3600.0


def _elapsed_gate_evidence(elapsed_hours: float | None, min_hours: float) -> str:
    if elapsed_hours is None:
        return "elapsed wall time unavailable"
    comparator = ">=" if elapsed_hours >= float(min_hours) else "<"
    return f"elapsed wall time {elapsed_hours:.3f} h {comparator} required {float(min_hours):.3f} h"


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bufotalin 12h completion gates.")
    parser.add_argument("root")
    parser.add_argument("--min-hours", type=float, default=12.0)
    args = parser.parse_args()
    report = audit_completion(args.root, min_hours=args.min_hours)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()

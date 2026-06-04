"""End-to-end SMILES-first literature strategic workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascade_planner.agent.case_trace import (
    case_bundle_from_p0_outputs,
    write_case_bundle,
)
from cascade_planner.agent.evidence_cards import (
    validation_summary,
    write_evidence_jsonl,
)
from cascade_planner.agent.literature_research import (
    build_triggered_literature_task,
    render_literature_report,
    retrieve_literature_evidence,
)
from cascade_planner.agent.route_package import (
    build_hybrid_route_package,
    render_route_map_svg,
    render_summary,
    validate_route_package,
    write_json,
)
from cascade_planner.agent.strategic_disconnection_miner import (
    mine_strategic_disconnection_cards,
    write_strategic_disconnection_cards_jsonl,
)
from cascade_planner.agent.strategic_candidate_generation import (
    generate_literature_candidates,
    write_candidates_jsonl,
)
from cascade_planner.agent.target_profile import (
    build_frontier_report,
    build_target_profile,
)


@dataclass
class SmilesFirstWorkflowConfig:
    target_smiles: str
    target_name: str = ""
    family_hint: str = ""
    objective: str = "route"
    output_dir: str | Path = "results/shared/smiles_first_literature_workflow"
    frontier_smiles: str = ""
    baseline_json: str | Path | None = None
    evidence_jsonl: str | Path | None = None
    db_paths: list[str | Path] | None = None
    query_budget: int = 12


def run_smiles_first_workflow(config: SmilesFirstWorkflowConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    profile = build_target_profile(
        config.target_smiles,
        target_name=config.target_name,
        family_hint=config.family_hint,
    )
    write_json(output_dir / "target_profile.json", profile.to_dict())
    if not profile.valid:
        validation = {
            "schema_version": "route_package_validation.v1",
            "case_id": profile.case_id,
            "accepted": False,
            "route_status": "invalid_package",
            "reasons": ["invalid_target_smiles"],
        }
        write_json(output_dir / "validation.json", validation)
        (output_dir / "summary.md").write_text(
            "# SMILES-First Literature Route Package Summary\n\n"
            "- Route status: `invalid_package`\n"
            "- Reason: invalid target SMILES.\n",
            encoding="utf-8",
        )
        return {"case_id": profile.case_id, "output_dir": str(output_dir), "validation": validation}

    baseline_routes = _load_baseline(config.baseline_json)
    write_json(output_dir / "baseline_routes.json", baseline_routes)
    frontier_report = build_frontier_report(
        profile,
        frontier_smiles=config.frontier_smiles,
        baseline_routes=baseline_routes,
    )
    write_json(output_dir / "frontier_report.json", frontier_report)
    primary_frontier = _primary_frontier_smiles(frontier_report, profile)

    user_requested_literature = not _baseline_solved_audit_passed(baseline_routes)
    task, trigger_report = build_triggered_literature_task(
        profile,
        primary_frontier,
        native_result=baseline_routes,
        route_audit=_route_audit_from_baseline(baseline_routes),
        frontier_report=frontier_report,
        user_requested=user_requested_literature,
        query_budget=config.query_budget,
    )
    write_json(output_dir / "literature_trigger_report.json", trigger_report)
    if task is None:
        validation = {
            "schema_version": "route_package_validation.v1",
            "case_id": profile.case_id,
            "accepted": True,
            "route_status": "native_solved_no_literature_trigger",
            "reasons": [],
            "trigger_report": trigger_report,
            "guards": {
                "native_solved_audit_passed_skips_literature": True,
                "p0_not_solved_without_stock_audit": True,
            },
        }
        write_json(output_dir / "literature_search_task.json", {"skipped": True, "trigger_report": trigger_report})
        write_evidence_jsonl([], output_dir / "evidence_cards.jsonl")
        write_json(output_dir / "literature_search_report.json", {
            "schema_version": "literature_search_report.v1",
            "case_id": profile.case_id,
            "skipped": True,
            "skip_reason": "native_solved_audit_passed",
            "trigger_report": trigger_report,
        })
        (output_dir / "literature_search_report.md").write_text(
            "# Literature Search Report\n\n- Skipped: `native_solved_audit_passed`\n",
            encoding="utf-8",
        )
        package = _native_solved_skip_package(profile, frontier_report, baseline_routes, validation)
        package_path = output_dir / f"{profile.case_id}_hybrid_retrosynthesis_route.json"
        write_json(package_path, package)
        write_json(output_dir / "validation.json", validation)
        (output_dir / "summary.md").write_text(render_summary(package, validation), encoding="utf-8")
        (figures_dir / f"{profile.case_id}_retrosynthesis_map.svg").write_text(
            render_route_map_svg(package),
            encoding="utf-8",
        )
        return {
            "case_id": profile.case_id,
            "output_dir": str(output_dir),
            "artifacts": {
                "target_profile": str(output_dir / "target_profile.json"),
                "baseline_routes": str(output_dir / "baseline_routes.json"),
                "frontier_report": str(output_dir / "frontier_report.json"),
                "literature_trigger_report": str(output_dir / "literature_trigger_report.json"),
                "literature_search_report": str(output_dir / "literature_search_report.md"),
                "evidence_cards": str(output_dir / "evidence_cards.jsonl"),
                "hybrid_route_package": str(package_path),
                "validation": str(output_dir / "validation.json"),
                "summary": str(output_dir / "summary.md"),
                "route_map": str(figures_dir / f"{profile.case_id}_retrosynthesis_map.svg"),
            },
            "validation": validation,
        }
    write_json(output_dir / "literature_search_task.json", task.to_dict())
    evidence_cards, literature_report = retrieve_literature_evidence(
        task,
        manual_evidence_jsonl=config.evidence_jsonl,
        db_paths=config.db_paths,
    )
    for card in evidence_cards:
        if not card.case_id:
            card.case_id = profile.case_id
    write_evidence_jsonl(evidence_cards, output_dir / "evidence_cards.jsonl")
    literature_report["evidence_validation"] = validation_summary(evidence_cards)
    write_json(output_dir / "literature_search_report.json", literature_report)
    (output_dir / "literature_search_report.md").write_text(
        render_literature_report(literature_report, evidence_cards),
        encoding="utf-8",
    )

    disconnection_cards = mine_strategic_disconnection_cards(
        case_id=profile.case_id,
        target_smiles=profile.isomeric_smiles,
        frontier_smiles=primary_frontier,
        evidence_cards=evidence_cards,
    )
    disconnection_path = output_dir / f"{profile.case_id}_strategic_disconnection_cards.jsonl"
    write_strategic_disconnection_cards_jsonl(disconnection_cards, disconnection_path)

    candidates = generate_literature_candidates(
        case_id=profile.case_id,
        target_smiles=profile.isomeric_smiles,
        frontier_smiles=primary_frontier,
        evidence_cards=evidence_cards,
    )
    candidate_path = output_dir / f"{profile.case_id}_literature_rxn_candidates.jsonl"
    write_candidates_jsonl(candidates, candidate_path)

    package = build_hybrid_route_package(
        profile=profile,
        frontier_report=frontier_report,
        evidence_cards=evidence_cards,
        candidates=candidates,
        baseline_routes=baseline_routes,
    )
    validation = validate_route_package(package, evidence_cards=evidence_cards, candidates=candidates)
    package["route_status"] = validation["route_status"]
    package_path = output_dir / f"{profile.case_id}_hybrid_retrosynthesis_route.json"
    write_json(package_path, package)
    write_json(output_dir / "validation.json", validation)
    (output_dir / "summary.md").write_text(render_summary(package, validation), encoding="utf-8")
    (figures_dir / f"{profile.case_id}_retrosynthesis_map.svg").write_text(
        render_route_map_svg(package),
        encoding="utf-8",
    )
    case_bundle = case_bundle_from_p0_outputs(
        route_package=package,
        validation=validation,
        evidence_cards=[card.to_dict() for card in evidence_cards],
        candidate_cards=[candidate.to_dict() for candidate in candidates],
        strategic_disconnection_cards=[card.to_dict() for card in disconnection_cards],
        summary_md=(output_dir / "summary.md").read_text(encoding="utf-8"),
    )
    write_case_bundle(case_bundle, output_dir / "case_bundle.json")

    return {
        "case_id": profile.case_id,
        "output_dir": str(output_dir),
        "artifacts": {
            "target_profile": str(output_dir / "target_profile.json"),
            "baseline_routes": str(output_dir / "baseline_routes.json"),
            "frontier_report": str(output_dir / "frontier_report.json"),
            "literature_trigger_report": str(output_dir / "literature_trigger_report.json"),
            "literature_search_report": str(output_dir / "literature_search_report.md"),
            "evidence_cards": str(output_dir / "evidence_cards.jsonl"),
            "strategic_disconnection_cards": str(disconnection_path),
            "literature_candidates": str(candidate_path),
            "hybrid_route_package": str(package_path),
            "validation": str(output_dir / "validation.json"),
            "summary": str(output_dir / "summary.md"),
            "route_map": str(figures_dir / f"{profile.case_id}_retrosynthesis_map.svg"),
            "case_bundle": str(output_dir / "case_bundle.json"),
        },
        "validation": validation,
    }


def _load_baseline(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {
            "schema_version": "baseline_routes.v1",
            "status": "not_run",
            "solved": False,
            "routes": [],
            "ordinary_steps": [],
            "note": "P0 fallback: no baseline JSON supplied.",
        }
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _primary_frontier_smiles(frontier_report: dict[str, Any], profile: Any) -> str:
    frontiers = frontier_report.get("frontiers") or []
    if frontiers:
        return str(frontiers[0].get("frontier_smiles") or "")
    return str(profile.isomeric_smiles or profile.input_smiles)


def _baseline_solved_audit_passed(baseline_routes: dict[str, Any]) -> bool:
    if not baseline_routes:
        return False
    route_audit = _route_audit_from_baseline(baseline_routes)
    return bool(
        baseline_routes.get("solved")
        and (baseline_routes.get("routes") or [])
        and route_audit.get("stock_audit_passed")
        and route_audit.get("route_status") in {"solved", "semisynthesis_closed"}
    )


def _route_audit_from_baseline(baseline_routes: dict[str, Any]) -> dict[str, Any]:
    audit = dict(baseline_routes.get("route_audit") or baseline_routes.get("audit") or {})
    if audit:
        return audit
    if baseline_routes.get("solved") and baseline_routes.get("stock_audit_passed"):
        return {
            "schema_version": "route_audit_report.v1",
            "route_status": "solved",
            "stock_audit_passed": True,
            "fake_closure_rejected": False,
            "reasons": [],
        }
    return {
        "schema_version": "route_audit_report.v1",
        "route_status": "unresolved" if not baseline_routes.get("solved") else "",
        "stock_audit_passed": bool(baseline_routes.get("stock_audit_passed")),
        "fake_closure_rejected": bool(baseline_routes.get("fake_closure_rejected")),
        "reasons": list(baseline_routes.get("audit_reasons") or []),
    }


def _native_solved_skip_package(
    profile: Any,
    frontier_report: dict[str, Any],
    baseline_routes: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "hybrid_route_package.v1",
        "case_id": profile.case_id,
        "target": {
            "name": profile.target_name,
            "smiles": profile.isomeric_smiles or profile.input_smiles,
            "profile_ref": "target_profile.json",
        },
        "baseline": {
            "status": baseline_routes.get("status") or "provided",
            "ordinary_steps": baseline_routes.get("ordinary_steps") or [],
            "route_count": len(baseline_routes.get("routes") or []),
            "solved": bool(baseline_routes.get("solved")),
        },
        "frontier": (frontier_report.get("frontiers") or [{}])[0] if frontier_report.get("frontiers") else {},
        "literature_evidence_refs": [],
        "literature_candidates": [],
        "strategy_templates": [],
        "route_graph": {"nodes": [], "edges": []},
        "route_status": validation["route_status"],
        "status_contract": "Native route audit passed; literature mode was not entered.",
    }

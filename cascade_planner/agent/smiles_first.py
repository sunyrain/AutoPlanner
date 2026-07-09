"""End-to-end SMILES-first literature strategic workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascade_planner.agent.artifact_validators import validate_typed_artifact
from cascade_planner.agent.case_trace import (
    ArtifactRecord,
    case_bundle_from_p0_outputs,
    write_case_bundle,
)
from cascade_planner.agent.evidence_cards import (
    EvidenceCard,
    evidence_from_dict,
    validate_evidence_card,
    validation_summary,
    write_evidence_jsonl,
)
from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerRunRecord,
    WorkerTask,
    run_codex_worker,
)
from cascade_planner.agent.literature_escalation import decide_literature_escalation
from cascade_planner.agent.literature_research import (
    build_triggered_literature_task,
    render_literature_report,
    retrieve_literature_evidence,
    retrieve_pubmed_evidence,
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
    literature_backend: str = "api_json"
    worker_timeout_s: float = 60.0
    worker_max_output_bytes: int = 200_000
    worker_max_tool_calls: int = 8


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

    route_audit = _route_audit_from_baseline(baseline_routes)
    escalation_decision = decide_literature_escalation(
        native_result=baseline_routes,
        route_audit=route_audit,
        frontier_report=frontier_report,
        user_objective=config.objective,
        user_requested_literature="literature" in str(config.objective or "").lower(),
    )
    write_json(output_dir / "literature_escalation_decision.json", escalation_decision.to_dict())
    user_requested_literature = "user_requested_literature" in set(escalation_decision.escalation_reason)
    task, trigger_report = build_triggered_literature_task(
        profile,
        primary_frontier,
        native_result=baseline_routes,
        route_audit=route_audit,
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
    evidence_cards, literature_report, worker_records = _retrieve_literature_evidence_with_backend(
        task,
        config=config,
        output_dir=output_dir,
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
    _append_worker_records_to_case_bundle(case_bundle, worker_records)
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


def _retrieve_literature_evidence_with_backend(
    task: Any,
    *,
    config: SmilesFirstWorkflowConfig,
    output_dir: Path,
) -> tuple[list[EvidenceCard], dict[str, Any], list[WorkerRunRecord]]:
    requested_backend = str(config.literature_backend or "api_json").lower()
    backend = _resolve_literature_backend(requested_backend)
    if backend in {"local", "manual", "local_curated"}:
        cards, report = retrieve_literature_evidence(
            task,
            manual_evidence_jsonl=config.evidence_jsonl,
            db_paths=config.db_paths,
        )
        report["backend"] = "manual" if backend == "manual" else "local_curated"
        report["backend_requested"] = requested_backend
        report["backend_resolved"] = report["backend"]
        return cards, report, []
    if backend == "pubmed":
        cards, report = retrieve_pubmed_evidence(task, retmax=config.query_budget)
        report["backend_requested"] = requested_backend
        report["backend_resolved"] = "pubmed"
        return cards, report, []
    if backend in {"local_pubmed", "pubmed_local"}:
        local_cards, local_report = retrieve_literature_evidence(
            task,
            manual_evidence_jsonl=config.evidence_jsonl,
            db_paths=config.db_paths,
        )
        local_report["backend"] = "local_curated"
        local_report["backend_requested"] = requested_backend
        local_report["backend_resolved"] = "local_curated"
        pubmed_cards, pubmed_report = retrieve_pubmed_evidence(task, retmax=config.query_budget)
        cards = [*local_cards, *pubmed_cards]
        report = _merge_literature_reports(
            task,
            backend=backend,
            requested_backend=requested_backend,
            reports=[local_report, pubmed_report],
            cards=cards,
        )
        return cards, report, []
    if backend not in {"codex", "api_json"}:
        raise ValueError(f"unsupported_literature_backend:{backend}")

    worker_task = _worker_task_for_literature(task, config=config, output_dir=output_dir, backend=backend)
    write_json(output_dir / "literature_worker_task.json", worker_task.to_dict())
    record = run_codex_worker(
        worker_task,
        use_codex_cli=backend == "codex",
        use_api_json=backend == "api_json",
    )
    write_json(output_dir / "literature_worker_run_record.json", record.to_dict())
    cards, artifact_validations = _evidence_cards_from_worker_record(record)
    write_json(output_dir / "literature_worker_artifact_validation.json", {
        "schema_version": "literature_worker_artifact_validation.v1",
        "case_id": task.case_id,
        "backend": record.backend,
        "status": record.status,
        "validations": artifact_validations,
    })
    report = {
        "schema_version": "literature_search_report.v1",
        "case_id": task.case_id,
        "task": task.to_dict(),
        "backend": record.backend,
        "backend_requested": requested_backend,
        "backend_resolved": backend,
        "worker_status": record.status,
        "worker_trace_path": str(output_dir / "literature_worker_run_record.json"),
        "provider": dict(record.metadata or {}).get("provider", ""),
        "base_url_fingerprint": dict(record.metadata or {}).get("base_url_fingerprint", ""),
        "model": dict(record.metadata or {}).get("model", ""),
        "searches": [{
            "query": task.frontier_smiles,
            "source": record.backend,
            "hits": len(cards),
            "status": record.status,
        }],
        "hit_count": len(cards),
        "evidence_levels": _evidence_level_counts_from_cards(cards),
        "unresolved_literature_gap": len(cards) == 0,
        "limitations": [] if cards else ["unresolved_literature_gap"],
        "artifact_validations": artifact_validations,
    }
    return cards, report, [record]


def _resolve_literature_backend(requested_backend: str) -> str:
    backend = str(requested_backend or "api_json").lower()
    if backend in {"auto", "default"}:
        return "api_json"
    return backend


def _merge_literature_reports(
    task: Any,
    *,
    backend: str,
    requested_backend: str,
    reports: list[dict[str, Any]],
    cards: list[EvidenceCard],
) -> dict[str, Any]:
    searches: list[dict[str, Any]] = []
    limitations: list[str] = []
    for report in reports:
        searches.extend(dict(item) for item in report.get("searches") or [])
        limitations.extend(str(item) for item in report.get("limitations") or [])
    return {
        "schema_version": "literature_search_report.v1",
        "case_id": task.case_id,
        "task": task.to_dict(),
        "backend": backend,
        "backend_requested": requested_backend,
        "backend_resolved": backend,
        "searches": searches,
        "hit_count": len(cards),
        "evidence_levels": _evidence_level_counts_from_cards(cards),
        "unresolved_literature_gap": len(cards) == 0,
        "limitations": sorted(set(item for item in limitations if item and item != "unresolved_literature_gap")),
        "component_backends": [
            str(report.get("backend") or report.get("backend_resolved") or "")
            for report in reports
            if report.get("backend") or report.get("backend_resolved")
        ],
    }


def _worker_task_for_literature(
    task: Any,
    *,
    config: SmilesFirstWorkflowConfig,
    output_dir: Path,
    backend: str,
) -> WorkerTask:
    task_context = json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True)
    return WorkerTask(
        task_id=f"{task.case_id}:literature:{backend}",
        case_id=task.case_id,
        task_type="target_research",
        required_artifact_type="EvidenceCard",
        input_refs=["target_profile.json", "frontier_report.json", "literature_search_task.json"],
        allowed_tools=["web_search", "local_search"],
        budget=WorkerBudget(
            timeout_s=float(config.worker_timeout_s),
            max_output_bytes=int(config.worker_max_output_bytes),
            max_tool_calls=int(config.worker_max_tool_calls),
            max_worker_runs=1,
        ),
        objective=(
            "Find traceable literature evidence for this exact target/frontier and return one "
            "EvidenceCard draft artifact. Prioritize route-relevant synthesis evidence: strategic "
            "disconnection, C17 2-pyrone/bufadienolide installation, semisynthesis anchor, or a "
            "stuck-frontier route precedent. Use route_role strategic_disconnection or route_anchor "
            "only when a traceable source supports it; otherwise mark the limitation and use a weaker "
            "role. Reject biological activity, pharmacology, toxicity, or assay-only papers as route anchors "
            "unless they also contain traceable synthesis/semisynthesis or structural route evidence. "
            "Do not propose raw reactions. The literature task context is: "
            f"{task_context}"
        ),
        allowed_workdir=str(output_dir),
        dry_run=False,
    )


def _evidence_cards_from_worker_record(record: WorkerRunRecord) -> tuple[list[EvidenceCard], list[dict[str, Any]]]:
    validations: list[dict[str, Any]] = []
    artifact = record.output_artifact if isinstance(record.output_artifact, dict) else None
    if not artifact:
        return [], validations
    typed_validation = validate_typed_artifact(artifact)
    validations.append({"stage": "typed_artifact", **typed_validation})
    if record.status != "accepted_draft" or not typed_validation.get("accepted"):
        return [], validations
    payload = dict(artifact.get("payload") or {})
    evidence_validation = validate_evidence_card(payload)
    validations.append({"stage": "evidence_card", **evidence_validation})
    if not evidence_validation.get("accepted"):
        return [], validations
    card = evidence_from_dict(payload)
    card.validation_status = str(evidence_validation.get("validation_status") or "validated")
    return [card], validations


def _append_worker_records_to_case_bundle(bundle: Any, records: list[WorkerRunRecord]) -> None:
    for record in records:
        payload = record.to_dict()
        bundle.append_artifact(ArtifactRecord(
            artifact_id=_unique_case_artifact_id(bundle, record.run_id),
            case_id=bundle.case_id,
            artifact_type="WorkerRunRecord",
            payload=payload,
            source=record.backend or "worker",
            validation_status="accepted" if record.status == "accepted_draft" else "rejected",
            input_refs=[record.task_id],
        ))
        if isinstance(record.output_artifact, dict):
            validation = validate_typed_artifact(record.output_artifact)
            status = "accepted" if validation.get("accepted") else "rejected"
            bundle.append_artifact(ArtifactRecord(
                artifact_id=_unique_case_artifact_id(bundle, str(record.output_artifact.get("artifact_id") or f"{record.run_id}:artifact")),
                case_id=bundle.case_id,
                artifact_type=str(record.output_artifact.get("artifact_type") or "WorkerOutputArtifact"),
                payload=record.output_artifact,
                source=str(record.output_artifact.get("source") or record.backend or "worker"),
                validation_status=status,
                input_refs=[str(ref) for ref in record.output_artifact.get("input_refs") or []],
                evidence_refs=[str(ref) for ref in record.output_artifact.get("evidence_refs") or []],
            ))


def _unique_case_artifact_id(bundle: Any, base: str) -> str:
    value = str(base or "artifact").replace("/", "_")
    existing = {artifact.artifact_id for artifact in bundle.artifacts}
    if value not in existing:
        return value
    idx = 2
    while f"{value}:{idx}" in existing:
        idx += 1
    return f"{value}:{idx}"


def _evidence_level_counts_from_cards(cards: list[EvidenceCard]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for card in cards:
        counts[card.target_relation] = counts.get(card.target_relation, 0) + 1
    return counts


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

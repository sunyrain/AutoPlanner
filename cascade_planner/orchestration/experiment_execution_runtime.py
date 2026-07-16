"""Audit one executor result against the current route innovation projection."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.experiment_execution_results import (
    audit_experiment_execution_result,
    release_experiment_validation_candidate,
)
from cascade_planner.orchestration.program_innovation_materials import (
    compile_route_program_innovation_materials,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def audit_current_route_experiment_result(
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    result: Mapping[str, Any],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Reproject current requests, audit one result, and release no facts."""

    result_value = dict(result)
    frontier, request = locate_current_route_experiment_request(
        graph,
        acceptance_spec=acceptance_spec,
        route_id=route_id,
        capabilities=capabilities,
        request_id=str(result_value.get("request_id") or ""),
        mechanism_proposals=mechanism_proposals,
        validations=validations,
    )
    audit = audit_experiment_execution_result(request, result_value)
    candidate = (
        release_experiment_validation_candidate(request, result_value)
        if audit["accepted_for_domain_gate"] is True
        else {}
    )
    payload = {
        "schema_version": "route_experiment_result_review.v1",
        "run_id": str(graph.get("run_id") or ""),
        "route_id": str(route_id),
        "experimental_work_frontier_sha256": str(frontier["content_sha256"]),
        "request": request,
        "result_audit": audit,
        "domain_validation_candidate": candidate,
        "next_boundary": (
            "submit_candidate_to_existing_domain_validation_gate"
            if candidate
            else "retain_result_envelope_without_domain_validation"
        ),
        "semantics": {
            "review_is_read_only": True,
            "current_frontier_reprojection_is_required": True,
            "released_candidate_is_not_accepted_validation": True,
            "no_claim_graph_proof_completion_or_catalog_write": True,
        },
    }
    payload["content_sha256"] = strict_canonical_json_sha256(payload)
    return payload


def locate_current_route_experiment_request(
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    request_id: str,
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one current request from the sole canonical work projection."""

    materials = compile_route_program_innovation_materials(
        graph,
        acceptance_spec=acceptance_spec,
        route_id=route_id,
        capabilities=capabilities,
        mechanism_proposals=mechanism_proposals,
        validations=validations,
    )
    frontier = dict(materials["experimental_work_frontier"])
    requests = {
        str(dict(item.get("execution_request") or {}).get("request_id") or ""): dict(
            item.get("execution_request") or {}
        )
        for item in dict(frontier.get("work_items") or {}).values()
        if isinstance(item, Mapping)
    }
    request = requests.get(str(request_id))
    if request is None:
        raise ValueError("experiment_result_request_not_in_current_frontier")
    return frontier, request


__all__ = [
    "audit_current_route_experiment_result",
    "locate_current_route_experiment_request",
]

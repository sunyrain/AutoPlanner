"""Compile whole-cell and hybrid capability matches into Program route drafts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.program_innovation_contracts import (
    ProgramInnovationContractError,
    strict_program_innovation_object,
    validate_program_innovation_inputs,
    with_program_innovation_digest,
)
from cascade_planner.application.program_span_substitutions import (
    ProgramSpanError,
    program_span_boundary,
    substitute_program_span,
)
from cascade_planner.application.execution_program_compilation import (
    bind_execution_operation_boundaries,
    execution_candidate_reasons,
)
from cascade_planner.application.execution_program_validations import (
    execution_validation_gate,
    strict_execution_validations,
)
from cascade_planner.application.transformation_programs import chemical_state_id
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXECUTION_PROGRAM_BUNDLE_SCHEMA = "execution_program_bundle.v1"
EXECUTION_PROGRAM_PROPOSAL_SCHEMA = "execution_program_proposal.v1"
EXECUTION_PROGRAM_ROUTE_SCHEMA = "execution_program_route_candidate.v1"
EXECUTION_PROGRAM_ORACLE_SCHEMA = "execution_program_bundle_oracle.v1"
EXECUTION_PROGRAM_SEMANTICS = {
    "read_only_program_proposals": True,
    "whole_cell_and_hybrid_are_execution_domains_not_search_labels": True,
    "programs_connect_exact_contiguous_route_boundaries": True,
    "replaced_edge_programs_remain_fallback": True,
    "operation_count_includes_preparation_workup_and_separation": True,
    "negative_operation_savings_remain_visible": True,
    "specialized_validation_is_required": True,
    "unvalidated_candidates_remain_exploration_only": True,
    "program_candidates_cannot_grant_route_completion": True,
    "edge_ids_remain_production_route_authority": True,
    "target_names_are_not_compilation_inputs": True,
}


class ExecutionProgramError(ValueError):
    """A whole-cell or hybrid draft cannot be safely compiled."""


def compile_execution_program_bundle(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    try:
        graph_value = strict_program_innovation_object(graph, "graph")
        route_value = strict_program_innovation_object(route, "route")
        projection_value = strict_program_innovation_object(projection, "projection")
        discovery_value = strict_program_innovation_object(discovery, "discovery")
        validate_program_innovation_inputs(
            graph_value, route_value, projection_value, discovery_value
        )
    except ProgramInnovationContractError as exc:
        raise ExecutionProgramError(str(exc)) from exc
    try:
        validation_rows = strict_execution_validations(validations)
    except ValueError as exc:
        raise ExecutionProgramError(str(exc)) from exc

    proposals: dict[str, dict[str, Any]] = {}
    routes: dict[str, dict[str, Any]] = {}
    rejections: list[dict[str, Any]] = []
    for raw in discovery_value.get("candidates") or []:
        if not isinstance(raw, Mapping) or raw.get("candidate_kind") != "program_execution_window":
            continue
        candidate = dict(raw)
        candidate_id = str(candidate.get("candidate_id") or "")
        try:
            proposal = _compile_proposal(
                graph_value,
                route_value,
                projection_value,
                candidate,
                validation_rows,
            )
            route_candidate = _compile_route(route_value, projection_value, proposal)
        except ExecutionProgramError as exc:
            rejections.append({"candidate_id": candidate_id, "reasons": str(exc).split(";")})
            continue
        proposals[proposal["program_id"]] = proposal
        routes[route_candidate["route_candidate_id"]] = route_candidate
    bound_program_ids = set(proposals)
    unbound = sorted(
        str(row.get("validation_id") or "")
        for row in validation_rows
        if str(row.get("program_id") or "") not in bound_program_ids
    )
    counts = {
        "program_proposals": len(proposals),
        "route_candidates": len(routes),
        "whole_cell": sum(row["execution_domain"] == "whole_cell" for row in proposals.values()),
        "hybrid": sum(row["execution_domain"] == "hybrid" for row in proposals.values()),
        "rejected_candidates": len(rejections),
        "validated_substitutions": sum(
            row["validation_plan"]["accepted"] is True for row in proposals.values()
        ),
        "unbound_validations": len(unbound),
    }
    return with_program_innovation_digest(
        {
            "schema_version": EXECUTION_PROGRAM_BUNDLE_SCHEMA,
            "run_id": str(projection_value["run_id"]),
            "source_graph_revision": int(projection_value["source_graph_revision"]),
            "source_graph_scientific_sha256": str(
                projection_value["source_graph_scientific_sha256"]
            ),
            "source_projection_sha256": str(projection_value["content_sha256"]),
            "source_route_id": str(route_value.get("route_id") or ""),
            "source_route_sha256": strict_canonical_json_sha256(route_value),
            "source_discovery_sha256": str(discovery_value["content_sha256"]),
            "program_proposals": proposals,
            "route_candidates": routes,
            "rejections": rejections,
            "unbound_validation_ids": unbound,
            "counts": counts,
            "semantics": dict(EXECUTION_PROGRAM_SEMANTICS),
        }
    )


def execution_program_bundle_oracle(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    try:
        expected = compile_execution_program_bundle(
            graph,
            route,
            projection,
            discovery,
            validations=validations,
        )
        observed_value = strict_program_innovation_object(observed, "observed")
    except (ExecutionProgramError, ProgramInnovationContractError) as exc:
        return _oracle_result(
            False,
            {"inputs_reprojectable": False},
            [f"bundle_inputs_invalid:{type(exc).__name__}"],
            "",
            "",
        )
    material = dict(observed_value)
    observed_digest = str(material.pop("content_sha256", ""))
    checks = {
        "inputs_reprojectable": True,
        "schema_equal": observed_value.get("schema_version") == EXECUTION_PROGRAM_BUNDLE_SCHEMA,
        "content_digest_valid": observed_digest == strict_canonical_json_sha256(material),
        "projection_equal": observed_value == expected,
        "authority_semantics_equal": observed_value.get("semantics") == EXECUTION_PROGRAM_SEMANTICS,
    }
    reasons = [key for key, accepted in checks.items() if accepted is not True]
    return _oracle_result(
        not reasons,
        checks,
        reasons,
        str(expected["content_sha256"]),
        observed_digest,
    )


def _compile_proposal(
    graph: dict[str, Any],
    route: dict[str, Any],
    projection: dict[str, Any],
    candidate: dict[str, Any],
    validations: list[dict[str, Any]],
) -> dict[str, Any]:
    capability = dict(candidate.get("execution_capability") or {})
    reasons = execution_candidate_reasons(candidate, capability)
    if reasons:
        raise ExecutionProgramError(";".join(reasons))
    boundary = dict(candidate.get("boundary") or {})
    span = [str(value) for value in boundary.get("replaced_edge_ids") or []]
    try:
        materialized = program_span_boundary(graph, route, span)
    except ProgramSpanError as exc:
        raise ExecutionProgramError(str(exc)) from exc
    if boundary.get("precursor_molecule_id") not in materialized["input_molecule_ids"]:
        raise ExecutionProgramError("execution_candidate_input_boundary_mismatch")
    if boundary.get("product_molecule_id") != materialized["output_molecule_ids"][0]:
        raise ExecutionProgramError("execution_candidate_output_boundary_mismatch")
    identity = {
        "run_id": projection["run_id"],
        "graph_revision": projection["source_graph_revision"],
        "route_id": route.get("route_id"),
        "candidate_id": candidate.get("candidate_id"),
        "capability_sha256": capability["content_sha256"],
        "equivalent_reference_span": span,
    }
    program_id = (
        f"program:{capability['execution_domain']}:{strict_canonical_json_sha256(identity)[:24]}"
    )
    input_states = [chemical_state_id(value) for value in materialized["input_molecule_ids"]]
    output_states = [chemical_state_id(value) for value in materialized["output_molecule_ids"]]
    operations = bind_execution_operation_boundaries(
        capability["operation_blueprints"], input_states, output_states
    )
    isolated_count = sum(row["isolated_operation"] is True for row in operations)
    domain = str(capability["execution_domain"])
    warnings: set[str] = {
        *[str(value) for value in candidate.get("warning_codes") or []],
    }
    if (
        domain == "hybrid"
        and sum(row["contributes_to_net_transform"] is True for row in operations) > 1
    ):
        warnings.add("HYBRID_INTERNAL_STATES_UNMATERIALIZED")
    proposal = {
        "schema_version": EXECUTION_PROGRAM_PROPOSAL_SCHEMA,
        "program_id": program_id,
        "proposal_kind": f"{domain}_execution",
        "execution_domain": domain,
        "source_candidate_id": str(candidate["candidate_id"]),
        "source_capability_id": str(capability["capability_id"]),
        "source_capability_sha256": str(capability["content_sha256"]),
        "input_state_ids": input_states,
        "output_state_ids": output_states,
        "operation_blueprints": operations,
        "equivalent_reference_span": span,
        "chemical_step_equivalent_count": len(span),
        "isolated_operation_count": isolated_count,
        "net_operation_savings": len(span) - isolated_count,
        "actors": dict(capability["actors"]),
        "cofactor_and_carrier_ledger": {
            "cofactor_requirements": dict(capability["cofactor_requirements"]),
            "cofactor_regenerations": dict(capability["cofactor_regenerations"]),
            "carrier_requirements": dict(capability["carrier_requirements"]),
            "carrier_regenerations": dict(capability["carrier_regenerations"]),
        },
        "selectivity_constraints": [str(capability["selectivity_objective"])],
        "claim_refs": {
            "precedent_refs": list(capability["precedent_refs"]),
            "exact_substrate_claim_refs": [],
            "analogy_only": True,
        },
        "validation_plan": {
            "required_checks": list(capability["validation_requirements"]),
        },
        "validation_vector": {
            "structure": "boundary_states_materialized",
            "precedent": "analogy_only",
            "execution": "operation_sequence_proposed",
            "specialized_execution": "validation_pending",
            "process": "not_assessed",
            "authoritative": False,
        },
        "warning_codes": [],
        "status": "proposal_only",
        "eligible_for_shadow_optimizer": False,
        "semantics": {
            "not_a_canonical_reaction_edge": True,
            "does_not_inherit_replaced_edge_proof": True,
            "cannot_grant_route_completion": True,
            "fallback_route_is_required": True,
        },
    }
    gate = execution_validation_gate(proposal, validations)
    proposal["validation_plan"] = {
        "required_checks": list(capability["validation_requirements"]),
        **gate,
        "grants_validation": gate["accepted"] is True,
    }
    validated = gate["accepted"] is True
    proposal["validation_vector"]["specialized_execution"] = (
        "validation_bound" if validated else "validation_required"
    )
    proposal["status"] = "shadow_ready" if validated else "proposal_only"
    proposal["eligible_for_shadow_optimizer"] = validated
    if validated:
        warnings.add("EXECUTION_VALIDATION_BOUND")
        warnings.discard("SPECIALIZED_EXECUTION_VALIDATION_REQUIRED")
    else:
        warnings.add("SPECIALIZED_EXECUTION_VALIDATION_REQUIRED")
    proposal["warning_codes"] = sorted(warnings)
    return with_program_innovation_digest(proposal)


def _compile_route(
    route: dict[str, Any], projection: dict[str, Any], proposal: dict[str, Any]
) -> dict[str, Any]:
    span = list(proposal["equivalent_reference_span"])
    try:
        substitution = substitute_program_span(route, span, str(proposal["program_id"]))
    except ProgramSpanError as exc:
        raise ExecutionProgramError(str(exc)) from exc
    fallback = substitution["fallback_program_ids"]
    selected = substitution["selected_program_ids"]
    physical = len(fallback) - len(span) + int(proposal["isolated_operation_count"])
    identity = {
        "source_route_id": route.get("route_id"),
        "execution_program_id": proposal["program_id"],
        "selected_program_ids": selected,
    }
    validated = proposal["validation_plan"]["accepted"] is True
    return with_program_innovation_digest(
        {
            "schema_version": EXECUTION_PROGRAM_ROUTE_SCHEMA,
            "route_candidate_id": (
                f"program-route:{proposal['execution_domain']}:"
                f"{strict_canonical_json_sha256(identity)[:24]}"
            ),
            "source_route_id": str(route.get("route_id") or ""),
            "source_route_family_id": str(route.get("route_family_id") or ""),
            "source_projection_sha256": str(projection["content_sha256"]),
            "source_edge_ids": [str(value) for value in route.get("edge_ids") or []],
            **substitution,
            "execution_program_id": str(proposal["program_id"]),
            "execution_domain": str(proposal["execution_domain"]),
            "replaced_edge_ids": span,
            "physical_operation_count": physical,
            "chemical_step_equivalent_count": len(fallback),
            "net_operation_savings": len(fallback) - physical,
            "full_candidate_route_restitched": True,
            "eligible_for_program_optimizer": validated,
            "eligible_for_route_completion": False,
            "status": "validated_candidate" if validated else "unvalidated_candidate",
            "semantics": {
                "fallback_route_is_retained": True,
                "negative_operation_savings_remain_visible": True,
                "does_not_replace_production_route": True,
            },
        }
    )


def _oracle_result(
    accepted: bool,
    checks: Mapping[str, bool],
    reasons: list[str],
    expected_digest: str,
    observed_digest: str,
) -> dict[str, Any]:
    return with_program_innovation_digest(
        {
            "schema_version": EXECUTION_PROGRAM_ORACLE_SCHEMA,
            "accepted": accepted,
            "checks": dict(checks),
            "reasons": reasons,
            "expected_bundle_sha256": expected_digest,
            "observed_bundle_sha256": observed_digest,
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_grants_no_scientific_authority": True,
            },
        }
    )


__all__ = [
    "EXECUTION_PROGRAM_BUNDLE_SCHEMA",
    "EXECUTION_PROGRAM_ORACLE_SCHEMA",
    "EXECUTION_PROGRAM_PROPOSAL_SCHEMA",
    "EXECUTION_PROGRAM_ROUTE_SCHEMA",
    "EXECUTION_PROGRAM_SEMANTICS",
    "ExecutionProgramError",
    "compile_execution_program_bundle",
    "execution_program_bundle_oracle",
]

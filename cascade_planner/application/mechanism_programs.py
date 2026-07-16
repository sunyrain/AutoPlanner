"""Compile one-hop mechanism hypotheses into fully restitched Program routes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.mechanism_program_compilation import (
    mechanism_candidate_reasons,
)
from cascade_planner.application.program_innovation_contracts import (
    ProgramInnovationContractError,
    strict_program_innovation_object,
    validate_program_innovation_inputs,
    with_program_innovation_digest,
)
from cascade_planner.application.program_span_substitutions import (
    ProgramSpanError,
    matching_program_spans,
    program_span_boundary,
    substitute_program_span,
)
from cascade_planner.application.mechanism_program_validations import (
    mechanism_required_checks,
    mechanism_support_state,
    mechanism_validation_gate,
    strict_mechanism_validations,
)
from cascade_planner.application.route_innovations import MECHANISM_EXTRAPOLATION
from cascade_planner.application.transformation_programs import chemical_state_id
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


MECHANISM_PROGRAM_BUNDLE_SCHEMA = "mechanism_program_bundle.v1"
MECHANISM_PROGRAM_PROPOSAL_SCHEMA = "mechanism_program_proposal.v1"
MECHANISM_PROGRAM_ROUTE_SCHEMA = "mechanism_restitched_program_route.v1"
MECHANISM_PROGRAM_ORACLE_SCHEMA = "mechanism_program_bundle_oracle.v1"
MECHANISM_PROGRAM_SEMANTICS = {
    "read_only_program_proposals": True,
    "one_hop_must_rejoin_known_route_states": True,
    "replaced_edge_programs_remain_fallback": True,
    "full_route_is_restitched_before_optimizer_visibility": True,
    "anchor_source_does_not_report_the_extrapolation": True,
    "specialized_mechanism_validation_is_required_for_shadow": True,
    "valid_failure_and_inconclusive_results_remain_feedback": True,
    "unvalidated_candidates_remain_exploration_visible": True,
    "program_candidates_cannot_grant_route_completion": True,
    "edge_ids_remain_production_route_authority": True,
    "target_names_are_not_compilation_inputs": True,
}


class MechanismProgramError(ValueError):
    """A mechanism hypothesis cannot be safely restitched into a route."""


def compile_mechanism_program_bundle(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Materialize restitchable one-hop candidates without canonical writes."""

    try:
        graph_value = strict_program_innovation_object(graph, "graph")
        route_value = strict_program_innovation_object(route, "route")
        projection_value = strict_program_innovation_object(projection, "projection")
        discovery_value = strict_program_innovation_object(discovery, "discovery")
        validate_program_innovation_inputs(
            graph_value, route_value, projection_value, discovery_value
        )
    except ProgramInnovationContractError as exc:
        raise MechanismProgramError(str(exc)) from exc
    try:
        validation_rows = strict_mechanism_validations(validations)
    except ValueError as exc:
        raise MechanismProgramError(str(exc)) from exc

    proposals: dict[str, dict[str, Any]] = {}
    routes: dict[str, dict[str, Any]] = {}
    rejections: list[dict[str, Any]] = []
    for raw in discovery_value.get("candidates") or []:
        if not isinstance(raw, Mapping) or raw.get("candidate_kind") != "mechanism_one_hop":
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
            restitched = _compile_route(route_value, projection_value, proposal)
        except MechanismProgramError as exc:
            rejections.append({"candidate_id": candidate_id, "reasons": str(exc).split(";")})
            continue
        proposals[proposal["program_id"]] = proposal
        routes[restitched["route_candidate_id"]] = restitched
    bound_program_ids = set(proposals)
    unbound = sorted(
        str(row.get("validation_id") or "")
        for row in validation_rows
        if str(row.get("program_id") or "") not in bound_program_ids
    )
    payload = {
        "schema_version": MECHANISM_PROGRAM_BUNDLE_SCHEMA,
        "run_id": str(projection_value["run_id"]),
        "source_graph_revision": int(projection_value["source_graph_revision"]),
        "source_graph_scientific_sha256": str(projection_value["source_graph_scientific_sha256"]),
        "source_projection_sha256": str(projection_value["content_sha256"]),
        "source_route_id": str(route_value.get("route_id") or ""),
        "source_route_sha256": strict_canonical_json_sha256(route_value),
        "source_discovery_sha256": str(discovery_value["content_sha256"]),
        "program_proposals": proposals,
        "route_candidates": routes,
        "rejections": rejections,
        "unbound_validation_ids": unbound,
        "counts": {
            "program_proposals": len(proposals),
            "route_candidates": len(routes),
            "rejected_candidates": len(rejections),
            "validated_substitutions": sum(
                row["validation_plan"]["accepted"] is True for row in proposals.values()
            ),
            "unbound_validations": len(unbound),
        },
        "semantics": dict(MECHANISM_PROGRAM_SEMANTICS),
    }
    return with_program_innovation_digest(payload)


def mechanism_program_bundle_oracle(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Recompile the complete bundle and compare exact content and authority."""

    try:
        expected = compile_mechanism_program_bundle(
            graph,
            route,
            projection,
            discovery,
            validations=validations,
        )
        observed_value = strict_program_innovation_object(observed, "observed")
    except (MechanismProgramError, ProgramInnovationContractError) as exc:
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
        "schema_equal": observed_value.get("schema_version") == MECHANISM_PROGRAM_BUNDLE_SCHEMA,
        "content_digest_valid": observed_digest == strict_canonical_json_sha256(material),
        "projection_equal": observed_value == expected,
        "authority_semantics_equal": observed_value.get("semantics") == MECHANISM_PROGRAM_SEMANTICS,
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
    reasons = mechanism_candidate_reasons(route, candidate)
    if reasons:
        raise MechanismProgramError(";".join(reasons))
    boundary = dict(candidate["boundary"])
    spans = matching_program_spans(
        graph,
        route,
        precursor_smiles=str(boundary["precursor_smiles"]),
        product_smiles=str(boundary["product_smiles"]),
    )
    if not spans:
        raise MechanismProgramError("mechanism_full_route_restitch_missing")
    if len(spans) != 1:
        raise MechanismProgramError("mechanism_full_route_restitch_ambiguous")
    span = spans[0]
    try:
        materialized_boundary = program_span_boundary(graph, route, span)
    except ProgramSpanError as exc:
        raise MechanismProgramError(str(exc)) from exc
    innovation = dict(candidate["route_innovation"])
    identity = {
        "run_id": projection["run_id"],
        "graph_revision": projection["source_graph_revision"],
        "route_id": route.get("route_id"),
        "candidate_id": candidate.get("candidate_id"),
        "innovation_id": innovation.get("innovation_id"),
        "equivalent_reference_span": span,
    }
    proposal_id = "program:mechanism:" + strict_canonical_json_sha256(identity)[:24]
    input_states = [
        chemical_state_id(value) for value in materialized_boundary["input_molecule_ids"]
    ]
    output_states = [
        chemical_state_id(value) for value in materialized_boundary["output_molecule_ids"]
    ]
    proposal = {
        "schema_version": MECHANISM_PROGRAM_PROPOSAL_SCHEMA,
        "program_id": proposal_id,
        "proposal_kind": MECHANISM_EXTRAPOLATION,
        "source_candidate_id": str(candidate["candidate_id"]),
        "source_innovation_id": str(innovation["innovation_id"]),
        "input_state_ids": input_states,
        "output_state_ids": output_states,
        "operation_blueprints": [
            {
                "operation_kind": "mechanism_hypothesis",
                "sequence_index": 0,
                "input_state_ids": input_states,
                "output_state_ids": output_states,
            }
        ],
        "execution_domain": "chemical",
        "equivalent_reference_span": span,
        "chemical_step_equivalent_count": len(span),
        "isolated_operation_count": 1,
        "net_step_savings": len(span) - 1,
        "anchor": dict(innovation["anchor"]),
        "mechanistic_rationale": str(innovation["mechanistic_rationale"]),
        "elementary_steps": list(innovation["elementary_steps"]),
        "falsifiable_checks": list(innovation["falsifiable_checks"]),
        "validation_plan": {},
        "validation_vector": {
            "structure": "known_route_boundary_states_materialized",
            "reaction": "unvalidated_mechanism_hypothesis",
            "mechanism": "hypothesis_only",
            "source": "anchor_only_not_extrapolated_reaction",
            "conditions": "missing",
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
            "full_route_boundary_was_restitched": True,
        },
    }
    required_checks = mechanism_required_checks(proposal)
    gate = mechanism_validation_gate(proposal, validations)
    proposal["validation_plan"] = {
        "required_checks": required_checks,
        **gate,
        "grants_validation": gate["accepted"] is True,
    }
    validated = gate["accepted"] is True
    proposal["validation_vector"]["reaction"] = (
        "exact_boundary_experiment_bound" if validated else "unvalidated_mechanism_hypothesis"
    )
    proposal["validation_vector"]["mechanism"] = mechanism_support_state(gate)
    warnings = {
        *[str(value) for value in candidate.get("warning_codes") or []],
        "ANCHOR_SOURCE_DOES_NOT_REPORT_EXTRAPOLATION",
        ("MECHANISM_VALIDATION_BOUND" if validated else "MECHANISM_EXTRAPOLATION_UNVALIDATED"),
    }
    proposal["warning_codes"] = sorted(warnings)
    proposal["status"] = "shadow_ready" if validated else "proposal_only"
    proposal["eligible_for_shadow_optimizer"] = validated
    return with_program_innovation_digest(proposal)


def _compile_route(
    route: dict[str, Any], projection: dict[str, Any], proposal: dict[str, Any]
) -> dict[str, Any]:
    span = list(proposal["equivalent_reference_span"])
    try:
        substitution = substitute_program_span(route, span, str(proposal["program_id"]))
    except ProgramSpanError as exc:
        raise MechanismProgramError(str(exc)) from exc
    selected = substitution["selected_program_ids"]
    fallback = substitution["fallback_program_ids"]
    identity = {
        "source_route_id": route.get("route_id"),
        "mechanism_program_id": proposal["program_id"],
        "selected_program_ids": selected,
    }
    validated = dict(proposal.get("validation_plan") or {}).get("accepted") is True
    return with_program_innovation_digest(
        {
            "schema_version": MECHANISM_PROGRAM_ROUTE_SCHEMA,
            "route_candidate_id": (
                "program-route:mechanism:" + strict_canonical_json_sha256(identity)[:24]
            ),
            "source_route_id": str(route.get("route_id") or ""),
            "source_route_family_id": str(route.get("route_family_id") or ""),
            "source_projection_sha256": str(projection["content_sha256"]),
            "source_edge_ids": [str(value) for value in route.get("edge_ids") or []],
            **substitution,
            "mechanism_program_id": str(proposal["program_id"]),
            "replaced_edge_ids": span,
            "physical_step_count": len(selected),
            "chemical_step_equivalent_count": len(fallback),
            "net_step_savings": len(span) - 1,
            "full_candidate_route_restitched": True,
            "eligible_for_program_optimizer": validated,
            "eligible_for_route_completion": False,
            "status": "validated_candidate" if validated else "unvalidated_candidate",
            "semantics": {
                "fallback_route_is_retained": True,
                "does_not_replace_production_route": True,
                "anchor_source_does_not_report_the_extrapolation": True,
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
            "schema_version": MECHANISM_PROGRAM_ORACLE_SCHEMA,
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
    "MECHANISM_PROGRAM_BUNDLE_SCHEMA",
    "MECHANISM_PROGRAM_ORACLE_SCHEMA",
    "MECHANISM_PROGRAM_PROPOSAL_SCHEMA",
    "MECHANISM_PROGRAM_ROUTE_SCHEMA",
    "MECHANISM_PROGRAM_SEMANTICS",
    "MechanismProgramError",
    "compile_mechanism_program_bundle",
    "mechanism_program_bundle_oracle",
]

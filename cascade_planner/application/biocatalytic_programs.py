"""Compile enzyme-window discoveries into read-only TransformationProgram drafts.

A superstep is a new input-state to output-state program.  It never becomes a
special annotation on the last reaction edge of the replaced interval.  The
canonical edge route remains an explicit fallback until a later admission
phase switches route authority to programs.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from cascade_planner.application.biocatalytic_program_contracts import (
    BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
    BIOCATALYTIC_PROGRAM_BUNDLE_SCHEMA,
    BIOCATALYTIC_PROGRAM_ORACLE_SCHEMA,
    BIOCATALYTIC_PROGRAM_PROPOSAL_SCHEMA,
    BIOCATALYTIC_PROGRAM_ROUTE_SCHEMA,
    BIOCATALYTIC_PROGRAM_SEMANTICS,
    BiocatalyticProgramError,
    biocatalysis_validation_gate as _validation_gate,
    biocatalytic_span_boundary as _span_boundary,
    strict_object as _strict_object,
    validate_biocatalytic_innovation as _validate_innovation,
    validate_biocatalytic_program_inputs as _validate_inputs,
    with_biocatalysis_program_validation_digest,
    with_digest as _with_digest,
)
from cascade_planner.application.transformation_programs import (
    chemical_state_id,
    program_id,
)
from cascade_planner.application.program_span_substitutions import (
    ProgramSpanError,
    substitute_program_span,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


_SEMANTICS = BIOCATALYTIC_PROGRAM_SEMANTICS


def compile_biocatalytic_program_bundle(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compile all enzyme-window candidates and their fallback route variants."""

    graph_value = _strict_object(graph, "graph")
    route_value = _strict_object(route, "route")
    projection_value = _strict_object(projection, "projection")
    discovery_value = _strict_object(discovery, "discovery")
    _validate_inputs(graph_value, route_value, projection_value, discovery_value)
    validation_rows = [_strict_object(value, "validation") for value in validations]

    proposals: dict[str, dict[str, Any]] = {}
    route_candidates: dict[str, dict[str, Any]] = {}
    bound_validation_ids: set[str] = set()
    rejections: list[dict[str, Any]] = []
    for raw in discovery_value.get("candidates") or []:
        if not isinstance(raw, Mapping) or raw.get("candidate_kind") != "enzyme_window":
            continue
        candidate = _strict_object(raw, "candidate")
        try:
            proposal = _compile_proposal(
                graph_value,
                route_value,
                projection_value,
                candidate,
                validation_rows,
            )
            variant = _compile_route_candidate(route_value, projection_value, proposal)
        except BiocatalyticProgramError as exc:
            rejections.append(
                {
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "reasons": str(exc).split(";"),
                }
            )
            continue
        proposals[proposal["program_id"]] = proposal
        route_candidates[variant["route_candidate_id"]] = variant
        bound_validation_ids.update(proposal["validation_gate"]["validation_ids"])

    unbound = sorted(
        str(row.get("validation_id") or "")
        for row in validation_rows
        if str(row.get("validation_id") or "") not in bound_validation_ids
    )
    accounting_valid = all(
        row["chemical_step_equivalent_count"] - row["physical_step_count"]
        == row["net_step_savings"]
        for row in route_candidates.values()
    )
    payload = {
        "schema_version": BIOCATALYTIC_PROGRAM_BUNDLE_SCHEMA,
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
        "route_candidates": route_candidates,
        "rejections": rejections,
        "unbound_validation_ids": unbound,
        "counts": {
            "program_proposals": len(proposals),
            "route_candidates": len(route_candidates),
            "validated_substitutions": sum(
                row["substitution_validated"] is True
                for row in route_candidates.values()
            ),
            "unvalidated_substitutions": sum(
                row["substitution_validated"] is not True
                for row in route_candidates.values()
            ),
            "rejected_candidates": len(rejections),
            "unbound_validations": len(unbound),
        },
        "compression_accounting": {
            "accepted": accounting_valid,
            "physical_step_mismatch_is_allowed_only_with_explicit_span": True,
        },
        "semantics": dict(_SEMANTICS),
    }
    return _with_digest(payload)


def biocatalytic_program_bundle_oracle(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Recompile a bundle and compare every proposal and accounting field."""

    try:
        expected = compile_biocatalytic_program_bundle(
            graph,
            route,
            projection,
            discovery,
            validations=validations,
        )
        observed_value = _strict_object(observed, "observed")
    except (BiocatalyticProgramError, TypeError, ValueError) as exc:
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
        "schema_equal": observed_value.get("schema_version")
        == BIOCATALYTIC_PROGRAM_BUNDLE_SCHEMA,
        "content_digest_valid": observed_digest
        == strict_canonical_json_sha256(material),
        "projection_equal": observed_value == expected,
        "compression_accounting_accepted": dict(
            observed_value.get("compression_accounting") or {}
        ).get("accepted")
        is True,
        "authority_semantics_equal": observed_value.get("semantics") == _SEMANTICS,
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
    validations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    innovation = dict(candidate.get("route_innovation") or {})
    _validate_innovation(candidate, innovation)
    span = [str(value) for value in innovation.get("replaced_step_ids") or []]
    boundary = _span_boundary(graph, route, span)
    observed_boundary = dict(candidate.get("boundary") or {})
    if observed_boundary.get("replaced_edge_ids") != span:
        raise BiocatalyticProgramError("candidate_replacement_span_mismatch")
    if observed_boundary.get("product_molecule_id") != boundary["output_molecule_ids"][0]:
        raise BiocatalyticProgramError("candidate_output_boundary_mismatch")
    if observed_boundary.get("precursor_molecule_id") not in boundary["input_molecule_ids"]:
        raise BiocatalyticProgramError("candidate_input_boundary_mismatch")
    identity = {
        "run_id": projection["run_id"],
        "graph_revision": projection["source_graph_revision"],
        "route_id": route.get("route_id"),
        "candidate_id": candidate.get("candidate_id"),
        "innovation_id": innovation.get("innovation_id"),
        "equivalent_reference_span": span,
    }
    proposal_id = f"program:biocatalytic:{strict_canonical_json_sha256(identity)[:24]}"
    input_states = [chemical_state_id(value) for value in boundary["input_molecule_ids"]]
    output_states = [chemical_state_id(value) for value in boundary["output_molecule_ids"]]
    gate = _validation_gate(
        proposal_id,
        innovation,
        input_states,
        output_states,
        validations,
    )
    proposal = {
        "schema_version": BIOCATALYTIC_PROGRAM_PROPOSAL_SCHEMA,
        "program_id": proposal_id,
        "proposal_kind": str(innovation["kind"]),
        "source_candidate_id": str(candidate.get("candidate_id") or ""),
        "source_innovation_id": str(innovation["innovation_id"]),
        "source_capability_id": str(candidate.get("capability_id") or ""),
        "input_state_ids": input_states,
        "output_state_ids": output_states,
        "operation_blueprints": [
            {
                "operation_kind": "enzyme_reaction",
                "sequence_index": 0,
                "input_state_ids": input_states,
                "output_state_ids": output_states,
            }
        ],
        "execution_domain": "enzymatic",
        "equivalent_reference_span": span,
        "replaced_program_ids": [program_id(value) for value in span],
        "chemical_step_equivalent_count": len(span),
        "isolated_operation_count": 1,
        "net_step_savings": len(span) - 1,
        "cofactor_and_carrier_ledger": {
            "requirements": dict(innovation.get("cofactor_requirements") or {}),
            "regenerations": dict(innovation.get("cofactor_regenerations") or {}),
        },
        "selectivity_constraints": [str(innovation["selectivity_objective"])],
        "claim_refs": {
            "precedent_refs": list(innovation.get("precedent_refs") or []),
            "exact_substrate_claim_refs": [],
            "analogy_only": True,
        },
        "validation_gate": gate,
        "validation_vector": {
            "structure": "boundary_states_materialized",
            "precedent": "analogy_only",
            "execution": "single_enzyme_operation_proposed",
            "biocatalysis": (
                "specialized_validation_bound" if gate["accepted"] else "screen_required"
            ),
            "process": "not_assessed",
            "authoritative": False,
        },
        "warning_codes": list(candidate.get("warning_codes") or []),
        "status": "admission_ready" if gate["accepted"] else "proposal_only",
        "eligible_for_shadow_admission": gate["accepted"],
        "semantics": {
            "not_a_canonical_reaction_edge": True,
            "does_not_inherit_replaced_edge_proof": True,
            "cannot_grant_route_completion": True,
        },
    }
    return _with_digest(proposal)


def _compile_route_candidate(
    route: dict[str, Any],
    projection: dict[str, Any],
    proposal: dict[str, Any],
) -> dict[str, Any]:
    edge_ids = [str(value) for value in route.get("edge_ids") or []]
    replaced = list(proposal["equivalent_reference_span"])
    try:
        substitution = substitute_program_span(
            route, replaced, str(proposal["program_id"])
        )
    except ProgramSpanError as exc:
        raise BiocatalyticProgramError(str(exc)) from exc
    fallback = substitution["fallback_program_ids"]
    selected = substitution["selected_program_ids"]
    route_identity = {
        "source_route_id": route.get("route_id"),
        "superstep_program_id": proposal["program_id"],
        "selected_program_ids": selected,
    }
    validated = proposal["validation_gate"]["accepted"] is True
    row = {
        "schema_version": BIOCATALYTIC_PROGRAM_ROUTE_SCHEMA,
        "route_candidate_id": (
            f"program-route:biocatalytic:{strict_canonical_json_sha256(route_identity)[:24]}"
        ),
        "source_route_id": str(route.get("route_id") or ""),
        "source_route_family_id": str(route.get("route_family_id") or ""),
        "source_projection_sha256": str(projection["content_sha256"]),
        "source_edge_ids": edge_ids,
        "fallback_program_ids": fallback,
        "selected_program_ids": selected,
        "superstep_program_id": str(proposal["program_id"]),
        "replaced_edge_ids": replaced,
        "replaced_program_ids": list(proposal["replaced_program_ids"]),
        "physical_step_count": len(selected),
        "chemical_step_equivalent_count": len(fallback),
        "net_step_savings": len(replaced) - 1,
        "substitution_validated": validated,
        "eligible_for_program_optimizer": validated,
        "eligible_for_route_completion": False,
        "status": "validated_candidate" if validated else "unvalidated_candidate",
        "semantics": {
            "fallback_route_is_retained": True,
            "non_equivalent_step_counts_are_intentional": len(replaced) > 1,
            "does_not_replace_production_route": True,
        },
    }
    return _with_digest(row)


def _oracle_result(
    accepted: bool,
    checks: Mapping[str, bool],
    reasons: list[str],
    expected_digest: str,
    observed_digest: str,
) -> dict[str, Any]:
    return _with_digest(
        {
            "schema_version": BIOCATALYTIC_PROGRAM_ORACLE_SCHEMA,
            "accepted": accepted,
            "checks": dict(checks),
            "reasons": sorted(set(reasons)),
            "expected_bundle_sha256": expected_digest,
            "observed_bundle_sha256": observed_digest,
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_cannot_admit_programs_or_complete_routes": True,
            },
        }
    )


__all__ = [
    "BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA",
    "BIOCATALYTIC_PROGRAM_BUNDLE_SCHEMA",
    "BIOCATALYTIC_PROGRAM_ORACLE_SCHEMA",
    "BIOCATALYTIC_PROGRAM_PROPOSAL_SCHEMA",
    "BIOCATALYTIC_PROGRAM_ROUTE_SCHEMA",
    "BiocatalyticProgramError",
    "biocatalytic_program_bundle_oracle",
    "compile_biocatalytic_program_bundle",
    "with_biocatalysis_program_validation_digest",
]

"""Proof-vector and inspector projections for the route workbench."""
from __future__ import annotations

import json
from typing import Any, Mapping


PROOF_VECTOR_SCHEMA = "retrosynthesis_proof_vector.v1"


def route_inspector(
    route: Mapping[str, Any],
    *,
    edge_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": route.get("stage"),
        "proof_level": route.get("proof_level"),
        "stock_closure_rate": route.get("stock_closure_rate"),
        "risk_score": route.get("risk_score"),
        "independent_source_groups": list(route.get("independent_source_groups") or []),
        "closure_profile": route.get("closure_profile"),
        "condition_complete": route.get("condition_complete") is True,
        "proof_vector": _copy_json(route.get("proof_vector") or {}),
        "edge_badges": {
            edge_id: list(dict(edge_rows.get(edge_id) or {}).get("badges") or [])
            for edge_id in route.get("edge_ids") or []
        },
    }


def edge_inspector(
    edge_id: str,
    *,
    graph: Mapping[str, Any],
    proof: Mapping[str, Any] | None,
) -> dict[str, Any]:
    proof_row = dict(proof or {})
    edge = dict(dict(graph.get("edges") or {}).get(edge_id) or {})
    source_ids = [str(value) for value in proof_row.get("source_binding_ids") or []]
    record_ids = [str(value) for value in proof_row.get("exact_record_ids") or []]
    conflict_ids = [str(value) for value in proof_row.get("conflict_ids") or []]
    exact_records = [
        _copy_json(dict(graph.get("exact_records") or {}).get(value) or {})
        for value in record_ids
    ]
    proof_vector = edge_proof_vector(edge=edge, proof=proof_row, graph=graph)
    return {
        "proof": _copy_json(proof_row),
        "proof_vector": proof_vector,
        "condition_status": proof_vector["conditions"],
        "condition_gap": (
            "no_replayable_reaction_conditions_bound"
            if proof_vector["conditions"] == "missing"
            else ""
        ),
        "reaction_proofs": _copy_json(edge.get("reaction_proofs") or []),
        "sources": [
            _copy_json(dict(graph.get("source_bindings") or {}).get(value) or {})
            for value in source_ids
        ],
        "exact_records": exact_records,
        "conflicts": [
            _copy_json(dict(graph.get("conflicts") or {}).get(value) or {})
            for value in conflict_ids
        ],
        "provenance": _copy_json(edge.get("origin_records") or []),
        "rejection_reasons": list(proof_row.get("reasons") or []),
    }


def edge_proof_vector(
    *, edge: Mapping[str, Any], proof: Mapping[str, Any], graph: Mapping[str, Any]
) -> dict[str, Any]:
    records = [
        dict(dict(graph.get("exact_records") or {}).get(str(record_id)) or {})
        for record_id in proof.get("exact_record_ids") or []
    ]
    records = [value for value in records if value]
    condition_records = [
        value
        for value in records
        if isinstance(value.get("conditions"), Mapping) and bool(value.get("conditions"))
    ]
    exact_procedure_records = [
        value
        for value in condition_records
        if str(
            value.get("procedure_authority_scope")
            or value.get("authority_scope")
            or ""
        )
        in {
            "source_exact_reaction_procedure",
            "source_exact_procedure_observation",
        }
    ]
    complete_procedure_records = [
        value
        for value in exact_procedure_records
        if dict(value.get("condition_completeness") or {}).get("complete") is True
    ]
    predicted = bool(
        edge.get("condition_predictions")
        or dict(edge.get("metadata") or {}).get("condition_predictions")
    )
    condition_state = (
        "source_exact"
        if exact_procedure_records
        else "source_recorded_unverified"
        if condition_records
        else "model_predicted"
        if predicted
        else "missing"
    )
    source_groups = {
        str(value)
        for value in proof.get("independent_source_groups") or []
        if str(value)
    }
    conflicted = bool(proof.get("conflict_ids"))
    source_state = (
        "conflicted"
        if conflicted
        else "independent_2_plus"
        if len(source_groups) >= 2
        else "single_group"
        if source_groups
        else "none"
    )
    reaction_state = (
        "source_reaction_exact"
        if exact_procedure_records and proof.get("reaction_validated") is True
        else "host_validated"
        if proof.get("reaction_validated") is True
        else "mapped"
        if edge.get("reaction_proofs")
        else "untested"
    )
    process_state = (
        "procedure_bound_candidate"
        if reaction_state in {"host_validated", "source_reaction_exact"}
        and bool(complete_procedure_records)
        and not conflicted
        else "blocked"
    )
    return {
        "schema_version": PROOF_VECTOR_SCHEMA,
        "identity": "source_exact" if records else "materialized",
        "reaction": reaction_state,
        "conditions": condition_state,
        "sources": source_state,
        "stock": "not_applicable_to_edge",
        "process": process_state,
        "condition_record_count": len(condition_records),
        "exact_procedure_record_count": len(exact_procedure_records),
        "complete_procedure_record_count": len(complete_procedure_records),
        "condition_completeness": (
            "complete"
            if complete_procedure_records
            else "partial"
            if condition_records
            else "missing"
        ),
        "semantics": {
            "axes_are_independent": True,
            "exact_structure_does_not_imply_exact_conditions": True,
            "display_projection_grants_no_authority": True,
        },
    }


def route_proof_vector(
    edges: list[Mapping[str, Any]],
    *,
    independent_source_groups: Any,
    closure_profile: str,
) -> dict[str, Any]:
    vectors = [dict(value.get("proof_vector") or {}) for value in edges]
    conditions = [str(value.get("conditions") or "missing") for value in vectors]
    condition_completeness = [
        str(value.get("condition_completeness") or "missing") for value in vectors
    ]
    reaction = [str(value.get("reaction") or "untested") for value in vectors]
    identities = [str(value.get("identity") or "proposed") for value in vectors]
    source_groups = {str(value) for value in independent_source_groups if str(value)}
    condition_state = (
        "missing"
        if not conditions or "missing" in conditions
        else "source_exact"
        if all(value == "source_exact" for value in conditions)
        else "mixed_supported"
    )
    return {
        "schema_version": PROOF_VECTOR_SCHEMA,
        "identity": (
            "all_source_exact"
            if identities and all(value == "source_exact" for value in identities)
            else "all_materialized"
            if identities
            and all(value in {"materialized", "source_exact"} for value in identities)
            else "incomplete"
        ),
        "reaction": (
            "all_validated"
            if reaction
            and all(
                value in {"host_validated", "source_reaction_exact"}
                for value in reaction
            )
            else "incomplete"
        ),
        "conditions": condition_state,
        "condition_completeness": (
            "complete"
            if condition_completeness
            and all(value == "complete" for value in condition_completeness)
            else "partial"
            if any(value == "partial" for value in condition_completeness)
            else "missing"
        ),
        "sources": (
            "independent_2_plus"
            if len(source_groups) >= 2
            else "single_group"
            if source_groups
            else "none"
        ),
        "stock": closure_profile,
        "process": "not_ready",
        "semantics": {
            "weakest_edge_controls_route_axis": True,
            "configured_boundary_closure_is_not_process_readiness": True,
            "display_projection_grants_no_authority": True,
        },
    }


def molecule_inspector(
    molecule_id: str, *, graph: Mapping[str, Any]
) -> dict[str, Any]:
    molecule = dict(dict(graph.get("molecules") or {}).get(molecule_id) or {})
    observation_ids = [
        str(value) for value in molecule.get("stock_observation_ids") or []
    ]
    return {
        "canonical_smiles": str(molecule.get("canonical_smiles") or ""),
        "stock_closed": molecule.get("stock_closed") is True,
        "stock_observations": [
            _copy_json(dict(graph.get("stock_observations") or {}).get(value) or {})
            for value in observation_ids
        ],
    }


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


__all__ = [
    "PROOF_VECTOR_SCHEMA",
    "edge_inspector",
    "edge_proof_vector",
    "molecule_inspector",
    "route_inspector",
    "route_proof_vector",
]

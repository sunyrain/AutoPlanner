"""Fail-closed validation for the Phase-1 TransformationProgram projection."""

from __future__ import annotations

import json
from typing import Any, Mapping

from cascade_planner.application.transformation_programs import (
    CHEMICAL_STATE_SCHEMA,
    OPERATION_NODE_SCHEMA,
    PROGRAM_PROJECTION_SCHEMA,
    TRANSFORMATION_PROGRAM_SCHEMA,
    chemical_state_id,
    operation_id,
    program_id,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


PROGRAM_PROJECTION_VALIDATION_SCHEMA = "transformation_program_projection_validation.v1"

_PROJECTION_KEYS = {
    "schema_version",
    "run_id",
    "source_graph_revision",
    "source_graph_scientific_sha256",
    "target_state_id",
    "chemical_states",
    "operation_nodes",
    "programs",
    "routes",
    "source_counts",
    "counts",
    "semantics",
    "content_sha256",
}
_STATE_KEYS = {
    "schema_version",
    "state_id",
    "parent_molecule_id",
    "canonical_smiles",
    "material_form",
    "mixture_state",
    "stock_observation_ids",
    "status",
    "semantics",
    "content_sha256",
}
_OPERATION_KEYS = {
    "schema_version",
    "operation_id",
    "operation_kind",
    "sequence_index",
    "input_state_ids",
    "output_state_ids",
    "procedure_record_ids",
    "condition_envelope",
    "source_edge_id",
    "status",
    "content_sha256",
}
_PROGRAM_KEYS = {
    "schema_version",
    "program_id",
    "input_state_ids",
    "output_state_ids",
    "net_atom_mapping",
    "reaction_centres",
    "operation_node_ids",
    "execution_domain",
    "equivalent_reference_span",
    "cofactor_and_carrier_ledger",
    "selectivity_constraints",
    "condition_envelope",
    "separation_plan",
    "claim_refs",
    "validation_vector",
    "source_edge_id",
    "status",
    "semantics",
    "content_sha256",
}
_ROUTE_KEYS = {
    "route_id",
    "source_edge_ids",
    "program_ids",
    "source_closed",
    "status",
    "content_sha256",
}
_PROJECTION_SEMANTICS = {
    "read_only_compatibility_projection": True,
    "one_program_per_canonical_edge": True,
    "edge_ids_remain_production_route_authority": True,
    "programs_cannot_change_proof_or_acceptance": True,
    "route_innovation_options_are_not_selected_by_this_projection": True,
}


def validate_program_projection(
    projection: Mapping[str, Any],
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate identities, references, multiplicity, digests, and authority."""

    reasons: list[str] = []
    try:
        value = _json_value(projection)
    except (TypeError, ValueError):
        return _result(
            accepted=False,
            reasons=["projection_not_strict_json"],
            projection_sha256="",
            counts={},
        )
    if not isinstance(value, dict):
        return _result(
            accepted=False,
            reasons=["projection_not_object"],
            projection_sha256="",
            counts={},
        )
    _exact_keys(value, _PROJECTION_KEYS, "projection", reasons)
    if value.get("schema_version") != PROGRAM_PROJECTION_SCHEMA:
        reasons.append("projection_schema_invalid")
    if expected_run_id is not None and value.get("run_id") != expected_run_id:
        reasons.append("projection_run_id_mismatch")
    _content_digest_valid(value, "projection", reasons)
    if value.get("semantics") != _PROJECTION_SEMANTICS:
        reasons.append("projection_authority_semantics_invalid")

    states = _object_map(value.get("chemical_states"), "chemical_states", reasons)
    operations = _object_map(value.get("operation_nodes"), "operation_nodes", reasons)
    programs = _object_map(value.get("programs"), "programs", reasons)
    routes = _object_map(value.get("routes"), "routes", reasons)
    state_ids = set(states)
    operation_ids = set(operations)
    program_ids = set(programs)

    for key, state in states.items():
        _exact_keys(state, _STATE_KEYS, f"state:{key}", reasons)
        _content_digest_valid(state, f"state:{key}", reasons)
        molecule_id = str(state.get("parent_molecule_id") or "")
        if (
            state.get("schema_version") != CHEMICAL_STATE_SCHEMA
            or not molecule_id
            or key != state.get("state_id")
            or key != chemical_state_id(molecule_id)
            or state.get("status") != "compatibility_projection"
            or not _string_list(state.get("stock_observation_ids"))
        ):
            reasons.append(f"state_contract_invalid:{key}")

    for key, operation in operations.items():
        _exact_keys(operation, _OPERATION_KEYS, f"operation:{key}", reasons)
        _content_digest_valid(operation, f"operation:{key}", reasons)
        edge_id = str(operation.get("source_edge_id") or "")
        inputs = operation.get("input_state_ids")
        outputs = operation.get("output_state_ids")
        if (
            operation.get("schema_version") != OPERATION_NODE_SCHEMA
            or not edge_id
            or key != operation.get("operation_id")
            or key != operation_id(edge_id)
            or operation.get("operation_kind") != "reaction"
            or operation.get("sequence_index") != 0
            or operation.get("status") != "compatibility_projection"
            or not _string_list(inputs)
            or not _string_list(outputs)
            or len(outputs) != 1
            or any(item not in state_ids for item in [*(inputs or []), *(outputs or [])])
            or not _string_list(operation.get("procedure_record_ids"))
        ):
            reasons.append(f"operation_contract_invalid:{key}")

    for key, program in programs.items():
        _exact_keys(program, _PROGRAM_KEYS, f"program:{key}", reasons)
        _content_digest_valid(program, f"program:{key}", reasons)
        edge_id = str(program.get("source_edge_id") or "")
        node_ids = program.get("operation_node_ids")
        inputs = program.get("input_state_ids")
        outputs = program.get("output_state_ids")
        node = operations.get(operation_id(edge_id)) if edge_id else None
        validation = program.get("validation_vector")
        semantics = program.get("semantics")
        if (
            program.get("schema_version") != TRANSFORMATION_PROGRAM_SCHEMA
            or not edge_id
            or key != program.get("program_id")
            or key != program_id(edge_id)
            or program.get("execution_domain") != "chemical"
            or program.get("status") != "compatibility_projection"
            or not _string_list(inputs)
            or not _string_list(outputs)
            or not _string_list(node_ids)
            or node_ids != [operation_id(edge_id)]
            or any(item not in state_ids for item in [*(inputs or []), *(outputs or [])])
            or any(item not in operation_ids for item in (node_ids or []))
            or program.get("equivalent_reference_span") != [edge_id]
            or not isinstance(validation, dict)
            or validation.get("authoritative") is not False
            or semantics
            != {
                "does_not_replace_source_edge": True,
                "does_not_select_route_innovation_option": True,
                "cannot_grant_acceptance": True,
            }
            or node is None
            or inputs != node.get("input_state_ids")
            or outputs != node.get("output_state_ids")
        ):
            reasons.append(f"program_contract_invalid:{key}")

    for key, route in routes.items():
        _exact_keys(route, _ROUTE_KEYS, f"route:{key}", reasons)
        _content_digest_valid(route, f"route:{key}", reasons)
        edge_ids = route.get("source_edge_ids")
        route_program_ids = route.get("program_ids")
        if (
            key != route.get("route_id")
            or route.get("status") != "compatibility_projection"
            or not _string_list(edge_ids)
            or not _string_list(route_program_ids)
            or route_program_ids != [program_id(edge_id) for edge_id in (edge_ids or [])]
            or any(item not in program_ids for item in (route_program_ids or []))
        ):
            reasons.append(f"route_contract_invalid:{key}")

    target_state_id = str(value.get("target_state_id") or "")
    if target_state_id and target_state_id not in state_ids:
        reasons.append("target_state_missing")
    expected_counts = {
        "chemical_states": len(states),
        "operation_nodes": len(operations),
        "programs": len(programs),
        "routes": len(routes),
    }
    if value.get("counts") != expected_counts:
        reasons.append("projection_counts_invalid")
    if value.get("source_counts") != {
        "molecules": expected_counts["chemical_states"],
        "edges": expected_counts["programs"],
        "route_families": expected_counts["routes"],
    }:
        reasons.append("projection_source_counts_invalid")
    return _result(
        accepted=not reasons,
        reasons=reasons,
        projection_sha256=str(value.get("content_sha256") or ""),
        counts=expected_counts,
    )


def _object_map(value: Any, label: str, reasons: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or any(not isinstance(row, dict) for row in value.values()):
        reasons.append(f"{label}_not_object_map")
        return {}
    return {str(key): dict(row) for key, row in value.items()}


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str, reasons: list[str]
) -> None:
    if set(value) != expected:
        reasons.append(f"{label}_fields_invalid")


def _content_digest_valid(value: Mapping[str, Any], label: str, reasons: list[str]) -> None:
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    if observed != strict_canonical_json_sha256(material):
        reasons.append(f"{label}_content_digest_invalid")


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _result(
    *, accepted: bool, reasons: list[str], projection_sha256: str, counts: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        "schema_version": PROGRAM_PROJECTION_VALIDATION_SCHEMA,
        "accepted": bool(accepted),
        "reasons": sorted(set(reasons)),
        "projection_sha256": projection_sha256,
        "counts": dict(counts),
        "semantics": {
            "contract_validation_only": True,
            "grants_no_scientific_authority": True,
        },
    }
    result["content_sha256"] = strict_canonical_json_sha256(result)
    return result


__all__ = ["PROGRAM_PROJECTION_VALIDATION_SCHEMA", "validate_program_projection"]

"""Read-only Phase-1 projection from V4 reaction edges to program entities.

The projection is deliberately non-authoritative.  It establishes stable
identities and a dual-read oracle before any production route switches from
``edge_ids`` to ``program_ids``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


CHEMICAL_STATE_SCHEMA = "chemical_state.v1"
OPERATION_NODE_SCHEMA = "operation_node.v1"
TRANSFORMATION_PROGRAM_SCHEMA = "transformation_program.v1"
PROGRAM_PROJECTION_SCHEMA = "transformation_program_projection.v1"
PROGRAM_PROJECTION_ORACLE_SCHEMA = "transformation_program_projection_oracle.v1"
_CANONICAL_GRAPH_SCHEMA = "canonical_retrosynthesis_hypergraph.v1"


def project_canonical_graph_to_programs(
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic compatibility projection without mutating V4."""

    if str(graph.get("schema_version") or "") != _CANONICAL_GRAPH_SCHEMA:
        raise ValueError("program_projection_requires_canonical_v4_graph")
    states = {
        chemical_state_id(molecule_id): _chemical_state(molecule_id, molecule)
        for molecule_id, molecule in sorted(
            dict(graph.get("molecules") or {}).items(), key=lambda row: str(row[0])
        )
    }
    operations: dict[str, dict[str, Any]] = {}
    programs: dict[str, dict[str, Any]] = {}
    for edge_id, edge in sorted(
        dict(graph.get("edges") or {}).items(), key=lambda row: str(row[0])
    ):
        operation = _operation_node(str(edge_id), dict(edge))
        operations[operation["operation_id"]] = operation
        program = _transformation_program(str(edge_id), dict(edge), operation)
        programs[program["program_id"]] = program
    routes = {
        str(route_id): _route_projection(str(route_id), dict(route))
        for route_id, route in sorted(
            dict(graph.get("route_families") or {}).items(),
            key=lambda row: str(row[0]),
        )
    }
    target_molecule_id = str(graph.get("target_molecule_id") or "")
    projection = {
        "schema_version": PROGRAM_PROJECTION_SCHEMA,
        "run_id": str(graph.get("run_id") or ""),
        "source_graph_revision": int(graph.get("revision") or 0),
        "source_graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "target_state_id": (chemical_state_id(target_molecule_id) if target_molecule_id else ""),
        "chemical_states": states,
        "operation_nodes": operations,
        "programs": programs,
        "routes": routes,
        "source_counts": {
            "molecules": len(dict(graph.get("molecules") or {})),
            "edges": len(dict(graph.get("edges") or {})),
            "route_families": len(dict(graph.get("route_families") or {})),
        },
        "counts": {
            "chemical_states": len(states),
            "operation_nodes": len(operations),
            "programs": len(programs),
            "routes": len(routes),
        },
        "semantics": {
            "read_only_compatibility_projection": True,
            "one_program_per_canonical_edge": True,
            "edge_ids_remain_production_route_authority": True,
            "programs_cannot_change_proof_or_acceptance": True,
            "route_innovation_options_are_not_selected_by_this_projection": True,
        },
    }
    return _with_digest(projection)


def program_projection_oracle(
    graph: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare every projected entity with a fresh canonical projection."""

    expected = project_canonical_graph_to_programs(graph)
    observed = _json_value(projection)
    checks = {
        "source_graph_bound": all(
            observed.get(key) == expected.get(key)
            for key in (
                "run_id",
                "source_graph_revision",
                "source_graph_scientific_sha256",
                "target_state_id",
            )
        ),
        "chemical_states_equal": observed.get("chemical_states") == expected["chemical_states"],
        "operation_nodes_equal": observed.get("operation_nodes") == expected["operation_nodes"],
        "programs_equal": observed.get("programs") == expected["programs"],
        "routes_equal": observed.get("routes") == expected["routes"],
        "source_counts_equal": observed.get("source_counts") == expected["source_counts"],
        "counts_equal": observed.get("counts") == expected["counts"],
        "content_digest_equal": observed.get("content_sha256") == expected["content_sha256"],
    }
    return _with_digest(
        {
            "schema_version": PROGRAM_PROJECTION_ORACLE_SCHEMA,
            "accepted": all(checks.values()),
            "checks": checks,
            "reasons": sorted(key for key, accepted in checks.items() if not accepted),
            "expected_projection_sha256": expected["content_sha256"],
            "observed_projection_sha256": str(observed.get("content_sha256") or ""),
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_does_not_authorize_program_writes": True,
            },
        }
    )


def chemical_state_id(molecule_id: str) -> str:
    identity = str(molecule_id or "").strip()
    if not identity:
        raise ValueError("chemical_state_molecule_id_required")
    return f"state:{identity}"


def program_id(edge_id: str) -> str:
    identity = str(edge_id or "").strip()
    if not identity:
        raise ValueError("transformation_program_edge_id_required")
    return f"program:{identity}"


def operation_id(edge_id: str) -> str:
    identity = str(edge_id or "").strip()
    if not identity:
        raise ValueError("operation_node_edge_id_required")
    return f"operation:reaction:{identity}"


def _chemical_state(molecule_id: str, molecule: Mapping[str, Any]) -> dict[str, Any]:
    return _with_digest(
        {
            "schema_version": CHEMICAL_STATE_SCHEMA,
            "state_id": chemical_state_id(str(molecule_id)),
            "parent_molecule_id": str(molecule_id),
            "canonical_smiles": str(molecule.get("canonical_smiles") or ""),
            "material_form": "unspecified",
            "mixture_state": "unspecified",
            "stock_observation_ids": _strings(molecule.get("stock_observation_ids")),
            "status": "compatibility_projection",
            "semantics": {
                "identity_inherited_from_canonical_v4": True,
                "not_yet_a_production_chemical_state": True,
            },
        }
    )


def _operation_node(
    edge_id: str,
    edge: Mapping[str, Any],
) -> dict[str, Any]:
    return _with_digest(
        {
            "schema_version": OPERATION_NODE_SCHEMA,
            "operation_id": operation_id(edge_id),
            "operation_kind": "reaction",
            "sequence_index": 0,
            "input_state_ids": [
                chemical_state_id(str(value)) for value in edge.get("precursor_molecule_ids") or []
            ],
            "output_state_ids": [chemical_state_id(str(edge.get("product_molecule_id") or ""))],
            "procedure_record_ids": _strings(edge.get("procedure_record_ids")),
            "condition_envelope": {},
            "source_edge_id": edge_id,
            "status": "compatibility_projection",
        }
    )


def _transformation_program(
    edge_id: str,
    edge: Mapping[str, Any],
    operation: Mapping[str, Any],
) -> dict[str, Any]:
    proof_digests = _strings(
        value.get("proof_digest")
        for value in edge.get("reaction_proofs") or []
        if isinstance(value, Mapping)
    )
    return _with_digest(
        {
            "schema_version": TRANSFORMATION_PROGRAM_SCHEMA,
            "program_id": program_id(edge_id),
            "input_state_ids": list(operation["input_state_ids"]),
            "output_state_ids": list(operation["output_state_ids"]),
            "net_atom_mapping": {},
            "reaction_centres": [],
            "operation_node_ids": [str(operation["operation_id"])],
            "execution_domain": "chemical",
            "equivalent_reference_span": [edge_id],
            "cofactor_and_carrier_ledger": {},
            "selectivity_constraints": [],
            "condition_envelope": {},
            "separation_plan": [],
            "claim_refs": {
                "source_binding_ids": _strings(edge.get("source_binding_ids")),
                "exact_record_ids": _strings(edge.get("exact_record_ids")),
                "procedure_record_ids": _strings(edge.get("procedure_record_ids")),
                "reaction_proof_digests": proof_digests,
            },
            "validation_vector": {
                "structure": "materialized",
                "precedent": "inherited_not_recomputed",
                "mechanism": "not_assessed",
                "execution": "single_reaction_edge_projected",
                "biocatalysis": "not_assessed",
                "process": "not_assessed",
                "procurement": "not_assessed",
                "conflict": "inherited_not_recomputed",
                "authoritative": False,
            },
            "source_edge_id": edge_id,
            "status": "compatibility_projection",
            "semantics": {
                "does_not_replace_source_edge": True,
                "does_not_select_route_innovation_option": True,
                "cannot_grant_acceptance": True,
            },
        }
    )


def _route_projection(route_id: str, route: Mapping[str, Any]) -> dict[str, Any]:
    edge_ids = _strings(route.get("edge_ids"))
    return _with_digest(
        {
            "route_id": route_id,
            "source_edge_ids": edge_ids,
            "program_ids": [program_id(value) for value in edge_ids],
            "source_closed": route.get("closed") is True,
            "status": "compatibility_projection",
        }
    )


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    return sorted(str(item) for item in values if str(item))


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


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _json_value(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return row


__all__ = [
    "CHEMICAL_STATE_SCHEMA",
    "OPERATION_NODE_SCHEMA",
    "PROGRAM_PROJECTION_ORACLE_SCHEMA",
    "PROGRAM_PROJECTION_SCHEMA",
    "TRANSFORMATION_PROGRAM_SCHEMA",
    "chemical_state_id",
    "operation_id",
    "program_id",
    "program_projection_oracle",
    "project_canonical_graph_to_programs",
]

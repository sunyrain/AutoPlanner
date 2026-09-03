"""Bounded route, edge, and molecule inspectors for the V4 workbench."""
from __future__ import annotations

import json
from typing import Any, Mapping

from cascade_planner.application.fact_lifecycle import graph_fact_lifecycle_state
from cascade_planner.application.route_innovations import merge_route_innovations
from cascade_planner.application.route_workbench_proof_vectors import (
    PROOF_VECTOR_SCHEMA as PROOF_VECTOR_SCHEMA,
    edge_proof_vector as edge_proof_vector,
    route_proof_vector as route_proof_vector,
)


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
        "acceptance_profiles": _copy_json(route.get("acceptance_profiles") or {}),
        "achieved_profiles": list(route.get("achieved_profiles") or []),
        "condition_complete": route.get("condition_complete") is True,
        "process_ready": route.get("process_ready") is True,
        "proof_vector": _copy_json(route.get("proof_vector") or {}),
        "inactive_fact_count": int(route.get("inactive_fact_count") or 0),
        "inactive_facts": _copy_json(route.get("inactive_facts") or []),
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
    procedure_source = (
        proof_row.get("procedure_record_ids")
        if "procedure_record_ids" in proof_row
        else edge.get("procedure_record_ids")
    )
    procedure_ids = [str(value) for value in procedure_source or []]
    exact_records = [
        _copy_json(dict(graph.get("exact_records") or {}).get(value) or {})
        for value in record_ids
    ]
    procedures = [
        _copy_json(dict(graph.get("procedure_records") or {}).get(value) or {})
        for value in procedure_ids
    ]
    procedures = [value for value in procedures if value]
    observation_source = (
        proof_row.get("source_observation_record_ids")
        if "source_observation_record_ids" in proof_row
        else edge.get("source_observation_record_ids")
    )
    source_observations = [
        _copy_json(
            dict(graph.get("source_observation_records") or {}).get(str(value))
            or {}
        )
        for value in observation_source or []
    ]
    source_observations = [value for value in source_observations if value]
    vector = edge_proof_vector(edge=edge, proof=proof_row, graph=graph)
    condition_gap = ""
    if not procedures and vector["conditions"] == "missing":
        condition_gap = "no_hash_bound_source_procedure"
    elif procedures and vector["condition_completeness"] == "missing":
        condition_gap = "source_procedure_located_conditions_unparsed"
    elif vector["condition_completeness"] == "partial":
        condition_gap = "source_procedure_conditions_incomplete"
    return {
        "proof": _copy_json(proof_row),
        "proof_vector": vector,
        "condition_status": vector["conditions"],
        "condition_gap": condition_gap,
        "condition_missing_required_groups": list(
            vector.get("condition_missing_required_groups") or []
        ),
        "condition_predictions": _copy_json(
            edge.get("condition_predictions")
            or dict(edge.get("metadata") or {}).get("condition_predictions")
            or []
        ),
        "condition_prediction_attempts": _copy_json(
            edge.get("condition_prediction_attempts") or []
        ),
        "reaction_proofs": _copy_json(edge.get("reaction_proofs") or []),
        "sources": [
            _copy_json(dict(graph.get("source_bindings") or {}).get(value) or {})
            for value in source_ids
        ],
        "exact_records": exact_records,
        "procedure_records": procedures,
        "source_observation_records": source_observations,
        "validation_findings": _copy_json(edge.get("validation_findings") or []),
        "route_innovations": _copy_json(
            merge_route_innovations((), edge.get("route_innovations") or [])
        ),
        "innovation_proof_gate": _copy_json(
            proof_row.get("innovation_proof_gate") or {}
        ),
        "inactive_facts": _copy_json(proof_row.get("inactive_facts") or []),
        "inactive_fact_count": int(proof_row.get("inactive_fact_count") or 0),
        "conflicts": [
            _copy_json(dict(graph.get("conflicts") or {}).get(value) or {})
            for value in proof_row.get("conflict_ids") or []
        ],
        "provenance": _copy_json(edge.get("origin_records") or []),
        "rejection_reasons": list(proof_row.get("reasons") or []),
    }


def molecule_inspector(
    molecule_id: str, *, graph: Mapping[str, Any]
) -> dict[str, Any]:
    molecule = dict(dict(graph.get("molecules") or {}).get(molecule_id) or {})
    observation_ids = [
        str(value) for value in molecule.get("stock_observation_ids") or []
    ]
    observations = [
        _copy_json(dict(graph.get("stock_observations") or {}).get(value) or {})
        for value in observation_ids
    ]
    lifecycle_states = [
        graph_fact_lifecycle_state(graph, "stock_observation", observation_id, observation)
        for observation_id, observation in zip(
            observation_ids, observations, strict=True
        )
        if observation
    ]
    return {
        "canonical_smiles": str(molecule.get("canonical_smiles") or ""),
        "stock_closed": molecule.get("stock_closed") is True,
        "stock_observations": observations,
        "stock_lifecycle_states": _copy_json(lifecycle_states),
        "inactive_fact_count": sum(
            value.get("active") is not True for value in lifecycle_states
        ),
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

"""Read-only Program projection for complete but scientifically open routes.

Canonical V4 must continue to reject malformed reaction edges.  A reported
route, however, may still be useful when one or more displayed transformations
omit atom-contributing reagents or have only visual structure support.  This
module preserves that distinction: it projects a digest-bound Workbench route
into non-authoritative candidate Programs without writing the canonical graph.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from rdkit import Chem

from cascade_planner.application.candidate_route_observations import (
    CANDIDATE_ROUTE_OBSERVATION_SCHEMA,
    CandidateProgramError,
    candidate_route_observation_from_workbench,
)
from cascade_planner.routes.admission import audit_retrosynthetic_candidate
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


CANDIDATE_CHEMICAL_STATE_SCHEMA = "candidate_chemical_state.v1"
CANDIDATE_OPERATION_NODE_SCHEMA = "candidate_operation_node.v1"
CANDIDATE_TRANSFORMATION_PROGRAM_SCHEMA = "candidate_transformation_program.v1"
CANDIDATE_PROGRAM_PROJECTION_SCHEMA = "candidate_program_projection.v1"
CANDIDATE_PROGRAM_ORACLE_SCHEMA = "candidate_program_projection_oracle.v1"
_RECOVERABLE_ADMISSION_REASONS = {
    "element_inventory_not_conserved",
    "large_atom_jump",
}
_FATAL_ADMISSION_REASONS = {
    "invalid_or_missing_material",
    "target_or_current_node_self_loop",
    "ancestor_or_target_cycle",
}


def project_candidate_route_to_programs(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a complete candidate route while isolating inadmissible spans."""

    source = _json_value(observation)
    if source.get("schema_version") != CANDIDATE_ROUTE_OBSERVATION_SCHEMA:
        raise CandidateProgramError("candidate_route_observation_schema_invalid")
    source_sha256 = _verified_digest(source, "candidate_route_observation_digest_invalid")
    molecules = _object_map(source.get("molecules"), "candidate_route_molecules_invalid")
    transformations = _object_map(
        source.get("transformations"), "candidate_route_transformations_invalid"
    )
    routes = _object_map(source.get("routes"), "candidate_route_routes_invalid")
    target = dict(source.get("target") or {})
    target_molecule_id = str(target.get("molecule_id") or "")
    _validate_observation_contract(
        source,
        target_molecule_id=target_molecule_id,
        molecules=molecules,
        transformations=transformations,
        routes=routes,
    )
    _validate_molecules(molecules)

    state_ids = {molecule_id: _state_id(molecule_id, row) for molecule_id, row in molecules.items()}
    states = {
        state_ids[molecule_id]: _with_digest(
            {
                "schema_version": CANDIDATE_CHEMICAL_STATE_SCHEMA,
                "state_id": state_ids[molecule_id],
                "source_molecule_id": molecule_id,
                "canonical_smiles": str(row["canonical_smiles"]),
                "label": str(row.get("label") or ""),
                "status": "candidate_route_observation",
                "authoritative": False,
            }
        )
        for molecule_id, row in sorted(molecules.items())
    }
    operations: dict[str, dict[str, Any]] = {}
    programs: dict[str, dict[str, Any]] = {}
    program_by_edge: dict[str, str] = {}
    for edge_id, row in sorted(transformations.items()):
        product_id = str(row.get("product_molecule_id") or "")
        precursor_ids = _strings(row.get("precursor_molecule_ids"))
        if product_id not in molecules or any(value not in molecules for value in precursor_ids):
            raise CandidateProgramError(f"candidate_route_edge_reference_invalid:{edge_id}")
        audit = audit_retrosynthetic_candidate(
            molecules[product_id]["canonical_smiles"],
            [molecules[value]["canonical_smiles"] for value in precursor_ids],
        )
        reasons = set(audit.get("reasons") or [])
        if reasons & _FATAL_ADMISSION_REASONS:
            raise CandidateProgramError(
                f"candidate_route_edge_fatal:{edge_id}:" + ",".join(sorted(reasons))
            )
        promotion_state = _promotion_state(audit)
        program_id = f"candidate-program:{strict_canonical_json_sha256({'edge_id': edge_id, 'edge_identity': audit['edge_identity']})[:24]}"
        operation_id = f"candidate-operation:{program_id.rsplit(':', 1)[-1]}"
        inputs = [state_ids[value] for value in precursor_ids]
        outputs = [state_ids[product_id]]
        conditions = _rows(row.get("condition_observations"))
        operations[operation_id] = _with_digest(
            {
                "schema_version": CANDIDATE_OPERATION_NODE_SCHEMA,
                "operation_id": operation_id,
                "operation_kind": "candidate_reaction",
                "input_state_ids": inputs,
                "output_state_ids": outputs,
                "condition_status": str(row.get("condition_status") or ""),
                "condition_observations": conditions,
                "source_transformation_id": edge_id,
                "status": promotion_state,
                "authoritative": False,
            }
        )
        warning_codes = sorted({*reasons, *_strings(row.get("warning_codes"))})
        proof_vector = dict(row.get("proof_vector") or {})
        programs[program_id] = _with_digest(
            {
                "schema_version": CANDIDATE_TRANSFORMATION_PROGRAM_SCHEMA,
                "program_id": program_id,
                "input_state_ids": inputs,
                "output_state_ids": outputs,
                "operation_node_ids": [operation_id],
                "source_transformation_id": edge_id,
                "source_refs": _strings(row.get("source_refs")),
                "warning_codes": warning_codes,
                "canonical_admission": _json_value(audit),
                "promotion_state": promotion_state,
                "validation_vector": {
                    "structure": "workbench_snapshot_materialized",
                    "canonical_search_admission": (
                        "passed" if audit.get("accepted") is True else "failed"
                    ),
                    "element_inventory": (
                        "gap" if reasons & _RECOVERABLE_ADMISSION_REASONS else "passed"
                    ),
                    "source": str(proof_vector.get("sources") or "none"),
                    "reaction": str(proof_vector.get("reaction") or "unvalidated"),
                    "conditions": str(proof_vector.get("conditions") or "missing"),
                    "authoritative": False,
                },
                "status": "candidate_only",
                "semantics": {
                    "cannot_enter_canonical_graph_without_host_admission": True,
                    "cannot_grant_route_closure_or_acceptance": True,
                    "inventory_gap_is_visible_not_silently_discarded": True,
                },
            }
        )
        program_by_edge[edge_id] = program_id

    projected_routes: dict[str, dict[str, Any]] = {}
    for route_id, route in sorted(routes.items()):
        edge_ids = _strings(route.get("edge_ids"))
        _validate_route_topology(
            route_id,
            edge_ids=edge_ids,
            target_molecule_id=target_molecule_id,
            transformations=transformations,
        )
        program_ids = [program_by_edge[value] for value in edge_ids]
        gap_ids = [
            value for value in program_ids if programs[value]["promotion_state"] == "inventory_gap"
        ]
        blocked_ids = [
            value
            for value in program_ids
            if programs[value]["promotion_state"] == "blocked_candidate"
        ]
        route_warnings = set(_strings(route.get("warning_codes")))
        if gap_ids:
            route_warnings.add("candidate_route_contains_inventory_gaps")
        if blocked_ids:
            route_warnings.add("candidate_route_contains_blocked_programs")
        projected_routes[route_id] = _with_digest(
            {
                "route_id": route_id,
                "source_edge_ids": edge_ids,
                "program_ids": program_ids,
                "canonical_admissible_program_ids": [
                    value
                    for value in program_ids
                    if programs[value]["promotion_state"] == "canonical_admissible"
                ],
                "inventory_gap_program_ids": gap_ids,
                "blocked_program_ids": blocked_ids,
                "source_exploration_closed": route.get("source_complete") is True,
                "production_closed": False,
                "accepted": False,
                "warning_codes": sorted(route_warnings),
                "status": "candidate_route_only",
            }
        )

    promotion_counts = {
        state: sum(program["promotion_state"] == state for program in programs.values())
        for state in ("canonical_admissible", "inventory_gap", "blocked_candidate")
    }
    projection = {
        "schema_version": CANDIDATE_PROGRAM_PROJECTION_SCHEMA,
        "observation_id": str(source.get("observation_id") or ""),
        "source_observation_sha256": source_sha256,
        "target_state_id": state_ids.get(target_molecule_id, ""),
        "chemical_states": states,
        "operation_nodes": operations,
        "programs": programs,
        "routes": projected_routes,
        "counts": {
            "chemical_states": len(states),
            "operation_nodes": len(operations),
            "programs": len(programs),
            "routes": len(projected_routes),
            **promotion_counts,
        },
        "semantics": {
            "read_only_candidate_projection": True,
            "canonical_v4_rejections_remain_in_force": True,
            "complete_display_route_does_not_mean_scientific_closure": True,
            "all_programs_are_non_authoritative": True,
            "no_program_store_admission_performed": True,
        },
    }
    return _with_digest(projection)


def candidate_program_projection_oracle(
    observation: Mapping[str, Any], projection: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the complete candidate projection and compare exact JSON."""

    expected = project_candidate_route_to_programs(observation)
    observed = _json_value(projection)
    checks = {
        "source_observation_bound": all(
            observed.get(key) == expected.get(key)
            for key in ("observation_id", "source_observation_sha256", "target_state_id")
        ),
        "chemical_states_equal": observed.get("chemical_states") == expected["chemical_states"],
        "operation_nodes_equal": observed.get("operation_nodes") == expected["operation_nodes"],
        "programs_equal": observed.get("programs") == expected["programs"],
        "routes_equal": observed.get("routes") == expected["routes"],
        "counts_equal": observed.get("counts") == expected["counts"],
        "content_digest_equal": observed.get("content_sha256") == expected["content_sha256"],
    }
    return _with_digest(
        {
            "schema_version": CANDIDATE_PROGRAM_ORACLE_SCHEMA,
            "accepted": all(checks.values()),
            "checks": checks,
            "reasons": sorted(key for key, value in checks.items() if not value),
            "expected_projection_sha256": expected["content_sha256"],
            "observed_projection_sha256": str(observed.get("content_sha256") or ""),
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_grants_no_scientific_authority": True,
            },
        }
    )


def _validate_molecules(molecules: Mapping[str, Mapping[str, Any]]) -> None:
    for molecule_id, row in molecules.items():
        if str(row.get("molecule_id") or "") != molecule_id:
            raise CandidateProgramError(f"candidate_route_molecule_id_invalid:{molecule_id}")
        supplied = str(row.get("canonical_smiles") or "")
        if not supplied or supplied != _canonical_smiles(supplied):
            raise CandidateProgramError(f"candidate_route_molecule_smiles_invalid:{molecule_id}")


def _validate_observation_contract(
    source: Mapping[str, Any],
    *,
    target_molecule_id: str,
    molecules: Mapping[str, Mapping[str, Any]],
    transformations: Mapping[str, Mapping[str, Any]],
    routes: Mapping[str, Mapping[str, Any]],
) -> None:
    if target_molecule_id not in molecules:
        raise CandidateProgramError("candidate_route_target_state_missing")
    expected_counts = {
        "molecules": len(molecules),
        "transformations": len(transformations),
        "routes": len(routes),
    }
    if source.get("counts") != expected_counts:
        raise CandidateProgramError("candidate_route_counts_invalid")
    expected_semantics = {
        "read_only_route_observation": True,
        "workbench_is_not_promoted_to_scientific_authority": True,
        "reported_route_may_remain_visible_with_open_transformations": True,
    }
    if source.get("semantics") != expected_semantics:
        raise CandidateProgramError("candidate_route_authority_semantics_invalid")
    for molecule_id, row in molecules.items():
        _verified_digest(row, f"candidate_route_molecule_digest_invalid:{molecule_id}")
    for edge_id, row in transformations.items():
        _verified_digest(row, f"candidate_route_edge_digest_invalid:{edge_id}")
        if str(row.get("transformation_id") or "") != edge_id:
            raise CandidateProgramError(f"candidate_route_edge_id_invalid:{edge_id}")
    referenced_edges: set[str] = set()
    for route_id, row in routes.items():
        _verified_digest(row, f"candidate_route_route_digest_invalid:{route_id}")
        if str(row.get("route_id") or "") != route_id:
            raise CandidateProgramError(f"candidate_route_route_id_invalid:{route_id}")
        referenced_edges.update(_strings(row.get("edge_ids")))
    if referenced_edges != set(transformations):
        raise CandidateProgramError("candidate_route_transformation_membership_invalid")


def _validate_route_topology(
    route_id: str,
    *,
    edge_ids: list[str],
    target_molecule_id: str,
    transformations: Mapping[str, Mapping[str, Any]],
) -> None:
    if not edge_ids or any(value not in transformations for value in edge_ids):
        raise CandidateProgramError(f"candidate_route_edges_invalid:{route_id}")
    products = [str(transformations[value].get("product_molecule_id") or "") for value in edge_ids]
    if len(products) != len(set(products)):
        raise CandidateProgramError(f"candidate_route_product_expanded_twice:{route_id}")
    reachable = {target_molecule_id}
    remaining = set(edge_ids)
    while remaining:
        consumed = sorted(
            value
            for value in remaining
            if str(transformations[value].get("product_molecule_id") or "") in reachable
        )
        if not consumed:
            break
        for edge_id in consumed:
            reachable.update(_strings(transformations[edge_id].get("precursor_molecule_ids")))
            remaining.remove(edge_id)
    if remaining:
        raise CandidateProgramError(
            f"candidate_route_disconnected:{route_id}:" + ",".join(sorted(remaining))
        )
    adjacency = {
        str(transformations[edge_id].get("product_molecule_id") or ""): _strings(
            transformations[edge_id].get("precursor_molecule_ids")
        )
        for edge_id in edge_ids
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(molecule_id: str) -> None:
        if molecule_id in visiting:
            raise CandidateProgramError(f"candidate_route_cycle:{route_id}")
        if molecule_id in visited:
            return
        visiting.add(molecule_id)
        for precursor_id in adjacency.get(molecule_id, []):
            visit(precursor_id)
        visiting.remove(molecule_id)
        visited.add(molecule_id)

    visit(target_molecule_id)


def _promotion_state(audit: Mapping[str, Any]) -> str:
    if audit.get("accepted") is True:
        return "canonical_admissible"
    reasons = set(audit.get("reasons") or [])
    if reasons and reasons <= _RECOVERABLE_ADMISSION_REASONS:
        return "inventory_gap"
    return "blocked_candidate"


def _state_id(molecule_id: str, molecule: Mapping[str, Any]) -> str:
    identity = {"molecule_id": molecule_id, "canonical_smiles": molecule["canonical_smiles"]}
    return f"candidate-state:{strict_canonical_json_sha256(identity)[:24]}"


def _verified_digest(value: Mapping[str, Any], error: str) -> str:
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    if not observed or observed != strict_canonical_json_sha256(material):
        raise CandidateProgramError(error)
    return observed


def _object_map(value: Any, error: str, *, allow_empty: bool = False) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or any(
        not isinstance(row, Mapping) for row in value.values()
    ):
        raise CandidateProgramError(error)
    result = {str(key): dict(row) for key, row in value.items()}
    if not result and not allow_empty:
        raise CandidateProgramError(error)
    return result


def _rows(value: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in value or [] if isinstance(row, Mapping)]


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    rows = [value] if isinstance(value, str) else value
    return [str(item) for item in rows if str(item)]


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if molecule is not None
        else ""
    )


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
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = [
    "CANDIDATE_PROGRAM_ORACLE_SCHEMA",
    "CANDIDATE_PROGRAM_PROJECTION_SCHEMA",
    "CANDIDATE_ROUTE_OBSERVATION_SCHEMA",
    "CandidateProgramError",
    "candidate_program_projection_oracle",
    "candidate_route_observation_from_workbench",
    "project_candidate_route_to_programs",
]

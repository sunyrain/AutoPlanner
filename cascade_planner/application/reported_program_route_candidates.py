"""Adapt digest-bound reported Candidate Programs into the common route space."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from rdkit import Chem

from cascade_planner.application.candidate_programs import (
    candidate_program_projection_oracle,
)
from cascade_planner.application.program_route_candidate_contracts import (
    ProgramRouteCandidateError,
)
from cascade_planner.application.program_route_candidate_factory import (
    build_program_route_candidate,
    canonical_route_authority_snapshot,
    normalize_strings,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


REPORTED_PROGRAM_ROUTE_PACK_SCHEMA = "reported_program_route_pack.v1"


def compile_reported_program_route_candidates(
    graph: Mapping[str, Any],
    canonical_route: Mapping[str, Any],
    *,
    baseline_program_ids: list[str],
    packs: Iterable[Mapping[str, Any]] = (),
) -> dict[str, dict[str, Any]]:
    """Compile full reported routes as exploration-only Program alternatives."""

    graph_value = _strict_object(graph, "graph")
    route_value = _strict_object(canonical_route, "canonical_route")
    target = _canonical_target(graph_value)
    candidates: dict[str, dict[str, Any]] = {}
    for pack_index, raw_pack in enumerate(packs):
        pack = _strict_object(raw_pack, f"reported_pack_{pack_index}")
        if pack.get("schema_version") != REPORTED_PROGRAM_ROUTE_PACK_SCHEMA:
            raise ProgramRouteCandidateError(
                f"reported_program_pack_schema_invalid:{pack_index}"
            )
        if set(pack) != {"schema_version", "observation", "projection", "route_ids"}:
            raise ProgramRouteCandidateError(
                f"reported_program_pack_fields_invalid:{pack_index}"
            )
        observation = _mapping(pack.get("observation"), "observation", pack_index)
        projection = _mapping(pack.get("projection"), "projection", pack_index)
        oracle = candidate_program_projection_oracle(observation, projection)
        if oracle.get("accepted") is not True:
            raise ProgramRouteCandidateError(
                f"reported_program_projection_not_current:{pack_index}"
            )
        _validate_target(target, projection, pack_index=pack_index)
        selected_route_ids = _selected_route_ids(pack, projection, pack_index)
        for reported_route_id in selected_route_ids:
            candidate = _compile_route(
                target=target,
                canonical_route=route_value,
                baseline_program_ids=baseline_program_ids,
                observation=observation,
                projection=projection,
                reported_route_id=reported_route_id,
            )
            candidates[candidate["candidate_id"]] = candidate
    return candidates


def _compile_route(
    *,
    target: dict[str, str],
    canonical_route: dict[str, Any],
    baseline_program_ids: list[str],
    observation: dict[str, Any],
    projection: dict[str, Any],
    reported_route_id: str,
) -> dict[str, Any]:
    projected_routes = dict(projection.get("routes") or {})
    observed_routes = dict(observation.get("routes") or {})
    reported_route = dict(projected_routes.get(reported_route_id) or {})
    observed_route = dict(observed_routes.get(reported_route_id) or {})
    if not reported_route or not observed_route:
        raise ProgramRouteCandidateError(
            f"reported_program_route_missing:{reported_route_id}"
        )
    programs = dict(projection.get("programs") or {})
    operations = dict(projection.get("operation_nodes") or {})
    program_ids = _ordered_strings(reported_route.get("program_ids"))
    if not program_ids or any(value not in programs for value in program_ids):
        raise ProgramRouteCandidateError(
            f"reported_program_route_programs_invalid:{reported_route_id}"
        )
    source_edge_ids = _ordered_strings(reported_route.get("source_edge_ids"))
    transformations = dict(observation.get("transformations") or {})
    if not source_edge_ids or any(value not in transformations for value in source_edge_ids):
        raise ProgramRouteCandidateError(
            f"reported_program_route_edges_invalid:{reported_route_id}"
        )
    source_refs = sorted(
        {
            *normalize_strings(observed_route.get("source_refs")),
            *(
                source_ref
                for program_id in program_ids
                for source_ref in normalize_strings(
                    dict(programs[program_id]).get("source_refs")
                )
            ),
        }
    )
    source_kind = "literature" if source_refs else "chemical"
    metrics = _reported_metrics(
        observation=observation,
        projection=projection,
        reported_route=reported_route,
        program_ids=program_ids,
        source_edge_ids=source_edge_ids,
        programs=programs,
        operations=operations,
    )
    observation_sha256 = str(observation.get("content_sha256") or "")
    projection_sha256 = str(projection.get("content_sha256") or "")
    identity = {
        "observation_sha256": observation_sha256,
        "projection_sha256": projection_sha256,
        "reported_route_id": reported_route_id,
    }
    warning_codes = {
        *normalize_strings(observed_route.get("warning_codes")),
        *normalize_strings(reported_route.get("warning_codes")),
        "REPORTED_CANDIDATE_NONAUTHORITATIVE",
        "CURRENT_CANONICAL_REPLAY_REQUIRED",
    }
    if not source_refs:
        warning_codes.add("SOURCE_PROVENANCE_MISSING")
    authority_snapshot = {
        "canonical_target": target,
        "canonical_route": canonical_route_authority_snapshot(canonical_route),
        "reported_observation_id": str(observation.get("observation_id") or ""),
        "reported_observation_sha256": observation_sha256,
        "reported_projection_sha256": projection_sha256,
        "reported_route_id": reported_route_id,
        "reported_program_ids": program_ids,
        "reported_inventory_gap_program_ids": normalize_strings(
            reported_route.get("inventory_gap_program_ids")
        ),
        "reported_blocked_program_ids": normalize_strings(
            reported_route.get("blocked_program_ids")
        ),
    }
    return build_program_route_candidate(
        candidate_id=(
            "program-route:reported:"
            + strict_canonical_json_sha256(identity)[:24]
        ),
        source_kind=source_kind,
        source_route_id=reported_route_id,
        program_ids=program_ids,
        fallback_program_ids=list(baseline_program_ids),
        substitution_program_ids=program_ids,
        execution_domains=["chemical"],
        metrics=metrics,
        shadow_optimizer=False,
        specialized_validation_ids=[],
        source_refs=source_refs,
        source_artifact_sha256s=[observation_sha256, projection_sha256],
        warning_codes=sorted(warning_codes),
        authority_snapshot=authority_snapshot,
    )


def _reported_metrics(
    *,
    observation: dict[str, Any],
    projection: dict[str, Any],
    reported_route: dict[str, Any],
    program_ids: list[str],
    source_edge_ids: list[str],
    programs: dict[str, Any],
    operations: dict[str, Any],
) -> dict[str, float | int]:
    transformations = dict(observation.get("transformations") or {})
    proof_levels = [
        max(0, int(dict(transformations[value]).get("proof_level") or 0))
        for value in source_edge_ids
    ]
    reaction_deficit = sum(
        dict(transformations[value]).get("source_step_accepted") is not True
        for value in source_edge_ids
    )
    condition_deficit = sum(
        not _program_conditions_complete(dict(programs[program_id]), operations)
        for program_id in program_ids
    )
    source_deficit = sum(
        not normalize_strings(dict(programs[program_id]).get("source_refs"))
        for program_id in program_ids
    )
    observed_route = dict(
        dict(observation.get("routes") or {}).get(
            str(reported_route.get("route_id") or "")
        )
        or {}
    )
    molecules = dict(observation.get("molecules") or {})
    leaves = normalize_strings(observed_route.get("leaf_molecule_ids"))
    procurement_deficit = max(
        1,
        sum(
            value not in molecules
            or dict(molecules[value]).get("source_stock_closed") is not True
            for value in leaves
        ),
    )
    inventory_gaps = len(
        normalize_strings(reported_route.get("inventory_gap_program_ids"))
    )
    blocked = len(normalize_strings(reported_route.get("blocked_program_ids")))
    physical = len(program_ids)
    return {
        "physical_operation_count": physical,
        "chemical_step_equivalent_count": physical,
        "net_step_savings": 0,
        "minimum_proof_level": min(proof_levels, default=0),
        "reaction_validation_deficit_count": reaction_deficit,
        "condition_deficit_count": condition_deficit,
        "specialized_validation_deficit_count": 0,
        "procurement_deficit_count": procurement_deficit,
        "process_deficit_count": physical,
        "source_deficit_count": source_deficit,
        "cofactor_system_count": 0,
        "risk_burden": float(inventory_gaps + 2 * blocked),
        "risk_data_deficit_count": 1,
    }


def _program_conditions_complete(
    program: dict[str, Any], operations: dict[str, Any]
) -> bool:
    operation_ids = normalize_strings(program.get("operation_node_ids"))
    if not operation_ids or any(value not in operations for value in operation_ids):
        return False
    observations = [
        dict(observation)
        for operation_id in operation_ids
        for observation in dict(operations[operation_id]).get("condition_observations")
        or []
        if isinstance(observation, Mapping)
    ]
    return bool(observations) and all(
        dict(value.get("condition_completeness") or {}).get("complete") is True
        for value in observations
    )


def _canonical_target(graph: dict[str, Any]) -> dict[str, str]:
    target_id = str(graph.get("target_molecule_id") or "")
    molecule = dict(dict(graph.get("molecules") or {}).get(target_id) or {})
    smiles = _canonical_smiles(molecule.get("canonical_smiles"))
    if not target_id or not smiles:
        raise ProgramRouteCandidateError("reported_program_canonical_target_invalid")
    return {"molecule_id": target_id, "canonical_smiles": smiles}


def _validate_target(
    target: dict[str, str], projection: dict[str, Any], *, pack_index: int
) -> None:
    target_state_id = str(projection.get("target_state_id") or "")
    state = dict(dict(projection.get("chemical_states") or {}).get(target_state_id) or {})
    reported_smiles = _canonical_smiles(state.get("canonical_smiles"))
    if not target_state_id or reported_smiles != target["canonical_smiles"]:
        raise ProgramRouteCandidateError(
            f"reported_program_target_mismatch:{pack_index}"
        )


def _selected_route_ids(
    pack: dict[str, Any], projection: dict[str, Any], pack_index: int
) -> list[str]:
    routes = dict(projection.get("routes") or {})
    route_ids = pack.get("route_ids")
    if not isinstance(route_ids, list):
        raise ProgramRouteCandidateError(
            f"reported_program_route_selection_invalid:{pack_index}"
        )
    selected = _ordered_strings(route_ids) if route_ids else sorted(routes)
    if not selected or any(value not in routes for value in selected):
        raise ProgramRouteCandidateError(
            f"reported_program_route_selection_invalid:{pack_index}"
        )
    return selected


def _ordered_strings(value: Any) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        return []
    rows = list(value)
    return rows if len(rows) == len(set(rows)) else []


def _mapping(value: Any, label: str, pack_index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgramRouteCandidateError(
            f"reported_program_pack_{label}_invalid:{pack_index}"
        )
    return dict(value)


def _strict_object(value: Any, label: str) -> dict[str, Any]:
    try:
        copied = json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ProgramRouteCandidateError(
            f"reported_program_{label}_not_strict_json"
        ) from exc
    if not isinstance(copied, dict):
        raise ProgramRouteCandidateError(f"reported_program_{label}_not_object")
    return copied


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if molecule is not None
        else ""
    )


__all__ = [
    "REPORTED_PROGRAM_ROUTE_PACK_SCHEMA",
    "compile_reported_program_route_candidates",
]

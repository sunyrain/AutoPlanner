"""Extract digest-bound candidate routes from a V4 Workbench snapshot."""

from __future__ import annotations

import json
from typing import Any, Mapping

from rdkit import Chem

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


CANDIDATE_ROUTE_OBSERVATION_SCHEMA = "candidate_route_observation.v1"
_WORKBENCH_SCHEMA = "retrosynthesis_route_workbench.v1"


class CandidateProgramError(ValueError):
    """A candidate route snapshot cannot be projected safely."""


def candidate_route_observation_from_workbench(
    workbench: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract only route-relevant, non-authoritative facts from a Workbench."""

    source = _json_value(workbench)
    if source.get("schema_version") != _WORKBENCH_SCHEMA:
        raise CandidateProgramError("candidate_workbench_schema_invalid")
    source_sha256 = _verified_digest(source, "candidate_workbench_digest_invalid")
    target = dict(source.get("target") or {})
    target_id = str(target.get("molecule_id") or "")
    target_smiles = _canonical_smiles(target.get("canonical_smiles"))
    if not target_id or not target_smiles:
        raise CandidateProgramError("candidate_workbench_target_invalid")

    source_routes = _object_map(source.get("routes"), "candidate_workbench_routes_invalid")
    source_edges = _object_map(source.get("edges"), "candidate_workbench_edges_invalid")
    source_molecules = _object_map(source.get("molecules"), "candidate_workbench_molecules_invalid")
    inspector_edges = _object_map(
        dict(source.get("inspectors") or {}).get("edges") or {},
        "candidate_workbench_inspectors_invalid",
        allow_empty=True,
    )
    route_edge_ids = {
        str(edge_id)
        for route in source_routes.values()
        for edge_id in route.get("edge_ids") or []
        if str(edge_id)
    }
    missing_edges = sorted(route_edge_ids - set(source_edges))
    if missing_edges:
        raise CandidateProgramError(
            "candidate_workbench_route_edges_missing:" + ",".join(missing_edges)
        )

    transformations: dict[str, dict[str, Any]] = {}
    referenced_molecules = {target_id}
    for edge_id in sorted(route_edge_ids):
        edge = source_edges[edge_id]
        product_id = str(edge.get("product_molecule_id") or "")
        precursor_ids = _strings(edge.get("precursor_molecule_ids"))
        if not product_id or not precursor_ids:
            raise CandidateProgramError(f"candidate_workbench_edge_invalid:{edge_id}")
        referenced_molecules.update({product_id, *precursor_ids})
        inspector = inspector_edges.get(edge_id, {})
        observations = [
            _condition_observation(value)
            for value in inspector.get("source_observation_records") or []
            if isinstance(value, Mapping)
        ]
        sources = [
            dict(value) for value in inspector.get("sources") or [] if isinstance(value, Mapping)
        ]
        transformations[edge_id] = _with_digest(
            {
                "transformation_id": edge_id,
                "product_molecule_id": product_id,
                "precursor_molecule_ids": precursor_ids,
                "source_step_accepted": edge.get("accepted") is True,
                "proof_level": int(edge.get("proof_level") or 0),
                "proof_vector": _json_value(edge.get("proof_vector") or {}),
                "provenance": _rows(inspector.get("provenance")),
                "source_refs": sorted(
                    {
                        str(value.get("source_ref") or "")
                        for value in sources
                        if str(value.get("source_ref") or "")
                    }
                ),
                "condition_status": str(inspector.get("condition_status") or ""),
                "condition_observations": observations,
                "warning_codes": sorted(
                    {
                        *_strings(edge.get("warning_codes")),
                        *_strings(inspector.get("rejection_reasons")),
                    }
                ),
            }
        )

    missing_molecules = sorted(referenced_molecules - set(source_molecules))
    if missing_molecules:
        raise CandidateProgramError(
            "candidate_workbench_route_molecules_missing:" + ",".join(missing_molecules)
        )
    molecules: dict[str, dict[str, Any]] = {}
    for molecule_id in sorted(referenced_molecules):
        source_molecule = source_molecules[molecule_id]
        canonical_smiles = _canonical_smiles(source_molecule.get("canonical_smiles"))
        if not canonical_smiles:
            raise CandidateProgramError(
                f"candidate_workbench_molecule_smiles_invalid:{molecule_id}"
            )
        molecules[molecule_id] = _with_digest(
            {
                "molecule_id": molecule_id,
                "canonical_smiles": canonical_smiles,
                "label": str(source_molecule.get("label") or ""),
                "role": str(source_molecule.get("role") or ""),
                "source_stock_closed": source_molecule.get("stock_closed") is True,
            }
        )
    routes = {
        route_id: _with_digest(
            {
                "route_id": route_id,
                "edge_ids": _strings(route.get("edge_ids")),
                "root_edge_ids": _strings(route.get("root_edge_ids")),
                "leaf_molecule_ids": _strings(route.get("leaf_molecule_ids")),
                "source_complete": route.get("complete") is True,
                "source_closure_profile": str(route.get("closure_profile") or ""),
                "source_refs": _strings(route.get("reported_source_refs")),
                "warning_codes": _strings(route.get("warning_codes")),
            }
        )
        for route_id, route in sorted(source_routes.items())
    }
    return _with_digest(
        {
            "schema_version": CANDIDATE_ROUTE_OBSERVATION_SCHEMA,
            "observation_id": f"candidate-route:{source_sha256[:24]}",
            "source_snapshot": {
                "schema_version": _WORKBENCH_SCHEMA,
                "run_id": str(source.get("run_id") or ""),
                "content_sha256": source_sha256,
                "revision": _json_value(source.get("revision") or {}),
            },
            "target": {
                "molecule_id": target_id,
                "canonical_smiles": target_smiles,
                "name": str(target.get("name") or ""),
            },
            "molecules": molecules,
            "transformations": transformations,
            "routes": routes,
            "counts": {
                "molecules": len(molecules),
                "transformations": len(transformations),
                "routes": len(routes),
            },
            "semantics": {
                "read_only_route_observation": True,
                "workbench_is_not_promoted_to_scientific_authority": True,
                "reported_route_may_remain_visible_with_open_transformations": True,
            },
        }
    )


def _condition_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    return _with_digest(
        {
            "record_id": str(row.get("record_id") or ""),
            "source_ref": str(row.get("source_ref") or ""),
            "location_refs": _strings(row.get("location_refs")),
            "conditions": _json_value(row.get("conditions") or {}),
            "condition_completeness": _json_value(row.get("condition_completeness") or {}),
            "grants_exact_structure_identity": False,
            "grants_reaction_validation": False,
        }
    )


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
    "CANDIDATE_ROUTE_OBSERVATION_SCHEMA",
    "CandidateProgramError",
    "candidate_route_observation_from_workbench",
]

"""Screen Candidate Program routes for enzyme windows without canonical writes."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.candidate_programs import (
    candidate_program_projection_oracle,
    project_candidate_route_to_programs,
)
from cascade_planner.application.route_innovation_capabilities import (
    normalize_biocatalysis_catalog,
)
from cascade_planner.application.route_innovation_discovery import (
    discover_route_innovations,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


CANDIDATE_INNOVATION_SCREEN_SCHEMA = "candidate_route_innovation_screen.v1"


def screen_candidate_route_innovations(
    observation: Mapping[str, Any],
    *,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals_by_route: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    max_window_steps: int = 8,
) -> dict[str, Any]:
    """Run proposal-only innovation discovery over validated candidate routes."""

    source = _json_value(observation)
    projection = project_candidate_route_to_programs(source)
    oracle = candidate_program_projection_oracle(source, projection)
    if oracle.get("accepted") is not True:
        raise ValueError("candidate_innovation_projection_oracle_failed")
    capability_rows, capability_rejections = normalize_biocatalysis_catalog(capabilities)
    graph = _screen_graph(source)
    proposal_map = dict(mechanism_proposals_by_route or {})
    screens: dict[str, dict[str, Any]] = {}
    for route_id, route in sorted(dict(source.get("routes") or {}).items()):
        route_row = dict(route)
        discovery = discover_route_innovations(
            graph,
            {
                "route_id": route_id,
                "route_family_id": f"candidate-family:{route_id}",
                "edge_ids": list(route_row.get("edge_ids") or []),
                "reported_source_refs": list(route_row.get("source_refs") or []),
            },
            capabilities=capability_rows,
            mechanism_proposals=proposal_map.get(route_id, ()),
            max_window_steps=max_window_steps,
        )
        enzyme_candidates = [
            row
            for row in discovery["candidates"]
            if row.get("candidate_kind") == "enzyme_window"
        ]
        mechanism_candidates = [
            row
            for row in discovery["candidates"]
            if row.get("candidate_kind") == "mechanism_one_hop"
        ]
        window_count = int(discovery["window_enumeration"]["count"])
        if not window_count:
            status = "not_screenable"
        elif not capability_rows:
            status = "no_accepted_capabilities"
        elif enzyme_candidates:
            status = "enzyme_candidates_found"
        else:
            status = "no_applicable_enzyme_capability"
        screens[route_id] = {
            "route_id": route_id,
            "screen_status": status,
            "negative_control_eligible": (
                status == "no_applicable_enzyme_capability" and bool(capability_rows)
            ),
            "enzyme_candidate_count": len(enzyme_candidates),
            "mechanism_candidate_count": len(mechanism_candidates),
            "discovery": discovery,
        }

    screen_rows = list(screens.values())
    capability_audit = {
        "accepted_capabilities": capability_rows,
        "rejected_capabilities": capability_rejections,
    }
    result = {
        "schema_version": CANDIDATE_INNOVATION_SCREEN_SCHEMA,
        "observation_sha256": str(source.get("content_sha256") or ""),
        "candidate_projection_sha256": projection["content_sha256"],
        "capability_audit_sha256": strict_canonical_json_sha256(capability_audit),
        "accepted_capability_count": len(capability_rows),
        "accepted_capability_ids": [row["capability_id"] for row in capability_rows],
        "rejected_capabilities": capability_rejections,
        "route_screens": screens,
        "counts": {
            "routes": len(screens),
            "screenable_routes": sum(
                row["screen_status"] != "not_screenable" for row in screen_rows
            ),
            "no_applicable_enzyme_routes": sum(
                row["negative_control_eligible"] is True for row in screen_rows
            ),
            "enzyme_candidates": sum(row["enzyme_candidate_count"] for row in screen_rows),
            "mechanism_candidates": sum(
                row["mechanism_candidate_count"] for row in screen_rows
            ),
        },
        "semantics": {
            "read_only": True,
            "candidate_graph_is_screening_input_not_scientific_authority": True,
            "target_names_are_not_matching_inputs": True,
            "zero_candidates_is_a_valid_screen_result": True,
            "enzyme_matches_are_proposal_only": True,
            "enzyme_windows_target_program_drafts_not_canonical_edges": True,
            "canonical_graph_not_modified": True,
            "program_store_admission_performed": False,
        },
    }
    return _with_digest(result)


def _screen_graph(source: Mapping[str, Any]) -> dict[str, Any]:
    molecules = {
        molecule_id: {
            "canonical_smiles": str(dict(row).get("canonical_smiles") or ""),
        }
        for molecule_id, row in dict(source.get("molecules") or {}).items()
    }
    edges = {
        edge_id: {
            "precursor_molecule_ids": list(dict(row).get("precursor_molecule_ids") or []),
            "product_molecule_id": str(dict(row).get("product_molecule_id") or ""),
            "innovation_boundary_proof_level": int(dict(row).get("proof_level") or 0),
        }
        for edge_id, row in dict(source.get("transformations") or {}).items()
    }
    return {"molecules": molecules, "edges": edges}


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


__all__ = ["CANDIDATE_INNOVATION_SCREEN_SCHEMA", "screen_candidate_route_innovations"]

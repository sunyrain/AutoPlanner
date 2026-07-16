"""Discover route-level enzyme windows and admit one-hop mechanism proposals."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem

from cascade_planner.application.route_innovation_capabilities import (
    match_biocatalysis_capability,
    normalize_biocatalysis_catalog,
)
from cascade_planner.application.route_execution_discovery import (
    discover_program_execution_windows,
    split_route_innovation_capabilities,
)
from cascade_planner.application.route_innovations import (
    BIOCATALYTIC_STEP,
    BIOCATALYTIC_SUPERSTEP,
    MECHANISM_EXTRAPOLATION,
    normalize_route_innovation,
)
from cascade_planner.application.route_innovation_windows import (
    enumerate_route_windows,
    molecule_smiles,
    route_window_boundary,
)


ROUTE_INNOVATION_DISCOVERY_SCHEMA = "route_innovation_discovery.v1"


def discover_route_innovations(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    *,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    max_window_steps: int = 8,
    max_candidates: int = 24,
) -> dict[str, Any]:
    """Return proposal-only innovations and canonical ingestion hypotheses."""

    biocatalytic_input, execution_input = split_route_innovation_capabilities(
        capabilities
    )
    capability_rows, capability_rejections = normalize_biocatalysis_catalog(
        biocatalytic_input
    )
    route_edge_ids = [str(value) for value in route.get("edge_ids") or []]
    enumeration = enumerate_route_windows(
        graph,
        route_edge_ids,
        max_window_steps=max(1, int(max_window_steps)),
    )
    paths = list(enumeration["windows"])
    execution_candidates, execution_rejections = discover_program_execution_windows(
        graph,
        route,
        paths,
        execution_input,
    )
    candidates: list[dict[str, Any]] = list(execution_candidates)
    rejected: list[dict[str, Any]] = [
        {"kind": "capability", **value} for value in capability_rejections
    ] + [{"kind": "execution_capability", **value} for value in execution_rejections]
    for path in paths:
        boundary = route_window_boundary(graph, path)
        if not boundary:
            continue
        for capability in capability_rows:
            audit = match_biocatalysis_capability(
                capability,
                boundary["precursor_smiles"],
                boundary["product_smiles"],
                window_steps=len(path),
            )
            if audit["accepted"] is not True:
                continue
            candidate = _enzyme_candidate(
                route,
                capability=capability,
                boundary=boundary,
                edge_ids=path,
                match_audit=audit,
            )
            if candidate:
                candidates.append(candidate)
    for proposal in mechanism_proposals:
        candidate, reasons = _mechanism_candidate(graph, route, proposal)
        if reasons:
            rejected.append(
                {
                    "kind": "mechanism_proposal",
                    "proposal_id": str(proposal.get("proposal_id") or ""),
                    "reasons": reasons,
                }
            )
        elif candidate:
            candidates.append(candidate)
    candidates = _rank_and_dedupe(candidates)[: max(1, int(max_candidates))]
    hypotheses = [
        _ingestion_hypothesis(route, value)
        for value in candidates
        if value.get("candidate_kind") == "mechanism_one_hop"
    ]
    result = {
        "schema_version": ROUTE_INNOVATION_DISCOVERY_SCHEMA,
        "route_id": str(route.get("route_id") or ""),
        "route_family_id": str(route.get("route_family_id") or ""),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "rejected": rejected,
        "window_enumeration": {
            key: enumeration[key] for key in ("count", "maximum", "truncated")
        },
        "program_draft_candidate_ids": sorted(
            str(value.get("candidate_id") or "")
            for value in candidates
            if value.get("candidate_kind") == "enzyme_window"
        ),
        "execution_program_draft_candidate_ids": sorted(
            str(value.get("candidate_id") or "")
            for value in candidates
            if value.get("candidate_kind") == "program_execution_window"
        ),
        "ingestion_hypotheses": hypotheses,
        "semantics": {
            "target_names_are_not_matching_inputs": True,
            "capabilities_are_data_not_code": True,
            "analog_matches_are_proposal_only": True,
            "mechanism_generation_and_host_admission_are_separate": True,
            "enzyme_windows_compile_to_program_drafts_not_reaction_edges": True,
            "whole_cell_and_hybrid_windows_compile_to_program_drafts": True,
            "canonical_ingestion_remains_the_only_scientific_write_path": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def canonical_innovation_batch(discovery: Mapping[str, Any]) -> dict[str, Any]:
    source_hypotheses = [
        dict(value) for value in discovery.get("ingestion_hypotheses") or []
    ]
    hypotheses = [
        value
        for value in source_hypotheses
        if value.get("origin_kind") == "mechanism_hypothesis"
    ]
    batch = {
        "schema_version": "canonical_route_innovation_ingestion_batch.v1",
        "route_id": str(discovery.get("route_id") or ""),
        "hypotheses": hypotheses,
        "excluded_program_candidate_ids": sorted(
            str(value)
            for value in discovery.get("program_draft_candidate_ids") or []
            if str(value)
        ),
        "semantics": {
            "submit_via_canonical_ingestion_batch": True,
            "materialization_does_not_grant_proof": True,
            "biocatalysis_requires_specialized_validation": True,
            "biocatalytic_supersteps_are_not_canonical_reaction_hypotheses": True,
        },
    }
    batch["content_sha256"] = _digest(batch)
    return batch


def _enzyme_candidate(
    route: Mapping[str, Any],
    *,
    capability: Mapping[str, Any],
    boundary: Mapping[str, Any],
    edge_ids: Sequence[str],
    match_audit: Mapping[str, Any],
) -> dict[str, Any]:
    kind = BIOCATALYTIC_SUPERSTEP if len(edge_ids) > 1 else BIOCATALYTIC_STEP
    enzyme = dict(capability.get("enzyme") or {})
    innovation, reasons = normalize_route_innovation(
        {
            "kind": kind,
            "chemical_step_equivalent_count": len(edge_ids),
            "replaced_step_ids": list(edge_ids),
            "enzyme": enzyme,
            "selectivity_objective": capability.get("selectivity_objective"),
            "substrate_scope_basis": capability.get("substrate_scope_basis"),
            "cofactor_requirements": capability.get("cofactor_requirements"),
            "cofactor_regenerations": capability.get("cofactor_regenerations"),
            "precedent_refs": capability.get("precedent_refs"),
            "validation_status": "proposed_screen_required",
            "route_family_id": route.get("route_family_id"),
        }
    )
    if reasons or not innovation:
        return {}
    identity = {
        "capability_id": capability["capability_id"],
        "edge_ids": list(edge_ids),
        "innovation_id": innovation["innovation_id"],
    }
    boundary_ready = int(boundary.get("minimum_boundary_proof_level") or 0) >= 1
    return {
        "candidate_id": f"route-innovation:{_digest(identity)[:24]}",
        "candidate_kind": "enzyme_window",
        "review_status": (
            "ready_for_enzyme_screen"
            if boundary_ready
            else "requires_boundary_materialization"
        ),
        "priority_score": round(float(match_audit.get("match_score") or 0.0), 6),
        "capability_id": capability["capability_id"],
        "boundary": dict(boundary),
        "route_innovation": innovation,
        "match_audit": dict(match_audit),
        "not_program_yet": True,
        "warning_codes": [
            "EXACT_SUBSTRATE_UNVALIDATED",
            *([] if boundary_ready else ["ROUTE_BOUNDARY_BELOW_L1"]),
        ],
    }


def _mechanism_candidate(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    proposal = dict(raw)
    precursor = _canonical(proposal.get("precursor_smiles"))
    product = _canonical(proposal.get("product_smiles"))
    route_edge_ids = {str(value) for value in route.get("edge_ids") or []}
    anchors = [str(value) for value in proposal.get("anchor_edge_ids") or []]
    source_refs = {str(value) for value in proposal.get("anchor_source_refs") or []}
    route_source_refs = {str(value) for value in route.get("reported_source_refs") or []}
    anchor_products = {
        _canonical(molecule_smiles(graph, dict(graph.get("edges") or {}).get(edge_id, {}).get("product_molecule_id")))
        for edge_id in anchors
        if edge_id in route_edge_ids
    }
    reasons: list[str] = []
    if not precursor or not product:
        reasons.append("mechanism_proposal_structure_invalid")
    if precursor == product:
        reasons.append("mechanism_proposal_self_loop")
    if not anchors or any(edge_id not in route_edge_ids for edge_id in anchors):
        reasons.append("mechanism_proposal_anchor_not_on_route")
    if precursor not in anchor_products:
        reasons.append("mechanism_proposal_not_one_hop_from_anchor_product")
    if not source_refs or not source_refs.intersection(route_source_refs):
        reasons.append("mechanism_proposal_source_not_bound_to_route")
    innovation, innovation_reasons = normalize_route_innovation(
        {
            **proposal,
            "kind": MECHANISM_EXTRAPOLATION,
            "hypothesis_depth": 1,
            "route_family_id": route.get("route_family_id"),
        }
    )
    reasons.extend(innovation_reasons)
    if reasons:
        return {}, sorted(set(reasons))
    boundary = {
        "precursor_molecule_id": _molecule_id_for_smiles(graph, precursor),
        "precursor_smiles": precursor,
        "product_molecule_id": "",
        "product_smiles": product,
        "replaced_edge_ids": [],
        "anchor_edge_ids": anchors,
        "new_connectivity_hypothesis": True,
    }
    return (
        {
            "candidate_id": str(
                proposal.get("proposal_id")
                or f"route-innovation:{_digest(boundary)[:24]}"
            ),
            "candidate_kind": "mechanism_one_hop",
            "review_status": "mechanism_review_only",
            "priority_score": float(proposal.get("priority_score") or 0.5),
            "boundary": boundary,
            "route_innovation": innovation,
            "not_canonical_edge_yet": True,
            "warning_codes": ["MECHANISM_HYPOTHESIS_UNVALIDATED"],
        },
        [],
    )


def _ingestion_hypothesis(
    route: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    boundary = dict(candidate.get("boundary") or {})
    innovation = dict(candidate.get("route_innovation") or {})
    return {
        "step_id": str(candidate.get("candidate_id") or ""),
        "product_smiles": str(boundary.get("product_smiles") or ""),
        "precursor_smiles": [str(boundary.get("precursor_smiles") or "")],
        "route_family_id": str(route.get("route_family_id") or ""),
        "origin_kind": (
            "mechanism_hypothesis"
            if innovation.get("kind") == MECHANISM_EXTRAPOLATION
            else "biocatalysis_hypothesis"
        ),
        "frontier_priority": float(candidate.get("priority_score") or 0.0),
        "route_innovation": innovation,
        "authority_scope": "proposal_only",
    }


def _rank_and_dedupe(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(
        {
            str(value.get("candidate_id") or ""): dict(value) for value in values
        }.values(),
        key=lambda row: (
            -int(
                dict(row.get("route_innovation") or {}).get("step_savings")
                or row.get("estimated_net_operation_savings")
                or 0
            ),
            -float(row.get("priority_score") or 0.0),
            str(row.get("candidate_id") or ""),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_boundaries: set[tuple[str, ...]] = set()
    for row in rows:
        innovation = dict(row.get("route_innovation") or {})
        edges = tuple(dict(row.get("boundary") or {}).get("replaced_edge_ids") or [])
        boundary_key = (
            str(row.get("candidate_kind") or ""),
            str(row.get("execution_domain") or innovation.get("kind") or ""),
            *edges,
        )
        if edges and boundary_key in seen_boundaries:
            continue
        edge_set = set(edges)
        if len(edges) > 1 and any(
            edge_set < set(dict(value.get("boundary") or {}).get("replaced_edge_ids") or [])
            and value.get("candidate_kind") == "enzyme_window"
            for value in selected
        ):
            continue
        selected.append(row)
        if edges:
            seen_boundaries.add(boundary_key)
    return selected


def _molecule_id_for_smiles(graph: Mapping[str, Any], smiles: str) -> str:
    return next(
        (
            str(molecule_id)
            for molecule_id, row in dict(graph.get("molecules") or {}).items()
            if _canonical(dict(row).get("canonical_smiles")) == smiles
        ),
        "",
    )


def _canonical(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or ""))
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else ""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "ROUTE_INNOVATION_DISCOVERY_SCHEMA",
    "canonical_innovation_batch",
    "discover_route_innovations",
]

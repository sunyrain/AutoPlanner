"""Validate mechanism discovery rows before Program materialization."""

from __future__ import annotations

from typing import Any, Mapping

from rdkit import Chem

from cascade_planner.application.route_innovations import (
    MECHANISM_EXTRAPOLATION,
    ROUTE_INNOVATION_SCHEMA,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def mechanism_candidate_reasons(
    route: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[str]:
    """Return fail-closed reasons for one mechanism discovery candidate."""

    innovation = dict(candidate.get("route_innovation") or {})
    material = dict(innovation)
    observed = str(material.pop("content_sha256", ""))
    boundary = dict(candidate.get("boundary") or {})
    anchor = dict(innovation.get("anchor") or {})
    reasons: list[str] = []
    if not str(candidate.get("candidate_id") or ""):
        reasons.append("mechanism_candidate_id_missing")
    if candidate.get("not_canonical_edge_yet") is not True:
        reasons.append("mechanism_candidate_canonical_boundary_invalid")
    if innovation.get("schema_version") != ROUTE_INNOVATION_SCHEMA:
        reasons.append("mechanism_innovation_schema_invalid")
    if observed != strict_canonical_json_sha256(material):
        reasons.append("mechanism_innovation_digest_invalid")
    if innovation.get("kind") != MECHANISM_EXTRAPOLATION:
        reasons.append("mechanism_innovation_kind_invalid")
    if innovation.get("hypothesis_depth") != 1:
        reasons.append("mechanism_innovation_depth_invalid")
    if innovation.get("reported_in_anchor_source") is not False:
        reasons.append("mechanism_anchor_authority_invalid")
    if not innovation.get("falsifiable_checks"):
        reasons.append("mechanism_falsifiable_checks_missing")
    precursor = _canonical_smiles(boundary.get("precursor_smiles"))
    product = _canonical_smiles(boundary.get("product_smiles"))
    if not precursor or not product or precursor == product:
        reasons.append("mechanism_boundary_structure_invalid")
    if boundary.get("new_connectivity_hypothesis") is not True:
        reasons.append("mechanism_boundary_hypothesis_flag_missing")
    if boundary.get("anchor_edge_ids") != anchor.get("edge_ids"):
        reasons.append("mechanism_anchor_boundary_mismatch")
    route_refs = {str(value) for value in route.get("reported_source_refs") or []}
    anchor_refs = {str(value) for value in anchor.get("source_refs") or []}
    if not anchor_refs or not anchor_refs.intersection(route_refs):
        reasons.append("mechanism_anchor_source_not_on_route")
    return sorted(set(reasons))


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if molecule is not None
        else ""
    )


__all__ = ["mechanism_candidate_reasons"]

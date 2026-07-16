"""Shared freshness and exact-state checks for Program validation frontiers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cascade_planner.application.route_innovation_discovery import (
    ROUTE_INNOVATION_DISCOVERY_SCHEMA,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


class ProgramValidationFrontierError(ValueError):
    """Validation frontier inputs are stale or lack exact boundary states."""


def validate_program_validation_frontier_inputs(
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    expected_bundle_schema: str,
) -> None:
    reasons: list[str] = []
    if discovery.get("schema_version") != ROUTE_INNOVATION_DISCOVERY_SCHEMA:
        reasons.append("validation_frontier_discovery_schema_invalid")
    if bundle.get("schema_version") != expected_bundle_schema:
        reasons.append("validation_frontier_bundle_schema_invalid")
    for label, value in (("discovery", discovery), ("bundle", bundle)):
        material = dict(value)
        observed = str(material.pop("content_sha256", ""))
        if observed != strict_canonical_json_sha256(material):
            reasons.append(f"validation_frontier_{label}_digest_invalid")
    if bundle.get("source_discovery_sha256") != discovery.get("content_sha256"):
        reasons.append("validation_frontier_discovery_binding_mismatch")
    if reasons:
        raise ProgramValidationFrontierError(";".join(sorted(set(reasons))))


def program_validation_state_snapshots(
    graph: Mapping[str, Any], state_ids: Sequence[Any]
) -> list[dict[str, str]]:
    molecules = dict(graph.get("molecules") or {})
    rows: list[dict[str, str]] = []
    for raw in state_ids:
        state_id = str(raw)
        if not state_id.startswith("state:"):
            raise ProgramValidationFrontierError(
                "validation_frontier_state_id_invalid"
            )
        molecule_id = state_id.removeprefix("state:")
        smiles = str(dict(molecules.get(molecule_id) or {}).get("canonical_smiles") or "")
        if not smiles:
            raise ProgramValidationFrontierError(
                "validation_frontier_state_structure_missing"
            )
        rows.append(
            {
                "state_id": state_id,
                "molecule_id": molecule_id,
                "canonical_smiles": smiles,
            }
        )
    if not rows:
        raise ProgramValidationFrontierError(
            "validation_frontier_boundary_states_missing"
        )
    return rows


__all__ = [
    "ProgramValidationFrontierError",
    "program_validation_state_snapshots",
    "validate_program_validation_frontier_inputs",
]

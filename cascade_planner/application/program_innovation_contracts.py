"""Shared fail-closed contracts for Program-level route innovations."""

from __future__ import annotations

import json
from typing import Any, Mapping

from cascade_planner.application.route_innovation_discovery import (
    ROUTE_INNOVATION_DISCOVERY_SCHEMA,
)
from cascade_planner.application.transformation_program_validation import (
    validate_program_projection,
)
from cascade_planner.application.transformation_programs import program_projection_oracle
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


CANONICAL_GRAPH_SCHEMA = "canonical_retrosynthesis_hypergraph.v1"


class ProgramInnovationContractError(ValueError):
    """Program innovation inputs are stale, malformed, or cross-boundary."""


def strict_program_innovation_object(
    value: Mapping[str, Any], label: str
) -> dict[str, Any]:
    try:
        copied = json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ProgramInnovationContractError(f"{label}_not_strict_json") from exc
    if not isinstance(copied, dict):
        raise ProgramInnovationContractError(f"{label}_not_object")
    return copied


def validate_program_innovation_inputs(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
) -> None:
    """Validate the shared graph/route/projection/discovery review boundary."""

    reasons: list[str] = []
    if graph.get("schema_version") != CANONICAL_GRAPH_SCHEMA:
        reasons.append("canonical_graph_schema_invalid")
    if validate_program_projection(
        projection, expected_run_id=str(graph.get("run_id") or "")
    ).get("accepted") is not True:
        reasons.append("program_projection_invalid")
    if program_projection_oracle(graph, projection).get("accepted") is not True:
        reasons.append("program_projection_not_current")
    route_id = str(route.get("route_id") or "")
    edge_ids = route.get("edge_ids")
    if not route_id or not _string_list(edge_ids) or not edge_ids:
        reasons.append("source_route_invalid")
    if discovery.get("schema_version") != ROUTE_INNOVATION_DISCOVERY_SCHEMA:
        reasons.append("innovation_discovery_schema_invalid")
    if discovery.get("route_id") != route_id:
        reasons.append("innovation_discovery_route_mismatch")
    material = dict(discovery)
    observed = str(material.pop("content_sha256", ""))
    if observed != strict_canonical_json_sha256(material):
        reasons.append("innovation_discovery_digest_invalid")
    if reasons:
        raise ProgramInnovationContractError(";".join(sorted(set(reasons))))


def with_program_innovation_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, str) and item for item in value
    )


__all__ = [
    "CANONICAL_GRAPH_SCHEMA",
    "ProgramInnovationContractError",
    "strict_program_innovation_object",
    "validate_program_innovation_inputs",
    "with_program_innovation_digest",
]

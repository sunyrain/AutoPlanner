"""Read-only route-to-Program overlay for an exact Workbench revision.

The canonical Workbench remains the UI and scientific authority.  This module
only proves whether every displayed edge can be read through the current
TransformationProgram projection without changing route identity, step count,
proof, conditions, or acceptance.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from cascade_planner.application.route_workbench import ROUTE_WORKBENCH_SCHEMA
from cascade_planner.application.transformation_program_validation import (
    validate_program_projection,
)
from cascade_planner.application.transformation_programs import (
    chemical_state_id,
    program_id,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


ROUTE_PROGRAM_DUAL_READ_SCHEMA = "route_program_dual_read.v1"
ROUTE_PROGRAM_DUAL_READ_ORACLE_SCHEMA = "route_program_dual_read_oracle.v1"

_ROUTE_COLLECTIONS = ("routes", "replacement_routes")
_ROUTE_AUTHORITY_FIELDS = (
    "route_family_id",
    "edge_ids",
    "root_edge_ids",
    "complete",
    "search_closed",
    "reaction_validated",
    "condition_complete",
    "literature_grounded",
    "process_ready",
    "procurement_closed",
    "configured_boundary_closed",
    "closure_profile",
    "proof_level",
    "proof_name",
    "proof_level_counts",
    "proof_vector",
    "acceptance_profiles",
    "achieved_profiles",
    "physical_step_count",
    "chemical_step_equivalent_count",
    "biocatalytic_step_count",
    "biocatalytic_superstep_count",
    "net_step_savings",
    "stock_boundary",
    "stock_closure_rate",
    "reported_in_source",
    "reported_source_refs",
    "warning_codes",
    "unproven_edge_ids",
    "unvalidated_biocatalytic_edge_ids",
)
_PORTFOLIO_AUTHORITY_FIELDS = (
    "route_ids",
    "default_route_id",
    "route_count",
    "accepted",
    "stock_boundary",
    "closure_profile",
    "achieved_profile",
    "acceptance_profile_counts",
    "process_ready",
    "closeout",
    "metrics",
)
_SEMANTICS = {
    "read_only_overlay": True,
    "edge_ids_remain_route_authority": True,
    "program_ids_are_secondary_identifiers": True,
    "proof_conditions_and_acceptance_are_copied_not_recomputed": True,
    "route_innovation_options_are_not_selected": True,
    "overlay_cannot_grant_completion": True,
}


class RouteProgramDualReadError(ValueError):
    """The Workbench and Program projection cannot be read as one revision."""


def project_workbench_routes_to_programs(
    workbench: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a route-row overlay without mutating either input."""

    workbench_value = _strict_object(workbench, "workbench")
    projection_value = _strict_object(projection, "program_projection")
    _validate_inputs(workbench_value, projection_value)

    programs = dict(projection_value["programs"])
    collections: dict[str, dict[str, Any]] = {}
    all_program_ids: set[str] = set()
    edge_reference_count = 0
    step_count_mismatches: list[str] = []
    for collection_name in _ROUTE_COLLECTIONS:
        source_rows = dict(workbench_value.get(collection_name) or {})
        rows: dict[str, Any] = {}
        for route_id, source_route in sorted(source_rows.items()):
            row = _route_overlay(
                collection_name=collection_name,
                route_id=str(route_id),
                source_route=source_route,
                programs=programs,
            )
            rows[str(route_id)] = row
            all_program_ids.update(row["program_ids"])
            edge_reference_count += len(row["edge_ids"])
            if row["step_count_equal"] is not True:
                step_count_mismatches.append(str(route_id))
        collections[collection_name] = rows

    portfolio = dict(workbench_value.get("portfolio") or {})
    portfolio_authority = {
        field: _copy_json(portfolio.get(field)) for field in _PORTFOLIO_AUTHORITY_FIELDS
    }
    route_ids = list(portfolio.get("route_ids") or [])
    selected_ids = list(collections["routes"])
    portfolio_identity_equal = (
        len(route_ids) == len(selected_ids) == len(set(route_ids))
        and set(route_ids) == set(selected_ids)
        and int(portfolio.get("route_count") or 0) == len(selected_ids)
    )
    checks = {
        "graph_revision_equal": True,
        "graph_scientific_digest_equal": True,
        "run_id_equal": True,
        "target_identity_equal": True,
        "all_displayed_edges_mapped": True,
        "portfolio_route_identity_equal": portfolio_identity_equal,
        "all_physical_step_counts_equal": not step_count_mismatches,
    }
    payload = {
        "schema_version": ROUTE_PROGRAM_DUAL_READ_SCHEMA,
        "run_id": str(workbench_value["run_id"]),
        "source_workbench_sha256": str(workbench_value["content_sha256"]),
        "source_projection_sha256": str(projection_value["content_sha256"]),
        "source_graph_revision": int(dict(workbench_value["revision"])["graph"]),
        "source_graph_scientific_sha256": str(
            dict(workbench_value["revision"])["graph_scientific_sha256"]
        ),
        "target_state_id": str(projection_value["target_state_id"]),
        "display_route_ids": route_ids,
        "collections": collections,
        "portfolio_authority": portfolio_authority,
        "portfolio_authority_sha256": strict_canonical_json_sha256(portfolio_authority),
        "counts": {
            "displayed_routes": len(collections["routes"]),
            "replacement_routes": len(collections["replacement_routes"]),
            "edge_references": edge_reference_count,
            "distinct_programs": len(all_program_ids),
            "physical_step_count_mismatches": len(step_count_mismatches),
        },
        "equivalence": {
            "accepted": all(checks.values()),
            "checks": checks,
            "step_count_mismatch_route_ids": sorted(step_count_mismatches),
        },
        "semantics": dict(_SEMANTICS),
    }
    return _with_digest(payload)


def route_program_dual_read_oracle(
    workbench: Mapping[str, Any],
    projection: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and compare a dual-read overlay under the current contracts."""

    reasons: list[str] = []
    try:
        expected = project_workbench_routes_to_programs(workbench, projection)
    except (RouteProgramDualReadError, TypeError, ValueError) as exc:
        return _oracle_result(
            accepted=False,
            checks={"inputs_reprojectable": False},
            reasons=[f"dual_read_inputs_invalid:{type(exc).__name__}"],
            expected_sha256="",
            observed_sha256="",
        )
    try:
        observed_value = _strict_object(observed, "observed")
    except RouteProgramDualReadError:
        return _oracle_result(
            accepted=False,
            checks={"inputs_reprojectable": True, "observed_strict_json": False},
            reasons=["dual_read_observed_not_strict_json_object"],
            expected_sha256=str(expected["content_sha256"]),
            observed_sha256="",
        )

    material = dict(observed_value)
    observed_sha256 = str(material.pop("content_sha256", ""))
    checks = {
        "inputs_reprojectable": True,
        "observed_schema_equal": (
            observed_value.get("schema_version") == ROUTE_PROGRAM_DUAL_READ_SCHEMA
        ),
        "observed_content_digest_valid": (
            observed_sha256 == strict_canonical_json_sha256(material)
        ),
        "projection_equal": observed_value == expected,
        "route_equivalence_accepted": (
            dict(observed_value.get("equivalence") or {}).get("accepted") is True
        ),
        "authority_semantics_equal": observed_value.get("semantics") == _SEMANTICS,
    }
    reasons.extend(key for key, accepted in checks.items() if accepted is not True)
    return _oracle_result(
        accepted=not reasons,
        checks=checks,
        reasons=reasons,
        expected_sha256=str(expected["content_sha256"]),
        observed_sha256=observed_sha256,
    )


def _validate_inputs(workbench: dict[str, Any], projection: dict[str, Any]) -> None:
    reasons: list[str] = []
    if workbench.get("schema_version") != ROUTE_WORKBENCH_SCHEMA:
        reasons.append("workbench_schema_invalid")
    material = dict(workbench)
    observed_workbench_sha256 = str(material.pop("content_sha256", ""))
    if observed_workbench_sha256 != strict_canonical_json_sha256(material):
        reasons.append("workbench_content_digest_invalid")
    if dict(workbench.get("semantics") or {}).get("canonical_graph_is_authority") is not True:
        reasons.append("workbench_authority_semantics_invalid")

    projection_validation = validate_program_projection(
        projection,
        expected_run_id=str(workbench.get("run_id") or ""),
    )
    if projection_validation.get("accepted") is not True:
        reasons.append("program_projection_invalid")
    revision = dict(workbench.get("revision") or {})
    if int(revision.get("graph") or 0) != int(projection.get("source_graph_revision") or 0):
        reasons.append("source_graph_revision_mismatch")
    if str(revision.get("graph_scientific_sha256") or "") != str(
        projection.get("source_graph_scientific_sha256") or ""
    ):
        reasons.append("source_graph_scientific_digest_mismatch")
    target = dict(workbench.get("target") or {})
    target_molecule_id = str(target.get("molecule_id") or "")
    if not target_molecule_id or projection.get("target_state_id") != chemical_state_id(
        target_molecule_id
    ):
        reasons.append("target_state_mismatch")

    programs = dict(projection.get("programs") or {})
    workbench_edges = dict(workbench.get("edges") or {})
    for collection_name in _ROUTE_COLLECTIONS:
        rows = workbench.get(collection_name)
        if not isinstance(rows, dict):
            reasons.append(f"{collection_name}_not_object_map")
            continue
        for route_id, route in rows.items():
            if not isinstance(route, dict) or route.get("route_id") != route_id:
                reasons.append(f"route_identity_invalid:{collection_name}:{route_id}")
                continue
            edge_ids = route.get("edge_ids")
            if not _string_list(edge_ids) or not edge_ids:
                reasons.append(f"route_edges_invalid:{collection_name}:{route_id}")
                continue
            for edge_id in edge_ids:
                expected_program_id = program_id(edge_id)
                program = dict(programs.get(expected_program_id) or {})
                if edge_id not in workbench_edges:
                    reasons.append(f"workbench_edge_missing:{route_id}:{edge_id}")
                if program.get("source_edge_id") != edge_id:
                    reasons.append(f"program_edge_mapping_missing:{route_id}:{edge_id}")
    if reasons:
        raise RouteProgramDualReadError(";".join(sorted(set(reasons))))


def _route_overlay(
    *,
    collection_name: str,
    route_id: str,
    source_route: Any,
    programs: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(source_route, Mapping):
        raise RouteProgramDualReadError(f"route_not_object:{collection_name}:{route_id}")
    source = _copy_json(source_route)
    edge_ids = [str(edge_id) for edge_id in source.get("edge_ids") or []]
    program_ids = [program_id(edge_id) for edge_id in edge_ids]
    if any(item not in programs for item in program_ids):
        raise RouteProgramDualReadError(f"route_program_missing:{collection_name}:{route_id}")
    authority_snapshot = {
        field: _copy_json(source.get(field)) for field in _ROUTE_AUTHORITY_FIELDS
    }
    source_physical_steps = int(source.get("physical_step_count") or len(edge_ids))
    row = {
        "collection": collection_name,
        "route_id": route_id,
        "route_family_id": str(source.get("route_family_id") or ""),
        "edge_ids": edge_ids,
        "program_ids": program_ids,
        "edge_program_pairs": [
            {"edge_id": edge_id, "program_id": mapped_program_id}
            for edge_id, mapped_program_id in zip(edge_ids, program_ids, strict=True)
        ],
        "edge_step_count": len(edge_ids),
        "program_step_count": len(program_ids),
        "source_physical_step_count": source_physical_steps,
        "step_count_equal": (
            len(edge_ids) == len(program_ids) == source_physical_steps
        ),
        "source_route_sha256": strict_canonical_json_sha256(source),
        "authority_snapshot": authority_snapshot,
        "authority_snapshot_sha256": strict_canonical_json_sha256(authority_snapshot),
        "semantics": {
            "source_route_row_unchanged": True,
            "program_ids_are_derived_from_edge_ids": True,
            "proof_and_conditions_are_not_recomputed": True,
        },
    }
    return _with_digest(row)


def _oracle_result(
    *,
    accepted: bool,
    checks: Mapping[str, bool],
    reasons: list[str],
    expected_sha256: str,
    observed_sha256: str,
) -> dict[str, Any]:
    return _with_digest(
        {
            "schema_version": ROUTE_PROGRAM_DUAL_READ_ORACLE_SCHEMA,
            "accepted": accepted,
            "checks": dict(checks),
            "reasons": sorted(set(reasons)),
            "expected_overlay_sha256": expected_sha256,
            "observed_overlay_sha256": observed_sha256,
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_cannot_switch_route_authority": True,
            },
        }
    )


def _strict_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        copied = json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise RouteProgramDualReadError(f"{label}_not_strict_json") from exc
    if not isinstance(copied, dict):
        raise RouteProgramDualReadError(f"{label}_not_object")
    return copied


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = strict_canonical_json_sha256(payload)
    return payload


__all__ = [
    "ROUTE_PROGRAM_DUAL_READ_ORACLE_SCHEMA",
    "ROUTE_PROGRAM_DUAL_READ_SCHEMA",
    "RouteProgramDualReadError",
    "project_workbench_routes_to_programs",
    "route_program_dual_read_oracle",
]

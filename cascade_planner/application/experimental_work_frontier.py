"""Read-only experiment subtasks bound to the single canonical work frontier.

This projection does not replace ``deficit_frontier.v1`` and cannot be
published to ``RunKernel``.  It only makes specialized Program validation
plans and exact-boundary calibration dirtiness schedulable by an external
executor adapter without inventing a second authority-owning queue.
"""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.capability_applicability_calibration import (
    CAPABILITY_CALIBRATION_SCHEMA,
)
from cascade_planner.application.deficit_frontier import DEFICIT_FRONTIER_SCHEMA
from cascade_planner.application.experiment_execution_contracts import (
    build_experiment_execution_request,
)
from cascade_planner.application.biocatalysis_validation_frontier import (
    BIOCATALYSIS_VALIDATION_FRONTIER_SCHEMA,
)
from cascade_planner.application.execution_validation_frontier import (
    EXECUTION_VALIDATION_FRONTIER_SCHEMA,
)
from cascade_planner.application.mechanism_validation_frontier import (
    MECHANISM_VALIDATION_FRONTIER_SCHEMA,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENTAL_WORK_FRONTIER_SCHEMA = "experimental_work_frontier.v1"
EXPERIMENTAL_WORK_ITEM_SCHEMA = "experimental_work_item.v1"
EXPERIMENTAL_WORK_FRONTIER_ORACLE_SCHEMA = "experimental_work_frontier_oracle.v1"

WORK_FRONTIER_SEMANTICS = {
    "projection_is_read_only": True,
    "canonical_deficit_frontier_remains_single_work_authority": True,
    "projection_cannot_be_published_to_run_kernel": True,
    "dirty_hints_schedule_recomputation_only": True,
    "work_items_grant_no_validation_claim_proof_or_completion": True,
    "executor_results_require_separate_domain_gate": True,
}


class ExperimentalWorkFrontierError(ValueError):
    """Validation plans cannot be safely projected into experiment work."""


def compile_experimental_work_frontier(
    canonical_frontier: Mapping[str, Any],
    biocatalysis_frontier: Mapping[str, Any],
    execution_frontier: Mapping[str, Any],
    mechanism_frontier: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    resource_hints: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile three domain plans into one non-authoritative subtask view."""

    canonical = _strict_digest_object(
        canonical_frontier, DEFICIT_FRONTIER_SCHEMA, "canonical_frontier"
    )
    domains = {
        "biocatalytic": _strict_digest_object(
            biocatalysis_frontier,
            BIOCATALYSIS_VALIDATION_FRONTIER_SCHEMA,
            "biocatalysis_frontier",
        ),
        "execution": _strict_digest_object(
            execution_frontier,
            EXECUTION_VALIDATION_FRONTIER_SCHEMA,
            "execution_frontier",
        ),
        "mechanism": _strict_digest_object(
            mechanism_frontier,
            MECHANISM_VALIDATION_FRONTIER_SCHEMA,
            "mechanism_frontier",
        ),
    }
    calibration_value = _strict_digest_object(
        calibration, CAPABILITY_CALIBRATION_SCHEMA, "calibration"
    )
    run_ids = {str(row.get("run_id") or "") for row in domains.values()}
    route_ids = {str(row.get("route_id") or "") for row in domains.values()}
    run_ids.add(str(calibration_value.get("run_id") or ""))
    route_ids.add(str(calibration_value.get("route_id") or ""))
    if len(run_ids) != 1 or "" in run_ids or len(route_ids) != 1 or "" in route_ids:
        raise ExperimentalWorkFrontierError("experimental_work_identity_mismatch")
    run_id = next(iter(run_ids))
    route_id = next(iter(route_ids))
    canonical_sha256 = str(canonical["content_sha256"])
    work_items: dict[str, dict[str, Any]] = {}
    hints_by_scope = _dirty_hints_by_scope(calibration_value)
    for domain, frontier in domains.items():
        for plan_id, raw_plan in sorted(dict(frontier.get("plans") or {}).items()):
            plan = _strict_plan(raw_plan, plan_id)
            item_identity = {
                "run_id": run_id,
                "route_id": route_id,
                "domain": domain,
                "plan_id": plan_id,
                "source_plan_sha256": plan["content_sha256"],
                "canonical_frontier_sha256": canonical_sha256,
            }
            item_id = "experimental-work:" + strict_canonical_json_sha256(item_identity)[:32]
            linked_deficits = _linked_deficit_ids(canonical, plan)
            dirty_ids = _matching_dirty_hint_ids(domain, plan, hints_by_scope)
            request = build_experiment_execution_request(
                run_id=run_id,
                route_id=route_id,
                work_item_id=item_id,
                domain=domain,
                plan=plan,
                canonical_frontier_sha256=canonical_sha256,
                resource_hints=dict(resource_hints or {}).get(domain),
            )
            item = {
                "schema_version": EXPERIMENTAL_WORK_ITEM_SCHEMA,
                "work_item_id": item_id,
                "work_kind": "experimental_validation_subtask",
                "domain": domain,
                "plan_id": plan_id,
                "program_id": str(plan.get("program_id") or ""),
                "source_plan_sha256": str(plan["content_sha256"]),
                "canonical_frontier_sha256": canonical_sha256,
                "linked_canonical_deficit_ids": linked_deficits,
                "canonical_anchor_kind": (
                    "linked_canonical_deficit" if linked_deficits else "route_scoped_shadow_work"
                ),
                "dirty_hint_ids": dirty_ids,
                "execution_request": request,
                "status": "executor_candidate",
                "grants_validation": False,
                "eligible_for_kernel_publication": False,
            }
            work_items[item_id] = _with_digest(item)
    dirty_hints = _compile_dirty_recompute_hints(
        calibration_value, work_items
    )
    payload = {
        "schema_version": EXPERIMENTAL_WORK_FRONTIER_SCHEMA,
        "run_id": run_id,
        "route_id": route_id,
        "canonical_frontier_ref": {
            "schema_version": DEFICIT_FRONTIER_SCHEMA,
            "graph_scientific_sha256": str(canonical.get("graph_scientific_sha256") or ""),
            "content_sha256": canonical_sha256,
        },
        "source_validation_frontiers": {
            domain: {
                "schema_version": str(frontier.get("schema_version") or ""),
                "content_sha256": str(frontier.get("content_sha256") or ""),
            }
            for domain, frontier in sorted(domains.items())
        },
        "source_calibration_sha256": str(calibration_value["content_sha256"]),
        "work_items": work_items,
        "dirty_recompute_hints": dirty_hints,
        "counts": {
            "work_items": len(work_items),
            "biocatalytic": sum(row["domain"] == "biocatalytic" for row in work_items.values()),
            "execution": sum(row["domain"] == "execution" for row in work_items.values()),
            "mechanism": sum(row["domain"] == "mechanism" for row in work_items.values()),
            "linked_canonical_deficits": sum(bool(row["linked_canonical_deficit_ids"]) for row in work_items.values()),
            "dirty_recompute_hints": len(dirty_hints),
            "kernel_publications": 0,
        },
        "semantics": dict(WORK_FRONTIER_SEMANTICS),
    }
    return _with_digest(payload)


def experimental_work_frontier_oracle(
    canonical_frontier: Mapping[str, Any],
    biocatalysis_frontier: Mapping[str, Any],
    execution_frontier: Mapping[str, Any],
    mechanism_frontier: Mapping[str, Any],
    calibration: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    resource_hints: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        expected = compile_experimental_work_frontier(
            canonical_frontier,
            biocatalysis_frontier,
            execution_frontier,
            mechanism_frontier,
            calibration,
            resource_hints=resource_hints,
        )
        observed_value = _strict_digest_object(
            observed, EXPERIMENTAL_WORK_FRONTIER_SCHEMA, "observed"
        )
        checks = {
            "inputs_reprojectable": True,
            "projection_equal": observed_value == expected,
            "authority_semantics_equal": observed_value.get("semantics")
            == WORK_FRONTIER_SEMANTICS,
        }
    except (ExperimentalWorkFrontierError, TypeError, ValueError):
        expected = {}
        observed_value = dict(observed) if isinstance(observed, Mapping) else {}
        checks = {"inputs_reprojectable": False}
    reasons = [key for key, accepted in checks.items() if accepted is not True]
    return _with_digest(
        {
            "schema_version": EXPERIMENTAL_WORK_FRONTIER_ORACLE_SCHEMA,
            "accepted": not reasons,
            "checks": checks,
            "reasons": reasons,
            "expected_frontier_sha256": str(expected.get("content_sha256") or ""),
            "observed_frontier_sha256": str(observed_value.get("content_sha256") or ""),
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_grants_no_work_or_scientific_authority": True,
            },
        }
    )


def _dirty_hints_by_scope(calibration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = dict(calibration.get("calibrations") or {})
    return {
        str(hint.get("calibration_id") or ""): {
            **dict(rows.get(str(hint.get("calibration_id") or "")) or {}),
            "hint": dict(hint),
        }
        for hint in calibration.get("dirty_domain_hints") or []
        if isinstance(hint, Mapping)
    }


def _matching_dirty_hint_ids(
    domain: str,
    plan: Mapping[str, Any],
    hints: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    boundary = dict(plan.get("exact_boundary") or {})
    input_ids = sorted(str(row.get("state_id") or "") for row in boundary.get("input_states") or [])
    output_ids = sorted(str(row.get("state_id") or "") for row in boundary.get("output_states") or [])
    matches = []
    plan_subjects = _plan_subject_refs(domain, plan)
    for calibration_id, row in hints.items():
        scope = dict(row.get("boundary") or {})
        subjects = dict(row.get("subject_refs") or {})
        if (
            row.get("domain") == domain
            and scope.get("input_state_ids") == input_ids
            and scope.get("output_state_ids") == output_ids
            and subjects == plan_subjects
        ):
            matches.append(calibration_id)
    return sorted(matches)


def _plan_subject_refs(domain: str, plan: Mapping[str, Any]) -> dict[str, str]:
    if domain == "execution":
        return {
            "capability_id": str(plan.get("capability_id") or ""),
            "execution_domain": str(plan.get("execution_domain") or ""),
        }
    if domain == "mechanism":
        return {"innovation_id": str(plan.get("innovation_id") or "")}
    return {
        "capability_id": str(plan.get("capability_id") or ""),
        "innovation_id": str(plan.get("innovation_id") or ""),
    }


def _compile_dirty_recompute_hints(
    calibration: Mapping[str, Any], work_items: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    scopes = _dirty_hints_by_scope(calibration)
    rows = []
    for calibration_id, scope in sorted(scopes.items()):
        hint = dict(scope.get("hint") or {})
        affected = sorted(
            item_id
            for item_id, item in work_items.items()
            if calibration_id in item.get("dirty_hint_ids", [])
        )
        rows.append(
            {
                "hint_id": "experimental-recompute:" + strict_canonical_json_sha256(hint)[:24],
                "calibration_id": calibration_id,
                "domain": str(scope.get("domain") or ""),
                "change_kind": str(hint.get("change_kind") or ""),
                "recompute_scope": "exact_boundary_only",
                "affected_work_item_ids": affected,
                "action": "recompute_route_program_review",
                "read_only": True,
            }
        )
    return rows


def _linked_deficit_ids(frontier: Mapping[str, Any], plan: Mapping[str, Any]) -> list[str]:
    replaced = {
        str(value)
        for value in dict(plan.get("canonical_context") or {}).get(
            "replaced_edge_ids"
        )
        or []
        if str(value)
    }
    return sorted(
        str(item.get("deficit_id") or "")
        for item in frontier.get("items") or []
        if isinstance(item, Mapping)
        and item.get("kind") == "validation"
        and replaced.intersection(str(value) for value in item.get("entity_ids") or [])
    )


def _strict_plan(value: Any, plan_id: str) -> dict[str, Any]:
    row = _strict_digest_object(value, "", "plan", allow_any_schema=True)
    if row.get("plan_id") != plan_id or not str(row.get("program_id") or ""):
        raise ExperimentalWorkFrontierError("experimental_work_plan_identity_invalid")
    return row


def _strict_digest_object(
    value: Any,
    schema: str,
    label: str,
    *,
    allow_any_schema: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentalWorkFrontierError(f"experimental_work_{label}_not_object")
    row = dict(value)
    material = dict(row)
    observed = str(material.pop("content_sha256", ""))
    if (
        (not allow_any_schema and row.get("schema_version") != schema)
        or not str(row.get("schema_version") or "")
        or not observed
        or observed != strict_canonical_json_sha256(material)
    ):
        raise ExperimentalWorkFrontierError(f"experimental_work_{label}_invalid")
    return row


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = [
    "EXPERIMENTAL_WORK_FRONTIER_ORACLE_SCHEMA",
    "EXPERIMENTAL_WORK_FRONTIER_SCHEMA",
    "EXPERIMENTAL_WORK_ITEM_SCHEMA",
    "ExperimentalWorkFrontierError",
    "compile_experimental_work_frontier",
    "experimental_work_frontier_oracle",
]

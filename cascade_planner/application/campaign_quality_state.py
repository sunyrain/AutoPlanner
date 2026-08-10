"""Full, non-exclusive quality projection for every campaign snapshot."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


CAMPAIGN_QUALITY_STATE_SCHEMA = "campaign_quality_state.v1"
CAMPAIGN_QUALITY_AXES = (
    "topology",
    "reaction_validation",
    "exact_evidence",
    "stock",
    "conditions",
    "procurement",
    "program_validation",
    "diversity",
)


def compile_campaign_quality_state(
    *,
    workbench: Mapping[str, Any] | None = None,
    gates: Mapping[str, Any] | None = None,
    minimum_routes: int | None = None,
    program_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile all quality axes without turning any axis into a solver mode."""

    snapshot = dict(workbench or {})
    gate_report = dict(gates or {})
    gate_values = dict(gate_report.get("gates") or {})
    routes = [
        dict(value)
        for value in dict(snapshot.get("routes") or {}).values()
        if isinstance(value, Mapping)
    ]
    required = max(
        1,
        int(
            minimum_routes
            or gate_report.get("minimum_routes")
            or dict(snapshot.get("campaign_summary") or {}).get("minimum_routes")
            or 2
        ),
    )
    route_counts = {
        "topology": len(routes),
        "reaction_validation": sum(
            row.get("reaction_validated") is True for row in routes
        ),
        "exact_evidence": sum(
            row.get("literature_grounded") is True for row in routes
        ),
        "stock": sum(
            row.get("configured_boundary_closed") is True for row in routes
        ),
        "conditions": sum(row.get("condition_complete") is True for row in routes),
        "procurement": sum(
            row.get("procurement_closed") is True for row in routes
        ),
    }
    count_overrides = {
        "topology": _gate_count(
            gate_report,
            "target_rooted_distinct_skeletons",
            route_counts["topology"],
        ),
        "reaction_validation": _gate_count(
            gate_report,
            "reaction_validated_skeletons",
            route_counts["reaction_validation"],
        ),
        "exact_evidence": _gate_count(
            gate_report,
            "evidence_closed_skeletons",
            route_counts["exact_evidence"],
        ),
        "stock": _gate_count(
            gate_report,
            "stock_closed_skeletons",
            route_counts["stock"],
        ),
    }
    route_counts.update(count_overrides)
    distinct_edge_sets = {
        tuple(sorted(str(value) for value in row.get("edge_ids") or [] if str(value)))
        for row in routes
        if row.get("edge_ids")
    }
    distinct_families = {
        str(row.get("route_family_id") or "")
        for row in routes
        if str(row.get("route_family_id") or "")
    }
    diversity_count = max(len(distinct_edge_sets), len(distinct_families))
    validation = dict(program_validation or {})
    program_observed = int(
        validation.get("validated_count")
        or validation.get("accepted_count")
        or 0
    )
    program_required = max(1, int(validation.get("required_count") or 1))
    program_assessed = bool(validation) or program_observed > 0

    axes = {
        "topology": _axis(
            route_counts["topology"],
            required,
            gate_values.get("B1_global_multi_route"),
            "canonical target-rooted route topology",
        ),
        "reaction_validation": _axis(
            route_counts["reaction_validation"],
            required,
            gate_values.get("B2_host_validated_routes"),
            "host reaction validation",
        ),
        "exact_evidence": _axis(
            route_counts["exact_evidence"],
            required,
            gate_values.get("B3_exact_multi_source"),
            "exact independently grouped source evidence",
        ),
        "stock": _axis(
            route_counts["stock"],
            required,
            gate_values.get("B4_stock_boundary"),
            "configured stock-oracle boundary",
            metadata={"boundary": _stock_boundary(snapshot, gate_report)},
        ),
        "conditions": _axis(
            route_counts["conditions"],
            required,
            None,
            "complete source or advisory reaction conditions",
            assessed=bool(routes),
        ),
        "procurement": _axis(
            route_counts["procurement"],
            required,
            None,
            "verified procurement or in-house availability",
            assessed=bool(routes),
        ),
        "program_validation": _axis(
            program_observed,
            program_required,
            validation.get("accepted"),
            "specialized Program validation",
            assessed=program_assessed,
        ),
        "diversity": _axis(
            diversity_count,
            required,
            None,
            "distinct edge sets or route families",
            assessed=bool(routes),
            metadata={
                "distinct_edge_set_count": len(distinct_edge_sets),
                "distinct_route_family_count": len(distinct_families),
            },
        ),
    }
    configured_acceptance = gate_values.get("B5_configured_portfolio_acceptance")
    if configured_acceptance not in {True, False}:
        configured_acceptance = dict(snapshot.get("portfolio") or {}).get("accepted")
    report = {
        "schema_version": CAMPAIGN_QUALITY_STATE_SCHEMA,
        "axes": axes,
        "configured_acceptance": configured_acceptance is True,
        "assessed_axis_count": sum(
            row["state"] != "not_assessed" for row in axes.values()
        ),
        "satisfied_axis_count": sum(row["satisfied"] is True for row in axes.values()),
        "semantics": {
            "axes_are_independent": True,
            "missing_assessment_is_not_failure_or_success": True,
            "configured_acceptance_is_a_snapshot_projection": True,
            "quality_state_does_not_select_actions": True,
        },
    }
    report["content_sha256"] = _digest(report)
    return report


def _axis(
    observed: int,
    required: int,
    gate_value: Any,
    basis: str,
    *,
    assessed: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    satisfied = bool(gate_value) if gate_value in {True, False} else observed >= required
    if not assessed and gate_value not in {True, False}:
        state = "not_assessed"
        satisfied_value: bool | None = None
    else:
        state = "satisfied" if satisfied else "open"
        satisfied_value = satisfied
    return {
        "state": state,
        "satisfied": satisfied_value,
        "observed_count": max(0, int(observed)),
        "required_count": max(1, int(required)),
        "basis": basis,
        "metadata": dict(metadata or {}),
    }


def _gate_count(
    report: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    counts = dict(report.get("counts") or {})
    return max(0, int(counts.get(key, fallback)))


def _stock_boundary(
    workbench: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> str:
    return str(
        gates.get("stock_boundary")
        or dict(workbench.get("portfolio") or {}).get("stock_boundary")
        or ""
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CAMPAIGN_QUALITY_AXES",
    "CAMPAIGN_QUALITY_STATE_SCHEMA",
    "compile_campaign_quality_state",
]

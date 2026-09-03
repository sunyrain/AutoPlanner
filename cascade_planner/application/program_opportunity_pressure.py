"""Target-blind value pressure for Program discovery and review Actions."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.condition_predictions import (
    edge_has_complete_source_procedure,
    edge_has_usable_condition_prediction,
)
from cascade_planner.application.deficit_frontier import DeficitScore
from cascade_planner.application.route_innovation_discovery import (
    discover_route_innovations,
)
from cascade_planner.application.route_innovation_windows import (
    enumerate_route_windows,
)


PROGRAM_OPPORTUNITY_PRESSURE_SCHEMA = "campaign_program_opportunity_pressure.v1"
PROGRAM_REVIEW_PRESSURE_SCHEMA = "campaign_program_review_pressure.v1"


def compile_program_opportunity_pressure(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    *,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    max_window_steps: int = 8,
) -> dict[str, Any]:
    """Estimate Program value from canonical route state and capability data."""

    route_row = dict(route)
    preview = discover_route_innovations(
        graph,
        route_row,
        capabilities=capabilities,
        mechanism_proposals=mechanism_proposals,
        max_window_steps=max(1, int(max_window_steps)),
    )
    candidates = [
        dict(value)
        for value in preview.get("candidates") or []
        if isinstance(value, Mapping)
    ]
    capability_candidates = [
        value
        for value in candidates
        if value.get("candidate_kind")
        in {"enzyme_window", "program_execution_window"}
    ]
    route_edges = dict(graph.get("edges") or {})
    open_selectivity_edges = {
        str(value) for value in route_row.get("unproven_edge_ids") or []
    }
    for edge_id in route_row.get("edge_ids") or []:
        edge = dict(route_edges.get(str(edge_id)) or {})
        if edge and not (
            edge_has_complete_source_procedure(graph, edge)
            or edge_has_usable_condition_prediction(edge)
        ):
            open_selectivity_edges.add(str(edge_id))
    matched_scores = [
        _unit(value.get("priority_score")) for value in capability_candidates
    ]
    selectivity_scores = [
        _unit(value.get("priority_score"))
        for value in capability_candidates
        if _selectivity_objective(value)
        and open_selectivity_edges.intersection(
            str(edge_id)
            for edge_id in dict(value.get("boundary") or {}).get(
                "replaced_edge_ids"
            )
            or []
        )
    ]
    window_enumeration = enumerate_route_windows(
        graph,
        [str(value) for value in route_row.get("edge_ids") or []],
        max_window_steps=max(1, int(max_window_steps)),
    )
    longest_span = max(
        (len(value) for value in window_enumeration["windows"]),
        default=0,
    )
    route_risk = _unit(route_row.get("risk_score"))
    span_scale = min(1.0, max(0.0, (longest_span - 2) / 4.0))
    high_cost_span = round(
        span_scale * (0.6 + 0.4 * route_risk),
        6,
    )
    known_match = max(matched_scores, default=0.0)
    selectivity_bottleneck = max(selectivity_scores, default=0.0)
    step_savings = max(
        (
            int(
                dict(value.get("route_innovation") or {}).get(
                    "step_savings"
                )
                or value.get("estimated_net_operation_savings")
                or 0
            )
            for value in candidates
        ),
        default=0,
    )
    replacement_gain = min(1.0, max(0, step_savings) / 4.0)
    mechanism_count = sum(
        value.get("candidate_kind") == "mechanism_one_hop"
        for value in candidates
    )
    mechanism_gain = min(1.0, mechanism_count / 3.0)
    components = {
        "high_cost_contiguous_span": high_cost_span,
        "known_capability_match": round(known_match, 6),
        "selectivity_bottleneck": round(selectivity_bottleneck, 6),
        "multi_step_replacement_gain": round(replacement_gain, 6),
        "mechanism_hypothesis_gain": round(mechanism_gain, 6),
    }
    pressure_total = round(
        min(
            1.0,
            0.24 * high_cost_span
            + 0.28 * known_match
            + 0.22 * selectivity_bottleneck
            + 0.18 * replacement_gain
            + 0.08 * mechanism_gain,
        ),
        6,
    )
    score = DeficitScore(
        expected_portfolio_gain=_unit(
            0.12
            + 0.18 * high_cost_span
            + 0.22 * known_match
            + 0.18 * selectivity_bottleneck
            + 0.12 * replacement_gain
            + 0.06 * mechanism_gain
        ),
        distance_to_closure=_unit(
            0.08
            + 0.12 * high_cost_span
            + 0.14 * known_match
            + 0.16 * selectivity_bottleneck
            + 0.12 * replacement_gain
        ),
        evidence_gain=_unit(0.08 + 0.08 * selectivity_bottleneck),
        route_diversity_gain=_unit(
            0.70 + 0.12 * known_match + 0.08 * mechanism_gain
        ),
        cost_penalty=0.10,
        failure_risk_penalty=0.10,
    ).to_dict()
    result = {
        "schema_version": PROGRAM_OPPORTUNITY_PRESSURE_SCHEMA,
        "route_id": str(route_row.get("route_id") or ""),
        "route_family_id": str(route_row.get("route_family_id") or ""),
        "route_innovation_preview_sha256": str(
            preview.get("content_sha256") or ""
        ),
        "candidate_count": len(candidates),
        "matched_capability_count": len(capability_candidates),
        "matched_capability_ids": sorted(
            {
                str(value.get("capability_id") or "")
                for value in capability_candidates
                if str(value.get("capability_id") or "")
            }
        ),
        "mechanism_candidate_count": mechanism_count,
        "longest_contiguous_window_steps": longest_span,
        "open_selectivity_edge_count": len(open_selectivity_edges),
        "maximum_step_savings": max(0, step_savings),
        "components": components,
        "pressure_total": pressure_total,
        "score": score,
        "legacy_priority": round(280.0 + 160.0 * pressure_total, 6),
        "semantics": {
            "target_labels_and_dataset_metadata_are_not_inputs": True,
            "route_cost_is_a_host_proxy_not_a_vendor_quote": True,
            "capability_matches_are_search_priors_only": True,
            "pressure_grants_no_program_or_reaction_authority": True,
            "conventional_route_remains_the_explicit_fallback": True,
            "single_action_frontier_and_budget_are_unchanged": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def compile_program_review_pressure(
    route_pressures: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate digest-valid route pressures into one read-only review value."""

    rows = [
        row
        for value in route_pressures
        for row in (_verified_pressure(value),)
        if row
    ]
    component_names = (
        "high_cost_contiguous_span",
        "known_capability_match",
        "selectivity_bottleneck",
        "multi_step_replacement_gain",
        "mechanism_hypothesis_gain",
    )
    components = {
        name: max(
            (float(dict(row.get("components") or {}).get(name) or 0.0) for row in rows),
            default=0.0,
        )
        for name in component_names
    }
    candidate_route_fraction = (
        sum(int(row.get("candidate_count") or 0) > 0 for row in rows)
        / max(1, len(rows))
    )
    pressure_total = round(
        min(
            1.0,
            max((float(row.get("pressure_total") or 0.0) for row in rows), default=0.0)
            + 0.15 * candidate_route_fraction,
        ),
        6,
    )
    score = DeficitScore(
        expected_portfolio_gain=_unit(0.10 + 0.22 * pressure_total),
        distance_to_closure=_unit(0.10 + 0.16 * pressure_total),
        evidence_gain=_unit(0.10 + 0.08 * components["selectivity_bottleneck"]),
        route_diversity_gain=_unit(0.20 + 0.28 * pressure_total),
        cost_penalty=0.05,
        failure_risk_penalty=0.02,
    ).to_dict()
    result = {
        "schema_version": PROGRAM_REVIEW_PRESSURE_SCHEMA,
        "route_pressure_count": len(rows),
        "candidate_route_fraction": round(candidate_route_fraction, 6),
        "components": components,
        "pressure_total": pressure_total,
        "route_pressure_sha256s": sorted(
            str(row.get("content_sha256") or "") for row in rows
        ),
        "score": score,
        "legacy_priority": round(260.0 + 140.0 * pressure_total, 6),
        "semantics": {
            "only_digest_valid_route_pressures_are_aggregated": True,
            "review_is_read_only_and_grants_no_authority": True,
            "conventional_fallbacks_are_retained": True,
            "budget_and_action_count_are_not_expanded": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def _selectivity_objective(candidate: Mapping[str, Any]) -> str:
    return str(
        dict(candidate.get("route_innovation") or {}).get(
            "selectivity_objective"
        )
        or dict(candidate.get("execution_capability") or {}).get(
            "selectivity_objective"
        )
        or ""
    )


def _verified_pressure(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    content_sha256 = str(row.pop("content_sha256", ""))
    if (
        row.get("schema_version") != PROGRAM_OPPORTUNITY_PRESSURE_SCHEMA
        or len(content_sha256) != 64
        or _digest(row) != content_sha256
    ):
        return {}
    return {**row, "content_sha256": content_sha256}


def _unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(1.0, max(0.0, number)), 6)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "PROGRAM_OPPORTUNITY_PRESSURE_SCHEMA",
    "PROGRAM_REVIEW_PRESSURE_SCHEMA",
    "compile_program_opportunity_pressure",
    "compile_program_review_pressure",
]

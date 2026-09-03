"""Content-bound anytime snapshots for one target-blind campaign trajectory."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from cascade_planner.application.route_workbench_edge_proof_vector import (
    edge_proof_vector,
)


LEGACY_CAMPAIGN_SNAPSHOT_SCHEMA = "campaign_anytime_snapshot.v1"
CAMPAIGN_SNAPSHOT_SCHEMA = "campaign_anytime_snapshot.v2"
CAMPAIGN_TRAJECTORY_SCHEMA = "campaign_trajectory.v2"
TRAJECTORY_BINDINGS_SCHEMA = "campaign_trajectory_bindings.v1"
TRAJECTORY_CUTOFF_PROJECTION_SCHEMA = "campaign_trajectory_cutoff_projection.v1"

_CUTOFF_FIELDS = {
    "wall_time_s",
    "event_sequence",
    "attempt_count",
    "accepted_expansion_count",
    "settled_task_count",
    "model_invocations",
    "visual_invocations",
    "input_tokens",
    "output_tokens",
    "model_wall_time_s",
    "native_search_invocations",
}
_FLOAT_CUTOFF_FIELDS = {"wall_time_s", "model_wall_time_s"}


def compile_trajectory_bindings(
    *,
    code: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    input_summary: Mapping[str, Any] | None = None,
    stock_oracle: Mapping[str, Any] | None = None,
    providers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the runtime identities that make a snapshot reproducible.

    Missing sections remain explicitly unobserved rather than receiving a
    fabricated identity. Production callers are expected to bind all five.
    """

    sections = {
        "code": _binding_section("code", code),
        "config": _binding_section("config", config),
        "input": _binding_section("input", input_summary),
        "stock_oracle": _binding_section("stock_oracle", stock_oracle),
        "providers": _binding_section("providers", providers),
    }
    result = {
        "schema_version": TRAJECTORY_BINDINGS_SCHEMA,
        **sections,
        "complete": all(row["observed"] is True for row in sections.values()),
        "semantics": {
            "bindings_describe_observed_runtime_identity": True,
            "missing_binding_is_not_inferred": True,
            "binding_changes_create_a_new_trajectory_epoch": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def compile_route_snapshot(
    *,
    graph: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile compact route counts and the non-dominated route archive."""

    candidates = [
        dict(value)
        for value in portfolio.get("route_candidates") or []
        if isinstance(value, Mapping)
    ]
    selected = [
        dict(value)
        for value in portfolio.get("selected_routes") or []
        if isinstance(value, Mapping)
    ]
    pareto = [value for value in candidates if value.get("pareto_optimal") is True]
    gate_counts = dict(gates.get("counts") or {})
    proofs = dict(portfolio.get("edge_proofs") or {})
    minimum_sources = max(
        1,
        int(
            dict(portfolio.get("proof_policy") or {}).get(
                "minimum_independent_source_groups"
            )
            or 1
        ),
    )
    closure_axes = [
        _route_closure_axes(
            value,
            graph=graph,
            proofs=proofs,
            minimum_sources=minimum_sources,
        )
        for value in candidates
    ]
    counts = {
        "route_family_count": len(dict(graph.get("route_families") or {})),
        "candidate_route_count": len(candidates),
        "selected_route_count": len(selected),
        "pareto_route_count": len(pareto),
        "configured_complete_route_count": sum(
            value.get("complete") is True for value in candidates
        ),
        "target_rooted_route_count": int(
            gate_counts.get("target_rooted_distinct_skeletons") or 0
        ),
        "canonical_materialized_route_count": int(
            gate_counts.get("materialized_skeletons") or 0
        ),
        "host_validated_route_count": int(
            gate_counts.get("reaction_validated_skeletons") or 0
        ),
        "strict_host_validated_route_count": sum(value["C2"] for value in closure_axes),
        "exact_evidence_route_count": int(
            gate_counts.get("evidence_closed_skeletons") or 0
        ),
        "exact_procedure_route_count": sum(value["C3"] for value in closure_axes),
        "condition_complete_route_count": sum(value["C4"] for value in closure_axes),
        "stock_closed_route_count": int(gate_counts.get("stock_closed_skeletons") or 0),
        "strict_stock_closed_route_count": sum(value["C5"] for value in closure_axes),
    }
    archive = [
        {
            key: _json_value(value[key])
            for key in (
                "route_id",
                "route_family_id",
                "edge_ids",
                "leaf_molecule_ids",
                "minimum_edge_proof_level",
                "stock_closure_rate",
                "independent_source_groups",
                "length",
                "physical_step_count",
                "chemical_step_equivalent_count",
                "net_step_savings",
                "convergence_score",
                "risk_score",
                "complete",
                "selected",
            )
            if key in value
        }
        for value in sorted(pareto, key=lambda row: str(row.get("route_id") or ""))
    ]
    result = {
        "counts": counts,
        "pareto_archive": archive,
        "portfolio_sha256": str(portfolio.get("content_sha256") or ""),
        "graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
    }
    result["content_sha256"] = _digest(result)
    return result


def _route_closure_axes(
    route: Mapping[str, Any],
    *,
    graph: Mapping[str, Any],
    proofs: Mapping[str, Mapping[str, Any]],
    minimum_sources: int,
) -> dict[str, bool]:
    """Compile strict C1-C5 axes without allowing one axis to imply another."""

    edges = dict(graph.get("edges") or {})
    edge_ids = [str(value) for value in route.get("edge_ids") or [] if str(value)]
    materialized = bool(edge_ids) and all(edge_id in edges for edge_id in edge_ids)
    if not materialized:
        return {key: False for key in ("C1", "C2", "C3", "C4", "C5")}
    vectors = [
        edge_proof_vector(
            edge=dict(edges[edge_id]),
            proof=dict(proofs.get(edge_id) or {}),
            graph=graph,
        )
        for edge_id in edge_ids
    ]
    reaction_validated = all(
        vector.get("reaction") in {"host_validated", "source_reaction_exact"}
        for vector in vectors
    )
    source_groups = {
        str(value)
        for value in route.get("independent_source_groups") or []
        if str(value)
    }
    exact_procedure = bool(
        reaction_validated
        and len(source_groups) >= minimum_sources
        and all(
            vector.get("identity") == "source_exact"
            and int(vector.get("procedure_record_count") or 0) > 0
            for vector in vectors
        )
    )
    complete_conditions = bool(
        exact_procedure
        and all(
            vector.get("conditions") == "source_exact"
            and vector.get("condition_completeness") == "complete"
            and vector.get("process") == "procedure_bound_candidate"
            for vector in vectors
        )
    )
    return {
        "C1": True,
        "C2": reaction_validated,
        "C3": exact_procedure,
        "C4": complete_conditions,
        "C5": route.get("all_leaves_stock_closed") is True,
    }


def compile_action_counts(
    executions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count unique Action executions by kind and terminal status."""

    unique: dict[str, dict[str, Any]] = {}
    for raw in executions:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        action = dict(row.get("action") or {})
        outcome = dict(row.get("outcome") or {})
        execution_id = str(
            action.get("execution_id")
            or outcome.get("action_execution_id")
            or row.get("execution_id")
            or ""
        )
        if not execution_id:
            continue
        unique.setdefault(execution_id, row)
    by_kind: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for row in unique.values():
        action = dict(row.get("action") or {})
        outcome = dict(row.get("outcome") or {})
        kind = str(action.get("kind") or "unknown")
        status = str(outcome.get("status") or row.get("status") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "total": len(unique),
        "by_kind": {key: by_kind[key] for key in sorted(by_kind)},
        "by_status": {key: by_status[key] for key in sorted(by_status)},
    }


def compile_campaign_snapshot(
    *,
    phase: str,
    observed_at: str,
    graph_revision: int,
    gates: Mapping[str, Any],
    resource_usage: Mapping[str, Any],
    event_sequence: int = 0,
    wall_time_s: float = 0.0,
    action_counts: Mapping[str, Any] | None = None,
    route_counts: Mapping[str, Any] | None = None,
    pareto_archive: Iterable[Mapping[str, Any]] = (),
    bindings: Mapping[str, Any] | None = None,
    program_milestones: Mapping[str, Any] | None = None,
    action_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(graph_revision) < 0 or int(event_sequence) < 0:
        raise ValueError("campaign snapshot revisions cannot be negative")
    elapsed = float(wall_time_s)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("campaign snapshot wall time must be finite and non-negative")
    gate_values = {
        str(key): value is True for key, value in dict(gates.get("gates") or {}).items()
    }
    normalized_route_counts = _count_mapping(route_counts or {})
    milestones = {
        **gate_values,
        "route:first_target_rooted": (
            int(normalized_route_counts.get("target_rooted_route_count") or 0) > 0
        ),
        "route:first_host_validated": (
            int(normalized_route_counts.get("host_validated_route_count") or 0) > 0
        ),
        **{
            str(key): value is True
            for key, value in dict(program_milestones or {}).items()
        },
    }
    resolved_bindings = dict(bindings or compile_trajectory_bindings())
    if resolved_bindings.get("schema_version") != TRAJECTORY_BINDINGS_SCHEMA:
        raise ValueError("campaign snapshot bindings schema is invalid")
    if not _digest_valid(resolved_bindings):
        raise ValueError("campaign snapshot bindings digest is invalid")
    row = {
        "schema_version": CAMPAIGN_SNAPSHOT_SCHEMA,
        "phase": str(phase),
        "observed_at": str(observed_at),
        "event_sequence": int(event_sequence),
        "graph_revision": int(graph_revision),
        "wall_time_s": round(elapsed, 6),
        "milestones": milestones,
        "highest_contiguous_gate": str(gates.get("highest_contiguous_gate") or "none"),
        "gate_counts": _count_mapping(dict(gates.get("counts") or {})),
        "resource_usage": _json_value(resource_usage),
        "action_counts": _json_value(action_counts or {}),
        "route_counts": normalized_route_counts,
        "pareto_archive": sorted(
            (_json_value(value) for value in pareto_archive),
            key=lambda value: str(value.get("route_id") or ""),
        ),
        "bindings": resolved_bindings,
        "next_action": {
            key: value
            for key, value in dict(action_decision or {}).items()
            if key
            in {
                "selected_action_id",
                "selected_action",
                "candidate_count",
                "eligible_candidate_count",
                "content_sha256",
            }
        },
        "semantics": {
            "one_trajectory_for_all_result_views": True,
            "wall_time_is_cumulative_run_kernel_time": True,
            "milestones_do_not_select_solver_control_flow": True,
            "snapshot_grants_no_additional_scientific_authority": True,
            "pareto_archive_is_a_historical_projection": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


def compile_campaign_trajectory(
    snapshots: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    by_digest: dict[str, dict[str, Any]] = {}
    for raw in snapshots:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if row.get("schema_version") not in {
            CAMPAIGN_SNAPSHOT_SCHEMA,
            LEGACY_CAMPAIGN_SNAPSHOT_SCHEMA,
        }:
            continue
        if not _digest_valid(row):
            raise ValueError("campaign snapshot digest is invalid")
        if row.get("schema_version") == CAMPAIGN_SNAPSHOT_SCHEMA:
            bindings = dict(row.get("bindings") or {})
            if bindings.get(
                "schema_version"
            ) != TRAJECTORY_BINDINGS_SCHEMA or not _digest_valid(bindings):
                raise ValueError("campaign snapshot bindings are invalid")
        by_digest[str(row["content_sha256"])] = row
    rows = sorted(by_digest.values(), key=_snapshot_sort_key)
    first_achieved: dict[str, dict[str, Any]] = {}
    for row in rows:
        for milestone, achieved in dict(row.get("milestones") or {}).items():
            if achieved is True and milestone not in first_achieved:
                first_achieved[str(milestone)] = _milestone_observation(row)
    v2_rows = [
        row for row in rows if row.get("schema_version") == CAMPAIGN_SNAPSHOT_SCHEMA
    ]
    continuity = _continuity(v2_rows)
    continuity["legacy_snapshot_count"] = len(rows) - len(v2_rows)
    binding_epochs = _binding_epochs(v2_rows)
    result = {
        "schema_version": CAMPAIGN_TRAJECTORY_SCHEMA,
        "snapshot_count": len(rows),
        "snapshots": rows,
        "first_achieved": {key: first_achieved[key] for key in sorted(first_achieved)},
        "time_to_first": _time_to_first(first_achieved),
        "binding_epochs": binding_epochs,
        "continuity": continuity,
        "resource_curve": [
            {
                "snapshot_sha256": str(row.get("content_sha256") or ""),
                "event_sequence": int(row.get("event_sequence") or 0),
                "graph_revision": int(row.get("graph_revision") or 0),
                "wall_time_s": row.get("wall_time_s"),
                "resource_usage": _json_value(row.get("resource_usage") or {}),
                "action_counts": _json_value(row.get("action_counts") or {}),
                "route_counts": _json_value(row.get("route_counts") or {}),
            }
            for row in rows
        ],
        "semantics": {
            "benchmark_metrics_are_fixed_cutoff_projections": True,
            "trajectory_is_independent_of_result_view": True,
            "resume_uses_cumulative_kernel_time": True,
            "historical_milestones_are_not_erased_by_later_state": True,
            "legacy_v1_snapshots_remain_readable": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def project_campaign_trajectory_at_cutoff(
    trajectory: Mapping[str, Any],
    *,
    cutoff: Mapping[str, int | float],
) -> dict[str, Any]:
    """Read one immutable trajectory at a fixed cumulative resource cutoff.

    The projection is deliberately downstream of the solver.  It never
    changes action selection, and it refuses trajectories whose cumulative
    resource coordinates regress across resume boundaries.
    """

    source = dict(trajectory)
    if source.get("schema_version") != CAMPAIGN_TRAJECTORY_SCHEMA:
        raise ValueError("campaign trajectory v2 is required for cutoff projection")
    if not _digest_valid(source):
        raise ValueError("campaign trajectory digest is invalid")
    limits = _normalize_cutoff(cutoff)
    rows = [
        (index, dict(row))
        for index, raw in enumerate(source.get("snapshots") or [])
        if isinstance(raw, Mapping)
        and (row := dict(raw)).get("schema_version") == CAMPAIGN_SNAPSHOT_SCHEMA
    ]
    for _, row in rows:
        if not _digest_valid(row):
            raise ValueError("campaign trajectory contains an invalid snapshot digest")
    observations = [
        (index, row, _snapshot_resource_observation(row)) for index, row in rows
    ]
    regressions = _resource_regressions(observations, limits)
    continuity = dict(source.get("continuity") or {})
    unavailable_reason = ""
    if continuity.get("resume_baseline_preserved") is False or regressions:
        unavailable_reason = "trajectory_resource_continuity_invalid"
    eligible = [
        (index, row, observed)
        for index, row, observed in observations
        if _within_cutoff(observed, limits)
    ]
    selected = eligible[-1] if eligible and not unavailable_reason else None
    prefix = [
        row
        for index, row, _ in observations
        if selected is not None and index <= selected[0]
    ]
    first_achieved = _first_achieved(prefix)
    result: dict[str, Any] = {
        "schema_version": TRAJECTORY_CUTOFF_PROJECTION_SCHEMA,
        "trajectory_sha256": str(source.get("content_sha256") or ""),
        "cutoff": limits,
        "available": selected is not None,
        "unavailable_reason": (
            unavailable_reason
            or ("" if selected is not None else "no_v2_snapshot_within_cutoff")
        ),
        "resource_regressions": regressions,
        "selected_snapshot_index": selected[0] if selected is not None else None,
        "selected_snapshot_sha256": (
            str(selected[1].get("content_sha256") or "") if selected is not None else ""
        ),
        "observed_resources": selected[2] if selected is not None else {},
        "milestones": (
            _json_value(selected[1].get("milestones") or {})
            if selected is not None
            else {}
        ),
        "first_achieved": {key: first_achieved[key] for key in sorted(first_achieved)},
        "time_to_first": _time_to_first(first_achieved),
        "gate_summary": (
            _gate_aliases(selected[1].get("milestones") or {})
            if selected is not None
            else {}
        ),
        "gate_counts": (
            _json_value(selected[1].get("gate_counts") or {})
            if selected is not None
            else {}
        ),
        "route_counts": (
            _json_value(selected[1].get("route_counts") or {})
            if selected is not None
            else {}
        ),
        "pareto_archive": (
            _json_value(selected[1].get("pareto_archive") or [])
            if selected is not None
            else []
        ),
        "resource_usage": (
            _json_value(selected[1].get("resource_usage") or {})
            if selected is not None
            else {}
        ),
        "action_counts": (
            _json_value(selected[1].get("action_counts") or {})
            if selected is not None
            else {}
        ),
        "semantics": {
            "projection_is_read_only": True,
            "projection_does_not_select_solver_actions": True,
            "last_observation_within_all_cutoffs_is_selected": True,
            "later_milestones_are_censored": True,
            "scientific_authority_is_not_upgraded": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def snapshots_from_stages(
    stages: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(dict(row).get("detail") or {})
        for row in stages
        if isinstance(row, Mapping)
        and str(row.get("stage") or "").startswith("campaign_snapshot_")
        and isinstance(row.get("detail"), Mapping)
    ]


def _binding_section(name: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _json_value(value or {})
    payload.pop("content_sha256", None)
    row = {
        "schema_version": f"campaign_trajectory_{name}_binding.v1",
        "observed": bool(value),
        "value": payload,
    }
    row["content_sha256"] = _digest(row)
    return row


def _snapshot_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    if row.get("schema_version") == CAMPAIGN_SNAPSHOT_SCHEMA:
        return (
            1,
            int(row.get("event_sequence") or 0),
            float(row.get("wall_time_s") or 0.0),
            int(row.get("graph_revision") or 0),
            str(row.get("phase") or ""),
            str(row.get("content_sha256") or ""),
        )
    return (
        0,
        int(row.get("graph_revision") or 0),
        str(row.get("observed_at") or ""),
        str(row.get("phase") or ""),
        str(row.get("content_sha256") or ""),
    )


def _milestone_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase": str(row.get("phase") or ""),
        "observed_at": str(row.get("observed_at") or ""),
        "event_sequence": int(row.get("event_sequence") or 0),
        "graph_revision": int(row.get("graph_revision") or 0),
        "elapsed_wall_time_s": row.get("wall_time_s"),
        "snapshot_sha256": str(row.get("content_sha256") or ""),
    }


def _first_achieved(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        for milestone, achieved in dict(row.get("milestones") or {}).items():
            if achieved is True and str(milestone) not in result:
                result[str(milestone)] = _milestone_observation(row)
    return result


def _normalize_cutoff(value: Mapping[str, int | float]) -> dict[str, int | float]:
    raw = dict(value or {})
    unknown = sorted(set(raw) - _CUTOFF_FIELDS)
    if unknown:
        raise ValueError(f"unsupported trajectory cutoff fields: {','.join(unknown)}")
    if not raw:
        raise ValueError("at least one trajectory cutoff is required")
    normalized: dict[str, int | float] = {}
    for key in sorted(raw):
        item = raw[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"trajectory cutoff {key} must be numeric")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"trajectory cutoff {key} must be finite and non-negative")
        normalized[key] = (
            round(number, 6) if key in _FLOAT_CUTOFF_FIELDS else int(number)
        )
    return normalized


def _snapshot_resource_observation(
    row: Mapping[str, Any],
) -> dict[str, int | float | None]:
    usage = dict(row.get("resource_usage") or {})
    model = dict(usage.get("model") or {})
    native = dict(usage.get("native_search") or {})
    tasks = dict(usage.get("tasks") or {})
    total_tasks = dict(dict(tasks.get("dimensions") or {}).get("total") or {})
    return {
        "wall_time_s": _number(row.get("wall_time_s"), floating=True),
        "event_sequence": _number(row.get("event_sequence")),
        "attempt_count": _number(usage.get("attempt_count")),
        "accepted_expansion_count": _number(usage.get("accepted_expansion_count")),
        "settled_task_count": _number(
            usage.get(
                "settled_task_count", total_tasks.get("settled", tasks.get("settled"))
            )
        ),
        "model_invocations": _number(model.get("model_invocations")),
        "visual_invocations": _number(model.get("visual_invocations")),
        "input_tokens": _number(model.get("input_tokens")),
        "output_tokens": _number(model.get("output_tokens")),
        "model_wall_time_s": _number(model.get("wall_time_s"), floating=True),
        "native_search_invocations": _number(native.get("committed_total")),
    }


def _number(value: Any, *, floating: bool = False) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return round(number, 6) if floating else int(number)


def _resource_regressions(
    observations: Iterable[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    limits: Mapping[str, int | float],
) -> list[str]:
    regressions: set[str] = set()
    previous: dict[str, int | float] = {}
    for _, _, observed in observations:
        for key in limits:
            current = observed.get(key)
            if current is None:
                continue
            if key in previous and current < previous[key]:
                regressions.add(key)
            previous[key] = current
    return sorted(regressions)


def _within_cutoff(
    observed: Mapping[str, Any],
    limits: Mapping[str, int | float],
) -> bool:
    return all(
        observed.get(key) is not None and observed[key] <= maximum
        for key, maximum in limits.items()
    )


def _gate_aliases(milestones: Mapping[str, Any]) -> dict[str, bool]:
    return {
        key.split("_", 1)[0]: value is True
        for key, value in dict(milestones).items()
        if key[:2] in {"B0", "B1", "B2", "B3", "B4", "B5"}
    }


def _time_to_first(first: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    aliases = {
        "first_route": "route:first_target_rooted",
        "B1": "B1_global_multi_route",
        "first_host_valid_route": "route:first_host_validated",
        "B2": "B2_host_validated_routes",
        "B3": "B3_exact_multi_source",
        "B4": "B4_stock_boundary",
        "B5": "B5_configured_portfolio_acceptance",
    }
    result = {
        alias: _json_value(first.get(milestone)) if milestone in first else None
        for alias, milestone in aliases.items()
    }
    result["program"] = {
        key: _json_value(first[key])
        for key in sorted(first)
        if key.startswith("program:")
    }
    return result


def _continuity(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    event_sequences = [int(row.get("event_sequence") or 0) for row in rows]
    wall_times = [float(row.get("wall_time_s") or 0.0) for row in rows]
    event_monotonic = all(
        right >= left for left, right in zip(event_sequences, event_sequences[1:])
    )
    wall_monotonic = all(
        right >= left for left, right in zip(wall_times, wall_times[1:])
    )
    return {
        "event_sequence_monotonic": event_monotonic,
        "wall_time_monotonic": wall_monotonic,
        "resume_baseline_preserved": event_monotonic and wall_monotonic,
    }


def _binding_epochs(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    epochs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        bindings = dict(row.get("bindings") or {})
        sha256 = str(bindings.get("content_sha256") or "")
        if epochs and epochs[-1]["bindings_sha256"] == sha256:
            epochs[-1]["last_snapshot_index"] = index
            epochs[-1]["last_event_sequence"] = int(row.get("event_sequence") or 0)
            continue
        epochs.append(
            {
                "bindings_sha256": sha256,
                "complete": bindings.get("complete") is True,
                "first_snapshot_index": index,
                "last_snapshot_index": index,
                "first_event_sequence": int(row.get("event_sequence") or 0),
                "last_event_sequence": int(row.get("event_sequence") or 0),
            }
        )
    return epochs


def _count_mapping(value: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(key): max(0, int(item or 0)) for key, item in sorted(dict(value).items())
    }


def _digest_valid(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    return len(supplied) == 64 and supplied == _digest(row)


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CAMPAIGN_SNAPSHOT_SCHEMA",
    "CAMPAIGN_TRAJECTORY_SCHEMA",
    "TRAJECTORY_BINDINGS_SCHEMA",
    "TRAJECTORY_CUTOFF_PROJECTION_SCHEMA",
    "compile_action_counts",
    "compile_campaign_snapshot",
    "compile_campaign_trajectory",
    "compile_route_snapshot",
    "compile_trajectory_bindings",
    "project_campaign_trajectory_at_cutoff",
    "snapshots_from_stages",
]

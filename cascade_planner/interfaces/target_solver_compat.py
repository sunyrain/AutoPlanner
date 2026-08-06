"""Compatibility-only projections for the unified V4 target solver.

Legacy objective labels, completed-checkpoint cursors, and external feedback
envelopes are translated here.  This module owns no scheduler or execution
control flow; every accepted work item still enters the canonical frontier and
the single ``CampaignActionRuntime.run_anytime()`` loop.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Literal, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
)


TARGET_SOLVE_CHECKPOINT_SCHEMA = "target_only_solve_checkpoint.v1"
TARGET_RESUME_CURSOR_SCHEMA = "target_solver_resume_cursor.v1"
TARGET_RESUME_WORK_SCHEMA = "target_solver_resume_work.v1"
TargetObjectiveMode = Literal[
    "benchmark_search",
    "scientific_proof",
    "procurement_delivery",
]
_OBJECTIVE_MODES = frozenset(
    {"benchmark_search", "scientific_proof", "procurement_delivery"}
)


def validate_target_objective_mode(value: str) -> TargetObjectiveMode:
    mode = str(value or "")
    if mode not in _OBJECTIVE_MODES:
        raise ValueError("target solver objective mode is invalid")
    return mode  # type: ignore[return-value]


def compile_target_claim_projection(
    gates: Mapping[str, Any],
    acceptance: RetrosynthesisAcceptanceSpec,
    resource_envelope: Mapping[str, Any],
    *,
    objective_mode: TargetObjectiveMode = "scientific_proof",
    workbench: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project legacy objective labels without changing solver decisions."""

    mode = validate_target_objective_mode(objective_mode)
    values = dict(gates.get("gates") or {})
    scientifically_accepted = bool(
        values.get("B5_configured_portfolio_acceptance") is True
        and resource_envelope.get("within_budget") is True
    )
    stock_milestone_achieved = bool(
        values.get("B4_stock_boundary") is True
        and resource_envelope.get("within_budget") is True
    )
    acceptance_profile = {
        "benchmark_search": "exploration_closed",
        "procurement": "procurement_closed",
        "in_house": "in_house_closed",
    }.get(acceptance.stock_boundary, "configured_boundary_closed")
    workbench_portfolio = dict(dict(workbench or {}).get("portfolio") or {})
    profile_counts = {
        str(key): int(value or 0)
        for key, value in dict(
            workbench_portfolio.get("acceptance_profile_counts") or {}
        ).items()
    }
    achieved_profile = str(
        workbench_portfolio.get("achieved_profile") or "unresolved"
    )
    milestones = {
        str(name): values.get(name) is True
        for name in (
            "B0_blind_input",
            "B1_global_multi_route",
            "B2_host_validated_routes",
            "B3_exact_multi_source",
            "B4_stock_boundary",
            "B5_configured_portfolio_acceptance",
        )
    }
    return {
        "generated_route_portfolio": values.get("B1_global_multi_route") is True,
        "host_validated_route_portfolio": (
            values.get("B2_host_validated_routes") is True
        ),
        "exact_multi_source_grade": values.get("B3_exact_multi_source") is True,
        "configured_stock_boundary_closed": values.get("B4_stock_boundary") is True,
        "accepted_under_configured_policy": scientifically_accepted,
        "scientific_proof_accepted": scientifically_accepted,
        "milestones": milestones,
        "objective_mode": mode,
        "objective_gate": "B5_configured_portfolio_acceptance",
        "objective_achieved": scientifically_accepted,
        "benchmark_search_completed": stock_milestone_achieved,
        "acceptance_profile": acceptance_profile,
        "achieved_profile": achieved_profile,
        "product_profile_counts": profile_counts,
        "literature_grounded": profile_counts.get("literature_grounded", 0) > 0,
        "procurement_ready": profile_counts.get("procurement_closed", 0) > 0,
        "within_resource_budget": resource_envelope.get("within_budget") is True,
        "condition_complete": profile_counts.get("condition_complete", 0) > 0,
        "process_ready": profile_counts.get("process_ready", 0) > 0,
        "no_unqualified_solved_claim": True,
        "no_unqualified_complete_claim": True,
        "semantics": {
            "objective_mode_is_compatibility_metadata_only": True,
            "B4_is_an_anytime_milestone_not_a_solver_terminal": True,
            "one_configured_acceptance_rule_for_all_targets": True,
        },
    }


def compile_program_validation_feedback_signals(
    feedback: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Translate external validation envelopes into canonical action signals."""

    signals: list[dict[str, Any]] = []
    for raw_envelope in feedback:
        envelope = dict(raw_envelope)
        route_id = str(envelope.get("route_id") or "")
        validation = dict(envelope.get("validation") or {})
        if not route_id or not validation:
            raise ValueError(
                "program validation feedback requires route_id and validation"
            )
        payload = {"route_id": route_id, "validation": validation}
        payload_sha256 = _digest(payload)
        program_id = str(validation.get("program_id") or payload_sha256)
        signals.append(
            {
                "signal_id": f"event-deficit:experiment-feedback:{payload_sha256}",
                "kind": "experiment_feedback",
                "object_id": str(
                    validation.get("validation_id") or payload_sha256
                ),
                "entity_ids": [program_id],
                "route_family_ids": [route_id],
                "dependency_ids": [],
                "deterministic": True,
                "model_allowed": False,
                "reason": "external_program_validation_feedback_available",
                "score": {
                    "expected_portfolio_gain": 0.05,
                    "distance_to_closure": 0.05,
                    "evidence_gain": 0.75,
                    "route_diversity_gain": 0.05,
                    "cost_penalty": 0.02,
                    "failure_risk_penalty": 0.05,
                },
                "metadata": {"experiment_feedback": True, **payload},
            }
        )
    return tuple(sorted(signals, key=lambda row: str(row["signal_id"])))


def build_target_resume_cursor(graph: Mapping[str, Any]) -> dict[str, Any]:
    signals = {
        str(signal_id): str(dict(signal).get("content_sha256") or _digest(signal))
        for signal_id, signal in sorted(
            dict(graph.get("action_signals") or {}).items()
        )
        if isinstance(signal, Mapping)
    }
    result = {
        "schema_version": TARGET_RESUME_CURSOR_SCHEMA,
        "graph_revision": int(graph.get("revision") or 0),
        "action_signals": signals,
    }
    result["content_sha256"] = _digest(result)
    return result


def classify_target_resume_work(
    checkpoint: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    feedback_signals: Iterable[Mapping[str, Any]] = (),
    available_signal_kinds: Iterable[str] = (),
) -> dict[str, Any]:
    """Detect only work newer than the completed checkpoint cursor."""

    available = {str(value) for value in available_signal_kinds if str(value)}
    baseline = dict(dict(checkpoint.get("resume_cursor") or {}).get("action_signals") or {})
    current_signals = {
        str(signal_id): dict(signal)
        for signal_id, signal in dict(graph.get("action_signals") or {}).items()
        if isinstance(signal, Mapping)
    }
    pending = []
    if baseline:
        for signal_id, signal in sorted(current_signals.items()):
            kind = str(signal.get("kind") or "")
            digest = str(signal.get("content_sha256") or _digest(signal))
            if (
                str(signal.get("status") or "open") == "open"
                and kind in available
                and baseline.get(signal_id) != digest
            ):
                pending.append(
                    {"signal_id": signal_id, "kind": kind, "content_sha256": digest}
                )
    external = []
    for raw_signal in feedback_signals:
        signal = dict(raw_signal)
        signal_id = str(signal.get("signal_id") or "")
        existing = current_signals.get(signal_id)
        if not signal_id or (
            existing is not None
            and str(existing.get("status") or "open") == "resolved"
        ):
            continue
        digest = str(signal.get("content_sha256") or _digest(signal))
        if existing is None or baseline.get(signal_id) != str(
            existing.get("content_sha256") or _digest(existing)
        ):
            external.append(
                {
                    "signal_id": signal_id,
                    "kind": str(signal.get("kind") or "experiment_feedback"),
                    "content_sha256": digest,
                }
            )
    work_items = sorted(
        [*pending, *external],
        key=lambda row: (str(row["kind"]), str(row["signal_id"])),
    )
    result = {
        "schema_version": TARGET_RESUME_WORK_SCHEMA,
        "has_new_work": bool(work_items),
        "pending_canonical_signals": pending,
        "external_feedback_signals": external,
        "work_items": work_items,
        "reasons": sorted(
            {
                *("new_pending_canonical_action_signal" for _ in pending[:1]),
                *("new_program_validation_feedback" for _ in external[:1]),
            }
        ),
    }
    result["work_fingerprint"] = _digest(work_items)
    return result


def compile_target_solver_checkpoint(
    run_id: str,
    stages: Iterable[Mapping[str, Any]],
    outcomes: Iterable[Mapping[str, Any]],
    *,
    complete: bool,
    resume_cursor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": TARGET_SOLVE_CHECKPOINT_SCHEMA,
        "run_id": str(run_id),
        "complete": bool(complete),
        "stages": [dict(value) for value in stages],
        "director_outcomes": [dict(value) for value in outcomes],
    }
    if resume_cursor:
        result["resume_cursor"] = dict(resume_cursor)
    return result


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "TARGET_RESUME_CURSOR_SCHEMA",
    "TARGET_RESUME_WORK_SCHEMA",
    "TARGET_SOLVE_CHECKPOINT_SCHEMA",
    "TargetObjectiveMode",
    "build_target_resume_cursor",
    "classify_target_resume_work",
    "compile_program_validation_feedback_signals",
    "compile_target_claim_projection",
    "compile_target_solver_checkpoint",
    "validate_target_objective_mode",
]

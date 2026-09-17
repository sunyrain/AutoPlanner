"""Bounded report/checkpoint views of durable campaign action receipts.

This module owns presentation only; canonical artifacts retain full execution
history and scientific authority. It neither schedules nor persists work.
"""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.campaign_actions import CampaignActionKind


def _bounded_detail(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return "<depth-limited>"
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if name == "route_lineage" and isinstance(child, list | tuple):
                # Provider lineage is a compact identity record consumed by
                # final canonical reconciliation.  Reset the display nesting
                # depth so outer action envelopes cannot replace proposal IDs
                # with the generic depth sentinel.
                out[name] = _bounded_detail(child)
                continue
            if name in {"graph", "portfolio", "snapshot"} and isinstance(child, Mapping):
                out[f"{name}_summary"] = {
                    "revision": child.get("revision") or child.get("graph_revision"),
                    "molecule_count": len(child.get("molecules") or {}),
                    "edge_count": len(child.get("edges") or {}),
                    "accepted": child.get("accepted"),
                    "content_sha256": child.get("content_sha256"),
                }
                continue
            out[name] = _bounded_detail(child, depth=depth + 1)
        return out
    if isinstance(value, list | tuple):
        rows = [_bounded_detail(child, depth=depth + 1) for child in value[:64]]
        if len(value) > 64:
            rows.append({"omitted_count": len(value) - 64})
        return rows
    if isinstance(value, str) and len(value) > 2_000:
        return value[:2_000] + f"... <{len(value) - 2_000} chars omitted>"
    return value


_COMPACT_ACTION_HANDLER_FIELDS: dict[str, frozenset[str]] = {
    CampaignActionKind.CHEMENZY_TARGET_EXPAND.value: frozenset(
        {
            "status",
            "mode",
            "scope",
            "request",
            "provider_invocation_count",
            "proposal_count",
            "route_lineage",
            "provider_envelope",
            "provider_registration",
            "request_sha256",
            "raw_proposal_sha256",
            "raw_result_sha256",
            "replay_key_sha256",
            "random_seed",
            "provider_invocation_binding",
            "runtime_preflight",
            "failure_reasons",
            "reasons",
        }
    ),
    CampaignActionKind.EXPERIMENT_FEEDBACK_INGEST.value: frozenset(
        {
            "status",
            "accepted",
            "validation_id",
            "reasons",
            "resolved_program_validation_signal_ids",
            "experimental_claims",
            "experimental_claims_oracle",
        }
    ),
    CampaignActionKind.PROGRAM_VALIDATE.value: frozenset(
        {"status", "accepted", "validated_count", "reasons"}
    ),
    CampaignActionKind.PROGRAM_ADMIT.value: frozenset(
        {"status", "accepted", "admission", "reasons"}
    ),
}


def _compact_campaign_action_stage(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep checkpoint/reports small while the CAS retains the full receipt."""

    current = dict(row)
    if not str(current.get("stage") or "").startswith("campaign_action_unified_core_"):
        return current
    detail = dict(current.get("detail") or {})
    action = dict(detail.get("action") or {})
    outcome = dict(detail.get("outcome") or {})
    kind = str(action.get("kind") or "")
    projected_action = {
        key: action[key]
        for key in (
            "action_id",
            "execution_id",
            "kind",
            "producer",
            "resource_class",
            "input_revision",
        )
        if key in action
    }
    action_metadata = dict(action.get("metadata") or {})
    projected_metadata = {
        key: action_metadata[key]
        for key in (
            "replan_pressure",
            "program_opportunity_pressure",
            "program_review_pressure",
        )
        if key in action_metadata
    }
    if projected_metadata:
        projected_action["metadata"] = projected_metadata
    projected_outcome = {
        key: outcome[key]
        for key in (
            "status",
            "output_revision",
            "elapsed_s",
            "material_events",
            "failure_type",
            "failure_reasons",
            "immutable_artifact_refs",
        )
        if key in outcome
    }
    retained_handler_fields = _COMPACT_ACTION_HANDLER_FIELDS.get(kind, frozenset())
    handler_result = dict(outcome.get("handler_result") or {})
    projected_handler = {
        key: handler_result[key] for key in retained_handler_fields if key in handler_result
    }
    if projected_handler:
        projected_outcome["handler_result"] = projected_handler
    if kind == CampaignActionKind.CHEMENZY_TARGET_EXPAND.value:
        nested_lineage = []
        for value in handler_result.get("results") or ():
            if not isinstance(value, Mapping) or not value.get("route_lineage"):
                continue
            nested_lineage.append(
                {
                    key: value[key]
                    for key in (
                        "mode",
                        "scope",
                        "request_sha256",
                        "raw_proposal_sha256",
                        "raw_result_sha256",
                        "replay_key_sha256",
                        "random_seed",
                        "route_lineage",
                    )
                    if key in value
                }
            )
        if nested_lineage:
            projected_outcome.setdefault("handler_result", {})["results"] = nested_lineage
    projected_detail = {
        "schema_version": "campaign_action_checkpoint_projection.v1",
        "status": str(detail.get("status") or outcome.get("status") or ""),
        "action": projected_action,
        "outcome": projected_outcome,
        "semantics": {
            "full_receipt_is_content_addressed": bool(detail.get("outcome_ref")),
            "projection_grants_no_scientific_authority": True,
        },
    }
    decision = dict(detail.get("decision") or {})
    projected_decision = {
        key: decision[key] for key in ("scientific_closure_pressure",) if key in decision
    }
    selected_action = dict(decision.get("selected_action") or {})
    if selected_action:
        projected_decision["selected_action"] = {
            key: selected_action[key]
            for key in ("kind", "schedule_components")
            if key in selected_action
        }
    if projected_decision:
        projected_detail["decision"] = projected_decision
    for key in (
        "outcome_ref",
        "cache_hit",
        "recovered_from_action_history",
        "outcome_pointer_recovered",
    ):
        if key in detail:
            projected_detail[key] = detail[key]
    current["detail"] = projected_detail
    return current

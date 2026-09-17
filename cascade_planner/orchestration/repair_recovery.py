"""Repair action semantics; the existing search tree and Editor own execution.

These are requests about provisional work, never chemical verdicts or terminal
success signals. Deterministic validation protects the retained route boundary.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


RECOVERY_GUIDANCE = (
    "Choose recovery.action=expand for a chemically justified next move or corrected current-node "
    "retry. A same-connectivity stereoisomer is not an exact reconnection; it may be a temporary "
    "intermediate only if an explicit subsequent reaction connects it. If an earlier provisional "
    "choice caused the dead end, use backtrack with its step_id from reversible_step_ids. If the "
    "repair requires changing retained preparations or retiring a now-unused supply branch, "
    "use expand_scope and explain the required "
    "boundary change; the Editor will choose original-route steps. For backtrack/expand_scope, "
    "emit no reaction_operations or conditions. Do not request backtracking merely for a different "
    "CIP letter, missing evidence, or one invalid edit that can be corrected here. These requests "
    "consume the ordinary budget and do not reject chemistry or claim completion."
)


def recovery_request(
    value: Any, *, reversible_step_ids: Sequence[str],
) -> tuple[dict[str, str] | None, str]:
    if value is None:
        return None, ""  # Historical one-step records imply expand.
    if not isinstance(value, Mapping):
        return None, "repair_recovery_not_object"
    action = str(value.get("action") or "")
    step_id = str(value.get("step_id") or "")
    reason = str(value.get("reason") or "").strip()
    if action not in {"expand", "backtrack", "expand_scope"}:
        return None, "repair_recovery_action_invalid"
    if action == "expand":
        # The graph edit selects its own product. An incidental reference ID
        # cannot mutate another step and must not discard an otherwise valid
        # reaction; only a backtrack request gives this field control meaning.
        return None, ""
    if not reason:
        return None, "repair_recovery_reason_missing"
    if action == "backtrack" and step_id not in reversible_step_ids:
        return None, "repair_recovery_step_outside_provisional_path"
    return {"action": action, "step_id": step_id, "reason": reason}, ""


def previous_repair_feedback(
    transactions: Sequence[Mapping[str, Any]],
    *, critique_history: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Read the transaction itself instead of maintaining another failure ledger."""
    if not transactions:
        return {}
    last = transactions[-1]
    feedback = {
        key: last[key] for key in (
            "status", "reason", "requested_change_step_ids", "repair_goal",
            "recovery_request", "replay_failures", "provisional_open_leaf_states",
        ) if last.get(key)
    }
    review_id = dict(last.get("recritic") or {}).get("critic_task_id")
    if review_id:
        review = next((row.get("critic") or {} for row in reversed(critique_history)
                       if dict(row.get("critic") or {}).get("critic_task_id") == review_id), {})
        # The review history owns the actual findings. Project chemical goals,
        # not vanished candidate IDs that Editor could mistake for live steps.
        findings = [
            {key: row[key] for key in ("blocking_type", "reasons", "suggested_revision") if row.get(key)}
            for row in review.get("step_assessments") or []
            if row.get("verdict") == "reject"
        ]
        if findings:
            feedback["rejected_replacement_findings"] = findings
    return feedback

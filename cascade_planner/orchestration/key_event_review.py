"""Pure Key Critic interpretation and selected-path checkpoint state.

The append-only Critic history owns chemical judgments; worker failures are
attempts, not judgments. Pure selectors identify review work; the Director
owns dispatch and budget admission.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.route_review_context import _strategy_card_digest


def _bind_key_event_focus_assessment(
    critique: Mapping[str, Any], focus_step_id: str
) -> dict[str, Any]:
    """Bind the one key-event assessment to the Host-owned focus identity.

    The compact provider wire does not author opaque route step identifiers;
    the Host already owns that identity when it dispatches the single-focus
    audit.  Bind only one otherwise-unidentified assessment.  A conflicting
    identity or multiple assessments is an invalid/ambiguous response and is
    deliberately left unchanged for the normal unavailable path.
    """

    bound = dict(critique)
    raw_assessments = list(critique.get("step_assessments") or [])
    assessments = [
        dict(value) if isinstance(value, Mapping) else value for value in raw_assessments
    ]
    bound["step_assessments"] = assessments
    focus_id = str(focus_step_id or "")
    if not focus_id:
        return bound
    rows = [value for value in assessments if isinstance(value, Mapping)]
    if any(str(row.get("step_id") or "") == focus_id for row in rows):
        return bound
    if len(rows) != 1 or len(assessments) != 1:
        return bound
    if str(rows[0].get("step_id") or ""):
        return bound
    rows[0]["step_id"] = focus_id
    return bound


def _key_event_focus_assessment(
    critique: Mapping[str, Any], focus_step_id: str
) -> dict[str, Any] | None:
    assessments = list(critique.get("step_assessments") or [])
    if critique.get("status") == "unavailable" or len(assessments) != 1:
        return None
    value = assessments[0]
    if (
        not isinstance(value, Mapping)
        or not focus_step_id
        or str(value.get("step_id") or "") != str(focus_step_id)
        or value.get("verdict") not in {"pass", "uncertain", "reject"}
    ):
        return None
    return dict(value)


def key_event_review_update(
    critique: Mapping[str, Any], *, focus_step_id: str, task_id: str, worker_status: str
) -> dict[str, Any]:
    """Interpret initial and follow-up Critic attempts with the same semantics."""
    bound = _bind_key_event_focus_assessment(critique, focus_step_id)
    assessment = _key_event_focus_assessment(bound, focus_step_id)
    if assessment is None or not isinstance(bound.get("checkpoint_match"), bool):
        return {
            "task_id": task_id,
            "worker_status": worker_status,
            "status": "review_unavailable",
            "critic_status": "unavailable",
            "checkpoint_match": None,
            "assessment": {},
            "reason": str(bound.get("reason") or "critic_focus_assessment_invalid"),
        }
    verdict = assessment["verdict"]
    match = bound["checkpoint_match"]
    return {
        "task_id": task_id,
        "worker_status": worker_status,
        "status": (
            "rejected" if verdict == "reject" or assessment.get("blocking") is True
            else "not_checkpoint" if not match
            else "completed" if verdict == "pass" else "uncertain"
        ),
        "critic_status": str(bound.get("status") or verdict),
        "checkpoint_match": match,
        "assessment": assessment,
    }


def key_event_history_has_assessment(row: Mapping[str, Any]) -> bool:
    """Read chemical authority, including histories predating attempt statuses.

    Legacy failures have status=not_checkpoint, critic_status=unavailable and
    an empty assessment. Display status alone cannot establish a judgment.
    """
    assessment = row.get("assessment")
    return (
        row.get("critic_status") != "unavailable"
        and isinstance(row.get("checkpoint_match"), bool)
        and isinstance(assessment, Mapping)
        and assessment.get("verdict") in {"pass", "uncertain", "reject"}
    )


def pending_key_event_runtime_retry(
    branch: Mapping[str, Any], *, steps: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """One recovery for a failed dispatched review on a still-selected scope.

    Missing output is not a chemical verdict. Do not retry genuine judgments,
    removed evidence, provider-wide pauses, or a failed recovery itself.
    """
    selected = {str(step.get("step_id") or "") for step in steps}
    history = [dict(row) for row in branch.get("key_event_critic_history") or ()
               if isinstance(row, Mapping)]
    for index, row in enumerate(history):
        task_id = str(row.get("task_id") or "")
        if (not task_id or row.get("runtime_retry_of_task_id")
                or row.get("worker_status") == "provider_error"
                or key_event_history_has_assessment(row)):
            continue
        focus = str(row.get("focus_step_id") or "")
        required = set(row.get("required_selected_step_ids") or [focus]) | {focus}
        if not focus or focus not in selected or not required.issubset(selected):
            continue
        obligation = str(row.get("review_of_obligation_id") or _key_event_obligation_id(row))
        if any(later.get("runtime_retry_of_task_id") == task_id and later.get("task_id")
               for later in history[index + 1:]):
            continue
        resolved = False
        for later in history[index + 1:]:
            if not key_event_history_has_assessment(later):
                continue
            if str(later.get("review_of_obligation_id") or _key_event_obligation_id(later)) != obligation:
                continue
            covered = set(later.get("required_selected_step_ids") or [later.get("focus_step_id")])
            if later.get("assessment", {}).get("verdict") == "reject" or (
                covered.issubset(selected) and required.issubset(covered)
            ):
                resolved = True
                break
        if not resolved:
            return row
    return {}


def _key_event_obligation_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("obligation_id") or "").strip()
    if explicit:
        return explicit
    payload = {
        "strategy_digest": str(row.get("strategy_digest") or ""),
        "strategy_milestone_index": int(row.get("strategy_milestone_index") or 1),
        "focus_step_id": str(row.get("focus_step_id") or ""),
        "lineage_root_mapped_smiles": str(row.get("lineage_root_mapped_smiles") or ""),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]


def _strategy_milestone_index(branch: Mapping[str, Any], strategy_card: Mapping[str, Any]) -> int:
    digest = _strategy_card_digest(strategy_card)
    cards = [
        dict(row)
        for row in branch.get("strategy_milestone_cards") or []
        if isinstance(row, Mapping)
    ]
    for index, card in enumerate(cards, start=1):
        if digest and _strategy_card_digest(card) == digest:
            return index
    return 1


def _selected_path_strategy_checkpoint_state(
    branch: Mapping[str, Any],
    *,
    strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive execution and confidence from append-only selected-path audits.

    ``status`` is a display projection and is deliberately ignored.  A later
    valid applicable follow-up for the same obligation supersedes its earlier row.
    Failed attempts retain the last applicable judgment and remain visibly
    pending for their own evidence scope; they never review new evidence by proxy.
    Non-reject evidence is path-local and cannot survive removal of its focus
    step; a reject remains authoritative while that focus is still selected.
    """

    digest = _strategy_card_digest(strategy_card)
    milestone_index = _strategy_milestone_index(branch, strategy_card)
    selected_step_ids = {
        str(row.get("step_id") or "")
        for row in steps
        if isinstance(row, Mapping) and str(row.get("step_id") or "")
    }
    latest_by_obligation: dict[str, tuple[int, dict[str, Any]]] = {}
    pending: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}
    for position, raw in enumerate(branch.get("key_event_critic_history") or []):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        row_digest = str(row.get("strategy_digest") or "")
        if row_digest:
            if not digest or row_digest != digest:
                continue
        elif int(row.get("strategy_milestone_index") or 1) != milestone_index:
            continue
        focus_step_id = str(row.get("focus_step_id") or "")
        required_ids = {
            str(value)
            for value in row.get("required_selected_step_ids") or (focus_step_id,)
            if str(value)
        }
        verdict = str(dict(row.get("assessment") or {}).get("verdict") or "")
        if not focus_step_id or focus_step_id not in selected_step_ids:
            continue
        # Pass/uncertain confidence is evidence-path local and disappears when
        # any required evidence edge is pruned.  A rejected follow-up remains
        # the latest chemical authority for the still-selected focus edge even
        # after the rejected evidence edge itself has been removed; otherwise
        # the older uncertain row would incorrectly become current again.
        if verdict != "reject" and not required_ids.issubset(selected_step_ids):
            continue
        obligation_id = str(
            row.get("review_of_obligation_id")
            or row.get("obligation_id")
            or _key_event_obligation_id(row)
        )
        if not key_event_history_has_assessment(row):
            pending[(obligation_id, frozenset(required_ids))] = row
            continue
        # A successful review covers only the evidence it actually inspected.
        # A reject settles the focus obligation even if its evidence was pruned.
        for key in list(pending):
            if key[0] == obligation_id and (verdict == "reject" or key[1].issubset(required_ids)):
                del pending[key]
        latest_by_obligation[obligation_id] = (position, row)
    pending_rows = list(pending.values())
    applicable = [
        (position, row)
        for position, row in latest_by_obligation.values()
        if row.get("checkpoint_match") is True
    ]
    if not applicable:
        return {
            "checkpoint_executed": False,
            "checkpoint_critic_passed": False,
            "chemical_confidence": "",
            "source_row": {},
            "review_pending": bool(pending_rows),
            "pending_review_rows": pending_rows,
        }
    _, source_row = max(applicable, key=lambda value: value[0])
    verdict = str(dict(source_row.get("assessment") or {}).get("verdict") or "")
    executed = verdict in {"pass", "uncertain"}
    return {
        "checkpoint_executed": executed,
        "checkpoint_critic_passed": verdict == "pass",
        "chemical_confidence": verdict,
        "source_row": source_row,
        "review_pending": bool(pending_rows),
        "pending_review_rows": pending_rows,
    }

"""Append-only recovery journal for agentic blackboard controller state.

``agent_blackboard.json`` is a mutable UI/closeout projection.  It is never
read by this module.  Cross-process recovery comes only from the digest-chained
events below, and the reducer restores a deliberately bounded subset of
in-progress facts.  Final verdicts, parent-route proof, and closeout artifacts
are outside this recovery authority.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TypeVar

from cascade_planner.legacy.harness_runtime.schemas import write_json


BLACKBOARD_EVENT_SCHEMA = "agent_blackboard_event.v1"
BLACKBOARD_CHECKPOINT_SCHEMA = "agent_blackboard_recovery_checkpoint.v1"
BLACKBOARD_JOURNAL_SUMMARY_SCHEMA = "agent_blackboard_event_journal_summary.v1"
BLACKBOARD_REHYDRATION_SCHEMA = "agent_blackboard_rehydration_report.v1"
BLACKBOARD_ACTION_STARTED_SCHEMA = "agent_blackboard_action_started.v1"
BLACKBOARD_ACTION_PREPARED_SCHEMA = "agent_blackboard_action_result_prepared.v1"
BLACKBOARD_ACTION_COMMITTED_SCHEMA = "agent_blackboard_action_committed.v1"
BLACKBOARD_TOMBSTONE_SCHEMA = "agent_blackboard_recovery_tombstone.v1"
BLACKBOARD_TORN_TAIL_QUARANTINE_SCHEMA = (
    "agent_blackboard_torn_tail_quarantine.v1"
)
BLACKBOARD_CHECKPOINT_REF_SCHEMA = "agent_blackboard_checkpoint_ref.v1"
_CHECKPOINT_INLINE_MAX_BYTES = 128 * 1024

_F = TypeVar("_F", bound=Callable[..., Any])

_RECOVERABLE_FIELDS = (
    "route_failures",
    "plugin_runtime_diagnostics",
    "literature_evidence",
    "analogical_hypotheses",
    "analogical_hypothesis_ranking",
    "analogical_templates",
    "analogical_template_ranking",
    "template_applications",
    "template_cache_refs",
    "template_failure_memory",
    "route_objective_summary",
    "endpoint_candidates",
    "objective_evidence_cards",
    "broad_transform_templates",
    "reaction_idea_cards",
    "semisynthesis_anchors",
    "recursive_hypothesis_tasks",
    "route_expansion_subgoals",
    "bridge_tasks",
    "terminal_blacklist",
    "planner_history",
    "action_history",
    "controller_action_batches",
    "controller_action_batch_validations",
    "budget_state",
    "current_belief",
    "target_side_disconnection_hypotheses",
    "chemenzy_route_proof_banks",
    "blackboard_migrations",
    "safety_flags",
    "artifact_refs",
    "retrosynthesis_run_contract",
    "route_deficit_queue",
    "retrosynthesis_acceptance",
    # Narrow, revision-only campaign authority survives controller restart.
    # Full team/graph/ledger payloads remain rebuildable from immutable
    # campaign commits and the live queue and are intentionally not journaled.
    "campaign_projection_binding",
    "codex_campaign_authority_projection",
)

_BLOCKED_ARTIFACT_REF_TOKENS = (
    "final_verdict",
    "closeout",
    "artifact_revision",
    "manifest",
    "parent_route_proof",
    "route_forest",
    "route_consensus",
    "codex_retrosynthesis",
    "reaction_proof",
    "stock_closure",
    "route_verifier",
    "agent_blackboard_snapshot",
    "agentic_run_audit",
    "agentic_capability_audit",
)


class BlackboardJournalError(RuntimeError):
    """Raised when an existing recovery journal cannot be trusted."""


class _DuplicateJSONKey(ValueError):
    """Raised before event validation when JSON contains an ambiguous key."""


class _NonFiniteJSONNumber(ValueError):
    """Raised before event validation for NaN, Infinity or overflowed floats."""


def blackboard_event_journal_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "blackboard_events" / "events.jsonl"


def blackboard_controller_single_writer(func: _F) -> _F:
    """Serialize a complete controller invocation for one run directory."""

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        output_dir = kwargs.get("output_dir") or kwargs.get("run_dir")
        if output_dir is None and args:
            output_dir = args[0]
        if output_dir is None:
            raise BlackboardJournalError("blackboard_controller_output_dir_missing")
        try:
            timeout_seconds = float(
                os.environ.get(
                    "AUTOPLANNER_CONTROLLER_LOCK_TIMEOUT_SECONDS",
                    "30",
                )
            )
        except ValueError:
            timeout_seconds = 30.0
        with blackboard_controller_lock(
            output_dir,
            timeout_seconds=max(0.0, timeout_seconds),
        ):
            return func(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


@contextmanager
def blackboard_controller_lock(
    run_dir: str | Path,
    *,
    timeout_seconds: float = 30.0,
) -> Iterator[None]:
    """Hold the run-wide lock, independently from short journal appends."""

    path = Path(run_dir) / "blackboard_events" / "controller.lock"
    with _exclusive_file_lock(
        path,
        timeout_seconds=timeout_seconds,
        timeout_reason="blackboard_controller_lock_timeout",
    ):
        yield


def append_blackboard_checkpoint(
    run_dir: str | Path,
    blackboard: Mapping[str, Any],
    *,
    stage: str,
    metadata: Mapping[str, Any] | None = None,
    expected_previous_event_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append one complete recoverable projection and return board + event.

    Existing events are fully validated before append.  This is intentionally
    conservative: a corrupt or target-mismatched history must never be
    extended into an apparently valid chain.
    """

    stage_name = str(stage or "").strip()
    if not stage_name:
        raise BlackboardJournalError("blackboard_event_stage_missing")
    if any(token in stage_name.lower() for token in ("final", "closeout")):
        raise BlackboardJournalError(
            f"blackboard_event_forbidden_closeout_stage:{stage_name}"
        )
    board = deepcopy(dict(blackboard))
    identity = _blackboard_identity(board)
    path = blackboard_event_journal_path(run_dir)
    checkpoint = _recovery_checkpoint(board)
    payload_sha256 = _canonical_sha256(checkpoint)
    # The complete validate -> sequence allocation -> append transition must be
    # serialized.  Locking only the write would still allow two controller
    # processes to allocate the same sequence from the same previous digest.
    with _exclusive_journal_lock(path):
        events = _load_and_validate_events(path, expected_identity=identity)
        _require_expected_head(
            events,
            board=board,
            explicit_expected=expected_previous_event_sha256,
        )
        sequence = len(events) + 1
        previous_sha256 = (
            str(events[-1].get("event_sha256") or "") if events else ""
        )
        event = {
            "schema_version": BLACKBOARD_EVENT_SCHEMA,
            "sequence": sequence,
            "event_id": f"blackboard-event:{sequence:08d}:{payload_sha256[:16]}",
            "event_type": "blackboard_checkpoint",
            "stage": stage_name,
            "case_id": identity["case_id"],
            "target_identity": identity["target_identity"],
            "target_identity_sha256": identity["target_identity_sha256"],
            "previous_event_sha256": previous_sha256,
            "checkpoint_sha256": payload_sha256,
            "metadata": deepcopy(dict(metadata or {})),
        }
        event.update(
            _checkpoint_storage_fields(
                path,
                checkpoint=checkpoint,
                checkpoint_sha256=payload_sha256,
            )
        )
        event["event_sha256"] = _canonical_sha256(event)
        _append_json_line_durable(path, event)
    board["blackboard_event_journal"] = _journal_summary(
        path,
        event_count=sequence,
        last_event=event,
        rehydrated=bool(
            (board.get("blackboard_rehydration") or {}).get("recovered")
            if isinstance(board.get("blackboard_rehydration"), Mapping)
            else (board.get("blackboard_event_journal") or {}).get("rehydrated")
            if isinstance(board.get("blackboard_event_journal"), Mapping)
            else False
        ),
    )
    return board, event


def begin_blackboard_action(
    run_dir: str | Path,
    blackboard: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    round_index: int,
    reserved_budget_state: Mapping[str, Any],
    allow_idempotent_retry: bool = False,
    max_idempotent_recovery_attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Claim an action attempt or return its durable prepared/commit state.

    A prior ``started`` event without a prepared result is indeterminate by
    default, because an external side effect may have completed before the
    process died.  Callers may explicitly authorize a bounded, uncharged
    retry for an idempotent local/read-only action.
    """

    board = deepcopy(dict(blackboard))
    identity = _blackboard_identity(board)
    path = blackboard_event_journal_path(run_dir)
    binding = _action_binding(action, round_index=round_index)
    with _exclusive_journal_lock(path):
        events = _load_and_validate_events(path, expected_identity=identity)
        _require_expected_head(events, board=board, explicit_expected=None)
        lifecycle = _action_lifecycle(events, action_key=binding["action_key"])
        if lifecycle.get("status") in {"prepared", "committed"}:
            return board, lifecycle

        attempts = list(lifecycle.get("started_events") or [])
        if attempts:
            maximum_attempts = max(
                1,
                int(max_idempotent_recovery_attempts or 1),
            )
            if not allow_idempotent_retry or len(attempts) >= maximum_attempts:
                return board, {
                    **lifecycle,
                    "status": "indeterminate",
                    "reason": "prior_action_started_without_prepared_result",
                    "automatic_retry_allowed": False,
                    "charged_attempt_count": len(attempts),
                }
            prior_event = dict(attempts[-1])
            attempt_index = len(attempts) + 1
            current_budget = deepcopy(dict(board.get("budget_state") or {}))
            action_payload = {
                "schema_version": BLACKBOARD_ACTION_STARTED_SCHEMA,
                **binding,
                "action": deepcopy(dict(action)),
                "attempt_index": attempt_index,
                "retry_of_event_id": str(prior_event.get("event_id") or ""),
                "retry_reason": "idempotent_local_action_recovery",
                "budget_pre_state": current_budget,
                "budget_after_reservation": current_budget,
                "budget_reservation_sha256": _canonical_sha256(
                    current_budget
                ),
            }
            event = _append_event_locked(
                path,
                events=events,
                identity=identity,
                event_type="action_started",
                stage="agent_action_idempotent_retry_started",
                body_key="action_execution",
                body=action_payload,
                metadata={
                    "round_index": int(round_index),
                    "action_id": binding["action_id"],
                    "action_type": binding["action_type"],
                    "attempt_index": attempt_index,
                    "charged_retry": False,
                    "idempotent_recovery_retry": True,
                },
            )
            board["blackboard_event_journal"] = _journal_summary(
                path,
                event_count=int(event["sequence"]),
                last_event=event,
                rehydrated=_board_was_rehydrated(board),
            )
            return board, {
                "status": "started",
                "action_key": binding["action_key"],
                "started_event": event,
                "attempt_index": attempt_index,
                "charged_retry": False,
                "idempotent_recovery_retry": True,
            }
        attempt_index = len(attempts) + 1
        action_payload = {
            "schema_version": BLACKBOARD_ACTION_STARTED_SCHEMA,
            **binding,
            "action": deepcopy(dict(action)),
            "attempt_index": attempt_index,
            "retry_of_event_id": "",
            "retry_reason": "",
            "budget_pre_state": deepcopy(dict(board.get("budget_state") or {})),
            "budget_after_reservation": deepcopy(dict(reserved_budget_state)),
            "budget_reservation_sha256": _canonical_sha256(
                dict(reserved_budget_state)
            ),
        }
        event = _append_event_locked(
            path,
            events=events,
            identity=identity,
            event_type="action_started",
            stage="agent_action_started",
            body_key="action_execution",
            body=action_payload,
            metadata={
                "round_index": int(round_index),
                "action_id": binding["action_id"],
                "action_type": binding["action_type"],
                "attempt_index": attempt_index,
                "charged_retry": False,
            },
        )
    board["budget_state"] = deepcopy(dict(reserved_budget_state))
    board["blackboard_event_journal"] = _journal_summary(
        path,
        event_count=int(event["sequence"]),
        last_event=event,
        rehydrated=_board_was_rehydrated(board),
    )
    return board, {
        "status": "started",
        "action_key": binding["action_key"],
        "started_event": event,
        "attempt_index": attempt_index,
        "charged_retry": False,
    }


def prepare_blackboard_action_result(
    run_dir: str | Path,
    blackboard: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    round_index: int,
    started_event: Mapping[str, Any],
    action_result: Mapping[str, Any],
    tool_records: list[Mapping[str, Any]],
    artifact_updates: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Durably place a completed action result in the replay outbox."""

    board = deepcopy(dict(blackboard))
    identity = _blackboard_identity(board)
    path = blackboard_event_journal_path(run_dir)
    binding = _action_binding(action, round_index=round_index)
    started_sha256 = str(started_event.get("event_sha256") or "")
    prepared = {
        "schema_version": BLACKBOARD_ACTION_PREPARED_SCHEMA,
        **binding,
        "started_event_id": str(started_event.get("event_id") or ""),
        "started_event_sha256": started_sha256,
        "attempt_index": int(
            ((started_event.get("action_execution") or {}).get("attempt_index") or 0)
            if isinstance(started_event.get("action_execution"), Mapping)
            else 0
        ),
        "action_result": deepcopy(dict(action_result)),
        "replayable_action_result": _sanitize_prepared_action_result(
            dict(action_result)
        ),
        "tool_records": deepcopy([dict(row) for row in tool_records]),
        "artifact_updates": deepcopy(dict(artifact_updates)),
        "replayable_artifact_updates": _sanitize_prepared_artifact_updates(
            artifact_updates
        ),
    }
    prepared["artifact_refs"] = _collect_artifact_refs(
        prepared["artifact_updates"]
    )
    prepared["result_sha256"] = _canonical_sha256(prepared["action_result"])
    prepared["replayable_result_sha256"] = _canonical_sha256(
        prepared["replayable_action_result"]
    )
    prepared["tool_records_sha256"] = _canonical_sha256(prepared["tool_records"])
    prepared["artifact_updates_sha256"] = _canonical_sha256(
        prepared["artifact_updates"]
    )
    prepared["replayable_artifact_updates_sha256"] = _canonical_sha256(
        prepared["replayable_artifact_updates"]
    )
    prepared["artifact_refs_sha256"] = _canonical_sha256(
        prepared["artifact_refs"]
    )
    with _exclusive_journal_lock(path):
        events = _load_and_validate_events(path, expected_identity=identity)
        _require_expected_head(events, board=board, explicit_expected=started_sha256)
        if not events or events[-1].get("event_sha256") != started_sha256:
            raise BlackboardJournalError("blackboard_action_started_head_mismatch")
        if str(
            ((events[-1].get("action_execution") or {}).get("action_key") or "")
        ) != binding["action_key"]:
            raise BlackboardJournalError("blackboard_action_started_binding_mismatch")
        event = _append_event_locked(
            path,
            events=events,
            identity=identity,
            event_type="action_result_prepared",
            stage="agent_action_result_prepared",
            body_key="action_execution",
            body=prepared,
            metadata={
                "round_index": int(round_index),
                "action_id": binding["action_id"],
                "action_type": binding["action_type"],
            },
        )
    board["blackboard_event_journal"] = _journal_summary(
        path,
        event_count=int(event["sequence"]),
        last_event=event,
        rehydrated=_board_was_rehydrated(board),
    )
    return board, event


def commit_prepared_blackboard_action(
    run_dir: str | Path,
    blackboard: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    round_index: int,
    prepared_event: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Atomically commit action result, budget and recovery checkpoint."""

    board = deepcopy(dict(blackboard))
    identity = _blackboard_identity(board)
    path = blackboard_event_journal_path(run_dir)
    binding = _action_binding(action, round_index=round_index)
    prepared_sha256 = str(prepared_event.get("event_sha256") or "")
    prepared_execution = dict(prepared_event.get("action_execution") or {})
    checkpoint = _recovery_checkpoint(board)
    committed = {
        "schema_version": BLACKBOARD_ACTION_COMMITTED_SCHEMA,
        **binding,
        "prepared_event_id": str(prepared_event.get("event_id") or ""),
        "prepared_event_sha256": prepared_sha256,
        "result_sha256": str(prepared_execution.get("result_sha256") or ""),
        "budget_committed": deepcopy(dict(board.get("budget_state") or {})),
        "checkpoint_sha256": _canonical_sha256(checkpoint),
    }
    committed.update(
        _checkpoint_storage_fields(
            path,
            checkpoint=checkpoint,
            checkpoint_sha256=str(committed["checkpoint_sha256"]),
        )
    )
    with _exclusive_journal_lock(path):
        events = _load_and_validate_events(path, expected_identity=identity)
        _require_expected_head(events, board=board, explicit_expected=None)
        lifecycle = _action_lifecycle(events, action_key=binding["action_key"])
        current_prepared = dict(lifecycle.get("prepared_event") or {})
        if (
            lifecycle.get("status") != "prepared"
            or current_prepared.get("event_sha256") != prepared_sha256
        ):
            raise BlackboardJournalError("blackboard_action_prepared_state_mismatch")
        if str(prepared_execution.get("action_key") or "") != binding["action_key"]:
            raise BlackboardJournalError("blackboard_action_prepared_binding_mismatch")
        event = _append_event_locked(
            path,
            events=events,
            identity=identity,
            event_type="action_committed",
            stage="agent_action_committed",
            body_key="action_execution",
            body=committed,
            metadata={
                "round_index": int(round_index),
                "action_id": binding["action_id"],
                "action_type": binding["action_type"],
            },
        )
    board["blackboard_event_journal"] = _journal_summary(
        path,
        event_count=int(event["sequence"]),
        last_event=event,
        rehydrated=_board_was_rehydrated(board),
    )
    return board, event


def replay_prepared_blackboard_action(
    prepared_event: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-derive the non-authoritative replay view on the current host."""

    execution = dict(prepared_event.get("action_execution") or {})
    if execution.get("schema_version") != BLACKBOARD_ACTION_PREPARED_SCHEMA:
        raise BlackboardJournalError("blackboard_action_replay_payload_invalid")
    return {
        "action_result": _sanitize_prepared_action_result(
            dict(execution.get("action_result") or {})
        ),
        "tool_records": [
            deepcopy(dict(row))
            for row in execution.get("tool_records") or []
            if isinstance(row, Mapping)
        ],
        "artifact_updates": _sanitize_prepared_artifact_updates(
            dict(execution.get("artifact_updates") or {})
        ),
        "requires_current_host_scientific_replay": True,
    }


def rehydrate_blackboard_from_events(
    initial_blackboard: Mapping[str, Any],
    *,
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reduce the validated journal over a fresh target-bound blackboard."""

    initial = deepcopy(dict(initial_blackboard))
    identity = _blackboard_identity(initial)
    path = blackboard_event_journal_path(run_dir)
    # Readers share the writer lock while copying the chain into memory.  This
    # prevents startup recovery from observing a partially flushed JSON line.
    with _exclusive_journal_lock(path):
        events = _load_and_validate_events(path, expected_identity=identity)
    if not events:
        report = {
            "schema_version": BLACKBOARD_REHYDRATION_SCHEMA,
            "recovered": False,
            "case_id": identity["case_id"],
            "target_identity_sha256": identity["target_identity_sha256"],
            "journal_path": str(path),
            "event_count": 0,
            "last_event_sha256": "",
            "restored_fields": [],
            "projection_source_used": False,
            "final_or_closeout_authority_restored": False,
        }
        _write_recovery_report(path, report)
        return initial, report

    reduced = deepcopy(initial)
    restored_fields: list[str] = []
    tombstoned_fields: list[str] = []
    for event in events:
        checkpoint = _event_checkpoint(event, journal_path=path)
        if checkpoint is None:
            if event.get("event_type") == "action_started":
                execution = dict(event.get("action_execution") or {})
                budget = execution.get("budget_after_reservation")
                if isinstance(budget, Mapping):
                    reduced["budget_state"] = deepcopy(dict(budget))
                    if "budget_state" not in restored_fields:
                        restored_fields.append("budget_state")
            continue
        projection = dict(checkpoint.get("recoverable_blackboard") or {})
        for field in _RECOVERABLE_FIELDS:
            value = projection[field]
            if _is_tombstone(value):
                reduced.pop(field, None)
                if field not in tombstoned_fields:
                    tombstoned_fields.append(field)
                continue
            reduced[field] = deepcopy(value)
            if field not in restored_fields:
                restored_fields.append(field)

    reduced["budget_state"] = _merge_recovered_budget_state(
        current=dict(initial.get("budget_state") or {}),
        recovered=dict(reduced.get("budget_state") or {}),
    )
    last_event = events[-1]
    for field in (
        "final_verdict",
        "parent_route_proof",
        "route_proof_bundle",
        "codex_agent_team",
        "codex_precursor_frontier_injection",
        "route_consensus",
        "route_consensus_graph",
    ):
        reduced.pop(field, None)
    reduced["parent_route_proof"] = {}
    reduced["route_proof_bundle"] = {}
    reduced["current_belief"] = _sanitize_current_belief(
        reduced.get("current_belief")
    )
    reduced["chemenzy_route_proof_banks"] = _sanitize_chemenzy_banks(
        reduced.get("chemenzy_route_proof_banks")
    )
    lifecycle = _all_action_lifecycles(events)
    pending_prepared = [
        deepcopy(dict(row["prepared_event"]))
        for row in lifecycle.values()
        if row.get("status") == "prepared"
        and isinstance(row.get("prepared_event"), Mapping)
    ]
    started_without_result = [
        deepcopy(dict(row["started_events"][-1]))
        for row in lifecycle.values()
        if row.get("status") == "started"
        and row.get("started_events")
    ]
    reduced["blackboard_event_journal"] = _journal_summary(
        path,
        event_count=len(events),
        last_event=last_event,
        rehydrated=True,
    )
    report = {
        "schema_version": BLACKBOARD_REHYDRATION_SCHEMA,
        "recovered": True,
        "case_id": identity["case_id"],
        "target_identity_sha256": identity["target_identity_sha256"],
        "journal_path": str(path),
        "event_count": len(events),
        "last_event_sha256": str(last_event.get("event_sha256") or ""),
        "last_stage": str(last_event.get("stage") or ""),
        "restored_fields": restored_fields,
        "tombstoned_fields": tombstoned_fields,
        "pending_prepared_action_count": len(pending_prepared),
        "started_without_result_count": len(started_without_result),
        "pending_prepared_actions": pending_prepared,
        "started_without_result_actions": started_without_result,
        "projection_source_used": False,
        "final_or_closeout_authority_restored": False,
        "semantics": {
            "deterministic_last_checkpoint_reducer": True,
            "agent_blackboard_json_ignored": True,
            "target_and_case_must_match": True,
            "journal_is_not_scientific_trust_root": True,
            "codex_consensus_and_solved_state_not_recoverable": True,
            "campaign_authority_revision_locator_is_recoverable": True,
            "prepared_results_require_controller_commit": True,
        },
    }
    _write_recovery_report(path, report)
    return reduced, report


def _recoverable_blackboard_projection(
    blackboard: Mapping[str, Any],
) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for field in _RECOVERABLE_FIELDS:
        if field not in blackboard:
            projection[field] = _tombstone(field)
            continue
        value = deepcopy(blackboard[field])
        if field == "artifact_refs":
            value = _recoverable_artifact_refs(value)
        elif field == "current_belief":
            value = _sanitize_current_belief(value)
        elif field == "chemenzy_route_proof_banks":
            value = _sanitize_chemenzy_banks(value)
        projection[field] = value
    return projection


def _recovery_checkpoint(blackboard: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": BLACKBOARD_CHECKPOINT_SCHEMA,
        "recoverable_blackboard": _recoverable_blackboard_projection(blackboard),
        "semantics": {
            "mutable_agent_blackboard_json_is_not_recovery_authority": True,
            "journal_is_not_scientific_trust_root": True,
            "final_verdict_not_recoverable": True,
            "parent_route_proof_not_recoverable": True,
            "codex_consensus_not_recoverable": True,
            "closeout_success_not_recoverable": True,
            "missing_fields_are_explicit_tombstones": True,
        },
    }


def _checkpoint_storage_fields(
    journal_path: Path,
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    encoded = _canonical_json_bytes(checkpoint)
    if len(encoded) <= _CHECKPOINT_INLINE_MAX_BYTES:
        return {"checkpoint": deepcopy(dict(checkpoint))}
    if hashlib.sha256(encoded).hexdigest() != checkpoint_sha256:
        raise BlackboardJournalError("blackboard_checkpoint_canonical_digest_mismatch")
    object_dir = journal_path.parent / "checkpoint_objects"
    object_path = object_dir / f"{checkpoint_sha256}.json"
    object_dir.mkdir(parents=True, exist_ok=True)
    if object_path.exists():
        try:
            if object_path.read_bytes() != encoded:
                raise BlackboardJournalError(
                    "blackboard_checkpoint_object_digest_collision"
                )
        except OSError as exc:
            raise BlackboardJournalError(
                f"blackboard_checkpoint_object_unreadable:{type(exc).__name__}"
            ) from exc
    else:
        temporary = object_path.with_name(
            f".{object_path.name}.{os.getpid()}.{time.time_ns():x}.tmp"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, object_path)
            _fsync_directory(object_dir)
        except OSError as exc:
            raise BlackboardJournalError(
                f"blackboard_checkpoint_object_write_failed:{type(exc).__name__}"
            ) from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return {
        "checkpoint_ref": {
            "schema_version": BLACKBOARD_CHECKPOINT_REF_SCHEMA,
            "relative_path": f"checkpoint_objects/{checkpoint_sha256}.json",
            "checkpoint_sha256": checkpoint_sha256,
            "byte_count": len(encoded),
            "storage": "immutable_content_addressed_json",
        }
    }


def _resolve_checkpoint_storage(
    container: Mapping[str, Any],
    *,
    journal_path: Path,
) -> dict[str, Any]:
    inline = container.get("checkpoint")
    raw_ref = container.get("checkpoint_ref")
    if inline is not None and raw_ref is not None:
        raise BlackboardJournalError("blackboard_checkpoint_storage_ambiguous")
    if inline is not None:
        return dict(inline) if isinstance(inline, Mapping) else {}
    if not isinstance(raw_ref, Mapping):
        return {}
    ref = dict(raw_ref)
    digest = str(ref.get("checkpoint_sha256") or "")
    relative = str(ref.get("relative_path") or "")
    expected_relative = f"checkpoint_objects/{digest}.json"
    root = journal_path.parent.resolve()
    path = (root / Path(relative)).resolve()
    if (
        ref.get("schema_version") != BLACKBOARD_CHECKPOINT_REF_SCHEMA
        or not _is_sha256(digest)
        or relative.replace("\\", "/") != expected_relative
        or path.parent != (root / "checkpoint_objects").resolve()
        or not path.is_file()
    ):
        raise BlackboardJournalError("blackboard_checkpoint_ref_invalid")
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise BlackboardJournalError(
            f"blackboard_checkpoint_object_unreadable:{type(exc).__name__}"
        ) from exc
    if (
        len(encoded) != int(ref.get("byte_count") or -1)
        or hashlib.sha256(encoded).hexdigest() != digest
    ):
        raise BlackboardJournalError("blackboard_checkpoint_object_digest_mismatch")
    try:
        parsed = _strict_json_loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BlackboardJournalError("blackboard_checkpoint_object_json_invalid") from exc
    if (
        not isinstance(parsed, dict)
        or _canonical_sha256(parsed) != digest
        or _canonical_json_bytes(parsed) != encoded
    ):
        raise BlackboardJournalError("blackboard_checkpoint_object_not_canonical")
    return dict(parsed)


def _event_checkpoint(
    event: Mapping[str, Any],
    *,
    journal_path: Path,
) -> dict[str, Any] | None:
    if event.get("event_type") == "blackboard_checkpoint":
        return _resolve_checkpoint_storage(event, journal_path=journal_path)
    if event.get("event_type") == "action_committed":
        execution = dict(event.get("action_execution") or {})
        return _resolve_checkpoint_storage(
            execution,
            journal_path=journal_path,
        )
    return None


def _tombstone(field: str) -> dict[str, Any]:
    return {
        "schema_version": BLACKBOARD_TOMBSTONE_SCHEMA,
        "deleted": True,
        "field": str(field),
    }


def _is_tombstone(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("schema_version") == BLACKBOARD_TOMBSTONE_SCHEMA
        and value.get("deleted") is True
    )


def _sanitize_current_belief(value: Any) -> dict[str, Any]:
    belief = dict(value) if isinstance(value, Mapping) else {}
    sanitized = _strip_authority_hints(belief)
    return dict(sanitized) if isinstance(sanitized, Mapping) else {}


def _strip_authority_hints(value: Any) -> Any:
    blocked_tokens = (
        "solved",
        "closeout",
        "final_verdict",
        "parent_route_verifier",
        "parent_route_proof",
        "route_consensus",
        "codex_agent_team",
    )
    if isinstance(value, Mapping):
        return {
            str(key): _strip_authority_hints(item)
            for key, item in value.items()
            if not any(token in str(key).lower() for token in blocked_tokens)
        }
    if isinstance(value, list):
        return [
            _strip_authority_hints(item)
            for item in value
            if not (
                isinstance(item, str)
                and any(token in item.lower() for token in blocked_tokens)
            )
        ]
    if isinstance(value, str) and any(
        token in value.lower() for token in blocked_tokens
    ):
        return ""
    return deepcopy(value)


def _sanitize_chemenzy_banks(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    banks: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        sanitized = _strip_authority_hints(dict(raw))
        row = dict(sanitized) if isinstance(sanitized, Mapping) else {}
        row["requires_current_host_replay"] = True
        row["no_solved_claim"] = True
        banks.append(row)
    return banks


def _sanitize_prepared_action_result(value: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = _strip_authority_hints(dict(value))
    result = dict(sanitized) if isinstance(sanitized, Mapping) else {}
    result["restored_from_untrusted_operation_journal"] = True
    result["scientific_authority_requires_current_host_replay"] = True
    return result


def _sanitize_prepared_artifact_updates(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    blocked_tokens = (
        "final",
        "closeout",
        "parent_route_proof",
        "route_proof_bundle",
        "route_consensus",
        "codex_retrosynthesis_team",
        "frontier_ledger",
        "reaction_proof",
        "stock_closure",
        "route_verifier",
    )
    replayable: dict[str, Any] = {}
    for key, item in value.items():
        if any(token in str(key).lower() for token in blocked_tokens):
            continue
        replayable[str(key)] = _strip_authority_hints(item)
    return replayable


def _action_binding(
    action: Mapping[str, Any],
    *,
    round_index: int,
) -> dict[str, Any]:
    normalized = deepcopy(dict(action))
    payload = dict(normalized.get("payload") or {})
    payload.pop("timestamp", None)
    normalized["payload"] = payload
    signature_sha256 = _canonical_sha256(normalized)
    action_id = str(action.get("action_id") or "").strip()
    action_type = str(action.get("action_type") or "").strip()
    action_key = _canonical_sha256(
        {
            "round_index": int(round_index),
            "action_id": action_id,
            "action_type": action_type,
            "action_signature_sha256": signature_sha256,
        }
    )
    return {
        "round_index": int(round_index),
        "action_id": action_id,
        "action_type": action_type,
        "action_signature_sha256": signature_sha256,
        "action_key": action_key,
    }


def _action_lifecycle(
    events: list[dict[str, Any]],
    *,
    action_key: str,
) -> dict[str, Any]:
    started: list[dict[str, Any]] = []
    prepared_event: dict[str, Any] | None = None
    committed_event: dict[str, Any] | None = None
    for event in events:
        execution = event.get("action_execution")
        if not isinstance(execution, Mapping) or str(
            execution.get("action_key") or ""
        ) != action_key:
            continue
        if event.get("event_type") == "action_started":
            started.append(event)
            prepared_event = None
            committed_event = None
        elif event.get("event_type") == "action_result_prepared":
            prepared_event = event
            committed_event = None
        elif event.get("event_type") == "action_committed":
            committed_event = event
    status = (
        "committed"
        if committed_event is not None
        else "prepared"
        if prepared_event is not None
        else "started"
        if started
        else "missing"
    )
    return {
        "status": status,
        "action_key": action_key,
        "started_events": started,
        "prepared_event": prepared_event,
        "committed_event": committed_event,
    }


def _all_action_lifecycles(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    keys = {
        str((event.get("action_execution") or {}).get("action_key") or "")
        for event in events
        if isinstance(event.get("action_execution"), Mapping)
    }
    return {
        key: _action_lifecycle(events, action_key=key)
        for key in keys
        if key
    }


def _collect_artifact_refs(value: Any) -> list[str]:
    refs: set[str] = set()

    def visit(item: Any, key: str = "") -> None:
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif isinstance(item, (str, Path)) and (
            key.lower().endswith("_ref")
            or key.lower() in {"artifact_ref", "record_ref", "path"}
        ):
            text = str(item).strip()
            if text:
                refs.add(text)

    visit(value)
    return sorted(refs)


def _board_was_rehydrated(board: Mapping[str, Any]) -> bool:
    report = board.get("blackboard_rehydration")
    if isinstance(report, Mapping) and report.get("recovered") is True:
        return True
    summary = board.get("blackboard_event_journal")
    return bool(isinstance(summary, Mapping) and summary.get("rehydrated"))


def _require_expected_head(
    events: list[dict[str, Any]],
    *,
    board: Mapping[str, Any],
    explicit_expected: str | None,
) -> None:
    actual = str(events[-1].get("event_sha256") or "") if events else ""
    summary = board.get("blackboard_event_journal")
    board_expected = (
        str(summary.get("last_event_sha256") or "")
        if isinstance(summary, Mapping)
        else ""
    )
    expected = (
        str(explicit_expected)
        if explicit_expected is not None
        else board_expected
    )
    if events and not expected:
        raise BlackboardJournalError("blackboard_event_expected_head_missing")
    if expected != actual:
        raise BlackboardJournalError(
            "blackboard_event_stale_head:" f"expected={expected}:actual={actual}"
        )


def _append_event_locked(
    path: Path,
    *,
    events: list[dict[str, Any]],
    identity: Mapping[str, Any],
    event_type: str,
    stage: str,
    body_key: str,
    body: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sequence = len(events) + 1
    previous_sha256 = str(events[-1].get("event_sha256") or "") if events else ""
    body_digest = _canonical_sha256(body)
    event = {
        "schema_version": BLACKBOARD_EVENT_SCHEMA,
        "sequence": sequence,
        "event_id": f"blackboard-event:{sequence:08d}:{body_digest[:16]}",
        "event_type": event_type,
        "stage": stage,
        "case_id": identity["case_id"],
        "target_identity": deepcopy(dict(identity["target_identity"])),
        "target_identity_sha256": identity["target_identity_sha256"],
        "previous_event_sha256": previous_sha256,
        body_key: deepcopy(dict(body)),
        "metadata": deepcopy(dict(metadata or {})),
    }
    event["event_sha256"] = _canonical_sha256(event)
    _append_json_line_durable(path, event)
    return event


def _recoverable_artifact_refs(value: Any) -> dict[str, str]:
    refs = dict(value) if isinstance(value, Mapping) else {}
    return {
        str(key): str(ref)
        for key, ref in refs.items()
        if str(key)
        and str(ref)
        and not any(
            token in str(key).lower() for token in _BLOCKED_ARTIFACT_REF_TOKENS
        )
    }


def _merge_recovered_budget_state(
    *,
    current: dict[str, Any],
    recovered: dict[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(recovered)
    # A resumed invocation may intentionally extend or tighten its caps.  The
    # counters/history are durable, while current invocation limits remain the
    # caller's explicit policy.
    for key, value in current.items():
        if key == "schema_version" or key.startswith("max_") or key in {
            "enable_analogical_templates",
            "template_radius_policy",
            "analog_template_confidence_threshold",
        }:
            merged[key] = deepcopy(value)
    return merged


def _load_and_validate_events(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        journal_bytes = path.read_bytes()
    except OSError as exc:
        raise BlackboardJournalError(
            f"blackboard_event_journal_unreadable:{type(exc).__name__}"
        ) from exc
    if not journal_bytes:
        return []

    has_final_newline = journal_bytes.endswith(b"\n")
    records = journal_bytes.split(b"\n")
    if has_final_newline:
        records.pop()
    events: list[dict[str, Any]] = []
    previous_sha256 = ""
    record_start = 0
    for record_index, record_bytes in enumerate(records):
        line_number = record_index + 1
        is_unterminated_tail = (
            record_index == len(records) - 1 and not has_final_newline
        )
        try:
            line = record_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            reason = f"blackboard_event_utf8_invalid:{line_number}"
            if is_unterminated_tail:
                _quarantine_torn_tail(
                    path,
                    fragment=record_bytes,
                    valid_prefix_length=record_start,
                    failure_reason=reason,
                )
                break
            raise BlackboardJournalError(reason) from exc
        if not line.strip():
            reason = f"blackboard_event_blank_record:{line_number}"
            if is_unterminated_tail:
                _quarantine_torn_tail(
                    path,
                    fragment=record_bytes,
                    valid_prefix_length=record_start,
                    failure_reason=reason,
                )
                break
            raise BlackboardJournalError(reason)
        try:
            raw = _strict_json_loads(line)
        except _DuplicateJSONKey as exc:
            raise BlackboardJournalError(
                f"blackboard_event_json_duplicate_key:{line_number}"
            ) from exc
        except _NonFiniteJSONNumber as exc:
            raise BlackboardJournalError(
                f"blackboard_event_json_non_finite_number:{line_number}"
            ) from exc
        except json.JSONDecodeError as exc:
            reason = f"blackboard_event_json_invalid:{line_number}"
            if is_unterminated_tail:
                _quarantine_torn_tail(
                    path,
                    fragment=record_bytes,
                    valid_prefix_length=record_start,
                    failure_reason=reason,
                )
                break
            raise BlackboardJournalError(reason) from exc
        if not isinstance(raw, dict):
            reason = f"blackboard_event_record_invalid:{line_number}"
            if is_unterminated_tail:
                _quarantine_torn_tail(
                    path,
                    fragment=record_bytes,
                    valid_prefix_length=record_start,
                    failure_reason=reason,
                )
                break
            raise BlackboardJournalError(reason)
        event = dict(raw)
        try:
            _validate_event(
                event,
                line_number=line_number,
                previous_sha256=previous_sha256,
                expected_identity=expected_identity,
                journal_path=path,
            )
        except BlackboardJournalError as exc:
            if is_unterminated_tail and _tail_validation_failure_can_quarantine(
                exc
            ):
                _quarantine_torn_tail(
                    path,
                    fragment=record_bytes,
                    valid_prefix_length=record_start,
                    failure_reason=str(exc),
                )
                break
            raise
        previous_sha256 = str(event["event_sha256"])
        events.append(event)
        record_start += len(record_bytes) + 1
    return events


def _strict_json_loads(value: str) -> Any:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise _DuplicateJSONKey(key)
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise _NonFiniteJSONNumber(value)

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise _NonFiniteJSONNumber(value)
        return parsed

    parsed = json.loads(
        value,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
        parse_float=parse_finite_float,
    )
    if _contains_non_finite_number(parsed):
        raise _NonFiniteJSONNumber("overflow")
    return parsed


def _contains_non_finite_number(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_non_finite_number(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite_number(item) for item in value)
    return False


def _tail_validation_failure_can_quarantine(
    error: BlackboardJournalError,
) -> bool:
    """Keep identity/ordering/digest ambiguity fail-closed, even at EOF."""

    reason = str(error).lower()
    fail_closed_tokens = (
        "sequence_invalid",
        "case_id_mismatch",
        "identity",
        "digest",
        "parent_mismatch",
        "parent_invalid",
        "binding",
    )
    return not any(token in reason for token in fail_closed_tokens)


def _quarantine_torn_tail(
    path: Path,
    *,
    fragment: bytes,
    valid_prefix_length: int,
    failure_reason: str,
) -> Path:
    """Preserve then remove one invalid, unterminated final write fragment.

    Callers hold ``events.lock`` across validation, evidence persistence and
    truncation.  The sidecar is forensic evidence only; no recovery reducer
    reads it and it can never grant scientific or action authority.
    """

    fragment_sha256 = hashlib.sha256(fragment).hexdigest()
    quarantined_at_unix_ns = time.time_ns()
    evidence = {
        "schema_version": BLACKBOARD_TORN_TAIL_QUARANTINE_SCHEMA,
        "journal_path": str(path),
        "valid_prefix_length": int(valid_prefix_length),
        "fragment_length": len(fragment),
        "fragment_sha256": fragment_sha256,
        "fragment_base64": base64.b64encode(fragment).decode("ascii"),
        "failure_reason": str(failure_reason),
        "quarantined_at_unix_ns": quarantined_at_unix_ns,
        "quarantined_under_exclusive_journal_lock": True,
        "scientific_authority": False,
        "action_authority": False,
        "replay_eligible": False,
    }
    sidecar = path.with_name(
        f"{path.stem}.torn-tail.{fragment_sha256[:16]}."
        f"{quarantined_at_unix_ns:x}.json"
    )
    _write_json_sidecar_durable(sidecar, evidence)
    try:
        with path.open("r+b") as handle:
            handle.truncate(valid_prefix_length)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise BlackboardJournalError(
            f"blackboard_event_torn_tail_truncate_failed:{type(exc).__name__}"
        ) from exc
    _fsync_directory(path.parent)
    return sidecar


def _write_json_sidecar_durable(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(
        f".bb-tail-{os.getpid()}-{time.time_ns():x}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise BlackboardJournalError(
            f"blackboard_event_torn_tail_evidence_failed:{type(exc).__name__}"
        ) from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_event(
    event: Mapping[str, Any],
    *,
    line_number: int,
    previous_sha256: str,
    expected_identity: Mapping[str, Any],
    journal_path: Path,
) -> None:
    prefix = f"blackboard_event:{line_number}"
    if event.get("schema_version") != BLACKBOARD_EVENT_SCHEMA:
        raise BlackboardJournalError(f"{prefix}:schema_invalid")
    sequence = event.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != line_number:
        raise BlackboardJournalError(f"{prefix}:sequence_invalid")
    event_type = str(event.get("event_type") or "")
    if event_type not in {
        "blackboard_checkpoint",
        "action_started",
        "action_result_prepared",
        "action_committed",
    }:
        raise BlackboardJournalError(f"{prefix}:event_type_invalid")
    if str(event.get("case_id") or "") != str(expected_identity["case_id"]):
        raise BlackboardJournalError(f"{prefix}:case_id_mismatch")
    if str(event.get("target_identity_sha256") or "") != str(
        expected_identity["target_identity_sha256"]
    ):
        raise BlackboardJournalError(f"{prefix}:target_identity_mismatch")
    target_identity = event.get("target_identity")
    if not isinstance(target_identity, Mapping) or _canonical_sha256(
        target_identity
    ) != str(event.get("target_identity_sha256") or ""):
        raise BlackboardJournalError(f"{prefix}:target_identity_digest_mismatch")
    if str(event.get("previous_event_sha256") or "") != previous_sha256:
        raise BlackboardJournalError(f"{prefix}:previous_digest_mismatch")
    if event_type == "blackboard_checkpoint":
        checkpoint = _resolve_checkpoint_storage(
            event,
            journal_path=journal_path,
        )
        _validate_checkpoint(
            checkpoint,
            supplied_digest=str(event.get("checkpoint_sha256") or ""),
            prefix=prefix,
        )
    elif event_type == "action_started":
        _validate_action_started(event, prefix=prefix)
    elif event_type == "action_result_prepared":
        _validate_action_prepared(event, prefix=prefix)
    elif event_type == "action_committed":
        _validate_action_committed(
            event,
            prefix=prefix,
            journal_path=journal_path,
        )
    event_without_digest = dict(event)
    supplied_digest = str(event_without_digest.pop("event_sha256", ""))
    if not _is_sha256(supplied_digest) or _canonical_sha256(
        event_without_digest
    ) != supplied_digest:
        raise BlackboardJournalError(f"{prefix}:event_digest_mismatch")


def _validate_checkpoint(
    checkpoint: Any,
    *,
    supplied_digest: str,
    prefix: str,
) -> None:
    if not isinstance(checkpoint, Mapping) or checkpoint.get(
        "schema_version"
    ) != BLACKBOARD_CHECKPOINT_SCHEMA:
        raise BlackboardJournalError(f"{prefix}:checkpoint_schema_invalid")
    projection = checkpoint.get("recoverable_blackboard")
    if not isinstance(projection, Mapping):
        raise BlackboardJournalError(f"{prefix}:checkpoint_payload_invalid")
    missing = [field for field in _RECOVERABLE_FIELDS if field not in projection]
    if missing:
        raise BlackboardJournalError(
            f"{prefix}:checkpoint_incomplete:{','.join(missing)}"
        )
    if _canonical_sha256(checkpoint) != supplied_digest:
        raise BlackboardJournalError(f"{prefix}:checkpoint_digest_mismatch")


def _validate_action_started(event: Mapping[str, Any], *, prefix: str) -> None:
    execution = event.get("action_execution")
    if not isinstance(execution, Mapping) or execution.get(
        "schema_version"
    ) != BLACKBOARD_ACTION_STARTED_SCHEMA:
        raise BlackboardJournalError(f"{prefix}:action_started_schema_invalid")
    _validate_action_binding(execution, prefix=prefix)
    if not isinstance(execution.get("action"), Mapping):
        raise BlackboardJournalError(f"{prefix}:action_started_payload_invalid")
    expected_binding = _action_binding(
        dict(execution["action"]),
        round_index=int(execution.get("round_index") or 0),
    )
    if any(
        str(execution.get(key) or "") != str(expected_binding[key])
        for key in (
            "action_key",
            "action_id",
            "action_type",
            "action_signature_sha256",
        )
    ):
        raise BlackboardJournalError(f"{prefix}:action_started_binding_mismatch")
    if _canonical_sha256(execution.get("budget_after_reservation")) != str(
        execution.get("budget_reservation_sha256") or ""
    ):
        raise BlackboardJournalError(f"{prefix}:action_budget_digest_mismatch")


def _validate_action_prepared(event: Mapping[str, Any], *, prefix: str) -> None:
    execution = event.get("action_execution")
    if not isinstance(execution, Mapping) or execution.get(
        "schema_version"
    ) != BLACKBOARD_ACTION_PREPARED_SCHEMA:
        raise BlackboardJournalError(f"{prefix}:action_prepared_schema_invalid")
    _validate_action_binding(execution, prefix=prefix)
    if str(execution.get("started_event_sha256") or "") != str(
        event.get("previous_event_sha256") or ""
    ):
        raise BlackboardJournalError(f"{prefix}:action_prepared_parent_mismatch")
    for value_key, digest_key in (
        ("action_result", "result_sha256"),
        ("replayable_action_result", "replayable_result_sha256"),
        ("tool_records", "tool_records_sha256"),
        ("artifact_updates", "artifact_updates_sha256"),
        (
            "replayable_artifact_updates",
            "replayable_artifact_updates_sha256",
        ),
        ("artifact_refs", "artifact_refs_sha256"),
    ):
        if _canonical_sha256(execution.get(value_key)) != str(
            execution.get(digest_key) or ""
        ):
            raise BlackboardJournalError(
                f"{prefix}:action_prepared_{value_key}_digest_mismatch"
            )


def _validate_action_committed(
    event: Mapping[str, Any],
    *,
    prefix: str,
    journal_path: Path,
) -> None:
    execution = event.get("action_execution")
    if not isinstance(execution, Mapping) or execution.get(
        "schema_version"
    ) != BLACKBOARD_ACTION_COMMITTED_SCHEMA:
        raise BlackboardJournalError(f"{prefix}:action_committed_schema_invalid")
    _validate_action_binding(execution, prefix=prefix)
    if not _is_sha256(str(execution.get("prepared_event_sha256") or "")):
        raise BlackboardJournalError(f"{prefix}:action_committed_parent_invalid")
    checkpoint = _resolve_checkpoint_storage(
        execution,
        journal_path=journal_path,
    )
    _validate_checkpoint(
        checkpoint,
        supplied_digest=str(execution.get("checkpoint_sha256") or ""),
        prefix=prefix,
    )


def _validate_action_binding(
    execution: Mapping[str, Any],
    *,
    prefix: str,
) -> None:
    if not str(execution.get("action_key") or "") or not _is_sha256(
        str(execution.get("action_signature_sha256") or "")
    ):
        raise BlackboardJournalError(f"{prefix}:action_binding_invalid")


def _blackboard_identity(blackboard: Mapping[str, Any]) -> dict[str, Any]:
    if str(blackboard.get("schema_version") or "") != "agent_blackboard.v1":
        raise BlackboardJournalError("blackboard_recovery_schema_invalid")
    case_id = str(blackboard.get("case_id") or "").strip()
    profile = blackboard.get("target_profile")
    if not case_id or not isinstance(profile, Mapping):
        raise BlackboardJournalError("blackboard_recovery_identity_missing")
    target_identity = {
        "schema_version": "agent_blackboard_target_identity.v1",
        "target_name": str(profile.get("target_name") or ""),
        "target_smiles": str(profile.get("target_smiles") or ""),
        "canonical_smiles": str(profile.get("canonical_smiles") or ""),
        "isomeric_smiles": str(profile.get("isomeric_smiles") or ""),
        "inchi_key": str(profile.get("inchi_key") or ""),
    }
    if not any(
        str(target_identity[key] or "")
        for key in ("target_smiles", "canonical_smiles", "isomeric_smiles", "inchi_key")
    ):
        raise BlackboardJournalError("blackboard_recovery_target_missing")
    return {
        "case_id": case_id,
        "target_identity": target_identity,
        "target_identity_sha256": _canonical_sha256(target_identity),
    }


def _journal_summary(
    path: Path,
    *,
    event_count: int,
    last_event: Mapping[str, Any],
    rehydrated: bool,
) -> dict[str, Any]:
    return {
        "schema_version": BLACKBOARD_JOURNAL_SUMMARY_SCHEMA,
        "authority": "digest_chained_blackboard_events",
        "journal_path": str(path),
        "event_count": int(event_count),
        "last_event_id": str(last_event.get("event_id") or ""),
        "last_event_sha256": str(last_event.get("event_sha256") or ""),
        "last_stage": str(last_event.get("stage") or ""),
        "rehydrated": bool(rehydrated),
        "agent_blackboard_json_is_projection_only": True,
        "final_or_closeout_authority_restored": False,
    }


def _append_json_line_durable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        needs_separator = False
        if size:
            handle.seek(-1, os.SEEK_END)
            needs_separator = handle.read(1) != b"\n"
        handle.seek(0, os.SEEK_END)
        if needs_separator:
            encoded = b"\n" + encoded
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _exclusive_journal_lock(
    journal_path: Path,
    *,
    timeout_seconds: float = 10.0,
) -> Iterator[None]:
    """Hold a cross-process exclusive lock adjacent to the event journal."""

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = journal_path.parent / "events.lock"
    with _exclusive_file_lock(
        lock_path,
        timeout_seconds=timeout_seconds,
        timeout_reason="blackboard_event_journal_lock_timeout",
    ):
        yield


@contextmanager
def _exclusive_file_lock(
    lock_path: Path,
    *,
    timeout_seconds: float,
    timeout_reason: str,
) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            # ``msvcrt.locking`` locks a byte range from the current file
            # position, so retain one byte in the stable sidecar lock file.
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())

        acquired = False
        while not acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(
                        handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                acquired = True
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise BlackboardJournalError(timeout_reason) from exc
                time.sleep(0.05)

        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_recovery_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path.parent / "recovery_report.json", dict(report))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)

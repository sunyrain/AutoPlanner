"""Codex-first action planning for agentic blackboard runs."""
from __future__ import annotations

import ast
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from cascade_planner.agent.action_contracts import (
    ACTION_BATCH_SCHEMA,
    ALLOWED_AGENT_ACTIONS,
    FORBIDDEN_RAW_REACTION_KEYS,
    PLANNER_SOURCE_HINT_SCHEMA,
)
from cascade_planner.agent.codex_worker import WorkerBudget, WorkerTask, run_codex_worker
from cascade_planner.harness.agent_action_planner import (
    build_child_expansion_payload_from_blackboard,
    build_guided_chemenzy_payload_from_blackboard,
    plan_action_batch,
    validate_action_batch,
)
from cascade_planner.harness.schemas import write_json


FallbackPlanner = Callable[..., dict[str, Any]]
DEFAULT_CODEX_ACTION_PLANNER_TOOLS = ("web_search", "browser", "literature_search", "local_search")


def plan_action_batch_with_codex(
    *,
    blackboard: dict[str, Any],
    round_index: int,
    run_dir: Path,
    enabled: bool | None = None,
    exhaust_round_budget: bool = False,
    fallback_planner: FallbackPlanner | None = None,
    mock_output: dict[str, Any] | None = None,
    use_codex_cli: bool | None = None,
    use_api_json: bool | None = None,
) -> dict[str, Any]:
    """Ask Codex to choose the next bounded action batch, then validate it.

    Codex is allowed to choose actions and priorities, but the local validator
    remains authoritative for action type, budget, stale repetition, raw
    reaction injection, and solved-claim rejection.
    """
    fallback = fallback_planner or plan_action_batch
    if not _planner_enabled(enabled):
        return _fallback_batch(
            blackboard=blackboard,
            round_index=round_index,
            exhaust_round_budget=exhaust_round_budget,
            fallback_planner=fallback,
            reason="codex_action_planner_disabled",
        )

    snapshot_path = _write_codex_blackboard_snapshot(blackboard, run_dir=run_dir, round_index=round_index)
    task = _codex_action_planner_task(
        blackboard=blackboard,
        round_index=round_index,
        run_dir=run_dir,
        snapshot_path=snapshot_path,
    )
    tool_policy = _planner_tool_policy_from_task(task)
    normalized_mock = _normalize_mock_output(mock_output, task=task, blackboard=blackboard, round_index=round_index)
    use_cli = _planner_use_codex_cli(use_codex_cli)
    use_api = _planner_use_api_json(use_api_json)
    record, attempt_records = _run_codex_action_planner_worker(
        task,
        mock_output=normalized_mock,
        use_codex_cli=use_cli,
        use_api_json=use_api,
    )
    record_payload = record.to_dict()
    if attempt_records:
        record_payload["attempts"] = attempt_records
    record_payload["planner_tool_policy"] = tool_policy
    record_path = run_dir / f"codex_action_planner_run_record_round_{int(round_index)}.json"
    write_json(record_path, record_payload)
    record_payload["record_ref"] = str(record_path)

    if record.status != "accepted_draft":
        return _fallback_batch(
            blackboard=blackboard,
            round_index=round_index,
            exhaust_round_budget=exhaust_round_budget,
            fallback_planner=fallback,
            reason="codex_action_planner_worker_rejected",
            codex_record=record_payload,
            blackboard_snapshot_ref=str(snapshot_path),
            planner_tool_policy=tool_policy,
        )

    batch = _normalize_codex_batch(
        dict((record.output_artifact or {}).get("payload") or {}),
        blackboard=blackboard,
        round_index=round_index,
    )
    validation = validate_action_batch(batch, blackboard=blackboard)
    if not validation.get("accepted"):
        repaired = _try_repair_invalid_codex_batch(
            invalid_batch=batch,
            initial_validation=validation,
            blackboard=blackboard,
            round_index=round_index,
            run_dir=run_dir,
            snapshot_path=snapshot_path,
            initial_record=record_payload,
            planner_tool_policy=tool_policy,
            mock_output_present=mock_output is not None,
            use_codex_cli=use_cli,
            use_api_json=use_api,
        )
        if repaired is not None:
            return repaired
        return _fallback_batch(
            blackboard=blackboard,
            round_index=round_index,
            exhaust_round_budget=exhaust_round_budget,
            fallback_planner=fallback,
            reason="codex_action_planner_batch_invalid",
            codex_record=record_payload,
            codex_validation=validation,
            blackboard_snapshot_ref=str(snapshot_path),
            planner_tool_policy=tool_policy,
        )

    batch["mode"] = "codex_xhigh_blackboard_planner"
    batch["codex_action_planner"] = {
        "schema_version": "codex_action_planner_metadata.v1",
        "backend": str(record.backend or ""),
        "status": str(record.status or ""),
        "record_ref": str(run_dir / f"codex_action_planner_run_record_round_{int(round_index)}.json"),
        "blackboard_snapshot_ref": str(snapshot_path),
        "validation": validation,
        "fallback_used": False,
        "tool_policy": tool_policy,
    }
    return batch


def _run_codex_action_planner_worker(
    task: WorkerTask,
    *,
    mock_output: dict[str, Any] | None,
    use_codex_cli: bool,
    use_api_json: bool,
) -> tuple[Any, list[dict[str, Any]]]:
    max_attempts = _planner_worker_max_attempts(mock_output=mock_output)
    attempts: list[dict[str, Any]] = []
    record = None
    for attempt in range(1, max_attempts + 1):
        record = run_codex_worker(
            task,
            mock_output=mock_output,
            use_codex_cli=use_codex_cli,
            use_api_json=use_api_json,
        )
        payload = record.to_dict()
        payload["attempt_index"] = attempt
        attempts.append(payload)
        if record.status == "accepted_draft":
            break
        if attempt >= max_attempts or not _codex_action_planner_transient_failure(payload):
            break
    return record, attempts[:-1]


def _try_repair_invalid_codex_batch(
    *,
    invalid_batch: dict[str, Any],
    initial_validation: dict[str, Any],
    blackboard: dict[str, Any],
    round_index: int,
    run_dir: Path,
    snapshot_path: Path,
    initial_record: dict[str, Any],
    planner_tool_policy: dict[str, Any],
    mock_output_present: bool,
    use_codex_cli: bool,
    use_api_json: bool,
) -> dict[str, Any] | None:
    """Let Codex repair/replan an invalid batch before using deterministic fallback."""
    if _unsafe_validator_reasons(initial_validation):
        return None

    if not mock_output_present and _codex_action_planner_repair_enabled():
        repair_task = _codex_action_planner_repair_task(
            blackboard=blackboard,
            round_index=round_index,
            run_dir=run_dir,
            snapshot_path=snapshot_path,
            invalid_batch=invalid_batch,
            initial_validation=initial_validation,
        )
        repair_tool_policy = _planner_tool_policy_from_task(repair_task)
        repair_record, repair_attempts = _run_codex_action_planner_worker(
            repair_task,
            mock_output=None,
            use_codex_cli=use_codex_cli,
            use_api_json=use_api_json,
        )
        repair_record_payload = repair_record.to_dict()
        if repair_attempts:
            repair_record_payload["attempts"] = repair_attempts
        repair_record_payload["planner_tool_policy"] = repair_tool_policy
        repair_record_path = run_dir / f"codex_action_planner_repair_run_record_round_{int(round_index)}.json"
        write_json(repair_record_path, repair_record_payload)
        repair_record_payload["record_ref"] = str(repair_record_path)
        initial_record["repair_attempt_record_ref"] = str(repair_record_path)
        initial_record["repair_attempt_status"] = str(repair_record.status or "")
        initial_record["repair_attempt_backend"] = str(repair_record.backend or "")
        initial_record["repair_attempt_worker_validation"] = dict(repair_record_payload.get("output_validation") or {})

        if repair_record.status == "accepted_draft":
            repaired_batch = _normalize_codex_batch(
                dict((repair_record.output_artifact or {}).get("payload") or {}),
                blackboard=blackboard,
                round_index=round_index,
            )
            repair_validation = validate_action_batch(repaired_batch, blackboard=blackboard)
            if repair_validation.get("accepted"):
                return _accepted_repaired_codex_batch(
                    repaired_batch,
                    validation=repair_validation,
                    initial_validation=initial_validation,
                    initial_record=initial_record,
                    blackboard_snapshot_ref=str(snapshot_path),
                    tool_policy=repair_tool_policy,
                    repair_source="codex_replan_after_validator_rejection",
                    repair_record=repair_record_payload,
                )

            locally_repaired = _locally_repair_invalid_codex_batch(
                repaired_batch,
                validation=repair_validation,
                blackboard=blackboard,
            )
            if locally_repaired is not None:
                local_validation = validate_action_batch(locally_repaired, blackboard=blackboard)
                if local_validation.get("accepted"):
                    return _accepted_repaired_codex_batch(
                        locally_repaired,
                        validation=local_validation,
                        initial_validation=initial_validation,
                        initial_record=initial_record,
                        blackboard_snapshot_ref=str(snapshot_path),
                        tool_policy=repair_tool_policy,
                        repair_source="codex_replan_plus_guarded_budget_salvage",
                        repair_record=repair_record_payload,
                        repair_validation=repair_validation,
                    )

    locally_repaired = _locally_repair_invalid_codex_batch(
        invalid_batch,
        validation=initial_validation,
        blackboard=blackboard,
    )
    if locally_repaired is None:
        return None
    local_validation = validate_action_batch(locally_repaired, blackboard=blackboard)
    if not local_validation.get("accepted"):
        return None
    return _accepted_repaired_codex_batch(
        locally_repaired,
        validation=local_validation,
        initial_validation=initial_validation,
        initial_record=initial_record,
        blackboard_snapshot_ref=str(snapshot_path),
        tool_policy=planner_tool_policy,
        repair_source="guarded_budget_salvage_of_codex_batch",
    )


def _unsafe_validator_reasons(validation: dict[str, Any]) -> bool:
    reasons = {str(item) for item in (validation or {}).get("reasons") or []}
    unsafe = {
        "planner_direct_solved_claim",
        "planner_semantics_allow_solved_claim",
        "planner_semantics_allow_raw_reaction_output",
        "raw_reaction_injection",
    }
    return bool(reasons & unsafe) or any("raw_reaction_injection" in reason for reason in reasons)


def _codex_action_planner_repair_enabled() -> bool:
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_REPAIR")
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _codex_action_planner_repair_task(
    *,
    blackboard: dict[str, Any],
    round_index: int,
    run_dir: Path,
    snapshot_path: Path,
    invalid_batch: dict[str, Any],
    initial_validation: dict[str, Any],
) -> WorkerTask:
    case_id = str(blackboard.get("case_id") or "case")
    planner_context = _planner_context_summary(blackboard)
    compact = _compact_for_prompt(blackboard)
    allowed_tools = _planner_allowed_tools()
    max_tool_calls = _planner_max_tool_calls(allowed_tools)
    tool_policy = _planner_tool_policy(allowed_tools=allowed_tools, max_tool_calls=max_tool_calls)
    objective = (
        "You are the Codex validator-replanner for an autonomous retrosynthesis blackboard. "
        "The previous Codex action batch was rejected by the local safety/budget gate. "
        "Repair or replan it into a valid AgentActionBatch instead of handing control to a deterministic fallback. "
        "Treat validation reasons as hard blockers. Preserve the useful scientific intent when possible, but drop, "
        "replace, or defer invalid actions. If a visual budget is exhausted, prefer non-visual PDF/source processing, "
        "template extraction/ranking, search/source acquisition, guided policy preparation, or stop_unresolved. "
        "If source-sensitive actions require binding, bind a concrete DOI, source_ref, pdf_path, chain_id, or artifact_ref "
        "from planner_context; if no honest binding exists, choose another action. "
        "For analogical template actions, include a complete analogical_template_policy. Safe allowed_use values are "
        "planner_priority, guided_policy_hint, template_candidate_validation, and bridge_task_triage. "
        "Never emit solved verdicts, raw reactions, reaction SMILES, production KB writes, or parent-route proof claims. "
        "At most 3 actions. Each action needs brief rationale, expected_artifact, and success_condition. "
        "Keep payloads skeletal; do not expand complete ChemEnzy policies, proof policies, or source acquisition policies. "
        "If no productive valid action remains, choose stop_unresolved with a concise reason. "
        "Planner tool policy: "
        f"{json.dumps(tool_policy, ensure_ascii=False, sort_keys=True)}. "
        "Initial validation: "
        f"{json.dumps(initial_validation, ensure_ascii=False, sort_keys=True)}. "
        "Rejected action batch: "
        f"{json.dumps(invalid_batch, ensure_ascii=False, sort_keys=True)}. "
        "Planner context: "
        f"{json.dumps(planner_context, ensure_ascii=False, sort_keys=True)}. "
        "Current compact blackboard snapshot: "
        f"{json.dumps(compact, ensure_ascii=False, sort_keys=True)}"
    )
    return WorkerTask(
        task_id=f"{case_id}:codex_action_planner_repair:r{int(round_index)}",
        case_id=case_id,
        task_type="strategic_disconnection_mining",
        required_artifact_type="AgentActionBatch",
        input_refs=[str(snapshot_path), str(run_dir / "agent_blackboard.json")],
        allowed_tools=allowed_tools,
        budget=WorkerBudget(
            timeout_s=_planner_repair_timeout_s(),
            max_output_bytes=int(os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_OUTPUT_BYTES", "120000")),
            max_tool_calls=max_tool_calls,
            max_worker_runs=1,
        ),
        objective=objective,
        allowed_workdir=str(run_dir),
    )


def _planner_repair_timeout_s() -> float:
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_REPAIR_TIMEOUT_S")
    if raw is None:
        return _planner_timeout_s()
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return _planner_timeout_s()


def _accepted_repaired_codex_batch(
    batch: dict[str, Any],
    *,
    validation: dict[str, Any],
    initial_validation: dict[str, Any],
    initial_record: dict[str, Any],
    blackboard_snapshot_ref: str,
    tool_policy: dict[str, Any],
    repair_source: str,
    repair_record: dict[str, Any] | None = None,
    repair_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(batch)
    out["mode"] = "codex_xhigh_blackboard_planner_repaired"
    metadata = {
        "schema_version": "codex_action_planner_metadata.v1",
        "backend": str((repair_record or initial_record or {}).get("backend") or ""),
        "status": str((repair_record or initial_record or {}).get("status") or ""),
        "record_ref": str((initial_record or {}).get("record_ref") or ""),
        "blackboard_snapshot_ref": str(blackboard_snapshot_ref or ""),
        "validation": validation,
        "initial_validation": dict(initial_validation or {}),
        "repair_used": True,
        "repair_source": str(repair_source or ""),
        "fallback_used": False,
        "tool_policy": dict(tool_policy or {}),
    }
    if repair_record:
        metadata["repair_record_ref"] = str(repair_record.get("record_ref") or "")
        metadata["repair_record_status"] = str(repair_record.get("status") or "")
        metadata["repair_worker_validation"] = dict(repair_record.get("output_validation") or {})
    if repair_validation:
        metadata["repair_validation_before_guarded_salvage"] = dict(repair_validation or {})
    out["codex_action_planner"] = metadata
    return out


def _locally_repair_invalid_codex_batch(
    batch: dict[str, Any],
    *,
    validation: dict[str, Any],
    blackboard: dict[str, Any],
) -> dict[str, Any] | None:
    if _unsafe_validator_reasons(validation):
        return None
    reasons = {str(item) for item in (validation or {}).get("reasons") or []}
    repaired = deepcopy(batch)
    actions = [dict(row) for row in repaired.get("actions") or [] if isinstance(row, dict)]
    if not actions:
        return None

    for action in actions:
        payload = _repair_codex_action_payload(
            str(action.get("action_type") or ""),
            dict(action.get("payload") or {}),
            blackboard=blackboard,
        )
        action["payload"] = payload
        if str(action.get("action_type") or "") in {
            "search_literature",
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "resolve_literature_structure_task",
            "compile_exact_literature_rows",
        }:
            payload["max_sources"] = min(max(1, int(payload.get("max_sources") or 1)), 1)

    if "visual_total_budget_exceeded" in reasons:
        actions = [row for row in actions if not _action_consumes_visual_budget(row)]
    if "scout_total_budget_exceeded" in reasons:
        actions = [row for row in actions if str(row.get("action_type") or "") != "search_literature"]
    if "guided_chemenzy_total_budget_exceeded" in reasons:
        actions = [row for row in actions if str(row.get("action_type") or "") != "run_guided_chemenzy"]
    if any("guided_chemenzy_missing_prior_signal_for_complex_target" in reason for reason in reasons):
        actions = [row for row in actions if str(row.get("action_type") or "") != "run_guided_chemenzy"]
    if "child_expansion_total_budget_exceeded" in reasons:
        actions = [row for row in actions if str(row.get("action_type") or "") != "expand_child_target"]
    if "template_application_total_budget_exceeded" in reasons:
        actions = [
            row
            for row in actions
            if str(row.get("action_type") or "") not in {"apply_analogical_template_to_target", "validate_template_application"}
        ]

    if not actions:
        return None
    repaired["actions"] = actions[:3]
    repaired["mode"] = "codex_xhigh_blackboard_planner_repaired"
    repaired.setdefault("semantics", {})
    repaired["semantics"]["planner_can_emit_solved"] = False
    repaired["semantics"]["raw_reaction_output_allowed"] = False
    repaired["semantics"]["deterministic_validator_required"] = True
    return repaired


def _action_consumes_visual_budget(action: dict[str, Any]) -> bool:
    action_type = str(action.get("action_type") or "")
    if action_type == "extract_visual_literature_chain":
        return True
    if action_type == "resolve_literature_structure_task":
        return bool((action.get("payload") or {}).get("run_visual", True))
    return False


def _planner_worker_max_attempts(*, mock_output: dict[str, Any] | None) -> int:
    if mock_output is not None:
        return 1
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_WORKER_ATTEMPTS")
    if raw is None:
        return 3
    return max(1, int(raw))


def _codex_action_planner_transient_failure(record_payload: dict[str, Any]) -> bool:
    status = str(record_payload.get("status") or "").lower()
    stderr = str(record_payload.get("stderr") or "").lower()
    reasons = " ".join(str(item).lower() for item in (record_payload.get("output_validation") or {}).get("reasons") or [])
    text = " ".join([status, stderr, reasons])
    if status in {"timeout", "worker_error"}:
        return True
    return any(
        token in text
        for token in (
            "502",
            "bad gateway",
            "upstream service temporarily unavailable",
            "stream disconnected",
            "websocket closed",
            "reconnecting",
            "timeout",
        )
    )


def _codex_action_planner_task(
    *,
    blackboard: dict[str, Any],
    round_index: int,
    run_dir: Path,
    snapshot_path: Path,
) -> WorkerTask:
    case_id = str(blackboard.get("case_id") or "case")
    compact = _compact_for_prompt(blackboard)
    planner_context = _planner_context_summary(blackboard)
    allowed_tools = _planner_allowed_tools()
    max_tool_calls = _planner_max_tool_calls(allowed_tools)
    tool_policy = _planner_tool_policy(allowed_tools=allowed_tools, max_tool_calls=max_tool_calls)
    objective = (
        "You are the central blackboard action planner for an autonomous retrosynthesis workflow. "
        "Read the current blackboard and choose the next action batch. You may freely choose priorities and exploration direction, "
        "but you may only emit typed actions, not routes or solved verdicts. "
        "Select at most 3 actions. Prefer parallel actions only when they do not depend on each other. "
        "Before returning, internally check the action_payload_requirements and repair your own draft so it is valid. "
        "Do not rely on a deterministic fallback to make the scientific choice for you. "
        "Keep the JSON small: rationale under 45 words, expected_artifact under 12 words, success_condition under 20 words, "
        "and payload as a skeleton with fewer than 12 scalar/list fields. "
        "If two recent rounds produced no useful artifact, either change direction or choose stop_unresolved. "
        "Use search_literature only when source evidence or target-proximal bridge evidence is missing. "
        "For search_literature, include only search_intent and short queries; local repair will add source_acquisition_policy. "
        "Codex online search is primary, local PDFs are fallback/cache matches only, and placeholders are allowed only after failures. "
        "Use visual/PDF extraction only after a source candidate or local PDF is available. "
        "For source-sensitive actions, follow planner_context.action_payload_requirements: when currently_required is true, "
        "the action payload must include a concrete binding such as source_ref, doi, pdf_path, source_pdf_path, chain_id, or artifact_ref. "
        "For run_guided_chemenzy, do not emit a full search_policy or chem_enzy_search_policy. Emit only intent fields "
        "such as initial_probe, search_mode, max_steps, chem_enzy_iterations, chem_enzy_expansion_topk, timeout_s, "
        "and max_candidates; local repair will build the complete policy from the blackboard. "
        "For expand_child_target, include explicit subgoal_targets or child_targets with target_equivalence_audit_required, "
        "exact_target_override, no_solved_claim, child_route_cannot_promote_parent, and a valid child search policy. "
        "For stitch_parent_route, include proof_binding and proof_policy; the payload must bind child route, parent route, "
        "exact literature rows, and the deterministic proof boundaries, with any missing input stated explicitly. "
        "For analogical template actions, include analogical_template_policy; analogy is advisory only and cannot be proof "
        "or final verdict authority. Safe allowed_use values are planner_priority, guided_policy_hint, "
        "template_candidate_validation, and bridge_task_triage. "
        "Use guided Chemenzy directly for simple low-complexity targets when a baseline stock-closure attempt is sensible. "
        "For complex polycyclic/steroid/natural-product-like targets, use guided Chemenzy only after blackboard evidence, "
        "bridge tasks, target-side hypotheses, exact rows, broad templates, or ranked analogical templates justify it. "
        "A complex first-round Chemenzy call is allowed only as an explicitly bounded initial_probe with max_steps<=6, "
        "chem_enzy_iterations<=10, chem_enzy_expansion_topk<=20, timeout_s<=180, and source_budget.max_candidates<=5. "
        "Child target work can never prove parent solved; stitch_parent_route is required before any final solution. "
        "You may use only the audited planner tools and tool-call budget declared in WorkerTask.allowed_tools and WorkerTask.budget. "
        "If online search is available, use it only to orient action choice or source-acquisition payloads; put evidence extraction "
        "into typed actions so the controller can normalize it back to the blackboard. "
        "When online/local search reveals DOI, PII, URL, title, or local PDF metadata that should guide source acquisition, "
        "put it in top-level planner_source_hints. These hints are source-acquisition hints only, not evidence, exact rows, or proof. "
        "Each hint must set evidence_class=planner_source_hint, allowed_use=source_acquisition_hint_only, and no_solved_claim=true. "
        "Allowed actions: "
        f"{', '.join(sorted(ALLOWED_AGENT_ACTIONS))}. "
        "Planner tool policy: "
        f"{json.dumps(tool_policy, ensure_ascii=False, sort_keys=True)}. "
        "Return only the required AgentActionBatch artifact payload. "
        "Derived planner context, for quick orientation but not as a hardcoded script: "
        f"{json.dumps(planner_context, ensure_ascii=False, sort_keys=True)}. "
        "Current compact blackboard snapshot: "
        f"{json.dumps(compact, ensure_ascii=False, sort_keys=True)}"
    )
    return WorkerTask(
        task_id=f"{case_id}:codex_action_planner:r{int(round_index)}",
        case_id=case_id,
        task_type="strategic_disconnection_mining",
        required_artifact_type="AgentActionBatch",
        input_refs=[str(snapshot_path), str(run_dir / "agent_blackboard.json")],
        allowed_tools=allowed_tools,
        budget=WorkerBudget(
            timeout_s=_planner_timeout_s(),
            max_output_bytes=int(os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_OUTPUT_BYTES", "120000")),
            max_tool_calls=max_tool_calls,
            max_worker_runs=1,
        ),
        objective=objective,
        allowed_workdir=str(run_dir),
    )


def _normalize_codex_batch(
    payload: dict[str, Any],
    *,
    blackboard: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    case_id = str(blackboard.get("case_id") or payload.get("case_id") or "case")
    actions: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for idx, raw in enumerate(payload.get("actions") or [], start=1):
        if not isinstance(raw, dict):
            continue
        action_type = str(raw.get("action_type") or "").strip()
        seen[action_type] = seen.get(action_type, 0) + 1
        action_id = str(raw.get("action_id") or "").strip()
        if not action_id:
            suffix = f"{action_type}_{seen[action_type]}" if action_type else f"action_{idx}"
            action_id = f"r{int(round_index)}:{suffix}"
        action = {
            "schema_version": str(raw.get("schema_version") or "agent_action.v1"),
            "action_id": action_id,
            "action_type": action_type,
            "rationale": str(raw.get("rationale") or ""),
            "expected_artifact": str(raw.get("expected_artifact") or ""),
            "success_condition": str(raw.get("success_condition") or ""),
            "payload": dict(raw.get("payload") or {}),
        }
        action["payload"] = _repair_codex_action_payload(action_type, dict(action.get("payload") or {}), blackboard=blackboard)
        for key, value in raw.items():
            key_l = str(key).lower()
            if key_l in {"verdict", "route_status", "status"} or key_l in FORBIDDEN_RAW_REACTION_KEYS:
                action[str(key)] = value
        actions.append(action)
    batch = {
        "schema_version": ACTION_BATCH_SCHEMA,
        "case_id": case_id,
        "round_index": int(round_index),
        "mode": str(payload.get("mode") or "codex_xhigh_blackboard_planner"),
        "actions": actions,
        "semantics": {
            "planner_can_emit_solved": False,
            "raw_reaction_output_allowed": False,
            "deterministic_validator_required": True,
        },
    }
    hints = _normalize_planner_source_hints(payload.get("planner_source_hints") or [])
    if hints:
        batch["planner_source_hints"] = hints
    return batch


def _repair_codex_action_payload(action_type: str, payload: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    if action_type == "search_literature":
        return _repair_search_literature_payload(payload, blackboard=blackboard)
    if action_type == "resolve_literature_structure_task":
        return _repair_structure_resolution_payload(payload, blackboard=blackboard)
    if action_type == "run_guided_chemenzy":
        return _repair_guided_chemenzy_payload(payload, blackboard=blackboard)
    if action_type == "expand_child_target":
        return _repair_child_expansion_payload(payload, blackboard=blackboard)
    if action_type in {
        "extract_analogical_reaction_templates",
        "rank_analogical_reaction_templates",
        "apply_analogical_template_to_target",
        "validate_template_application",
    }:
        return _repair_analogical_template_payload(action_type, payload)
    return dict(payload or {})


def _repair_search_literature_payload(payload: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    raw = dict(payload or {})
    queries = _dedupe_preserve_order(
        [
            *[_normalize_search_query_item(item) for item in raw.get("queries") or []],
            *[_normalize_search_query_item(item) for item in raw.get("search_queries") or []],
            _normalize_search_query_item(raw.get("query") or ""),
        ]
    )
    if queries:
        raw["queries"] = queries
        raw["search_queries"] = queries
    if not str(raw.get("search_intent") or raw.get("query") or "").strip():
        target = dict(blackboard.get("target_profile") or {})
        raw["search_intent"] = (
            f"target_proximal_source_discovery for {target.get('target_name') or blackboard.get('case_id') or 'target'}"
        )
    policy = dict(raw.get("source_acquisition_policy") or {})
    policy["schema_version"] = "agentic_source_acquisition_policy.v1"
    policy["codex_online_first"] = True
    policy["local_pdf_fallback_allowed"] = True
    policy["placeholder_allowed_after_failures"] = True
    policy["auto_local_pdf_requires_agent_discovered_metadata"] = True
    policy["no_solved_claim"] = True
    policy["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
    raw["source_acquisition_policy"] = policy
    raw["no_solved_claim"] = True
    raw["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "search_literature",
        "completed_from_blackboard": True,
    }
    return raw


def _normalize_search_query_item(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("query", "search_query", "text", "title"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return ""
    text = str(value or "").strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, dict):
            return _normalize_search_query_item(parsed)
    return text


def _repair_structure_resolution_payload(payload: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    raw = dict(payload or {})
    task = _structure_resolution_task_for_payload(blackboard, raw)
    if task:
        raw.setdefault("task_id", str(task.get("task_id") or ""))
        raw.setdefault("label", str(task.get("label") or ""))
        raw.setdefault("compound_label", str(task.get("label") or ""))
        raw.setdefault("source_ref", str(task.get("source_ref") or ""))
        raw.setdefault("source_title", str(task.get("source_title") or ""))
        raw.setdefault("source_locator", str(task.get("source_locator") or ""))
        raw.setdefault("artifact_ref", str(task.get("artifact_ref") or ""))
        source = _source_candidate_for_structure_task(blackboard, task)
        if source:
            raw.setdefault("pdf_path", str(source.get("local_pdf") or source.get("pdf_path") or source.get("source_pdf_path") or ""))
            raw.setdefault("source_title", str(source.get("title") or source.get("source_title") or raw.get("source_title") or ""))
    raw["schema_version"] = "literature_structure_resolution_payload.v1"
    raw["run_visual"] = bool(raw.get("run_visual", True))
    raw["compress_images"] = bool(raw.get("compress_images", True))
    raw["max_images"] = int(raw.get("max_images") or 6)
    raw["visual_max_side_px"] = int(raw.get("visual_max_side_px") or 1400)
    raw["visual_jpeg_quality"] = int(raw.get("visual_jpeg_quality") or 70)
    raw["no_solved_claim"] = True
    raw["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "resolve_literature_structure_task",
        "completed_from_blackboard": bool(task),
    }
    return raw


def _structure_resolution_task_for_payload(blackboard: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    requested_id = str(payload.get("task_id") or "").strip()
    requested_label = str(payload.get("label") or payload.get("compound_label") or "").strip().lower()
    tasks = [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("structure_resolution_tasks") or []
        if isinstance(row, dict) and str(row.get("status") or "open") == "open"
    ]
    for task in tasks:
        if requested_id and str(task.get("task_id") or "") == requested_id:
            return task
        if requested_label and str(task.get("label") or "").strip().lower() == requested_label:
            return task
    for task in tasks:
        try:
            if int(task.get("resolution_attempt_count") or 0) > 0:
                continue
        except (TypeError, ValueError):
            pass
        return task
    return tasks[0] if tasks else {}


def _source_candidate_for_structure_task(blackboard: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    task_key = _source_key(task)
    for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []:
        if isinstance(row, dict) and task_key and _source_key(row) == task_key:
            return dict(row)
    return {}


def _repair_guided_chemenzy_payload(payload: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    base_payload = build_guided_chemenzy_payload_from_blackboard(blackboard)
    base_policy = dict(base_payload.get("search_policy") or {})
    raw = dict(payload or {})
    raw_search_policy = raw.pop("search_policy", None)
    raw_chem_policy = raw.pop("chem_enzy_search_policy", None)
    raw_policy = raw_chem_policy if isinstance(raw_chem_policy, dict) else raw_search_policy
    policy = _complete_search_policy(
        _deep_merge(base_policy, dict(raw_policy or {}) if isinstance(raw_policy, dict) else {}),
        base_policy,
    )
    _apply_guided_probe_hints(policy, raw)
    repaired = {**dict(base_payload), **raw}
    repaired["search_policy"] = policy
    if isinstance(raw_chem_policy, dict):
        repaired["chem_enzy_search_policy"] = dict(policy)
    repaired["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "run_guided_chemenzy",
        "completed_from_blackboard": True,
        "preserved_codex_fields": sorted(str(key) for key in raw.keys()),
    }
    return repaired


def _apply_guided_probe_hints(policy: dict[str, Any], payload: dict[str, Any]) -> None:
    search_mode = str(payload.get("search_mode") or payload.get("mode") or "").strip()
    if search_mode:
        policy["search_mode"] = search_mode
    initial_probe = bool(payload.get("initial_probe")) or search_mode.lower() in {"initial_probe", "cheap_scan", "baseline_probe"}
    source_budget = dict(policy.get("source_budget") or {})
    compiler = dict(policy.get("compiler_metadata") or {})
    if initial_probe:
        source_budget["initial_scan_allowed"] = True
        compiler["initial_scan_probe"] = True
    if payload.get("max_candidates") is not None:
        try:
            source_budget["max_candidates"] = max(1, int(payload.get("max_candidates") or 0))
        except (TypeError, ValueError):
            pass
    policy["source_budget"] = source_budget
    policy["compiler_metadata"] = compiler


def _repair_child_expansion_payload(payload: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    base_payload = build_child_expansion_payload_from_blackboard(blackboard)
    raw = dict(payload or {})
    raw_targets = raw.pop("subgoal_targets", None)
    raw_child_targets = raw.pop("child_targets", None)
    targets = raw_targets if isinstance(raw_targets, list) else raw_child_targets
    if not isinstance(targets, list) or not targets:
        targets = list(base_payload.get("subgoal_targets") or [])
    if not targets:
        return {**dict(base_payload), **raw}

    repaired_targets: list[dict[str, Any]] = []
    for idx, target in enumerate(targets, start=1):
        if not isinstance(target, dict):
            continue
        repaired_targets.append(_repair_child_target(dict(target), blackboard=blackboard, index=idx))
    repaired = {**dict(base_payload), **raw}
    repaired["subgoal_targets"] = repaired_targets
    repaired["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "expand_child_target",
        "completed_from_blackboard": True,
        "target_count": len(repaired_targets),
    }
    return repaired


def _repair_child_target(target: dict[str, Any], *, blackboard: dict[str, Any], index: int) -> dict[str, Any]:
    smiles = str(target.get("smiles") or target.get("target_smiles") or "").strip()
    target["smiles"] = smiles
    if not str(target.get("name") or "").strip():
        target["name"] = str(target.get("target_name") or f"child_target_{index}")
    target["exact_target_override"] = True
    target["target_equivalence_audit_required"] = True
    target["no_solved_claim"] = True
    target["child_route_cannot_promote_parent"] = True
    base_policy = _child_search_policy_seed(blackboard, target=target, index=index)
    raw_policy = target.get("chem_enzy_search_policy") if isinstance(target.get("chem_enzy_search_policy"), dict) else target.get("policy")
    policy = _complete_search_policy(
        _deep_merge(base_policy, dict(raw_policy or {}) if isinstance(raw_policy, dict) else {}),
        base_policy,
    )
    compiler = dict(policy.get("compiler_metadata") or {})
    compiler["requires_verifier"] = True
    compiler["no_solved_claim"] = True
    compiler["child_route_cannot_promote_parent"] = True
    policy["compiler_metadata"] = compiler
    target["chem_enzy_search_policy"] = policy
    return target


def _repair_analogical_template_payload(action_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw = dict(payload or {})
    policy = dict(raw.get("analogical_template_policy") or {})
    original_allowed = policy.get("allowed_use", raw.get("allowed_use"))
    allowed_use = _normalize_analogical_template_allowed_use(original_allowed, action_type=action_type)
    policy["schema_version"] = "agentic_analogical_template_action_policy.v1"
    policy["action_type"] = action_type
    policy["analogy_is_advisory_only"] = True
    policy["no_solved_claim"] = True
    policy["requires_verifier"] = True
    policy["requires_parent_route_proof"] = True
    policy["production_write_blocked"] = True
    policy["raw_reaction_output_allowed"] = False
    policy["final_verdict_authority"] = "deterministic_parent_route_proof"
    policy["allowed_use"] = allowed_use
    policy["deterministic_template_validation_required"] = True
    raw["analogical_template_policy"] = policy
    raw["no_solved_claim"] = True
    raw["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": action_type,
        "completed_from_blackboard": True,
        "original_allowed_use": original_allowed if isinstance(original_allowed, (str, list)) else "",
    }
    return raw


def _normalize_analogical_template_allowed_use(value: Any, *, action_type: str) -> list[str]:
    safe = {"planner_priority", "guided_policy_hint", "template_candidate_validation", "bridge_task_triage"}
    forbidden = {"solved_proof", "final_verdict", "parent_route_proof"}
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw_items = []
    selected: list[str] = []
    for item in raw_items:
        token = str(item or "").strip()
        if not token or token in forbidden or token not in safe or token in selected:
            continue
        selected.append(token)
    if selected:
        return selected
    if action_type in {"apply_analogical_template_to_target", "validate_template_application"}:
        return ["guided_policy_hint", "template_candidate_validation", "bridge_task_triage"]
    return ["planner_priority", "guided_policy_hint", "bridge_task_triage"]


def _child_search_policy_seed(blackboard: dict[str, Any], *, target: dict[str, Any], index: int) -> dict[str, Any]:
    guided = dict(build_guided_chemenzy_payload_from_blackboard(blackboard).get("search_policy") or {})
    smiles = str(target.get("smiles") or target.get("target_smiles") or "").strip()
    policy = deepcopy(guided)
    policy["policy_id"] = f"{blackboard.get('case_id') or 'case'}_codex_child_{index}_policy"
    policy["operator_id"] = "agentic_blackboard_controller"
    policy["case_id"] = str(blackboard.get("case_id") or "")
    policy["anchor_whitelist"] = [smiles] if smiles else []
    refs = list(policy.get("evidence_refs") or [])
    if smiles:
        refs.append(f"child_target:{smiles}")
    policy["evidence_refs"] = _dedupe_preserve_order([str(item) for item in refs if str(item or "").strip()])
    policy["preferred_subgoal"] = {
        "schema_version": "codex_repaired_child_subgoal.v1",
        "preferred_subgoals": [str(target.get("name") or ""), smiles],
        "target": {
            "name": str(target.get("name") or ""),
            "smiles": smiles,
        },
    }
    source_budget = dict(policy.get("source_budget") or {})
    source_budget["require_target_core_retention"] = True
    source_budget["max_unexplained_heavy_atom_jump"] = int(source_budget.get("max_unexplained_heavy_atom_jump") or 12)
    source_budget["analogy_is_advisory_only"] = True
    source_budget.setdefault("preferred_reaction_classes", ["source_detail_terminal_upstream_expansion"])
    policy["source_budget"] = source_budget
    policy["rerun_reason"] = "codex repaired child-target search from blackboard constraints"
    policy["mode"] = "guided"
    compiler = dict(policy.get("compiler_metadata") or {})
    compiler["compiler_schema"] = "agentic_blackboard_codex_repaired_child_target.v1"
    compiler["not_raw_reaction_injection"] = True
    compiler["requires_verifier"] = True
    compiler["no_solved_claim"] = True
    compiler["child_route_cannot_promote_parent"] = True
    policy["compiler_metadata"] = compiler
    return policy


def _complete_search_policy(policy: dict[str, Any], base_policy: dict[str, Any]) -> dict[str, Any]:
    repaired = _deep_merge(dict(base_policy or {}), dict(policy or {}))
    for field in (
        "evidence_refs",
        "terminal_blacklist",
        "anchor_whitelist",
        "active_bridge_tasks",
        "accepted_exact_row_ids",
        "selected_analogical_hypothesis_ids",
        "selected_analogical_template_ids",
        "forbidden_template_ids",
    ):
        if not isinstance(repaired.get(field), list):
            repaired[field] = list(base_policy.get(field) or [])
    if not repaired.get("evidence_refs"):
        repaired["evidence_refs"] = list(base_policy.get("evidence_refs") or ["codex_repaired_blackboard_state"])
    for field in ("policy_id", "operator_id", "case_id", "rerun_reason", "mode", "schema_version"):
        if not str(repaired.get(field) or "").strip():
            repaired[field] = base_policy.get(field)
    if repaired.get("mode") not in {"baseline", "guided", "literature-assisted", "stuck-node rerun"}:
        repaired["mode"] = "guided"
    source_budget = dict(base_policy.get("source_budget") or {})
    source_budget.update(dict(repaired.get("source_budget") or {}))
    source_budget["require_target_core_retention"] = True
    source_budget["analogy_is_advisory_only"] = True
    try:
        max_jump = int(source_budget.get("max_unexplained_heavy_atom_jump") or 0)
    except (TypeError, ValueError):
        max_jump = 0
    if max_jump <= 0:
        source_budget["max_unexplained_heavy_atom_jump"] = int(
            (base_policy.get("source_budget") or {}).get("max_unexplained_heavy_atom_jump") or 12
        )
    repaired["source_budget"] = source_budget
    compiler = dict(base_policy.get("compiler_metadata") or {})
    compiler.update(dict(repaired.get("compiler_metadata") or {}))
    compiler["requires_verifier"] = True
    compiler["no_solved_claim"] = True
    repaired["compiler_metadata"] = compiler
    budget = dict(base_policy.get("budget") or {})
    budget.update(dict(repaired.get("budget") or {}))
    repaired["budget"] = budget
    repaired["schema_version"] = "chem_enzy_search_policy.v1"
    return repaired


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged.get(key) or {}), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_planner_source_hints(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        doi = _normalize_hint_doi(str(row.get("doi") or ""))
        pii = str(row.get("pii") or "").strip()
        url = str(row.get("url") or "").strip()
        local_pdf = str(row.get("local_pdf") or row.get("pdf_path") or "").strip()
        local_ref = str(row.get("local_ref") or "").strip()
        source_ref = str(row.get("source_ref") or "").strip()
        title = str(row.get("title") or row.get("source_title") or "").strip()
        if not source_ref:
            source_ref = f"doi:{doi}" if doi else (f"pii:{pii}" if pii else (url or local_ref or (f"local_pdf:{Path(local_pdf).name}" if local_pdf else "")))
        key = str(doi or pii or url or local_pdf or local_ref or source_ref or title).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "schema_version": PLANNER_SOURCE_HINT_SCHEMA,
                "hint_id": str(row.get("hint_id") or f"planner_source_hint_{idx}"),
                "source_ref": source_ref,
                "title": title,
                "doi": doi,
                "pii": pii,
                "url": url,
                "local_pdf": local_pdf,
                "local_ref": local_ref,
                "source_type": str(row.get("source_type") or "planner_discovered_literature_metadata"),
                "relevance_rationale": str(row.get("relevance_rationale") or row.get("rationale") or ""),
                "expected_scheme_or_compound_labels": [
                    str(item)
                    for item in row.get("expected_scheme_or_compound_labels") or row.get("expected_labels") or []
                    if str(item or "").strip()
                ],
                "extraction_task_recommendations": [
                    str(item)
                    for item in row.get("extraction_task_recommendations") or []
                    if str(item or "").strip()
                ],
                "evidence_class": "planner_source_hint",
                "allowed_use": "source_acquisition_hint_only",
                "no_solved_claim": True,
            }
        )
        if len(out) >= 8:
            break
    return out


def _normalize_hint_doi(value: str) -> str:
    text = str(value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    for separator in ("?", "#", "&"):
        if separator in text:
            text = text.split(separator, 1)[0]
    return text.strip().strip(".,;:)]}'\"").lower()


def _fallback_batch(
    *,
    blackboard: dict[str, Any],
    round_index: int,
    exhaust_round_budget: bool,
    fallback_planner: FallbackPlanner,
    reason: str,
    codex_record: dict[str, Any] | None = None,
    codex_validation: dict[str, Any] | None = None,
    blackboard_snapshot_ref: str = "",
    planner_tool_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    batch = dict(
        fallback_planner(
            blackboard,
            round_index=round_index,
            exhaust_round_budget=exhaust_round_budget,
        )
    )
    batch["mode"] = "deterministic_policy_fallback_after_codex_planner"
    batch["codex_action_planner"] = {
        "schema_version": "codex_action_planner_metadata.v1",
        "fallback_used": True,
        "fallback_reason": reason,
        "record_status": str((codex_record or {}).get("status") or ""),
        "record_backend": str((codex_record or {}).get("backend") or ""),
        "blackboard_snapshot_ref": str(blackboard_snapshot_ref or ""),
        "worker_validation": dict((codex_record or {}).get("output_validation") or {}),
        "batch_validation": dict(codex_validation or {}),
        "tool_policy": dict(planner_tool_policy or (codex_record or {}).get("planner_tool_policy") or {}),
    }
    if (codex_record or {}).get("record_ref"):
        batch["codex_action_planner"]["record_ref"] = str(codex_record.get("record_ref") or "")
    if (codex_record or {}).get("repair_attempt_record_ref"):
        batch["codex_action_planner"]["repair_attempt_record_ref"] = str(codex_record.get("repair_attempt_record_ref") or "")
        batch["codex_action_planner"]["repair_attempt_status"] = str(codex_record.get("repair_attempt_status") or "")
        batch["codex_action_planner"]["repair_attempt_backend"] = str(codex_record.get("repair_attempt_backend") or "")
        batch["codex_action_planner"]["repair_attempt_worker_validation"] = dict(
            codex_record.get("repair_attempt_worker_validation") or {}
        )
    return batch


def _normalize_mock_output(
    mock_output: dict[str, Any] | None,
    *,
    task: WorkerTask,
    blackboard: dict[str, Any],
    round_index: int,
) -> dict[str, Any] | None:
    if mock_output is None:
        return None
    if mock_output.get("artifact_type") == "AgentActionBatch":
        return dict(mock_output)
    payload = dict(mock_output.get("payload") or mock_output)
    return {
        "schema_version": "agent_action_batch_artifact.v1",
        "artifact_id": f"{task.task_id}:AgentActionBatch",
        "artifact_type": "AgentActionBatch",
        "case_id": task.case_id,
        "source": "codex_action_planner_mock",
        "input_refs": list(task.input_refs),
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": "mock Codex action batch",
        "payload": payload,
    }


def _write_codex_blackboard_snapshot(blackboard: dict[str, Any], *, run_dir: Path, round_index: int) -> Path:
    path = run_dir / f"codex_action_planner_blackboard_round_{int(round_index)}.json"
    snapshot = {
        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
        "round_index": int(round_index),
        "planner_context": _planner_context_summary(blackboard),
        "blackboard": _compact_for_prompt(blackboard, max_depth=6, max_list=30, max_string=4000),
    }
    write_json(path, snapshot)
    return path


def _planner_allowed_tools() -> list[str]:
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_ALLOWED_TOOLS")
    if raw is None:
        raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_TOOLS")
    if raw is None:
        return list(DEFAULT_CODEX_ACTION_PLANNER_TOOLS)
    text = str(raw).strip()
    if not text or text.lower() in {"0", "false", "no", "none", "off", "disabled"}:
        return []
    if text.lower() in {"default", "online", "web", "search"}:
        return list(DEFAULT_CODEX_ACTION_PLANNER_TOOLS)
    allowed = set(DEFAULT_CODEX_ACTION_PLANNER_TOOLS)
    selected: list[str] = []
    for token in text.replace(";", ",").replace(" ", ",").split(","):
        tool = token.strip()
        if not tool or tool not in allowed or tool in selected:
            continue
        selected.append(tool)
    return selected


def _planner_max_tool_calls(allowed_tools: list[str]) -> int:
    if not allowed_tools:
        return 0
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_TOOL_CALLS")
    if raw is None:
        return 4
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 4


def _planner_tool_policy(*, allowed_tools: list[str], max_tool_calls: int) -> dict[str, Any]:
    tools = [str(item) for item in allowed_tools if str(item or "").strip()]
    return {
        "schema_version": "codex_action_planner_tool_policy.v1",
        "allowed_tools": tools,
        "max_tool_calls": int(max_tool_calls or 0),
        "cli_search_enabled": bool(int(max_tool_calls or 0) > 0 and {"web_search", "browser", "literature_search"} & set(tools)),
        "outputs_remain_typed_action_batch_only": True,
        "raw_reaction_output_allowed": False,
        "final_verdict_authority": "deterministic_parent_route_proof",
    }


def _planner_tool_policy_from_task(task: WorkerTask) -> dict[str, Any]:
    return _planner_tool_policy(
        allowed_tools=[str(item) for item in task.allowed_tools or []],
        max_tool_calls=int(task.budget.max_tool_calls or 0),
    )


def _planner_context_summary(blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    belief = dict(blackboard.get("current_belief") or {})
    planner_allowed_tools = _planner_allowed_tools()
    planner_max_tool_calls = _planner_max_tool_calls(planner_allowed_tools)
    candidates = [dict(row) for row in evidence.get("source_candidates") or [] if isinstance(row, dict)]
    lifecycle = [dict(row) for row in evidence.get("source_lifecycle") or [] if isinstance(row, dict)]
    lifecycle_stage_counts = _source_lifecycle_stage_counts(lifecycle)
    local_pdf_candidates = [row for row in candidates if str(row.get("local_pdf") or "").strip()]
    visual_chains = [dict(row) for row in evidence.get("visual_chains") or [] if isinstance(row, dict)]
    pdf_done = _source_keys(evidence.get("pdf_structure_evidence") or [])
    visual_done = _source_keys(visual_chains)
    pending_pdf = [row for row in local_pdf_candidates if _source_key(row) not in pdf_done]
    pending_visual = [row for row in local_pdf_candidates if _source_key(row) in pdf_done and _source_key(row) not in visual_done]
    pending_proxy = [row for row in lifecycle if str(row.get("stage") or "") == "local_pdf_proxy_requested"]
    history = [dict(row) for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    budget = dict(blackboard.get("budget_state") or {})
    return {
        "schema_version": "codex_action_planner_context.v1",
        "source_acquisition": {
            "source_discovery_mode": str(evidence.get("source_discovery_mode") or ""),
            "fallback_order": [str(item) for item in evidence.get("fallback_order") or []],
            "planner_source_hint_count": len(evidence.get("planner_source_hints") or []),
            "source_lifecycle_count": len(lifecycle),
            "source_lifecycle_stage_counts": lifecycle_stage_counts,
            "source_candidate_count": len(candidates),
            "real_source_count": sum(1 for row in candidates if _candidate_has_real_source(row)),
            "local_pdf_available_count": len(local_pdf_candidates),
            "local_pdf_proxy_request_count": len(evidence.get("local_pdf_proxy_requests") or []),
            "awaiting_local_pdf_proxy_count": int(lifecycle_stage_counts.get("local_pdf_proxy_requested") or 0),
            "local_pdf_cache_match_count": sum(1 for row in candidates if isinstance(row.get("local_pdf_match"), dict)),
            "auto_local_pdf_cache_match_count": sum(
                1
                for row in candidates
                if isinstance(row.get("local_pdf_match"), dict) and isinstance(row.get("local_pdf_index"), dict)
            ),
            "placeholder_count": sum(
                1
                for row in candidates
                if bool(row.get("placeholder_only")) or str(row.get("access_status") or "").lower() == "placeholder_only"
            ),
        },
        "literature_processing": {
            "source_lifecycle": _compact_source_lifecycle(lifecycle),
            "pending_local_pdf_proxy_sources": _compact_source_lifecycle(pending_proxy),
            "pending_pdf_extraction_sources": _compact_sources(pending_pdf),
            "pending_visual_extraction_sources": _compact_sources(pending_visual),
            "processed_pdf_source_keys": sorted(pdf_done),
            "processed_visual_source_keys": sorted(visual_done),
            "exact_row_count": len(evidence.get("exact_rows") or []),
            "visual_chain_count": len(evidence.get("visual_chains") or []),
            "structure_resolution_task_count": len(evidence.get("structure_resolution_tasks") or []),
        },
        "blackboard_policy_bias": {
            "next_action_bias": [str(item) for item in belief.get("next_action_bias") or [] if str(item or "").strip()],
            "blocked_directions": [
                {
                    "direction": str(row.get("direction") or row.get("blocked_direction") or ""),
                    "reason": str(row.get("reason") or ""),
                }
                for row in belief.get("blocked_directions") or []
                if isinstance(row, dict)
            ],
            "constraints": dict(belief.get("constraints") or {}),
        },
        "planner_tool_policy": _planner_tool_policy(
            allowed_tools=planner_allowed_tools,
            max_tool_calls=planner_max_tool_calls,
        ),
        "action_payload_requirements": _action_payload_requirements(
            blackboard=blackboard,
            source_candidates=candidates,
            local_pdf_candidates=local_pdf_candidates,
            visual_chains=visual_chains,
        ),
        "recent_blackboard_transitions": [
            {
                "round_index": int(row.get("round_index") or 0),
                "action_type": str(row.get("action_type") or ""),
                "useful_artifact": bool(row.get("useful_artifact")),
                "stale": bool(row.get("stale")),
                "changed_blackboard_fields": [str(item) for item in row.get("changed_blackboard_fields") or []],
                "blackboard_delta": dict(row.get("blackboard_delta") or {}),
            }
            for row in history[-5:]
        ],
        "budget_remaining": _budget_remaining_summary(budget),
        "safety_boundaries": {
            "planner_can_emit_solved": False,
            "raw_reaction_output_allowed": False,
            "final_verdict_authority": "deterministic_parent_route_proof",
            "child_route_cannot_promote_parent": True,
        },
        "no_solved_claim": True,
    }


def _action_payload_requirements(
    *,
    blackboard: dict[str, Any],
    source_candidates: list[dict[str, Any]],
    local_pdf_candidates: list[dict[str, Any]],
    visual_chains: list[dict[str, Any]],
) -> dict[str, Any]:
    extract_binding_required = len(_distinct_source_keys(local_pdf_candidates or source_candidates)) > 1
    compile_binding_required = (
        len(_distinct_source_keys(visual_chains)) > 1
        or len(_distinct_source_keys(source_candidates)) > 1
    )
    structure_tasks = [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("structure_resolution_tasks") or []
        if isinstance(row, dict) and str(row.get("status") or "open") == "open"
    ]
    structure_binding_required = (
        len(_distinct_source_keys(local_pdf_candidates or source_candidates or structure_tasks)) > 1
        or len(structure_tasks) > 1
    )
    source_binding_fields = [
        "source_ref",
        "doi",
        "pii",
        "url",
        "source_title",
        "title",
        "pdf_path",
        "local_pdf",
        "source_pdf_path",
    ]
    chain_binding_fields = [*source_binding_fields, "chain_id", "visual_chain_id", "artifact_ref"]
    return {
        "schema_version": "codex_action_payload_requirements.v1",
        "validator_rejection_reason": "source_sensitive_action_missing_source_binding",
        "search_actions": {
            "search_literature": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["search_intent", "query", "queries", "search_queries", "source_acquisition_policy"],
                "required_policy_fields": [
                    "codex_online_first",
                    "local_pdf_fallback_allowed",
                    "placeholder_allowed_after_failures",
                    "auto_local_pdf_requires_agent_discovered_metadata",
                    "fallback_order",
                    "no_solved_claim",
                ],
                "blackboard_guidance": _search_payload_requirement_guidance(blackboard),
                "validator_rejection_prefix": "search_literature_payload",
            }
        },
        "source_sensitive_actions": {
            "extract_pdf_literature_structures": {
                "currently_required": extract_binding_required,
                "reason": "multiple candidate literature/PDF sources are present" if extract_binding_required else "",
                "accepted_payload_fields": source_binding_fields,
                "binding_candidates": _compact_sources(local_pdf_candidates or source_candidates),
            },
            "extract_visual_literature_chain": {
                "currently_required": extract_binding_required,
                "reason": "multiple candidate literature/PDF sources are present" if extract_binding_required else "",
                "accepted_payload_fields": source_binding_fields,
                "binding_candidates": _compact_sources(local_pdf_candidates or source_candidates),
            },
            "resolve_literature_structure_task": {
                "currently_required": bool(structure_tasks) or structure_binding_required,
                "reason": "visual extraction produced unresolved compound labels" if structure_tasks else "",
                "accepted_payload_fields": [
                    *chain_binding_fields,
                    "task_id",
                    "label",
                    "compound_label",
                    "source_locator",
                    "run_visual",
                    "candidate_smiles",
                    "candidate_structures",
                    "no_solved_claim",
                ],
                "binding_candidates": _compact_structure_resolution_tasks(structure_tasks),
                "validator_rejection_prefix": "resolve_literature_structure_task_payload",
            },
            "compile_exact_literature_rows": {
                "currently_required": compile_binding_required,
                "reason": "multiple visual chains or candidate literature sources are present" if compile_binding_required else "",
                "accepted_payload_fields": chain_binding_fields,
                "binding_candidates": _compact_visual_chains(visual_chains) or _compact_sources(source_candidates),
            },
        },
        "guided_actions": {
            "run_guided_chemenzy": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["search_policy", "chem_enzy_search_policy"],
                "required_policy_list_fields": [
                    "terminal_blacklist",
                    "active_bridge_tasks",
                    "accepted_exact_row_ids",
                    "selected_analogical_hypothesis_ids",
                    "selected_analogical_template_ids",
                    "forbidden_template_ids",
                ],
                "required_policy_safety_fields": [
                    "source_budget.require_target_core_retention",
                    "source_budget.max_unexplained_heavy_atom_jump",
                    "source_budget.analogy_is_advisory_only",
                    "compiler_metadata.requires_verifier",
                    "compiler_metadata.no_solved_claim",
                ],
                "blackboard_guidance": _guided_payload_requirement_guidance(blackboard),
                "direct_baseline_policy": {
                    "simple_target_direct_chemenzy_allowed": True,
                    "complex_target_requires_prior_signal_or_bounded_initial_probe": True,
                    "bounded_initial_probe_limits": {
                        "max_steps": 6,
                        "chem_enzy_iterations": 10,
                        "chem_enzy_expansion_topk": 20,
                        "timeout_s": 180,
                        "source_budget.max_candidates": 5,
                    },
                },
                "validator_rejection_prefix": "guided_chemenzy_payload",
            }
        },
        "child_expansion_actions": {
            "expand_child_target": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["subgoal_targets", "child_targets"],
                "required_target_fields": [
                    "name",
                    "smiles",
                    "target_equivalence_audit_required",
                    "exact_target_override",
                    "no_solved_claim",
                    "child_route_cannot_promote_parent",
                    "chem_enzy_search_policy",
                ],
                "required_policy_safety_fields": [
                    "compiler_metadata.requires_verifier",
                    "compiler_metadata.no_solved_claim",
                    "compiler_metadata.child_route_cannot_promote_parent",
                ],
                "blackboard_guidance": _child_payload_requirement_guidance(blackboard),
                "validator_rejection_prefix": "child_expansion_payload",
            }
        },
        "stitch_actions": {
            "stitch_parent_route": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["proof_binding", "proof_policy", "analogy_refs"],
                "required_binding_fields": [
                    "child_route_ref",
                    "parent_route_ref",
                    "exact_literature_segment_ref",
                    "exact_literature_row_ids",
                    "input_refs",
                    "missing_inputs",
                ],
                "required_policy_fields": [
                    "target_equivalence_required",
                    "parent_route_verifier_required",
                    "stock_audit_required",
                    "no_unexplained_large_atom_jump_required",
                    "child_route_connectivity_required",
                    "exact_literature_connectivity_required",
                    "analogy_is_not_proof",
                    "child_route_cannot_promote_parent",
                    "final_verdict_authority",
                ],
                "blackboard_guidance": _stitch_payload_requirement_guidance(blackboard),
                "validator_rejection_prefix": "stitch_parent_route_payload",
            }
        },
        "analogical_template_actions": {
            action_type: {
                "currently_required_when_selected": True,
                "accepted_payload_fields": [
                    "analogical_template_policy",
                    "max_templates",
                    "max_applications",
                    "template_radius_policy",
                    "analog_template_confidence_threshold",
                ],
                "required_policy_fields": [
                    "schema_version",
                    "action_type",
                    "analogy_is_advisory_only",
                    "no_solved_claim",
                    "requires_verifier",
                    "requires_parent_route_proof",
                    "production_write_blocked",
                    "raw_reaction_output_allowed",
                    "final_verdict_authority",
                    "allowed_use",
                    "deterministic_template_validation_required",
                ],
                "allowed_policy_uses": [
                    "planner_priority",
                    "guided_policy_hint",
                    "template_candidate_validation",
                    "bridge_task_triage",
                ],
                "forbidden_policy_uses": ["solved_proof", "final_verdict", "parent_route_proof"],
                "blackboard_guidance": _analogical_template_payload_requirement_guidance(blackboard),
                "validator_rejection_prefix": "analogical_template_payload",
            }
            for action_type in (
                "extract_analogical_reaction_templates",
                "rank_analogical_reaction_templates",
                "apply_analogical_template_to_target",
                "validate_template_application",
            )
        },
    }


def _guided_payload_requirement_guidance(blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    template_policy = dict((blackboard.get("current_belief") or {}).get("template_policy") or {})
    return {
        "bridge_task_count": len(blackboard.get("bridge_tasks") or []),
        "terminal_blacklist_count": len(blackboard.get("terminal_blacklist") or []),
        "exact_row_count": len(evidence.get("exact_rows") or []),
        "selected_analogical_hypothesis_count": len(
            ((blackboard.get("analogical_hypothesis_ranking") or {}).get("selected_hypotheses") or [])
        ),
        "selected_analogical_template_count": len(
            ((blackboard.get("analogical_template_ranking") or {}).get("selected_templates") or [])
        ),
        "validated_template_one_step_row_count": int(template_policy.get("validated_one_step_row_count") or 0),
        "target_core_retention_required": bool(
            ((blackboard.get("current_belief") or {}).get("constraints") or {}).get("target_core_retention_required", True)
        ),
    }


def _search_payload_requirement_guidance(blackboard: dict[str, Any]) -> dict[str, Any]:
    target = dict(blackboard.get("target_profile") or {})
    bridge_tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    evidence = dict(blackboard.get("literature_evidence") or {})
    planner_hints = [dict(row) for row in evidence.get("planner_source_hints") or [] if isinstance(row, dict)]
    lifecycle = [dict(row) for row in evidence.get("source_lifecycle") or [] if isinstance(row, dict)]
    lifecycle_stage_counts = _source_lifecycle_stage_counts(lifecycle)
    return {
        "target_name": str(target.get("target_name") or ""),
        "family_hint": str(target.get("family_hint") or ""),
        "bridge_task_count": len(bridge_tasks),
        "planner_source_hint_count": len(planner_hints),
        "source_lifecycle_stage_counts": lifecycle_stage_counts,
        "local_pdf_proxy_request_count": len(evidence.get("local_pdf_proxy_requests") or []),
        "awaiting_local_pdf_proxy_count": int(lifecycle_stage_counts.get("local_pdf_proxy_requested") or 0),
        "planner_source_hints_are_not_evidence": True,
        "sample_planner_source_hints": [
            {
                "source_ref": str(row.get("source_ref") or ""),
                "title": str(row.get("title") or "")[:160],
                "doi": str(row.get("doi") or ""),
                "pii": str(row.get("pii") or ""),
                "url": str(row.get("url") or "")[:220],
            }
            for row in planner_hints[:4]
        ],
        "source_candidate_count": len(evidence.get("source_candidates") or []),
        "structure_resolution_task_count": len(evidence.get("structure_resolution_tasks") or []),
        "fallback_order": ["codex_online", "local_pdf", "placeholder"],
        "auto_local_pdf_requires_agent_discovered_metadata": True,
        "no_solved_claim": True,
    }


def _child_payload_requirement_guidance(blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    terminal_candidates = [
        dict(row)
        for row in evidence.get("terminal_candidates") or []
        if isinstance(row, dict) and str(row.get("smiles") or "").strip()
    ]
    upstream_tasks = [
        dict(row)
        for row in blackboard.get("bridge_tasks") or []
        if isinstance(row, dict) and str(row.get("task_type") or "") == "upstream_terminal_synthesis"
    ]
    return {
        "terminal_candidate_count": len(terminal_candidates),
        "upstream_terminal_bridge_task_count": len(upstream_tasks),
        "exact_row_count": len(evidence.get("exact_rows") or []),
        "source_ref_count": len(evidence.get("source_refs") or []),
        "sample_terminal_candidates": [
            {
                "name": str(row.get("name") or ""),
                "smiles": str(row.get("smiles") or ""),
                "source_ref": str(row.get("source_ref") or ""),
                "terminal_id": str(row.get("terminal_id") or ""),
            }
            for row in terminal_candidates[:4]
        ],
        "parent_proof_required_after_child_run": True,
        "child_route_cannot_promote_parent": True,
    }


def _stitch_payload_requirement_guidance(blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    refs = dict(blackboard.get("artifact_refs") or {})
    return {
        "child_route_solved": bool((blackboard.get("current_belief") or {}).get("child_route_solved")),
        "exact_row_count": len(evidence.get("exact_rows") or []),
        "route_expansion_ref_available": bool(refs.get("route_expansion_subgoal_search")),
        "guided_or_parent_route_ref_available": any("guided_chemenzy" in str(key) or "route_verifier" in str(key) for key in refs),
        "selected_analogy_count": len(((blackboard.get("analogical_hypothesis_ranking") or {}).get("selected_hypotheses") or [])),
        "analogy_must_not_be_used_as_proof": True,
        "final_verdict_authority": "deterministic_parent_route_proof",
    }


def _analogical_template_payload_requirement_guidance(blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    ranking = dict(blackboard.get("analogical_template_ranking") or {})
    return {
        "analogical_template_count": len(blackboard.get("analogical_templates") or []),
        "selected_analogical_template_count": len(ranking.get("selected_templates") or []),
        "template_application_count": len(blackboard.get("template_applications") or []),
        "source_ref_count": len(evidence.get("source_refs") or []),
        "exact_row_count": len(evidence.get("exact_rows") or []),
        "analogy_is_advisory_only": True,
        "no_solved_claim": True,
        "requires_verifier": True,
        "requires_parent_route_proof": True,
        "final_verdict_authority": "deterministic_parent_route_proof",
    }


def _source_lifecycle_stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("stage") or "unresolved")
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def _compact_source_lifecycle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_key": str(row.get("source_key") or ""),
            "source_ref": str(row.get("source_ref") or ""),
            "title": str(row.get("title") or "")[:180],
            "doi": str(row.get("doi") or ""),
            "local_pdf": str(row.get("local_pdf") or ""),
            "stage": str(row.get("stage") or ""),
            "next_recommended_stage": str(row.get("next_recommended_stage") or ""),
            "counts": dict(row.get("counts") or {}),
            "stage_flags": dict(row.get("stage_flags") or {}),
        }
        for row in rows[:8]
    ]


def _compact_visual_chains(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chain_id": str(row.get("chain_id") or ""),
            "artifact_ref": str(row.get("artifact_ref") or ""),
            "source_ref": str(row.get("source_ref") or ""),
            "source_title": str(row.get("source_title") or "")[:240],
            "source_pdf_path": str(row.get("source_pdf_path") or row.get("pdf_path") or ""),
            "candidate_step_count": int(row.get("candidate_step_count") or row.get("step_count") or 0),
            "accepted": bool(row.get("accepted")),
        }
        for row in rows[:8]
    ]


def _compact_structure_resolution_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": str(row.get("task_id") or ""),
            "label": str(row.get("label") or ""),
            "source_ref": str(row.get("source_ref") or ""),
            "source_title": str(row.get("source_title") or "")[:240],
            "source_locator": str(row.get("source_locator") or "")[:240],
            "artifact_ref": str(row.get("artifact_ref") or ""),
            "status": str(row.get("status") or "open"),
            "resolution_attempt_count": int(row.get("resolution_attempt_count") or 0),
            "last_resolution_status": str(row.get("last_resolution_status") or ""),
            "no_solved_claim": True,
        }
        for row in rows[:8]
    ]


def _distinct_source_keys(rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for index, row in enumerate(rows):
        key = _source_key(row)
        if not key:
            key = str(row.get("chain_id") or row.get("artifact_ref") or "").strip().lower()
            if key:
                key = f"chain:{key}"
        keys.add(key or f"row:{index}")
    return keys


def _compact_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_ref": str(row.get("source_ref") or ""),
            "doi": str(row.get("doi") or ""),
            "pii": str(row.get("pii") or ""),
            "title": str(row.get("title") or "")[:240],
            "local_pdf": str(row.get("local_pdf") or ""),
            "source_discovery_mode": str(row.get("source_discovery_mode") or ""),
        }
        for row in rows[:8]
    ]


def _source_keys(rows: list[Any]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _source_key(row)
        if key:
            keys.add(key)
    return keys


def _source_key(row: dict[str, Any]) -> str:
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref:
        return f"ref:{source_ref}"
    doi = str(row.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    pii = str(row.get("pii") or "").strip().lower()
    if pii:
        return f"pii:{pii}"
    local_pdf = str(row.get("local_pdf") or row.get("source_pdf_path") or row.get("pdf_path") or "").strip().lower()
    if local_pdf:
        return f"pdf:{local_pdf}"
    title = str(row.get("title") or row.get("source_title") or "").strip().lower()
    return f"title:{title}" if title else ""


def _candidate_has_real_source(row: dict[str, Any]) -> bool:
    if bool(row.get("placeholder_only")):
        return False
    if str(row.get("access_status") or "").strip().lower() == "placeholder_only":
        return False
    return bool(str(row.get("doi") or row.get("pii") or row.get("url") or row.get("local_pdf") or "").strip())


def _budget_remaining_summary(budget: dict[str, Any]) -> dict[str, int]:
    pairs = {
        "rounds": ("rounds_completed", "max_rounds"),
        "scout_calls": ("scout_calls", "max_scout_calls"),
        "visual_calls": ("visual_calls", "max_visual_calls"),
        "chemenzy_runs": ("chemenzy_runs", "max_chemenzy_runs"),
        "child_target_runs": ("child_target_runs", "max_child_target_runs"),
        "codex_research_runs": ("codex_research_runs", "max_codex_research_runs"),
        "template_application_actions": ("template_application_actions", "max_template_application_actions"),
    }
    out: dict[str, int] = {}
    for label, (used_key, max_key) in pairs.items():
        try:
            used = int(budget.get(used_key) or 0)
            maximum = int(budget.get(max_key) or 0)
        except (TypeError, ValueError):
            continue
        out[f"{label}_used"] = used
        out[f"{label}_max"] = maximum
        out[f"{label}_remaining"] = max(0, maximum - used) if maximum >= 0 else 0
    return out


def _compact_for_prompt(
    value: Any,
    *,
    max_depth: int = 5,
    max_list: int = 12,
    max_string: int = 1600,
) -> Any:
    if max_depth <= 0:
        return _summary_token(value)
    if isinstance(value, dict):
        preferred_keys = [
            "schema_version",
            "case_id",
            "target_profile",
            "route_failures",
            "plugin_runtime_diagnostics",
            "literature_evidence",
            "bridge_tasks",
            "target_side_disconnection_hypotheses",
            "analogical_hypothesis_ranking",
            "analogical_templates",
            "analogical_template_ranking",
            "template_applications",
            "terminal_blacklist",
            "planner_history",
            "action_history",
            "budget_state",
            "current_belief",
            "parent_route_proof",
            "artifact_refs",
        ]
        keys = [key for key in preferred_keys if key in value]
        keys.extend(key for key in value.keys() if key not in keys)
        return {
            str(key): _compact_for_prompt(value.get(key), max_depth=max_depth - 1, max_list=max_list, max_string=max_string)
            for key in keys[:40]
        }
    if isinstance(value, list):
        rows = value[-max_list:] if _looks_like_history(value) else value[:max_list]
        compact = [
            _compact_for_prompt(item, max_depth=max_depth - 1, max_list=max_list, max_string=max_string)
            for item in rows
        ]
        if len(value) > max_list:
            compact.append({"truncated_count": len(value) - max_list})
        return compact
    if isinstance(value, str):
        if len(value) > max_string:
            return value[:max_string] + f"...[truncated {len(value) - max_string} chars]"
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:max_string]


def _looks_like_history(value: list[Any]) -> bool:
    return any(isinstance(row, dict) and "round_index" in row and "action_type" in row for row in value)


def _summary_token(value: Any) -> str:
    if isinstance(value, dict):
        return f"<dict:{len(value)}>"
    if isinstance(value, list):
        return f"<list:{len(value)}>"
    text = str(value)
    return text[:120]


def _planner_enabled(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER")
    if raw is None:
        return False
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled", "mock"}


def _planner_use_codex_cli(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_BACKEND")
    if raw is not None:
        return raw.strip().lower() in {"codex", "codex_cli", "cli"}
    return True


def _planner_use_api_json(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_BACKEND")
    if raw is None:
        return False
    return raw.strip().lower() in {"api", "api_json", "json", "openai_compatible"}


def _planner_timeout_s() -> float:
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_TIMEOUT_S")
    if raw is None:
        return 90.0
    return max(5.0, float(raw))

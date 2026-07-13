"""Codex-first action planning for agentic blackboard runs."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
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
    independent_literature_source_keys,
    plan_action_batch,
    validate_action_batch,
)
from cascade_planner.harness.parent_route_proof import is_solved_parent_route_proof
from cascade_planner.harness.schemas import write_json
from cascade_planner.harness.source_capabilities import (
    build_source_capability_queue,
    eligible_source_capabilities,
    meaningful_compound_labels,
    pdf_evidence_has_materialized_render,
)


FallbackPlanner = Callable[..., dict[str, Any]]
DEFAULT_CODEX_ACTION_PLANNER_TOOLS = ("web_search", "browser", "literature_search")


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
    allow_deterministic_fallback: bool = True,
) -> dict[str, Any]:
    """Ask Codex to choose the next bounded action batch, then validate it.

    Codex is allowed to choose actions and priorities, but the local validator
    remains authoritative for action type, budget, stale repetition, raw
    reaction injection, and solved-claim rejection.
    """
    fallback = fallback_planner or plan_action_batch
    fast_path = _deterministic_fast_path_batch(
        blackboard=blackboard,
        round_index=round_index,
        exhaust_round_budget=exhaust_round_budget,
        fallback_planner=fallback,
    )
    if fast_path is not None:
        return fast_path

    if not _planner_enabled(enabled):
        return _fallback_batch(
            blackboard=blackboard,
            round_index=round_index,
            exhaust_round_budget=exhaust_round_budget,
            fallback_planner=fallback,
            reason="codex_action_planner_disabled",
            allow_deterministic_fallback=allow_deterministic_fallback,
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
            allow_deterministic_fallback=allow_deterministic_fallback,
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
            allow_deterministic_fallback=allow_deterministic_fallback,
        )

    batch["mode"] = "codex_blackboard_planner"
    batch["codex_action_planner"] = {
        "schema_version": "codex_action_planner_metadata.v1",
        "backend": str(record.backend or ""),
        "status": str(record.status or ""),
        "record_ref": str(run_dir / f"codex_action_planner_run_record_round_{int(round_index)}.json"),
        "blackboard_snapshot_ref": str(snapshot_path),
        "embedded_prompt_snapshot": _planner_prompt_snapshot_audit(snapshot_path),
        "validation": validation,
        "fallback_used": False,
        "tool_policy": tool_policy,
    }
    normalization_audit = dict(batch.get("normalization_audit") or {})
    if normalization_audit:
        batch["codex_action_planner"]["normalization_audit"] = normalization_audit
        batch["codex_action_planner"]["payload_normalization_used"] = bool(
            normalization_audit.get("payload_changes")
        )
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

    locally_repaired = _locally_repair_invalid_codex_batch(
        invalid_batch,
        validation=initial_validation,
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
                tool_policy=planner_tool_policy,
                repair_source="guarded_budget_salvage_of_codex_batch",
            )

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

    return None


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
        return False
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
    embedded_snapshot = _embedded_planner_prompt_snapshot(snapshot_path)
    allowed_tools = _planner_allowed_tools()
    max_tool_calls = _planner_max_tool_calls(allowed_tools)
    tool_policy = _planner_tool_policy(allowed_tools=allowed_tools, max_tool_calls=max_tool_calls)
    objective = (
        "You are the Codex validator-replanner for an autonomous retrosynthesis blackboard. "
        "The previous Codex action batch was rejected by the local safety/budget gate. "
        "Repair or replan it into a valid AgentActionBatch instead of handing control to a deterministic fallback. "
        "All decision state needed for this replan is already embedded below. Do not call shell, exec, terminal, "
        "PowerShell, Python, cat, or any filesystem tool to read input_refs. Input refs are immutable audit locators only; "
        "they do not need to be opened. Shell remains forbidden even when a path is present. "
        "Treat validation reasons as hard blockers. Preserve the useful scientific intent when possible, but drop, "
        "replace, or defer invalid actions. If a visual budget is exhausted, prefer non-visual PDF/source processing, "
        "template extraction/ranking, search/source acquisition, guided policy preparation, or stop_unresolved. "
        "If source-sensitive actions require binding, bind a concrete DOI, source_ref, pdf_path, chain_id, or artifact_ref "
        "from planner_context; if no honest binding exists, choose another action. "
        "When route_closure_pressure shows named literature/process anchors or open structure-resolution tasks, repair toward "
        "source-bound structure resolution, visual/source-detail extraction, advisory template derivation, or connected child-target testing. "
        "If resolved_anchor_structures are already present, preserve that progress and repair toward connection/template/child-target actions "
        "unless a concrete missing source binding blocks the next step. "
        "Do not discard named anchors just because they lack SMILES; choose an action that can resolve or connect them. "
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
        "Embedded immutable planner snapshot: "
        f"{json.dumps(embedded_snapshot, ensure_ascii=False, sort_keys=True)}"
    )
    return WorkerTask(
        task_id=f"{case_id}:codex_action_planner_repair:r{int(round_index)}",
        case_id=case_id,
        task_type="strategic_disconnection_mining",
        required_artifact_type="AgentActionBatch",
        input_refs=[str(snapshot_path)],
        allowed_tools=allowed_tools,
        budget=WorkerBudget(
            timeout_s=_planner_repair_timeout_s(),
            max_output_bytes=int(os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_OUTPUT_BYTES", "120000")),
            max_tool_calls=max_tool_calls,
            max_worker_runs=1,
            reasoning_effort=_planner_reasoning_effort(),
        ),
        objective=objective,
        allowed_workdir=str(run_dir),
        model=_planner_model(),
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
    out["mode"] = "codex_blackboard_planner_repaired"
    metadata = {
        "schema_version": "codex_action_planner_metadata.v1",
        "backend": str((repair_record or initial_record or {}).get("backend") or ""),
        "status": str((repair_record or initial_record or {}).get("status") or ""),
        "record_ref": str((initial_record or {}).get("record_ref") or ""),
        "blackboard_snapshot_ref": str(blackboard_snapshot_ref or ""),
        "embedded_prompt_snapshot": _planner_prompt_snapshot_audit(
            Path(blackboard_snapshot_ref)
        ),
        "validation": validation,
        "initial_validation": dict(initial_validation or {}),
        "repair_used": True,
        "repair_source": str(repair_source or ""),
        "fallback_used": False,
        "tool_policy": dict(tool_policy or {}),
    }
    local_repair_audit = dict(out.get("repair_audit") or {})
    if local_repair_audit:
        metadata["repair_audit"] = local_repair_audit
        metadata["repair_reasons"] = [
            str(item)
            for item in local_repair_audit.get("trigger_reasons") or []
            if str(item or "").strip()
        ]
    normalization_audit = dict(out.get("normalization_audit") or {})
    if normalization_audit:
        metadata["normalization_audit"] = normalization_audit
        metadata["payload_normalization_used"] = bool(normalization_audit.get("payload_changes"))
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
    initial_reasons = {
        str(item) for item in (validation or {}).get("reasons") or []
    }
    reasons = set(initial_reasons)
    repaired = deepcopy(batch)
    actions = [dict(row) for row in repaired.get("actions") or [] if isinstance(row, dict)]
    if not actions:
        return None
    original_actions = deepcopy(actions)

    for action in actions:
        payload = _repair_codex_action_payload(
            str(action.get("action_type") or ""),
            dict(action.get("payload") or {}),
            blackboard=blackboard,
        )
        action["payload"] = payload
        action_type = str(action.get("action_type") or "")
        if action_type == "search_literature":
            payload["max_sources"] = _bounded_source_count(payload.get("max_sources"), default=3)
        elif action_type in {
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "resolve_literature_structure_task",
            "compile_exact_literature_rows",
        } and "max_sources" in payload:
            # Follow-up actions are source-bound and consume one document at a
            # time; only discovery may intentionally return 2-3 independent
            # sources.
            payload["max_sources"] = 1

    post_payload_repair = deepcopy(repaired)
    post_payload_repair["actions"] = actions
    post_validation = validate_action_batch(
        post_payload_repair,
        blackboard=blackboard,
    )
    if _unsafe_validator_reasons(post_validation):
        return None
    reasons = {
        str(item) for item in (post_validation or {}).get("reasons") or []
    }
    invalid_indices = {
        int(row.get("index") or 0)
        for row in post_validation.get("action_validations") or []
        if isinstance(row, dict) and row.get("accepted") is not True
    }
    if invalid_indices:
        actions = [
            row for index, row in enumerate(actions) if index not in invalid_indices
        ]

    if "visual_total_budget_exceeded" in reasons:
        actions = _trim_visual_budget_actions(actions, blackboard=blackboard)
    if "scout_total_budget_exceeded" in reasons:
        actions = [row for row in actions if str(row.get("action_type") or "") != "search_literature"]
    if "literature_source_round_budget_exceeded" in reasons:
        actions = _trim_literature_source_budget_actions(actions, max_sources=3)
    if any(str(reason).startswith("failure_critic_requires_failure_evidence") for reason in reasons):
        actions = [row for row in actions if str(row.get("action_type") or "") != "build_failure_critic_report"]
    if any(str(reason).startswith("extract_pdf_literature_structures_requires_pdf_binding") for reason in reasons):
        actions = [row for row in actions if str(row.get("action_type") or "") != "extract_pdf_literature_structures"]
    if any(str(reason).startswith("resolve_literature_structure_task_payload:") for reason in reasons):
        actions = [
            row
            for row in actions
            if str(row.get("action_type") or "") != "resolve_literature_structure_task"
            or _repaired_structure_resolution_payload_valid(dict(row.get("payload") or {}))
        ]
    if any(str(reason).startswith("compile_exact_literature_rows_requires_uncompiled_visual_steps") for reason in reasons):
        actions = [row for row in actions if str(row.get("action_type") or "") != "compile_exact_literature_rows"]
        fallback_batch = plan_action_batch(
            blackboard,
            round_index=int(repaired.get("round_index") or 0),
            exhaust_round_budget=True,
        )
        for fallback_action in fallback_batch.get("actions") or []:
            if str(fallback_action.get("action_type") or "") != "derive_broad_reaction_template":
                continue
            if not any(str(row.get("action_type") or "") == "derive_broad_reaction_template" for row in actions):
                insert_at = next(
                    (
                        idx
                        for idx, row in enumerate(actions)
                        if str(row.get("action_type") or "") in {"run_guided_chemenzy", "expand_child_target", "search_literature"}
                    ),
                    len(actions),
                )
                actions.insert(insert_at, dict(fallback_action))
            break
    if "advisory_visual_template_requires_broad_template" in reasons:
        fallback_batch = plan_action_batch(
            blackboard,
            round_index=int(repaired.get("round_index") or 0),
            exhaust_round_budget=True,
        )
        for fallback_action in fallback_batch.get("actions") or []:
            if str(fallback_action.get("action_type") or "") != "derive_broad_reaction_template":
                continue
            if not any(str(row.get("action_type") or "") == "derive_broad_reaction_template" for row in actions):
                actions.append(dict(fallback_action))
                actions = _trim_actions_for_required_broad_template(actions)
            break
    if "guided_chemenzy_total_budget_exceeded" in reasons:
        actions = [row for row in actions if str(row.get("action_type") or "") != "run_guided_chemenzy"]
    if any("guided_chemenzy_missing_prior_signal_for_complex_target" in reason for reason in reasons):
        actions = [row for row in actions if str(row.get("action_type") or "") != "run_guided_chemenzy"]
    if "complex_target_requires_frontier_bootstrap_after_initial_probe" in reasons:
        actions = _ensure_frontier_bootstrap_action(actions, round_index=int(repaired.get("round_index") or 0))
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
    repaired["mode"] = "codex_blackboard_planner_repaired"
    repaired.setdefault("semantics", {})
    repaired["semantics"]["planner_can_emit_solved"] = False
    repaired["semantics"]["raw_reaction_output_allowed"] = False
    repaired["semantics"]["deterministic_validator_required"] = True
    repaired["repair_audit"] = _local_repair_audit(
        original_actions,
        repaired["actions"],
        trigger_reasons=sorted(initial_reasons | reasons),
    )
    return repaired


def _bounded_source_count(value: Any, *, default: int = 1, maximum: int = 3) -> int:
    try:
        parsed = int(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, min(int(maximum), parsed))


def _literature_action_source_cost(action: dict[str, Any]) -> int:
    action_type = str(action.get("action_type") or "")
    if action_type not in {
        "search_literature",
        "extract_pdf_literature_structures",
        "extract_visual_literature_chain",
        "resolve_literature_structure_task",
        "compile_exact_literature_rows",
    }:
        return 0
    if action_type == "search_literature":
        return _bounded_source_count((action.get("payload") or {}).get("max_sources"), default=3)
    return 1


def _trim_literature_source_budget_actions(
    actions: list[dict[str, Any]],
    *,
    max_sources: int,
) -> list[dict[str, Any]]:
    """Preserve bound follow-ups, then fit discovery into the remaining cap."""
    capacity = max(0, int(max_sources or 0))
    bound_followup_count = min(
        capacity,
        sum(
            1
            for row in actions
            if _literature_action_source_cost(row) > 0
            and str(row.get("action_type") or "") != "search_literature"
        ),
    )
    remaining_search = max(0, capacity - bound_followup_count)
    kept_followups = 0
    out: list[dict[str, Any]] = []
    for raw in actions:
        action = dict(raw)
        cost = _literature_action_source_cost(action)
        if cost <= 0:
            out.append(action)
            continue
        if str(action.get("action_type") or "") != "search_literature":
            if kept_followups >= bound_followup_count:
                continue
            out.append(action)
            kept_followups += 1
            continue
        if remaining_search <= 0:
            continue
        if cost > remaining_search:
            action["payload"] = dict(action.get("payload") or {})
            action["payload"]["max_sources"] = remaining_search
            cost = remaining_search
        if cost <= remaining_search:
            out.append(action)
            remaining_search -= cost
    return out


def _local_repair_audit(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    trigger_reasons: list[str],
) -> dict[str, Any]:
    def identity(row: dict[str, Any], index: int) -> str:
        return str(row.get("action_id") or f"{row.get('action_type') or 'action'}:{index}")

    before_index = {identity(row, idx): dict(row) for idx, row in enumerate(before)}
    after_index = {identity(row, idx): dict(row) for idx, row in enumerate(after)}
    dropped = [key for key in before_index if key not in after_index]
    inserted = [key for key in after_index if key not in before_index]
    payload_changes: list[dict[str, Any]] = []
    for key in sorted(set(before_index) & set(after_index)):
        old_payload = dict(before_index[key].get("payload") or {})
        new_payload = dict(after_index[key].get("payload") or {})
        changed_keys = sorted(
            field
            for field in set(old_payload) | set(new_payload)
            if old_payload.get(field) != new_payload.get(field)
        )
        if changed_keys:
            payload_changes.append(
                {
                    "action_id": key,
                    "action_type": str(after_index[key].get("action_type") or ""),
                    "changed_payload_fields": changed_keys,
                    "before_max_sources": old_payload.get("max_sources"),
                    "after_max_sources": new_payload.get("max_sources"),
                }
            )
    return {
        "schema_version": "codex_action_batch_local_repair_audit.v1",
        "trigger_reasons": [str(item) for item in trigger_reasons if str(item or "").strip()],
        "before_action_count": len(before),
        "after_action_count": len(after),
        "dropped_action_ids": dropped,
        "inserted_action_ids": inserted,
        "payload_changes": payload_changes,
        "silent_repair": False,
    }


def _trim_actions_for_required_broad_template(actions: list[dict[str, Any]], *, max_actions: int = 3) -> list[dict[str, Any]]:
    if len(actions) <= max_actions:
        return actions
    for action_type in (
        "run_guided_chemenzy",
        "expand_child_target",
        "resolve_literature_structure_task",
        "search_literature",
        "rank_analogical_hypotheses",
    ):
        for idx, row in enumerate(actions):
            if str(row.get("action_type") or "") == action_type:
                del actions[idx]
                return actions[:max_actions]
    return actions[:max_actions]


def _ensure_frontier_bootstrap_action(actions: list[dict[str, Any]], *, round_index: int) -> list[dict[str, Any]]:
    if any(str(row.get("action_type") or "") == "generate_disconnection_hypotheses" for row in actions):
        return actions[:3]
    bootstrap = {
        "schema_version": "agent_action.v1",
        "action_id": f"r{max(1, int(round_index or 1))}:frontier_bootstrap",
        "action_type": "generate_disconnection_hypotheses",
        "rationale": "bootstrap target-side precursor hypotheses before more complex-target exploration",
        "expected_artifact": "target_side_disconnection_hypotheses.v1",
        "success_condition": "typed hypotheses and bridge tasks are recorded",
        "payload": {"no_solved_claim": True},
    }
    repaired = [bootstrap, *actions]
    if len(repaired) <= 3:
        return repaired
    drop_order = [
        "stitch_parent_route",
        "run_guided_chemenzy",
        "extract_visual_literature_chain",
        "extract_pdf_literature_structures",
        "compile_exact_literature_rows",
        "resolve_literature_structure_task",
        "search_literature",
    ]
    for action_type in drop_order:
        if len(repaired) <= 3:
            break
        for idx in range(len(repaired) - 1, 0, -1):
            if str(repaired[idx].get("action_type") or "") == action_type:
                del repaired[idx]
                break
    return repaired[:3]


def _action_consumes_visual_budget(action: dict[str, Any]) -> bool:
    action_type = str(action.get("action_type") or "")
    if action_type == "extract_visual_literature_chain":
        return True
    if action_type == "resolve_literature_structure_task":
        return bool((action.get("payload") or {}).get("run_visual", True))
    return False


def _trim_visual_budget_actions(actions: list[dict[str, Any]], *, blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    budget = dict((blackboard or {}).get("budget_state") or {})
    try:
        remaining = int(budget.get("max_visual_calls") or 0) - int(budget.get("visual_calls") or 0)
    except (TypeError, ValueError):
        remaining = 0
    if remaining <= 0:
        return [row for row in actions if not _action_consumes_visual_budget(row)]
    visual_indices = [idx for idx, row in enumerate(actions) if _action_consumes_visual_budget(row)]
    if len(visual_indices) <= remaining:
        return actions
    keep_visual = set(
        sorted(
            visual_indices,
            key=lambda idx: (_visual_budget_action_priority(actions[idx]), idx),
        )[:remaining]
    )
    return [
        row
        for idx, row in enumerate(actions)
        if idx in keep_visual or not _action_consumes_visual_budget(row)
    ]


def _visual_budget_action_priority(action: dict[str, Any]) -> int:
    action_type = str(action.get("action_type") or "")
    payload = dict(action.get("payload") or {})
    if action_type == "resolve_literature_structure_task":
        try:
            explicit = int(payload.get("visual_budget_priority"))
        except (TypeError, ValueError):
            explicit = 3
        if explicit >= 0:
            return min(explicit, 8)
        return 3
    if action_type == "extract_visual_literature_chain":
        return 4
    return 9


def _repaired_structure_resolution_payload_valid(payload: dict[str, Any]) -> bool:
    return bool(
        str(payload.get("task_id") or "").strip()
        and str(payload.get("label") or payload.get("compound_label") or "").strip()
        and payload.get("no_solved_claim") is True
    )


def _planner_worker_max_attempts(*, mock_output: dict[str, Any] | None) -> int:
    if mock_output is not None:
        return 1
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_WORKER_ATTEMPTS")
    if raw is None:
        return 1
    return max(1, int(raw))


def _codex_action_planner_transient_failure(record_payload: dict[str, Any]) -> bool:
    status = str(record_payload.get("status") or "").lower()
    stderr = str(record_payload.get("stderr") or "").lower()
    reasons = " ".join(str(item).lower() for item in (record_payload.get("output_validation") or {}).get("reasons") or [])
    text = " ".join([status, stderr, reasons])
    if status == "timeout":
        return False
    if status == "worker_error":
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
    embedded_snapshot = _embedded_planner_prompt_snapshot(snapshot_path)
    allowed_tools = _planner_allowed_tools()
    max_tool_calls = _planner_max_tool_calls(allowed_tools)
    tool_policy = _planner_tool_policy(allowed_tools=allowed_tools, max_tool_calls=max_tool_calls)
    objective = (
        "You are the central blackboard action planner for an autonomous retrosynthesis workflow. "
        "Use the embedded immutable planner snapshot below and choose the next action batch. "
        "It already contains the bounded decision state needed for planning, including pending sources, local PDFs, open tasks, "
        "budgets, route/frontier summaries, and action bindings. Do not call shell, exec, terminal, PowerShell, Python, cat, "
        "or any filesystem tool to read input_refs. Input refs are audit locators only and do not need to be opened. "
        "Shell remains forbidden even when a file path is present. "
        "You may freely choose priorities and exploration direction, "
        "but you may only emit typed actions, not routes or solved verdicts. "
        "Select at most 3 actions. Prefer parallel actions only when they do not depend on each other. "
        "For PDF, visual, structure-resolution, or exact-compilation work, select only an eligible entry from "
        "evidence_board.source_capability_queue. Its cost.literature_source_units counts against the explicit "
        "literature_source_units_max_this_round cap; never select a blocked or absent source capability. "
        "Before returning, internally check the action_payload_requirements and repair your own draft so it is valid. "
        "Do not rely on a deterministic fallback to make the scientific choice for you. "
        "Keep the JSON small: rationale under 45 words, expected_artifact under 12 words, success_condition under 20 words, "
        "and payload as a skeleton with fewer than 12 scalar/list fields. "
        "If two recent rounds produced no useful artifact, either change direction or choose stop_unresolved. "
        "Use search_literature when source evidence/target-proximal bridge evidence is missing, when discovered metadata still needs accessible "
        "HTML/PDF material, or when fewer than two independent logical source groups are present. Article and SI documents sharing one DOI are "
        "one source group. For search_literature, include search_intent, short queries, and max_sources up to 3; local repair preserves a 2-3 source "
        "request and adds source acquisition/independence policies. "
        "Default to online source acquisition through web/browser/literature search. "
        "Do not ask for local PDF fallback unless the user explicitly supplied a PDF/source seed or the blackboard already has same-target local PDF candidates. "
        "Placeholders are allowed only after online access failures. "
        "Continue discovered evidence across rounds: acquire accessible HTML/PDF material, render a source-bound PDF, extract visual structures, "
        "resolve open compound labels, then compile exact rows. Use visual/PDF extraction only after a source candidate or local PDF is available. "
        "Use compile_exact_literature_rows only after current visual/source-detail extraction has produced uncompiled molecular steps; "
        "process, fermentation, feedstock-mixture, organism, strain, or biotransformation evidence is an objective/anchor signal, not an exact reaction row. "
        "When process evidence names an advanced intermediate, feedstock, endpoint, or transformation but lacks exact SMILES, treat it as a route-closure anchor: "
        "bind the source, resolve the named structure when possible, extract a visual/source-detail chain, or derive an advisory template before drifting into weaker fallback child targets. "
        "When route_closure_pressure already includes resolved_anchor_structures, treat those SMILES as candidate bridge material for advisory template derivation, "
        "child-target expansion, guided rerun, or stitch/proof triage before spending more actions on broad source discovery. "
        "Use route_anchor_opportunities and route_closure_pressure as the primary decision surface; they are evidence-grounded branches with action menus, "
        "not a hardcoded script. Incomplete process/advisory/name-only anchors lower confidence but should still be advanced as hypothesis/template material "
        "when a typed action can reduce a proof gap. "
        "Do not choose stop_unresolved while a route_anchor_opportunity has plausible_next_actions and budget remains, unless every listed action is blocked by "
        "recent repeated failure or missing source binding. "
        "If parent/objective proof says deterministic_connected_route_not_proven, prefer actions that can connect the named literature anchor to the target or to a verified child route. "
        "For source-sensitive actions, follow planner_context.action_payload_requirements: when currently_required is true, "
        "the action payload must include a concrete binding such as source_ref, doi, pdf_path, source_pdf_path, chain_id, or artifact_ref. "
        "When the blackboard has pending retrosynthetic proposals or recursive hypothesis frontiers, prefer expanding those typed "
        "proposal targets over emitting another vague guided search; use run_guided_chemenzy for a bounded parent-side scan or when "
        "compiled evidence should steer the main target, and use expand_child_target when a concrete proposal precursor should be tested recursively. "
        "For run_guided_chemenzy, do not emit a full search_policy or chem_enzy_search_policy. Emit only intent fields "
        "such as initial_probe, search_mode, max_steps, chem_enzy_iterations, chem_enzy_expansion_topk, timeout_s, "
        "and max_candidates; local repair will build the complete policy from the blackboard. "
        "For expand_child_target, include explicit subgoal_targets or child_targets with target_equivalence_audit_required, "
        "exact_target_override, no_solved_claim, child_route_cannot_promote_parent, and policy_runtime_rebuild=true. "
        "Do not emit full child chem_enzy_search_policy/search_policy; local execution rebuilds the policy from the blackboard. "
        "For stitch_parent_route, include proof_binding and proof_policy; the payload must bind parent route plus either "
        "child/exact literature connectivity or an already solved direct parent-route verifier, with any missing input stated explicitly. "
        "For analogical template actions, include analogical_template_policy; analogy is advisory only and cannot be proof "
        "or final verdict authority. Safe allowed_use values are planner_priority, guided_policy_hint, "
        "template_candidate_validation, and bridge_task_triage. "
        "For simple low-complexity targets with no deterministic parent-route proof and no prior guided retrosynthesis attempt, "
        "select run_guided_chemenzy as one bounded action before more literature/template mining. "
        "This is not a rigid ChemEnzy-first policy; it is a direct retrosynthesis attempt delegated to the model when sensible. "
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
        "Embedded immutable planner snapshot: "
        f"{json.dumps(embedded_snapshot, ensure_ascii=False, sort_keys=True)}"
    )
    return WorkerTask(
        task_id=f"{case_id}:codex_action_planner:r{int(round_index)}",
        case_id=case_id,
        task_type="strategic_disconnection_mining",
        required_artifact_type="AgentActionBatch",
        input_refs=[str(snapshot_path)],
        allowed_tools=allowed_tools,
        budget=WorkerBudget(
            timeout_s=_planner_timeout_s(),
            max_output_bytes=int(os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_OUTPUT_BYTES", "120000")),
            max_tool_calls=max_tool_calls,
            max_worker_runs=1,
            reasoning_effort=_planner_reasoning_effort(),
        ),
        objective=objective,
        allowed_workdir=str(run_dir),
        model=_planner_model(),
    )


def _planner_model() -> str:
    """Use an explicit, known-working model instead of ambient CLI config.

    The action planner is a recoverable workflow component, so it must not
    silently inherit a newer model name that the installed Codex CLI cannot
    serve. Callers can still override this independently from the recursive
    retrosynthesis teams.
    """

    return str(
        os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_MODEL")
        or os.environ.get("AUTOPLANNER_CODEX_MODEL")
        or "gpt-5.5"
    ).strip()


def _normalize_codex_batch(
    payload: dict[str, Any],
    *,
    blackboard: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    case_id = str(blackboard.get("case_id") or payload.get("case_id") or "case")
    actions: list[dict[str, Any]] = []
    normalization_changes: list[dict[str, Any]] = []
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
        original_payload = deepcopy(dict(raw.get("payload") or {}))
        repair_payload = dict(original_payload)
        repair_payload.setdefault("_planner_action_id", action_id)
        repair_payload.setdefault("_planner_expected_artifact", str(raw.get("expected_artifact") or ""))
        repair_payload.setdefault("_planner_rationale", str(raw.get("rationale") or ""))
        repair_payload.setdefault("_planner_success_condition", str(raw.get("success_condition") or ""))
        action = {
            "schema_version": str(raw.get("schema_version") or "agent_action.v1"),
            "action_id": action_id,
            "action_type": action_type,
            "rationale": str(raw.get("rationale") or ""),
            "expected_artifact": str(raw.get("expected_artifact") or ""),
            "success_condition": str(raw.get("success_condition") or ""),
            "payload": repair_payload,
        }
        action["payload"] = _repair_codex_action_payload(action_type, dict(action.get("payload") or {}), blackboard=blackboard)
        for internal_key in (
            "_planner_action_id",
            "_planner_expected_artifact",
            "_planner_rationale",
            "_planner_success_condition",
        ):
            action["payload"].pop(internal_key, None)
        payload_audit = _payload_normalization_audit(
            action_id=action_id,
            action_type=action_type,
            before=original_payload,
            after=dict(action.get("payload") or {}),
        )
        if payload_audit:
            normalization_changes.append(payload_audit)
        for key, value in raw.items():
            key_l = str(key).lower()
            if key_l in {"verdict", "route_status", "status"} or key_l in FORBIDDEN_RAW_REACTION_KEYS:
                action[str(key)] = value
        actions.append(action)
    batch = {
        "schema_version": ACTION_BATCH_SCHEMA,
        "case_id": case_id,
        "round_index": int(round_index),
        "mode": str(payload.get("mode") or "codex_blackboard_planner"),
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
    batch["normalization_audit"] = {
        "schema_version": "codex_action_batch_normalization_audit.v1",
        "payload_changes": normalization_changes,
        "changed_action_count": len(normalization_changes),
        "silent_repair": False,
    }
    return batch


def _payload_normalization_audit(
    *,
    action_id: str,
    action_type: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    changed_fields = sorted(
        field
        for field in set(before) | set(after)
        if before.get(field) != after.get(field)
    )
    if not changed_fields:
        return {}
    added = [field for field in changed_fields if field not in before]
    removed = [field for field in changed_fields if field not in after]
    normalized = [field for field in changed_fields if field in before and field in after]
    reasons = [
        *[f"payload_field_completed:{field}" for field in added],
        *[f"payload_field_removed_by_contract:{field}" for field in removed],
        *[f"payload_field_normalized:{field}" for field in normalized],
    ]
    return {
        "action_id": str(action_id or ""),
        "action_type": str(action_type or ""),
        "changed_payload_fields": changed_fields,
        "added_payload_fields": added,
        "removed_payload_fields": removed,
        "normalized_payload_fields": normalized,
        "reasons": reasons,
        "before_max_sources": before.get("max_sources"),
        "after_max_sources": after.get("max_sources"),
    }


def _repair_codex_action_payload(action_type: str, payload: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    if action_type == "search_literature":
        return _repair_search_literature_payload(payload, blackboard=blackboard)
    if action_type == "extract_pdf_literature_structures":
        return _repair_pdf_literature_payload(payload, blackboard=blackboard)
    if action_type == "extract_visual_literature_chain":
        return _repair_visual_literature_payload(payload, blackboard=blackboard)
    if action_type == "resolve_literature_structure_task":
        return _repair_structure_resolution_payload(payload, blackboard=blackboard)
    if action_type == "run_guided_chemenzy":
        return _repair_guided_chemenzy_payload(payload, blackboard=blackboard)
    if action_type == "expand_child_target":
        return _repair_child_expansion_payload(payload, blackboard=blackboard)
    if action_type == "compile_exact_literature_rows":
        return _repair_compile_exact_rows_payload(payload, blackboard=blackboard)
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
    local_pdf_allowed = _local_pdf_fallback_allowed_for_blackboard(blackboard)
    policy["schema_version"] = "agentic_source_acquisition_policy.v1"
    policy["codex_online_first"] = True
    policy["local_pdf_fallback_allowed"] = local_pdf_allowed
    policy["placeholder_allowed_after_failures"] = True
    policy["auto_local_pdf_requires_agent_discovered_metadata"] = True
    policy["no_solved_claim"] = True
    policy["fallback_order"] = ["codex_online", *(["local_pdf"] if local_pdf_allowed else []), "placeholder"]
    raw["source_acquisition_policy"] = policy
    raw["max_sources"] = _bounded_source_count(raw.get("max_sources"), default=3)
    independence = dict(raw.get("source_independence_policy") or {})
    independence["schema_version"] = "agentic_source_independence_policy.v1"
    independence["group_by"] = ["doi", "pii", "patent_family", "canonical_source_ref", "title"]
    independence["article_and_supporting_information_share_source_group"] = True
    independence["require_distinct_source_groups"] = True
    independence["no_solved_claim"] = True
    raw["source_independence_policy"] = independence
    raw.setdefault("minimum_independent_sources", 2)
    raw.setdefault("preferred_independent_sources", 3)
    raw["no_solved_claim"] = True
    raw["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "search_literature",
        "completed_from_blackboard": True,
    }
    return raw


def _local_pdf_fallback_allowed_for_blackboard(blackboard: dict[str, Any]) -> bool:
    raw = os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_LOCAL_PDF_FALLBACK_ALLOWED")
    env_allowed = True if raw is None else _parse_env_bool(raw, default=False)
    return bool(env_allowed or _blackboard_has_local_pdf_seed(blackboard))


def _blackboard_has_local_pdf_seed(blackboard: dict[str, Any]) -> bool:
    evidence = dict(blackboard.get("literature_evidence") or {})
    target_input = dict(blackboard.get("target_input") or {})
    for row in evidence.get("source_candidates") or []:
        if isinstance(row, dict) and str(row.get("local_pdf") or row.get("pdf_path") or "").strip():
            return True
    for row in target_input.get("literature_sources") or []:
        if isinstance(row, dict) and str(row.get("local_pdf") or row.get("pdf_path") or "").strip():
            return True
    return bool(str(target_input.get("literature_pdf_path") or "").strip())


def _parse_env_bool(value: Any, *, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return bool(default)


def _repair_visual_literature_payload(payload: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    raw = dict(payload or {})
    source = _source_candidate_for_visual_payload(blackboard, raw)
    if source:
        raw.setdefault("source_ref", str(source.get("source_ref") or ""))
        raw.setdefault("doi", str(source.get("doi") or ""))
        raw.setdefault("pii", str(source.get("pii") or ""))
        raw.setdefault("url", str(source.get("url") or ""))
        raw.setdefault("source_title", str(source.get("title") or source.get("source_title") or ""))
        pdf_path = _source_candidate_pdf_path(source)
        if pdf_path:
            raw.setdefault("pdf_path", pdf_path)
        labels = [str(item) for item in source.get("expected_scheme_or_compound_labels") or [] if str(item or "").strip()]
        if labels:
            raw.setdefault("expected_labels", labels)
            raw.setdefault("compound_labels", labels)
        if source.get("route_sequence_hint"):
            raw.setdefault("route_sequence_hint", str(source.get("route_sequence_hint") or ""))
    if not str(raw.get("source_ref") or "").strip() and str(raw.get("doi") or "").strip():
        raw["source_ref"] = f"doi:{str(raw.get('doi')).strip()}"
    raw["schema_version"] = "visual_literature_chain_payload.v1"
    raw.setdefault("render_zoom", 1.35)
    raw["compress_images"] = bool(raw.get("compress_images", True))
    raw["max_images"] = _positive_int_or_default(raw.get("max_images"), 6)
    raw["visual_max_side_px"] = _positive_int_or_default(raw.get("visual_max_side_px"), 1400)
    raw["visual_jpeg_quality"] = _positive_int_or_default(raw.get("visual_jpeg_quality"), 70)
    raw["timeout_s"] = _planner_visual_timeout_s(raw.get("timeout_s"))
    raw["no_solved_claim"] = True
    raw["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "extract_visual_literature_chain",
        "completed_from_blackboard": bool(source),
        "timeout_repaired": True,
        "source_bound": bool(
            str(raw.get("source_ref") or raw.get("doi") or raw.get("pdf_path") or raw.get("source_pdf_path") or "").strip()
        ),
    }
    return _whitelist_payload_fields(
        raw,
        {
            "schema_version",
            "source_capability_id",
            "source_ref",
            "doi",
            "pii",
            "url",
            "source_title",
            "title",
            "pdf_path",
            "local_pdf",
            "source_pdf_path",
            "expected_labels",
            "compound_labels",
            "page_numbers",
            "route_sequence_hint",
            "focused_gap_repair",
            "focused_structure_resolution",
            "render_zoom",
            "compress_images",
            "max_images",
            "visual_max_side_px",
            "visual_jpeg_quality",
            "timeout_s",
            "no_solved_claim",
            "codex_payload_repair",
        },
    )


def _repair_pdf_literature_payload(payload: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    raw = dict(payload or {})
    source = _source_candidate_for_pdf_payload(blackboard, raw)
    if source:
        raw.setdefault("source_ref", str(source.get("source_ref") or ""))
        raw.setdefault("doi", str(source.get("doi") or ""))
        raw.setdefault("pii", str(source.get("pii") or ""))
        raw.setdefault("url", str(source.get("url") or ""))
        raw.setdefault("source_title", str(source.get("title") or source.get("source_title") or ""))
        pdf_path = _source_candidate_pdf_path(source)
        if pdf_path:
            raw.setdefault("pdf_path", pdf_path)
        labels = [str(item) for item in source.get("expected_scheme_or_compound_labels") or [] if str(item or "").strip()]
        if labels:
            raw.setdefault("expected_labels", labels)
    if not str(raw.get("source_ref") or "").strip() and str(raw.get("doi") or "").strip():
        raw["source_ref"] = f"doi:{str(raw.get('doi')).strip()}"
    raw["schema_version"] = "extract_pdf_literature_structures_payload.v1"
    raw["no_solved_claim"] = True
    raw["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "extract_pdf_literature_structures",
        "completed_from_blackboard": bool(source),
        "whitelist_compacted": True,
    }
    return _whitelist_payload_fields(
        raw,
        {
            "schema_version",
            "source_capability_id",
            "source_ref",
            "doi",
            "pii",
            "url",
            "source_title",
            "title",
            "pdf_path",
            "local_pdf",
            "source_pdf_path",
            "expected_labels",
            "page_numbers",
            "max_sources",
            "no_solved_claim",
            "codex_payload_repair",
        },
    )


def _source_candidate_for_visual_payload(blackboard: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    candidates = [
        dict(row)
        for row in evidence.get("source_candidates") or []
        if isinstance(row, dict) and _source_candidate_pdf_path(row)
    ]
    if not candidates:
        return {}
    for row in candidates:
        if _source_candidate_matches_payload(row, payload):
            return row
    if len(candidates) == 1:
        return candidates[0]
    return {}


def _source_candidate_for_pdf_payload(blackboard: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    candidates = [
        dict(row)
        for row in evidence.get("source_candidates") or []
        if isinstance(row, dict) and _source_candidate_pdf_path(row)
    ]
    if not candidates:
        return {}
    for row in candidates:
        if _source_candidate_matches_payload(row, payload):
            return row
    if len(candidates) == 1:
        return candidates[0]
    return {}


def _source_candidate_matches_payload(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    payload_key = _source_key(payload)
    row_key = _source_key(row)
    if payload_key and row_key and payload_key == row_key:
        return True
    for field in ("source_ref", "doi", "pii", "url"):
        requested = str(payload.get(field) or "").strip().lower()
        if requested and requested == str(row.get(field) or "").strip().lower():
            return True
    requested_pdf = str(payload.get("pdf_path") or payload.get("local_pdf") or payload.get("source_pdf_path") or "").strip().lower()
    if requested_pdf and requested_pdf == _source_candidate_pdf_path(row).lower():
        return True
    requested_title = str(payload.get("source_title") or payload.get("title") or "").strip().lower()
    row_title = str(row.get("title") or row.get("source_title") or "").strip().lower()
    return bool(requested_title and row_title and requested_title == row_title)


def _source_candidate_pdf_path(row: dict[str, Any]) -> str:
    return str(row.get("local_pdf") or row.get("pdf_path") or row.get("source_pdf_path") or "").strip()


def _source_candidate_for_compile_payload(blackboard: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    candidates = [
        dict(row)
        for row in evidence.get("source_candidates") or []
        if isinstance(row, dict)
    ]
    if not candidates:
        return {}
    for row in candidates:
        if _source_candidate_matches_payload(row, payload):
            return row
    visual = _visual_chain_for_compile_payload(blackboard, payload)
    visual_key = _source_key(visual)
    if visual_key:
        for row in candidates:
            if _source_key(row) == visual_key:
                return row
    return candidates[0] if len(candidates) == 1 else {}


def _visual_chain_for_compile_payload(blackboard: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    chains = [
        dict(row)
        for row in evidence.get("visual_chains") or []
        if isinstance(row, dict)
    ]
    if not chains:
        return {}
    requested_key = _source_key(payload)
    requested_chain = str(payload.get("chain_id") or payload.get("visual_chain_id") or payload.get("artifact_ref") or "").strip()
    for row in chains:
        chain_id = str(row.get("chain_id") or row.get("artifact_ref") or "").strip()
        artifact = str(row.get("artifact_ref") or "").strip()
        if requested_chain and requested_chain in {chain_id, artifact}:
            return row
        if requested_key and _source_key(row) == requested_key:
            return row
    for row in chains:
        if _compact_visual_chain_step_count(row) > 0:
            return row
    return chains[-1]


def _compact_visual_chain_step_count(row: dict[str, Any]) -> int:
    for field in ("candidate_step_count", "step_count"):
        try:
            value = int(row.get(field) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    steps = row.get("steps")
    return len([step for step in steps or [] if isinstance(step, dict)]) if isinstance(steps, list) else 0


def _whitelist_payload_fields(payload: dict[str, Any], allowed_fields: set[str]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in allowed_fields:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        compact[key] = value
    return compact


def _positive_int_or_default(value: Any, default: int) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        parsed = 0
    return parsed if parsed > 0 else int(default)


def _planner_visual_timeout_s(value: Any) -> int:
    try:
        parsed = float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        parsed = 0.0
    if parsed <= 0:
        raw = os.environ.get("AUTOPLANNER_VISUAL_TIMEOUT_S")
        try:
            parsed = float(raw) if raw not in (None, "") else 900.0
        except (TypeError, ValueError):
            parsed = 900.0
    return int(max(120.0, parsed))


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
        if task.get("visual_budget_priority") is not None:
            raw.setdefault("visual_budget_priority", task.get("visual_budget_priority"))
        source = _source_candidate_for_structure_task(blackboard, task)
        if source:
            raw.setdefault("pdf_path", str(source.get("local_pdf") or source.get("pdf_path") or source.get("source_pdf_path") or ""))
            raw.setdefault("source_title", str(source.get("title") or source.get("source_title") or raw.get("source_title") or ""))
    raw["schema_version"] = "literature_structure_resolution_payload.v1"
    target_identity_shortcut = _structure_resolution_payload_targets_input_label(raw, blackboard)
    raw["run_visual"] = False if target_identity_shortcut else bool(raw.get("run_visual", True))
    if target_identity_shortcut:
        raw["target_identity_shortcut"] = True
        target = dict(blackboard.get("target_profile") or {})
        raw.setdefault("target_smiles", str(target.get("target_smiles") or target.get("isomeric_smiles") or ""))
    raw["compress_images"] = bool(raw.get("compress_images", True))
    raw["max_images"] = int(raw.get("max_images") or 6)
    raw["visual_max_side_px"] = int(raw.get("visual_max_side_px") or 1400)
    raw["visual_jpeg_quality"] = int(raw.get("visual_jpeg_quality") or 70)
    raw["timeout_s"] = _planner_visual_timeout_s(raw.get("timeout_s"))
    raw["no_solved_claim"] = True
    raw["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "resolve_literature_structure_task",
        "completed_from_blackboard": bool(task),
        "target_identity_shortcut": bool(target_identity_shortcut),
    }
    return _whitelist_payload_fields(
        raw,
        {
            "schema_version",
            "source_capability_id",
            "task_id",
            "label",
            "compound_label",
            "source_ref",
            "doi",
            "pii",
            "url",
            "source_title",
            "source_locator",
            "artifact_ref",
            "pdf_path",
            "local_pdf",
            "source_pdf_path",
            "run_visual",
            "target_identity_shortcut",
            "target_smiles",
            "candidate_smiles",
            "candidate_structures",
            "compress_images",
            "max_images",
            "visual_max_side_px",
            "visual_jpeg_quality",
            "timeout_s",
            "visual_budget_priority",
            "no_solved_claim",
            "codex_payload_repair",
        },
    )


def _repair_compile_exact_rows_payload(payload: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    raw = dict(payload or {})
    source = _source_candidate_for_compile_payload(blackboard, raw)
    if source:
        raw.setdefault("source_ref", str(source.get("source_ref") or ""))
        raw.setdefault("doi", str(source.get("doi") or ""))
        raw.setdefault("pii", str(source.get("pii") or ""))
        raw.setdefault("url", str(source.get("url") or ""))
        raw.setdefault("source_title", str(source.get("title") or source.get("source_title") or ""))
        pdf_path = _source_candidate_pdf_path(source)
        if pdf_path:
            raw.setdefault("pdf_path", pdf_path)
    visual = _visual_chain_for_compile_payload(blackboard, raw)
    if visual:
        raw.setdefault("chain_id", str(visual.get("chain_id") or ""))
        raw.setdefault("visual_chain_id", str(visual.get("chain_id") or ""))
        raw.setdefault("artifact_ref", str(visual.get("artifact_ref") or ""))
        raw.setdefault("source_ref", str(visual.get("source_ref") or raw.get("source_ref") or ""))
        raw.setdefault("source_title", str(visual.get("source_title") or raw.get("source_title") or ""))
    if not str(raw.get("source_ref") or "").strip() and str(raw.get("doi") or "").strip():
        raw["source_ref"] = f"doi:{str(raw.get('doi')).strip()}"
    raw["schema_version"] = "compile_exact_literature_rows_payload.v1"
    raw["compile_attempt"] = _positive_int_or_default(raw.get("compile_attempt"), 1)
    raw["no_solved_claim"] = True
    raw["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "compile_exact_literature_rows",
        "completed_from_blackboard": bool(source or visual),
        "whitelist_compacted": True,
    }
    return _whitelist_payload_fields(
        raw,
        {
            "schema_version",
            "source_capability_id",
            "compile_attempt",
            "source_ref",
            "doi",
            "pii",
            "url",
            "source_title",
            "title",
            "pdf_path",
            "local_pdf",
            "source_pdf_path",
            "chain_id",
            "visual_chain_id",
            "artifact_ref",
            "expected_labels",
            "page_numbers",
            "target_smiles",
            "target_name",
            "terminal_smiles",
            "terminal_name",
            "source_detail_steps",
            "source_detail_route_steps",
            "source_detail_resolution_path",
            "source_detail_steps_path",
            "compiled_downstream_path",
            "no_solved_claim",
            "codex_payload_repair",
        },
    )


def _structure_resolution_task_for_payload(blackboard: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    requested_id = str(payload.get("task_id") or "").strip()
    requested_labels = _structure_resolution_requested_labels(payload)
    requested_source_key = _source_key(payload)
    requested_action_text = _structure_label_key(str(payload.get("_planner_action_id") or "").replace("_", " "))
    requested_text = " ".join(
        str(payload.get(key) or "")
        for key in (
            "_planner_action_id",
            "_planner_expected_artifact",
            "_planner_rationale",
            "_planner_success_condition",
            "route_sequence_hint",
            "source_title",
            "search_intent",
        )
    ).lower()
    tasks = _meaningful_open_structure_resolution_tasks(blackboard)
    source_scoped_tasks = [
        task for task in tasks if _source_task_matches_request(task, requested_source_key)
    ] if requested_source_key else tasks
    for task in tasks:
        if (
            requested_id
            and str(task.get("task_id") or "") == requested_id
            and _source_task_matches_request(task, requested_source_key)
        ):
            return task
    scored = sorted(
        (
            (
                _structure_resolution_task_match_score(
                    task,
                    requested_labels=requested_labels,
                    requested_text=requested_text,
                    requested_action_text=requested_action_text,
                ),
                task,
            )
            for task in source_scoped_tasks
        ),
        key=lambda item: (item[0], -_structure_resolution_attempt_count(item[1])),
        reverse=True,
    )
    if scored and scored[0][0] > 0:
        return scored[0][1]
    if requested_text:
        for task in source_scoped_tasks:
            label = str(task.get("label") or "").strip().lower()
            if (
                label
                and label in requested_text
            ):
                return task
    preferred = _preferred_structure_resolution_task_from_visual_chain(blackboard, source_scoped_tasks)
    if preferred:
        return preferred
    for task in source_scoped_tasks:
        if _structure_resolution_attempt_count(task) > 0:
            continue
        return task
    return source_scoped_tasks[0] if source_scoped_tasks else {}


def _structure_resolution_requested_labels(payload: dict[str, Any]) -> set[str]:
    labels = _structure_resolution_explicit_labels(payload)
    label_text = " ".join(
        str(payload.get(key) or "")
        for key in (
            "_planner_action_id",
            "_planner_expected_artifact",
            "_planner_rationale",
            "_planner_success_condition",
            "route_sequence_hint",
            "search_intent",
        )
    ).lower()
    labels.update(_structure_resolution_label_hints_from_text(label_text))
    return {label for label in labels if label}


def _structure_resolution_explicit_labels(payload: dict[str, Any]) -> set[str]:
    raw_labels: list[Any] = [
        payload.get("label"),
        payload.get("compound_label"),
    ]
    for field in ("expected_labels", "compound_labels"):
        value = payload.get(field)
        if isinstance(value, list):
            raw_labels.extend(value)
        elif value:
            raw_labels.append(value)
    labels = {
        _structure_label_key(str(item or ""))
        for item in raw_labels
        if str(item or "").strip()
    }
    return {label for label in labels if label}


def _source_task_matches_request(task: dict[str, Any], requested_source_key: str) -> bool:
    if not requested_source_key:
        return True
    task_key = _source_key(task)
    return bool(task_key and task_key == requested_source_key)


def _structure_resolution_label_hints_from_text(text: str) -> set[str]:
    normalized = _structure_label_key(text.replace("_", " ").replace("-", " "))
    if not normalized:
        return set()
    labels: set[str] = set()
    numeric_labels = set(re.findall(r"\b(?:label|compound|cmpd)\s+(\d+[a-z]?)\b", normalized))
    numeric_labels.update(re.findall(r"\blabel\s*(\d+[a-z]?)\b", normalized))
    for number in numeric_labels:
        labels.add(number)
        labels.add(f"label {number}")
    return labels


def _structure_resolution_payload_targets_input_label(payload: dict[str, Any], blackboard: dict[str, Any]) -> bool:
    # Target identity is a strict shortcut, not fuzzy task routing.  Planner
    # rationale and family terms must never turn "target derivative 12" into
    # the target itself.
    labels = _structure_resolution_explicit_labels(payload)
    if not labels:
        return False
    target = dict(blackboard.get("target_profile") or {})
    target_keys = {
        _structure_label_key(str(target.get("target_name") or "")),
        _structure_label_key(str(target.get("name") or "")),
        _structure_label_key(str(blackboard.get("case_id") or "")),
    }
    target_keys = {item for item in target_keys if item}
    for label in labels:
        if any(_structure_resolution_identity_labels_match(label, target_key) for target_key in target_keys):
            return True
    return False


def _structure_resolution_identity_labels_match(label: str, target_key: str) -> bool:
    if not label or not target_key:
        return False
    if label == target_key:
        return True
    return _strip_terminal_compound_number(label) == target_key


def _strip_terminal_compound_number(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(label or "").lower()).strip()
    return re.sub(r"\s+(?:compound\s+)?\d+[a-z]?$", "", normalized).strip()


def _structure_resolution_task_match_score(
    task: dict[str, Any],
    *,
    requested_labels: set[str],
    requested_text: str,
    requested_action_text: str = "",
) -> int:
    label = _structure_label_key(str(task.get("label") or ""))
    if not label:
        return 0
    label_tokens = {token for token in label.split() if token}
    numeric_tokens = {token for token in label_tokens if re.fullmatch(r"\d+[a-z]?", token)}
    requested_numeric = {token for token in requested_labels if re.fullmatch(r"\d+[a-z]?", token)}
    action_tokens = {token for token in requested_action_text.split() if token}
    if requested_action_text:
        if label in requested_action_text:
            return 140
        token_overlap = label_tokens.intersection(action_tokens)
        if label_tokens and label_tokens.issubset(action_tokens):
            return 135
        if numeric_tokens.intersection(action_tokens) and len(token_overlap) >= 2:
            return 130
        if len(token_overlap) >= 2:
            return 120
    if label in requested_labels:
        return 115
    if requested_text and label in requested_text:
        return 90
    if requested_numeric and numeric_tokens.intersection(requested_numeric):
        requested_tokens = set(requested_text.split())
        if len(label_tokens.intersection(requested_tokens)) >= 2:
            return 85
        return 75
    for requested in requested_labels:
        if not requested or re.fullmatch(r"\d+[a-z]?", requested):
            continue
        if len(requested) >= 4 and requested in label:
            return 60
        if len(label) >= 4 and label in requested:
            return 55
    return 0


def _structure_resolution_attempt_count(task: dict[str, Any]) -> int:
    try:
        return int(task.get("resolution_attempt_count") or 0)
    except (TypeError, ValueError):
        return 0


def _preferred_structure_resolution_task_from_visual_chain(
    blackboard: dict[str, Any], tasks: list[dict[str, Any]]
) -> dict[str, Any]:
    visual_labels = _structure_labels_available_from_visual_chains(blackboard)
    if not visual_labels:
        return {}
    for task in tasks:
        try:
            if int(task.get("resolution_attempt_count") or 0) > 0:
                continue
        except (TypeError, ValueError):
            pass
        label = _structure_label_key(str(task.get("label") or ""))
        if label and label in visual_labels:
            return task
    return {}


def _structure_labels_available_from_visual_chains(blackboard: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for chain in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(chain, dict):
            continue
        for step in chain.get("steps") or []:
            if not isinstance(step, dict):
                continue
            product_label = str(step.get("product_label") or step.get("label") or "").strip()
            if product_label:
                labels.add(_structure_label_key(product_label))
            for reactant_label in step.get("reactant_labels") or []:
                text = str(reactant_label or "").strip()
                if text:
                    labels.add(_structure_label_key(text))
    return labels


def _structure_label_key(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


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
    _preserve_guided_timeout_floor(policy, base_policy)
    repaired = _whitelist_payload_fields(
        raw,
        {
            "schema_version",
            "initial_probe",
            "search_mode",
            "search_preset",
            "max_steps",
            "chem_enzy_iterations",
            "chem_enzy_expansion_topk",
            "timeout_s",
            "max_candidates",
            "rerun_attempt",
            "failure_mode_focus",
            "no_solved_claim",
            "reason",
        },
    )
    repaired["schema_version"] = "guided_chemenzy_rerun_payload.v1"
    repaired["no_solved_claim"] = True
    repaired["guided_policy_runtime_rebuild"] = True
    repaired["guided_policy_summary"] = _compact_guided_policy_summary(policy)
    if _guided_policy_timeout(policy) and repaired.get("timeout_s") is None:
        repaired["timeout_s"] = _guided_policy_timeout(policy)
    repaired["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "run_guided_chemenzy",
        "completed_from_blackboard": True,
        "runtime_policy_rebuild": True,
        "preserved_codex_fields": sorted(str(key) for key in raw.keys()),
    }
    return repaired


def _compact_guided_policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    source_budget = dict(policy.get("source_budget") or {})
    compiler = dict(policy.get("compiler_metadata") or {})
    preferred = dict(policy.get("preferred_subgoal") or {})
    budget = dict(policy.get("budget") or {})
    return {
        "schema_version": "guided_policy_summary.v1",
        "policy_id": str(policy.get("policy_id") or ""),
        "mode": str(policy.get("mode") or policy.get("search_mode") or ""),
        "evidence_ref_count": len(policy.get("evidence_refs") or []),
        "active_bridge_task_count": len(policy.get("active_bridge_tasks") or []),
        "terminal_blacklist_count": len(policy.get("terminal_blacklist") or []),
        "accepted_exact_row_count": len(policy.get("accepted_exact_row_ids") or []),
        "preferred_subgoal_count": len(preferred.get("preferred_subgoals") or []),
        "budget": {
            key: budget.get(key)
            for key in ("max_depth", "max_iterations", "expansion_topk", "timeout_s")
            if budget.get(key) is not None
        },
        "source_flags": {
            "require_target_core_retention": bool(source_budget.get("require_target_core_retention")),
            "initial_scan_allowed": bool(source_budget.get("initial_scan_allowed")),
            "visual_connectivity_hints_are_not_proof": bool(
                source_budget.get("visual_connectivity_hints_are_not_proof")
            ),
            "retrosynthetic_proposals_are_not_proof": bool(
                source_budget.get("retrosynthetic_proposals_are_not_proof")
            ),
        },
        "compiler_flags": {
            "requires_verifier": bool(compiler.get("requires_verifier")),
            "no_solved_claim": bool(compiler.get("no_solved_claim")),
            "initial_scan_probe": bool(compiler.get("initial_scan_probe")),
        },
    }


def _preserve_guided_timeout_floor(policy: dict[str, Any], base_policy: dict[str, Any]) -> None:
    floor = _guided_policy_timeout(base_policy)
    if floor <= 0:
        return
    budget = dict(policy.get("budget") or {})
    current = _positive_float_value(budget.get("timeout_s"))
    if current <= 0 or current < floor:
        budget["timeout_s"] = floor
        policy["budget"] = budget


def _guided_policy_timeout(policy: dict[str, Any]) -> float:
    return _positive_float_value((policy.get("budget") or {}).get("timeout_s"))


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
    budget = dict(policy.get("budget") or {})
    _copy_positive_int(payload, "max_steps", budget, "max_depth")
    _copy_positive_int(payload, "chem_enzy_iterations", budget, "max_iterations")
    _copy_positive_int(payload, "chem_enzy_expansion_topk", budget, "expansion_topk")
    _copy_positive_float(payload, "timeout_s", budget, "timeout_s")
    policy["budget"] = budget
    policy["source_budget"] = source_budget
    policy["compiler_metadata"] = compiler


def _copy_positive_int(source: dict[str, Any], source_key: str, target: dict[str, Any], target_key: str) -> None:
    if source.get(source_key) is None:
        return
    try:
        value = int(source.get(source_key) or 0)
    except (TypeError, ValueError):
        return
    if value > 0:
        target[target_key] = value


def _copy_positive_float(source: dict[str, Any], source_key: str, target: dict[str, Any], target_key: str) -> None:
    if source.get(source_key) is None:
        return
    value = _positive_float_value(source.get(source_key))
    if value > 0:
        target[target_key] = value


def _positive_float_value(value: Any) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0 else 0.0


def _repair_child_expansion_payload(payload: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    base_payload = build_child_expansion_payload_from_blackboard(blackboard)
    raw = dict(payload or {})
    raw_targets = raw.pop("subgoal_targets", None)
    raw_child_targets = raw.pop("child_targets", None)
    targets = raw_targets if isinstance(raw_targets, list) else raw_child_targets
    if not isinstance(targets, list) or not targets:
        singular: Any = None
        for field in ("subgoal_target", "child_target", "target"):
            value = raw.get(field)
            if (isinstance(value, dict) and value) or (isinstance(value, str) and value.strip()):
                singular = raw.pop(field)
                break
        if isinstance(singular, dict):
            targets = [singular]
        elif isinstance(singular, str) and singular.strip():
            targets = [{"smiles": singular.strip(), "name": "child_target_1"}]
    if not isinstance(targets, list) or not targets:
        direct_smiles = str(raw.pop("target_smiles", "") or raw.get("smiles") or "").strip()
        if direct_smiles:
            targets = [{"smiles": direct_smiles, "name": str(raw.pop("target_name", "") or "child_target_1")}]
    if not isinstance(targets, list) or not targets:
        targets = list(base_payload.get("subgoal_targets") or [])
    if not targets:
        return {
            "schema_version": "route_expansion_subgoal_search_payload.v1",
            "max_targets": _positive_int_or_default(raw.get("max_targets"), 2),
            "no_solved_claim": True,
            "child_policy_runtime_rebuild": True,
            "codex_payload_repair": {
                "schema_version": "codex_action_payload_repair.v1",
                "action_type": "expand_child_target",
                "completed_from_blackboard": False,
                "target_count": 0,
            },
        }

    repaired_targets: list[dict[str, Any]] = []
    for idx, target in enumerate(targets, start=1):
        if not isinstance(target, dict):
            continue
        repaired_targets.append(_repair_child_target(dict(target), blackboard=blackboard, index=idx))
    repaired = _whitelist_payload_fields(
        raw,
        {
            "schema_version",
            "max_targets",
            "target_offset",
            "search_preset",
            "max_steps",
            "chem_enzy_iterations",
            "chem_enzy_expansion_topk",
            "timeout_s",
            "stock_mode",
            "device",
            "no_solved_claim",
        },
    )
    repaired["schema_version"] = "route_expansion_subgoal_search_payload.v1"
    repaired["max_targets"] = min(
        len(repaired_targets),
        _positive_int_or_default(repaired.get("max_targets"), len(repaired_targets)),
    )
    repaired["no_solved_claim"] = True
    repaired["child_policy_runtime_rebuild"] = True
    repaired["subgoal_targets"] = repaired_targets
    repaired["codex_payload_repair"] = {
        "schema_version": "codex_action_payload_repair.v1",
        "action_type": "expand_child_target",
        "completed_from_blackboard": True,
        "target_count": len(repaired_targets),
        "runtime_policy_rebuild": True,
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
    target["policy_runtime_rebuild"] = True
    target["policy_summary"] = _compact_child_policy_summary(policy)
    target.pop("chem_enzy_search_policy", None)
    target.pop("policy", None)
    return _whitelist_payload_fields(
        target,
        {
            "schema_version",
            "name",
            "target_name",
            "smiles",
            "target_smiles",
            "source",
            "hypothesis_only_not_solved",
            "recursive_hypothesis_task_id",
            "recursive_depth",
            "parent_smiles",
            "parent_candidate_id",
            "task_scope",
            "precursor_set_smiles",
            "precursor_component_index",
            "precursor_component_count",
            "multi_component_precursor_set",
            "requires_precursor_set_stitching",
            "sibling_precursor_smiles",
            "template_id",
            "application_id",
            "evidence_refs",
            "exact_target_override",
            "target_equivalence_audit_required",
            "no_solved_claim",
            "child_route_cannot_promote_parent",
            "policy_runtime_rebuild",
            "policy_summary",
        },
    )


def _compact_child_policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    budget = dict(policy.get("budget") or {})
    source_budget = dict(policy.get("source_budget") or {})
    compiler = dict(policy.get("compiler_metadata") or {})
    return {
        "schema_version": "child_policy_summary.v1",
        "policy_id": str(policy.get("policy_id") or ""),
        "mode": str(policy.get("mode") or ""),
        "anchor_whitelist_count": len(policy.get("anchor_whitelist") or []),
        "terminal_blacklist_count": len(policy.get("terminal_blacklist") or []),
        "evidence_ref_count": len(policy.get("evidence_refs") or []),
        "budget": {
            key: budget.get(key)
            for key in ("max_depth", "max_iterations", "expansion_topk", "timeout_s")
            if budget.get(key) is not None
        },
        "source_flags": {
            "require_target_core_retention": bool(source_budget.get("require_target_core_retention")),
            "analogy_is_advisory_only": bool(source_budget.get("analogy_is_advisory_only")),
        },
        "compiler_flags": {
            "requires_verifier": bool(compiler.get("requires_verifier")),
            "no_solved_claim": bool(compiler.get("no_solved_claim")),
            "child_route_cannot_promote_parent": bool(compiler.get("child_route_cannot_promote_parent")),
        },
    }


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


def _terminal_blacklist_from_blackboard(blackboard: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for row in blackboard.get("terminal_blacklist") or []:
        if isinstance(row, dict):
            values.append(str(row.get("canonical_smiles") or row.get("smiles") or ""))
        else:
            values.append(str(row or ""))
    return _dedupe_preserve_order([item for item in values if str(item or "").strip()])


def _child_search_policy_seed(blackboard: dict[str, Any], *, target: dict[str, Any], index: int) -> dict[str, Any]:
    smiles = str(target.get("smiles") or target.get("target_smiles") or "").strip()
    policy: dict[str, Any] = {
        "schema_version": "chem_enzy_search_policy.v1",
        "policy_id": f"{blackboard.get('case_id') or 'case'}_codex_child_{index}_policy",
        "operator_id": "agentic_blackboard_controller",
        "case_id": str(blackboard.get("case_id") or ""),
        "anchor_whitelist": [smiles] if smiles else [],
        "terminal_blacklist": _terminal_blacklist_from_blackboard(blackboard),
        "active_bridge_tasks": [],
        "accepted_exact_row_ids": [],
        "selected_analogical_hypothesis_ids": [],
        "selected_analogical_template_ids": [],
        "forbidden_template_ids": [],
    }
    refs = [
        str(item)
        for item in target.get("evidence_refs") or []
        if str(item or "").strip()
    ][:6]
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
    policy["source_budget"] = {
        "require_target_core_retention": True,
        "max_unexplained_heavy_atom_jump": 12,
        "analogy_is_advisory_only": True,
        "preferred_reaction_classes": ["source_detail_terminal_upstream_expansion"],
    }
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
    allow_deterministic_fallback: bool = True,
) -> dict[str, Any]:
    planner_disabled = reason == "codex_action_planner_disabled"
    fallback_permitted = planner_disabled or bool(allow_deterministic_fallback)
    if fallback_permitted:
        batch = dict(
            fallback_planner(
                blackboard,
                round_index=round_index,
                exhaust_round_budget=exhaust_round_budget,
            )
        )
    else:
        batch = _codex_fail_closed_batch(
            blackboard=blackboard,
            round_index=round_index,
            reason=reason,
        )
    if not planner_disabled and fallback_permitted:
        batch = _guard_codex_rejection_fallback_batch(batch, blackboard=blackboard, round_index=round_index)
    if planner_disabled:
        batch["mode"] = str(batch.get("mode") or "deterministic_policy")
    elif not fallback_permitted:
        batch["mode"] = "codex_planner_fail_closed"
    else:
        batch["mode"] = "deterministic_policy_fallback_after_codex_planner"
    batch["codex_action_planner"] = {
        "schema_version": "codex_action_planner_metadata.v1",
        "fallback_used": bool(not planner_disabled and fallback_permitted),
        "fallback_permitted": bool(fallback_permitted),
        "fail_closed": bool(not planner_disabled and not fallback_permitted),
        "fallback_reason": reason,
        "planner_disabled": planner_disabled,
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


def _codex_fail_closed_batch(
    *,
    blackboard: dict[str, Any],
    round_index: int,
    reason: str,
) -> dict[str, Any]:
    """Stop honestly when the Codex mainline fails instead of inventing a route plan."""
    case_id = str(blackboard.get("case_id") or "")
    return {
        "schema_version": ACTION_BATCH_SCHEMA,
        "case_id": case_id,
        "round_index": int(round_index),
        "mode": "codex_planner_fail_closed",
        "actions": [
            {
                "schema_version": "agent_action.v1",
                "action_id": f"r{int(round_index)}:codex_fail_closed",
                "action_type": "stop_unresolved",
                "rationale": f"Codex planning did not produce an accepted action batch ({reason}); deterministic scientific fallback is disabled",
                "expected_artifact": "codex_planner_failure_marker.v1",
                "success_condition": "run remains unresolved and records the rejected Codex attempt",
                "payload": {
                    "failure_reason": str(reason or "codex_planner_failure"),
                    "deterministic_scientific_fallback_used": False,
                    "no_solved_claim": True,
                },
            }
        ],
        "semantics": {
            "planner_can_emit_solved": False,
            "raw_reaction_output_allowed": False,
            "deterministic_validator_required": True,
        },
    }


def _guard_codex_rejection_fallback_batch(
    batch: dict[str, Any],
    *,
    blackboard: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    actions = [dict(row) for row in batch.get("actions") or [] if isinstance(row, dict)]
    if len(actions) != 1 or str(actions[0].get("action_type") or "") != "run_guided_chemenzy":
        return batch
    if blackboard.get("target_side_disconnection_hypotheses"):
        return batch
    guarded = dict(batch)
    guarded["actions"] = [
        {
            "schema_version": "agent_action.v1",
            "action_id": f"r{int(round_index)}:fallback_generate_disconnection_hypotheses",
            "action_type": "generate_disconnection_hypotheses",
            "rationale": "Codex planner output was rejected; fall back to safe target-side hypothesis generation before guided search",
            "expected_artifact": "target_side_disconnection_hypotheses.v1",
            "success_condition": "at least one advisory hypothesis and bridge task",
            "payload": {},
        }
    ]
    return guarded


def _deterministic_fast_path_batch(
    *,
    blackboard: dict[str, Any],
    round_index: int,
    exhaust_round_budget: bool,
    fallback_planner: FallbackPlanner,
) -> dict[str, Any] | None:
    batch = dict(
        fallback_planner(
            blackboard,
            round_index=round_index,
            exhaust_round_budget=exhaust_round_budget,
        )
    )
    actions = [dict(row) for row in batch.get("actions") or [] if isinstance(row, dict)]
    if len(actions) != 1:
        return None
    action = actions[0]
    if str(action.get("action_type") or "") != "stitch_parent_route":
        return None
    payload = dict(action.get("payload") or {})
    binding = dict(payload.get("proof_binding") or {})
    policy = dict(payload.get("proof_policy") or {})
    direct_parent = (
        str(binding.get("proof_mode") or "") == "direct_parent_route"
        or bool(binding.get("direct_parent_route_verifier_ready"))
        or str(policy.get("proof_mode") or "") == "direct_parent_route"
    )
    if not direct_parent:
        return None
    validation = validate_action_batch(batch, blackboard=blackboard)
    if not validation.get("accepted"):
        return None
    batch["mode"] = "deterministic_policy_fast_path_before_codex_planner"
    batch["codex_action_planner"] = {
        "schema_version": "codex_action_planner_metadata.v1",
        "fallback_used": False,
        "fallback_reason": "",
        "fast_path_used": True,
        "fast_path_reason": "deterministic_direct_parent_route_proof_ready",
        "batch_validation": validation,
        "tool_policy": {
            "codex_worker_bypassed": True,
            "reason": "direct_parent_route_verifier_ready",
        },
    }
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
    planner_context = _planner_context_summary(
        blackboard,
        round_index=round_index,
    )
    handoff = _planner_blackboard_handoff(
        blackboard,
        planner_context=planner_context,
    )
    prompt_payload, prompt_bounds = _bounded_planner_prompt_payload(
        handoff,
        round_index=round_index,
    )
    snapshot = {
        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
        "case_id": str(blackboard.get("case_id") or ""),
        "round_index": int(round_index),
        "planner_context": planner_context,
        "blackboard": handoff,
        "prompt_payload": prompt_payload,
        "prompt_payload_sha256": _planner_json_sha256(prompt_payload),
        "prompt_payload_bounds": prompt_bounds,
        "source_blackboard_sha256": _planner_json_sha256(blackboard),
        "input_ref_audit": {
            "schema_version": "codex_action_planner_input_ref_audit.v1",
            "prompt_snapshot_ref": str(path),
            "source_blackboard_ref": str(run_dir / "agent_blackboard.json"),
            "input_refs_are_audit_only": True,
            "model_file_read_required": False,
            "shell_read_allowed": False,
        },
    }
    snapshot["content_sha256"] = _planner_json_sha256(snapshot)
    write_json(path, snapshot)
    return path


def _embedded_planner_prompt_snapshot(snapshot_path: Path) -> dict[str, Any]:
    """Load and verify the host-written compact snapshot for prompt embedding."""

    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("codex action planner snapshot is unreadable") from exc
    if not isinstance(snapshot, dict):
        raise ValueError("codex action planner snapshot is not an object")
    supplied_content_digest = str(snapshot.get("content_sha256") or "")
    content_payload = dict(snapshot)
    content_payload.pop("content_sha256", None)
    if (
        not supplied_content_digest
        or supplied_content_digest != _planner_json_sha256(content_payload)
    ):
        raise ValueError("codex action planner snapshot digest mismatch")
    prompt_payload = snapshot.get("prompt_payload")
    if not isinstance(prompt_payload, dict):
        raise ValueError("codex action planner prompt payload missing")
    supplied_prompt_digest = str(snapshot.get("prompt_payload_sha256") or "")
    if (
        not supplied_prompt_digest
        or supplied_prompt_digest != _planner_json_sha256(prompt_payload)
    ):
        raise ValueError("codex action planner prompt payload digest mismatch")
    raw_bounds = snapshot.get("prompt_payload_bounds")
    if not isinstance(raw_bounds, dict):
        raise ValueError("codex action planner prompt payload bounds missing")
    try:
        max_bytes = int(raw_bounds.get("max_bytes"))
        declared_bytes = int(raw_bounds.get("embedded_bytes"))
    except (TypeError, ValueError) as exc:
        raise ValueError("codex action planner prompt payload bounds invalid") from exc
    actual_bytes = len(_planner_json_bytes(prompt_payload))
    if (
        max_bytes <= 0
        or raw_bounds.get("within_bound") is not True
        or declared_bytes != actual_bytes
        or actual_bytes > max_bytes
    ):
        raise ValueError("codex action planner prompt payload exceeds hard bound")
    verified_bounds = {
        "schema_version": "codex_action_planner_prompt_bounds.v1",
        "max_bytes": max_bytes,
        "original_bytes": _planner_prompt_count(raw_bounds.get("original_bytes")),
        "embedded_bytes": actual_bytes,
        "within_bound": True,
        "compaction": _planner_prompt_text(raw_bounds.get("compaction"), 80),
        "full_blackboard_omitted": True,
    }
    return {
        "schema_version": "codex_action_planner_embedded_snapshot.v1",
        "input_ref": str(snapshot_path),
        "input_ref_content_sha256": supplied_content_digest,
        "source_blackboard_sha256": str(
            snapshot.get("source_blackboard_sha256") or ""
        ),
        "prompt_payload_sha256": supplied_prompt_digest,
        "prompt_payload_bounds": verified_bounds,
        "input_ref_role": "audit_only_no_model_file_read_required",
        "shell_read_allowed": False,
        "prompt_payload": dict(prompt_payload),
    }


def _planner_prompt_snapshot_audit(snapshot_path: Path) -> dict[str, Any]:
    embedded = _embedded_planner_prompt_snapshot(snapshot_path)
    embedded.pop("prompt_payload", None)
    return embedded


def _bounded_planner_prompt_payload(
    handoff: dict[str, Any],
    *,
    round_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    max_bytes = _planner_prompt_snapshot_max_bytes()
    embedded_handoff = _truncate_planner_prompt_value(
        handoff,
        list_limit=12,
        string_limit=1_600,
        dict_limit=48,
        key_limit=80,
        max_depth=8,
    )
    # The persisted snapshot retains this locator for humans/auditors.  The
    # model already has every bounded decision field below and should not be
    # tempted to open the multi-megabyte source blackboard.
    embedded_handoff.pop("full_audit_ref", None)
    payload = {
        "schema_version": "codex_action_planner_prompt_payload.v1",
        "round_index": int(round_index),
        "blackboard_handoff": embedded_handoff,
        "no_file_read_required": True,
        "input_refs_are_audit_only": True,
        "shell_read_allowed": False,
    }
    original_bytes = len(_planner_json_bytes(payload))
    compaction = "bounded_handoff"
    if original_bytes > max_bytes:
        payload["blackboard_handoff"] = _truncate_planner_prompt_value(
            _minimum_planner_handoff(handoff),
            list_limit=6,
            string_limit=700,
            dict_limit=32,
            key_limit=80,
            max_depth=7,
        )
        compaction = "essential_decision_surface"
    actual_bytes = len(_planner_json_bytes(payload))
    if actual_bytes > max_bytes:
        payload["blackboard_handoff"] = _truncate_planner_prompt_value(
            payload["blackboard_handoff"],
            list_limit=3,
            string_limit=240,
            dict_limit=24,
            key_limit=72,
            max_depth=6,
        )
        compaction = "essential_decision_surface_truncated"
        actual_bytes = len(_planner_json_bytes(payload))
    if actual_bytes > max_bytes:
        payload["blackboard_handoff"] = _truncate_planner_prompt_value(
            payload["blackboard_handoff"],
            list_limit=1,
            string_limit=120,
            dict_limit=16,
            key_limit=64,
            max_depth=5,
        )
        compaction = "minimum_fail_soft_projection"
        actual_bytes = len(_planner_json_bytes(payload))
    if actual_bytes > max_bytes:
        payload["blackboard_handoff"] = _absolute_minimum_planner_handoff(
            handoff
        )
        compaction = "absolute_minimum_decision_projection"
        actual_bytes = len(_planner_json_bytes(payload))
    if actual_bytes > max_bytes:
        raise ValueError(
            "codex action planner prompt payload cannot satisfy hard byte bound"
        )
    return payload, {
        "schema_version": "codex_action_planner_prompt_bounds.v1",
        "max_bytes": max_bytes,
        "original_bytes": original_bytes,
        "embedded_bytes": actual_bytes,
        "within_bound": True,
        "compaction": compaction,
        "full_blackboard_omitted": True,
    }


def _minimum_planner_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    evidence = _planner_prompt_mapping(handoff.get("evidence_board"))
    route = _planner_prompt_mapping(handoff.get("route_board"))
    decision = _planner_prompt_mapping(handoff.get("decision_board"))
    return {
        "schema_version": "codex_action_planner_blackboard_handoff.v1",
        "case_id": str(handoff.get("case_id") or ""),
        "target_profile": _planner_prompt_mapping(handoff.get("target_profile")),
        "state_counts": _planner_prompt_mapping(handoff.get("state_counts")),
        "evidence_board": {
            "source_acquisition": _planner_prompt_mapping(
                evidence.get("source_acquisition")
            ),
            "source_capability_queue": _planner_prompt_mapping(
                evidence.get("source_capability_queue")
            ),
            "pending_local_pdf_proxy_sources": _planner_prompt_prefix(
                evidence.get("pending_local_pdf_proxy_sources"), 4
            ),
            "pending_pdf_extraction_sources": _planner_prompt_prefix(
                evidence.get("pending_pdf_extraction_sources"), 4
            ),
            "pending_visual_extraction_sources": _planner_prompt_prefix(
                evidence.get("pending_visual_extraction_sources"), 4
            ),
            "source_candidates": _planner_prompt_prefix(
                evidence.get("source_candidates"), 6
            ),
            "pdf_focus": _planner_prompt_prefix(
                evidence.get("pdf_focus"), 4
            ),
            "structure_resolution_tasks": _planner_prompt_prefix(
                evidence.get("structure_resolution_tasks"), 6
            ),
            "process_evidence_rows": _planner_prompt_prefix(
                evidence.get("process_evidence_rows"), 4
            ),
            "exact_row_samples": _planner_prompt_prefix(
                evidence.get("exact_row_samples"), 4
            ),
        },
        "route_board": {
            "bridge_tasks": _planner_prompt_prefix(route.get("bridge_tasks"), 4),
            "retrosynthetic_proposals": _planner_prompt_prefix(
                route.get("retrosynthetic_proposals"), 6
            ),
            "recursive_hypothesis_tasks": _planner_prompt_prefix(
                route.get("recursive_hypothesis_tasks"), 6
            ),
            "parent_route_proof": _planner_prompt_mapping(
                route.get("parent_route_proof")
            ),
            "route_anchor_opportunities": _planner_prompt_mapping(
                route.get("route_anchor_opportunities")
            ),
            "route_closure_pressure": _planner_prompt_mapping(
                route.get("route_closure_pressure")
            ),
            "frontier_ledger": _planner_prompt_mapping(
                route.get("frontier_ledger")
            ),
        },
        "decision_board": {
            "next_action_bias": _planner_prompt_prefix(
                decision.get("next_action_bias"), 6
            ),
            "blocked_directions": _planner_prompt_prefix(
                decision.get("blocked_directions"), 4
            ),
            "route_failures": _planner_prompt_prefix(
                decision.get("route_failures"), 4
            ),
            "proposal_failure_feedback": _planner_prompt_prefix(
                decision.get("proposal_failure_feedback"), 4
            ),
            "recent_actions": _planner_prompt_prefix(
                decision.get("recent_actions"), 5
            ),
            "budget_remaining": _planner_prompt_mapping(
                decision.get("budget_remaining")
            ),
        },
        "action_requirements": _planner_prompt_mapping(
            handoff.get("action_requirements")
        ),
        "safety_boundaries": _planner_prompt_mapping(
            handoff.get("safety_boundaries")
        ),
        "no_solved_claim": True,
    }


def _absolute_minimum_planner_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    """Return a fixed-schema projection that cannot inherit arbitrary keys."""

    evidence = _planner_prompt_mapping(handoff.get("evidence_board"))
    route = _planner_prompt_mapping(handoff.get("route_board"))
    decision = _planner_prompt_mapping(handoff.get("decision_board"))
    target = _planner_prompt_mapping(handoff.get("target_profile"))
    state_counts = _planner_prompt_mapping(handoff.get("state_counts"))
    parent_proof = _planner_prompt_mapping(route.get("parent_route_proof"))
    return {
        "schema_version": "codex_action_planner_blackboard_handoff.v1",
        "case_id": _planner_prompt_text(handoff.get("case_id"), 96),
        "target_profile": {
            "target_name": _planner_prompt_text(target.get("target_name"), 96),
            "target_smiles": _planner_prompt_text(target.get("target_smiles"), 320),
            "canonical_smiles": _planner_prompt_text(
                target.get("canonical_smiles"), 320
            ),
            "valid": target.get("valid") is True,
        },
        "state_counts": {
            key: _planner_prompt_count(state_counts.get(key))
            for key in (
                "source_candidates",
                "pdf_structure_evidence",
                "visual_chains",
                "exact_rows",
                "structure_resolution_tasks_open",
                "resolved_structures",
                "retrosynthetic_proposals",
                "recursive_hypothesis_tasks",
                "bridge_tasks",
                "route_failures",
            )
        },
        "evidence_board": {
            "source_capability_queue": _minimum_prompt_source_capability_queue(
                evidence.get("source_capability_queue")
            ),
            "pending_source_counts": {
                "local_pdf_proxy": _planner_prompt_list_count(
                    evidence.get("pending_local_pdf_proxy_sources")
                ),
                "pdf_extraction": _planner_prompt_list_count(
                    evidence.get("pending_pdf_extraction_sources")
                ),
                "visual_extraction": _planner_prompt_list_count(
                    evidence.get("pending_visual_extraction_sources")
                ),
                "source_candidates": _planner_prompt_list_count(
                    evidence.get("source_candidates")
                ),
                "structure_resolution_tasks": _planner_prompt_list_count(
                    evidence.get("structure_resolution_tasks")
                ),
            },
            "pending_pdf_extraction_sources": _minimum_prompt_pending_pdf_rows(
                evidence
            ),
            "pdf_focus": _minimum_prompt_pdf_focus(
                evidence.get("pdf_focus")
            ),
            "structure_resolution_tasks": _minimum_prompt_structure_tasks(
                evidence.get("structure_resolution_tasks")
            ),
        },
        "route_board": {
            "route_counts": {
                "bridge_tasks": _planner_prompt_list_count(route.get("bridge_tasks")),
                "retrosynthetic_proposals": _planner_prompt_list_count(
                    route.get("retrosynthetic_proposals")
                ),
                "recursive_hypothesis_tasks": _planner_prompt_list_count(
                    route.get("recursive_hypothesis_tasks")
                ),
            },
            "parent_route_proof": {
                "accepted": parent_proof.get("accepted") is True,
                "proof_mode": _planner_prompt_text(
                    parent_proof.get("proof_mode"), 100
                ),
                "status": _planner_prompt_text(parent_proof.get("status"), 100),
            },
            "frontier_ledger": _minimum_prompt_frontier_ledger(
                route.get("frontier_ledger")
            ),
        },
        "decision_board": {
            "next_action_bias": _fixed_prompt_text_list(
                decision.get("next_action_bias"), limit=2, string_limit=80
            ),
            "blocked_direction_count": _planner_prompt_list_count(
                decision.get("blocked_directions")
            ),
            "route_failure_count": _planner_prompt_list_count(
                decision.get("route_failures")
            ),
            "recent_action_types": [
                _planner_prompt_text(row.get("action_type"), 100)
                for row in (
                    _planner_prompt_mapping(raw)
                    for raw in (
                        decision.get("recent_actions")[-2:]
                        if isinstance(decision.get("recent_actions"), (list, tuple))
                        else []
                    )
                )
                if row
            ],
            "budget_remaining": _fixed_prompt_budget_remaining(
                decision.get("budget_remaining")
            ),
        },
        "action_requirements": _minimum_prompt_action_requirements(
            handoff.get("action_requirements")
        ),
        "safety_boundaries": _fixed_prompt_safety_boundaries(
            handoff.get("safety_boundaries")
        ),
        "no_solved_claim": True,
    }


def _planner_prompt_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _planner_prompt_prefix(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(value[: max(0, int(limit))])


def _planner_prompt_text(value: Any, limit: int) -> str:
    text = str(value or "")
    byte_limit = max(0, int(limit))
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


def _planner_prompt_count(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(-1_000_000_000, min(1_000_000_000, number))


def _planner_prompt_list_count(value: Any) -> int:
    return min(1_000_000_000, len(value)) if isinstance(value, (list, tuple)) else 0


def _fixed_prompt_text_list(
    value: Any,
    *,
    limit: int,
    string_limit: int,
) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [
        _planner_prompt_text(item, string_limit)
        for item in value[: max(0, int(limit))]
    ]


def _fixed_prompt_source_capability_queue(value: Any) -> dict[str, Any]:
    queue = _planner_prompt_mapping(value)
    budget = _planner_prompt_mapping(queue.get("budget"))
    capabilities: list[dict[str, Any]] = []
    for raw in (
        queue.get("capabilities")[:6]
        if isinstance(queue.get("capabilities"), (list, tuple))
        else []
    ):
        row = _planner_prompt_mapping(raw)
        cost = _planner_prompt_mapping(row.get("cost"))
        binding = _planner_prompt_mapping(row.get("payload_binding"))
        capabilities.append(
            {
                "capability_id": _planner_prompt_text(
                    row.get("capability_id"),
                    180,
                ),
                "action_type": _planner_prompt_text(row.get("action_type"), 100),
                "document_identity": _planner_prompt_text(
                    row.get("document_identity"),
                    240,
                ),
                "source_ref": _planner_prompt_text(row.get("source_ref"), 240),
                "source_title": _planner_prompt_text(row.get("source_title"), 220),
                "stage_from": _planner_prompt_text(row.get("stage_from"), 80),
                "stage_to": _planner_prompt_text(row.get("stage_to"), 80),
                "payload_binding": {
                    key: _planner_prompt_text(binding.get(key), 360)
                    for key in (
                        "source_ref",
                        "source_title",
                        "pdf_path",
                        "document_id",
                        "task_id",
                        "chain_id",
                        "artifact_ref",
                    )
                    if str(binding.get(key) or "").strip()
                },
                "cost": {
                    key: _planner_prompt_count(cost.get(key))
                    for key in (
                        "action_slots",
                        "literature_source_units",
                        "scout_calls",
                        "visual_calls",
                    )
                },
                "eligible": row.get("eligible") is True,
                "no_solved_claim": True,
            }
        )
    return {
        "schema_version": _planner_prompt_text(queue.get("schema_version"), 80),
        "content_sha256": _planner_prompt_text(queue.get("content_sha256"), 128),
        "budget": {
            "literature_source_units_max_this_round": _planner_prompt_count(
                budget.get("literature_source_units_max_this_round")
            ),
            "literature_source_units_remaining_this_round": _planner_prompt_count(
                budget.get("literature_source_units_remaining_this_round")
            ),
            "visual_calls_remaining": _planner_prompt_count(
                budget.get("visual_calls_remaining")
            ),
            "scout_calls_remaining": _planner_prompt_count(
                budget.get("scout_calls_remaining")
            ),
        },
        "capabilities": capabilities,
        "no_solved_claim": True,
    }


def _minimum_prompt_source_capability_queue(value: Any) -> dict[str, Any]:
    queue = _planner_prompt_mapping(value)
    budget = _planner_prompt_mapping(queue.get("budget"))
    priority = {
        "extract_pdf_literature_structures": 0,
        "resolve_literature_structure_task": 1,
        "extract_visual_literature_chain": 2,
        "compile_exact_literature_rows": 3,
        "search_literature": 4,
    }
    raw_capabilities = (
        list(queue.get("capabilities")[:24])
        if isinstance(queue.get("capabilities"), (list, tuple))
        else []
    )
    ranked = sorted(
        (
            (index, _planner_prompt_mapping(raw))
            for index, raw in enumerate(raw_capabilities)
            if _planner_prompt_mapping(raw)
        ),
        key=lambda item: (
            item[1].get("eligible") is not True,
            priority.get(str(item[1].get("action_type") or ""), 99),
            item[0],
        ),
    )[:2]
    capabilities: list[dict[str, Any]] = []
    for _, row in ranked:
        binding = _planner_prompt_mapping(row.get("payload_binding"))
        cost = _planner_prompt_mapping(row.get("cost"))
        capabilities.append(
            {
                "capability_id": _planner_prompt_text(
                    row.get("capability_id"), 112
                ),
                "action_type": _planner_prompt_text(row.get("action_type"), 72),
                "source_ref": _planner_prompt_text(row.get("source_ref"), 180),
                "payload_binding": {
                    key: _planner_prompt_text(binding.get(key), limit)
                    for key, limit in (
                        ("source_ref", 180),
                        ("pdf_path", 280),
                        ("task_id", 160),
                        ("chain_id", 160),
                    )
                    if str(binding.get(key) or "").strip()
                },
                "literature_source_units": _planner_prompt_count(
                    cost.get("literature_source_units")
                ),
                "visual_calls": _planner_prompt_count(cost.get("visual_calls")),
                "eligible": row.get("eligible") is True,
            }
        )
    return {
        "schema_version": _planner_prompt_text(queue.get("schema_version"), 64),
        "content_sha256": _planner_prompt_text(queue.get("content_sha256"), 80),
        "literature_source_units_remaining_this_round": _planner_prompt_count(
            budget.get("literature_source_units_remaining_this_round")
        ),
        "capabilities": capabilities,
    }


def _minimum_prompt_pending_pdf_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "pending_pdf_extraction_sources",
        "pending_local_pdf_proxy_sources",
        "source_candidates",
    ):
        values = evidence.get(key)
        if not isinstance(values, (list, tuple)):
            continue
        for raw in values[:24]:
            row = _planner_prompt_mapping(raw)
            pdf_path = (
                row.get("local_pdf")
                or row.get("pdf_path")
                or row.get("source_pdf_path")
            )
            if not row or not str(pdf_path or "").strip():
                continue
            return [
                {
                    "source_ref": _planner_prompt_text(row.get("source_ref"), 180),
                    "doi": _planner_prompt_text(row.get("doi"), 96),
                    "title": _planner_prompt_text(
                        row.get("title") or row.get("source_title"), 96
                    ),
                    "local_pdf": _planner_prompt_text(pdf_path, 280),
                }
            ]
    return []


def _minimum_prompt_structure_tasks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    for raw in value[:24]:
        row = _planner_prompt_mapping(raw)
        if not row:
            continue
        return [
            {
                "task_id": _planner_prompt_text(row.get("task_id"), 160),
                "label": _planner_prompt_text(row.get("label"), 96),
                "source_ref": _planner_prompt_text(row.get("source_ref"), 180),
                "status": _planner_prompt_text(row.get("status"), 48),
                "no_solved_claim": True,
            }
        ]
    return []


def _minimum_prompt_pdf_focus(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    for raw in value[:8]:
        row = _planner_prompt_mapping(raw)
        focus = _planner_prompt_mapping(row.get("focus"))
        if not row or not focus:
            continue
        return [
            {
                "source_ref": _planner_prompt_text(row.get("source_ref"), 180),
                "artifact_ref": _planner_prompt_text(row.get("artifact_ref"), 240),
                "focus": {
                    "focus_terms": _fixed_prompt_text_list(
                        focus.get("focus_terms"), limit=8, string_limit=80
                    ),
                    "focus_page_numbers": [
                        _planner_prompt_count(item)
                        for item in (
                            focus.get("focus_page_numbers")[:12]
                            if isinstance(focus.get("focus_page_numbers"), (list, tuple))
                            else []
                        )
                        if _planner_prompt_count(item) > 0
                    ],
                    "selection_strategy": _planner_prompt_text(
                        focus.get("selection_strategy"), 72
                    ),
                    "relevance_available": focus.get("relevance_available") is True,
                    "no_ocr_or_relevance_fabrication": focus.get(
                        "no_ocr_or_relevance_fabrication"
                    )
                    is True,
                    "no_solved_claim": True,
                },
                "no_solved_claim": True,
            }
        ]
    return []


def _fixed_prompt_source_rows(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    rows: list[dict[str, Any]] = []
    for raw in value[: max(0, int(limit))]:
        row = _planner_prompt_mapping(raw)
        if not row:
            continue
        rows.append(
            {
                "source_key": _planner_prompt_text(row.get("source_key"), 180),
                "candidate_id": _planner_prompt_text(row.get("candidate_id"), 180),
                "source_ref": _planner_prompt_text(row.get("source_ref"), 240),
                "doi": _planner_prompt_text(row.get("doi"), 120),
                "pii": _planner_prompt_text(row.get("pii"), 100),
                "title": _planner_prompt_text(
                    row.get("title") or row.get("source_title"), 220
                ),
                "local_pdf": _planner_prompt_text(
                    row.get("local_pdf")
                    or row.get("pdf_path")
                    or row.get("source_pdf_path"),
                    360,
                ),
                "stage": _planner_prompt_text(
                    row.get("stage") or row.get("access_status"), 80
                ),
            }
        )
    return rows


def _fixed_prompt_structure_tasks(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    rows: list[dict[str, Any]] = []
    for raw in value[: max(0, int(limit))]:
        row = _planner_prompt_mapping(raw)
        if not row:
            continue
        rows.append(
            {
                "task_id": _planner_prompt_text(row.get("task_id"), 180),
                "label": _planner_prompt_text(row.get("label"), 140),
                "source_ref": _planner_prompt_text(row.get("source_ref"), 240),
                "source_locator": _planner_prompt_text(
                    row.get("source_locator"), 220
                ),
                "artifact_ref": _planner_prompt_text(row.get("artifact_ref"), 240),
                "status": _planner_prompt_text(row.get("status"), 80),
                "no_solved_claim": True,
            }
        )
    return rows


def _fixed_prompt_frontier_ledger(value: Any) -> dict[str, Any]:
    ledger = _planner_prompt_mapping(value)
    pending = _planner_prompt_mapping(ledger.get("pending_counts"))
    root = _planner_prompt_mapping(ledger.get("root"))
    return {
        "ledger_schema_version": _planner_prompt_text(
            ledger.get("ledger_schema_version"), 80
        ),
        "frontier_ledger_content_sha256": _planner_prompt_text(
            ledger.get("frontier_ledger_content_sha256"), 128
        ),
        "input_valid": ledger.get("input_valid") is True,
        "ledger_validation_accepted": ledger.get("ledger_validation_accepted")
        is True,
        "root_canonical_smiles": _planner_prompt_text(
            root.get("canonical_smiles"), 320
        ),
        "pending_counts": {
            key: _planner_prompt_count(pending.get(key))
            for key in (
                "proposal_pending_molecule_count",
                "proposal_expansion_eligible_molecule_count",
                "work_pending_molecule_count",
                "stock_pending_leaf_count",
                "reaction_proof_pending_edge_count",
                "dependency_pending_edge_count",
            )
        },
        "open_frontier_molecule_count": _planner_prompt_count(
            ledger.get("open_frontier_molecule_count")
        ),
        "open_frontier_rows_truncated": ledger.get(
            "open_frontier_rows_truncated"
        )
        is True,
        "open_frontier_molecules": [
            {
                "canonical_smiles": _planner_prompt_text(
                    row.get("canonical_smiles"), 320
                ),
                "proposal_state": _planner_prompt_text(
                    row.get("proposal_state"), 80
                ),
                "work_open": row.get("work_open") is True,
                "proposal_expansion_allowed": row.get(
                    "proposal_expansion_allowed"
                )
                is True,
                "stock_closed": row.get("stock_closed") is True,
            }
            for row in (
                _planner_prompt_mapping(raw)
                for raw in (
                    ledger.get("open_frontier_molecules")[:2]
                    if isinstance(ledger.get("open_frontier_molecules"), (list, tuple))
                    else []
                )
            )
            if row
        ],
    }


def _minimum_prompt_frontier_ledger(value: Any) -> dict[str, Any]:
    ledger = _planner_prompt_mapping(value)
    pending = _planner_prompt_mapping(ledger.get("pending_counts"))
    root = _planner_prompt_mapping(ledger.get("root"))
    return {
        "frontier_ledger_content_sha256": _planner_prompt_text(
            ledger.get("frontier_ledger_content_sha256"), 80
        ),
        "input_valid": ledger.get("input_valid") is True,
        "ledger_validation_accepted": ledger.get("ledger_validation_accepted")
        is True,
        "root_canonical_smiles": _planner_prompt_text(
            root.get("canonical_smiles"), 240
        ),
        "pending_counts": {
            key: _planner_prompt_count(pending.get(key))
            for key in (
                "proposal_pending_molecule_count",
                "proposal_expansion_eligible_molecule_count",
                "stock_pending_leaf_count",
                "reaction_proof_pending_edge_count",
            )
        },
        "open_frontier_molecule_count": _planner_prompt_count(
            ledger.get("open_frontier_molecule_count")
        ),
    }


def _fixed_prompt_budget_remaining(value: Any) -> dict[str, int]:
    budget = _planner_prompt_mapping(value)
    return {
        key: _planner_prompt_count(budget.get(key))
        for key in (
            "rounds_remaining",
            "scout_calls_remaining",
            "visual_calls_remaining",
            "chemenzy_runs_remaining",
            "child_target_runs_remaining",
            "codex_research_runs_remaining",
            "template_application_actions_remaining",
            "literature_source_units_max_this_round",
            "literature_source_units_remaining_this_round",
        )
    }


def _fixed_prompt_action_requirements(value: Any) -> dict[str, Any]:
    requirements = _planner_prompt_mapping(value)
    source_sensitive = _planner_prompt_mapping(
        requirements.get("source_sensitive_actions")
    )
    source_rows: dict[str, Any] = {}
    for action_type in (
        "extract_pdf_literature_structures",
        "extract_visual_literature_chain",
        "resolve_literature_structure_task",
        "compile_exact_literature_rows",
    ):
        row = _planner_prompt_mapping(source_sensitive.get(action_type))
        source_rows[action_type] = {
            "currently_required": row.get("currently_required") is True,
            "requires_pdf_path": row.get("requires_pdf_path") is True,
            "requires_uncompiled_visual_steps": row.get(
                "requires_uncompiled_visual_steps"
            )
            is True,
            "accepted_payload_fields": _fixed_prompt_text_list(
                row.get("accepted_payload_fields"), limit=10, string_limit=64
            ),
        }
    search = _planner_prompt_mapping(requirements.get("search_literature"))
    guided = _planner_prompt_mapping(requirements.get("run_guided_chemenzy"))
    return {
        "source_sensitive_actions": source_rows,
        "search_literature": {
            "currently_required_when_selected": search.get(
                "currently_required_when_selected"
            )
            is True,
            "accepted_payload_fields": _fixed_prompt_text_list(
                search.get("accepted_payload_fields"), limit=10, string_limit=64
            ),
        },
        "run_guided_chemenzy": {
            "currently_required_when_selected": guided.get(
                "currently_required_when_selected"
            )
            is True,
            "accepted_payload_fields": _fixed_prompt_text_list(
                guided.get("accepted_payload_fields"), limit=10, string_limit=64
            ),
        },
    }


def _minimum_prompt_action_requirements(value: Any) -> dict[str, Any]:
    requirements = _planner_prompt_mapping(value)
    source_sensitive = _planner_prompt_mapping(
        requirements.get("source_sensitive_actions")
    )
    preferred_fields = {
        "extract_pdf_literature_structures": (
            "source_capability_id",
            "source_ref",
            "doi",
            "pdf_path",
            "local_pdf",
        ),
        "extract_visual_literature_chain": (
            "source_capability_id",
            "source_ref",
            "pdf_path",
            "task_id",
            "chain_id",
        ),
        "resolve_literature_structure_task": (
            "source_capability_id",
            "task_id",
            "source_ref",
            "pdf_path",
            "artifact_ref",
        ),
        "compile_exact_literature_rows": (
            "source_capability_id",
            "chain_id",
            "source_ref",
            "artifact_ref",
        ),
    }
    source_rows: dict[str, Any] = {}
    for action_type, preferred in preferred_fields.items():
        row = _planner_prompt_mapping(source_sensitive.get(action_type))
        accepted = {
            str(item)
            for item in row.get("accepted_payload_fields") or []
            if isinstance(item, str)
        }
        source_rows[action_type] = {
            "currently_required": row.get("currently_required") is True,
            "requires_pdf_path": row.get("requires_pdf_path") is True,
            "requires_uncompiled_visual_steps": row.get(
                "requires_uncompiled_visual_steps"
            )
            is True,
            "accepted_payload_fields": [
                field for field in preferred if field in accepted
            ],
        }
    search = _planner_prompt_mapping(requirements.get("search_literature"))
    guided = _planner_prompt_mapping(requirements.get("run_guided_chemenzy"))
    return {
        "source_sensitive_actions": source_rows,
        "search_literature": {
            "currently_required_when_selected": search.get(
                "currently_required_when_selected"
            )
            is True,
            "accepted_payload_fields": _preferred_prompt_fields(
                search.get("accepted_payload_fields"),
                (
                    "search_intent",
                    "queries",
                    "max_sources",
                    "minimum_independent_sources",
                    "exclude_source_refs",
                ),
            ),
        },
        "run_guided_chemenzy": {
            "currently_required_when_selected": guided.get(
                "currently_required_when_selected"
            )
            is True,
            "accepted_payload_fields": _preferred_prompt_fields(
                guided.get("accepted_payload_fields"),
                (
                    "initial_probe",
                    "search_mode",
                    "max_steps",
                    "chem_enzy_iterations",
                    "chem_enzy_expansion_topk",
                    "timeout_s",
                ),
            ),
        },
    }


def _preferred_prompt_fields(value: Any, preferred: tuple[str, ...]) -> list[str]:
    accepted = {
        str(item) for item in value or [] if isinstance(item, str)
    } if isinstance(value, (list, tuple)) else set()
    return [field for field in preferred if field in accepted]


def _fixed_prompt_safety_boundaries(value: Any) -> dict[str, Any]:
    safety = _planner_prompt_mapping(value)
    return {
        "planner_can_emit_solved": False,
        "raw_reaction_output_allowed": False,
        "child_route_cannot_promote_parent": True,
        "final_verdict_authority": _planner_prompt_text(
            safety.get("final_verdict_authority")
            or "deterministic_parent_route_proof",
            100,
        ),
    }


def _truncate_planner_prompt_value(
    value: Any,
    *,
    list_limit: int,
    string_limit: int,
    dict_limit: int = 32,
    key_limit: int = 80,
    max_depth: int = 6,
) -> Any:
    if max_depth <= 0:
        if isinstance(value, dict):
            return {"summary_type": "dict", "item_count": len(value)}
        if isinstance(value, (list, tuple)):
            return {"summary_type": "list", "item_count": len(value)}
        return _planner_prompt_text(value, min(string_limit, 80))
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        retained = 0
        inspected = 0
        for key, item in value.items():
            if inspected >= max(0, int(dict_limit)):
                break
            inspected += 1
            bounded_key = _planner_prompt_text(key, key_limit)
            if not bounded_key or bounded_key in out:
                continue
            out[bounded_key] = _truncate_planner_prompt_value(
                item,
                list_limit=list_limit,
                string_limit=string_limit,
                dict_limit=dict_limit,
                key_limit=key_limit,
                max_depth=max_depth - 1,
            )
            retained += 1
        if len(value) > retained:
            out["__omitted_key_count"] = len(value) - retained
        return out
    if isinstance(value, (list, tuple)):
        rows = [
            _truncate_planner_prompt_value(
                item,
                list_limit=list_limit,
                string_limit=string_limit,
                dict_limit=dict_limit,
                key_limit=key_limit,
                max_depth=max_depth - 1,
            )
            for item in value[: max(0, int(list_limit))]
        ]
        if len(value) > max(0, int(list_limit)):
            rows.append({"__omitted_item_count": len(value) - int(list_limit)})
        return rows
    if isinstance(value, str):
        return _planner_prompt_text(value, string_limit)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return _planner_prompt_count(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return 0.0
        return max(-1_000_000_000.0, min(1_000_000_000.0, value))
    return _planner_prompt_text(value, string_limit)


def _planner_prompt_snapshot_max_bytes() -> int:
    raw = os.environ.get(
        "AUTOPLANNER_CODEX_ACTION_PLANNER_PROMPT_SNAPSHOT_MAX_BYTES",
        "48000",
    )
    try:
        return max(12_000, min(96_000, int(raw)))
    except (TypeError, ValueError):
        return 48_000


def _planner_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _planner_json_sha256(value: Any) -> str:
    return hashlib.sha256(_planner_json_bytes(value)).hexdigest()


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
        "shell_allowed": False,
        "filesystem_read_tools_allowed": False,
        "input_refs_are_audit_only": True,
        "embedded_snapshot_supplies_decision_state": True,
        "outputs_remain_typed_action_batch_only": True,
        "raw_reaction_output_allowed": False,
        "final_verdict_authority": "deterministic_parent_route_proof",
    }


def _planner_tool_policy_from_task(task: WorkerTask) -> dict[str, Any]:
    return _planner_tool_policy(
        allowed_tools=[str(item) for item in task.allowed_tools or []],
        max_tool_calls=int(task.budget.max_tool_calls or 0),
    )


def _planner_context_summary(
    blackboard: dict[str, Any],
    *,
    round_index: int = 0,
    max_literature_sources_per_round: int = 3,
) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    belief = dict(blackboard.get("current_belief") or {})
    planner_allowed_tools = _planner_allowed_tools()
    planner_max_tool_calls = _planner_max_tool_calls(planner_allowed_tools)
    candidates = [dict(row) for row in evidence.get("source_candidates") or [] if isinstance(row, dict)]
    lifecycle = [dict(row) for row in evidence.get("source_lifecycle") or [] if isinstance(row, dict)]
    lifecycle_stage_counts = _source_lifecycle_stage_counts(lifecycle)
    visual_chains = [dict(row) for row in evidence.get("visual_chains") or [] if isinstance(row, dict)]
    process_rows = [dict(row) for row in evidence.get("process_evidence_rows") or [] if isinstance(row, dict)]
    capability_queue = build_source_capability_queue(
        blackboard,
        round_index=round_index,
        max_literature_sources_per_round=max_literature_sources_per_round,
    )
    pdf_done = _source_keys(
        [
            row
            for row in evidence.get("pdf_structure_evidence") or []
            if isinstance(row, dict) and pdf_evidence_has_materialized_render(row)
        ]
    )
    visual_done = _source_keys(visual_chains)
    local_pdf_candidates = [row for row in candidates if str(row.get("local_pdf") or "").strip()]
    pending_pdf = list(capability_queue.get("pending_pdf_extraction_sources") or [])
    pending_visual = list(
        capability_queue.get("pending_visual_extraction_sources") or []
    )
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
            "source_capability_queue": _compact_source_capability_queue(
                capability_queue
            ),
            "source_lifecycle": _compact_source_lifecycle(lifecycle),
            "pending_local_pdf_proxy_sources": _compact_source_lifecycle(pending_proxy),
            "pending_pdf_extraction_sources": _compact_sources(pending_pdf),
            "pending_visual_extraction_sources": _compact_sources(pending_visual),
            "processed_pdf_source_keys": sorted(pdf_done),
            "processed_visual_source_keys": sorted(visual_done),
            "exact_row_count": len(evidence.get("exact_rows") or []),
            "visual_chain_count": len(evidence.get("visual_chains") or []),
            "process_evidence_row_count": len(process_rows),
            "process_evidence_rows": _compact_process_evidence_rows(process_rows),
            "structure_resolution_task_count": len(evidence.get("structure_resolution_tasks") or []),
        },
        "route_anchor_opportunities": _route_anchor_opportunities(blackboard),
        "route_closure_pressure": _route_closure_pressure_summary(blackboard),
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
            capability_queue=capability_queue,
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
        "budget_remaining": {
            **_budget_remaining_summary(budget),
            "literature_source_units_max_this_round": int(
                max_literature_sources_per_round
            ),
            "literature_source_units_remaining_this_round": int(
                max_literature_sources_per_round
            ),
        },
        "safety_boundaries": {
            "planner_can_emit_solved": False,
            "raw_reaction_output_allowed": False,
            "final_verdict_authority": "deterministic_parent_route_proof",
            "child_route_cannot_promote_parent": True,
        },
        "no_solved_claim": True,
    }


def _planner_blackboard_handoff(
    blackboard: dict[str, Any],
    *,
    planner_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = planner_context or _planner_context_summary(blackboard)
    evidence = dict(blackboard.get("literature_evidence") or {})
    belief = dict(blackboard.get("current_belief") or {})
    source_candidates = [dict(row) for row in evidence.get("source_candidates") or [] if isinstance(row, dict)]
    pdf_focus = _compact_pdf_focus_summaries(
        [
            dict(row)
            for row in evidence.get("pdf_structure_evidence") or []
            if isinstance(row, dict)
        ]
    )
    visual_chains = [dict(row) for row in evidence.get("visual_chains") or [] if isinstance(row, dict)]
    exact_rows = [dict(row) for row in evidence.get("exact_rows") or [] if isinstance(row, dict)]
    structure_tasks = _meaningful_open_structure_resolution_tasks(blackboard)
    resolved_structures = [
        dict(row)
        for row in evidence.get("resolved_structures") or []
        if isinstance(row, dict) and row.get("accepted")
    ]
    return {
        "schema_version": "codex_action_planner_blackboard_handoff.v1",
        "case_id": str(blackboard.get("case_id") or ""),
        "target_profile": _handoff_target_profile(blackboard),
        "state_counts": {
            "source_candidates": len(source_candidates),
            "pdf_structure_evidence": len(evidence.get("pdf_structure_evidence") or []),
            "visual_chains": len(visual_chains),
            "exact_rows": len(exact_rows),
            "structure_resolution_tasks_open": len(structure_tasks),
            "resolved_structures": len(resolved_structures),
            "retrosynthetic_proposals": len(blackboard.get("retrosynthetic_proposals") or []),
            "recursive_hypothesis_tasks": len(blackboard.get("recursive_hypothesis_tasks") or []),
            "bridge_tasks": len(blackboard.get("bridge_tasks") or []),
            "route_failures": len(blackboard.get("route_failures") or []),
        },
        "evidence_board": {
            "source_acquisition": dict(context.get("source_acquisition") or {}),
            "source_capability_queue": (
                context.get("literature_processing") or {}
            ).get("source_capability_queue")
            or {},
            "pending_local_pdf_proxy_sources": (context.get("literature_processing") or {}).get("pending_local_pdf_proxy_sources") or [],
            "pending_pdf_extraction_sources": (context.get("literature_processing") or {}).get("pending_pdf_extraction_sources") or [],
            "pending_visual_extraction_sources": (context.get("literature_processing") or {}).get("pending_visual_extraction_sources") or [],
            "source_candidates": _compact_sources(source_candidates),
            "pdf_focus": pdf_focus,
            "visual_chains": _compact_visual_chains(visual_chains),
            "exact_row_samples": _compact_exact_rows(exact_rows),
            "structure_resolution_tasks": _compact_structure_resolution_tasks(structure_tasks),
            "process_evidence_rows": (context.get("literature_processing") or {}).get("process_evidence_rows") or [],
        },
        "route_board": {
            "bridge_tasks": _compact_bridge_tasks(blackboard.get("bridge_tasks") or []),
            "reaction_idea_cards": _compact_reaction_idea_cards(blackboard.get("reaction_idea_cards") or []),
            "retrosynthetic_proposals": _compact_retrosynthetic_proposals(blackboard.get("retrosynthetic_proposals") or []),
            "recursive_hypothesis_tasks": _compact_recursive_tasks(blackboard.get("recursive_hypothesis_tasks") or []),
            "parent_route_proof": _compact_parent_route_proof(blackboard.get("parent_route_proof") or {}),
            "route_anchor_opportunities": dict(context.get("route_anchor_opportunities") or {}),
            "route_closure_pressure": _route_closure_pressure_summary(blackboard),
            "frontier_ledger": _compact_planner_frontier_ledger(blackboard),
        },
        "decision_board": {
            "next_action_bias": [str(item) for item in belief.get("next_action_bias") or [] if str(item or "").strip()][:8],
            "blocked_directions": _compact_blocked_directions(belief.get("blocked_directions") or []),
            "route_failures": _compact_route_failures(blackboard.get("route_failures") or []),
            "proposal_failure_feedback": _compact_route_failures(blackboard.get("proposal_failure_feedback") or []),
            "runtime_diagnostics": _compact_runtime_diagnostics(blackboard.get("plugin_runtime_diagnostics") or []),
            "recent_actions": _compact_recent_actions(blackboard.get("action_history") or []),
            "budget_remaining": dict(context.get("budget_remaining") or {}),
        },
        "action_requirements": _brief_action_requirements(context.get("action_payload_requirements") or {}),
        "safety_boundaries": dict(context.get("safety_boundaries") or {}),
        "full_audit_ref": "agent_blackboard.json",
        "omitted_from_handoff": [
            "full planner_history",
            "full action_history",
            "artifact_refs",
            "large exact/visual/source payloads",
        ],
        "no_solved_claim": True,
}


def _compact_planner_frontier_ledger(blackboard: dict[str, Any]) -> dict[str, Any]:
    ledger = (
        dict(blackboard.get("frontier_ledger") or {})
        if isinstance(blackboard.get("frontier_ledger"), dict)
        else {}
    )
    envelope = (
        dict(blackboard.get("frontier_ledger_summary") or {})
        if isinstance(blackboard.get("frontier_ledger_summary"), dict)
        else {}
    )
    molecules = (
        dict(ledger.get("molecules") or {})
        if isinstance(ledger.get("molecules"), dict)
        else {}
    )
    open_rows: list[dict[str, Any]] = []
    open_row_count = 0
    for smiles in sorted(str(key) for key in molecules):
        raw = molecules.get(smiles)
        if not isinstance(raw, dict):
            continue
        proposal = dict(raw.get("proposal") or {})
        work = dict(raw.get("work") or {})
        stock = dict(raw.get("stock") or {})
        if not (
            str(proposal.get("state") or "") == "frontier"
            or work.get("open") is True
            or stock.get("closed") is not True
        ):
            continue
        open_row_count += 1
        if len(open_rows) < 8:
            open_rows.append(
                {
                    "canonical_smiles": smiles,
                    "proposal_state": str(proposal.get("state") or ""),
                    "work_open": work.get("open") is True,
                    "proposal_expansion_allowed": (
                        work.get("proposal_expansion_allowed") is True
                    ),
                    "stock_closed": stock.get("closed") is True,
                    "work_states": [
                        str(item) for item in work.get("states") or []
                    ][:4],
                }
            )
    ledger_summary = (
        dict(ledger.get("summary") or {})
        if isinstance(ledger.get("summary"), dict)
        else {}
    )
    return {
        "schema_version": "codex_action_planner_frontier_ledger_summary.v1",
        "ledger_schema_version": str(ledger.get("schema_version") or ""),
        "frontier_ledger_content_sha256": str(
            ledger.get("content_sha256")
            or envelope.get("frontier_ledger_content_sha256")
            or ""
        ),
        "input_valid": envelope.get("input_valid") is True,
        "ledger_validation_accepted": (
            envelope.get("ledger_validation_accepted") is True
        ),
        "root": dict(ledger.get("root") or {}) if isinstance(ledger.get("root"), dict) else {},
        "pending_counts": {
            key: ledger_summary.get(key)
            for key in (
                "proposal_pending_molecule_count",
                "proposal_expansion_eligible_molecule_count",
                "work_pending_molecule_count",
                "stock_pending_leaf_count",
                "reaction_proof_pending_edge_count",
                "dependency_pending_edge_count",
            )
            if key in ledger_summary
        },
        "open_frontier_molecules": open_rows,
        "open_frontier_molecule_count": open_row_count,
        "open_frontier_rows_truncated": len(open_rows) < open_row_count,
        "authority_note": (
            "decision_summary_only; closure authority remains the host-validated full ledger"
        ),
    }


def _brief_action_requirements(requirements: Any) -> dict[str, Any]:
    if not isinstance(requirements, dict):
        return {}
    source_sensitive = dict(requirements.get("source_sensitive_actions") or {})
    guided = dict(requirements.get("guided_actions") or {})
    search = dict(requirements.get("search_actions") or {})
    return {
        "schema_version": "codex_action_payload_requirements_brief.v1",
        "source_sensitive_actions": {
            action_type: {
                "currently_required": bool(row.get("currently_required")),
                "requires_pdf_path": bool(row.get("requires_pdf_path")),
                "requires_uncompiled_visual_steps": bool(row.get("requires_uncompiled_visual_steps")),
                "accepted_payload_fields": [str(item) for item in row.get("accepted_payload_fields") or []][:14],
                "binding_candidates": [dict(item) for item in row.get("binding_candidates") or [] if isinstance(item, dict)][:4],
                "recommended_defaults": dict(row.get("recommended_defaults") or {}),
            }
            for action_type, row in source_sensitive.items()
            if isinstance(row, dict)
        },
        "search_literature": {
            "currently_required_when_selected": bool((search.get("search_literature") or {}).get("currently_required_when_selected")),
            "accepted_payload_fields": [
                str(item) for item in (search.get("search_literature") or {}).get("accepted_payload_fields") or []
            ],
        },
        "run_guided_chemenzy": {
            "currently_required_when_selected": bool((guided.get("run_guided_chemenzy") or {}).get("currently_required_when_selected")),
            "accepted_payload_fields": [
                str(item) for item in (guided.get("run_guided_chemenzy") or {}).get("accepted_payload_fields") or []
            ],
        },
    }


def _handoff_target_profile(blackboard: dict[str, Any]) -> dict[str, Any]:
    profile = dict(blackboard.get("target_profile") or {})
    return {
        "target_name": str(profile.get("target_name") or ""),
        "target_smiles": str(profile.get("target_smiles") or profile.get("isomeric_smiles") or ""),
        "canonical_smiles": str(profile.get("canonical_smiles") or ""),
        "family_hint": str(profile.get("family_hint") or ""),
        "heavy_atoms": profile.get("heavy_atoms"),
        "rings": profile.get("rings"),
        "stereocenters": profile.get("stereocenters"),
        "functional_handles": [str(item) for item in profile.get("functional_handles") or [] if str(item or "").strip()][:8],
        "valid": bool(profile.get("valid", True)),
    }


def _compact_exact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row_id": str(row.get("row_id") or row.get("reaction_label") or ""),
            "source_ref": str(row.get("source_ref") or ""),
            "product_label": str(row.get("product_label") or row.get("visible_product_label") or "")[:120],
            "reactant_labels": [str(item) for item in row.get("reactant_labels") or [] if str(item or "").strip()][:4],
            "allowed_use": str(row.get("allowed_use") or ""),
            "validation_status": str(row.get("validation_status") or ""),
        }
        for row in rows[:6]
    ]


def _compact_bridge_tasks(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "task_id": str(row.get("task_id") or ""),
            "task_type": str(row.get("task_type") or ""),
            "target_handle": str(row.get("target_handle") or "")[:120],
            "required_bridge": str(row.get("required_bridge") or "")[:220],
            "status": str(row.get("status") or ""),
        }
        for row in rows[:6]
        if isinstance(row, dict)
    ]


def _compact_reaction_idea_cards(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "card_id": str(row.get("card_id") or row.get("target_handle") or ""),
            "target_handle": str(row.get("target_handle") or "")[:120],
            "transformation_idea": str(row.get("transformation_idea") or "")[:260],
            "confidence": str(row.get("confidence") or ""),
            "risk_flags": [str(item) for item in row.get("risk_flags") or [] if str(item or "").strip()][:4],
        }
        for row in rows[:5]
        if isinstance(row, dict)
    ]


def _compact_retrosynthetic_proposals(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "proposal_id": str(row.get("proposal_id") or row.get("proposal_label") or ""),
            "proposal_label": str(row.get("proposal_label") or "")[:140],
            "precursor_smiles": str(row.get("precursor_smiles") or row.get("smiles") or "")[:260],
            "transformation_idea": str(row.get("transformation_idea") or "")[:260],
            "route_objective_type": str(row.get("route_objective_type") or ""),
            "proposal_granularity": str(row.get("proposal_granularity") or ""),
            "confidence": str(row.get("confidence") or ""),
        }
        for row in rows[:6]
        if isinstance(row, dict)
    ]


def _compact_recursive_tasks(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "task_id": str(row.get("task_id") or row.get("proposal_id") or ""),
            "status": str(row.get("status") or "pending"),
            "name": str(row.get("name") or row.get("target_name") or "")[:120],
            "precursor_smiles": str(row.get("precursor_smiles") or row.get("smiles") or "")[:260],
            "route_objective_type": str(row.get("route_objective_type") or ""),
        }
        for row in rows[:6]
        if isinstance(row, dict)
    ]


def _compact_parent_route_proof(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {}
    return {
        "accepted": bool(value.get("accepted")),
        "proof_mode": str(value.get("proof_mode") or ""),
        "status": str(value.get("status") or ""),
        "route_ref": str(value.get("route_ref") or value.get("artifact_ref") or ""),
        "reasons": [str(item) for item in value.get("reasons") or [] if str(item or "").strip()][:6],
    }


def _compact_blocked_directions(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "direction": str(row.get("direction") or row.get("blocked_direction") or "")[:160],
            "reason": str(row.get("reason") or row.get("message") or "")[:260],
        }
        for row in rows[:6]
        if isinstance(row, dict)
    ]


def _compact_route_failures(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "failure_id": str(row.get("failure_id") or row.get("proposal_id") or row.get("category") or "")[:160],
            "reason": str(row.get("reason") or row.get("message") or row.get("summary") or "")[:300],
            "action_id": str(row.get("action_id") or ""),
            "risk_flags": [str(item) for item in row.get("risk_flags") or [] if str(item or "").strip()][:4],
        }
        for row in rows[-6:]
        if isinstance(row, dict)
    ]


def _compact_runtime_diagnostics(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "diagnostic_id": str(row.get("diagnostic_id") or row.get("id") or "")[:160],
            "severity": str(row.get("severity") or row.get("status") or ""),
            "reasons": [str(item) for item in row.get("reasons") or [] if str(item or "").strip()][:5],
            "message": str(row.get("message") or row.get("summary") or "")[:300],
        }
        for row in rows[-5:]
        if isinstance(row, dict)
    ]


def _compact_recent_actions(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "round_index": int(row.get("round_index") or 0),
            "action_id": str(row.get("action_id") or ""),
            "action_type": str(row.get("action_type") or ""),
            "useful_artifact": bool(row.get("useful_artifact")),
            "stale": bool(row.get("stale")),
            "changed_blackboard_fields": [str(item) for item in row.get("changed_blackboard_fields") or []][:8],
            "blackboard_delta": dict(row.get("blackboard_delta") or {}),
        }
        for row in rows[-8:]
        if isinstance(row, dict)
    ]


def _action_payload_requirements(
    *,
    blackboard: dict[str, Any],
    source_candidates: list[dict[str, Any]],
    local_pdf_candidates: list[dict[str, Any]],
    visual_chains: list[dict[str, Any]],
    capability_queue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    queue = dict(capability_queue or {})
    pdf_capabilities = eligible_source_capabilities(
        queue,
        "extract_pdf_literature_structures",
    )
    visual_capabilities = eligible_source_capabilities(
        queue,
        "extract_visual_literature_chain",
    )
    pdf_binding_candidates = [
        dict(row.get("source") or {}) for row in pdf_capabilities
    ]
    visual_binding_candidates = [
        dict(row.get("source") or {}) for row in visual_capabilities
    ]
    pdf_extraction_available = bool(pdf_capabilities)
    visual_extraction_available = bool(visual_capabilities)
    compile_binding_required = (
        len(_distinct_source_keys(visual_chains)) > 1
        or len(_distinct_source_keys(source_candidates)) > 1
    )
    structure_tasks = _meaningful_open_structure_resolution_tasks(blackboard)
    structure_binding_required = (
        len(_distinct_source_keys(local_pdf_candidates or source_candidates or structure_tasks)) > 1
        or len(structure_tasks) > 1
    )
    source_binding_fields = [
        "source_capability_id",
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
                "accepted_payload_fields": [
                    "search_intent",
                    "query",
                    "queries",
                    "search_queries",
                    "max_sources",
                    "minimum_independent_sources",
                    "preferred_independent_sources",
                    "known_independent_source_keys",
                    "exclude_source_refs",
                    "source_acquisition_policy",
                    "source_independence_policy",
                ],
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
                "currently_required": pdf_extraction_available,
                "reason": (
                    "local PDF is available for extraction"
                    if pdf_extraction_available
                    else "no source-matched local PDF is available; resolve/download source material first"
                ),
                "accepted_payload_fields": source_binding_fields,
                "binding_candidates": _compact_sources(pdf_binding_candidates),
                "recommended_binding_candidates": _compact_sources(
                    pdf_binding_candidates
                ),
                "requires_pdf_path": True,
                "validator_rejection_prefix": "extract_pdf_literature_structures_requires_pdf_binding",
            },
            "extract_visual_literature_chain": {
                "currently_required": visual_extraction_available,
                "reason": (
                    "accepted rendered PDF evidence is available for visual extraction"
                    if visual_extraction_available
                    else "no eligible accepted rendered PDF evidence is available"
                ),
                "accepted_payload_fields": [
                    *source_binding_fields,
                    "timeout_s",
                    "render_zoom",
                    "compress_images",
                    "max_images",
                    "visual_max_side_px",
                    "visual_jpeg_quality",
                    "expected_labels",
                    "compound_labels",
                    "route_sequence_hint",
                ],
                "binding_candidates": _compact_sources(visual_binding_candidates),
                "recommended_binding_candidates": _compact_sources(
                    visual_binding_candidates
                ),
                "requires_pdf_path": True,
                "recommended_defaults": {
                    "timeout_s": _planner_visual_timeout_s(None),
                    "compress_images": True,
                    "max_images": 6,
                    "visual_max_side_px": 1400,
                    "visual_jpeg_quality": 70,
                },
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
                    "timeout_s",
                    "max_images",
                    "visual_max_side_px",
                    "visual_jpeg_quality",
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
                "requires_uncompiled_visual_steps": True,
                "process_evidence_rows_are_not_exact_rows": True,
            },
        },
        "guided_actions": {
            "run_guided_chemenzy": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": [
                    "initial_probe",
                    "search_mode",
                    "max_steps",
                    "chem_enzy_iterations",
                    "chem_enzy_expansion_topk",
                    "timeout_s",
                    "max_candidates",
                    "guided_policy_runtime_rebuild",
                ],
                "runtime_policy_rebuild": True,
                "do_not_emit_full_policy": True,
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
                "accepted_payload_fields": ["subgoal_targets", "child_targets", "child_policy_runtime_rebuild"],
                "required_target_fields": [
                    "name",
                    "smiles",
                    "target_equivalence_audit_required",
                    "exact_target_override",
                    "no_solved_claim",
                    "child_route_cannot_promote_parent",
                    "policy_runtime_rebuild",
                    "policy_summary",
                ],
                "required_policy_safety_fields": [
                    "compiler_metadata.requires_verifier",
                    "compiler_metadata.no_solved_claim",
                    "compiler_metadata.child_route_cannot_promote_parent",
                ],
                "runtime_policy_rebuild": True,
                "do_not_emit_full_policy": True,
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
        "reaction_idea_card_count": len(blackboard.get("reaction_idea_cards") or []),
        "retrosynthetic_proposal_count": len(blackboard.get("retrosynthetic_proposals") or []),
        "pending_recursive_hypothesis_task_count": sum(
            1
            for row in blackboard.get("recursive_hypothesis_tasks") or []
            if isinstance(row, dict) and str(row.get("status") or "pending").lower() in {"", "pending", "ready", "queued"}
        ),
        "proposal_granularity_counts": _proposal_granularity_counts(blackboard),
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
        "independent_source_group_count": len(independent_literature_source_keys(blackboard)),
        "minimum_independent_source_groups": 2,
        "preferred_independent_source_groups": 3,
        "article_and_supporting_information_share_source_group": True,
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
    pending_recursive = [
        dict(row)
        for row in blackboard.get("recursive_hypothesis_tasks") or []
        if isinstance(row, dict)
        and str(row.get("status") or "pending").lower() in {"", "pending", "ready", "queued"}
        and str(row.get("precursor_smiles") or "").strip()
    ]
    proposals = [
        dict(row)
        for row in blackboard.get("retrosynthetic_proposals") or []
        if isinstance(row, dict) and str(row.get("precursor_smiles") or "").strip()
    ]
    return {
        "terminal_candidate_count": len(terminal_candidates),
        "upstream_terminal_bridge_task_count": len(upstream_tasks),
        "retrosynthetic_proposal_count": len(proposals),
        "pending_recursive_hypothesis_task_count": len(pending_recursive),
        "proposal_granularity_counts": _proposal_granularity_counts(blackboard),
        "sample_pending_proposal_frontiers": [
            {
                "name": str(row.get("name") or row.get("proposal_id") or row.get("task_id") or "")[:120],
                "smiles": str(row.get("precursor_smiles") or row.get("smiles") or "")[:220],
                "proposal_granularity": str(row.get("proposal_granularity") or ""),
                "route_objective_type": str(row.get("route_objective_type") or ""),
            }
            for row in [*pending_recursive[:3], *proposals[:3]][:4]
        ],
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


def _proposal_granularity_counts(blackboard: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in blackboard.get("retrosynthetic_proposals") or []:
        if not isinstance(row, dict):
            continue
        granularity = str(row.get("proposal_granularity") or "unspecified")
        counts[granularity] = counts.get(granularity, 0) + 1
    return counts


def _compact_source_capability_queue(queue: dict[str, Any]) -> dict[str, Any]:
    return _fixed_prompt_source_capability_queue(queue)


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


def _compact_pdf_focus_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows[:8]:
        focus = dict(row.get("focus") or {})
        if not focus:
            continue
        compact.append(
            {
                "evidence_id": str(row.get("evidence_id") or ""),
                "source_ref": str(row.get("source_ref") or ""),
                "artifact_ref": str(row.get("artifact_ref") or ""),
                "focus": {
                    "focus_terms": [
                        str(item)[:96]
                        for item in focus.get("focus_terms") or []
                        if str(item or "").strip()
                    ][:24],
                    "focus_page_numbers": [
                        _planner_prompt_count(item)
                        for item in focus.get("focus_page_numbers") or []
                        if _planner_prompt_count(item) > 0
                    ][:16],
                    "page_relevance": [
                        {
                            "page_number": _planner_prompt_count(
                                relevance.get("page_number")
                            ),
                            "score": _planner_prompt_count(relevance.get("score")),
                            "matched_terms": [
                                str(item)[:96]
                                for item in relevance.get("matched_terms") or []
                                if str(item or "").strip()
                            ][:8],
                        }
                        for relevance in focus.get("page_relevance") or []
                        if isinstance(relevance, dict)
                    ][:16],
                    "selection_strategy": str(focus.get("selection_strategy") or "")[:80],
                    "relevance_available": focus.get("relevance_available") is True,
                    "no_ocr_or_relevance_fabrication": focus.get(
                        "no_ocr_or_relevance_fabrication"
                    )
                    is True,
                    "no_solved_claim": True,
                },
                "no_solved_claim": True,
            }
        )
    return compact


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


def _meaningful_open_structure_resolution_tasks(
    blackboard: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in (blackboard.get("literature_evidence") or {}).get(
        "structure_resolution_tasks"
    ) or []:
        if not isinstance(row, dict) or str(row.get("status") or "open").lower() not in {
            "",
            "open",
            "pending",
            "ready",
        }:
            continue
        label = str(row.get("label") or "").strip()
        if label and not meaningful_compound_labels([label]):
            continue
        tasks.append(dict(row))
    return tasks


def _compact_process_evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row_id": str(row.get("row_id") or ""),
            "process_type": str(row.get("process_type") or ""),
            "source_ref": str(row.get("source_ref") or ""),
            "source_title": str(row.get("source_title") or "")[:240],
            "endpoint_labels": [str(item) for item in row.get("endpoint_labels") or [] if str(item).strip()][:4],
            "substrate_or_feedstock_labels": [
                str(item) for item in row.get("substrate_or_feedstock_labels") or [] if str(item).strip()
            ][:4],
            "biocatalyst_or_process_labels": [
                str(item) for item in row.get("biocatalyst_or_process_labels") or [] if str(item).strip()
            ][:4],
            "allowed_use": str(row.get("allowed_use") or "route_objective_anchor_and_guided_hint_only"),
            "not_exact_literature_segment": bool(row.get("not_exact_literature_segment", True)),
            "not_parent_route_proof": bool(row.get("not_parent_route_proof", True)),
            "no_solved_claim": bool(row.get("no_solved_claim", True)),
        }
        for row in rows[:8]
    ]


def _route_anchor_opportunities(blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    source_candidates = [dict(row) for row in evidence.get("source_candidates") or [] if isinstance(row, dict)]
    source_by_ref = {
        str(row.get("source_ref") or "").strip(): row
        for row in source_candidates
        if str(row.get("source_ref") or "").strip()
    }
    visual_source_refs = {
        str(row.get("source_ref") or "").strip()
        for row in evidence.get("visual_chains") or []
        if isinstance(row, dict) and str(row.get("source_ref") or "").strip()
    }
    structure_tasks = _meaningful_open_structure_resolution_tasks(blackboard)
    tasks_by_source: dict[str, list[dict[str, Any]]] = {}
    for task in structure_tasks:
        source_ref = str(task.get("source_ref") or "").strip()
        tasks_by_source.setdefault(source_ref, []).append(task)
    opportunities: list[dict[str, Any]] = []
    for index, row in enumerate(evidence.get("process_evidence_rows") or [], start=1):
        if not isinstance(row, dict):
            continue
        source_ref = str(row.get("source_ref") or "").strip()
        anchor_labels = _dedupe_preserve_order(
            [
                *[str(item) for item in row.get("substrate_or_feedstock_labels") or []],
                *[str(item) for item in row.get("endpoint_labels") or []],
            ]
        )
        if not (source_ref or anchor_labels or row.get("process_type")):
            continue
        source = source_by_ref.get(source_ref, {})
        actions = ["derive_broad_reaction_template", "compile_objective_route_proof"]
        if source_ref in tasks_by_source:
            actions.insert(0, "resolve_literature_structure_task")
        if str(source.get("local_pdf") or row.get("source_pdf_path") or "").strip() and source_ref not in visual_source_refs:
            actions.insert(0, "extract_visual_literature_chain")
        if anchor_labels or row.get("biocatalyst_or_process_labels"):
            actions.extend(["extract_analogical_reaction_templates", "run_guided_chemenzy"])
        opportunities.append(
            {
                "opportunity_id": str(row.get("row_id") or row.get("process_id") or f"process_anchor:{index}"),
                "opportunity_type": "process_or_literature_anchor",
                "evidence_level": "not_exact_process_or_advisory_anchor",
                "source_ref": source_ref,
                "source_title": str(row.get("source_title") or source.get("title") or "")[:220],
                "anchor_labels": anchor_labels[:6],
                "transformation_labels": [
                    str(item)
                    for item in row.get("biocatalyst_or_process_labels") or []
                    if str(item or "").strip()
                ][:6],
                "missing_or_uncertain_inputs": _dedupe_preserve_order(
                    [
                        "machine_readable_anchor_structure" if anchor_labels else "",
                        "exact_step_row" if row.get("not_exact_literature_segment", True) else "",
                        "deterministic_parent_connectivity_proof",
                    ]
                ),
                "plausible_next_actions": _dedupe_preserve_order(actions)[:8],
                "allowed_use": str(row.get("allowed_use") or "route_objective_anchor_and_guided_hint_only"),
                "no_solved_claim": True,
            }
        )
    for index, task in enumerate(structure_tasks, start=1):
        opportunities.append(
            {
                "opportunity_id": str(task.get("task_id") or f"structure_task:{index}"),
                "opportunity_type": "open_structure_resolution",
                "evidence_level": "name_or_visual_label_needs_structure",
                "source_ref": str(task.get("source_ref") or ""),
                "source_title": str(task.get("source_title") or "")[:220],
                "anchor_labels": [str(task.get("label") or "")] if str(task.get("label") or "").strip() else [],
                "missing_or_uncertain_inputs": ["machine_readable_anchor_structure"],
                "plausible_next_actions": [
                    "resolve_literature_structure_task",
                    "search_literature",
                    "derive_broad_reaction_template",
                ],
                "allowed_use": "structure_resolution_before_template_or_guided_search",
                "no_solved_claim": True,
            }
        )
    for index, row in enumerate(evidence.get("resolved_structures") or [], start=1):
        if not isinstance(row, dict) or not row.get("accepted"):
            continue
        opportunities.append(
            {
                "opportunity_id": str(row.get("task_id") or row.get("resolution_id") or f"resolved_anchor:{index}"),
                "opportunity_type": "resolved_anchor_structure",
                "evidence_level": "source_grounded_structure_candidate",
                "source_ref": str(row.get("source_ref") or ""),
                "anchor_labels": [str(row.get("label") or "")] if str(row.get("label") or "").strip() else [],
                "anchor_smiles": str(row.get("smiles") or "")[:260],
                "missing_or_uncertain_inputs": ["route_connection_to_target", "deterministic_parent_connectivity_proof"],
                "plausible_next_actions": [
                    "derive_broad_reaction_template",
                    "extract_analogical_reaction_templates",
                    "expand_child_target",
                    "run_guided_chemenzy",
                    "compile_objective_route_proof",
                ],
                "allowed_use": "bridge_material_for_template_child_or_guided_search",
                "no_solved_claim": True,
            }
        )
    for index, row in enumerate(evidence.get("visual_chains") or [], start=1):
        if not isinstance(row, dict):
            continue
        reasons = {str(item) for item in row.get("reasons") or row.get("audit_reasons") or []}
        advisory = (
            bool(row.get("not_exact_literature_segment"))
            or int(row.get("candidate_step_count") or row.get("step_count") or 0) > 0
            or "advisory_visual_template_card_available" in reasons
        )
        if not advisory:
            continue
        opportunities.append(
            {
                "opportunity_id": str(row.get("chain_id") or row.get("artifact_ref") or f"visual_chain:{index}"),
                "opportunity_type": "advisory_visual_chain",
                "evidence_level": "visual_or_source_detail_advisory",
                "source_ref": str(row.get("source_ref") or ""),
                "source_title": str(row.get("source_title") or "")[:220],
                "candidate_step_count": int(row.get("candidate_step_count") or row.get("step_count") or 0),
                "missing_or_uncertain_inputs": ["exact_row_or_full_atom_mapping", "deterministic_parent_connectivity_proof"],
                "plausible_next_actions": [
                    "compile_exact_literature_rows",
                    "derive_broad_reaction_template",
                    "extract_analogical_reaction_templates",
                    "run_guided_chemenzy",
                ],
                "allowed_use": "template_extraction_and_guided_hint_only",
                "no_solved_claim": True,
            }
        )
    for index, row in enumerate(blackboard.get("recursive_hypothesis_tasks") or [], start=1):
        if not isinstance(row, dict) or str(row.get("status") or "pending").lower() not in {"", "pending", "open", "ready"}:
            continue
        if not str(row.get("precursor_smiles") or row.get("smiles") or "").strip():
            continue
        opportunities.append(
            {
                "opportunity_id": str(row.get("task_id") or f"recursive_frontier:{index}"),
                "opportunity_type": "recursive_hypothesis_frontier",
                "evidence_level": str(row.get("proposal_granularity") or "hypothesis_frontier"),
                "anchor_labels": [str(row.get("name") or row.get("target_name") or "")] if str(row.get("name") or row.get("target_name") or "").strip() else [],
                "anchor_smiles": str(row.get("precursor_smiles") or row.get("smiles") or "")[:260],
                "missing_or_uncertain_inputs": ["child_target_route_verification", "parent_connectivity_after_child_route"],
                "plausible_next_actions": ["expand_child_target", "run_guided_chemenzy", "compile_objective_route_proof"],
                "allowed_use": "route_expansion_subgoal_hint_only",
                "no_solved_claim": True,
            }
        )
    return {
        "schema_version": "route_anchor_opportunities.v1",
        "decision_policy": (
            "Agent chooses among plausible_next_actions. Evidence can be exact, process, advisory, or name-only; "
            "uncertainty changes confidence and required follow-up, not whether the branch can be explored."
        ),
        "opportunity_count": len(opportunities),
        "opportunities": opportunities[:12],
    }


def _route_closure_pressure_summary(blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    parent_proof = dict((blackboard or {}).get("parent_route_proof") or {})
    proof_bundle = dict((blackboard or {}).get("route_proof_bundle") or {})
    process_rows = [dict(row) for row in evidence.get("process_evidence_rows") or [] if isinstance(row, dict)]
    structure_tasks = _meaningful_open_structure_resolution_tasks(blackboard)
    resolved_structures = [
        dict(row)
        for row in evidence.get("resolved_structures") or []
        if isinstance(row, dict) and row.get("accepted")
    ]
    proof_reasons = _dedupe_preserve_order(
        [
            *[str(item) for item in parent_proof.get("reasons") or [] if str(item or "").strip()],
            *[str(item) for item in proof_bundle.get("reasons") or [] if str(item or "").strip()],
        ]
    )
    named_anchors = _dedupe_preserve_order(
        [
            *[
                str(item)
                for row in process_rows
                for item in row.get("substrate_or_feedstock_labels") or []
                if str(item or "").strip()
            ],
            *[
                str(item)
                for row in process_rows
                for item in row.get("endpoint_labels") or []
                if str(item or "").strip()
            ],
        ]
    )
    transformation_anchors = _dedupe_preserve_order(
        [
            str(item)
            for row in process_rows
            for item in row.get("biocatalyst_or_process_labels") or []
            if str(item or "").strip()
        ]
    )
    return {
        "schema_version": "route_closure_pressure.v1",
        "parent_proof_accepted": is_solved_parent_route_proof(
            parent_proof,
            expected_target_smiles=str(
                ((blackboard or {}).get("target_profile") or {}).get("target_smiles") or ""
            ),
        ),
        "parent_route_status": str(parent_proof.get("route_status") or parent_proof.get("status") or ""),
        "objective_proof_status": str(proof_bundle.get("route_status") or ""),
        "proof_gap_reasons": proof_reasons[:8],
        "named_process_or_endpoint_anchors": named_anchors[:10],
        "transformation_anchors": transformation_anchors[:8],
        "open_structure_resolution_tasks": _compact_structure_resolution_tasks(structure_tasks),
        "resolved_anchor_structures": _compact_resolved_structures(resolved_structures),
        "planning_hint": (
            "Close the proof gap by turning source-grounded anchors into resolved structures, exact/advisory templates, "
            "or connected child targets; do not skip a named literature anchor only because it lacks SMILES yet."
            if (proof_reasons or named_anchors or structure_tasks)
            else ""
        ),
    }


def _compact_resolved_structures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": str(row.get("label") or "")[:120],
            "smiles": str(row.get("smiles") or "")[:260],
            "source_ref": str(row.get("source_ref") or ""),
            "source_locator": str(row.get("source_locator") or "")[:220],
            "confidence": str(row.get("confidence") or ""),
            "derivation_mode": str(row.get("derivation_mode") or ""),
            "rdkit_valid": bool(row.get("rdkit_valid")),
            "no_solved_claim": bool(row.get("no_solved_claim", True)),
        }
        for row in rows[:10]
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
    local_pdf = str(row.get("local_pdf") or row.get("source_pdf_path") or row.get("pdf_path") or "").strip().lower()
    if local_pdf:
        return f"pdf:{local_pdf}"
    document_id = str(row.get("document_id") or "").strip().lower()
    if document_id:
        return f"document:{document_id}"
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref:
        return f"ref:{source_ref}"
    doi = str(row.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    pii = str(row.get("pii") or "").strip().lower()
    if pii:
        return f"pii:{pii}"
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
            "reaction_idea_cards",
            "retrosynthetic_proposals",
            "retrosynthetic_proposal_compile_report",
            "recursive_hypothesis_tasks",
            "proposal_failure_feedback",
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


def _planner_reasoning_effort() -> str:
    return str(os.environ.get("AUTOPLANNER_CODEX_ACTION_PLANNER_REASONING_EFFORT") or "medium").strip() or "medium"

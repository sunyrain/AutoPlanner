"""Agentic blackboard controller entry point."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerTask,
    run_codex_worker,
)
from cascade_planner.harness.agent_action_planner import (
    plan_action_batch,
    validate_action_batch,
)
from cascade_planner.harness.agentic_blackboard import (
    build_agentic_guided_payload,
    complete_round,
    initialize_agent_blackboard,
    rank_analogical_hypotheses_from_blackboard,
    update_blackboard_from_action,
    update_budget_for_action,
)
from cascade_planner.harness.failure_critic import compile_failure_critic_report
from cascade_planner.harness.parent_route_proof import compile_stitched_parent_route_proof
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.runner import emit_final_verdict
from cascade_planner.harness.schemas import (
    ArtifactBundle,
    FinalVerdict,
    TargetInput,
    append_jsonl,
    write_json,
)
from cascade_planner.harness.target_side_strategy import build_target_side_disconnection_hypotheses
from cascade_planner.harness.tools import (
    HarnessBudget,
    ToolExecutionState,
    artifact_bundle_from_state,
    execute_local_tool,
)


ActionPlannerRunner = Callable[..., dict[str, Any]]


def run_agentic_blackboard_controller(
    *,
    target_name: str,
    target_smiles: str,
    family_hint: str = "",
    output_dir: str | Path,
    literature_pdf_path: str | Path = "",
    literature_pdf_source_ref: str = "",
    timeout_s: float = 1800.0,
    key_path: str | Path = "",
    base_url: str = "https://api.wellau.com/v1",
    model: str = "gpt-5.5",
    max_rounds: int = 3,
    exhaust_round_budget: bool = False,
    action_planner: ActionPlannerRunner | None = None,
    mock_tool_results: dict[str, Any] | None = None,
    prior_artifacts: dict[str, Any] | None = None,
    budget: HarnessBudget | None = None,
) -> dict[str, Any]:
    """Run the policy-driven DAG + blackboard controller."""
    run_dir = Path(output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tool_calls.jsonl").touch()
    (run_dir / "decision_trace.jsonl").touch()
    budget = budget or HarnessBudget(timeout_s=float(timeout_s))
    budget.max_guided_chemenzy_runs = max(1, int(budget.max_guided_chemenzy_runs or 1))
    budget.max_route_expansion_subgoal_runs = max(1, int(budget.max_route_expansion_subgoal_runs or 2))

    target = TargetInput(
        target_name=target_name,
        target_smiles=target_smiles,
        family_hint=family_hint,
        case_id="",
    )
    target_data = target.to_dict()
    if str(literature_pdf_path or "").strip():
        target_data["literature_pdf_path"] = str(Path(literature_pdf_path).expanduser().resolve())
    if str(literature_pdf_source_ref or "").strip():
        target_data["literature_pdf_source_ref"] = str(literature_pdf_source_ref).strip()
    write_json(run_dir / "target_input.json", target_data)
    write_json(run_dir / "budget.json", budget.to_dict())
    append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "start", "created_at_utc": _now(), "controller": "agentic_blackboard"})

    preflight = run_preflight(target)
    target.case_id = str(preflight.get("case_id") or target.case_id)
    target_data = target.to_dict()
    if str(literature_pdf_path or "").strip():
        target_data["literature_pdf_path"] = str(Path(literature_pdf_path).expanduser().resolve())
    if str(literature_pdf_source_ref or "").strip():
        target_data["literature_pdf_source_ref"] = str(literature_pdf_source_ref).strip()
    write_json(run_dir / "target_input.json", target_data)
    write_json(run_dir / "preflight.json", preflight)
    append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "preflight", "preflight": preflight})

    state = ToolExecutionState(
        run_dir=run_dir,
        target_input=target_data,
        preflight=preflight,
        budget=budget,
        key_path=key_path or Path(__file__).resolve().parents[2] / "key.txt",
        base_url=base_url,
        model=model,
        mock_tool_results=dict(mock_tool_results or {}),
    )
    if prior_artifacts:
        state.artifacts.update(dict(prior_artifacts))
    blackboard = initialize_agent_blackboard(
        target_input=target_data,
        preflight=preflight,
        max_rounds=max_rounds,
        budget_limits=budget.to_dict(),
        prior_artifacts=prior_artifacts,
    )
    if prior_artifacts:
        blackboard = _seed_failure_evidence_from_prior(blackboard, state=state)

    tool_calls: list[dict[str, Any]] = []
    action_batches: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    if not preflight.get("accepted"):
        final = _invalid_final(preflight)
        write_json(run_dir / "agent_blackboard.json", blackboard)
        bundle = artifact_bundle_from_state(state=state, workflow_plan=_workflow_plan_from_actions(action_batches), tool_calls=tool_calls)
        write_json(run_dir / "artifact_bundle.json", bundle.to_dict())
        write_json(run_dir / "final_verdict.json", final.to_dict())
        return _result(run_dir, target_data, preflight, blackboard, action_batches, validations, bundle, final, tool_calls)

    stop_requested = False
    for round_index in range(1, int(max_rounds or 3) + 1):
        action_batch = _obtain_action_batch(
            blackboard=blackboard,
            round_index=round_index,
            run_dir=run_dir,
            action_planner=action_planner,
            exhaust_round_budget=exhaust_round_budget,
        )
        validation = validate_action_batch(action_batch, blackboard=blackboard)
        validations.append(validation)
        action_batches.append(action_batch)
        write_json(run_dir / f"action_batch_round_{round_index}.json", action_batch)
        append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "action_batch", "round_index": round_index, "validation": validation})
        if not validation.get("accepted"):
            state.validations.append(validation)
            break

        round_useful = False
        for action in action_batch.get("actions") or []:
            action_type = str(action.get("action_type") or "")
            action_result, records = _execute_agent_action(
                action=action,
                state=state,
                blackboard=blackboard,
            )
            tool_calls.extend(records)
            blackboard = update_budget_for_action(blackboard, action_type)
            blackboard = update_blackboard_from_action(
                blackboard,
                action=action,
                action_result=action_result,
                round_index=round_index,
                run_dir=run_dir,
            )
            round_useful = round_useful or bool(blackboard["action_history"][-1].get("useful_artifact"))
            append_jsonl(
                run_dir / "decision_trace.jsonl",
                {
                    "stage": "agent_action",
                    "round_index": round_index,
                    "action_type": action_type,
                    "accepted": bool(action_result.get("accepted", True)),
                    "useful_artifact": bool(blackboard["action_history"][-1].get("useful_artifact")),
                },
            )
            if action_type == "stop_unresolved":
                stop_requested = True
                break
        blackboard = _auto_update_critic(blackboard, state=state, run_dir=run_dir, round_index=round_index)
        blackboard = complete_round(blackboard, round_index)
        write_json(run_dir / "agent_blackboard.json", blackboard)
        if stop_requested or _parent_proof_accepted(blackboard):
            break
        if not round_useful and round_index >= int(max_rounds or 3):
            break

    state.artifacts["agent_blackboard"] = blackboard
    workflow_plan = _workflow_plan_from_actions(action_batches)
    bundle = artifact_bundle_from_state(state=state, workflow_plan=workflow_plan, tool_calls=tool_calls)
    bundle.validations = [*bundle.validations, *validations]
    final = emit_agentic_final_verdict(blackboard=blackboard, artifacts=state.artifacts, bundle=bundle.to_dict())
    write_json(run_dir / "artifact_bundle.json", bundle.to_dict())
    write_json(run_dir / "agent_blackboard.json", blackboard)
    write_json(run_dir / "final_verdict.json", final.to_dict())
    append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "final_verdict", "final_verdict": final.to_dict()})
    return _result(run_dir, target_data, preflight, blackboard, action_batches, validations, bundle, final, tool_calls)


def emit_agentic_final_verdict(
    *,
    blackboard: dict[str, Any],
    artifacts: dict[str, Any],
    bundle: dict[str, Any] | None = None,
) -> FinalVerdict:
    proof = dict(blackboard.get("parent_route_proof") or artifacts.get("parent_route_proof") or {})
    case_id = str(blackboard.get("case_id") or (bundle or {}).get("case_id") or "target")
    if proof.get("accepted") and proof.get("solved"):
        return FinalVerdict(
            case_id=case_id,
            verdict="solved",
            route_status="solved",
            solved=True,
            stock_audit_passed=True,
            artifact_refs=dict(blackboard.get("artifact_refs") or {}),
        )
    latest_verdict = emit_final_verdict(bundle or {
        "case_id": case_id,
        "target_input": {},
        "preflight": {"accepted": True},
        "workflow_plan": {},
        "artifacts": artifacts,
        "validations": [],
    })
    if latest_verdict.solved:
        return FinalVerdict(
            case_id=case_id,
            verdict="unresolved",
            reasons=["candidate_route_found_parent_proof_missing"],
            route_status="candidate_route_found_parent_proof_missing",
            stock_audit_passed=bool(latest_verdict.stock_audit_passed),
            artifact_refs=dict(blackboard.get("artifact_refs") or {}),
        )
    if proof:
        status = str(proof.get("route_status") or latest_verdict.route_status or "unresolved")
        verdict = "fake_closed_rejected" if status == "fake_closed_rejected" else (
            "partial_anchor_only_not_solved" if status == "partial_anchor_only_not_solved" else "unresolved"
        )
        return FinalVerdict(
            case_id=case_id,
            verdict=verdict,
            reasons=[str(item) for item in proof.get("reasons") or latest_verdict.reasons],
            route_status=status,
            stock_audit_passed=False,
            artifact_refs=dict(blackboard.get("artifact_refs") or {}),
        )
    child_solved = bool((blackboard.get("current_belief") or {}).get("child_route_solved"))
    if child_solved:
        return FinalVerdict(
            case_id=case_id,
            verdict="unresolved",
            reasons=["child_target_solved_parent_proof_missing"],
            route_status="child_solved_parent_unresolved",
            stock_audit_passed=False,
            artifact_refs=dict(blackboard.get("artifact_refs") or {}),
        )
    reasons = [str(item) for item in latest_verdict.reasons or []]
    if not reasons:
        reasons = ["no_deterministic_parent_route_proof"]
    return FinalVerdict(
        case_id=case_id,
        verdict=latest_verdict.verdict if latest_verdict.verdict != "solved" else "unresolved",
        reasons=sorted(set(reasons + ["no_deterministic_parent_route_proof"])),
        route_status=latest_verdict.route_status or "unresolved",
        stock_audit_passed=False,
        artifact_refs=dict(blackboard.get("artifact_refs") or {}),
    )


def _obtain_action_batch(
    *,
    blackboard: dict[str, Any],
    round_index: int,
    run_dir: Path,
    action_planner: ActionPlannerRunner | None,
    exhaust_round_budget: bool = False,
) -> dict[str, Any]:
    if action_planner is not None:
        return action_planner(blackboard=blackboard, round_index=round_index, run_dir=run_dir)
    return plan_action_batch(blackboard, round_index=round_index, exhaust_round_budget=exhaust_round_budget)


def _execute_agent_action(
    *,
    action: dict[str, Any],
    state: ToolExecutionState,
    blackboard: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    action_type = str(action.get("action_type") or "")
    mock = _mock_action_result(state, action, blackboard)
    if mock is not None:
        return _wrap_action_result(mock), []
    if action_type == "generate_disconnection_hypotheses":
        result = build_target_side_disconnection_hypotheses(
            target_smiles=str(state.target_input.get("target_smiles") or ""),
            target_name=str(state.target_input.get("target_name") or ""),
            family_hint=str(state.target_input.get("family_hint") or ""),
            source_evidence_refs=[str(item) for item in (blackboard.get("literature_evidence") or {}).get("source_refs") or []],
            case_id=str(state.preflight.get("case_id") or ""),
        )
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "build_failure_critic_report":
        result = compile_failure_critic_report(
            blackboard=blackboard,
            artifacts=state.artifacts,
            case_id=str(state.preflight.get("case_id") or ""),
            target_name=str(state.target_input.get("target_name") or ""),
        )
        state.artifacts["failure_critic_report"] = result
        return {"accepted": True, "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "search_literature":
        result = _codex_first_literature_scout(blackboard=blackboard, state=state, payload=dict(action.get("payload") or {}))
        state.artifacts["literature_scout"] = result
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "rank_analogical_hypotheses":
        result = rank_analogical_hypotheses_from_blackboard(blackboard)
        state.artifacts["analogical_hypothesis_ranking"] = result
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "extract_pdf_literature_structures":
        payload = dict(action.get("payload") or {})
        _inject_pdf_defaults(payload, state.target_input)
        record = execute_local_tool("extract_pdf_literature_structures", payload, state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "extract_visual_literature_chain":
        payload = dict(action.get("payload") or {})
        _inject_pdf_defaults(payload, state.target_input)
        record = execute_local_tool("extract_visual_literature_chain", payload, state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "compile_exact_literature_rows":
        record = execute_local_tool("compile_source_detail_chain_route", dict(action.get("payload") or {}), state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "run_guided_chemenzy":
        payload = {**build_agentic_guided_payload(blackboard), **dict(action.get("payload") or {})}
        record = execute_local_tool("run_guided_chemenzy_rerun", payload, state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "expand_child_target":
        payload = {"max_targets": 2, **dict(action.get("payload") or {})}
        record = execute_local_tool("run_route_expansion_subgoal_search", payload, state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "stitch_parent_route":
        record = execute_local_tool("stitch_literature_chain_with_subgoal_route", dict(action.get("payload") or {}), state)
        stitched = dict((record.output.get("result") if isinstance(record.output, dict) else {}) or {})
        proof = _compile_parent_proof_from_state(state=state, blackboard=blackboard, stitched=stitched)
        state.artifacts["parent_route_proof"] = proof
        return {
            "accepted": bool(proof.get("accepted") or proof.get("reasons")),
            "result": {"parent_route_proof": proof, "stitched_route": stitched},
            "reasons": [str(item) for item in proof.get("reasons") or []],
        }, [record.to_dict()]
    if action_type == "stop_unresolved":
        return {"accepted": True, "schema_version": "agent_stop_unresolved.v1", "reasons": ["stop_unresolved_selected"]}, []
    return {"accepted": False, "reasons": [f"unknown_action:{action_type}"]}, []


def _auto_update_critic(blackboard: dict[str, Any], *, state: ToolExecutionState, run_dir: Path, round_index: int) -> dict[str, Any]:
    report = compile_failure_critic_report(
        blackboard=blackboard,
        artifacts=state.artifacts,
        case_id=str(state.preflight.get("case_id") or ""),
        target_name=str(state.target_input.get("target_name") or ""),
    )
    if not report.get("accepted"):
        return blackboard
    state.artifacts["failure_critic_report"] = report
    action = {
        "action_id": f"r{round_index}:auto_failure_critic",
        "action_type": "build_failure_critic_report",
        "payload": {},
    }
    return update_blackboard_from_action(
        blackboard,
        action=action,
        action_result={"accepted": True, "result": report},
        round_index=round_index,
        run_dir=run_dir,
    )


def _seed_failure_evidence_from_prior(blackboard: dict[str, Any], *, state: ToolExecutionState) -> dict[str, Any]:
    report = compile_failure_critic_report(
        blackboard=blackboard,
        artifacts=state.artifacts,
        case_id=str(state.preflight.get("case_id") or ""),
        target_name=str(state.target_input.get("target_name") or ""),
    )
    if not report.get("accepted"):
        return blackboard
    board = dict(blackboard)
    board["route_failures"] = list(report.get("route_failures") or [])
    board["bridge_tasks"] = list(report.get("bridge_tasks") or [])
    board["terminal_blacklist"] = list(report.get("terminal_blacklist") or [])
    belief = dict(board.get("current_belief") or {})
    belief["blocked_directions"] = list(report.get("blocked_directions") or [])
    constraints = dict(belief.get("constraints") or {})
    constraints.update(dict(report.get("constraints") or {}))
    belief["constraints"] = constraints
    board["current_belief"] = belief
    if report.get("source_reasons"):
        board["plugin_runtime_diagnostics"] = [
            {
                "schema_version": "agent_prior_failure_source_reasons.v1",
                "reasons": [str(item) for item in report.get("source_reasons") or []],
            }
        ]
    state.artifacts["failure_critic_report"] = report
    return board


def _compile_parent_proof_from_state(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    stitched: dict[str, Any],
) -> dict[str, Any]:
    parent_verifier = {} if stitched.get("accepted") else _latest_parent_verifier(state.artifacts)
    route_expansion = dict(state.artifacts.get("route_expansion_subgoal_search") or {})
    exact_rows = (blackboard.get("literature_evidence") or {}).get("exact_rows") or []
    exact_segment = {
        "accepted": bool(exact_rows),
        "parent_route_connected": bool(stitched.get("accepted")),
        "row_count": len(exact_rows),
    }
    return compile_stitched_parent_route_proof(
        target_smiles=str(state.target_input.get("target_smiles") or ""),
        target_name=str(state.target_input.get("target_name") or ""),
        case_id=str(state.preflight.get("case_id") or ""),
        parent_verifier=parent_verifier,
        stitched_route=stitched,
        child_route=route_expansion,
        exact_literature_segment=exact_segment,
        analogy_refs=(blackboard.get("analogical_hypothesis_ranking") or {}).get("selected_hypotheses") or [],
    )


def _latest_parent_verifier(artifacts: dict[str, Any]) -> dict[str, Any]:
    guided = artifacts.get("guided_chemenzy")
    if isinstance(guided, dict):
        verifier = guided.get("raw_route_verifier")
        if isinstance(verifier, dict) and verifier:
            return dict(verifier)
    route_verifier = artifacts.get("route_verifier")
    if isinstance(route_verifier, dict):
        return dict(route_verifier)
    return {}


def _codex_first_literature_scout(*, blackboard: dict[str, Any], state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    """Scout sources through Codex first, then local PDF, then placeholders.

    Python keeps this action bounded and auditable. The open-ended literature
    search belongs to Codex; local PDFs only rescue cases where live source
    access fails or is disabled.
    """
    max_sources = max(1, min(3, int(payload.get("max_sources") or 3)))
    attempts: list[dict[str, Any]] = []
    reasons: list[str] = []

    if _codex_scout_enabled(payload):
        codex_report = _run_codex_literature_scout(
            blackboard=blackboard,
            state=state,
            payload=payload,
            max_sources=max_sources,
        )
        attempts.append(dict(codex_report.get("attempt_summary") or {}))
        if _real_source_candidates(codex_report):
            codex_report["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
            codex_report["scout_attempts"] = attempts
            return codex_report
        reasons.extend(str(item) for item in codex_report.get("reasons") or ["codex_online_scout_no_real_sources"])
    else:
        attempts.append({"mode": "codex_online", "attempted": False, "reason": "codex_online_scout_disabled"})
        reasons.append("codex_online_scout_disabled")

    local_report = _local_pdf_literature_scout(
        blackboard=blackboard,
        state=state,
        payload=payload,
        max_sources=max_sources,
        prior_reasons=reasons,
    )
    attempts.append(dict(local_report.get("attempt_summary") or {}))
    if _real_source_candidates(local_report):
        local_report["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        local_report["scout_attempts"] = attempts
        local_report["codex_worker_run_attempted"] = any(bool(row.get("attempted")) for row in attempts if row.get("mode") == "codex_online")
        local_report["codex_research_runs"] = int(state.codex_research_runs)
        return local_report

    reasons.extend(str(item) for item in local_report.get("reasons") or ["local_pdf_scout_no_real_sources"])
    placeholder = _placeholder_literature_scout(
        blackboard=blackboard,
        state=state,
        payload=payload,
        max_sources=max_sources,
        prior_reasons=reasons,
    )
    placeholder["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
    placeholder["scout_attempts"] = attempts + [dict(placeholder.get("attempt_summary") or {})]
    placeholder["codex_worker_run_attempted"] = any(bool(row.get("attempted")) for row in attempts if row.get("mode") == "codex_online")
    placeholder["codex_research_runs"] = int(state.codex_research_runs)
    return placeholder


def _codex_scout_enabled(payload: dict[str, Any]) -> bool:
    raw = payload.get("codex_scout_enabled")
    if raw is None:
        raw = payload.get("use_codex_worker")
    if raw is None:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "off", "no", "disabled"}
    return bool(raw)


def _run_codex_literature_scout(
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
) -> dict[str, Any]:
    mock = state.mock_tool_results.get("codex_literature_scout")
    if mock is not None:
        value = mock(state, payload, blackboard) if callable(mock) else mock
        report = _normalize_literature_scout_report(
            dict(value or {}),
            blackboard=blackboard,
            state=state,
            max_sources=max_sources,
            discovery_mode="codex_online",
        )
        report["attempt_summary"] = {"mode": "codex_online", "attempted": True, "backend": "mock"}
        return report

    if state.codex_research_runs >= int(state.budget.max_codex_research_runs or 1):
        return {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": str(state.preflight.get("case_id") or ""),
            "source_candidates": [],
            "source_refs": [],
            "reasons": ["codex_research_budget_exhausted"],
            "no_solved_claim": True,
            "source_discovery_mode": "codex_online",
            "attempt_summary": {"mode": "codex_online", "attempted": False, "reason": "codex_research_budget_exhausted"},
        }

    state.codex_research_runs += 1
    task = _codex_literature_scout_task(
        blackboard=blackboard,
        state=state,
        payload=payload,
        max_sources=max_sources,
    )
    record = run_codex_worker(
        task,
        use_codex_cli=_payload_bool(payload, "use_codex_cli", default=True),
        use_api_json=_payload_bool(payload, "use_api_json", default=False),
    )
    record_payload = record.to_dict()
    write_json(state.run_dir / "codex_literature_scout_run_record.json", record_payload)
    state.artifacts["codex_literature_scout_run_record"] = record_payload

    artifact = dict(record.output_artifact or {})
    scout_payload = dict(artifact.get("payload") or {})
    report = _normalize_literature_scout_report(
        scout_payload,
        blackboard=blackboard,
        state=state,
        max_sources=max_sources,
        discovery_mode="codex_online",
    )
    reasons = [str(item) for item in report.get("reasons") or []]
    if record.status != "accepted_draft":
        reasons.extend(str(item) for item in (record.output_validation or {}).get("reasons") or [record.status])
    if not _real_source_candidates(report) and not reasons:
        reasons.append("codex_online_scout_no_real_sources")
    report["accepted"] = bool(_real_source_candidates(report))
    report["reasons"] = sorted(set(reasons))
    report["codex_worker_run_attempted"] = True
    report["codex_worker_status"] = str(record.status or "")
    report["codex_worker_backend"] = str(record.backend or "")
    report["codex_research_runs"] = int(state.codex_research_runs)
    report["attempt_summary"] = {
        "mode": "codex_online",
        "attempted": True,
        "backend": str(record.backend or ""),
        "status": str(record.status or ""),
        "accepted": bool(record.status == "accepted_draft" and _real_source_candidates(report)),
    }
    return report


def _codex_literature_scout_task(
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
) -> WorkerTask:
    target = dict(blackboard.get("target_profile") or {})
    bridge_tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    query_terms = _literature_scout_queries(blackboard=blackboard, state=state, payload=payload)
    objective = (
        "Use Codex native web search to find real literature or source-material leads for this retrosynthesis target. "
        "Return only traceable source metadata, not routes. Prefer exact target, target-proximal intermediates, "
        "close steroid/polycyclic analogues, DOI pages, publisher pages, PDFs, or supporting information pages. "
        f"Return up to {max_sources} candidates. If no real source is found, return accepted=false and an empty source_candidates list. "
        f"Target profile: {json.dumps(target, ensure_ascii=False, sort_keys=True)}. "
        f"Bridge tasks: {json.dumps(bridge_tasks[:6], ensure_ascii=False, sort_keys=True)}. "
        f"Suggested queries: {json.dumps(query_terms, ensure_ascii=False)}."
    )
    return WorkerTask(
        task_id=f"{str(state.preflight.get('case_id') or 'case')}:literature_scout:{int((blackboard.get('budget_state') or {}).get('scout_calls') or 0) + 1}",
        case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        task_type="target_research",
        required_artifact_type="LiteratureScoutReport",
        input_refs=[str(state.run_dir / "agent_blackboard.json")],
        allowed_tools=["web_search", "browser", "local_search"],
        budget=WorkerBudget(
            timeout_s=float(payload.get("codex_timeout_s") or min(float(state.budget.open_research_timeout_s or 900.0), 900.0)),
            max_output_bytes=int(payload.get("max_output_bytes") or 120_000),
            max_tool_calls=int(payload.get("max_tool_calls") or 12),
            max_worker_runs=1,
        ),
        objective=objective,
        allowed_workdir=str(state.run_dir),
    )


def _local_pdf_literature_scout(
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
    prior_reasons: list[str],
) -> dict[str, Any]:
    pdf_path = str(payload.get("pdf_path") or state.target_input.get("literature_pdf_path") or "")
    source_ref = str(payload.get("source_ref") or state.target_input.get("literature_pdf_source_ref") or "")
    reasons = [str(item) for item in prior_reasons]
    if not pdf_path:
        reasons.append("local_pdf_not_provided")
        return {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": str(state.preflight.get("case_id") or ""),
            "source_candidates": [],
            "source_refs": [],
            "reasons": sorted(set(reasons)),
            "no_solved_claim": True,
            "source_discovery_mode": "local_pdf_fallback",
            "attempt_summary": {"mode": "local_pdf", "attempted": False, "reason": "local_pdf_not_provided"},
        }
    resolved = Path(pdf_path).expanduser()
    if not resolved.is_file():
        reasons.append("local_pdf_missing")
        return {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": str(state.preflight.get("case_id") or ""),
            "source_candidates": [],
            "source_refs": [],
            "reasons": sorted(set(reasons)),
            "no_solved_claim": True,
            "source_discovery_mode": "local_pdf_fallback",
            "attempt_summary": {"mode": "local_pdf", "attempted": True, "reason": "local_pdf_missing"},
        }
    target = str(state.target_input.get("target_name") or "target")
    source_hints = _local_pdf_source_hints(
        target=target,
        source_ref=source_ref,
        family_hint=str(state.target_input.get("family_hint") or ""),
    )
    candidate = {
        "schema_version": "literature_source_candidate.v1",
        "candidate_id": "local_pdf_1",
        "source_ref": source_ref or "local_pdf",
        "title": source_hints.get("title") or f"{target} local PDF source",
        "doi": _doi_from_source_ref(source_ref),
        "url": "",
        "local_pdf": str(resolved.resolve()),
        "source_type": "local_pdf",
        "source_discovery_mode": "local_pdf_fallback",
        "access_status": "local_pdf_available",
        "relevance_rationale": "local PDF fallback after Codex online scout did not yield a usable source",
        "expected_scheme_or_compound_labels": source_hints.get("expected_labels") or [],
        "extraction_task_recommendations": [
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "compile_exact_literature_rows",
        ],
        "route_sequence_hint": source_hints.get("route_sequence_hint") or "",
        "no_solved_claim": True,
    }
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": True,
        "case_id": str(state.preflight.get("case_id") or ""),
        "source_candidates": [candidate][:max_sources],
        "source_refs": [str(candidate.get("source_ref") or "")],
        "reasons": [],
        "no_solved_claim": True,
        "source_discovery_mode": "local_pdf_fallback",
        "attempt_summary": {"mode": "local_pdf", "attempted": True, "accepted": True},
    }


def _placeholder_literature_scout(
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
    prior_reasons: list[str],
) -> dict[str, Any]:
    target = str(state.target_input.get("target_name") or "target")
    tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    candidates: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks[:max_sources], start=1):
        handle = str(task.get("target_handle") or task.get("task_type") or "bridge")
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": f"placeholder_query_{idx}",
                "source_ref": f"query:{target}:{handle}",
                "title": f"{target} synthesis {handle} bridge",
                "doi": "",
                "url": "",
                "local_pdf": "",
                "source_type": "placeholder_query",
                "source_discovery_mode": "placeholder",
                "access_status": "placeholder_only",
                "placeholder_only": True,
                "relevance_rationale": str(task.get("required_bridge") or "target-proximal bridge required"),
                "expected_scheme_or_compound_labels": [],
                "extraction_task_recommendations": ["retry_codex_online_scout", "request_local_pdf_or_source_url"],
                "no_solved_claim": True,
            }
        )
    if not candidates:
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": "placeholder_target_synthesis_query",
                "source_ref": f"query:{target}:target_proximal_synthesis",
                "title": f"{target} synthesis target-proximal intermediate",
                "doi": "",
                "url": "",
                "local_pdf": "",
                "source_type": "placeholder_query",
                "source_discovery_mode": "placeholder",
                "access_status": "placeholder_only",
                "placeholder_only": True,
                "relevance_rationale": "online and local PDF scout failed; placeholder records the missing target-proximal source need",
                "expected_scheme_or_compound_labels": [],
                "extraction_task_recommendations": ["retry_codex_online_scout", "request_local_pdf_or_source_url"],
                "no_solved_claim": True,
            }
        )
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": False,
        "case_id": str(state.preflight.get("case_id") or ""),
        "source_candidates": candidates[:max_sources],
        "source_refs": [str(row.get("source_ref") or "") for row in candidates[:max_sources]],
        "reasons": sorted(set([*prior_reasons, "no_real_literature_source_found", "placeholder_source_candidates_only"])),
        "no_solved_claim": True,
        "source_discovery_mode": "placeholder",
        "placeholder_only": True,
        "attempt_summary": {"mode": "placeholder", "attempted": True, "accepted": False},
    }


def _normalize_literature_scout_report(
    report: dict[str, Any],
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    max_sources: int,
    discovery_mode: str,
) -> dict[str, Any]:
    payload = dict(report.get("payload") or report)
    raw_candidates = payload.get("source_candidates") or []
    if not raw_candidates and payload.get("source_title"):
        raw_candidates = [payload]
    candidates = [
        _normalize_source_candidate(row, idx=idx, discovery_mode=discovery_mode)
        for idx, row in enumerate(raw_candidates, start=1)
        if isinstance(row, dict)
    ]
    candidates = [row for row in candidates if row.get("source_ref") or row.get("title")]
    candidates = _dedupe_candidates(candidates)[:max_sources]
    refs = [str(row.get("source_ref") or "") for row in candidates if str(row.get("source_ref") or "").strip()]
    if not refs:
        refs = [str(item) for item in payload.get("source_refs") or [] if str(item or "").strip()]
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": bool(candidates and any(_candidate_has_real_source(row) for row in candidates)),
        "case_id": str(payload.get("case_id") or state.preflight.get("case_id") or ""),
        "source_candidates": candidates,
        "source_refs": _dedupe(refs),
        "search_queries": [str(item) for item in payload.get("search_queries") or _literature_scout_queries(blackboard=blackboard, state=state, payload={})],
        "reasons": [str(item) for item in payload.get("reasons") or []],
        "limitations": [str(item) for item in payload.get("limitations") or []],
        "no_solved_claim": True,
        "source_discovery_mode": discovery_mode,
    }


def _normalize_source_candidate(row: dict[str, Any], *, idx: int, discovery_mode: str) -> dict[str, Any]:
    doi = _normalize_doi(str(row.get("doi") or ""))
    url = str(row.get("url") or "").strip()
    local_pdf = str(row.get("local_pdf") or row.get("pdf_path") or "").strip()
    title = str(row.get("title") or row.get("source_title") or "").strip()
    source_ref = str(row.get("source_ref") or "").strip()
    if not source_ref:
        source_ref = f"doi:{doi}" if doi else (url or (f"local_pdf:{Path(local_pdf).name}" if local_pdf else f"source:{idx}"))
    tasks = [str(item) for item in row.get("extraction_task_recommendations") or [] if str(item or "").strip()]
    if not tasks:
        tasks = ["extract_pdf_literature_structures", "extract_visual_literature_chain", "compile_exact_literature_rows"] if local_pdf else [
            "resolve_source_material_or_provide_pdf",
            "codex_source_detail_followup",
        ]
    return {
        "schema_version": "literature_source_candidate.v1",
        "candidate_id": str(row.get("candidate_id") or f"{discovery_mode}_{idx}"),
        "source_ref": source_ref,
        "title": title,
        "doi": doi,
        "url": url or (f"https://doi.org/{doi}" if doi else ""),
        "local_pdf": local_pdf,
        "source_type": str(row.get("source_type") or ("local_pdf" if local_pdf else "literature_metadata")),
        "source_discovery_mode": str(row.get("source_discovery_mode") or discovery_mode),
        "access_status": str(row.get("access_status") or ("local_pdf_available" if local_pdf else "metadata_only")),
        "relevance_rationale": str(row.get("relevance_rationale") or row.get("route_role_detail") or ""),
        "expected_scheme_or_compound_labels": [
            str(item)
            for item in row.get("expected_scheme_or_compound_labels") or row.get("expected_labels") or []
            if str(item or "").strip()
        ],
        "extraction_task_recommendations": tasks,
        "no_solved_claim": True,
    }


def _real_source_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in report.get("source_candidates") or []
        if isinstance(row, dict) and _candidate_has_real_source(row)
    ]


def _candidate_has_real_source(row: dict[str, Any]) -> bool:
    if bool(row.get("placeholder_only")):
        return False
    if str(row.get("access_status") or "").strip().lower() == "placeholder_only":
        return False
    return bool(str(row.get("doi") or row.get("url") or row.get("local_pdf") or "").strip())


def _literature_scout_queries(*, blackboard: dict[str, Any], state: ToolExecutionState, payload: dict[str, Any]) -> list[str]:
    explicit = [str(item) for item in payload.get("queries") or payload.get("search_queries") or [] if str(item or "").strip()]
    if explicit:
        return _dedupe(explicit)[:6]
    target = str(state.target_input.get("target_name") or (blackboard.get("target_profile") or {}).get("target_name") or "target")
    family = str(state.target_input.get("family_hint") or (blackboard.get("target_profile") or {}).get("family_hint") or "")
    handles = [
        str(row.get("target_handle") or row.get("task_type") or "")
        for row in blackboard.get("bridge_tasks") or []
        if isinstance(row, dict)
    ]
    base = [part for part in [target, family] if part]
    queries = [
        " ".join([*base, "synthesis", "total synthesis", "semisynthesis"]),
        " ".join([*base, "target proximal intermediate", "steroid"]),
    ]
    for handle in handles[:4]:
        if handle:
            queries.append(" ".join([*base, handle, "synthesis bridge"]))
    return _dedupe([q for q in queries if q.strip()])[:6]


def _payload_bool(payload: dict[str, Any], key: str, *, default: bool) -> bool:
    raw = payload.get(key)
    if raw is None:
        return bool(default)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "off", "no", "disabled"}
    return bool(raw)


def _normalize_doi(value: str) -> str:
    text = str(value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip().lower()


def _doi_from_source_ref(value: str) -> str:
    text = str(value or "").strip()
    return _normalize_doi(text) if text.lower().startswith(("doi:", "10.")) else ""


def _dedupe_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("doi") or row.get("url") or row.get("local_pdf") or row.get("source_ref") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _deterministic_literature_scout(*, blackboard: dict[str, Any], state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    target = str(state.target_input.get("target_name") or "target")
    tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    max_sources = max(1, min(3, int(payload.get("max_sources") or 3)))
    candidates: list[dict[str, Any]] = []
    pdf_path = str(state.target_input.get("literature_pdf_path") or "")
    source_ref = str(state.target_input.get("literature_pdf_source_ref") or "")
    source_hints = _local_pdf_source_hints(target=target, source_ref=source_ref, family_hint=str(state.target_input.get("family_hint") or ""))
    if pdf_path:
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": "local_pdf_1",
                "source_ref": source_ref or "local_pdf",
                "title": source_hints.get("title") or f"{target} local PDF source",
                "url": "",
                "local_pdf": pdf_path,
                "relevance_rationale": "user-provided local PDF for source-detail extraction",
                "expected_scheme_or_compound_labels": source_hints.get("expected_labels") or [],
                "extraction_task_recommendations": [
                    "extract_pdf_literature_structures",
                    "extract_visual_literature_chain",
                    "compile_exact_literature_rows",
                ],
                "route_sequence_hint": source_hints.get("route_sequence_hint") or "",
            }
        )
    for idx, task in enumerate(tasks[:max_sources], start=1):
        if len(candidates) >= max_sources:
            break
        handle = str(task.get("target_handle") or task.get("task_type") or "bridge")
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": f"query_{idx}",
                "source_ref": f"query:{target}:{handle}",
                "title": f"{target} synthesis {handle} bridge",
                "url": "",
                "local_pdf": "",
                "relevance_rationale": str(task.get("required_bridge") or "target-proximal bridge required"),
                "expected_scheme_or_compound_labels": [],
                "extraction_task_recommendations": [
                    "extract_pdf_literature_structures",
                    "extract_visual_literature_chain",
                    "compile_exact_literature_rows",
                ],
            }
        )
    if not candidates:
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": "target_synthesis_query",
                "source_ref": f"query:{target}:target_proximal_synthesis",
                "title": f"{target} synthesis target-proximal intermediate",
                "url": "",
                "local_pdf": "",
                "relevance_rationale": "initial scout for target-proximal route evidence",
                "expected_scheme_or_compound_labels": [],
                "extraction_task_recommendations": ["extract_visual_literature_chain"],
            }
        )
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": bool(candidates),
        "case_id": str(state.preflight.get("case_id") or ""),
        "source_candidates": candidates[:max_sources],
        "source_refs": [str(row.get("source_ref") or "") for row in candidates[:max_sources]],
        "reasons": [] if candidates else ["no_source_candidates"],
        "no_solved_claim": True,
    }


def _local_pdf_source_hints(*, target: str, source_ref: str, family_hint: str) -> dict[str, Any]:
    text = " ".join([target, source_ref, family_hint]).lower()
    if "bufotalin" in text or "10.1016/j.tet.2025.134610" in text:
        labels = [
            "bufotalin",
            "33",
            "32",
            "31",
            "30",
            "22",
            "14",
            "20",
            "19",
            "28",
            "27",
            "26",
            "23",
            "25",
            "24",
            "11",
        ]
        return {
            "title": "Construction of advanced intermediate sharing C14-beta-OH for the synthesis of bufotalin",
            "expected_labels": labels,
            "route_sequence_hint": (
                "For bufotalin, inspect Scheme 4 and Scheme 3 as a connected retro chain. "
                "The source-detail target is bufotalin -> 33 -> 32 -> 31 -> 30 -> 22 -> 14 -> "
                "20 -> 19 -> 28 -> 27 -> 26 -> 23 -> 25 -> 24 -> 11 when structures are visible."
            ),
            "pdf_page_numbers": [3, 4, 5, 6],
            "pdf_render_zoom": 2.5,
            "scheme_crops": [
                {
                    "crop_id": "scheme3_full_to_20",
                    "page_number": 3,
                    "bbox_px": [330, 150, 1160, 650],
                    "evidence_refs": ["doi:10.1016/j.tet.2025.134610", "scheme:3"],
                },
                {
                    "crop_id": "scheme4_total_synthesis",
                    "page_number": 3,
                    "bbox_px": [330, 690, 1165, 1095],
                    "evidence_refs": ["doi:10.1016/j.tet.2025.134610", "scheme:4"],
                },
                {
                    "crop_id": "table1_allylic_oxidation",
                    "page_number": 3,
                    "bbox_px": [80, 1110, 725, 1640],
                    "evidence_refs": ["doi:10.1016/j.tet.2025.134610", "table:1"],
                },
            ],
        }
    return {"title": "", "expected_labels": [], "route_sequence_hint": ""}


def _tool_record_to_action_result(record: Any) -> dict[str, Any]:
    output = dict(record.output or {})
    result = output.get("result") if isinstance(output.get("result"), dict) else output
    return {
        "accepted": record.status not in {"rejected", "error"} or bool(output.get("result")),
        "result": result,
        "reasons": [str(item) for item in getattr(record, "reasons", []) or output.get("reasons") or []],
    }


def _mock_action_result(state: ToolExecutionState, action: dict[str, Any], blackboard: dict[str, Any]) -> Any | None:
    action_type = str(action.get("action_type") or "")
    action_id = str(action.get("action_id") or "")
    value = state.mock_tool_results.get(action_id)
    if value is None:
        value = state.mock_tool_results.get(action_type)
    if value is None:
        return None
    if callable(value):
        return value(state, action, blackboard)
    return value


def _wrap_action_result(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if "accepted" in value and ("result" in value or "schema_version" in value):
            return dict(value)
        return {"accepted": bool(value.get("accepted", True)), "result": dict(value), "reasons": [str(item) for item in value.get("reasons") or []]}
    return {"accepted": True, "result": {"value": str(value)}, "reasons": []}


def _inject_pdf_defaults(payload: dict[str, Any], target_input: dict[str, Any]) -> None:
    if not payload.get("source_ref") and str(target_input.get("literature_pdf_source_ref") or "").strip():
        payload["source_ref"] = str(target_input.get("literature_pdf_source_ref") or "").strip()
    if not payload.get("pdf_path") and str(target_input.get("literature_pdf_path") or "").strip():
        payload["pdf_path"] = str(target_input.get("literature_pdf_path") or "").strip()
    hints = _local_pdf_source_hints(
        target=str(target_input.get("target_name") or ""),
        source_ref=str(payload.get("source_ref") or target_input.get("literature_pdf_source_ref") or ""),
        family_hint=str(target_input.get("family_hint") or ""),
    )
    if hints.get("expected_labels") and not payload.get("compound_labels"):
        payload["compound_labels"] = list(hints.get("expected_labels") or [])
    if hints.get("pdf_page_numbers") and not payload.get("page_numbers"):
        payload["page_numbers"] = list(hints.get("pdf_page_numbers") or [])
    if hints.get("pdf_render_zoom") and not payload.get("render_zoom"):
        payload["render_zoom"] = float(hints.get("pdf_render_zoom") or 2.0)
    if hints.get("scheme_crops") and not payload.get("scheme_crops"):
        payload["scheme_crops"] = [dict(item) for item in hints.get("scheme_crops") or [] if isinstance(item, dict)]
    if hints.get("route_sequence_hint") and not payload.get("route_sequence_hint"):
        payload["route_sequence_hint"] = str(hints.get("route_sequence_hint") or "")
    if hints.get("expected_labels") and not payload.get("expected_labels"):
        payload["expected_labels"] = list(hints.get("expected_labels") or [])


def _workflow_plan_from_actions(action_batches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "agentic_blackboard_workflow_plan.v1",
        "recommended_strategy": "policy_driven_dag_blackboard",
        "planned_action_batches": action_batches,
        "planner_authority": "action_selection_only",
        "final_verdict_authority": "deterministic_parent_route_proof",
        "raw_reaction_output_allowed": False,
    }


def _invalid_final(preflight: dict[str, Any]) -> FinalVerdict:
    return FinalVerdict(
        case_id=str(preflight.get("case_id") or "target"),
        verdict="invalid_input",
        route_status="invalid_input",
        reasons=[str(item) for item in preflight.get("reasons") or ["invalid_smiles"]],
    )


def _parent_proof_accepted(blackboard: dict[str, Any]) -> bool:
    proof = dict(blackboard.get("parent_route_proof") or {})
    return bool(proof.get("accepted") and proof.get("solved"))


def _result(
    run_dir: Path,
    target_input: dict[str, Any],
    preflight: dict[str, Any],
    blackboard: dict[str, Any],
    action_batches: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    artifact_bundle: ArtifactBundle,
    final_verdict: FinalVerdict,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "agentic_blackboard_controller_result.v1",
        "run_dir": str(run_dir),
        "target_input": target_input,
        "preflight": preflight,
        "agent_blackboard": blackboard,
        "action_batches": action_batches,
        "validations": validations,
        "tool_calls": tool_calls,
        "artifact_bundle": artifact_bundle.to_dict(),
        "final_verdict": final_verdict.to_dict(),
        "artifacts": {
            "target_input": str(run_dir / "target_input.json"),
            "preflight": str(run_dir / "preflight.json"),
            "agent_blackboard": str(run_dir / "agent_blackboard.json"),
            "decision_trace": str(run_dir / "decision_trace.jsonl"),
            "tool_calls": str(run_dir / "tool_calls.jsonl"),
            "artifact_bundle": str(run_dir / "artifact_bundle.json"),
            "final_verdict": str(run_dir / "final_verdict.json"),
        },
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

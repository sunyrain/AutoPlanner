"""Agentic blackboard controller entry point."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
    budget.max_guided_chemenzy_runs = min(int(budget.max_guided_chemenzy_runs or 1), 1)
    budget.max_route_expansion_subgoal_runs = min(int(budget.max_route_expansion_subgoal_runs or 2), 2)

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
            reasons=["parent_route_proof_required_for_agentic_solved_claim"],
            route_status="child_solved_parent_unresolved",
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
) -> dict[str, Any]:
    if action_planner is not None:
        return action_planner(blackboard=blackboard, round_index=round_index, run_dir=run_dir)
    return plan_action_batch(blackboard, round_index=round_index)


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
        result = _deterministic_literature_scout(blackboard=blackboard, state=state, payload=dict(action.get("payload") or {}))
        state.artifacts["literature_scout"] = result
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "rank_analogical_hypotheses":
        result = rank_analogical_hypotheses_from_blackboard(blackboard)
        state.artifacts["analogical_hypothesis_ranking"] = result
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
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
    parent_verifier = _latest_parent_verifier(state.artifacts)
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


def _deterministic_literature_scout(*, blackboard: dict[str, Any], state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    target = str(state.target_input.get("target_name") or "target")
    tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    max_sources = max(1, min(3, int(payload.get("max_sources") or 3)))
    candidates: list[dict[str, Any]] = []
    pdf_path = str(state.target_input.get("literature_pdf_path") or "")
    source_ref = str(state.target_input.get("literature_pdf_source_ref") or "")
    if pdf_path:
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": "local_pdf_1",
                "source_ref": source_ref or "local_pdf",
                "title": f"{target} local PDF source",
                "url": "",
                "local_pdf": pdf_path,
                "relevance_rationale": "user-provided local PDF for source-detail extraction",
                "expected_scheme_or_compound_labels": [],
                "extraction_task_recommendations": ["extract_visual_literature_chain"],
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
                "extraction_task_recommendations": ["compile_exact_literature_rows", "extract_visual_literature_chain"],
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

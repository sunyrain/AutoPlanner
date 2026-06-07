"""Codex-entry harness runner."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cascade_planner.harness.codex_plan import (
    DEFAULT_BASE_URL,
    DEFAULT_KEY_PATH,
    DEFAULT_MODEL,
    deterministic_workflow_plan,
    plan_workflow_with_codex,
)
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.progress import write_progress_panel
from cascade_planner.harness.schemas import (
    CANONICAL_RUN_SEMANTICS,
    FINAL_VERDICTS,
    WORKFLOW_PLAN_SCHEMA,
    ArtifactBundle,
    FinalVerdict,
    TargetInput,
    WorkflowPlan,
    append_jsonl,
    validate_workflow_plan,
    workflow_plan_from_dict,
    write_json,
)
from cascade_planner.harness.tools import (
    HarnessBudget,
    ToolExecutionState,
    artifact_bundle_from_state,
    execute_local_tool,
)


PlannerRunner = Callable[..., dict[str, Any]]


def run_codex_entry_controller(
    *,
    target_name: str,
    target_smiles: str,
    family_hint: str = "",
    output_dir: str | Path,
    timeout_s: float = 1800.0,
    key_path: str | Path = DEFAULT_KEY_PATH,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    use_live_planner: bool = True,
    planner_plan: dict[str, Any] | WorkflowPlan | None = None,
    planner_runner: PlannerRunner | None = None,
    mock_tool_results: dict[str, Any] | None = None,
    budget: HarnessBudget | None = None,
) -> dict[str, Any]:
    """Run the full Codex-entry harness and write a self-contained run dir."""
    run_dir = Path(output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tool_calls.jsonl").touch()
    budget = budget or HarnessBudget(timeout_s=float(timeout_s))
    target = TargetInput(
        target_name=target_name,
        target_smiles=target_smiles,
        family_hint=family_hint,
        case_id="",
    )
    write_json(run_dir / "target_input.json", target.to_dict())
    write_json(run_dir / "budget.json", budget.to_dict())
    append_jsonl(run_dir / "decision_trace.jsonl", {
        "stage": "start",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_name": target_name,
    })

    preflight = run_preflight(target)
    target.case_id = str(preflight.get("case_id") or target.case_id)
    target_data = target.to_dict()
    write_json(run_dir / "target_input.json", target_data)
    write_json(run_dir / "preflight.json", preflight)
    append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "preflight", "preflight": preflight})

    tool_calls: list[dict[str, Any]] = []
    if not preflight.get("accepted"):
        plan = deterministic_workflow_plan(target_input=target_data, preflight=preflight)
        plan_data = plan.to_dict()
        write_json(run_dir / "codex_workflow_plan.json", plan_data)
        state = ToolExecutionState(
            run_dir=run_dir,
            target_input=target_data,
            preflight=preflight,
            budget=budget,
            key_path=key_path,
            base_url=base_url,
            model=model,
            mock_tool_results=dict(mock_tool_results or {}),
        )
        bundle = artifact_bundle_from_state(state=state, workflow_plan=plan_data, tool_calls=tool_calls)
        write_json(run_dir / "artifact_bundle.json", bundle.to_dict())
        verdict = FinalVerdict(
            case_id=str(preflight.get("case_id") or "target"),
            verdict="invalid_input",
            reasons=[str(item) for item in preflight.get("reasons") or ["invalid_smiles"]],
            route_status="invalid_input",
        )
        write_json(run_dir / "final_verdict.json", verdict.to_dict())
        write_progress_panel(
            run_dir=run_dir,
            target_input=target_data,
            preflight=preflight,
            workflow_plan=plan_data,
            final_verdict=verdict.to_dict(),
            tool_calls=tool_calls,
        )
        return _run_result(run_dir, target_data, preflight, plan_data, bundle, verdict, tool_calls)

    plan_record = _obtain_plan(
        target_data=target_data,
        preflight=preflight,
        run_dir=run_dir,
        timeout_s=timeout_s,
        key_path=key_path,
        base_url=base_url,
        model=model,
        use_live_planner=use_live_planner,
        planner_plan=planner_plan,
        planner_runner=planner_runner,
    )
    plan_data = dict(plan_record.get("workflow_plan") or {})
    write_json(run_dir / "codex_workflow_plan.json", plan_data)
    write_json(run_dir / "codex_planner_run_record.json", plan_record)
    append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "codex_plan", "planner_record": plan_record})

    plan_validation = validate_workflow_plan(plan_data, case_id=str(preflight.get("case_id") or ""))
    if not plan_record.get("accepted") or not plan_validation.get("accepted"):
        state = ToolExecutionState(
            run_dir=run_dir,
            target_input=target_data,
            preflight=preflight,
            budget=budget,
            key_path=key_path,
            base_url=base_url,
            model=model,
            mock_tool_results=dict(mock_tool_results or {}),
        )
        state.validations.append(plan_validation)
        bundle = artifact_bundle_from_state(state=state, workflow_plan=plan_data, tool_calls=tool_calls)
        write_json(run_dir / "artifact_bundle.json", bundle.to_dict())
        verdict = FinalVerdict(
            case_id=str(preflight.get("case_id") or "target"),
            verdict="needs_followup",
            reasons=[str(item) for item in plan_validation.get("reasons") or plan_record.get("reasons") or ["planner_rejected"]],
            route_status="planner_rejected",
        )
        write_json(run_dir / "final_verdict.json", verdict.to_dict())
        write_progress_panel(
            run_dir=run_dir,
            target_input=target_data,
            preflight=preflight,
            workflow_plan=plan_data,
            final_verdict=verdict.to_dict(),
            tool_calls=tool_calls,
        )
        return _run_result(run_dir, target_data, preflight, plan_data, bundle, verdict, tool_calls)

    state = ToolExecutionState(
        run_dir=run_dir,
        target_input=target_data,
        preflight=preflight,
        budget=budget,
        key_path=key_path,
        base_url=base_url,
        model=model,
        mock_tool_results=dict(mock_tool_results or {}),
    )
    for row in plan_data.get("planned_tools") or []:
        tool_name = str(row.get("tool_name") or row.get("name") or "")
        payload = dict(row.get("payload") or row.get("input") or {})
        if tool_name == "emit_final_verdict":
            append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "emit_final_verdict_deferred"})
            continue
        record = execute_local_tool(tool_name, payload, state)
        tool_calls.append(record.to_dict())
        can_continue = _tool_rejection_has_continuation(tool_name=tool_name, record=record, state=state)
        if record.status in {"rejected", "error"} and tool_name not in {
            "run_chemenzy",
            "run_guided_chemenzy_rerun",
            "run_route_expansion_subgoal_search",
        } and not can_continue:
            break

    bundle = artifact_bundle_from_state(state=state, workflow_plan=plan_data, tool_calls=tool_calls)
    bundle_validation = _ensure_bundle_validation(bundle)
    if bundle_validation:
        state.validations.append(bundle_validation)
        bundle.validations = list(state.validations)
    verdict = emit_final_verdict(bundle)
    write_json(run_dir / "artifact_bundle.json", bundle.to_dict())
    write_json(run_dir / "final_verdict.json", verdict.to_dict())
    append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "final_verdict", "final_verdict": verdict.to_dict()})
    write_progress_panel(
        run_dir=run_dir,
        target_input=target_data,
        preflight=preflight,
        workflow_plan=plan_data,
        final_verdict=verdict.to_dict(),
        tool_calls=tool_calls,
    )
    return _run_result(run_dir, target_data, preflight, plan_data, bundle, verdict, tool_calls)


def emit_final_verdict(bundle_or_data: ArtifactBundle | dict[str, Any]) -> FinalVerdict:
    bundle = bundle_or_data if isinstance(bundle_or_data, ArtifactBundle) else ArtifactBundle(
        case_id=str(bundle_or_data.get("case_id") or "target"),
        target_input=dict(bundle_or_data.get("target_input") or {}),
        preflight=dict(bundle_or_data.get("preflight") or {}),
        workflow_plan=dict(bundle_or_data.get("workflow_plan") or {}),
        tool_calls=[dict(item) for item in bundle_or_data.get("tool_calls") or []],
        artifacts=dict(bundle_or_data.get("artifacts") or {}),
        validations=[dict(item) for item in bundle_or_data.get("validations") or []],
        safety_flags=[str(item) for item in bundle_or_data.get("safety_flags") or []],
        run_semantics=str(bundle_or_data.get("run_semantics") or CANONICAL_RUN_SEMANTICS),
    )
    if not bundle.preflight.get("accepted"):
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="invalid_input",
            reasons=[str(item) for item in bundle.preflight.get("reasons") or ["invalid_smiles"]],
            route_status="invalid_input",
        )
    artifacts = dict(bundle.artifacts or {})
    audit = _latest_route_audit(artifacts)
    route_status = str(audit.get("route_status") or "")
    reasons = [str(item) for item in audit.get("reasons") or []]
    validation_reasons = [
        str(reason)
        for validation in bundle.validations
        for reason in validation.get("reasons") or []
    ]
    if audit.get("fake_closure_rejected") or route_status == "fake_closed_rejected" or "fake_closure_evidence_present" in validation_reasons:
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="fake_closed_rejected",
            reasons=sorted(set(reasons + validation_reasons + ["fake_closure_evidence_present"])),
            route_status="fake_closed_rejected",
            stock_audit_passed=bool(audit.get("stock_audit_passed")),
        )
    if _artifact_tree_contains_raw_reaction(artifacts) or "raw_reaction_injection" in validation_reasons:
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="fake_closed_rejected",
            reasons=sorted(set(validation_reasons + ["raw_reaction_injection"])),
            route_status=route_status or "artifact_rejected",
            stock_audit_passed=bool(audit.get("stock_audit_passed")),
        )
    if route_status == "solved" and audit.get("stock_audit_passed") and bundle.run_semantics != CANONICAL_RUN_SEMANTICS:
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="needs_followup",
            reasons=sorted(set(reasons + ["noncanonical_run_cannot_claim_solved"])),
            route_status="needs_followup",
            stock_audit_passed=bool(audit.get("stock_audit_passed")),
            run_semantics=bundle.run_semantics,
        )
    if route_status == "solved" and audit.get("stock_audit_passed"):
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="solved",
            reasons=[],
            route_status="solved",
            solved=True,
            stock_audit_passed=True,
            run_semantics=bundle.run_semantics,
        )
    if route_status == "solved" and not audit.get("stock_audit_passed"):
        reasons.append("solved_requires_stock_audit")
    tool_failed = any(row.get("status") in {"rejected", "error"} for row in bundle.tool_calls)
    if tool_failed:
        tool_failure_reasons = [
            str(reason)
            for row in bundle.tool_calls
            if row.get("status") in {"rejected", "error"}
            for reason in row.get("reasons") or []
        ]
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="needs_followup",
            reasons=sorted(set(reasons + validation_reasons + tool_failure_reasons + ["tool_execution_failed"])),
            route_status=route_status or "needs_followup",
            stock_audit_passed=bool(audit.get("stock_audit_passed")),
        )
    if _has_literature_or_template_anchor(artifacts) or route_status in {"partial_anchor", "semisynthesis_closed"}:
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="partial_anchor_only_not_solved",
            reasons=sorted(set(reasons + ["literature_anchor_without_executable_stock_closure"])),
            route_status=route_status or "partial_anchor",
            stock_audit_passed=bool(audit.get("stock_audit_passed")),
        )
    return FinalVerdict(
        case_id=bundle.case_id,
        verdict="unresolved",
        reasons=sorted(set(reasons + validation_reasons + ["no_deterministic_stock_closed_route"])),
        route_status=route_status or "unresolved",
        stock_audit_passed=bool(audit.get("stock_audit_passed")),
    )


def _obtain_plan(
    *,
    target_data: dict[str, Any],
    preflight: dict[str, Any],
    run_dir: Path,
    timeout_s: float,
    key_path: str | Path,
    base_url: str,
    model: str,
    use_live_planner: bool,
    planner_plan: dict[str, Any] | WorkflowPlan | None,
    planner_runner: PlannerRunner | None,
) -> dict[str, Any]:
    if planner_plan is not None:
        plan = planner_plan if isinstance(planner_plan, WorkflowPlan) else workflow_plan_from_dict(planner_plan)
        validation = validate_workflow_plan(plan, case_id=str(preflight.get("case_id") or ""))
        return {
            "schema_version": "codex_entry_planner_run.v1",
            "accepted": bool(validation.get("accepted")),
            "workflow_plan": plan.to_dict(),
            "validation": validation,
            "mode": "injected",
        }
    if planner_runner is not None:
        return planner_runner(
            target_input=target_data,
            preflight=preflight,
            run_dir=run_dir,
            timeout_s=timeout_s,
            key_path=key_path,
            base_url=base_url,
            model=model,
        )
    if use_live_planner:
        return plan_workflow_with_codex(
            target_input=target_data,
            preflight=preflight,
            run_dir=run_dir,
            timeout_s=timeout_s,
            key_path=key_path,
            base_url=base_url,
            model=model,
        )
    plan = deterministic_workflow_plan(target_input=target_data, preflight=preflight)
    validation = validate_workflow_plan(plan, case_id=str(preflight.get("case_id") or ""))
    return {
        "schema_version": "codex_entry_planner_run.v1",
        "accepted": bool(validation.get("accepted")),
        "workflow_plan": plan.to_dict(),
        "validation": validation,
        "mode": "deterministic_offline",
    }


def _ensure_bundle_validation(bundle: ArtifactBundle) -> dict[str, Any] | None:
    if any(validation.get("schema_version") == "codex_entry_artifact_bundle_validation.v1" for validation in bundle.validations):
        return None
    reasons: list[str] = []
    audit = _latest_route_audit(bundle.artifacts)
    if audit.get("route_status") == "solved" and not audit.get("stock_audit_passed"):
        reasons.append("solved_without_stock_audit")
    if audit.get("fake_closure_rejected"):
        reasons.append("fake_closure_evidence_present")
    open_research = bundle.artifacts.get("open_structure_research")
    if isinstance(open_research, dict) and not open_research.get("accepted", True):
        reasons.extend(str(item) for item in open_research.get("reasons") or ["open_structure_research_failed"])
    subgoal_search = bundle.artifacts.get("route_expansion_subgoal_search")
    if isinstance(subgoal_search, dict) and not subgoal_search.get("accepted", True):
        reasons.extend(str(item) for item in subgoal_search.get("reasons") or ["route_expansion_subgoal_search_failed"])
    self_evo_replay = bundle.artifacts.get("self_evo_replay")
    if isinstance(self_evo_replay, dict):
        if self_evo_replay.get("target_run") and int(self_evo_replay.get("production_promoted_count") or 0):
            reasons.append("self_evo_target_run_promoted_production")
        if self_evo_replay.get("target_run") and not self_evo_replay.get("production_write_blocked"):
            reasons.append("self_evo_target_run_production_not_blocked")
    if _artifact_tree_contains_raw_reaction(bundle.artifacts):
        reasons.append("raw_reaction_injection")
    return {
        "schema_version": "codex_entry_artifact_bundle_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "artifact_keys": sorted(bundle.artifacts),
    }


def _tool_rejection_has_continuation(*, tool_name: str, record: Any, state: ToolExecutionState) -> bool:
    if tool_name != "run_open_structure_research_agent":
        return False
    if not isinstance(getattr(record, "output", None), dict):
        return False
    result = dict(record.output)
    if not result.get("continuation_available"):
        return False
    compiled = state.artifacts.get("compiled_downstream")
    return isinstance(compiled, dict) and bool(compiled.get("accepted"))


def _latest_route_audit(artifacts: dict[str, Any]) -> dict[str, Any]:
    guided = artifacts.get("guided_chemenzy")
    if isinstance(guided, dict):
        verifier = dict(guided.get("raw_route_verifier") or {})
        if verifier:
            route_status = str(verifier.get("route_status") or "")
            reasons = [str(item) for item in verifier.get("reasons") or []]
            if not verifier.get("accepted"):
                reasons.append("route_verifier_rejected_raw_routes")
            return {
                "schema_version": "route_audit_report.v1",
                "case_id": guided.get("case_id") or verifier.get("case_id") or "",
                "route_status": "solved" if verifier.get("accepted") else route_status or "fake_closed_rejected",
                "stock_audit_passed": bool(verifier.get("accepted")),
                "fake_closure_rejected": bool(not verifier.get("accepted") and route_status == "fake_closed_rejected"),
                "reasons": sorted(set(reasons)),
                "rejected_terminal_list": list(verifier.get("rejected_terminal_list") or []),
                "failure_events": list(verifier.get("failure_events") or []),
            }
    verifier_sources: list[dict[str, Any]] = []
    route_verifier = artifacts.get("route_verifier")
    if isinstance(route_verifier, dict):
        verifier_sources.append(route_verifier)
    chemenzy = artifacts.get("chemenzy")
    if isinstance(chemenzy, dict):
        embedded = chemenzy.get("raw_route_verifier")
        if isinstance(embedded, dict):
            verifier_sources.append(embedded)
        result = chemenzy.get("result")
        if isinstance(result, dict) and isinstance(result.get("raw_route_verifier"), dict):
            verifier_sources.append(dict(result["raw_route_verifier"]))
    for verifier in verifier_sources:
        if verifier:
            route_status = str(verifier.get("route_status") or "")
            reasons = [str(item) for item in verifier.get("reasons") or []]
            if not verifier.get("accepted"):
                reasons.append("route_verifier_rejected_raw_routes")
            return {
                "schema_version": "route_audit_report.v1",
                "case_id": verifier.get("case_id") or "",
                "route_status": "solved" if verifier.get("accepted") else route_status or "fake_closed_rejected",
                "stock_audit_passed": bool(verifier.get("accepted")),
                "fake_closure_rejected": bool(not verifier.get("accepted") and route_status == "fake_closed_rejected"),
                "reasons": sorted(set(reasons)),
                "rejected_terminal_list": list(verifier.get("rejected_terminal_list") or []),
                "failure_events": list(verifier.get("failure_events") or []),
            }
    audit = artifacts.get("route_audit")
    if isinstance(audit, dict):
        return dict(audit)
    if isinstance(chemenzy, dict) and isinstance(chemenzy.get("route_audit"), dict):
        return dict(chemenzy["route_audit"])
    return {}


def _has_literature_or_template_anchor(artifacts: dict[str, Any]) -> bool:
    smiles_first = artifacts.get("smiles_first")
    if isinstance(smiles_first, dict):
        validation = dict(smiles_first.get("validation") or {})
        if validation.get("route_status") in {"partial_anchor", "ready_for_guided_rerun"}:
            return True
        if smiles_first.get("artifacts") or smiles_first.get("template_artifacts") or smiles_first.get("literature_candidates"):
            return True
    open_research = artifacts.get("open_structure_research")
    if isinstance(open_research, dict):
        if open_research.get("artifacts") or open_research.get("structure_template_candidates"):
            return True
    return False


def _artifact_tree_contains_raw_reaction(value: Any, *, deterministic_context: bool = False) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            child_deterministic = deterministic_context or key_text in {"chemenzy", "route_audit"}
            if not deterministic_context and key_text in {
                "rxn",
                "rxn_smiles",
                "reaction_smiles",
                "raw_reaction",
                "raw_reactions",
                "raw_reaction_candidates",
                "reaction_candidates",
            }:
                return True
            if _artifact_tree_contains_raw_reaction(item, deterministic_context=child_deterministic):
                return True
    if isinstance(value, list):
        return any(_artifact_tree_contains_raw_reaction(item, deterministic_context=deterministic_context) for item in value)
    return False


def _run_result(
    run_dir: Path,
    target_input: dict[str, Any],
    preflight: dict[str, Any],
    workflow_plan: dict[str, Any],
    artifact_bundle: ArtifactBundle,
    final_verdict: FinalVerdict,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    assert final_verdict.verdict in FINAL_VERDICTS
    return {
        "schema_version": "codex_entry_controller_result.v1",
        "run_dir": str(run_dir),
        "target_input": target_input,
        "preflight": preflight,
        "workflow_plan": workflow_plan,
        "tool_calls": tool_calls,
        "artifact_bundle": artifact_bundle.to_dict(),
        "final_verdict": final_verdict.to_dict(),
        "artifacts": {
            "target_input": str(run_dir / "target_input.json"),
            "preflight": str(run_dir / "preflight.json"),
            "codex_workflow_plan": str(run_dir / "codex_workflow_plan.json"),
            "decision_trace": str(run_dir / "decision_trace.jsonl"),
            "tool_calls": str(run_dir / "tool_calls.jsonl"),
            "artifact_bundle": str(run_dir / "artifact_bundle.json"),
            "final_verdict": str(run_dir / "final_verdict.json"),
            "progress_panel": str(run_dir / "progress_panel.html"),
        },
    }

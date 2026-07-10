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
from cascade_planner.harness.parent_route_proof import (
    compile_stitched_parent_route_proof,
    is_solved_parent_route_proof,
)
from cascade_planner.harness.route_verifier import is_accepted_route_verifier_report
from cascade_planner.harness.schemas import (
    CANONICAL_RUN_SEMANTICS,
    FINAL_VERDICTS,
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
    literature_pdf_path: str | Path = "",
    literature_pdf_source_ref: str = "",
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
    target_data = target.to_dict()
    if str(literature_pdf_path or "").strip():
        target_data["literature_pdf_path"] = str(Path(literature_pdf_path).expanduser().resolve())
    if str(literature_pdf_source_ref or "").strip():
        target_data["literature_pdf_source_ref"] = str(literature_pdf_source_ref).strip()
    write_json(run_dir / "target_input.json", target_data)
    write_json(run_dir / "budget.json", budget.to_dict())
    append_jsonl(run_dir / "decision_trace.jsonl", {
        "stage": "start",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_name": target_name,
    })

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
    normalized_plan_data, controller_plan_audit = _normalize_controller_plan_for_execution(
        plan_data,
        target_data=target_data,
        preflight=preflight,
    )
    if controller_plan_audit.get("changed"):
        plan_data = normalized_plan_data
        plan_record = dict(plan_record)
        plan_record["workflow_plan"] = plan_data
        plan_record["controller_plan_normalization"] = controller_plan_audit
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
        _inject_controller_payload_defaults(tool_name=tool_name, payload=payload, target_data=target_data)
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


def _inject_controller_payload_defaults(*, tool_name: str, payload: dict[str, Any], target_data: dict[str, Any]) -> None:
    if tool_name == "extract_pdf_literature_structures":
        pdf_path = str(target_data.get("literature_pdf_path") or "").strip()
        if pdf_path and not payload.get("pdf_path"):
            payload["pdf_path"] = pdf_path
            payload.setdefault("render_zoom", 2.0)
    if tool_name == "extract_visual_literature_chain":
        if not payload.get("source_ref"):
            source_ref = str(target_data.get("literature_pdf_source_ref") or "").strip()
            if source_ref:
                payload["source_ref"] = source_ref


def _normalize_controller_plan_for_execution(
    plan_data: dict[str, Any],
    *,
    target_data: dict[str, Any],
    preflight: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep live planner output aligned with deterministic execution guardrails."""
    plan = dict(plan_data or {})
    tools = [dict(row) for row in plan.get("planned_tools") or [] if isinstance(row, dict)]
    audit = {
        "schema_version": "controller_plan_normalization.v1",
        "changed": False,
        "reason": "",
        "inserted_tools": [],
    }
    if not _literature_first_needs_native_baseline(plan, target_data=target_data, preflight=preflight):
        return plan, audit

    tool_names = [str(row.get("tool_name") or row.get("name") or "") for row in tools]
    if "run_chemenzy" in tool_names and "audit_route_and_extract_frontier" in tool_names:
        chemenzy_index = tool_names.index("run_chemenzy")
        audit_index = tool_names.index("audit_route_and_extract_frontier")
        first_research_index = min(
            [
                idx
                for idx, name in enumerate(tool_names)
                if name in {"run_smiles_first_literature_workflow", "run_open_structure_research_agent"}
            ]
            or [len(tool_names)]
        )
        if chemenzy_index < audit_index < first_research_index:
            return plan, audit

    deterministic = deterministic_workflow_plan(target_input=target_data, preflight=preflight).to_dict()
    default_tools = [dict(row) for row in deterministic.get("planned_tools") or [] if isinstance(row, dict)]
    chemenzy_row = next((dict(row) for row in tools if str(row.get("tool_name") or "") == "run_chemenzy"), None)
    if chemenzy_row is None:
        chemenzy_row = next((dict(row) for row in default_tools if str(row.get("tool_name") or "") == "run_chemenzy"), None)
    audit_row = next((dict(row) for row in tools if str(row.get("tool_name") or "") == "audit_route_and_extract_frontier"), None)
    if audit_row is None:
        audit_row = {"tool_name": "audit_route_and_extract_frontier", "payload": {}}
    baseline_rows = [row for row in [chemenzy_row, audit_row] if row]
    baseline_names = {str(row.get("tool_name") or row.get("name") or "") for row in baseline_rows}
    remaining = [
        row
        for row in tools
        if str(row.get("tool_name") or row.get("name") or "") not in baseline_names
    ]
    plan["planned_tools"] = [*baseline_rows, *remaining]
    plan["recommended_strategy"] = "hybrid"
    plan["controller_normalized_from_strategy"] = str(plan_data.get("recommended_strategy") or "")
    audit.update(
        {
            "changed": True,
            "reason": "non_glycoside_literature_first_requires_native_chemenzy_baseline",
            "inserted_tools": [name for name in ["run_chemenzy", "audit_route_and_extract_frontier"] if name not in tool_names],
            "original_strategy": str(plan_data.get("recommended_strategy") or ""),
            "normalized_strategy": "hybrid",
        }
    )
    return plan, audit


def _literature_first_needs_native_baseline(
    plan_data: dict[str, Any],
    *,
    target_data: dict[str, Any],
    preflight: dict[str, Any],
) -> bool:
    if str(plan_data.get("recommended_strategy") or "") != "literature_first":
        return False
    reason = str(plan_data.get("planner_decision_reason") or "")
    family = str(target_data.get("family_hint") or "").lower()
    flags = {str(item) for item in preflight.get("initial_risk_flags") or []}
    if "glycoside" in family or "glycoside_or_o_glycoside_like" in flags or reason == "glycoside_or_o_glycoside_like":
        return False
    if reason in {"user_requested_literature", "known_backend_unsuitable"}:
        return False
    return reason in {"natural_product_like", "macrocycle_or_steroid_like", "steroid_or_polycyclic_core"} or bool(flags)


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
    validation_reasons = [
        str(reason)
        for validation in bundle.validations
        for reason in validation.get("reasons") or []
    ]
    if _artifact_tree_contains_raw_reaction(artifacts) or "raw_reaction_injection" in validation_reasons:
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="fake_closed_rejected",
            reasons=sorted(set(validation_reasons + ["raw_reaction_injection"])),
            route_status="artifact_rejected",
            stock_audit_passed=False,
        )
    parent_proof = _parent_route_proof_artifact(artifacts)
    expected_target_smiles = str(bundle.target_input.get("target_smiles") or "")
    if not is_solved_parent_route_proof(
        parent_proof,
        expected_target_smiles=expected_target_smiles,
    ):
        verifier = _route_verifier_artifact(artifacts)
        if verifier:
            parent_proof = compile_stitched_parent_route_proof(
                target_smiles=expected_target_smiles,
                target_name=str(bundle.target_input.get("target_name") or ""),
                case_id=bundle.case_id,
                parent_verifier=verifier,
            )
    stitched = _stitched_route_audit(artifacts)
    if not is_solved_parent_route_proof(
        parent_proof,
        expected_target_smiles=expected_target_smiles,
    ) and stitched:
        parent_proof = compile_stitched_parent_route_proof(
            target_smiles=expected_target_smiles,
            target_name=str(bundle.target_input.get("target_name") or ""),
            case_id=bundle.case_id,
            stitched_route=stitched,
        )
    if is_solved_parent_route_proof(
        parent_proof,
        expected_target_smiles=expected_target_smiles,
    ):
        if bundle.run_semantics != CANONICAL_RUN_SEMANTICS:
            return FinalVerdict(
                case_id=bundle.case_id,
                verdict="needs_followup",
                reasons=["noncanonical_run_cannot_claim_solved"],
                route_status="needs_followup",
                stock_audit_passed=True,
                artifact_refs=dict(parent_proof.get("artifact_refs") or {}),
                run_semantics=bundle.run_semantics,
            )
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="solved",
            reasons=[],
            route_status="solved",
            solved=True,
            stock_audit_passed=True,
            artifact_refs=dict(parent_proof.get("artifact_refs") or {}),
            run_semantics=bundle.run_semantics,
        )
    audit = _latest_route_audit(
        artifacts,
        expected_target_smiles=expected_target_smiles,
    )
    route_status = str(audit.get("route_status") or "")
    reasons = [str(item) for item in audit.get("reasons") or []]
    deterministic_verifier_accepted = audit.get("_deterministic_route_verifier_accepted") is True
    if audit.get("fake_closure_rejected") or route_status == "fake_closed_rejected" or "fake_closure_evidence_present" in validation_reasons:
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="fake_closed_rejected",
            reasons=sorted(set(reasons + validation_reasons + ["fake_closure_evidence_present"])),
            route_status="fake_closed_rejected",
            stock_audit_passed=bool(audit.get("stock_audit_passed")),
        )
    if route_status == "solved" and audit.get("stock_audit_passed") and not deterministic_verifier_accepted:
        reasons.append("solved_requires_deterministic_parent_route_proof")
        route_status = "unresolved"
    if route_status == "solved" and deterministic_verifier_accepted and bundle.run_semantics != CANONICAL_RUN_SEMANTICS:
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="needs_followup",
            reasons=sorted(set(reasons + ["noncanonical_run_cannot_claim_solved"])),
            route_status="needs_followup",
            stock_audit_passed=bool(audit.get("stock_audit_passed")),
            run_semantics=bundle.run_semantics,
        )
    if route_status == "solved" and deterministic_verifier_accepted:
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
    tool_failed = any(_tool_call_is_execution_failure(row) for row in bundle.tool_calls)
    if tool_failed:
        tool_failure_reasons = [
            str(reason)
            for row in bundle.tool_calls
            if _tool_call_is_execution_failure(row)
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
        hypothesis_report = _hypothesis_route_report(artifacts)
        if hypothesis_report:
            return FinalVerdict(
                case_id=bundle.case_id,
                verdict="hypothesis_route_proposed",
                reasons=sorted(set(reasons + ["hypothesis_only_retrosynthesis_available"])),
                route_status=_hypothesis_route_status(artifacts) or "hypothesis_route_proposed",
                solved=False,
                stock_audit_passed=bool(audit.get("stock_audit_passed")),
                artifact_refs=dict(hypothesis_report.get("artifact_refs") or {}),
            )
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="partial_anchor_only_not_solved",
            reasons=sorted(set(reasons + ["literature_anchor_without_executable_stock_closure"])),
            route_status=route_status or "partial_anchor",
            stock_audit_passed=bool(audit.get("stock_audit_passed")),
        )
    hypothesis_report = _hypothesis_route_report(artifacts)
    if hypothesis_report:
        return FinalVerdict(
            case_id=bundle.case_id,
            verdict="hypothesis_route_proposed",
            reasons=sorted(set(reasons + validation_reasons + ["hypothesis_only_retrosynthesis_available"])),
            route_status=_hypothesis_route_status(artifacts) or "hypothesis_route_proposed",
            solved=False,
            stock_audit_passed=bool(audit.get("stock_audit_passed")),
            artifact_refs=dict(hypothesis_report.get("artifact_refs") or {}),
        )
    return FinalVerdict(
        case_id=bundle.case_id,
        verdict="unresolved",
        reasons=sorted(set(reasons + validation_reasons + ["no_deterministic_stock_closed_route"])),
        route_status=route_status or "unresolved",
        stock_audit_passed=bool(audit.get("stock_audit_passed")),
    )


def _tool_call_is_execution_failure(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "")
    if status == "error":
        return True
    if status != "rejected":
        return False
    tool_name = str(row.get("tool_name") or "")
    reasons = {str(reason) for reason in row.get("reasons") or []}
    hard_failure_markers = {
        "tool_exception",
        "forbidden_tool",
        "chem_enzy_nonzero_exit",
        "chem_enzy_timeout",
        "codex_executable_or_api_key_missing",
    }
    if reasons & hard_failure_markers:
        return True
    if any("tool_exception" in reason for reason in reasons):
        return True
    audit_rejection_tools = {
        "compile_source_detail_chain_route",
        "resolve_literature_structure_task",
        "validate_literature_intermediate_chain",
        "validate_artifact_bundle",
    }
    return tool_name not in audit_rejection_tools


def _hypothesis_route_report(artifacts: dict[str, Any]) -> dict[str, Any]:
    report = artifacts.get("hypothesis_only_retrosynthesis_report")
    if not isinstance(report, dict):
        return {}
    payload = dict(report.get("payload") or report)
    if not (
        payload.get("accepted")
        and int(payload.get("candidate_precursor_count") or 0) > 0
        and payload.get("no_solved_claim") is True
    ):
        return {}
    return dict(report)


def _hypothesis_route_status(artifacts: dict[str, Any]) -> str:
    execution = artifacts.get("hypothesis_execution_report")
    if isinstance(execution, dict):
        payload = dict(execution.get("payload") or execution)
        status = str(payload.get("route_status") or "")
        if status and status != "no_hypothesis_candidates":
            return status
    return ""


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
    if not isinstance(getattr(record, "output", None), dict):
        return False
    if tool_name == "run_open_structure_research_agent":
        result = dict(record.output)
        if not result.get("continuation_available"):
            return False
        compiled = state.artifacts.get("compiled_downstream")
        return isinstance(compiled, dict) and bool(compiled.get("accepted"))
    if tool_name == "extract_pdf_literature_structures":
        reasons = [str(item) for item in getattr(record, "reasons", []) or []]
        if "pdf_or_image_input_missing" in reasons:
            return _compiled_downstream_or_curator_records_available(state)
        return False
    if tool_name in {
        "extract_visual_literature_chain",
        "validate_literature_intermediate_chain",
        "build_source_detail_curator_records",
    }:
        return _compiled_downstream_or_curator_records_available(state)
    return False


def _compiled_downstream_or_curator_records_available(state: ToolExecutionState) -> bool:
    compiled = state.artifacts.get("compiled_downstream")
    if isinstance(compiled, dict) and bool(compiled.get("accepted")):
        return True
    refs = dict(compiled.get("artifact_refs") or {}) if isinstance(compiled, dict) else {}
    compiled_path = refs.get("compiled_downstream_consumables")
    if compiled_path and Path(str(compiled_path)).exists():
        return True
    open_dir = state.run_dir / "open_structure_research"
    return (open_dir / "evidence" / "source_detail_curator_records.json").exists()


def _latest_route_audit(
    artifacts: dict[str, Any],
    *,
    expected_target_smiles: str = "",
) -> dict[str, Any]:
    guided = artifacts.get("guided_chemenzy")
    if isinstance(guided, dict):
        verifier = dict(guided.get("raw_route_verifier") or {})
        if verifier:
            verifier_accepted = is_accepted_route_verifier_report(
                verifier,
                expected_target_smiles=expected_target_smiles,
            )
            route_status = str(verifier.get("route_status") or "")
            reasons = [str(item) for item in verifier.get("reasons") or []]
            if not verifier_accepted:
                reasons.append("route_verifier_rejected_raw_routes")
            return {
                "schema_version": "route_audit_report.v1",
                "case_id": guided.get("case_id") or verifier.get("case_id") or "",
                "route_status": "solved"
                if verifier_accepted
                else (route_status if route_status and route_status != "solved" else "fake_closed_rejected"),
                "stock_audit_passed": verifier_accepted,
                "fake_closure_rejected": bool(not verifier_accepted),
                "reasons": sorted(set(reasons)),
                "rejected_terminal_list": list(verifier.get("rejected_terminal_list") or []),
                "failure_events": list(verifier.get("failure_events") or []),
                "_deterministic_route_verifier_accepted": verifier_accepted,
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
            verifier_accepted = is_accepted_route_verifier_report(
                verifier,
                expected_target_smiles=expected_target_smiles,
            )
            route_status = str(verifier.get("route_status") or "")
            reasons = [str(item) for item in verifier.get("reasons") or []]
            if not verifier_accepted:
                reasons.append("route_verifier_rejected_raw_routes")
            return {
                "schema_version": "route_audit_report.v1",
                "case_id": verifier.get("case_id") or "",
                "route_status": "solved"
                if verifier_accepted
                else (route_status if route_status and route_status != "solved" else "fake_closed_rejected"),
                "stock_audit_passed": verifier_accepted,
                "fake_closure_rejected": bool(not verifier_accepted),
                "reasons": sorted(set(reasons)),
                "rejected_terminal_list": list(verifier.get("rejected_terminal_list") or []),
                "failure_events": list(verifier.get("failure_events") or []),
                "_deterministic_route_verifier_accepted": verifier_accepted,
            }
    audit = artifacts.get("route_audit")
    if isinstance(audit, dict):
        return {**dict(audit), "_deterministic_route_verifier_accepted": False}
    if isinstance(chemenzy, dict) and isinstance(chemenzy.get("route_audit"), dict):
        return {**dict(chemenzy["route_audit"]), "_deterministic_route_verifier_accepted": False}
    return {}


def _stitched_route_audit(artifacts: dict[str, Any]) -> dict[str, Any]:
    stitched = artifacts.get("stitched_semisynthesis_route")
    if isinstance(stitched, dict):
        return dict(stitched.get("result") or stitched)
    return {}


def _parent_route_proof_artifact(artifacts: dict[str, Any]) -> dict[str, Any]:
    for key in ("parent_route_proof", "stitched_parent_route_proof"):
        value = artifacts.get(key)
        if isinstance(value, dict):
            nested = value.get("result")
            return dict(nested) if isinstance(nested, dict) else dict(value)
    return {}


def _route_verifier_artifact(artifacts: dict[str, Any]) -> dict[str, Any]:
    guided = artifacts.get("guided_chemenzy")
    if isinstance(guided, dict) and isinstance(guided.get("raw_route_verifier"), dict):
        return dict(guided["raw_route_verifier"])
    verifier = artifacts.get("route_verifier")
    if isinstance(verifier, dict):
        return dict(verifier)
    chemenzy = artifacts.get("chemenzy")
    if isinstance(chemenzy, dict):
        if isinstance(chemenzy.get("raw_route_verifier"), dict):
            return dict(chemenzy["raw_route_verifier"])
        result = chemenzy.get("result")
        if isinstance(result, dict) and isinstance(result.get("raw_route_verifier"), dict):
            return dict(result["raw_route_verifier"])
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
            child_deterministic = deterministic_context or key_text in {
                "chemenzy",
                "guided_chemenzy",
                "route_audit",
                "route_verifier",
                "parent_route_proof",
                "stitched_semisynthesis_route",
                "source_detail_chain_route",
            }
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

"""Repo-controlled local tool wrappers for the Codex-entry harness."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from cascade_planner.agent.route_auditor import audit_route_package, validate_route_audit_report
from cascade_planner.agent.smiles_first import SmilesFirstWorkflowConfig, run_smiles_first_workflow
from cascade_planner.harness.downstream_compiler import (
    compile_downstream_consumables,
    write_compiled_downstream_artifacts,
)
from cascade_planner.harness.analogical_retrosynthesis import build_analogical_retrosynthesis_hypotheses
from cascade_planner.harness.open_research_contract import (
    REQUIRED_OPEN_RESEARCH_ARTIFACTS,
    REQUIRED_OPEN_RESEARCH_JSON_ARTIFACTS,
    validate_open_research_json_payload,
)
from cascade_planner.harness.open_research_experience import (
    audit_open_research_boundary,
    extract_open_research_experience,
)
from cascade_planner.harness.open_research_retrieval import validate_retrieval_prefetch_consumption
from cascade_planner.harness.literature_pdf_extraction import extract_literature_pdf_assets
from cascade_planner.harness.visual_structure_extraction import validate_visual_structure_chain
from cascade_planner.harness.visual_literature_chain_agent import run_visual_literature_chain_agent
from cascade_planner.harness.source_detail_chain_builder import (
    build_source_detail_curator_records_from_chain,
    compile_hybrid_route_set as compile_hybrid_route_set_artifact,
    compile_source_detail_chain_route as compile_source_detail_chain_route_artifact,
    probe_literature_plugin_chain,
    resolve_curator_records_to_source_detail_steps,
)
from cascade_planner.harness.stitched_route import compile_stitched_semisynthesis_route
from cascade_planner.harness.source_detail_resolution import (
    resolve_source_detail_extraction_pack,
    source_detail_resolution_manifest_entry,
    source_detail_curator_records_path,
)
from cascade_planner.harness.route_failure_feedback import (
    compile_route_failure_feedback,
    write_route_failure_feedback,
)
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from cascade_planner.harness.schemas import (
    ArtifactBundle,
    CANONICAL_RUN_SEMANTICS,
    ToolCallRecord,
    append_jsonl,
    validate_tool_payload,
    write_json,
)
from cascade_planner.harness.self_evo_memory import compile_self_evo_memory, write_self_evo_memory
from cascade_planner.harness.self_evo_replay import run_self_evo_replay_gate as run_self_evo_replay_gate_report


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHEMENZY_ENV_PREFIX = Path("/root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env")


@dataclass
class HarnessBudget:
    max_chem_enzy_runs: int = 1
    max_guided_chemenzy_runs: int = 1
    max_route_expansion_subgoal_runs: int = 2
    max_codex_research_runs: int = 1
    max_scout_calls: int = 3
    max_visual_calls: int = 3
    max_template_application_actions: int = 3
    max_template_applications_per_round: int = 5
    timeout_s: float = 1800.0
    chem_enzy_timeout_s: float = 1200.0
    guided_chemenzy_timeout_s: float = 1800.0
    smiles_first_timeout_s: float = 600.0
    open_research_timeout_s: float = 900.0
    schema_version: str = "codex_entry_budget.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_chem_enzy_runs": self.max_chem_enzy_runs,
            "max_guided_chemenzy_runs": self.max_guided_chemenzy_runs,
            "max_route_expansion_subgoal_runs": self.max_route_expansion_subgoal_runs,
            "max_codex_research_runs": self.max_codex_research_runs,
            "max_scout_calls": self.max_scout_calls,
            "max_visual_calls": self.max_visual_calls,
            "max_template_application_actions": self.max_template_application_actions,
            "max_template_applications_per_round": self.max_template_applications_per_round,
            "timeout_s": self.timeout_s,
            "chem_enzy_timeout_s": self.chem_enzy_timeout_s,
            "guided_chemenzy_timeout_s": self.guided_chemenzy_timeout_s,
            "smiles_first_timeout_s": self.smiles_first_timeout_s,
            "open_research_timeout_s": self.open_research_timeout_s,
        }


@dataclass
class ToolExecutionState:
    run_dir: Path
    target_input: dict[str, Any]
    preflight: dict[str, Any]
    budget: HarnessBudget = field(default_factory=HarnessBudget)
    key_path: str | Path = ROOT / "key.txt"
    base_url: str = "https://api.wellau.com/v1"
    model: str = "gpt-5.5"
    run_semantics: str = CANONICAL_RUN_SEMANTICS
    mock_tool_results: dict[str, Any] = field(default_factory=dict)
    chem_enzy_runs: int = 0
    guided_chemenzy_runs: int = 0
    route_expansion_subgoal_runs: int = 0
    codex_research_runs: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)
    validations: list[dict[str, Any]] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)


ToolHandler = Callable[[ToolExecutionState, dict[str, Any]], dict[str, Any]]


def _online_anchor_resolution_enabled(target_input: dict[str, Any]) -> bool:
    value = target_input.get("enable_online_anchor_resolution")
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "off", "no"}
    return bool(value)


def execute_local_tool(tool_name: str, payload: dict[str, Any], state: ToolExecutionState) -> ToolCallRecord:
    started = time.monotonic()
    validation = validate_tool_payload(tool_name, payload)
    if not validation["accepted"]:
        record = ToolCallRecord(
            tool_name=tool_name,
            status="rejected",
            input_payload=payload,
            output={"validation": validation},
            reasons=list(validation["reasons"]),
            elapsed_s=round(time.monotonic() - started, 3),
        )
        _write_tool_record(state, record)
        return record

    handlers: dict[str, ToolHandler] = {
        "run_chemenzy": run_chemenzy,
        "audit_route_and_extract_frontier": audit_route_and_extract_frontier,
        "run_smiles_first_literature_workflow": run_smiles_first_literature_workflow_tool,
        "run_open_structure_research_agent": run_open_structure_research_agent,
        "extract_pdf_literature_structures": extract_pdf_literature_structures_tool,
        "extract_visual_literature_chain": extract_visual_literature_chain_tool,
        "resolve_literature_structure_task": resolve_literature_structure_task_tool,
        "apply_source_text_condition_repairs": apply_source_text_condition_repairs_tool,
        "validate_literature_intermediate_chain": validate_literature_intermediate_chain_tool,
        "build_source_detail_curator_records": build_source_detail_curator_records_tool,
        "build_analogical_retrosynthesis_hypotheses": build_analogical_retrosynthesis_hypotheses_tool,
        "compile_source_detail_chain_route": compile_source_detail_chain_route_tool,
        "stitch_literature_chain_with_subgoal_route": stitch_literature_chain_with_subgoal_route_tool,
        "compile_hybrid_route_set": compile_hybrid_route_set_tool,
        "run_guided_chemenzy_rerun": run_guided_chemenzy_rerun,
        "run_route_expansion_subgoal_search": run_route_expansion_subgoal_search,
        "run_self_evo_replay_gate": run_self_evo_replay_gate_tool,
        "validate_artifact_bundle": validate_artifact_bundle_tool,
        "emit_final_verdict": lambda st, pl: {"status": "deferred_to_runner", "accepted": True},
    }
    handler = handlers.get(tool_name)
    if handler is None:
        output = {"accepted": False, "reasons": ["forbidden_tool"]}
        status = "rejected"
    else:
        try:
            output = handler(state, payload)
            status = "accepted" if output.get("accepted", True) else "rejected"
        except Exception as exc:
            output = {
                "accepted": False,
                "reasons": ["tool_exception"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            status = "error"
    record = ToolCallRecord(
        tool_name=tool_name,
        status=status,
        input_payload=payload,
        output=output,
        reasons=[str(item) for item in output.get("reasons") or []],
        elapsed_s=round(time.monotonic() - started, 3),
    )
    _write_tool_record(state, record)
    return record


def run_chemenzy(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "run_chemenzy", payload)
    if mock is not None:
        result = dict(mock)
        _attach_native_route_verifier(state, result)
        state.artifacts["chemenzy"] = result
        write_json(state.run_dir / "chemenzy_native_raw_result.json", result)
        return {"accepted": True, "mode": "mock", "result": result}

    if state.chem_enzy_runs >= state.budget.max_chem_enzy_runs:
        return {"accepted": False, "reasons": ["chem_enzy_budget_exhausted"]}
    state.chem_enzy_runs += 1

    request = _chemenzy_request_from_payload(state, payload)
    result = _execute_chemenzy_request(
        state=state,
        request=request,
        request_path=state.run_dir / "chemenzy_request.json",
        output_path=state.run_dir / "chemenzy_native_raw_result.json",
        timeout_s=float(payload.get("timeout_s") or state.budget.chem_enzy_timeout_s),
    )
    _attach_native_route_verifier(state, result)
    state.artifacts["chemenzy"] = result
    write_json(state.run_dir / "chemenzy_native_raw_result.json", result)
    return {"accepted": bool(result.get("ok") or result.get("accepted", result.get("exit_code") == 0)), "result": result}


def _attach_native_route_verifier(state: ToolExecutionState, result: dict[str, Any]) -> dict[str, Any]:
    """Attach the deterministic raw-route verifier for any native ChemEnzy routes."""
    routes = result.get("routes") or (dict(result.get("result") or {}).get("routes") if isinstance(result.get("result"), dict) else [])
    if not routes:
        return {}
    verifier = verify_chemenzy_raw_routes(
        result,
        target_smiles=str(state.target_input.get("target_smiles") or result.get("target") or ""),
        case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
    )
    if verifier:
        result["raw_route_verifier"] = verifier
        state.artifacts["route_verifier"] = verifier
        write_json(state.run_dir / "route_verifier_report.json", verifier)
        if not verifier.get("accepted"):
            feedback = compile_route_failure_feedback(
                verifier,
                case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
                target_name=str(state.target_input.get("target_name") or ""),
            )
            write_route_failure_feedback(feedback, output_dir=state.run_dir)
            state.artifacts["route_failure_feedback"] = feedback
    return verifier


def run_guided_chemenzy_rerun(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "run_guided_chemenzy_rerun", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["guided_chemenzy"] = result
        write_json(state.run_dir / "guided_chemenzy_result.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    if _native_route_verified_solved(state):
        result = _skip_result(
            schema_version="guided_chemenzy_rerun_result.v1",
            reasons=["native_route_verified_solved"],
            route_status="solved",
            solved=True,
        )
        state.artifacts["guided_chemenzy"] = result
        write_json(state.run_dir / "guided_chemenzy_result.json", result)
        return {"accepted": True, "result": result, "reasons": result["reasons"]}

    if state.guided_chemenzy_runs >= state.budget.max_guided_chemenzy_runs:
        return {"accepted": False, "reasons": ["guided_chemenzy_budget_exhausted"]}
    policy = _guided_policy_from_payload_or_artifacts(state, payload)
    if not policy:
        plugin_flags = _literature_template_plugin_flags_from_artifacts(state)
        if plugin_flags:
            policy = _plugin_only_guided_policy(state, plugin_flags=plugin_flags, payload=payload)
    if not policy:
        result = _skip_result(
            schema_version="guided_chemenzy_rerun_result.v1",
            reasons=["guided_policy_missing"],
            route_status="unresolved",
            solved=False,
        )
        state.artifacts["guided_chemenzy"] = result
        write_json(state.run_dir / "guided_chemenzy_result.json", result)
        return {"accepted": True, "result": result, "reasons": result["reasons"]}
    policy = _merge_route_failure_feedback_policy(state, policy, payload)
    policy = _merge_analogical_retrosynthesis_policy(state, policy, payload)
    state.guided_chemenzy_runs += 1

    budget = dict(policy.get("budget") or {})
    request_payload = {
        "search_preset": payload.get("search_preset", "thorough"),
        "max_steps": int(payload.get("max_steps") or budget.get("max_depth") or 15),
        "chem_enzy_iterations": int(payload.get("chem_enzy_iterations") or budget.get("max_iterations") or 50),
        "chem_enzy_expansion_topk": int(payload.get("chem_enzy_expansion_topk") or budget.get("expansion_topk") or 100),
        "stock_mode": payload.get("stock_mode", "building-block"),
        "device": payload.get("device", "cpu"),
        "chem_enzy_search_policy": policy,
    }
    plugin_flags = _literature_template_plugin_flags_from_artifacts(state)
    plugin_flags = _merge_self_evo_memory_plugin_flags(state, plugin_flags, payload)
    if plugin_flags:
        request_payload["literature_template_plugin"] = plugin_flags
    request = _chemenzy_request_from_payload(state, request_payload)
    result = _execute_chemenzy_request(
        state=state,
        request=request,
        request_path=state.run_dir / "guided_chemenzy_request.json",
        output_path=state.run_dir / "guided_chemenzy_raw_result.json",
        timeout_s=float(payload.get("timeout_s") or state.budget.guided_chemenzy_timeout_s),
    )
    verifier = verify_chemenzy_raw_routes(
        result,
        target_smiles=str(state.target_input.get("target_smiles") or result.get("target") or ""),
        case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
    ) if (result.get("routes") or (result.get("result") or {}).get("routes")) else {}
    plugin_runtime = _literature_template_plugin_runtime_diagnostics(result, request)
    proof_blockers = _guided_route_proof_blockers(verifier, plugin_runtime)
    verifier_for_output = _guided_hardened_verifier(verifier, proof_blockers=proof_blockers)
    verifier_accepted = bool(verifier_for_output.get("accepted"))
    out = {
        "schema_version": "guided_chemenzy_rerun_result.v1",
        "accepted": bool(result.get("ok") or result.get("accepted", result.get("exit_code") == 0)) and (verifier_accepted if verifier else True),
        "policy": policy,
        "request": request,
        "result": result,
        "raw_route_verifier": verifier_for_output,
        "literature_template_plugin_runtime": plugin_runtime,
        "route_status": str(
            verifier_for_output.get("route_status")
            or ("solved" if (result.get("search_status") or {}).get("solved") and verifier_accepted else "unresolved")
        ),
        "solved": verifier_accepted,
    }
    if proof_blockers:
        out["backend_raw_route_verifier"] = verifier
        out["route_proof_blockers"] = proof_blockers
    if verifier and not verifier_for_output.get("accepted"):
        out["accepted"] = False
        out["reasons"] = sorted(
            set(
                [str(item) for item in verifier_for_output.get("reasons") or ["route_verifier_rejected_raw_routes"]]
                + [str(item) for item in plugin_runtime.get("reasons") or []]
            )
        )
        feedback = compile_route_failure_feedback(
            verifier_for_output,
            case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
            target_name=str(state.target_input.get("target_name") or ""),
        )
        feedback_path = write_route_failure_feedback(feedback, output_dir=state.run_dir)
        out["route_failure_feedback"] = {
            "accepted": bool(feedback.get("accepted")),
            "path": feedback_path,
            "terminal_blacklist_count": len(feedback.get("terminal_blacklist") or []),
            "frontier_research_target_count": len(feedback.get("frontier_research_targets") or []),
            "query_hint_count": len(feedback.get("query_hints") or []),
        }
        state.artifacts["route_failure_feedback"] = feedback
    state.artifacts["guided_chemenzy"] = out
    write_json(state.run_dir / "guided_chemenzy_result.json", out)
    if verifier:
        write_json(state.run_dir / "guided_route_verifier_report.json", verifier_for_output)
    # A verifier rejection is chemistry feedback, not a harness execution
    # failure; runtime/transport failures without verifier evidence still
    # reject the tool call.
    tool_accepted = bool(out.get("accepted")) or bool(verifier)
    return {"accepted": tool_accepted, "result": out, "reasons": [str(item) for item in out.get("reasons") or []]}


def _guided_route_proof_blockers(verifier: dict[str, Any], plugin_runtime: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not verifier:
        return blockers
    reason_text = json.dumps(
        {
            "reasons": verifier.get("reasons") or [],
            "failure_events": verifier.get("failure_events") or [],
            "rejected_terminal_list": verifier.get("rejected_terminal_list") or [],
        },
        sort_keys=True,
        default=str,
    )
    if "large_atom_jump" in reason_text:
        blockers.extend(["large_atom_jump", "guided_route_verifier_rejected_large_atom_jump"])
    runtime_reasons = {str(item) for item in plugin_runtime.get("reasons") or [] if str(item or "").strip()}
    if "literature_template_plugin_not_invoked" in runtime_reasons:
        blockers.append("literature_template_plugin_not_invoked")
    return sorted(set(blockers))


def _guided_hardened_verifier(verifier: dict[str, Any], *, proof_blockers: list[str]) -> dict[str, Any]:
    if not verifier or not proof_blockers:
        return dict(verifier or {})
    out = dict(verifier)
    reasons = [str(item) for item in out.get("reasons") or [] if str(item or "").strip()]
    reasons.extend(str(item) for item in proof_blockers if str(item or "").strip())
    out["accepted"] = False
    if "large_atom_jump" in proof_blockers:
        out["route_status"] = "fake_closed_rejected"
    elif "literature_template_plugin_not_invoked" in proof_blockers:
        out["route_status"] = "partial_anchor_only_not_solved"
    else:
        out["route_status"] = str(out.get("route_status") or "unresolved")
    out["reasons"] = sorted(set(reasons or ["route_proof_blocked"]))
    out["route_proof_blocked"] = True
    return out


def run_self_evo_replay_gate_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "run_self_evo_replay_gate", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["self_evo_replay"] = result
        write_json(state.run_dir / "self_evo_replay_report.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    if _native_route_verified_solved(state):
        result = _skip_result(
            schema_version="harness_self_evo_replay_report.v1",
            reasons=["native_route_verified_solved"],
            production_write_blocked=True,
            production_promoted_count=0,
        )
        state.artifacts["self_evo_replay"] = result
        write_json(state.run_dir / "self_evo_replay_report.json", result)
        return {"accepted": True, "result": result, "reasons": result["reasons"]}

    staging = _self_evo_staging_from_artifacts(state)
    if not staging:
        result = {
            "schema_version": "harness_self_evo_replay_report.v1",
            "accepted": True,
            "status": "skipped",
            "skipped": True,
            "reasons": ["self_evo_staging_missing"],
            "production_write_blocked": True,
            "production_promoted_count": 0,
        }
        state.artifacts["self_evo_replay"] = result
        write_json(state.run_dir / "self_evo_replay_report.json", result)
        return {"accepted": True, "result": result, "reasons": result["reasons"]}

    result = run_self_evo_replay_gate_report(
        staging,
        replay_metrics=dict(payload.get("replay_metrics") or {}),
        target_run=bool(payload.get("target_run", True)),
        allow_production=bool(payload.get("allow_production", False)),
    )
    memory = compile_self_evo_memory(
        result,
        compiled_downstream=_compiled_downstream_from_state(state),
        case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
    )
    memory_path = write_self_evo_memory(memory, output_dir=state.run_dir)
    result["self_evo_memory"] = {
        "accepted": bool(memory.get("accepted")),
        "path": memory_path,
        "reusable_template_card_count": len(memory.get("reusable_template_cards") or []),
        "reusable_one_step_row_count": len(memory.get("reusable_one_step_rows") or []),
        "reusable_route_expansion_task_count": len(memory.get("reusable_route_expansion_tasks") or []),
    }
    state.artifacts["self_evo_memory"] = memory
    state.artifacts["self_evo_replay"] = result
    write_json(state.run_dir / "self_evo_replay_report.json", result)
    return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}


def run_route_expansion_subgoal_search(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "run_route_expansion_subgoal_search", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["route_expansion_subgoal_search"] = result
        write_json(state.run_dir / "route_expansion_subgoal_search_result.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    if _native_route_verified_solved(state):
        result = _skip_result(
            schema_version="route_expansion_subgoal_search_result.v1",
            reasons=["native_route_verified_solved"],
            solved=True,
            subgoal_count=0,
        )
        state.artifacts["route_expansion_subgoal_search"] = result
        write_json(state.run_dir / "route_expansion_subgoal_search_result.json", result)
        return {"accepted": True, "result": result, "reasons": result["reasons"]}

    compiled = _compiled_downstream_from_state(state)
    targets = _route_expansion_child_targets(state=state, payload=payload, compiled=compiled)
    if not targets:
        result = _skip_result(
            schema_version="route_expansion_subgoal_search_result.v1",
            reasons=["route_expansion_child_targets_missing"],
            solved=False,
            subgoal_count=0,
        )
        state.artifacts["route_expansion_subgoal_search"] = result
        write_json(state.run_dir / "route_expansion_subgoal_search_result.json", result)
        return {"accepted": True, "result": result, "reasons": result["reasons"]}

    prior_result = dict(state.artifacts.get("route_expansion_subgoal_search") or {})
    prior_rows = [
        dict(item)
        for item in prior_result.get("subgoals") or []
        if isinstance(item, dict)
    ]
    remaining_budget = max(0, int(state.budget.max_route_expansion_subgoal_runs) - int(state.route_expansion_subgoal_runs))
    if remaining_budget <= 0:
        return {"accepted": False, "reasons": ["route_expansion_subgoal_budget_exhausted"]}
    max_targets = min(max(1, int(payload.get("max_targets") or 2)), remaining_budget)
    if payload.get("target_offset") is not None:
        try:
            target_offset = max(0, int(payload.get("target_offset") or 0))
        except (TypeError, ValueError):
            target_offset = len(prior_rows)
    else:
        target_offset = len(prior_rows)
    selected_targets = targets[target_offset:target_offset + max_targets]
    explicit_target_payload = bool(payload.get("child_targets") or payload.get("subgoal_targets"))
    result_offset = len(prior_rows) if explicit_target_payload else target_offset
    if not selected_targets:
        accepted_prior = [row for row in prior_rows if row.get("accepted") or row.get("solved")]
        result = {
            "schema_version": "route_expansion_subgoal_search_result.v1",
            "accepted": bool(accepted_prior),
            "status": "solved" if accepted_prior else "exhausted",
            "solved": bool(accepted_prior),
            "subgoal_count": len(prior_rows),
            "accepted_subgoal_count": len(accepted_prior),
            "rejected_subgoal_count": len(prior_rows) - len(accepted_prior),
            "subgoals": prior_rows,
            "reasons": [] if accepted_prior else ["route_expansion_child_targets_exhausted"],
        }
        state.artifacts["route_expansion_subgoal_search"] = result
        write_json(state.run_dir / "route_expansion_subgoal_search_result.json", result)
        return {"accepted": True, "result": result, "reasons": list(result.get("reasons") or [])}
    rows: list[dict[str, Any]] = []
    for local_idx, target in enumerate(selected_targets):
        absolute_idx = result_offset + local_idx
        sub_dir = state.run_dir / "route_expansion_subgoals"
        sub_dir.mkdir(parents=True, exist_ok=True)
        request_payload = {
            "target_smiles": target["smiles"],
            "target_name": target["name"],
            "family_hint": str(state.target_input.get("family_hint") or ""),
            "search_preset": payload.get("search_preset", "thorough"),
            "max_steps": int(payload.get("max_steps") or target.get("max_depth") or 20),
            "chem_enzy_iterations": int(payload.get("chem_enzy_iterations") or target.get("max_iterations") or 50),
            "chem_enzy_expansion_topk": int(payload.get("chem_enzy_expansion_topk") or target.get("expansion_topk") or 100),
            "stock_mode": payload.get("stock_mode", "building-block"),
            "device": payload.get("device", "cpu"),
            "chem_enzy_search_policy": target.get("policy") or {},
        }
        if _exact_target_audit_required(target):
            request_payload.update(
                {
                    "exact_target_override": True,
                    "target_equivalence_audit_required": True,
                    "requested_exact_target_smiles": target["smiles"],
                    "requested_exact_target_name": target["name"],
                }
            )
        plugin_flags = _literature_template_plugin_flags_from_artifacts(state)
        plugin_flags = _merge_self_evo_memory_plugin_flags(state, plugin_flags, payload)
        if plugin_flags:
            request_payload["literature_template_plugin"] = plugin_flags
        request = _chemenzy_request_from_payload(state, request_payload)
        safe_name = _safe_file_stem(target["name"] or f"subgoal_{absolute_idx + 1}")
        raw_path = sub_dir / f"{absolute_idx + 1:02d}_{safe_name}_raw_result.json"
        req_path = sub_dir / f"{absolute_idx + 1:02d}_{safe_name}_request.json"
        state.route_expansion_subgoal_runs += 1
        raw = _execute_chemenzy_request(
            state=state,
            request=request,
            request_path=req_path,
            output_path=raw_path,
            timeout_s=float(payload.get("timeout_s") or state.budget.guided_chemenzy_timeout_s),
        )
        verifier = verify_chemenzy_raw_routes(
            raw,
            target_smiles=target["smiles"],
            case_id=f"{state.preflight.get('case_id') or state.target_input.get('target_name') or 'case'}:{safe_name}",
        ) if (raw.get("routes") or (raw.get("result") or {}).get("routes")) else {}
        row = {
            "schema_version": "route_expansion_subgoal_result.v1",
            "accepted": bool(verifier.get("accepted")),
            "subgoal_index": absolute_idx,
            "subgoal": target,
            "request_path": str(req_path),
            "raw_result_path": str(raw_path),
            "raw_ok": bool(raw.get("ok") or raw.get("accepted", raw.get("exit_code") == 0)),
            "raw_solved": bool((raw.get("search_status") or {}).get("solved")),
            "route_count": len(raw.get("routes") or []),
            "verifier": verifier,
            "route_status": str(verifier.get("route_status") or ("solved" if (raw.get("search_status") or {}).get("solved") else "unresolved")),
            "solved": bool(verifier.get("accepted")),
        }
        rows.append(row)
        write_json(sub_dir / f"{absolute_idx + 1:02d}_{safe_name}_verifier.json", verifier)

    all_rows = _dedupe_subgoal_results([*prior_rows, *rows])
    accepted_rows = [row for row in all_rows if row.get("accepted") or row.get("solved")]
    result = {
        "schema_version": "route_expansion_subgoal_search_result.v1",
        "accepted": bool(accepted_rows),
        "status": "solved" if accepted_rows else "failed",
        "solved": bool(accepted_rows),
        "subgoal_count": len(all_rows),
        "accepted_subgoal_count": len(accepted_rows),
        "rejected_subgoal_count": len(all_rows) - len(accepted_rows),
        "subgoals": all_rows,
        "reasons": [] if accepted_rows else ["no_route_expansion_subgoal_verified_solved"],
    }
    state.artifacts["route_expansion_subgoal_search"] = result
    write_json(state.run_dir / "route_expansion_subgoal_search_result.json", result)
    return {"accepted": True, "result": result, "reasons": list(result.get("reasons") or [])}


def _chemenzy_request_from_payload(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    payload = _chemenzy_payload_with_harness_defaults(state, payload)
    defaults = _chemenzy_default_search_defaults(state)
    request = {
        "target_smiles": payload.get("target_smiles") or state.target_input["target_smiles"],
        "target_name": payload.get("target_name") or state.target_input.get("target_name", ""),
        "family_hint": payload.get("family_hint") or state.target_input.get("family_hint", ""),
        "planner_backend": "chem_enzy_native",
        "search_preset": payload.get("search_preset", defaults["search_preset"]),
        "max_steps": int(payload.get("max_steps") or defaults["max_steps"]),
        "chem_enzy_iterations": int(payload.get("chem_enzy_iterations") or defaults["chem_enzy_iterations"]),
        "chem_enzy_expansion_topk": int(payload.get("chem_enzy_expansion_topk") or defaults["chem_enzy_expansion_topk"]),
        "stock_mode": payload.get("stock_mode", "building-block"),
        "device": payload.get("device", "cpu"),
        "enable_condition_prediction": bool(payload.get("enable_condition_prediction", False)),
        "enable_enzyme_assignment": bool(payload.get("enable_enzyme_assignment", False)),
        "enable_easifa": bool(payload.get("enable_easifa", False)),
    }
    if payload.get("stock_names"):
        request["stock_names"] = [str(item) for item in payload.get("stock_names") or [] if str(item or "").strip()]
    if payload.get("harness_search_boundary"):
        request["harness_search_boundary"] = dict(payload.get("harness_search_boundary") or {})
    for key in (
        "chem_enzy_search_policy",
        "search_policy",
        "literature_template_plugin",
        "autoplanner_literature_template_plugin",
        "exact_target_override",
        "target_equivalence_audit_required",
        "requested_exact_target_smiles",
        "requested_exact_target_name",
    ):
        if key in payload:
            request[key] = payload[key]
    return request


def _chemenzy_payload_with_harness_defaults(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload or {})
    defaults = _chemenzy_default_search_defaults(state)
    out.setdefault("search_preset", defaults["search_preset"])
    out.setdefault("max_steps", defaults["max_steps"])
    out.setdefault("chem_enzy_iterations", defaults["chem_enzy_iterations"])
    out.setdefault("chem_enzy_expansion_topk", defaults["chem_enzy_expansion_topk"])

    requested_stock_mode = str(out.get("stock_mode") or "").strip()
    requested_stock_names = [str(item) for item in out.get("stock_names") or [] if str(item or "").strip()]
    actions: list[str] = []
    if not bool(out.get("allow_broad_stock")):
        if requested_stock_names and requested_stock_names != ["PaRotes_n1-stock"]:
            out["stock_names"] = ["PaRotes_n1-stock"]
            actions.append("coerced_explicit_stock_names_to_small_stock")
        if not requested_stock_mode or requested_stock_mode.lower() not in {"building-block", "building_block", "strict", "paroutes-n1", "n1"}:
            if requested_stock_mode:
                actions.append("coerced_broad_or_unknown_stock_mode_to_building_block")
            out["stock_mode"] = "building-block"
    else:
        out.setdefault("stock_mode", "building-block")

    boundary = dict(out.get("harness_search_boundary") or {})
    boundary.update(
        {
            "schema_version": "chemenzy_harness_search_boundary.v1",
            "small_stock_default": not bool(out.get("allow_broad_stock")),
            "broad_stock_requires_explicit_allow": True,
            "requested_stock_mode": requested_stock_mode or "unset",
            "requested_stock_names": requested_stock_names,
            "effective_stock_mode": str(out.get("stock_mode") or ""),
            "effective_stock_names": [str(item) for item in out.get("stock_names") or []]
            or (["PaRotes_n1-stock"] if str(out.get("stock_mode") or "").lower() in {"building-block", "building_block", "strict", "paroutes-n1", "n1"} else []),
            "target_heavy_atoms": _target_heavy_atoms_from_state(state),
            "complex_target_deep_defaults": _target_heavy_atoms_from_state(state) >= 25,
            "stock_policy_actions": actions,
        }
    )
    out["harness_search_boundary"] = boundary
    return out


def _chemenzy_default_search_defaults(state: ToolExecutionState) -> dict[str, Any]:
    if _target_heavy_atoms_from_state(state) >= 25:
        return {
            "search_preset": "thorough",
            "max_steps": 20,
            "chem_enzy_iterations": 50,
            "chem_enzy_expansion_topk": 100,
        }
    return {
        "search_preset": "quick",
        "max_steps": 6,
        "chem_enzy_iterations": 10,
        "chem_enzy_expansion_topk": 50,
    }


def _target_heavy_atoms_from_state(state: ToolExecutionState) -> int:
    profile = dict(state.preflight.get("target_profile") or {})
    if profile.get("heavy_atoms"):
        return int(profile.get("heavy_atoms") or 0)
    try:
        from rdkit import Chem
    except Exception:
        return 0
    mol = Chem.MolFromSmiles(str(state.target_input.get("target_smiles") or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def _target_search_name(state: ToolExecutionState) -> str:
    text = " ".join(
        str(value or "")
        for value in (
            state.target_input.get("target_name"),
            state.preflight.get("case_id"),
            state.target_input.get("family_hint"),
        )
    )
    for token in text.replace("_", " ").replace("-", " ").split():
        lower = token.strip().lower()
        if len(lower) > 6 and lower.endswith("statin"):
            return lower
    name = str(state.target_input.get("target_name") or state.preflight.get("case_id") or "").strip()
    if "_" in name:
        prefix = name.split("_", 1)[0].strip()
        if prefix.isalpha() and len(prefix) > 2 and prefix.lower() not in {"target", "case"}:
            return prefix
    if _usable_target_search_name(name):
        return name
    return _target_search_fallback(state)


def _usable_target_search_name(name: str) -> bool:
    clean = str(name or "").strip()
    if not clean or clean.lower() in {"target", "case"}:
        return False
    if "_" not in clean:
        return True
    alpha_tokens = [token for token in clean.replace("-", "_").split("_") if token.isalpha()]
    return any(len(token) > 2 for token in alpha_tokens)


def _target_search_fallback(state: ToolExecutionState) -> str:
    family = str(state.target_input.get("family_hint") or "").strip()
    profile = dict(state.preflight.get("target_profile") or {})
    formula = str(profile.get("formula") or "").strip()
    flags = {str(item) for item in state.preflight.get("initial_risk_flags") or []}
    descriptors: list[str] = []
    if formula:
        descriptors.append(formula)
    if family:
        descriptors.append(family)
    if any("steroid" in flag or "polycyclic" in flag for flag in flags):
        descriptors.append("steroid")
    if descriptors:
        return " ".join(_dedupe_texts(descriptors)).strip()
    return "target"


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _execute_chemenzy_request(
    *,
    state: ToolExecutionState,
    request: dict[str, Any],
    request_path: Path,
    output_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    write_json(request_path, request)

    python_bin = _chem_enzy_python_bin()
    if python_bin is None:
        return {
            "schema_version": "chemenzy_run_result.v1",
            "accepted": False,
            "status": "runtime_unavailable",
            "reasons": ["chem_enzy_runtime_python_not_found"],
            "request_path": str(request_path),
        }

    cmd = [
        str(python_bin),
        str(ROOT / "scripts/run_chem_enzy_plan_for_web.py"),
        "--input",
        str(request_path),
        "--output",
        str(output_path),
        "--vendor-root",
        str(ROOT / "vendor/ChemEnzyRetroPlanner"),
        "--gpu",
        "-1",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    stdout_path = output_path.with_suffix(output_path.suffix + ".stdout.log")
    stderr_path = output_path.with_suffix(output_path.suffix + ".stderr.log")
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=float(timeout_s),
                check=False,
                env=env,
                start_new_session=True,
            )
            returncode = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        del exc
        return {
            "schema_version": "chemenzy_run_result.v1",
            "accepted": False,
            "status": "timeout",
            "reasons": ["chem_enzy_timeout"],
            "command": cmd,
            "stdout": stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else "",
            "stderr": stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else "",
        }

    if output_path.exists():
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {"accepted": False, "status": "invalid_json_output", "reasons": ["chemenzy_invalid_json_output"]}
    else:
        result = {"accepted": False, "status": "missing_output", "reasons": ["chemenzy_missing_output"]}
    result.setdefault("schema_version", "chemenzy_web_result.v1")
    result.setdefault("command", cmd)
    result.setdefault("stdout_path", str(stdout_path))
    result.setdefault("stderr_path", str(stderr_path))
    result.setdefault("exit_code", int(returncode))
    if returncode != 0:
        result.setdefault("accepted", False)
        result.setdefault("reasons", []).append("chemenzy_nonzero_exit")
        result["stderr"] = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    return result


def audit_route_and_extract_frontier(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "audit_route_and_extract_frontier", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["route_audit"] = result
        write_json(state.run_dir / "route_audit.json", result)
        return {"accepted": True, "audit": result}

    route_package = _route_package_from_artifacts(state)
    raw_route_verifier = _raw_route_verifier_from_artifacts(state)
    validation = _validation_from_artifacts(state, route_package)
    if raw_route_verifier and not raw_route_verifier.get("accepted"):
        validation = _validation_rejected_by_route_verifier(validation, raw_route_verifier)
    stock_audit_passed = bool(
        (payload.get("stock_audit_passed") or _stock_audit_from_artifacts(state))
        and (not raw_route_verifier or raw_route_verifier.get("accepted"))
    )
    report = audit_route_package(
        route_package,
        validation=validation,
        stock_audit_passed=stock_audit_passed,
        target_match=True,
        condition_candidates=[dict(item) for item in payload.get("condition_candidates") or []],
    )
    audit = report.to_dict()
    audit_validation = validate_route_audit_report(report)
    frontier = _frontier_from_route_package(route_package)
    result = {
        "schema_version": "codex_entry_route_audit_tool_result.v1",
        "accepted": bool(audit_validation.get("accepted")),
        "audit": audit,
        "audit_validation": audit_validation,
        "frontier_smiles": frontier,
        "route_package": route_package,
        "validation": validation,
        "raw_route_verifier": raw_route_verifier,
    }
    state.artifacts["route_audit"] = audit
    state.artifacts["frontier_smiles"] = frontier
    if raw_route_verifier:
        state.artifacts["route_verifier"] = raw_route_verifier
        write_json(state.run_dir / "route_verifier_report.json", raw_route_verifier)
    write_json(state.run_dir / "route_audit.json", audit)
    write_json(state.run_dir / "route_audit_tool_result.json", result)
    return result


def run_smiles_first_literature_workflow_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "run_smiles_first_literature_workflow", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["smiles_first"] = result
        write_json(state.run_dir / "smiles_first_workflow_result.json", result)
        return {"accepted": True, "result": result}

    if _native_route_verified_solved(state):
        result = _skip_result(
            schema_version="smiles_first_workflow_result.v1",
            reasons=["native_route_verified_solved"],
            route_status="solved",
        )
        state.artifacts["smiles_first"] = result
        write_json(state.run_dir / "smiles_first_workflow_result.json", result)
        return {"accepted": True, "result": result, "reasons": result["reasons"]}

    output_dir = state.run_dir / "smiles_first_literature_workflow"
    frontier = str(payload.get("frontier_smiles") or state.artifacts.get("frontier_smiles") or state.target_input["target_smiles"])
    result = run_smiles_first_workflow(
        SmilesFirstWorkflowConfig(
            target_smiles=state.target_input["target_smiles"],
            target_name=state.target_input.get("target_name", ""),
            family_hint=state.target_input.get("family_hint", ""),
            objective=str(payload.get("objective") or "codex-entry literature/template workflow"),
            output_dir=output_dir,
            frontier_smiles=frontier,
            baseline_json=_baseline_json_path(state),
            query_budget=int(payload.get("query_budget") or 6),
            literature_backend=str(payload.get("literature_backend") or "local"),
            worker_timeout_s=float(payload.get("worker_timeout_s") or state.budget.smiles_first_timeout_s),
            worker_max_output_bytes=200_000,
            worker_max_tool_calls=8,
        )
    )
    state.artifacts["smiles_first"] = result
    write_json(state.run_dir / "smiles_first_workflow_result.json", result)
    return {"accepted": True, "result": result}


def run_open_structure_research_agent(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "run_open_structure_research_agent", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["open_structure_research"] = result
        write_json(state.run_dir / "open_structure_research_result.json", result)
        return {"accepted": True, "mode": "mock", "result": result}

    if _native_route_verified_solved(state):
        result = _skip_result(
            schema_version="open_structure_research_result.v1",
            reasons=["native_route_verified_solved"],
            output_dir=str((state.run_dir / "open_structure_research").resolve()),
        )
        state.artifacts["open_structure_research"] = result
        write_json(state.run_dir / "open_structure_research_result.json", result)
        return result

    if state.codex_research_runs >= state.budget.max_codex_research_runs:
        return {"accepted": False, "reasons": ["codex_research_budget_exhausted"]}
    state.codex_research_runs += 1

    open_dir = (state.run_dir / "open_structure_research").resolve()
    open_dir.mkdir(parents=True, exist_ok=True)
    if not _is_relative_to(open_dir, state.run_dir.resolve()):
        return {"accepted": False, "reasons": ["open_research_output_outside_run_dir"]}
    context_root = state.run_dir.resolve()
    frontier = str(payload.get("frontier_smiles") or state.artifacts.get("frontier_smiles") or state.target_input["target_smiles"])
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_open_structure_template_agent.py"),
        "--output-dir",
        str(open_dir),
        "--target-name",
        str(state.target_input.get("target_name") or state.preflight.get("case_id") or "target"),
        "--search-name",
        str(payload.get("search_name") or _target_search_name(state)),
        "--target-smiles",
        str(state.target_input["target_smiles"]),
        "--frontier-smiles",
        frontier,
        "--context-root",
        str(context_root),
        "--key-path",
        str(state.key_path),
        "--base-url",
        str(state.base_url),
        "--model",
        str(state.model),
        "--timeout-s",
        str(float(payload.get("timeout_s") or state.budget.open_research_timeout_s)),
    ]
    experience_path = Path(str(payload.get("experience_path") or state.run_dir / "open_research_experience.json"))
    if not experience_path.is_absolute():
        experience_path = state.run_dir / experience_path
    if experience_path.exists():
        if not _is_relative_to(experience_path.resolve(), state.run_dir.resolve()):
            return {"accepted": False, "reasons": ["open_research_experience_outside_run_dir"]}
        cmd.extend(["--experience-path", str(experience_path.resolve())])
    if payload.get("prompt_path"):
        prompt_path = Path(str(payload["prompt_path"])).resolve()
        if not _is_relative_to(prompt_path, state.run_dir.resolve()):
            return {"accepted": False, "reasons": ["open_research_prompt_outside_run_dir"]}
        cmd.extend(["--prompt-path", str(prompt_path)])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=float(payload.get("timeout_s") or state.budget.open_research_timeout_s) + 30.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result = {
            "schema_version": "open_structure_research_result.v1",
            "accepted": False,
            "status": "timeout",
            "reasons": ["open_structure_research_timeout"],
            "command": cmd,
            "stdout": _timeout_stream(exc.stdout),
            "stderr": _timeout_stream(exc.stderr),
        }
        state.artifacts["open_structure_research"] = result
        write_json(state.run_dir / "open_structure_research_result.json", result)
        return result

    artifacts = _open_research_artifacts(open_dir)
    run_record = _load_open_research_run_record(open_dir)
    output_validation = _validate_open_research_output(open_dir=open_dir, run_record=run_record)
    experience = extract_open_research_experience(run_dir=open_dir, run_record=run_record)
    if (open_dir / "open_research_experience.json").exists():
        artifacts["open_research_experience"] = str(open_dir / "open_research_experience.json")
        write_json(state.run_dir / "open_research_experience.json", experience)
    reasons = []
    if proc.returncode != 0 and not output_validation.get("accepted"):
        reasons.append("open_structure_research_nonzero_exit")
    reasons.extend(str(item) for item in output_validation.get("reasons") or [])
    accepted = bool(output_validation.get("accepted")) and (
        proc.returncode == 0 or bool(output_validation.get("checkpoint_after_timeout"))
    )
    result = {
        "schema_version": "open_structure_research_result.v1",
        "accepted": accepted,
        "status": "completed" if accepted else "failed",
        "command": cmd,
        "exit_code": int(proc.returncode),
        "output_dir": str(open_dir),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "artifacts": artifacts,
        "run_record": run_record,
        "output_validation": output_validation,
    }
    compiled = _compile_open_research_downstream(
        state=state,
        open_dir=open_dir,
        target_smiles=str(state.target_input.get("target_smiles") or ""),
        prefer_local_seed=not accepted,
    )
    if compiled:
        result["compiled_downstream"] = compiled
        state.artifacts["compiled_downstream"] = compiled
        if accepted and not compiled.get("accepted"):
            result["accepted"] = False
            result["status"] = "failed"
            reasons.extend(str(item) for item in compiled.get("reasons") or ["compiled_downstream_unusable"])
        if not accepted and compiled.get("accepted"):
            result["continuation_available"] = True
            result["continuation_source"] = str(compiled.get("source") or "compiled_downstream")
    if reasons:
        result["reasons"] = sorted(set(reasons))
    state.artifacts["open_structure_research"] = result
    write_json(state.run_dir / "open_structure_research_result.json", result)
    return result


def extract_pdf_literature_structures_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "extract_pdf_literature_structures", payload)
    if mock is not None:
        result = dict(mock)
        _attach_literature_source_metadata(result, payload)
        _record_pdf_structure_evidence(state, result)
        write_json(state.run_dir / "literature_pdf_structure_evidence.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    out = _tool_output_dir(state, payload, default_name="literature_pdf_structure_extraction")
    pdf_path = _input_path(state, payload.get("pdf_path")) if payload.get("pdf_path") else None
    raw_image_values = [value for value in payload.get("image_paths") or [] if str(value or "").strip()]
    scheme_crops = [dict(item) for item in payload.get("scheme_crops") or [] if isinstance(item, dict)]
    image_paths = [str(_input_path(state, value)) for value in raw_image_values]
    if pdf_path is None and not image_paths and not scheme_crops:
        result = {
            "schema_version": "literature_pdf_structure_evidence.v1",
            "accepted": False,
            "status": "input_missing",
            "source_ref": str(payload.get("source_ref") or ""),
            "source_title": str(payload.get("source_title") or ""),
            "source_pdf_path": "",
            "rendered_pages": [],
            "indexed_images": [],
            "scheme_crops": [],
            "compound_text_snippets": [],
            "summary": {
                "rendered_page_count": 0,
                "indexed_image_count": 0,
                "scheme_crop_count": 0,
                "compound_text_snippet_count": 0,
            },
            "reasons": ["pdf_or_image_input_missing"],
        }
        _record_pdf_structure_evidence(state, result, output_dir=out)
        write_json(state.run_dir / "literature_pdf_structure_evidence.json", result)
        return {
            "accepted": False,
            "result": result,
            "artifact_refs": {
                "literature_pdf_structure_evidence": str(out / "literature_pdf_structure_evidence.json"),
            },
            "reasons": result["reasons"],
        }
    result = extract_literature_pdf_assets(
        pdf_path=pdf_path,
        output_dir=out,
        page_numbers=[int(item) for item in payload.get("page_numbers") or []],
        render_zoom=float(payload.get("render_zoom") or 2.0),
        image_paths=image_paths,
        scheme_crops=scheme_crops,
        compound_labels=[str(item) for item in payload.get("compound_labels") or [] if str(item).strip()],
    )
    _attach_literature_source_metadata(result, payload)
    _record_pdf_structure_evidence(state, result, output_dir=out)
    write_json(state.run_dir / "literature_pdf_structure_evidence.json", result)
    return {
        "accepted": bool(result.get("accepted")),
        "result": result,
        "artifact_refs": {
            "literature_pdf_structure_evidence": str(out / "literature_pdf_structure_evidence.json"),
        },
        "reasons": [str(item) for item in result.get("reasons") or []],
    }


def _attach_literature_source_metadata(result: dict[str, Any], payload: dict[str, Any]) -> None:
    if str(payload.get("source_ref") or "").strip():
        result["source_ref"] = str(payload.get("source_ref") or "").strip()
    if str(payload.get("source_title") or "").strip():
        result["source_title"] = str(payload.get("source_title") or "").strip()
    if str(payload.get("pdf_path") or "").strip() and not str(result.get("source_pdf_path") or "").strip():
        result["source_pdf_path"] = str(payload.get("pdf_path") or "").strip()


def _record_pdf_structure_evidence(state: ToolExecutionState, result: dict[str, Any], *, output_dir: Path | None = None) -> None:
    state.artifacts["literature_pdf_structure_evidence"] = result
    if output_dir is not None:
        state.artifacts["literature_pdf_structure_evidence_dir"] = str(output_dir)

    history = state.artifacts.setdefault("literature_pdf_structure_evidence_history", [])
    if isinstance(history, list):
        key = _pdf_evidence_key(result)
        if key:
            history[:] = [
                dict(row)
                for row in history
                if isinstance(row, dict) and _pdf_evidence_key(row) != key
            ]
        history.append(dict(result))

    by_source = state.artifacts.setdefault("literature_pdf_structure_evidence_by_source", {})
    if isinstance(by_source, dict):
        for key in _pdf_evidence_keys(result):
            by_source[key] = dict(result)


def extract_visual_literature_chain_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "extract_visual_literature_chain", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["visual_literature_chain_extraction"] = result
        if result.get("candidate_chain"):
            candidate = dict(result.get("candidate_chain") or {})
            state.artifacts["visual_structure_candidate_chain"] = candidate
        _record_visual_chain_result(state, result)
        write_json(state.run_dir / "visual_literature_chain_extraction_result.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    out = _tool_output_dir(state, payload, default_name="visual_literature_chain_extraction")
    pdf_evidence = _pdf_evidence_from_payload_or_artifacts(state, payload)
    image_paths = _visual_chain_image_paths(state, payload, pdf_evidence)
    if not image_paths:
        result = {
            "schema_version": "visual_literature_chain_extraction_result.v1",
            "accepted": False,
            "status": "failed",
            "target_name": str(payload.get("target_name") or state.target_input.get("target_name") or state.preflight.get("case_id") or "target"),
            "target_smiles": str(payload.get("target_smiles") or state.target_input.get("target_smiles") or ""),
            "image_paths": [],
            "candidate_chain": {},
            "candidate_step_count": 0,
            "reasons": ["visual_input_images_missing"],
            "extraction_policy": {
                "pdf_reuse_allowed": True,
                "prior_candidate_chain_reuse_allowed": False,
                "prior_source_detail_records_reuse_allowed": False,
                "must_derive_from_current_images": True,
                "no_solved_claim": True,
            },
        }
        state.artifacts["visual_literature_chain_extraction"] = result
        _record_visual_chain_result(state, result)
        write_json(state.run_dir / "visual_literature_chain_extraction_result.json", result)
        return {
            "accepted": False,
            "result": result,
            "artifact_refs": {
                "visual_literature_chain_extraction": str(out / "visual_literature_chain_extraction_result.json"),
            },
            "reasons": result["reasons"],
        }
    result = run_visual_literature_chain_agent(
        image_paths=image_paths,
        output_dir=out,
        target_name=str(payload.get("target_name") or state.target_input.get("target_name") or state.preflight.get("case_id") or "target"),
        target_smiles=_literature_tool_target_smiles(state, payload),
        source_ref=str(payload.get("source_ref") or "doi:10.1016/j.tet.2025.134610"),
        source_title=str(payload.get("source_title") or ""),
        expected_labels=[str(item) for item in payload.get("expected_labels") or [] if str(item).strip()],
        route_sequence_hint=str(payload.get("route_sequence_hint") or ""),
        text_snippets=[dict(item) for item in (pdf_evidence.get("compound_text_snippets") or []) if isinstance(item, dict)],
        key_path=state.key_path,
        base_url=state.base_url,
        model=state.model,
        timeout_s=_visual_literature_timeout_s(state, payload),
        allow_repair=not bool(payload.get("focused_gap_repair")) and _visual_literature_repair_enabled(payload),
    )
    if str(payload.get("source_ref") or "").strip():
        result["source_ref"] = str(payload.get("source_ref") or "").strip()
    if str(payload.get("source_title") or "").strip():
        result["source_title"] = str(payload.get("source_title") or "").strip()
    if str(payload.get("pdf_path") or "").strip():
        result["source_pdf_path"] = str(payload.get("pdf_path") or "").strip()
    candidate_path_value = str(result.get("candidate_chain_path") or "").strip()
    candidate_path = Path(candidate_path_value) if candidate_path_value else None
    if candidate_path is not None and candidate_path.is_file():
        try:
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            candidate = {}
        if isinstance(candidate, dict):
            state.artifacts["visual_structure_candidate_chain"] = dict(candidate)
            state.artifacts["visual_structure_candidate_chain_path"] = str(candidate_path)
    state.artifacts["visual_literature_chain_extraction"] = result
    _record_visual_chain_result(state, result)
    write_json(state.run_dir / "visual_literature_chain_extraction_result.json", result)
    return {
        "accepted": bool(result.get("accepted")),
        "result": result,
        "artifact_refs": {
            "visual_literature_chain_extraction": str(out / "visual_literature_chain_extraction_result.json"),
            **({"visual_structure_candidate_chain": str(candidate_path)} if candidate_path is not None and candidate_path.is_file() else {}),
        },
        "reasons": [str(item) for item in result.get("reasons") or []],
    }


def _visual_literature_timeout_s(state: ToolExecutionState, payload: dict[str, Any]) -> float:
    explicit = payload.get("timeout_s")
    if explicit is None:
        explicit = os.environ.get("AUTOPLANNER_VISUAL_TIMEOUT_S")
    if explicit is not None:
        return max(10.0, float(explicit))
    return min(float(state.budget.open_research_timeout_s or 240.0), 240.0)


def _visual_literature_repair_enabled(payload: dict[str, Any]) -> bool:
    raw = payload.get("allow_repair")
    if raw is None:
        raw = os.environ.get("AUTOPLANNER_VISUAL_ALLOW_REPAIR")
    if raw is None:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "off", "no", "disabled"}
    return bool(raw)


def resolve_literature_structure_task_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "resolve_literature_structure_task", payload)
    if mock is not None:
        result = dict(mock)
        _record_structure_resolution_result(state, result)
        write_json(state.run_dir / "literature_structure_resolution_result.json", result)
        return result

    out = _tool_output_dir(state, payload, default_name="literature_structure_resolution")
    task_id = str(payload.get("task_id") or "").strip()
    label = str(payload.get("label") or payload.get("compound_label") or "").strip()
    source_ref = str(payload.get("source_ref") or "").strip()
    source_title = str(payload.get("source_title") or payload.get("title") or "").strip()
    reasons: list[str] = []
    if not task_id:
        reasons.append("structure_resolution_task_id_missing")
    if not label:
        reasons.append("structure_resolution_label_missing")

    pdf_evidence = _pdf_evidence_from_payload_or_artifacts(state, payload)
    image_paths = _visual_chain_image_paths(state, payload, pdf_evidence) if label else []
    candidate_rows = _structure_resolution_candidate_rows_from_payload(
        payload,
        task_id=task_id,
        label=label,
        source_ref=source_ref,
        source_title=source_title,
    )
    visual_attempt: dict[str, Any] = {}
    if not any(row.get("accepted") for row in candidate_rows) and label and _structure_resolution_visual_enabled(payload):
        visual_attempt = run_visual_literature_chain_agent(
            image_paths=image_paths,
            output_dir=out,
            target_name=str(payload.get("target_name") or label),
            target_smiles=str(payload.get("target_smiles") or state.target_input.get("target_smiles") or ""),
            source_ref=source_ref or "structure_resolution_source",
            source_title=source_title,
            expected_labels=[label],
            route_sequence_hint=_structure_resolution_visual_prompt_hint(label=label, payload=payload),
            text_snippets=[dict(item) for item in (pdf_evidence.get("compound_text_snippets") or []) if isinstance(item, dict)],
            key_path=state.key_path,
            base_url=state.base_url,
            model=state.model,
            timeout_s=_structure_resolution_timeout_s(state, payload),
            allow_repair=False,
        )
        candidate_rows.extend(
            _structure_resolution_candidate_rows_from_visual_attempt(
                visual_attempt,
                task_id=task_id,
                label=label,
                source_ref=source_ref,
                source_title=source_title,
            )
        )
    if _structure_resolution_visual_enabled(payload) and not image_paths and not any(row.get("accepted") for row in candidate_rows):
        reasons.append("structure_resolution_visual_images_missing")

    accepted_candidates = [row for row in candidate_rows if row.get("accepted")]
    unresolved = []
    if not accepted_candidates:
        unresolved.append(
            {
                "schema_version": "literature_structure_unresolved_task.v1",
                "task_id": task_id,
                "label": label,
                "source_ref": source_ref,
                "source_title": source_title,
                "status": "unresolved",
                "reason": "no_rdkit_valid_source_grounded_structure_candidate",
                "next_actions": [
                    "search_supplementary_information_for_label",
                    "crop_higher_resolution_scheme_region",
                    "ask_user_for_source_detail_or_structure",
                ],
                "no_solved_claim": True,
            }
        )
        reasons.append("no_rdkit_valid_structure_candidate")

    result = {
        "schema_version": "literature_structure_resolution_result.v1",
        "accepted": bool(accepted_candidates),
        "status": "resolved" if accepted_candidates else "unresolved",
        "task_id": task_id,
        "label": label,
        "source_ref": source_ref,
        "source_title": source_title,
        "resolved_structures": accepted_candidates,
        "rejected_candidates": [row for row in candidate_rows if not row.get("accepted")],
        "unresolved_tasks": unresolved,
        "selected_image_paths": [str(path) for path in image_paths],
        "visual_attempt": visual_attempt,
        "artifact_refs": {
            "literature_structure_resolution": str(out / "literature_structure_resolution_result.json"),
        },
        "source_policy": {
            "structure_candidates_require_rdkit_valid_smiles": True,
            "source_grounding_required": True,
            "no_solved_claim": True,
            "production_write_blocked": True,
            "does_not_emit_exact_literature_rows": True,
        },
        "reasons": sorted(set(reasons)),
        "no_solved_claim": True,
    }
    _record_structure_resolution_result(state, result)
    write_json(out / "literature_structure_resolution_result.json", result)
    write_json(state.run_dir / "literature_structure_resolution_result.json", result)
    return result


def _record_structure_resolution_result(state: ToolExecutionState, result: dict[str, Any]) -> None:
    state.artifacts["literature_structure_resolution"] = dict(result)
    history = state.artifacts.setdefault("literature_structure_resolution_history", [])
    if isinstance(history, list):
        history.append(dict(result))


def _structure_resolution_visual_enabled(payload: dict[str, Any]) -> bool:
    raw = payload.get("run_visual")
    if raw is None:
        raw = payload.get("use_visual")
    if raw is None:
        raw = os.environ.get("AUTOPLANNER_STRUCTURE_RESOLUTION_VISUAL", "1")
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "off", "no", "disabled"}
    return bool(raw)


def _structure_resolution_timeout_s(state: ToolExecutionState, payload: dict[str, Any]) -> float:
    explicit = payload.get("timeout_s") or os.environ.get("AUTOPLANNER_STRUCTURE_RESOLUTION_TIMEOUT_S")
    if explicit is not None:
        return max(10.0, float(explicit))
    return min(float(state.budget.open_research_timeout_s or 180.0), 180.0)


def _structure_resolution_visual_prompt_hint(*, label: str, payload: dict[str, Any]) -> str:
    source_locator = str(payload.get("source_locator") or "")
    hint = str(payload.get("route_sequence_hint") or "")
    parts = [
        f"Resolve only compound label {label}.",
        "Return a visual_structure_candidate_chain JSON with one step when the drawn structure can be converted to RDKit-valid SMILES.",
        "If stereochemistry is not fully legible but atom connectivity/protecting groups are visible, return an achiral/connectivity-only SMILES and mark it as not_exact_literature_segment=true, stereochemistry_status=unspecified_or_partial, allowed_use=exploratory_template_and_guided_hint_only.",
        "If protecting groups or atom connectivity are not legible even at connectivity-only level, omit the step and add an extraction_gaps row.",
        "Do not infer a route, do not invent a reaction, and do not claim solved.",
    ]
    if source_locator:
        parts.append(f"Prior source locator: {source_locator}.")
    if hint:
        parts.append(hint)
    return " ".join(parts)


def _structure_resolution_candidate_rows_from_payload(
    payload: dict[str, Any],
    *,
    task_id: str,
    label: str,
    source_ref: str,
    source_title: str,
) -> list[dict[str, Any]]:
    raw_candidates: list[Any] = []
    if payload.get("candidate_smiles"):
        raw_candidates.append({"smiles": payload.get("candidate_smiles"), "source_locator": payload.get("source_locator")})
    raw_candidates.extend(payload.get("candidate_structures") or [])
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_candidates, start=1):
        if isinstance(raw, str):
            candidate = {"smiles": raw}
        elif isinstance(raw, dict):
            candidate = dict(raw)
        else:
            continue
        rows.append(
            _structure_resolution_candidate_row(
                candidate,
                task_id=task_id,
                label=label,
                source_ref=source_ref,
                source_title=source_title,
                candidate_index=idx,
                derivation_mode="payload_candidate",
            )
        )
    return rows


def _structure_resolution_candidate_rows_from_visual_attempt(
    visual_attempt: dict[str, Any],
    *,
    task_id: str,
    label: str,
    source_ref: str,
    source_title: str,
) -> list[dict[str, Any]]:
    chain = _load_visual_candidate_chain_from_result(visual_attempt)
    rows: list[dict[str, Any]] = []
    for idx, step in enumerate(chain.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        product_label = str(step.get("product_label") or step.get("label") or "").strip()
        if product_label and not _structure_label_matches(product_label, label):
            continue
        rows.append(
            _structure_resolution_candidate_row(
                {
                    "smiles": step.get("product_smiles"),
                    "source_locator": step.get("source_locator") or (step.get("structure_derivation") or {}).get("source_locator"),
                    "evidence_refs": step.get("evidence_refs"),
                    "confidence": (step.get("structure_derivation") or {}).get("confidence") or step.get("confidence"),
                },
                task_id=task_id,
                label=label,
                source_ref=source_ref,
                source_title=source_title,
                candidate_index=idx,
                derivation_mode="focused_visual_structure_resolution",
            )
        )
    return rows


def _load_visual_candidate_chain_from_result(result: dict[str, Any]) -> dict[str, Any]:
    path_value = str(result.get("candidate_chain_path") or "").strip()
    if path_value:
        path = Path(path_value)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict):
                return data
    candidate = result.get("candidate_chain")
    if isinstance(candidate, dict):
        return dict(candidate)
    parsed = result.get("parsed_output")
    if isinstance(parsed, dict):
        return dict(parsed)
    return {}


def _structure_resolution_candidate_row(
    candidate: dict[str, Any],
    *,
    task_id: str,
    label: str,
    source_ref: str,
    source_title: str,
    candidate_index: int,
    derivation_mode: str,
) -> dict[str, Any]:
    smiles = str(candidate.get("smiles") or candidate.get("product_smiles") or "").strip()
    valid = _valid_smiles(smiles)
    source_locator = str(candidate.get("source_locator") or candidate.get("locator") or "").strip()
    reasons: list[str] = []
    if not smiles:
        reasons.append("candidate_smiles_missing")
    elif not valid:
        reasons.append("candidate_smiles_invalid")
    if not source_locator:
        reasons.append("source_locator_missing")
    return {
        "schema_version": "literature_resolved_structure_candidate.v1",
        "structure_id": f"{_resolution_safe_id(task_id or label)}:{candidate_index}",
        "task_id": task_id,
        "label": label,
        "smiles": smiles,
        "source_ref": source_ref,
        "source_title": source_title,
        "source_locator": source_locator,
        "evidence_refs": [str(item) for item in candidate.get("evidence_refs") or [] if str(item or "").strip()],
        "confidence": str(candidate.get("confidence") or "low"),
        "derivation_mode": derivation_mode,
        "rdkit_valid": bool(valid),
        "accepted": bool(valid and source_locator),
        "reasons": reasons,
        "no_solved_claim": True,
    }


def _structure_label_matches(observed: str, expected: str) -> bool:
    observed_text = " ".join(str(observed or "").lower().split())
    expected_text = " ".join(str(expected or "").lower().split())
    return bool(observed_text and expected_text and (observed_text == expected_text or expected_text in observed_text or observed_text in expected_text))


def _resolution_safe_id(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "structure"))
    return "_".join(part for part in safe.split("_") if part)[:100] or "structure"


def apply_source_text_condition_repairs_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "apply_source_text_condition_repairs", payload)
    if mock is not None:
        result = dict(mock)
        if result.get("candidate_chain"):
            state.artifacts["visual_structure_candidate_chain"] = dict(result.get("candidate_chain") or {})
        write_json(state.run_dir / "source_text_condition_repairs_result.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    out = _tool_output_dir(state, payload, default_name="source_text_condition_repairs")
    candidate = _candidate_chain_payload_from_payload_or_artifacts(state, payload)
    if not candidate:
        return {"accepted": False, "reasons": ["candidate_chain_missing"]}
    repairs = _condition_repair_rows(payload.get("condition_repairs") or payload.get("repairs"))
    if not repairs:
        return {"accepted": False, "reasons": ["condition_repairs_missing"]}

    by_step_id = {str(row.get("step_id") or ""): row for row in repairs if str(row.get("step_id") or "")}
    by_product = {str(row.get("product_label") or ""): row for row in repairs if str(row.get("product_label") or "")}
    steps = [dict(item) for item in candidate.get("steps") or [] if isinstance(item, dict)]
    applied: list[dict[str, Any]] = []
    unmatched = set(id(row) for row in repairs)
    repaired_steps: list[dict[str, Any]] = []
    for step in steps:
        repaired = dict(step)
        repair = by_step_id.get(str(step.get("step_id") or "")) or by_product.get(str(step.get("product_label") or ""))
        if repair:
            unmatched.discard(id(repair))
            repaired = _apply_condition_repair_to_step(repaired, repair)
            applied.append(
                {
                    "step_id": str(step.get("step_id") or ""),
                    "product_label": str(step.get("product_label") or ""),
                    "source_locator": str(repair.get("source_locator") or ""),
                }
            )
        repaired_steps.append(repaired)
    repaired_chain = {
        **candidate,
        "steps": repaired_steps,
        "condition_repair_audit": {
            "schema_version": "source_text_condition_repair_audit.v1",
            "repair_scope": "condition_candidate_source_excerpt_locator_only",
            "structure_smiles_unchanged": _step_smiles_signature(steps) == _step_smiles_signature(repaired_steps),
            "applied_repair_count": len(applied),
            "unmatched_repair_count": len(unmatched),
            "source_ref": str(payload.get("source_ref") or candidate.get("source_ref") or ""),
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
    }
    accepted = bool(applied) and bool(repaired_chain["condition_repair_audit"]["structure_smiles_unchanged"])
    reasons: list[str] = []
    if not applied:
        reasons.append("condition_repairs_no_matching_steps")
    if not repaired_chain["condition_repair_audit"]["structure_smiles_unchanged"]:
        reasons.append("condition_repair_changed_smiles")
    path = out / "visual_structure_candidate_chain_condition_repaired.json"
    path.write_text(json.dumps(repaired_chain, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": "source_text_condition_repairs_result.v1",
        "accepted": accepted,
        "status": "completed" if accepted else "failed",
        "candidate_chain_path": str(path),
        "candidate_chain": repaired_chain,
        "applied_repairs": applied,
        "summary": {
            "input_step_count": len(steps),
            "repair_count": len(repairs),
            "applied_repair_count": len(applied),
            "unmatched_repair_count": len(unmatched),
        },
        "source_policy": {
            "smiles_mutation_allowed": False,
            "condition_source_text_repair_allowed": True,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
        "artifact_refs": {
            "visual_structure_candidate_chain": str(path),
        },
        "reasons": reasons,
    }
    state.artifacts["visual_structure_candidate_chain"] = repaired_chain
    state.artifacts["visual_structure_candidate_chain_path"] = str(path)
    state.artifacts["source_text_condition_repairs"] = result
    write_json(state.run_dir / "source_text_condition_repairs_result.json", result)
    return result


def validate_literature_intermediate_chain_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "validate_literature_intermediate_chain", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["visual_structure_chain_validation"] = result
        write_json(state.run_dir / "visual_structure_chain_validation.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    out = _tool_output_dir(state, payload, default_name="literature_intermediate_chain_validation")
    candidate_chain: dict[str, Any] | Path
    if payload.get("candidate_chain"):
        candidate_chain = dict(payload.get("candidate_chain") or {})
    elif payload.get("candidate_chain_path"):
        candidate_chain = _input_path(state, payload.get("candidate_chain_path"))
    elif state.artifacts.get("visual_structure_candidate_chain"):
        candidate_chain = dict(state.artifacts.get("visual_structure_candidate_chain") or {})
    elif state.artifacts.get("visual_structure_candidate_chain_path"):
        candidate_chain = _input_path(state, state.artifacts.get("visual_structure_candidate_chain_path"))
    else:
        return {"accepted": False, "reasons": ["candidate_chain_missing"]}
    result = validate_visual_structure_chain(
        candidate_chain,
        output_dir=out,
        target_smiles=_literature_tool_target_smiles(state, payload),
        require_contiguous=bool(payload.get("require_contiguous", True)),
    )
    state.artifacts["visual_structure_chain_validation"] = result
    state.artifacts["visual_structure_chain_validation_dir"] = str(out)
    write_json(state.run_dir / "visual_structure_chain_validation.json", result)
    return {
        "accepted": bool(result.get("accepted")),
        "result": result,
        "artifact_refs": {
            "visual_structure_chain_validation": str(out / "visual_structure_chain_validation.json"),
        },
        "reasons": [str(item) for item in result.get("reasons") or []],
    }


def build_source_detail_curator_records_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "build_source_detail_curator_records", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["source_detail_curator_records"] = result
        write_json(state.run_dir / "source_detail_curator_records_tool_result.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    out = _tool_output_dir(state, payload, default_name="open_structure_research")
    validation = _validation_payload_from_payload_or_artifacts(state, payload)
    if not validation:
        return {"accepted": False, "reasons": ["visual_structure_chain_validation_missing"]}
    curator_records = build_source_detail_curator_records_from_chain(
        validation,
        output_dir=out,
        source_ref=str(payload.get("source_ref") or validation.get("source_ref") or ""),
        source_title=str(payload.get("source_title") or validation.get("source_title") or ""),
        evidence_refs=[str(item) for item in payload.get("evidence_refs") or [] if str(item).strip()],
        record_id=str(payload.get("record_id") or ""),
        provenance=str(payload.get("provenance") or "codex_source_text_translation"),
        main_reactant_only=bool(payload.get("main_reactant_only", True)),
        write_file=True,
    )
    resolution = resolve_curator_records_to_source_detail_steps(
        curator_records,
        output_dir=out,
        target_name=str(state.target_input.get("target_name") or state.preflight.get("case_id") or ""),
        target_smiles=str(state.target_input.get("target_smiles") or ""),
        source_ref=str(payload.get("source_ref") or validation.get("source_ref") or ""),
    )
    downstream_payload = {
        "schema_version": "open_downstream_consumables.v1",
        "case_id": str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        "planner_handoff": {
            "next_action": "template_plugin_rerun",
            "solved": False,
            "production_kb_promotion": False,
        },
        "guided_rerun_requests": [],
        "literature_template_cards": [],
        "literature_route_segments": [],
        "executable_template_candidates": [],
        "source_detail_route_steps": resolution.get("downstream_patch", {}).get("source_detail_route_steps") or [],
        "route_expansion_tasks": [],
        "evolution_candidates": [],
        "rejected_consumables": [],
    }
    compiled = compile_downstream_consumables(
        downstream_payload,
        target_smiles=str(state.target_input.get("target_smiles") or ""),
        case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        enable_online_anchor_resolution=_online_anchor_resolution_enabled(state.target_input),
    )
    write_compiled_downstream_artifacts(compiled, output_dir=out)
    result = {
        "schema_version": "source_detail_curator_records_tool_result.v1",
        "accepted": bool(resolution.get("accepted")) and bool(compiled.get("accepted")),
        "curator_records": curator_records,
        "source_detail_resolution": resolution,
        "compiled_downstream": compiled,
        "artifact_refs": {
            "source_detail_curator_records": str(source_detail_curator_records_path(out)),
            "source_detail_resolution_pack": str(out / "evidence" / "source_detail_resolution_pack.json"),
            "compiled_downstream_consumables": str(out / "compiled_downstream_consumables.json"),
            "compiled_literature_template_plugin": str(out / "compiled_literature_template_plugin.json"),
        },
        "summary": {
            "curator_record_count": len(curator_records.get("records") or []),
            "source_detail_route_step_count": len(resolution.get("source_detail_route_steps") or []),
            "one_step_row_count": len(((compiled.get("literature_template_plugin") or {}).get("one_step_rows") or [])),
        },
        "reasons": [str(item) for item in compiled.get("reasons") or []],
    }
    state.artifacts["source_detail_curator_records"] = curator_records
    state.artifacts["source_detail_resolution"] = resolution
    state.artifacts["compiled_downstream_result"] = _compiled_downstream_harness_result(
        compiled,
        output_dir=out,
        source="source_detail_curator_records_tool",
    )
    state.artifacts["compiled_downstream"] = compiled
    state.artifacts["compiled_downstream_payload"] = compiled
    write_json(state.run_dir / "source_detail_curator_records_tool_result.json", result)
    return result


def build_analogical_retrosynthesis_hypotheses_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "build_analogical_retrosynthesis_hypotheses", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["analogical_retrosynthesis_hypotheses"] = result
        write_json(state.run_dir / "analogical_retrosynthesis_hypotheses.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    compiled = _compiled_downstream_from_state(state)
    if isinstance(payload.get("compiled_downstream"), dict):
        compiled = dict(payload.get("compiled_downstream") or {})
    elif payload.get("compiled_downstream_path"):
        explicit = _json_payload_or_path(state, payload.get("compiled_downstream_path"))
        if explicit:
            compiled = explicit
    result = build_analogical_retrosynthesis_hypotheses(
        compiled_downstream=compiled,
        target_smiles=str(payload.get("target_smiles") or state.target_input.get("target_smiles") or ""),
        target_name=str(payload.get("target_name") or state.target_input.get("target_name") or ""),
        case_id=str(payload.get("case_id") or state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        max_hypotheses=int(payload.get("max_hypotheses") or 12),
    )
    state.artifacts["analogical_retrosynthesis_hypotheses"] = result
    write_json(state.run_dir / "analogical_retrosynthesis_hypotheses.json", result)
    return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}


def compile_source_detail_chain_route_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "compile_source_detail_chain_route", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["source_detail_chain_route"] = result
        write_json(state.run_dir / "source_detail_chain_route_result.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    out = _tool_output_dir(state, payload, default_name="source_detail_chain_route")
    compiled = _compiled_payload_from_payload_or_artifacts(state, payload)
    steps = _source_detail_steps_from_payload_or_artifacts(state, payload)
    result = compile_source_detail_chain_route_artifact(
        source_detail_steps=steps,
        compiled_downstream=compiled,
        output_dir=out,
        target_smiles=str(payload.get("target_smiles") or state.target_input.get("target_smiles") or ""),
        case_id=str(payload.get("case_id") or state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        terminal_smiles=str(payload.get("terminal_smiles") or ""),
        terminal_name=str(payload.get("terminal_name") or ""),
    )
    plugin_payload = ((result.get("compiled_downstream") or {}).get("literature_template_plugin") or {})
    validation = _validation_payload_from_payload_or_artifacts(state, payload)
    if plugin_payload and validation:
        result["plugin_probe"] = probe_literature_plugin_chain(
            plugin_payload=plugin_payload,
            validation=validation,
            output_dir=out,
        )
    state.artifacts["source_detail_chain_route"] = result
    compiled_downstream = result.get("compiled_downstream")
    if isinstance(compiled_downstream, dict) and compiled_downstream:
        state.artifacts["compiled_downstream"] = dict(compiled_downstream)
        state.artifacts["compiled_downstream_payload"] = dict(compiled_downstream)
    write_json(state.run_dir / "source_detail_chain_route_result.json", result)
    return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}


def stitch_literature_chain_with_subgoal_route_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "stitch_literature_chain_with_subgoal_route", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["stitched_semisynthesis_route"] = result
        write_json(state.run_dir / "stitched_semisynthesis_route_result.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    out = _tool_output_dir(state, payload, default_name="stitched_semisynthesis_route")
    literature_chain = _json_payload_or_path(
        state,
        payload.get("literature_chain_audit") or payload.get("literature_chain_audit_path"),
    )
    if not literature_chain:
        route_result = dict(state.artifacts.get("source_detail_chain_route") or {})
        literature_chain = dict(route_result.get("chain_audit") or {})
    route_expansion = _json_payload_or_path(
        state,
        payload.get("route_expansion_result") or payload.get("route_expansion_result_path"),
    )
    if not route_expansion:
        route_expansion = dict(state.artifacts.get("route_expansion_subgoal_search") or {})
    subgoal_verifier = _json_payload_or_path(
        state,
        payload.get("subgoal_verifier") or payload.get("subgoal_verifier_path"),
    )
    subgoal_raw = _json_payload_or_path(
        state,
        payload.get("subgoal_raw_result") or payload.get("subgoal_raw_result_path"),
    )
    result = compile_stitched_semisynthesis_route(
        literature_chain_audit=literature_chain,
        subgoal_verifier=subgoal_verifier,
        subgoal_raw_result=subgoal_raw,
        route_expansion_result=route_expansion,
        output_dir=out,
        case_id=str(payload.get("case_id") or state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        target_smiles=str(payload.get("target_smiles") or state.target_input.get("target_smiles") or ""),
        target_name=str(payload.get("target_name") or state.target_input.get("target_name") or ""),
        subgoal_name=str(payload.get("subgoal_name") or ""),
    )
    state.artifacts["stitched_semisynthesis_route"] = result
    write_json(state.run_dir / "stitched_semisynthesis_route_result.json", result)
    return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}


def compile_hybrid_route_set_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "compile_hybrid_route_set", payload)
    if mock is not None:
        result = dict(mock)
        state.artifacts["hybrid_route_set"] = result
        write_json(state.run_dir / "hybrid_route_set.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    out = _tool_output_dir(state, payload, default_name="hybrid_route_set")
    literature_chain = _json_payload_or_path(state, payload.get("literature_chain_audit") or payload.get("literature_chain_audit_path"))
    if not literature_chain:
        route_result = dict(state.artifacts.get("source_detail_chain_route") or {})
        literature_chain = dict(route_result.get("chain_audit") or {})
    chemenzy = _json_payload_or_path(state, payload.get("chemenzy_result") or payload.get("chemenzy_result_path"))
    if not chemenzy:
        chemenzy = dict(state.artifacts.get("guided_chemenzy") or state.artifacts.get("chemenzy") or {})
        if chemenzy.get("result") and not chemenzy.get("routes"):
            chemenzy = dict(chemenzy.get("result") or {})
    verifier = _json_payload_or_path(state, payload.get("verifier_report") or payload.get("verifier_report_path"))
    if not verifier:
        verifier = dict(state.artifacts.get("route_verifier") or {})
    result = compile_hybrid_route_set_artifact(
        output_dir=out,
        case_id=str(payload.get("case_id") or state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        target_smiles=str(payload.get("target_smiles") or state.target_input.get("target_smiles") or ""),
        literature_chain_audit=literature_chain,
        chemenzy_result=chemenzy,
        verifier_report=verifier,
    )
    state.artifacts["hybrid_route_set"] = result
    write_json(state.run_dir / "hybrid_route_set.json", result)
    return {"accepted": bool(result.get("accepted")), "result": result}


def validate_artifact_bundle_tool(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    mock = _mock_result(state, "validate_artifact_bundle", payload)
    if mock is not None:
        validation = dict(mock)
        state.validations.append(validation)
        write_json(state.run_dir / "artifact_bundle_validation.json", validation)
        return {"accepted": bool(validation.get("accepted", True)), "validation": validation}

    reasons: list[str] = []
    route_audit = dict(state.artifacts.get("route_audit") or {})
    if route_audit.get("route_status") == "solved" and not route_audit.get("stock_audit_passed"):
        reasons.append("solved_without_stock_audit")
    if route_audit.get("fake_closure_rejected"):
        reasons.append("fake_closure_evidence_present")
    guided = dict(state.artifacts.get("guided_chemenzy") or {})
    guided_verifier = dict(guided.get("raw_route_verifier") or {})
    if guided_verifier and not guided_verifier.get("accepted"):
        reasons.extend(str(item) for item in guided_verifier.get("reasons") or ["guided_route_verifier_rejected"])
    subgoal_search = dict(state.artifacts.get("route_expansion_subgoal_search") or {})
    if subgoal_search and not subgoal_search.get("accepted", True):
        reasons.extend(str(item) for item in subgoal_search.get("reasons") or ["route_expansion_subgoal_search_failed"])
    open_research = state.artifacts.get("open_structure_research")
    if isinstance(open_research, dict) and not open_research.get("accepted", True):
        reasons.extend(str(item) for item in open_research.get("reasons") or ["open_structure_research_failed"])
    compiled_downstream = state.artifacts.get("compiled_downstream")
    open_research_skipped = isinstance(open_research, dict) and bool(open_research.get("skipped"))
    if isinstance(open_research, dict) and open_research.get("accepted") and not open_research_skipped and not compiled_downstream:
        reasons.append("open_research_missing_compiled_downstream")
    if isinstance(compiled_downstream, dict) and not compiled_downstream.get("accepted", True):
        reasons.extend(str(item) for item in compiled_downstream.get("reasons") or ["compiled_downstream_unusable"])
    self_evo_replay = dict(state.artifacts.get("self_evo_replay") or {})
    if self_evo_replay:
        if self_evo_replay.get("target_run") and int(self_evo_replay.get("production_promoted_count") or 0):
            reasons.append("self_evo_target_run_promoted_production")
        if self_evo_replay.get("target_run") and not self_evo_replay.get("production_write_blocked"):
            reasons.append("self_evo_target_run_production_not_blocked")
    if _artifact_tree_contains_raw_reaction(state.artifacts):
        reasons.append("raw_reaction_injection")
    validation = {
        "schema_version": "codex_entry_artifact_bundle_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "artifact_keys": sorted(state.artifacts),
    }
    state.validations.append(validation)
    write_json(state.run_dir / "artifact_bundle_validation.json", validation)
    return {"accepted": bool(validation["accepted"]), "validation": validation, "reasons": validation["reasons"]}


def artifact_bundle_from_state(
    *,
    state: ToolExecutionState,
    workflow_plan: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> ArtifactBundle:
    return ArtifactBundle(
        case_id=str(state.preflight.get("case_id") or state.target_input.get("case_id") or "target"),
        target_input=dict(state.target_input),
        preflight=dict(state.preflight),
        workflow_plan=dict(workflow_plan),
        tool_calls=tool_calls,
        artifacts=dict(state.artifacts),
        validations=list(state.validations),
        safety_flags=sorted(set(state.safety_flags)),
        run_semantics=str(state.run_semantics or CANONICAL_RUN_SEMANTICS),
    )


def _route_package_from_artifacts(state: ToolExecutionState) -> dict[str, Any]:
    chemenzy = dict(state.artifacts.get("chemenzy") or {})
    if isinstance(chemenzy.get("route_package"), dict):
        return dict(chemenzy["route_package"])
    if isinstance(chemenzy.get("result"), dict) and isinstance(chemenzy["result"].get("route_package"), dict):
        return dict(chemenzy["result"]["route_package"])
    smiles_first = dict(state.artifacts.get("smiles_first") or {})
    package_path = ((smiles_first.get("artifacts") or {}).get("hybrid_route_package") or "")
    if package_path and Path(package_path).exists():
        return json.loads(Path(package_path).read_text(encoding="utf-8"))
    target = state.target_input.get("target_smiles")
    frontier = _frontier_from_chemenzy_web(chemenzy) or target
    status = _route_status_from_chemenzy_web(chemenzy)
    return {
        "schema_version": "codex_entry_minimal_route_package.v1",
        "case_id": state.preflight.get("case_id") or state.target_input.get("target_name") or "target",
        "route_status": status,
        "target": {"name": state.target_input.get("target_name", ""), "smiles": target},
        "frontier": {
            "frontier_smiles": frontier,
            "flags": _frontier_flags_from_chemenzy(chemenzy, status),
        },
        "literature_evidence_refs": [],
        "literature_candidates": [],
    }


def _validation_from_artifacts(state: ToolExecutionState, route_package: dict[str, Any]) -> dict[str, Any]:
    chemenzy = dict(state.artifacts.get("chemenzy") or {})
    if isinstance(chemenzy.get("validation"), dict):
        return dict(chemenzy["validation"])
    if isinstance(chemenzy.get("result"), dict) and isinstance(chemenzy["result"].get("validation"), dict):
        return dict(chemenzy["result"]["validation"])
    return {
        "schema_version": "codex_entry_route_validation.v1",
        "case_id": route_package.get("case_id"),
        "accepted": not bool(chemenzy.get("reasons")),
        "route_status": route_package.get("route_status") or "unresolved",
        "reasons": [str(item) for item in chemenzy.get("reasons") or []],
    }


def _raw_route_verifier_from_artifacts(state: ToolExecutionState) -> dict[str, Any]:
    chemenzy = dict(state.artifacts.get("chemenzy") or {})
    if not chemenzy:
        return {}
    result = dict(chemenzy.get("result") or chemenzy)
    if not result.get("routes"):
        return {}
    return verify_chemenzy_raw_routes(
        result,
        target_smiles=str(state.target_input.get("target_smiles") or result.get("target") or ""),
        case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
    )


def _validation_rejected_by_route_verifier(validation: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    reasons = [str(item) for item in validation.get("reasons") or []]
    reasons.extend(str(item) for item in report.get("reasons") or [])
    return {
        **dict(validation),
        "accepted": False,
        "route_status": str(report.get("route_status") or "fake_closed_rejected"),
        "reasons": sorted(set(reasons + ["route_verifier_rejected_raw_routes"])),
        "route_verifier": dict(report),
    }


def _native_route_verified_solved(state: ToolExecutionState) -> bool:
    verifier = dict(state.artifacts.get("route_verifier") or {})
    if verifier:
        return bool(verifier.get("accepted")) and str(verifier.get("route_status") or "") == "solved"
    audit = dict(state.artifacts.get("route_audit") or {})
    if not audit:
        return False
    embedded_verifier = dict(audit.get("route_verifier") or audit.get("raw_route_verifier") or {})
    if embedded_verifier:
        return bool(embedded_verifier.get("accepted")) and str(embedded_verifier.get("route_status") or "") == "solved"
    return (
        str(audit.get("route_status") or "") == "solved"
        and bool(audit.get("stock_audit_passed"))
        and not bool(audit.get("fake_closure_rejected"))
    )


def _skip_result(*, schema_version: str, reasons: list[str], **fields: Any) -> dict[str, Any]:
    result = {
        "schema_version": schema_version,
        "accepted": True,
        "status": "skipped",
        "skipped": True,
        "reasons": [str(item) for item in reasons],
    }
    result.update(fields)
    return result


def _guided_policy_from_payload_or_artifacts(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    explicit = payload.get("chem_enzy_search_policy") or payload.get("search_policy")
    if isinstance(explicit, dict):
        return dict(explicit)
    compiled = _compiled_downstream_from_state(state)
    guided = dict((compiled.get("guided_chemenzy") or {}))
    policies = guided.get("policy_payloads") or []
    for item in policies:
        if isinstance(item, dict):
            return dict(item)
    return {}


def _plugin_only_guided_policy(
    state: ToolExecutionState,
    *,
    plugin_flags: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    budget = {
        "max_reruns": 1,
        "max_iterations": int(payload.get("chem_enzy_iterations") or 50),
        "max_depth": int(payload.get("max_steps") or 15),
        "expansion_topk": int(payload.get("chem_enzy_expansion_topk") or 100),
    }
    return {
        "schema_version": "chem_enzy_search_policy.v1",
        "policy_id": str(payload.get("policy_id") or f"{state.preflight.get('case_id') or state.target_input.get('target_name') or 'case'}_literature_plugin_only"),
        "operator_id": "literature_template_plugin_only",
        "case_id": str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        "evidence_refs": _plugin_evidence_refs(plugin_flags),
        "terminal_blacklist": [],
        "anchor_whitelist": [],
        "preferred_subgoal": {
            "target": {
                "name": state.target_input.get("target_name"),
                "smiles": state.target_input.get("target_smiles"),
            },
            "preferred_subgoals": [],
            "resolved_advisory_anchor_targets": [],
            "blocked_advisory_anchor_targets": [],
        },
        "source_budget": {
            "preferred_reaction_classes": ["literature_template_plugin_replay"],
            "plugin_only_guided_rerun": True,
        },
        "rerun_reason": "compiled_literature_template_plugin_available",
        "budget": budget,
        "mode": "guided",
        "compiler_metadata": {
            "source": "compiled_literature_template_plugin",
            "one_step_row_count": len(plugin_flags.get("one_step_rows") or []),
            "template_card_count": len(plugin_flags.get("template_cards") or []),
            "no_solved_claim": True,
            "requires_verifier": True,
        },
    }


def _plugin_evidence_refs(plugin_flags: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for row in plugin_flags.get("one_step_rows") or []:
        if not isinstance(row, dict):
            continue
        trace = dict(row.get("literature_template_trace") or {})
        refs.extend(str(item) for item in trace.get("evidence_refs") or [])
        template = row.get("template") if isinstance(row.get("template"), dict) else row.get("templates")
        if isinstance(template, dict):
            refs.extend(str(item) for item in template.get("evidence_refs") or [])
    for card in plugin_flags.get("template_cards") or []:
        if isinstance(card, dict):
            refs.extend(str(item) for item in card.get("evidence_refs") or [])
    seen: set[str] = set()
    out: list[str] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        out.append(ref)
    return out


def _compiled_downstream_from_state(state: ToolExecutionState) -> dict[str, Any]:
    compiled = _unwrap_compiled_downstream(state.artifacts.get("compiled_downstream_payload"))
    if compiled:
        return compiled
    compiled = _unwrap_compiled_downstream(state.artifacts.get("compiled_downstream"))
    if compiled:
        return compiled
    open_research = state.artifacts.get("open_structure_research")
    if isinstance(open_research, dict):
        compiled = _unwrap_compiled_downstream(open_research.get("compiled_downstream"))
        if compiled:
            return compiled
    path = state.run_dir / "open_structure_research" / "compiled_downstream_consumables.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}
    return {}


def _unwrap_compiled_downstream(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    payload = dict(value)
    if payload.get("schema_version") == "compiled_downstream_consumables.v1":
        return payload
    refs = dict(payload.get("artifact_refs") or {})
    path_value = str(refs.get("compiled_downstream_consumables") or "")
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(loaded, dict) and loaded.get("schema_version") == "compiled_downstream_consumables.v1":
        return dict(loaded)
    return {}


def _tool_output_dir(state: ToolExecutionState, payload: dict[str, Any], *, default_name: str) -> Path:
    raw = str(payload.get("output_dir") or default_name)
    path = Path(raw)
    if not path.is_absolute():
        path = state.run_dir / path
    resolved = path.resolve()
    run_root = state.run_dir.resolve()
    if not _is_relative_to(resolved, run_root):
        resolved = run_root / default_name
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _pdf_evidence_from_payload_or_artifacts(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("pdf_evidence"), dict):
        return dict(payload.get("pdf_evidence") or {})
    if payload.get("pdf_evidence_path"):
        data = _json_payload_or_path(state, payload.get("pdf_evidence_path"))
        if data:
            return data
    for row in _pdf_evidence_candidates_from_artifacts(state):
        if _pdf_evidence_matches_payload(row, payload):
            return dict(row)
    artifact = state.artifacts.get("literature_pdf_structure_evidence")
    return dict(artifact) if isinstance(artifact, dict) else {}


def _pdf_evidence_candidates_from_artifacts(state: ToolExecutionState) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    history = state.artifacts.get("literature_pdf_structure_evidence_history")
    if isinstance(history, list):
        candidates.extend(dict(row) for row in history if isinstance(row, dict))
    by_source = state.artifacts.get("literature_pdf_structure_evidence_by_source")
    if isinstance(by_source, dict):
        candidates.extend(dict(row) for row in by_source.values() if isinstance(row, dict))
    latest = state.artifacts.get("literature_pdf_structure_evidence")
    if isinstance(latest, dict):
        candidates.append(dict(latest))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        key = _pdf_evidence_key(row) or str(id(row))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _pdf_evidence_matches_payload(evidence: dict[str, Any], payload: dict[str, Any]) -> bool:
    source_ref = str(payload.get("source_ref") or "").strip().lower()
    if source_ref and source_ref == str(evidence.get("source_ref") or "").strip().lower():
        return True
    pdf_path = _normalized_path_key(payload.get("pdf_path"))
    evidence_pdf = _normalized_path_key(evidence.get("source_pdf_path") or evidence.get("pdf_path"))
    if pdf_path and evidence_pdf and pdf_path == evidence_pdf:
        return True
    source_title = _text_key(payload.get("source_title"))
    evidence_title = _text_key(evidence.get("source_title"))
    return bool(source_title and evidence_title and source_title == evidence_title)


def _pdf_evidence_key(evidence: dict[str, Any]) -> str:
    keys = _pdf_evidence_keys(evidence)
    return keys[0] if keys else ""


def _pdf_evidence_keys(evidence: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    source_ref = str(evidence.get("source_ref") or "").strip().lower()
    if source_ref:
        keys.append(f"ref:{source_ref}")
    pdf_key = _normalized_path_key(evidence.get("source_pdf_path") or evidence.get("pdf_path"))
    if pdf_key:
        keys.append(f"pdf:{pdf_key}")
    title = _text_key(evidence.get("source_title"))
    if title:
        keys.append(f"title:{title}")
    return keys


def _normalized_path_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve()).lower()


def _text_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _visual_chain_image_paths(state: ToolExecutionState, payload: dict[str, Any], pdf_evidence: dict[str, Any]) -> list[Path]:
    raw_paths = [str(item) for item in payload.get("image_paths") or [] if str(item).strip()]
    if not raw_paths:
        page_filter = _visual_page_filter(payload)
        for row in pdf_evidence.get("scheme_crops") or []:
            if isinstance(row, dict) and row.get("image_path"):
                raw_paths.append(str(row["image_path"]))
        for row in pdf_evidence.get("rendered_pages") or []:
            if not isinstance(row, dict) or not row.get("image_path"):
                continue
            if page_filter and int(row.get("page_number") or 0) not in page_filter:
                continue
            raw_paths.append(str(row["image_path"]))
    paths: list[Path] = []
    seen: set[str] = set()
    for raw in raw_paths:
        path = _input_path(state, raw).resolve()
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        paths.append(path)
    max_images = _visual_max_images(payload)
    if max_images > 0:
        paths = paths[:max_images]
    return _prepared_visual_image_paths(state, paths, payload)


def _visual_page_filter(payload: dict[str, Any]) -> set[int]:
    values = payload.get("page_numbers") or payload.get("visual_page_numbers") or []
    out: set[int] = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            out.add(number)
    return out


def _visual_max_images(payload: dict[str, Any]) -> int:
    raw = payload.get("max_images")
    if raw is None:
        raw = os.environ.get("AUTOPLANNER_VISUAL_MAX_IMAGES", "6")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 6


def _prepared_visual_image_paths(state: ToolExecutionState, paths: list[Path], payload: dict[str, Any]) -> list[Path]:
    if not _visual_image_compression_enabled(payload):
        return paths
    try:
        from PIL import Image
    except Exception:
        return paths
    out_dir = state.run_dir / "_visual_prepared_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_side = _visual_max_side_px(payload)
    quality = _visual_jpeg_quality(payload)
    prepared: list[Path] = []
    for idx, path in enumerate(paths, start=1):
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                width, height = image.size
                scale = min(1.0, float(max_side) / float(max(width, height))) if max_side > 0 else 1.0
                if scale < 1.0:
                    image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
                target = out_dir / f"{path.stem}_vinput_{idx}.jpg"
                image.save(target, format="JPEG", quality=quality, optimize=True)
                prepared.append(target.resolve())
        except Exception:
            prepared.append(path)
    return prepared


def _visual_image_compression_enabled(payload: dict[str, Any]) -> bool:
    raw = payload.get("compress_images")
    if raw is None:
        raw = os.environ.get("AUTOPLANNER_VISUAL_COMPRESS_IMAGES", "1")
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "off", "no", "disabled"}
    return bool(raw)


def _visual_max_side_px(payload: dict[str, Any]) -> int:
    raw = payload.get("visual_max_side_px") or os.environ.get("AUTOPLANNER_VISUAL_MAX_SIDE_PX", "1400")
    try:
        return max(256, int(raw))
    except (TypeError, ValueError):
        return 1400


def _visual_jpeg_quality(payload: dict[str, Any]) -> int:
    raw = payload.get("visual_jpeg_quality") or os.environ.get("AUTOPLANNER_VISUAL_JPEG_QUALITY", "70")
    try:
        return min(95, max(35, int(raw)))
    except (TypeError, ValueError):
        return 70


def _input_path(state: ToolExecutionState, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    run_candidate = (state.run_dir / path).resolve()
    if run_candidate.exists():
        return run_candidate
    return (ROOT / path).resolve()


def _json_payload_or_path(state: ToolExecutionState, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    path = _input_path(state, value)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _validation_payload_from_payload_or_artifacts(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    explicit = payload.get("validation") or payload.get("visual_structure_chain_validation")
    if isinstance(explicit, dict):
        return dict(explicit)
    path_value = payload.get("validation_path") or payload.get("visual_structure_chain_validation_path")
    if path_value:
        data = _json_payload_or_path(state, path_value)
        if data:
            return data
    artifact = state.artifacts.get("visual_structure_chain_validation")
    if isinstance(artifact, dict):
        return dict(artifact)
    for path in [
        state.run_dir / "visual_structure_chain_validation.json",
        state.run_dir / "literature_intermediate_chain_validation" / "visual_structure_chain_validation.json",
    ]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return dict(data)
    return {}


def _literature_tool_target_smiles(state: ToolExecutionState, payload: dict[str, Any]) -> str:
    if "target_smiles" in payload:
        return str(payload.get("target_smiles") or "")
    if bool(payload.get("allow_partial_chain_without_target_match")) or payload.get("target_match_required") is False:
        return ""
    return str(state.target_input.get("target_smiles") or "")


def _candidate_chain_payload_from_payload_or_artifacts(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    explicit = payload.get("candidate_chain")
    if isinstance(explicit, dict):
        return dict(explicit)
    path_value = payload.get("candidate_chain_path")
    if path_value:
        data = _json_payload_or_path(state, path_value)
        if data:
            return data
    candidates: list[dict[str, Any]] = []
    candidates.extend(
        dict(item)
        for item in state.artifacts.get("visual_structure_candidate_chain_history") or []
        if isinstance(item, dict)
    )
    artifact = state.artifacts.get("visual_structure_candidate_chain")
    if isinstance(artifact, dict):
        candidates.append(dict(artifact))
    path_value = state.artifacts.get("visual_structure_candidate_chain_path")
    if path_value:
        data = _json_payload_or_path(state, path_value)
        if data:
            candidates.append(data)
    if candidates:
        candidates.sort(key=_visual_candidate_quality_score, reverse=True)
        return dict(candidates[0])
    return {}


def _record_visual_chain_result(state: ToolExecutionState, result: dict[str, Any]) -> None:
    history = state.artifacts.setdefault("visual_literature_chain_extraction_history", [])
    if isinstance(history, list):
        history.append(dict(result))
    candidate = _candidate_chain_from_visual_result(state, result)
    if isinstance(candidate, dict) and candidate:
        candidate_history = state.artifacts.setdefault("visual_structure_candidate_chain_history", [])
        if isinstance(candidate_history, list):
            candidate_history.append(dict(candidate))


def _best_visual_candidate_from_history(state: ToolExecutionState) -> dict[str, Any]:
    candidates = [
        dict(item)
        for item in state.artifacts.get("visual_structure_candidate_chain_history") or []
        if isinstance(item, dict)
    ]
    if not candidates:
        return {}
    candidates.sort(key=_visual_candidate_quality_score, reverse=True)
    return dict(candidates[0])


def _candidate_chain_from_visual_result(state: ToolExecutionState, result: dict[str, Any]) -> dict[str, Any]:
    path_value = str(result.get("candidate_chain_path") or "").strip()
    if path_value:
        loaded = _json_payload_or_path(state, path_value)
        if isinstance(loaded, dict) and loaded:
            return dict(loaded)
    candidate = result.get("candidate_chain")
    if isinstance(candidate, dict) and candidate:
        return dict(candidate)
    parsed = result.get("parsed_output")
    if not isinstance(parsed, dict) or not parsed:
        return {}
    enriched = dict(parsed)
    if result.get("source_ref") and not enriched.get("source_ref"):
        enriched["source_ref"] = str(result.get("source_ref") or "")
    if result.get("source_title") and not enriched.get("source_title"):
        enriched["source_title"] = str(result.get("source_title") or "")
    if not enriched.get("evidence_refs"):
        refs = [
            f"current_image:{item}"
            for item in result.get("image_paths") or []
            if str(item or "").strip()
        ]
        enriched["evidence_refs"] = _dedupe_texts(refs)
    if result.get("candidate_chain_path") and not enriched.get("candidate_chain_path"):
        enriched["candidate_chain_path"] = str(result.get("candidate_chain_path") or "")
    return enriched


def _visual_candidate_quality_score(candidate: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    steps = _visual_candidate_steps(candidate)
    valid_steps = 0
    labels: set[str] = set()
    has_terminal_11 = 0
    has_target_product = 0
    has_source_ref = 1 if str(candidate.get("source_ref") or _doi_source_ref(candidate) or "").strip() else 0
    evidence_count = len([item for item in candidate.get("evidence_refs") or [] if str(item or "").strip()])
    for step in steps:
        product = str(step.get("product_smiles") or "")
        reactants = [str(item) for item in step.get("reactant_smiles") or [] if str(item or "").strip()]
        if product and reactants and _valid_smiles(product) and any(_valid_smiles(item) for item in reactants):
            valid_steps += 1
        if str(step.get("source_ref") or "").strip():
            has_source_ref = 1
        evidence_count += len([item for item in step.get("evidence_refs") or [] if str(item or "").strip()])
        product_label = str(step.get("product_label") or "").strip().lower()
        if product_label:
            labels.add(product_label)
            if product_label == "bufotalin":
                has_target_product = 1
        for label in step.get("reactant_labels") or []:
            clean = str(label or "").strip().lower()
            if clean:
                labels.add(clean)
                if clean == "11":
                    has_terminal_11 = 1
    return (valid_steps, has_source_ref, min(evidence_count, 25), len(labels), has_terminal_11, has_target_product)


def _condition_repair_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("repairs"), list):
            return [dict(item) for item in value.get("repairs") or [] if isinstance(item, dict)]
        return [dict(row) for row in value.values() if isinstance(row, dict)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _apply_condition_repair_to_step(step: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
    condition = repair.get("condition_candidate")
    if not isinstance(condition, dict):
        condition = {
            "schema_version": "condition_candidate.v1",
            "source_type": "exact",
            "condition_status": "evidence_backed",
            "reagent": str(repair.get("reagent") or ""),
            "solvent": str(repair.get("solvent") or ""),
            "temperature": str(repair.get("temperature") or ""),
            "duration": str(repair.get("duration") or repair.get("time") or ""),
            "reported_yield": str(repair.get("reported_yield") or repair.get("yield") or ""),
            "source_grounding": str(repair.get("source_grounding") or repair.get("source_excerpt") or ""),
        }
        condition = {key: val for key, val in condition.items() if val not in ("", [])}
    repaired = dict(step)
    repaired["condition_candidate"] = condition
    if repair.get("source_locator"):
        repaired["source_locator"] = str(repair.get("source_locator") or "")
    if repair.get("source_excerpt"):
        repaired["source_excerpt"] = str(repair.get("source_excerpt") or "")
    evidence_refs = [str(item) for item in repair.get("evidence_refs") or [] if str(item).strip()]
    if evidence_refs:
        repaired["evidence_refs"] = evidence_refs
    derivation = dict(repaired.get("structure_derivation") or {})
    if repair.get("source_locator"):
        derivation["source_locator"] = str(repair.get("source_locator") or "")
    derivation.setdefault("basis", "current_pdf_image_to_smiles")
    derivation.setdefault("confidence", str(repair.get("confidence") or repaired.get("confidence") or "low"))
    checks = [str(item) for item in derivation.get("tool_checks") or [] if str(item).strip()]
    if "condition fields repaired from source text manifest" not in checks:
        checks.append("condition fields repaired from source text manifest")
    derivation["tool_checks"] = checks
    repaired["structure_derivation"] = derivation
    return repaired


def _step_smiles_signature(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step_id": str(step.get("step_id") or ""),
            "product_smiles": str(step.get("product_smiles") or ""),
            "reactant_smiles": [str(item) for item in step.get("reactant_smiles") or []],
            "main_reactant_smiles": str(step.get("main_reactant_smiles") or ""),
        }
        for step in steps
    ]


def _compiled_payload_from_payload_or_artifacts(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    explicit = payload.get("compiled_downstream")
    if isinstance(explicit, dict):
        return dict(explicit)
    path_value = payload.get("compiled_downstream_path")
    if path_value:
        data = _json_payload_or_path(state, path_value)
        if data:
            return data
    return _compiled_downstream_from_state(state)


def _source_detail_steps_from_payload_or_artifacts(state: ToolExecutionState, payload: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = payload.get("source_detail_steps") or payload.get("source_detail_route_steps")
    if isinstance(explicit, list):
        return [dict(item) for item in explicit if isinstance(item, dict)]
    resolution = payload.get("source_detail_resolution")
    if isinstance(resolution, dict):
        return [dict(item) for item in resolution.get("source_detail_route_steps") or [] if isinstance(item, dict)]
    path_value = payload.get("source_detail_resolution_path") or payload.get("source_detail_steps_path")
    if path_value:
        data = _json_payload_or_path(state, path_value)
        if data.get("source_detail_route_steps"):
            return [dict(item) for item in data.get("source_detail_route_steps") or [] if isinstance(item, dict)]
        if data.get("downstream_patch"):
            return [
                dict(item)
                for item in (data.get("downstream_patch") or {}).get("source_detail_route_steps") or []
                if isinstance(item, dict)
            ]
    artifact = state.artifacts.get("source_detail_resolution")
    if isinstance(artifact, dict):
        return [dict(item) for item in artifact.get("source_detail_route_steps") or [] if isinstance(item, dict)]
    visual_steps = _source_detail_steps_from_visual_candidate(state, payload)
    if visual_steps:
        return visual_steps
    return []


def _source_detail_steps_from_visual_candidate(state: ToolExecutionState, payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = _candidate_chain_payload_from_payload_or_artifacts(state, payload)
    if not candidate:
        visual = state.artifacts.get("visual_literature_chain_extraction")
        if isinstance(visual, dict):
            candidate = _candidate_chain_from_visual_result(state, visual)
    if not candidate:
        return []
    visual_artifact = state.artifacts.get("visual_literature_chain_extraction")
    visual_artifact = dict(visual_artifact) if isinstance(visual_artifact, dict) else {}
    source_ref = str(
        payload.get("source_ref")
        or candidate.get("source_ref")
        or visual_artifact.get("source_ref")
        or _doi_source_ref(candidate)
        or ""
    )
    source_title = str(payload.get("source_title") or candidate.get("source_title") or visual_artifact.get("source_title") or "")
    global_evidence = _dedupe_texts(
        [
            *[
                str(item)
                for item in candidate.get("evidence_refs") or []
                if str(item or "").strip()
            ],
            *[
                f"current_image:{item}"
                for item in visual_artifact.get("image_paths") or []
                if str(item or "").strip()
            ],
            str(candidate.get("source_locator") or ""),
            str(visual_artifact.get("candidate_chain_path") or ""),
        ]
    )
    out: list[dict[str, Any]] = []
    for idx, step in enumerate(_visual_candidate_steps(candidate), start=1):
        if not isinstance(step, dict):
            continue
        product = str(step.get("product_smiles") or "")
        raw_reactants = step.get("reactant_smiles") or []
        if isinstance(raw_reactants, str):
            raw_reactants = [raw_reactants]
        reactants = [str(item) for item in raw_reactants if str(item or "").strip()]
        if not reactants and str(step.get("main_reactant_smiles") or "").strip():
            reactants = [str(step.get("main_reactant_smiles") or "").strip()]
        step_ref = str(step.get("source_ref") or source_ref or candidate.get("source_ref") or "")
        raw_step_evidence = step.get("evidence_refs") or []
        if isinstance(raw_step_evidence, str):
            raw_step_evidence = [raw_step_evidence]
        evidence_refs = _dedupe_texts(
            [
                *[str(item) for item in raw_step_evidence if str(item or "").strip()],
                *global_evidence,
                str(step.get("source_locator") or ""),
            ]
        )
        if not product or not reactants or not step_ref or not evidence_refs:
            continue
        raw_condition = step.get("condition_candidate") or step.get("condition") or step.get("conditions") or {}
        if isinstance(raw_condition, dict):
            condition = dict(raw_condition)
        elif str(raw_condition or "").strip():
            condition = {"reagent": str(raw_condition or "").strip()}
        else:
            condition = {}
        if not condition.get("reagent") and condition.get("reagents"):
            condition["reagent"] = str(condition.get("reagents") or "")
        if not condition.get("reported_yield") and condition.get("yield"):
            condition["reported_yield"] = str(condition.get("yield") or "")
        condition.setdefault("schema_version", "condition_candidate.v1")
        condition.setdefault("source_type", "exact")
        condition.setdefault("condition_status", "evidence_backed")
        condition.setdefault("source_grounding", str(step.get("source_locator") or "current PDF visual extraction"))
        condition.setdefault("step_id", str(step.get("step_id") or f"visual_source_detail_step_{idx}"))
        if "evidence_refs" not in condition and evidence_refs:
            condition["evidence_refs"] = list(evidence_refs)
        derivation = dict(step.get("structure_derivation") or {})
        derivation.setdefault("basis", "current_pdf_image_to_smiles")
        derivation.setdefault("confidence", str(step.get("confidence") or candidate.get("confidence") or "low"))
        if step.get("source_locator"):
            derivation.setdefault("source_locator", str(step.get("source_locator") or ""))
        checks = [str(item) for item in derivation.get("tool_checks") or [] if str(item or "").strip()]
        if "visual candidate promoted to draft source-detail step" not in checks:
            checks.append("visual candidate promoted to draft source-detail step")
        derivation["tool_checks"] = checks
        not_exact_visual = _visual_step_is_exploratory(step, derivation)
        out.append(
            {
                "schema_version": "source_detail_route_step.v1",
                "step_id": str(step.get("step_id") or f"visual_source_detail_step_{idx}"),
                "segment_id": str(step.get("segment_id") or candidate.get("case_id") or "visual_literature_chain"),
                "source_ref": step_ref,
                "source_title": str(step.get("source_title") or source_title),
                "evidence_refs": evidence_refs,
                "product_name": str(step.get("product_label") or ""),
                "reactant_names": [str(item) for item in step.get("reactant_labels") or [] if str(item or "").strip()],
                "product_smiles": product,
                "reactant_smiles": reactants,
                "relation_type": "visual_connectivity_approximation" if not_exact_visual else "exact",
                "condition_candidate": condition,
                "applicability": {
                    "status": "hypothesis_only" if not_exact_visual else "passed",
                    "product_reconstruction_passed": not not_exact_visual,
                    "reconstructed_product_smiles": product,
                },
                "provenance": "visual_candidate_chain_current_pdf",
                "source_excerpt": str(step.get("source_excerpt") or step.get("source_locator") or ""),
                "structure_derivation": derivation,
                "validation_status": "draft_rdkit_valid_visual_approximation" if not_exact_visual else "draft_validated_by_rdkit_chain",
                "curation_status": "visual_candidate_for_exploratory_template_hint" if not_exact_visual else "visual_candidate_promoted_for_exact_row_compile",
                "not_exact_literature_segment": bool(not_exact_visual),
                "allowed_use": "exploratory_template_and_guided_hint_only" if not_exact_visual else "exact_candidate",
                "full_text_content_stored": False,
                "procedure_text_stored": False,
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    return out


def _visual_step_is_exploratory(step: dict[str, Any], derivation: dict[str, Any] | None = None) -> bool:
    derivation = dict(derivation or step.get("structure_derivation") or {})
    text = " ".join(
        [
            str(step.get("allowed_use") or ""),
            str(step.get("stereochemistry_status") or ""),
            str(derivation.get("basis") or ""),
            str(derivation.get("allowed_use") or ""),
            str(derivation.get("stereochemistry_status") or ""),
            " ".join(str(item) for item in step.get("risk_flags") or []),
            " ".join(str(item) for item in derivation.get("risk_flags") or []),
        ]
    ).lower()
    return bool(
        step.get("not_exact_literature_segment")
        or derivation.get("not_exact_literature_segment")
        or derivation.get("approximate_structure")
        or "exploratory" in text
        or "achiral" in text
        or "connectivity" in text
        or "unspecified" in text
        or "partial" in text
    )


def _visual_candidate_steps(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    steps = candidate.get("steps")
    if isinstance(steps, list) and steps:
        return [dict(item) for item in steps if isinstance(item, dict)]
    candidate_steps = candidate.get("candidate_steps")
    if isinstance(candidate_steps, list) and candidate_steps:
        return [dict(item) for item in candidate_steps if isinstance(item, dict)]
    chain = candidate.get("candidate_chain")
    if not isinstance(chain, list):
        chain = candidate.get("chain")
        if isinstance(chain, list) and any(isinstance(item, dict) and item.get("product_smiles") for item in chain):
            out: list[dict[str, Any]] = []
            for idx, item in enumerate(chain, start=1):
                if not isinstance(item, dict):
                    continue
                product_smiles = str(item.get("product_smiles") or "").strip()
                reactant_smiles = str(
                    item.get("reactant_smiles")
                    or item.get("main_reactant_smiles")
                    or item.get("precursor_smiles")
                    or ""
                ).strip()
                if not product_smiles or not reactant_smiles:
                    continue
                label = str(item.get("product_label") or item.get("label") or "").strip()
                reactant_label = str(item.get("reactant_label") or item.get("precursor_label") or "").strip()
                reactant_labels = [str(value) for value in item.get("reactant_labels") or [] if str(value).strip()]
                if reactant_label and not reactant_labels:
                    reactant_labels = [reactant_label]
                out.append(
                    {
                        "schema_version": "visual_structure_candidate_step.v1",
                        "step_id": str(item.get("step_id") or f"visual_step_{idx}_{_safe_file_stem(label)}"),
                        "segment_id": str(item.get("segment_id") or candidate.get("case_id") or "visual_literature_chain"),
                        "product_label": label,
                        "product_smiles": product_smiles,
                        "reactant_labels": reactant_labels,
                        "reactant_smiles": [reactant_smiles],
                        "main_reactant_smiles": reactant_smiles,
                        "source_ref": str(item.get("source_ref") or candidate.get("source_ref") or _doi_source_ref(candidate)),
                        "source_title": str(item.get("source_title") or candidate.get("source_title") or ""),
                        "evidence_refs": [str(value) for value in item.get("evidence_refs") or candidate.get("evidence_refs") or []],
                        "source_locator": str(item.get("source_locator") or item.get("source_location") or candidate.get("source_locator") or ""),
                        "condition_candidate": (
                            item.get("condition_candidate")
                            or item.get("condition")
                            or item.get("conditions")
                            or item.get("forward_conditions")
                            or {}
                        ),
                        "source_excerpt": str(
                            item.get("source_excerpt")
                            or item.get("source_locator")
                            or item.get("source_location")
                            or candidate.get("source_excerpt")
                            or ""
                        ),
                        "confidence": str(item.get("confidence") or candidate.get("confidence") or "low"),
                    }
                )
            return out
        return []
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(chain, start=1):
        if not isinstance(item, dict):
            continue
        precursor_smiles = str(item.get("precursor_smiles") or item.get("reactant_smiles") or "").strip()
        if not precursor_smiles:
            continue
        label = str(item.get("label") or item.get("product_label") or item.get("target_label") or "").strip()
        precursor_label = str(item.get("precursor_label") or item.get("reactant_label") or "").strip()
        out.append(
            {
                "schema_version": "visual_structure_candidate_step.v1",
                "step_id": str(item.get("step_id") or f"visual_step_{idx}_{_safe_file_stem(label)}"),
                "segment_id": str(item.get("segment_id") or candidate.get("case_id") or "visual_literature_chain"),
                "product_label": label,
                "product_smiles": str(item.get("smiles") or item.get("product_smiles") or "").strip(),
                "reactant_labels": [precursor_label] if precursor_label else [],
                "reactant_smiles": [precursor_smiles],
                "main_reactant_smiles": precursor_smiles,
                "source_ref": str(item.get("source_ref") or candidate.get("source_ref") or _doi_source_ref(candidate)),
                "source_title": str(item.get("source_title") or candidate.get("source_title") or ""),
                "evidence_refs": [str(value) for value in item.get("evidence_refs") or candidate.get("evidence_refs") or []],
                "source_locator": str(item.get("source_locator") or candidate.get("source_locator") or ""),
                "condition_candidate": item.get("condition_candidate") or item.get("condition") or item.get("conditions") or {},
                "source_excerpt": str(item.get("source_excerpt") or item.get("source_locator") or candidate.get("source_excerpt") or ""),
                "confidence": str(item.get("confidence") or candidate.get("confidence") or "low"),
            }
        )
    return out


def _doi_source_ref(candidate: dict[str, Any]) -> str:
    doi = str(candidate.get("doi") or "").strip()
    if not doi:
        return ""
    return doi if doi.startswith("doi:") else f"doi:{doi}"


def _route_expansion_child_targets(
    *,
    state: ToolExecutionState,
    payload: dict[str, Any],
    compiled: dict[str, Any],
) -> list[dict[str, Any]]:
    explicit = payload.get("child_targets") or payload.get("subgoal_targets")
    rows: list[dict[str, Any]] = []
    for item in explicit or []:
        if not isinstance(item, dict):
            continue
        smiles = str(item.get("smiles") or item.get("target_smiles") or "").strip()
        if _valid_smiles(smiles):
            rows.append({
                "schema_version": "route_expansion_child_target.v1",
                "name": str(item.get("name") or item.get("target_name") or f"subgoal_{len(rows) + 1}"),
                "smiles": smiles,
                "source": str(item.get("source") or "explicit_payload"),
                "explicit_payload": True,
                "hypothesis_only_not_solved": bool(item.get("hypothesis_only_not_solved")),
                "recursive_hypothesis_task_id": str(item.get("recursive_hypothesis_task_id") or ""),
                "recursive_depth": int(item.get("recursive_depth") or 0),
                "parent_smiles": str(item.get("parent_smiles") or ""),
                "parent_candidate_id": str(item.get("parent_candidate_id") or ""),
                "template_id": str(item.get("template_id") or ""),
                "application_id": str(item.get("application_id") or ""),
                "policy": dict(item.get("chem_enzy_search_policy") or item.get("policy") or {}),
                "exact_target_override": bool(
                    item.get("exact_target_override")
                    or item.get("strict_exact_target")
                    or item.get("target_equivalence_audit_required")
                ),
                "target_equivalence_audit_required": bool(
                    item.get("target_equivalence_audit_required")
                    or item.get("exact_target_override")
                    or item.get("strict_exact_target")
                ),
            })
    feedback = _route_failure_feedback_from_payload_or_artifacts(state, payload)
    for item in feedback.get("frontier_research_targets") or []:
        if not isinstance(item, dict):
            continue
        smiles = str(item.get("canonical_smiles") or item.get("smiles") or "").strip()
        if _valid_smiles(smiles):
            rows.append({
                "schema_version": "route_expansion_child_target.v1",
                "name": str(item.get("frontier_role") or f"feedback_subgoal_{len(rows) + 1}"),
                "smiles": smiles,
                "source": "route_failure_feedback",
                "frontier_role": str(item.get("frontier_role") or ""),
                "required_action": str(item.get("required_action") or ""),
                "reason": str(item.get("reason") or ""),
                "policy": {},
            })
    route_expansion = dict((compiled.get("route_expansion") or {}))
    rows.extend(_route_expansion_task_child_targets(route_expansion, exact_only=True))
    for item in route_expansion.get("child_targets") or []:
        if not isinstance(item, dict):
            continue
        smiles = str(item.get("smiles") or item.get("target_smiles") or "").strip()
        if not _valid_smiles(smiles):
            continue
        rows.append({
            "schema_version": "route_expansion_child_target.v1",
            "name": str(item.get("name") or item.get("child_target_id") or f"compiled_child_target_{len(rows) + 1}"),
            "smiles": smiles,
            "source": str(item.get("source") or "compiled_route_expansion_child_target"),
            "child_target_id": str(item.get("child_target_id") or ""),
            "source_template_id": str(item.get("source_template_id") or ""),
            "source_ref": str(item.get("source_ref") or ""),
            "parent_product_smiles": str(item.get("parent_product_smiles") or ""),
            "evidence_refs": [str(ref) for ref in item.get("evidence_refs") or []],
            "policy": dict(item.get("chem_enzy_search_policy") or item.get("policy") or {}),
            "max_depth": item.get("max_depth"),
            "max_iterations": item.get("max_iterations"),
            "expansion_topk": item.get("expansion_topk"),
            "no_solved_claim": True,
            "production_write_blocked": True,
            "exact_target_override": bool(
                item.get("exact_target_override")
                or item.get("strict_exact_target")
                or item.get("target_equivalence_audit_required")
            ),
            "target_equivalence_audit_required": bool(
                item.get("target_equivalence_audit_required")
                or item.get("exact_target_override")
                or item.get("strict_exact_target")
            ),
        })
    rows.extend(_route_expansion_task_child_targets(route_expansion, exact_only=False))
    return _dedupe_child_targets(_prioritize_route_expansion_child_targets(rows))


def _route_expansion_task_child_targets(route_expansion: dict[str, Any], *, exact_only: bool) -> list[dict[str, Any]]:
    policies_by_id = {
        str(policy.get("policy_id") or ""): dict(policy)
        for policy in route_expansion.get("policy_payloads") or []
        if isinstance(policy, dict)
    }
    rows: list[dict[str, Any]] = []
    for task in route_expansion.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        policy = policies_by_id.get(str(task.get("policy_id") or ""), {})
        exact_smiles = str(task.get("exact_target_smiles") or task.get("exact_terminal_smiles") or "").strip()
        exact_override = bool(
            task.get("exact_target_override")
            or task.get("strict_exact_target")
            or task.get("target_equivalence_audit_required")
            or exact_smiles
        )
        candidates: list[tuple[str, str, bool]] = []
        if exact_override and _valid_smiles(exact_smiles):
            candidates.append((exact_smiles, "route_expansion_exact_target_override", True))
        if exact_only:
            pass
        else:
            candidates.extend(
                [
                    (str(task.get("frontier_smiles") or ""), "route_expansion_frontier", False),
                    *[
                        (str(item or ""), "route_expansion_preferred_subgoal", False)
                        for item in task.get("preferred_subgoals") or []
                    ],
                ]
            )
        for smiles, source, is_exact in candidates:
            if not _valid_smiles(smiles):
                continue
            rows.append(
                {
                    "schema_version": "route_expansion_child_target.v1",
                    "name": str(task.get("task_id") or f"route_expansion_subgoal_{len(rows) + 1}"),
                    "smiles": smiles,
                    "source": source,
                    "task_id": str(task.get("task_id") or ""),
                    "policy": policy,
                    "max_depth": task.get("max_depth"),
                    "max_iterations": task.get("max_iterations"),
                    "expansion_topk": task.get("expansion_topk"),
                    "exact_target_override": bool(is_exact or exact_override),
                    "target_equivalence_audit_required": bool(
                        is_exact or exact_override or task.get("target_equivalence_audit_required")
                    ),
                    "no_solved_claim": bool(task.get("no_solved_claim", True)),
                    "production_write_blocked": bool(task.get("production_write_blocked", True)),
                }
            )
    return rows


def _dedupe_child_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("smiles") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _dedupe_subgoal_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        subgoal = row.get("subgoal") if isinstance(row.get("subgoal"), dict) else {}
        key = str(subgoal.get("smiles") or row.get("request_path") or row.get("subgoal_index") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(dict(row))
    return out


def _prioritize_route_expansion_child_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = [(idx, dict(row)) for idx, row in enumerate(rows)]
    indexed.sort(key=lambda item: (_child_target_priority(item[1]), item[0]))
    return [row for _, row in indexed]


def _child_target_priority(row: dict[str, Any]) -> int:
    source = str(row.get("source") or "")
    text = " ".join(
        str(row.get(key) or "")
        for key in ("name", "child_target_id", "source_template_id", "task_id")
    ).lower()
    if row.get("explicit_payload") or source == "explicit_payload":
        return 0
    if "from_11" in text or "reactant_11" in text or text.endswith("_11"):
        return 1
    if source == "route_failure_feedback":
        return 3
    if "source_detail_exact_step" in text or "source_detail" in source:
        return 5
    return 10


def _exact_target_audit_required(target: dict[str, Any]) -> bool:
    return bool(
        target.get("exact_target_override")
        or target.get("strict_exact_target")
        or target.get("target_equivalence_audit_required")
        or str(target.get("source") or "") == "route_expansion_exact_target_override"
    )


def _valid_smiles(value: str) -> bool:
    try:
        from rdkit import Chem
    except Exception:
        return bool(str(value).strip())
    return Chem.MolFromSmiles(str(value or "")) is not None


def _safe_file_stem(value: str) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").lower()).strip("_")
    return text[:80] or "subgoal"


def _merge_route_failure_feedback_policy(
    state: ToolExecutionState,
    policy: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    feedback = _route_failure_feedback_from_payload_or_artifacts(state, payload)
    if not feedback or not feedback.get("accepted"):
        return policy
    merged = dict(policy or {})
    patch = dict(feedback.get("next_guided_policy_patch") or {})
    blacklist = [str(item) for item in patch.get("terminal_blacklist") or [] if item]
    if blacklist:
        merged["terminal_blacklist"] = sorted(set([*(merged.get("terminal_blacklist") or []), *blacklist]))
    preferred = dict(merged.get("preferred_subgoal") or {})
    subgoals = [str(item) for item in preferred.get("preferred_subgoals") or []]
    for item in patch.get("preferred_subgoals") or []:
        text = str(item or "")
        if text and text not in subgoals:
            subgoals.append(text)
    if subgoals:
        preferred["preferred_subgoals"] = subgoals
        merged["preferred_subgoal"] = preferred
    source_budget = dict(merged.get("source_budget") or {})
    patch_budget = dict(patch.get("source_budget") or {})
    for key, values in patch_budget.items():
        existing = [str(item) for item in source_budget.get(key) or []] if isinstance(source_budget.get(key), list) else []
        for item in values or []:
            text = str(item or "")
            if text and text not in existing:
                existing.append(text)
        if existing:
            source_budget[key] = existing
    if source_budget:
        merged["source_budget"] = source_budget
    merged["route_failure_feedback"] = {
        "enabled": True,
        "terminal_blacklist_count": len(blacklist),
        "frontier_research_target_count": len(feedback.get("frontier_research_targets") or []),
        "source_route_status": feedback.get("source_route_status"),
        "source_reasons": list(feedback.get("source_reasons") or []),
    }
    return merged


def _merge_analogical_retrosynthesis_policy(
    state: ToolExecutionState,
    policy: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    report = _analogical_retrosynthesis_from_payload_or_artifacts(state, payload)
    if not report or not report.get("accepted"):
        return policy
    patch = dict(report.get("search_policy_patch") or {})
    if not patch.get("enabled"):
        return policy
    merged = dict(policy or {})
    source_budget = dict(merged.get("source_budget") or {})
    for key in ("preferred_reaction_classes", "active_failure_modes"):
        existing = [str(item) for item in source_budget.get(key) or []] if isinstance(source_budget.get(key), list) else []
        for item in patch.get(key) or []:
            text = str(item or "")
            if text and text not in existing:
                existing.append(text)
        if existing:
            source_budget[key] = existing
    source_budget["analogical_inspiration_enabled"] = True
    source_budget["require_target_core_retention"] = bool(patch.get("require_target_core_retention", True))
    source_budget["max_unexplained_heavy_atom_delta"] = int(patch.get("max_unexplained_heavy_atom_delta") or 20)
    source_budget["analogical_hypothesis_count"] = int(report.get("hypothesis_count") or 0)
    merged["source_budget"] = source_budget

    preferred = dict(merged.get("preferred_subgoal") or {})
    preferred["analogical_retrosynthesis_hypotheses"] = [
        {
            "hypothesis_id": str(item.get("hypothesis_id") or ""),
            "inspiration_type": str(item.get("inspiration_type") or ""),
            "reaction_family": str(item.get("reaction_family") or ""),
            "target_side_attempt": dict(item.get("target_side_attempt") or {}),
            "required_verification": [str(value) for value in item.get("required_verification") or []],
        }
        for item in (report.get("hypotheses") or [])[:6]
        if isinstance(item, dict)
    ]
    merged["preferred_subgoal"] = preferred
    metadata = dict(merged.get("compiler_metadata") or {})
    metadata["analogical_retrosynthesis"] = {
        "enabled": True,
        "schema_version": report.get("schema_version"),
        "hypothesis_count": int(report.get("hypothesis_count") or 0),
        "source_row_count": int(report.get("source_row_count") or 0),
        "mode": report.get("mode"),
        "no_solved_claim": True,
        "requires_verifier": True,
    }
    merged["compiler_metadata"] = metadata
    merged["analogical_retrosynthesis"] = {
        "enabled": True,
        "hypothesis_count": int(report.get("hypothesis_count") or 0),
        "policy_patch_schema_version": patch.get("schema_version"),
        "not_raw_reaction_injection": True,
        "no_solved_claim": True,
    }
    return merged


def _analogical_retrosynthesis_from_payload_or_artifacts(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    explicit = payload.get("analogical_retrosynthesis_hypotheses")
    if isinstance(explicit, dict):
        return dict(explicit)
    path_value = payload.get("analogical_retrosynthesis_hypotheses_path")
    if path_value:
        data = _json_payload_or_path(state, path_value)
        if data:
            return data
    artifact = state.artifacts.get("analogical_retrosynthesis_hypotheses")
    if isinstance(artifact, dict):
        return dict(artifact)
    path = state.run_dir / "analogical_retrosynthesis_hypotheses.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return dict(data) if isinstance(data, dict) else {}
    return {}


def _literature_template_plugin_runtime_diagnostics(result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    plugin = dict(request.get("literature_template_plugin") or {})
    stats = dict(((result.get("raw_backend_metadata") or {}).get("literature_template_plugin") or {}))
    row_count = len(plugin.get("one_step_rows") or [])
    enabled = bool(plugin.get("enabled"))
    reasons: list[str] = []
    calls = int(stats.get("calls") or 0)
    added = int(stats.get("added_candidates") or 0)
    if enabled and row_count and stats and calls == 0:
        reasons.append("literature_template_plugin_not_invoked")
    elif enabled and row_count and stats and calls > 0 and added == 0:
        reasons.append("literature_template_plugin_no_candidates_added")
    return {
        "schema_version": "literature_template_plugin_runtime_diagnostics.v1",
        "enabled_in_request": enabled,
        "request_one_step_row_count": row_count,
        "backend_stats_present": bool(stats),
        "calls": calls,
        "candidate_templates": int(stats.get("candidate_templates") or 0),
        "instantiated_candidates": int(stats.get("instantiated_candidates") or 0),
        "added_candidates": added,
        "reasons": reasons,
    }


def _route_failure_feedback_from_payload_or_artifacts(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    explicit = payload.get("route_failure_feedback")
    if isinstance(explicit, dict):
        return dict(explicit)
    path_value = payload.get("route_failure_feedback_path")
    if path_value:
        path = Path(str(path_value))
        if not path.is_absolute():
            path = state.run_dir / path
        resolved = path.resolve()
        if _is_relative_to(resolved, state.run_dir.resolve()) and resolved.exists():
            try:
                data = json.loads(resolved.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
            return dict(data) if isinstance(data, dict) else {}
        return {}
    feedback = state.artifacts.get("route_failure_feedback")
    if isinstance(feedback, dict):
        return dict(feedback)
    path = state.run_dir / "route_failure_feedback.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return dict(data) if isinstance(data, dict) else {}
    return {}


def _literature_template_plugin_flags_from_artifacts(state: ToolExecutionState) -> dict[str, Any]:
    compiled = _compiled_downstream_from_state(state)
    plugin = dict((compiled.get("literature_template_plugin") or {}))
    flags = dict(plugin.get("plugin_flags") or {})
    return flags if flags.get("enabled") else {}


def _merge_self_evo_memory_plugin_flags(
    state: ToolExecutionState,
    flags: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    memory = _self_evo_memory_from_payload_or_artifacts(state, payload)
    if not memory or not memory.get("accepted"):
        return flags
    merged = dict(flags or {})
    memory_cards = [dict(item) for item in memory.get("reusable_template_cards") or [] if isinstance(item, dict)]
    memory_rows = [dict(item) for item in memory.get("reusable_one_step_rows") or [] if isinstance(item, dict)]
    if not memory_cards and not memory_rows:
        return merged
    merged["enabled"] = True
    merged["template_cards"] = _dedupe_dicts([*(merged.get("template_cards") or []), *memory_cards])
    merged["one_step_rows"] = _dedupe_dicts([*(merged.get("one_step_rows") or []), *memory_rows])
    merged["top_k"] = max(1, int(merged.get("top_k") or min(6, len(merged["template_cards"]) or 1)))
    merged["max_added"] = max(1, int(merged.get("max_added") or min(6, len(merged["one_step_rows"]) or len(merged["template_cards"]) or 1)))
    merged["requires_audit"] = True
    merged["not_raw_reaction_injection"] = True
    merged["self_evo_memory"] = {
        "enabled": True,
        "case_id": memory.get("case_id"),
        "template_card_count": len(memory_cards),
        "one_step_row_count": len(memory_rows),
        "future_use_policy": dict(memory.get("future_use_policy") or {}),
    }
    return merged


def _self_evo_memory_from_payload_or_artifacts(state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    explicit = payload.get("self_evo_memory")
    if isinstance(explicit, dict):
        return dict(explicit)
    path_value = payload.get("self_evo_memory_path")
    if path_value:
        path = Path(str(path_value))
        if not path.is_absolute():
            path = state.run_dir / path
        resolved = path.resolve()
        if _is_relative_to(resolved, state.run_dir.resolve()) and resolved.exists():
            try:
                data = json.loads(resolved.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
            return dict(data) if isinstance(data, dict) else {}
        return {}
    memory = state.artifacts.get("self_evo_memory")
    if isinstance(memory, dict):
        return dict(memory)
    path = state.run_dir / "self_evo_memory.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return dict(data) if isinstance(data, dict) else {}
    return {}


def _dedupe_dicts(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _self_evo_staging_from_artifacts(state: ToolExecutionState) -> dict[str, Any]:
    compiled = _compiled_downstream_from_state(state)
    self_evo = dict(compiled.get("self_evo") or {})
    return self_evo


def _stock_audit_from_artifacts(state: ToolExecutionState) -> bool:
    chemenzy = dict(state.artifacts.get("chemenzy") or {})
    if chemenzy.get("stock_audit_passed"):
        return True
    if isinstance(chemenzy.get("route_audit"), dict) and chemenzy["route_audit"].get("stock_audit_passed"):
        return True
    result = dict(chemenzy.get("result") or chemenzy)
    search_status = dict(result.get("search_status") or {})
    if search_status.get("solved"):
        return True
    return any(bool(((route.get("metrics") or {}).get("route_solved"))) for route in result.get("routes") or [])


def _frontier_from_route_package(route_package: dict[str, Any]) -> str:
    frontier = dict(route_package.get("frontier") or {})
    return str(frontier.get("frontier_smiles") or (route_package.get("target") or {}).get("smiles") or "")


def _frontier_from_chemenzy_web(chemenzy: dict[str, Any]) -> str:
    result = dict(chemenzy.get("result") or chemenzy)
    for route in result.get("routes") or []:
        for key in ("unresolved_frontiers", "open_leaves", "terminal_reactants"):
            values = route.get(key) or []
            if values:
                return str(values[0])
    return str(result.get("target") or result.get("target_smiles") or "")


def _route_status_from_chemenzy_web(chemenzy: dict[str, Any]) -> str:
    if chemenzy.get("route_status"):
        return str(chemenzy["route_status"])
    result = dict(chemenzy.get("result") or chemenzy)
    status = dict(result.get("search_status") or {})
    if status.get("solved"):
        return "solved"
    if result.get("ok") and result.get("routes"):
        return "partial"
    return "unresolved"


def _frontier_flags_from_chemenzy(chemenzy: dict[str, Any], status: str) -> list[str]:
    if status == "solved":
        return []
    reasons = [str(item) for item in chemenzy.get("reasons") or []]
    flags = ["unresolved_core"]
    if any("fake" in item or "same_scaffold" in item or "no_complexity" in item for item in reasons):
        flags.extend(["advanced_same_scaffold", "no_complexity_drop"])
    return sorted(set(flags))


def _baseline_json_path(state: ToolExecutionState) -> str | None:
    path = state.run_dir / "chemenzy_baseline_routes.json"
    if path.exists():
        return str(path)
    chemenzy = state.artifacts.get("chemenzy")
    if isinstance(chemenzy, dict):
        baseline = {
            "schema_version": "baseline_routes.v1",
            "status": str(chemenzy.get("status") or "provided"),
            "solved": _stock_audit_from_artifacts(state),
            "routes": (chemenzy.get("result") or chemenzy).get("routes") or [],
            "stock_audit_passed": _stock_audit_from_artifacts(state),
            "audit_reasons": list(chemenzy.get("reasons") or []),
        }
        write_json(path, baseline)
        return str(path)
    return None


def _open_research_artifacts(open_dir: Path) -> dict[str, str]:
    names = list(REQUIRED_OPEN_RESEARCH_ARTIFACTS) + [
        "codex_events.jsonl",
        "open_research_manifest.json",
        "open_research_experience.json",
        "compiled_downstream_consumables.json",
        "compiled_guided_chemenzy_requests.json",
        "compiled_route_expansion_tasks.json",
        "compiled_literature_template_plugin.json",
        "self_evo_staging_kb.json",
    ]
    return {name: str(open_dir / name) for name in names if (open_dir / name).exists()}


def _compile_open_research_downstream(
    *,
    state: ToolExecutionState,
    open_dir: Path,
    target_smiles: str,
    prefer_local_seed: bool = False,
) -> dict[str, Any]:
    path = open_dir / "downstream_consumables.json"
    candidates: list[tuple[str, dict[str, Any] | Path]] = []
    curator_augmented = _open_research_curator_augmented_downstream(open_dir)
    if curator_augmented:
        candidates.append(("curator_augmented_downstream", curator_augmented))
    local_seed = _open_research_local_downstream_seed(open_dir)
    if prefer_local_seed and local_seed:
        candidates.append(("harness_local_downstream_seed", local_seed))
    if path.exists():
        candidates.append(("downstream_consumables", path))
    if not prefer_local_seed and local_seed:
        candidates.append(("harness_local_downstream_seed", local_seed))

    rejected: list[dict[str, Any]] = []
    for source, payload_or_path in candidates:
        compiled = compile_downstream_consumables(
            payload_or_path,
            target_smiles=target_smiles,
            case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
            enable_online_anchor_resolution=_online_anchor_resolution_enabled(state.target_input),
        )
        result = _compiled_downstream_harness_result(compiled, output_dir=open_dir, source=source)
        if result.get("accepted"):
            return result
        rejected.append({"source": source, "reasons": result.get("reasons") or []})
    if not rejected:
        return {}
    result = _compiled_downstream_harness_result(
        compile_downstream_consumables(
            candidates[0][1],
            target_smiles=target_smiles,
            case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
            enable_online_anchor_resolution=_online_anchor_resolution_enabled(state.target_input),
        ),
        output_dir=open_dir,
        source=str(candidates[0][0]),
    )
    result["rejected_sources"] = rejected
    return result


def _open_research_curator_augmented_downstream(open_dir: Path) -> dict[str, Any]:
    if not source_detail_curator_records_path(open_dir).exists():
        return {}
    manifest_path = open_dir / "open_research_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    pack_path = Path(str(((manifest.get("retrieval_prefetch") or {}).get("source_detail_extraction_pack_path") or "")))
    if not pack_path.exists():
        return {}
    try:
        resolution = resolve_source_detail_extraction_pack(
            pack_path,
            output_dir=open_dir,
            timeout_s=10.0,
            max_items=10,
        )
    except Exception:
        return {}
    manifest["source_detail_resolution"] = source_detail_resolution_manifest_entry(
        resolution,
        output_dir=open_dir,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    patch = dict(resolution.get("downstream_patch") or {})
    steps = [dict(item) for item in patch.get("source_detail_route_steps") or [] if isinstance(item, dict)]
    rejected = [dict(item) for item in patch.get("rejected_consumables") or [] if isinstance(item, dict)]
    if not steps and not rejected:
        return {}
    base = _load_downstream_payload(open_dir / "downstream_consumables.json") or _open_research_local_downstream_seed(open_dir)
    if not base:
        case_id = str(((manifest.get("target") or {}).get("name") or "case"))
        base = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": case_id,
            "planner_handoff": {
                "next_action": "template_plugin_rerun" if steps else "chemist_review",
                "solved": False,
                "production_kb_promotion": False,
                "generated_by": "curator_augmented_downstream",
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }
    out = dict(base)
    out["source_detail_route_steps"] = _dedupe_dicts([
        *[dict(item) for item in out.get("source_detail_route_steps") or [] if isinstance(item, dict)],
        *steps,
    ])
    out["rejected_consumables"] = _dedupe_dicts([
        *[dict(item) for item in out.get("rejected_consumables") or [] if isinstance(item, dict)],
        *rejected,
    ])
    handoff = dict(out.get("planner_handoff") or {})
    if steps:
        handoff["next_action"] = "template_plugin_rerun"
        handoff["reason"] = "source-detail curator records produced source-grounded route steps"
    handoff["solved"] = False
    handoff["production_kb_promotion"] = False
    out["planner_handoff"] = handoff
    return out


def _load_downstream_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _open_research_local_downstream_seed(open_dir: Path) -> dict[str, Any]:
    path = open_dir / "harness_local_downstream_seed.json"
    if not path.exists():
        return {}
    try:
        seed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(seed, dict):
        return {}
    downstream = seed.get("downstream_consumables")
    return dict(downstream) if isinstance(downstream, dict) else {}


def _compiled_downstream_harness_result(
    compiled: dict[str, Any],
    *,
    output_dir: Path,
    source: str,
) -> dict[str, Any]:
    refs = write_compiled_downstream_artifacts(compiled, output_dir=output_dir)
    return {
        "schema_version": "compiled_downstream_harness_result.v1",
        "accepted": bool(compiled.get("accepted")),
        "source": str(source),
        "reasons": [str(item) for item in compiled.get("reasons") or []],
        "artifact_refs": refs,
        "summary": {
            "guided_policy_count": len(((compiled.get("guided_chemenzy") or {}).get("policy_payloads") or [])),
            "route_expansion_task_count": len(((compiled.get("route_expansion") or {}).get("tasks") or [])),
            "template_card_count": len(((compiled.get("literature_template_plugin") or {}).get("template_cards") or [])),
            "one_step_row_count": len(((compiled.get("literature_template_plugin") or {}).get("one_step_rows") or [])),
            "self_evo_staging_candidate_count": int(
                ((compiled.get("self_evo") or {}).get("staging_candidate_count") or 0)
            ),
        },
    }


def _load_open_research_run_record(open_dir: Path) -> dict[str, Any]:
    path = open_dir / "open_agent_run_record.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "schema_version": "open_codex_structure_template_run.v1",
            "error": "invalid_open_agent_run_record_json",
            "decode_error": str(exc),
            "path": str(path),
        }


def _validate_open_research_output(*, open_dir: Path, run_record: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    transport_reasons: list[str] = []
    warnings: list[str] = []
    if not run_record:
        reasons.append("open_agent_run_record_missing")
    elif run_record.get("error"):
        transport_reasons.append(f"open_agent_{run_record['error']}")
    if run_record and run_record.get("exit_code") != 0:
        transport_reasons.append("open_agent_nonzero_or_missing_exit")

    event_summary: dict[str, Any] = {}
    metadata = dict(run_record.get("metadata") or {}) if isinstance(run_record, dict) else {}
    if metadata.get("stream_jsonl"):
        event_log = Path(str(metadata.get("event_log_path") or open_dir / "codex_events.jsonl"))
        event_summary = _summarize_codex_events(event_log)
        if not event_summary.get("turn_completed"):
            transport_reasons.append("codex_events_missing_turn_completed")
        if not event_summary.get("usage"):
            transport_reasons.append("codex_events_missing_usage")

    boundary_audit = audit_open_research_boundary(run_dir=open_dir)

    missing = [name for name in REQUIRED_OPEN_RESEARCH_ARTIFACTS if not (open_dir / name).exists()]
    reasons.extend(f"missing_open_agent_artifact:{name}" for name in missing)

    invalid_json: list[str] = []
    schema_reasons: list[str] = []
    for name in REQUIRED_OPEN_RESEARCH_JSON_ARTIFACTS:
        path = open_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            invalid_json.append(name)
            continue
        schema_reasons.extend(validate_open_research_json_payload(name=name, payload=payload))
    reasons.extend(f"invalid_open_agent_json:{name}" for name in invalid_json)
    reasons.extend(schema_reasons)
    reasons.extend(str(item) for item in boundary_audit.get("reasons") or [])
    retrieval_consumption = validate_retrieval_prefetch_consumption(run_dir=open_dir)
    reasons.extend(str(item) for item in retrieval_consumption.get("reasons") or [])

    checkpoint_valid = (
        not missing
        and not invalid_json
        and not schema_reasons
        and bool(boundary_audit.get("accepted", True))
        and bool(retrieval_consumption.get("accepted", True))
    )
    checkpoint_after_timeout = bool(run_record.get("error") == "timeout" and checkpoint_valid)
    if checkpoint_after_timeout:
        warnings.extend(sorted(set(transport_reasons + ["checkpoint_valid_but_turn_timeout"])))
    else:
        reasons.extend(transport_reasons)

    return {
        "schema_version": "open_structure_research_output_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
        "checkpoint_valid": checkpoint_valid,
        "checkpoint_after_timeout": checkpoint_after_timeout,
        "required_artifacts": list(REQUIRED_OPEN_RESEARCH_ARTIFACTS),
        "missing_artifacts": missing,
        "invalid_json_artifacts": invalid_json,
        "schema_reasons": schema_reasons,
        "event_summary": event_summary,
        "boundary_audit": boundary_audit,
        "retrieval_prefetch_consumption": retrieval_consumption,
    }


def _summarize_codex_events(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "event_count": 0,
        "turn_completed": False,
        "usage": None,
        "last_event_type": "",
    }
    if not path.exists():
        return summary
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        summary["event_count"] = int(summary["event_count"]) + 1
        summary["last_event_type"] = str(event.get("type") or "")
        if event.get("type") == "turn.completed":
            summary["turn_completed"] = True
            summary["usage"] = event.get("usage")
    return summary


def _mock_result(state: ToolExecutionState, tool_name: str, payload: dict[str, Any]) -> Any | None:
    value = state.mock_tool_results.get(tool_name)
    if callable(value):
        return value(state, payload)
    return value


def _write_tool_record(state: ToolExecutionState, record: ToolCallRecord) -> None:
    append_jsonl(state.run_dir / "tool_calls.jsonl", record.to_dict())
    append_jsonl(state.run_dir / "decision_trace.jsonl", {"stage": "tool_call", "tool_call": record.to_dict()})


def _chem_enzy_python_bin() -> Path | None:
    env_prefix = Path(os.environ.get("CHEMENZY_ENV_PREFIX", str(DEFAULT_CHEMENZY_ENV_PREFIX)))
    candidate = env_prefix / "bin" / "python"
    if candidate.exists():
        return candidate
    return Path(sys.executable) if (ROOT / "vendor/ChemEnzyRetroPlanner").exists() else None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def _artifact_tree_contains_raw_reaction(value: Any, *, deterministic_context: bool = False) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            child_deterministic = deterministic_context or key_text in {
                "chemenzy",
                "guided_chemenzy",
                "route_audit",
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

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
from cascade_planner.harness.source_detail_chain_builder import (
    build_source_detail_curator_records_from_chain,
    compile_hybrid_route_set as compile_hybrid_route_set_artifact,
    compile_source_detail_chain_route as compile_source_detail_chain_route_artifact,
    probe_literature_plugin_chain,
    resolve_curator_records_to_source_detail_steps,
)
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
        "validate_literature_intermediate_chain": validate_literature_intermediate_chain_tool,
        "build_source_detail_curator_records": build_source_detail_curator_records_tool,
        "compile_source_detail_chain_route": compile_source_detail_chain_route_tool,
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
    out = {
        "schema_version": "guided_chemenzy_rerun_result.v1",
        "accepted": bool(result.get("ok") or result.get("accepted", result.get("exit_code") == 0)),
        "policy": policy,
        "request": request,
        "result": result,
        "raw_route_verifier": verifier,
        "route_status": str(verifier.get("route_status") or ("solved" if (result.get("search_status") or {}).get("solved") else "unresolved")),
        "solved": bool(verifier.get("accepted")),
    }
    if verifier and not verifier.get("accepted"):
        out["accepted"] = False
        out["reasons"] = [str(item) for item in verifier.get("reasons") or ["route_verifier_rejected_raw_routes"]]
        feedback = compile_route_failure_feedback(
            verifier,
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
        write_json(state.run_dir / "guided_route_verifier_report.json", verifier)
    # A verifier rejection is chemistry feedback, not a harness execution
    # failure; runtime/transport failures without verifier evidence still
    # reject the tool call.
    tool_accepted = bool(out.get("accepted")) or bool(verifier)
    return {"accepted": tool_accepted, "result": out, "reasons": [str(item) for item in out.get("reasons") or []]}


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

    remaining_budget = max(0, int(state.budget.max_route_expansion_subgoal_runs) - int(state.route_expansion_subgoal_runs))
    if remaining_budget <= 0:
        return {"accepted": False, "reasons": ["route_expansion_subgoal_budget_exhausted"]}
    max_targets = min(max(1, int(payload.get("max_targets") or 2)), remaining_budget)
    rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for idx, target in enumerate(targets[:max_targets]):
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
        safe_name = _safe_file_stem(target["name"] or f"subgoal_{idx + 1}")
        raw_path = sub_dir / f"{idx + 1:02d}_{safe_name}_raw_result.json"
        req_path = sub_dir / f"{idx + 1:02d}_{safe_name}_request.json"
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
            "subgoal_index": idx,
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
        if row["accepted"]:
            accepted_rows.append(row)
        write_json(sub_dir / f"{idx + 1:02d}_{safe_name}_verifier.json", verifier)

    result = {
        "schema_version": "route_expansion_subgoal_search_result.v1",
        "accepted": bool(accepted_rows),
        "status": "solved" if accepted_rows else "failed",
        "solved": bool(accepted_rows),
        "subgoal_count": len(rows),
        "accepted_subgoal_count": len(accepted_rows),
        "rejected_subgoal_count": len(rows) - len(accepted_rows),
        "subgoals": rows,
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
    name = str(state.target_input.get("target_name") or state.preflight.get("case_id") or "target").strip()
    if "_" in name:
        prefix = name.split("_", 1)[0].strip()
        if prefix:
            return prefix
    return name or "target"


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
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "schema_version": "chemenzy_run_result.v1",
            "accepted": False,
            "status": "timeout",
            "reasons": ["chem_enzy_timeout"],
            "command": cmd,
            "stdout": _timeout_stream(exc.stdout),
            "stderr": _timeout_stream(exc.stderr),
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
    result.setdefault("exit_code", int(proc.returncode))
    if proc.returncode != 0:
        result.setdefault("accepted", False)
        result.setdefault("reasons", []).append("chemenzy_nonzero_exit")
        result["stderr"] = proc.stderr
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
        state.artifacts["literature_pdf_structure_evidence"] = result
        write_json(state.run_dir / "literature_pdf_structure_evidence.json", result)
        return {"accepted": bool(result.get("accepted", True)), "result": result}

    out = _tool_output_dir(state, payload, default_name="literature_pdf_structure_extraction")
    pdf_path = _input_path(state, payload.get("pdf_path")) if payload.get("pdf_path") else None
    image_paths = [
        str(_input_path(state, value))
        for value in payload.get("image_paths") or []
        if str(value or "").strip()
    ]
    result = extract_literature_pdf_assets(
        pdf_path=pdf_path,
        output_dir=out,
        page_numbers=[int(item) for item in payload.get("page_numbers") or []],
        render_zoom=float(payload.get("render_zoom") or 2.0),
        image_paths=image_paths,
        scheme_crops=[dict(item) for item in payload.get("scheme_crops") or [] if isinstance(item, dict)],
        compound_labels=[str(item) for item in payload.get("compound_labels") or [] if str(item).strip()],
    )
    state.artifacts["literature_pdf_structure_evidence"] = result
    state.artifacts["literature_pdf_structure_evidence_dir"] = str(out)
    write_json(state.run_dir / "literature_pdf_structure_evidence.json", result)
    return {
        "accepted": bool(result.get("accepted")),
        "result": result,
        "artifact_refs": {
            "literature_pdf_structure_evidence": str(out / "literature_pdf_structure_evidence.json"),
        },
        "reasons": [str(item) for item in result.get("reasons") or []],
    }


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
    else:
        return {"accepted": False, "reasons": ["candidate_chain_missing"]}
    result = validate_visual_structure_chain(
        candidate_chain,
        output_dir=out,
        target_smiles=str(payload.get("target_smiles") or state.target_input.get("target_smiles") or ""),
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
    write_json(state.run_dir / "source_detail_chain_route_result.json", result)
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
    if not path.exists():
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
    return []


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
                "source": "explicit_payload",
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
    return _dedupe_child_targets(rows)


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

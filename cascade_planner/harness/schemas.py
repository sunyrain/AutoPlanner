"""Small typed records for the Codex-entry harness."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TARGET_INPUT_SCHEMA = "codex_entry_target_input.v1"
WORKFLOW_PLAN_SCHEMA = "codex_entry_workflow_plan.v1"
TOOL_CALL_SCHEMA = "codex_entry_tool_call.v1"
ARTIFACT_BUNDLE_SCHEMA = "codex_entry_artifact_bundle.v1"
FINAL_VERDICT_SCHEMA = "codex_entry_final_verdict.v1"
CANONICAL_RUN_SEMANTICS = "canonical_agent_controller"

RUN_SEMANTICS = {
    CANONICAL_RUN_SEMANTICS,
    "replay",
    "probe",
    "showcase",
    "legacy",
}

LITERATURE_FIRST_REASONS = {
    "user_requested_literature",
    "glycoside_or_o_glycoside_like",
    "natural_product_like",
    "macrocycle_or_steroid_like",
    "steroid_or_polycyclic_core",
    "known_backend_unsuitable",
}

ALLOWED_STRATEGIES = {
    "chem_enzy_first",
    "literature_first",
    "hybrid",
    "reject_invalid_input",
}

ALLOWED_LOCAL_TOOLS = {
    "run_chemenzy",
    "audit_route_and_extract_frontier",
    "run_smiles_first_literature_workflow",
    "run_open_structure_research_agent",
    "extract_pdf_literature_structures",
    "validate_literature_intermediate_chain",
    "build_source_detail_curator_records",
    "compile_source_detail_chain_route",
    "compile_hybrid_route_set",
    "run_guided_chemenzy_rerun",
    "run_route_expansion_subgoal_search",
    "run_self_evo_replay_gate",
    "validate_artifact_bundle",
    "emit_final_verdict",
}

FINAL_VERDICTS = {
    "solved",
    "partial_anchor_only_not_solved",
    "unresolved",
    "fake_closed_rejected",
    "invalid_input",
    "needs_followup",
}

FORBIDDEN_RAW_REACTION_KEYS = {
    "rxn",
    "rxn_smiles",
    "rxn_smiles_list",
    "reaction_smiles",
    "raw_reaction",
    "raw_reactions",
    "raw_reaction_candidates",
    "reaction_candidates",
    "route_tree_actions",
    "candidate_actions",
}


@dataclass
class TargetInput:
    target_name: str
    target_smiles: str
    family_hint: str = ""
    case_id: str = ""
    schema_version: str = TARGET_INPUT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowPlan:
    case_id: str
    recommended_strategy: str
    planned_tools: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    risk_flags: list[str] = field(default_factory=list)
    expected_verdict_floor: str = "needs_audit"
    planner_decision_reason: str = ""
    run_semantics: str = CANONICAL_RUN_SEMANTICS
    schema_version: str = WORKFLOW_PLAN_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallRecord:
    tool_name: str
    status: str
    input_payload: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    schema_version: str = TOOL_CALL_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactBundle:
    case_id: str
    target_input: dict[str, Any]
    preflight: dict[str, Any]
    workflow_plan: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    validations: list[dict[str, Any]] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    run_semantics: str = CANONICAL_RUN_SEMANTICS
    schema_version: str = ARTIFACT_BUNDLE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalVerdict:
    case_id: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    route_status: str = ""
    solved: bool = False
    stock_audit_passed: bool = False
    artifact_refs: dict[str, str] = field(default_factory=dict)
    run_semantics: str = CANONICAL_RUN_SEMANTICS
    schema_version: str = FINAL_VERDICT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def target_input_from_dict(data: dict[str, Any]) -> TargetInput:
    return TargetInput(
        target_name=str(data.get("target_name") or ""),
        target_smiles=str(data.get("target_smiles") or ""),
        family_hint=str(data.get("family_hint") or ""),
        case_id=str(data.get("case_id") or ""),
        schema_version=str(data.get("schema_version") or TARGET_INPUT_SCHEMA),
    )


def workflow_plan_from_dict(data: dict[str, Any]) -> WorkflowPlan:
    return WorkflowPlan(
        case_id=str(data.get("case_id") or ""),
        recommended_strategy=str(data.get("recommended_strategy") or ""),
        planned_tools=[dict(item) for item in data.get("planned_tools") or [] if isinstance(item, dict)],
        rationale=str(data.get("rationale") or ""),
        risk_flags=[str(item) for item in data.get("risk_flags") or []],
        expected_verdict_floor=str(data.get("expected_verdict_floor") or "needs_audit"),
        planner_decision_reason=str(data.get("planner_decision_reason") or ""),
        run_semantics=str(data.get("run_semantics") or CANONICAL_RUN_SEMANTICS),
        schema_version=str(data.get("schema_version") or WORKFLOW_PLAN_SCHEMA),
    )


def validate_workflow_plan(plan_or_data: WorkflowPlan | dict[str, Any], *, case_id: str = "") -> dict[str, Any]:
    plan = plan_or_data if isinstance(plan_or_data, WorkflowPlan) else workflow_plan_from_dict(plan_or_data)
    reasons: list[str] = []
    if plan.schema_version != WORKFLOW_PLAN_SCHEMA:
        reasons.append("invalid_workflow_plan_schema")
    if not plan.case_id:
        reasons.append("missing_case_id")
    if case_id and plan.case_id != case_id:
        reasons.append("case_id_mismatch")
    if plan.recommended_strategy not in ALLOWED_STRATEGIES:
        reasons.append("invalid_recommended_strategy")
    if plan.run_semantics not in RUN_SEMANTICS:
        reasons.append("invalid_run_semantics")
    if plan.recommended_strategy == "reject_invalid_input" and plan.planned_tools:
        reasons.append("reject_invalid_input_must_not_plan_tools")
    if plan.recommended_strategy == "literature_first" and plan.planner_decision_reason not in LITERATURE_FIRST_REASONS:
        reasons.append("literature_first_requires_accepted_reason")
    if not isinstance(plan.planned_tools, list):
        reasons.append("planned_tools_not_list")
    tool_names = [str(tool.get("tool_name") or tool.get("name") or "") for tool in plan.planned_tools]
    if plan.recommended_strategy == "chem_enzy_first":
        first = next((name for name in tool_names if name and name != "emit_final_verdict"), "")
        if first and first != "run_chemenzy":
            reasons.append("chem_enzy_first_must_start_with_run_chemenzy")
    if plan.recommended_strategy in {"chem_enzy_first", "hybrid"}:
        for gated in ("run_smiles_first_literature_workflow", "run_open_structure_research_agent"):
            if gated not in tool_names:
                continue
            gated_index = tool_names.index(gated)
            before = set(tool_names[:gated_index])
            if "run_chemenzy" not in before or "audit_route_and_extract_frontier" not in before:
                reasons.append(f"{gated}_requires_native_audit_or_literature_first_reason")
    for idx, tool in enumerate(plan.planned_tools):
        name = str(tool.get("tool_name") or tool.get("name") or "")
        if name not in ALLOWED_LOCAL_TOOLS:
            reasons.append(f"forbidden_planner_tool:{idx}:{name or 'missing'}")
        payload = dict(tool.get("payload") or tool.get("input") or {})
        if _contains_raw_reaction_payload(payload):
            reasons.append("raw_reaction_injection")
        if _contains_raw_reaction_payload(tool):
            reasons.append("raw_reaction_injection")
        if tool.get("route_status") == "solved" or tool.get("verdict") == "solved":
            reasons.append("planner_direct_solved_claim")
    return {
        "schema_version": "codex_entry_workflow_plan_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "case_id": plan.case_id,
        "planned_tool_count": len(plan.planned_tools),
    }


def validate_tool_payload(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if tool_name not in ALLOWED_LOCAL_TOOLS:
        reasons.append("forbidden_tool")
    if _contains_raw_reaction_payload(payload):
        reasons.append("raw_reaction_injection")
    if payload.get("verdict") == "solved" or payload.get("route_status") == "solved":
        reasons.append("tool_payload_direct_solved_claim")
    return {
        "schema_version": "codex_entry_tool_payload_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "tool_name": tool_name,
    }


def _contains_raw_reaction_payload(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_RAW_REACTION_KEYS:
                return True
            if _contains_raw_reaction_payload(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_reaction_payload(item) for item in value)
    return False


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")

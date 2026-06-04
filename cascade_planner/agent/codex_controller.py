"""Minimal event-driven Codex controller shell.

The controller is deliberately a safety/control wrapper. It can select bounded
actions and call existing tool-level functions, but it cannot access ChemEnzy
step-level hooks, run online LLM rerank/judges, inject raw reactions, or write
production KB entries.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from cascade_planner.agent.artifact_validators import validate_typed_artifact
from cascade_planner.agent.case_trace import CaseBundle, RouteStatus
from cascade_planner.agent.codex_worker import WorkerTask, run_codex_worker
from cascade_planner.agent.evolution_manager import EvolutionCandidate, LayeredKnowledgeBase


CONTROLLER_OBSERVATION_SCHEMA = "controller_observation.v1"
CONTROLLER_ACTION_SCHEMA = "controller_action.v1"
CONTROLLER_TRACE_SCHEMA = "controller_trace.v1"

ALLOWED_ACTIONS = {
    "RESOLVE_TARGET",
    "PROFILE_STRUCTURE",
    "RUN_BASELINE_CHEMENZY",
    "AUDIT_ROUTE",
    "DIAGNOSE_FAILURE",
    "RESEARCH_TARGET",
    "RESEARCH_STUCK_NODE",
    "EXTRACT_EVIDENCE",
    "VALIDATE_EVIDENCE",
    "COMPILE_STRATEGIC_OPERATOR",
    "RUN_GUIDED_CHEMENZY",
    "DESIGN_CONDITIONS",
    "SUBMIT_EVOLUTION_CANDIDATE",
    "FINAL_UNRESOLVED",
    "FINAL_PARTIAL_ANCHOR",
    "FINAL_SOLVED_BY_AUDIT",
}

FORBIDDEN_ACTIONS = {
    "LLM_RERANK_CANDIDATES",
    "ONLINE_LLM_PROPOSAL_JUDGE",
    "CHEMENZY_STEP_HOOK",
    "RAW_REACTION_INJECTION",
    "WRITE_PRODUCTION_KB",
}


@dataclass
class ControllerBudget:
    max_chem_enzy_runs: int = 2
    max_codex_worker_runs: int = 2
    max_literature_rounds: int = 2
    max_stuck_nodes: int = 4
    max_wall_time_s: float = 120.0
    max_tool_calls: int = 20
    max_reruns_without_improvement: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ControllerAction:
    action_type: str
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CONTROLLER_ACTION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ControllerTrace:
    case_id: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    chem_enzy_runs: int = 0
    codex_worker_runs: int = 0
    literature_rounds: int = 0
    reruns_without_improvement: int = 0
    started_at_monotonic: float = field(default_factory=time.monotonic)
    final_route_status: str = RouteStatus.UNRESOLVED.value
    schema_version: str = CONTROLLER_TRACE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "actions": list(self.actions),
            "tool_calls": self.tool_calls,
            "chem_enzy_runs": self.chem_enzy_runs,
            "codex_worker_runs": self.codex_worker_runs,
            "literature_rounds": self.literature_rounds,
            "reruns_without_improvement": self.reruns_without_improvement,
            "elapsed_s": round(time.monotonic() - self.started_at_monotonic, 3),
            "final_route_status": self.final_route_status,
        }


ToolHandler = Callable[[ControllerAction, CaseBundle, ControllerTrace], dict[str, Any]]


def observe_case(bundle: CaseBundle, trace: ControllerTrace | None = None) -> dict[str, Any]:
    accepted_types = [artifact.artifact_type for artifact in bundle.accepted_artifacts()]
    failure_reasons = [event.reason for event in bundle.failure_events]
    return {
        "schema_version": CONTROLLER_OBSERVATION_SCHEMA,
        "case_id": bundle.case_id,
        "route_status": bundle.route_status.value,
        "accepted_artifact_types": accepted_types,
        "failure_reasons": failure_reasons,
        "has_evidence": "EvidenceCardList" in accepted_types or "EvidenceCard" in accepted_types,
        "has_package_validation": "RoutePackageValidation" in accepted_types,
        "budget_trace": trace.to_dict() if trace is not None else {},
    }


def decide_next_action(observation: dict[str, Any], budget_state: dict[str, Any] | None = None) -> ControllerAction:
    if _budget_exhausted(dict(budget_state or {})):
        return ControllerAction("FINAL_UNRESOLVED", "budget_exhausted")
    status = str(observation.get("route_status") or "")
    accepted = set(observation.get("accepted_artifact_types") or [])
    failures = set(observation.get("failure_reasons") or [])
    if status == RouteStatus.PARTIAL_ANCHOR.value and "RoutePackageValidation" in accepted:
        return ControllerAction("COMPILE_STRATEGIC_OPERATOR", "validated_partial_anchor")
    if "unresolved_core" in failures and not observation.get("has_evidence"):
        return ControllerAction("RESEARCH_STUCK_NODE", "unresolved_core_without_evidence")
    if status == RouteStatus.FAKE_CLOSED_REJECTED.value:
        return ControllerAction("DIAGNOSE_FAILURE", "fake_closure_rejected")
    if status == RouteStatus.UNRESOLVED.value:
        return ControllerAction("RESEARCH_TARGET", "unresolved_case")
    return ControllerAction("AUDIT_ROUTE", "status_requires_audit")


def execute_action(
    action: ControllerAction,
    bundle: CaseBundle,
    trace: ControllerTrace,
    *,
    handlers: dict[str, ToolHandler] | None = None,
) -> dict[str, Any]:
    action_validation = validate_controller_action(action)
    if not action_validation["accepted"]:
        result = {"accepted": False, "reasons": action_validation["reasons"], "action": action.to_dict()}
        trace.actions.append(result)
        return result
    trace.tool_calls += 1
    if action.action_type in {"RUN_BASELINE_CHEMENZY", "RUN_GUIDED_CHEMENZY"}:
        trace.chem_enzy_runs += 1
    if action.action_type in {"RESEARCH_TARGET", "RESEARCH_STUCK_NODE"}:
        trace.codex_worker_runs += 1
        trace.literature_rounds += 1
    handler = dict(handlers or {}).get(action.action_type)
    if handler is not None:
        result = handler(action, bundle, trace)
    else:
        result = _default_action_result(action, bundle, trace)
    trace.actions.append({"action": action.to_dict(), "result": result})
    if action.action_type.startswith("FINAL_"):
        trace.final_route_status = _final_status_for_action(action.action_type)
    return result


def validate_controller_action(action_or_data: ControllerAction | dict[str, Any]) -> dict[str, Any]:
    action = action_or_data if isinstance(action_or_data, ControllerAction) else ControllerAction(
        action_type=str(action_or_data.get("action_type") or ""),
        reason=str(action_or_data.get("reason") or ""),
        payload=dict(action_or_data.get("payload") or {}),
        schema_version=str(action_or_data.get("schema_version") or CONTROLLER_ACTION_SCHEMA),
    )
    reasons: list[str] = []
    if action.schema_version != CONTROLLER_ACTION_SCHEMA:
        reasons.append("invalid_controller_action_schema")
    if action.action_type in FORBIDDEN_ACTIONS:
        reasons.append("forbidden_controller_action")
    if action.action_type not in ALLOWED_ACTIONS:
        reasons.append("unknown_controller_action")
    if _contains_raw_reaction(action.payload):
        reasons.append("raw_reaction_injection")
    if action.payload.get("llm_rerank") or action.payload.get("online_llm_judge"):
        reasons.append("online_llm_decision_not_allowed")
    if action.payload.get("write_layer") == "production" or action.payload.get("kb_layer") == "production":
        reasons.append("controller_direct_production_write")
    return {
        "schema_version": "controller_action_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "action_type": action.action_type,
    }


def validate_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return validate_typed_artifact(artifact)


def update_blackboard(bundle: CaseBundle, result: dict[str, Any]) -> CaseBundle:
    # P0/P1a CaseBundle is append-only elsewhere. This shell only records
    # state-free results and returns the same bundle.
    if result.get("final_route_status"):
        bundle.route_status = RouteStatus(str(result["final_route_status"]))
    return bundle


def stop_or_continue(bundle: CaseBundle, trace: ControllerTrace, budget: ControllerBudget) -> dict[str, Any]:
    exhausted = _budget_exhausted(_budget_state(trace, budget))
    terminal = bundle.route_status in {
        RouteStatus.SOLVED,
        RouteStatus.SEMISYNTHESIS_CLOSED,
        RouteStatus.FAKE_CLOSED_REJECTED,
    }
    return {
        "schema_version": "controller_stop_decision.v1",
        "stop": bool(exhausted or terminal),
        "route_status": bundle.route_status.value,
        "budget_exhausted": exhausted,
        "trace": trace.to_dict(),
    }


def run_controller_once(
    bundle: CaseBundle,
    *,
    budget: ControllerBudget | None = None,
    handlers: dict[str, ToolHandler] | None = None,
) -> dict[str, Any]:
    budget = budget or ControllerBudget()
    trace = ControllerTrace(case_id=bundle.case_id)
    observation = observe_case(bundle, trace)
    action = decide_next_action(observation, _budget_state(trace, budget))
    result = execute_action(action, bundle, trace, handlers=handlers)
    bundle = update_blackboard(bundle, result)
    return {
        "schema_version": "controller_once_result.v1",
        "observation": observation,
        "action": action.to_dict(),
        "result": result,
        "stop_decision": stop_or_continue(bundle, trace, budget),
        "trace": trace.to_dict(),
    }


def run_worker_action(task: WorkerTask) -> dict[str, Any]:
    record = run_codex_worker(task)
    return record.to_dict()


def submit_evolution_candidate(
    kb: LayeredKnowledgeBase,
    candidate: EvolutionCandidate,
    *,
    target_run: bool = True,
) -> dict[str, Any]:
    kb.add_candidate(candidate, target_run=target_run)
    return {
        "schema_version": "submit_evolution_candidate_result.v1",
        "accepted": True,
        "candidate_id": candidate.candidate_id,
        "layer": "candidate",
        "production_write": False,
    }


def _default_action_result(action: ControllerAction, bundle: CaseBundle, trace: ControllerTrace) -> dict[str, Any]:
    if action.action_type == "FINAL_UNRESOLVED":
        return {"accepted": True, "final_route_status": RouteStatus.UNRESOLVED.value, "reason": action.reason}
    if action.action_type == "FINAL_PARTIAL_ANCHOR":
        return {"accepted": True, "final_route_status": RouteStatus.PARTIAL_ANCHOR.value, "reason": action.reason}
    if action.action_type == "FINAL_SOLVED_BY_AUDIT":
        if not action.payload.get("stock_audit_passed"):
            return {"accepted": False, "reasons": ["solved_requires_stock_audit"]}
        return {"accepted": True, "final_route_status": RouteStatus.SOLVED.value, "reason": action.reason}
    return {
        "accepted": True,
        "action_type": action.action_type,
        "case_id": bundle.case_id,
        "note": "controller shell recorded bounded action; external handler not supplied",
    }


def _budget_state(trace: ControllerTrace, budget: ControllerBudget) -> dict[str, Any]:
    return {
        "chem_enzy_runs": trace.chem_enzy_runs,
        "codex_worker_runs": trace.codex_worker_runs,
        "literature_rounds": trace.literature_rounds,
        "tool_calls": trace.tool_calls,
        "reruns_without_improvement": trace.reruns_without_improvement,
        "elapsed_s": time.monotonic() - trace.started_at_monotonic,
        "budget": budget.to_dict(),
    }


def _budget_exhausted(state: dict[str, Any]) -> bool:
    budget = dict(state.get("budget") or {})
    if not budget:
        return False
    checks = [
        ("chem_enzy_runs", "max_chem_enzy_runs"),
        ("codex_worker_runs", "max_codex_worker_runs"),
        ("literature_rounds", "max_literature_rounds"),
        ("tool_calls", "max_tool_calls"),
        ("reruns_without_improvement", "max_reruns_without_improvement"),
    ]
    for actual_key, max_key in checks:
        if int(state.get(actual_key) or 0) >= int(budget.get(max_key) or 0):
            return True
    return float(state.get("elapsed_s") or 0.0) >= float(budget.get("max_wall_time_s") or 0.0)


def _final_status_for_action(action_type: str) -> str:
    if action_type == "FINAL_SOLVED_BY_AUDIT":
        return RouteStatus.SOLVED.value
    if action_type == "FINAL_PARTIAL_ANCHOR":
        return RouteStatus.PARTIAL_ANCHOR.value
    return RouteStatus.UNRESOLVED.value


def _contains_raw_reaction(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"rxn", "rxn_smiles", "reaction_smiles", "raw_reaction", "reaction_candidates"}:
                return True
            if _contains_raw_reaction(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_reaction(item) for item in value)
    if isinstance(value, str):
        return ">>" in value
    return False

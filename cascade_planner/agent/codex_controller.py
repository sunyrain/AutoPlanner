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
from cascade_planner.agent.case_trace import ArtifactRecord, CaseBundle, RouteStatus
from cascade_planner.agent.codex_worker import WorkerBudget, WorkerTask, run_codex_worker
from cascade_planner.agent.evolution_manager import EvolutionCandidate, LayeredKnowledgeBase
from cascade_planner.agent.route_auditor import audit_route_package


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
    "MINE_STRATEGIC_DISCONNECTION",
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


def _validate_artifact_for_bundle(bundle: CaseBundle, artifact: dict[str, Any]) -> dict[str, Any]:
    return validate_typed_artifact(
        artifact,
        validated_evidence_refs=_accepted_evidence_refs(bundle),
        validated_disconnection_refs=_accepted_disconnection_refs(bundle),
    )


def update_blackboard(bundle: CaseBundle, result: dict[str, Any]) -> CaseBundle:
    worker_trace = result.get("worker_trace")
    if isinstance(worker_trace, dict):
        _append_controller_artifact(
            bundle,
            ArtifactRecord(
                artifact_id=_unique_artifact_id(bundle, str(worker_trace.get("run_id") or "worker_run")),
                case_id=bundle.case_id,
                artifact_type="WorkerRunRecord",
                payload=worker_trace,
                source="codex_controller",
                validation_status="accepted" if _worker_record_accepted(worker_trace) else "rejected",
                input_refs=[str(worker_trace.get("task_id") or "")],
            ),
        )
    output_artifact = result.get("output_artifact")
    if isinstance(output_artifact, dict):
        source_fallback = "worker"
        if isinstance(worker_trace, dict):
            source_fallback = str(worker_trace.get("backend") or source_fallback)
        validation = _validate_artifact_for_bundle(bundle, output_artifact)
        status = "accepted" if validation.get("accepted") else "rejected"
        _append_controller_artifact(
            bundle,
            ArtifactRecord(
                artifact_id=_unique_artifact_id(bundle, str(output_artifact.get("artifact_id") or "worker_artifact")),
                case_id=bundle.case_id,
                artifact_type=str(output_artifact.get("artifact_type") or "WorkerOutputArtifact"),
                payload=output_artifact,
                source=str(output_artifact.get("source") or source_fallback),
                validation_status=status,
                input_refs=[str(ref) for ref in output_artifact.get("input_refs") or []],
                evidence_refs=[str(ref) for ref in output_artifact.get("evidence_refs") or []],
            ),
        )
        _append_controller_artifact(
            bundle,
            ArtifactRecord(
                artifact_id=_unique_artifact_id(bundle, f"{output_artifact.get('artifact_id') or 'worker_artifact'}:validation"),
                case_id=bundle.case_id,
                artifact_type="ArtifactValidationRecord",
                payload=validation,
                source="artifact_validator",
                validation_status="accepted" if validation.get("accepted") else "rejected",
                input_refs=[str(output_artifact.get("artifact_id") or "")],
            ),
        )
    validation_record = result.get("validation")
    if isinstance(validation_record, dict):
        _append_controller_artifact(
            bundle,
            ArtifactRecord(
                artifact_id=_unique_artifact_id(bundle, str(validation_record.get("artifact_id") or "controller_validation")),
                case_id=bundle.case_id,
                artifact_type="ArtifactValidationRecord",
                payload=validation_record,
                source="artifact_validator",
                validation_status="accepted" if validation_record.get("accepted") else "rejected",
            ),
        )
    rerun_history = result.get("rerun_history")
    if isinstance(rerun_history, dict):
        _append_controller_artifact(
            bundle,
            ArtifactRecord(
                artifact_id=_unique_artifact_id(bundle, str(rerun_history.get("policy_id") or "guided_rerun_history")),
                case_id=bundle.case_id,
                artifact_type="GuidedRerunHistory",
                payload=rerun_history,
                source="codex_controller",
                validation_status="accepted",
            ),
        )
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


def run_controller_loop(
    bundle: CaseBundle,
    *,
    budget: ControllerBudget | None = None,
    handlers: dict[str, ToolHandler] | None = None,
    research_action: ControllerAction | None = None,
) -> dict[str, Any]:
    """Run the bounded default controller loop for one unresolved case.

    The loop is intentionally coarse grained: worker research can only append a
    draft typed artifact, then deterministic validators/gates decide whether it
    can feed policy compilation and a guided ChemEnzy rerun request.
    """
    budget = budget or ControllerBudget()
    trace = ControllerTrace(case_id=bundle.case_id)
    observations: list[dict[str, Any]] = []
    executed: list[dict[str, Any]] = []

    first_observation = observe_case(bundle, trace)
    observations.append(first_observation)
    first_action = research_action or decide_next_action(first_observation, _budget_state(trace, budget))
    if first_action.action_type not in {"RESEARCH_TARGET", "RESEARCH_STUCK_NODE", "COMPILE_STRATEGIC_OPERATOR", "FINAL_UNRESOLVED"}:
        first_action = ControllerAction("RESEARCH_TARGET", "controller_loop_default_research")
    actions: list[ControllerAction] = []
    if first_action.action_type in {"RESEARCH_TARGET", "RESEARCH_STUCK_NODE", "FINAL_UNRESOLVED"}:
        actions.append(first_action)
    if first_action.action_type != "FINAL_UNRESOLVED":
        actions.extend([
            ControllerAction("EXTRACT_EVIDENCE", "extract_worker_evidence"),
            ControllerAction("VALIDATE_EVIDENCE", "validate_worker_evidence"),
            ControllerAction("MINE_STRATEGIC_DISCONNECTION", "mine_validated_evidence"),
            ControllerAction("COMPILE_STRATEGIC_OPERATOR", "compile_validated_policy"),
            ControllerAction("RUN_GUIDED_CHEMENZY", "request_guided_rerun"),
            ControllerAction("AUDIT_ROUTE", "audit_after_guided_rerun_request"),
        ])
    for action in actions:
        if stop_or_continue(bundle, trace, budget)["budget_exhausted"]:
            break
        if action.action_type == "FINAL_UNRESOLVED":
            result = execute_action(action, bundle, trace, handlers=handlers)
            bundle = update_blackboard(bundle, result)
            executed.append({"action": action.to_dict(), "result": result})
            break
        result = execute_action(action, bundle, trace, handlers=handlers)
        bundle = update_blackboard(bundle, result)
        executed.append({"action": action.to_dict(), "result": result})
        observations.append(observe_case(bundle, trace))
        if action.action_type in {"RESEARCH_TARGET", "RESEARCH_STUCK_NODE"} and not result.get("accepted"):
            diagnosis = execute_action(ControllerAction("DIAGNOSE_FAILURE", "worker_research_failed"), bundle, trace, handlers=handlers)
            bundle = update_blackboard(bundle, diagnosis)
            executed.append({"action": ControllerAction("DIAGNOSE_FAILURE", "worker_research_failed").to_dict(), "result": diagnosis})
            break
        if action.action_type not in {"EXTRACT_EVIDENCE", "AUDIT_ROUTE"} and not result.get("accepted"):
            break

    trace.final_route_status = bundle.route_status.value
    return {
        "schema_version": "controller_loop_result.v1",
        "case_id": bundle.case_id,
        "observations": observations,
        "executed": executed,
        "stop_decision": stop_or_continue(bundle, trace, budget),
        "trace": trace.to_dict(),
        "case_bundle": bundle.to_dict(),
    }


def run_worker_action(task: WorkerTask) -> dict[str, Any]:
    record = run_codex_worker(task, use_codex_cli=True)
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
    if action.action_type in {"RESEARCH_TARGET", "RESEARCH_STUCK_NODE"}:
        return _run_default_research_worker(action, bundle, trace)
    if action.action_type == "EXTRACT_EVIDENCE":
        return _extract_evidence_default(action, bundle)
    if action.action_type == "VALIDATE_EVIDENCE":
        return _validate_evidence_default(action, bundle)
    if action.action_type == "MINE_STRATEGIC_DISCONNECTION":
        return _mine_strategic_disconnection_default(action, bundle)
    if action.action_type == "DIAGNOSE_FAILURE":
        return _diagnose_failure_default(action, bundle)
    if action.action_type == "DESIGN_CONDITIONS":
        return _design_conditions_default(action, bundle)
    if action.action_type == "COMPILE_STRATEGIC_OPERATOR":
        return _compile_strategic_operator_default(action, bundle)
    if action.action_type == "RUN_GUIDED_CHEMENZY":
        return _run_guided_chemenzy_default(action, bundle)
    if action.action_type == "AUDIT_ROUTE":
        return _audit_route_default(action, bundle)
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


def _run_default_research_worker(action: ControllerAction, bundle: CaseBundle, trace: ControllerTrace) -> dict[str, Any]:
    payload = dict(action.payload or {})
    task_payload = dict(payload.get("worker_task") or {})
    if task_payload:
        task = _worker_task_from_action_payload(task_payload, action, bundle, trace)
    else:
        task_type = "target_research" if action.action_type == "RESEARCH_TARGET" else "stuck_node_research"
        task = WorkerTask(
            task_id=str(payload.get("task_id") or f"{bundle.case_id}:{task_type}:{trace.codex_worker_runs}"),
            case_id=bundle.case_id,
            task_type=task_type,
            required_artifact_type=str(payload.get("required_artifact_type") or "ResearchReport"),
            input_refs=[str(item) for item in payload.get("input_refs") or _default_worker_input_refs(bundle)],
            allowed_tools=[str(item) for item in payload.get("allowed_tools") or ["web_search", "local_search"]],
            budget=WorkerBudget(
                timeout_s=float(payload.get("timeout_s") or 60.0),
                max_output_bytes=int(payload.get("max_output_bytes") or 200_000),
                max_tool_calls=int(payload.get("max_tool_calls") or 8),
                max_worker_runs=1,
            ),
            objective=str(payload.get("objective") or action.reason or action.action_type),
            allowed_workdir=str(payload.get("allowed_workdir") or "."),
            dry_run=bool(payload.get("dry_run", False)),
        )
    record = run_codex_worker(task, use_codex_cli=True)
    return {
        "accepted": record.status == "accepted_draft",
        "action_type": action.action_type,
        "case_id": bundle.case_id,
        "worker_trace": record.to_dict(),
        "output_artifact": record.output_artifact,
        "reasons": list(record.output_validation.get("reasons") or []),
    }


def _worker_task_from_action_payload(
    task_payload: dict[str, Any],
    action: ControllerAction,
    bundle: CaseBundle,
    trace: ControllerTrace,
) -> WorkerTask:
    budget = dict(task_payload.get("budget") or {})
    task_type = str(task_payload.get("task_type") or ("target_research" if action.action_type == "RESEARCH_TARGET" else "stuck_node_research"))
    return WorkerTask(
        task_id=str(task_payload.get("task_id") or f"{bundle.case_id}:{task_type}:{trace.codex_worker_runs}"),
        case_id=str(task_payload.get("case_id") or bundle.case_id),
        task_type=task_type,
        required_artifact_type=str(task_payload.get("required_artifact_type") or "ResearchReport"),
        input_refs=[str(item) for item in task_payload.get("input_refs") or _default_worker_input_refs(bundle)],
        allowed_tools=[str(item) for item in task_payload.get("allowed_tools") or ["web_search", "local_search"]],
        budget=WorkerBudget(
            timeout_s=float(budget.get("timeout_s") or task_payload.get("timeout_s") or 60.0),
            max_output_bytes=int(budget.get("max_output_bytes") or task_payload.get("max_output_bytes") or 200_000),
            max_tool_calls=int(budget.get("max_tool_calls") or task_payload.get("max_tool_calls") or 8),
            max_worker_runs=int(budget.get("max_worker_runs") or 1),
        ),
        objective=str(task_payload.get("objective") or action.reason or action.action_type),
        allowed_workdir=str(task_payload.get("allowed_workdir") or "."),
        dry_run=bool(task_payload.get("dry_run", False)),
    )


def _default_worker_input_refs(bundle: CaseBundle) -> list[str]:
    refs = [artifact.artifact_id for artifact in bundle.accepted_artifacts()]
    refs.extend(event.failure_id for event in bundle.failure_events)
    return refs or [bundle.case_id]


def _extract_evidence_default(action: ControllerAction, bundle: CaseBundle) -> dict[str, Any]:
    evidence: list[Any] = []
    for artifact in bundle.artifacts:
        if artifact.validation_status not in {"draft", "accepted"}:
            continue
        if artifact.artifact_type == "EvidenceCard":
            evidence.append(artifact.payload)
        elif artifact.artifact_type == "EvidenceCardList" and isinstance(artifact.payload, list):
            evidence.extend(artifact.payload)
    return {
        "accepted": True,
        "action_type": action.action_type,
        "case_id": bundle.case_id,
        "evidence_count": len(evidence),
        "evidence": evidence,
    }


def _validate_evidence_default(action: ControllerAction, bundle: CaseBundle) -> dict[str, Any]:
    payload = dict(action.payload or {})
    artifact_id = str(payload.get("artifact_id") or "")
    selected = None
    if artifact_id:
        selected = next((artifact for artifact in bundle.artifacts if artifact.artifact_id == artifact_id), None)
    if selected is None:
        selected = next((artifact for artifact in reversed(bundle.artifacts) if artifact.artifact_type == "EvidenceCard"), None)
    if selected is None:
        selected = next((artifact for artifact in reversed(bundle.artifacts) if artifact.artifact_type == "EvidenceCardList"), None)
    if selected is None:
        return {
            "accepted": False,
            "action_type": action.action_type,
            "case_id": bundle.case_id,
            "reasons": ["missing_evidence_artifact"],
        }
    artifact = _evidence_artifact_from_record(selected, bundle.case_id)
    if artifact.get("artifact_type") != "EvidenceCard":
        artifact = {
            "schema_version": "evidence_card_artifact.v1",
            "artifact_id": selected.artifact_id,
            "artifact_type": "EvidenceCard",
            "case_id": bundle.case_id,
            "source": selected.source,
            "input_refs": selected.input_refs,
            "evidence_refs": selected.evidence_refs,
            "validation_status": "draft",
            "payload": artifact,
        }
    validation = validate_typed_artifact(artifact)
    return {
        "accepted": bool(validation.get("accepted")),
        "action_type": action.action_type,
        "case_id": bundle.case_id,
        "validation": validation,
        "output_artifact": artifact,
        "reasons": list(validation.get("reasons") or []),
    }


def _evidence_artifact_from_record(record: ArtifactRecord, case_id: str) -> dict[str, Any]:
    payload = record.payload
    if record.artifact_type == "EvidenceCardList" and isinstance(payload, list):
        first_card = next((dict(item) for item in payload if isinstance(item, dict)), {})
        evidence_id = str(first_card.get("evidence_id") or "evidence")
        return {
            "schema_version": "evidence_card_artifact.v1",
            "artifact_id": f"{record.artifact_id}:{evidence_id}",
            "artifact_type": "EvidenceCard",
            "case_id": case_id,
            "source": record.source,
            "input_refs": [record.artifact_id],
            "evidence_refs": record.evidence_refs,
            "validation_status": "draft",
            "payload": first_card,
        }
    return dict(payload or {})


def _mine_strategic_disconnection_default(action: ControllerAction, bundle: CaseBundle) -> dict[str, Any]:
    evidence_refs = _accepted_evidence_ref_list(bundle)
    if not evidence_refs:
        return {
            "accepted": False,
            "action_type": action.action_type,
            "case_id": bundle.case_id,
            "reasons": ["missing_validated_evidence"],
        }
    artifact = {
        "schema_version": "strategic_disconnection_card.v1",
        "artifact_id": f"{bundle.case_id}:controller_disconnection",
        "artifact_type": "StrategicDisconnectionCard",
        "case_id": bundle.case_id,
        "source": "codex_controller",
        "input_refs": evidence_refs,
        "evidence_refs": evidence_refs,
        "validation_status": "draft",
        "payload": {
            "card_id": f"{bundle.case_id}_controller_disconnection",
            "case_id": bundle.case_id,
            "candidate_kind": "controller_mined_disconnection",
            "retrosynthetic_move": "validated_evidence_guided_frontier",
            "evidence_refs": evidence_refs,
            "limitations": ["controller default does not inject raw reactions"],
        },
    }
    validation = _validate_artifact_for_bundle(bundle, artifact)
    return {
        "accepted": bool(validation.get("accepted")),
        "action_type": action.action_type,
        "case_id": bundle.case_id,
        "output_artifact": artifact,
        "validation": validation,
        "reasons": list(validation.get("reasons") or []),
    }


def _diagnose_failure_default(action: ControllerAction, bundle: CaseBundle) -> dict[str, Any]:
    events = [event.to_dict() for event in bundle.failure_events]
    reason = action.reason or (events[0]["reason"] if events else "route_unresolved")
    artifact = {
        "schema_version": "failure_diagnosis.v1",
        "artifact_id": f"{bundle.case_id}:failure_diagnosis",
        "artifact_type": "FailureDiagnosis",
        "case_id": bundle.case_id,
        "source": "codex_controller",
        "input_refs": [event.get("failure_id", "") for event in events] or [bundle.case_id],
        "evidence_refs": [],
        "validation_status": "draft",
        "payload": {
            "failure_mode": reason,
            "failure_events": events,
            "next_actions": ["research_or_extend_frontier"],
        },
    }
    validation = validate_typed_artifact(artifact)
    return {
        "accepted": bool(validation.get("accepted")),
        "action_type": action.action_type,
        "case_id": bundle.case_id,
        "output_artifact": artifact,
        "validation": validation,
        "reasons": list(validation.get("reasons") or []),
    }


def _design_conditions_default(action: ControllerAction, bundle: CaseBundle) -> dict[str, Any]:
    payload = dict(action.payload or {})
    step_id = str(payload.get("step_id") or "step_1")
    evidence_refs = [artifact.artifact_id for artifact in bundle.accepted_artifacts("EvidenceCard")]
    artifact = {
        "schema_version": "condition_candidate.v1",
        "artifact_id": f"{bundle.case_id}:{step_id}:condition_candidate",
        "artifact_type": "ConditionCandidate",
        "case_id": bundle.case_id,
        "source": "codex_controller",
        "input_refs": [step_id],
        "evidence_refs": evidence_refs,
        "validation_status": "draft",
        "payload": {
            "step_id": step_id,
            "source_type": "exact" if evidence_refs else "unknown",
            "condition_status": "evidence_backed" if evidence_refs else "gap",
            "evidence_refs": evidence_refs,
            "risk_flags": [],
            "confidence": "medium",
        },
    }
    validation = validate_typed_artifact(artifact)
    return {
        "accepted": bool(validation.get("accepted")),
        "action_type": action.action_type,
        "case_id": bundle.case_id,
        "output_artifact": artifact,
        "validation": validation,
        "reasons": list(validation.get("reasons") or []),
    }


def _compile_strategic_operator_default(action: ControllerAction, bundle: CaseBundle) -> dict[str, Any]:
    evidence_refs = _accepted_evidence_ref_list(bundle)
    disconnection_refs = sorted(_accepted_disconnection_refs(bundle))
    if not evidence_refs:
        return {
            "accepted": False,
            "action_type": action.action_type,
            "case_id": bundle.case_id,
            "reasons": ["missing_validated_evidence"],
        }
    artifact = {
        "schema_version": "strategic_operator_artifact.v1",
        "artifact_id": f"{bundle.case_id}:strategic_operator",
        "artifact_type": "StrategicOperator",
        "case_id": bundle.case_id,
        "source": "codex_controller",
        "input_refs": [*evidence_refs, *disconnection_refs],
        "evidence_refs": evidence_refs,
        "validation_status": "draft",
        "payload": {
            "operator_id": f"{bundle.case_id}_controller_operator",
            "case_id": bundle.case_id,
            "evidence_refs": evidence_refs,
            "terminal_blacklist": [],
            "anchor_whitelist": [],
            "preferred_subgoal": {},
            "source_budget": {"max_external_steps": 1},
            "rerun_reason": action.reason or "validated_evidence_available",
            "budget": {"max_reruns": 1, "max_iterations": 16, "max_depth": 6, "expansion_topk": 50},
            "input_artifact_refs": [*evidence_refs, *disconnection_refs],
            "mode": "literature_guided_rerun",
            "schema_version": "strategic_operator.v1",
        },
    }
    validation = _validate_artifact_for_bundle(bundle, artifact)
    return {
        "accepted": bool(validation.get("accepted")),
        "action_type": action.action_type,
        "case_id": bundle.case_id,
        "output_artifact": artifact,
        "validation": validation,
        "reasons": list(validation.get("reasons") or []),
    }


def _run_guided_chemenzy_default(action: ControllerAction, bundle: CaseBundle) -> dict[str, Any]:
    policy_id = str(action.payload.get("policy_id") or f"{bundle.case_id}:guided_rerun_request")
    return {
        "accepted": True,
        "action_type": action.action_type,
        "case_id": bundle.case_id,
        "rerun_history": {
            "schema_version": "guided_rerun_history.v1",
            "policy_id": policy_id,
            "route_status_before": bundle.route_status.value,
            "requested": True,
            "executed": False,
            "reason": action.reason or "controller_default_guided_rerun_request",
        },
        "final_route_status": bundle.route_status.value,
    }


def _audit_route_default(action: ControllerAction, bundle: CaseBundle) -> dict[str, Any]:
    package = _latest_payload(bundle, "HybridRoutePackage")
    validation = _latest_payload(bundle, "RoutePackageValidation")
    if not package:
        package = {
            "case_id": bundle.case_id,
            "route_status": bundle.route_status.value,
            "frontier": {},
            "literature_evidence_refs": _accepted_evidence_ref_list(bundle),
            "literature_candidates": [],
        }
    report = audit_route_package(package, validation=validation)
    payload = report.to_dict()
    artifact = {
        "schema_version": "route_audit_report.v1",
        "artifact_id": f"{bundle.case_id}:controller_route_audit",
        "artifact_type": "RouteAuditReport",
        "case_id": bundle.case_id,
        "source": "codex_controller",
        "input_refs": [
            artifact.artifact_id
            for artifact in bundle.accepted_artifacts()
            if artifact.artifact_type in {"HybridRoutePackage", "RoutePackageValidation", "StrategicOperator", "GuidedRerunHistory"}
        ] or [bundle.case_id],
        "evidence_refs": _accepted_evidence_ref_list(bundle),
        "validation_status": "draft",
        "payload": payload,
    }
    validation_result = validate_typed_artifact(artifact)
    return {
        "accepted": bool(validation_result.get("accepted")),
        "action_type": action.action_type,
        "case_id": bundle.case_id,
        "output_artifact": artifact,
        "validation": validation_result,
        "final_route_status": payload.get("route_status") or bundle.route_status.value,
        "reasons": list(validation_result.get("reasons") or []),
    }


def _worker_record_accepted(worker_trace: dict[str, Any]) -> bool:
    return bool(worker_trace.get("status") == "accepted_draft" and worker_trace.get("output_validation", {}).get("accepted"))


def _append_controller_artifact(bundle: CaseBundle, artifact: ArtifactRecord) -> None:
    bundle.append_artifact(artifact)


def _accepted_evidence_refs(bundle: CaseBundle) -> set[str]:
    refs = set(_accepted_evidence_ref_list(bundle))
    return refs


def _accepted_evidence_ref_list(bundle: CaseBundle) -> list[str]:
    refs: list[str] = []
    for artifact in bundle.accepted_artifacts("EvidenceCard"):
        refs.append(artifact.artifact_id)
        if isinstance(artifact.payload, dict):
            payload = dict(artifact.payload.get("payload") or artifact.payload)
            if payload.get("evidence_id"):
                refs.append(str(payload["evidence_id"]))
    for artifact in bundle.accepted_artifacts("EvidenceCardList"):
        refs.append(artifact.artifact_id)
        for card in artifact.payload or []:
            if isinstance(card, dict) and card.get("evidence_id"):
                refs.append(str(card["evidence_id"]))
    return sorted(set(refs))


def _accepted_disconnection_refs(bundle: CaseBundle) -> set[str]:
    refs = {artifact.artifact_id for artifact in bundle.accepted_artifacts("StrategicDisconnectionCard")}
    for artifact in bundle.accepted_artifacts("StrategicDisconnectionCard"):
        if isinstance(artifact.payload, dict):
            payload = dict(artifact.payload.get("payload") or artifact.payload)
            if payload.get("card_id"):
                refs.add(str(payload["card_id"]))
    for artifact in bundle.accepted_artifacts("StrategicDisconnectionCardList"):
        refs.add(artifact.artifact_id)
        for card in artifact.payload or []:
            if isinstance(card, dict) and card.get("card_id"):
                refs.add(str(card["card_id"]))
            elif isinstance(card, dict) and card.get("artifact_id"):
                refs.add(str(card["artifact_id"]))
    return refs


def _unique_artifact_id(bundle: CaseBundle, base: str) -> str:
    base = str(base or "artifact").replace("/", "_")
    existing = {artifact.artifact_id for artifact in bundle.artifacts}
    if base not in existing:
        return base
    idx = 2
    while f"{base}:{idx}" in existing:
        idx += 1
    return f"{base}:{idx}"


def _latest_payload(bundle: CaseBundle, artifact_type: str) -> dict[str, Any]:
    for artifact in reversed(bundle.accepted_artifacts(artifact_type)):
        if isinstance(artifact.payload, dict):
            return dict(artifact.payload)
    return {}


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

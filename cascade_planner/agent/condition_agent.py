"""Condition worker task, candidate, and condition-audit helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cascade_planner.agent.codex_worker import WorkerBudget, WorkerTask

CONDITION_WORKER_TASK_SCHEMA = "condition_worker_task.v1"
CONDITION_CANDIDATE_SCHEMA = "condition_candidate.v1"
CONDITION_AUDIT_SCHEMA = "condition_audit.v1"
CONDITION_BENCHMARK_REPORT_SCHEMA = "condition_benchmark_report.v1"

ALLOWED_CONDITION_SOURCE_TYPES = {"exact", "analog", "template", "model-only", "unknown"}
ALLOWED_CONDITION_STATUSES = {"evidence_backed", "analog_scope_gap", "feasibility_hint", "gap"}


@dataclass
class ConditionWorkerTask:
    task_id: str
    case_id: str
    step_id: str
    route_step: dict[str, Any]
    required_artifact_type: str = "ConditionCandidate"
    allowed_source_types: list[str] = field(default_factory=lambda: ["exact", "analog", "model-only"])
    evidence_refs: list[str] = field(default_factory=list)
    objective: str = "extract bounded reaction conditions for one route step"
    schema_version: str = CONDITION_WORKER_TASK_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionCandidate:
    step_id: str
    source_type: str = "unknown"
    condition_status: str = "gap"
    reagent: str = ""
    catalyst: str = ""
    enzyme: str = ""
    solvent: str = ""
    temperature: str = ""
    ph: str = ""
    buffer: str = ""
    atmosphere: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    scope_gap: str = ""
    risk_flags: list[str] = field(default_factory=list)
    confidence: str = "medium"
    schema_version: str = CONDITION_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if self.condition_status == "gap":
            if self.source_type == "exact" and self.evidence_refs:
                self.condition_status = "evidence_backed"
            elif self.source_type == "analog" and self.scope_gap:
                self.condition_status = "analog_scope_gap"
            elif self.source_type == "model-only" and any(
                [self.reagent, self.catalyst, self.enzyme, self.solvent, self.temperature, self.ph, self.buffer]
            ):
                self.condition_status = "feasibility_hint"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_condition_candidate(candidate_or_data: ConditionCandidate | dict[str, Any]) -> dict[str, Any]:
    candidate = candidate_or_data if isinstance(candidate_or_data, ConditionCandidate) else condition_candidate_from_dict(candidate_or_data)
    reasons: list[str] = []
    if candidate.schema_version != CONDITION_CANDIDATE_SCHEMA:
        reasons.append("invalid_condition_candidate_schema")
    if not candidate.step_id:
        reasons.append("missing_step_id")
    if candidate.source_type not in ALLOWED_CONDITION_SOURCE_TYPES:
        reasons.append("invalid_condition_source_type")
    if candidate.source_type == "exact" and not candidate.evidence_refs:
        reasons.append("exact_condition_missing_evidence")
    if candidate.source_type == "analog" and not candidate.scope_gap:
        reasons.append("analog_condition_missing_scope_gap")
    if candidate.source_type == "model-only" and candidate.condition_status != "feasibility_hint":
        reasons.append("model_only_condition_must_be_hint")
    if candidate.condition_status not in ALLOWED_CONDITION_STATUSES:
        reasons.append("invalid_condition_status")
    if not any([candidate.reagent, candidate.catalyst, candidate.enzyme, candidate.solvent, candidate.temperature, candidate.ph]):
        reasons.append("condition_gap")
    if not isinstance(candidate.risk_flags, list):
        reasons.append("risk_flags_not_list")
    return {
        "schema_version": "condition_candidate_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "step_id": candidate.step_id,
    }


def condition_candidate_from_dict(data: dict[str, Any]) -> ConditionCandidate:
    return ConditionCandidate(
        step_id=str(data.get("step_id") or ""),
        source_type=str(data.get("source_type") or data.get("condition_source_type") or "unknown"),
        condition_status=str(data.get("condition_status") or _default_condition_status(data)),
        reagent=str(data.get("reagent") or ""),
        catalyst=str(data.get("catalyst") or ""),
        enzyme=str(data.get("enzyme") or ""),
        solvent=str(data.get("solvent") or ""),
        temperature=str(data.get("temperature") or data.get("temperature_c") or ""),
        ph=str(data.get("ph") or data.get("pH") or ""),
        buffer=str(data.get("buffer") or ""),
        atmosphere=str(data.get("atmosphere") or ""),
        evidence_refs=[str(ref) for ref in data.get("evidence_refs") or []],
        scope_gap=str(data.get("scope_gap") or data.get("analog_scope_gap") or ""),
        risk_flags=[str(flag) for flag in data.get("risk_flags") or data.get("hazard_flags") or []],
        confidence=str(data.get("confidence") or "medium"),
        schema_version=str(data.get("schema_version") or CONDITION_CANDIDATE_SCHEMA),
    )


def audit_conditions(candidates: list[ConditionCandidate | dict[str, Any]]) -> dict[str, Any]:
    rows = [
        candidate if isinstance(candidate, ConditionCandidate) else condition_candidate_from_dict(candidate)
        for candidate in candidates
    ]
    validations = [validate_condition_candidate(candidate) for candidate in rows]
    accepted = [candidate for candidate, result in zip(rows, validations) if result["accepted"]]
    risk_flags = sorted({flag for candidate in rows for flag in candidate.risk_flags})
    route_risk = "high" if any(flag in {"hazard", "incompatible", "unsafe", "extreme_temperature"} for flag in risk_flags) else "ok"
    if not rows or len(accepted) < len(rows) or any("condition_gap" in result["reasons"] for result in validations):
        route_risk = "gap" if route_risk == "ok" else route_risk
    source_order = {"exact": 0, "analog": 1, "template": 2, "model-only": 3, "unknown": 4}
    best_source = min((candidate.source_type for candidate in accepted), key=lambda item: source_order.get(item, 99), default="unknown")
    return {
        "schema_version": CONDITION_AUDIT_SCHEMA,
        "candidate_count": len(rows),
        "accepted_count": len(accepted),
        "validations": validations,
        "best_source_type": best_source,
        "route_risk": route_risk,
        "condition_gap": route_risk == "gap",
        "risk_flags": risk_flags,
        "audit_downgrade": route_risk in {"gap", "high"},
    }


def condition_worker_task_from_route_step(
    *,
    case_id: str,
    route_step: dict[str, Any],
    evidence_refs: list[str] | None = None,
    timeout_s: float = 60.0,
) -> ConditionWorkerTask:
    step_id = str(route_step.get("step_id") or route_step.get("id") or "step_1")
    return ConditionWorkerTask(
        task_id=f"{case_id}:{step_id}:condition_research",
        case_id=case_id,
        step_id=step_id,
        route_step=dict(route_step or {}),
        evidence_refs=[str(ref) for ref in evidence_refs or []],
        objective=(
            "Extract exact or analog reaction conditions for this route step. "
            "Model-only output is only a feasibility hint and cannot be an executable procedure."
        ),
    )


def condition_worker_task_to_worker_task(task: ConditionWorkerTask, *, timeout_s: float = 60.0) -> WorkerTask:
    return WorkerTask(
        task_id=task.task_id,
        case_id=task.case_id,
        task_type="condition_research",
        required_artifact_type=task.required_artifact_type,
        input_refs=[task.step_id, *task.evidence_refs],
        allowed_tools=["web_search", "local_search"],
        budget=WorkerBudget(timeout_s=timeout_s, max_output_bytes=80_000, max_tool_calls=6, max_worker_runs=1),
        objective=task.objective,
        dry_run=False,
    )


def benchmark_condition_candidates(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    gaps = 0
    risky_expected = 0
    risky_true_positive = 0
    audit_downgrades = 0
    for idx, case in enumerate(cases, start=1):
        candidates = [condition_candidate_from_dict(item) for item in case.get("candidates") or []]
        audit = audit_conditions(candidates)
        expected_gap = bool(case.get("expected_gap"))
        expected_risky = bool(case.get("expected_risky"))
        has_gap = bool(audit.get("condition_gap"))
        has_risk = bool(set(audit.get("risk_flags") or []).intersection({"hazard", "incompatible", "unsafe", "extreme_temperature"}))
        if has_gap:
            gaps += 1
        if expected_risky:
            risky_expected += 1
            if has_risk:
                risky_true_positive += 1
        if audit.get("audit_downgrade"):
            audit_downgrades += 1
        rows.append({
            "case_id": str(case.get("case_id") or idx),
            "expected_gap": expected_gap,
            "observed_gap": has_gap,
            "expected_risky": expected_risky,
            "observed_risky": has_risk,
            "audit": audit,
        })
    total = len(cases)
    return {
        "schema_version": CONDITION_BENCHMARK_REPORT_SCHEMA,
        "case_count": total,
        "gap_rate": (gaps / total) if total else 0.0,
        "risky_flag_precision": (risky_true_positive / risky_expected) if risky_expected else 1.0,
        "audit_downgrade_count": audit_downgrades,
        "rows": rows,
    }


def _default_condition_status(data: dict[str, Any]) -> str:
    source_type = str(data.get("source_type") or data.get("condition_source_type") or "unknown")
    if source_type == "exact":
        return "evidence_backed"
    if source_type == "analog":
        return "analog_scope_gap"
    if source_type == "model-only":
        return "feasibility_hint"
    return "gap"

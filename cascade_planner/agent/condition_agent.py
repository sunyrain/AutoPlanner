"""Minimal condition candidate and condition-audit helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CONDITION_CANDIDATE_SCHEMA = "condition_candidate.v1"
CONDITION_AUDIT_SCHEMA = "condition_audit.v1"

ALLOWED_CONDITION_SOURCE_TYPES = {"exact", "analog", "template", "model-only", "unknown"}


@dataclass
class ConditionCandidate:
    step_id: str
    source_type: str = "unknown"
    reagent: str = ""
    catalyst: str = ""
    enzyme: str = ""
    solvent: str = ""
    temperature: str = ""
    ph: str = ""
    buffer: str = ""
    atmosphere: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    confidence: str = "medium"
    schema_version: str = CONDITION_CANDIDATE_SCHEMA

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
        reagent=str(data.get("reagent") or ""),
        catalyst=str(data.get("catalyst") or ""),
        enzyme=str(data.get("enzyme") or ""),
        solvent=str(data.get("solvent") or ""),
        temperature=str(data.get("temperature") or data.get("temperature_c") or ""),
        ph=str(data.get("ph") or data.get("pH") or ""),
        buffer=str(data.get("buffer") or ""),
        atmosphere=str(data.get("atmosphere") or ""),
        evidence_refs=[str(ref) for ref in data.get("evidence_refs") or []],
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
    }

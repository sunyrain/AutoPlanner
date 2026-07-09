"""Validated literature route segments for recursive planning.

The segment layer keeps multi-step literature information as audited planning
material.  It does not inject raw reaction strings into ChemEnzy; every edge is
checked as structured product/reactant/condition/source data before recursive
unroll is allowed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.agent.condition_agent import audit_conditions


RDLogger.DisableLog("rdApp.*")

LITERATURE_ROUTE_SEGMENT_CARD_SCHEMA = "literature_route_segment_card.v1"
SEGMENT_STEP_CANDIDATE_SCHEMA = "segment_step_candidate.v1"
SEGMENT_VALIDATION_SCHEMA = "literature_route_segment_validation.v1"
SEGMENT_UNROLL_TRACE_SCHEMA = "literature_route_segment_unroll_trace.v1"

ALLOWED_SEGMENT_RELATIONS = {"exact", "analog", "mismatch"}


@dataclass
class SegmentStepCandidate:
    step_id: str
    product_smiles: str
    reactant_smiles: list[str]
    evidence_refs: list[str]
    source_ref: str
    relation_type: str = "exact"
    applicability: dict[str, Any] = field(default_factory=dict)
    condition_candidate: dict[str, Any] = field(default_factory=dict)
    scope_gap: str = ""
    schema_version: str = SEGMENT_STEP_CANDIDATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiteratureRouteSegmentCard:
    segment_id: str
    case_id: str
    target_smiles: str
    steps: list[SegmentStepCandidate]
    evidence_refs: list[str]
    source_title: str = ""
    source_type: str = "literature"
    trigger_reasons: list[str] = field(default_factory=list)
    validation_status: str = "draft"
    schema_version: str = LITERATURE_ROUTE_SEGMENT_CARD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        return data


def segment_step_from_dict(data: dict[str, Any]) -> SegmentStepCandidate:
    return SegmentStepCandidate(
        step_id=str(data.get("step_id") or ""),
        product_smiles=str(data.get("product_smiles") or ""),
        reactant_smiles=[str(item) for item in data.get("reactant_smiles") or []],
        evidence_refs=[str(item) for item in data.get("evidence_refs") or []],
        source_ref=str(data.get("source_ref") or ""),
        relation_type=str(data.get("relation_type") or "exact"),
        applicability=dict(data.get("applicability") or {}),
        condition_candidate=dict(data.get("condition_candidate") or {}),
        scope_gap=str(data.get("scope_gap") or ""),
        schema_version=str(data.get("schema_version") or SEGMENT_STEP_CANDIDATE_SCHEMA),
    )


def literature_route_segment_from_dict(data: dict[str, Any]) -> LiteratureRouteSegmentCard:
    return LiteratureRouteSegmentCard(
        segment_id=str(data.get("segment_id") or ""),
        case_id=str(data.get("case_id") or ""),
        target_smiles=str(data.get("target_smiles") or ""),
        steps=[segment_step_from_dict(item) for item in data.get("steps") or []],
        evidence_refs=[str(item) for item in data.get("evidence_refs") or []],
        source_title=str(data.get("source_title") or ""),
        source_type=str(data.get("source_type") or "literature"),
        trigger_reasons=[str(item) for item in data.get("trigger_reasons") or []],
        validation_status=str(data.get("validation_status") or "draft"),
        schema_version=str(data.get("schema_version") or LITERATURE_ROUTE_SEGMENT_CARD_SCHEMA),
    )


def validate_segment_step(step_or_data: SegmentStepCandidate | dict[str, Any]) -> dict[str, Any]:
    step = step_or_data if isinstance(step_or_data, SegmentStepCandidate) else segment_step_from_dict(step_or_data)
    reasons: list[str] = []
    if step.schema_version != SEGMENT_STEP_CANDIDATE_SCHEMA:
        reasons.append("invalid_segment_step_schema")
    if not step.step_id:
        reasons.append("missing_step_id")
    if step.relation_type not in ALLOWED_SEGMENT_RELATIONS:
        reasons.append("invalid_relation_type")
    if step.relation_type == "analog" and not step.scope_gap:
        reasons.append("analog_step_missing_scope_gap")
    if step.relation_type == "mismatch":
        reasons.append("segment_step_mismatch")
    if not step.evidence_refs:
        reasons.append("missing_evidence_refs")
    if not step.source_ref:
        reasons.append("missing_source_ref")
    if not _valid_smiles(step.product_smiles):
        reasons.append("invalid_product_smiles")
    if not step.reactant_smiles:
        reasons.append("missing_reactant_smiles")
    for smiles in step.reactant_smiles:
        if not _valid_smiles(smiles):
            reasons.append("invalid_reactant_smiles")
            break
    app_reasons = _applicability_reasons(step)
    reasons.extend(app_reasons)
    atom_reasons = _atom_accounting_reasons(step)
    reasons.extend(atom_reasons)
    condition_audit = audit_conditions([step.condition_candidate] if step.condition_candidate else [])
    if condition_audit.get("route_risk") == "gap":
        reasons.append("condition_gap")
    if condition_audit.get("route_risk") == "high":
        reasons.append("condition_high_risk")
    return {
        "schema_version": "segment_step_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "step_id": step.step_id,
        "condition_audit": condition_audit,
    }


def validate_literature_route_segment(
    segment_or_data: LiteratureRouteSegmentCard | dict[str, Any],
) -> dict[str, Any]:
    segment = (
        segment_or_data
        if isinstance(segment_or_data, LiteratureRouteSegmentCard)
        else literature_route_segment_from_dict(segment_or_data)
    )
    reasons: list[str] = []
    if segment.schema_version != LITERATURE_ROUTE_SEGMENT_CARD_SCHEMA:
        reasons.append("invalid_segment_schema")
    if not segment.segment_id:
        reasons.append("missing_segment_id")
    if not segment.case_id:
        reasons.append("missing_case_id")
    if not _valid_smiles(segment.target_smiles):
        reasons.append("invalid_target_smiles")
    if not segment.evidence_refs:
        reasons.append("missing_segment_evidence_refs")
    if not (2 <= len(segment.steps) <= 5):
        reasons.append("segment_step_count_out_of_range")
    step_results = [validate_segment_step(step) for step in segment.steps]
    for result in step_results:
        reasons.extend(result.get("reasons") or [])
    if any(step.relation_type == "analog" for step in segment.steps):
        reasons.append("analog_segment_not_executable")
    if any(step.relation_type == "mismatch" for step in segment.steps):
        reasons.append("mismatch_segment_not_executable")
    accepted = not reasons
    return {
        "schema_version": SEGMENT_VALIDATION_SCHEMA,
        "accepted": accepted,
        "reasons": sorted(set(reasons)),
        "segment_id": segment.segment_id,
        "step_results": step_results,
        "allowed_for_recursive_unroll": accepted,
        "validation_status": "validated" if accepted else "rejected",
    }


def unroll_literature_route_segment(
    segment_or_data: LiteratureRouteSegmentCard | dict[str, Any],
    *,
    max_steps: int = 5,
    native_solved_audit_passed: bool = False,
) -> dict[str, Any]:
    segment = (
        segment_or_data
        if isinstance(segment_or_data, LiteratureRouteSegmentCard)
        else literature_route_segment_from_dict(segment_or_data)
    )
    trace = {
        "schema_version": SEGMENT_UNROLL_TRACE_SCHEMA,
        "segment_id": segment.segment_id,
        "case_id": segment.case_id,
        "expanded_steps": [],
        "rejected_steps": [],
        "final_status": "not_started",
        "stop_reason": "",
        "false_uplift_blocked": False,
    }
    if native_solved_audit_passed:
        trace["final_status"] = "skipped"
        trace["stop_reason"] = "native_solved_audit_passed"
        trace["false_uplift_blocked"] = True
        return trace
    if max_steps <= 0:
        trace["final_status"] = "stopped"
        trace["stop_reason"] = "budget_exhausted"
        return trace

    for step in segment.steps:
        if len(trace["expanded_steps"]) >= max_steps:
            trace["final_status"] = "partial"
            trace["stop_reason"] = "budget_exhausted"
            return trace
        result = validate_segment_step(step)
        if not result["accepted"]:
            trace["rejected_steps"].append({"step": step.to_dict(), "validation": result})
            trace["final_status"] = "partial" if trace["expanded_steps"] else "rejected"
            trace["stop_reason"] = _stop_reason_from_step_validation(result)
            return trace
        trace["expanded_steps"].append({
            "step_id": step.step_id,
            "product_smiles": step.product_smiles,
            "reactant_smiles": list(step.reactant_smiles),
            "evidence_refs": list(step.evidence_refs),
            "source_ref": step.source_ref,
            "validation": result,
        })
    segment_validation = validate_literature_route_segment(segment)
    if not segment_validation["accepted"]:
        trace["final_status"] = "partial" if trace["expanded_steps"] else "rejected"
        trace["stop_reason"] = "audit_failed"
        trace["segment_validation"] = segment_validation
        return trace
    trace["final_status"] = "segment_unrolled"
    trace["stop_reason"] = "complete"
    trace["segment_validation"] = segment_validation
    return trace


def _applicability_reasons(step: SegmentStepCandidate) -> list[str]:
    app = dict(step.applicability or {})
    reasons: list[str] = []
    if app.get("status") not in {"passed", "exact"}:
        reasons.append("applicability_failed")
    if not bool(app.get("product_reconstruction_passed")):
        reconstructed = str(app.get("reconstructed_product_smiles") or "")
        if not reconstructed or _canonical(reconstructed) != _canonical(step.product_smiles):
            reasons.append("product_reconstruction_failed")
    if bool(app.get("audit_failed")):
        reasons.append("segment_step_audit_failed")
    return reasons


def _atom_accounting_reasons(step: SegmentStepCandidate) -> list[str]:
    product_counts = _element_counts(step.product_smiles)
    reactant_counts: dict[int, int] = {}
    for smiles in step.reactant_smiles:
        for atomic_num, count in _element_counts(smiles).items():
            reactant_counts[atomic_num] = reactant_counts.get(atomic_num, 0) + count
    for atomic_num, product_count in product_counts.items():
        if reactant_counts.get(atomic_num, 0) < product_count:
            return ["atom_accounting_failed"]
    return []


def _stop_reason_from_step_validation(result: dict[str, Any]) -> str:
    reasons = set(result.get("reasons") or [])
    if "segment_step_mismatch" in reasons:
        return "mismatch"
    if "condition_high_risk" in reasons:
        return "high_risk_condition"
    if "segment_step_audit_failed" in reasons:
        return "audit_failed"
    return "validation_failed"


def _valid_smiles(smiles: str) -> bool:
    return bool(smiles) and Chem.MolFromSmiles(str(smiles)) is not None


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _element_counts(smiles: str) -> dict[int, int]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {}
    counts: dict[int, int] = {}
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 1:
            continue
        counts[atomic_num] = counts.get(atomic_num, 0) + 1
    return counts

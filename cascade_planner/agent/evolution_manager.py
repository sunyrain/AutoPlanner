"""Evidence-gated evolution manager for P3.

Knowledge discovered during a target run can enter candidate/shadow/staging
layers, but production promotion requires deterministic benchmark gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cascade_planner.agent.action_contracts import contains_raw_reaction_payload
from cascade_planner.agent.condition_agent import validate_condition_candidate
from cascade_planner.agent.literature_segments import (
    validate_literature_route_segment,
    validate_segment_step,
)

EVOLUTION_CANDIDATE_SCHEMA = "evolution_candidate.v1"
EVOLUTION_GATE_REPORT_SCHEMA = "evolution_gate_report.v1"
EVOLUTION_LAYERED_KB_SCHEMA = "evolution_layered_kb.v1"

ALLOWED_CANDIDATE_TYPES = {
    "ReactionRecordCandidate",
    "TemplateCandidate",
    "ConditionCandidate",
    "LiteratureRouteSegmentCard",
    "SegmentStepCandidate",
    "AnchorCandidate",
    "ControllerPolicyTrace",
}
KB_LAYERS = ("candidate", "shadow", "staging", "production")


@dataclass
class EvolutionCandidate:
    candidate_id: str
    candidate_type: str
    payload: dict[str, Any]
    evidence_refs: list[str] = field(default_factory=list)
    validation_status: str = "draft"
    source: str = "codex_worker"
    schema_version: str = EVOLUTION_CANDIDATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkGateReport:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    true_solved_rate_delta: float = 0.0
    fake_closure_rate_delta: float = 0.0
    condition_quality_delta: float = 0.0
    schema_version: str = EVOLUTION_GATE_REPORT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayeredKnowledgeBase:
    layers: dict[str, dict[str, EvolutionCandidate]] = field(default_factory=lambda: {layer: {} for layer in KB_LAYERS})
    history: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = EVOLUTION_LAYERED_KB_SCHEMA

    def add_candidate(self, candidate: EvolutionCandidate, *, target_run: bool = True) -> None:
        validation = validate_evolution_candidate(candidate)
        if not validation["accepted"]:
            raise ValueError(f"invalid evolution candidate: {validation['reasons']}")
        self.layers["candidate"][candidate.candidate_id] = candidate
        self.history.append({
            "event": "add_candidate",
            "candidate_id": candidate.candidate_id,
            "target_run": bool(target_run),
            "layer": "candidate",
        })

    def promote(
        self,
        candidate_id: str,
        *,
        from_layer: str,
        to_layer: str,
        gate_report: BenchmarkGateReport | dict[str, Any] | None = None,
        target_run: bool = False,
    ) -> None:
        if from_layer not in KB_LAYERS or to_layer not in KB_LAYERS:
            raise ValueError("invalid_kb_layer")
        if to_layer == "production" and target_run:
            raise ValueError("target_run_cannot_write_production")
        candidate = self.layers[from_layer].get(candidate_id)
        if candidate is None:
            raise ValueError("candidate_not_found")
        if to_layer == "production":
            report = _gate_report_from_raw(gate_report)
            if not report.accepted:
                raise ValueError(f"benchmark_gate_failed:{report.reasons}")
        self.layers[to_layer][candidate_id] = candidate
        self.history.append({
            "event": "promote",
            "candidate_id": candidate_id,
            "from_layer": from_layer,
            "to_layer": to_layer,
            "gate_report": _gate_report_payload(gate_report),
        })

    def rollback(self, candidate_id: str, *, layer: str = "production") -> None:
        if layer not in KB_LAYERS:
            raise ValueError("invalid_kb_layer")
        removed = self.layers[layer].pop(candidate_id, None)
        self.history.append({
            "event": "rollback",
            "candidate_id": candidate_id,
            "layer": layer,
            "removed": removed is not None,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layers": {
                layer: {cid: candidate.to_dict() for cid, candidate in rows.items()}
                for layer, rows in self.layers.items()
            },
            "history": list(self.history),
        }


def evaluate_benchmark_gate(metrics: dict[str, Any]) -> BenchmarkGateReport:
    """Accept only non-regressing benchmark reports."""
    solved_delta = _float(metrics.get("true_solved_rate_delta"), 0.0)
    fake_delta = _float(metrics.get("fake_closure_rate_delta"), 0.0)
    condition_delta = _float(metrics.get("condition_quality_delta"), 0.0)
    reasons: list[str] = []
    if solved_delta < 0:
        reasons.append("true_solved_rate_regressed")
    if fake_delta > 0:
        reasons.append("fake_closure_rate_regressed")
    if condition_delta < 0:
        reasons.append("condition_quality_regressed")
    if not bool(metrics.get("template_replay_passes", True)):
        reasons.append("template_replay_failed")
    if not bool(metrics.get("segment_replay_passes", True)):
        reasons.append("segment_replay_failed")
    if not bool(metrics.get("condition_quality_passes", True)):
        reasons.append("condition_quality_failed")
    if not bool(metrics.get("structure_validated", True)):
        reasons.append("structure_not_validated")
    if not bool(metrics.get("evidence_source_credible", True)):
        reasons.append("evidence_source_not_credible")
    if not bool(metrics.get("role_assignment_checked", True)):
        reasons.append("role_assignment_not_checked")
    if bool(metrics.get("overgeneralization_detected", False)):
        reasons.append("overgeneralization_detected")
    return BenchmarkGateReport(
        accepted=not reasons,
        reasons=sorted(set(reasons)),
        true_solved_rate_delta=solved_delta,
        fake_closure_rate_delta=fake_delta,
        condition_quality_delta=condition_delta,
    )


def validate_evolution_candidate(candidate_or_data: EvolutionCandidate | dict[str, Any]) -> dict[str, Any]:
    candidate = candidate_or_data if isinstance(candidate_or_data, EvolutionCandidate) else evolution_candidate_from_dict(candidate_or_data)
    reasons: list[str] = []
    if candidate.schema_version != EVOLUTION_CANDIDATE_SCHEMA:
        reasons.append("invalid_evolution_candidate_schema")
    if not candidate.candidate_id:
        reasons.append("missing_candidate_id")
    if candidate.candidate_type not in ALLOWED_CANDIDATE_TYPES:
        reasons.append("invalid_candidate_type")
    if not isinstance(candidate.payload, dict):
        reasons.append("payload_not_object")
    if not candidate.evidence_refs:
        reasons.append("missing_evidence_refs")
    if candidate.validation_status not in {"draft", "validated"}:
        reasons.append("invalid_validation_status")
    if _contains_raw_reaction_injection(candidate.payload):
        reasons.append("raw_reaction_injection")
    reasons.extend(_candidate_payload_reasons(candidate))
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "candidate_id": candidate.candidate_id,
        "schema_version": EVOLUTION_CANDIDATE_SCHEMA,
    }


def evolution_candidate_from_dict(data: dict[str, Any]) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=str(data.get("candidate_id") or ""),
        candidate_type=str(data.get("candidate_type") or ""),
        payload=dict(data.get("payload") or {}),
        evidence_refs=[str(item) for item in data.get("evidence_refs") or []],
        validation_status=str(data.get("validation_status") or "draft"),
        source=str(data.get("source") or "codex_worker"),
        schema_version=str(data.get("schema_version") or EVOLUTION_CANDIDATE_SCHEMA),
    )


def _gate_report_from_raw(report: BenchmarkGateReport | dict[str, Any] | None) -> BenchmarkGateReport:
    if isinstance(report, BenchmarkGateReport):
        return report
    if not isinstance(report, dict):
        return BenchmarkGateReport(accepted=False, reasons=["missing_benchmark_gate"])
    if "accepted" in report:
        return BenchmarkGateReport(
            accepted=bool(report.get("accepted")),
            reasons=[str(item) for item in report.get("reasons") or []],
            true_solved_rate_delta=_float(report.get("true_solved_rate_delta"), 0.0),
            fake_closure_rate_delta=_float(report.get("fake_closure_rate_delta"), 0.0),
            condition_quality_delta=_float(report.get("condition_quality_delta"), 0.0),
        )
    return evaluate_benchmark_gate(report)


def _gate_report_payload(report: BenchmarkGateReport | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(report, BenchmarkGateReport):
        return report.to_dict()
    if isinstance(report, dict):
        return dict(report)
    return {}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _contains_raw_reaction_injection(value: Any) -> bool:
    return contains_raw_reaction_payload(value)


def _candidate_payload_reasons(candidate: EvolutionCandidate) -> list[str]:
    if candidate.candidate_type == "ConditionCandidate":
        result = validate_condition_candidate(candidate.payload)
        return [f"condition:{reason}" for reason in result.get("reasons") or []]
    if candidate.candidate_type == "LiteratureRouteSegmentCard":
        result = validate_literature_route_segment(candidate.payload)
        return [f"segment:{reason}" for reason in result.get("reasons") or []]
    if candidate.candidate_type == "SegmentStepCandidate":
        result = validate_segment_step(candidate.payload)
        return [f"segment_step:{reason}" for reason in result.get("reasons") or []]
    return []

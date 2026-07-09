"""Typed artifact schemas for the agentic CASP control layer.

These records are intentionally small wrappers around payload dictionaries.
They give Codex/worker/controller outputs a common contract without making
natural-language text part of the decision path.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


ARTIFACT_SCHEMA_VERSION = "typed_artifact.v1"


@dataclass
class ArtifactBase:
    artifact_id: str
    case_id: str
    source: str
    input_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    validation_status: str = "draft"
    payload: dict[str, Any] = field(default_factory=dict)

    artifact_type: ClassVar[str] = "Artifact"
    schema_version: ClassVar[str] = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("missing_artifact_id")
        if not self.case_id:
            raise ValueError("missing_case_id")
        if not self.source:
            raise ValueError("missing_source")
        if self.validation_status not in {"draft", "draft_only", "accepted", "validated", "rejected"}:
            raise ValueError("invalid_validation_status")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact_type"] = self.artifact_type
        data["schema_version"] = self.schema_version
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


class TargetResolution(ArtifactBase):
    artifact_type = "TargetResolution"
    schema_version = "target_resolution.v1"


class StructureProfile(ArtifactBase):
    artifact_type = "StructureProfile"
    schema_version = "structure_profile.v1"


class TargetTriage(ArtifactBase):
    artifact_type = "TargetTriage"
    schema_version = "target_triage.v1"


class AgentActionBatch(ArtifactBase):
    artifact_type = "AgentActionBatch"
    schema_version = "agent_action_batch_artifact.v1"


class AgentActionBatchValidation(ArtifactBase):
    artifact_type = "AgentActionBatchValidation"
    schema_version = "agent_action_batch_validation_artifact.v1"


class AgentBlackboardSnapshot(ArtifactBase):
    artifact_type = "AgentBlackboardSnapshot"
    schema_version = "agent_blackboard_snapshot_artifact.v1"


class AgenticRunAudit(ArtifactBase):
    artifact_type = "AgenticRunAudit"
    schema_version = "agentic_run_audit_artifact.v1"


class AgenticCapabilityAudit(ArtifactBase):
    artifact_type = "AgenticCapabilityAudit"
    schema_version = "agentic_capability_audit_artifact.v1"


class AgenticFinalVerdictValidation(ArtifactBase):
    artifact_type = "AgenticFinalVerdictValidation"
    schema_version = "agentic_final_verdict_validation_artifact.v1"


class HypothesisOnlyRetrosynthesisReport(ArtifactBase):
    artifact_type = "HypothesisOnlyRetrosynthesisReport"
    schema_version = "hypothesis_only_retrosynthesis_report_artifact.v1"


class HypothesisExecutionReport(ArtifactBase):
    artifact_type = "HypothesisExecutionReport"
    schema_version = "hypothesis_execution_report_artifact.v1"


class ResearchReport(ArtifactBase):
    artifact_type = "ResearchReport"
    schema_version = "research_report.v1"


class LiteratureScoutReport(ArtifactBase):
    artifact_type = "LiteratureScoutReport"
    schema_version = "literature_scout_report_artifact.v1"


class EvidenceCard(ArtifactBase):
    artifact_type = "EvidenceCard"
    schema_version = "evidence_card_artifact.v1"


class StrategicDisconnectionCard(ArtifactBase):
    artifact_type = "StrategicDisconnectionCard"
    schema_version = "strategic_disconnection_card.v1"


class LiteratureRouteSegmentCard(ArtifactBase):
    artifact_type = "LiteratureRouteSegmentCard"
    schema_version = "literature_route_segment_card.v1"


class SegmentStepCandidate(ArtifactBase):
    artifact_type = "SegmentStepCandidate"
    schema_version = "segment_step_candidate.v1"


class LiteratureTemplateCard(ArtifactBase):
    artifact_type = "LiteratureTemplateCard"
    schema_version = "literature_template_card.v1"


class TemplateApplicabilityReport(ArtifactBase):
    artifact_type = "TemplateApplicabilityReport"
    schema_version = "template_applicability_report.v1"


class ExecutableTemplateCandidate(ArtifactBase):
    artifact_type = "ExecutableTemplateCandidate"
    schema_version = "executable_template_candidate.v1"


class TemplateValidationReport(ArtifactBase):
    artifact_type = "TemplateValidationReport"
    schema_version = "template_validation_report.v1"


class AnalogicalReactionTemplateReport(ArtifactBase):
    artifact_type = "AnalogicalReactionTemplateReport"
    schema_version = "analogical_reaction_template_report.v1"


class AnalogicalReactionTemplateRanking(ArtifactBase):
    artifact_type = "AnalogicalReactionTemplateRanking"
    schema_version = "analogical_reaction_template_ranking_artifact.v1"


class AnalogicalTemplateApplicationReport(ArtifactBase):
    artifact_type = "AnalogicalTemplateApplicationReport"
    schema_version = "analogical_template_application_report_artifact.v1"


class AnalogicalTemplateApplicationValidation(ArtifactBase):
    artifact_type = "AnalogicalTemplateApplicationValidation"
    schema_version = "analogical_template_application_validation_artifact.v1"


class LiteratureTriggerReport(ArtifactBase):
    artifact_type = "LiteratureTriggerReport"
    schema_version = "literature_trigger_report.v1"


class RouteAnchorExpansionTask(ArtifactBase):
    artifact_type = "RouteAnchorExpansionTask"
    schema_version = "route_anchor_expansion_task.v1"


class FailureEvent(ArtifactBase):
    artifact_type = "FailureEvent"
    schema_version = "failure_event_artifact.v1"


class FailureDiagnosis(ArtifactBase):
    artifact_type = "FailureDiagnosis"
    schema_version = "failure_diagnosis.v1"


class StrategicOperator(ArtifactBase):
    artifact_type = "StrategicOperator"
    schema_version = "strategic_operator_artifact.v1"


class ChemEnzySearchPolicy(ArtifactBase):
    artifact_type = "ChemEnzySearchPolicy"
    schema_version = "chem_enzy_search_policy_artifact.v1"


class JudgePolicy(ArtifactBase):
    artifact_type = "JudgePolicy"
    schema_version = "judge_policy_artifact.v1"


class ConditionCandidate(ArtifactBase):
    artifact_type = "ConditionCandidate"
    schema_version = "condition_candidate.v1"


class RouteAuditReport(ArtifactBase):
    artifact_type = "RouteAuditReport"
    schema_version = "route_audit_report.v1"


class RouteStatus(ArtifactBase):
    artifact_type = "RouteStatus"
    schema_version = "route_status_artifact.v1"


class EvolutionCandidate(ArtifactBase):
    artifact_type = "EvolutionCandidate"
    schema_version = "evolution_candidate_artifact.v1"


class StatinPanelSelfEvoReport(ArtifactBase):
    artifact_type = "StatinPanelSelfEvoReport"
    schema_version = "statin_panel_self_evo_report_artifact.v1"


class StatinFullflowOverview(ArtifactBase):
    artifact_type = "StatinFullflowOverview"
    schema_version = "statin_fullflow_overview_artifact.v1"


class StatinFullflowDossier(ArtifactBase):
    artifact_type = "StatinFullflowDossier"
    schema_version = "statin_fullflow_dossier_artifact.v1"


class StatinRouteTemplate(ArtifactBase):
    artifact_type = "StatinRouteTemplate"
    schema_version = "statin_route_template_artifact.v1"


class StatinRouteClosureAudit(ArtifactBase):
    artifact_type = "StatinRouteClosureAudit"
    schema_version = "statin_route_closure_audit_artifact.v1"


class StatinRouteClosureMatrix(ArtifactBase):
    artifact_type = "StatinRouteClosureMatrix"
    schema_version = "statin_route_closure_matrix_artifact.v1"


class StatinClosureLeadCurationPacket(ArtifactBase):
    artifact_type = "StatinClosureLeadCurationPacket"
    schema_version = "statin_closure_lead_curation_packet_artifact.v1"


class StatinClosureCurationResultSet(ArtifactBase):
    artifact_type = "StatinClosureCurationResultSet"
    schema_version = "statin_closure_curation_result_set_artifact.v1"


class WorkerRunRecord(ArtifactBase):
    artifact_type = "WorkerRunRecord"
    schema_version = "worker_run_record_artifact.v1"


ARTIFACT_CLASSES: dict[str, type[ArtifactBase]] = {
    cls.artifact_type: cls
    for cls in (
        TargetResolution,
        StructureProfile,
        TargetTriage,
        AgentActionBatch,
        AgentActionBatchValidation,
        AgentBlackboardSnapshot,
        AgenticRunAudit,
        AgenticCapabilityAudit,
        AgenticFinalVerdictValidation,
        HypothesisOnlyRetrosynthesisReport,
        HypothesisExecutionReport,
        ResearchReport,
        LiteratureScoutReport,
        EvidenceCard,
        StrategicDisconnectionCard,
        LiteratureRouteSegmentCard,
        SegmentStepCandidate,
        LiteratureTemplateCard,
        TemplateApplicabilityReport,
        ExecutableTemplateCandidate,
        TemplateValidationReport,
        AnalogicalReactionTemplateReport,
        AnalogicalReactionTemplateRanking,
        AnalogicalTemplateApplicationReport,
        AnalogicalTemplateApplicationValidation,
        LiteratureTriggerReport,
        RouteAnchorExpansionTask,
        FailureEvent,
        FailureDiagnosis,
        StrategicOperator,
        ChemEnzySearchPolicy,
        JudgePolicy,
        ConditionCandidate,
        RouteAuditReport,
        RouteStatus,
        EvolutionCandidate,
        StatinPanelSelfEvoReport,
        StatinFullflowOverview,
        StatinFullflowDossier,
        StatinRouteTemplate,
        StatinRouteClosureAudit,
        StatinRouteClosureMatrix,
        StatinClosureLeadCurationPacket,
        StatinClosureCurationResultSet,
        WorkerRunRecord,
    )
}


def artifact_from_dict(data: dict[str, Any]) -> ArtifactBase:
    artifact_type = str(data.get("artifact_type") or "")
    cls = ARTIFACT_CLASSES.get(artifact_type)
    if cls is None:
        raise ValueError(f"unknown_artifact_type:{artifact_type}")
    schema_version = str(data.get("schema_version") or "")
    if schema_version and schema_version != cls.schema_version:
        raise ValueError(f"schema_version_mismatch:{artifact_type}")
    return cls(
        artifact_id=str(data.get("artifact_id") or ""),
        case_id=str(data.get("case_id") or ""),
        source=str(data.get("source") or ""),
        input_refs=[str(item) for item in data.get("input_refs") or []],
        evidence_refs=[str(item) for item in data.get("evidence_refs") or []],
        validation_status=str(data.get("validation_status") or "draft"),
        payload=dict(data.get("payload") or {}),
    )


def artifact_json_round_trip(artifact: ArtifactBase) -> ArtifactBase:
    return artifact_from_dict(json.loads(artifact.to_json()))


def artifact_type_names() -> list[str]:
    return sorted(ARTIFACT_CLASSES)

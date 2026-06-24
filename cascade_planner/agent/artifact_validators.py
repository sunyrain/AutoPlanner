"""Deterministic validators for agent typed artifacts."""
from __future__ import annotations

from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

from cascade_planner.agent.action_contracts import (
    ACTION_BATCH_SCHEMA as AGENT_ACTION_BATCH_SCHEMA,
    ALLOWED_AGENT_ACTIONS,
    contains_raw_reaction_payload as _contains_raw_reaction,
    planner_source_hint_reasons,
)
from cascade_planner.agent.artifact_schemas import ArtifactBase, artifact_from_dict
from cascade_planner.agent.chem_enzy_policy import (
    validate_chem_enzy_search_policy_payload,
    validate_strategic_operator,
)
from cascade_planner.agent.condition_agent import validate_condition_candidate
from cascade_planner.agent.evidence_cards import validate_evidence_card
from cascade_planner.agent.evolution_manager import validate_evolution_candidate
from cascade_planner.agent.literature_segments import (
    validate_literature_route_segment,
    validate_segment_step,
)
from cascade_planner.agent.strategic_candidate_generation import validate_literature_candidate
from cascade_planner.agent.terminal_judge import validate_judge_policy


RDLogger.DisableLog("rdApp.*")

VALIDATOR_SCHEMA = "artifact_validator.v1"


def validate_typed_artifact(
    artifact_or_data: ArtifactBase | dict[str, Any],
    *,
    validated_evidence_refs: set[str] | None = None,
    validated_disconnection_refs: set[str] | None = None,
) -> dict[str, Any]:
    """Validate one typed artifact without using model calls."""
    reasons: list[str] = []
    try:
        artifact = artifact_or_data if isinstance(artifact_or_data, ArtifactBase) else artifact_from_dict(artifact_or_data)
    except ValueError as exc:
        return _result(False, [str(exc)], "", "")

    data = artifact.to_dict()
    payload = dict(data.get("payload") or {})
    artifact_type = str(data.get("artifact_type") or "")
    reasons.extend(_common_reasons(data))

    if artifact_type == "AgentActionBatch":
        reasons.extend(_agent_action_batch_reasons(payload))
    elif artifact_type == "AgentActionBatchValidation":
        reasons.extend(_agent_action_batch_validation_reasons(payload))
    elif artifact_type == "AgentBlackboardSnapshot":
        reasons.extend(_agent_blackboard_snapshot_reasons(payload))
    elif artifact_type == "AgenticRunAudit":
        reasons.extend(_agentic_run_audit_reasons(payload))
    elif artifact_type == "AgenticCapabilityAudit":
        reasons.extend(_agentic_capability_audit_reasons(payload))
    elif artifact_type == "AgenticFinalVerdictValidation":
        reasons.extend(_agentic_final_verdict_validation_reasons(payload))
    elif artifact_type == "HypothesisOnlyRetrosynthesisReport":
        reasons.extend(_hypothesis_only_retrosynthesis_report_reasons(payload))
    elif artifact_type == "HypothesisExecutionReport":
        reasons.extend(_hypothesis_execution_report_reasons(payload))
    elif artifact_type == "LiteratureScoutReport":
        reasons.extend(_literature_scout_report_reasons(payload))
    elif artifact_type == "AnalogicalReactionTemplateReport":
        reasons.extend(_analogical_reaction_template_report_reasons(payload))
    elif artifact_type == "AnalogicalReactionTemplateRanking":
        reasons.extend(_analogical_reaction_template_ranking_reasons(payload))
    elif artifact_type == "AnalogicalTemplateApplicationReport":
        reasons.extend(_analogical_template_application_report_reasons(payload))
    elif artifact_type == "AnalogicalTemplateApplicationValidation":
        reasons.extend(_analogical_template_application_validation_reasons(payload))
    elif artifact_type in {"TargetResolution", "StructureProfile"}:
        reasons.extend(_structure_reasons(payload))
    elif artifact_type == "EvidenceCard":
        reasons.extend(_evidence_reasons(payload))
    elif artifact_type == "StrategicDisconnectionCard":
        reasons.extend(_strategic_disconnection_reasons(payload, validated_evidence_refs or set()))
    elif artifact_type == "LiteratureRouteSegmentCard":
        reasons.extend(_literature_route_segment_reasons(payload))
    elif artifact_type == "SegmentStepCandidate":
        reasons.extend(_segment_step_reasons(payload))
    elif artifact_type == "StrategicOperator":
        reasons.extend(_strategic_operator_reasons(payload, validated_evidence_refs or set(), validated_disconnection_refs or set()))
    elif artifact_type == "ChemEnzySearchPolicy":
        reasons.extend(_search_policy_reasons(payload, validated_evidence_refs or set()))
    elif artifact_type == "JudgePolicy":
        reasons.extend(_judge_policy_reasons(payload))
    elif artifact_type == "ConditionCandidate":
        reasons.extend(_condition_reasons(payload))
    elif artifact_type == "RouteAuditReport":
        reasons.extend(_route_audit_reasons(payload))
    elif artifact_type == "RouteStatus":
        reasons.extend(_route_status_reasons(payload))
    elif artifact_type == "EvolutionCandidate":
        reasons.extend(_evolution_reasons(payload))
    elif artifact_type == "StatinPanelSelfEvoReport":
        reasons.extend(_statin_panel_self_evo_report_reasons(payload))
    elif artifact_type == "StatinFullflowOverview":
        reasons.extend(_statin_fullflow_overview_reasons(payload))
    elif artifact_type == "StatinFullflowDossier":
        reasons.extend(_statin_fullflow_dossier_reasons(payload))
    elif artifact_type == "StatinRouteTemplate":
        reasons.extend(_statin_route_template_reasons(payload))
    elif artifact_type == "StatinRouteClosureAudit":
        reasons.extend(_statin_route_closure_audit_reasons(payload))
    elif artifact_type == "StatinRouteClosureMatrix":
        reasons.extend(_statin_route_closure_matrix_reasons(payload))
    elif artifact_type == "StatinClosureLeadCurationPacket":
        reasons.extend(_statin_closure_lead_curation_packet_reasons(payload))
    elif artifact_type == "StatinClosureCurationResultSet":
        reasons.extend(_statin_closure_curation_result_set_reasons(payload))
    elif artifact_type == "FailureDiagnosis":
        reasons.extend(_failure_diagnosis_reasons(payload))
    elif artifact_type == "WorkerRunRecord":
        reasons.extend(_worker_run_record_reasons(payload))

    return _result(not reasons, sorted(set(reasons)), data.get("artifact_id", ""), artifact_type)


def validate_artifact_list(
    artifacts: list[ArtifactBase | dict[str, Any]],
) -> dict[str, Any]:
    """Validate artifacts in order, carrying accepted evidence/disconnection refs forward."""
    accepted_evidence: set[str] = set()
    accepted_disconnections: set[str] = set()
    rows = []
    for item in artifacts:
        result = validate_typed_artifact(
            item,
            validated_evidence_refs=accepted_evidence,
            validated_disconnection_refs=accepted_disconnections,
        )
        rows.append(result)
        if result["accepted"]:
            artifact_type = result["artifact_type"]
            artifact_id = result["artifact_id"]
            if artifact_type == "EvidenceCard":
                accepted_evidence.add(artifact_id)
            if artifact_type == "StrategicDisconnectionCard":
                accepted_disconnections.add(artifact_id)
    return {
        "schema_version": "artifact_list_validation.v1",
        "accepted": all(row["accepted"] for row in rows),
        "results": rows,
        "accepted_evidence_refs": sorted(accepted_evidence),
        "accepted_disconnection_refs": sorted(accepted_disconnections),
    }


def _common_reasons(data: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not data.get("schema_version"):
        reasons.append("missing_schema_version")
    if not data.get("artifact_id"):
        reasons.append("missing_artifact_id")
    if not data.get("case_id"):
        reasons.append("missing_case_id")
    if not data.get("source"):
        reasons.append("missing_source")
    if data.get("validation_status") not in {"draft", "draft_only", "accepted", "validated", "rejected"}:
        reasons.append("invalid_validation_status")
    if not data.get("evidence_refs") and not data.get("input_refs"):
        reasons.append("missing_evidence_or_input_refs")
    return reasons


def _agent_action_batch_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != AGENT_ACTION_BATCH_SCHEMA:
        reasons.append("invalid_action_batch_schema")
    semantics = payload.get("semantics") or {}
    if isinstance(semantics, dict):
        if bool(semantics.get("planner_can_emit_solved")):
            reasons.append("planner_semantics_allow_solved_claim")
        if bool(semantics.get("raw_reaction_output_allowed")):
            reasons.append("planner_semantics_allow_raw_reaction_output")
    if payload.get("verdict") == "solved" or payload.get("route_status") == "solved":
        reasons.append("planner_direct_solved_claim")
    if _contains_raw_reaction({key: value for key, value in payload.items() if key != "actions"}):
        reasons.append("raw_reaction_injection")
    reasons.extend(planner_source_hint_reasons(payload.get("planner_source_hints")))
    actions = payload.get("actions")
    if not isinstance(actions, list):
        reasons.append("actions_not_list")
        actions = []
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            reasons.append(f"action_not_object:{idx}")
            continue
        action_type = str(action.get("action_type") or "")
        if action_type not in ALLOWED_AGENT_ACTIONS:
            reasons.append(f"unknown_action:{idx}:{action_type or 'missing'}")
        for field in ("rationale", "expected_artifact", "success_condition"):
            if not str(action.get(field) or "").strip():
                reasons.append(f"action_missing_{field}:{idx}")
        if action.get("verdict") == "solved" or action.get("route_status") == "solved":
            reasons.append("planner_direct_solved_claim")
        if _contains_raw_reaction(action):
            reasons.append("raw_reaction_injection")
    return reasons


def _agent_action_batch_validation_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "agent_action_batch_validation.v1":
        reasons.append("invalid_action_batch_validation_schema")
    if not isinstance(payload.get("accepted"), bool):
        reasons.append("action_batch_validation_missing_accepted_boolean")
    if not isinstance(payload.get("reasons"), list):
        reasons.append("action_batch_validation_reasons_not_list")
    if not str(payload.get("case_id") or ""):
        reasons.append("action_batch_validation_missing_case_id")
    return reasons


def _agent_blackboard_snapshot_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "agent_blackboard.v1":
        reasons.append("invalid_agent_blackboard_schema")
    if not str(payload.get("case_id") or ""):
        reasons.append("agent_blackboard_missing_case_id")
    required_fields = {
        "target_profile": dict,
        "literature_evidence": dict,
        "budget_state": dict,
        "current_belief": dict,
        "planner_history": list,
        "action_history": list,
        "artifact_refs": dict,
    }
    for field, expected in required_fields.items():
        if not isinstance(payload.get(field), expected):
            reasons.append(f"agent_blackboard_invalid_field:{field}")
    target = dict(payload.get("target_profile") or {})
    if target.get("schema_version") != "agent_target_profile_summary.v1":
        reasons.append("agent_blackboard_invalid_target_profile_schema")
    if not str(target.get("target_smiles") or "").strip():
        reasons.append("agent_blackboard_missing_target_smiles")
    literature = dict(payload.get("literature_evidence") or {})
    if literature.get("schema_version") != "agent_literature_evidence_summary.v1":
        reasons.append("agent_blackboard_invalid_literature_evidence_schema")
    if "source_lifecycle" in literature and not isinstance(literature.get("source_lifecycle"), list):
        reasons.append("agent_blackboard_source_lifecycle_not_list")
    budget = dict(payload.get("budget_state") or {})
    if budget.get("schema_version") != "agent_blackboard_budget_state.v1":
        reasons.append("agent_blackboard_invalid_budget_schema")
    reasons.extend(_agent_blackboard_budget_reasons(budget))
    belief = dict(payload.get("current_belief") or {})
    if belief.get("schema_version") != "agent_current_belief.v1":
        reasons.append("agent_blackboard_invalid_current_belief_schema")
    template_policy = dict(belief.get("template_policy") or {})
    if template_policy.get("analogy_is_advisory_only", True) is not True:
        reasons.append("agent_blackboard_analogy_not_advisory_only")
    if payload.get("verdict") == "solved" or payload.get("route_status") == "solved" or payload.get("solved") is True:
        reasons.append("agent_blackboard_direct_solved_claim")
    proof = dict(payload.get("parent_route_proof") or {})
    if proof.get("solved") and not proof.get("accepted"):
        reasons.append("agent_blackboard_parent_proof_solved_without_accepted")
    for idx, row in enumerate(payload.get("planner_history") or []):
        if not isinstance(row, dict):
            reasons.append(f"agent_blackboard_planner_history_row_not_object:{idx}")
            continue
        if row.get("planner_can_emit_solved"):
            reasons.append("agent_blackboard_planner_history_allows_solved")
        if row.get("raw_reaction_output_allowed"):
            reasons.append("agent_blackboard_planner_history_allows_raw_reaction")
    for idx, row in enumerate(payload.get("action_history") or []):
        if not isinstance(row, dict):
            reasons.append(f"agent_blackboard_action_history_row_not_object:{idx}")
            continue
        if not str(row.get("action_type") or "").strip():
            reasons.append(f"agent_blackboard_action_history_missing_action_type:{idx}")
    if _contains_raw_reaction(payload):
        reasons.append("raw_reaction_injection")
    return reasons


def _agent_blackboard_budget_reasons(budget: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    checks = [
        ("rounds_completed", "max_rounds"),
        ("scout_calls", "max_scout_calls"),
        ("visual_calls", "max_visual_calls"),
        ("chemenzy_runs", "max_chemenzy_runs"),
        ("child_target_runs", "max_child_target_runs"),
        ("template_application_actions", "max_template_application_actions"),
    ]
    for used_key, max_key in checks:
        if used_key not in budget or max_key not in budget:
            continue
        try:
            used = int(budget.get(used_key) or 0)
            maximum = int(budget.get(max_key) or 0)
        except (TypeError, ValueError):
            reasons.append(f"agent_blackboard_invalid_budget_value:{used_key}")
            continue
        if used < 0 or maximum < 0:
            reasons.append(f"agent_blackboard_negative_budget_value:{used_key}")
        if maximum >= 0 and used > maximum:
            reasons.append(f"agent_blackboard_budget_exceeded:{used_key}")
    return reasons


def _agentic_run_audit_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "agentic_blackboard_run_audit.v1":
        reasons.append("invalid_agentic_run_audit_schema")
    if payload.get("audit_authority") != "diagnostic_only":
        reasons.append("agentic_run_audit_authority_not_diagnostic_only")
    if payload.get("deterministic_final_verdict_required") is not True:
        reasons.append("agentic_run_audit_missing_final_verdict_guard")
    if not isinstance(payload.get("round_summaries"), list):
        reasons.append("agentic_run_audit_round_summaries_not_list")
    if not isinstance(payload.get("budget_state"), dict):
        reasons.append("agentic_run_audit_missing_budget_state")
    source_acquisition = payload.get("source_acquisition_summary")
    if source_acquisition is not None and not isinstance(source_acquisition, dict):
        reasons.append("agentic_run_audit_source_acquisition_summary_not_object")
    if isinstance(source_acquisition, dict):
        if source_acquisition.get("no_solved_claim") is not True:
            reasons.append("agentic_run_audit_source_acquisition_summary_missing_no_solved_claim")
        if source_acquisition.get("auto_local_pdf_blind_fallback_used"):
            reasons.append("agentic_run_audit_auto_local_pdf_blind_fallback_used")
        if int(source_acquisition.get("planner_source_hint_count") or 0) > 0 and source_acquisition.get("planner_source_hints_are_not_evidence") is not True:
            reasons.append("agentic_run_audit_planner_source_hints_not_boundary_marked")
    transition_summary = payload.get("blackboard_transition_summary")
    if transition_summary is not None:
        if not isinstance(transition_summary, dict):
            reasons.append("agentic_run_audit_transition_summary_not_object")
        elif transition_summary.get("no_solved_claim") is not True:
            reasons.append("agentic_run_audit_transition_summary_missing_no_solved_claim")
    typed_summary = payload.get("typed_artifact_validation_summary")
    if typed_summary is not None:
        if not isinstance(typed_summary, dict):
            reasons.append("agentic_run_audit_typed_validation_summary_not_object")
        elif int(typed_summary.get("failed_artifact_count") or 0) and not typed_summary.get("failed_artifact_keys"):
            reasons.append("agentic_run_audit_typed_validation_failures_missing_keys")
    analogical_summary = payload.get("analogical_template_summary")
    if analogical_summary is not None:
        if not isinstance(analogical_summary, dict):
            reasons.append("agentic_run_audit_analogical_template_summary_not_object")
        else:
            if analogical_summary.get("no_solved_claim") is not True:
                reasons.append("agentic_run_audit_analogical_template_summary_missing_no_solved_claim")
            if analogical_summary.get("final_verdict_authority") not in {"", "none", None}:
                reasons.append("agentic_run_audit_analogical_template_as_final_authority")
    final = dict(payload.get("final_verdict") or {})
    proof = dict(payload.get("parent_route_proof") or {})
    if final.get("solved") or final.get("verdict") == "solved" or final.get("route_status") == "solved":
        if not (proof.get("accepted") and proof.get("solved")):
            reasons.append("agentic_run_audit_solved_without_parent_proof")
    if _contains_raw_reaction(payload):
        reasons.append("raw_reaction_injection")
    return reasons


def _agentic_capability_audit_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "agentic_capability_audit.v1":
        reasons.append("invalid_agentic_capability_audit_schema")
    if payload.get("audit_authority") != "diagnostic_only":
        reasons.append("agentic_capability_audit_authority_not_diagnostic_only")
    if payload.get("no_solved_claim") is not True:
        reasons.append("agentic_capability_audit_missing_no_solved_claim")
    if payload.get("deterministic_final_verdict_required") is not True:
        reasons.append("agentic_capability_audit_missing_final_verdict_guard")
    if payload.get("final_verdict_authority") != "deterministic_parent_route_proof":
        reasons.append("agentic_capability_audit_invalid_final_verdict_authority")
    if not isinstance(payload.get("accepted"), bool):
        reasons.append("agentic_capability_audit_missing_accepted_boolean")
    checks = payload.get("requirement_checks")
    if not isinstance(checks, list):
        reasons.append("agentic_capability_audit_requirement_checks_not_list")
        checks = []
    failed_ids: list[str] = []
    for idx, check in enumerate(checks):
        if not isinstance(check, dict):
            reasons.append(f"agentic_capability_audit_check_not_object:{idx}")
            continue
        requirement_id = str(check.get("requirement_id") or "").strip()
        if not requirement_id:
            reasons.append(f"agentic_capability_audit_check_missing_requirement_id:{idx}")
        if not isinstance(check.get("accepted"), bool):
            reasons.append(f"agentic_capability_audit_check_missing_accepted_boolean:{idx}")
        elif not check.get("accepted"):
            failed_ids.append(requirement_id or f"check_{idx}")
        if not isinstance(check.get("evidence"), list):
            reasons.append(f"agentic_capability_audit_check_evidence_not_list:{idx}")
        if not isinstance(check.get("reasons"), list):
            reasons.append(f"agentic_capability_audit_check_reasons_not_list:{idx}")
        if check.get("verdict") == "solved" or check.get("route_status") == "solved" or check.get("solved") is True:
            reasons.append("agentic_capability_audit_direct_solved_claim")
        if _contains_raw_reaction(check):
            reasons.append("raw_reaction_injection")
    failed_requirements = [str(item) for item in payload.get("failed_requirements") or []]
    if payload.get("accepted") is True and failed_ids:
        reasons.append("agentic_capability_audit_accepted_with_failed_requirements")
    if payload.get("accepted") is False and failed_ids and not failed_requirements:
        reasons.append("agentic_capability_audit_missing_failed_requirements")
    if _contains_raw_reaction(payload):
        reasons.append("raw_reaction_injection")
    return reasons


def _agentic_final_verdict_validation_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "agentic_final_verdict_validation.v1":
        reasons.append("invalid_agentic_final_verdict_validation_schema")
    if not isinstance(payload.get("accepted"), bool):
        reasons.append("agentic_final_verdict_validation_missing_accepted_boolean")
    if not isinstance(payload.get("reasons"), list):
        reasons.append("agentic_final_verdict_validation_reasons_not_list")
    if not isinstance(payload.get("final_verdict"), dict):
        reasons.append("agentic_final_verdict_validation_missing_final_verdict")
    if not isinstance(payload.get("parent_route_proof_summary"), dict):
        reasons.append("agentic_final_verdict_validation_missing_parent_proof_summary")
    if not isinstance(payload.get("checked_invariants"), list):
        reasons.append("agentic_final_verdict_validation_missing_checked_invariants")
    if payload.get("accepted") is False and not payload.get("reasons"):
        reasons.append("agentic_final_verdict_validation_rejected_without_reasons")
    final = dict(payload.get("final_verdict") or {})
    proof = dict(payload.get("parent_route_proof_summary") or {})
    if final.get("solved") or final.get("verdict") == "solved" or final.get("route_status") == "solved":
        if not (proof.get("accepted") and proof.get("solved")):
            reasons.append("agentic_final_verdict_validation_solved_without_parent_proof")
    if _contains_raw_reaction(payload):
        reasons.append("raw_reaction_injection")
    return reasons


def _hypothesis_only_retrosynthesis_report_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "hypothesis_only_retrosynthesis_report.v1":
        reasons.append("invalid_hypothesis_only_retrosynthesis_report_schema")
    reasons.extend(_advisory_payload_guard_reasons(payload, prefix="hypothesis_only_retrosynthesis_report"))
    if payload.get("final_verdict_authority") not in {"none", "", None}:
        reasons.append("hypothesis_only_report_has_final_verdict_authority")
    if payload.get("not_parent_route_proof") is not True:
        reasons.append("hypothesis_only_report_missing_parent_proof_boundary")
    if payload.get("not_exact_literature_segment") is not True:
        reasons.append("hypothesis_only_report_missing_exact_literature_boundary")
    candidates = payload.get("candidate_precursors")
    if not isinstance(candidates, list):
        reasons.append("hypothesis_only_report_candidates_not_list")
        candidates = []
    sketches = payload.get("route_sketches")
    if not isinstance(sketches, list):
        reasons.append("hypothesis_only_report_route_sketches_not_list")
        sketches = []
    if payload.get("accepted") and not candidates:
        reasons.append("accepted_hypothesis_only_report_without_candidates")
    for idx, row in enumerate(candidates):
        if not isinstance(row, dict):
            reasons.append(f"hypothesis_only_candidate_not_object:{idx}")
            continue
        if row.get("schema_version") != "hypothesis_only_precursor_candidate.v1":
            reasons.append(f"invalid_hypothesis_only_candidate_schema:{idx}")
        if not str(row.get("candidate_id") or "").strip():
            reasons.append(f"hypothesis_only_candidate_missing_id:{idx}")
        if row.get("no_solved_claim") is not True:
            reasons.append(f"hypothesis_only_candidate_missing_no_solved_claim:{idx}")
        if row.get("not_parent_route_proof") is not True:
            reasons.append(f"hypothesis_only_candidate_missing_parent_proof_boundary:{idx}")
        if row.get("not_exact_literature_segment") is not True:
            reasons.append(f"hypothesis_only_candidate_missing_exact_literature_boundary:{idx}")
        if str(row.get("allowed_use") or "") != "guided_search_seed_only":
            reasons.append(f"hypothesis_only_candidate_invalid_allowed_use:{idx}")
        precursor = str(row.get("precursor_smiles") or "")
        if precursor and Chem.MolFromSmiles(precursor) is None:
            reasons.append(f"hypothesis_only_candidate_invalid_precursor_smiles:{idx}")
        target = str(row.get("target_smiles") or "")
        if target and Chem.MolFromSmiles(target) is None:
            reasons.append(f"hypothesis_only_candidate_invalid_target_smiles:{idx}")
        if row.get("verdict") == "solved" or row.get("route_status") == "solved" or row.get("solved") is True:
            reasons.append("hypothesis_only_candidate_direct_solved_claim")
        if _contains_raw_reaction(row):
            reasons.append("raw_reaction_injection")
    for idx, sketch in enumerate(sketches):
        if not isinstance(sketch, dict):
            reasons.append(f"hypothesis_only_route_sketch_not_object:{idx}")
            continue
        if sketch.get("schema_version") != "hypothesis_only_route_sketch.v1":
            reasons.append(f"invalid_hypothesis_only_route_sketch_schema:{idx}")
        if sketch.get("no_solved_claim") is not True:
            reasons.append(f"hypothesis_only_route_sketch_missing_no_solved_claim:{idx}")
        if sketch.get("not_parent_route_proof") is not True:
            reasons.append(f"hypothesis_only_route_sketch_missing_parent_proof_boundary:{idx}")
        if sketch.get("verdict") == "solved" or sketch.get("route_status") == "solved" or sketch.get("solved") is True:
            reasons.append("hypothesis_only_route_sketch_direct_solved_claim")
        if _contains_raw_reaction(sketch):
            reasons.append("raw_reaction_injection")
    if _contains_raw_reaction(payload):
        reasons.append("raw_reaction_injection")
    return reasons


def _hypothesis_execution_report_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "hypothesis_execution_report.v1":
        reasons.append("invalid_hypothesis_execution_report_schema")
    if not isinstance(payload.get("accepted"), bool):
        reasons.append("hypothesis_execution_report_missing_accepted_boolean")
    if payload.get("solved") is True or payload.get("route_status") == "solved":
        reasons.append("hypothesis_execution_report_direct_solved_claim")
    if payload.get("no_parent_solved_claim") is not True:
        reasons.append("hypothesis_execution_report_missing_parent_solved_boundary")
    if payload.get("hypotheses_must_be_executed") is not True:
        reasons.append("hypothesis_execution_report_missing_execution_contract")
    rows = payload.get("candidate_executions")
    if not isinstance(rows, list):
        reasons.append("hypothesis_execution_report_rows_not_list")
        rows = []
    try:
        candidate_count = int(payload.get("candidate_count") or 0)
    except (TypeError, ValueError):
        candidate_count = -1
        reasons.append("hypothesis_execution_report_invalid_candidate_count")
    if candidate_count != len(rows):
        reasons.append("hypothesis_execution_report_candidate_count_mismatch")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            reasons.append(f"hypothesis_execution_row_not_object:{idx}")
            continue
        if row.get("schema_version") != "hypothesis_candidate_execution.v1":
            reasons.append(f"invalid_hypothesis_execution_row_schema:{idx}")
        if not str(row.get("candidate_id") or "").strip():
            reasons.append(f"hypothesis_execution_row_missing_candidate_id:{idx}")
        if str(row.get("execution_status") or "") not in {
            "not_executed",
            "executed_rejected",
            "executed_verified_child_route",
        }:
            reasons.append(f"hypothesis_execution_row_invalid_status:{idx}")
        if row.get("no_parent_solved_claim") is not True:
            reasons.append(f"hypothesis_execution_row_missing_parent_boundary:{idx}")
        if row.get("solved") is True and row.get("verifier_accepted") is not True:
            reasons.append(f"hypothesis_execution_row_solved_without_verifier:{idx}")
        if str(row.get("precursor_smiles") or "") and Chem.MolFromSmiles(str(row.get("precursor_smiles") or "")) is None:
            reasons.append(f"hypothesis_execution_row_invalid_precursor_smiles:{idx}")
        if _contains_raw_reaction(row):
            reasons.append("raw_reaction_injection")
    if _contains_raw_reaction(payload):
        reasons.append("raw_reaction_injection")
    return reasons


def _literature_scout_report_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "literature_scout_report.v1":
        reasons.append("invalid_literature_scout_report_schema")
    if not isinstance(payload.get("accepted"), bool):
        reasons.append("literature_scout_missing_accepted_boolean")
    if not isinstance(payload.get("source_candidates"), list):
        reasons.append("literature_scout_source_candidates_not_list")
        candidates: list[Any] = []
    else:
        candidates = list(payload.get("source_candidates") or [])
    if not isinstance(payload.get("source_refs"), list):
        reasons.append("literature_scout_source_refs_not_list")
    if payload.get("no_solved_claim") is not True:
        reasons.append("literature_scout_missing_no_solved_claim")
    if payload.get("verdict") == "solved" or payload.get("route_status") == "solved":
        reasons.append("literature_scout_direct_solved_claim")
    if str(payload.get("source_discovery_mode") or "") == "placeholder" and payload.get("accepted"):
        reasons.append("placeholder_scout_marked_accepted")
    if bool(payload.get("placeholder_only")) and payload.get("accepted"):
        reasons.append("placeholder_scout_marked_accepted")
    real_candidate_count = 0
    placeholder_count = 0
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            reasons.append(f"literature_source_candidate_not_object:{idx}")
            continue
        if candidate.get("schema_version") and candidate.get("schema_version") != "literature_source_candidate.v1":
            reasons.append(f"invalid_literature_source_candidate_schema:{idx}")
        if candidate.get("no_solved_claim") is not True:
            reasons.append(f"literature_source_candidate_missing_no_solved_claim:{idx}")
        if not str(candidate.get("source_ref") or candidate.get("title") or "").strip():
            reasons.append(f"literature_source_candidate_missing_identity:{idx}")
        if _scout_candidate_is_placeholder(candidate):
            placeholder_count += 1
        elif _scout_candidate_has_real_source(candidate):
            real_candidate_count += 1
        if candidate.get("verdict") == "solved" or candidate.get("route_status") == "solved":
            reasons.append("literature_scout_direct_solved_claim")
        if _contains_raw_reaction(candidate):
            reasons.append("raw_reaction_injection")
    if payload.get("accepted") and not real_candidate_count:
        reasons.append("accepted_literature_scout_without_real_source")
    if payload.get("accepted") and placeholder_count and not real_candidate_count:
        reasons.append("accepted_literature_scout_placeholder_only")
    fallback_order = payload.get("fallback_order")
    if fallback_order is not None:
        if not isinstance(fallback_order, list):
            reasons.append("literature_scout_fallback_order_not_list")
        elif [str(item) for item in fallback_order] != ["codex_online", "local_pdf", "placeholder"]:
            reasons.append("literature_scout_invalid_fallback_order")
    attempts = payload.get("scout_attempts")
    if attempts is not None and not isinstance(attempts, list):
        reasons.append("literature_scout_attempts_not_list")
    if _contains_raw_reaction(payload):
        reasons.append("raw_reaction_injection")
    return reasons


def _scout_candidate_is_placeholder(candidate: dict[str, Any]) -> bool:
    if bool(candidate.get("placeholder_only")):
        return True
    if str(candidate.get("access_status") or "").strip().lower() == "placeholder_only":
        return True
    return str(candidate.get("source_type") or "").strip().lower() == "placeholder_query"


def _scout_candidate_has_real_source(candidate: dict[str, Any]) -> bool:
    if _scout_candidate_is_placeholder(candidate):
        return False
    return bool(str(candidate.get("doi") or candidate.get("pii") or candidate.get("url") or candidate.get("local_pdf") or "").strip())


def _analogical_reaction_template_report_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "analogical_reaction_template_report.v1":
        reasons.append("invalid_analogical_template_report_schema")
    reasons.extend(_advisory_payload_guard_reasons(payload, prefix="analogical_template_report"))
    templates = payload.get("templates")
    if not isinstance(templates, list):
        reasons.append("analogical_template_report_templates_not_list")
        templates = []
    for idx, template in enumerate(templates):
        if not isinstance(template, dict):
            reasons.append(f"analogical_template_not_object:{idx}")
            continue
        reasons.extend(_analogical_template_reasons(template, idx=idx))
    if payload.get("accepted") and not templates:
        reasons.append("accepted_analogical_template_report_without_templates")
    return reasons


def _analogical_template_reasons(template: dict[str, Any], *, idx: int) -> list[str]:
    reasons: list[str] = []
    if template.get("schema_version") != "analogical_reaction_template.v1":
        reasons.append(f"invalid_analogical_template_schema:{idx}")
    if not str(template.get("template_id") or "").strip():
        reasons.append(f"analogical_template_missing_template_id:{idx}")
    if str(template.get("relation_type") or "") in {"analog", "family_precedent"} and not str(template.get("scope_gap") or "").strip():
        reasons.append(f"analogical_template_missing_scope_gap:{idx}")
    if template.get("no_solved_claim") is not True:
        reasons.append(f"analogical_template_missing_no_solved_claim:{idx}")
    if template.get("not_raw_reaction_injection") is not True:
        reasons.append(f"analogical_template_missing_raw_reaction_guard:{idx}")
    if not (template.get("source_refs") or template.get("evidence_refs")):
        reasons.append(f"analogical_template_missing_source_refs:{idx}")
    center = dict(template.get("reaction_center") or {})
    if not str(center.get("product_retron_type") or "").strip():
        reasons.append(f"analogical_template_missing_product_retron_type:{idx}")
    if template.get("verdict") == "solved" or template.get("route_status") == "solved":
        reasons.append("analogical_template_direct_solved_claim")
    if _contains_raw_reaction(template):
        reasons.append("raw_reaction_injection")
    return reasons


def _analogical_reaction_template_ranking_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "analogical_reaction_template_ranking.v1":
        reasons.append("invalid_analogical_template_ranking_schema")
    reasons.extend(_advisory_payload_guard_reasons(payload, prefix="analogical_template_ranking"))
    ranked = payload.get("ranked_templates")
    selected = payload.get("selected_templates")
    if not isinstance(ranked, list):
        reasons.append("analogical_template_ranking_ranked_templates_not_list")
        ranked = []
    if not isinstance(selected, list):
        reasons.append("analogical_template_ranking_selected_templates_not_list")
        selected = []
    for idx, row in enumerate([*ranked, *selected]):
        if not isinstance(row, dict):
            reasons.append(f"analogical_template_ranking_row_not_object:{idx}")
            continue
        if not str(row.get("template_id") or "").strip():
            reasons.append(f"analogical_template_ranking_missing_template_id:{idx}")
        if row.get("no_solved_claim") is not True:
            reasons.append(f"analogical_template_ranking_missing_no_solved_claim:{idx}")
        if row.get("verdict") == "solved" or row.get("route_status") == "solved":
            reasons.append("analogical_template_ranking_direct_solved_claim")
        if _contains_raw_reaction(row):
            reasons.append("raw_reaction_injection")
    if payload.get("accepted") and not selected:
        reasons.append("accepted_analogical_template_ranking_without_selection")
    return reasons


def _analogical_template_application_report_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "analogical_template_application_report.v1":
        reasons.append("invalid_analogical_template_application_report_schema")
    reasons.extend(_advisory_payload_guard_reasons(payload, prefix="analogical_template_application_report"))
    applications = payload.get("applications")
    if not isinstance(applications, list):
        reasons.append("analogical_template_application_report_applications_not_list")
        applications = []
    for idx, row in enumerate(applications):
        if not isinstance(row, dict):
            reasons.append(f"analogical_template_application_not_object:{idx}")
            continue
        if row.get("schema_version") != "analogical_template_application.v1":
            reasons.append(f"invalid_analogical_template_application_schema:{idx}")
        if not str(row.get("application_id") or "").strip():
            reasons.append(f"analogical_template_application_missing_application_id:{idx}")
        if not str(row.get("template_id") or "").strip():
            reasons.append(f"analogical_template_application_missing_template_id:{idx}")
        if str(row.get("allowed_use") or "") not in {
            "forbidden",
            "advisory_or_rerank_only",
            "executable_candidate",
            "hypothesis_only_not_solved",
        }:
            reasons.append(f"invalid_analogical_template_application_allowed_use:{idx}")
        if row.get("no_solved_claim") is not True:
            reasons.append(f"analogical_template_application_missing_no_solved_claim:{idx}")
        if row.get("verdict") == "solved" or row.get("route_status") == "solved":
            reasons.append("analogical_template_application_direct_solved_claim")
        if row.get("candidate_payload_redacted") is False and row.get("executable_candidate_available") and row.get("allowed_use") != "executable_candidate":
            reasons.append(f"analogical_template_application_unredacted_nonexecutable_candidate:{idx}")
        if _contains_raw_reaction(row):
            reasons.append("raw_reaction_injection")
    if payload.get("accepted") and not applications:
        reasons.append("accepted_analogical_template_application_report_without_applications")
    return reasons


def _analogical_template_application_validation_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != "analogical_template_application_validation.v1":
        reasons.append("invalid_analogical_template_application_validation_schema")
    reasons.extend(_advisory_payload_guard_reasons(payload, prefix="analogical_template_application_validation"))
    if not isinstance(payload.get("compiled_downstream_refs"), dict):
        reasons.append("analogical_template_validation_missing_compiled_downstream_refs")
    if int(payload.get("one_step_row_count") or 0) > 0 and not bool(payload.get("compiled_downstream")):
        reasons.append("analogical_template_validation_rows_missing_compiled_downstream")
    if payload.get("accepted") and int(payload.get("one_step_row_count") or 0) <= 0:
        reasons.append("accepted_analogical_template_validation_without_rows")
    reasons.extend(_analogical_template_guided_hint_boundary_reasons(payload))
    return reasons


def _analogical_template_guided_hint_boundary_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("evidence_class") != "analogical_template_hint":
        reasons.append("analogical_template_validation_missing_hint_evidence_class")
    if payload.get("allowed_use") != "guided_search_hint_only":
        reasons.append("analogical_template_validation_not_guided_hint_only")
    for flag in (
        "analogy_is_advisory_only",
        "not_exact_literature_segment",
        "not_parent_route_proof",
        "requires_verifier",
        "requires_parent_route_proof",
        "production_write_blocked",
    ):
        if payload.get(flag) is not True:
            reasons.append(f"analogical_template_validation_missing_{flag}")
    compiled = dict(payload.get("compiled_downstream") or {})
    if not compiled:
        return reasons
    if compiled.get("schema_version") == "compiled_downstream_consumables.v1":
        reasons.append("analogical_template_validation_uses_exact_downstream_schema")
    if compiled.get("schema_version") != "analogical_template_guided_hints.v1":
        reasons.append("analogical_template_validation_invalid_guided_hint_schema")
    for flag in (
        "analogy_is_advisory_only",
        "not_exact_literature_segment",
        "not_parent_route_proof",
        "requires_verifier",
        "requires_parent_route_proof",
        "production_write_blocked",
    ):
        if compiled.get(flag) is not True:
            reasons.append(f"analogical_template_guided_hints_missing_{flag}")
    if compiled.get("allowed_use") != "guided_search_hint_only":
        reasons.append("analogical_template_guided_hints_not_guided_hint_only")
    exact_plugin = dict(compiled.get("literature_template_plugin") or {})
    exact_flags = dict(exact_plugin.get("plugin_flags") or {})
    if exact_plugin.get("enabled") or exact_flags.get("enabled"):
        reasons.append("analogical_template_guided_hints_enabled_exact_literature_plugin")
    if exact_plugin.get("one_step_rows") or exact_flags.get("one_step_rows"):
        reasons.append("analogical_template_guided_hints_exposes_exact_one_step_rows")
    hints = dict(compiled.get("analogical_template_hints") or compiled.get("guided_search_hints") or {})
    hint_flags = dict(hints.get("plugin_flags") or {})
    hint_rows = [dict(row) for row in hints.get("one_step_rows") or [] if isinstance(row, dict)]
    if int(payload.get("one_step_row_count") or 0) > 0 and not hint_rows:
        reasons.append("analogical_template_guided_hints_missing_hint_rows")
    if hint_flags.get("enabled") is True:
        reasons.append("analogical_template_guided_hints_plugin_flags_enable_exact_replay")
    if hint_flags.get("guided_hint_enabled") is not bool(hint_rows):
        reasons.append("analogical_template_guided_hints_inconsistent_hint_enabled_flag")
    for idx, row in enumerate(hint_rows):
        reasons.extend(_analogical_template_hint_row_reasons(row, idx=idx))
    return reasons


def _analogical_template_hint_row_reasons(row: dict[str, Any], *, idx: int) -> list[str]:
    reasons: list[str] = []
    for field, expected in (
        ("row_source", "analogical_template_application"),
        ("evidence_class", "analogical_template_hint"),
        ("allowed_use", "guided_search_hint_only"),
        ("source_policy_decision", "analogical_guided_hint_only"),
    ):
        if row.get(field) != expected:
            reasons.append(f"analogical_template_hint_row_{idx}_invalid_{field}")
    for flag in (
        "used_as_proof",
        "not_exact_literature_segment",
        "not_parent_route_proof",
        "requires_verifier",
        "requires_parent_route_proof",
        "production_write_blocked",
        "no_solved_claim",
    ):
        value = row.get(flag)
        if flag == "used_as_proof":
            if value is not False:
                reasons.append(f"analogical_template_hint_row_{idx}_used_as_proof")
        elif value is not True:
            reasons.append(f"analogical_template_hint_row_{idx}_missing_{flag}")
    if row.get("source_policy_decision") == "enabled_literature_template_plugin":
        reasons.append(f"analogical_template_hint_row_{idx}_enabled_exact_literature_plugin")
    template = dict(row.get("template") or row.get("templates") or {})
    if template.get("source_policy_decision") == "enabled_literature_template_plugin":
        reasons.append(f"analogical_template_hint_row_{idx}_template_enabled_exact_literature_plugin")
    if template.get("evidence_class") != "analogical_template_hint":
        reasons.append(f"analogical_template_hint_row_{idx}_template_missing_hint_evidence_class")
    if template.get("used_as_proof") is not False:
        reasons.append(f"analogical_template_hint_row_{idx}_template_used_as_proof")
    trace = dict(row.get("literature_template_trace") or template.get("literature_template_trace") or {})
    if trace.get("source_detail_exact_step") is True or trace.get("structured_segment_step") is True:
        reasons.append(f"analogical_template_hint_row_{idx}_claims_exact_source_detail_step")
    if trace.get("analogical_template_hint") is not True:
        reasons.append(f"analogical_template_hint_row_{idx}_trace_missing_analogical_hint_flag")
    return reasons


def _advisory_payload_guard_reasons(payload: dict[str, Any], *, prefix: str) -> list[str]:
    reasons: list[str] = []
    if not isinstance(payload.get("accepted"), bool):
        reasons.append(f"{prefix}_missing_accepted_boolean")
    if payload.get("no_solved_claim") is not True:
        reasons.append(f"{prefix}_missing_no_solved_claim")
    if payload.get("verdict") == "solved" or payload.get("route_status") == "solved" or payload.get("solved") is True:
        reasons.append(f"{prefix}_direct_solved_claim")
    if _contains_raw_reaction(payload):
        reasons.append("raw_reaction_injection")
    return reasons


def _structure_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    smiles = str(payload.get("target_smiles") or payload.get("smiles") or payload.get("input_smiles") or "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["invalid_smiles"]
    canonical = payload.get("canonical_smiles")
    if canonical:
        can_mol = Chem.MolFromSmiles(str(canonical))
        if can_mol is None or Chem.MolToSmiles(can_mol, isomericSmiles=False) != Chem.MolToSmiles(mol, isomericSmiles=False):
            reasons.append("canonical_smiles_mismatch")
    inchi_key = payload.get("inchi_key")
    if inchi_key and str(inchi_key) != MolToInchiKey(mol):
        reasons.append("inchi_key_mismatch")
    chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    has_unassigned = any(flag == "?" for _, flag in chiral)
    if has_unassigned and not payload.get("target_ambiguity"):
        reasons.append("target_ambiguity_not_marked")
    return reasons


def _evidence_reasons(payload: dict[str, Any]) -> list[str]:
    result = validate_evidence_card(payload)
    reasons = list(result.get("reasons") or [])
    if payload.get("validation_status") not in {"draft", "validated"}:
        reasons.append("invalid_evidence_validation_status")
    return reasons


def _strategic_disconnection_reasons(payload: dict[str, Any], evidence_refs: set[str]) -> list[str]:
    reasons: list[str] = []
    refs = {str(ref) for ref in payload.get("evidence_refs") or []}
    if not refs:
        reasons.append("missing_evidence_refs")
    missing = sorted(refs.difference(evidence_refs))
    if missing:
        reasons.append("unvalidated_evidence_refs")
    if not payload.get("candidate_kind") and not payload.get("retrosynthetic_move"):
        reasons.append("missing_disconnection_content")
    if _contains_raw_reaction(payload):
        reasons.append("raw_reaction_injection")
    return reasons


def _literature_route_segment_reasons(payload: dict[str, Any]) -> list[str]:
    result = validate_literature_route_segment(payload)
    return list(result.get("reasons") or [])


def _segment_step_reasons(payload: dict[str, Any]) -> list[str]:
    result = validate_segment_step(payload)
    return list(result.get("reasons") or [])


def _strategic_operator_reasons(
    payload: dict[str, Any],
    evidence_refs: set[str],
    disconnection_refs: set[str],
) -> list[str]:
    result = validate_strategic_operator(payload)
    reasons = list(result.get("reasons") or [])
    refs = {str(ref) for ref in payload.get("evidence_refs") or []}
    if refs and refs.difference(evidence_refs):
        reasons.append("unvalidated_evidence_refs")
    input_refs = {str(ref) for ref in payload.get("input_artifact_refs") or []}
    if disconnection_refs and input_refs and not input_refs.intersection(disconnection_refs):
        reasons.append("missing_validated_strategic_disconnection_ref")
    if payload.get("budget", {}).get("max_reruns", 0) > 1:
        reasons.append("rerun_budget_not_bounded")
    return reasons


def _search_policy_reasons(payload: dict[str, Any], evidence_refs: set[str]) -> list[str]:
    result = validate_chem_enzy_search_policy_payload(payload)
    reasons = list(result.get("reasons") or [])
    refs = {str(ref) for ref in payload.get("evidence_refs") or []}
    if refs and refs.difference(evidence_refs):
        reasons.append("unvalidated_evidence_refs")
    if _contains_raw_reaction(payload):
        reasons.append("raw_reaction_injection")
    return reasons


def _judge_policy_reasons(payload: dict[str, Any]) -> list[str]:
    result = validate_judge_policy(payload)
    return list(result.get("reasons") or [])


def _condition_reasons(payload: dict[str, Any]) -> list[str]:
    result = validate_condition_candidate(payload)
    return list(result.get("reasons") or [])


def _route_audit_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    status = str(payload.get("route_status") or "")
    if not status:
        reasons.append("missing_route_status")
    reasons.extend(_route_status_reasons(payload))
    return reasons


def _route_status_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    status = str(payload.get("route_status") or payload.get("status") or "")
    allowed = {"solved", "semisynthesis_closed", "partial_anchor", "fake_closed_rejected", "unresolved"}
    if status not in allowed:
        reasons.append("invalid_route_status")
        return reasons
    if status == "solved" and not payload.get("stock_audit_passed"):
        reasons.append("solved_without_stock_audit")
    if status == "semisynthesis_closed" and not payload.get("anchor_evidence_refs"):
        reasons.append("semisynthesis_closed_without_anchor_evidence")
    if status == "fake_closed_rejected" and not (payload.get("rejected_terminal_refs") or payload.get("rejected_step_refs")):
        reasons.append("fake_closed_rejected_without_rejection_ref")
    if status == "unresolved" and not payload.get("reason"):
        reasons.append("unresolved_without_reason")
    return reasons


def _evolution_reasons(payload: dict[str, Any]) -> list[str]:
    result = validate_evolution_candidate(payload)
    return list(result.get("reasons") or [])


def _statin_panel_self_evo_report_reasons(payload: dict[str, Any]) -> list[str]:
    from cascade_planner.agent.statin_panel import STATIN_PANEL_REPORT_SCHEMA

    reasons: list[str] = []
    if payload.get("schema_version") != STATIN_PANEL_REPORT_SCHEMA:
        reasons.append("invalid_statin_panel_report_schema")
    if payload.get("run_semantics") != "replay":
        reasons.append("statin_panel_report_must_be_replay_semantics")
    if int(payload.get("target_count") or 0) != 9:
        reasons.append("statin_panel_report_not_all_nine")
    if int(payload.get("failed") or 0) != 0:
        reasons.append("statin_panel_report_has_failed_targets")
    hard_gates = dict(payload.get("hard_gates") or {})
    if not hard_gates:
        reasons.append("missing_statin_panel_hard_gates")
    for key, value in hard_gates.items():
        if not bool(value):
            reasons.append(f"statin_panel_hard_gate_failed:{key}")
    targets = list(payload.get("targets") or [])
    if len(targets) != 9:
        reasons.append("statin_panel_target_rows_not_all_nine")
    for row in targets:
        safe = str(row.get("safe") or row.get("name") or "unknown")
        if row.get("route_status") == "solved" or row.get("claims_solved"):
            reasons.append(f"statin_target_claimed_solved:{safe}")
        dossier = dict(row.get("fullflow_dossier") or {})
        if not (dossier.get("validation") or {}).get("accepted"):
            reasons.append(f"statin_target_dossier_not_validated:{safe}")
        self_evo = dict(row.get("self_evolution") or {})
        if self_evo.get("kb_target_layer") != "staging":
            reasons.append(f"statin_target_self_evo_not_staging:{safe}")
        if not self_evo.get("production_write_blocked"):
            reasons.append(f"statin_target_production_write_not_blocked:{safe}")
    aggregation = dict(payload.get("self_evolution_aggregation") or {})
    if not aggregation.get("accepted"):
        reasons.append("statin_self_evo_aggregation_not_accepted")
    if int(aggregation.get("production_promoted_count") or 0) != 0:
        reasons.append("statin_replay_promoted_production")
    if not all((row.get("staging_promoted") and not row.get("production_promoted")) for row in aggregation.get("families") or []):
        reasons.append("statin_self_evo_family_templates_not_staging_only")
    production = set(
        (payload.get("self_evolution_kb") or {})
        .get("layers", {})
        .get("production", {})
        .keys()
    )
    if production:
        reasons.append("statin_self_evo_replay_production_layer_not_empty")
    return reasons


def _statin_fullflow_dossier_reasons(payload: dict[str, Any]) -> list[str]:
    from cascade_planner.agent.statin_panel import validate_statin_fullflow_dossier

    result = validate_statin_fullflow_dossier(payload)
    return list(result.get("reasons") or [])


def _statin_fullflow_overview_reasons(payload: dict[str, Any]) -> list[str]:
    from cascade_planner.agent.statin_panel import validate_statin_fullflow_overview

    result = validate_statin_fullflow_overview(payload)
    return list(result.get("reasons") or [])


def _statin_route_template_reasons(payload: dict[str, Any]) -> list[str]:
    from cascade_planner.agent.statin_panel import validate_statin_route_template

    result = validate_statin_route_template(payload)
    return list(result.get("reasons") or [])


def _statin_route_closure_audit_reasons(payload: dict[str, Any]) -> list[str]:
    from cascade_planner.agent.statin_panel import validate_statin_route_closure_audit

    result = validate_statin_route_closure_audit(payload)
    return list(result.get("reasons") or [])


def _statin_route_closure_matrix_reasons(payload: dict[str, Any]) -> list[str]:
    from cascade_planner.agent.statin_panel import validate_statin_route_closure_matrix

    result = validate_statin_route_closure_matrix(payload)
    return list(result.get("reasons") or [])


def _statin_closure_lead_curation_packet_reasons(payload: dict[str, Any]) -> list[str]:
    from cascade_planner.agent.statin_panel import validate_statin_closure_lead_curation_packet

    result = validate_statin_closure_lead_curation_packet(payload)
    return list(result.get("reasons") or [])


def _statin_closure_curation_result_set_reasons(payload: dict[str, Any]) -> list[str]:
    from cascade_planner.agent.statin_panel import validate_statin_closure_curation_result_set

    result = validate_statin_closure_curation_result_set(payload)
    return list(result.get("reasons") or [])


def _failure_diagnosis_reasons(payload: dict[str, Any]) -> list[str]:
    if not payload.get("failure_mode") and not payload.get("reason"):
        return ["missing_failure_mode"]
    if _contains_raw_reaction(payload):
        return ["raw_reaction_injection"]
    return []


def _worker_run_record_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("status") == "accepted_draft" and not payload.get("output_validation", {}).get("accepted"):
        reasons.append("accepted_worker_run_without_output_validation")
    if payload.get("final_route_status") == "solved":
        reasons.append("worker_run_direct_solved_claim")
    return reasons


def _result(accepted: bool, reasons: list[str], artifact_id: str, artifact_type: str) -> dict[str, Any]:
    return {
        "schema_version": VALIDATOR_SCHEMA,
        "accepted": accepted,
        "reasons": reasons,
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
    }


def validate_literature_candidate_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    return validate_literature_candidate(payload)

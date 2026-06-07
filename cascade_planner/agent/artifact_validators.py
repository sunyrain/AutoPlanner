"""Deterministic validators for agent typed artifacts."""
from __future__ import annotations

from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

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

    if artifact_type in {"TargetResolution", "StructureProfile"}:
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


def _contains_raw_reaction(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {
                "rxn",
                "rxn_smiles",
                "rxn_smiles_list",
                "reaction_smiles",
                "raw_reaction",
                "raw_reactions",
                "raw_reaction_candidates",
                "reaction_candidates",
            }:
                return True
            if _contains_raw_reaction(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_reaction(item) for item in value)
    if isinstance(value, str):
        return ">>" in value
    return False


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

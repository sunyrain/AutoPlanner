"""Contracts for literature-derived executable template artifacts.

The classes here separate advisory planning material from deterministic
one-step proposal material.  A literature record can influence search policy as
an advisory strategy, but it cannot enter ChemEnzy one-step expansion unless it
has passed product-specific applicability and validation gates.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

LITERATURE_TEMPLATE_CARD_SCHEMA = "literature_template_card.v1"
TEMPLATE_APPLICABILITY_REPORT_SCHEMA = "template_applicability_report.v1"
EXECUTABLE_TEMPLATE_CANDIDATE_SCHEMA = "executable_template_candidate.v1"
TEMPLATE_VALIDATION_REPORT_SCHEMA = "template_validation_report.v1"
LITERATURE_TRIGGER_REPORT_SCHEMA = "literature_trigger_report.v1"
ROUTE_ANCHOR_EXPANSION_TASK_SCHEMA = "route_anchor_expansion_task.v1"
ROUTE_ANCHOR_STITCH_REPORT_SCHEMA = "route_anchor_stitch_report.v1"
TEMPLATE_COMPLIANCE_GATE_SCHEMA = "template_compliance_gate.v1"
TEMPLATE_KB_PROMOTION_GATE_SCHEMA = "template_kb_promotion_gate.v1"

LITERATURE_TEMPLATE_PLUGIN_SOURCE = "literature_template_plugin"
LITERATURE_TEMPLATE_PLUGIN_MODEL = "autoplanner.literature_template_plugin"


class LiteratureTriggerReason(str, Enum):
    NATIVE_FAILED = "native_failed"
    UNCLOSED_ROUTE = "unclosed_route"
    FAKE_CLOSURE_RISK = "fake_closure_risk"
    ADVANCED_FRONTIER_DETECTED = "advanced_frontier_detected"
    ROUTE_AUDIT_FAILED = "route_audit_failed"
    USER_REQUESTED_LITERATURE = "user_requested_literature"


class LiteratureTemplateLevel(str, Enum):
    ADVISORY_STRATEGY = "advisory_strategy"
    RETRON_PATTERN = "retron_pattern"
    EXECUTABLE_TEMPLATE_CANDIDATE = "executable_template_candidate"
    VALIDATED_EXECUTABLE_TEMPLATE = "validated_executable_template"
    ROUTE_ANCHOR_ONLY = "route_anchor_only"


class RouteAnchorStatus(str, Enum):
    SOLVED = "solved"
    SEMISYNTHESIS_CLOSED = "semisynthesis_closed"
    PARTIAL_ANCHOR = "partial_anchor"
    UNRESOLVED = "unresolved"


@dataclass
class LiteratureTemplateCard:
    template_id: str
    evidence_refs: list[str]
    reaction_class: str
    template_level: str
    product_retron: dict[str, Any]
    break_bonds: list[dict[str, Any]] = field(default_factory=list)
    precursor_roles: list[str] = field(default_factory=list)
    applicability: dict[str, Any] = field(default_factory=dict)
    scope_limits: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    promotion_status: str = "draft"
    source_family: str = ""
    condition_source: str = "unknown"
    not_raw_reaction_injection: bool = True
    schema_version: str = LITERATURE_TEMPLATE_CARD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TemplateApplicabilityReport:
    target_smiles: str
    frontier_smiles: str
    matched_retron_atoms: list[list[int]] = field(default_factory=list)
    matched_bonds: list[dict[str, Any]] = field(default_factory=list)
    match_confidence: str = "none"
    mismatch_reasons: list[str] = field(default_factory=list)
    allowed_use: str = "forbidden"
    ambiguity_count: int = 0
    selected_bond: dict[str, Any] | None = None
    cut_fragments: list[str] = field(default_factory=list)
    retron_type: str = ""
    template_id: str = ""
    schema_version: str = TEMPLATE_APPLICABILITY_REPORT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutableTemplateCandidate:
    product_smiles: str
    reactant_smiles: list[str]
    rxn_smiles: str
    atom_mapping_status: str
    template_smarts: str
    source_template_id: str
    not_lab_procedure: bool
    proposal_source: str = LITERATURE_TEMPLATE_PLUGIN_SOURCE
    evidence_refs: list[str] = field(default_factory=list)
    precursor_roles: list[dict[str, Any]] = field(default_factory=list)
    applicability_report: dict[str, Any] = field(default_factory=dict)
    validation_report: dict[str, Any] = field(default_factory=dict)
    literature_template_trace: dict[str, Any] = field(default_factory=dict)
    requires_audit: bool = True
    condition_source: str = "unknown"
    schema_version: str = EXECUTABLE_TEMPLATE_CANDIDATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TemplateValidationReport:
    accepted: bool
    reasons: list[str]
    confidence: str
    allowed_for_one_step_source: bool
    source_template_id: str = ""
    reconstruction_report: dict[str, Any] = field(default_factory=dict)
    chemical_sanity: dict[str, Any] = field(default_factory=dict)
    audit_required: bool = True
    no_solved_claim: bool = True
    schema_version: str = TEMPLATE_VALIDATION_REPORT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteAnchorExpansionTask:
    anchor_smiles: str
    source_evidence: list[str]
    required_closure_type: str
    parent_route_reference: str
    anchor_name: str = ""
    child_target_name: str = ""
    child_target_smiles: str = ""
    source_template_id: str = ""
    native_first: bool = True
    trigger_literature_only_after_native_gap: bool = True
    schema_version: str = ROUTE_ANCHOR_EXPANSION_TASK_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_native_run_for_literature(
    native_result: Any | None = None,
    *,
    route_audit: Any | None = None,
    frontier_report: dict[str, Any] | None = None,
    user_requested: bool = False,
) -> dict[str, Any]:
    """Return explicit reasons that allow literature-template work to start."""
    result = _result_dict(native_result)
    audit = _audit_dict(route_audit)
    reasons: list[str] = []
    route_count = int(result.get("route_count") or len(result.get("routes") or []))
    solved = bool(result.get("solved"))
    failures = [dict(item) for item in result.get("failures") or [] if isinstance(item, dict)]
    route_status = str(audit.get("route_status") or "")
    stock_audit_passed = bool(audit.get("stock_audit_passed"))
    fake_rejected = bool(audit.get("fake_closure_rejected")) or bool(audit.get("rejected_terminal_list"))

    if user_requested:
        reasons.append(LiteratureTriggerReason.USER_REQUESTED_LITERATURE.value)
    if route_count == 0 or any(str(f.get("category") or "") == "no_route_found" for f in failures):
        reasons.append(LiteratureTriggerReason.NATIVE_FAILED.value)
    if route_count > 0 and (not solved or not _all_route_leaves_stock_closed(result)):
        reasons.append(LiteratureTriggerReason.UNCLOSED_ROUTE.value)
    if fake_rejected or _has_fake_closure_reason(audit):
        reasons.append(LiteratureTriggerReason.FAKE_CLOSURE_RISK.value)
    if _advanced_frontier_detected(frontier_report or {}, audit):
        reasons.append(LiteratureTriggerReason.ADVANCED_FRONTIER_DETECTED.value)
    if route_status in {"fake_closed_rejected", "unresolved"} and audit.get("reasons"):
        reasons.append(LiteratureTriggerReason.ROUTE_AUDIT_FAILED.value)

    native_audit_passed = bool(
        solved
        and route_count > 0
        and stock_audit_passed
        and not fake_rejected
        and route_status in {"", "solved", "semisynthesis_closed"}
        and _all_route_leaves_stock_closed(result)
    )
    if native_audit_passed and not user_requested:
        reasons = []
    reasons = _dedupe_ordered(reasons)
    return {
        "schema_version": LITERATURE_TRIGGER_REPORT_SCHEMA,
        "should_trigger": bool(reasons),
        "trigger_reasons": reasons,
        "native_audit_passed": native_audit_passed,
        "route_count": route_count,
        "solved": solved,
        "audit_summary": {
            "route_status": route_status,
            "stock_audit_passed": stock_audit_passed,
            "fake_closure_rejected": fake_rejected,
            "frontier_reasons": list((frontier_report or {}).get("reasons") or []),
        },
    }


def validate_literature_template_card(card_or_data: LiteratureTemplateCard | dict[str, Any]) -> dict[str, Any]:
    card = card_or_data if isinstance(card_or_data, LiteratureTemplateCard) else template_card_from_dict(card_or_data)
    reasons: list[str] = []
    if card.schema_version != LITERATURE_TEMPLATE_CARD_SCHEMA:
        reasons.append("invalid_template_card_schema")
    if not card.template_id:
        reasons.append("missing_template_id")
    if not card.evidence_refs:
        reasons.append("missing_evidence_refs")
    if not card.reaction_class:
        reasons.append("missing_reaction_class")
    if card.template_level not in {level.value for level in LiteratureTemplateLevel}:
        reasons.append("invalid_template_level")
    if not isinstance(card.product_retron, dict):
        reasons.append("missing_product_retron")
    elif card.template_level in {
        LiteratureTemplateLevel.RETRON_PATTERN.value,
        LiteratureTemplateLevel.EXECUTABLE_TEMPLATE_CANDIDATE.value,
        LiteratureTemplateLevel.VALIDATED_EXECUTABLE_TEMPLATE.value,
    } and not card.product_retron.get("retron_type"):
        reasons.append("missing_retron_type")
    if not card.not_raw_reaction_injection:
        reasons.append("raw_reaction_injection_not_allowed")
    if _template_is_route_anchor_only(card) and card.promotion_status == "validated_for_one_step":
        reasons.append("route_anchor_cannot_be_promoted_for_one_step")
    if _template_is_advisory_only(card) and direct_consumption_allowed(card):
        reasons.append("advisory_template_cannot_be_direct_consumed")
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "template_id": card.template_id,
        "direct_consumption_allowed": direct_consumption_allowed(card),
        "schema_version": "literature_template_card_validation.v1",
    }


def validate_template_applicability_report(report_or_data: TemplateApplicabilityReport | dict[str, Any]) -> dict[str, Any]:
    report = (
        report_or_data
        if isinstance(report_or_data, TemplateApplicabilityReport)
        else applicability_report_from_dict(report_or_data)
    )
    reasons: list[str] = []
    if report.schema_version != TEMPLATE_APPLICABILITY_REPORT_SCHEMA:
        reasons.append("invalid_applicability_report_schema")
    if not _valid_smiles(report.target_smiles):
        reasons.append("invalid_target_smiles")
    if not _valid_smiles(report.frontier_smiles):
        reasons.append("invalid_frontier_smiles")
    if report.allowed_use == "executable_candidate" and not report.selected_bond:
        reasons.append("executable_use_without_selected_bond")
    if report.allowed_use == "executable_candidate" and report.mismatch_reasons:
        reasons.append("executable_use_with_mismatch")
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "allowed_use": report.allowed_use,
        "schema_version": "template_applicability_report_validation.v1",
    }


def validate_executable_template_candidate(candidate_or_data: ExecutableTemplateCandidate | dict[str, Any]) -> dict[str, Any]:
    candidate = (
        candidate_or_data
        if isinstance(candidate_or_data, ExecutableTemplateCandidate)
        else executable_candidate_from_dict(candidate_or_data)
    )
    reasons: list[str] = []
    if candidate.schema_version != EXECUTABLE_TEMPLATE_CANDIDATE_SCHEMA:
        reasons.append("invalid_executable_candidate_schema")
    if not _valid_smiles(candidate.product_smiles):
        reasons.append("invalid_product_smiles")
    if not candidate.reactant_smiles:
        reasons.append("missing_reactant_smiles")
    for smiles in candidate.reactant_smiles:
        if not _valid_smiles(smiles):
            reasons.append("invalid_reactant_smiles")
            break
    if not candidate.rxn_smiles or candidate.rxn_smiles.count(">>") != 1:
        reasons.append("invalid_rxn_smiles")
    if not candidate.source_template_id:
        reasons.append("missing_source_template_id")
    if not candidate.not_lab_procedure:
        reasons.append("missing_not_lab_procedure_guard")
    if candidate.proposal_source != LITERATURE_TEMPLATE_PLUGIN_SOURCE:
        reasons.append("invalid_proposal_source")
    app_validation = validate_template_applicability_report(candidate.applicability_report or {})
    if not app_validation["accepted"]:
        reasons.append("invalid_applicability_report")
    validation = dict(candidate.validation_report or {})
    if validation and not validation.get("allowed_for_one_step_source"):
        reasons.append("validation_report_disallows_one_step")
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "source_template_id": candidate.source_template_id,
        "schema_version": "executable_template_candidate_validation.v1",
    }


def direct_consumption_allowed(card_or_data: LiteratureTemplateCard | dict[str, Any]) -> bool:
    card = card_or_data if isinstance(card_or_data, LiteratureTemplateCard) else template_card_from_dict(card_or_data)
    if card.template_level not in {
        LiteratureTemplateLevel.EXECUTABLE_TEMPLATE_CANDIDATE.value,
        LiteratureTemplateLevel.VALIDATED_EXECUTABLE_TEMPLATE.value,
    }:
        return False
    if _template_is_route_anchor_only(card):
        return False
    if str(card.promotion_status or "") in {"rejected", "draft_only", "route_anchor_only"}:
        return False
    return True


def template_card_from_dict(data: dict[str, Any]) -> LiteratureTemplateCard:
    return LiteratureTemplateCard(
        template_id=str(data.get("template_id") or ""),
        evidence_refs=[str(item) for item in data.get("evidence_refs") or []],
        reaction_class=str(data.get("reaction_class") or ""),
        template_level=str(data.get("template_level") or LiteratureTemplateLevel.ADVISORY_STRATEGY.value),
        product_retron=dict(data.get("product_retron") or {}),
        break_bonds=[dict(item) for item in data.get("break_bonds") or [] if isinstance(item, dict)],
        precursor_roles=[str(item) for item in data.get("precursor_roles") or []],
        applicability=dict(data.get("applicability") or {}),
        scope_limits=[str(item) for item in data.get("scope_limits") or []],
        safety_flags=[str(item) for item in data.get("safety_flags") or []],
        promotion_status=str(data.get("promotion_status") or "draft"),
        source_family=str(data.get("source_family") or ""),
        condition_source=str(data.get("condition_source") or "unknown"),
        not_raw_reaction_injection=bool(data.get("not_raw_reaction_injection", True)),
        schema_version=str(data.get("schema_version") or LITERATURE_TEMPLATE_CARD_SCHEMA),
    )


def applicability_report_from_dict(data: dict[str, Any]) -> TemplateApplicabilityReport:
    return TemplateApplicabilityReport(
        target_smiles=str(data.get("target_smiles") or ""),
        frontier_smiles=str(data.get("frontier_smiles") or ""),
        matched_retron_atoms=[[int(v) for v in item] for item in data.get("matched_retron_atoms") or []],
        matched_bonds=[dict(item) for item in data.get("matched_bonds") or [] if isinstance(item, dict)],
        match_confidence=str(data.get("match_confidence") or "none"),
        mismatch_reasons=[str(item) for item in data.get("mismatch_reasons") or []],
        allowed_use=str(data.get("allowed_use") or "forbidden"),
        ambiguity_count=int(data.get("ambiguity_count") or 0),
        selected_bond=dict(data["selected_bond"]) if isinstance(data.get("selected_bond"), dict) else None,
        cut_fragments=[str(item) for item in data.get("cut_fragments") or []],
        retron_type=str(data.get("retron_type") or ""),
        template_id=str(data.get("template_id") or ""),
        schema_version=str(data.get("schema_version") or TEMPLATE_APPLICABILITY_REPORT_SCHEMA),
    )


def executable_candidate_from_dict(data: dict[str, Any]) -> ExecutableTemplateCandidate:
    return ExecutableTemplateCandidate(
        product_smiles=str(data.get("product_smiles") or ""),
        reactant_smiles=[str(item) for item in data.get("reactant_smiles") or []],
        rxn_smiles=str(data.get("rxn_smiles") or ""),
        atom_mapping_status=str(data.get("atom_mapping_status") or ""),
        template_smarts=str(data.get("template_smarts") or ""),
        source_template_id=str(data.get("source_template_id") or ""),
        not_lab_procedure=bool(data.get("not_lab_procedure")),
        proposal_source=str(data.get("proposal_source") or LITERATURE_TEMPLATE_PLUGIN_SOURCE),
        evidence_refs=[str(item) for item in data.get("evidence_refs") or []],
        precursor_roles=[dict(item) for item in data.get("precursor_roles") or [] if isinstance(item, dict)],
        applicability_report=dict(data.get("applicability_report") or {}),
        validation_report=dict(data.get("validation_report") or {}),
        literature_template_trace=dict(data.get("literature_template_trace") or {}),
        requires_audit=bool(data.get("requires_audit", True)),
        condition_source=str(data.get("condition_source") or "unknown"),
        schema_version=str(data.get("schema_version") or EXECUTABLE_TEMPLATE_CANDIDATE_SCHEMA),
    )


def template_card_from_advisory_strategy(
    strategy_template: dict[str, Any],
    *,
    template_id: str = "",
    evidence_refs: list[str] | None = None,
) -> LiteratureTemplateCard:
    payload = dict(strategy_template or {})
    seed = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return LiteratureTemplateCard(
        template_id=template_id or f"advisory_{digest}",
        evidence_refs=list(evidence_refs or payload.get("evidence_refs") or ["advisory_source"]),
        reaction_class=str(payload.get("reaction_class") or "strategic_disconnection"),
        template_level=LiteratureTemplateLevel.ADVISORY_STRATEGY.value,
        product_retron={},
        break_bonds=[
            {"label": str(item), "source": "advisory_strategy_template"}
            for item in payload.get("break_bonds") or []
        ],
        precursor_roles=[str(item) for item in payload.get("suggested_precursor_roles") or []],
        applicability={"direct_one_step_consumption": False},
        scope_limits=["advisory_strategy_template.v1 cannot be consumed directly by ChemEnzy one-step"],
        safety_flags=["not_raw_reaction_injection"],
        promotion_status="advisory_only",
        not_raw_reaction_injection=True,
    )


def default_literature_template_cards() -> list[LiteratureTemplateCard]:
    """Return deterministic template cards for the checklist/MVP families."""
    return [
        LiteratureTemplateCard(
            template_id="lit_tpl_o_glycoside_split_v1",
            evidence_refs=["ev_glycosylation_precedent"],
            reaction_class="glycosylation",
            template_level=LiteratureTemplateLevel.EXECUTABLE_TEMPLATE_CANDIDATE.value,
            product_retron={
                "retron_type": "o_glycoside",
                "description": "anomeric C-O linkage from sugar ring to O/N/S acceptor",
                "smarts": "[O,N,S]-[C;R]1-[O;R]-[C;R]-[C;R]-[C;R]-[C;R]-1",
            },
            precursor_roles=["aglycone_acceptor", "sugar_donor_or_precursor"],
            break_bonds=[{"role": "glycosidic_linkage", "atoms": ["hetero_acceptor", "anomeric_carbon"]}],
            promotion_status="candidate",
            source_family="glycoside",
            condition_source="literature_analog",
        ),
        LiteratureTemplateCard(
            template_id="lit_tpl_c_glycoside_split_v1",
            evidence_refs=["ev_c_glycoside_precedent"],
            reaction_class="C_glycosylation",
            template_level=LiteratureTemplateLevel.EXECUTABLE_TEMPLATE_CANDIDATE.value,
            product_retron={
                "retron_type": "c_glycoside",
                "description": "anomeric C-C aryl linkage",
                "smarts": "[c]-[C;R]1-[O;R]-[C;R]-[C;R]-[C;R]-[C;R]-1",
            },
            precursor_roles=["aryl_acceptor", "sugar_coupling_partner"],
            break_bonds=[{"role": "anomeric_c_aryl_bond", "atoms": ["aryl_carbon", "anomeric_carbon"]}],
            promotion_status="candidate",
            source_family="glycoside",
            condition_source="literature_analog",
        ),
        LiteratureTemplateCard(
            template_id="lit_tpl_macrolactone_split_v1",
            evidence_refs=["ev_macrolactonization_precedent"],
            reaction_class="macrolactonization",
            template_level=LiteratureTemplateLevel.EXECUTABLE_TEMPLATE_CANDIDATE.value,
            product_retron={
                "retron_type": "macrolactone",
                "description": "macrocyclic ester acyl C-O bond",
                "smarts": "[C;R](=O)-[O;R]",
            },
            precursor_roles=["seco_acid", "hydroxy_acid_precursor"],
            break_bonds=[{"role": "macrocyclic_ester_acyl_o_bond", "atoms": ["carbonyl_carbon", "ester_oxygen"]}],
            promotion_status="candidate",
            source_family="macrolide",
            condition_source="literature_analog",
        ),
        LiteratureTemplateCard(
            template_id="lit_tpl_taxane_c13_side_chain_split_v1",
            evidence_refs=["ev_taxane_semisynthesis_precedent"],
            reaction_class="taxane_side_chain_acylation",
            template_level=LiteratureTemplateLevel.EXECUTABLE_TEMPLATE_CANDIDATE.value,
            product_retron={
                "retron_type": "taxane_c13_side_chain",
                "description": "taxane ring alcohol to C13 side-chain ester/carbamate boundary",
                "smarts": "[C;R]-[O]-[C](=O)-[#6,#7]",
            },
            precursor_roles=["taxane_core", "side_chain_fragment"],
            break_bonds=[{"role": "c13_ester_or_carbamate", "atoms": ["taxane_oxygen", "side_chain_carbonyl"]}],
            scope_limits=["does not mark baccatin/10-DAB child route as solved"],
            promotion_status="candidate",
            source_family="taxane",
            condition_source="literature_known",
        ),
        LiteratureTemplateCard(
            template_id="lit_tpl_bufadienolide_c17_pyrone_split_v1",
            evidence_refs=["ev_bufadienolide_c17_pyrone_precedent"],
            reaction_class="C_C_coupling",
            template_level=LiteratureTemplateLevel.EXECUTABLE_TEMPLATE_CANDIDATE.value,
            product_retron={
                "retron_type": "bufadienolide_c17_pyrone",
                "description": "steroid C17 to 2-pyrone C-C boundary",
                "smarts": "[C;R]-[c;R]1[c,o][c][c][c,o]1",
            },
            precursor_roles=["steroid_core", "pyrone_coupling_partner"],
            break_bonds=[{"role": "steroid_c17_pyrone_c_c_bond", "atoms": ["steroid_c17", "pyrone_carbon"]}],
            promotion_status="candidate",
            source_family="bufadienolide_steroid",
            condition_source="literature_analog",
        ),
        LiteratureTemplateCard(
            template_id="lit_tpl_corey_lactone_side_chain_split_v1",
            evidence_refs=["ev_corey_lactone_precedent"],
            reaction_class="corey_lactone_sidechain_installation",
            template_level=LiteratureTemplateLevel.EXECUTABLE_TEMPLATE_CANDIDATE.value,
            product_retron={
                "retron_type": "corey_lactone_side_chain",
                "description": "Corey lactone side-chain installation boundary",
                "smarts": "[C;R]-[C;!R]",
            },
            precursor_roles=["corey_lactone", "side_chain_fragment"],
            break_bonds=[{"role": "corey_lactone_side_chain_bond", "atoms": ["lactone_core", "side_chain_atom"]}],
            promotion_status="candidate",
            source_family="prostaglandin",
            condition_source="literature_known",
        ),
        LiteratureTemplateCard(
            template_id="lit_tpl_artemisinin_peroxide_anchor_v1",
            evidence_refs=["ev_artemisinin_peroxide_precedent"],
            reaction_class="late_stage_peroxide_formation",
            template_level=LiteratureTemplateLevel.ROUTE_ANCHOR_ONLY.value,
            product_retron={
                "retron_type": "artemisinin_peroxide_anchor",
                "description": "peroxide-bearing anchor for recursive child planning only",
            },
            precursor_roles=["dihydroartemisinic_acid_anchor"],
            break_bonds=[],
            scope_limits=["route_anchor_only", "no one-step executable candidate"],
            promotion_status="route_anchor_only",
            source_family="artemisinin",
            condition_source="literature_known",
        ),
    ]


def route_anchor_expansion_tasks_from_templates(
    template_cards: list[LiteratureTemplateCard | dict[str, Any]],
    *,
    parent_route_reference: str,
) -> list[RouteAnchorExpansionTask]:
    tasks: list[RouteAnchorExpansionTask] = []
    for raw in template_cards:
        card = raw if isinstance(raw, LiteratureTemplateCard) else template_card_from_dict(raw)
        retron = str((card.product_retron or {}).get("retron_type") or "")
        reaction_class = card.reaction_class.lower()
        if "taxane" in retron or "taxane" in reaction_class:
            tasks.append(RouteAnchorExpansionTask(
                anchor_smiles="",
                anchor_name="baccatin_or_10_DAB",
                child_target_name="baccatin_or_10_DAB_child",
                source_evidence=list(card.evidence_refs),
                required_closure_type="stock_or_semisynthesis_anchor",
                parent_route_reference=parent_route_reference,
                source_template_id=card.template_id,
            ))
        elif "bufadienolide" in retron or "steroid" in card.source_family:
            tasks.append(RouteAnchorExpansionTask(
                anchor_smiles="",
                anchor_name="androstenedione_like_steroid_core",
                child_target_name="steroid_core_child",
                source_evidence=list(card.evidence_refs),
                required_closure_type="recursive_native_then_literature",
                parent_route_reference=parent_route_reference,
                source_template_id=card.template_id,
            ))
        elif "macrolactone" in retron or "macrolactonization" in reaction_class:
            tasks.append(RouteAnchorExpansionTask(
                anchor_smiles="",
                anchor_name="seco_acid_or_macrolactone_precursor",
                child_target_name="seco_acid_child",
                source_evidence=list(card.evidence_refs),
                required_closure_type="seco_acid_stock_or_route",
                parent_route_reference=parent_route_reference,
                source_template_id=card.template_id,
            ))
        elif "corey" in retron or "prostaglandin" in card.source_family:
            tasks.append(RouteAnchorExpansionTask(
                anchor_smiles="",
                anchor_name="corey_lactone",
                child_target_name="corey_lactone_child",
                source_evidence=list(card.evidence_refs),
                required_closure_type="corey_lactone_route_or_stock",
                parent_route_reference=parent_route_reference,
                source_template_id=card.template_id,
            ))
    return tasks


def route_status_after_anchor_expansion(
    *,
    parent_status: str,
    child_statuses: list[str],
    all_leaf_audit_passed: bool,
) -> str:
    """Conservative route-status promotion for recursive anchor expansion."""
    if all_leaf_audit_passed and parent_status in {"solved", "semisynthesis_closed"} and all(
        status in {"solved", "semisynthesis_closed"} for status in child_statuses
    ):
        return RouteAnchorStatus.SOLVED.value
    if child_statuses and all(status in {"solved", "semisynthesis_closed"} for status in child_statuses):
        return RouteAnchorStatus.SEMISYNTHESIS_CLOSED.value
    if parent_status in {"partial_anchor", "semisynthesis_closed"} or child_statuses:
        return RouteAnchorStatus.PARTIAL_ANCHOR.value
    return RouteAnchorStatus.UNRESOLVED.value


def stitch_parent_child_routes(
    parent_route: dict[str, Any],
    child_routes: list[dict[str, Any]],
    *,
    anchor_tasks: list[RouteAnchorExpansionTask | dict[str, Any]] | None = None,
    all_leaf_audit_passed: bool = False,
) -> dict[str, Any]:
    """Join parent and recursive anchor routes without upgrading status early."""
    parent = dict(parent_route or {})
    children = [dict(route or {}) for route in child_routes or []]
    tasks = [
        task.to_dict() if isinstance(task, RouteAnchorExpansionTask) else dict(task)
        for task in anchor_tasks or []
    ]
    child_statuses = [str(route.get("route_status") or route.get("status") or "unresolved") for route in children]
    stitched_status = route_status_after_anchor_expansion(
        parent_status=str(parent.get("route_status") or parent.get("status") or "unresolved"),
        child_statuses=child_statuses,
        all_leaf_audit_passed=all_leaf_audit_passed,
    )
    parent_steps = [dict(step) for step in parent.get("steps") or []]
    child_steps = [
        {
            "anchor_task_id": (tasks[idx] if idx < len(tasks) else {}).get("source_template_id")
            or (tasks[idx] if idx < len(tasks) else {}).get("anchor_name")
            or f"child_{idx + 1}",
            "route_status": child_statuses[idx] if idx < len(child_statuses) else "unresolved",
            "steps": list(route.get("steps") or []),
            "source_evidence": list((tasks[idx] if idx < len(tasks) else {}).get("source_evidence") or []),
        }
        for idx, route in enumerate(children)
    ]
    unresolved = [
        item
        for item in child_steps
        if item["route_status"] not in {RouteAnchorStatus.SOLVED.value, RouteAnchorStatus.SEMISYNTHESIS_CLOSED.value}
    ]
    return {
        "schema_version": ROUTE_ANCHOR_STITCH_REPORT_SCHEMA,
        "parent_route_reference": parent.get("route_id") or parent.get("id") or "",
        "route_status": stitched_status,
        "all_leaf_audit_passed": bool(all_leaf_audit_passed),
        "parent_steps": parent_steps,
        "child_routes": child_steps,
        "unresolved_anchor_count": len(unresolved),
        "solved_claim_allowed": stitched_status == RouteAnchorStatus.SOLVED.value and bool(all_leaf_audit_passed),
        "status_contract": "Unclosed child anchors keep the stitched route partial/unresolved.",
    }


def template_compliance_gate(card_or_data: LiteratureTemplateCard | dict[str, Any]) -> dict[str, Any]:
    """Conservative safety gate for dangerous/controlled/dual-use templates."""
    card = card_or_data if isinstance(card_or_data, LiteratureTemplateCard) else template_card_from_dict(card_or_data)
    flags = {str(flag).lower() for flag in card.safety_flags or []}
    text = json.dumps(card.to_dict(), sort_keys=True, default=str).lower()
    blocking = sorted(
        flag for flag in flags
        if flag in {"dangerous", "controlled", "dual_use", "explosive", "toxin", "scheduled"}
    )
    keyword_hits = [
        keyword
        for keyword in ("controlled substance", "dual-use", "explosive", "nerve agent", "scheduled")
        if keyword in text
    ]
    reasons = [f"safety_flag:{flag}" for flag in blocking]
    reasons.extend(f"safety_keyword:{keyword}" for keyword in keyword_hits)
    return {
        "schema_version": TEMPLATE_COMPLIANCE_GATE_SCHEMA,
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "template_id": card.template_id,
        "requires_human_compliance_review": bool(reasons),
        "allowed_for_one_step_source": not reasons,
    }


def production_kb_promotion_gate(
    card_or_data: LiteratureTemplateCard | dict[str, Any],
    *,
    replicated_case_count: int = 0,
    negative_controls_passed: bool = False,
    source_evidence_stable: bool = False,
    from_target_run: bool = True,
    validation_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Separate production-KB promotion from target-run proposal use."""
    card = card_or_data if isinstance(card_or_data, LiteratureTemplateCard) else template_card_from_dict(card_or_data)
    reports = [dict(report or {}) for report in validation_reports or []]
    reasons: list[str] = []
    if replicated_case_count < 2:
        reasons.append("insufficient_multi_case_replication")
    if not negative_controls_passed:
        reasons.append("negative_controls_not_passed")
    if not source_evidence_stable:
        reasons.append("source_evidence_not_stable")
    if from_target_run:
        reasons.append("target_run_direct_kb_write_forbidden")
    if not reports or not all(report.get("allowed_for_one_step_source") for report in reports):
        reasons.append("missing_or_failed_validation_reports")
    compliance = template_compliance_gate(card)
    if not compliance["accepted"]:
        reasons.append("compliance_gate_failed")
    return {
        "schema_version": TEMPLATE_KB_PROMOTION_GATE_SCHEMA,
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "template_id": card.template_id,
        "allowed_for_production_kb_write": not reasons,
        "proposal_use_only": bool(reasons),
        "no_target_run_direct_write": True,
    }


def _result_dict(native_result: Any | None) -> dict[str, Any]:
    if native_result is None:
        return {}
    if isinstance(native_result, dict):
        return dict(native_result)
    to_dict = getattr(native_result, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:
            pass
    return {
        "route_count": int(getattr(native_result, "route_count", 0) or 0),
        "solved": bool(getattr(native_result, "solved", False)),
        "routes": [route.to_dict() for route in getattr(native_result, "routes", []) or [] if hasattr(route, "to_dict")],
        "failures": [
            failure.to_dict() for failure in getattr(native_result, "failures", []) or [] if hasattr(failure, "to_dict")
        ],
    }


def _audit_dict(route_audit: Any | None) -> dict[str, Any]:
    if route_audit is None:
        return {}
    if isinstance(route_audit, dict):
        return dict(route_audit)
    to_dict = getattr(route_audit, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:
            return {}
    return {}


def _all_route_leaves_stock_closed(result: dict[str, Any]) -> bool:
    routes = result.get("routes") or []
    if not routes:
        return False
    for route in routes:
        if not isinstance(route, dict):
            continue
        route_stock = route.get("stock_status") or {}
        if route_stock and any(value is False or value is None for value in route_stock.values()):
            return False
        for step in route.get("steps") or []:
            stock = (step or {}).get("stock_status") or {}
            if stock and any(value is False or value is None for value in stock.values()):
                return False
    return True


def _has_fake_closure_reason(audit: dict[str, Any]) -> bool:
    text = json.dumps(audit, sort_keys=True, default=str).lower()
    return any(token in text for token in ("fake_closure", "advanced_same_scaffold", "no_complexity_drop"))


def _advanced_frontier_detected(frontier_report: dict[str, Any], audit: dict[str, Any]) -> bool:
    if bool(frontier_report.get("advanced_frontier_found")):
        return True
    reasons = {str(item) for item in frontier_report.get("reasons") or []}
    if reasons.intersection({"advanced_frontier", "advanced_same_scaffold", "unresolved_core"}):
        return True
    text = json.dumps(audit.get("top_route_summary") or {}, sort_keys=True, default=str).lower()
    return "advanced_frontier" in text or "unresolved_core" in text


def _template_is_route_anchor_only(card: LiteratureTemplateCard) -> bool:
    return card.template_level == LiteratureTemplateLevel.ROUTE_ANCHOR_ONLY.value or "route_anchor_only" in {
        item.lower() for item in card.scope_limits
    }


def _template_is_advisory_only(card: LiteratureTemplateCard) -> bool:
    return card.template_level == LiteratureTemplateLevel.ADVISORY_STRATEGY.value


def _valid_smiles(smiles: str) -> bool:
    if not smiles:
        return False
    return Chem.MolFromSmiles(str(smiles)) is not None


def _dedupe_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out

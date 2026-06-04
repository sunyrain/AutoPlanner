"""Strategic-disconnection cards mined from validated evidence cards."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from cascade_planner.agent.evidence_cards import EvidenceCard, validate_evidence_card


STRATEGIC_DISCONNECTION_CARD_SCHEMA = "strategic_disconnection_card.v1"

ALLOWED_TARGET_RELATIONS = {
    "exact_target_or_intermediate",
    "family_precedent",
    "reaction_precedent",
    "analogy_only",
}
ALLOWED_ROUTE_CLAIMS = {
    "total_synthesis",
    "semisynthesis",
    "biosynthesis-inspired",
    "failed",
    "route_policy",
    "unknown",
}
ALLOWED_DISCONNECTION_TYPES = {
    "fragment_coupling",
    "macrocyclization",
    "glycosylation",
    "semisynthesis_anchor",
    "late_stage_oxidation",
    "side_chain_installation",
    "negative_guidance",
    "route_policy",
    "unknown",
}


@dataclass
class StrategicDisconnectionCard:
    card_id: str
    case_id: str
    target_smiles: str
    frontier_smiles: str
    target_relation: str
    route_claim: str
    disconnection_type: str
    product_side_pattern: str = ""
    precursor_side_pattern: str = ""
    strategic_subgoal: str = ""
    anchor_candidate: str = ""
    forbidden_fake_terminal_implication: str = ""
    confidence: str = "medium"
    usable_for_search: bool = False
    evidence_refs: list[str] = field(default_factory=list)
    source_record_refs: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "draft"
    schema_version: str = STRATEGIC_DISCONNECTION_CARD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mine_strategic_disconnection_cards(
    *,
    case_id: str,
    target_smiles: str,
    frontier_smiles: str,
    evidence_cards: list[EvidenceCard],
) -> list[StrategicDisconnectionCard]:
    cards: list[StrategicDisconnectionCard] = []
    for evidence in evidence_cards:
        if evidence.validation_status == "draft_only":
            continue
        if evidence.route_role not in {"strategic_disconnection", "route_anchor", "negative_guidance"}:
            continue
        card = _card_from_evidence(
            case_id=case_id,
            target_smiles=target_smiles,
            frontier_smiles=frontier_smiles,
            evidence=evidence,
        )
        validation = validate_strategic_disconnection_card(
            card,
            validated_evidence_refs={evidence.evidence_id} if validate_evidence_card(evidence)["accepted"] else set(),
        )
        card.validation_status = validation["validation_status"]
        cards.append(card)
    return cards


def validate_strategic_disconnection_card(
    card_or_data: StrategicDisconnectionCard | dict[str, Any],
    *,
    validated_evidence_refs: set[str] | None = None,
) -> dict[str, Any]:
    card = card_or_data if isinstance(card_or_data, StrategicDisconnectionCard) else strategic_disconnection_card_from_dict(card_or_data)
    validated = set(validated_evidence_refs or [])
    reasons: list[str] = []
    if card.schema_version != STRATEGIC_DISCONNECTION_CARD_SCHEMA:
        reasons.append("invalid_strategic_disconnection_card_schema")
    if not card.card_id:
        reasons.append("missing_card_id")
    if not card.case_id:
        reasons.append("missing_case_id")
    if card.target_relation not in ALLOWED_TARGET_RELATIONS:
        reasons.append("invalid_target_relation")
    if card.route_claim not in ALLOWED_ROUTE_CLAIMS:
        reasons.append("invalid_route_claim")
    if card.disconnection_type not in ALLOWED_DISCONNECTION_TYPES:
        reasons.append("invalid_disconnection_type")
    if not card.evidence_refs:
        reasons.append("missing_evidence_refs")
    elif validated and any(ref not in validated for ref in card.evidence_refs):
        reasons.append("unvalidated_evidence_refs")
    if card.route_claim == "failed" and card.usable_for_search:
        reasons.append("failed_route_must_be_negative_guidance")
    if "isolation" in card.route_claim.lower() and card.usable_for_search:
        reasons.append("isolation_claim_not_synthesis_disconnection")
    if card.target_relation == "analogy_only" and card.usable_for_search:
        reasons.append("analogy_only_not_search_ready")
    if card.usable_for_search and not (card.strategic_subgoal or card.anchor_candidate):
        reasons.append("search_ready_card_needs_subgoal_or_anchor")
    if card.usable_for_search and not card.forbidden_fake_terminal_implication:
        reasons.append("missing_fake_terminal_guard")
    return {
        "schema_version": "strategic_disconnection_card_validation.v1",
        "accepted": not reasons,
        "validation_status": "validated" if not reasons else "rejected",
        "reasons": sorted(set(reasons)),
        "card_id": card.card_id,
        "usable_for_search": card.usable_for_search and not reasons,
    }


def strategic_disconnection_card_from_dict(data: dict[str, Any]) -> StrategicDisconnectionCard:
    return StrategicDisconnectionCard(
        card_id=str(data.get("card_id") or ""),
        case_id=str(data.get("case_id") or ""),
        target_smiles=str(data.get("target_smiles") or ""),
        frontier_smiles=str(data.get("frontier_smiles") or ""),
        target_relation=str(data.get("target_relation") or "analogy_only"),
        route_claim=str(data.get("route_claim") or "unknown"),
        disconnection_type=str(data.get("disconnection_type") or "unknown"),
        product_side_pattern=str(data.get("product_side_pattern") or ""),
        precursor_side_pattern=str(data.get("precursor_side_pattern") or ""),
        strategic_subgoal=str(data.get("strategic_subgoal") or ""),
        anchor_candidate=str(data.get("anchor_candidate") or ""),
        forbidden_fake_terminal_implication=str(data.get("forbidden_fake_terminal_implication") or ""),
        confidence=str(data.get("confidence") or "medium"),
        usable_for_search=bool(data.get("usable_for_search")),
        evidence_refs=[str(ref) for ref in data.get("evidence_refs") or []],
        source_record_refs=[str(ref) for ref in data.get("source_record_refs") or []],
        limitations=[str(item) for item in data.get("limitations") or []],
        source_metadata=dict(data.get("source_metadata") or {}),
        validation_status=str(data.get("validation_status") or "draft"),
        schema_version=str(data.get("schema_version") or STRATEGIC_DISCONNECTION_CARD_SCHEMA),
    )


def write_strategic_disconnection_cards_jsonl(
    cards: Iterable[StrategicDisconnectionCard],
    path: str | Path,
) -> None:
    rows = [json.dumps(card.to_dict(), ensure_ascii=False, sort_keys=True) for card in cards]
    Path(path).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def load_strategic_disconnection_cards_jsonl(path: str | Path) -> list[StrategicDisconnectionCard]:
    cards: list[StrategicDisconnectionCard] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cards.append(strategic_disconnection_card_from_dict(json.loads(line)))
    return cards


def _card_from_evidence(
    *,
    case_id: str,
    target_smiles: str,
    frontier_smiles: str,
    evidence: EvidenceCard,
) -> StrategicDisconnectionCard:
    record = (evidence.source_metadata or {}).get("record") or {}
    move = record.get("retrosynthetic_move") or {}
    break_bonds = [str(item) for item in move.get("break_bonds") or []]
    precursor_roles = [str(item) for item in move.get("suggested_precursor_roles") or []]
    route_claim = _route_claim(evidence, record)
    disconnection_type = _disconnection_type(evidence, record)
    usable = (
        evidence.target_relation != "analogy_only"
        and evidence.route_role in {"strategic_disconnection", "route_anchor"}
        and route_claim != "failed"
        and bool(break_bonds or evidence.route_role == "route_anchor")
    )
    anchor_candidate = ""
    if evidence.route_role == "route_anchor":
        anchor_candidate = str(record.get("name") or record.get("anchor_id") or evidence.route_role_detail or "")
    elif precursor_roles:
        anchor_candidate = "; ".join(role for role in precursor_roles if "anchor" in role.lower() or "core" in role.lower())
    product_pattern = "; ".join(break_bonds) or str((record.get("applicability") or {}).get("target_features") or "")
    precursor_pattern = "; ".join(precursor_roles)
    subgoal = str(move.get("planner_hint") or record.get("strategic_principle") or evidence.route_role_detail or "")
    return StrategicDisconnectionCard(
        card_id=f"sd_{evidence.source_record_id or evidence.evidence_id}",
        case_id=case_id or evidence.case_id,
        target_smiles=target_smiles,
        frontier_smiles=frontier_smiles,
        target_relation=evidence.target_relation,
        route_claim=route_claim,
        disconnection_type=disconnection_type,
        product_side_pattern=product_pattern,
        precursor_side_pattern=precursor_pattern,
        strategic_subgoal=subgoal,
        anchor_candidate=anchor_candidate,
        forbidden_fake_terminal_implication=_fake_terminal_guard(evidence, record),
        confidence=evidence.confidence,
        usable_for_search=usable,
        evidence_refs=[evidence.evidence_id],
        source_record_refs=[evidence.source_record_id] if evidence.source_record_id else [],
        limitations=list(evidence.limitations or []),
        source_metadata={"evidence_source_metadata": evidence.source_metadata, "record": record},
    )


def _route_claim(evidence: EvidenceCard, record: dict[str, Any]) -> str:
    text = json.dumps(record, ensure_ascii=False).lower()
    if evidence.route_role == "negative_guidance" or "failed" in text:
        return "failed"
    if "semisynthesis" in text or evidence.route_role == "route_anchor":
        return "semisynthesis"
    if "biosynthetic" in text or "biomimetic" in text:
        return "biosynthesis-inspired"
    if "total synthesis" in text:
        return "total_synthesis"
    if "policy" in str(record.get("use_policy") or "").lower():
        return "route_policy"
    return "unknown"


def _disconnection_type(evidence: EvidenceCard, record: dict[str, Any]) -> str:
    text = json.dumps(record, ensure_ascii=False).lower()
    family = str(evidence.family_id or "").lower()
    if evidence.route_role == "negative_guidance":
        return "negative_guidance"
    if evidence.route_role == "route_anchor" or "anchor" in text:
        return "semisynthesis_anchor"
    if "macrolactonization" in text or "macrocyclization" in text:
        return "macrocyclization"
    if "glycos" in text:
        return "glycosylation"
    if "oxidation" in text or "oxygenation" in text or "peroxide" in text:
        return "late_stage_oxidation"
    if "side chain" in text or "side-chain" in text or "statin" in family or "taxane" in family:
        return "side_chain_installation"
    if "coupling" in text or (record.get("retrosynthetic_move") or {}).get("break_bonds"):
        return "fragment_coupling"
    if "policy" in text:
        return "route_policy"
    return "unknown"


def _fake_terminal_guard(evidence: EvidenceCard, record: dict[str, Any]) -> str:
    policy = record.get("use_policy") or {}
    hard_reject = policy.get("hard_reject_counterexamples") or []
    if hard_reject:
        return "; ".join(str(item) for item in hard_reject)
    text = json.dumps(record, ensure_ascii=False).lower()
    if "not stock" in text or "do not mark" in text or "product-like" in text:
        return "do not treat product-like advanced analogues as solved stock"
    if evidence.route_role == "route_anchor":
        return "route anchor is not ordinary stock closure"
    return "same-scaffold/product-like terminal does not prove solved route"

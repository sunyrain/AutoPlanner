"""Evidence-card schema for SMILES-first literature-assisted planning."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


EVIDENCE_SCHEMA_VERSION = "evidence_card.v1"
ALLOWED_TARGET_RELATIONS = {
    "exact_target_or_intermediate",
    "family_precedent",
    "reaction_precedent",
    "analogy_only",
}
ALLOWED_ROUTE_ROLES = {
    "scaffold_family",
    "strategic_disconnection",
    "route_anchor",
    "condition_hint",
    "negative_guidance",
    "unknown",
}
ALLOWED_CONFIDENCE = {"low", "medium", "medium_high", "high"}


@dataclass
class EvidenceCard:
    evidence_id: str
    case_id: str
    source_type: str
    source_title: str
    target_relation: str
    claim_type: str
    route_role: str
    confidence: str = "medium"
    url: str = ""
    doi: str = ""
    local_ref: str = ""
    source_record_id: str = ""
    family_id: str = ""
    route_role_detail: str = ""
    limitations: list[str] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    validation_status: str = "draft"

    def normalize(self) -> "EvidenceCard":
        if self.target_relation not in ALLOWED_TARGET_RELATIONS:
            self.limitations.append(f"unsupported_target_relation:{self.target_relation}")
            self.target_relation = "analogy_only"
        if self.route_role not in ALLOWED_ROUTE_ROLES:
            self.limitations.append(f"unsupported_route_role:{self.route_role}")
            self.route_role = "unknown"
        if self.confidence not in ALLOWED_CONFIDENCE:
            self.confidence = "medium"
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalize())


def evidence_from_dict(data: dict[str, Any]) -> EvidenceCard:
    allowed = set(EvidenceCard.__dataclass_fields__)
    kwargs = {key: value for key, value in data.items() if key in allowed}
    return EvidenceCard(**kwargs).normalize()


def validate_evidence_card(card_or_data: EvidenceCard | dict[str, Any]) -> dict[str, Any]:
    card = card_or_data if isinstance(card_or_data, EvidenceCard) else evidence_from_dict(card_or_data)
    reasons: list[str] = []
    if not card.evidence_id:
        reasons.append("missing_evidence_id")
    if not card.case_id:
        reasons.append("missing_case_id")
    if not card.source_title:
        reasons.append("missing_source_title")
    if card.target_relation not in ALLOWED_TARGET_RELATIONS:
        reasons.append("invalid_target_relation")
    if card.route_role not in ALLOWED_ROUTE_ROLES:
        reasons.append("invalid_route_role")
    if not (card.url or card.doi or card.local_ref):
        reasons.append("untraceable_source")
    if card.target_relation == "analogy_only" and card.route_role == "strategic_disconnection":
        reasons.append("analogy_only_disconnection_not_search_ready")
    accepted = not reasons
    return {
        "accepted": accepted,
        "validation_status": "validated" if accepted else "draft_only",
        "reasons": reasons,
        "evidence_id": card.evidence_id,
    }


def load_evidence_jsonl(path: str | Path) -> list[EvidenceCard]:
    cards: list[EvidenceCard] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        card = evidence_from_dict(data)
        if not card.evidence_id:
            card.evidence_id = f"{Path(path).stem}:{line_no}"
        cards.append(card)
    return cards


def write_evidence_jsonl(cards: Iterable[EvidenceCard], path: str | Path) -> None:
    rows = []
    for card in cards:
        validation = validate_evidence_card(card)
        card.validation_status = validation["validation_status"]
        rows.append(json.dumps(card.to_dict(), ensure_ascii=False, sort_keys=True))
    Path(path).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def validation_summary(cards: Iterable[EvidenceCard]) -> dict[str, Any]:
    results = [validate_evidence_card(card) for card in cards]
    return {
        "total": len(results),
        "accepted": sum(1 for item in results if item["accepted"]),
        "rejected": sum(1 for item in results if not item["accepted"]),
        "results": results,
    }

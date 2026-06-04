"""Local literature-retrieval adapter for SMILES-first P0 planning.

This module consumes curated strategic-disconnection records and optional
manual/Codex evidence JSONL.  It deliberately emits evidence cards rather than
raw reactions so later stages can validate and downgrade weak evidence.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cascade_planner.agent.evidence_cards import EvidenceCard, load_evidence_jsonl
from cascade_planner.agent.literature_templates import audit_native_run_for_literature
from cascade_planner.agent.target_profile import TargetProfile


LITERATURE_TASK_SCHEMA = "literature_search_task.v1"
DEFAULT_DB_GLOB = "data/strategic_disconnections/strategic_disconnections*.json"


@dataclass
class LiteratureSearchTask:
    case_id: str
    target_profile: dict[str, Any]
    frontier_smiles: str
    family_hints: list[str] = field(default_factory=list)
    query_budget: int = 12
    allowed_source_types: list[str] = field(default_factory=lambda: ["literature", "local_curated"])
    required_output_schema: str = "evidence_cards.jsonl"
    trigger_reasons: list[str] = field(default_factory=list)
    schema_version: str = LITERATURE_TASK_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_literature_task(
    profile: TargetProfile,
    frontier_smiles: str,
    *,
    query_budget: int = 12,
    trigger_reasons: list[str] | None = None,
) -> LiteratureSearchTask:
    return LiteratureSearchTask(
        case_id=profile.case_id,
        target_profile=profile.to_dict(),
        frontier_smiles=frontier_smiles,
        family_hints=list(profile.family_hints),
        query_budget=int(query_budget),
        trigger_reasons=[str(reason) for reason in trigger_reasons or []],
    )


def build_triggered_literature_task(
    profile: TargetProfile,
    frontier_smiles: str,
    *,
    native_result: dict[str, Any] | None = None,
    route_audit: dict[str, Any] | None = None,
    frontier_report: dict[str, Any] | None = None,
    user_requested: bool = False,
    query_budget: int = 12,
) -> tuple[LiteratureSearchTask | None, dict[str, Any]]:
    """Build a literature task only when native/audit evidence justifies it."""
    trigger_report = audit_native_run_for_literature(
        native_result or {},
        route_audit=route_audit or {},
        frontier_report=frontier_report or {},
        user_requested=user_requested,
    )
    if not trigger_report["should_trigger"]:
        return None, trigger_report
    return (
        build_literature_task(
            profile,
            frontier_smiles,
            query_budget=query_budget,
            trigger_reasons=list(trigger_report.get("trigger_reasons") or []),
        ),
        trigger_report,
    )


def retrieve_literature_evidence(
    task: LiteratureSearchTask,
    *,
    manual_evidence_jsonl: str | Path | None = None,
    db_paths: list[str | Path] | None = None,
) -> tuple[list[EvidenceCard], dict[str, Any]]:
    cards: list[EvidenceCard] = []
    search_rows: list[dict[str, Any]] = []

    if manual_evidence_jsonl:
        manual_cards = load_evidence_jsonl(manual_evidence_jsonl)
        cards.extend(manual_cards)
        search_rows.append({
            "query": str(manual_evidence_jsonl),
            "source": "manual_evidence_jsonl",
            "hits": len(manual_cards),
        })

    db = load_strategic_disconnection_db(db_paths)
    query_terms = _query_terms(task)
    matches = query_strategic_records(db, query_terms=query_terms, family_hints=task.family_hints)
    search_rows.append({
        "query": " OR ".join(query_terms[:8]),
        "source": "local_strategic_disconnections",
        "hits": len(matches["disconnections"]) + len(matches["anchors"]),
        "matched_families": [item.get("family_id") for item in matches["families"][:8]],
    })

    cards.extend(_cards_from_records(task.case_id, matches, limit=task.query_budget))
    report = {
        "schema_version": "literature_search_report.v1",
        "case_id": task.case_id,
        "task": task.to_dict(),
        "searches": search_rows,
        "hit_count": len(cards),
        "evidence_levels": _evidence_level_counts(cards),
        "unresolved_literature_gap": len(cards) == 0,
        "limitations": _limitations(cards),
    }
    return cards, report


def load_strategic_disconnection_db(paths: list[str | Path] | None = None) -> dict[str, Any]:
    db_paths = [Path(path) for path in paths] if paths else sorted(Path().glob(DEFAULT_DB_GLOB))
    merged: dict[str, Any] = {
        "schema_version": "strategic_disconnections.merged",
        "sources": [str(path) for path in db_paths],
        "families": [],
        "anchors": [],
        "disconnections": [],
    }
    seen: dict[str, set[str]] = {"families": set(), "anchors": set(), "disconnections": set()}
    keys = {"families": "family_id", "anchors": "anchor_id", "disconnections": "id"}
    for path in db_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for section, key in keys.items():
            for item in data.get(section, []) or []:
                item_id = str(item.get(key) or "")
                if item_id and item_id in seen[section]:
                    continue
                if item_id:
                    seen[section].add(item_id)
                merged[section].append(item)
    return merged


def query_strategic_records(
    db: dict[str, Any],
    *,
    query_terms: list[str],
    family_hints: list[str],
) -> dict[str, Any]:
    terms = [term.lower() for term in query_terms if term]
    hinted_families = _hinted_family_ids(db, family_hints, terms)
    family_score = {
        str(item.get("family_id") or ""): _score_record(item, terms, family_hints)
        for item in db.get("families", [])
    }
    families = _sort_records([
        item for item in db.get("families", [])
        if item.get("family_id") in hinted_families or _record_matches(item, terms)
    ], terms, family_hints, family_score)
    family_ids = {item.get("family_id") for item in families}
    anchors = _sort_records([
        item for item in db.get("anchors", [])
        if item.get("family_id") in family_ids or _record_matches(item, terms)
    ], terms, family_hints, family_score)
    disconnections = _sort_records([
        item for item in db.get("disconnections", [])
        if item.get("family_id") in family_ids or _record_matches(item, terms)
    ], terms, family_hints, family_score)
    return {
        "families": families,
        "anchors": anchors,
        "disconnections": disconnections,
    }


def render_literature_report(report: dict[str, Any], cards: list[EvidenceCard]) -> str:
    lines = [
        "# Literature Search Report",
        "",
        f"- Case: `{report.get('case_id')}`",
        f"- Hit count: `{report.get('hit_count')}`",
        f"- Unresolved gap: `{bool(report.get('unresolved_literature_gap'))}`",
        f"- Evidence levels: `{json.dumps(report.get('evidence_levels') or {}, sort_keys=True)}`",
        "",
        "## Searches",
    ]
    for item in report.get("searches") or []:
        lines.append(f"- `{item.get('source')}` query `{item.get('query')}` -> {item.get('hits')} hits")
    lines.extend(["", "## Evidence Cards"])
    for card in cards:
        lines.append(
            f"- `{card.evidence_id}` {card.target_relation} / {card.route_role}: "
            f"{card.source_title} ({card.url or card.doi or card.local_ref})"
        )
        if card.limitations:
            lines.append(f"  - Limitations: {', '.join(card.limitations[:4])}")
    if report.get("limitations"):
        lines.extend(["", "## Limitations"])
        for limitation in report["limitations"]:
            lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def _cards_from_records(case_id: str, matches: dict[str, Any], *, limit: int) -> list[EvidenceCard]:
    cards: list[EvidenceCard] = []
    family_names = {item.get("family_id"): item.get("name") for item in matches.get("families", [])}
    for item in matches.get("disconnections", [])[: max(0, limit)]:
        evidence = item.get("evidence") or []
        source = evidence[0] if evidence else {}
        relation = _target_relation_for_record(item)
        cards.append(EvidenceCard(
            evidence_id=f"ev_{item.get('id')}",
            case_id=case_id,
            source_type=str(source.get("type") or "local_curated"),
            source_title=str(source.get("citation") or item.get("name") or item.get("id")),
            url=str(source.get("url") or ""),
            doi=str(source.get("doi") or ""),
            local_ref=str(source.get("local_ref") or f"data/strategic_disconnections:{item.get('id')}"),
            source_record_id=str(item.get("id") or ""),
            family_id=str(item.get("family_id") or ""),
            target_relation=relation,
            claim_type="strategic_disconnection",
            route_role="strategic_disconnection",
            confidence=str(item.get("confidence") or "medium"),
            route_role_detail=str((item.get("retrosynthetic_move") or {}).get("planner_hint") or ""),
            limitations=list(item.get("risks") or []),
            source_metadata={
                "record_type": "disconnection",
                "record": item,
                "family_name": family_names.get(item.get("family_id"), ""),
            },
        ))
    remaining = max(0, limit - len(cards))
    for item in matches.get("anchors", [])[:remaining]:
        evidence = item.get("evidence") or []
        source = evidence[0] if evidence else {}
        cards.append(EvidenceCard(
            evidence_id=f"ev_{item.get('anchor_id')}",
            case_id=case_id,
            source_type=str(source.get("type") or "local_curated"),
            source_title=str(source.get("citation") or item.get("name") or item.get("anchor_id")),
            url=str(source.get("url") or ""),
            doi=str(source.get("doi") or ""),
            local_ref=str(source.get("local_ref") or f"data/strategic_disconnections:{item.get('anchor_id')}"),
            source_record_id=str(item.get("anchor_id") or ""),
            family_id=str(item.get("family_id") or ""),
            target_relation="family_precedent",
            claim_type="route_anchor",
            route_role="route_anchor",
            confidence="medium_high",
            route_role_detail=str(item.get("role") or ""),
            limitations=[str(item.get("acceptance_policy") or "anchor_not_stock")],
            source_metadata={"record_type": "anchor", "record": item},
        ))
    return cards


def _query_terms(task: LiteratureSearchTask) -> list[str]:
    profile = task.target_profile or {}
    terms = list(task.family_hints)
    terms.extend(str(x) for x in profile.get("family_hints") or [])
    terms.extend(str(x) for x in [profile.get("target_name"), profile.get("formula")] if x)
    if task.frontier_smiles:
        terms.append(task.frontier_smiles)
    return [term for term in terms if term]


def _hinted_family_ids(db: dict[str, Any], hints: list[str], terms: list[str]) -> set[str]:
    text_terms = " ".join([*hints, *terms]).lower().replace("-", "_")
    family_ids: set[str] = set()
    for family in db.get("families", []):
        fid = str(family.get("family_id") or "")
        keywords = [fid, str(family.get("name") or "")]
        keywords.extend(str(x) for x in family.get("pattern_keywords") or [])
        if any(_token_match(text_terms, keyword) for keyword in keywords):
            family_ids.add(fid)
    return family_ids


def _record_matches(item: Any, terms: list[str]) -> bool:
    if not terms:
        return False
    text = json.dumps(item, ensure_ascii=False).lower()
    return any(term and term in text for term in terms)


def _sort_records(
    records: list[dict[str, Any]],
    terms: list[str],
    family_hints: list[str],
    family_score: dict[str, int],
) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            family_score.get(str(item.get("family_id") or ""), 0),
            _score_record(item, terms, family_hints),
            str(item.get("family_id") or ""),
            str(item.get("id") or item.get("anchor_id") or ""),
        ),
        reverse=True,
    )


def _score_record(item: dict[str, Any], terms: list[str], family_hints: list[str]) -> int:
    text = json.dumps(item, ensure_ascii=False).lower().replace("-", "_")
    fid = str(item.get("family_id") or "").lower().replace("-", "_")
    hints = [str(hint).lower().replace("-", "_") for hint in family_hints if str(hint).strip()]
    score = 0
    for hint in hints:
        if not hint:
            continue
        if hint == fid:
            score += 200
        if hint in fid:
            score += 80
        if hint in text:
            score += 45 if len(hint) >= 8 else 18
        for part in hint.split("_"):
            if len(part) >= 8 and part in fid:
                score += 35
            elif len(part) >= 8 and part in text:
                score += 12
    for term in terms:
        term = str(term).lower().replace("-", "_")
        if not term:
            continue
        if term == fid:
            score += 120
        elif len(term) >= 8 and term in fid:
            score += 45
        elif len(term) >= 8 and term in text:
            score += 10
    # Specific strategic words should outrank broad policy records.
    for keyword in ("bufadienolide", "macrolactonization", "glycosylation", "pictet", "pyrone"):
        if keyword in hints and keyword in text:
            score += 90
    return score


def _token_match(haystack: str, keyword: str) -> bool:
    key = str(keyword or "").lower().replace("-", "_")
    if not key:
        return False
    if key in haystack:
        return True
    return any(part and part in haystack for part in key.split("_") if len(part) >= 5)


def _target_relation_for_record(item: dict[str, Any]) -> str:
    confidence = str(item.get("confidence") or "").lower()
    if confidence == "high":
        return "family_precedent"
    if confidence in {"medium_high", "medium"}:
        return "reaction_precedent"
    return "analogy_only"


def _evidence_level_counts(cards: list[EvidenceCard]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for card in cards:
        counts[card.target_relation] = counts.get(card.target_relation, 0) + 1
    return counts


def _limitations(cards: list[EvidenceCard]) -> list[str]:
    if not cards:
        return ["unresolved_literature_gap"]
    values = []
    if all(card.target_relation == "analogy_only" for card in cards):
        values.append("only_analogy_evidence")
    if not any(card.route_role == "strategic_disconnection" for card in cards):
        values.append("no_strategic_disconnection_evidence")
    return values

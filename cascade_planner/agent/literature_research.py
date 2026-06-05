"""Local literature-retrieval adapter for SMILES-first P0 planning.

This module consumes curated strategic-disconnection records and optional
manual/Codex evidence JSONL.  It deliberately emits evidence cards rather than
raw reactions so later stages can validate and downgrade weak evidence.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from cascade_planner.agent.evidence_cards import EvidenceCard, load_evidence_jsonl
from cascade_planner.agent.literature_templates import audit_native_run_for_literature
from cascade_planner.agent.target_profile import TargetProfile


LITERATURE_TASK_SCHEMA = "literature_search_task.v1"
DEFAULT_DB_GLOB = "data/strategic_disconnections/strategic_disconnections*.json"
PUBMED_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
STATIN_DRUG_NAMES = {
    "atorvastatin",
    "cerivastatin",
    "fluvastatin",
    "lovastatin",
    "mevastatin",
    "pitavastatin",
    "pravastatin",
    "rosuvastatin",
    "simvastatin",
    "compactin",
}
PUBMED_ABSTRACT_ROUTE_SIGNAL_TERMS = [
    "synthesis",
    "synthetic",
    "semisynthesis",
    "semi-synthesis",
    "fermentation",
    "biotransformation",
    "intermediate",
    "intermediates",
    "process",
    "process chemistry",
    "route",
    "preparation",
    "lactone",
    "salt",
    "impurity",
    "crystallization",
    "resolution",
    "hydrolysis",
    "esterification",
    "side chain",
    "side-chain",
    "deprotection",
    "scale-up",
]


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


def retrieve_pubmed_evidence(
    task: LiteratureSearchTask,
    *,
    retmax: int | None = None,
    timeout_s: float = 10.0,
) -> tuple[list[EvidenceCard], dict[str, Any]]:
    """Retrieve lightweight PubMed evidence cards through NCBI E-utilities.

    PubMed summaries are evidence leads, not parsed reaction records.  The
    generated cards are therefore conservative and must pass the ordinary
    EvidenceCard validation before they can support route planning.
    """
    limit = max(1, min(int(retmax or task.query_budget or 5), 20))
    query = _pubmed_query(task)
    search_rows: list[dict[str, Any]] = []
    cards: list[EvidenceCard] = []
    try:
        esearch_url = _pubmed_url(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": str(limit),
                "sort": "relevance",
                "tool": "AutoPlanner",
            },
        )
        search_payload = _fetch_pubmed_json(esearch_url, timeout_s=timeout_s)
        id_list = [
            str(uid)
            for uid in (search_payload.get("esearchresult") or {}).get("idlist") or []
            if str(uid).strip()
        ][:limit]
        search_rows.append({
            "query": query,
            "source": "pubmed_esearch",
            "hits": len(id_list),
            "retrieved_ids": id_list,
        })
        if id_list:
            esummary_url = _pubmed_url(
                "esummary.fcgi",
                {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "json",
                    "tool": "AutoPlanner",
                },
            )
            summary_payload = _fetch_pubmed_json(esummary_url, timeout_s=timeout_s)
            cards = _pubmed_cards_from_summary(task, summary_payload, id_list, query=query)
            search_rows.append({
                "query": ",".join(id_list),
                "source": "pubmed_esummary",
                "hits": len(cards),
            })
    except Exception as exc:  # pragma: no cover - covered by runtime reports.
        search_rows.append({
            "query": query,
            "source": "pubmed_eutils",
            "hits": 0,
            "error": type(exc).__name__,
        })
    report = {
        "schema_version": "literature_search_report.v1",
        "case_id": task.case_id,
        "task": task.to_dict(),
        "backend": "pubmed",
        "backend_requested": "pubmed",
        "backend_resolved": "pubmed",
        "searches": search_rows,
        "hit_count": len(cards),
        "evidence_levels": _evidence_level_counts(cards),
        "unresolved_literature_gap": len(cards) == 0,
        "limitations": [] if cards else ["unresolved_literature_gap"],
    }
    return cards, report


def retrieve_pubmed_query_evidence(
    *,
    case_id: str,
    query: str,
    family_hints: list[str] | None = None,
    query_variants: list[str] | None = None,
    retmax: int = 3,
    timeout_s: float = 10.0,
    include_abstract_signals: bool = False,
) -> tuple[list[EvidenceCard], dict[str, Any]]:
    """Retrieve PubMed evidence leads for one explicit follow-up query."""
    task = LiteratureSearchTask(
        case_id=case_id,
        target_profile={"target_name": case_id, "family_hints": list(family_hints or [])},
        frontier_smiles="",
        family_hints=list(family_hints or []),
        query_budget=int(retmax),
    )
    limit = max(1, min(int(retmax or 3), 20))
    search_rows: list[dict[str, Any]] = []
    cards: list[EvidenceCard] = []
    attempted_queries = _dedupe_queries([str(query), *[str(item) for item in query_variants or []]])
    resolved_query = ""
    abstract_signal_audit: list[dict[str, Any]] = []
    abstract_signal_status = "not_requested"
    try:
        for attempt_index, attempt_query in enumerate(attempted_queries, start=1):
            esearch_url = _pubmed_url(
                "esearch.fcgi",
                {
                    "db": "pubmed",
                    "term": attempt_query,
                    "retmode": "json",
                    "retmax": str(limit),
                    "sort": "relevance",
                    "tool": "AutoPlanner",
                },
            )
            search_payload = _fetch_pubmed_json(esearch_url, timeout_s=timeout_s)
            id_list = [
                str(uid)
                for uid in (search_payload.get("esearchresult") or {}).get("idlist") or []
                if str(uid).strip()
            ][:limit]
            search_rows.append({
                "attempt": attempt_index,
                "query": attempt_query,
                "source": "pubmed_followup_esearch",
                "hits": len(id_list),
                "retrieved_ids": id_list,
                "query_strategy": "primary" if attempt_index == 1 else "fallback",
            })
            if not id_list:
                continue
            esummary_url = _pubmed_url(
                "esummary.fcgi",
                {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "json",
                    "tool": "AutoPlanner",
                },
            )
            summary_payload = _fetch_pubmed_json(esummary_url, timeout_s=timeout_s)
            cards = _pubmed_cards_from_summary(task, summary_payload, id_list, query=attempt_query)
            if include_abstract_signals:
                try:
                    abstract_signal_audit = _pubmed_abstract_signal_audit(id_list, timeout_s=timeout_s)
                    search_rows.append({
                        "attempt": attempt_index,
                        "query": ",".join(id_list),
                        "source": "pubmed_followup_efetch_abstract_signal",
                        "hits": len(abstract_signal_audit),
                        "signal_hits": sum(
                            1 for item in abstract_signal_audit if item.get("route_signal_terms")
                        ),
                        "query_strategy": "primary" if attempt_index == 1 else "fallback",
                    })
                    _attach_abstract_signal_audit(cards, abstract_signal_audit)
                except Exception as exc:
                    abstract_signal_status = "abstract_signal_audit_error"
                    search_rows.append({
                        "attempt": attempt_index,
                        "query": ",".join(id_list),
                        "source": "pubmed_followup_efetch_abstract_signal",
                        "hits": 0,
                        "error": type(exc).__name__,
                        "query_strategy": "primary" if attempt_index == 1 else "fallback",
                    })
            resolved_query = attempt_query
            search_rows.append({
                "attempt": attempt_index,
                "query": ",".join(id_list),
                "source": "pubmed_followup_esummary",
                "hits": len(cards),
                "query_strategy": "primary" if attempt_index == 1 else "fallback",
            })
            break
    except Exception as exc:  # pragma: no cover - runtime report records failure.
        search_rows.append({
            "query": str(query),
            "source": "pubmed_followup_eutils",
            "hits": 0,
            "error": type(exc).__name__,
        })
    abstract_signal_terms = sorted({
        str(term)
        for item in abstract_signal_audit
        for term in item.get("route_signal_terms") or []
        if str(term).strip()
    })
    if include_abstract_signals and abstract_signal_status != "abstract_signal_audit_error":
        if abstract_signal_terms:
            abstract_signal_status = "abstract_route_signal_detected"
        elif abstract_signal_audit:
            abstract_signal_status = "abstract_missing_or_no_route_signal"
        else:
            abstract_signal_status = "no_pubmed_hits_for_abstract_signal_audit"
    report = {
        "schema_version": "literature_followup_search_report.v1",
        "case_id": case_id,
        "backend": "pubmed_followup",
        "query": str(query),
        "query_variants": attempted_queries,
        "query_attempt_count": len([row for row in search_rows if row.get("source") == "pubmed_followup_esearch"]),
        "resolved_query": resolved_query,
        "fallback_used": bool(resolved_query and resolved_query != str(query)),
        "searches": search_rows,
        "hit_count": len(cards),
        "evidence_levels": _evidence_level_counts(cards),
        "unresolved_literature_gap": len(cards) == 0,
        "limitations": [] if cards else ["unresolved_literature_gap"],
        "abstract_signal_audit_requested": bool(include_abstract_signals),
        "abstract_signal_status": abstract_signal_status,
        "abstract_signal_record_count": len(abstract_signal_audit),
        "abstract_signal_hit_count": sum(1 for item in abstract_signal_audit if item.get("route_signal_terms")),
        "abstract_signal_terms": abstract_signal_terms,
        "abstract_signal_audit": abstract_signal_audit,
    }
    return cards, report


def _dedupe_queries(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        query = " ".join(str(value or "").split())
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        out.append(query)
    return out


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
    family_constraint = _preferred_family_constraint(db, family_hints, terms)
    hinted_families = _hinted_family_ids(db, family_hints, terms, family_constraint=family_constraint)
    family_rows = _family_rows(db, family_constraint)
    anchor_rows = [
        item for item in _family_filtered_rows(db.get("anchors", []), family_constraint)
        if _target_scope_matches(item, terms, family_hints)
    ]
    disconnection_rows = [
        item for item in _family_filtered_rows(db.get("disconnections", []), family_constraint)
        if _target_scope_matches(item, terms, family_hints)
    ]
    family_score = {
        str(item.get("family_id") or ""): _score_record(item, terms, family_hints)
        for item in family_rows
    }
    families = _sort_records([
        item for item in family_rows
        if item.get("family_id") in hinted_families or _record_matches(item, terms)
    ], terms, family_hints, family_score)
    family_ids = {item.get("family_id") for item in families}
    anchors = _sort_records([
        item for item in anchor_rows
        if item.get("family_id") in family_ids or _record_matches(item, terms)
    ], terms, family_hints, family_score)
    disconnections = _sort_records([
        item for item in disconnection_rows
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


def _pubmed_query(task: LiteratureSearchTask) -> str:
    terms = _query_terms(task)
    drug_terms = sorted(_statin_drugs_in_text(terms))
    route_terms = [
        "synthesis",
        "semisynthesis",
        "route",
        "fermentation",
        "biotransformation",
        "intermediate",
    ]
    if drug_terms:
        target_part = " OR ".join(f'"{term}"' for term in drug_terms[:4])
    else:
        profile = task.target_profile or {}
        target_name = str(profile.get("target_name") or task.case_id or "").replace("_", " ").strip()
        target_part = f'"{target_name}"' if target_name else "statin"
    return f"({target_part}) AND ({' OR '.join(route_terms)})"


def _pubmed_url(endpoint: str, params: dict[str, str]) -> str:
    return f"{PUBMED_EUTILS_BASE}/{endpoint}?{urlencode(params)}"


def _fetch_pubmed_json(url: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_pubmed_text(url: str, *, timeout_s: float = 10.0) -> str:
    with urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8", errors="replace")


def _pubmed_abstract_signal_audit(id_list: list[str], *, timeout_s: float = 10.0) -> list[dict[str, Any]]:
    clean_ids = [str(uid).strip() for uid in id_list if str(uid).strip()]
    if not clean_ids:
        return []
    efetch_url = _pubmed_url(
        "efetch.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(clean_ids),
            "retmode": "xml",
            "tool": "AutoPlanner",
        },
    )
    xml_text = _fetch_pubmed_text(efetch_url, timeout_s=timeout_s)
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for article in root.findall(".//PubmedArticle"):
        pmid = str(article.findtext("./MedlineCitation/PMID") or "").strip()
        if not pmid:
            continue
        abstract_text = _pubmed_abstract_text(article)
        rows.append(_pubmed_abstract_signal_row(pmid, abstract_text))
        seen.add(pmid)
    for pmid in clean_ids:
        if pmid in seen:
            continue
        rows.append({
            "schema_version": "pubmed_abstract_route_signal_audit.v1",
            "pmid": pmid,
            "abstract_available": False,
            "abstract_text_char_count": 0,
            "route_signal_terms": [],
            "route_signal_count": 0,
            "route_signal_status": "efetch_record_missing",
            "limitations": [
                "abstract_text_not_stored",
                "not_reaction_extraction",
                "requires_full_text_route_audit",
            ],
        })
    return rows


def _pubmed_abstract_text(article: ET.Element) -> str:
    parts: list[str] = []
    for node in article.findall("./MedlineCitation/Article/Abstract/AbstractText"):
        text = " ".join(" ".join(node.itertext()).split())
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _pubmed_abstract_signal_row(pmid: str, abstract_text: str) -> dict[str, Any]:
    terms = _pubmed_route_signal_terms(abstract_text)
    if terms:
        status = "abstract_route_signal_detected"
    elif abstract_text:
        status = "abstract_available_no_route_signal"
    else:
        status = "abstract_missing"
    return {
        "schema_version": "pubmed_abstract_route_signal_audit.v1",
        "pmid": pmid,
        "abstract_available": bool(abstract_text),
        "abstract_text_char_count": len(abstract_text),
        "route_signal_terms": terms,
        "route_signal_count": len(terms),
        "route_signal_status": status,
        "limitations": [
            "abstract_text_not_stored",
            "not_reaction_extraction",
            "requires_full_text_route_audit",
        ],
    }


def _pubmed_route_signal_terms(text: str) -> list[str]:
    lowered = str(text or "").lower()
    return [term for term in PUBMED_ABSTRACT_ROUTE_SIGNAL_TERMS if term in lowered]


def _attach_abstract_signal_audit(cards: list[EvidenceCard], audit_rows: list[dict[str, Any]]) -> None:
    audit_by_pmid = {str(item.get("pmid") or ""): item for item in audit_rows}
    for card in cards:
        pmid = str((card.source_metadata or {}).get("pmid") or "")
        if not pmid:
            continue
        card.source_metadata["abstract_signal_audit"] = dict(audit_by_pmid.get(pmid) or {})


def _pubmed_cards_from_summary(
    task: LiteratureSearchTask,
    payload: dict[str, Any],
    id_list: list[str],
    *,
    query: str,
) -> list[EvidenceCard]:
    result = payload.get("result") or {}
    cards: list[EvidenceCard] = []
    for uid in id_list:
        item = result.get(uid) or {}
        if not item:
            continue
        title = str(item.get("title") or f"PubMed PMID {uid}")
        doi = _pubmed_doi(item)
        pubdate = str(item.get("pubdate") or "")
        journal = str(item.get("fulljournalname") or item.get("source") or "")
        cards.append(EvidenceCard(
            evidence_id=f"ev_pubmed_{uid}",
            case_id=task.case_id,
            source_type="literature",
            source_title=title,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            doi=doi,
            local_ref="",
            source_record_id=f"pubmed:{uid}",
            family_id=_pubmed_family_id(task),
            target_relation=_pubmed_target_relation(task, title),
            claim_type="literature_search_result",
            route_role=_pubmed_route_role(title),
            confidence="medium",
            route_role_detail="PubMed title/summary evidence lead; requires full-text or curator audit before route promotion.",
            limitations=[
                "pubmed_summary_only",
                "not_reaction_extraction",
                "requires_full_text_route_audit",
            ],
            source_metadata={
                "backend": "pubmed",
                "pmid": uid,
                "journal": journal,
                "pubdate": pubdate,
                "query": query,
                "articleids": item.get("articleids") or [],
            },
        ))
    return cards


def _pubmed_doi(item: dict[str, Any]) -> str:
    for article_id in item.get("articleids") or []:
        if str(article_id.get("idtype") or "").lower() == "doi":
            return str(article_id.get("value") or "")
    return ""


def _pubmed_family_id(task: LiteratureSearchTask) -> str:
    text = " ".join(_query_terms(task)).lower().replace("-", "_")
    if "statin" in text:
        if any(term in text for term in ("lovastatin", "mevastatin", "pravastatin", "simvastatin", "compactin", "fermentation")):
            return "natural_statin_semisynthesis"
        return "synthetic_statin"
    return ""


def _pubmed_target_relation(task: LiteratureSearchTask, title: str) -> str:
    title_text = title.lower().replace("-", "_")
    drugs = _statin_drugs_in_text(_query_terms(task))
    if drugs and any(drug in title_text for drug in drugs):
        return "exact_target_or_intermediate"
    if "statin" in title_text:
        return "family_precedent"
    return "reaction_precedent"


def _pubmed_route_role(title: str) -> str:
    text = title.lower().replace("-", "_")
    if any(term in text for term in ("synthesis", "semisynthesis", "fermentation", "biotransformation")):
        return "route_anchor"
    if any(term in text for term in ("condition", "process", "intermediate")):
        return "condition_hint"
    return "unknown"


def _query_terms(task: LiteratureSearchTask) -> list[str]:
    profile = task.target_profile or {}
    terms = list(task.family_hints)
    terms.extend(str(x) for x in profile.get("family_hints") or [])
    terms.extend(str(x) for x in [profile.get("target_name"), profile.get("formula")] if x)
    if task.frontier_smiles:
        terms.append(task.frontier_smiles)
    return [term for term in terms if term]


def _hinted_family_ids(
    db: dict[str, Any],
    hints: list[str],
    terms: list[str],
    *,
    family_constraint: set[str] | None = None,
) -> set[str]:
    text_terms = " ".join([*hints, *terms]).lower().replace("-", "_")
    family_ids: set[str] = set()
    for family in _family_rows(db, family_constraint):
        fid = str(family.get("family_id") or "")
        keywords = [fid, str(family.get("name") or "")]
        keywords.extend(str(x) for x in family.get("pattern_keywords") or [])
        if any(_token_match(text_terms, keyword) for keyword in keywords):
            family_ids.add(fid)
    return family_ids


def _preferred_family_constraint(
    db: dict[str, Any],
    hints: list[str],
    terms: list[str],
) -> set[str] | None:
    """Apply high-precision constraints for families with broad generic terms.

    Statin prompts often include words like semisynthesis, fermentation, side
    chain, and convergence.  Those words are useful inside a statin family but
    too broad for first-pass family retrieval, so explicit statin cases are
    narrowed before record matching.
    """
    text = " ".join([*hints, *terms]).lower().replace("-", "_")
    statin_families = {str(f.get("family_id") or "") for f in db.get("families", []) if "statin" in str(f.get("family_id") or "").lower()}
    if not statin_families:
        return None
    natural_terms = {"lovastatin", "simvastatin", "pravastatin", "mevastatin", "compactin"}
    synthetic_terms = {"atorvastatin", "fluvastatin", "pitavastatin", "rosuvastatin", "cerivastatin"}
    natural_signal = (
        "natural_statin" in text
        or "fermentation_core" in text
        or "fermentation-derived" in text
        or any(term in text for term in natural_terms)
    )
    synthetic_signal = (
        "synthetic_statin" in text
        or "syn_3,5" in text
        or "hwe" in text
        or "wittig" in text
        or any(term in text for term in synthetic_terms)
    )
    if natural_signal and not synthetic_signal:
        return {"natural_statin_semisynthesis"}.intersection(statin_families)
    if synthetic_signal and not natural_signal:
        return {"synthetic_statin"}.intersection(statin_families)
    if natural_signal or synthetic_signal or "statin" in text:
        return statin_families
    return None


def _family_rows(db: dict[str, Any], family_constraint: set[str] | None) -> list[dict[str, Any]]:
    rows = [item for item in db.get("families", []) if isinstance(item, dict)]
    if family_constraint is None:
        return rows
    return [item for item in rows if str(item.get("family_id") or "") in family_constraint]


def _family_filtered_rows(rows: list[Any], family_constraint: set[str] | None) -> list[dict[str, Any]]:
    out = [item for item in rows if isinstance(item, dict)]
    if family_constraint is None:
        return out
    return [item for item in out if str(item.get("family_id") or "") in family_constraint]


def _record_matches(item: Any, terms: list[str]) -> bool:
    if not terms:
        return False
    text = json.dumps(item, ensure_ascii=False).lower()
    return any(term and term in text for term in terms)


def _target_scope_matches(item: dict[str, Any], terms: list[str], family_hints: list[str]) -> bool:
    target_drugs = _statin_drugs_in_text([*terms, *family_hints])
    if not target_drugs:
        return True
    record_drugs = _record_applicability_drugs(item)
    if not record_drugs:
        return True
    return bool(target_drugs & record_drugs)


def _record_applicability_drugs(item: dict[str, Any]) -> set[str]:
    applicability = item.get("applicability") if isinstance(item, dict) else {}
    if not isinstance(applicability, dict):
        return set()
    classes = [str(value) for value in applicability.get("target_classes") or []]
    return _statin_drugs_in_text(classes)


def _statin_drugs_in_text(values: list[str]) -> set[str]:
    text = " ".join(str(value or "") for value in values).lower().replace("-", "_")
    return {drug for drug in STATIN_DRUG_NAMES if drug in text}


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
    record_key = " ".join(str(item.get(key) or "") for key in ("id", "anchor_id", "name")).lower().replace("-", "_")
    hints = [str(hint).lower().replace("-", "_") for hint in family_hints if str(hint).strip()]
    drug_name_hints = {
        "atorvastatin",
        "fluvastatin",
        "pitavastatin",
        "rosuvastatin",
        "cerivastatin",
        "lovastatin",
        "simvastatin",
        "pravastatin",
        "mevastatin",
        "compactin",
    }
    score = 0
    for hint in hints:
        if not hint:
            continue
        if hint in drug_name_hints and hint in record_key:
            score += 260
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

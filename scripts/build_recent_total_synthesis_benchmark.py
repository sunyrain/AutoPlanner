#!/usr/bin/env python3
"""Build an auditable recent-total-synthesis discovery and curation dataset.

The builder deliberately separates three claims that are often conflated:

1. metadata records discovered in public indexes;
2. papers admitted by human bibliographic/scope review; and
3. runnable targets with source-concordant structures and route evidence.

Crossref, OpenAlex, and Europe PMC are merged by DOI and normalized title.  The
existing manual curation is imported only as a seed; automated candidates never
become benchmark truth merely because a title contains "total synthesis".
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import html
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FREEZE_START = "2025-01-01"
FREEZE_END = "2026-09-01"
STRICT_POST_CUTOFF = "2026-02-16"
SYNTHEX_CUTOFF_PROXY = "2025-01-31"

QUERY_TERMS = {
    "total_synthesis": "total synthesis",
    "formal_synthesis": "formal synthesis",
    "collective_synthesis": "collective synthesis",
    "divergent_synthesis": "divergent synthesis natural product",
    "biomimetic_synthesis": "biomimetic synthesis natural product",
    "bioinspired_synthesis": "bioinspired synthesis natural product",
    "chemoenzymatic_synthesis": "chemoenzymatic total synthesis",
    "concise_synthesis": "concise synthesis natural product",
    "natural_product_synthesis": "natural product synthesis",
    "stereochemical_revision": "synthesis stereochemical revision",
    "structure_revision": "synthesis structure revision",
    "unified_synthesis": "unified synthesis natural product",
    "biocatalytic_synthesis": "biocatalytic total synthesis",
    "enzymatic_synthesis": "enzymatic total synthesis",
    "chemical_synthesis": "chemical synthesis natural product",
}

REVIEW_RE = re.compile(
    r"\b(review|perspective|overview|recent (?:advances|progress|highlights|update)|"
    r"applications? in|account|retrospective|strategies for|advanced strategies|"
    r"advances in|an update on|key contributions and perspectives|lessons from|"
    r"tales of|occurrences and opportunities)\b",
    re.IGNORECASE,
)
INCOMPLETE_RE = re.compile(
    r"\b(toward|towards|studies toward|approach(?:es)? toward|synthetic studies|"
    r"progress toward|attempted synthesis|founding strategic bases for)\b",
    re.IGNORECASE,
)
SECONDARY_RE = re.compile(r"\b(synfacts|retraction|correction|erratum|corrigendum)\b", re.I)
NON_CORE_RE = re.compile(
    r"\b(proteins?|peptides?|cyclo\w*peptides?|glycans?|oligosaccharides?|polysaccharides?|"
    r"peptidic|polypeptides?|depsipeptides?|tetrapeptides?|docosapeptides?|"
    r"pentasaccharide|nucleoside component|"
    r"o-antigen|antigenic determinant|gene|polymer|nanoparticle|energetic material|"
    r"fluorescent compound|bioisostere|drug substance|process route)\b",
    re.IGNORECASE,
)
METHOD_ONLY_RE = re.compile(
    r"\b(application in|applications? to|inspired by|methodology|scope|catalyzed .{0,80}"
    r"(?:reaction|functionalization|hydrogenation|coupling)|preparation of analogues|"
    r"strategy in natural product total synthesis)\b",
    re.IGNORECASE,
)
NATURAL_PRODUCT_SIGNAL_RE = re.compile(
    r"\b(natural product|alkaloid|terpen(?:e|oid)|sesquiterpen|diterpen|triterpen|"
    r"isoprenoid|steroid|polyketide|macrolide|lactone|lactam|flavonoid|lignan|"
    r"meroterpenoid|antibiotic|toxin|marine-derived|fungal|metabolite|phytocannabinoid|"
    r"mushroom|biosynthesis)\b",
    re.IGNORECASE,
)
CHEMISTRY_VENUE_RE = re.compile(
    r"\b(chem|organic|jacs|tetrahedron|synthesis|synlett|catalysis|natural products?|"
    r"molecular diversity|bioscience|fitoterapia)\b",
    re.IGNORECASE,
)
COMPLETED_RE = re.compile(
    r"\b(total synthesis|total syntheses|formal synthesis|formal syntheses|"
    r"collective synthesis|collective syntheses|divergent synthesis|divergent syntheses|"
    r"unified synthesis|concise synthesis|biomimetic synthesis|bioinspired synthesis|"
    r"chemoenzymatic synthesis|enantioselective synthesis|asymmetric synthesis)\b",
    re.IGNORECASE,
)
OMISSION_ROUTE_RE = re.compile(
    r"\b(synthesis and stereochemical revision of|synthesis and structure revision of|"
    r"chemical synthesis of (?:the )?.{0,80}natural product|"
    r"synthesis of the natural product|synthesis of natural product)\b",
    re.IGNORECASE,
)
FORMAL_SYNTHESIS_RE = re.compile(r"\bformal (?:total )?synthes(?:is|es)\b", re.IGNORECASE)
ROUTE_IMPROVEMENT_RE = re.compile(
    r"\b(?:improved|improvements? in|revised) (?:the )?(?:formal )?"
    r"(?:total )?synthes(?:is|es)\b",
    re.IGNORECASE,
)
CONTROL_SCOPE_RE = re.compile(
    r"\b(?:major human metabolite|clinical candidate|key synthetic precursor|"
    r"key intermediate|oral contraceptive|photoswitchable|unnatural|"
    r"minimized analog|natural product analogues?|hypercholesterolemia drug)\b",
    re.IGNORECASE,
)
ABSTRACT_REVIEW_RE = re.compile(
    r"\b(?:this review|aim of review|review aims|we review|comprehensive review)\b",
    re.IGNORECASE,
)
ABSTRACT_CONTROL_RE = re.compile(
    r"\b(?:important pharmaceutical|pharmacological agent used|approved drug|"
    r"unnatural .{0,50} analog)\b",
    re.IGNORECASE,
)
METHOD_APPLICATION_TOTAL_RE = re.compile(
    r"\b(?:application to (?:the )?total synthesis|using .{0,80}(?:reactors?|"
    r"immobilized (?:enzyme|transaminase)))\b",
    re.IGNORECASE,
)
COMPLETION_CUE_RE = re.compile(
    r"\b(?:we (?:report|describe|present|achieved|accomplished)|"
    r"has been achieved|have been achieved|was achieved|were achieved|"
    r"was accomplished|were accomplished|first (?:asymmetric )?total synthes(?:is|es)|"
    r"total synthesis .{0,80} (?:is|are) reported|"
    r"synthesis .{0,80} was completed)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("tmp/literature-benchmark-metadata-cache"),
    )
    parser.add_argument(
        "--manual-seed",
        type=Path,
        default=Path("benchmarks/literature_strategy_rediscovery_v0_1/manual_curation_v0_1.json"),
    )
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(part) for part in value)
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_doi(value: Any) -> str:
    doi = clean_text(value).lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi.strip().rstrip(".,")


def normalize_title(value: Any) -> str:
    text = clean_text(value).casefold()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def request_json(url: str, cache: Path, *, offline: bool) -> Any:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if offline:
        raise FileNotFoundError(f"offline cache missing: {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "AutoPlanner-literature-benchmark/0.2",
                    "Accept": "application/json",
                },
            )
            with urlopen(request, timeout=90) as response:
                payload = response.read()
            cache.write_bytes(payload)
            return json.loads(payload)
        except Exception as exc:  # bounded network retry with cached receipts
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def crossref_date(item: dict[str, Any]) -> str:
    for field in ("published-online", "published-print", "issued"):
        parts = ((item.get(field) or {}).get("date-parts") or [[]])[0]
        if not parts:
            continue
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return f"{year:04d}-{month:02d}-{day:02d}"
    return ""


def openalex_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    positions = [
        int(position)
        for values in inverted_index.values()
        if isinstance(values, list)
        for position in values
    ]
    if not positions:
        return ""
    words = [""] * (max(positions) + 1)
    for word, values in inverted_index.items():
        if not isinstance(values, list):
            continue
        for position in values:
            index = int(position)
            if 0 <= index < len(words):
                words[index] = str(word)
    return clean_text(" ".join(words))[:6000]


def crossref_records(cache_dir: Path, *, offline: bool) -> Iterable[dict[str, Any]]:
    for query_id, term in QUERY_TERMS.items():
        params = {
            "query.title": term,
            "filter": (
                f"from-pub-date:{FREEZE_START},until-pub-date:{FREEZE_END},type:journal-article"
            ),
            "rows": 1000,
            "select": (
                "DOI,title,container-title,published-online,published-print,issued,"
                "URL,author,abstract,link,type"
            ),
        }
        url = "https://api.crossref.org/works?" + urlencode(params)
        payload = request_json(
            url,
            cache_dir / f"crossref-{query_id}.json",
            offline=offline,
        )
        for item in payload.get("message", {}).get("items", []):
            authors = item.get("author") or []
            first_author = ""
            if authors:
                first_author = clean_text(
                    f"{authors[0].get('given', '')} {authors[0].get('family', '')}"
                )
            links = item.get("link") or []
            yield {
                "provider": "crossref",
                "provider_query": query_id,
                "doi": normalize_doi(item.get("DOI")),
                "title": clean_text(item.get("title")),
                "journal": clean_text(item.get("container-title")),
                "publication_date": crossref_date(item),
                "first_author": first_author,
                "abstract": clean_text(item.get("abstract"))[:6000],
                "source_url": clean_text(item.get("URL")),
                "fulltext_link_count": len(links),
                "fulltext_links": [
                    {
                        "url": clean_text(link.get("URL")),
                        "content_type": clean_text(link.get("content-type")),
                        "content_version": clean_text(link.get("content-version")),
                        "intended_application": clean_text(link.get("intended-application")),
                    }
                    for link in links
                    if clean_text(link.get("URL"))
                ],
            }


def openalex_records(cache_dir: Path, *, offline: bool) -> Iterable[dict[str, Any]]:
    for query_id, term in QUERY_TERMS.items():
        cursor = "*"
        page = 0
        while cursor:
            filters = ",".join(
                [
                    f"title.search:{term}",
                    f"from_publication_date:{FREEZE_START}",
                    f"to_publication_date:{FREEZE_END}",
                    "type:article",
                ]
            )
            params = {"filter": filters, "per-page": 200, "cursor": cursor}
            url = "https://api.openalex.org/works?" + urlencode(params)
            payload = request_json(
                url,
                cache_dir / f"openalex-{query_id}-{page:03d}.json",
                offline=offline,
            )
            for item in payload.get("results", []):
                authorships = item.get("authorships") or []
                first_author = ""
                if authorships:
                    first_author = clean_text(
                        (authorships[0].get("author") or {}).get("display_name")
                    )
                location = item.get("primary_location") or {}
                source = location.get("source") or {}
                oa = item.get("open_access") or {}
                best_oa = item.get("best_oa_location") or {}
                yield {
                    "provider": "openalex",
                    "provider_query": query_id,
                    "doi": normalize_doi(item.get("doi")),
                    "title": clean_text(item.get("title")),
                    "journal": clean_text(source.get("display_name")),
                    "publication_date": clean_text(item.get("publication_date")),
                    "first_author": first_author,
                    "abstract": openalex_abstract(item.get("abstract_inverted_index")),
                    "source_url": clean_text(location.get("landing_page_url")),
                    "open_access": bool(oa.get("is_oa")),
                    "oa_landing_url": clean_text(best_oa.get("landing_page_url")),
                    "oa_pdf_url": clean_text(best_oa.get("pdf_url")),
                    "openalex_id": clean_text(item.get("id")),
                }
            next_cursor = clean_text((payload.get("meta") or {}).get("next_cursor"))
            if not next_cursor or next_cursor == cursor or not payload.get("results"):
                break
            cursor = next_cursor
            page += 1
            if page >= 30:
                raise RuntimeError(f"OpenAlex pagination cap reached for {query_id}")


def europe_pmc_records(cache_dir: Path, *, offline: bool) -> Iterable[dict[str, Any]]:
    for query_id, term in QUERY_TERMS.items():
        query = f'TITLE:"{term}" AND FIRST_PDATE:[{FREEZE_START} TO {FREEZE_END}] AND SRC:MED'
        params = {
            "query": query,
            "format": "json",
            "pageSize": 1000,
            "resultType": "core",
        }
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode(params)
        payload = request_json(
            url,
            cache_dir / f"europe-pmc-{query_id}.json",
            offline=offline,
        )
        for item in (payload.get("resultList") or {}).get("result", []):
            journal = item.get("journalInfo") or {}
            journal_title = (journal.get("journal") or {}).get("title")
            yield {
                "provider": "europe_pmc",
                "provider_query": query_id,
                "doi": normalize_doi(item.get("doi")),
                "title": clean_text(item.get("title")),
                "journal": clean_text(journal_title or item.get("journalTitle")),
                "publication_date": clean_text(
                    item.get("firstPublicationDate") or item.get("firstIndexDate")
                )[:10],
                "first_author": clean_text(item.get("authorString")),
                "abstract": clean_text(item.get("abstractText"))[:6000],
                "source_url": (
                    f"https://europepmc.org/article/MED/{item.get('pmid')}"
                    if item.get("pmid")
                    else ""
                ),
                "open_access": str(item.get("isOpenAccess") or "").upper() == "Y",
                "repository_fulltext": bool(item.get("pmcid")),
                "pmid": clean_text(item.get("pmid")),
                "pmcid": clean_text(item.get("pmcid")),
            }


def _prefer(current: str, candidate: str) -> str:
    if not current:
        return candidate
    if not candidate:
        return current
    return candidate if len(candidate) > len(current) else current


def merge_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for raw in records:
        title = clean_text(raw.get("title"))
        doi = normalize_doi(raw.get("doi"))
        if not title:
            continue
        fallback = f"{normalize_title(title)}|{raw.get('publication_date', '')[:4]}"
        key = f"doi:{doi}" if doi else f"title:{fallback}"
        row = merged.setdefault(
            key,
            {
                "paper_id": stable_id("paper", key),
                "doi": doi,
                "title": title,
                "journal": clean_text(raw.get("journal")),
                "publication_date": clean_text(raw.get("publication_date"))[:10],
                "first_author": clean_text(raw.get("first_author")),
                "abstract": clean_text(raw.get("abstract")),
                "source_url": clean_text(raw.get("source_url")),
                "providers": [],
                "provider_queries": [],
                "open_access": False,
                "repository_fulltext": False,
                "pmid": "",
                "pmcid": "",
                "openalex_id": "",
                "oa_landing_url": "",
                "oa_pdf_url": "",
                "fulltext_link_count": 0,
                "fulltext_links": [],
            },
        )
        row["title"] = _prefer(row["title"], title)
        row["journal"] = _prefer(row["journal"], clean_text(raw.get("journal")))
        candidate_date = clean_text(raw.get("publication_date"))[:10]
        if candidate_date and (
            not row["publication_date"] or candidate_date < row["publication_date"]
        ):
            row["publication_date"] = candidate_date
        row["first_author"] = _prefer(row["first_author"], clean_text(raw.get("first_author")))
        row["abstract"] = _prefer(row["abstract"], clean_text(raw.get("abstract")))
        row["source_url"] = _prefer(row["source_url"], clean_text(raw.get("source_url")))
        row["providers"] = sorted(set(row["providers"]) | {clean_text(raw.get("provider"))})
        row["provider_queries"] = sorted(
            set(row["provider_queries"]) | {clean_text(raw.get("provider_query"))}
        )
        row["open_access"] = bool(row["open_access"] or raw.get("open_access"))
        row["repository_fulltext"] = bool(
            row["repository_fulltext"] or raw.get("repository_fulltext")
        )
        for field in (
            "pmid",
            "pmcid",
            "openalex_id",
            "oa_landing_url",
            "oa_pdf_url",
        ):
            row[field] = _prefer(row[field], clean_text(raw.get(field)))
        row["fulltext_link_count"] = max(
            int(row["fulltext_link_count"] or 0),
            int(raw.get("fulltext_link_count") or 0),
        )
        fulltext_by_url = {
            str(link.get("url") or ""): dict(link)
            for link in row["fulltext_links"]
            if link.get("url")
        }
        for link in raw.get("fulltext_links") or []:
            url = str(link.get("url") or "")
            if url:
                fulltext_by_url[url] = dict(link)
        row["fulltext_links"] = [fulltext_by_url[url] for url in sorted(fulltext_by_url)]

    rows = list(merged.values())
    title_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family_key = normalize_title(row["title"])
        title_groups.setdefault(family_key, []).append(row)
    for family_key, family in title_groups.items():
        family_id = stable_id("family", family_key)
        family.sort(
            key=lambda row: (
                0 if row["doi"].startswith("10.1002/anie.") else 1,
                0 if row["doi"] else 1,
                row["doi"],
            )
        )
        for index, row in enumerate(family):
            row["article_family_id"] = family_id
            row["preferred_family_record"] = index == 0
            row["family_record_count"] = len(family)
    rows.sort(key=lambda row: (row["publication_date"], row["doi"], row["title"]))
    return rows


def load_manual_seed(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {
        normalize_doi(row.get("doi")): dict(row)
        for row in payload.get("records", [])
        if normalize_doi(row.get("doi"))
    }
    return records, payload


def title_target_phrase(title: str) -> str:
    match = re.search(
        r"\b(?:total|formal|collective|divergent|unified|concise|biomimetic|"
        r"bioinspired|chemoenzymatic|enantioselective|asymmetric) synthes(?:is|es) of\s+(.+)",
        title,
        re.IGNORECASE,
    )
    if not match:
        return ""
    value = match.group(1)
    value = re.split(
        r"\s+(?:via|using|through|enabled by|employing|and (?:revision|evaluation|"
        r"assessment|discovery|functional analysis|biological evaluation))\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return value.strip(" .:;")[:500]


def automated_screen(row: dict[str, Any]) -> tuple[str, str]:
    title = row["title"]
    journal = row["journal"]
    combined = f"{title} {journal}"
    abstract = str(row.get("abstract") or "")
    scope_context = f"{title} {abstract}"
    doi = row["doi"]
    if (
        doi.startswith("10.5281/zenodo.")
        or re.search(r"\.s\d{2,}$", doi)
        or "zenodo" in journal.casefold()
    ):
        return (
            "exclude_repository_or_supplement",
            "repository record or supplementary-material DOI, not a primary paper",
        )
    if doi.startswith(("10.17638/", "10.7907/")):
        return (
            "scope_review_thesis_or_repository",
            "institutional-repository record requires primary-paper resolution",
        )
    if SECONDARY_RE.search(combined):
        return "exclude_secondary_or_correction", "secondary summary or correction"
    if REVIEW_RE.search(title) or ABSTRACT_REVIEW_RE.search(abstract):
        return "exclude_review", "review, perspective, or account"
    if INCOMPLETE_RE.search(title):
        return "exclude_incomplete", "toward/studies rather than completed synthesis"
    if FORMAL_SYNTHESIS_RE.search(title):
        return "scope_review_formal_synthesis", "formal synthesis is a conditional stratum"
    if ROUTE_IMPROVEMENT_RE.search(title):
        return (
            "scope_review_route_improvement",
            "route improvement is not a new target-level total-synthesis report",
        )
    if CONTROL_SCOPE_RE.search(title) or ABSTRACT_CONTROL_RE.search(abstract):
        return (
            "scope_review_control_target",
            "drug, metabolite, precursor, unnatural target, or analogue control signal",
        )
    if METHOD_APPLICATION_TOTAL_RE.search(title):
        return (
            "scope_review_method_application",
            "method/platform paper with a total-synthesis application",
        )
    if METHOD_ONLY_RE.search(title) and not re.search(r"\btotal synthesis of\b", title, re.I):
        return "scope_review_method_application", "method/application wording"
    if NON_CORE_RE.search(scope_context):
        return "scope_review_noncore", "peptide/glycan/material/process scope signal"
    if re.search(r"\btotal synthes(?:is|es)\b", title, re.IGNORECASE):
        return "high_priority_primary_candidate", "completed-synthesis title signal"
    if OMISSION_ROUTE_RE.search(title):
        return (
            "high_priority_omission_candidate",
            "completed natural-product route wording outside the standard total-synthesis phrase",
        )
    if COMPLETED_RE.search(title):
        context = f"{title} {row.get('abstract', '')}"
        if NATURAL_PRODUCT_SIGNAL_RE.search(context):
            return (
                "high_priority_primary_candidate",
                "completed-synthesis wording with natural-product context",
            )
        return (
            "manual_title_review",
            "completed-synthesis wording without natural-product context",
        )
    return "manual_title_review", "discovery-query match without a completed-route signal"


def annotate_rows(rows: list[dict[str, Any]], manual: dict[str, dict[str, Any]]) -> None:
    manual_status_map = {
        "benchmark_primary": "admitted_metadata_primary",
        "benchmark_conditional": "admitted_metadata_conditional",
        "benchmark_control": "control_manual",
        "exclude": "excluded_manual",
    }
    for row in rows:
        row["after_synthex_cutoff_proxy"] = bool(
            row["publication_date"] and row["publication_date"] > SYNTHEX_CUTOFF_PROXY
        )
        row["after_strict_model_cutoff"] = bool(
            row["publication_date"] and row["publication_date"] > STRICT_POST_CUTOFF
        )
        row["title_target_phrase"] = title_target_phrase(row["title"])
        automatic_status, automatic_reason = automated_screen(row)
        row["automated_status"] = automatic_status
        row["automated_reason"] = automatic_reason
        seed = manual.get(row["doi"])
        if seed:
            row["curation_status"] = manual_status_map.get(
                seed.get("status", ""), "manual_unmapped"
            )
            row["curation_basis"] = clean_text(seed.get("evidence_basis"))
            row["curation_reason"] = clean_text(seed.get("reason"))
            row["primary_target_count"] = int(seed.get("primary_target_count") or 0)
            row["conditional_target_count"] = int(seed.get("conditional_target_count") or 0)
            row["control_target_count"] = int(seed.get("control_target_count") or 0)
            row["target_names"] = list(seed.get("target_names") or [])
            row["identity_status"] = clean_text(seed.get("identity_status"))
        else:
            row["curation_status"] = "unreviewed"
            row["curation_basis"] = ""
            row["curation_reason"] = ""
            row["primary_target_count"] = 0
            row["conditional_target_count"] = 0
            row["control_target_count"] = 0
            row["target_names"] = []
            row["identity_status"] = "not_reviewed"
        row["source_concordant_structure_count"] = 0
        row["key_step_evidence_count"] = 0
        row["runnable_target_count"] = 0


def build_target_slots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for row in rows:
        counts = [
            ("primary", int(row["primary_target_count"])),
            ("conditional", int(row["conditional_target_count"])),
            ("control", int(row["control_target_count"])),
        ]
        names = list(row.get("target_names") or [])
        cursor = 0
        for slot_class, count in counts:
            for _ in range(count):
                target_name = names[cursor] if cursor < len(names) else ""
                cursor += 1
                slot_id = stable_id(
                    "target-slot", row["paper_id"], slot_class, str(cursor), target_name
                )
                slots.append(
                    {
                        "target_slot_id": slot_id,
                        "paper_id": row["paper_id"],
                        "doi": row["doi"],
                        "article_family_id": row["article_family_id"],
                        "publication_title": row["title"],
                        "publication_date": row["publication_date"],
                        "source_url": row["source_url"],
                        "slot_class": slot_class,
                        "target_name": target_name,
                        "target_identity_status": row["identity_status"],
                        "target_smiles": "",
                        "structure_status": "pending_source_concordant_structure",
                        "route_evidence_status": "pending_fulltext_and_si_extraction",
                        "runnable": False,
                    }
                )
    return slots


def _review_payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _review_source_artifact(
    value: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    raw_path = str(value.get("source_artifact_path") or "")
    relative = Path(raw_path)
    expected_sha = str(value.get("source_artifact_sha256") or "").lower()
    locator = value.get("source_locator")
    if not raw_path or relative.is_absolute() or not expected_sha or not locator:
        raise RuntimeError("human_review_source_binding_incomplete")
    root = repo_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("human_review_source_outside_repository") from exc
    if not resolved.is_file() or sha256(resolved) != expected_sha:
        raise RuntimeError("human_review_source_hash_invalid")
    return {
        "source_artifact_path": relative.as_posix(),
        "source_artifact_sha256": expected_sha,
        "source_locator": locator,
    }


def normalize_human_review_source_artifact(
    value: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Public entry point shared by submission tooling and the materializer."""

    return _review_source_artifact(value, repo_root=repo_root)


def _normalize_structure_review_record(
    value: dict[str, Any],
    *,
    target: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(value.get("isomeric_smiles") or ""))
    if molecule is None:
        raise RuntimeError("human_structure_review_smiles_invalid")
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    source_doi = normalize_doi(value.get("source_doi"))
    if source_doi != normalize_doi(target.get("doi")):
        raise RuntimeError("human_structure_review_doi_mismatch")
    if value.get("identity_confirmed") is not True:
        raise RuntimeError("human_structure_review_identity_not_confirmed")
    stereo_status = str(value.get("absolute_stereochemistry_status") or "")
    if stereo_status not in {"confirmed", "not_applicable"}:
        raise RuntimeError("human_structure_review_stereochemistry_incomplete")
    if value.get("relative_stereochemistry_confirmed") is not True:
        raise RuntimeError("human_structure_review_relative_stereo_incomplete")
    source = _review_source_artifact(value, repo_root=repo_root)
    return {
        "isomeric_smiles": smiles,
        "source_doi": source_doi,
        **source,
        "identity_confirmed": True,
        "relative_stereochemistry_confirmed": True,
        "absolute_stereochemistry_status": stereo_status,
    }


def normalize_human_structure_review_record(
    value: dict[str, Any],
    *,
    target: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Validate and canonicalize one accepted structure-review record."""

    return _normalize_structure_review_record(value, target=target, repo_root=repo_root)


def _normalize_route_review_record(
    value: dict[str, Any],
    *,
    target: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    source_doi = normalize_doi(value.get("source_doi"))
    if source_doi != normalize_doi(target.get("doi")):
        raise RuntimeError("human_route_review_doi_mismatch")
    reference_scope = str(value.get("reference_scope") or "")
    if reference_scope not in {"ordered_route", "strategic_key_step"}:
        raise RuntimeError("human_route_review_scope_invalid")
    raw_sources = list(value.get("source_artifacts") or [])
    if not raw_sources or not all(isinstance(row, dict) for row in raw_sources):
        raise RuntimeError("human_route_review_sources_missing")
    sources = [
        _review_source_artifact(dict(row), repo_root=repo_root)
        for row in raw_sources
    ]
    steps = [dict(row) for row in value.get("steps") or [] if isinstance(row, dict)]
    events = [
        dict(row)
        for row in value.get("strategic_events") or []
        if isinstance(row, dict)
    ]
    if reference_scope == "ordered_route" and not steps:
        raise RuntimeError("human_route_review_ordered_steps_missing")
    if not events:
        raise RuntimeError("human_route_review_strategic_events_missing")
    step_ids: list[str] = []
    for step in steps:
        step_id = str(step.get("step_id") or "")
        precursor_labels = list(step.get("precursor_labels") or [])
        if (
            not step_id
            or not str(step.get("product_label") or "")
            or not precursor_labels
            or not str(step.get("transformation_class") or "")
            or not str(step.get("strategic_role") or "")
            or not step.get("source_locator")
        ):
            raise RuntimeError("human_route_review_step_incomplete")
        step_ids.append(step_id)
    if len(step_ids) != len(set(step_ids)):
        raise RuntimeError("human_route_review_step_ids_not_unique")
    for event in events:
        if (
            not str(event.get("event_id") or "")
            or not str(event.get("description") or "")
            or not str(event.get("transformation_class") or "")
            or not event.get("source_locator")
        ):
            raise RuntimeError("human_route_review_strategic_event_incomplete")
    return {
        "source_doi": source_doi,
        "reference_scope": reference_scope,
        "source_artifacts": sources,
        "steps": steps,
        "strategic_events": events,
    }


def normalize_human_route_review_record(
    value: dict[str, Any],
    *,
    target: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Validate and canonicalize one accepted route-review record."""

    return _normalize_route_review_record(value, target=target, repo_root=repo_root)


def _review_row_identity(
    row: dict[str, Any],
    *,
    target_ids: set[str],
    reviewer_field: str,
) -> tuple[str, str, str]:
    target_id = str(row.get("target_slot_id") or "")
    reviewer_id = str(row.get(reviewer_field) or "")
    decision = str(row.get("decision") or "")
    if target_id not in target_ids or not reviewer_id:
        raise RuntimeError("human_review_identity_invalid")
    if decision not in {"accept", "reject", "needs_revision"}:
        raise RuntimeError("human_review_decision_invalid")
    if not str(row.get("reviewed_at") or "") or row.get("reviewer_attestation") is not True:
        raise RuntimeError("human_review_attestation_missing")
    return target_id, reviewer_id, decision


def _select_human_admission(
    *,
    subject_type: str,
    target: dict[str, Any],
    reviews: list[dict[str, Any]],
    adjudications: list[dict[str, Any]],
    normalizer: Any,
    repo_root: Path,
) -> tuple[dict[str, Any] | None, list[str], str, str]:
    accepted: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for review in reviews:
        if str(review.get("decision") or "") != "accept":
            continue
        record = normalizer(
            dict(review.get("record") or {}),
            target=target,
            repo_root=repo_root,
        )
        digest = _review_payload_digest(record)
        accepted.setdefault(digest, (record, []))[1].append(
            str(review.get("reviewer_id") or "")
        )
    consensus = [value for value in accepted.values() if len(set(value[1])) >= 2]
    if len(consensus) > 1:
        raise RuntimeError(f"human_{subject_type}_review_multiple_consensus_records")
    if consensus:
        record, reviewer_ids = consensus[0]
        return record, sorted(set(reviewer_ids)), "two_matching_accepts", "accepted"

    if len({str(row.get("reviewer_id") or "") for row in reviews}) >= 2:
        matching_adjudications = [
            row
            for row in adjudications
            if str(row.get("subject_type") or "") == subject_type
        ]
        if len(matching_adjudications) > 1:
            raise RuntimeError(f"human_{subject_type}_review_multiple_adjudications")
        if matching_adjudications:
            adjudication = matching_adjudications[0]
            if str(adjudication.get("decision") or "") == "accept":
                record = normalizer(
                    dict(adjudication.get("record") or {}),
                    target=target,
                    repo_root=repo_root,
                )
                return (
                    record,
                    sorted(
                        {
                            str(row.get("reviewer_id") or "") for row in reviews
                        }
                        | {str(adjudication.get("adjudicator_id") or "")}
                    ),
                    "independent_reviews_plus_adjudication",
                    "accepted",
                )
            if str(adjudication.get("decision") or "") == "reject":
                return None, [], "independent_reviews_plus_adjudication", "rejected"
    if len([row for row in reviews if row.get("decision") == "reject"]) >= 2:
        return None, [], "two_rejects", "rejected"
    return None, [], "insufficient_consensus", "in_progress" if reviews else "not_started"


def materialize_paper_review_states(
    papers: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    review_decisions: dict[str, Any],
) -> dict[str, str]:
    """Resolve dual human paper-scope decisions without promoting AI triage."""

    paper_by_id = {str(row["paper_id"]): row for row in papers}
    target_ids_by_paper: dict[str, set[str]] = {}
    for target in targets:
        target_ids_by_paper.setdefault(str(target["paper_id"]), set()).add(
            str(target["target_slot_id"])
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    keys: set[tuple[str, str]] = set()
    allowed = {"primary", "conditional", "control", "exclude", "needs_revision"}
    for raw in review_decisions.get("paper_reviews") or []:
        row = dict(raw)
        paper_id = str(row.get("paper_id") or "")
        reviewer_id = str(row.get("reviewer_id") or "")
        decision = str(row.get("decision") or "")
        if paper_id not in paper_by_id or not reviewer_id or decision not in allowed:
            raise RuntimeError("human_paper_review_identity_invalid")
        if not str(row.get("reviewed_at") or "") or row.get("reviewer_attestation") is not True:
            raise RuntimeError("human_paper_review_attestation_missing")
        key = (paper_id, reviewer_id)
        if key in keys:
            raise RuntimeError("human_paper_review_duplicate_reviewer")
        keys.add(key)
        target_slot_ids = sorted(
            {str(value) for value in row.get("target_slot_ids") or [] if str(value)}
        )
        if decision in {"primary", "conditional", "control"}:
            if not target_slot_ids or not set(target_slot_ids).issubset(
                target_ids_by_paper.get(paper_id, set())
            ):
                raise RuntimeError("human_paper_review_target_enumeration_invalid")
        if not list(row.get("evidence_locators") or []):
            raise RuntimeError("human_paper_review_evidence_locator_missing")
        row["target_slot_ids"] = target_slot_ids
        grouped.setdefault(paper_id, []).append(row)

    adjudications: dict[str, dict[str, Any]] = {}
    for raw in review_decisions.get("adjudications") or []:
        row = dict(raw)
        if str(row.get("subject_type") or "") != "paper":
            continue
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in paper_by_id or paper_id in adjudications:
            raise RuntimeError("human_paper_review_adjudication_identity_invalid")
        if (
            not str(row.get("adjudicator_id") or "")
            or not str(row.get("reviewed_at") or "")
            or row.get("reviewer_attestation") is not True
            or str(row.get("decision") or "") not in allowed - {"needs_revision"}
        ):
            raise RuntimeError("human_paper_review_adjudication_incomplete")
        target_slot_ids = sorted(
            {str(value) for value in row.get("target_slot_ids") or [] if str(value)}
        )
        if str(row["decision"]) in {"primary", "conditional", "control"} and (
            not target_slot_ids
            or not set(target_slot_ids).issubset(target_ids_by_paper.get(paper_id, set()))
        ):
            raise RuntimeError("human_paper_review_adjudication_targets_invalid")
        if not list(row.get("evidence_locators") or []):
            raise RuntimeError("human_paper_review_adjudication_locator_missing")
        row["target_slot_ids"] = target_slot_ids
        adjudications[paper_id] = row

    states: dict[str, str] = {}
    for paper_id, paper in paper_by_id.items():
        reviews = grouped.get(paper_id, [])
        candidates: dict[tuple[str, tuple[str, ...]], set[str]] = {}
        for row in reviews:
            if row["decision"] == "needs_revision":
                continue
            key = (str(row["decision"]), tuple(row["target_slot_ids"]))
            candidates.setdefault(key, set()).add(str(row["reviewer_id"]))
        consensus = [key for key, reviewers in candidates.items() if len(reviewers) >= 2]
        if len(consensus) > 1:
            raise RuntimeError("human_paper_review_multiple_consensus_decisions")
        if consensus:
            state = consensus[0][0]
        elif len({str(row["reviewer_id"]) for row in reviews}) >= 2 and paper_id in adjudications:
            state = str(adjudications[paper_id]["decision"])
        elif reviews:
            state = "in_progress"
        else:
            state = "not_started"
        states[paper_id] = state
        paper["human_review_status"] = state
    for target in targets:
        target["paper_review_status"] = states[str(target["paper_id"])]
    return states


def materialize_human_admissions(
    targets: list[dict[str, Any]],
    review_decisions: dict[str, Any],
    *,
    repo_root: Path,
    paper_review_states: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile signed dual review into derived structure and route registries."""

    target_by_id = {str(row["target_slot_id"]): row for row in targets}
    target_ids = set(target_by_id)
    reviews_by_subject: dict[str, dict[str, list[dict[str, Any]]]] = {
        "structure": {},
        "route": {},
    }
    review_keys: set[tuple[str, str, str]] = set()
    for subject_type, field in (
        ("structure", "structure_reviews"),
        ("route", "route_reviews"),
    ):
        for raw in review_decisions.get(field) or []:
            row = dict(raw)
            target_id, reviewer_id, _ = _review_row_identity(
                row,
                target_ids=target_ids,
                reviewer_field="reviewer_id",
            )
            key = (subject_type, target_id, reviewer_id)
            if key in review_keys:
                raise RuntimeError("human_review_duplicate_reviewer_target")
            review_keys.add(key)
            reviews_by_subject[subject_type].setdefault(target_id, []).append(row)

    adjudications_by_target: dict[str, list[dict[str, Any]]] = {}
    adjudication_keys: set[tuple[str, str]] = set()
    for raw in review_decisions.get("adjudications") or []:
        row = dict(raw)
        subject_type = str(row.get("subject_type") or "")
        if subject_type == "paper":
            continue
        if subject_type not in {"structure", "route"}:
            raise RuntimeError("human_review_adjudication_subject_invalid")
        target_id, adjudicator_id, _ = _review_row_identity(
            row,
            target_ids=target_ids,
            reviewer_field="adjudicator_id",
        )
        key = (subject_type, target_id)
        if key in adjudication_keys:
            raise RuntimeError("human_review_duplicate_adjudication")
        adjudication_keys.add(key)
        row["adjudicator_id"] = adjudicator_id
        adjudications_by_target.setdefault(target_id, []).append(row)

    structures: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    for target_id, target in target_by_id.items():
        structure, structure_reviewers, structure_basis, structure_state = (
            _select_human_admission(
                subject_type="structure",
                target=target,
                reviews=reviews_by_subject["structure"].get(target_id, []),
                adjudications=adjudications_by_target.get(target_id, []),
                normalizer=_normalize_structure_review_record,
                repo_root=repo_root,
            )
        )
        route, route_reviewers, route_basis, route_state = _select_human_admission(
            subject_type="route",
            target=target,
            reviews=reviews_by_subject["route"].get(target_id, []),
            adjudications=adjudications_by_target.get(target_id, []),
            normalizer=_normalize_route_review_record,
            repo_root=repo_root,
        )
        if structure is not None:
            structure_record = {
                "schema_version": "recent_total_synthesis_structure_record.v1",
                "target_slot_id": target_id,
                **structure,
                "reviewer_ids": structure_reviewers,
                "admission_basis": structure_basis,
            }
            structure_record["content_sha256"] = _review_payload_digest(
                structure_record
            )
            structures.append(structure_record)
            target["target_smiles"] = structure["isomeric_smiles"]
            target["structure_status"] = "verified_source_concordant"
        elif structure_state == "rejected":
            target["structure_status"] = "rejected_by_human_review"
        elif structure_state == "in_progress":
            target["structure_status"] = "human_review_in_progress"

        if route is not None:
            route_record = {
                "schema_version": "recent_total_synthesis_route_record.v1",
                "target_slot_id": target_id,
                **route,
                "reviewer_ids": route_reviewers,
                "admission_basis": route_basis,
            }
            route_record["content_sha256"] = _review_payload_digest(route_record)
            routes.append(route_record)
            target["route_evidence_status"] = "verified_route_or_key_step"
        elif route_state == "rejected":
            target["route_evidence_status"] = "rejected_by_human_review"
        elif route_state == "in_progress":
            target["route_evidence_status"] = "human_review_in_progress"

        target["runnable"] = bool(
            (paper_review_states or {}).get(str(target["paper_id"])) == "primary"
            and target["slot_class"] == "primary"
            and target["structure_status"] == "verified_source_concordant"
            and target["route_evidence_status"] == "verified_route_or_key_step"
        )
    return structures, routes


def source_access_class(row: dict[str, Any]) -> str:
    if row.get("repository_fulltext"):
        return "repository_fulltext"
    if row.get("open_access"):
        return "open_access_landing"
    if int(row.get("fulltext_link_count") or 0) > 0:
        return "publisher_fulltext_link"
    if row.get("abstract"):
        return "abstract_and_metadata"
    return "metadata_only"


def review_priority(row: dict[str, Any]) -> tuple[str, int, list[str]]:
    reasons: list[str] = []
    status = row["automated_status"]
    if row["curation_status"] == "admitted_metadata_primary":
        tier, score = "P0_source_extraction", 100
        reasons.append("manually curated P0 primary seed")
    elif status == "high_priority_primary_candidate":
        tier, score = "P1_scope_review", 70
        reasons.append("completed-synthesis title signal")
    elif status == "high_priority_omission_candidate":
        tier, score = "P1_scope_review", 68
        reasons.append("nonstandard completed-route wording")
    elif status.startswith("scope_review"):
        tier, score = "P2_scope_boundary", 40
        reasons.append("conditional scope signal")
    elif status == "manual_title_review":
        tier, score = "P3_omission_audit", 20
        reasons.append("broad discovery match")
    else:
        tier, score = "excluded_or_resolved", 0
        reasons.append(status)

    access = source_access_class(row)
    access_bonus = {
        "repository_fulltext": 15,
        "open_access_landing": 10,
        "publisher_fulltext_link": 6,
        "abstract_and_metadata": 3,
        "metadata_only": 0,
    }[access]
    score += access_bonus
    if access_bonus:
        reasons.append(access)
    if row.get("title_target_phrase"):
        score += 4
        reasons.append("target phrase extractable from title")
    if len(row.get("providers") or []) >= 2:
        score += 2
        reasons.append("multi-index confirmation")
    if row.get("after_strict_model_cutoff"):
        score += 3
        reasons.append("strict post-cutoff")
    return tier, score, reasons


def first_pass_scope_status(row: dict[str, Any], tier: str) -> str:
    """Prioritize human review without converting automation into admission."""

    if tier == "P0_source_extraction":
        return "human_admitted_source_evidence_pending"
    if tier == "P1_scope_review":
        if row["automated_status"] == "high_priority_omission_candidate":
            return "nonstandard_route_title_needs_scope_confirmation"
        abstract = str(row.get("abstract") or "")
        if not abstract:
            return "title_only_source_required"
        if COMPLETION_CUE_RE.search(abstract):
            return "likely_completed_route_needs_dual_review"
        return "abstract_scope_review_required"
    return "conditional_or_control_boundary_review"


def has_completed_route_title_signal(row: dict[str, Any]) -> bool:
    title = str(row.get("title") or "")
    return bool(
        COMPLETED_RE.search(title)
        or FORMAL_SYNTHESIS_RE.search(title)
        or OMISSION_ROUTE_RE.search(title)
        or ROUTE_IMPROVEMENT_RE.search(title)
        or METHOD_APPLICATION_TOTAL_RE.search(title)
    )


def build_paper_review_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for row in rows:
        if not row["preferred_family_record"]:
            continue
        if row["curation_status"] not in {"unreviewed", "admitted_metadata_primary"}:
            continue
        # The strict post-cutoff slice governs new candidate review, not evidence
        # extraction for papers already admitted to the frozen parent cohort.
        if row["curation_status"] == "unreviewed" and not row["after_strict_model_cutoff"]:
            continue
        tier, score, reasons = review_priority(row)
        if tier in {"excluded_or_resolved", "P3_omission_audit"}:
            continue
        if tier == "P2_scope_boundary" and not has_completed_route_title_signal(row):
            continue
        scope_status = first_pass_scope_status(row, tier)
        if scope_status == "likely_completed_route_needs_dual_review":
            score += 5
            reasons.append("explicit completion cue in abstract")
        elif scope_status == "title_only_source_required":
            score -= 3
            reasons.append("title-only record")
        queue.append(
            {
                "paper_id": row["paper_id"],
                "article_family_id": row["article_family_id"],
                "doi": row["doi"],
                "title": row["title"],
                "journal": row["journal"],
                "publication_date": row["publication_date"],
                "first_author": row["first_author"],
                "pmid": row.get("pmid", ""),
                "pmcid": row.get("pmcid", ""),
                "source_url": row["source_url"],
                "oa_landing_url": row.get("oa_landing_url", ""),
                "oa_pdf_url": row.get("oa_pdf_url", ""),
                "fulltext_links": row.get("fulltext_links", []),
                "providers": row["providers"],
                "source_access_class": source_access_class(row),
                "open_access": row["open_access"],
                "repository_fulltext": row["repository_fulltext"],
                "has_abstract": bool(row["abstract"]),
                "title_target_phrase": row["title_target_phrase"],
                "curation_status": row["curation_status"],
                "automated_status": row["automated_status"],
                "review_tier": tier,
                "review_priority_score": score,
                "review_priority_reasons": reasons,
                "first_pass_scope_status": scope_status,
                "automated_screening_is_admission": False,
                "source_acquisition_status": "not_started",
                "required_next_action": (
                    "extract_source_concordant_targets_and_route"
                    if tier == "P0_source_extraction"
                    else "independent_scope_and_target_count_review"
                ),
            }
        )
    queue.sort(
        key=lambda row: (
            -int(row["review_priority_score"]),
            row["publication_date"],
            row["doi"],
        )
    )
    for rank, row in enumerate(queue, start=1):
        row["review_rank"] = rank
    return queue


def build_work_items(
    paper_queue: list[dict[str, Any]], target_slots: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    work: list[dict[str, Any]] = []
    for row in paper_queue:
        task_type = (
            "paper_source_and_route_extraction"
            if row["review_tier"] == "P0_source_extraction"
            else "paper_scope_review"
        )
        work.append(
            {
                "work_item_id": stable_id("work", task_type, row["paper_id"]),
                "task_type": task_type,
                "paper_id": row["paper_id"],
                "target_slot_id": "",
                "priority": row["review_priority_score"],
                "status": (
                    "source_acquired_route_extraction_pending"
                    if row.get("source_acquisition_status") == "source_package_partially_acquired"
                    else "pending"
                ),
                "blocking_requirement": row["required_next_action"],
                "source_access_class": row["source_access_class"],
                "source_url": row["source_url"],
            }
        )
    for row in target_slots:
        if row["slot_class"] != "primary":
            continue
        work.append(
            {
                "work_item_id": stable_id(
                    "work", "target_structure_and_route", row["target_slot_id"]
                ),
                "task_type": "target_structure_and_route",
                "paper_id": row["paper_id"],
                "target_slot_id": row["target_slot_id"],
                "priority": 120,
                "status": "pending",
                "blocking_requirement": (
                    "verify source-concordant isomeric SMILES and route/key-step evidence"
                ),
                "source_access_class": "resolve_from_parent_paper",
                "source_url": row["source_url"],
            }
        )
    work.sort(key=lambda row: (-int(row["priority"]), row["task_type"], row["work_item_id"]))
    return work


def validate_dataset(
    papers: list[dict[str, Any]],
    target_slots: list[dict[str, Any]],
    paper_queue: list[dict[str, Any]],
    manual_record_count: int,
    source_receipts: list[dict[str, Any]],
    source_packages: list[dict[str, Any]],
    authorized_fetch_attempts: list[dict[str, Any]],
    screening_annotations: list[dict[str, Any]],
    review_decisions: dict[str, Any],
    structure_candidates: list[dict[str, Any]],
    route_candidates: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    paper_ids = [row["paper_id"] for row in papers]
    doi_rows = [row["doi"] for row in papers if row["doi"]]
    family_counts = Counter(row["article_family_id"] for row in papers)
    preferred_counts = Counter(
        row["article_family_id"] for row in papers if row["preferred_family_record"]
    )
    target_ids = [row["target_slot_id"] for row in target_slots]
    manual_rows = [row for row in papers if row["curation_status"] != "unreviewed"]
    p0_paper_ids = {
        row["paper_id"] for row in paper_queue if row["review_tier"] == "P0_source_extraction"
    }
    source_package_ids = [row.get("paper_id", "") for row in source_packages]
    authorized_fetch_paper_ids = [
        str(row.get("paper_id") or "") for row in authorized_fetch_attempts
    ]
    annotation_dois = [normalize_doi(row.get("doi")) for row in screening_annotations]
    known_dois = {row["doi"] for row in papers if row["doi"]}
    paper_review_rows = list(review_decisions.get("paper_reviews") or [])
    paper_review_keys = [
        (str(row.get("paper_id") or ""), str(row.get("reviewer_id") or ""))
        for row in paper_review_rows
    ]
    known_paper_ids = set(paper_ids)
    primary_target_ids = {
        row["target_slot_id"] for row in target_slots if row["slot_class"] == "primary"
    }
    candidate_target_ids = [str(row.get("target_slot_id") or "") for row in structure_candidates]
    route_candidate_target_ids = [str(row.get("target_slot_id") or "") for row in route_candidates]
    source_artifacts_valid = True
    for package in source_packages:
        for artifact in package.get("artifacts") or []:
            path = repo_root / str(artifact.get("cache_path") or "")
            if (
                not path.is_file()
                or sha256(path) != artifact.get("sha256")
                or path.stat().st_size != int(artifact.get("size_bytes") or -1)
            ):
                source_artifacts_valid = False
    structure_candidate_cache_valid = True
    for candidate in structure_candidates:
        cache_path = str(candidate.get("cache_path") or "")
        if not cache_path:
            continue
        path = repo_root / cache_path
        if not path.is_file() or sha256(path) != candidate.get("cache_sha256"):
            structure_candidate_cache_valid = False
    route_candidate_sources_valid = True
    for candidate in route_candidates:
        bound_sources = list(candidate.get("source_artifacts") or [])
        if not bound_sources:
            bound_sources = [candidate]
        for source in bound_sources:
            source_path = repo_root / str(source.get("source_artifact_path") or "")
            if not source_path.is_file() or sha256(source_path) != source.get(
                "source_artifact_sha256"
            ):
                route_candidate_sources_valid = False
    checks = {
        "paper_ids_unique": len(paper_ids) == len(set(paper_ids)),
        "doi_rows_unique": len(doi_rows) == len(set(doi_rows)),
        "one_preferred_record_per_family": all(
            preferred_counts[family_id] == 1 for family_id in family_counts
        ),
        "target_slot_ids_unique": len(target_ids) == len(set(target_ids)),
        "manual_seed_rows_recovered": len(manual_rows) == manual_record_count,
        "no_runnable_target_missing_structure_or_route": all(
            (not row["runnable"])
            or (
                bool(row["target_smiles"])
                and row["structure_status"] == "verified_source_concordant"
                and row["route_evidence_status"] == "verified_route_or_key_step"
            )
            for row in target_slots
        ),
        "paper_queue_ranks_unique": len(paper_queue)
        == len({row["review_rank"] for row in paper_queue}),
        "paper_queue_only_preferred_and_in_scope": all(
            next(
                row["preferred_family_record"]
                and (
                    row["after_strict_model_cutoff"]
                    or row["curation_status"] == "admitted_metadata_primary"
                )
                for row in papers
                if row["paper_id"] == queued["paper_id"]
            )
            for queued in paper_queue
        ),
        "all_admitted_primary_papers_have_source_tasks": {
            row["paper_id"]
            for row in papers
            if row["curation_status"] == "admitted_metadata_primary"
            and row["preferred_family_record"]
        }.issubset(
            {row["paper_id"] for row in paper_queue if row["review_tier"] == "P0_source_extraction"}
        ),
        "complete_enumeration_queries_not_truncated": all(
            row["retrieval_complete"]
            for row in source_receipts
            if row["retrieval_mode"] != "top_k_relevance_sample"
        ),
        "incomplete_queries_are_declared_supplemental": all(
            row["retrieval_complete"] or row["retrieval_mode"] == "top_k_relevance_sample"
            for row in source_receipts
        ),
        "source_package_receipts_consistent": (
            not source_packages
            or (
                len(source_package_ids) == len(set(source_package_ids))
                and set(source_package_ids) == p0_paper_ids
                and source_artifacts_valid
            )
        ),
        "authorized_fetch_audit_is_unique_and_p0_scoped": (
            not authorized_fetch_attempts
            or (
                len(authorized_fetch_paper_ids) == len(set(authorized_fetch_paper_ids))
                and set(authorized_fetch_paper_ids) == p0_paper_ids
                and all(
                    row.get("schema_version") == "recent_total_synthesis_authorized_fetch_batch.v1"
                    and isinstance(row.get("accepted"), bool)
                    for row in authorized_fetch_attempts
                )
            )
        ),
        "screening_annotations_are_unique_known_nonadmission_rows": (
            len(annotation_dois) == len(set(annotation_dois))
            and set(annotation_dois).issubset(known_dois)
            and all(str(row.get("preliminary_disposition") or "") for row in screening_annotations)
        ),
        "persistent_review_ledger_is_well_formed": (
            review_decisions.get("schema_version") == "recent_total_synthesis_review_decisions.v1"
            and all(
                isinstance(review_decisions.get(field), list)
                for field in (
                    "paper_reviews",
                    "structure_reviews",
                    "route_reviews",
                    "adjudications",
                )
            )
            and len(paper_review_keys) == len(set(paper_review_keys))
            and all(
                paper_id in known_paper_ids and bool(reviewer_id)
                for paper_id, reviewer_id in paper_review_keys
            )
        ),
        "structure_resolution_candidates_are_nonadmitting_and_complete": (
            not structure_candidates
            or (
                len(candidate_target_ids) == len(set(candidate_target_ids))
                and set(candidate_target_ids) == primary_target_ids
                and all(
                    row.get("admission_authority") is False
                    and row.get("source_concordance_checked") is False
                    and row.get("stereochemistry_checked_against_paper") is False
                    for row in structure_candidates
                )
                and structure_candidate_cache_valid
            )
        ),
        "route_evidence_candidates_are_nonadmitting_and_source_bound": (
            not route_candidates
            or (
                len(route_candidate_target_ids) == len(set(route_candidate_target_ids))
                and set(route_candidate_target_ids).issubset(primary_target_ids)
                and all(
                    row.get("admission_authority") is False
                    and row.get("route_or_key_step_admitted") is False
                    for row in route_candidates
                )
                and route_candidate_sources_valid
            )
        ),
    }
    return {
        "schema_version": "recent_total_synthesis_quality_report.v1",
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "paper_rows": len(papers),
            "article_families": len(family_counts),
            "duplicate_family_records": sum(count - 1 for count in family_counts.values()),
            "target_slots": len(target_slots),
            "paper_queue_rows": len(paper_queue),
            "manual_seed_rows_recovered": len(manual_rows),
            "source_query_groups": len(
                {(row["provider"], row["query_id"]) for row in source_receipts}
            ),
            "incomplete_supplemental_query_groups": len(
                {
                    (row["provider"], row["query_id"])
                    for row in source_receipts
                    if not row["retrieval_complete"]
                }
            ),
            "source_package_receipts": len(source_packages),
            "source_packages_acquired": sum(
                bool(row.get("source_package_acquired")) for row in source_packages
            ),
            "authorized_fetch_attempts": len(authorized_fetch_attempts),
            "authorized_fetch_attempts_accepted": sum(
                bool(row.get("accepted")) for row in authorized_fetch_attempts
            ),
            "preliminary_screening_annotations": len(screening_annotations),
            "submitted_paper_reviews": len(paper_review_rows),
            "structure_resolution_candidate_rows": len(structure_candidates),
            "pubchem_candidates_found_unverified": sum(
                row.get("lookup_status") == "candidate_found_unverified"
                for row in structure_candidates
            ),
            "route_evidence_candidate_rows": len(route_candidates),
            "route_evidence_candidates_with_passages": sum(
                bool(row.get("evidence_passages")) for row in route_candidates
            ),
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_cache_receipts(cache_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem
        if name.startswith("crossref-"):
            provider = "crossref"
            query_id = name.removeprefix("crossref-")
            page_index = 0
            record_count = len((payload.get("message") or {}).get("items") or [])
            provider_total = (payload.get("message") or {}).get("total-results")
        elif name.startswith("openalex-"):
            provider = "openalex"
            query_id = re.sub(r"-\d{3}$", "", name.removeprefix("openalex-"))
            page_match = re.search(r"-(\d{3})$", name)
            page_index = int(page_match.group(1)) if page_match else 0
            record_count = len(payload.get("results") or [])
            provider_total = (payload.get("meta") or {}).get("count")
        elif name.startswith("europe-pmc-"):
            provider = "europe_pmc"
            query_id = name.removeprefix("europe-pmc-")
            page_index = 0
            record_count = len(((payload.get("resultList") or {}).get("result") or []))
            provider_total = payload.get("hitCount")
        else:
            continue
        receipts.append(
            {
                "provider": provider,
                "query_id": query_id,
                "query_term": QUERY_TERMS.get(query_id, ""),
                "page_index": page_index,
                "cache_path": str(path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": sha256(path),
                "cache_record_count": record_count,
                "provider_reported_total": provider_total,
            }
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in receipts:
        grouped.setdefault((row["provider"], row["query_id"]), []).append(row)
    for (provider, _query_id), group in grouped.items():
        returned = sum(int(row["cache_record_count"]) for row in group)
        totals = [
            int(row["provider_reported_total"])
            for row in group
            if row["provider_reported_total"] is not None
        ]
        provider_total = max(totals) if totals else returned
        mode = {
            "crossref": "top_k_relevance_sample",
            "openalex": "cursor_paginated_enumeration",
            "europe_pmc": "single_page_enumeration",
        }[provider]
        complete = returned >= provider_total
        for row in group:
            row["query_returned_record_count"] = returned
            row["query_provider_reported_total"] = provider_total
            row["retrieval_mode"] = mode
            row["retrieval_complete"] = complete
    return receipts


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    cache_dir = (repo_root / args.cache_dir).resolve()
    manual_path = (repo_root / args.manual_seed).resolve()

    records: list[dict[str, Any]] = []
    records.extend(crossref_records(cache_dir, offline=args.offline))
    records.extend(openalex_records(cache_dir, offline=args.offline))
    records.extend(europe_pmc_records(cache_dir, offline=args.offline))
    papers = merge_records(records)
    manual, manual_payload = load_manual_seed(manual_path)
    annotate_rows(papers, manual)
    targets = build_target_slots(papers)

    preferred = [row for row in papers if row["preferred_family_record"]]
    post_cutoff = [row for row in preferred if row["after_strict_model_cutoff"]]
    admitted_primary = [
        row for row in preferred if row["curation_status"] == "admitted_metadata_primary"
    ]
    admitted_conditional = [
        row for row in preferred if row["curation_status"] == "admitted_metadata_conditional"
    ]
    high_priority_unreviewed = [
        row
        for row in post_cutoff
        if row["curation_status"] == "unreviewed"
        and row["automated_status"]
        in {"high_priority_primary_candidate", "high_priority_omission_candidate"}
    ]
    source_package_path = output_dir / "source_package_receipts.jsonl"
    source_packages = read_jsonl(source_package_path)
    authorized_fetch_batch_path = output_dir / "authorized_source_fetch_batch.jsonl"
    authorized_fetch_attempts = read_jsonl(authorized_fetch_batch_path)
    structure_candidate_path = output_dir / "structure_resolution_candidates.jsonl"
    structure_candidates = read_jsonl(structure_candidate_path)
    visual_candidate_path = output_dir / "visual_structure_candidates.jsonl"
    visual_candidates = read_jsonl(visual_candidate_path)
    visual_candidate_summary_path = output_dir / "visual_structure_candidates.summary.json"
    visual_candidate_summary = (
        json.loads(visual_candidate_summary_path.read_text(encoding="utf-8"))
        if visual_candidate_summary_path.exists()
        else {}
    )
    route_candidate_path = output_dir / "route_evidence_candidates.jsonl"
    route_candidates = read_jsonl(route_candidate_path)
    p1_source_package_path = output_dir / "p1_source_package_receipts.jsonl"
    p1_source_packages = read_jsonl(p1_source_package_path)
    p1_authorized_fetch_batch_path = output_dir / "p1_authorized_source_fetch_batch.jsonl"
    p1_authorized_fetch_attempts = read_jsonl(p1_authorized_fetch_batch_path)
    p1_candidate_root = output_dir / "curation_candidates" / "p1_scope"
    p1_scope_consensus_path = p1_candidate_root / "scope-consensus.jsonl"
    p1_scope_consensus = read_jsonl(p1_scope_consensus_path)
    p1_scope_disagreements_path = p1_candidate_root / "scope-disagreements.jsonl"
    p1_scope_disagreements = read_jsonl(p1_scope_disagreements_path)
    p1_candidate_slots_path = p1_candidate_root / "candidate-target-slots.jsonl"
    p1_candidate_slots = read_jsonl(p1_candidate_slots_path)
    p1_structure_candidates_path = p1_candidate_root / "structure-resolution-candidates.jsonl"
    p1_structure_candidates = read_jsonl(p1_structure_candidates_path)
    p1_visual_candidates_path = p1_candidate_root / "visual-structure-candidates.jsonl"
    p1_visual_candidates = read_jsonl(p1_visual_candidates_path)
    p1_visual_summary_path = p1_candidate_root / "visual-structure-candidates.summary.json"
    p1_visual_summary = (
        json.loads(p1_visual_summary_path.read_text(encoding="utf-8"))
        if p1_visual_summary_path.exists()
        else {}
    )
    p1_route_candidates_path = p1_candidate_root / "route-evidence-candidates.jsonl"
    p1_route_candidates = read_jsonl(p1_route_candidates_path)
    screening_annotations_path = output_dir / "curation_inputs" / "screening_annotations.json"
    screening_annotations_payload = (
        json.loads(screening_annotations_path.read_text(encoding="utf-8"))
        if screening_annotations_path.exists()
        else {"admission_authority": False, "records": []}
    )
    if screening_annotations_payload.get("admission_authority") is not False:
        raise RuntimeError("preliminary screening annotations cannot grant admission")
    screening_annotations = list(screening_annotations_payload.get("records") or [])
    review_decisions_path = output_dir / "curation_inputs" / "review_decisions.json"
    review_decisions = (
        json.loads(review_decisions_path.read_text(encoding="utf-8"))
        if review_decisions_path.exists()
        else {
            "schema_version": "recent_total_synthesis_review_decisions.v1",
            "paper_reviews": [],
            "structure_reviews": [],
            "route_reviews": [],
            "adjudications": [],
        }
    )
    paper_review_states = materialize_paper_review_states(
        papers,
        targets,
        review_decisions,
    )
    admitted_structures, admitted_routes = materialize_human_admissions(
        targets,
        review_decisions,
        repo_root=repo_root,
        paper_review_states=paper_review_states,
    )
    target_counts_by_paper: dict[str, Counter[str]] = {}
    for target in targets:
        counts = target_counts_by_paper.setdefault(
            str(target["paper_id"]),
            Counter(),
        )
        counts["structures"] += int(
            target["structure_status"] == "verified_source_concordant"
        )
        counts["routes"] += int(
            target["route_evidence_status"] == "verified_route_or_key_step"
        )
        counts["runnable"] += int(bool(target["runnable"]))
    for paper in papers:
        counts = target_counts_by_paper.get(str(paper["paper_id"]), Counter())
        paper["source_concordant_structure_count"] = counts["structures"]
        paper["key_step_evidence_count"] = counts["routes"]
        paper["runnable_target_count"] = counts["runnable"]
    source_packages_by_paper = {row["paper_id"]: row for row in source_packages}
    paper_queue = build_paper_review_queue(papers)
    paper_review_counts = Counter(
        str(row.get("paper_id") or "") for row in review_decisions.get("paper_reviews") or []
    )
    for row in paper_queue:
        submitted_reviews = int(paper_review_counts[row["paper_id"]])
        row["submitted_paper_review_count"] = submitted_reviews
        row["review_state"] = "not_started" if submitted_reviews == 0 else "review_in_progress"
        package = source_packages_by_paper.get(row["paper_id"])
        if package:
            row["source_acquisition_status"] = package["status"]
            row["source_package_acquired"] = bool(package["source_package_acquired"])
    work_items = build_work_items(paper_queue, targets)
    target_rows_by_id = {str(row["target_slot_id"]): row for row in targets}
    visual_candidates_by_target = {
        str(row.get("target_slot_id") or ""): row for row in visual_candidates
    }
    route_candidates_by_target = {
        str(row.get("target_slot_id") or ""): row for row in route_candidates
    }
    for item in work_items:
        target_id = str(item.get("target_slot_id") or "")
        if not target_id:
            continue
        target = target_rows_by_id[target_id]
        visual = visual_candidates_by_target.get(target_id, {})
        route_lead = route_candidates_by_target.get(target_id, {})
        visual_status = str(visual.get("visual_status") or "unresolved")
        rdkit_valid = (
            (visual.get("rdkit_validation") or {}).get("status")
            == "roundtrip_valid"
        )
        route_passages = len(route_lead.get("evidence_passages") or [])
        if target["runnable"]:
            packet_status = "admitted_runnable"
        elif rdkit_valid and route_passages and visual_status == "exact_source_structure_candidate":
            packet_status = "dual_review_ready_exact_source"
        elif rdkit_valid and route_passages:
            packet_status = "dual_review_ready_stereochemistry_resolution"
        elif not rdkit_valid:
            packet_status = "structure_resolution_required"
        else:
            packet_status = "route_source_reconstruction_required"
        item.update(
            {
                "review_packet_status": packet_status,
                "visual_structure_candidate_status": visual_status,
                "visual_structure_rdkit_valid": rdkit_valid,
                "route_evidence_passage_count": route_passages,
                "structure_review_status": target["structure_status"],
                "route_review_status": target["route_evidence_status"],
                "projection_grants_admission": False,
            }
        )
    cache_receipts = build_cache_receipts(cache_dir, repo_root)
    quality = validate_dataset(
        papers,
        targets,
        paper_queue,
        len(manual),
        cache_receipts,
        source_packages,
        authorized_fetch_attempts,
        screening_annotations,
        review_decisions,
        structure_candidates,
        route_candidates,
        repo_root,
    )
    admitted_structure_target_ids = [
        str(row["target_slot_id"]) for row in admitted_structures
    ]
    admitted_route_target_ids = [str(row["target_slot_id"]) for row in admitted_routes]
    target_by_id = {str(row["target_slot_id"]): row for row in targets}
    quality["checks"].update(
        {
            "admitted_structures_are_unique_review_derived_targets": (
                len(admitted_structure_target_ids)
                == len(set(admitted_structure_target_ids))
                and all(
                    target_by_id[target_id]["structure_status"]
                    == "verified_source_concordant"
                    for target_id in admitted_structure_target_ids
                )
            ),
            "admitted_routes_are_unique_review_derived_targets": (
                len(admitted_route_target_ids) == len(set(admitted_route_target_ids))
                and all(
                    target_by_id[target_id]["route_evidence_status"]
                    == "verified_route_or_key_step"
                    for target_id in admitted_route_target_ids
                )
            ),
            "runnable_targets_have_both_admitted_records": all(
                (not row["runnable"])
                or (
                    row["target_slot_id"] in set(admitted_structure_target_ids)
                    and row["target_slot_id"] in set(admitted_route_target_ids)
                )
                for row in targets
            ),
        }
    )
    quality["counts"].update(
        {
            "admitted_source_concordant_structures": len(admitted_structures),
            "admitted_literature_routes_or_key_steps": len(admitted_routes),
            "runnable_primary_targets": sum(
                row["slot_class"] == "primary" and bool(row["runnable"])
                for row in targets
            ),
        }
    )
    p1_queue_ids = {
        str(row.get("paper_id") or "")
        for row in paper_queue
        if row.get("review_tier") == "P1_scope_review"
    }
    p1_nonrepository_ids = {
        str(row.get("paper_id") or "")
        for row in paper_queue
        if row.get("review_tier") == "P1_scope_review"
        and row.get("source_access_class") != "repository_fulltext"
    }
    p1_candidate_slot_ids = [
        str(row.get("target_slot_id") or "") for row in p1_candidate_slots
    ]
    p1_candidate_slot_id_set = set(p1_candidate_slot_ids)
    primary_target_slot_ids = {
        str(row.get("target_slot_id") or "")
        for row in targets
        if row.get("slot_class") == "primary"
    }
    p1_checks = {
        "visual_structure_candidates_are_complete_portable_and_nonadmitting": (
            not visual_candidates
            or (
                {
                    str(row.get("target_slot_id") or "")
                    for row in visual_candidates
                }
                == primary_target_slot_ids
                and len(visual_candidates) == len(primary_target_slot_ids)
                and all(
                    row.get("admission_authority") is False
                    and not Path(
                        str((row.get("source_image") or {}).get("image_path") or "")
                    ).is_absolute()
                    for row in visual_candidates
                )
                and int(visual_candidate_summary.get("target_rows") or 0)
                == len(visual_candidates)
            )
        ),
        "p1_source_receipts_are_unique_and_queue_scoped": (
            not p1_source_packages
            or (
                len(p1_source_packages) == len(p1_queue_ids)
                and len({str(row.get("paper_id") or "") for row in p1_source_packages})
                == len(p1_source_packages)
                and {
                    str(row.get("paper_id") or "") for row in p1_source_packages
                }
                == p1_queue_ids
            )
        ),
        "p1_authorized_fetch_audit_is_unique_and_nonrepository_scoped": (
            not p1_authorized_fetch_attempts
            or (
                len(p1_authorized_fetch_attempts) == len(p1_nonrepository_ids)
                and len(
                    {
                        str(row.get("paper_id") or "")
                        for row in p1_authorized_fetch_attempts
                    }
                )
                == len(p1_authorized_fetch_attempts)
                and {
                    str(row.get("paper_id") or "")
                    for row in p1_authorized_fetch_attempts
                }
                == p1_nonrepository_ids
            )
        ),
        "p1_scope_consensus_is_complete_and_nonadmitting": (
            not p1_scope_consensus
            or (
                len(p1_scope_consensus) == len(p1_queue_ids)
                and len(
                    {str(row.get("paper_id") or "") for row in p1_scope_consensus}
                )
                == len(p1_scope_consensus)
                and all(row.get("admission_authority") is False for row in p1_scope_consensus)
            )
        ),
        "p1_scope_disagreements_are_consensus_subset": (
            not p1_scope_disagreements
            or {
                str(row.get("paper_id") or "") for row in p1_scope_disagreements
            }
            <= {str(row.get("paper_id") or "") for row in p1_scope_consensus}
        ),
        "p1_candidate_target_slots_are_unique_and_nonadmitting": (
            not p1_candidate_slots
            or (
                len(p1_candidate_slot_ids) == len(p1_candidate_slot_id_set)
                and all(
                    row.get("formal_benchmark_eligible") is False
                    for row in p1_candidate_slots
                )
            )
        ),
        "p1_structure_candidates_are_complete_and_nonadmitting": (
            not p1_structure_candidates
            or (
                {
                    str(row.get("target_slot_id") or "")
                    for row in p1_structure_candidates
                }
                == p1_candidate_slot_id_set
                and all(
                    row.get("admission_authority") is False
                    and row.get("source_concordance_checked") is False
                    for row in p1_structure_candidates
                )
            )
        ),
        "p1_visual_structure_candidates_are_complete_portable_and_nonadmitting": (
            not p1_visual_candidates
            or (
                {
                    str(row.get("target_slot_id") or "")
                    for row in p1_visual_candidates
                }
                == p1_candidate_slot_id_set
                and len(p1_visual_candidates) == len(p1_candidate_slot_id_set)
                and all(
                    row.get("admission_authority") is False
                    and not Path(
                        str((row.get("source_image") or {}).get("image_path") or "")
                    ).is_absolute()
                    for row in p1_visual_candidates
                )
                and int(p1_visual_summary.get("target_rows") or 0)
                == len(p1_visual_candidates)
            )
        ),
        "p1_route_candidates_are_source_bound_nonadmitting_subset": (
            not p1_route_candidates
            or (
                {
                    str(row.get("target_slot_id") or "") for row in p1_route_candidates
                }
                <= p1_candidate_slot_id_set
                and all(
                    row.get("admission_authority") is False
                    for row in p1_route_candidates
                )
            )
        ),
    }
    quality["checks"].update(p1_checks)
    quality["counts"].update(
        {
            "p1_source_package_receipts": len(p1_source_packages),
            "p1_source_packages_acquired": sum(
                bool(row.get("source_package_acquired")) for row in p1_source_packages
            ),
            "p1_authorized_fetch_attempts": len(p1_authorized_fetch_attempts),
            "p1_authorized_fetch_attempts_accepted": sum(
                bool(row.get("accepted")) for row in p1_authorized_fetch_attempts
            ),
            "p1_dual_ai_scope_consensus_rows": len(p1_scope_consensus),
            "p1_dual_ai_scope_disagreement_rows": len(p1_scope_disagreements),
            "p1_candidate_target_slots_nonadmitting": len(p1_candidate_slots),
            "p1_structure_candidates_found_unverified": sum(
                row.get("lookup_status") == "candidate_found_unverified"
                for row in p1_structure_candidates
            ),
            "visual_structure_candidate_rows_nonadmitting": len(visual_candidates),
            "visual_structure_candidates_rdkit_valid": sum(
                (row.get("rdkit_validation") or {}).get("status") == "roundtrip_valid"
                for row in visual_candidates
            ),
            "p1_visual_structure_candidate_rows_nonadmitting": len(p1_visual_candidates),
            "p1_visual_structure_candidates_rdkit_valid": sum(
                (row.get("rdkit_validation") or {}).get("status") == "roundtrip_valid"
                for row in p1_visual_candidates
            ),
            "p1_route_evidence_candidate_rows": len(p1_route_candidates),
            "p1_route_evidence_candidates_with_passages": sum(
                bool(row.get("evidence_passages")) for row in p1_route_candidates
            ),
        }
    )
    quality["all_checks_passed"] = all(quality["checks"].values())
    if not quality["all_checks_passed"]:
        raise RuntimeError(f"dataset quality checks failed: {quality['checks']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    papers_path = output_dir / "papers.jsonl"
    targets_path = output_dir / "target_slots.jsonl"
    queue_path = output_dir / "paper_review_queue.jsonl"
    work_path = output_dir / "work_items.jsonl"
    receipts_path = output_dir / "source_query_receipts.jsonl"
    planner_path = output_dir / "planner_targets.jsonl"
    evaluator_path = output_dir / "evaluator_references.jsonl"
    admitted_routes_by_target = {
        str(row["target_slot_id"]): row for row in admitted_routes
    }
    write_jsonl(papers_path, papers)
    write_jsonl(targets_path, targets)
    write_jsonl(queue_path, paper_queue)
    write_jsonl(work_path, work_items)
    write_jsonl(receipts_path, cache_receipts)
    write_jsonl(
        planner_path,
        [
            {
                "target_slot_id": row["target_slot_id"],
                "target_smiles": row["target_smiles"],
            }
            for row in targets
            if row["slot_class"] == "primary" and row["runnable"]
        ],
    )
    write_jsonl(
        evaluator_path,
        [
            {
                "target_slot_id": row["target_slot_id"],
                "paper_id": row["paper_id"],
                "doi": row["doi"],
                "target_name": row["target_name"],
                "route_evidence_status": row["route_evidence_status"],
                "reference_route": admitted_routes_by_target[row["target_slot_id"]],
            }
            for row in targets
            if row["slot_class"] == "primary" and row["runnable"]
        ],
    )
    write_json(
        output_dir / "structures.json",
        {
            "schema_version": "recent_total_synthesis_structures.v1",
            "records": admitted_structures,
            "claim_boundary": (
                "Only source-concordant, stereochemically verified structures may be added."
            ),
        },
    )
    write_json(
        output_dir / "route_evidence.json",
        {
            "schema_version": "recent_total_synthesis_route_evidence.v1",
            "records": admitted_routes,
            "claim_boundary": (
                "LLM extraction is a lead; a human-confirmed article/SI route or key step is required."
            ),
        },
    )
    legacy_cohort_root = repo_root / "benchmarks" / "literature_strategy_rediscovery_v0_1"
    legacy_manifest_path = legacy_cohort_root / "manifest.json"
    legacy_manifest = (
        json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        if legacy_manifest_path.exists()
        else {}
    )
    legacy_counts = legacy_manifest.get("counts") or {}
    write_json(
        output_dir / "external_cohorts.json",
        {
            "schema_version": "recent_total_synthesis_external_cohorts.v1",
            "cohorts": [
                {
                    "cohort_id": "synthex_reported_literature_targets_or_routes",
                    "reported_count": 145,
                    "count_unit": "reported_curated_targets_or_routes",
                    "independent_paper_count": None,
                    "row_level_identifiers_publicly_recovered": False,
                    "source_url": "https://arxiv.org/abs/2608.07454",
                },
                {
                    "cohort_id": "local_human_reviewed_primary_seed",
                    "reported_count": len(admitted_primary),
                    "count_unit": "independent_preferred_article_families",
                    "target_slot_count": sum(
                        row["primary_target_count"] for row in admitted_primary
                    ),
                    "row_level_identifiers_publicly_recovered": True,
                },
                {
                    "cohort_id": "synthatlas_public_target_frontier",
                    "reported_count": legacy_counts.get("synthatlas_frontier_targets", 0),
                    "count_unit": "distinct_target_structures",
                    "human_literature_route_truth_available": False,
                    "intended_use": "blind target-only reach and closure evaluation",
                    "artifact_path": str(
                        (legacy_cohort_root / "target_only.csv").relative_to(repo_root)
                    ).replace("\\", "/"),
                },
                {
                    "cohort_id": "local_post_hoc_positive_pilots",
                    "reported_count": legacy_counts.get("local_verified_cases", 0),
                    "count_unit": "case_studies",
                    "eligible_for_hit_rate_denominator": False,
                    "intended_use": "protocol and visualization checks only",
                    "artifact_path": str(
                        (legacy_cohort_root / "evaluator_only.csv").relative_to(repo_root)
                    ).replace("\\", "/"),
                },
            ],
        },
    )
    write_json(output_dir / "quality_report.json", quality)

    deprecated_queue = output_dir / "high_priority_review_queue.jsonl"
    if deprecated_queue.exists():
        deprecated_queue.unlink()

    provider_counts = Counter(
        provider for row in preferred for provider in row.get("providers", [])
    )
    automated_counts = Counter(row["automated_status"] for row in post_cutoff)
    curation_counts = Counter(row["curation_status"] for row in preferred)
    queue_tier_counts = Counter(row["review_tier"] for row in paper_queue)
    queue_access_counts = Counter(row["source_access_class"] for row in paper_queue)
    queue_scope_counts = Counter(row["first_pass_scope_status"] for row in paper_queue)
    high_priority_non_crossref = sum(
        "crossref" not in row["providers"] for row in high_priority_unreviewed
    )
    p0_candidate_paper_ids = {
        str(row["paper_id"])
        for row in admitted_primary
    }
    p1_candidate_paper_ids = {
        str(row.get("paper_id") or "")
        for row in p1_candidate_slots
        if str(row.get("paper_id") or "")
    }
    acquired_candidate_paper_ids = {
        str(row.get("paper_id") or "")
        for row in (*source_packages, *p1_source_packages)
        if row.get("source_package_acquired") is True
    }
    combined_visual_candidates = [*visual_candidates, *p1_visual_candidates]
    combined_route_candidates = [*route_candidates, *p1_route_candidates]
    p0_route_passage_target_ids = {
        str(row.get("target_slot_id") or "")
        for row in route_candidates
        if row.get("evidence_passages")
    }
    p0_dual_evidence_review_candidates = [
        row
        for row in visual_candidates
        if (row.get("rdkit_validation") or {}).get("status") == "roundtrip_valid"
        and str(row.get("target_slot_id") or "") in p0_route_passage_target_ids
    ]
    runnable_primary_targets = sum(
        1 for row in targets if row["slot_class"] == "primary" and row["runnable"]
    )

    readme = f"""# Recent total-synthesis benchmark

This is a stable, auditable benchmark-construction workspace for recent published
total syntheses. It independently reconstructs a public-source literature universe;
it does not claim access to the unreleased row-level SynthEx literature-145 cohort.

## Frozen scope and current state

- Discovery window: {FREEZE_START} through {FREEZE_END}
- Strict post-model-cutoff slice: after {STRICT_POST_CUTOFF}
- Metadata providers: Crossref, OpenAlex, and Europe PMC
- OpenAlex/Europe PMC query groups are exhaustively enumerated within their APIs;
  Crossref is an explicitly capped top-k recall supplement
- Curated candidate universe: {len(p0_candidate_paper_ids | p1_candidate_paper_ids):,} papers / {len(combined_visual_candidates):,} target slots
- Candidate-universe source packages acquired: {len((p0_candidate_paper_ids | p1_candidate_paper_ids) & acquired_candidate_paper_ids):,}/{len(p0_candidate_paper_ids | p1_candidate_paper_ids):,}
- Combined Codex visual extraction (non-admitting): {sum((row.get("rdkit_validation") or {}).get("status") == "roundtrip_valid" for row in combined_visual_candidates):,} RDKit-valid; {sum(row.get("visual_status") == "exact_source_structure_candidate" for row in combined_visual_candidates):,} exact-source, {sum(row.get("visual_status") == "partial_stereo_candidate" for row in combined_visual_candidates):,} partial-stereochemistry, {sum(row.get("visual_status") == "unresolved" for row in combined_visual_candidates):,} unresolved
- Combined route-evidence leads (non-admitting): {len(combined_route_candidates):,} rows; {sum(bool(row.get("evidence_passages")) for row in combined_route_candidates):,} with candidate passages
- P0 targets ready for dual human structure/route review (still non-admitting): {len(p0_dual_evidence_review_candidates):,}; {sum(row.get("visual_status") == "exact_source_structure_candidate" for row in p0_dual_evidence_review_candidates):,} exact-source and {sum(row.get("visual_status") == "partial_stereo_candidate" for row in p0_dual_evidence_review_candidates):,} partial-stereochemistry
- Preferred DOI/title article families: {len(preferred):,}
- Preferred records after the strict cutoff: {len(post_cutoff):,}
- Curated P0 primary-paper seed: {len(admitted_primary):,}
- Curated P0 primary target slots: {sum(row["primary_target_count"] for row in admitted_primary):,}
- Primary slots with an extracted target name: {sum(bool(row.get("target_name")) for row in targets if row["slot_class"] == "primary"):,}
- PubChem structure leads found (unverified, non-admitting): {sum(row.get("lookup_status") == "candidate_found_unverified" for row in structure_candidates):,}
- P0 Codex visual candidates (non-admitting): {sum((row.get("rdkit_validation") or {}).get("status") == "roundtrip_valid" for row in visual_candidates):,}/{len(visual_candidates):,} RDKit-valid
- Target slots linked to acquired full text (unverified, non-admitting): {len(route_candidates):,}
- Target slots with extracted transformation passages (unverified, non-admitting): {sum(bool(row.get("evidence_passages")) for row in route_candidates):,}
- Primary-paper source packages acquired: {sum(bool(row.get("source_package_acquired")) for row in source_packages):,}/{len(admitted_primary):,}
- High-priority unreviewed post-cutoff papers: {len(high_priority_unreviewed):,}
- P1 papers with an explicit abstract completion cue: {queue_scope_counts.get("likely_completed_route_needs_dual_review", 0):,}
- P1 title-only records requiring source acquisition: {queue_scope_counts.get("title_only_source_required", 0):,}
- P1 dual-AI metadata consensus rows (non-admitting): {len(p1_scope_consensus):,}
- P1 metadata disagreements requiring review: {len(p1_scope_disagreements):,}
- P1 candidate target slots (non-admitting): {len(p1_candidate_slots):,}
- P1 source packages acquired: {sum(bool(row.get("source_package_acquired")) for row in p1_source_packages):,}/{len(p1_source_packages):,}
- P1 PubChem structure leads found (unverified): {sum(row.get("lookup_status") == "candidate_found_unverified" for row in p1_structure_candidates):,}
- P1 Codex visual candidates (non-admitting): {sum((row.get("rdkit_validation") or {}).get("status") == "roundtrip_valid" for row in p1_visual_candidates):,}/{len(p1_visual_candidates):,} RDKit-valid
- P1 route-evidence candidate rows (non-admitting): {len(p1_route_candidates):,}
- AI-assisted preliminary screening annotations (non-admitting): {len(screening_annotations):,}
- High-priority papers missed by Crossref but recovered elsewhere: {high_priority_non_crossref:,}
- Runnable primary targets with both admitted structure and route evidence: {runnable_primary_targets:,}

## Authoritative files

- `papers.jsonl`: immutable provider-merged paper registry.
- `paper_review_queue.jsonl`: ranked paper-level review and source-acquisition queue.
- `target_slots.jsonl`: target slots projected from the curated P0 paper seed.
- `work_items.jsonl`: actionable paper, structure, and route tasks.
- `REVIEW_PROTOCOL.md`: expert-facing input, output, acceptance, blind-review, and
  submission rules.
- `structures.json`: source-concordant structure registry; currently empty by design.
- `route_evidence.json`: human-confirmed route/key-step registry; currently empty.
- `planner_targets.jsonl`: leakage-safe planner input view; currently empty.
- `evaluator_references.jsonl`: withheld evaluator view; currently empty.
- `source_query_receipts.jsonl`: query-cache hashes, retrieval modes, completeness,
  and provider counts.
- `source_package_receipts.jsonl`: openly acquired article artifacts and explicit
  pending/failure states; present after the source-acquisition step.
- `SOURCE_STORAGE.md`: local article/SI cache layout, immutable receipt boundary,
  backup guidance, and the files that intentionally remain outside Git.
- `authorized_source_fetch_batch.jsonl`: one audited authorized-browser acquisition
  attempt per P0 paper; acquisition provenance only, not chemical evidence.
- `structure_resolution_candidates.jsonl`: PubChem name-resolution leads with RDKit
  round-trip checks; explicitly non-admitting and never exposed to the planner.
- `visual_structure_candidates.jsonl`: exact-paper Codex visual transcription
  candidates with source locators/hashes, RDKit postchecks, and PubChem comparison;
  explicitly non-admitting and never exposed to the planner.
- `route_evidence_candidates.jsonl`: source-hashed target-linked article passages;
  these accelerate SI review but are not route truth.
- `curation_candidates/p1_scope/`: dual-AI metadata consensus, disagreements,
  candidate target slots, source/structure/route leads, and the P1 visual projection;
  none has admission authority.
- `quality_report.json`: machine-checkable dataset invariants.
- `manifest.json`: release counts, hashes, sources, and claim boundaries.

## Expert handoff

`python scripts/build_recent_total_synthesis_review_packets.py` creates a local HTML
index, paper-scope packets, target truth packets, RDKit candidate depictions, and one
editable `submission.json` per packet under
`output/recent_total_synthesis_review_packets/`. Validate a returned file with
`python scripts/validate_recent_total_synthesis_review_submission.py --submission
<path>`; only the dataset administrator uses `--merge`.

## Admission rule

A title or abstract is discovery evidence only. A runnable target requires an exact
target identity, dual-review primary-paper admission, source-concordant
stereochemistry, verified isomeric SMILES, primary article/SI provenance, and a
human-confirmed route or strategic key step. Planner inputs are not emitted until all
requirements are satisfied.

## Counting rule

Always report independent paper/article-family counts separately from target-slot
counts. Keep all targets from one collective/divergent synthesis and shared scaffold
family in the same split. Parallel journal editions and duplicate index records are
one article family, not independent papers.
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    source_storage_path = output_dir / "SOURCE_STORAGE.md"

    methods = f"""# Dataset construction and admission protocol

## Scope

The discovery window is frozen from {FREEZE_START} through {FREEZE_END}. The core
cohort contains discrete natural products and similarly complex small molecules with
a completed published synthesis. Peptides, glycans, formal syntheses, route
improvements, and method-only demonstrations remain conditional/control strata unless
the review protocol explicitly admits them.

## Discovery and reconciliation

{len(QUERY_TERMS)} title-query families are run against Crossref, OpenAlex, and
Europe PMC. Raw provider responses are cached and hashed. OpenAlex is cursor-paginated
to exhaustion and Europe PMC is complete where its reported count fits the retrieved
page. Crossref's search semantics return a very broad relevance universe, so its
first 1,000 results per query are retained only as a declared top-k recall supplement,
not as an exhaustive enumeration. Records are merged by normalized DOI;
DOI-less records use normalized title and year. Identical normalized titles form
article families, with the English Angewandte DOI preferred for display while every
source record remains preserved.

## Evidence ladder

1. **Discovered metadata** - at least one provider returned a matching record.
2. **Paper candidate** - general rules assign a review tier; no truth claim is made.
3. **Curated P0 seed paper** - deliberate curation nominates source-extraction
   priorities; this is not yet a frozen benchmark-denominator decision.
4. **Paper-admitted record** - two independent reviewers confirm primary synthesis
   scope and target count, with adjudication for disagreement.
5. **Structure-admitted target** - the source structure and stereochemistry produce a
   verified isomeric SMILES with provenance.
6. **Route-admitted target** - the article/SI yields a reconstructable route or
   human-confirmed strategic key step.
7. **Runnable benchmark target** - both structure and route evidence are frozen and
   planner/evaluator views are generated.

The present release reaches provisional level 3 for {len(admitted_primary):,} papers
and {sum(row["primary_target_count"] for row in admitted_primary):,} primary target
slots. The independent paper-review ledger remains empty, so level 4 is not claimed.
It reports {runnable_primary_targets:,} runnable targets because missing evidence is
never silently imputed.

## Visual structure candidate extraction

Source-concordant structure review is accelerated with at most one independent Codex
visual call per paper. The host prefers the main article over supporting information,
excludes text-only reference pages, ranks body pages by target-name and chemical-
graphics evidence, and renders complete pages at fixed resolution. All primary
targets from the paper are requested together.

The visual subprocess receives only the rendered paper pages and a compact JSON
contract. Shell, code execution, browsing, plugins, external validators, and other
structure-recognition models are disabled. It may return an exact-source candidate,
a partial-stereochemistry candidate, or unresolved; guessing is forbidden. RDKit is
used only after the response to test SMILES parsing and canonical round trip.

Visual candidates retain the paper, page/figure locator, source-artifact hash,
rendered-image hash, model, prompt-input hash, and transcription note. A separate
projection places them beside PubChem name-resolution leads. Neither source has
admission authority; two independent source-image reviews are still required.

## Leakage and splitting

The strict novelty slice contains publications after {STRICT_POST_CUTOFF}. Dates after
{SYNTHEX_CUTOFF_PROXY} are also labeled as a proxy for SynthEx's January 2025 cutoff;
this is not an exact training-data guarantee. DOI, title, target name, literature
strategy, and route evidence remain evaluator-only. Splits are grouped by paper,
article family, and scaffold family.

## Exploratory Strategy rediscovery protocol

The exploratory candidate-coverage runner is not the formal benchmark. It filters
the non-admitting PubChem leads to unique, RDKit-round-trip-valid, conflict-free
candidate structures and invokes the same three-card Strategy Generator followed by
the independent Strategy Critic. Durable task journals bind target structure, exact
Generator and Critic prompts, model, and reasoning effort, so interrupted runs resume
without reusing results from a changed prompt.

The planner input contains only target-slot ID and candidate isomeric SMILES. A
separate evaluator-only pass then receives the frozen cards plus source-hashed paper
passages. `exact` requires the same route-defining scaffold construction or skeletal
reorganization and key transformation/control logic; `partial` requires a substantive
shared strategic element but permits a missing or substituted route-defining event.
Generic plausibility and peripheral overlap count as `none`. Missing or target-
ambiguous route evidence is `non_comparable` and is excluded from the rate
denominator. The resulting labels remain provisional until source-concordant
structures, route admission, and independent chemist review are complete.

## Reproducibility boundary

This release is reproducible with respect to its dated retrieval contract and cached
responses. It is not exhaustive over Crossref's broad relevance-ranked hit universe
or all worldwide chemistry literature. Reaxys, SciFinder, and the unreleased SynthEx
row list are not available, so global exhaustiveness must not be claimed.
"""
    methods_path = output_dir / "METHODS.md"
    methods_path.write_text(methods, encoding="utf-8")

    schema = """# Data contract

## Paper registry

`papers.jsonl` has one row per provider-reconciled DOI record. `article_family_id`
groups parallel editions/duplicate titles; exactly one row per family has
`preferred_family_record=true`. Automated status is triage metadata, never admission.
The review queue's `first_pass_scope_status` only prioritizes human work; the field
`automated_screening_is_admission` is always false.

`curation_inputs/review_decisions.json` is the only persistent review ledger. Queue
status fields are derived projections and must not be edited as ground truth.

Paper review rows bind `paper_id`, `reviewer_id`, `reviewed_at`,
`reviewer_attestation`, `decision`, `target_slot_ids`, and source
`evidence_locators`. Two reviewers must agree on the scope decision and target
enumeration, or a third adjudicator resolves the disagreement. Automated scope
consensus and the manually seeded P0 queue do not satisfy this paper-admission gate.

Structure and route review rows use the common fields `target_slot_id`,
`reviewer_id`, `reviewed_at`, `reviewer_attestation`, `decision`, and `record`.
`decision` is `accept`, `reject`, or `needs_revision`. Two different reviewers must
submit matching normalized accepted records; otherwise the item remains pending until
an independent `adjudications` row binds `subject_type`, `target_slot_id`,
`adjudicator_id`, the same attestation fields, and the final record. The builder, not
the reviewer, derives admitted registries and runnable status.

## Source receipts

`source_query_receipts.jsonl` records every cached response hash, query-level returned
and provider-reported counts, retrieval mode, and completeness flag. An incomplete
`top_k_relevance_sample` is permitted only as a supplemental source; an enumeration
source must be complete or the dataset build fails.

## Target registry

`target_slots.jsonl` has one row per curated P0 literature target slot. These rows are
source-extraction work units, not frozen benchmark truth. A target is
runnable only when the parent paper is dual-review admitted as primary,
`target_smiles` is non-empty, `structure_status` is
`verified_source_concordant`, and `route_evidence_status` is
`verified_route_or_key_step`.

## Structure evidence

An accepted structure review record contains isomeric SMILES, source DOI, relative
and absolute stereochemistry decisions, identity confirmation, and a repository-
relative source artifact path/hash/locator. `structures.json` is a derived registry:
it contains only records admitted by two matching reviewers or by adjudication after
two independent submissions. PubChem or title/name resolution alone is insufficient.

`structure_resolution_candidates.jsonl` is a disposable review accelerator. Its
rows must cover all primary target slots, carry `admission_authority=false`, and
retain the PubChem response hash. A name match or valid RDKit round trip does not
establish source concordance.

`visual_structure_candidates.jsonl` and the P1
`curation_candidates/p1_scope/visual-structure-candidates.jsonl` counterpart are
one-row-per-target review accelerators derived from exact-paper images. Each row
retains the visual status and SMILES, source page/figure locator, source-artifact and
rendered-image hashes, RDKit postcheck, and its relation to any PubChem candidate.
They must carry `admission_authority=false`; model transcription plus a valid round
trip does not establish source identity or stereochemistry.

## Route evidence

An accepted route review record declares `ordered_route` or `strategic_key_step`,
binds every source artifact by path/hash, and contains at least one strategic event.
Ordered routes also contain source-located steps with stable step IDs, product and
precursor labels, transformation class, and strategic role. `route_evidence.json` is
derived from dual consensus or adjudication; LLM output alone is insufficient.

`route_evidence_candidates.jsonl` contains deterministic target-linked full-text
passages. Every row is bound to the source artifact hash and has no admission
authority; scheme/SI reconstruction and two independent route reviews are still
required.

## Blind views

`planner_targets.jsonl` contains only opaque target ID and target SMILES.
`evaluator_references.jsonl` contains withheld literature identity and route evidence.
The two views are generated only from runnable primary targets.

## Evaluation outputs

Literature truth and planner evaluation are different authorities. Source admission
remains in `curation_inputs/review_decisions.json`; blinded Strategy/route-value
reviews, baseline results, reaction audits, and experimental outcomes belong to the
frozen evaluation run that produced them. They must bind the opaque target ID, route
artifact hash, rubric version, reviewer role, and blinded packet ID without copying a
new structure or route truth into the dataset ledger.
"""
    schema_path = output_dir / "SCHEMA.md"
    canonical_schema = (
        repo_root / "benchmarks" / "recent_total_synthesis" / "SCHEMA.md"
    )
    if canonical_schema.is_file():
        schema = canonical_schema.read_text(encoding="utf-8")
    schema_path.write_text(schema, encoding="utf-8")

    review_protocol = """# Human review and experimental escalation protocol

## Paper admission

Two reviewers independently label primary/conditional/control/exclude, enumerate
targets, and record evidence locators. Disagreements require adjudication; automated
triage and abstracts cannot settle admission.
All signed decisions are stored in `curation_inputs/review_decisions.json`; rebuilding
the dataset must never erase or replace that ledger.

## Structure admission

Reviewers verify the final reported target, compound numbering, relative and absolute
stereochemistry against the article/SI. The resulting isomeric SMILES must round-trip
through RDKit and match the cited source drawing. Name-service matches remain leads.

## Route and key-step admission

Reviewers reconstruct the longest target-producing branch, preserve convergent side
branches, and identify the literature-nominated or human-judged strategic key step.
Every reaction needs source locators and compound identity continuity. Routine FGI or
protecting-group steps may be omitted only from key-step scoring, not from route
provenance.

## Separation of authorities

Dataset curators establish what the paper reports. They do not decide whether an AI
route is valuable. Planner outputs, literature routes, and baseline routes enter a
separate blinded evaluation only after target identity and route artifacts are frozen.
The same individual should not both adjudicate source truth and make the final
experimental investment decision for that item.

## Blinded route-value review

Each randomized packet shows the target, a route diagram, the route-defining steps,
concise conditions or evidence when available, and explicit unresolved risks. It hides
system identity, publication identity, route order within the comparison, model
confidence, stock-closed labels, and previous reviewer verdicts. Literature similarity
is scored separately and only after intrinsic value review; a credible novel route can
therefore succeed without copying the paper.

The reviewer form is intentionally short:

1. route-defining insight (1--5);
2. key-transformation plausibility (1--5);
3. stereochemical and selectivity logic (1--5);
4. convergence and route economy (1--5);
5. risk concentration and quality of contingencies (1--5);
6. laboratory decision: `pursue`, `redesign`, or `stop`;
7. one decisive reason and, if present, one fatal blocker.

At least three independent synthetic chemists review each primary comparison. The
frozen analysis reports ratings, pairwise preferences, `pursue` rate, inter-rater
heterogeneity, and full failures; it must not collapse the result into stock closure or
an unblinded literature-match score. Disagreements on a factual identity or reaction
error return to the source/reaction authority. Differences in taste remain observed
rater variation and are not adjudicated into artificial consensus.

## Reaction audit and experimental handoff

No route advances because it is merely target-connected or stock-closed. Before wet
lab work, the candidate must have a Host-valid graph, no critical identity or stereo
defect, an independent audit of the key transformation, reviewable precedent and
conditions or an explicit novelty hypothesis, a predeclared route-value threshold,
and a hazard/operations review.

The synthesis team receives one compact packet: target and route version, exact
structures and provenance, proposed forward transformation, scale and analytical
readouts, precedent/conditions, unresolved selectivity and compatibility risks,
materials, hazards, stop criteria, and fallback experiment. Experimental escalation
is staged: first one or two route-defining transformations at microscale, then a short
route segment, and only then a complete synthesis. Negative experiments are retained
as scientific outcomes rather than removed from the denominator.

## Freeze and evaluation

Freeze paper/article-family/scaffold splits before any planner run. The planner sees
only opaque target IDs and structures. Strategy similarity, route validity, stock
closure, and expert preference are separate endpoints; no post-hoc positive-only
denominator is allowed.
"""
    canonical_review_protocol = (
        repo_root / "benchmarks" / "recent_total_synthesis" / "REVIEW_PROTOCOL.md"
    )
    if canonical_review_protocol.is_file():
        # Keep the operational expert protocol in one reviewable Markdown file.
        # Alternate build directories receive the same protocol instead of a stale copy.
        review_protocol = canonical_review_protocol.read_text(encoding="utf-8")
    review_path = output_dir / "REVIEW_PROTOCOL.md"
    review_path.write_text(review_protocol, encoding="utf-8")

    dataset_card = f"""# Dataset card

## Purpose

This workspace supports three evaluation questions without conflating them:

1. Can a planner discover and preserve route-defining synthetic insight?
2. Can it independently rediscover strategic ideas found in recent human total
   syntheses, while still receiving credit for credible novel alternatives?
3. Can it materialize, validate, and eventually test those ideas rather than merely
   reach purchasable leaves?

## Cohorts and admissible claims

| Cohort | Current size | Evidence available | Valid use |
|---|---:|---|---|
| Curated recent-literature P0 seed | {len(admitted_primary)} papers / {sum(row["primary_target_count"] for row in admitted_primary)} target slots | manually curated scope/target slots; 0 completed independent paper reviews; {sum(bool(row.get("source_package_acquired")) for row in source_packages)} source package(s) | source extraction and dual review; route-strategy benchmark only after all admissions |
| P1 curation candidates | {len(p1_scope_consensus)} dual-AI metadata rows / {len(p1_candidate_slots)} exact-name candidate slots | {len(p1_scope_disagreements)} disagreement rows; {sum(bool(row.get("source_package_acquired")) for row in p1_source_packages)} source package(s); no admission authority | source-concordant structure/route review and independent adjudication |
| Strict post-cutoff review queue | {len(high_priority_unreviewed)} high-priority papers | public metadata and ranked source access | prospective denominator construction, not planner evaluation yet |
| SynthAtlas public frontier | {legacy_counts.get("synthatlas_frontier_targets", 0)} target structures | target-only structures; published planner artifacts are not experimental truth | blind reach, validity, and stock-closure evaluation |
| SynthEx reported literature cohort | 145 target/route entries (aggregate only) | aggregate counts, no recovered row list | paper-level aggregate context only |
| Local positive pilots | {legacy_counts.get("local_verified_cases", 0)} post-hoc cases | local evaluator evidence | smoke tests and case studies, never hit-rate denominator |

## Unit of analysis

Paper/article-family counts, target-slot counts, route-family counts, and runnable
target counts are reported separately. Targets from one collective/divergent paper
or shared scaffold remain in one split group.

## Leakage control

The planner view contains only an opaque target ID and verified isomeric SMILES. DOI,
title, target name, article route, compound labels, and strategic key-step annotations
remain evaluator-only. No planner view is emitted for a target lacking both verified
structure and route evidence.

## Current release boundary

The discovery registry and review workflow are operational and reproducible. The
human-route benchmark currently has {runnable_primary_targets:,} runnable targets.
This is an explicit curation state, not a negative planning result.

## Exploratory Strategy coverage

A separate, explicitly non-formal coverage run may use RDKit-valid, conflict-free
structure leads before human admission. Its exact eligible count belongs to that
frozen run rather than this dataset card, because candidate extraction can continue
without changing admitted benchmark truth. The planner receives only the opaque
target-slot ID and candidate SMILES; target names, DOIs, titles, and route passages
remain withheld.

After Strategy generation is frozen, an evaluator-only worker may compare each
three-card portfolio with source-bound target passages. It reports exact, partial,
none, or non-comparable at target level. These rates are provisional diagnostics:
the structures are not source-concordance-admitted, the passages are automatically
extracted leads rather than human route truth, and the match labels are not dual
chemist review. Non-comparable targets are excluded from match-rate denominators.
"""
    dataset_card_path = output_dir / "DATASET_CARD.md"
    dataset_card_path.write_text(dataset_card, encoding="utf-8")

    nonstandard_rows = [
        row
        for row in paper_queue
        if row["first_pass_scope_status"] == "nonstandard_route_title_needs_scope_confirmation"
    ]
    nonstandard_table = (
        "\n".join(
            f"| {row['doi']} | {row['title'].replace('|', '/')} |" for row in nonstandard_rows
        )
        or "| - | None |"
    )
    screening_report = f"""# Screening status

## Current funnel

| Layer | Count | Meaning |
|---|---:|---|
| Preferred article families in the discovery registry | {len(preferred)} | high-recall metadata universe, mostly not benchmark papers |
| Strict post-cutoff preferred records | {len(post_cutoff)} | novelty slice before chemistry review |
| Curated P0 primary-paper seed | {len(admitted_primary)} | source extraction plus independent paper review required |
| P1 likely primary papers | {queue_tier_counts.get("P1_scope_review", 0)} | dual scope review required |
| P2 formal/noncore/control/method boundaries | {queue_tier_counts.get("P2_scope_boundary", 0)} | conditional/control adjudication required |
| Runnable recent-literature targets | 0 | no target has passed both structure and route admission |

## P1 review batches

- Batch A: {queue_scope_counts.get("likely_completed_route_needs_dual_review", 0)} papers with explicit completion language in the abstract.
- Batch B: {queue_scope_counts.get("nonstandard_route_title_needs_scope_confirmation", 0)} nonstandard route titles requiring scope confirmation.
- Batch C: {queue_scope_counts.get("abstract_scope_review_required", 0)} papers whose abstracts require chemical reading.
- Batch D: {queue_scope_counts.get("title_only_source_required", 0)} title-only records requiring article acquisition.

Automated batches prioritize review and never admit a paper. Two independent reviewer
decisions and adjudication remain mandatory.

{len(screening_annotations)} initial annotations are stored as non-admitting screening evidence; they do not
create target slots or planner inputs.

## Nonstandard-title omission candidates

| DOI | Title |
|---|---|
{nonstandard_table}

## Immediate curation order

1. Review Batch A for paper scope and exact target count.
2. Resolve the five nonstandard-title candidates to prevent systematic omission.
3. Acquire sources for Batch D and all P0 papers.
4. Verify source-concordant target stereochemistry before creating planner rows.
5. Extract route/key-step evidence into the evaluator-only registry.
"""
    screening_report_path = output_dir / "SCREENING_REPORT.md"
    screening_report_path.write_text(screening_report, encoding="utf-8")

    data_paths = {
        "papers": papers_path,
        "target_slots": targets_path,
        "paper_review_queue": queue_path,
        "work_items": work_path,
        "source_query_receipts": receipts_path,
        "structures": output_dir / "structures.json",
        "route_evidence": output_dir / "route_evidence.json",
        "planner_targets": planner_path,
        "evaluator_references": evaluator_path,
        "external_cohorts": output_dir / "external_cohorts.json",
        "quality_report": output_dir / "quality_report.json",
        "readme": readme_path,
        "methods": methods_path,
        "schema": schema_path,
        "review_protocol": review_path,
        "dataset_card": dataset_card_path,
        "screening_report": screening_report_path,
    }
    if source_package_path.exists():
        data_paths["source_package_receipts"] = source_package_path
    if source_storage_path.exists():
        data_paths["source_storage"] = source_storage_path
    if authorized_fetch_batch_path.exists():
        data_paths["authorized_source_fetch_batch"] = authorized_fetch_batch_path
    if structure_candidate_path.exists():
        data_paths["structure_resolution_candidates"] = structure_candidate_path
    if visual_candidate_path.exists():
        data_paths["visual_structure_candidates"] = visual_candidate_path
    if visual_candidate_summary_path.exists():
        data_paths["visual_structure_candidates_summary"] = visual_candidate_summary_path
    if route_candidate_path.exists():
        data_paths["route_evidence_candidates"] = route_candidate_path
    if p1_source_package_path.exists():
        data_paths["p1_source_package_receipts"] = p1_source_package_path
    if p1_authorized_fetch_batch_path.exists():
        data_paths["p1_authorized_source_fetch_batch"] = p1_authorized_fetch_batch_path
    if p1_scope_consensus_path.exists():
        data_paths["p1_scope_consensus"] = p1_scope_consensus_path
    if p1_scope_disagreements_path.exists():
        data_paths["p1_scope_disagreements"] = p1_scope_disagreements_path
    if p1_candidate_slots_path.exists():
        data_paths["p1_candidate_target_slots"] = p1_candidate_slots_path
    if p1_structure_candidates_path.exists():
        data_paths["p1_structure_resolution_candidates"] = p1_structure_candidates_path
    if p1_visual_candidates_path.exists():
        data_paths["p1_visual_structure_candidates"] = p1_visual_candidates_path
    if p1_visual_summary_path.exists():
        data_paths["p1_visual_structure_candidates_summary"] = p1_visual_summary_path
    if p1_route_candidates_path.exists():
        data_paths["p1_route_evidence_candidates"] = p1_route_candidates_path
    if screening_annotations_path.exists():
        data_paths["screening_annotations"] = screening_annotations_path
    if review_decisions_path.exists():
        data_paths["review_decisions"] = review_decisions_path
    manifest = {
        "schema_version": "recent_total_synthesis_benchmark.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "freeze_window": {"start": FREEZE_START, "end": FREEZE_END},
        "cutoffs": {
            "strict_model_cutoff": STRICT_POST_CUTOFF,
            "synthex_model_cutoff_proxy": SYNTHEX_CUTOFF_PROXY,
        },
        "claim_boundary": {
            "metadata_discovery_is_complete_benchmark_truth": False,
            "manual_seed_is_complete_denominator": False,
            "p0_seed_has_completed_dual_paper_review": False,
            "p1_dual_ai_scope_consensus_is_human_admission": False,
            "source_concordant_structures_verified": bool(admitted_structures),
            "literature_routes_or_key_steps_extracted": bool(admitted_routes),
            "runnable_primary_targets": runnable_primary_targets,
            "synthex_145_rows_publicly_recovered": False,
        },
        "counts": {
            "raw_provider_records": len(records),
            "deduplicated_paper_records": len(papers),
            "preferred_article_families": len(preferred),
            "preferred_post_strict_cutoff_records": len(post_cutoff),
            "manually_admitted_primary_papers": len(admitted_primary),
            "manually_admitted_conditional_papers": len(admitted_conditional),
            "manually_admitted_primary_target_slots": sum(
                row["primary_target_count"] for row in admitted_primary
            ),
            "high_priority_unreviewed_post_cutoff_papers": len(high_priority_unreviewed),
            "high_priority_non_crossref_recoveries": high_priority_non_crossref,
            "paper_review_queue_rows": len(paper_queue),
            "work_item_rows": len(work_items),
            "target_slots_from_manual_seed": len(targets),
            "preliminary_screening_annotations": len(screening_annotations),
            "submitted_paper_reviews": len(review_decisions.get("paper_reviews") or []),
            "admitted_source_concordant_structures": len(admitted_structures),
            "admitted_literature_routes_or_key_steps": len(admitted_routes),
            "admitted_ordered_literature_routes": sum(
                row.get("reference_scope") == "ordered_route"
                for row in admitted_routes
            ),
            "named_primary_target_slots": sum(
                bool(row.get("target_name")) for row in targets if row["slot_class"] == "primary"
            ),
            "structure_resolution_candidates_found_unverified": sum(
                row.get("lookup_status") == "candidate_found_unverified"
                for row in structure_candidates
            ),
            "route_evidence_candidate_rows": len(route_candidates),
            "route_evidence_candidates_with_passages": sum(
                bool(row.get("evidence_passages")) for row in route_candidates
            ),
            "p1_source_package_receipts": len(p1_source_packages),
            "p1_source_packages_acquired": sum(
                bool(row.get("source_package_acquired")) for row in p1_source_packages
            ),
            "p1_authorized_fetch_attempts": len(p1_authorized_fetch_attempts),
            "p1_authorized_fetch_attempts_accepted": sum(
                bool(row.get("accepted")) for row in p1_authorized_fetch_attempts
            ),
            "p1_dual_ai_scope_consensus_rows": len(p1_scope_consensus),
            "p1_dual_ai_scope_disagreement_rows": len(p1_scope_disagreements),
            "p1_candidate_target_slots_nonadmitting": len(p1_candidate_slots),
            "p1_structure_candidates_found_unverified": sum(
                row.get("lookup_status") == "candidate_found_unverified"
                for row in p1_structure_candidates
            ),
            "visual_structure_candidate_rows_nonadmitting": len(visual_candidates),
            "visual_structure_candidates_rdkit_valid": sum(
                (row.get("rdkit_validation") or {}).get("status") == "roundtrip_valid"
                for row in visual_candidates
            ),
            "p1_visual_structure_candidate_rows_nonadmitting": len(p1_visual_candidates),
            "p1_visual_structure_candidates_rdkit_valid": sum(
                (row.get("rdkit_validation") or {}).get("status") == "roundtrip_valid"
                for row in p1_visual_candidates
            ),
            "p1_route_evidence_candidate_rows": len(p1_route_candidates),
            "p1_route_evidence_candidates_with_passages": sum(
                bool(row.get("evidence_passages")) for row in p1_route_candidates
            ),
            "combined_candidate_papers": len(
                p0_candidate_paper_ids | p1_candidate_paper_ids
            ),
            "combined_candidate_target_slots": len(combined_visual_candidates),
            "combined_candidate_source_packages_acquired": len(
                (p0_candidate_paper_ids | p1_candidate_paper_ids)
                & acquired_candidate_paper_ids
            ),
            "combined_visual_candidates_rdkit_valid": sum(
                (row.get("rdkit_validation") or {}).get("status") == "roundtrip_valid"
                for row in combined_visual_candidates
            ),
            "combined_route_evidence_candidate_rows": len(
                combined_route_candidates
            ),
            "combined_route_evidence_candidates_with_passages": sum(
                bool(row.get("evidence_passages"))
                for row in combined_route_candidates
            ),
            "p0_dual_evidence_review_candidates": len(
                p0_dual_evidence_review_candidates
            ),
            "runnable_primary_targets": runnable_primary_targets,
            "authorized_fetch_attempts": len(authorized_fetch_attempts),
            "authorized_fetch_attempts_accepted": sum(
                bool(row.get("accepted")) for row in authorized_fetch_attempts
            ),
            "paper_queue_tiers": dict(sorted(queue_tier_counts.items())),
            "paper_queue_source_access": dict(sorted(queue_access_counts.items())),
            "paper_queue_first_pass_scope": dict(sorted(queue_scope_counts.items())),
            "source_query_groups": len(
                {(row["provider"], row["query_id"]) for row in cache_receipts}
            ),
            "complete_source_query_groups": len(
                {
                    (row["provider"], row["query_id"])
                    for row in cache_receipts
                    if row["retrieval_complete"]
                }
            ),
            "supplemental_truncated_query_groups": len(
                {
                    (row["provider"], row["query_id"])
                    for row in cache_receipts
                    if not row["retrieval_complete"]
                    and row["retrieval_mode"] == "top_k_relevance_sample"
                }
            ),
            "provider_coverage": dict(sorted(provider_counts.items())),
            "post_cutoff_automated_status": dict(sorted(automated_counts.items())),
            "curation_status": dict(sorted(curation_counts.items())),
        },
        "manual_seed": {
            "path": str(manual_path.relative_to(repo_root)).replace("\\", "/"),
            "schema_version": manual_payload.get("schema_version", ""),
            "record_count": len(manual),
        },
        "sources": [
            {"provider": "Crossref", "url": "https://api.crossref.org/works"},
            {"provider": "OpenAlex", "url": "https://api.openalex.org/works"},
            {
                "provider": "Europe PMC",
                "url": "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            },
            {"provider": "SynthEx", "url": "https://arxiv.org/abs/2608.07454"},
        ],
        "query_terms": QUERY_TERMS,
        "files": {
            name: {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256(path),
            }
            for name, path in data_paths.items()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

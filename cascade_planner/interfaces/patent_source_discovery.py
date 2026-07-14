"""Bounded Google Patents discovery without evidence authority."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Protocol
from urllib.parse import quote

import requests

from cascade_planner.interfaces.live_evidence import LiveEvidenceConnectorError


class PatentSearchConfig(Protocol):
    seed_publications: tuple[str, ...]
    timeout_s: float
    max_search_queries: int
    max_search_pages_per_query: int
    max_html_bytes: int


def evidence_queries(
    request: Mapping[str, Any],
    *,
    limit: int,
) -> list[str]:
    values = [str(request.get("target_name") or "").strip()]
    values.extend(
        str(row.get("query") or "").strip()
        for row in request.get("source_tasks") or []
        if isinstance(row, Mapping)
    )
    values.extend(
        str(row.get("source_ref") or "").removeprefix("patent:").strip()
        for row in request.get("source_hints") or []
        if isinstance(row, Mapping)
    )
    out: list[str] = []
    for value in values:
        compact = " ".join(value.split())[:800]
        if compact and compact.casefold() not in {row.casefold() for row in out}:
            out.append(compact)
        if len(out) >= limit:
            break
    return out


def google_patent_candidate_provider(
    config: PatentSearchConfig,
):
    def search(queries: Iterable[str]) -> Iterable[Mapping[str, Any]]:
        rows: list[dict[str, Any]] = []
        all_queries = [*config.seed_publications, *list(queries)]
        for query in all_queries[: config.max_search_queries]:
            explicit = _patent_publications(query)
            rows.extend(_direct_pdf_candidates(query, explicit))
            for publication in explicit:
                direct = _resolve_google_patent_publication(
                    publication,
                    timeout_s=config.timeout_s,
                    max_html_bytes=config.max_html_bytes,
                )
                if direct:
                    rows.append(direct)
            search_values = [*explicit, query]
            for search_value in search_values:
                nested = f"q=({search_value})"
                for page in range(config.max_search_pages_per_query):
                    nested_page = f"{nested}&page={page}" if page else nested
                    url = (
                        "https://patents.google.com/xhr/query?url="
                        + quote(nested_page, safe="")
                    )
                    response = requests.get(
                        url,
                        headers={
                            "User-Agent": "AutoPlanner/1.0 patent-evidence"
                        },
                        timeout=config.timeout_s,
                    )
                    if response.status_code != 200:
                        continue
                    try:
                        payload = response.json()
                    except requests.JSONDecodeError:
                        continue
                    rows.extend(_xhr_candidates(payload, query=query))
        return rows

    return search


def select_independent_candidates(
    values: Iterable[Mapping[str, Any]],
    *,
    queries: Iterable[str],
    limit: int,
) -> list[dict[str, Any]]:
    query_rows = list(queries)
    query_tokens = {
        token
        for query in query_rows
        for token in re.findall(r"[a-z0-9]{4,}", query.casefold())
        if token not in {"synthesis", "process", "preparation", "patent"}
    }
    explicit_queries = {value.casefold() for value in query_rows}
    by_publication: dict[str, tuple[int, dict[str, Any]]] = {}
    for raw in values:
        row = dict(raw)
        publication = _publication(row.get("publication_number"))
        if not publication:
            continue
        title = _plain_text(row.get("title")).casefold()
        snippet = _plain_text(row.get("snippet")).casefold()
        matched = sum(token in f"{title} {snippet}" for token in query_tokens)
        process = sum(
            term in title
            for term in (
                "synthesis",
                "synthetic",
                "process",
                "preparation",
                "intermediate",
            )
        )
        noise = sum(
            term in title
            for term in ("medical use", "combination", "formulation", "treatment")
        )
        explicit = int(publication.casefold() in explicit_queries)
        score = 100 * explicit + 12 * matched + 8 * process - 8 * noise
        normalized = {**row, "publication_number": publication}
        prior = by_publication.get(publication)
        if prior is None:
            by_publication[publication] = (score, normalized)
            continue
        prior_score, prior_row = prior
        merged = dict(prior_row)
        for key, value in normalized.items():
            if not merged.get(key) and value:
                merged[key] = value
        by_publication[publication] = (max(score, prior_score), merged)

    candidates = list(by_publication.values())
    candidates.sort(key=lambda item: (-item[0], item[1]["publication_number"]))
    selected: list[dict[str, Any]] = []
    families: set[str] = set()
    for _score, row in candidates:
        family = str(row.get("family_id") or row["publication_number"])
        if family in families:
            continue
        families.add(family)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def fetch_bounded_bytes(url: str, timeout_s: float, limit: int) -> bytes:
    response = requests.get(
        url,
        headers={"User-Agent": "AutoPlanner/1.0 patent-evidence"},
        timeout=timeout_s,
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in response.iter_content(64 * 1024):
            total += len(chunk)
            if total > limit:
                raise LiveEvidenceConnectorError(
                    "source_artifact_size_limit_exceeded"
                )
            chunks.append(bytes(chunk))
    finally:
        response.close()
    return b"".join(chunks)


def _xhr_candidates(
    payload: Mapping[str, Any],
    *,
    query: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in dict(payload.get("results") or {}).get("cluster") or []:
        for result in dict(cluster).get("result") or []:
            patent = dict(dict(result).get("patent") or {})
            publication = _publication(patent.get("publication_number"))
            pdf = str(patent.get("pdf") or "").strip("/")
            if not publication:
                continue
            family_seed = {
                "priority_date": patent.get("priority_date"),
                "assignee": patent.get("assignee"),
                "title": _plain_text(patent.get("title")),
            }
            rows.append(
                {
                    "publication_number": publication,
                    "title": _plain_text(patent.get("title")),
                    "snippet": _plain_text(patent.get("snippet")),
                    "priority_date": str(patent.get("priority_date") or ""),
                    "assignee": _plain_text(patent.get("assignee")),
                    "pdf_url": (
                        "https://patentimages.storage.googleapis.com/" + pdf
                        if pdf
                        else ""
                    ),
                    "html_url": (
                        "https://patents.google.com/patent/"
                        f"{publication}/en"
                    ),
                    "family_id": "search-family:" + _digest(family_seed)[:24],
                    "query": query,
                }
            )
    return rows


def _direct_pdf_candidates(
    query: str,
    publications: Iterable[str],
) -> list[dict[str, Any]]:
    urls = re.findall(
        r"https://patentimages\.storage\.googleapis\.com/[^\s;]+?\.pdf",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    publication_rows = list(publications)
    rows: list[dict[str, Any]] = []
    for url in urls:
        publication = next(
            (
                value
                for value in publication_rows
                if value.casefold() in url.casefold()
            ),
            _publication(Path(url).stem),
        )
        if not publication:
            continue
        rows.append(
            {
                "publication_number": publication,
                "title": f"Patent {publication}",
                "snippet": "direct primary patent PDF locator",
                "pdf_url": url,
                "html_url": (
                    "https://patents.google.com/patent/"
                    f"{publication}/en"
                ),
                "family_id": f"publication:{publication}",
                "query": query,
            }
        )
    return rows


def _resolve_google_patent_publication(
    publication: str,
    *,
    timeout_s: float,
    max_html_bytes: int,
) -> dict[str, Any]:
    url = f"https://patents.google.com/patent/{publication}/en"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "AutoPlanner/1.0 patent-evidence"},
            timeout=timeout_s,
        )
    except requests.RequestException:
        return {}
    if response.status_code != 200 or len(response.content) > max_html_bytes:
        return {}
    text = response.text
    pdf_match = re.search(
        r'<meta\s+name="citation_pdf_url"\s+content="([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )
    title_match = re.search(
        r'<meta\s+(?:scheme="[^\"]+"\s+)?name="DC\.title"\s+content="([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )
    priority_match = re.search(
        r'<meta\s+scheme="dateSubmitted"\s+content="([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )
    title = _plain_text(title_match.group(1) if title_match else publication)
    priority = str(priority_match.group(1) if priority_match else "")
    family_seed = {"priority_date": priority, "title": title}
    return {
        "publication_number": publication,
        "title": title,
        "snippet": "resolved from primary patent publication page",
        "priority_date": priority,
        "html_url": url,
        "_primary_html_bytes": bytes(response.content),
        "pdf_url": html.unescape(pdf_match.group(1)) if pdf_match else "",
        "family_id": "publication-family:" + _digest(family_seed)[:24],
        "query": publication,
    }


def _patent_publications(value: str) -> list[str]:
    rows = re.findall(
        r"\b(?:US|WO|EP|CN|JP|KR|CA|AU|DE|GB)\s*[-/]?\s*"
        r"\d{5,12}\s*[A-Z]\d?\b",
        str(value or "").upper(),
    )
    out: list[str] = []
    for row in rows:
        publication = _publication(row)
        if publication and publication not in out:
            out.append(publication)
    return out


def _publication(value: Any) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return compact if re.fullmatch(r"[A-Z]{2}\d{5,12}[A-Z]\d?", compact) else ""


def _plain_text(value: Any) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(str(value or ""))).split())


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "evidence_queries",
    "fetch_bounded_bytes",
    "google_patent_candidate_provider",
    "select_independent_candidates",
]

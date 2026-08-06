"""Bounded Google Patents discovery without evidence authority."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Protocol
from urllib.parse import quote

import requests

from cascade_planner.interfaces.epo_family_discovery import (
    epo_family_pdf_candidates,
)
from cascade_planner.interfaces.live_evidence import LiveEvidenceConnectorError
from cascade_planner.interfaces.literature_search import europe_pmc_metadata_search


HttpRequester = Callable[..., Any]
PatentMetadataSearch = Callable[[str, int], Iterable[Mapping[str, Any]]]


class PatentSearchConfig(Protocol):
    seed_publications: tuple[str, ...]
    timeout_s: float
    max_search_queries: int
    max_search_pages_per_query: int
    max_html_bytes: int
    max_patents: int


def evidence_queries(
    request: Mapping[str, Any],
    *,
    limit: int,
) -> list[str]:
    identity = dict(request.get("target_identity") or {})
    structure_patents = [
        str(value).strip()
        for value in identity.get("patent_ids") or []
        if str(value).strip()
    ]
    name_linked_patents = [
        str(value).strip()
        for value in identity.get("name_linked_patent_ids") or []
        if str(value).strip()
    ]
    # Director source tasks express the actual transformation being audited.
    # PubChem's structure-linked patent list is useful only as a fallback: it
    # frequently contains formulation, medical-use and unrelated compound
    # families.  Putting those identifiers first used to consume the entire
    # bounded query budget before a precise process query was attempted.
    # A Director-verified publication token is strictly more actionable than
    # a prose search query and must survive the bounded query limit.  It still
    # grants no evidence authority; it only selects the primary document to
    # download and replay.
    patent_hints = [
        dict(row)
        for row in request.get("source_hints") or []
        if isinstance(row, Mapping)
        and str(row.get("source_kind") or "patent").casefold() == "patent"
        and _patent_publications(str(row.get("source_ref") or ""))
    ]
    patent_hints.sort(key=_patent_hint_rank)
    values: list[str] = [
        str(row.get("source_ref") or "").removeprefix("patent:").strip()
        for row in patent_hints
    ]
    for row in request.get("source_tasks") or []:
        if not isinstance(row, Mapping) or (
            row.get("source_types")
            and not any(
                str(kind).casefold() in {"patent", "patents"}
                for kind in row.get("source_types") or []
            )
        ):
            continue
        values.extend(
            str(ref).strip()
            for ref in row.get("source_refs") or []
            if _patent_publications(str(ref))
        )
        if str(row.get("query") or "").strip():
            values.append(str(row.get("query") or "").strip())
    target_name = str(request.get("target_name") or "").strip()
    if target_name:
        values.append(f'"{target_name}" synthesis process')
    # Reserve bounded slots for direct structure/name-linked publications.
    # Otherwise four Director prose queries can consume the full query budget
    # before an authoritative PubChem patent cross-reference is attempted.
    linked_patents = sorted(
        [*structure_patents, *name_linked_patents],
        key=_structure_patent_rank,
    )
    if linked_patents:
        # Explicit route-linked publications are more specific than broad
        # structure/name cross-references.  Preserve up to three of them and
        # let identity-linked patents occupy only the remaining bounded slots.
        protected_count = min(3, len(patent_hints), max(0, limit - 1))
        protected = values[:protected_count]
        remaining_slots = max(0, limit - len(protected))
        reserved_count = min(3, len(linked_patents), remaining_slots)
        prose_count = max(0, remaining_slots - reserved_count)
        reserved = linked_patents[:reserved_count]
        values = [
            *protected,
            *values[protected_count : protected_count + prose_count],
            *reserved,
        ]
    out: list[str] = []
    for value in values:
        compact = " ".join(value.split())[:800]
        if compact and compact.casefold() not in {row.casefold() for row in out}:
            out.append(compact)
        if len(out) >= limit:
            break
    return out


def _patent_hint_rank(row: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    """Prioritize independently corroborated target-edge hints."""

    return (
        -int(row.get("target_edge_occurrence_count") or 0),
        -int(row.get("corroborating_source_ref_count") or 0),
        -int(row.get("occurrence_count") or 0),
        -int(row.get("route_skeleton_count") or 0),
        str(row.get("source_ref") or ""),
    )


def _structure_patent_rank(value: str) -> tuple[int, int, str]:
    publication = _publication(value)
    match = re.match(r"([A-Z]{2})(\d+)", publication)
    if not match:
        return (1, 10**15, publication)
    number = int(match.group(2))
    return (0, number, publication)


def google_patent_candidate_provider(
    config: PatentSearchConfig,
    *,
    metadata_search: PatentMetadataSearch | None = None,
):
    search_metadata = metadata_search or europe_pmc_metadata_search

    def search(queries: Iterable[str]) -> Iterable[Mapping[str, Any]]:
        rows: list[dict[str, Any]] = []
        all_queries = [*config.seed_publications, *list(queries)]
        for query in all_queries[: config.max_search_queries]:
            explicit = _patent_publications(query)
            rows.extend(_direct_pdf_candidates(query, explicit))
            for publication in explicit:
                rows.extend(
                    epo_family_pdf_candidates(
                        publication,
                        timeout_s=config.timeout_s,
                        max_response_bytes=config.max_html_bytes,
                    )
                )
                direct = _resolve_google_patent_publication(
                    publication,
                    timeout_s=config.timeout_s,
                    max_html_bytes=config.max_html_bytes,
                )
                if direct:
                    rows.append(direct)
                else:
                    rows.extend(
                        _pubchem_family_pdf_candidates(
                            publication,
                            timeout_s=config.timeout_s,
                            max_response_bytes=config.max_html_bytes,
                        )
                    )
            if explicit:
                # The structure-bound publication has already gone through
                # direct, EPO-family and PubChem-family resolution.  Repeating
                # it through Google XHR adds no authority and is frequently
                # throttled, so move to the next exact identifier instead.
                if len(
                    select_independent_candidates(
                        rows,
                        queries=all_queries,
                        limit=max(1, int(config.max_patents)),
                    )
                ) >= max(1, int(config.max_patents)):
                    break
                continue
            # Start with a bounded metadata query.  A resolved EPO family is
            # both faster and more authoritative than waiting for a blocked
            # Google XHR endpoint, so Google now acts only as fallback.
            try:
                metadata_rows = list(
                    search_metadata(
                        f"({query}) AND SRC:PAT",
                        max(4, int(config.max_patents) * 3),
                    )
                )
            except (OSError, RuntimeError, ValueError, requests.RequestException):
                metadata_rows = []
            patent_metadata = [
                dict(row)
                for row in metadata_rows
                if str(row.get("source_kind") or "").casefold() == "patent"
            ]
            patent_metadata.sort(
                key=lambda row: _metadata_patent_rank(row, query=query)
            )
            for metadata in patent_metadata:
                publication = _publication(
                    metadata.get("publication_number") or metadata.get("id")
                )
                if not publication:
                    continue
                rows.extend(
                    _metadata_patent_candidates(
                        publication,
                        title=_plain_text(metadata.get("title")),
                        query=query,
                        config=config,
                    )
                )
                if _candidate_limit_reached(rows, all_queries, config.max_patents):
                    break
            if _candidate_limit_reached(rows, all_queries, config.max_patents):
                break
            search_values = [*explicit, query]
            for search_value in search_values:
                xhr_query = _google_patents_xhr_query(search_value)
                if not xhr_query:
                    continue
                nested = f"q=({xhr_query})"
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


def _candidate_limit_reached(
    rows: Iterable[Mapping[str, Any]],
    queries: Iterable[str],
    limit: int,
) -> bool:
    return len(
        select_independent_candidates(
            rows,
            queries=queries,
            limit=max(1, int(limit)),
        )
    ) >= max(1, int(limit))


def _metadata_patent_rank(
    row: Mapping[str, Any],
    *,
    query: str,
) -> tuple[int, int, bool]:
    title = _plain_text(row.get("title")).casefold()
    target_tokens = {
        token
        for token in re.findall(r"[a-z0-9]{4,}", query.casefold())
        if token
        not in {
            "synthesis",
            "synthetic",
            "process",
            "preparation",
            "production",
            "patent",
        }
    }
    matched = sum(token in title for token in target_tokens)
    process = sum(
        token in title
        for token in ("synthesis", "process", "preparation", "production")
    )
    publication = _publication(row.get("publication_number") or row.get("id"))
    return (-matched, -process, not publication.startswith("WO"))


def _metadata_patent_candidates(
    publication: str,
    *,
    title: str,
    query: str,
    config: PatentSearchConfig,
) -> list[dict[str, Any]]:
    """Resolve metadata-only patent ids to primary publication artifacts."""

    variants = [publication]
    if re.fullmatch(r"WO\d{5,12}", publication):
        variants = [f"{publication}A2", f"{publication}A1"]
    for variant in variants:
        family = epo_family_pdf_candidates(
            variant,
            timeout_s=config.timeout_s,
            max_response_bytes=config.max_html_bytes,
        )
        if family:
            return [
                {
                    **row,
                    "title": str(row.get("title") or title),
                    "query": query,
                    "metadata_provider": "europe_pmc",
                    "xml_url": str(
                        row.get("xml_url")
                        or _epo_xml_url(str(row.get("publication_number") or ""))
                    ),
                }
                for row in family
            ]
        direct = _resolve_google_patent_publication(
            variant,
            timeout_s=config.timeout_s,
            max_html_bytes=config.max_html_bytes,
        )
        if direct:
            return [
                {
                    **direct,
                    "title": str(direct.get("title") or title),
                    "query": query,
                    "metadata_provider": "europe_pmc",
                    "_source_priority": 18,
                }
            ]
    return []


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
        source_priority = max(0, min(50, int(row.get("_source_priority") or 0)))
        score = (
            100 * explicit
            + 12 * matched
            + 8 * process
            - 8 * noise
            + source_priority
        )
        normalized = {**row, "publication_number": publication}
        if not normalized.get("xml_url"):
            normalized["xml_url"] = _epo_xml_url(publication)
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


def _epo_xml_url(publication: str) -> str:
    match = re.fullmatch(r"EP(?P<number>\d{5,12})(?P<kind>A\d|B\d)", publication)
    if match is None:
        return ""
    document_id = f"EP{match.group('number')}NW{match.group('kind')}"
    return (
        "https://data.epo.org/publication-server/rest/v1.2/patents/"
        f"{document_id}/document.xml"
    )


def _xhr_candidates(
    payload: Mapping[str, Any],
    *,
    query: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in dict(payload.get("results") or {}).get("cluster") or []:
        for result in dict(cluster).get("result") or []:
            result_row = dict(result)
            patent = dict(result_row.get("patent") or {})
            publication = _publication(
                patent.get("publication_number")
                or str(result_row.get("id") or "").removeprefix("patent/").split(
                    "/", 1
                )[0]
            )
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


def _google_patents_xhr_query(value: Any) -> str:
    """Compile free chemical prose into Google Patents' bounded XHR syntax.

    The endpoint rejects otherwise valid chemical names when punctuation such
    as commas, parentheses, quotes, or hyphens is embedded directly inside a
    ``q=(...)`` expression.  Publication identifiers take the direct resolver
    path before this helper; free text is therefore reduced to stable lexical
    terms and joined with the endpoint's explicit ``+`` separator.
    """

    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", html.unescape(str(value or "")))
        if len(token) >= 3 and not token.isdigit()
    ]
    selected: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        selected.append(token)
        if len(selected) >= 16:
            break
    return "+".join(selected)


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


def _pubchem_family_pdf_candidates(
    publication: str,
    *,
    timeout_s: float,
    max_response_bytes: int,
    requester: HttpRequester = requests.get,
) -> list[dict[str, Any]]:
    """Resolve a blocked patent page through PubChem to an official EPO PDF."""
    canonical = _publication(publication)
    if not _hyphenated_publication(canonical):
        try:
            landing = requester(
                f"https://pubchem.ncbi.nlm.nih.gov/patent/{publication}",
                headers={"User-Agent": "AutoPlanner/1.0 patent-family-discovery"},
                timeout=timeout_s,
            )
        except requests.RequestException:
            return []
        landing_bytes = bytes(landing.content)
        if landing.status_code != 200 or len(landing_bytes) > max_response_bytes:
            return []
        match = re.search(
            rb'<meta\s+name="ncbi_pubchem_publication_number"\s+content="([^"]+)"',
            landing_bytes,
            flags=re.IGNORECASE,
        )
        canonical = (
            _publication(match.group(1).decode("ascii", errors="ignore"))
            if match
            else ""
        )
    if not canonical:
        return []
    root_payload = _pubchem_patent_payload(
        canonical,
        requester=requester,
        timeout_s=timeout_s,
        max_response_bytes=max_response_bytes,
    )
    if not root_payload:
        return []
    title = _plain_text(dict(root_payload.get("Record") or {}).get("RecordTitle"))
    family_members = _pubchem_family_members(root_payload)
    rows: list[dict[str, Any]] = []
    for member in sorted(
        family_members,
        key=lambda value: (
            not value.endswith(("B1", "B2")),
            not value.endswith(("A1", "A2")),
            value,
        ),
    )[:6]:
        payload = _pubchem_patent_payload(
            member,
            requester=requester,
            timeout_s=timeout_s,
            max_response_bytes=max_response_bytes,
        )
        publication_date = _pubchem_publication_date(payload)
        if not publication_date:
            continue
        parts = re.fullmatch(r"EP(\d{5,12})(A\d|B\d)", member)
        if not parts:
            continue
        number, kind = parts.groups()
        document_id = f"EP{number}NW{kind}"
        rows.append(
            {
                "publication_number": member,
                "title": title or f"Patent {member}",
                "snippet": f"official EPO family member resolved from PubChem patent {canonical}",
                "publication_date": publication_date,
                "pdf_url": (
                    "https://data.epo.org/publication-server/rest/v1.2/"
                    f"publication-dates/{publication_date.replace('-', '')}/"
                    f"patents/{document_id}/document.pdf"
                ),
                "html_url": (
                    "https://patents.google.com/patent/"
                    f"{member}/en"
                ),
                "family_id": f"pubchem-family:{canonical}",
                "query": publication,
                "source_authority": "pubchem_to_epo_publication_server",
                "_source_priority": 25 if kind.startswith("B") else 15,
            }
        )
        if len(rows) >= 2:
            break
    return rows


def _pubchem_patent_payload(
    publication: str,
    *,
    requester: HttpRequester,
    timeout_s: float,
    max_response_bytes: int,
) -> dict[str, Any]:
    token = _hyphenated_publication(publication)
    if not token:
        return {}
    try:
        response = requester(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/"
            f"patent/{token}/JSON",
            headers={"Accept": "application/json", "User-Agent": "AutoPlanner/1.0"},
            timeout=timeout_s,
        )
    except requests.RequestException:
        return {}
    content = bytes(response.content)
    if response.status_code != 200 or len(content) > max_response_bytes:
        return {}
    try:
        value = response.json()
    except (ValueError, requests.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _pubchem_family_members(payload: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for section in _walk_mappings(payload):
        if str(section.get("TOCHeading") or "") != "Patent Family":
            continue
        for information in section.get("Information") or []:
            if not isinstance(information, Mapping):
                continue
            for row in dict(information.get("Value") or {}).get("StringWithMarkup") or []:
                if not isinstance(row, Mapping):
                    continue
                publication = _publication(row.get("String"))
                if publication.startswith("EP") and publication not in values:
                    values.append(publication)
        break
    return values


def _pubchem_publication_date(payload: Mapping[str, Any]) -> str:
    for section in _walk_mappings(payload):
        if str(section.get("TOCHeading") or "") != "Publication Date":
            continue
        for information in section.get("Information") or []:
            if not isinstance(information, Mapping):
                continue
            values = dict(information.get("Value") or {}).get("DateISO8601") or []
            if values:
                value = str(values[0]).replace("/", "-")
                return value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else ""
    return ""


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _walk_mappings(child)


def _hyphenated_publication(publication: str) -> str:
    parsed = re.fullmatch(r"([A-Z]{2})(\d{5,12})([A-Z]\d?)", publication)
    return "-".join(parsed.groups()) if parsed else ""


def _patent_publications(value: str) -> list[str]:
    rows = re.findall(
        r"\b(?:US|WO|EP|CN|JP|KR|CA|AU|DE|GB)\s*[-/]?\s*"
        r"\d{5,12}(?:\s*[-/]?\s*[A-Z]\d?)?\b",
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
    return compact if re.fullmatch(r"[A-Z]{2}\d{5,12}(?:[A-Z]\d?)?", compact) else ""


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

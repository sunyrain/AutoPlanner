"""Resolve WO publications to bounded, official EPO family PDFs.

The EPO linked-data graph is discovery metadata only.  It is used to locate an
EP family member whose primary publication bytes can then be frozen and parsed
by the patent evidence connector.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping

import requests


EPO_LINKED_DATA_QUERY_URL = "https://data.epo.org/linked-data/query"
EPO_PUBLICATION_SERVER_BASE = "https://data.epo.org/publication-server/rest/v1.2"
JsonRequester = Callable[..., Any]


def epo_family_pdf_candidates(
    publication: str,
    *,
    timeout_s: float,
    max_response_bytes: int,
    requester: JsonRequester = requests.get,
) -> list[dict[str, Any]]:
    """Return same-simple-family EP members with deterministic PDF URLs."""

    parsed = _publication_parts(publication)
    if parsed is None or parsed[0] != "WO":
        return []
    authority, number, kind = parsed
    root_uri = (
        "http://data.epo.org/linked-data/data/publication/"
        f"{authority}/{number}/{kind}/-"
    )
    query = _family_query(root_uri)
    try:
        response = requester(
            EPO_LINKED_DATA_QUERY_URL,
            params={"query": query},
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "AutoPlanner/1.0 patent-family-discovery",
            },
            timeout=timeout_s,
        )
    except requests.RequestException:
        return []
    content = bytes(response.content)
    if response.status_code != 200 or len(content) > max_response_bytes:
        return []
    try:
        payload = response.json()
    except (ValueError, requests.JSONDecodeError):
        return []
    bindings = list(dict(dict(payload).get("results") or {}).get("bindings") or [])
    rows = _candidate_rows(bindings, root_publication=publication)
    rows.sort(
        key=lambda row: (
            -int(row["_source_priority"]),
            str(row.get("publication_number") or ""),
        )
    )
    return rows[:2]


def _candidate_rows(
    bindings: list[Any],
    *,
    root_publication: str,
) -> list[dict[str, Any]]:
    parsed_rows: list[tuple[str, str, str, str]] = []
    for raw in bindings:
        if not isinstance(raw, Mapping):
            continue
        uri = _binding_value(raw, "object")
        match = re.search(r"/publication/EP/(\d{5,12})/(A\d|B\d)/", uri)
        if not match:
            continue
        title = _binding_value(raw, "title")
        parsed_rows.append((match.group(1), match.group(2), _binding_value(raw, "date"), title))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, kind, date, title in parsed_rows:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            continue
        publication = f"EP{number}{kind}"
        if publication in seen:
            continue
        seen.add(publication)
        compact_date = date.replace("-", "")
        document_id = f"EP{number}NW{kind}"
        rows.append(
            {
                "publication_number": publication,
                "title": " ".join(title.split()) or f"Patent {publication}",
                "snippet": f"official EPO family member of {root_publication}",
                "publication_date": date,
                "pdf_url": (
                    f"{EPO_PUBLICATION_SERVER_BASE}/publication-dates/"
                    f"{compact_date}/patents/{document_id}/document.pdf"
                ),
                "html_url": "",
                "family_id": f"epo-family:{root_publication}",
                "query": root_publication,
                "source_authority": "epo_publication_server",
                "_source_priority": 30 if kind.startswith("B") else 20,
            }
        )
    return rows


def _family_query(root_uri: str) -> str:
    return f"""PREFIX patent: <http://data.epo.org/linked-data/def/patent/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT DISTINCT ?object ?date ?title {{
  <{root_uri}> patent:application ?root .
  ?root patent:familyMemberOf/patent:familyMember ?member .
  ?member patent:publication ?object .
  ?object rdf:type patent:Publication ;
    patent:publicationDate ?date ; patent:titleOfInvention ?title .
  FILTER(lang(?title) = 'en')
  FILTER(CONTAINS(STR(?object), '/publication/EP/'))
}} ORDER BY ?object LIMIT 32"""


def _publication_parts(value: str) -> tuple[str, str, str] | None:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    match = re.fullmatch(r"([A-Z]{2})(\d{5,12})([A-Z]\d?)", compact)
    return match.groups() if match else None


def _binding_value(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return str(dict(value).get("value") or "") if isinstance(value, Mapping) else ""


__all__ = ["epo_family_pdf_candidates"]

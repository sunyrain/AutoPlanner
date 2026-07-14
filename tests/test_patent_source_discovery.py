from __future__ import annotations

import json

from cascade_planner.interfaces.patent_source_discovery import (
    _patent_publications,
    _pubchem_family_pdf_candidates,
    evidence_queries,
    select_independent_candidates,
)


class _Response:
    def __init__(self, value: bytes | dict, status: int = 200) -> None:
        self.status_code = status
        self.content = value if isinstance(value, bytes) else json.dumps(value).encode()

    def json(self):
        return json.loads(self.content)


def test_pubchem_hyphenated_publication_is_normalized() -> None:
    assert _patent_publications("WO-2011123232-A1") == ["WO2011123232A1"]
    assert _patent_publications("US4681893") == ["US4681893"]


def test_evidence_queries_are_bounded_and_deduplicated() -> None:
    request = {
        "target_name": "Nirmatrelvir",
        "source_tasks": [
            {"query": "nirmatrelvir"},
            {"query": "WO2021250648A1 synthesis"},
        ],
        "source_hints": [{"source_ref": "patent:WO2021250648A1"}],
    }

    assert evidence_queries(request, limit=3) == [
        "nirmatrelvir",
        "WO2021250648A1 synthesis",
        '"Nirmatrelvir" synthesis process',
    ]


def test_route_specific_source_tasks_precede_structure_patent_fallbacks() -> None:
    request = {
        "target_name": "Zavegepant",
        "target_identity": {
            "patent_ids": ["WO-2012-079783-A1", "US-2012-0315304-A1"]
        },
        "source_tasks": [{"query": "generic amino acid synthesis"}],
    }

    assert evidence_queries(request, limit=4) == [
        "generic amino acid synthesis",
        '"Zavegepant" synthesis process',
        "WO-2012-079783-A1",
        "US-2012-0315304-A1",
    ]


def test_verified_patent_source_ref_precedes_its_search_query() -> None:
    request = {
        "target_name": "Simvastatin",
        "source_tasks": [
            {
                "query": "simvastatin monacolin J acylation",
                "source_types": ["patent", "journal"],
                "source_refs": ["patent:US8211664B2", "doi:10.1000/paper"],
            }
        ],
    }

    assert evidence_queries(request, limit=3) == [
        "patent:US8211664B2",
        "simvastatin monacolin J acylation",
        '"Simvastatin" synthesis process',
    ]


def test_three_structure_resolved_patents_fit_before_free_text() -> None:
    request = {
        "target_name": "Example",
        "target_identity": {
            "patent_ids": ["WO-1-A1", "US-2-A1", "EP-3-A1", "CN-4-A"]
        },
        "source_tasks": [{"query": "broad synthesis"}],
    }

    assert evidence_queries(request, limit=5) == [
        "broad synthesis",
        '"Example" synthesis process',
        "WO-1-A1",
        "US-2-A1",
        "EP-3-A1",
    ]


def test_long_structure_patent_list_prefers_original_low_number_families() -> None:
    request = {
        "target_name": "Atorvastatin",
        "target_identity": {
            "patent_ids": [
                "US12168069",
                "US11369567",
                "CA2220018",
                "US4681893",
                "US5969156",
            ]
        },
    }

    assert evidence_queries(request, limit=4) == [
        '"Atorvastatin" synthesis process',
        "CA2220018",
        "US4681893",
        "US5969156",
    ]


def test_candidate_dedup_merges_prefetched_html_with_pdf_fallback() -> None:
    html = b"<html>prefetched publication</html>"
    selected = select_independent_candidates(
        [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:one",
                "title": "Preparation of target",
                "html_url": (
                    "https://patents.google.com/patent/US1234567A1/en"
                ),
                "_primary_html_bytes": html,
            },
            {
                "publication_number": "US-1234567-A1",
                "family_id": "family:one",
                "pdf_url": "https://source.invalid/US1234567A1.pdf",
            },
            {
                "publication_number": "EP7654321A1",
                "family_id": "family:one",
                "title": "Same patent family",
            },
            {
                "publication_number": "WO7654321A1",
                "family_id": "family:two",
                "title": "Independent synthesis of target",
            },
        ],
        queries=["target synthesis"],
        limit=3,
    )

    assert len(selected) == 2
    us = next(
        row for row in selected if row["publication_number"] == "US1234567A1"
    )
    assert us["_primary_html_bytes"] == html
    assert us["pdf_url"] == "https://source.invalid/US1234567A1.pdf"
    assert {row["family_id"] for row in selected} == {
        "family:one",
        "family:two",
    }


def test_pubchem_family_fallback_builds_official_epo_pdf_locator() -> None:
    def requester(url: str, **_kwargs):
        if url.endswith("/patent/US4681893"):
            return _Response(
                b'<meta name="ncbi_pubchem_publication_number" content="US-4681893-A">'
            )
        if "/patent/US-4681893-A/JSON" in url:
            return _Response(
                {
                    "Record": {
                        "RecordTitle": "Atorvastatin process",
                        "Section": [
                            {
                                "TOCHeading": "Patent Family",
                                "Information": [
                                    {
                                        "Value": {
                                            "StringWithMarkup": [
                                                {"String": "EP-0247633-B1"}
                                            ]
                                        }
                                    }
                                ],
                            }
                        ],
                    }
                }
            )
        assert "/patent/EP-0247633-B1/JSON" in url
        return _Response(
            {
                "Record": {
                    "Section": [
                        {
                            "TOCHeading": "Publication Date",
                            "Information": [
                                {"Value": {"DateISO8601": ["1991/01/30"]}}
                            ],
                        }
                    ]
                }
            }
        )

    rows = _pubchem_family_pdf_candidates(
        "US4681893",
        timeout_s=10,
        max_response_bytes=1_000_000,
        requester=requester,
    )

    assert rows[0]["publication_number"] == "EP0247633B1"
    assert rows[0]["pdf_url"].endswith(
        "/19910130/patents/EP0247633NWB1/document.pdf"
    )
    assert rows[0]["source_authority"] == "pubchem_to_epo_publication_server"

from __future__ import annotations

from cascade_planner.interfaces.patent_source_discovery import (
    evidence_queries,
    select_independent_candidates,
)


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
        "Nirmatrelvir",
        "WO2021250648A1 synthesis",
        "WO2021250648A1",
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

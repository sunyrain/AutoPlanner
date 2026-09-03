from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "extract_recent_total_synthesis_route_candidates.py"
)
SPEC = importlib.util.spec_from_file_location(
    "recent_total_synthesis_route_candidates", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)


def test_target_linked_synthesis_passage_retains_source_locator_and_text() -> None:
    payload = b"""\
    <article><body><sec id="route-1"><title>Synthesis of Example A</title>
    <p>Treatment of precursor 7 followed by cyclization provided Example A in 81% yield.
    <xref rid="scheme-4" ref-type="fig">Scheme 4</xref></p>
    </sec></body></article>
    """
    rows = extractor.passage_candidates(payload, "Example A", max_passages=3)
    assert len(rows) == 1
    assert rows[0]["section_id"] == "route-1"
    assert rows[0]["cross_references"] == [{"rid": "scheme-4", "ref_type": "fig"}]
    assert "provided Example A" in rows[0]["verbatim_text"]


def test_unrelated_background_paragraph_is_not_route_evidence() -> None:
    payload = b"<article><body><sec><title>Introduction</title><p>Example A is a natural product.</p></sec></body></article>"
    assert extractor.passage_candidates(payload, "Example A", max_passages=3) == []


def test_structured_article_json_yields_target_linked_reaction_window() -> None:
    payload = b"""{
      "full_text": [{
        "title": "Completion of the synthesis",
        "text": "Intermediate 7 was prepared. Treatment of 7 with acid furnished Example A in 81% yield. The product was characterized."
      }]
    }"""
    rows = extractor.structured_json_passage_candidates(payload, "Example A", max_passages=3)
    assert len(rows) == 1
    assert rows[0]["section_id"] == "structured-section-1"
    assert "furnished Example A" in rows[0]["verbatim_text"]


def test_html_section_heading_links_route_paragraph_to_target() -> None:
    payload = b"""
    <article><h2>Completion of Example A</h2>
    <p>Precursor 12 underwent radical cyclization and afforded compound 13.</p>
    </article>
    """
    rows = extractor.html_passage_candidates(payload, "Example A", max_passages=3)
    assert len(rows) == 1
    assert rows[0]["target_mentioned_in_section_title"] is True
    assert rows[0]["source_locator"]["type"] == "html_section_paragraph"


def test_same_pdf_page_scope_can_link_target_caption_and_reaction_text() -> None:
    blocks = [
        {
            "scope_key": "pdf-page:1",
            "section_title": "",
            "paragraph_index": 1,
            "locator": {"type": "pdf_page_paragraph", "page_number": 1},
            "text": "Scheme 4. Completion of Example A.",
        },
        {
            "scope_key": "pdf-page:1",
            "section_title": "",
            "paragraph_index": 2,
            "locator": {"type": "pdf_page_paragraph", "page_number": 1},
            "text": "Treatment of precursor 12 followed by cyclization afforded 13.",
        },
    ]
    rows = extractor._passage_candidates_from_blocks(blocks, "Example A", max_passages=3)
    assert len(rows) == 1
    assert rows[0]["target_mentioned_in_locator_scope"] is True


def test_candidate_primary_slots_share_the_primary_extraction_path() -> None:
    assert extractor.PRIMARY_TARGET_SLOT_CLASSES == {"primary", "primary_candidate"}

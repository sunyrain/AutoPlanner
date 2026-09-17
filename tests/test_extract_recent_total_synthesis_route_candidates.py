from __future__ import annotations

import importlib.util
import hashlib
import json
import io
import zipfile
from pathlib import Path

import pytest


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


def test_collective_synthesis_parses_each_unique_source_once_and_keeps_si(tmp_path, monkeypatch):
    main = json.dumps({"full_text": [{"title": "Example A and Example B", "text": "\n\n".join(f"Reaction {n} afforded Example A and Example B in high yield." for n in range(12))}]}).encode()
    si = b'<article><body><sec><title>Example A and Example B</title><p>Reduction of 7 furnished 8 in the supporting information.</p></sec></body></article>'
    artifacts = []
    for name, payload, kind in [("article.json", main, "authorized_publisher_structured_text"), ("si.xml", si, "supporting_information"), ("duplicate-si.xml", si, "supporting_information")]:
        (tmp_path/name).write_bytes(payload)
        artifacts.append({"cache_path": name, "sha256": hashlib.sha256(payload).hexdigest(), "artifact_kind": kind})
    calls = []
    original = extractor._artifact_blocks
    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(extractor, "_artifact_blocks", counted)
    slots = [{"paper_id": "paper", "doi": "10.1/example", "target_slot_id": name, "target_name": name, "slot_class": "primary"} for name in ["Example A", "Example B"]]
    rows, stats = extractor.extract_candidate_rows(slots, [{"paper_id": "paper", "artifacts": artifacts}], repo_root=tmp_path, max_passages=2)
    assert len(calls) == stats["artifact_parse_calls"] == 2
    assert len(rows) == 2
    for row in rows:
        assert row["source_coverage"]["article_text_inspected"]
        assert row["source_coverage"]["si_text_inspected"]
        assert row["source_coverage"]["selected_si_passages"] == 1
        assert len(row["source_artifacts"]) == 3
        assert len(row["evidence_passages"]) == 2
        assert row["admission_authority"] is False


def test_source_missing_target_is_retained_as_explicit_gap(tmp_path):
    slots = [{"paper_id": "missing", "doi": "10.1/missing", "target_slot_id": "gap", "target_name": "Example", "slot_class": "primary_candidate"}]
    rows, stats = extractor.extract_candidate_rows(slots, [], repo_root=tmp_path, max_passages=2)
    assert rows[0]["extraction_status"] == "source_artifacts_missing"
    assert rows[0]["source_artifacts"] == []
    assert stats["target_rows"] == 1
    assert stats["target_rows_with_source_package"] == 0


def test_inspected_source_with_wrong_hash_is_not_used(tmp_path):
    (tmp_path/"article.json").write_text('{"full_text": []}', encoding="utf-8")
    slots = [{"paper_id": "paper", "doi": "10.1/x", "target_slot_id": "x", "target_name": "Example", "slot_class": "primary"}]
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        extractor.extract_candidate_rows(slots, [{"paper_id": "paper", "artifacts": [{"cache_path": "article.json", "artifact_kind": "authorized_publisher_structured_text", "sha256": "wrong"}]}], repo_root=tmp_path, max_passages=2)


def test_repackaged_article_is_not_counted_as_supporting_information(tmp_path):
    document = {"full_text": [{"title": "Example", "text": "The reaction furnished Example in good yield."}]}
    artifacts = []
    for name, kind, indent in [("article.json", "authorized_publisher_structured_text", None), ("fake-si.json", "supporting_information", 4)]:
        payload = json.dumps(document, indent=indent).encode()
        (tmp_path/name).write_bytes(payload)
        artifacts.append({"cache_path": name, "artifact_kind": kind, "sha256": hashlib.sha256(payload).hexdigest()})
    assert artifacts[0]["sha256"] != artifacts[1]["sha256"]
    slot = {"paper_id": "paper", "doi": "10.1/x", "target_slot_id": "x", "target_name": "Example", "slot_class": "primary"}
    rows, _ = extractor.extract_candidate_rows([slot], [{"paper_id": "paper", "artifacts": artifacts}], repo_root=tmp_path, max_passages=2)
    assert rows[0]["source_coverage"]["si_text_inspected"] is False
    assert rows[0]["source_coverage"]["si_duplicates_article"] is True


def test_docx_disguised_as_zip_keeps_word_paragraph_locators():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Reduction provided Example A.</w:t></w:r></w:p></w:body></w:document>')
    blocks = extractor._artifact_blocks(stream.getvalue(), ".zip", container="received-si.zip")
    assert blocks[0]["locator"] == {"type": "docx_paragraph", "paragraph_index": 1, "container": "received-si.zip"}

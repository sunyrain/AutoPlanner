from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "fetch_recent_total_synthesis_sources.py"
)
SPEC = importlib.util.spec_from_file_location("recent_total_synthesis_source_fetcher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


def test_elsevier_coredata_wrapper_is_not_admitted_as_fulltext() -> None:
    payload = b"""\
    <full-text-retrieval-response>
      <coredata>
        <identifier>doi:10.1016/j.example.2026.1</identifier>
        <title>Total synthesis of Example A</title>
      </coredata>
    </full-text-retrieval-response>
    """
    with pytest.raises(ValueError, match="metadata without article body"):
        fetcher.publisher_fulltext_metadata(payload, "text/xml", "10.1016/j.example.2026.1")


def test_xml_article_body_is_admitted_with_auditable_measurements() -> None:
    body = "route evidence " * 200
    payload = f"""\
    <full-text-retrieval-response>
      <coredata><identifier>doi:10.1016/j.example.2026.1</identifier></coredata>
      <originalText><article><body><section>{body}</section></body></article></originalText>
    </full-text-retrieval-response>
    """.encode()
    metadata = fetcher.publisher_fulltext_metadata(payload, "text/xml", "10.1016/j.example.2026.1")
    assert metadata["validation_basis"] == "xml_article_body"
    assert metadata["article_body_characters"] >= 2_000
    assert metadata["section_elements"] == 1


def test_structured_abstract_wrapper_is_not_admitted_as_fulltext() -> None:
    payload = json.dumps(
        {
            "metadata": {
                "doi": "10.1016/j.example.2026.1",
                "abstract": "A synthesis was reported.",
            },
            "full_text": [],
        }
    ).encode()
    with pytest.raises(ValueError, match="metadata without article body"):
        fetcher.publisher_structured_text_metadata(payload, "10.1016/j.example.2026.1")


def test_structured_article_sections_are_admitted() -> None:
    payload = json.dumps(
        {
            "metadata": {"doi": "10.1016/j.example.2026.1"},
            "full_text": [{"title": "Results", "text": "route " * 500}],
        }
    ).encode()
    metadata = fetcher.publisher_structured_text_metadata(payload, "10.1016/j.example.2026.1")
    assert metadata["validation_basis"] == "structured_full_text_sections"
    assert metadata["section_elements"] == 1


def test_europe_pmc_resolution_recovers_new_pmcid_by_exact_doi() -> None:
    payload = b"""{
      "resultList": {"result": [{
        "doi": "10.1021/jacs.example",
        "pmid": "123",
        "pmcid": "PMC456",
        "isOpenAccess": "Y",
        "hasSuppl": "Y"
      }]}
    }"""
    resolution = fetcher.europe_pmc_doi_resolution(payload, "10.1021/JACS.EXAMPLE")
    assert resolution == {
        "doi_exact_match": True,
        "pmid": "123",
        "pmcid": "PMC456",
        "is_open_access": True,
        "has_supplementary": True,
    }


def test_portable_diagnostic_removes_machine_local_repo_prefix(tmp_path: Path) -> None:
    missing = tmp_path / "tmp" / "source-cache" / "paper-a" / "article.pdf"

    diagnostic = fetcher.portable_diagnostic(
        f"offline source cache missing: {missing}", tmp_path
    )

    assert diagnostic == (
        "offline source cache missing: tmp/source-cache/paper-a/article.pdf"
    )


def test_nature_supplementary_filename_maps_to_static_content_endpoint() -> None:
    rows = fetcher.supplementary_download_urls(
        {"supplementary_links": ["41467_2026_70735_MOESM1_ESM.pdf"]},
        "10.1038/s41467-026-70735-2",
    )
    assert rows == [
        (
            "https://static-content.springer.com/esm/"
            "art%3A10.1038%2Fs41467-026-70735-2/MediaObjects/"
            "41467_2026_70735_MOESM1_ESM.pdf",
            "41467_2026_70735_MOESM1_ESM.pdf",
        )
    ]


def test_authorized_cache_imports_verified_fulltext_and_si(tmp_path: Path) -> None:
    paper = {
        "paper_id": "paper-a",
        "doi": "10.1002/example",
        "source_url": "https://doi.org/10.1002/example",
    }
    root = tmp_path / "authorized" / "paper-a"
    article = root / "article"
    supplement = article / "supplementary_materials"
    supplement.mkdir(parents=True)
    html = ("10.1002/example full article reaction procedure " * 100).encode()
    pdf = b"%PDF-1.7\n" + b"route" * 100
    (article / "article.html").write_bytes(html)
    (supplement / "si.pdf").write_bytes(pdf)
    artifacts = []
    for path in (article / "article.html", supplement / "si.pdf"):
        artifacts.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "authorized-literature-fetch.json").write_text(
        json.dumps({"accepted": True, "doi": paper["doi"], "artifacts": artifacts}),
        encoding="utf-8",
    )
    rows, errors = fetcher.authorized_source_artifacts(tmp_path, tmp_path / "authorized", paper)
    assert errors == []
    assert {row["artifact_kind"] for row in rows} == {
        "authorized_publisher_fulltext_html",
        "supporting_information",
    }


def test_authorized_cache_accepts_immutable_versioned_article_root(
    tmp_path: Path,
) -> None:
    paper = {
        "paper_id": "paper-a",
        "doi": "10.1002/example",
        "source_url": "https://doi.org/10.1002/example",
    }
    root = tmp_path / "authorized" / "paper-a"
    article = root / "versions" / "refetch-1" / "article"
    article.mkdir(parents=True)
    html = ("10.1002/example full article reaction procedure " * 100).encode()
    (article / "article.html").write_bytes(html)
    relative = "versions/refetch-1/article/article.html"
    (root / "authorized-literature-fetch.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "doi": paper["doi"],
                "artifacts": [
                    {
                        "relative_path": relative,
                        "size_bytes": len(html),
                        "sha256": hashlib.sha256(html).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows, errors = fetcher.authorized_source_artifacts(tmp_path, tmp_path / "authorized", paper)

    assert errors == []
    assert [row["artifact_kind"] for row in rows] == ["authorized_publisher_fulltext_html"]

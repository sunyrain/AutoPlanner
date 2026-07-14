from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any
import zipfile

from cascade_planner.interfaces.literature_evidence import (
    BuiltinLiteratureEvidenceConfig,
    _interleave_candidates,
    _request_source_candidates,
    build_builtin_literature_evidence_connector,
)
from cascade_planner.interfaces.literature_candidates import (
    target_relevant_candidates,
)
from cascade_planner.interfaces.live_evidence import compose_evidence_connectors
from cascade_planner.interfaces import literature_html
from cascade_planner.interfaces.visual_evidence import compile_visual_evidence_request
from cascade_planner.interfaces.literature_search import (
    europe_pmc_metadata_search,
    europe_pmc_open_access_fulltext,
    europe_pmc_open_access_pdf,
    europe_pmc_repository_html,
    primary_literature_search,
)
from cascade_planner.harness.local_pdf_proxy import (
    local_pdf_proxy_download_manifest_path,
    local_pdf_proxy_request_queue_path,
)


def _request() -> dict[str, Any]:
    return {
        "schema_version": "evidence_acquisition_request.v1",
        "content_sha256": "a" * 64,
        "target_name": "bufotalin",
        "target_smiles": "CC",
        "edges": [{"edge_id": "edge:one"}],
        "source_tasks": [
            {
                "query": "bufotalin total synthesis",
                "source_types": ["paper_si"],
            }
        ],
    }


def test_literature_candidates_are_interleaved_across_queries() -> None:
    rows = _interleave_candidates(
        [
            [{"doi": "name-1"}, {"doi": "name-2"}],
            [{"doi": "route-1"}, {"doi": "route-2"}],
        ]
    )

    assert [row["doi"] for row in rows] == [
        "name-1",
        "route-1",
        "name-2",
        "route-2",
    ]


def test_target_relevance_rejects_clinical_metadata_and_keeps_route_sources() -> None:
    rows = target_relevant_candidates(
        [
            {
                "doi": "10.1/clinical",
                "title": "Cholesterol absorption and synthesis during pravastatin",
            },
            {
                "doi": "10.1/route",
                "title": "An asymmetric synthesis of pravastatin",
            },
            {
                "doi": "10.1/other",
                "title": "Total synthesis of an unrelated natural product",
            },
            {
                "doi": "10.1/channel",
                "title": (
                    "Cholesterol synthesis inhibitors pravastatin and "
                    "triparanol regulate channel function"
                ),
            },
            {
                "doi": "10.1/fibroblast",
                "title": (
                    "Residual cholesterol synthesis and pravastatin induction "
                    "in syndrome fibroblasts"
                ),
            },
            {
                "doi": "10.1/lipid",
                "title": (
                    "Pravastatin enhances linoleic acid conversion and "
                    "triglyceride synthesis"
                ),
            },
            {
                "doi": "10.1/chitosan",
                "title": (
                    "Synthesis and properties of mucoadhesive thiolated "
                    "chitosan for pravastatin"
                ),
            },
        ],
        target_name="pravastatin",
    )

    assert [row["doi"] for row in rows] == ["10.1/route"]


def test_europe_pmc_metadata_search_normalizes_papers_and_patents() -> None:
    class Response:
        content = b"{}"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "resultList": {
                    "result": [
                        {
                            "source": "MED",
                            "doi": "10.1128/AEM.02820-06",
                            "pmcid": "PMC1855665",
                            "isOpenAccess": "Y",
                            "inPMC": "Y",
                            "title": "Efficient synthesis of simvastatin",
                        },
                        {
                            "source": "PAT",
                            "id": "WO2011044496",
                            "title": "LovD mutants for simvastatin synthesis",
                        },
                    ]
                }
            }

    rows = europe_pmc_metadata_search(
        "simvastatin LovD",
        5,
        requester=lambda *_args, **_kwargs: Response(),
    )

    assert rows[0]["doi"] == "10.1128/AEM.02820-06"
    assert rows[0]["is_open_access"] is True
    assert rows[0]["has_repository_fulltext"] is True
    assert rows[0]["access_class"] == "open_access"
    assert rows[1]["publication_number"] == "WO2011044496"
    assert rows[1]["source_ref"] == "patent:WO2011044496"


def test_primary_literature_search_rewrites_named_synthesis_and_ranks_route_source(
    monkeypatch: Any,
) -> None:
    observed: list[str] = []

    def metadata(query: str, _limit: int):
        observed.append(query)
        return [
            {
                "doi": "10.1/therapy",
                "title": "Simvastatin therapy in cancer",
            },
            {
                "doi": "10.1/synthesis",
                "title": "Efficient synthesis of simvastatin by biocatalysis",
            },
        ]

    monkeypatch.setattr(
        "cascade_planner.interfaces.literature_search.europe_pmc_metadata_search",
        metadata,
    )

    rows = primary_literature_search('"Simvastatin" synthesis', 1)

    assert observed == ['TITLE:"synthesis of Simvastatin"']
    assert rows[0]["doi"] == "10.1/synthesis"


def test_verified_literature_refs_become_direct_candidates() -> None:
    request = _request()
    request["source_tasks"][0]["source_refs"] = [
        "doi:10.1128/AEM.02820-06",
        "https://example.test/supporting-information.pdf",
        "patent:US8211664B2",
    ]

    assert _request_source_candidates(request) == [
        {
            "doi": "10.1128/AEM.02820-06",
            "title": "10.1128/AEM.02820-06",
            "source_ref": "doi:10.1128/AEM.02820-06",
        },
        {
            "pdf_url": "https://example.test/supporting-information.pdf",
            "title": "https://example.test/supporting-information.pdf",
            "source_ref": "https://example.test/supporting-information.pdf",
        },
    ]


def test_europe_pmc_open_access_resolver_reads_nested_si_pdf() -> None:
    nested_buffer = BytesIO()
    with zipfile.ZipFile(nested_buffer, "w") as nested:
        nested.writestr("paper-supplementary.pdf", b"%PDF-1.7\nopen-access-si")
    outer_buffer = BytesIO()
    with zipfile.ZipFile(outer_buffer, "w") as outer:
        outer.writestr("paper-s001.zip", nested_buffer.getvalue())

    def fetch(url: str, _timeout: float, _maximum: int) -> bytes:
        if "search?" in url:
            return json.dumps(
                {
                    "resultList": {
                        "result": [
                            {
                                "doi": "10.1000/open",
                                "pmcid": "PMC123",
                                "isOpenAccess": "Y",
                            }
                        ]
                    }
                }
            ).encode()
        return outer_buffer.getvalue()

    content, receipt = europe_pmc_open_access_pdf(
        "10.1000/open",
        timeout_s=5.0,
        max_bytes=1_000_000,
        fetch=fetch,
    )

    assert content.startswith(b"%PDF-")
    assert receipt["pmcid"] == "PMC123"
    assert receipt["archive_member"] == (
        "paper-s001.zip!/paper-supplementary.pdf"
    )


def test_europe_pmc_resolver_returns_structured_fulltext_and_figure_archive() -> None:
    archive_buffer = BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("figure-1.jpg", b"\xff\xd8figure")
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <article xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta><article-id pub-id-type="doi">10.1000/open</article-id>
      </article-meta></front><body><fig><graphic xlink:href="figure-1.jpg"/></fig>
      </body></article>"""

    def fetch(url: str, _timeout: float, _maximum: int) -> bytes:
        if "search?" in url:
            return json.dumps(
                {
                    "resultList": {
                        "result": [
                            {
                                "doi": "10.1000/open",
                                "pmcid": "PMC123",
                                "isOpenAccess": "Y",
                            }
                        ]
                    }
                }
            ).encode()
        if "fullTextXML" in url:
            return xml
        return archive_buffer.getvalue()

    fulltext, archive, receipt = europe_pmc_open_access_fulltext(
        "10.1000/open",
        timeout_s=5.0,
        max_bytes=1_000_000,
        fetch=fetch,
    )

    assert fulltext == xml
    assert archive.startswith(b"PK")
    assert receipt["pmcid"] == "PMC123"
    assert receipt["fulltext_sha256"] == hashlib.sha256(xml).hexdigest()
    assert receipt["archive_error"] == ""


def test_europe_pmc_resolver_accepts_free_pmc_fulltext_without_oa_licence() -> None:
    xml = b"""<?xml version="1.0"?><article><front><article-meta>
    <article-id pub-id-type="doi">10.1128/aem.02820-06</article-id>
    </article-meta></front><body><p>full text</p></body></article>"""

    def fetch(url: str, _timeout: float, _maximum: int) -> bytes:
        if "search?" in url:
            return json.dumps(
                {
                    "resultList": {
                        "result": [
                            {
                                "doi": "10.1128/aem.02820-06",
                                "pmcid": "PMC1855665",
                                "isOpenAccess": "N",
                                "inEPMC": "Y",
                                "inPMC": "Y",
                            }
                        ]
                    }
                }
            ).encode()
        if "fullTextXML" in url:
            return xml
        raise OSError("figure archive unavailable")

    fulltext, archive, receipt = europe_pmc_open_access_fulltext(
        "10.1128/AEM.02820-06",
        timeout_s=5.0,
        max_bytes=1_000_000,
        fetch=fetch,
    )

    assert fulltext == xml
    assert archive == b""
    assert receipt["pmcid"] == "PMC1855665"
    assert receipt["open_access"] is False
    assert receipt["repository_fulltext"] is True
    assert receipt["access_class"] == "free_repository_fulltext"


def test_europe_pmc_repository_html_preserves_non_oa_access_semantics() -> None:
    html = b"""<!doctype html><html><head>
    <meta name="citation_doi" content="10.1128/AEM.02820-06"></head>
    <body><main><h2>Materials and methods</h2><p>Simvastatin was purified
    after the reaction mixture was incubated for two hours.</p></main></body></html>"""

    def fetch(url: str, _timeout: float, _maximum: int) -> bytes:
        if "search?" in url:
            return json.dumps(
                {
                    "resultList": {
                        "result": [
                            {
                                "doi": "10.1128/aem.02820-06",
                                "pmcid": "PMC1855665",
                                "isOpenAccess": "N",
                                "inPMC": "Y",
                            }
                        ]
                    }
                }
            ).encode()
        assert url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC1855665/"
        return html

    content, receipt = europe_pmc_repository_html(
        "10.1128/AEM.02820-06",
        timeout_s=5.0,
        max_bytes=1_000_000,
        fetch=fetch,
    )

    assert content == html
    assert receipt["open_access"] is False
    assert receipt["repository_fulltext"] is True
    assert receipt["html_sha256"] == hashlib.sha256(html).hexdigest()


def test_literature_connector_uses_pmc_html_before_pdf_or_browser(
    tmp_path: Path,
) -> None:
    html = b"""<!doctype html><html><head>
    <meta name="citation_doi" content="10.1128/AEM.02820-06"></head><body>
    <h2>Materials and methods</h2><h3>Synthesis of DMB-S-MMP</h3>
    <p>Dimethylbutyryl chloride was added slowly at 0 degrees C and the
    reaction mixture was stirred for 2 h, purified by chromatography, and
    isolated in 81 percent yield for simvastatin production.</p></body></html>"""
    calls: list[str] = []

    def fetch(url: str, _timeout: float, _maximum: int) -> bytes:
        calls.append(url)
        if "search?" in url:
            return json.dumps(
                {
                    "resultList": {
                        "result": [
                            {
                                "doi": "10.1128/aem.02820-06",
                                "pmcid": "PMC1855665",
                                "isOpenAccess": "N",
                                "inEPMC": "Y",
                            }
                        ]
                    }
                }
            ).encode()
        if "fullTextXML" in url:
            raise OSError("legacy deposit has no Europe PMC XML")
        if "pmc.ncbi.nlm.nih.gov" in url:
            return html
        raise AssertionError("PDF and browser fallback must not run")

    connector = build_builtin_literature_evidence_connector(
        BuiltinLiteratureEvidenceConfig(
            cache_dir=tmp_path / "cache",
            seed_dois=("10.1128/AEM.02820-06",),
            max_sources=1,
        ),
        searcher=lambda _query, _limit: [],
        fetcher=fetch,
    )
    result = connector(_request())

    source = result["discovery"]["sources"][0]
    assert source["acquisition_method"] == "pmc_repository_fulltext_html"
    assert source["pmcid"] == "PMC1855665"
    assert source["procedure_inventory"][0]["source_artifact_kind"] == (
        "pmc_fulltext_html"
    )
    assert not any("doi.org" in url for url in calls)

    def offline_fetch(_url: str, _timeout: float, _maximum: int) -> bytes:
        raise OSError("network unavailable during deterministic replay")

    cached_connector = build_builtin_literature_evidence_connector(
        BuiltinLiteratureEvidenceConfig(
            cache_dir=tmp_path / "cache",
            seed_dois=("10.1128/AEM.02820-06",),
            max_sources=1,
        ),
        searcher=lambda _query, _limit: [],
        fetcher=offline_fetch,
    )
    cached = cached_connector(_request())
    cached_source = cached["discovery"]["sources"][0]

    assert cached_source["acquisition_method"] == "pmc_repository_fulltext_html"
    assert cached_source["acquisition_receipt"]["cache_hit"] is True
    assert cached["receipt"]["queued_source_count"] == 0


def test_literature_connector_uses_isolated_browser_after_pmc_challenge(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    challenge = (
        b'<!doctype html><html><head><base href="https://www.google.com/'
        b'recaptcha/challenge"></head><body>'
        + (b"challenge " * 32)
        + b"</body></html>"
    )
    full_html = b"""<!doctype html><html><head>
    <meta name="citation_doi" content="10.1128/AEM.02820-06"></head><body>
    <h2>Materials and methods</h2><h3>Whole-cell biocatalysis</h3>
    <p>Simvastatin production was incubated for 24 h and the reaction mixture
    was purified by chromatography in 81 percent yield.</p></body></html>"""
    browser_calls: list[str] = []

    def fetch(url: str, _timeout: float, _maximum: int) -> bytes:
        if "search?" in url:
            return json.dumps(
                {
                    "resultList": {
                        "result": [
                            {
                                "doi": "10.1128/aem.02820-06",
                                "pmcid": "PMC1855665",
                                "isOpenAccess": "N",
                                "inPMC": "Y",
                            }
                        ]
                    }
                }
            ).encode()
        if "fullTextXML" in url:
            raise OSError("legacy deposit has no Europe PMC XML")
        if "pmc.ncbi.nlm.nih.gov" in url:
            return challenge
        raise AssertionError(f"unexpected URL: {url}")

    def browser_fetch(url: str, _timeout: float, _maximum: int) -> bytes:
        browser_calls.append(url)
        return full_html

    monkeypatch.setattr(
        literature_html,
        "fetch_repository_html_with_browser",
        browser_fetch,
    )
    source = literature_html.materialize_pmc_repository_html(
        {
            "doi": "10.1128/AEM.02820-06",
            "title": "Efficient synthesis of simvastatin by use of whole-cell biocatalysis.",
        },
        request=_request(),
        source_ref="doi:10.1128/AEM.02820-06",
        source_dir=tmp_path / "source",
        fulltext_cache_dir=tmp_path / "cache",
        config=BuiltinLiteratureEvidenceConfig(cache_dir=tmp_path / "evidence"),
        fetch=fetch,
    )
    assert source["acquisition_method"] == "pmc_repository_fulltext_html"
    assert source["acquisition_receipt"]["transport"] == (
        "isolated_playwright_repository_fallback"
    )
    assert source["acquisition_receipt"]["http_challenge_sha256"] == (
        hashlib.sha256(challenge).hexdigest()
    )
    assert browser_calls == [
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC1855665/"
    ]


def test_literature_connector_uses_structured_fulltext_and_original_figures_before_pdf(
    tmp_path: Path,
) -> None:
    archive_buffer = BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("figure-1.jpg", b"\xff\xd8\xff\xe0original-figure")
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <article xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta><article-id pub-id-type="doi">10.1000/monacolin</article-id>
      </article-meta></front><body>
        <sec><title>Materials and Methods</title><p>The monacolin J reaction
        mixture was incubated for 2 h and purified in 81% yield.</p></sec>
        <fig id="f1"><label>Figure 1</label><caption><p>Chemical structure of
        monacolin J and its biotransformation product.</p></caption>
        <graphic xlink:href="figure-1.jpg"/></fig>
      </body></article>"""
    calls: list[str] = []

    def fetch(url: str, _timeout: float, _maximum: int) -> bytes:
        calls.append(url)
        if "search?" in url:
            return json.dumps(
                {
                    "resultList": {
                        "result": [
                            {
                                "doi": "10.1000/monacolin",
                                "pmcid": "PMC456",
                                "isOpenAccess": "Y",
                            }
                        ]
                    }
                }
            ).encode()
        if "fullTextXML" in url:
            return xml
        if "supplementaryFiles" in url:
            return archive_buffer.getvalue()
        raise AssertionError("PDF fallback must not be fetched")

    request = _request()
    request["target_name"] = "monacolin J"
    connector = build_builtin_literature_evidence_connector(
        BuiltinLiteratureEvidenceConfig(
            cache_dir=tmp_path / "cache",
            max_sources=1,
        ),
        searcher=lambda _query, _limit: [
            {
                "doi": "10.1000/monacolin",
                "title": "A monacolin J biotransformation",
            }
        ],
        fetcher=fetch,
    )

    result = connector(request)

    source = result["discovery"]["sources"][0]
    assert source["acquisition_method"] == "europe_pmc_structured_fulltext_xml"
    assert source["source_fulltext_sha256"] == hashlib.sha256(xml).hexdigest()
    assert source["source_pdf_sha256"] == ""
    assert source["procedure_inventory"][0]["source_artifact_kind"] == (
        "europe_pmc_fulltext_xml"
    )
    figure = source["visual_candidate_pages"][0]
    assert Path(figure["image_path"]).read_bytes() == b"\xff\xd8\xff\xe0original-figure"
    assert figure["caption"].startswith("Figure 1 Chemical structure")
    assert not any(".pdf" in value.casefold() for value in calls)
    assert result["receipt"]["audits"][0]["fulltext_sha256"]
    visual_request = compile_visual_evidence_request(
        evidence_request=request,
        discovery=result["discovery"],
        max_pages=2,
    )
    assert visual_request["source"]["source_artifact_kind"] == (
        "europe_pmc_fulltext_xml"
    )
    assert visual_request["source"]["source_pdf_sha256"] == ""
    assert visual_request["source"]["source_artifact_sha256"] == (
        source["source_fulltext_sha256"]
    )
    assert visual_request["source"]["expected_labels"] == []
    network_call_count = len(calls)
    repeated = connector(request)
    assert len(calls) == network_call_count
    assert repeated["discovery"]["sources"][0]["acquisition_receipt"][
        "cache_hit"
    ] is True


def test_literature_connector_discovers_freezes_and_focuses_pdf(
    tmp_path: Path, monkeypatch: Any
) -> None:
    page = tmp_path / "page-7.png"
    page.write_bytes(b"page")
    page_sha = hashlib.sha256(page.read_bytes()).hexdigest()

    def fake_materialize(**_kwargs: Any) -> dict[str, Any]:
        return {
            "accepted": True,
            "rendered_pages": [
                {
                    "page_number": 7,
                    "image_path": str(page),
                    "sha256": page_sha,
                }
            ],
            "focus_page_numbers": [7],
        }

    monkeypatch.setattr(
        "cascade_planner.interfaces.literature_materialization.extract_literature_pdf_assets",
        fake_materialize,
    )
    monkeypatch.setattr(
        "cascade_planner.interfaces.literature_materialization.pdf_page_count",
        lambda _path: 12,
    )
    monkeypatch.setattr(
        "cascade_planner.interfaces.literature_materialization.rebuild_literature_pdf_page_focus",
        lambda *_args, **_kwargs: {"focus_page_numbers": [7]},
    )
    connector = build_builtin_literature_evidence_connector(
        BuiltinLiteratureEvidenceConfig(cache_dir=tmp_path / "cache"),
        searcher=lambda query, limit: [
            {
                "doi": "10.1000/bufotalin",
                "title": query,
                "pdf_url": "https://example.test/paper.pdf",
            }
        ],
        fetcher=lambda url, timeout, maximum: b"%PDF-1.7\nfixture",
    )

    result = connector(_request())

    source = result["discovery"]["sources"][0]
    assert source["source_kind"] == "paper_si"
    assert source["source_ref"] == "doi:10.1000/bufotalin"
    assert source["visual_candidate_pages"][0]["page_number"] == 7
    assert source["exact_row_count"] == 0
    assert result["receipt"]["model_invocations"] == 0


def test_composed_connector_keeps_paper_when_patent_provider_fails() -> None:
    def failed(_request: Any) -> Any:
        raise ValueError("patent unavailable")

    def paper(request: Any) -> dict[str, Any]:
        discovery = {
            "schema_version": "source_discovery_observation.v1",
            "provider_id": "paper",
            "request_sha256": request["content_sha256"],
            "sources": [{"source_kind": "paper_si", "source_ref": "doi:10.1/x"}],
        }
        return {"discovery": discovery, "receipt": {"provider_id": "paper"}}

    result = compose_evidence_connectors(failed, paper)(_request())

    assert result["discovery"]["sources"][0]["source_ref"] == "doi:10.1/x"
    assert result["receipt"]["failures"]


def test_restricted_paper_is_queued_then_consumed_on_resume(
    tmp_path: Path, monkeypatch: Any
) -> None:
    proxy_root = tmp_path / "authorized-proxy"
    config = BuiltinLiteratureEvidenceConfig(
        cache_dir=tmp_path / "cache",
        authorized_proxy_output_dir=proxy_root,
        max_sources=1,
    )
    candidate = {
        "doi": "10.1000/restricted",
        "title": "Restricted synthesis route for bufotalin",
        "pdf_url": "https://publisher.test/restricted.pdf",
    }
    connector = build_builtin_literature_evidence_connector(
        config,
        searcher=lambda _query, _limit: [candidate],
        fetcher=lambda _url, _timeout, _maximum: b"<html>institution login</html>",
    )

    queued = connector(_request())

    assert queued["receipt"]["queued_source_count"] == 1
    assert queued["discovery"]["sources"][0]["acquisition_status"] == (
        "queued_for_authorized_browser"
    )
    assert local_pdf_proxy_request_queue_path(proxy_root).is_file()

    downloaded = proxy_root / "paper.pdf"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_bytes(b"%PDF-1.7\nproxy fixture")
    manifest = local_pdf_proxy_download_manifest_path(proxy_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "{\"accepted\":true,\"status\":\"downloaded\","
        "\"doi\":\"10.1000/restricted\","
        f"\"pdf_path\":{downloaded.as_posix()!r}}}\n".replace("'", '"'),
        encoding="utf-8",
    )
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "cascade_planner.interfaces.literature_materialization.pdf_page_count",
        lambda _path: 10,
    )
    monkeypatch.setattr(
        "cascade_planner.interfaces.literature_materialization.rebuild_literature_pdf_page_focus",
        lambda *_args, **_kwargs: {"focus_page_numbers": [8]},
    )
    monkeypatch.setattr(
        "cascade_planner.interfaces.literature_materialization.extract_literature_pdf_assets",
        lambda **_kwargs: {
            "rendered_pages": [
                {"page_number": 8, "image_path": str(image), "sha256": image_sha}
            ],
            "focus_page_numbers": [8],
        },
    )

    resumed = connector(_request())

    assert resumed["receipt"]["accepted_source_count"] == 1
    assert resumed["discovery"]["sources"][0]["acquisition_status"] == "materialized"

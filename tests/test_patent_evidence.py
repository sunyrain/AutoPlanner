from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw
import pytest

from cascade_planner.interfaces import patent_evidence
from cascade_planner.interfaces.live_evidence import LiveEvidenceConnectorError
from cascade_planner.interfaces.patent_evidence import (
    BuiltinPatentEvidenceConfig,
    build_builtin_patent_evidence_connector,
)


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Example 1. Preparation of ethyl acetate")
    value = document.tobytes()
    document.close()
    return value


def _image_only_pdf_bytes() -> bytes:
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (60, 80),
        "Ethyl acetate (T1). Ethanol and acetic acid were added.",
        fill="black",
    )
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_image(page.rect, stream=image_buffer.getvalue())
    value = document.tobytes()
    document.close()
    return value


def _patent_html_bytes(publication: str = "US1234567A1") -> bytes:
    return f"""
    <html><head><meta name="DC.relation" content="{publication}"></head>
    <body>
      <div id="p0001" class="description-paragraph">Example 1</div>
      <div id="p0002" class="description-paragraph">
        Ethyl acetate (T1). Ethanol and acetic acid were added and the
        reaction mixture was stirred to afford T1 in 85 percent yield.
      </div>
      <div id="p0003" class="description-paragraph">
        The product was isolated and characterized.
      </div>
    </body></html>
    """.encode()


def _request(*, validated: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "evidence_acquisition_request.v1",
        "run_id": "blind-patent",
        "target_name": "ethyl acetate",
        "target_smiles": "CCOC(C)=O",
        "content_sha256": "a" * 64,
        "edges": [
            {
                "edge_id": "edge:ester",
                "edge_digest": "b" * 64,
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)O"],
                "current_host_reaction_validated": validated,
            }
        ],
        "source_tasks": [
            {
                "query": "ethyl acetate synthesis patent",
                "priority": 1.0,
            }
        ],
        "source_hints": [],
    }


def _compiler(steps: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
    records = []
    for step in steps:
        records.append(
            {
                "accepted": True,
                "step_id": step["step_id"],
                "binding": {
                    "page_number": 1,
                    "image_sha256": "c" * 64,
                    "synthesis_projection": {
                        "product_smiles": step["product_smiles"],
                        "reactant_smiles": sorted(step["reactant_smiles"]),
                    },
                },
            }
        )
    return {
        "schema_version": "deterministic_literature_registry_audit.v1",
        "content_sha256": "d" * 64,
        "records": records,
    }


def test_builtin_patent_connector_freezes_independent_pdfs_and_emits_exact_rows(
    tmp_path: Path,
) -> None:
    def candidates(_queries: Any) -> list[dict[str, Any]]:
        return [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:one",
                "title": "Process for preparation of ethyl acetate",
                "snippet": "ethyl acetate synthesis",
                "pdf_url": "https://source.invalid/one.pdf",
            },
            {
                "publication_number": "EP1234567A1",
                "family_id": "family:one",
                "title": "Process for preparation of ethyl acetate",
                "snippet": "same patent family",
                "pdf_url": "https://source.invalid/duplicate.pdf",
            },
            {
                "publication_number": "WO7654321A1",
                "family_id": "family:two",
                "title": "Synthetic method for ethyl acetate",
                "snippet": "independent process",
                "pdf_url": "https://source.invalid/two.pdf",
            },
        ]

    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(cache_dir=tmp_path, max_patents=3),
        candidate_provider=candidates,
        bytes_fetcher=lambda _url, _timeout, _limit: _pdf_bytes(),
        registry_compiler=_compiler,
        structure_resolver=lambda _name: "CCOC(C)=O",
        candidate_name_resolver=lambda _smiles: ["ethyl acetate"],
    )
    result = connector(_request())

    sources = result["document"]["sources"]
    assert len(sources) == 2
    assert {
        source["binding"]["patent_family"] for source in sources
    } == {"family:one", "family:two"}
    assert all(
        source["extraction"]["rows"][0]["relation_type"] == "exact"
        for source in sources
    )
    assert result["receipt"]["model_invocations"] == 0
    assert result["receipt"]["candidate_count"] == 2
    assert len(list(tmp_path.rglob("*.pdf"))) == 2
    assert len(list(tmp_path.rglob("*.png"))) == 2


def test_builtin_patent_connector_reuses_hashed_source_bytes_across_runs(
    tmp_path: Path,
) -> None:
    fetch_count = 0

    def fetch_pdf(_url: str, _timeout: float, _limit: int) -> bytes:
        nonlocal fetch_count
        fetch_count += 1
        return _pdf_bytes()

    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(cache_dir=tmp_path, max_patents=1),
        candidate_provider=lambda _queries: [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:one",
                "title": "Process for preparation of ethyl acetate",
                "snippet": "ethyl acetate synthesis",
                "pdf_url": "https://source.invalid/one.pdf",
            }
        ],
        bytes_fetcher=fetch_pdf,
        registry_compiler=_compiler,
        structure_resolver=lambda _name: "CCOC(C)=O",
        candidate_name_resolver=lambda _smiles: ["ethyl acetate"],
    )

    first = connector(_request())
    second_request = _request()
    second_request["run_id"] = "blind-patent-independent-second-run"
    second = connector(second_request)

    assert fetch_count == 1
    assert first["receipt"]["audits"][0]["source_byte_cache"]["pdf"][
        "cache_hit"
    ] is False
    assert second["receipt"]["audits"][0]["source_byte_cache"]["pdf"][
        "cache_hit"
    ] is True
    assert second["discovery"]["sources"][0]["source_byte_cache"]["pdf"][
        "semantics"
    ]["target_derived_extraction_is_not_shared"] is True


def test_builtin_patent_connector_refuses_unvalidated_edges(tmp_path: Path) -> None:
    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(cache_dir=tmp_path),
        candidate_provider=lambda _queries: [],
    )

    with pytest.raises(
        LiveEvidenceConnectorError,
        match="no_validated_edges",
    ):
        connector(_request(validated=False))


def test_builtin_patent_connector_allows_authority_free_target_prefetch(
    tmp_path: Path,
) -> None:
    request = _request()
    request["edges"] = []
    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(cache_dir=tmp_path, max_patents=1),
        candidate_provider=lambda _queries: [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:one",
                "title": "Process for preparation of ethyl acetate",
                "snippet": "ethyl acetate synthesis",
                "pdf_url": "https://source.invalid/one.pdf",
            }
        ],
        bytes_fetcher=lambda _url, _timeout, _limit: _pdf_bytes(),
        registry_compiler=_compiler,
    )

    result = connector(request)

    assert "document" not in result
    assert result["receipt"]["accepted_source_count"] == 0
    assert result["receipt"]["semantics"][
        "target_only_prefetch_grants_no_evidence_authority"
    ] is True
    source = result["discovery"]["sources"][0]
    assert source["pdf_sha256"]
    assert source["exact_row_count"] == 0
    assert source["approved_exact_row_count"] == 0
    assert source["source_route_exact_row_count"] == 0


def test_builtin_patent_connector_returns_unbound_procedures_for_global_replan(
    tmp_path: Path,
) -> None:
    def rejected_compiler(
        steps: list[dict[str, Any]], **_: Any
    ) -> dict[str, Any]:
        return {
            "schema_version": "deterministic_literature_registry_audit.v1",
            "content_sha256": "d" * 64,
            "records": [
                {
                    "accepted": False,
                    "step_id": step["step_id"],
                    "reasons": ["product_not_reconstructed_from_source_heading"],
                }
                for step in steps
            ],
            "source_procedure_inventory": [
                {
                    "procedures": [
                        {
                            "label": "1",
                            "name": "ethyl acetate",
                            "page_number": 1,
                            "procedure_excerpt": "Ethanol and acetic acid were combined.",
                        }
                    ]
                }
            ],
        }

    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(cache_dir=tmp_path, max_patents=1),
        candidate_provider=lambda _queries: [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:one",
                "title": "Alternative preparation of ethyl acetate",
                "snippet": "source material only",
                "pdf_url": "https://source.invalid/one.pdf",
            }
        ],
        bytes_fetcher=lambda _url, _timeout, _limit: _pdf_bytes(),
        registry_compiler=rejected_compiler,
        structure_resolver=lambda _name: "",
        candidate_name_resolver=lambda _smiles: ["ethyl acetate"],
    )
    result = connector(_request())

    assert "document" not in result
    assert result["receipt"]["accepted_source_count"] == 0
    observation = result["discovery"]["sources"][0]
    assert observation["exact_row_count"] == 0
    assert observation["procedure_inventory"][0]["name"] == "ethyl acetate"
    assert result["discovery"]["semantics"][
        "discovery_does_not_grant_exact_evidence"
    ] is True


def test_builtin_patent_connector_reuses_persistent_default_resolver_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def resolvers(**kwargs: Any) -> tuple[Any, Any]:
        observed.update(kwargs)
        return (
            lambda _name: "CCOC(C)=O",
            lambda _smiles: ["ethyl acetate"],
        )

    monkeypatch.setattr(
        patent_evidence,
        "build_deterministic_literature_resolvers",
        resolvers,
    )
    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(cache_dir=tmp_path, max_patents=1),
        candidate_provider=lambda _queries: [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:one",
                "title": "Preparation of ethyl acetate",
                "snippet": "source material",
                "pdf_url": "https://source.invalid/one.pdf",
            }
        ],
        bytes_fetcher=lambda _url, _timeout, _limit: _pdf_bytes(),
        registry_compiler=_compiler,
    )

    result = connector(_request())

    assert observed["persistent_cache"] is not None
    assert result["receipt"]["resolver_cache"] == {
        "flushed": False,
        "entry_count": 0,
    }


def test_builtin_patent_connector_reuses_run_scoped_search_pdf_and_render(
    tmp_path: Path,
) -> None:
    calls = {"search": 0, "fetch": 0}

    def candidates(_queries: Any) -> list[dict[str, Any]]:
        calls["search"] += 1
        return [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:one",
                "title": "Preparation of ethyl acetate",
                "snippet": "source material",
                "pdf_url": "https://source.invalid/one.pdf",
            }
        ]

    def fetch(_url: str, _timeout: float, _limit: int) -> bytes:
        calls["fetch"] += 1
        return _pdf_bytes()

    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(cache_dir=tmp_path, max_patents=1),
        candidate_provider=candidates,
        bytes_fetcher=fetch,
        registry_compiler=_compiler,
        structure_resolver=lambda _name: "CCOC(C)=O",
        candidate_name_resolver=lambda _smiles: ["ethyl acetate"],
    )

    connector(_request())
    replayed = connector(_request())

    assert calls == {"search": 1, "fetch": 1}
    assert replayed["receipt"]["candidate_cache_hit"] is True
    audit = replayed["receipt"]["audits"][0]
    assert audit["pdf_cache_hit"] is True
    assert audit["manifest_cache_hit"] is True


def test_builtin_patent_connector_ocr_closes_image_only_exact_row_without_model(
    tmp_path: Path,
) -> None:
    structures = {
        "ethyl acetate": "CCOC(C)=O",
        "ethanol": "CCO",
        "acetic acid": "CC(=O)O",
    }
    names = {
        "CCOC(C)=O": ["ethyl acetate"],
        "CCO": ["ethanol"],
        "CC(=O)O": ["acetic acid"],
    }
    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(
            cache_dir=tmp_path,
            max_patents=1,
            max_ocr_pages=1,
        ),
        candidate_provider=lambda _queries: [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:ocr",
                "title": "Image-only preparation of ethyl acetate",
                "snippet": "primary source",
                "pdf_url": "https://source.invalid/scanned.pdf",
            }
        ],
        bytes_fetcher=lambda _url, _timeout, _limit: _image_only_pdf_bytes(),
        structure_resolver=lambda value: structures[str(value).casefold()],
        candidate_name_resolver=lambda value: names.get(str(value), []),
        ocr_runner=lambda *_args: {
            "text": (
                "Ethyl acetate (T1). Ethanol and acetic acid were added. "
                "The reaction mixture was stirred to afford T1."
            ),
            "engine_id": "tesseract",
            "engine_version": "fixture",
        },
    )

    result = connector(_request())

    assert result["receipt"]["model_invocations"] == 0
    assert result["document"]["sources"][0]["extraction"]["rows"][0][
        "relation_type"
    ] == "exact"
    discovery = result["discovery"]["sources"][0]
    assert discovery["ocr_audit"]["status"] == "completed"
    assert discovery["exact_row_count"] == 1
    assert discovery["approved_exact_row_count"] == 1
    assert discovery["unresolved_edge_count"] == 0


def test_builtin_patent_connector_html_closes_edge_without_fetching_pdf(
    tmp_path: Path,
) -> None:
    publication = "US1234567A1"

    def pdf_must_not_be_fetched(*_args: Any) -> bytes:
        raise AssertionError("PDF fallback must not run after HTML closure")

    def html_must_not_be_refetched(*_args: Any) -> bytes:
        raise AssertionError("prefetched publication HTML must be reused")

    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(cache_dir=tmp_path, max_patents=1),
        candidate_provider=lambda _queries: [
            {
                "publication_number": publication,
                "family_id": "family:html",
                "title": "Preparation of ethyl acetate",
                "snippet": "search metadata only",
                "html_url": (
                    f"https://patents.google.com/patent/{publication}/en"
                ),
                "_primary_html_bytes": _patent_html_bytes(),
                "pdf_url": "https://source.invalid/must-not-run.pdf",
            }
        ],
        bytes_fetcher=pdf_must_not_be_fetched,
        html_fetcher=html_must_not_be_refetched,
        structure_resolver=lambda name: {
            "Ethyl acetate": "CCOC(C)=O",
        }[name],
        candidate_name_resolver=lambda smiles: {
            "CCOC(C)=O": ["ethyl acetate"],
            "CCO": ["ethanol"],
            "CC(=O)O": ["acetic acid"],
        }.get(smiles, []),
    )

    result = connector(_request())

    source = result["document"]["sources"][0]
    row = source["extraction"]["rows"][0]
    discovery = result["discovery"]["sources"][0]
    assert source["binding"]["provenance"] == (
        "builtin_deterministic_primary_patent_html"
    )
    assert row["location_ref"].startswith(f"{publication}:html:p")
    assert {value.split(":", 1)[0] for value in row["evidence_refs"]} == {
        "html_sha256",
        "text_sha256",
    }
    assert discovery["html_sha256"]
    assert discovery["pdf_sha256"] == ""
    assert discovery["exact_row_count"] == 1
    assert list(tmp_path.rglob("*.pdf")) == []
    assert list(tmp_path.rglob("*.png")) == []


def test_builtin_patent_connector_pdf_fallback_receives_only_unresolved_html_edges(
    tmp_path: Path,
) -> None:
    request = _request()
    request["edges"].append(
        {
            "edge_id": "edge:second",
            "edge_digest": "e" * 64,
            "product_smiles": "CCO",
            "precursor_smiles": ["CC=O"],
            "current_host_reaction_validated": True,
        }
    )
    compiled_step_ids: list[list[str]] = []
    pdf_fetch_count = 0

    def compiler(steps: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        compiled_step_ids.append([str(step["step_id"]) for step in steps])
        is_pdf = bool(steps and steps[0].get("source_evidence"))
        selected = steps if is_pdf else steps[:1]
        records = []
        for step in selected:
            if is_pdf:
                evidence = dict(step["source_evidence"][0])
                artifact = {
                    "source_artifact_kind": "pdf",
                    "source_pdf_sha256": evidence["source_pdf_sha256"],
                    "page_number": evidence["page_number"],
                    "image_sha256": evidence["image_sha256"],
                }
            else:
                companion = dict(step["source_text_companions"][0])
                section = dict(companion["sections"][0])
                artifact = {
                    "source_artifact_kind": "html",
                    "source_artifact_sha256": companion["artifact_sha256"],
                    "source_location": {
                        "kind": "html_paragraph_range",
                        "start_element_id": section["start_element_id"],
                        "end_element_id": section["end_element_id"],
                        "text_sha256": "f" * 64,
                    },
                }
            records.append(
                {
                    "accepted": True,
                    "step_id": step["step_id"],
                    "binding": {
                        **artifact,
                        "synthesis_projection": {
                            "product_smiles": step["product_smiles"],
                            "reactant_smiles": sorted(step["reactant_smiles"]),
                        },
                    },
                }
            )
        return {
            "schema_version": "deterministic_literature_registry_audit.v1",
            "content_sha256": "d" * 64,
            "records": records,
        }

    def fetch_pdf(_url: str, _timeout: float, _limit: int) -> bytes:
        nonlocal pdf_fetch_count
        pdf_fetch_count += 1
        return _pdf_bytes()

    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(cache_dir=tmp_path, max_patents=1),
        candidate_provider=lambda _queries: [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:mixed",
                "title": "Mixed source",
                "html_url": (
                    "https://patents.google.com/patent/US1234567A1/en"
                ),
                "pdf_url": "https://source.invalid/fallback.pdf",
            }
        ],
        bytes_fetcher=fetch_pdf,
        html_fetcher=lambda _url, _timeout, _limit: _patent_html_bytes(),
        registry_compiler=compiler,
        structure_resolver=lambda _name: "CCOC(C)=O",
        candidate_name_resolver=lambda smiles: {
            "CCOC(C)=O": ["ethyl acetate"],
            "CCO": ["ethanol"],
            "CC(=O)O": ["acetic acid"],
            "CC=O": ["acetaldehyde"],
        }.get(smiles, []),
    )

    result = connector(request)

    assert compiled_step_ids == [
        ["edge:ester", "edge:second"],
        ["edge:second"],
    ]
    assert pdf_fetch_count == 1
    source = result["document"]["sources"][0]
    assert source["binding"]["provenance"] == (
        "builtin_patent_html_first_with_pdf_fallback"
    )
    assert {row["step_id"] for row in source["extraction"]["rows"]} == {
        "edge:ester",
        "edge:second",
    }
    assert result["discovery"]["sources"][0]["unresolved_edge_count"] == 0


def test_builtin_patent_connector_falls_back_when_html_registry_is_unavailable(
    tmp_path: Path,
) -> None:
    compiled_kinds: list[str] = []

    def compiler(steps: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if not steps[0].get("source_evidence"):
            compiled_kinds.append("html")
            raise RuntimeError("temporary HTML resolver outage")
        compiled_kinds.append("pdf")
        return _compiler(steps, **kwargs)

    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(cache_dir=tmp_path, max_patents=1),
        candidate_provider=lambda _queries: [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:fallback",
                "title": "Preparation of ethyl acetate",
                "html_url": (
                    "https://patents.google.com/patent/US1234567A1/en"
                ),
                "pdf_url": "https://source.invalid/fallback.pdf",
            }
        ],
        bytes_fetcher=lambda _url, _timeout, _limit: _pdf_bytes(),
        html_fetcher=lambda _url, _timeout, _limit: _patent_html_bytes(),
        registry_compiler=compiler,
        structure_resolver=lambda _name: "CCOC(C)=O",
        candidate_name_resolver=lambda _smiles: ["ethyl acetate"],
    )

    result = connector(_request())

    assert compiled_kinds == ["html", "pdf"]
    assert result["document"]["sources"][0]["binding"]["provenance"] == (
        "builtin_deterministic_patent_pdf_extraction"
    )
    discovery = result["discovery"]["sources"][0]
    assert discovery["html_audit"]["status"] == "failed"
    assert discovery["pdf_sha256"]
    assert discovery["exact_row_count"] == 1

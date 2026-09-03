from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz

from cascade_planner.harness.source_text_companion import (
    PRIMARY_HTML_AUTHORITY_MODE,
    SOURCE_TEXT_COMPANION_SPEC_SCHEMA,
    STRUCTURED_FULLTEXT_HTML_FORMAT,
    materialize_source_text_companion_pages,
    validate_source_text_companion_binding,
)
from cascade_planner.interfaces.literature_route_binding import (
    bind_materialized_literature_source,
)
from cascade_planner.interfaces.evidence_import import (
    validate_structured_evidence_document,
)


def _fixture(tmp_path: Path) -> tuple[Path, str, str]:
    excerpt = (
        "Acetic acid (60 mg, 1 mmol) and ethanol (46 mg, 1 mmol) were added "
        "to a flask. The reaction mixture was stirred at 25 C for 2 h and "
        "purified to yield ethyl acetate."
    )
    html = (
        "<!doctype html><html><body><p>PMC123</p>"
        "<h2>Synthesis of ethyl acetate from acetic acid and ethanol</h2>"
        f"<p>{excerpt}</p></body></html>"
    ).encode()
    path = tmp_path / "fulltext.html"
    path.write_bytes(html)
    return path, hashlib.sha256(html).hexdigest(), excerpt


def test_structured_fulltext_companion_is_hash_replayable(tmp_path: Path) -> None:
    path, digest, excerpt = _fixture(tmp_path)
    spec = {
        "schema_version": SOURCE_TEXT_COMPANION_SPEC_SCHEMA,
        "artifact_path": str(path),
        "artifact_sha256": digest,
        "document_identity": "PMC123",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
        "format": STRUCTURED_FULLTEXT_HTML_FORMAT,
        "authority_mode": PRIMARY_HTML_AUTHORITY_MODE,
        "sections": [
            {
                "page_number": 1,
                "label": "html-section-1",
                "name": "Synthesis of ethyl acetate from acetic acid and ethanol",
                "text": excerpt,
                "text_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
            }
        ],
    }

    pages, binding, reasons = materialize_source_text_companion_pages(
        spec,
        source_ref="doi:10.1000/example",
    )

    assert reasons == ()
    assert pages[0]["name"].startswith("Synthesis of ethyl acetate")
    assert validate_source_text_companion_binding(
        binding,
        expected_source_ref="doi:10.1000/example",
    )


def test_paper_procedure_compiles_route_and_exact_row(tmp_path: Path) -> None:
    path, digest, excerpt = _fixture(tmp_path)
    structures = {
        "ethyl acetate": "CCOC(C)=O",
        "acetic acid": "CC(=O)O",
        "ethanol": "CCO",
    }

    def resolve_structure(name: str) -> str:
        return structures.get(" ".join(name.casefold().split()), "")

    names = {value: [key] for key, value in structures.items()}
    source = {
        "source_kind": "paper_si",
        "source_ref": "doi:10.1000/example",
        "doi": "10.1000/example",
        "pmcid": "PMC123",
        "title": "A preparation of ethyl acetate",
        "source_fulltext_sha256": digest,
        "fulltext_html_path": str(path),
        "visual_candidate_pages": [],
        "procedure_inventory": [
            {
                "label": "html-section-1",
                "name": "Synthesis of ethyl acetate from acetic acid and ethanol",
                "page_number": 1,
                "procedure_excerpt": excerpt,
            }
        ],
        "acquisition_receipt": {
            "html_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
            "pmcid": "PMC123",
        },
    }
    request = {
        "edges": [
            {
                "edge_id": "edge:existing",
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CC(=O)O", "CCO"],
                "current_host_reaction_validated": True,
            }
        ]
    }

    enriched, structured, audit = bind_materialized_literature_source(
        source,
        request=request,
        output_dir=tmp_path / "registry",
        structure_resolver=resolve_structure,
        candidate_name_resolver=lambda smiles: names.get(smiles, []),
        timeout_s=1.0,
        provider_version="test",
    )

    assert audit["status"] == "completed"
    assert enriched["source_route_proposal_count"] == 1
    assert enriched["exact_row_count"] == 1
    assert structured["extraction"]["rows"][0]["relation_type"] == "exact"
    assert structured["binding"]["source_ref"] == "doi:10.1000/example"
    validated = validate_structured_evidence_document(
        {"schema_version": "structured_evidence_import.v1", "sources": [structured]}
    )
    assert len(validated["sources"]) == 1


def test_pdf_text_procedure_keeps_unresolved_route_observation(tmp_path: Path) -> None:
    _html_path, _html_digest, excerpt = _fixture(tmp_path)
    text_path = tmp_path / "fulltext.txt"
    text_path.write_text(excerpt, encoding="utf-8")
    digest = hashlib.sha256(text_path.read_bytes()).hexdigest()
    structures = {
        "ethyl acetate": "CCOC(C)=O",
        "acetic acid": "CC(=O)O",
        "ethanol": "CCO",
    }
    names = {value: [key] for key, value in structures.items()}
    source = {
        "source_kind": "paper_si",
        "source_ref": "doi:10.1000/example.s001",
        "doi": "10.1000/example.s001",
        "title": "Supporting information for ethyl acetate",
        "source_fulltext_sha256": digest,
        "fulltext_text_path": str(text_path),
        "visual_candidate_pages": [],
        "procedure_inventory": [
            {
                "label": "pdf-section-1",
                "name": "Synthesis of ethyl acetate from acetic acid and ethanol",
                "page_number": 1,
                "procedure_excerpt": excerpt,
            }
        ],
        "acquisition_receipt": {},
    }
    request = {
        "edges": [
            {
                "edge_id": "edge:existing",
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CC(=O)O", "CCO"],
                "current_host_reaction_validated": True,
            }
        ]
    }

    enriched, structured, audit = bind_materialized_literature_source(
        source,
        request=request,
        output_dir=tmp_path / "registry",
        structure_resolver=lambda name: structures.get(name.casefold(), ""),
        candidate_name_resolver=lambda smiles: names.get(smiles, []),
        timeout_s=1.0,
        provider_version="test",
    )

    assert audit == {
        "status": "unresolved",
        "model_invocations": 0,
        "proposal_count": 1,
        "reason": "replayable_source_artifact_missing",
    }
    assert enriched["source_route_proposal_count"] == 1
    assert len(enriched["source_route_observation"]["proposals"]) == 1
    assert structured == {}


def test_hash_bound_native_pdf_procedure_compiles_exact_row(tmp_path: Path) -> None:
    excerpt = (
        "Acetic acid (60 mg, 1 mmol) and ethanol (46 mg, 1 mmol) were added "
        "to a flask. The reaction mixture was stirred at 25 C for 2 h and "
        "purified to yield ethyl acetate."
    )
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(72, 72, 520, 300),
        f"Ethyl acetate (T1). {excerpt}",
    )
    document.save(pdf_path)
    document.close()
    pdf_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    image_path = tmp_path / "page-1.png"
    with fitz.open(pdf_path) as replay:
        replay[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    source_ref = "doi:10.1000/native-pdf"
    document_id = "paper:10.1000/native-pdf"
    manifest_path = tmp_path / "literature_pdf_structure_evidence.json"
    manifest = {
        "schema_version": "literature_pdf_structure_evidence.v1",
        "accepted": True,
        "source_ref": source_ref,
        "source_pdf_path": str(pdf_path),
        "source_pdf_sha256": pdf_sha256,
        "rendered_pages": [
            {
                "page_number": 1,
                "image_path": str(image_path),
                "sha256": image_sha256,
            }
        ],
        "source_binding_audit": {
            "schema_version": "local_pdf_source_binding_audit.v1",
            "accepted": True,
            "source_ref": source_ref,
            "matched_source_count": 1,
            "matched_document_ids": [document_id],
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    text_path = tmp_path / "fulltext.txt"
    text_path.write_text(excerpt, encoding="utf-8")
    text_sha256 = hashlib.sha256(text_path.read_bytes()).hexdigest()
    evidence = {
        "schema_version": "materialized_source_evidence.v1",
        "document_id": document_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "source_pdf_path": str(pdf_path),
        "source_pdf_sha256": pdf_sha256,
        "page_number": 1,
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "source_ref": source_ref,
    }
    structures = {
        "ethyl acetate": "CCOC(C)=O",
        "acetic acid": "CC(=O)O",
        "ethanol": "CCO",
    }
    names = {value: [key] for key, value in structures.items()}
    source = {
        "source_kind": "paper_si",
        "source_ref": source_ref,
        "doi": "10.1000/native-pdf",
        "title": "Native PDF preparation",
        "source_fulltext_sha256": text_sha256,
        "fulltext_text_path": str(text_path),
        "procedure_inventory": [
            {
                "label": "T1",
                "name": "Preparation of ethyl acetate from acetic acid and ethanol",
                "page_number": 1,
                "procedure_excerpt": excerpt,
            }
        ],
        "source_evidence": [evidence],
        "visual_candidate_pages": [],
        "acquisition_receipt": {},
    }
    request = {
        "edges": [
            {
                "edge_id": "edge:existing",
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CC(=O)O", "CCO"],
                "current_host_reaction_validated": True,
            }
        ]
    }

    enriched, structured, audit = bind_materialized_literature_source(
        source,
        request=request,
        output_dir=tmp_path / "registry",
        structure_resolver=lambda name: structures.get(name.casefold(), ""),
        candidate_name_resolver=lambda smiles: names.get(smiles, []),
        timeout_s=1.0,
        provider_version="test",
    )

    assert audit["status"] == "completed"
    assert enriched["exact_row_count"] == 1
    exact = structured["extraction"]["rows"][0]
    assert exact["location_ref"] == f"{source_ref}:pdf:page-1"
    assert f"pdf_sha256:{pdf_sha256}" in exact["evidence_refs"]

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
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

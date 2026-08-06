from __future__ import annotations

import json
from typing import Any, Mapping

import pytest

from cascade_planner.interfaces.live_evidence import (
    HttpEvidenceConnectorConfig,
    LiveEvidenceConnectorError,
    acquire_structured_evidence,
    build_http_evidence_connector,
    compile_evidence_acquisition_request,
)


def _document(*, provider_id: str = "tests.extractor") -> dict:
    return {
        "schema_version": "structured_evidence_import.v1",
        "sources": [
            {
                "binding": {
                    "source_kind": "patent",
                    "source_ref": "patent:US1234567A1",
                    "title": "Primary patent example",
                    "provenance": "typed_connector",
                },
                "extraction": {
                    "schema_version": "structured_exact_row_extraction.v1",
                    "extractor": {
                        "producer_kind": "typed_connector_structured_extraction",
                        "producer_id": provider_id,
                        "version": "2.0.0",
                    },
                    "rows": [
                        {
                            "product_smiles": "CCOC(C)=O",
                            "reactant_smiles": ["CCO", "CC(=O)Cl"],
                            "location_ref": "Example 7, step 2",
                        }
                    ],
                },
            }
        ],
    }


def test_request_contains_bounded_current_edges_and_source_tasks() -> None:
    request = compile_evidence_acquisition_request(
        run_id="blind-1",
        target_name="ethyl acetate",
        target_smiles="CCOC(C)=O",
        graph={
            "revision": 4,
            "edges": {
                "edge:one": {
                    "edge_id": "edge:one",
                    "edge_digest": "a" * 64,
                    "product_smiles": "CCOC(C)=O",
                    "precursor_smiles": ["CCO", "CC(=O)Cl"],
                    "reaction_proofs": [
                        {
                            "accepted": True,
                            "validator_version": (
                                "autoplanner.reaction_step_verifier.v10"
                            ),
                        }
                    ],
                }
            },
            "route_families": {"route:one": {"edge_ids": ["edge:one"]}},
        },
        source_frontier={
            "source_plan": [
                {
                    "source_task_id": "source:one",
                    "query": " exact ester patent ",
                    "priority": 2.0,
                    "source_types": ["patent"],
                    "target_claims": ["exact reaction row"],
                }
            ]
        },
        target_identity={
            "preferred_name": "ethyl acetate",
            "patent_ids": ["WO-EXACT-1"],
            "pubmed_ids": ["123"],
        },
    )

    assert request["edges"][0]["current_host_reaction_validated"] is True
    assert request["source_tasks"][0]["query"] == "exact ester patent"
    assert request["target_name"] == "ethyl acetate"
    assert request["target_identity"]["patent_ids"] == ["WO-EXACT-1"]
    assert request["target_identity"]["resolved_from_input_structure"] is True
    assert request["source_tasks"][0]["priority"] == 1.0
    assert len(request["content_sha256"]) == 64


def test_http_connector_freezes_receipt_without_recording_token(monkeypatch: Any) -> None:
    monkeypatch.setenv("TEST_EVIDENCE_TOKEN", "secret-value")

    def requester(
        method: str,
        url: str,
        *,
        json_body: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_s: float,
        max_response_bytes: int,
    ) -> tuple[int, bytes, Mapping[str, Any]]:
        assert method == "POST"
        assert url == "http://127.0.0.1:8080/extract?tenant=test"
        assert json_body["schema_version"] == "evidence_acquisition_request.v1"
        assert headers["Authorization"] == "Bearer secret-value"
        assert timeout_s == 12.0
        assert max_response_bytes == 50_000
        content = json.dumps(_document(), sort_keys=True).encode()
        return 200, content, {"content-type": "application/json"}

    config = HttpEvidenceConnectorConfig(
        endpoint="http://127.0.0.1:8080/extract?tenant=test",
        provider_id="tests.extractor",
        provider_version="2.0.0",
        token_env="TEST_EVIDENCE_TOKEN",
        timeout_s=12.0,
        max_response_bytes=50_000,
    )
    connector = build_http_evidence_connector(config, requester=requester)
    request = compile_evidence_acquisition_request(
        run_id="blind-1",
        target_smiles="CCOC(C)=O",
        graph={"revision": 1, "edges": {}, "route_families": {}},
        source_frontier={},
    )
    result = acquire_structured_evidence(request, connector=connector)

    assert result["document"]["sources"]
    assert result["receipt"]["endpoint"] == "http://127.0.0.1:8080/extract"
    assert result["receipt"]["credential_recorded"] is False
    assert "secret-value" not in json.dumps(result, sort_keys=True)


def test_connector_rejects_public_plain_http_and_wrong_extractor_identity() -> None:
    with pytest.raises(ValueError, match="endpoint_invalid"):
        HttpEvidenceConnectorConfig(
            endpoint="http://evidence.invalid/extract",
            provider_id="tests.extractor",
            provider_version="2.0.0",
        )

    def wrong_identity(_request: Mapping[str, Any]) -> Mapping[str, Any]:
        return _document(provider_id="other.extractor")

    request = compile_evidence_acquisition_request(
        run_id="blind-1",
        target_smiles="CCOC(C)=O",
        graph={"revision": 1, "edges": {}, "route_families": {}},
        source_frontier={},
    )
    config = HttpEvidenceConnectorConfig(
        endpoint="https://evidence.invalid/extract",
        provider_id="tests.extractor",
        provider_version="2.0.0",
    )
    connector = build_http_evidence_connector(
        config,
        requester=lambda *_args, **_kwargs: (
            200,
            json.dumps(wrong_identity({})).encode(),
            {},
        ),
    )
    with pytest.raises(LiveEvidenceConnectorError, match="identity_mismatch"):
        acquire_structured_evidence(request, connector=connector)


def test_connector_accepts_bounded_discovery_without_promoting_exact_rows() -> None:
    request = compile_evidence_acquisition_request(
        run_id="blind-discovery",
        target_name="ethyl acetate",
        target_smiles="CCOC(C)=O",
        graph={"revision": 1, "edges": {}, "route_families": {}},
        source_frontier={},
    )

    def connector(_request: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "discovery": {
                "schema_version": "source_discovery_observation.v1",
                "provider_id": "tests.discovery",
                "request_sha256": request["content_sha256"],
                "sources": [
                    {
                        "publication_number": "US1234567A1",
                        "procedure_inventory": [
                            {
                                "label": "7",
                                "name": "ethyl acetate",
                                "procedure_excerpt": "Untrusted source text.",
                            }
                        ],
                    }
                ],
            },
            "receipt": {"provider_id": "tests.discovery"},
        }

    result = acquire_structured_evidence(request, connector=connector)

    assert result["document"] is None
    assert result["document_sha256"] == ""
    assert result["discovery"]["sources"][0]["publication_number"] == (
        "US1234567A1"
    )
    assert result["receipt"]["provider_id"] == "tests.discovery"


def test_http_connector_normalizes_transport_failures() -> None:
    config = HttpEvidenceConnectorConfig(
        endpoint="https://evidence.example.test/extract",
        provider_id="example.extractor",
        provider_version="2026.07",
    )

    def fail(
        *_args: object, **_kwargs: object
    ) -> tuple[int, bytes, Mapping[str, Any]]:
        raise OSError("network details must not escape")

    connector = build_http_evidence_connector(config, requester=fail)
    with pytest.raises(
        LiveEvidenceConnectorError,
        match="evidence_connector_transport_failed:OSError",
    ):
        connector({"content_sha256": "request-digest"})

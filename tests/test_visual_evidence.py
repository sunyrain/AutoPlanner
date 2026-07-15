from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import pytest

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.run_kernel import RunLimits, RunSpec
from cascade_planner.interfaces.visual_evidence import (
    CodexVisualEvidenceConfig,
    VisualEvidenceError,
    acquire_visual_evidence_candidates,
    build_codex_visual_evidence_provider,
    compile_visual_evidence_request,
    materialize_visual_evidence_candidates,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)


TARGET = "CCOC(C)=O"


def _service(
    tmp_path: Path,
    *,
    max_model_invocations: int = 1,
    max_visual_invocations: int = 1,
) -> RetrosynthesisCampaignService:
    return RetrosynthesisCampaignService.create(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="visual-evidence-test",
            target_name="ethyl acetate",
            target_smiles=TARGET,
            limits=RunLimits(
                model=RetrosynthesisRunBudget(
                    max_model_invocations=max_model_invocations,
                    max_visual_invocations=max_visual_invocations,
                    max_total_input_tokens=10_000,
                    max_total_output_tokens=5_000,
                    max_total_wall_time_s=60,
                    max_attempt_runs=8,
                )
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        artifact_store_root=tmp_path / "cas",
        run_index_path=tmp_path / "index" / "runs.sqlite3",
    )


def _inputs(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    image = tmp_path / "page.png"
    image.write_bytes(b"visual-page-fixture")
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    evidence_request = {
        "schema_version": "evidence_acquisition_request.v1",
        "content_sha256": "a" * 64,
        "run_id": "visual-evidence-test",
        "target_name": "ethyl acetate",
        "target_smiles": TARGET,
        "edges": [
            {
                "edge_id": "edge:ester",
                "product_smiles": TARGET,
                "precursor_smiles": ["CCO", "CC(=O)O"],
                "current_host_reaction_validated": True,
            }
        ],
    }
    discovery = {
        "schema_version": "source_discovery_observation.v1",
        "request_sha256": evidence_request["content_sha256"],
        "sources": [
            {
                "publication_number": "US1234567A1",
                "family_id": "family:one",
                "title": "Preparation of ethyl acetate",
                "pdf_sha256": "b" * 64,
                "exact_row_count": 0,
                "unresolved_edge_count": 1,
                "procedure_inventory": [
                    {
                        "label": "T1",
                        "name": "ethyl acetate",
                        "page_number": 1,
                        "procedure_excerpt": (
                            "Ethanol and acetic acid were heated at reflux "
                            "to afford ethyl acetate."
                        ),
                    }
                ],
                "visual_candidate_pages": [
                    {
                        "page_number": 1,
                        "image_path": str(image),
                        "image_sha256": image_sha256,
                    }
                ],
            }
        ],
    }
    return evidence_request, discovery


def test_visual_request_prefers_route_rich_source_and_carries_page_text(
    tmp_path: Path,
) -> None:
    evidence_request, discovery = _inputs(tmp_path)
    rich = discovery["sources"][0]
    rich["source_ref"] = "patent:RICH1"
    rich["source_route_proposal_count"] = 1
    rich["source_route_observation"] = {
        "proposals": [
            {
                "proposal_id": "source-route:T1",
                "product_name": "ethyl acetate T1",
            }
        ]
    }
    poor = {
        **rich,
        "source_ref": "patent:POOR1",
        "publication_number": "POOR1",
        "unresolved_edge_count": 99,
        "procedure_inventory": [],
        "source_route_proposal_count": 0,
        "source_route_observation": {},
    }
    discovery["sources"] = [poor, rich]

    request = compile_visual_evidence_request(
        evidence_request=evidence_request,
        discovery=discovery,
        max_pages=2,
    )

    assert request["source"]["source_ref"] == "patent:RICH1"
    assert request["source"]["text_snippets"][0]["compound_label"] == "T1"
    assert "Ethanol and acetic acid" in request["source"]["text_snippets"][0][
        "snippet"
    ]
    assert request["source"]["route_sequence_hint"] == "ethyl acetate T1"

    captured: dict[str, Any] = {}

    def runner(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "status": "completed",
            "usage": {"model_invocations": 1, "visual_invocations": 1},
            "candidate_chain": {"steps": []},
        }

    provider = build_codex_visual_evidence_provider(
        CodexVisualEvidenceConfig(cache_dir=tmp_path / "visual", max_pages=2),
        runner=runner,
    )
    provider(request)

    assert captured["route_sequence_hint"] == "ethyl acetate T1"
    assert captured["text_snippets"][0]["compound_label"] == "T1"


def test_visual_candidate_is_one_call_host_normalized_and_never_exact(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    evidence_request, discovery = _inputs(tmp_path)
    calls = 0

    def provider(request: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "request_sha256": request["content_sha256"],
            "provider_status": "completed",
            "provider_receipt": {"provider_id": "tests.visual"},
            "usage": {
                "model_invocations": 1,
                "visual_invocations": 1,
                "input_tokens": 200,
                "output_tokens": 50,
                "wall_time_s": 0.2,
            },
            "candidate_chain": {
                "steps": [
                    {
                        "product_smiles": "CCOC(=O)C",
                        "reactant_smiles": ["O=C(O)C", "OCC"],
                        "product_label": "T1",
                        "reactant_labels": ["acid", "alcohol"],
                        "source_locator": "page 1",
                        "reaction_digest": "forged-provider-digest-is-ignored",
                    }
                ]
            },
        }

    stage = acquire_visual_evidence_candidates(
        service,
        evidence_request=evidence_request,
        discovery=discovery,
        provider=provider,
    )

    assert stage["status"] == "completed"
    assert calls == 1
    observation = stage["observation"]
    assert observation["matched_current_edge_count"] == 1
    assert observation["candidate_steps"][0]["matched_current_edge_id"] == "edge:ester"
    assert observation["candidate_steps"][0]["grants_exact_evidence"] is False
    assert observation["semantics"]["observation_cannot_grant_L2_L3_or_stock"] is True
    assert service.kernel.state.model_totals["model_invocations"] == 1
    assert service.kernel.state.model_totals["visual_invocations"] == 1

    repeated = acquire_visual_evidence_candidates(
        service,
        evidence_request=evidence_request,
        discovery=discovery,
        provider=provider,
    )
    assert repeated["status"] == "budget_blocked"
    assert repeated["reason"] == "campaign_visual_evidence_call_already_admitted"
    assert calls == 1


def test_visual_reference_annotation_cannot_masquerade_as_reaction_conditions(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    evidence_request, discovery = _inputs(tmp_path)

    stage = acquire_visual_evidence_candidates(
        service,
        evidence_request=evidence_request,
        discovery=discovery,
        provider=lambda request: {
            "request_sha256": request["content_sha256"],
            "provider_status": "completed",
            "provider_receipt": {"provider_id": "tests.reference-only"},
            "usage": {
                "model_invocations": 1,
                "visual_invocations": 1,
                "input_tokens": 100,
                "output_tokens": 50,
                "wall_time_s": 0.1,
            },
            "candidate_chain": {
                "steps": [
                    {
                        "product_smiles": TARGET,
                        "reactant_smiles": ["CCO", "CC(=O)O"],
                        "condition_candidate": {
                            "source_type": "exact",
                            "condition_status": "evidence_backed",
                            "condition_text_transcribed": "ref. 78,80",
                            "source_excerpt": "ref. 78,80",
                        },
                    }
                ]
            },
        },
    )

    condition = stage["observation"]["candidate_steps"][0][
        "condition_candidate"
    ]
    assert condition == {
        "schema_version": "visual_condition_candidate.v1",
        "source_reference_annotation": "ref. 78,80",
        "condition_status": "reference_citation_only",
        "source_type": "visual_hypothesis",
        "grants_exact_evidence": False,
    }


def test_paper_pdf_visual_chain_enters_canonical_graph_without_becoming_l3(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    evidence_request, discovery = _inputs(tmp_path)
    source = discovery["sources"][0]
    source.pop("publication_number")
    source["source_kind"] = "paper_si"
    source["source_ref"] = "doi:10.1000/visual-route"
    source["doi"] = "10.1000/visual-route"
    source["source_pdf_sha256"] = source.pop("pdf_sha256")

    request = compile_visual_evidence_request(
        evidence_request=evidence_request,
        discovery=discovery,
        max_pages=2,
    )
    assert request["source"]["source_ref"] == "doi:10.1000/visual-route"
    assert request["source"]["source_kind"] == "paper_si"

    stage = acquire_visual_evidence_candidates(
        service,
        evidence_request=evidence_request,
        discovery=discovery,
        provider=lambda request: {
            "request_sha256": request["content_sha256"],
            "provider_status": "completed",
            "provider_receipt": {"provider_id": "tests.paper-vision"},
            "usage": {
                "model_invocations": 1,
                "visual_invocations": 1,
                "input_tokens": 100,
                "output_tokens": 50,
                "wall_time_s": 0.1,
            },
            "candidate_chain": {
                "steps": [
                    {
                        "product_smiles": TARGET,
                        "reactant_smiles": ["CCO", "CC(=O)O"],
                        "source_locator": "Scheme 2, page 1",
                        "conditions": {"solvent": "toluene", "temperature_c": 80},
                    }
                ]
            },
        },
    )
    materialized = materialize_visual_evidence_candidates(
        service,
        observation=stage["observation"],
    )

    assert materialized["status"] == "completed"
    graph = service.graph_store.load()
    assert len(graph["edges"]) == 1
    edge = next(iter(graph["edges"].values()))
    assert edge["origin_records"][0]["origin_kind"] == "literature_visual_extraction"
    assert edge["condition_predictions"][0]["solvent"] == "toluene"
    assert edge["exact_record_ids"] == []
    assert graph["exact_records"] == {}


def test_visual_chain_with_wrong_root_formula_cannot_create_disconnected_route(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    evidence_request, discovery = _inputs(tmp_path)
    stage = acquire_visual_evidence_candidates(
        service,
        evidence_request=evidence_request,
        discovery=discovery,
        provider=lambda request: {
            "request_sha256": request["content_sha256"],
            "provider_status": "completed",
            "provider_receipt": {"provider_id": "tests.wrong-root"},
            "usage": {
                "model_invocations": 1,
                "visual_invocations": 1,
                "input_tokens": 100,
                "output_tokens": 50,
                "wall_time_s": 0.1,
            },
            "candidate_chain": {
                "steps": [
                    {
                        "product_smiles": "CCOC(=O)CO",
                        "reactant_smiles": ["CCO", "O=CC(=O)O"],
                        "source_locator": "Scheme 4",
                    }
                ]
            },
        },
    )

    observation = stage["observation"]
    assert observation["candidate_step_count"] == 1
    assert observation["admission_eligible_step_count"] == 0
    assert observation["chain_admission_accepted"] is False
    assert observation["chain_admission_reasons"] == [
        "visual_chain_root_not_target_connected"
    ]
    materialized = materialize_visual_evidence_candidates(
        service,
        observation=observation,
    )
    assert materialized["status"] == "not_needed"
    assert service.graph_store.load()["edges"] == {}


def test_visual_chain_can_anchor_to_existing_frontier_as_replacement_module(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    evidence_request, discovery = _inputs(tmp_path)
    evidence_request["edges"].append(
        {
            "edge_id": "edge:frontier",
            "product_smiles": "CCO",
            "precursor_smiles": ["CC"],
            "current_host_reaction_validated": True,
        }
    )
    stage = acquire_visual_evidence_candidates(
        service,
        evidence_request=evidence_request,
        discovery=discovery,
        provider=lambda request: {
            "request_sha256": request["content_sha256"],
            "provider_status": "completed",
            "provider_receipt": {"provider_id": "tests.frontier-anchor"},
            "usage": {
                "model_invocations": 1,
                "visual_invocations": 1,
                "input_tokens": 100,
                "output_tokens": 50,
                "wall_time_s": 0.1,
            },
            "candidate_chain": {
                "steps": [
                    {
                        "product_smiles": "OCC",
                        "reactant_smiles": ["C", "CO"],
                        "source_locator": "Scheme 7",
                    }
                ]
            },
        },
    )

    observation = stage["observation"]
    assert observation["chain_admission_accepted"] is True
    assert observation["frontier_anchored_step_count"] == 1
    assert observation["candidate_steps"][0]["root_anchor"] == (
        "canonical_frontier_identity"
    )


def test_visual_budget_zero_prevents_provider_call(tmp_path: Path) -> None:
    service = _service(tmp_path, max_visual_invocations=0)
    evidence_request, discovery = _inputs(tmp_path)

    def provider(_request: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("provider must not run when visual budget is zero")

    stage = acquire_visual_evidence_candidates(
        service,
        evidence_request=evidence_request,
        discovery=discovery,
        provider=provider,
    )

    assert stage["status"] == "budget_blocked"
    assert "visual_invocation_budget_exhausted" in stage["reason"]
    assert service.kernel.state.model_totals["visual_invocations"] == 0


def test_visual_request_skips_already_closed_source_and_tampered_image(
    tmp_path: Path,
) -> None:
    evidence_request, discovery = _inputs(tmp_path)
    discovery["sources"][0]["exact_row_count"] = 1
    discovery["sources"][0]["unresolved_edge_count"] = 0
    assert not compile_visual_evidence_request(
        evidence_request=evidence_request,
        discovery=discovery,
        max_pages=4,
    )

    discovery["request_sha256"] = "c" * 64
    assert not compile_visual_evidence_request(
        evidence_request=evidence_request,
        discovery=discovery,
        max_pages=4,
    )


def test_visual_provider_request_digest_mismatch_fails_and_consumes_call(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    evidence_request, discovery = _inputs(tmp_path)
    stage = acquire_visual_evidence_candidates(
        service,
        evidence_request=evidence_request,
        discovery=discovery,
        provider=lambda _request: {
            "request_sha256": "f" * 64,
            "usage": {
                "model_invocations": 1,
                "visual_invocations": 1,
            },
            "candidate_chain": {"steps": []},
        },
    )

    assert stage["status"] == "failed"
    assert "request_digest_mismatch" in stage["reason"]
    assert service.kernel.state.model_totals["model_invocations"] == 1
    assert service.kernel.state.model_totals["visual_invocations"] == 1
    assert service.kernel.state.in_flight_tasks == {}


def test_visual_provider_exception_cannot_strand_kernel_reservation(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    evidence_request, discovery = _inputs(tmp_path)

    def broken_provider(_request):
        raise KeyError("broken provider")

    stage = acquire_visual_evidence_candidates(
        service,
        evidence_request=evidence_request,
        discovery=discovery,
        provider=broken_provider,
    )

    assert stage["status"] == "failed"
    assert stage["model_invocations"] == 1
    assert stage["visual_invocations"] == 1
    assert service.kernel.state.in_flight_tasks == {}

    discovery["sources"][0]["exact_row_count"] = 0
    discovery["sources"][0]["visual_candidate_pages"][0]["image_sha256"] = "f" * 64
    assert not compile_visual_evidence_request(
        evidence_request=evidence_request,
        discovery=discovery,
        max_pages=4,
    )


def test_codex_visual_provider_rejects_candidate_without_usage_receipt(
    tmp_path: Path,
) -> None:
    evidence_request, discovery = _inputs(tmp_path)
    request = compile_visual_evidence_request(
        evidence_request=evidence_request,
        discovery=discovery,
        max_pages=1,
    )
    provider = build_codex_visual_evidence_provider(
        CodexVisualEvidenceConfig(cache_dir=tmp_path / "visual", max_pages=1),
        runner=lambda **_kwargs: {
            "status": "completed",
            "usage": {},
            "candidate_chain": {
                "steps": [
                    {
                        "product_smiles": TARGET,
                        "reactant_smiles": ["CCO", "CC(=O)O"],
                    }
                ]
            },
        },
    )

    with pytest.raises(VisualEvidenceError, match="usage_receipt_missing"):
        provider(request)

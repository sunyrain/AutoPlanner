from __future__ import annotations

import hashlib
import json
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cascade_planner.orchestration.admitted_hyperedges import (
    AdmittedHyperedgeJournalError,
    canonical_graph_step_id,
    canonical_graph_step_signature,
    load_external_hyperedge_events,
    record_external_hyperedges,
)
from cascade_planner.orchestration.codex_retrosynthesis import (
    RetrosynthesisTeamConfig,
    reconcile_codex_campaign_proof_state,
)
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from cascade_planner.routes.adapters import rebuild_consensus_graph_from_blackboard


_FIXTURES = Path(__file__).parent / "fixtures"
_SOURCE_PDF = _FIXTURES / "source_evidence_stub.pdf"
_SOURCE_PAGE = _FIXTURES / "source_page.ppm"
_SOURCE_MANIFEST = _FIXTURES / "source_evidence_manifest.json"
_TRUSTED_REGISTRY = _FIXTURES / "trusted_literature_step_registry.json"


def _external_graph() -> dict:
    step_id = canonical_graph_step_id("CCO", ["CC", "O"])
    return {
        "schema_version": "route_consensus_graph.v1",
        "case_id": "journal-case",
        "target_smiles": "CCO",
        "nodes": [
            {
                "schema_version": "route_consensus_molecule.v1",
                "node_id": "root:CCO",
                "smiles": "CCO",
                "min_depth": 0,
            },
            {
                "schema_version": "route_consensus_molecule.v1",
                "node_id": "leaf:CC=O",
                "smiles": "CC",
                "min_depth": 1,
            },
            {
                "schema_version": "route_consensus_molecule.v1",
                "node_id": "leaf:O",
                "smiles": "O",
                "min_depth": 1,
            },
        ],
        "steps": [
            {
                "schema_version": "route_consensus_step.v1",
                "step_id": step_id,
                "signature": canonical_graph_step_signature("CCO", ["CC", "O"]),
                "product_node_id": "root:CCO",
                "precursor_node_ids": ["leaf:CC=O", "leaf:O"],
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "reaction_family": "carbonyl reduction",
                "source_channels": ["literature_exact"],
                "source_refs": ["fixture:exact-row"],
                "rank_score": 0.8,
            }
        ],
    }


def _admission_receipts() -> dict:
    report = verify_chemenzy_raw_routes(
        {
            "target": "CCO",
            "routes": [
                {
                    "route_rank": 0,
                    "metrics": {
                        "terminal_reactants": ["CC", "O"],
                        "terminal_stock_status": {"CC": True, "O": True},
                    },
                    "steps": [
                        {
                            "index": 0,
                            "product": "CCO",
                            "reactant_smiles": ["CC", "O"],
                            "stock_status": {"CC": True, "O": True},
                        }
                    ],
                }
            ],
        },
        target_smiles="CCO",
        case_id="journal-case",
    )
    assert report["accepted"] is True
    rebuild = rebuild_consensus_graph_from_blackboard(
        {
            "case_id": "journal-case",
            "target_profile": {"target_smiles": "CCO"},
            "chemenzy_route_proof_banks": [
                {
                    "artifact_ref": "guided_chemenzy_result.json",
                    "route_proof_bank": report["route_proof_bank"],
                }
            ],
        }
    )
    return rebuild["admission_receipts"]


def _validated_exact_row() -> dict:
    template_id = "source_detail_exact_step:ethanol_hydration"
    return {
        "row_id": template_id,
        "step_id": "ethanol_hydration",
        "accepted": True,
        "product_smiles": "CCO",
        "reactant_smiles": ["CC", "O"],
        "source_template_id": template_id,
        "source_detail_exact_step": True,
        "relation_type": "exact",
        "source_ref": "doi:10.1000/revalidatable-stitch",
        "exact_step_validation": {
            "schema_version": "template_validation_report.v1",
            "accepted": True,
            "allowed_for_one_step_source": True,
            "source_template_id": template_id,
            "reasons": [],
        },
        "source_evidence": [
            {
                "schema_version": "materialized_source_evidence.v1",
                "document_id": "fixture:revalidatable-stitch",
                "manifest_path": str(_SOURCE_MANIFEST.resolve()),
                "manifest_sha256": hashlib.sha256(
                    _SOURCE_MANIFEST.read_bytes()
                ).hexdigest(),
                "source_pdf_path": str(_SOURCE_PDF.resolve()),
                "source_pdf_sha256": hashlib.sha256(
                    _SOURCE_PDF.read_bytes()
                ).hexdigest(),
                "page_number": 1,
                "image_path": str(_SOURCE_PAGE.resolve()),
                "image_sha256": hashlib.sha256(
                    _SOURCE_PAGE.read_bytes()
                ).hexdigest(),
                "source_ref": "doi:10.1000/revalidatable-stitch",
            }
        ],
    }


def _rehash(payload: dict) -> dict:
    row = dict(payload)
    row.pop("content_sha256", None)
    row["content_sha256"] = hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return row


def test_admitted_hyperedge_journal_fails_closed_on_drift_and_authority_field(
    tmp_path,
) -> None:
    journal_root = tmp_path / "admitted_hyperedges"
    identity_sha256 = "a" * 64
    policy_sha256 = "b" * 64
    report = record_external_hyperedges(
        journal_root,
        _external_graph(),
        case_id="journal-case",
        target_smiles="CCO",
        campaign_identity_sha256=identity_sha256,
        campaign_policy_sha256=policy_sha256,
        admission_receipts=_admission_receipts(),
    )

    assert report["new_event_count"] == 1
    assert len(
        load_external_hyperedge_events(
            journal_root,
            case_id="journal-case",
            target_smiles="CCO",
            campaign_identity_sha256=identity_sha256,
            campaign_policy_sha256=policy_sha256,
        )
    ) == 1
    with pytest.raises(
        AdmittedHyperedgeJournalError,
        match="event_campaign_binding_mismatch",
    ):
        load_external_hyperedge_events(
            journal_root,
            case_id="journal-case",
            target_smiles="CCO",
            campaign_identity_sha256=identity_sha256,
            campaign_policy_sha256="c" * 64,
        )

    event_path = Path(report["event_refs"][0])
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["proof"] = {"accepted": True, "solved": True}
    event_path.write_text(
        json.dumps(_rehash(event), sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        AdmittedHyperedgeJournalError,
        match="event_fields_invalid",
    ):
        load_external_hyperedge_events(
            journal_root,
            case_id="journal-case",
            target_smiles="CCO",
            campaign_identity_sha256=identity_sha256,
            campaign_policy_sha256=policy_sha256,
        )


def test_unreceipted_source_labels_are_quarantined_not_journaled(tmp_path) -> None:
    report = record_external_hyperedges(
        tmp_path / "admitted_hyperedges",
        _external_graph(),
        case_id="journal-case",
        target_smiles="CCO",
        campaign_identity_sha256="a" * 64,
        campaign_policy_sha256="b" * 64,
    )

    assert report["event_count"] == 0
    assert report["new_event_count"] == 0
    assert report["quarantined_edge_count"] == 1
    assert report["quarantined_edges"][0]["reasons"] == [
        "current_host_provenance_receipt_missing"
    ]


def test_mixed_receipted_and_unknown_edges_admit_only_receipted_edge(tmp_path) -> None:
    graph = _external_graph()
    unknown_id = canonical_graph_step_id("CCO", ["C", "CO"])
    graph["steps"].append(
        {
            "schema_version": "route_consensus_step.v1",
            "step_id": unknown_id,
            "signature": canonical_graph_step_signature("CCO", ["C", "CO"]),
            "product_node_id": "root:CCO",
            "precursor_node_ids": ["leaf:C", "leaf:CO"],
            "product_smiles": "CCO",
            "precursor_smiles": ["C", "CO"],
            "reaction_family": "unknown advisory split",
            "source_channels": ["other", "codex_strategy"],
            "rank_score": 0.4,
        }
    )

    report = record_external_hyperedges(
        tmp_path / "admitted_hyperedges",
        graph,
        case_id="journal-case",
        target_smiles="CCO",
        campaign_identity_sha256="a" * 64,
        campaign_policy_sha256="b" * 64,
        admission_receipts=_admission_receipts(),
    )

    assert report["new_event_count"] == 1
    assert report["event_count"] == 1
    assert report["quarantined_edge_count"] == 1
    assert report["quarantined_edges"][0]["step_id"] == unknown_id


def test_valid_receipt_survives_tampered_independent_sibling_material(
    tmp_path,
) -> None:
    receipts = _admission_receipts()
    edge_key = next(iter(receipts))
    tampered = copy.deepcopy(receipts[edge_key][0])
    tampered["artifact_ref"] = "tampered-without-rehash"
    receipts[edge_key].append(tampered)

    report = record_external_hyperedges(
        tmp_path / "admitted_hyperedges",
        _external_graph(),
        case_id="journal-case",
        target_smiles="CCO",
        campaign_identity_sha256="a" * 64,
        campaign_policy_sha256="b" * 64,
        admission_receipts=receipts,
    )

    assert report["new_event_count"] == 1
    assert report["quarantined_edge_count"] == 0
    assert report["rejected_material_count"] == 1
    assert report["rejected_materials"][0]["material_index"] == 1
    assert report["rejected_materials"][0]["semantics"][
        "cannot_veto_valid_independent_receipt"
    ] is True


def test_validated_exact_literature_receipt_replays_source_material_on_load(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(_TRUSTED_REGISTRY.resolve()),
    )
    rebuild = rebuild_consensus_graph_from_blackboard(
        {
            "case_id": "journal-case",
            "target_profile": {"target_smiles": "CCO"},
            "literature_evidence": {"exact_rows": [_validated_exact_row()]},
        }
    )
    report = record_external_hyperedges(
        tmp_path / "admitted_hyperedges",
        rebuild["graph"],
        case_id="journal-case",
        target_smiles="CCO",
        campaign_identity_sha256="a" * 64,
        campaign_policy_sha256="b" * 64,
        admission_receipts=rebuild["admission_receipts"],
    )

    assert report["new_event_count"] == 1
    event = load_external_hyperedge_events(
        tmp_path / "admitted_hyperedges",
        case_id="journal-case",
        target_smiles="CCO",
        campaign_identity_sha256="a" * 64,
        campaign_policy_sha256="b" * 64,
    )[0]
    assert event["provenance_receipt"]["source_kind"] == (
        "validated_exact_literature_adapter"
    )

    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(tmp_path / "missing-registry.json"),
    )
    events = load_external_hyperedge_events(
        tmp_path / "admitted_hyperedges",
        case_id="journal-case",
        target_smiles="CCO",
        campaign_identity_sha256="a" * 64,
        campaign_policy_sha256="b" * 64,
    )
    assert events == []
    replay_report = json.loads(
        (
            tmp_path
            / "admitted_hyperedges"
            / "replay_report.json"
        ).read_text(encoding="utf-8")
    )
    assert replay_report["inactive_event_count"] == 1
    assert replay_report["active_event_count"] == 0
    assert any(
        "exact_literature_source_row_host_replay_failed" in reason
        for reason in replay_report["inactive_events"][0]["reasons"]
    )
    assert replay_report["inactive_events"][0]["semantics"][
        "cannot_mutate_queue_proof_stock_or_completion"
    ] is True


def test_source_bound_literature_without_registry_enters_only_l0_search(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(tmp_path / "empty-registry.json"),
    )
    row = _validated_exact_row()
    rebuild = rebuild_consensus_graph_from_blackboard(
        {
            "case_id": "materialized-literature-case",
            "target_profile": {"target_smiles": "CCO"},
            "literature_evidence": {"exact_rows": [row]},
        },
        max_depth=2,
    )

    proposal = rebuild["consensus"]["proposals"][0]
    source = proposal["source_records"][0]
    assert source["source_channel"] == "literature_analogy"
    assert source["authority_evidence_level"] == "model_only"
    assert source["authority_bound"] is False
    material = next(iter(rebuild["admission_receipts"].values()))[0]
    assert material["schema_version"] == (
        "materialized_literature_search_admission.v1"
    )

    reconciled = reconcile_codex_campaign_proof_state(
        graph=rebuild["graph"],
        run_dir=tmp_path,
        case_id="materialized-literature-case",
        campaign_config=RetrosynthesisTeamConfig(max_depth=2),
        external_hyperedge_admission_receipts=rebuild["admission_receipts"],
    )

    assert len(reconciled["canonical_route_consensus_graph"]["steps"]) == 1
    proof_record = reconciled["reaction_proof_state"]["records"][0]
    assert proof_record["achieved_proof_level"] == 0
    assert proof_record["status"] == "pending"
    event = load_external_hyperedge_events(
        tmp_path / "codex_retrosynthesis_team" / "admitted_hyperedges",
        case_id="materialized-literature-case",
        target_smiles="CCO",
        campaign_identity_sha256=reconciled["campaign_identity_sha256"],
        campaign_policy_sha256=reconciled["campaign_policy_sha256"],
    )[0]
    receipt = event["provenance_receipt"]
    assert receipt["source_kind"] == "materialized_literature_search_admission"
    encoded_receipt = json.dumps(receipt, sort_keys=True).lower()
    assert '"accepted"' not in encoded_receipt
    assert '"validated"' not in encoded_receipt
    assert '"solved"' not in encoded_receipt


def test_source_claim_with_missing_artifact_is_quarantined(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(tmp_path / "empty-registry.json"),
    )
    row = copy.deepcopy(_validated_exact_row())
    row["source_evidence"][0]["image_path"] = str(
        tmp_path / "missing-source-page.png"
    )
    rebuild = rebuild_consensus_graph_from_blackboard(
        {
            "case_id": "missing-literature-artifact-case",
            "target_profile": {"target_smiles": "CCO"},
            "literature_evidence": {"exact_rows": [row]},
        }
    )
    assert rebuild["graph"]["steps"]
    assert rebuild["admission_receipts"] == {}

    report = record_external_hyperedges(
        tmp_path / "admitted_hyperedges",
        rebuild["graph"],
        case_id="missing-literature-artifact-case",
        target_smiles="CCO",
        campaign_identity_sha256="a" * 64,
        campaign_policy_sha256="b" * 64,
        admission_receipts=rebuild["admission_receipts"],
    )
    assert report["event_count"] == 0
    assert report["quarantined_edge_count"] == 1


def test_concurrent_same_edge_admission_publishes_one_event(tmp_path) -> None:
    journal_root = tmp_path / "admitted_hyperedges"
    receipts = _admission_receipts()

    def record() -> dict:
        return record_external_hyperedges(
            journal_root,
            _external_graph(),
            case_id="journal-case",
            target_smiles="CCO",
            campaign_identity_sha256="a" * 64,
            campaign_policy_sha256="b" * 64,
            admission_receipts=receipts,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda _: record(), range(2)))

    assert sorted(report["new_event_count"] for report in reports) == [0, 1]
    events = load_external_hyperedge_events(
        journal_root,
        case_id="journal-case",
        target_smiles="CCO",
        campaign_identity_sha256="a" * 64,
        campaign_policy_sha256="b" * 64,
    )
    assert len(events) == 1

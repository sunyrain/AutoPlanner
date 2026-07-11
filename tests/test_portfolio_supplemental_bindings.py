from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from cascade_planner.application.route_portfolio import derive_portfolio_bindings
from cascade_planner.harness.codex_edge_verification import (
    verify_codex_consensus_graph,
)
from cascade_planner.harness.reaction_step_verifier import verify_reaction_route
from cascade_planner.providers import (
    BenchmarkCatalogStockProvider,
    ProviderContext,
    SnapshotStockProvider,
    stock_snapshot_sha256,
)


def _overlay() -> dict:
    return {
        "schema_version": "route_hypergraph_overlay.v2",
        "root_molecule_id": "mol:product",
        "molecules": [
            {
                "molecule_id": "mol:product",
                "canonical_isomeric_smiles": "CC=O",
            },
            {
                "molecule_id": "mol:ethanol",
                "canonical_isomeric_smiles": "CCO",
            },
        ],
        "reaction_hyperedges": [
            {
                "hyperedge_id": "rxn:oxidation",
                "product_molecule_id": "mol:product",
                "precursor_molecule_ids": ["mol:ethanol"],
            }
        ],
    }


def test_host_edge_proof_and_hashed_stock_provider_bind_without_route_verifier(
    tmp_path: Path,
) -> None:
    edge_report = verify_codex_consensus_graph(
        {
            "schema_version": "codex_route_consensus_graph.v1",
            "target_smiles": "CC=O",
            "steps": [
                {
                    "step_id": "oxidation",
                    "product_smiles": "CC=O",
                    "precursor_smiles": ["CCO"],
                }
            ],
        },
        atom_mapper=lambda _: [
            "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
        ],
        enable_optional_rxnmapper=False,
    )
    catalog = tmp_path / "stock.csv"
    catalog.write_text("smiles\nCCO\n", encoding="utf-8")
    digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
    provider = BenchmarkCatalogStockProvider(
        catalog_artifact=catalog,
        catalog_sha256=digest,
        catalog_name="fixture-stock",
    )
    stock = provider.invoke(
        {"schema_version": "stock_lookup_request.v1", "smiles": "CCO"},
        context=ProviderContext(
            run_id="test",
            case_id="test",
            target_smiles="CC=O",
        ),
    ).to_dict()

    bindings = derive_portfolio_bindings(
        _overlay(),
        {},
        supplemental_edge_verification_reports=[edge_report],
        supplemental_stock_provider_results=[stock],
    )

    assert bindings["edge_proof_levels"] == {"rxn:oxidation": 2}
    edge_binding = bindings["exact_edge_proof_bindings"]["rxn:oxidation"]
    assert edge_binding["proof_source"] == (
        "supplemental_reaction_validation.v2_replayed"
    )
    assert edge_binding["verifier_source_sha256"] == (
        edge_report["reaction_validations"][0]["content_sha256"]
    )
    assert bindings["stock_molecule_ids"] == ["mol:ethanol"]
    assert bindings["accepted_supplemental_reaction_validation_count"] == 1
    assert bindings["accepted_supplemental_edge_verification_report_count"] == 1
    assert bindings["accepted_supplemental_stock_boundary_count"] == 1
    assert bindings["stock_binding_valid"] is True
    stock_binding = bindings["stock_bindings"]["mol:ethanol"]
    assert stock_binding["boundary_type"] == "benchmark_stock"
    assert stock_binding["benchmark_membership"] is True
    assert stock_binding["commercial_orderability_claimed"] is False
    assert stock_binding["artifact_hash_replayed"] is True
    assert stock_binding["canonical_membership_replayed"] is True
    assert stock_binding["provider_trust_authority"] == (
        "autoplanner_host_builtin_allowlist.v1"
    )


def test_digest_consistent_detached_reaction_booleans_are_not_authority() -> None:
    detached = verify_reaction_route(
        [
            {
                "step_id": "oxidation",
                "product_smiles": "CC=O",
                "reactant_smiles": ["CCO"],
                "atom_mapped_reaction_smiles": (
                    "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
                ),
            }
        ],
        graph_and_stock_closed=True,
    )

    bindings = derive_portfolio_bindings(
        _overlay(),
        {},
        supplemental_reaction_validations=[detached],
    )

    assert bindings["accepted_supplemental_reaction_validation_count"] == 0
    assert bindings["edge_proof_levels"] == {}


def test_rehashed_tampered_replay_comparison_is_rejected() -> None:
    edge_report = verify_codex_consensus_graph(
        {
            "schema_version": "codex_route_consensus_graph.v1",
            "target_smiles": "CC=O",
            "steps": [
                {
                    "step_id": "oxidation",
                    "product_smiles": "CC=O",
                    "precursor_smiles": ["CCO"],
                }
            ],
        },
        atom_mapper=lambda _: [
            "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
        ],
        enable_optional_rxnmapper=False,
    )
    forged = copy.deepcopy(edge_report["reaction_validations"][0])
    forged["claimed_step_proof"]["checks"]["bond_change_present"] = False
    unsigned = dict(forged)
    unsigned.pop("content_sha256")
    forged["content_sha256"] = _digest(unsigned)

    bindings = derive_portfolio_bindings(
        _overlay(),
        {},
        supplemental_reaction_validations=[forged],
    )

    assert bindings["accepted_supplemental_reaction_validation_count"] == 0
    assert bindings["edge_proof_levels"] == {}


def test_rehashed_report_with_unlinked_candidate_and_wrapper_is_rejected() -> None:
    edge_report = verify_codex_consensus_graph(
        {
            "schema_version": "codex_route_consensus_graph.v1",
            "target_smiles": "CC=O",
            "steps": [
                {
                    "step_id": "oxidation",
                    "product_smiles": "CC=O",
                    "precursor_smiles": ["CCO"],
                }
            ],
        },
        atom_mapper=lambda _: [
            "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
        ],
        enable_optional_rxnmapper=False,
    )
    forged = copy.deepcopy(edge_report)
    forged["edge_verifications"][0]["materialized_candidate"]["step_id"] = "other"
    unsigned = dict(forged)
    unsigned.pop("content_sha256")
    forged["content_sha256"] = _digest(unsigned)

    bindings = derive_portfolio_bindings(
        _overlay(),
        {},
        supplemental_edge_verification_reports=[forged],
    )

    assert bindings["accepted_supplemental_edge_verification_report_count"] == 0
    assert bindings["accepted_supplemental_reaction_validation_count"] == 0
    assert bindings["edge_proof_levels"] == {}


def test_unknown_stock_provider_cannot_gain_authority_by_rehashing(tmp_path: Path) -> None:
    catalog = tmp_path / "stock.csv"
    catalog.write_text("smiles\nCCO\n", encoding="utf-8")
    catalog_sha256 = hashlib.sha256(catalog.read_bytes()).hexdigest()
    stock = BenchmarkCatalogStockProvider(
        catalog_artifact=catalog,
        catalog_sha256=catalog_sha256,
    ).invoke(
        {"schema_version": "stock_lookup_request.v1", "smiles": "CCO"},
        context=ProviderContext(run_id="test", case_id="test", target_smiles="CC=O"),
    ).to_dict()
    stock["provider_id"] = "attacker.made_up_stock"
    unsigned = dict(stock)
    unsigned.pop("content_hash")
    stock["content_hash"] = _digest(unsigned)

    bindings = derive_portfolio_bindings(
        _overlay(),
        {},
        supplemental_stock_provider_results=[stock],
    )

    assert bindings["accepted_supplemental_stock_boundary_count"] == 0
    assert bindings["stock_molecule_ids"] == []


def test_benchmark_envelope_is_rejected_after_artifact_bytes_change(tmp_path: Path) -> None:
    catalog = tmp_path / "stock.csv"
    catalog.write_text("smiles\nCCO\n", encoding="utf-8")
    catalog_sha256 = hashlib.sha256(catalog.read_bytes()).hexdigest()
    provider = BenchmarkCatalogStockProvider(
        catalog_artifact=catalog,
        catalog_sha256=catalog_sha256,
    )
    stock = provider.invoke(
        {"schema_version": "stock_lookup_request.v1", "smiles": "CCO"},
        context=ProviderContext(run_id="test", case_id="test", target_smiles="CC=O"),
    ).to_dict()
    catalog.write_text("smiles\nC\n", encoding="utf-8")

    bindings = derive_portfolio_bindings(
        _overlay(),
        {},
        supplemental_stock_provider_results=[stock],
    )

    assert bindings["accepted_supplemental_stock_boundary_count"] == 0
    assert bindings["stock_molecule_ids"] == []


def test_benchmark_envelope_rechecks_canonical_membership(tmp_path: Path) -> None:
    catalog = tmp_path / "stock.csv"
    catalog.write_text("smiles\nC\n", encoding="utf-8")
    catalog_sha256 = hashlib.sha256(catalog.read_bytes()).hexdigest()
    provider = BenchmarkCatalogStockProvider(
        catalog_artifact=catalog,
        catalog_sha256=catalog_sha256,
    )
    forged = provider.invoke(
        {"schema_version": "stock_lookup_request.v1", "smiles": "C"},
        context=ProviderContext(run_id="test", case_id="test", target_smiles="CC=O"),
    ).to_dict()
    forged["payload"]["canonical_smiles"] = "CCO"
    forged["payload"]["catalog_bindings"][0]["canonical_smiles"] = "CCO"
    unsigned = dict(forged)
    unsigned.pop("content_hash")
    forged["content_hash"] = _digest(unsigned)

    bindings = derive_portfolio_bindings(
        _overlay(),
        {},
        supplemental_stock_provider_results=[forged],
    )

    assert bindings["accepted_supplemental_stock_boundary_count"] == 0
    assert bindings["stock_molecule_ids"] == []


def test_commercial_snapshot_signature_is_recomputed_at_consumption() -> None:
    snapshot = {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": "fixture-supplier",
        "catalog_number": "ETH-1",
        "smiles": "CCO",
        "checked_at": "2026-07-12T00:00:00Z",
        "available": True,
    }
    snapshot_sha256 = stock_snapshot_sha256(snapshot)
    stock = SnapshotStockProvider(trusted_snapshots=[snapshot]).invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "CCO",
            "offers": [{**snapshot, "snapshot_sha256": snapshot_sha256}],
        },
        context=ProviderContext(run_id="test", case_id="test", target_smiles="CC=O"),
    ).to_dict()
    untrusted_self_contained = derive_portfolio_bindings(
        _overlay(),
        {},
        supplemental_stock_provider_results=[stock],
    )
    assert untrusted_self_contained["accepted_supplemental_stock_boundary_count"] == 0
    assert untrusted_self_contained["stock_molecule_ids"] == []

    trusted_provider = SnapshotStockProvider(trusted_snapshots=[snapshot])
    valid = derive_portfolio_bindings(
        _overlay(),
        {},
        supplemental_stock_provider_results=[stock],
        trusted_stock_provider_instances={
            SnapshotStockProvider.descriptor.provider_id: trusted_provider,
        },
    )
    assert valid["stock_molecule_ids"] == ["mol:ethanol"]
    binding = valid["stock_bindings"]["mol:ethanol"]
    assert binding["boundary_type"] == "commercially_orderable"
    assert binding["commercial_orderability_claimed"] is True
    assert binding["snapshot_digest_replayed"] is True

    forged = copy.deepcopy(stock)
    forged["payload"]["offers"][0]["snapshot"]["catalog_number"] = "FORGED"
    unsigned = dict(forged)
    unsigned.pop("content_hash")
    forged["content_hash"] = _digest(unsigned)
    rejected = derive_portfolio_bindings(
        _overlay(),
        {},
        supplemental_stock_provider_results=[forged],
        trusted_stock_provider_instances={
            SnapshotStockProvider.descriptor.provider_id: trusted_provider,
        },
    )
    assert rejected["accepted_supplemental_stock_boundary_count"] == 0
    assert rejected["stock_molecule_ids"] == []


def _digest(value: object) -> str:
    import json

    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()

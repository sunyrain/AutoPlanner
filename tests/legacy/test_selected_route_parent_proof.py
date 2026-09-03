from __future__ import annotations

import copy
import hashlib
import json

import pytest

from cascade_planner.legacy.application_runtime.selected_route_parent_proof import (
    compile_selected_route_parent_proof,
    is_solved_selected_route_parent_proof,
    validate_selected_route_parent_proof,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _seal(value: dict, field: str = "content_sha256") -> dict:
    value.pop(field, None)
    value[field] = _digest(value)
    return value


def _overlay(*, single_route: bool = False) -> dict:
    edges = [
        {
            "hyperedge_id": "edge:direct",
            "product_molecule_id": "mol:target",
            "precursor_molecule_ids": ["mol:stock-a"],
        }
    ]
    if not single_route:
        edges.extend(
            [
                {
                    "hyperedge_id": "edge:via-middle",
                    "product_molecule_id": "mol:target",
                    "precursor_molecule_ids": ["mol:middle"],
                },
                {
                    "hyperedge_id": "edge:middle-leaf",
                    "product_molecule_id": "mol:middle",
                    "precursor_molecule_ids": ["mol:stock-b"],
                },
            ]
        )
    return {
        "schema_version": "route_hypergraph_overlay.v2",
        "root_molecule_id": "mol:target",
        # Persisted validation is intentionally not an authority for this module.
        "validation": {"valid": True, "errors": []},
        "molecules": [
            {
                "molecule_id": "mol:target",
                "canonical_isomeric_smiles": "CCO",
            },
            {
                "molecule_id": "mol:stock-a",
                "canonical_isomeric_smiles": "C",
            },
            {
                "molecule_id": "mol:middle",
                "canonical_isomeric_smiles": "CC",
            },
            {
                "molecule_id": "mol:stock-b",
                "canonical_isomeric_smiles": "N",
            },
        ],
        "reaction_hyperedges": edges,
    }


def _edge_binding(
    edge: dict,
    smiles_by_id: dict[str, str],
    *,
    level: int,
) -> dict:
    edge_id = str(edge["hyperedge_id"])
    product_id = str(edge["product_molecule_id"])
    precursor_ids = sorted(str(value) for value in edge["precursor_molecule_ids"])
    signature = _digest(
        {
            "product_canonical_isomeric_smiles": smiles_by_id[product_id],
            "reactant_canonical_isomeric_smiles": sorted(
                smiles_by_id[value] for value in precursor_ids
            ),
        }
    )
    named_level = {
        2: "L2_reaction_validated",
        3: "L3_precedent_supported",
        4: "L4_procurement_ready",
    }[level]
    return _seal(
        {
            "schema_version": "exact_edge_proof_binding.v1",
            "hyperedge_id": edge_id,
            "product_molecule_id": product_id,
            "precursor_molecule_ids": precursor_ids,
            "structure_signature_sha256": signature,
            "proof_level": named_level,
            "portfolio_proof_level": level,
            "advisory": False,
            "proof_accepted": True,
            "proof_digest": _digest(f"proof:{edge_id}"),
            "route_proof_digest": _digest(f"route-proof:{edge_id}"),
            "reaction_digest": signature,
            "trusted_precedent_sha256": _digest(f"precedent:{edge_id}"),
            "validator_version": "fixture.reaction-verifier.v1",
            "proof_source": "route_proof_bank.v1",
            "proof_bank_entry_id": f"proof-entry:{edge_id}",
            "proof_bank_entry_sha256": _digest(f"proof-entry:{edge_id}"),
            "verifier_report_id": "fixture-verifier",
            "verifier_source_sha256": _digest("fixture-verifier"),
            "verifier_target_smiles": "CCO",
        },
        "binding_sha256",
    )


def _stock_binding(
    molecule_id: str,
    smiles: str,
    *,
    commercial: bool,
) -> dict:
    boundary_type = "commercially_orderable" if commercial else "benchmark_stock"
    return _seal(
        {
            "schema_version": "exact_stock_binding.v1",
            "molecule_id": molecule_id,
            "canonical_isomeric_smiles": smiles,
            "catalog_id": f"catalog:{molecule_id}",
            "catalog_sha256": _digest(f"catalog:{molecule_id}"),
            "lookup_basis": (
                "verified_commercial_snapshot_provider"
                if commercial
                else "exact_canonical_smiles"
            ),
            "boundary_type": boundary_type,
            "benchmark_membership": not commercial,
            "commercial_orderability_claimed": commercial,
            "snapshot_digest_replayed": commercial,
            "provider_id": "builtin.snapshot-stock" if commercial else "",
            "provider_descriptor_sha256": (
                _digest("builtin.snapshot-stock") if commercial else ""
            ),
            "provider_trust_authority": (
                "autoplanner_host_builtin_allowlist.v1" if commercial else ""
            ),
            "stock_audit_sha256": _digest(f"stock-audit:{molecule_id}"),
            "evidence_sha256": _digest(f"stock-evidence:{molecule_id}"),
            "binding_authority": "strictly_replayed_route_proof_bank.v1",
            "proof_bank_authorities": [
                {
                    "proof_bank_entry_id": f"proof-entry:{molecule_id}",
                    "proof_bank_entry_sha256": _digest(
                        f"proof-entry:{molecule_id}"
                    ),
                    "stock_evidence_binding_sha256": _digest(
                        f"stock-evidence-binding:{molecule_id}"
                    ),
                }
            ],
        },
        "binding_sha256",
    )


def _bindings(
    overlay: dict,
    *,
    level: int = 3,
    commercial: bool = False,
) -> dict:
    smiles_by_id = {
        str(row["molecule_id"]): str(row["canonical_isomeric_smiles"])
        for row in overlay["molecules"]
    }
    exact_edges = {
        str(edge["hyperedge_id"]): _edge_binding(
            edge,
            smiles_by_id,
            level=level,
        )
        for edge in overlay["reaction_hyperedges"]
    }
    stock_ids = ["mol:stock-a"]
    if any(
        edge["hyperedge_id"] == "edge:middle-leaf"
        for edge in overlay["reaction_hyperedges"]
    ):
        stock_ids.append("mol:stock-b")
    stock = {
        molecule_id: _stock_binding(
            molecule_id,
            smiles_by_id[molecule_id],
            commercial=commercial,
        )
        for molecule_id in stock_ids
    }
    return _seal(
        {
            "schema_version": "route_portfolio_bindings.v1",
            "stock_molecule_ids": stock_ids,
            "edge_proof_levels": {
                edge_id: level for edge_id in exact_edges
            },
            "exact_edge_proof_bindings": exact_edges,
            "stock_bindings": stock,
            # These producer summaries are not used as proof authority.
            "stock_binding_valid": True,
            "all_materialized_terminals_proven": True,
            "accepted": True,
            "solved": True,
        }
    )


def test_two_distinct_l3_routes_compile_and_json_roundtrip_validate() -> None:
    overlay = _overlay()
    bindings = _bindings(overlay)

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["benchmark_solved"] is True
    assert proof["solved"] is True
    assert proof["procurement_ready"] is False
    assert proof["distinct_complete_route_count"] == 2
    assert len(set(proof["route_edge_set_sha256s"])) == 2
    assert all(route["connected"] is True for route in proof["routes"])
    assert all(route["acyclic"] is True for route in proof["routes"])
    assert all(route["non_zero_step_synthesis"] is True for route in proof["routes"])
    assert all(
        edge["edge_binding_sha256"] and edge["trusted_precedent_sha256"]
        for route in proof["routes"]
        for edge in route["selected_hyperedges"]
    )
    assert all(
        leaf["stock_binding_sha256"]
        for route in proof["routes"]
        for leaf in route["stock_terminals"]
    )

    roundtripped = json.loads(json.dumps(proof))
    assert validate_selected_route_parent_proof(roundtripped, overlay, bindings) == []
    assert is_solved_selected_route_parent_proof(
        roundtripped,
        overlay,
        bindings,
    )


def test_current_host_supplemental_l3_verifier_is_authoritative() -> None:
    overlay = _overlay()
    bindings = _bindings(overlay)
    for binding in bindings["exact_edge_proof_bindings"].values():
        binding["proof_source"] = (
            "supplemental_reaction_validation.v2_replayed"
        )
        binding["proof_bank_entry_id"] = ""
        binding["proof_bank_entry_sha256"] = ""
        _seal(binding, "binding_sha256")
    _seal(bindings)

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["benchmark_solved"] is True
    assert proof["distinct_complete_route_count"] == 2
    assert proof["binding_audit"]["rejected_edges"] == []


def test_l2_edges_cannot_establish_parent_proof() -> None:
    overlay = _overlay()
    bindings = _bindings(overlay, level=2)

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["benchmark_solved"] is False
    assert proof["distinct_complete_route_count"] == 0
    assert "no_complete_l3_stock_closed_synthesis_route" in proof["reasons"]
    assert not is_solved_selected_route_parent_proof(proof, overlay, bindings)


def test_single_complete_route_is_not_enough() -> None:
    overlay = _overlay(single_route=True)
    bindings = _bindings(overlay)

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["distinct_complete_route_count"] == 1
    assert proof["benchmark_solved"] is False
    assert "insufficient_distinct_complete_route_edge_sets" in proof["reasons"]


def test_duplicate_persisted_edge_set_claims_do_not_create_an_alternative() -> None:
    overlay = _overlay(single_route=True)
    bindings = _bindings(overlay)
    repeated = {
        "selected_hyperedge_ids": ["edge:direct"],
        "accepted": True,
        "solved": True,
    }
    bindings["persisted_portfolio_routes"] = [repeated, copy.deepcopy(repeated)]
    _seal(bindings)

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["producer_completion_flags_ignored"] is True
    assert proof["distinct_complete_route_count"] == 1
    assert proof["benchmark_solved"] is False


def test_missing_leaf_stock_binding_prevents_that_route_from_closing() -> None:
    overlay = _overlay()
    bindings = _bindings(overlay)
    del bindings["stock_bindings"]["mol:stock-b"]
    _seal(bindings)

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["distinct_complete_route_count"] == 1
    assert proof["benchmark_solved"] is False
    rejected = {
        row["molecule_id"]: row["reasons"]
        for row in proof["binding_audit"]["rejected_stock"]
    }
    assert "missing_exact_stock_binding" in rejected["mol:stock-b"]


def test_benchmark_closure_never_claims_procurement() -> None:
    overlay = _overlay()
    proof = compile_selected_route_parent_proof(overlay, _bindings(overlay))

    assert proof["benchmark_solved"] is True
    assert proof["any_procurement_route_ready"] is False
    assert proof["procurement_ready"] is False
    assert all(
        leaf["boundary_type"] == "benchmark_stock"
        for route in proof["routes"]
        for leaf in route["stock_terminals"]
    )


def test_all_l4_edges_and_commercial_leaves_are_procurement_ready() -> None:
    overlay = _overlay()
    bindings = _bindings(overlay, level=4, commercial=True)

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["benchmark_solved"] is True
    assert proof["procurement_route_count"] == 2
    assert proof["any_procurement_route_ready"] is True
    assert proof["procurement_ready"] is True
    assert all(route["procurement_ready"] is True for route in proof["routes"])


def test_rehashed_tampered_proof_fails_full_recompile() -> None:
    overlay = _overlay()
    bindings = _bindings(overlay)
    proof = compile_selected_route_parent_proof(overlay, bindings)
    forged = copy.deepcopy(proof)
    forged["routes"][0]["selected_hyperedges"][0]["hyperedge_id"] = "edge:forged"
    _seal(forged["routes"][0])
    _seal(forged)

    reasons = validate_selected_route_parent_proof(forged, overlay, bindings)

    assert "selected_route_parent_proof_full_recompile_mismatch" in reasons
    assert "selected_route_parent_proof_content_sha256_mismatch" not in reasons
    assert not is_solved_selected_route_parent_proof(forged, overlay, bindings)


def test_root_stock_binding_cannot_create_a_zero_step_route() -> None:
    overlay = _overlay(single_route=True)
    bindings = _bindings(overlay)
    bindings["stock_molecule_ids"].append("mol:target")
    bindings["stock_bindings"]["mol:target"] = _stock_binding(
        "mol:target",
        "CCO",
        commercial=True,
    )
    # Remove the only edge authority, leaving only target procurement.
    bindings["edge_proof_levels"] = {}
    bindings["exact_edge_proof_bindings"] = {}
    _seal(bindings)

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["distinct_complete_route_count"] == 0
    assert proof["benchmark_solved"] is False
    assert proof["procurement_ready"] is False


def test_cycle_is_not_a_route_witness() -> None:
    overlay = _overlay(single_route=True)
    overlay["reaction_hyperedges"] = [
        {
            "hyperedge_id": "edge:target-middle",
            "product_molecule_id": "mol:target",
            "precursor_molecule_ids": ["mol:middle"],
        },
        {
            "hyperedge_id": "edge:middle-target",
            "product_molecule_id": "mol:middle",
            "precursor_molecule_ids": ["mol:target"],
        },
    ]
    bindings = _bindings(overlay)

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["routes"] == []
    assert proof["benchmark_solved"] is False


def test_disconnected_missing_precursor_fails_overlay_contract() -> None:
    overlay = _overlay()
    overlay["reaction_hyperedges"][0]["precursor_molecule_ids"] = ["mol:missing"]
    bindings = _bindings(_overlay())

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["input_contract_valid"] is False
    assert any(
        reason.startswith("overlay_hyperedge_precursor_missing")
        for reason in proof["reasons"]
    )
    assert proof["benchmark_solved"] is False


def test_campaign_revision_and_digest_binding_is_recompiled() -> None:
    overlay = _overlay()
    bindings = _bindings(overlay)
    overlay_sha256 = _digest(overlay)
    campaign = _seal(
        {
            "schema_version": "selected_route_campaign_binding.v1",
            "campaign_id": "campaign:test",
            "revision": 7,
            "route_hypergraph_overlay_sha256": overlay_sha256,
            "route_portfolio_bindings_sha256": bindings["content_sha256"],
        }
    )

    proof = compile_selected_route_parent_proof(
        overlay,
        bindings,
        campaign_binding=campaign,
        expected_campaign_revision=7,
    )

    assert proof["benchmark_solved"] is True
    assert proof["input_bindings"]["campaign"]["revision"] == 7
    assert validate_selected_route_parent_proof(
        json.loads(json.dumps(proof)),
        overlay,
        bindings,
        campaign_binding=campaign,
        expected_campaign_revision=7,
    ) == []


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("trusted_precedent_sha256", "0" * 64),
        ("binding_sha256", "f" * 64),
    ],
)
def test_tampered_exact_edge_binding_is_excluded(
    field: str,
    new_value: str,
) -> None:
    overlay = _overlay()
    bindings = _bindings(overlay)
    bindings["exact_edge_proof_bindings"]["edge:direct"][field] = new_value
    _seal(bindings)

    proof = compile_selected_route_parent_proof(overlay, bindings)

    assert proof["distinct_complete_route_count"] == 1
    assert proof["benchmark_solved"] is False
    rejected_ids = {
        row["hyperedge_id"]
        for row in proof["binding_audit"]["rejected_edges"]
    }
    assert "edge:direct" in rejected_ids

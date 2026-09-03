from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from cascade_planner.legacy.application_runtime.route_portfolio import (
    build_route_verifier_bundle,
    derive_portfolio_bindings,
    solve_diverse_routes,
    validate_portfolio_replacements,
    validate_route_replacement,
)
from cascade_planner.harness import route_verifier as rv


FIXTURES = Path(__file__).parents[1] / "fixtures"
TRUSTED_REGISTRY = FIXTURES / "trusted_literature_step_registry.json"


def edge(edge_id: str, product: str, precursors: list[str], score: float) -> dict:
    return {
        "hyperedge_id": edge_id,
        "product_molecule_id": product,
        "precursor_molecule_ids": precursors,
        "rank_score": score,
        "source_channels": ["literature_exact"],
        "independent_support_groups": [f"paper:{edge_id}"],
    }


def overlay() -> dict:
    return {
        "schema_version": "route_hypergraph_overlay.v2",
        "root_molecule_id": "target",
        "validation": {"valid": True, "errors": []},
        "reaction_hyperedges": [
            edge("e-main", "target", ["shared", "side-a"], 0.95),
            edge("e-alt", "target", ["shared", "side-b"], 0.80),
            edge("e-shared", "shared", ["stock-1"], 0.9),
            edge("e-side-a", "side-a", ["stock-2"], 0.8),
            edge("e-side-b", "side-b", ["stock-3"], 0.75),
        ],
    }


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def reaction_validation(
    product: str,
    reactants: list[str],
    *,
    proof_level: str = "L3_precedent_supported",
) -> dict:
    reaction_digest = digest(
        {
            "product_canonical_isomeric_smiles": product,
            "reactant_canonical_isomeric_smiles": sorted(reactants),
        }
    )
    accepted = proof_level in {
        "L2_reaction_validated",
        "L3_precedent_supported",
        "L4_procurement_ready",
    }
    precedent = (
        {
            "schema_version": "trusted_precedent_binding.v1",
            "accepted": True,
            "authority": "human_curator",
            "authority_id": "fixture-curator",
            "binding_id": "fixture-binding",
            "source_ref": "doi:10.1000/fixture",
            "reaction_digest": reaction_digest,
        }
        if proof_level in {"L3_precedent_supported", "L4_procurement_ready"}
        else {}
    )
    mapping_checks = {
        key: True
        for key in (
            "structures_materialized",
            "mapped_reaction_present",
            "mapped_product_matches",
            "mapped_reactants_match",
            "atom_maps_complete",
            "atom_maps_unique",
            "product_atoms_have_reactant_provenance",
            "mapped_elements_preserved",
            "mapped_reactant_components_contribute",
            "scaffold_continuity_plausible",
            "ring_change_plausible",
            "bond_change_present",
            "reaction_edit_budget_plausible",
            "stereochemical_product_matches",
        )
    }
    proof = {
        "schema_version": "reaction_step_proof.v1",
        "step_id": "step:0",
        "step_index": 0,
        "product_smiles": product,
        "reactant_smiles": reactants,
        "proof_level": proof_level,
        "accepted": accepted,
        "checks": {
            **mapping_checks,
            "trusted_precedent_bound": bool(precedent),
            "procurement_bound": proof_level == "L4_procurement_ready",
        },
        "reasons": [],
        "trusted_precedent_binding": precedent,
        "reaction_digest": reaction_digest,
        "validator_version": "fixture.verifier.v1",
    }
    proof["proof_digest"] = digest(proof)
    route_proof = {
        "schema_version": "reaction_route_validation.v1",
        "accepted": accepted,
        "proof_level": proof_level,
        "weakest_link_policy": True,
        "step_count": 1,
        "reaction_validated_step_count": int(accepted),
        "step_proofs": [proof],
        "validator_version": "fixture.verifier.v1",
    }
    route_proof["proof_digest"] = digest(route_proof)
    return route_proof


def verified_route(
    product: str,
    reactants: list[str],
    *,
    step_extra: dict | None = None,
) -> dict:
    terminals = list(dict.fromkeys(reactants))
    return rv.verify_chemenzy_raw_routes(
        {
            "target": product,
            "routes": [
                {
                    "route_rank": 0,
                    "metrics": {
                        "terminal_reactants": terminals,
                        "terminal_stock_status": {item: True for item in terminals},
                    },
                    "steps": [
                        {
                            "index": 0,
                            "product": product,
                            "reactant_smiles": reactants,
                            "stock_status": {item: True for item in terminals},
                            **dict(step_extra or {}),
                        }
                    ],
                }
            ],
        },
        target_smiles=product,
    )


def strict_l3_step(
    *,
    step_id: str = "ethanol_oxidation",
    atom_mapped_reaction_smiles: str = (
        "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
    ),
) -> dict:
    source_pdf = FIXTURES / "source_evidence_stub.pdf"
    source_page = FIXTURES / "source_page.ppm"
    source_manifest = FIXTURES / "source_evidence_manifest.json"
    template_id = f"source_detail_exact_step:{step_id}"
    return {
        "step_id": step_id,
        "source_template_id": template_id,
        "source_detail_exact_step": True,
        "relation_type": "exact",
        "source_ref": "doi:10.1000/revalidatable-stitch",
        "atom_mapped_reaction_smiles": atom_mapped_reaction_smiles,
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
                "manifest_path": str(source_manifest.resolve()),
                "manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                "source_pdf_path": str(source_pdf.resolve()),
                "source_pdf_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
                "page_number": 1,
                "image_path": str(source_page.resolve()),
                "image_sha256": hashlib.sha256(source_page.read_bytes()).hexdigest(),
                "source_ref": "doi:10.1000/revalidatable-stitch",
            }
        ],
    }


def rehash(value: dict, *, field: str = "content_hash") -> None:
    payload = dict(value)
    payload.pop(field, None)
    value[field] = digest(payload)


class RoutePortfolioTest(unittest.TestCase):
    def test_exact_structure_signatures_bind_verifier_proof_and_stock(self) -> None:
        graph = overlay()
        graph["molecules"] = [
            {"molecule_id": molecule_id, "canonical_isomeric_smiles": smiles}
            for molecule_id, smiles in {
                "target": "CC=O",
                "ethanol": "CCO",
                "unrelated": "C",
            }.items()
        ]
        graph["reaction_hyperedges"] = [
            edge("e-main", "target", ["ethanol"], 0.9)
        ]
        with patch.dict(
            os.environ,
            {"AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(TRUSTED_REGISTRY)},
        ):
            verifier = verified_route("CC=O", ["CCO"], step_extra=strict_l3_step())
            # Deliberately malicious/unrelated producer claim. It must not
            # turn the overlay's methane molecule into a stock leaf.
            verifier["accepted_route"]["metrics"]["terminal_stock_status"]["C"] = True
            bindings = derive_portfolio_bindings(graph, verifier)
        self.assertEqual(bindings["edge_proof_levels"], {"e-main": 3})
        self.assertEqual(bindings["stock_molecule_ids"], ["ethanol"])
        self.assertNotIn("unrelated", bindings["stock_molecule_ids"])
        self.assertEqual(bindings["materialized_terminal_count"], 1)
        proof_binding = bindings["exact_edge_proof_bindings"]["e-main"]
        self.assertEqual(proof_binding["proof_level"], "L3_precedent_supported")
        self.assertEqual(proof_binding["portfolio_proof_level"], 3)
        self.assertRegex(proof_binding["proof_digest"], r"^[0-9a-f]{64}$")
        self.assertRegex(proof_binding["binding_sha256"], r"^[0-9a-f]{64}$")
        stock_binding = bindings["stock_bindings"]["ethanol"]
        self.assertRegex(stock_binding["catalog_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(stock_binding["evidence_sha256"], r"^[0-9a-f]{64}$")
        content_sha256 = bindings.pop("content_sha256")
        self.assertEqual(content_sha256, digest(bindings))

    def test_mapping_only_proof_is_exactly_bound_but_remains_advisory(self) -> None:
        graph = overlay()
        graph["molecules"] = [
            {"molecule_id": "target", "canonical_isomeric_smiles": "CCO"},
            {"molecule_id": "shared", "canonical_isomeric_smiles": "CC"},
            {"molecule_id": "side-a", "canonical_isomeric_smiles": "O"},
        ]
        graph["reaction_hyperedges"] = [
            edge("e-main", "target", ["shared", "side-a"], 0.9)
        ]
        verifier = verified_route(
            "CCO",
            ["CC", "O"],
            step_extra={
                "atom_mapped_reaction_smiles": (
                    "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
                )
            },
        )
        bindings = derive_portfolio_bindings(graph, verifier)

        self.assertEqual(bindings["edge_proof_levels"], {})
        exact = bindings["exact_edge_proof_bindings"]["e-main"]
        self.assertEqual(exact["proof_level"], "L2_mapping_consistent")
        self.assertEqual(exact["portfolio_proof_level"], 0)
        self.assertTrue(exact["advisory"])

    def test_replay_failed_bank_cannot_fallback_to_forged_legacy_l3(self) -> None:
        graph = overlay()
        graph["molecules"] = [
            {"molecule_id": "target", "canonical_isomeric_smiles": "CC=O"},
            {"molecule_id": "ethanol", "canonical_isomeric_smiles": "CCO"},
        ]
        graph["reaction_hyperedges"] = [
            edge("e-main", "target", ["ethanol"], 0.9)
        ]
        with patch.dict(
            os.environ,
            {"AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(TRUSTED_REGISTRY)},
        ):
            verifier = verified_route("CC=O", ["CCO"], step_extra=strict_l3_step())
            tampered = copy.deepcopy(verifier)
            entry = tampered["route_proof_bank"]["entries"][0]
            validation = entry["reaction_validation"]
            validation["step_proofs"][0]["checks"]["bond_change_present"] = False
            rehash(validation["step_proofs"][0], field="proof_digest")
            rehash(validation, field="proof_digest")
            entry["route_audit"]["reaction_validation"] = copy.deepcopy(validation)
            rehash(entry)
            rehash(tampered["route_proof_bank"])
            self.assertEqual(
                rv.validate_route_proof_bank(
                    tampered["route_proof_bank"],
                    expected_target_smiles="CC=O",
                ),
                [],
            )
            self.assertFalse(
                rv.is_replayable_route_proof_bank_entry(
                    tampered["route_proof_bank"],
                    expected_target_smiles="CC=O",
                )
            )
            # This otherwise digest-consistent legacy view must be ignored.
            tampered["reaction_validation"] = reaction_validation("CC=O", ["CCO"])
            bindings = derive_portfolio_bindings(graph, tampered)

        self.assertEqual(bindings["edge_proof_levels"], {})
        self.assertEqual(bindings["exact_edge_proof_bindings"], {})
        self.assertEqual(bindings["stock_molecule_ids"], [])
        self.assertTrue(bindings["proof_bank_fail_closed"])
        self.assertEqual(
            bindings["proof_binding_source"],
            "route_proof_bank_rejected_fail_closed",
        )

    def test_strictly_replayed_proof_bank_takes_priority_over_legacy_best(self) -> None:
        graph = overlay()
        graph["molecules"] = [
            {"molecule_id": "target", "canonical_isomeric_smiles": "CCO"},
            {"molecule_id": "shared", "canonical_isomeric_smiles": "CC"},
            {"molecule_id": "side-a", "canonical_isomeric_smiles": "O"},
        ]
        graph["reaction_hyperedges"] = [
            edge("e-main", "target", ["shared", "side-a"], 0.9)
        ]
        verifier = rv.verify_chemenzy_raw_routes(
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
        )
        # A stronger, digest-consistent legacy view must not supersede the
        # strictly replayed bank when that bank is available.
        verifier["reaction_validation"] = reaction_validation("CCO", ["CC", "O"])

        bindings = derive_portfolio_bindings(graph, verifier)

        self.assertEqual(bindings["proof_binding_source"], "strictly_replayed_route_proof_bank.v1")
        self.assertEqual(bindings["replayed_proof_bank_entry_count"], 1)
        self.assertEqual(bindings["edge_proof_levels"], {"e-main": 1})
        exact = bindings["exact_edge_proof_bindings"]["e-main"]
        self.assertEqual(exact["proof_source"], "route_proof_bank.v1")
        self.assertTrue(exact["proof_bank_entry_id"].startswith("route-proof:"))
        self.assertEqual(bindings["stock_molecule_ids"], ["shared", "side-a"])
        self.assertEqual(
            bindings["stock_bindings"]["shared"]["binding_authority"],
            "strictly_replayed_route_proof_bank.v1",
        )

    def test_strict_legacy_best_route_remains_compatible_when_bank_is_absent(self) -> None:
        graph = overlay()
        graph["molecules"] = [
            {"molecule_id": "target", "canonical_isomeric_smiles": "CCO"},
            {"molecule_id": "shared", "canonical_isomeric_smiles": "CC"},
            {"molecule_id": "side-a", "canonical_isomeric_smiles": "O"},
        ]
        graph["reaction_hyperedges"] = [
            edge("e-main", "target", ["shared", "side-a"], 0.9)
        ]
        verifier = verified_route("CCO", ["CC", "O"])
        verifier.pop("route_proof_bank")

        bindings = derive_portfolio_bindings(graph, verifier)

        self.assertFalse(bindings["proof_bank_present"])
        self.assertFalse(bindings["proof_bank_fail_closed"])
        self.assertEqual(
            bindings["proof_binding_source"],
            "strictly_replayed_legacy_best_accepted_route",
        )
        self.assertEqual(bindings["edge_proof_levels"], {"e-main": 1})
        self.assertEqual(bindings["stock_molecule_ids"], ["shared", "side-a"])

    def test_verifier_bundle_binds_two_child_targets_without_duplicate_inflation(self) -> None:
        graph = {
            "schema_version": "route_hypergraph_overlay.v2",
            "root_molecule_id": "ethanol",
            "validation": {"valid": True, "errors": []},
            "molecules": [
                {"molecule_id": "ethanol", "canonical_isomeric_smiles": "CCO"},
                {"molecule_id": "ethane", "canonical_isomeric_smiles": "CC"},
                {"molecule_id": "water", "canonical_isomeric_smiles": "O"},
                {"molecule_id": "oxygen", "canonical_isomeric_smiles": "O=O"},
            ],
            "reaction_hyperedges": [
                edge("e-ethanol", "ethanol", ["ethane", "water"], 0.9),
                edge("e-water", "water", ["oxygen"], 0.8),
            ],
        }
        with patch.dict(
            os.environ,
            {"AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(TRUSTED_REGISTRY)},
        ):
            ethanol = verified_route(
                "CCO",
                ["CC", "O"],
                step_extra=strict_l3_step(
                    step_id="common_stock_to_ethanol",
                    atom_mapped_reaction_smiles=(
                        "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
                    ),
                ),
            )
            water = verified_route(
                "O",
                ["O=O"],
                step_extra=strict_l3_step(
                    step_id="oxygen_to_water",
                    atom_mapped_reaction_smiles="[O:1]=[O:2]>>[OH2:1]",
                ),
            )
            bundle = build_route_verifier_bundle([ethanol, water, ethanol])
            bindings = derive_portfolio_bindings(graph, bundle)

        self.assertEqual(bundle["input_report_count"], 3)
        self.assertEqual(bundle["report_count"], 2)
        self.assertEqual(bundle["duplicate_report_count"], 1)
        bundle_payload = dict(bundle)
        bundle_sha256 = bundle_payload.pop("content_sha256")
        self.assertEqual(bundle_sha256, digest(bundle_payload))
        self.assertEqual(bindings["accepted_verifier_report_count"], 2)
        self.assertEqual(bindings["rejected_verifier_report_count"], 0)
        self.assertEqual(bindings["duplicate_verifier_report_count"], 1)
        self.assertEqual(bindings["proof_step_count"], 2)
        self.assertEqual(bindings["replayed_proof_bank_entry_count"], 2)
        self.assertEqual(
            bindings["edge_proof_levels"],
            {"e-ethanol": 3, "e-water": 3},
        )
        self.assertEqual(
            set(bindings["stock_molecule_ids"]),
            {"ethane", "water", "oxygen"},
        )
        self.assertEqual(
            {
                binding["verifier_target_smiles"]
                for binding in bindings["exact_edge_proof_bindings"].values()
            },
            {"CCO", "O"},
        )
        audit = bindings["verifier_bundle_audit"]
        audit_payload = dict(audit)
        audit_sha256 = audit_payload.pop("content_sha256")
        self.assertEqual(audit_sha256, digest(audit_payload))

    def test_invalid_child_report_is_rejected_without_erasing_valid_sibling(self) -> None:
        graph = {
            "schema_version": "route_hypergraph_overlay.v2",
            "root_molecule_id": "ethanol",
            "validation": {"valid": True, "errors": []},
            "molecules": [
                {"molecule_id": "ethanol", "canonical_isomeric_smiles": "CCO"},
                {"molecule_id": "ethane", "canonical_isomeric_smiles": "CC"},
                {"molecule_id": "water", "canonical_isomeric_smiles": "O"},
                {"molecule_id": "oxygen", "canonical_isomeric_smiles": "O=O"},
            ],
            "reaction_hyperedges": [
                edge("e-ethanol", "ethanol", ["ethane", "water"], 0.9),
                edge("e-water", "water", ["oxygen"], 0.8),
            ],
        }
        with patch.dict(
            os.environ,
            {"AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(TRUSTED_REGISTRY)},
        ):
            ethanol = verified_route(
                "CCO",
                ["CC", "O"],
                step_extra=strict_l3_step(
                    step_id="common_stock_to_ethanol",
                    atom_mapped_reaction_smiles=(
                        "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
                    ),
                ),
            )
            water = verified_route(
                "O",
                ["O=O"],
                step_extra=strict_l3_step(
                    step_id="oxygen_to_water",
                    atom_mapped_reaction_smiles="[O:1]=[O:2]>>[OH2:1]",
                ),
            )
            tampered_water = copy.deepcopy(water)
            bad_entry = tampered_water["route_proof_bank"]["entries"][0]
            bad_entry["materialized_route"]["steps"][0]["product"] = "N"
            rehash(bad_entry)
            rehash(tampered_water["route_proof_bank"])
            bundle = build_route_verifier_bundle([ethanol, tampered_water])
            bindings = derive_portfolio_bindings(graph, bundle)

        self.assertEqual(bindings["accepted_verifier_report_count"], 1)
        self.assertEqual(bindings["rejected_verifier_report_count"], 1)
        self.assertEqual(bindings["edge_proof_levels"], {"e-ethanol": 3})
        self.assertNotIn("e-water", bindings["exact_edge_proof_bindings"])
        self.assertNotIn("oxygen", bindings["stock_molecule_ids"])
        rejected = [
            row
            for row in bindings["verifier_bundle_audit"]["reports"]
            if row["accepted"] is not True
        ]
        self.assertEqual(len(rejected), 1)
        self.assertIn("route_proof_bank_invalid_or_replay_failed", rejected[0]["reasons"])

    def test_and_or_closure_returns_diverse_complete_routes(self) -> None:
        report = solve_diverse_routes(
            overlay(),
            stock_molecule_ids=["stock-1", "stock-2", "stock-3"],
            edge_proof_levels={
                "e-main": 3,
                "e-alt": 2,
                "e-shared": 2,
                "e-side-a": 2,
                "e-side-b": 2,
            },
            top_k=2,
        )
        self.assertEqual(len(report.routes), 2)
        self.assertTrue(all(route.complete for route in report.routes))
        self.assertTrue(all(route.reaction_validated for route in report.routes))
        self.assertEqual(
            {
                dict(route.selected_hyperedges)["target"]
                for route in report.routes
            },
            {"e-main", "e-alt"},
        )
        self.assertGreater(report.routes[1].diversity_score, 0)

        item = report.routes[0].to_dict()
        item_sha256 = item.pop("content_sha256")
        self.assertEqual(item_sha256, digest(item))
        item["base_score"] = 1.0 - item["base_score"]
        self.assertNotEqual(item_sha256, digest(item))

    def test_l3_reaction_proof_and_procurement_stock_are_orthogonal(self) -> None:
        stock_ids = ["stock-1", "stock-2", "stock-3"]
        report = solve_diverse_routes(
            overlay(),
            stock_molecule_ids=stock_ids,
            stock_bindings={
                molecule_id: {
                    "boundary_type": "commercially_orderable",
                    "commercial_orderability_claimed": True,
                }
                for molecule_id in stock_ids
            },
            edge_proof_levels={
                "e-main": 3,
                "e-alt": 3,
                "e-shared": 3,
                "e-side-a": 3,
                "e-side-b": 3,
            },
            top_k=2,
            min_reaction_proof_level=3,
        )

        self.assertEqual(len(report.routes), 2)
        for route in report.routes:
            self.assertTrue(route.complete)
            self.assertTrue(route.reaction_validated)
            self.assertTrue(route.benchmark_stock_closed)
            self.assertTrue(route.procurement_stock_closed)
            self.assertFalse(route.in_house_stock_closed)
            self.assertFalse(route.procurement_ready)
            self.assertEqual(route.weakest_proof_level, 3)

    def test_target_in_benchmark_stock_is_not_a_reaction_or_procurement_claim(self) -> None:
        graph = {
            "schema_version": "route_hypergraph_overlay.v2",
            "root_molecule_id": "target",
            "validation": {"valid": True, "errors": []},
            "reaction_hyperedges": [],
        }
        report = solve_diverse_routes(
            graph,
            stock_molecule_ids=["target"],
            stock_bindings={
                "target": {
                    "boundary_type": "benchmark_stock",
                    "benchmark_membership": True,
                    "commercial_orderability_claimed": False,
                }
            },
            edge_proof_levels={},
        )

        self.assertEqual(len(report.routes), 1)
        route = report.routes[0]
        self.assertTrue(route.complete)
        self.assertTrue(route.target_stock_available)
        self.assertTrue(route.target_benchmark_membership)
        self.assertEqual(route.target_stock_boundary_type, "benchmark_stock")
        self.assertFalse(route.target_commercially_orderable)
        self.assertFalse(route.reaction_validated)
        self.assertFalse(route.procurement_ready)
        self.assertEqual(route.weakest_proof_level, 0)
        self.assertEqual(route.selected_hyperedges, ())

    def test_candidate_stock_metrics_without_independent_audit_bind_nothing(self) -> None:
        graph = overlay()
        graph["molecules"] = [
            {"molecule_id": "target", "canonical_isomeric_smiles": "CCO"},
            {"molecule_id": "shared", "canonical_isomeric_smiles": "CC"},
            {"molecule_id": "side-a", "canonical_isomeric_smiles": "O"},
        ]
        bindings = derive_portfolio_bindings(
            graph,
            {
                "accepted_route": {
                    "steps": [{"product": "CCO", "reactant_smiles": ["CC", "O"]}],
                    "metrics": {"terminal_stock_status": {"CC": True, "O": True}},
                }
            },
        )

        self.assertEqual(bindings["stock_molecule_ids"], [])
        self.assertFalse(bindings["stock_binding_valid"])

    def test_unproved_edge_cannot_enter_complete_portfolio(self) -> None:
        report = solve_diverse_routes(
            overlay(),
            stock_molecule_ids=["stock-1", "stock-2", "stock-3"],
            edge_proof_levels={
                "e-main": 1,
                "e-alt": 1,
                "e-shared": 2,
                "e-side-a": 2,
                "e-side-b": 2,
            },
        )
        self.assertEqual(report.routes, ())
        self.assertIn("no_stock_closed_reaction_validated_route", report.reasons)

    def test_unproved_high_rank_flood_cannot_starve_low_rank_l3_route(self) -> None:
        graph = {
            "schema_version": "route_hypergraph_overlay.v2",
            "root_molecule_id": "target",
            "validation": {"valid": True, "errors": []},
            "reaction_hyperedges": [
                *[
                    edge(f"e-l0-{index:03d}", "target", [f"bad-{index:03d}"], 1.0)
                    for index in range(100)
                ],
                edge("e-valid-l3", "target", ["validated-mid"], 0.01),
                edge("e-mid-l3", "validated-mid", ["stock-good"], 0.01),
            ],
        }
        report = solve_diverse_routes(
            graph,
            stock_molecule_ids=["stock-good", *[f"bad-{index:03d}" for index in range(100)]],
            edge_proof_levels={
                **{f"e-l0-{index:03d}": 0 for index in range(100)},
                "e-valid-l3": 3,
                "e-mid-l3": 3,
            },
            max_enumerated_routes=1,
        )

        self.assertEqual(len(report.routes), 1)
        self.assertEqual(
            set(report.routes[0].hyperedge_ids),
            {"e-valid-l3", "e-mid-l3"},
        )
        self.assertEqual(report.enumerated_candidate_count, 1)
        self.assertFalse(report.truncated)

    def test_fixed_replacement_below_proof_floor_is_explicitly_rejected(self) -> None:
        validation = validate_route_replacement(
            overlay(),
            stock_molecule_ids=["stock-1", "stock-2", "stock-3"],
            edge_proof_levels={
                "e-main": 3,
                "e-alt": 1,
                "e-shared": 3,
                "e-side-a": 3,
                "e-side-b": 3,
            },
            base_selections={"target": "e-main"},
            product_molecule_id="target",
            replacement_hyperedge_id="e-alt",
        )

        self.assertFalse(validation["accepted"])
        self.assertIn(
            "fixed_edge_below_min_reaction_proof:target:e-alt",
            validation["reasons"],
        )

    def test_missing_stock_terminal_prevents_false_closure(self) -> None:
        report = solve_diverse_routes(
            overlay(),
            stock_molecule_ids=["stock-1"],
            edge_proof_levels={edge_id: 2 for edge_id in [
                "e-main", "e-alt", "e-shared", "e-side-a", "e-side-b"
            ]},
        )
        self.assertEqual(report.complete_candidate_count, 0)

    def test_replacement_is_resolved_not_spliced(self) -> None:
        accepted = validate_route_replacement(
            overlay(),
            stock_molecule_ids=["stock-1", "stock-2", "stock-3"],
            edge_proof_levels={edge_id: 2 for edge_id in [
                "e-main", "e-alt", "e-shared", "e-side-a", "e-side-b"
            ]},
            base_selections={"target": "e-main", "shared": "e-shared"},
            product_molecule_id="target",
            replacement_hyperedge_id="e-alt",
        )
        self.assertTrue(accepted["accepted"])
        selected = {
            row["product_molecule_id"]: row["hyperedge_id"]
            for row in accepted["route"]["selected_hyperedges"]
        }
        self.assertEqual(selected["target"], "e-alt")
        self.assertEqual(selected["side-b"], "e-side-b")

        rejected = validate_route_replacement(
            overlay(),
            stock_molecule_ids=["stock-1"],
            edge_proof_levels={edge_id: 2 for edge_id in [
                "e-main", "e-alt", "e-shared", "e-side-a", "e-side-b"
            ]},
            base_selections={"target": "e-main"},
            product_molecule_id="target",
            replacement_hyperedge_id="e-alt",
        )
        self.assertFalse(rejected["accepted"])
        self.assertTrue(rejected["connectivity_revalidated"])

    def test_replacement_catalog_accepts_changed_precursors_with_full_closure(self) -> None:
        levels = {
            edge_id: 3
            for edge_id in ["e-main", "e-alt", "e-shared", "e-side-a", "e-side-b"]
        }
        portfolio = solve_diverse_routes(
            overlay(),
            stock_molecule_ids=["stock-1", "stock-2", "stock-3"],
            edge_proof_levels=levels,
            top_k=1,
        )
        catalog = validate_portfolio_replacements(
            overlay(),
            portfolio=portfolio,
            stock_molecule_ids=["stock-1", "stock-2", "stock-3"],
            edge_proof_levels=levels,
        )

        self.assertEqual(catalog["schema_version"], "route_replacement_catalog.v1")
        self.assertEqual(catalog["candidate_count"], 1)
        candidate = catalog["candidates"][0]
        self.assertTrue(candidate["accepted"])
        selected = {
            row["product_molecule_id"]: row["hyperedge_id"]
            for row in candidate["route"]["selected_hyperedges"]
        }
        self.assertEqual(selected["target"], "e-alt")
        self.assertEqual(selected["side-b"], "e-side-b")
        self.assertNotIn("side-a", selected)
        self.assertEqual(
            set(candidate["route"]["stock_terminal_ids"]),
            {"stock-1", "stock-3"},
        )

    def test_replacement_catalog_rejects_open_or_l1_replacement(self) -> None:
        strong_levels = {
            edge_id: 3
            for edge_id in ["e-main", "e-alt", "e-shared", "e-side-a", "e-side-b"]
        }
        portfolio = solve_diverse_routes(
            overlay(),
            stock_molecule_ids=["stock-1", "stock-2", "stock-3"],
            edge_proof_levels=strong_levels,
            top_k=1,
        )
        weak_levels = {**strong_levels, "e-alt": 1}
        weak_catalog = validate_portfolio_replacements(
            overlay(),
            portfolio=portfolio,
            stock_molecule_ids=["stock-1", "stock-2", "stock-3"],
            edge_proof_levels=weak_levels,
        )
        self.assertFalse(weak_catalog["candidates"][0]["accepted"])

        open_catalog = validate_portfolio_replacements(
            overlay(),
            portfolio=portfolio,
            stock_molecule_ids=["stock-1"],
            edge_proof_levels=strong_levels,
        )
        self.assertFalse(open_catalog["candidates"][0]["accepted"])

    def test_replacement_catalog_is_bounded_deduplicated_and_stably_hashed(self) -> None:
        levels = {
            edge_id: 3
            for edge_id in ["e-main", "e-alt", "e-shared", "e-side-a", "e-side-b"]
        }
        portfolio = solve_diverse_routes(
            overlay(),
            stock_molecule_ids=["stock-1", "stock-2", "stock-3"],
            edge_proof_levels=levels,
            top_k=2,
        )
        kwargs = {
            "portfolio": portfolio,
            "stock_molecule_ids": ["stock-1", "stock-2", "stock-3"],
            "edge_proof_levels": levels,
            "max_candidates": 1,
        }
        first = validate_portfolio_replacements(overlay(), **kwargs)
        second = validate_portfolio_replacements(overlay(), **kwargs)

        self.assertTrue(first["truncated"])
        self.assertEqual(first["candidate_count"], 1)
        self.assertEqual(first["available_candidate_count"], 2)
        self.assertEqual(first["content_sha256"], second["content_sha256"])
        payload = dict(first)
        content_sha256 = payload.pop("content_sha256")
        self.assertEqual(content_sha256, digest(payload))


if __name__ == "__main__":
    unittest.main()

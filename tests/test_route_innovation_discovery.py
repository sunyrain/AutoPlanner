from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.application.route_innovation_capabilities import (
    match_biocatalysis_capability,
    normalize_biocatalysis_catalog,
)
from cascade_planner.application.route_innovation_discovery import (
    canonical_innovation_batch,
    discover_route_innovations,
)
from cascade_planner.application.route_innovation_precedents import (
    capabilities_from_enzyme_precedents,
)


ROOT = Path(__file__).resolve().parents[1]


def _linear_graph(smiles: list[str]) -> tuple[dict, dict]:
    molecules = {
        f"m:{index}": {"canonical_smiles": value}
        for index, value in enumerate(smiles)
    }
    edges = {
        f"edge:{index}": {
            "precursor_molecule_ids": [f"m:{index}"],
            "product_molecule_id": f"m:{index + 1}",
            "innovation_boundary_proof_level": 1,
        }
        for index in range(len(smiles) - 1)
    }
    route = {
        "route_id": "route:fixture",
        "route_family_id": "family:fixture",
        "edge_ids": list(edges),
        "reported_source_refs": ["doi:10.1000/anchor"],
    }
    return {"molecules": molecules, "edges": edges}, route


def _generic_reduction_capability() -> dict:
    return {
        "capability_id": "fixture:cyclic-ketone-reduction",
        "enzyme": {"classes": ["ketoreductase"]},
        "match": {
            "net_motif_delta": {"carbonyl": -1, "hydroxyl": 1},
            "preserved_motifs": ["alkene", "ester"],
            "element_delta": {"C": 0, "O": 0},
            "min_scaffold_similarity": 0.3,
            "max_abs_heavy_atom_delta": 0,
            "min_substrate_carbons": 6,
            "min_substrate_rings": 1,
            "min_window_steps": 2,
            "max_window_steps": 4,
            "reject_unlisted_motif_changes": True,
        },
        "selectivity_objective": "Reduce one cyclic ketone to the specified alcohol.",
        "substrate_scope_basis": "fixture analog only",
        "precedent_refs": ["doi:10.1000/fixture"],
    }


def test_discovery_finds_data_driven_multistep_window_without_target_name() -> None:
    graph, route = _linear_graph(
        [
            "CC(=O)C1CCCCC1",
            "CC(O)(O)C1CCCCC1",
            "CC(Cl)C1CCCCC1",
            "CC(O)C1CCCCC1",
        ]
    )

    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_generic_reduction_capability()],
    )

    assert discovery["candidate_count"] == 1
    candidate = discovery["candidates"][0]
    assert candidate["route_innovation"]["kind"] == "biocatalytic_superstep"
    assert candidate["route_innovation"]["step_savings"] == 2
    assert candidate["boundary"]["replaced_edge_ids"] == [
        "edge:0",
        "edge:1",
        "edge:2",
    ]
    assert discovery["program_draft_candidate_ids"] == [candidate["candidate_id"]]
    assert discovery["ingestion_hypotheses"] == []
    assert discovery["semantics"]["target_names_are_not_matching_inputs"] is True
    assert discovery["semantics"][
        "enzyme_windows_compile_to_program_drafts_not_reaction_edges"
    ] is True


def test_l0_boundary_keeps_candidate_visible_but_out_of_screen_queue() -> None:
    graph, route = _linear_graph(
        ["CC(=O)C1CCCCC1", "CC(O)(O)C1CCCCC1", "CC(O)C1CCCCC1"]
    )
    graph["edges"]["edge:0"]["innovation_boundary_proof_level"] = 0

    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_generic_reduction_capability()],
    )

    assert discovery["candidate_count"] == 1
    candidate = discovery["candidates"][0]
    assert candidate["review_status"] == "requires_boundary_materialization"
    assert "ROUTE_BOUNDARY_BELOW_L1" in candidate["warning_codes"]
    assert candidate["not_program_yet"] is True


def test_mechanism_proposal_requires_one_hop_from_route_anchor() -> None:
    graph, route = _linear_graph(["CC=O", "CCO"])
    base = {
        "proposal_id": "mechanism:oxidize-forward",
        "precursor_smiles": "CCO",
        "product_smiles": "CC(=O)O",
        "anchor_edge_ids": ["edge:0"],
        "anchor_source_refs": ["doi:10.1000/anchor"],
        "mechanistic_rationale": (
            "An alcohol oxidation can proceed through hydride transfer to the "
            "oxidant followed by hydrate oxidation."
        ),
        "elementary_steps": ["alcohol oxidation", "aldehyde hydrate oxidation"],
        "falsifiable_checks": ["LC-MS must show acid formation and mass balance"],
    }

    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[],
        mechanism_proposals=[base],
    )
    assert discovery["candidate_count"] == 1
    assert discovery["candidates"][0]["candidate_kind"] == "mechanism_one_hop"
    batch = canonical_innovation_batch(discovery)
    assert batch["hypotheses"][0]["origin_kind"] == "mechanism_hypothesis"

    rejected = discover_route_innovations(
        graph,
        route,
        capabilities=[],
        mechanism_proposals=[{**base, "precursor_smiles": "CO"}],
    )
    assert rejected["candidate_count"] == 0
    assert "mechanism_proposal_not_one_hop_from_anchor_product" in rejected[
        "rejected"
    ][0]["reasons"]


def test_primary_capability_catalog_matches_net_steroid_reduction() -> None:
    catalog = json.loads(
        (ROOT / "config" / "route_innovation_capabilities.v1.json").read_text(
            encoding="utf-8"
        )
    )
    capabilities, rejected = normalize_biocatalysis_catalog(catalog)
    hsdh = next(value for value in capabilities if value["capability_id"].startswith("hsdh:"))

    audit = match_biocatalysis_capability(
        hsdh,
        "CC12CCC3C(CCC4=CC(=O)CCC43C)C1CCC2=O",
        "CC12CCC3C(CCC4=CC(O)CCC43C)C1CCC2=O",
        window_steps=6,
    )

    assert rejected == []
    assert audit["accepted"] is True
    assert audit["transition"]["motif_delta"]["carbonyl"] == -1
    assert audit["transition"]["motif_delta"]["hydroxyl"] == 1


def test_retrieved_enzyme_precedent_becomes_capability_not_proof() -> None:
    adapted = capabilities_from_enzyme_precedents(
        [
            {
                "main_reactant": "CC(=O)C1CCCCC1",
                "precedent_reaction_id": "rxn:ketoreduction:1",
                "enzyme_ec_numbers": ["1.1.1.-"],
                "rhea_ids": ["12345"],
                "evidence": {
                    "precedent_product_main_smiles": "CC(O)C1CCCCC1",
                },
            }
        ]
    )

    assert adapted["rejected"] == []
    capability = adapted["capabilities"][0]
    assert capability["capability_id"] == "enzyme-precedent:rxn:ketoreduction:1"
    assert capability["precedent_refs"] == [
        "enzyme-reaction:rxn:ketoreduction:1",
        "rhea:12345",
    ]
    assert capability["authority_scope"] == "search_prior_only"
    assert capability["not_reaction_proof"] is True
    assert capability["exact_substrate_validated"] is False

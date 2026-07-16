from __future__ import annotations

from scripts.build_bufotalin_innovation_review import (
    build_ingestion_batch,
    build_review,
)


def _dossier() -> dict:
    smiles = [
        "CC12CCC3C(CCC4=CC(=O)CCC43C)C1CCC2=O",
        "CC12CCC(=O)C=C1CCC1C2CCC2(C)C1CCC21OCCO1",
        "CC12CCC(=O)CC1CCC1C2CCC2(C)C1CCC21OCCO1",
        "CC12CCC(O)CC1CCC1C2CCC2(C)C1CCC21OCCO1",
        "CC12CCC(O)CC1CCC1C2CCC2(C)C1C(Br)CC21OCCO1",
        "CC12CCC(O)C=C1CCC1C2CCC2(C)C1CCC21OCCO1",
        "CC12CCC3C(CCC4=CC(O)CCC43C)C1CCC2=O",
        *("C" * size for size in range(2, 16)),
    ]
    steps = []
    for index in range(20):
        steps.append(
            {
                "edge_id": f"edge:{index}",
                "reactant_molecule_id": f"m:{index}",
                "reactant_label": str(index),
                "reactant_smiles": smiles[index],
                "product_molecule_id": f"m:{index + 1}",
                "product_label": str(index + 1),
                "product_smiles": smiles[index + 1],
                "proof_level": 1,
            }
        )
    return {
        "route": {
            "paper_reported_step_count": 15,
            "planner_hypothesis_step_count": 5,
        },
        "source_evidence": {
            "procedure_sequences_agree": True,
            "source_ref": "doi:10.1000/fixture",
        },
        "steps": steps,
    }


def _mechanism_proposal() -> dict:
    return {
        "proposal_id": "mechanism:fixture-forward-hop",
        "precursor_smiles": "C" * 15,
        "product_smiles": "O" + "C" * 15,
        "anchor_edge_ids": ["edge:19"],
        "anchor_source_refs": ["doi:10.1000/fixture"],
        "mechanistic_rationale": (
            "A terminal oxidation hypothesis is submitted only to exercise the "
            "generic one-hop host admission boundary."
        ),
        "elementary_steps": ["hydrogen abstraction", "oxygen rebound"],
        "falsifiable_checks": ["LC-MS must show the proposed mass shift"],
    }


def test_benchmark_adapter_delegates_to_generic_discovery() -> None:
    review = build_review(
        _dossier(),
        mechanism_proposals=[_mechanism_proposal()],
        input_artifact="fixture.json",
    )

    assert review["schema_version"] == "bufotalin_route_innovation_review.v2"
    assert review["baseline"]["physical_step_count"] == 20
    assert review["baseline"]["maximum_single_window_step_savings"] == 5
    assert review["triage"]["ready_for_enzyme_screen"]
    assert review["triage"]["mechanism_review_only"] == [
        "mechanism:fixture-forward-hop"
    ]
    assert review["triage"]["canonical_edges_created"] == 0
    assert review["semantics"]["target_names_are_not_matching_inputs"] is True
    assert (
        review["semantics"]["benchmark_adapter_contains_no_chemistry_match_rules"]
        is True
    )

    superstep = next(
        value
        for value in review["candidates"]
        if value["route_innovation"]["step_savings"] == 5
    )
    assert superstep["route_innovation"]["kind"] == "biocatalytic_superstep"
    assert superstep["boundary"]["replaced_edge_ids"] == [
        "edge:0",
        "edge:1",
        "edge:2",
        "edge:3",
        "edge:4",
        "edge:5",
    ]
    mechanism = next(
        value
        for value in review["candidates"]
        if value["candidate_kind"] == "mechanism_one_hop"
    )
    assert mechanism["route_innovation"]["reported_in_anchor_source"] is False
    assert mechanism["boundary"]["new_connectivity_hypothesis"] is True


def test_review_separates_program_drafts_from_canonical_ingestion_hypotheses() -> None:
    review = build_review(
        _dossier(),
        mechanism_proposals=[_mechanism_proposal()],
    )
    batch = build_ingestion_batch(review)

    assert batch["schema_version"] == "canonical_route_innovation_ingestion_batch.v1"
    assert len(batch["hypotheses"]) == 1
    assert batch["hypotheses"][0]["origin_kind"] == "mechanism_hypothesis"
    assert batch["excluded_program_candidate_ids"] == review[
        "program_draft_candidate_ids"
    ]
    assert batch["semantics"]["materialization_does_not_grant_proof"] is True
    assert batch["semantics"][
        "biocatalytic_supersteps_are_not_canonical_reaction_hypotheses"
    ] is True

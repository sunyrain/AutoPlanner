from __future__ import annotations

from cascade_planner.harness.agentic_blackboard import _row_summary
from cascade_planner.harness.codex_edge_verification import (
    verify_codex_consensus_graph,
)


def _exact_row() -> dict:
    return {
        "row_id": "exact:oxidation",
        "product_smiles": "CC=O",
        "reactant_smiles": ["CCO"],
        "source_ref": "doi:10.1000/example",
        "source_template_id": "source_detail_exact_step:oxidation",
        "source_detail_exact_step": True,
        "relation_type": "exact",
        "condition_candidate": {
            "schema_version": "condition_candidate.v1",
            "reagent": "oxidant",
            "solvent": "ethyl acetate",
            "temperature": "25 C",
            "time": "2 h",
        },
        "exact_step_validation": {
            "accepted": True,
            "allowed_for_one_step_source": True,
            "reasons": [],
        },
        "source_evidence": [
            {
                "schema_version": "materialized_source_evidence.v1",
                "source_ref": "doi:10.1000/example",
                "document_id": "example-si",
                "manifest_sha256": "a" * 64,
                "source_pdf_sha256": "b" * 64,
                "page_number": 3,
                "image_sha256": "c" * 64,
            }
        ],
        "accepted": True,
        "validated": True,
    }


def test_blackboard_exact_row_keeps_structured_and_projected_conditions() -> None:
    summary = _row_summary(_exact_row(), 1, compilation_accepted=True)

    assert summary["condition_candidate"]["reagent"] == "oxidant"
    assert summary["exact_step_validation"]["accepted"] is True
    assert summary["source_evidence"][0]["page_number"] == 3
    assert summary["conditions"] == [
        "reagent=oxidant",
        "solvent=ethyl acetate",
        "temperature=25 C",
        "time=2 h",
    ]


def test_edge_materialization_consumes_structured_exact_row_conditions() -> None:
    report = verify_codex_consensus_graph(
        {
            "schema_version": "route_consensus_graph.v1",
            "target_smiles": "CC=O",
            "steps": [
                {
                    "step_id": "oxidation",
                    "product_smiles": "CC=O",
                    "precursor_smiles": ["CCO"],
                    "reaction_family": "alcohol oxidation",
                }
            ],
        },
        exact_rows=[_exact_row()],
        atom_mapper=lambda _: [
            "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
        ],
        enable_optional_rxnmapper=False,
    )

    candidate = report["edge_verifications"][0]["materialized_candidate"]
    assert candidate["condition_candidate"]["solvent"] == "ethyl acetate"
    assert candidate["conditions"] == [
        "reagent=oxidant",
        "solvent=ethyl acetate",
        "temperature=25 C",
        "time=2 h",
    ]
    assert candidate["exact_step_validation"]["accepted"] is True
    assert candidate["source_evidence"][0]["image_sha256"] == "c" * 64

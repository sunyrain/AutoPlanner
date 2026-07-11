from __future__ import annotations

from cascade_planner.harness.visual_structure_extraction import (
    validate_visual_structure_chain,
)


def test_source_local_compound_label_cannot_bind_two_structures() -> None:
    report = validate_visual_structure_chain(
        {
            "schema_version": "visual_structure_candidate_chain.v1",
            "source_ref": "patent:WO2021250648A1",
            "document_id": "WO2021250648A1-main",
            "steps": [
                {
                    "step_id": "first",
                    "product_label": "C16",
                    "product_smiles": "CCN",
                    "reactant_labels": ["start-a"],
                    "reactant_smiles": ["CC"],
                    "condition_candidate": "reagent A",
                    "source_locator": "page 10, example 1",
                },
                {
                    "step_id": "second",
                    "product_label": "c16",
                    "product_smiles": "CCO",
                    "reactant_labels": ["start-b"],
                    "reactant_smiles": ["CO"],
                    "condition_candidate": "reagent B",
                    "source_locator": "page 12, example 2",
                },
            ],
        },
        require_contiguous=False,
    )

    assert report["accepted"] is False
    audit = report["compound_binding_audit"]
    assert audit["accepted"] is False
    assert audit["independent_source_group"] == "patent:WO2021250648A1"
    assert audit["conflict_count"] == 1
    assert audit["conflicts"][0]["normalized_label"] == "c16"
    assert {row["canonical_smiles"] for row in audit["conflicts"][0]["structures"]} == {
        "CCN",
        "CCO",
    }
    assert all(step["accepted"] is False for step in report["steps"])

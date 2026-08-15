from cascade_planner.application.paper_equivalent_metric import (
    compile_paper_equivalent_metric,
)


def test_paper_equivalent_ignores_validation_evidence_and_minimum_route_count() -> None:
    result = compile_paper_equivalent_metric(
        {
            "minimum_routes": 2,
            "routes": [
                {
                    "route_family_id": "family:1",
                    "skeleton_id": "skeleton:1",
                    "edge_ids": ["edge:1"],
                    "leaf_molecule_ids": ["mol:a", "mol:b"],
                    "stock_closed": True,
                    "reaction_validated": False,
                    "evidence_closed": False,
                }
            ],
        },
        stock_oracle={
            "binding": {
                "catalog_name": "ZINC + eMolecules",
                "member_count": 39_684_411,
                "identity_key": "full_inchikey",
            }
        },
    )
    assert result["paper_equivalent_solved"] is True
    assert result["paper_equivalent_solved_route_count"] == 1
    assert result["stock_comparable_to_synthex"] is True


def test_paper_equivalent_marks_different_stock_as_not_comparable() -> None:
    result = compile_paper_equivalent_metric(
        {"routes": []},
        stock_oracle={
            "binding": {
                "catalog_name": "eMolecules-only",
                "member_count": 23_081_629,
            }
        },
    )
    assert result["paper_equivalent_solved"] is False
    assert result["stock_comparable_to_synthex"] is False

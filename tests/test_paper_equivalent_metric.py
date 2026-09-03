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
                "member_count": 39_478_827,
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


def test_paper_reach_solved_and_strict_b2_are_independent_axes() -> None:
    result = compile_paper_equivalent_metric(
        {
            "routes": [
                {
                    "route_family_id": "family:reach",
                    "skeleton_id": "skeleton:reach",
                    "edge_ids": ["edge:1"],
                    "leaf_molecule_ids": ["mol:a"],
                    "stock_closed": False,
                }
            ],
            "gates": {"B2_host_validated_routes": False},
            "counts": {"reaction_validated_skeletons": 0},
        },
        stock_oracle={"binding": {"catalog_name": "local", "member_count": 1}},
    )
    assert result["paper_reach"] is True
    assert result["paper_solved"] is False
    assert result["strict_b2"]["host_validated"] is False
    assert result["strict_b2"]["independent_from_paper_reach_and_solved"] is True


def test_strict_b2_can_be_true_without_changing_paper_topology_metric() -> None:
    result = compile_paper_equivalent_metric(
        {
            "routes": [
                {
                    "route_family_id": "family:solved",
                    "skeleton_id": "skeleton:solved",
                    "edge_ids": ["edge:1"],
                    "leaf_molecule_ids": ["mol:a"],
                    "stock_closed": True,
                }
            ],
            "gates": {"B2_host_validated_routes": True},
            "counts": {"reaction_validated_skeletons": 1},
        },
        stock_oracle={
            "binding": {
                "catalog_name": "ZINC+eMolecules",
                "member_count": 39_478_827,
                "identity_key": "full_inchikey",
            }
        },
    )
    assert result["paper_solved"] is True
    assert result["strict_b2"]["host_validated"] is True
    assert result["strict_b2"]["host_validated_route_count"] == 1


def test_paper_declared_entry_count_is_not_misused_as_unique_membership_count() -> None:
    result = compile_paper_equivalent_metric(
        {"routes": []},
        stock_oracle={
            "binding": {
                "catalog_name": "ZINC+eMolecules",
                "member_count": 39_684_411,
                "identity_key": "full_inchikey",
            }
        },
    )

    assert result["stock_comparable_to_synthex"] is False
    assert result["required_stock_unique_member_count"] == 39_478_827
    assert result["paper_declared_stock_entry_count"] == 39_684_411

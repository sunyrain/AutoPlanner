from __future__ import annotations

import pytest

from cascade_planner.application.reactionjson_replay import (
    PRIMITIVES,
    REACTIONJSON_PROFILE,
    ReactionJsonReplayError,
    replay_reactionjson,
)


@pytest.mark.parametrize(
    ("product", "operation", "expected"),
    [
        (
            "[CH3:1][CH3:2]",
            {"op": "break_bond", "map_a": 1, "map_b": 2},
            ["C", "C"],
        ),
        (
            "[CH3:1].[CH3:2]",
            {"op": "add_bond", "map_a": 1, "map_b": 2, "order": 1},
            ["CC"],
        ),
        (
            "[CH2:1]=[CH2:2]",
            {"op": "change_bond_order", "map_a": 1, "map_b": 2, "delta": -1},
            ["CC"],
        ),
        (
            "[CH3:1]",
            {"op": "change_atom", "map_idx": 1, "element": "N"},
            ["N"],
        ),
        (
            "[C:1]",
            {"op": "set_explicit_h", "map_idx": 1, "count": 4},
            ["C"],
        ),
        (
            "[CH3:1]",
            {"op": "add_group", "map_idx": 1, "fragment_smiles": "[*][CH3:2]"},
            ["CC"],
        ),
        (
            "[CH3:1][CH3:2]",
            {"op": "remove_group", "map_indices": [2]},
            ["C"],
        ),
        (
            "[C@H:1]([F:2])([Cl:3])[Br:4]",
            {"op": "invert_stereocenter", "map_idx": 1},
            ["F[C@H](Cl)Br"],
        ),
        (
            "[C@H:1]([F:2])([Cl:3])[Br:4]",
            {"op": "clear_stereocenter", "map_idx": 1},
            ["FC(Cl)Br"],
        ),
        (
            "[CH3:1][CH:2]=[CH:3][CH3:4]",
            {
                "op": "set_bond_stereo",
                "map_a": 2,
                "map_b": 3,
                "stereo": "E",
                "stereo_atom_maps": [1, 4],
            },
            ["C/C=C/C"],
        ),
    ],
)
def test_public_profile_replays_all_ten_primitives(
    product: str, operation: dict, expected: list[str]
) -> None:
    audit = replay_reactionjson(
        mapped_product_smiles=product,
        operations=[operation],
        expected_precursor_smiles=expected,
    )

    assert audit["accepted"] is True
    assert audit["profile"] == REACTIONJSON_PROFILE
    assert audit["primitive_counts"][operation["op"]] == 1
    assert audit["expected_precursors_match"] is True
    assert audit["semantics"]["replay_grants_no_reaction_proof"] is True


def test_public_profile_preserves_order_and_is_deterministic() -> None:
    operations = [
        {"op": "change_bond_order", "map_a": 1, "map_b": 2, "delta": -1},
        {"op": "break_bond", "map_a": 1, "map_b": 2},
    ]
    first = replay_reactionjson(
        mapped_product_smiles="[CH2:1]=[CH2:2]",
        operations=operations,
        expected_precursor_smiles=["C", "C"],
    )
    second = replay_reactionjson(
        mapped_product_smiles="[CH2:1]=[CH2:2]",
        operations=operations,
        expected_precursor_smiles=["C", "C"],
    )

    assert first == second
    assert first["operation_count"] == 2
    assert first["content_sha256"] == second["content_sha256"]


@pytest.mark.parametrize(
    ("product", "operations", "reason"),
    [
        ("CC", [{"op": "break_bond", "map_a": 1, "map_b": 2}], "product_maps_invalid"),
        (
            "[CH3:1][CH3:2]",
            [{"op": "magic", "map_a": 1, "map_b": 2}],
            "primitive_unknown",
        ),
        (
            "[CH3:1][CH3:2]",
            [{"op": "break_bond", "map_a": 1, "map_b": 2, "comment": "ignore"}],
            "operation_field_unknown",
        ),
        (
            "[CH3:1][CH3:2]",
            [{"op": "break_bond", "map_a": 1, "map_b": 99}],
            "map_not_found",
        ),
        (
            "[CH3:1]",
            [{"op": "add_group", "map_idx": 1, "fragment_smiles": "CC"}],
            "fragment_attachment_invalid",
        ),
        (
            "[CH3:1]",
            [{"op": "add_group", "map_idx": 1, "fragment_smiles": "[*][CH3:1]"}],
            "fragment_map_collision",
        ),
        (
            "[CH3:1][CH3:2]",
            [{"op": "add_bond", "map_a": 1, "map_b": 2}],
            "bond_already_exists",
        ),
    ],
)
def test_public_profile_fails_closed(
    product: str, operations: list[dict], reason: str
) -> None:
    with pytest.raises(ReactionJsonReplayError, match=reason):
        replay_reactionjson(mapped_product_smiles=product, operations=operations)


def test_expected_precursors_are_a_required_conformance_oracle_when_supplied() -> None:
    with pytest.raises(
        ReactionJsonReplayError, match="reactionjson_expected_precursors_mismatch"
    ):
        replay_reactionjson(
            mapped_product_smiles="[CH3:1][CH3:2]",
            operations=[{"op": "break_bond", "map_a": 1, "map_b": 2}],
            expected_precursor_smiles=["CC"],
        )


def test_public_primitive_inventory_is_frozen() -> None:
    assert PRIMITIVES == (
        "break_bond",
        "add_bond",
        "change_bond_order",
        "change_atom",
        "set_explicit_h",
        "add_group",
        "remove_group",
        "invert_stereocenter",
        "clear_stereocenter",
        "set_bond_stereo",
    )

from __future__ import annotations

import pytest

from cascade_planner.application.chemical_strategy_critic import (
    critique_strategy_candidate,
)
from cascade_planner.application.reactionjson_replay import (
    PRIMITIVES,
    REACTIONJSON_PROFILE,
    ReactionJsonReplayError,
    diagnose_reactionjson,
    replay_reactionjson,
)
from cascade_planner.application.routejson_compiler import RouteJSONCompiler


def test_critic_reuses_host_bound_routejson_atom_map_namespace() -> None:
    """The v36 RouteJSON replay must not be rerun in a fresh map namespace."""

    product = (
        "CCOC(=O)C1(O)c2cc(Br)c(C3CC(=O)C[C@@H](C)O3)cc2C(=O)C1(C)O"
    )
    precursor = (
        "CCOC(=O)C1(O)c2cc(Br)c(C=CC(=O)C[C@@H](C)O)cc2C(=O)C1(C)O"
    )
    operations = [
        {"map_a": 19, "map_b": 26, "op": "break_bond"},
        {"delta": 1, "map_a": 19, "map_b": 20, "op": "change_bond_order"},
    ]
    host_audit = replay_reactionjson(
        mapped_product_smiles=(
            "[CH3:1][CH2:2][O:3][C:4](=[O:5])[C:6]1([OH:7])"
            "[c:8]2[cH:9][c:10]([Br:31])[c:11]([CH:19]3[CH2:20]"
            "[C:21](=[O:22])[CH2:23][C@@H:24]([CH3:25])[O:26]3)"
            "[cH:12][c:13]2[C:14](=[O:15])[C:16]1([CH3:17])[OH:18]"
        ),
        operations=operations,
        expected_precursor_smiles=[precursor],
    )

    result = critique_strategy_candidate(
        product_smiles=product,
        precursor_smiles=[precursor],
        reaction_operations=operations,
        reactionjson_audit=host_audit,
        reaction_family="intramolecular hydroalkoxylation",
    )

    assert result["accepted"] is True
    assert "critic_reaction_operations_replay_failed" not in result[
        "blocking_reasons"
    ]
    assert "reactionjson_host_map_namespace_reused" in result["observations"]
    assert result["reactionjson_audit"]["accepted"] is True


def test_critic_reuses_host_map_when_compiler_only_normalized_stereo() -> None:
    """Stereo-only compiler normalization must not trigger a fresh-map replay."""

    operations = [{"op": "break_bond", "map_a": 2, "map_b": 4}]
    materialized = RouteJSONCompiler().compile_step(
        mapped_product_smiles="[CH3:1][C@H:2]([CH3:3])[OH:4]",
        operations=operations,
        expected_product_smiles="CC(C)O",
    )
    assert materialized.audit["mapped_product_stereo_normalized"] is True

    result = critique_strategy_candidate(
        product_smiles="CC(C)O",
        precursor_smiles=["CC(C)", "O"],
        reaction_operations=operations,
        reactionjson_audit=materialized.audit,
        reaction_family="host compiler stereo normalization probe",
    )

    assert result["accepted"] is True
    assert "critic_reactionjson_product_binding_mismatch" not in result[
        "blocking_reasons"
    ]
    assert "critic_reaction_operations_replay_failed" not in result[
        "blocking_reasons"
    ]
    assert "reactionjson_host_map_namespace_reused_stereo_normalized" in result[
        "observations"
    ]


def test_critic_accepts_only_replayed_external_oxygen_provenance() -> None:
    operations = [{"op": "remove_group", "map_indices": [3]}]
    host_audit = replay_reactionjson(
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        operations=operations,
        expected_precursor_smiles=["CC"],
    )
    bound_audit = {
        **host_audit,
        "external_atom_source_required": True,
        "external_atom_source_status": "declared_graph_edit_requires_validation",
        "external_atom_source_grants_reaction_proof": False,
    }

    accepted = critique_strategy_candidate(
        product_smiles="CCO",
        precursor_smiles=["CC"],
        reaction_operations=operations,
        reactionjson_audit=bound_audit,
        reaction_family="P450 C-H hydroxylation",
        enzyme="P450",
    )
    unbound = critique_strategy_candidate(
        product_smiles="CCO",
        precursor_smiles=["CC"],
        reaction_operations=operations,
        reactionjson_audit=host_audit,
        reaction_family="P450 C-H hydroxylation",
        enzyme="P450",
    )

    assert accepted["accepted"] is True
    assert "critic_atom_provenance_deficit" not in accepted["blocking_reasons"]
    assert "critic_external_atom_source_bound_by_reactionjson" in accepted[
        "observations"
    ]
    assert "critic_external_atom_source_requires_validation" in accepted[
        "uncertainties"
    ]
    assert unbound["accepted"] is False
    assert "critic_atom_provenance_deficit" in unbound["blocking_reasons"]


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
            {"op": "add_bond", "map_a": 1, "map_b": 2},
            ["CC"],
        ),
        (
            "[CH2:1]=[CH2:2]",
            {"op": "change_bond_order", "map_a": 1, "map_b": 2, "delta": -1},
            ["CC"],
        ),
        (
            "[NH4+:1]",
            {"op": "change_atom", "map_idx": 1, "formal_charge": 0},
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


def test_add_bond_then_change_bond_order_creates_a_double_bond() -> None:
    audit = replay_reactionjson(
        mapped_product_smiles="[CH2:1].[CH2:2]",
        operations=[
            {"op": "add_bond", "map_a": 1, "map_b": 2},
            {"op": "change_bond_order", "map_a": 1, "map_b": 2, "delta": 1},
        ],
        expected_precursor_smiles=["C=C"],
    )

    assert audit["accepted"] is True
    assert audit["expected_precursors_match"] is True


@pytest.mark.parametrize(
    "field",
    [{"element": "N"}, {"atomic_num": 7}],
)
def test_change_atom_rejects_element_transmutation(field: dict) -> None:
    with pytest.raises(
        ReactionJsonReplayError,
        match="reactionjson_change_atom_transmutation_forbidden",
    ):
        replay_reactionjson(
            mapped_product_smiles="[CH3:1]",
            operations=[{"op": "change_atom", "map_idx": 1, **field}],
        )


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


def test_add_group_assigns_deterministic_fresh_maps_to_unmapped_atoms() -> None:
    operation = {
        "op": "add_group",
        "map_idx": 7,
        "fragment_smiles": "[*][Mg]Br",
    }

    first = replay_reactionjson(
        mapped_product_smiles="[CH3:7]",
        operations=[operation],
        expected_precursor_smiles=["C[Mg]Br"],
    )
    second = replay_reactionjson(
        mapped_product_smiles="[CH3:7]",
        operations=[operation],
        expected_precursor_smiles=["C[Mg]Br"],
    )

    assert first == second
    assert first["accepted"] is True
    assert len(first["mapped_precursor_smiles"]) == 1
    assert ":7]" in first["mapped_precursor_smiles"][0]
    assert ":8]" in first["mapped_precursor_smiles"][0]
    assert ":9]" in first["mapped_precursor_smiles"][0]


def test_add_group_order_overrides_dummy_attachment_bond() -> None:
    audit = replay_reactionjson(
        mapped_product_smiles="[CH2:1]",
        operations=[
            {
                "op": "add_group",
                "map_idx": 1,
                "fragment_smiles": "[*]O",
                "order": 2,
            }
        ],
        expected_precursor_smiles=["C=O"],
    )

    assert audit["accepted"] is True
    assert audit["expected_precursors_match"] is True


def test_add_group_without_order_uses_dummy_attachment_bond() -> None:
    audit = replay_reactionjson(
        mapped_product_smiles="[CH2:1]",
        operations=[
            {
                "op": "add_group",
                "map_idx": 1,
                "fragment_smiles": "[*]=O",
            }
        ],
        expected_precursor_smiles=["C=O"],
    )

    assert audit["accepted"] is True
    assert audit["expected_precursors_match"] is True


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
        (
            "[CH3:1].[OH:2]",
            [{"op": "add_bond", "map_a": 1, "map_b": 2, "order": 1.5}],
            "aromatic_bond_requires_aromatic_atoms",
        ),
        (
            "[CH3:1]",
            [
                {
                    "op": "add_group",
                    "map_idx": 1,
                    "fragment_smiles": "[*]O",
                    "order": 1.5,
                }
            ],
            "aromatic_bond_requires_aromatic_atoms",
        ),
    ],
)
def test_public_profile_fails_closed(
    product: str, operations: list[dict], reason: str
) -> None:
    with pytest.raises(ReactionJsonReplayError, match=reason):
        replay_reactionjson(mapped_product_smiles=product, operations=operations)


@pytest.mark.parametrize(
    ("product", "operation", "endpoint_aromaticity"),
    [
        (
            "[CH3:1].[OH:2]",
            {"op": "add_bond", "map_a": 1, "map_b": 2, "order": 1.5},
            {"map_a": False, "map_b": False},
        ),
        (
            "[CH3:1]",
            {
                "op": "add_group",
                "map_idx": 1,
                "fragment_smiles": "[*]O",
                "order": 1.5,
            },
            {"anchor": False, "fragment_attachment": False},
        ),
    ],
)
def test_aromatic_order_diagnostic_identifies_only_the_failed_operation(
    product: str,
    operation: dict,
    endpoint_aromaticity: dict[str, bool],
) -> None:
    diagnostic = diagnose_reactionjson(
        mapped_product_smiles=product,
        operations=[operation],
    )

    assert diagnostic["replay_succeeded"] is False
    assert diagnostic["operation_index"] == 0
    assert diagnostic["failed_operation"] == operation
    assert diagnostic["endpoint_aromaticity"] == endpoint_aromaticity
    assert diagnostic["allowed_orders"] == [1, 2, 3]


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

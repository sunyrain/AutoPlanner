from __future__ import annotations

import pytest

from cascade_planner.application.reactionjson_replay import ReactionJsonReplayError
from cascade_planner.application.routejson_compiler import RouteJSONCompiler
from cascade_planner.routes.admission import audit_retrosynthetic_candidate


def test_compile_step_materializes_precursors_deterministically() -> None:
    compiler = RouteJSONCompiler()
    first = compiler.compile_step(
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        operations=[{"op": "break_bond", "map_a": 2, "map_b": 3}],
        expected_product_smiles="CCO",
    )
    second = compiler.compile_step(
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        operations=[{"op": "break_bond", "map_a": 2, "map_b": 3}],
        expected_product_smiles="CCO",
    )

    assert first == second
    assert first.product_smiles == "CCO"
    assert first.precursor_smiles == ("CC", "O")
    assert first.mapped_precursor_smiles == ("[CH3:1][CH3:2]", "[OH2:3]")


def test_compile_step_aligns_mapped_fragments_to_canonical_fragment_order() -> None:
    materialized = RouteJSONCompiler().compile_step(
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        operations=[{"op": "break_bond", "map_a": 1, "map_b": 2}],
        expected_product_smiles="CCO",
    )

    assert materialized.precursor_smiles == ("C", "CO")
    assert materialized.mapped_precursor_smiles == (
        "[CH4:1]",
        "[CH3:2][OH:3]",
    )


def test_compile_step_aligns_unique_constitution_when_symmetry_drops_stereo() -> None:
    materialized = RouteJSONCompiler().compile_step(
        mapped_product_smiles="[CH3:1][C@H:2]([CH3:3])[CH2:4][OH:5]",
        operations=[{"op": "break_bond", "map_a": 4, "map_b": 5}],
    )

    assert materialized.precursor_smiles == ("CC(C)C", "O")
    assert materialized.mapped_precursor_smiles == (
        "[CH3:1][C@H:2]([CH3:3])[CH3:4]",
        "[OH2:5]",
    )


def test_compile_step_uses_host_canonical_product_when_mapped_stereo_is_stale() -> None:
    materialized = RouteJSONCompiler().compile_step(
        mapped_product_smiles="[CH3:1][C@H:2]([CH3:3])[OH:4]",
        operations=[{"op": "break_bond", "map_a": 2, "map_b": 4}],
        expected_product_smiles="CC(C)O",
    )

    assert materialized.product_smiles == "CC(C)O"
    assert materialized.audit["mapped_product_stereo_normalized"] is True
    assert materialized.audit["canonical_product_smiles"] == "CC(C)O"


@pytest.mark.parametrize(
    ("mapped_product", "operations", "expected_product", "expected_precursor"),
    [
        (
            "[CH3:1][CH2:2][OH:3]",
            [{"op": "remove_group", "map_indices": [3]}],
            "CCO",
            "CC",
        ),
        (
            "[CH2:1]=[C:2]([CH3:3])[CH3:4]",
            [
                {
                    "op": "remove_group",
                    "map_indices": [1],
                },
                {
                    "op": "add_group",
                    "map_idx": 2,
                    "fragment_smiles": "[*]=[O:5]",
                },
            ],
            "C=C(C)C",
            "CC(C)=O",
        ),
        (
            "[CH3:1][CH:2]([I:3])[CH3:4]",
            [
                {
                    "op": "remove_group",
                    "map_indices": [3],
                },
                {
                    "op": "add_group",
                    "map_idx": 2,
                    "fragment_smiles": "[*][OH:5]",
                },
            ],
            "CC(C)I",
            "CC(C)O",
        ),
    ],
)
def test_compile_step_binds_replayed_external_atom_sources_once(
    mapped_product: str,
    operations: list[dict],
    expected_product: str,
    expected_precursor: str,
) -> None:
    materialized = RouteJSONCompiler().compile_step(
        mapped_product_smiles=mapped_product,
        operations=operations,
        expected_product_smiles=expected_product,
    )

    assert materialized.precursor_smiles == (expected_precursor,)
    assert materialized.audit["external_atom_source_required"] is True
    assert materialized.audit["external_atom_source_status"] == (
        "declared_graph_edit_requires_validation"
    )
    assert materialized.audit["external_atom_source_grants_reaction_proof"] is False


def test_forged_external_atom_flags_do_not_bypass_admission() -> None:
    mapped_product = "[CH3:1][CH2:2][OH:3]"
    audit = audit_retrosynthetic_candidate(
        "CCO",
        ["CC"],
        mapped_product_smiles=mapped_product,
        reaction_operations=[{"op": "break_bond", "map_a": 1, "map_b": 2}],
        reactionjson_audit={
            "accepted": True,
            "mapped_product_smiles": mapped_product,
            "external_atom_source_required": True,
            "external_atom_source_grants_reaction_proof": False,
        },
    )

    assert audit["accepted"] is False
    assert audit["replayed_external_atom_deficit_bound"] is False
    assert "element_inventory_not_conserved" in audit["reasons"]


def test_compile_linear_route_preserves_atom_maps_across_steps() -> None:
    compiler = RouteJSONCompiler()
    compiled = compiler.compile_linear_route(
        mapped_target_smiles="[CH3:10][CH2:20][OH:30]",
        steps=[
            {
                "product_smiles": "CCO",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 20, "map_b": 30}
                ],
            },
            {
                "product_smiles": "CC",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 10, "map_b": 20}
                ],
            },
        ],
    )

    assert compiled[1].mapped_product_smiles == "[CH3:10][CH3:20]"
    assert compiled[1].precursor_smiles == ("C", "C")
    assert set(compiled[1].mapped_precursor_smiles) == {"[CH4:10]", "[CH4:20]"}


def test_compile_route_graph_preserves_sibling_frontiers_and_map_namespaces() -> None:
    compiled = RouteJSONCompiler().compile_route_graph(
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        steps=[
            {
                "product_smiles": "CCOC",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 2, "map_b": 3}
                ],
            },
            {
                "product_smiles": "CC",
                "mapped_product_smiles": "[CH3:1][CH3:2]",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 2}
                ],
            },
            {
                # This is the root step's sibling precursor, not a precursor
                # of the immediately previous CC expansion.
                "product_smiles": "CO",
                "mapped_product_smiles": "[OH:3][CH3:4]",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 3, "map_b": 4}
                ],
            },
        ],
    )

    assert len(compiled) == 3
    assert compiled[1].mapped_product_smiles == "[CH3:1][CH3:2]"
    assert compiled[2].mapped_product_smiles == "[OH:3][CH3:4]"
    assert compiled[2].precursor_smiles == ("C", "O")


def test_compile_route_graph_state_returns_only_real_mapped_open_precursors() -> None:
    state = RouteJSONCompiler().compile_route_graph_state(
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        steps=[
            {
                "product_smiles": "CCOC",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 2, "map_b": 3}
                ],
            },
            {
                "product_smiles": "CC",
                # Advisory maps are deliberately wrong. The compiler must
                # consume the host-emitted [CH3:1][CH3:2] boundary instead.
                "mapped_product_smiles": "[CH3:18][CH3:19]",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 2}
                ],
            },
        ],
    )

    assert state.reactions[1].mapped_product_smiles == "[CH3:1][CH3:2]"
    assert {
        (row.product_smiles, row.mapped_product_smiles)
        for row in state.open_precursors
    } == {
        ("CO", "[OH:3][CH3:4]"),
        ("C", "[CH4:1]"),
        ("C", "[CH4:2]"),
    }


def test_compile_route_graph_rejects_a_disconnected_step() -> None:
    with pytest.raises(
        ReactionJsonReplayError,
        match="routejson_compiler_product_not_open_precursor",
    ):
        RouteJSONCompiler().compile_route_graph(
            mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
            steps=[
                {
                    "product_smiles": "CCO",
                    "reaction_operations": [
                        {"op": "break_bond", "map_a": 2, "map_b": 3}
                    ],
                },
                {
                    "product_smiles": "CO",
                    "reaction_operations": [
                        {"op": "break_bond", "map_a": 1, "map_b": 2}
                    ],
                },
            ],
        )


def test_compile_linear_route_rejects_non_replayed_next_product() -> None:
    with pytest.raises(
        ReactionJsonReplayError,
        match="routejson_compiler_next_product_not_previous_precursor",
    ):
        RouteJSONCompiler().compile_linear_route(
            mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
            steps=[
                {
                    "product_smiles": "CCO",
                    "reaction_operations": [
                        {"op": "break_bond", "map_a": 2, "map_b": 3}
                    ],
                },
                {
                    "product_smiles": "CO",
                    "reaction_operations": [
                        {"op": "break_bond", "map_a": 1, "map_b": 2}
                    ],
                },
            ],
        )


def test_compile_linear_route_uses_host_precursor_for_stereo_only_declaration() -> None:
    compiled = RouteJSONCompiler().compile_linear_route(
        mapped_target_smiles="[CH3:1][C@:2]([F:3])([Cl:4])[CH2:5][OH:6]",
        steps=[
            {
                "product_smiles": "C[C@](F)(Cl)CO",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 2, "map_b": 5}
                ],
            },
            {
                # Same constitution as the host precursor, but a model-
                # invented opposite stereo declaration.
                "product_smiles": "C[C@@H](F)Cl",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 2}
                ],
            },
        ],
    )

    assert compiled[1].product_smiles == "C[C@H](F)Cl"
    assert compiled[1].audit["declared_product_matches_host"] is False
    assert compiled[1].audit["declared_product_mismatch_type"] == (
        "stereochemistry_only"
    )


def test_compile_linear_route_rejects_constitutionally_different_next_product() -> None:
    with pytest.raises(
        ReactionJsonReplayError,
        match="routejson_compiler_next_product_not_previous_precursor",
    ):
        RouteJSONCompiler().compile_linear_route(
            mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
            steps=[
                {
                    "product_smiles": "CCO",
                    "reaction_operations": [
                        {"op": "break_bond", "map_a": 2, "map_b": 3}
                    ],
                },
                {
                    "product_smiles": "CBr",
                    "reaction_operations": [
                        {"op": "break_bond", "map_a": 1, "map_b": 2}
                    ],
                },
            ],
        )


def test_assemble_route_overrides_model_declared_structures() -> None:
    compiler = RouteJSONCompiler()
    compiled = compiler.compile_linear_route(
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        steps=[
            {
                "product_smiles": "CCO",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 2, "map_b": 3}
                ],
            }
        ],
    )
    route = compiler.assemble_route(
        compiled,
        metadata=[
            {
                "step_id": "root",
                "product_smiles": "wrong",
                "precursor_smiles": ["wrong"],
            }
        ],
    )

    assert route[0]["step_id"] == "root"
    assert route[0]["product_smiles"] == "CCO"
    assert route[0]["precursor_smiles"] == ["CC", "O"]
    assert route[0]["mapped_product_smiles"] == "[CH3:1][CH2:2][OH:3]"

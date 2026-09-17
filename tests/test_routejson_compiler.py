from __future__ import annotations

import pytest

from cascade_planner.application.reactionjson_replay import ReactionJsonReplayError
from cascade_planner.application.routejson_compiler import (
    RouteJSONCompiler,
    _mapped_atom_maps,
)
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


def test_assembled_route_persists_host_assigned_add_group_maps_across_steps() -> None:
    compiler = RouteJSONCompiler()
    first = compiler.compile_step(
        mapped_product_smiles="[CH3:1][Br:2]",
        operations=[
            {"op": "remove_group", "map_indices": [2]},
            {"op": "add_group", "map_idx": 1, "fragment_smiles": "[*]O"},
        ],
        expected_product_smiles="CBr",
        reserved_atom_maps=(1, 2, 31),
    )

    assert first.reaction_operations == (
        {"op": "remove_group", "map_indices": [2]},
        {
            "op": "add_group",
            "map_idx": 1,
            "fragment_smiles": "*[OH:32]",
        },
    )
    assembled = compiler.assemble_route(
        (first,),
        metadata=[{"step_id": "route:1"}],
    )
    assembled.append(
        {
            "step_id": "route:2",
            "product_smiles": "CO",
            "reaction_operations": [
                {"op": "break_bond", "map_a": 1, "map_b": 32}
            ],
        }
    )

    replayed = compiler.compile_route_graph(
        mapped_target_smiles="[CH3:1][Br:2]",
        steps=assembled,
    )

    assert replayed[0].reaction_operations == first.reaction_operations
    assert replayed[1].mapped_product_smiles == "[CH3:1][OH:32]"
    assert replayed[1].precursor_smiles == ("C",)
    assert replayed[1].reaction_input_smiles == ("C", "O")
    assert replayed[1].auxiliary_reagent_smiles == ("O",)


def test_collision_feedback_survives_builder_projection_and_recovers_without_relabeling():
    from cascade_planner.application.reactionjson_replay import reactionjson_failure_focus
    from cascade_planner.orchestration.sequential_strategy_director import _compact_builder_rejection

    compiler = RouteJSONCompiler()
    operations = [{"op": "remove_group", "map_indices": [2]},
                  {"op": "add_group", "map_idx": 1, "fragment_smiles": "[*][OH:31]"}]
    with pytest.raises(ReactionJsonReplayError, match="reactionjson_fragment_map_collision") as caught:
        compiler.compile_step(mapped_product_smiles="[CH3:1][Br:2]", operations=operations,
                              reserved_atom_maps=(31,))
    failure = reactionjson_failure_focus(caught.value)
    projected = _compact_builder_rejection({"reason": "candidate_does_not_extend_target_rooted_route",
                                            "routejson_replay_validation": {"compiler_error": str(caught.value), **failure}})
    diagnostic = projected["replay_diagnostic"]
    assert diagnostic["colliding_atom_map"] == 31
    assert diagnostic["fresh_atom_map_start"] == 32
    assert "Leave new atoms unmapped" in diagnostic["required_repair"]
    operations[1] = {**operations[1], "fragment_smiles": "[*]O"}
    resolved = compiler.compile_step(mapped_product_smiles="[CH3:1][Br:2]", operations=operations,
                                     reserved_atom_maps=(31,))
    assert resolved.mapped_product_smiles == "[CH3:1][Br:2]"
    assert resolved.mapped_precursor_smiles == ("[CH3:1][OH:32]",)
    assert resolved.reaction_operations[1]["fragment_smiles"] == "*[OH:32]"


def test_route_graph_replay_inherits_repair_reserved_atom_map_namespace() -> None:
    compiler = RouteJSONCompiler()
    rows = [
        {
            "step_id": "repair:1",
            "product_smiles": "CBr",
            "reaction_operations": [
                {"op": "remove_group", "map_indices": [2]},
                {"op": "add_group", "map_idx": 1, "fragment_smiles": "[*]O"},
            ],
        },
        {
            "step_id": "repair:2",
            "product_smiles": "CO",
            "reaction_operations": [
                {"op": "break_bond", "map_a": 1, "map_b": 32}
            ],
        },
    ]

    replayed = compiler.compile_route_graph(
        mapped_target_smiles="[CH3:1][Br:2]",
        steps=rows,
        reserved_atom_maps=(31,),
    )

    assert replayed[0].reaction_operations[-1]["fragment_smiles"] == "*[OH:32]"
    assert replayed[1].mapped_product_smiles == "[CH3:1][OH:32]"
    assert replayed[1].precursor_smiles == ("C",)
    assert replayed[1].reaction_input_smiles == ("C", "O")
    assert replayed[1].auxiliary_reagent_smiles == ("O",)


def test_target_atom_lineage_keeps_reagent_out_of_route_frontier() -> None:
    materialized = RouteJSONCompiler().compile_step(
        mapped_product_smiles="[CH3:1][Cu:36]",
        operations=[
            {"op": "break_bond", "map_a": 1, "map_b": 36},
            {
                "op": "add_group",
                "map_idx": 1,
                "fragment_smiles": "*[Li:40]",
            },
            {
                "op": "add_group",
                "map_idx": 36,
                "fragment_smiles": "*[I:41]",
            },
        ],
        expected_product_smiles="C[Cu]",
        target_atom_maps={1},
    )

    assert materialized.precursor_smiles == ("[Li]C",)
    assert materialized.reaction_input_smiles == ("[Cu]I", "[Li]C")
    assert materialized.auxiliary_reagent_smiles == ("[Cu]I",)
    assert materialized.mapped_precursor_smiles == ("[CH3:1][Li:40]",)
    assert materialized.mapped_auxiliary_reagent_smiles == ("[Cu:36][I:41]",)
    assert materialized.audit["reaction_component_ledger"] == {
        "schema_version": "reaction_component_ledger.v1",
        "authority": "host_target_atom_map_lineage",
        "target_atom_maps": [1],
        "route_precursor_indices": [1],
        "auxiliary_reagent_indices": [0],
        "route_precursor_count": 1,
        "auxiliary_reagent_count": 1,
        "zero_target_atom_components_never_enter_route_frontier": True,
        "component_role_grants_no_reaction_or_stock_proof": True,
    }


def test_target_atom_lineage_retains_every_target_contributing_partner() -> None:
    materialized = RouteJSONCompiler().compile_step(
        mapped_product_smiles="[CH3:1][CH2:2][CH2:3][OH:4]",
        operations=[{"op": "break_bond", "map_a": 2, "map_b": 3}],
        expected_product_smiles="CCCO",
        target_atom_maps={1, 2, 3, 4},
    )

    assert materialized.precursor_smiles == ("CC", "CO")
    assert materialized.reaction_input_smiles == materialized.precursor_smiles
    assert materialized.auxiliary_reagent_smiles == ()


def test_target_contributing_acetaldehyde_is_never_reclassified_as_reagent() -> None:
    materialized = RouteJSONCompiler().compile_step(
        mapped_product_smiles="[CH3:1][C@@H:2]([OH:3])[CH2:4][CH3:5]",
        operations=[
            {"op": "break_bond", "map_a": 2, "map_b": 4},
            {
                "op": "change_bond_order",
                "map_a": 2,
                "map_b": 3,
                "delta": 1,
            },
        ],
        expected_product_smiles="CCC(C)O",
        target_atom_maps={1, 2, 3, 4, 5},
    )

    assert set(materialized.precursor_smiles) == {"CC", "CC=O"}
    assert materialized.auxiliary_reagent_smiles == ()


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


def test_replay_materialized_route_rebases_sibling_local_fragment_maps() -> None:
    rows = [
        {
            "product_smiles": "CCOC",
            "mapped_product_smiles": "[CH3:1][CH2:2][O:3][CH3:4]",
            "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH:3][CH3:4]"],
            "reaction_operations": [
                {"op": "break_bond", "map_a": 2, "map_b": 3}
            ],
            "reactionjson_audit": {"accepted": True},
        },
        {
            "product_smiles": "CC",
            "mapped_product_smiles": "[CH3:1][CH3:2]",
            "mapped_precursor_smiles": ["[CH3:1][CH2:2][Cl:5]"],
            "reaction_operations": [
                {
                    "op": "add_group",
                    "map_idx": 2,
                    "fragment_smiles": "*[Cl:5]",
                }
            ],
            "reactionjson_audit": {"accepted": True},
        },
        {
            "product_smiles": "CO",
            "mapped_product_smiles": "[OH:3][CH3:4]",
            "mapped_precursor_smiles": ["[OH:3][CH2:4][Br:5]"],
            "reaction_operations": [
                {
                    "op": "add_group",
                    "map_idx": 4,
                    "fragment_smiles": "*[Br:5]",
                }
            ],
            "reactionjson_audit": {"accepted": True},
        },
        {
            "product_smiles": "OCBr",
            "mapped_product_smiles": "[OH:3][CH2:4][Br:5]",
            "mapped_precursor_smiles": ["[OH:3][CH2:4][I:6]"],
            "reaction_operations": [
                {"op": "remove_group", "map_indices": [5]},
                {
                    "op": "add_group",
                    "map_idx": 4,
                    "fragment_smiles": "*[I:6]",
                },
            ],
            "reactionjson_audit": {"accepted": True},
        },
    ]

    with pytest.raises(ReactionJsonReplayError, match="fragment_map_collision"):
        RouteJSONCompiler().compile_route_graph_state(
            mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
            steps=rows,
        )

    state = RouteJSONCompiler().compile_route_graph_state(
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        steps=rows,
        rebase_materialized_local_maps=True,
    )

    assert len(state.reactions) == 4
    sibling_addition = state.reactions[2]
    rebound_bromine_map = sibling_addition.audit["fragment_map_translation"][0][1]
    assert rebound_bromine_map > 6
    assert state.reactions[3].reaction_operations[0] == {
        "op": "remove_group",
        "map_indices": [rebound_bromine_map],
    }
    all_frontier_maps = [
        atom_map
        for precursor in state.open_precursors
        for atom_map in _mapped_atom_maps(precursor.mapped_product_smiles)
    ]
    assert len(all_frontier_maps) == len(set(all_frontier_maps))


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
    assert state.parent_step_indices == (None, 0)
    assert state.open_precursor_producer_step_indices == (0, 1, 1)
    assert {
        (row.product_smiles, row.mapped_product_smiles)
        for row in state.open_precursors
    } == {
        ("CO", "[OH:3][CH3:4]"),
        ("C", "[CH4:1]"),
        ("C", "[CH4:2]"),
    }


def test_compile_route_graph_state_reports_sibling_dependency_parents() -> None:
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
                "mapped_product_smiles": "[CH3:1][CH3:2]",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 2}
                ],
            },
            {
                "product_smiles": "CO",
                "mapped_product_smiles": "[OH:3][CH3:4]",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 3, "map_b": 4}
                ],
            },
        ],
    )

    assert state.parent_step_indices == (None, 0, 0)


def test_compile_route_graph_allows_duplicate_products_on_sibling_branches() -> None:
    compiled = RouteJSONCompiler().compile_route_graph(
        mapped_target_smiles="[CH3:1][CH2:2][CH2:3][CH3:4]",
        steps=[
            {
                "product_smiles": "CCCC",
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
                "product_smiles": "CC",
                "mapped_product_smiles": "[CH3:3][CH3:4]",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 3, "map_b": 4}
                ],
            },
        ],
    )

    assert len(compiled) == 3
    assert compiled[1].mapped_product_smiles == "[CH3:1][CH3:2]"
    assert compiled[2].mapped_product_smiles == "[CH3:3][CH3:4]"


def test_compile_route_graph_rejects_cycle_within_one_sibling_branch() -> None:
    with pytest.raises(
        ReactionJsonReplayError,
        match="routejson_compiler_product_cycle",
    ):
        RouteJSONCompiler().compile_route_graph(
            mapped_target_smiles="[CH3:1][CH2:2][CH2:3][CH3:4]",
            steps=[
                {
                    "product_smiles": "CCCC",
                    "reaction_operations": [
                        {"op": "break_bond", "map_a": 2, "map_b": 3}
                    ],
                },
                {
                    "product_smiles": "CC",
                    "mapped_product_smiles": "[CH3:1][CH3:2]",
                    "reaction_operations": [
                        {
                            "op": "add_group",
                            "map_idx": 2,
                            "fragment_smiles": "[*][CH2:5][CH3:6]",
                        }
                    ],
                },
            ],
        )


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

from __future__ import annotations

from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.interfaces.chemenzy_reactionjson_expansion import (
    ChemEnzyReactionJsonOrSearch,
    ReactionJsonOrCandidate,
)
from cascade_planner.orchestration.sequential_strategy_director import (
    _reactionjson_candidates_from_record,
)


def _candidate(candidate_id: str, operations: list[dict]) -> dict:
    return {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": candidate_id,
        "product_smiles": "CCO",
        "precursor_smiles": [],
        "reaction_family": "fragmentation",
        "transformation_rationale": "host replay probe",
        "conditions": [],
        "catalyst": "",
        "enzyme": "",
        "limitations": [],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "reaction_operations": operations,
        "route_json": None,
    }


def _record(candidates: list[dict]) -> WorkerRunRecord:
    return WorkerRunRecord(
        run_id="or-candidates:run",
        task_id="or-candidates",
        case_id="or-candidates:case",
        status="accepted_draft",
        output_artifact={
            "artifact_type": "RetrosynthesisProposalReport",
            "payload": {
                "schema_version": "retrosynthesis_proposal_report.v1",
                "no_solved_claim": True,
                "candidates": candidates,
            },
        },
        output_validation={"accepted": True, "reasons": []},
        usage={"input_tokens": 1, "output_tokens": 1},
    )


def test_invalid_candidate_does_not_discard_valid_reactionjson_sibling() -> None:
    accepted, rejected = _reactionjson_candidates_from_record(
        _record(
            [
                _candidate(
                    "invalid",
                    [{"op": "break_bond", "map_a": 2, "map_b": 99}],
                ),
                _candidate(
                    "valid",
                    [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                ),
            ]
        ),
        expected_product="CCO",
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        require_reaction_operations=True,
        max_candidates=3,
    )

    assert [row.candidate_id for row in accepted] == ["valid"]
    assert accepted[0].expansion.precursor_smiles == ("CC", "O")
    assert rejected[0]["candidate_id"] == "invalid"
    assert rejected[0]["reason"] == "strategy_graph_edit_replay_failed"


def test_chemenzy_or_tree_backtracks_to_preserved_reaction_candidate() -> None:
    search = ChemEnzyReactionJsonOrSearch(
        target_smiles="CCO",
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        max_depth=6,
    )
    root = search.select_open_node()
    assert root is not None
    inserted = search.expand(
        root,
        [
            ReactionJsonOrCandidate(
                candidate_id="route-a",
                precursor_smiles=("CC", "O"),
                mapped_precursor_smiles=("[CH3:1][CH3:2]", "[OH2:3]"),
                route_step={"step_id": "route-a", "product_smiles": "CCO"},
                score=1.0,
                cost=0.0,
                candidate_key="route-a-key",
            ),
            ReactionJsonOrCandidate(
                candidate_id="route-b",
                precursor_smiles=("C", "CO"),
                mapped_precursor_smiles=("[CH4:1]", "[CH3:2][OH:3]"),
                route_step={"step_id": "route-b", "product_smiles": "CCO"},
                score=0.5,
                cost=0.7,
                candidate_key="route-b-key",
            ),
        ],
        stock_smiles={"C", "O"},
    )

    assert inserted == 2
    assert len(search.tree.root.children) == 2
    assert search.project().steps[0]["step_id"] == "route-a"

    route_a_leaf = search.select_open_node()
    assert route_a_leaf.mol == "CC"
    search.defer_failed_node(route_a_leaf)

    after_backtrack = search.project()
    assert after_backtrack.steps[0]["step_id"] == "route-b"
    route_b_leaf = search.select_open_node()
    assert route_b_leaf.mol == "CO"

    search.expand(
        route_b_leaf,
        [
            ReactionJsonOrCandidate(
                candidate_id="route-b-tail",
                precursor_smiles=("C", "O"),
                mapped_precursor_smiles=("[CH4:2]", "[OH2:3]"),
                route_step={"step_id": "route-b-tail", "product_smiles": "CO"},
                score=1.0,
                cost=0.0,
                candidate_key="route-b-tail-key",
            )
        ],
        stock_smiles={"C", "O"},
    )

    solved = search.project()
    assert solved.complete is True
    assert [step["step_id"] for step in solved.steps] == [
        "route-b",
        "route-b-tail",
    ]
    assert solved.open_leaf_states == ()
    assert solved.summary["backtracks"] == 1


def test_edited_route_replay_recomputes_root_solution_from_fresh_tree() -> None:
    search = ChemEnzyReactionJsonOrSearch(
        target_smiles="CCO",
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        max_depth=6,
    )
    inserted = search.replay_route(
        (
            {
                "step_id": "editor:1",
                "product_smiles": "CCO",
                "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                "precursor_smiles": ["C", "CO"],
                "mapped_precursor_smiles": ["[CH4:1]", "[CH3:2][OH:3]"],
            },
        ),
        stock_smiles={"C"},
    )

    projection = search.project()
    assert inserted == 1
    assert projection.complete is False
    assert projection.summary["root_solved"] is False
    assert projection.open_leaf_states == (
        {"smiles": "CO", "mapped_smiles": "[CH3:2][OH:3]"},
    )

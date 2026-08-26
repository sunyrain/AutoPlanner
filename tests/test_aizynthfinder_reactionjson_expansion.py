from __future__ import annotations

import importlib.util
import json

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("aizynthfinder") is None,
    reason="runs in the isolated requirements_aizynth.txt environment",
)


def test_paper_strategy_query_does_not_turn_builder_role_into_search_control() -> None:
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        _node_satisfies_strategy_execution_contract,
    )

    class Action:
        def __init__(self, step):
            self.metadata = {"autoplanner_route_step": step}

    class Node:
        def __init__(self, steps):
            self._actions = [Action(step) for step in steps]

        def path_to(self):
            return self._actions, []

    strategy_text = json.dumps(
        {
            "strategy_query": "Construct the decalin through an IMDA.",
            "strategy_signature": "IMDA-decalin",
        }
    )
    enabling = Node(
        [{"step_role": "enabling", "strategy_anchor": False}]
    )
    key = Node(
        [
            {"step_role": "enabling", "strategy_anchor": False},
            {"step_role": "key", "strategy_anchor": True},
        ]
    )

    assert (
        _node_satisfies_strategy_execution_contract(
            enabling, strategy_text=strategy_text
        )
        is True
    )
    assert (
        _node_satisfies_strategy_execution_contract(key, strategy_text=strategy_text)
        is True
    )


def _imports():
    from aizynthfinder.chem import Molecule
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.context.stock.queries import StockQueryMixin
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        AiZynthFinderReactionJsonExpansionStrategy,
        ReactionJsonExpansionCandidate,
        ReactionJsonMctsSearchTree,
    )

    return (
        Molecule,
        Configuration,
        StockQueryMixin,
        AiZynthFinderReactionJsonExpansionStrategy,
        ReactionJsonExpansionCandidate,
        ReactionJsonMctsSearchTree,
    )


def test_aizynthfinder_reactionjson_policy_preserves_or_candidates_and_backtracks() -> None:
    (
        Molecule,
        Configuration,
        StockQueryMixin,
        Policy,
        Candidate,
        SearchTree,
    ) = _imports()

    class SetStock(StockQueryMixin):
        def __init__(self, smiles: set[str]) -> None:
            self.keys = {Molecule(smiles=value).inchi_key for value in smiles}

        def __contains__(self, mol) -> bool:
            return mol.inchi_key in self.keys

        def __len__(self) -> int:
            return len(self.keys)

    requests = []

    def candidate_provider(request):
        requests.append(request)
        selected = request.expandable_smiles[0]
        mapped = request.expandable_mapped_smiles[0]
        if selected == "CCO":
            return [
                Candidate(
                    candidate_id="dead-high-prior",
                    product_smiles="CCO",
                    mapped_product_smiles=mapped,
                    precursor_smiles=("CC", "O"),
                    mapped_precursor_smiles=("[CH3:1][CH3:2]", "[OH2:3]"),
                    route_step={"step_id": "dead-high-prior", "product_smiles": "CCO"},
                    prior=0.95,
                    candidate_key="dead-high-prior",
                ),
                Candidate(
                    candidate_id="solvable-lower-prior",
                    product_smiles="CCO",
                    mapped_product_smiles=mapped,
                    precursor_smiles=("C", "CO"),
                    mapped_precursor_smiles=("[CH4:1]", "[CH3:2][OH:3]"),
                    route_step={"step_id": "solvable-lower-prior", "product_smiles": "CCO"},
                    prior=0.2,
                    candidate_key="solvable-lower-prior",
                ),
            ]
        if selected == "CO":
            return [
                Candidate(
                    candidate_id="solvable-tail",
                    product_smiles="CO",
                    mapped_product_smiles=mapped,
                    precursor_smiles=("C", "O"),
                    mapped_precursor_smiles=("[CH4:2]", "[OH2:3]"),
                    route_step={"step_id": "solvable-tail", "product_smiles": "CO"},
                    prior=1.0,
                    candidate_key="solvable-tail",
                )
            ]
        return []

    config = Configuration()
    config.search.algorithm_config["C"] = 1.4
    config.search.algorithm_config["use_prior"] = True
    config.search.algorithm_config["prune_cycles_in_search"] = True
    config.search.max_transforms = 6
    config.scorers.create_default_scorers()
    config.stock.load(SetStock({"C", "O"}), "paper")
    config.stock.select("paper")
    policy = Policy(
        "codex_reactionjson",
        config,
        candidate_provider=candidate_provider,
        strategy_id="strategy-1",
        strategy_text="Prefer the convergent disconnection.",
        max_policy_calls=25,
        max_candidates_per_call=3,
    )
    config.expansion_policy.load(policy)
    config.expansion_policy.select("codex_reactionjson")

    tree = SearchTree(config, root_smiles="CCO")
    solved = False
    for _ in range(20):
        solved = tree.one_iteration()
        if solved:
            break

    assert solved is True
    assert tree.root is not None
    root_view = tree.root.children_view()
    assert [action.metadata["candidate_id"] for action in root_view["actions"]] == [
        "dead-high-prior",
        "solvable-lower-prior",
    ]
    assert root_view["visitations"][0] > 1
    assert root_view["visitations"][1] > 1
    assert policy.policy_calls <= 25
    assert policy.accepted_actions == 3
    assert any(request.route_steps for request in requests if request.depth > 0)


def test_aizynthfinder_requests_preserve_host_maps_for_new_atoms() -> None:
    from aizynthfinder.chem import Molecule
    from aizynthfinder.context.stock.queries import StockQueryMixin
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        ReactionJsonExpansionCandidate,
        run_reactionjson_branch,
    )

    class SetStock(StockQueryMixin):
        def __init__(self, smiles: set[str]) -> None:
            self.keys = {Molecule(smiles=value).inchi_key for value in smiles}

        def __contains__(self, mol) -> bool:
            return mol.inchi_key in self.keys

        def __len__(self) -> int:
            return len(self.keys)

    upstream_requests = []

    def candidate_provider(request):
        selected = request.expandable_smiles[0]
        mapped = request.expandable_mapped_smiles[0]
        if request.depth == 0:
            return [
                ReactionJsonExpansionCandidate(
                    candidate_id="make-grignard-leaf",
                    product_smiles="CC",
                    mapped_product_smiles=mapped,
                    precursor_smiles=("C[Mg]Br", "C"),
                    mapped_precursor_smiles=(
                        "[CH3:1][Mg:22][Br:23]",
                        "[CH4:2]",
                    ),
                    route_step={"step_id": "make-grignard-leaf", "product_smiles": "CC"},
                )
            ]
        if selected == "C[Mg]Br":
            upstream_requests.append(request)
        return []

    run_reactionjson_branch(
        target_smiles="CC",
        strategy_id="host-map-namespace",
        strategy_text="preserve maps for introduced reagent atoms",
        candidate_provider=candidate_provider,
        stock_query=SetStock({"C"}),
        max_policy_calls=2,
        max_candidates_per_call=1,
        max_transforms=3,
        max_mcts_iterations=6,
    )

    assert upstream_requests
    assert upstream_requests[0].expandable_mapped_smiles == (
        "[CH3:1][Mg:22][Br:23]",
    )


def test_flattened_aiz_route_checks_only_terminal_stock_leaves() -> None:
    from scripts.run_aizynthfinder_paper_search import _flatten_route

    route = {
        "type": "mol",
        "smiles": "CCO",
        "in_stock": False,
        "children": [
            {
                "type": "reaction",
                "metadata": {"policy_name": "uspto"},
                "children": [
                    {
                        "type": "mol",
                        "smiles": "CC",
                        "in_stock": False,
                        "children": [
                            {
                                "type": "reaction",
                                "metadata": {"policy_name": "uspto"},
                                "children": [
                                    {"type": "mol", "smiles": "C", "in_stock": True},
                                    {"type": "mol", "smiles": "C", "in_stock": True},
                                ],
                            }
                        ],
                    },
                    {"type": "mol", "smiles": "O", "in_stock": True},
                ],
            }
        ],
    }

    flattened = _flatten_route(route, route_index=1)

    assert flattened["step_count"] == 2
    assert flattened["steps"][0]["reactant_stock_status"] == [False, True]
    assert flattened["terminal_leaf_count"] == 3
    assert flattened["terminal_leaf_stock_status"] == [True, True, True]
    assert flattened["all_leaves_in_provider_stock"] is True


def test_paper_stock_query_reads_exact_full_inchikey_index() -> None:
    from aizynthfinder.chem import Molecule
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        FullInchiKeySqliteStockQuery,
    )

    query = FullInchiKeySqliteStockQuery(
        "data_external/synthatlas/zinc_synthelite_20260223_full_inchikey.sqlite3"
    )
    try:
        assert len(query) == 39_478_827
        # Methane is deliberately only an API probe here; the assertion binds
        # query semantics to an independently computed full InChIKey lookup.
        methane = Molecule(smiles="C")
        expected = query._connection.execute(
            "SELECT 1 FROM stock WHERE full_inchikey = ? LIMIT 1",
            (methane.inchi_key,),
        ).fetchone()
        assert (methane in query) is (expected is not None)
    finally:
        query.close()


def test_reactionjson_branch_excludes_target_from_stock_zero_step_closure() -> None:
    from aizynthfinder.chem import Molecule
    from aizynthfinder.context.stock.queries import StockQueryMixin
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        ReactionJsonExpansionCandidate,
        run_reactionjson_branch,
    )

    class SetStock(StockQueryMixin):
        def __init__(self, smiles: set[str]) -> None:
            self.keys = {Molecule(smiles=value).inchi_key for value in smiles}

        def __contains__(self, mol) -> bool:
            return mol.inchi_key in self.keys

        def __len__(self) -> int:
            return len(self.keys)

    def candidate_provider(request):
        assert request.expandable_smiles == ("CCO",)
        return [
            ReactionJsonExpansionCandidate(
                candidate_id="leave-target-out",
                product_smiles="CCO",
                mapped_product_smiles=request.expandable_mapped_smiles[0],
                precursor_smiles=("C", "O"),
                mapped_precursor_smiles=("[CH4:1]", "[OH2:2]"),
                route_step={"step_id": "leave-target-out", "product_smiles": "CCO"},
            )
        ]

    result = run_reactionjson_branch(
        target_smiles="CCO",
        strategy_id="target-in-stock",
        strategy_text="force a real retrosynthetic expansion",
        candidate_provider=candidate_provider,
        stock_query=SetStock({"CCO", "C", "O"}),
        max_policy_calls=3,
        max_candidates_per_call=1,
        max_transforms=3,
    )

    assert result.solved is True
    assert result.policy_calls == 1
    assert result.route_steps[0]["step_id"] == "leave-target-out"
    assert result.diagnostics["root_solved"] is False


def test_unsolved_branch_projects_best_replayed_descendant_not_empty_root() -> None:
    from aizynthfinder.context.stock.queries import StockQueryMixin
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        ReactionJsonExpansionCandidate,
        run_reactionjson_branch,
    )

    class EmptyStock(StockQueryMixin):
        def __contains__(self, mol) -> bool:
            del mol
            return False

        def __len__(self) -> int:
            return 0

    def candidate_provider(request):
        if request.depth:
            return []
        return [
            ReactionJsonExpansionCandidate(
                candidate_id="retain-partial-route",
                product_smiles="CCO",
                mapped_product_smiles=request.expandable_mapped_smiles[0],
                precursor_smiles=("CC", "O"),
                mapped_precursor_smiles=("[CH3:1][CH3:2]", "[OH2:3]"),
                route_step={
                    "step_id": "retain-partial-route",
                    "product_smiles": "CCO",
                },
            )
        ]

    result = run_reactionjson_branch(
        target_smiles="CCO",
        strategy_id="unsolved-partial",
        strategy_text="retain host-replayed partial routes",
        candidate_provider=candidate_provider,
        stock_query=EmptyStock(),
        max_policy_calls=2,
        max_candidates_per_call=1,
        max_transforms=3,
    )

    assert result.solved is False
    assert result.route_steps[0]["step_id"] == "retain-partial-route"
    assert result.diagnostics["selected_depth"] == 1


def test_empty_builder_responses_leave_termination_to_mcts_budget() -> None:
    from aizynthfinder.context.stock.queries import StockQueryMixin
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        ReactionJsonExpansionCandidate,
        ReactionJsonPolicyResponse,
        run_reactionjson_branch,
    )

    class EmptyStock(StockQueryMixin):
        def __contains__(self, mol) -> bool:
            del mol
            return False

        def __len__(self) -> int:
            return 0

    requests: list[tuple[str, ...]] = []

    def candidate_provider(request):
        requests.append(request.expandable_smiles)
        if request.depth == 0:
            return ReactionJsonPolicyResponse(
                candidates=(
                    ReactionJsonExpansionCandidate(
                        candidate_id="root-split",
                        product_smiles=request.expandable_smiles[0],
                        mapped_product_smiles=request.expandable_mapped_smiles[0],
                        precursor_smiles=("CC", "O"),
                        mapped_precursor_smiles=("[CH3:1][CH3:2]", "[OH2:3]"),
                        route_step={
                            "step_id": "root-split",
                            "product_smiles": request.expandable_smiles[0],
                        },
                    ),
                )
            )
        return ReactionJsonPolicyResponse()

    result = run_reactionjson_branch(
        target_smiles="CCO",
        strategy_id="host-termination",
        strategy_text="retain unresolved leaves until the Host budget ends",
        candidate_provider=candidate_provider,
        stock_query=EmptyStock(),
        max_policy_calls=4,
        max_candidates_per_call=1,
        max_transforms=3,
    )

    assert requests[0] == ("CCO",)
    assert len(requests) == 4
    assert all(len(request) == 2 for request in requests[1:])
    assert result.policy_calls == 4
    assert result.diagnostics["calls_exhausted"] is True
    assert not any("handoff" in key for key in result.diagnostics)


def test_unsolved_projection_prefers_deepest_connected_route_over_shallow_reward() -> None:
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        ReactionJsonExpansionCandidate,
        run_reactionjson_branch,
    )

    class EmptyStock:
        def __contains__(self, mol) -> bool:
            del mol
            return False

        def __len__(self) -> int:
            return 0

    def candidate_provider(request):
        selected = request.expandable_smiles[0]
        mapped = request.expandable_mapped_smiles[0]
        if request.depth == 0:
            return [
                ReactionJsonExpansionCandidate(
                    candidate_id="partial-step-1",
                    product_smiles=selected,
                    mapped_product_smiles=mapped,
                    precursor_smiles=("CC",),
                    mapped_precursor_smiles=("[CH3:1][CH3:2]",),
                    route_step={
                        "step_id": "partial-step-1",
                        "product_smiles": selected,
                    },
                )
            ]
        if request.depth == 1:
            return [
                ReactionJsonExpansionCandidate(
                    candidate_id="partial-step-2",
                    product_smiles=selected,
                    mapped_product_smiles=mapped,
                    precursor_smiles=("C",),
                    mapped_precursor_smiles=("[CH4:1]",),
                    route_step={
                        "step_id": "partial-step-2",
                        "product_smiles": selected,
                    },
                )
            ]
        return []

    result = run_reactionjson_branch(
        target_smiles="CCO",
        strategy_id="deepest-partial",
        strategy_text="retain the deepest connected partial route",
        candidate_provider=candidate_provider,
        stock_query=EmptyStock(),
        max_policy_calls=3,
        max_candidates_per_call=1,
        max_transforms=3,
    )

    assert result.solved is False
    assert [step["step_id"] for step in result.route_steps] == [
        "partial-step-1",
        "partial-step-2",
    ]
    assert result.diagnostics["selected_depth"] == 2


def test_hybrid_branch_projection_preserves_materialized_chemical_and_biological_steps() -> None:
    import json

    from aizynthfinder.context.stock.queries import StockQueryMixin
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        ReactionJsonExpansionCandidate,
        run_reactionjson_branch,
    )

    class EmptyStock(StockQueryMixin):
        def __contains__(self, mol) -> bool:
            del mol
            return False

        def __len__(self) -> int:
            return 0

    def candidate_provider(request):
        selected = request.expandable_smiles[0]
        mapped = request.expandable_mapped_smiles[0]
        if request.depth == 0:
            return [
                ReactionJsonExpansionCandidate(
                    candidate_id="chemical-core-closure",
                    product_smiles=selected,
                    mapped_product_smiles=mapped,
                    precursor_smiles=("CC",),
                    mapped_precursor_smiles=("[CH3:1][CH3:2]",),
                    route_step={
                        "step_id": "chemical-core-closure",
                        "product_smiles": selected,
                        "execution_domain": "chemical",
                        "strategy_anchor": True,
                        "reaction_operations": [
                            {"op": "break_bond", "map_a": 1, "map_b": 2}
                        ],
                    },
                )
            ]
        if request.depth == 1:
            return [
                ReactionJsonExpansionCandidate(
                    candidate_id="enzymatic-tailoring",
                    product_smiles=selected,
                    mapped_product_smiles=mapped,
                    precursor_smiles=("C",),
                    mapped_precursor_smiles=("[CH4:1]",),
                    route_step={
                        "step_id": "enzymatic-tailoring",
                        "product_smiles": selected,
                        "execution_domain": "enzymatic",
                        "enzyme": "P450 candidate",
                    },
                )
            ]
        return []

    result = run_reactionjson_branch(
        target_smiles="CCO",
        strategy_id="hybrid-partial",
        strategy_text=json.dumps(
            {
                "execution_domain": "hybrid",
                "key_bond_signature": ["map_pair:1:2"],
            }
        ),
        candidate_provider=candidate_provider,
        stock_query=EmptyStock(),
        max_policy_calls=3,
        max_candidates_per_call=1,
        max_transforms=3,
    )

    assert result.solved is False
    assert [step["step_id"] for step in result.route_steps] == [
        "chemical-core-closure",
        "enzymatic-tailoring",
    ]
    assert result.diagnostics["selected_depth"] == 2
    assert result.diagnostics["selected_strategy_execution_contract_satisfied"] is True


def test_unmapped_role_labels_do_not_count_as_realized_strategy_milestones() -> None:
    from aizynthfinder.context.stock.queries import StockQueryMixin
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        ReactionJsonExpansionCandidate,
        run_reactionjson_branch,
    )

    class EmptyStock(StockQueryMixin):
        def __contains__(self, mol) -> bool:
            del mol
            return False

        def __len__(self) -> int:
            return 0

    def candidate_provider(request):
        selected = request.expandable_smiles[0]
        mapped = request.expandable_mapped_smiles[0]
        if request.depth == 0:
            return [
                ReactionJsonExpansionCandidate(
                    candidate_id="root-milestone",
                    product_smiles=selected,
                    mapped_product_smiles=mapped,
                    precursor_smiles=("CO",),
                    mapped_precursor_smiles=("[CH3:2][OH:3]",),
                    route_step={
                        "step_id": "root-milestone",
                        "product_smiles": selected,
                        "strategy_anchor": True,
                        "strategy_milestone_index": 1,
                    },
                )
            ]
        if request.depth == 1:
            return [
                ReactionJsonExpansionCandidate(
                    candidate_id="upstream-milestone",
                    product_smiles=selected,
                    mapped_product_smiles=mapped,
                    precursor_smiles=("C",),
                    mapped_precursor_smiles=("[CH4:2]",),
                    route_step={
                        "step_id": "upstream-milestone",
                        "product_smiles": selected,
                        "strategy_anchor": True,
                        "strategy_milestone_index": 2,
                    },
                )
            ]
        return []

    result = run_reactionjson_branch(
        target_smiles="CCO",
        strategy_id="multi-milestone-partial",
        strategy_text="retain the connected multi-milestone path",
        candidate_provider=candidate_provider,
        stock_query=EmptyStock(),
        max_policy_calls=3,
        max_candidates_per_call=1,
        max_transforms=3,
    )

    assert result.solved is False
    assert [step["step_id"] for step in result.route_steps] == [
        "root-milestone",
        "upstream-milestone",
    ]
    assert result.diagnostics["selected_depth"] == 2
    assert result.diagnostics["selected_realized_strategic_milestones"] == 0
    assert result.diagnostics["maximum_realized_strategic_milestones_in_tree"] == 0


def test_director_worker_record_is_host_replayed_before_aiz_action() -> None:
    from cascade_planner.agent.codex_worker import WorkerRunRecord
    from cascade_planner.interfaces.aizynthfinder_director_adapter import (
        WorkerRecordReactionJsonCandidateProvider,
    )
    from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
        ReactionJsonExpansionRequest,
    )

    def record_provider(_request):
        return WorkerRunRecord(
            run_id="aiz-worker-record",
            task_id="aiz-worker-record:1",
            case_id="aiz-worker-record:case",
            status="accepted_draft",
            output_artifact={
                "artifact_type": "RetrosynthesisProposalReport",
                "payload": {
                    "schema_version": "retrosynthesis_proposal_report.v1",
                    "no_solved_claim": True,
                    "candidates": [
                        {
                            "schema_version": "retrosynthesis_candidate.v1",
                            "candidate_id": "break-co",
                            "product_smiles": "CCO",
                            # This declaration is intentionally empty.  The
                            # host replay, not model text, must produce CC + O.
                            "precursor_smiles": [],
                            "reaction_family": "C-O disconnection",
                            "transformation_rationale": "compiler canary",
                            "conditions": [],
                            "catalyst": "",
                            "enzyme": "",
                            "limitations": [],
                            "no_solved_claim": True,
                            "not_parent_route_proof": True,
                            "reaction_operations": [
                                {"op": "break_bond", "map_a": 2, "map_b": 3}
                            ],
                            "route_json": None,
                        }
                    ],
                },
            },
            output_validation={"accepted": True, "reasons": []},
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    adapter = WorkerRecordReactionJsonCandidateProvider(
        record_provider,
        max_candidates=1,
    )
    candidates = adapter(
        ReactionJsonExpansionRequest(
            strategy_id="strategy-1",
            strategy_text="disconnect C-O",
            call_index=1,
            max_calls=25,
            depth=0,
            expandable_smiles=("CCO",),
            expandable_mapped_smiles=("[CH3:1][CH2:2][OH:3]",),
            route_steps=(),
        )
    )

    assert len(candidates) == 1
    assert candidates[0].precursor_smiles == ("CC", "O")
    assert candidates[0].mapped_precursor_smiles == (
        "[CH3:1][CH3:2]",
        "[OH2:3]",
    )
    assert candidates[0].route_step["reactionjson_audit"]["accepted"] is True
    assert (
        candidates[0].route_step["reactionjson_audit"]["semantics"]
        ["deterministic_graph_edit_replay"]
        is True
    )
    assert adapter.diagnostics[0].rejected_candidates == ()

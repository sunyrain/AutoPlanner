from __future__ import annotations

import copy
import json
import re
import threading
import time
from dataclasses import replace

import cascade_planner.agent.codex_worker as codex_worker_module
import cascade_planner.orchestration.sequential_strategy_director as sequential_module

from cascade_planner.agent.codex_worker import WorkerBudget, WorkerRunRecord, WorkerTask
from cascade_planner.application.campaign_context import CampaignContext, CampaignContextDelta
from cascade_planner.application.run_kernel import RunRevision
from cascade_planner.application.reaction_proof_versions import (
    CURRENT_REACTION_VALIDATOR_VERSION,
)
from cascade_planner.application.strategy_contract import normalize_strategy_card
from cascade_planner.application.strategy_contract import normalize_reaction_operations
from cascade_planner.orchestration.global_campaign_director import (
    DirectorConfig,
    GlobalCampaignPlan,
    validate_global_campaign_plan,
)
from cascade_planner.orchestration.sequential_strategy_director import (
    SequentialStrategyDirectorRunner,
    compile_frontier_builder_context,
    _expansion_from_record,
    _editor_route_expansions_from_record,
    _expansions_from_record,
    _expansion_rejection_diagnostic,
    _apply_route_patch,
    _editor_route_scaffold,
    _has_atom_provenance_deficit,
    _node_prompt,
    _paper_strategy_portfolio_prompt,
    _strategy_cards_from_portfolio_record,
    _strategy_prompt,
    _strategy_conflicts,
    _branch_mandates_for_profile,
)
from cascade_planner.interfaces.chemenzy_reactionjson_expansion import (
    ChemEnzyReactionJsonOrSearch,
)
from cascade_planner.runtime import AgentSpec, AgentState, Budget


def _context() -> CampaignContext:
    return CampaignContext(
        run_id="sequential-test",
        target={"canonical_smiles": "CCO"},
        revision=RunRevision(
            run_id="sequential-test",
            revision=1,
            state_sha256="a" * 64,
            graph_revision=1,
            evidence_revision=0,
            deficit_sha256="b" * 64,
            acceptance_sha256="c" * 64,
            status="running",
            updated_at="2026-08-15T00:00:00Z",
        ),
        topology={},
        route_portfolio={},
        evidence={},
        stock={},
        deficits=(),
        proposal_history=(),
        failure_history=(),
        budget_state={},
        acceptance_state={},
        delta=CampaignContextDelta(),
    )


def _spec(context: CampaignContext) -> AgentSpec:
    return AgentSpec.from_context(
        run_id=context.run_id,
        agent_id="director:test",
        parent_agent_id="kernel:test",
        role="global_campaign_director",
        objective="compact",
        context=context.to_dict(),
        idempotency_key="sequential:test",
        budget=Budget(max_wall_time_s=60, max_tokens=20_000),
        context_refs=(context.content_sha256,),
        metadata={"model": "gpt-5.6-sol", "reasoning_effort": "medium"},
    )


def test_frontier_builder_context_binds_multi_precursor_identity_not_array_order() -> None:
    graph = {
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:target": {"canonical_smiles": "CCO"},
            "molecule:ethyl": {"canonical_smiles": "CC"},
            "molecule:water": {"canonical_smiles": "O"},
        },
        "route_families": {
            "route:one": {
                "route_family_id": "route:one",
                "edge_ids": ["edge:root"],
                "strategy_card": {},
            }
        },
        "edges": {
            "edge:root": {
                "edge_id": "edge:root",
                "product_molecule_id": "molecule:target",
                "product_smiles": "CCO",
                "precursor_molecule_ids": ["molecule:ethyl", "molecule:water"],
                # Canonical graph projections are allowed to sort these
                # independently. Identity, never array position, binds maps.
                "precursor_smiles": ["O", "CC"],
                "reactionjson_audit": {
                    "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                    "mapped_precursor_smiles": [
                        "[OH2:3]",
                        "[CH3:1][CH3:2]",
                    ],
                },
                "origin_records": [
                    {
                        "proposal_id": "step:root",
                        "origin_kind": "codex",
                        "canonical_route_family_ids": ["route:one"],
                    }
                ],
            }
        },
    }

    context, diagnostic = compile_frontier_builder_context(
        graph,
        frontier_molecule_id="molecule:ethyl",
        route_family_ids=("route:one",),
    )

    assert diagnostic == {}
    assert context is not None
    assert context.selected_product_smiles == "CC"
    assert context.selected_product_mapped == "[CH3:1][CH3:2]"
    assert context.connected_steps[0]["precursor_smiles"] == ["CC", "O"]
    assert context.connected_steps[0]["mapped_precursor_smiles"] == [
        "[CH3:1][CH3:2]",
        "[OH2:3]",
    ]


def test_path_repair_focus_leaf_identity_comes_from_rejected_graph_center() -> None:
    path_repair = {
        "repair_reference_span": [
            {
                "step_id": "old:passed",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 2}
                ],
                "prior_key_critic": {"status": "passed", "verdict": "pass"},
            },
            {
                "step_id": "old:rejected",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 18, "map_b": 19},
                    {"op": "change_bond_order", "map_a": 18, "map_b": 27},
                ],
                "prior_key_critic": {"status": "rejected", "verdict": "reject"},
            },
        ]
    }
    spectator = "[CH3:1][CH2:2][OH:15]"
    repair_core = "[C:18](=[O:27])[CH2:19][CH3:24]"

    assert sequential_module._path_repair_focus_atom_maps(path_repair) == {
        18,
        19,
        27,
    }
    assert sequential_module._path_repair_focus_leaf_indices(
        selectable_indices=(0, 1),
        mapped_product_smiles=(spectator, repair_core),
        path_repair=path_repair,
    ) == (1,)
    assert sequential_module._path_repair_focus_leaf_indices(
        selectable_indices=(0, 1),
        mapped_product_smiles=(repair_core, spectator),
        path_repair=path_repair,
    ) == (0,)


def test_path_repair_builder_follows_focus_component_not_aiz_array_order(
    monkeypatch,
) -> None:
    selected_products: list[str] = []
    builder_strategies: list[dict] = []

    def builder_executor(task: WorkerTask) -> WorkerRunRecord:
        assert task.task_type == "paper_matched_route_step"
        builder_context = json.loads(
            task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1]
        )
        builder_strategies.append(dict(builder_context["strategy"]))
        product = str(task.host_context.get("selected_product") or "")
        selected_products.append(product)
        operations = (
            [{"op": "break_bond", "map_a": 3, "map_b": 4}]
            if len(selected_products) == 1
            else [{"op": "break_bond", "map_a": 4, "map_b": 5}]
        )
        return replace(
            _proposal_record(
                {
                    "candidate_id": task.task_id,
                    "product_smiles": product,
                    "precursor_smiles": [],
                    "checkpoint_relation": (
                        "preparatory"
                        if len(selected_products) == 1
                        else "executes_checkpoint"
                    ),
                    "reaction_family": "mapped repair focus canary",
                    "conditions": ["test conditions"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "reaction_operations": operations,
                },
                target=product,
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        first = request_handler(
            {
                "expandable_smiles": ["CCOCC"],
                "expandable_mapped_smiles": [
                    "[CH3:1][CH2:2][O:3][CH2:4][CH3:5]"
                ],
                "route_steps": [],
            }
        )["candidates"][0]
        precursor_pairs = list(
            zip(first["precursor_smiles"], first["mapped_precursor_smiles"])
        )
        focus_pair = next(
            pair for pair in precursor_pairs if ":4]" in pair[1] and ":5]" in pair[1]
        )
        spectator_pair = next(pair for pair in precursor_pairs if pair != focus_pair)
        second = request_handler(
            {
                # Deliberately put the spectator first. The Host must bind the
                # continuation to the rejected graph center, not this order.
                "expandable_smiles": [spectator_pair[0], focus_pair[0]],
                "expandable_mapped_smiles": [spectator_pair[1], focus_pair[1]],
                "route_steps": [first["route_step"]],
            }
        )["candidates"][0]
        return {
            "route_steps": [first["route_step"], second["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    second["precursor_smiles"], second["mapped_precursor_smiles"]
                )
            ],
            "solved": False,
            "policy_calls": 2,
            "mcts_iterations": 2,
            "diagnostics": {"engine": "AiZynthFinder.MctsSearchTree"},
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    strategy = _strategy_card(1)
    foreign_strategy = _strategy_card(2)
    monkeypatch.setattr(
        sequential_module,
        "_strategy_horizon_for_leaf",
        lambda **_kwargs: (foreign_strategy, True),
    )
    branch = {
        "branch_index": 0,
        "lens": "repair the mapped reaction center",
        "steps": [],
        "target_mapped_smiles": "[CH3:1][CH2:2][O:3][CH2:4][CH3:5]",
        "strategy_card": strategy,
        "root_strategy_card": strategy,
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 0,
        "path_repair_builder_call_count": 0,
        "call_count": 0,
        "key_event_critic_completed": False,
        "open_leaf_states": [
            {
                "smiles": "CCOCC",
                "mapped_smiles": "[CH3:1][CH2:2][O:3][CH2:4][CH3:5]",
            }
        ],
        "open_leaves": ["CCOCC"],
        "expanded_products": set(),
        "_path_repair_resume": {
            "repair_frontier_mapped_product_smiles": (
                "[CH3:1][CH2:2][O:3][CH2:4][CH3:5]"
            ),
            "repair_goal": "continue on the component carrying maps 4 and 5",
            "active_constraints": [],
            "durable_steps": [],
            "reconnect_boundaries": [],
            "repair_reference_span": [
                {
                    "step_id": "old:rejected",
                    "reaction_operations": [
                        {"op": "break_bond", "map_a": 4, "map_b": 5}
                    ],
                    "prior_key_critic": {
                        "status": "rejected",
                        "verdict": "reject",
                    },
                }
            ],
            "reserved_atom_maps": [],
            "completion_mode": "strategy_checkpoint",
            "strategy_card": strategy,
        },
    }
    runner = SequentialStrategyDirectorRunner(
        node_executor=builder_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )

    runner._expand_seeded_branches_aizynthfinder(
        _spec(_context()),
        target="CCOCC",
        seeded=[branch],
        existing_records=[],
        route_quota=sequential_module._NodeCallBudget(
            model_invocations=10,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        critic_editor_call_reserve=0,
        critic_input_reserve=0,
        critic_output_reserve=0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            enable_key_event_critic=False,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            strategy_branch_workers=1,
            max_strategic_milestones_per_branch=4,
            max_node_expansions_per_branch=4,
            max_reactionjson_candidates_per_node=1,
        ),
        started=time.monotonic(),
    )

    assert selected_products == ["CCOCC", "CC"]
    assert all(
        row["strategy_query"] == strategy["strategy_query"]
        for row in builder_strategies
    )
    assert branch["key_event_critic_completed"] is False


def test_frontier_builder_inherits_selected_horizon_and_unresolved_repair() -> None:
    root = normalize_strategy_card(
        {
            "strategy_query": "construct the target-side scaffold",
            "critical_assumption": "the root construction is selective",
            "critic_checkpoint": "observe the root scaffold",
        }
    )
    milestone = normalize_strategy_card(
        {
            "strategy_query": "execute the upstream stereocontrolled annulation",
            "critical_assumption": "the annulation fixes the required junction",
            "critic_checkpoint": "observe the configured annulation product",
        }
    )
    milestone["host_lineage"] = {
        "root_mapped_smiles": "[CH3:1][CH3:2]",
        "milestone_index": 2,
    }
    graph = {
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:target": {"canonical_smiles": "CCO"},
            "molecule:ethyl": {"canonical_smiles": "CC"},
            "molecule:water": {"canonical_smiles": "O"},
        },
        "route_families": {
            "route:one": {
                "route_family_id": "route:one",
                "edge_ids": ["edge:root"],
                "strategy_card": root,
                "strategy_milestone_cards": [root, milestone],
                "key_event_critic_history": [
                    {
                        "status": "rejected",
                        "focus_step_id": "step:rejected-checkpoint",
                        "strategy_digest": milestone["strategy_digest"],
                        "strategy_milestone_index": 2,
                        "lineage_root_mapped_smiles": "[CH3:1][CH3:2]",
                        "checkpoint_match": True,
                        "assessment": {
                            "blocking": True,
                            "blocking_type": "stereochemistry",
                            "reasons": ["the junction configuration is not established"],
                            "suggested_revision": "rebuild the annulation sequence",
                        },
                    }
                ],
                "path_repair_transactions": [
                    {
                        "transaction_index": 1,
                        "status": "retained_uncommitted_prefix",
                        "reason": "path_repair_reconnect_boundary_not_reached",
                        "rollback_start_step_id": "step:root",
                        "rebuild_through_step_id": "step:rejected-checkpoint",
                        "repair_goal": "rebuild through the actual annulation boundary",
                        "active_constraints": [
                            "preserve the configured ring junction across the rebuilt span"
                        ],
                        "repair_frontier_mapped_product_smiles": "[CH3:1][CH3:2]",
                    }
                ],
            }
        },
        "edges": {
            "edge:root": {
                "edge_id": "edge:root",
                "product_molecule_id": "molecule:target",
                "product_smiles": "CCO",
                "precursor_molecule_ids": ["molecule:ethyl", "molecule:water"],
                "precursor_smiles": ["CC", "O"],
                "reactionjson_audit": {
                    "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                    "mapped_precursor_smiles": [
                        "[CH3:1][CH3:2]",
                        "[OH2:3]",
                    ],
                },
                "origin_records": [
                    {
                        "proposal_id": "step:root",
                        "canonical_route_family_ids": ["route:one"],
                    }
                ],
            }
        },
    }

    context, diagnostic = compile_frontier_builder_context(
        graph,
        frontier_molecule_id="molecule:ethyl",
        route_family_ids=("route:one",),
    )

    assert diagnostic == {}
    assert context is not None
    assert context.strategy_card["strategy_query"] == milestone["strategy_query"]
    assert context.pending_checkpoint_feedback is not None
    assert context.pending_checkpoint_feedback["active_constraints"][0][
        "blocking_type"
    ] == "stereochemistry"
    assert context.path_repair is not None
    assert context.path_repair["status"] == "retained_uncommitted_prefix"
    prompt = SequentialStrategyDirectorRunner().frontier_prompt_for(
        context,
        DirectorConfig(paper_matched_reach_profile=True),
    )
    payload = json.loads(prompt.rsplit("\n", 1)[1])
    assert payload["phase"] == "route_local_repair"
    assert payload["strategy"]["strategy_query"] == milestone["strategy_query"]
    assert payload["pending_checkpoint_feedback"]["active_constraints"]
    assert payload["path_repair"]["repair_goal"] == (
        "rebuild through the actual annulation boundary"
    )


def test_frontier_builder_context_refuses_multi_route_strategy_binding() -> None:
    context, diagnostic = compile_frontier_builder_context(
        {"route_families": {}},
        frontier_molecule_id="molecule:leaf",
        route_family_ids=("route:a", "route:b"),
    )

    assert context is None
    assert diagnostic["reason"] == ("frontier_builder_route_family_binding_ambiguous")


def test_frontier_builder_context_uses_current_host_proof_mapping() -> None:
    graph = {
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:target": {"canonical_smiles": "CCO"},
            "molecule:ethyl": {"canonical_smiles": "CC"},
            "molecule:water": {"canonical_smiles": "O"},
        },
        "route_families": {
            "route:one": {
                "route_family_id": "route:one",
                "edge_ids": ["edge:root"],
                "strategy_card": {},
            }
        },
        "edges": {
            "edge:root": {
                "edge_id": "edge:root",
                "product_molecule_id": "molecule:target",
                "product_smiles": "CCO",
                "precursor_molecule_ids": ["molecule:ethyl", "molecule:water"],
                "precursor_smiles": ["CC", "O"],
                "route_family_ids": ["route:one"],
                "reactionjson_audit": {},
                "reaction_proofs": [
                    {
                        "accepted": True,
                        "validator_version": CURRENT_REACTION_VALIDATOR_VERSION,
                        "mapped_reaction": ("[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"),
                        "checks": {
                            "mapped_reaction_present": True,
                            "mapped_product_matches": True,
                            "mapped_reactants_match": True,
                            "atom_maps_complete": True,
                            "product_atom_maps_complete": True,
                            "atom_maps_unique": True,
                        },
                    }
                ],
                "origin_records": [
                    {
                        "proposal_id": "step:root",
                        "canonical_route_family_ids": ["route:one"],
                    }
                ],
            }
        },
    }

    context, diagnostic = compile_frontier_builder_context(
        graph,
        frontier_molecule_id="molecule:ethyl",
        route_family_ids=("route:one",),
    )

    assert diagnostic == {}
    assert context is not None
    assert context.selected_product_mapped == "[CH3:1][CH3:2]"
    assert context.connected_steps[0]["mapped_precursor_smiles"] == [
        "[CH3:1][CH3:2]",
        "[OH2:3]",
    ]


def test_frontier_builder_missing_host_mapping_waits_for_validation() -> None:
    graph = {
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:target": {"canonical_smiles": "CCO"},
            "molecule:ethyl": {"canonical_smiles": "CC"},
            "molecule:water": {"canonical_smiles": "O"},
        },
        "route_families": {
            "route:one": {
                "route_family_id": "route:one",
                "edge_ids": ["edge:root"],
                "strategy_card": {},
            }
        },
        "edges": {
            "edge:root": {
                "edge_id": "edge:root",
                "product_molecule_id": "molecule:target",
                "product_smiles": "CCO",
                "precursor_molecule_ids": ["molecule:ethyl", "molecule:water"],
                "precursor_smiles": ["CC", "O"],
                "reactionjson_audit": {},
                "reaction_proofs": [],
            }
        },
    }

    context, diagnostic = compile_frontier_builder_context(
        graph,
        frontier_molecule_id="molecule:ethyl",
        route_family_ids=("route:one",),
    )

    assert context is None
    assert diagnostic == {
        "reason": "frontier_builder_mapped_path_incomplete",
        "edge_id": "edge:root",
        "retryable_after_reaction_validation": True,
        "prerequisite_kind": "reaction_validation",
    }


def test_strategy_portfolio_modes_keep_paper_prior_neutral_and_enzyme_arm_explicit() -> None:
    paper = _branch_mandates_for_profile("paper_independent")
    assert len(paper) == 1
    assert "enzymatic" not in paper[0]
    assert "whole-cell" not in paper[0]
    assert "execution-domain prior" in paper[0]

    hybrid = _branch_mandates_for_profile("autoplanner_hybrid")
    assert len(hybrid) == 3
    assert all("execution_domain=chemical" in row for row in hybrid[:2])
    assert "execution_domain=hybrid" in hybrid[2]
    assert "genuine chemoenzymatic route" in hybrid[2]
    assert "generic cyclase" in hybrid[2]

    enzyme = _branch_mandates_for_profile("enzyme_advantage")
    assert len(enzyme) == 1
    assert "explicit substrate-product boundary" in enzyme[0]
    assert "do not invent enzyme capability" in enzyme[0]

    fusion = _branch_mandates_for_profile("chemoenzymatic_fusion")
    assert len(fusion) == 1
    assert "at least one chemical scaffold-forming step" in fusion[0]
    assert "at least one enzymatic or whole-cell" in fusion[0]
    assert "biological label or cofactor list alone does not satisfy" in fusion[0]


def test_strategy_v2_has_three_structurally_distinct_slots_and_precursor_boundary() -> None:
    v2 = _branch_mandates_for_profile("autoplanner_strategy_v2")
    assert len(v2) == 3
    assert {"convergent", "topology", "reorganization"} <= {
        row.split("strategy_v2_slot=", 1)[1].split(":", 1)[0] for row in v2
    }
    prompt = _strategy_prompt(
        target="CCO",
        branch_index=1,
        lens=v2[1],
        forbidden_strategy_cards=(),
        prior_rejections=(),
    )
    assert "strategy_portfolio_generator_input.v2" in prompt
    assert "anchor_bond_changes" in prompt
    assert "precursor_only_bond_changes" in prompt
    assert "Do not use web search" in prompt


def test_paper_matched_strategy_prompt_builds_a_diverse_four_point_portfolio() -> None:
    prompt = _paper_strategy_portfolio_prompt(target="C1CCCC1")
    frozen_prompt = _paper_strategy_portfolio_prompt(
        target="C1CCCC1",
        enhanced=False,
    )

    assert "exactly three independent high-level strategies" in prompt
    assert "single call" in prompt
    assert "paper's four dimensions" in prompt
    assert "one strategy_query sentence" in prompt
    assert "one critical_assumption sentence" in prompt
    assert "one critic_checkpoint sentence" in prompt
    assert "compare and attack their weakest chemical assumptions" in prompt
    assert "skeletal construction or reorganization" in prompt
    assert "principal ring system" in prompt
    assert "same unexplained complex core" in prompt
    assert "It need not enumerate the complete route or every ring closure" in prompt
    assert "chiral-pool" in prompt
    assert "natural biosynthetic origin alone is not evidence" in prompt
    assert "reactive-handle motif" in prompt
    assert "stereochemical or functional-group control" in prompt
    assert "consumable reactive-handle motif" in prompt
    assert "unsupported C-H bond formations" in prompt
    assert "Do not output atom-map pairs" in prompt
    assert "mechanistic essay" in prompt
    assert '"ring_sizes":[5]' in prompt
    assert "earliest non-substitutable graph transformation" in prompt
    assert "campaign_target_mapped" not in prompt
    assert "execution_domain=chemical" not in prompt
    assert "ketyl" not in prompt.lower()
    assert "grob" not in prompt.lower()
    assert "principal ring system" not in frozen_prompt
    assert '"ring_systems"' not in frozen_prompt
    assert "skeletal construction or reorganization" in frozen_prompt


def test_strategy_topology_profile_distinguishes_ring_sizes_from_fusion() -> None:
    target = "C=C(C)[C@H]1CC[C@]2(C)C[C@@H]3[C@H](C)CC[C@@H]3/C(C)=C\\CC12"
    generator_prompt = _paper_strategy_portfolio_prompt(target=target)
    critic_prompt = sequential_module._paper_strategy_portfolio_critic_prompt(
        target=target,
        strategy_cards=[
            {
                "strategy_query": "test strategy",
                "critical_assumption": "test assumption",
                "critic_checkpoint": "test checkpoint",
            }
        ]
        * 3,
    )

    for prompt in (generator_prompt, critic_prompt):
        assert '"ring_sizes":[5,5,8]' in prompt
        assert '"ring_systems":[' in prompt
        assert '"pair_count":1,"ring_sizes":[5,5],"shared_atom_count":0' in prompt
        assert '"pair_count":2,"ring_sizes":[5,8],"shared_atom_count":2' in prompt
    assert "never infer a fused or spiro relationship from ring_sizes alone" in (critic_prompt)
    assert "shared unexplained complex core" in critic_prompt
    assert "Do not make an acceptable card more specific" in critic_prompt
    assert "named downstream reaction" in critic_prompt
    assert "change only the contradicted clause" in critic_prompt
    assert "replacement at the same high-level Strategy granularity" in critic_prompt
    assert "lacks consumable reactive handles" in critic_prompt
    assert "control is only an adjective" in critic_prompt
    assert "enolate" not in critic_prompt.casefold()
    assert "radical" not in critic_prompt.casefold()


def test_upstream_strategy_prompt_uses_real_leaf_and_compact_route_horizon() -> None:
    prompt = sequential_module._milestone_strategy_prompt(
        campaign_target="C1CCC2CCCCC2C1",
        selected_product="C1CCCCC1",
        selected_product_mapped="[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1",
        branch_index=0,
        milestone_index=2,
        strategy_mandate="retain chemically credible core simplification",
        completed_strategy_cards=[
            {
                "strategy_query": "Build the first fused-ring junction by annulation.",
                "critic_checkpoint": "Audit the annulation graph edit.",
            }
        ],
        route_steps=[],
    )

    assert '"selected_upstream_leaf":"C1CCCCC1"' in prompt
    assert '"selected_upstream_leaf_mapped":' in prompt
    assert '"selected_upstream_leaf_topology_profile":' in prompt
    assert '"strategy_query":"Build the first fused-ring junction by annulation."' in prompt
    assert "one Host-derived leaf-lineage projection" in prompt
    assert "complex principal ring system" in prompt
    assert "one concise strategy_query" in prompt
    assert "not a complete route" in prompt
    assert "operational at Strategy granularity" in prompt
    assert "selected_upstream_leaf_bond_pairs" not in prompt
    context = json.loads(prompt.split("BlindUpstreamStrategyMilestoneInput:\n", 1)[1])
    assert context["schema_version"] == "strategy_horizon_context.v1"
    assert context["phase"] == "strategy_horizon_generation"
    assert context["connected_path_reactions"] == []
    assert "accepted_target_rooted_prefix" not in context


def test_upstream_strategy_critic_receives_lineage_spine_and_keeps_same_event_control() -> None:
    prompt = sequential_module._upstream_strategy_critic_prompt(
        campaign_target="CCO",
        selected_product="CC",
        selected_product_mapped="[CH3:1][CH3:2]",
        branch_index=0,
        milestone_index=2,
        generated_card={
            "strategy_query": "Close the ring by a stereocontrolled aldol reaction.",
            "critical_assumption": "The aldol event sets the required alcohol configuration.",
            "critic_checkpoint": "Formation of the ring bond and required alcohol configuration.",
        },
        completed_strategy_cards=[
            {
                "strategy_query": "Build the downstream ring by cycloaddition.",
                "critical_assumption": "The tether selects one cycloadduct.",
                "critic_checkpoint": "Formation of both cycloaddition bonds.",
            }
        ],
        accepted_route_steps=[
            {
                "step_id": "accepted:1",
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
                "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                "reaction_family": "accepted downstream cleavage",
                "conditions": ["test conditions"],
            }
        ],
    )

    context = json.loads(prompt.split("UpstreamStrategyCheckpointReviewInput:\n", 1)[1])
    assert context["schema_version"] == "strategy_horizon_context.v1"
    assert context["phase"] == "strategy_horizon_review"
    assert context["campaign_target"] == "CCO"
    assert context["completed_milestones"][0]["critical_assumption"] == (
        "The tether selects one cycloadduct."
    )
    assert context["connected_path_reactions"] == [
        {
            "checkpoint_relation": "",
            "edit_summary": "break bond maps 2-3",
            "reaction_family": "accepted downstream cleavage",
            "step_id": "accepted:1",
        }
    ]
    assert "accepted_target_rooted_prefix" not in context
    assert "[CH3:1][CH2:2][OH:3]" not in json.dumps(context)
    assert "one Host-derived molecular-occurrence lineage" in prompt
    assert "not required to be the next Builder reaction" in prompt
    assert "without replacing the route-defining horizon" in prompt
    assert "every revision or replacement must retain route-defining" in prompt
    assert "principal scaffold is already simple" in prompt
    assert "do not weaken such a checkpoint to bond formation alone" in prompt
    assert "Do not invent a more specific named reaction" in prompt
    assert "lacks consumable reactive handles" in prompt


def test_strategy_generator_and_critic_share_selected_leaf_lineage_without_sibling_history() -> (
    None
):
    route_steps = [
        {
            "step_id": "root:split",
            "product_smiles": "CCO",
            "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "precursor_smiles": ["CC", "O"],
            "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
            "reaction_family": "target split",
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        },
        {
            "step_id": "sibling:expanded",
            "product_smiles": "O",
            "mapped_product_smiles": "[OH2:3]",
            "precursor_smiles": ["N"],
            "mapped_precursor_smiles": ["[NH3:3]"],
            "reaction_family": "sibling-only reaction",
            "reaction_operations": [{"op": "change_atom", "map_idx": 3, "formal_charge": 0}],
        },
    ]
    completed = [
        {
            "strategy_query": "Build the downstream bond.",
            "critical_assumption": "The split is selective.",
            "critic_checkpoint": "Observe the split products.",
        }
    ]
    generated = {
        "strategy_query": "Deconstruct the selected carbon fragment.",
        "critical_assumption": "The carbon fragment can be simplified.",
        "critic_checkpoint": "Observe the carbon-fragment disconnection.",
    }

    generator_prompt = sequential_module._milestone_strategy_prompt(
        campaign_target="CCO",
        selected_product="CC",
        selected_product_mapped="[CH3:1][CH3:2]",
        branch_index=0,
        milestone_index=2,
        strategy_mandate="select the next route-defining horizon",
        completed_strategy_cards=completed,
        route_steps=route_steps,
    )
    critic_prompt = sequential_module._upstream_strategy_critic_prompt(
        campaign_target="CCO",
        selected_product="CC",
        selected_product_mapped="[CH3:1][CH3:2]",
        branch_index=0,
        milestone_index=2,
        generated_card=generated,
        completed_strategy_cards=completed,
        accepted_route_steps=route_steps,
    )
    generator_context = json.loads(
        generator_prompt.split("BlindUpstreamStrategyMilestoneInput:\n", 1)[1]
    )
    critic_context = json.loads(
        critic_prompt.split("UpstreamStrategyCheckpointReviewInput:\n", 1)[1]
    )

    shared_fields = {
        "campaign_target",
        "selected_upstream_leaf",
        "selected_upstream_leaf_mapped",
        "selected_upstream_leaf_profile",
        "selected_upstream_leaf_topology_profile",
        "branch_id",
        "milestone_index",
        "completed_milestones",
        "connected_path_reactions",
        "current_split_context",
    }
    assert {key: generator_context[key] for key in shared_fields} == {
        key: critic_context[key] for key in shared_fields
    }
    assert [row["step_id"] for row in generator_context["connected_path_reactions"]] == [
        "root:split"
    ]
    assert "sibling-only reaction" not in json.dumps(generator_context)
    assert generator_context["current_split_context"]["co_precursors"] == [
        {
            "mapped_smiles": "[OH2:3]",
            "path_status": "expanded_on_current_path",
        }
    ]
    assert "accepted_target_rooted_prefix" not in generator_context
    assert "accepted_target_rooted_prefix" not in critic_context


def test_paper_matched_node_prompt_is_compact_single_reactionjson_policy() -> None:
    prompt = _node_prompt(
        target="CCO",
        branch_index=0,
        lens="neutral",
        selected_product="CCO",
        selected_product_mapped="[CH3:1][CH2:2][OH:3]",
        steps=(),
        open_leaves=("CCO",),
        prior_rejections=(),
        repair=False,
        strategy_card={
            "strategy_query": "Construct the C-O bond from complementary handles.",
            "scaffold_motif": "acyclic oxygenated scaffold",
            "key_forward_transformation": "C-O bond formation",
            "key_bond_changes": ["map 2-map 3"],
            "functional_group_conflicts": [],
            "protection_policy": "none",
            "stereochemical_plan": "none",
            "strategic_step_count": 1,
            "critical_assumption": "The complementary handles react selectively.",
            "critic_checkpoint": "Audit the step whose graph edit forms the C-O bond.",
        },
        forbidden_strategy_cards=(_strategy_card(2),),
        host_failure_feedback={
            "pending_checkpoint_feedback": {
                "active_constraints": [
                    {
                        "severity": "blocking",
                        "checkpoint_match": False,
                        "blocking_type": "sequence_dependency",
                        "reasons": ["preserve one tethered intramolecular precursor"],
                        "suggested_revision": ("use a disconnection that retains the tether"),
                        "source_focus_step_id": "step:false-key",
                    }
                ],
                "failure_basin": {
                    "distinct_rejected_attempt_count": 2,
                    "recurring_blocking_types": ["sequence_dependency"],
                    "recurrent_across_distinct_candidates": True,
                    "authority": "derived_diagnostic_only",
                },
            }
        },
        max_reactionjson_candidates=1,
        paper_matched=True,
    )

    assert "next-step expansion policy for one selected MCTS node" in prompt
    assert "steering hypothesis" in prompt
    assert "guides the whole pathway" in prompt
    assert "work out a complete chemically coherent pathway" in prompt
    assert "one-object output boundary does not limit route-level reasoning" in prompt
    assert "do not generate the remaining route" not in prompt
    assert "future steps" not in prompt
    assert "Do not plan" not in prompt
    assert "strategy_relation" not in prompt
    assert "checkpoint_relation=executes_checkpoint" in prompt
    assert "checkpoint_relation=preparatory" in prompt
    assert "complete connected bond-change pattern" in prompt
    assert "Do not telescope independent events" in prompt
    assert "separate reaction edge" in prompt
    assert "do not split one mechanistic event into fictitious intermediates" in prompt
    assert "actual net graph edit, not the reaction name" in prompt
    assert "cannot substitute for missing topology or stereochemical information" in prompt
    assert "feasibility_check" not in prompt
    assert "reaction_intent" in prompt
    assert "move_role" not in prompt
    assert "Privately challenge the chosen move" in prompt
    assert "include any catalyst there" in prompt
    assert "mentally replay it" in prompt
    assert "change_bond_order uses signed delta" in prompt
    assert "change_atom changes formal_charge or isotope only" in prompt
    assert "exactly one [*] attachment atom" in prompt
    assert "add_bond always creates a single bond and has no order field" in prompt
    assert "follow add_bond with change_bond_order delta 1 or 2" in prompt
    assert "[*]=O" in prompt
    assert "do not output order" in prompt
    assert "1.5" not in prompt
    assert "Host derives RDKit stereo reference neighbours" in prompt
    assert "Protection/deprotection" in prompt
    assert "precursor set -> selected_leaf_mapped product" in prompt
    assert "across preparatory moves" in prompt
    assert "a newer finding does not replace an older one" in prompt
    assert "only a later selected Critic pass retires the set" in prompt
    assert "do not merely rename reagents or draw another cosmetic checkpoint variant" in prompt
    assert "does not itself reject the Strategy" in prompt
    assert "current_split_context.co_precursors" not in prompt
    assert "complete RouteJSON" in prompt
    assert '"strategy_anchor_progress"' not in prompt
    assert '"remaining_map_pairs"' not in prompt
    assert "handoff" in prompt
    assert "has no handoff, fail, stop, or solved action" in prompt
    assert "fail_branch" not in prompt
    assert "route_complete" not in prompt
    assert "strategy_progress" not in prompt
    assert "key_step_seen" not in prompt
    assert "The Host/MCTS alone decides termination" in prompt
    assert "Expand exactly one retrosynthetic node" not in prompt
    assert '"schema_version":"sequential_route_builder_context.v1"' in prompt
    assert "forbidden_root_strategies" not in prompt
    assert "campaign_target_profile" not in prompt
    assert "biocatalytic_intent" not in prompt
    context = json.loads(prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
    assert set(context["strategy"]) == {
        "strategy_query",
        "critical_assumption",
        "critic_checkpoint",
    }
    assert "key_bond_changes" not in context["strategy"]
    assert "accepted_strategy_spine" not in context
    assert "strategy_progress" not in context
    assert "accepted_path" not in context
    assert "open_leaves" not in context
    assert "connected_path_reactions" not in context
    assert "ancestor_smiles" not in context
    assert "current_split_context" not in context
    assert "last_rejection_for_this_leaf" not in context
    assert context["pending_checkpoint_feedback"] == {
        "active_constraints": [
            {
                "severity": "blocking",
                "checkpoint_match": False,
                "blocking_type": "sequence_dependency",
                "reasons": ["preserve one tethered intramolecular precursor"],
                "suggested_revision": "use a disconnection that retains the tether",
                "source_focus_step_id": "step:false-key",
            }
        ],
        "failure_basin": {
            "distinct_rejected_attempt_count": 2,
            "recurring_blocking_types": ["sequence_dependency"],
            "recurrent_across_distinct_candidates": True,
            "authority": "derived_diagnostic_only",
        },
    }


def test_paper_path_repair_uses_same_builder_contract_with_compact_boundary() -> None:
    prompt = _node_prompt(
        target="CCO",
        branch_index=0,
        lens="local repair",
        selected_product="CC",
        selected_product_mapped="[CH3:1][CH3:2]",
        steps=(
            {
                "step_id": "route:1",
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
                "reaction_family": "C-O disconnection",
                "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
            },
        ),
        open_leaves=("CC",),
        prior_rejections=(),
        repair=True,
        strategy_card=_strategy_card(1),
        forbidden_strategy_cards=(),
        host_failure_feedback={
            "path_repair": {
                "rollback_start_step_id": "route:2",
                "rebuild_through_step_id": "route:3",
                "repair_goal": "preserve the carbonyl oxygen",
                "active_constraints": ["retain the key cyclization"],
                "reconnect_boundaries": [
                    {
                        "step_id": "route:4",
                        "product_smiles": "CCCl",
                        "mapped_product_smiles": "[CH3:1][CH2:2][Cl:4]",
                    }
                ],
                "replay_failures": [
                    {
                        "replay_error": "reactionjson_stereo_reference_neighbor_missing",
                        "failed_operation": {
                            "op": "set_bond_stereo",
                            "map_a": 12,
                            "map_b": 13,
                            "stereo": "E",
                        },
                        "occurrence_count": 2,
                    }
                ],
                "repair_reference_span": [
                    {
                        "step_id": "route:2",
                        "mapped_product_smiles": "[CH3:1][CH3:2]",
                        "mapped_precursor_smiles": ["[CH4:1]", "[CH4:2]"],
                        "reaction_operations": [
                            {"op": "break_bond", "map_a": 1, "map_b": 2}
                        ],
                        "prior_key_critic": {
                            "status": "completed",
                            "checkpoint_match": True,
                            "verdict": "pass",
                        },
                    }
                ],
            }
        },
        max_reactionjson_candidates=1,
        paper_matched=True,
    )

    assert "under a route-local repair" in prompt
    assert "hands off an eligible" not in prompt
    assert "or fails this Strategy branch" not in prompt
    assert "The Builder has no handoff, fail, stop, or solved action" in prompt
    assert "path_repair.repair_goal guides the replacement chemistry" in prompt
    assert "repair_reference_span is the compact Host-replayed mutable span" in prompt
    assert "reference, not accepted history" in prompt
    assert "bounded replacement transaction, not a new stock search" in prompt
    assert "shortest chemically coherent local replacement" in prompt
    assert "toward accessible precursors" not in prompt
    assert "transaction-wide negative memory" in prompt
    assert "never signals repair completion" in prompt
    assert "route Critic alone decides" in prompt
    context = json.loads(prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
    assert context["phase"] == "route_local_repair"
    assert context["path_repair"]["repair_goal"] == ("preserve the carbonyl oxygen")
    assert context["path_repair"]["rollback_start_step_id"] == "route:2"
    assert context["path_repair"]["rebuild_through_step_id"] == "route:3"
    assert len(context["path_repair"]["reconnect_boundaries"]) == 1
    assert context["path_repair"]["replay_failures"][0]["occurrence_count"] == 2
    assert context["path_repair"]["repair_reference_span"][0]["step_id"] == "route:2"
    assert context["connected_path_reactions"][0]["reaction_family"] == ("C-O disconnection")
    assert "accepted_path" not in context
    assert "open_leaves" not in context


def test_paper_builder_context_keeps_reaction_spine_roles_without_conditions() -> None:
    prompt = _node_prompt(
        target="CCO",
        branch_index=0,
        lens="neutral",
        selected_product="CO",
        selected_product_mapped="[CH3:2][OH:3]",
        steps=(
            {
                "step_id": "route:1",
                "product_smiles": "CCO",
                "precursor_smiles": ["C", "CO"],
                "transformation_hypothesis": "downstream ring opening",
                "strategic_role": "Expose the diene needed for the planned IMDA.",
                "step_role": "enabling",
                "condition_predictions": [{"reagents": ["bulky base"], "catalyst": "Pd complex"}],
            },
        ),
        open_leaves=("C", "CO", "unrelated-open-leaf"),
        prior_rejections=(),
        repair=False,
        strategy_card={
            "strategy_query": "Construct the decalin through an IMDA.",
            "critical_assumption": "The tethered diene and dienophile can cyclize selectively.",
        },
        forbidden_strategy_cards=(),
        host_failure_feedback={},
        paper_matched=True,
    )

    context = json.loads(prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
    assert "accepted_strategy_spine" not in context
    assert "strategy_progress" not in context
    serialized = json.dumps(context, ensure_ascii=False)
    assert "downstream ring opening" in serialized
    assert "claimed_move_role" not in serialized
    assert "Expose the diene" not in serialized
    assert "bulky base" not in serialized
    assert "Pd complex" not in serialized
    assert "unrelated-open-leaf" not in serialized
    assert '"precursor_smiles"' not in serialized


def test_paper_builder_context_reuses_compact_host_ring_paths() -> None:
    prompt = _node_prompt(
        target="CC1CCCCC1",
        branch_index=0,
        lens="ring construction",
        selected_product="CC1CCCCC1",
        selected_product_mapped=("[CH3:1][C@H:2]1[CH2:3][CH2:4][CH2:5][CH2:6][CH2:7]1"),
        steps=(),
        open_leaves=("CC1CCCCC1",),
        prior_rejections=(),
        repair=False,
        strategy_card={
            "strategy_query": "Construct the six-membered ring by cyclization.",
            "critical_assumption": "The tether closes selectively.",
            "critic_checkpoint": "Form the six-membered ring.",
        },
        forbidden_strategy_cards=(),
        host_failure_feedback={},
        paper_matched=True,
    )

    context = json.loads(prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
    topology = context["selected_leaf_topology"]
    assert topology["ring_sizes"] == [6]
    assert len(topology["ring_paths"]) == 1
    assert set(topology["ring_paths"][0]) == {2, 3, 4, 5, 6, 7}
    assert "atoms" not in topology
    assert "bonds" not in topology
    assert "selected_leaf_topology is the Host's compact RDKit ring-path" in prompt


def test_key_event_focus_binding_is_unique_and_host_owned() -> None:
    provider_critique = {
        "checkpoint_match": False,
        "step_assessments": [
            {
                "step_id": "",
                "verdict": "reject",
                "blocking": True,
                "blocking_type": "sequence_dependency",
            }
        ],
    }

    bound = sequential_module._bind_key_event_focus_assessment(
        provider_critique,
        "host:focus",
    )

    assert provider_critique["step_assessments"][0]["step_id"] == ""
    assert bound["step_assessments"][0]["step_id"] == "host:focus"
    assert (
        sequential_module._key_event_focus_assessment(
            bound,
            "host:focus",
        )["verdict"]
        == "reject"
    )

    conflicting = sequential_module._bind_key_event_focus_assessment(
        {"step_assessments": [{"step_id": "provider:other", "verdict": "reject"}]},
        "host:focus",
    )
    ambiguous = sequential_module._bind_key_event_focus_assessment(
        {
            "step_assessments": [
                {"step_id": "", "verdict": "reject"},
                {"step_id": "", "verdict": "reject"},
            ]
        },
        "host:focus",
    )

    assert (
        sequential_module._key_event_focus_assessment(
            conflicting,
            "host:focus",
        )
        is None
    )
    assert (
        sequential_module._key_event_focus_assessment(
            ambiguous,
            "host:focus",
        )
        is None
    )


def test_paper_builder_context_keeps_only_connected_structural_ancestors() -> None:
    prompt = _node_prompt(
        target="CCCO",
        branch_index=0,
        lens="neutral",
        selected_product="C",
        selected_product_mapped="[CH4:1]",
        steps=(
            {
                "product_smiles": "CCCO",
                "precursor_smiles": ["CC", "O"],
                "step_role": "key",
                "transformation_hypothesis": "root split",
            },
            {
                "product_smiles": "CC",
                "precursor_smiles": ["C"],
                "step_role": "enabling",
                "transformation_hypothesis": "connected edit",
            },
            {
                "product_smiles": "O",
                "precursor_smiles": ["[H][H]"],
                "step_role": "supporting",
                "transformation_hypothesis": "unrelated sibling edit",
            },
        ),
        open_leaves=("C",),
        prior_rejections=(
            {
                "reason": "candidate_returns_to_ancestor",
                "product_smiles": "C",
                "ancestor_smiles": ["CC"],
            },
            {
                "reason": "failure_from_other_leaf_must_not_leak",
                "product_smiles": "O",
            },
        ),
        repair=False,
        strategy_card={
            "strategy_query": "Disconnect the carbon chain.",
            "strategy_signature": "chain disconnection",
        },
        forbidden_strategy_cards=(),
        host_failure_feedback={},
        paper_matched=True,
    )

    context = json.loads(prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
    assert context["ancestor_smiles"] == ["CC", "CCCO"]
    assert [row["reaction_family"] for row in context["connected_path_reactions"]] == [
        "root split",
        "connected edit",
    ]
    assert context["last_rejection_for_this_leaf"] == {
        "ancestor_smiles": ["CC"],
        "product_smiles": "C",
        "reason": "candidate_returns_to_ancestor",
        "replay_diagnostic": {"reason": "candidate_returns_to_ancestor"},
    }
    serialized = json.dumps(context, ensure_ascii=False)
    assert "unrelated sibling edit" not in serialized
    assert "failure_from_other_leaf_must_not_leak" not in serialized
    assert all("claimed_move_role" not in row for row in context["connected_path_reactions"])
    assert '"step_role"' not in serialized


def test_editor_prompt_allows_paper_level_route_repair_without_truncation() -> None:
    prompt = _node_prompt(
        target="CCO",
        branch_index=0,
        lens="paper editor",
        selected_product="CCO",
        selected_product_mapped="[CH3:1][CH2:2][OH:3]",
        steps=(
            {
                "step_id": "route:1",
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
                "condition_predictions": [
                    {
                        "reagents": ["base"],
                        "catalyst": "Pd catalyst",
                    }
                ],
                "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                "strategy_anchor": True,
            },
            {
                "step_id": "route:2",
                "product_smiles": "CC",
                "precursor_smiles": ["C", "C"],
                "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
            },
        ),
        open_leaves=("C", "O"),
        prior_rejections=(),
        repair=True,
        strategy_card=_strategy_card(1),
        forbidden_strategy_cards=(),
        host_failure_feedback={"blocking_step": {"step_id": "route:2", "product_smiles": "CC"}},
        complete_route_json=True,
        editor_route_mutations=True,
        paper_matched=True,
    )

    assert "complete Host-replayed route" in prompt
    assert "smallest dependency-closed replace_span" in prompt
    assert "one row, several rows, or the whole route" in prompt
    assert "Choose the target-side steps" in prompt
    assert "directly generate every retained boundary" in prompt
    assert "include the incompatible retained step" in prompt
    assert "unsupported intermediate transformation" in prompt
    assert "rejected_net_edit_signatures" in prompt
    assert "Host preserves every unlisted row" in prompt
    assert "every later product_smiles must be emitted by an earlier row" in prompt
    assert "Never put a newly exposed precursor into the replacing row's product_smiles" in prompt
    assert "route_patch" not in prompt
    assert "do not repeat the failed edit unchanged" in prompt
    assert "every field change claimed in repair_summary" in prompt
    assert "add_bond always creates a single bond and has no order field" in prompt
    assert "follow add_bond with change_bond_order delta 1 or 2" in prompt
    assert "do not output order" in prompt
    assert "1.5" not in prompt
    assert "change_bond_order uses signed delta" in prompt
    assert "exactly one [*] attachment atom" in prompt
    assert "introduce or remove atoms explicitly through add_group/remove_group" in prompt
    assert "Do not output mapped_product_smiles" in prompt
    assert "step_role" not in prompt
    assert "New structures are allowed only when introduced through explicit" in prompt
    assert "repair_status=unrepairable" not in prompt
    assert "Do not output route_json" not in prompt
    assert "fewest rows" in prompt
    assert "supplied structures" not in prompt
    editor_context = json.loads(prompt.split("PaperMatchedRouteEditorContext:\n", 1)[1])
    assert set(editor_context["strategy"]) == {
        "strategy_query",
        "strategy_signature",
    }
    assert "route_json" in editor_context
    assert "critic_annotations" in editor_context
    assert "frozen_route" not in editor_context
    assert "critic_feedback" not in editor_context


def test_critic_prompt_separates_uncertainty_from_reject_and_keeps_full_route() -> None:
    prompt = sequential_module._critic_prompt(
        target="CCO",
        branch_index=0,
        strategy_card=_strategy_card(1),
        steps=(
            {
                "step_id": "route:1",
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
                "condition_predictions": [{"reagents": ["base"], "catalyst": "Pd catalyst"}],
                "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                "strategy_anchor": True,
            },
        ),
        paper_matched=True,
    )

    assert "pass means executable as written" in prompt
    assert "merely underspecified conditions are not blockers" in prompt
    assert "exact host-derived mapped products" in prompt
    assert "net structural/H/charge/redox plausibility" in prompt
    assert "preserve mapped element identity" in prompt
    assert "change_atom may change formal charge or isotope only" in prompt
    assert "non-route reagents" in prompt
    assert "at most two concrete reasons" in prompt
    assert "No Builder checkpoint_relation, role label, or host anchor claim is evidence" in prompt
    assert "strategy_adherence=false" in prompt
    assert "observation metadata only" in prompt
    assert "Strategy non-adherence alone is not a blocker" in prompt
    assert "opportunistic route such as a stock-closed short path" in prompt
    assert "overall_assessment reports chemical route validity only" in prompt
    assert "without complementary handles" in prompt
    assert "changed label, catalyst, or condition" in prompt
    assert "direct C-H/C-H bond formation" not in prompt
    assert "missing structural handle" in prompt
    assert "target-to-current-frontier boundary" in prompt
    assert "long mechanistic analysis" in prompt
    context = json.loads(prompt.split("PaperMatchedRouteCriticInput:\n", 1)[1])
    assert context["steps"][0]["mapped_product_smiles"] == ("[CH3:1][CH2:2][OH:3]")
    assert context["steps"][0]["mapped_precursor_smiles"] == [
        "[CH3:1][CH3:2]",
        "[OH2:3]",
    ]
    assert context["steps"][0]["conditions"] == ["base"]
    assert context["steps"][0]["catalyst"] == "Pd catalyst"
    assert "checkpoint_relation" not in context["steps"][0]
    assert "strategy_anchor" not in context["steps"][0]


def test_route_recritic_receives_host_bound_repair_checkpoint_focus() -> None:
    prompt = sequential_module._critic_prompt(
        target="C1CCCCC1",
        branch_index=0,
        strategy_card={
            "strategy_query": "form the principal ring",
            "critical_assumption": "the closure is selective",
            "critic_checkpoint": "form the six-membered ring",
        },
        steps=(
            {
                "step_id": "repair:key",
                "product_smiles": "C1CCCCC1",
                "precursor_smiles": ["CCCCCC"],
                "mapped_product_smiles": ("[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1"),
                "mapped_precursor_smiles": ["[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH3:6]"],
                "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 6}],
            },
        ),
        paper_matched=True,
        repair_completion={
            "completion_mode": "strategy_checkpoint",
            "required_checkpoint_step_id": "repair:key",
            "active_constraints": ["form the intended six-membered ring"],
        },
    )

    assert "repair_checkpoint_focus binds the one Host-replayed step" in prompt
    context = json.loads(prompt.split("PaperMatchedRouteCriticInput:\n", 1)[1])
    focus = context["repair_checkpoint_focus"]
    assert focus["step_id"] == "repair:key"
    assert focus["active_constraints"] == ["form the intended six-membered ring"]
    assert focus["topology"]["step_id"] == "repair:key"
    assert focus["topology"]["product"]["ring_sizes"] == [6]
    assert set(focus["topology"]["product"]["ring_paths"][0]) == {
        1,
        2,
        3,
        4,
        5,
        6,
    }
    assert focus["topology"]["precursors"] == [
        {"precursor_index": 0, "ring_sizes": [], "ring_paths": []}
    ]


def test_editor_feedback_includes_every_concrete_critic_blocker() -> None:
    steps = [
        {
            "step_id": "route:1",
            "product_smiles": "CCO",
            "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        },
        {
            "step_id": "route:2",
            "product_smiles": "CC",
            "mapped_product_smiles": "[CH3:1][CH3:2]",
            "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
        },
    ]
    critique = {
        "overall_assessment": "reject",
        "strategy_adherence": "partial",
        "step_assessments": [
            {
                "step_id": "route:1",
                "verdict": "reject",
                "blocking": True,
                "blocking_type": "missing_reactive_handle",
                "reasons": ["nucleophile handle is absent"],
                "suggested_revision": "install the explicit organometallic handle",
            },
            {
                "step_id": "route:2",
                "verdict": "reject",
                "blocking": True,
                "blocking_type": "sequence_ordering",
                "reasons": ["protection must precede coupling"],
                "suggested_revision": "insert protection and reorder the coupling",
            },
        ],
        "repair_actions": [
            "install the explicit organometallic handle",
            "insert protection and reorder the coupling",
        ],
        "route_level_risks": ["the two blockers require a coordinated edit"],
        "experimental_variables": ["protecting group choice"],
    }

    blockers = sequential_module._blocking_critic_steps(critique, steps)
    feedback = sequential_module._compact_critic_feedback(critique, blockers)

    assert [row["step_id"] for row in blockers] == ["route:1", "route:2"]
    assert [row["route_step"]["step_id"] for row in feedback["blocking_steps"]] == [
        "route:1",
        "route:2",
    ]
    assert feedback["repair_actions"] == critique["repair_actions"]
    assert feedback["failure_reasons"] == [
        "nucleophile handle is absent",
        "protection must precede coupling",
    ]

    paper_feedback = sequential_module._compact_critic_feedback(
        critique,
        blockers,
        paper_matched=True,
    )
    assert [row["step_id"] for row in paper_feedback["blocking_steps"]] == ["route:1", "route:2"]
    assert set(paper_feedback) == {
        "overall_assessment",
        "strategy_adherence",
        "step_annotations",
        "blocking_steps",
        "rejected_net_edit_signatures",
        "route_level_risks",
    }
    assert [row["step_id"] for row in paper_feedback["rejected_net_edit_signatures"]] == [
        "route:1",
        "route:2",
    ]
    assert paper_feedback["step_annotations"] == []
    assert all("route_step" not in row for row in paper_feedback["blocking_steps"])
    assert "failure_reasons" not in paper_feedback
    assert "repair_actions" not in paper_feedback
    assert "blocking_step" not in paper_feedback
    assert "step_assessment" not in paper_feedback


def test_worker_output_contract_uses_dependency_closed_editor_span() -> None:
    task = type(
        "PaperEditorTask",
        (),
        {
            "task_type": "paper_matched_route_editor",
            "case_id": "paper-editor-case",
        },
    )()
    instruction = codex_worker_module._artifact_payload_instruction(
        "RetrosynthesisProposalReport",
        task=task,
    )

    assert "dependency-closed replace_span" in instruction
    assert "preserves all unlisted rows" in instruction
    assert "route_patch" not in instruction

    schema = codex_worker_module._retrosynthesis_proposal_report_payload_json_schema(task)
    candidate = schema["properties"]["candidates"]["items"]
    assert "route_json" not in candidate["properties"]
    span = candidate["properties"]["replace_span"]
    assert set(span["properties"]) == {"remove_step_ids", "revised_steps"}
    assert "route_patch" not in candidate["properties"]
    assert "repair_status" not in candidate["properties"]
    assert "repair_summary" in candidate["properties"]
    assert "unrepairable_reason" not in candidate["properties"]
    assert span["properties"]["remove_step_ids"]["minItems"] == 1
    assert "uniqueItems" not in span["properties"]["remove_step_ids"]
    assert span["properties"]["revised_steps"]["minItems"] == 1
    assert span["properties"]["revised_steps"]["maxItems"] == 25
    assert "product_smiles" not in candidate["properties"]
    route_step = span["properties"]["revised_steps"]["items"]
    assert "mapped_product_smiles" not in route_step["properties"]
    assert "conditions" in route_step["properties"]
    assert "step_role" not in route_step["properties"]


def test_self_correcting_strategy_schema_is_query_assumption_and_checkpoint() -> None:
    task = type(
        "PaperStrategyTask",
        (),
        {
            "task_type": "paper_matched_strategy_generator",
            "case_id": "paper-strategy-case",
        },
    )()

    card_schema = codex_worker_module._paper_strategy_card_json_schema()
    assert set(card_schema["properties"]) == {
        "strategy_query",
        "critical_assumption",
        "critic_checkpoint",
    }

    portfolio_schema = codex_worker_module._strategy_portfolio_report_payload_json_schema(task)
    assert set(portfolio_schema["properties"]) == {
        "schema_version",
        "case_id",
        "target_smiles",
        "strategy_cards",
        "no_route_or_solved_claim",
    }
    assert portfolio_schema["properties"]["strategy_cards"]["minItems"] == 3
    assert "selection_rationale" not in portfolio_schema["properties"]
    assert "limitations" not in portfolio_schema["properties"]


def test_paper_critic_editor_output_windows_fit_complete_route_documents() -> None:
    context = _context()
    spec = _spec(context)
    builder_task = sequential_module._node_task(
        spec,
        prompt="paper builder",
        branch_index=0,
        node_index=0,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        timeout_s=600,
        paper_matched=True,
    )
    editor_task = sequential_module._node_task(
        spec,
        prompt="paper editor",
        branch_index=0,
        node_index=0,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        timeout_s=600,
        task_type="route_chemistry_edit",
        paper_matched=True,
    )
    critic_task = sequential_module._critic_task(
        spec,
        prompt="paper critic",
        branch_index=0,
        iteration=0,
        timeout_s=600,
        paper_matched=True,
    )
    strategy_task = sequential_module._strategy_portfolio_task(
        spec,
        prompt="paper strategy",
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        timeout_s=600,
    )

    assert editor_task.budget.max_output_bytes == 40_000
    assert critic_task.budget.max_output_bytes == 32_000
    assert builder_task.budget.max_tool_calls is None
    assert editor_task.budget.max_tool_calls is None
    assert critic_task.budget.max_tool_calls is None
    assert strategy_task.budget.max_tool_calls is None


def test_paper_worker_response_schema_preflight_accepts_all_reachable_schemas() -> None:
    context = _context()

    sequential_module._preflight_paper_matched_worker_schemas(
        _spec(context),
        target="CCO",
        config=DirectorConfig(
            paper_matched_reach_profile=True,
            allow_editor_route_mutations=True,
        ),
    )
    sequential_module._preflight_paper_matched_worker_schemas(
        _spec(context),
        target="CCO",
        config=DirectorConfig(
            paper_matched_reach_profile=True,
            enable_strategy_portfolio_critic=True,
            enable_key_event_critic=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
        ),
    )


def test_self_correcting_critic_tasks_use_distinct_provider_contracts() -> None:
    spec = _spec(_context())
    strategy_task = sequential_module._strategy_portfolio_critic_task(
        spec,
        prompt="review portfolio",
        model="gpt-5.6-sol",
        reasoning_effort="high",
        timeout_s=10.0,
        target_smiles="CCO",
    )
    key_task = sequential_module._critic_task(
        spec,
        prompt="review key event",
        branch_index=0,
        iteration=1,
        timeout_s=10.0,
        paper_matched=True,
        target_smiles="CCO",
        audit_kind="key_event",
        focus_step_id="step:key",
    )
    final_task = sequential_module._critic_task(
        spec,
        prompt="review route",
        branch_index=0,
        iteration=1,
        timeout_s=10.0,
        paper_matched=True,
        target_smiles="CCO",
    )

    assert strategy_task.task_type == "paper_matched_strategy_critic"
    assert strategy_task.required_artifact_type == "StrategyPortfolioReport"
    assert key_task.task_type == "paper_matched_key_event_critic"
    assert final_task.task_type == "paper_matched_route_critic"
    assert key_task.task_id != final_task.task_id
    key_schema = codex_worker_module._chemical_strategy_critique_payload_json_schema(key_task)
    final_schema = codex_worker_module._chemical_strategy_critique_payload_json_schema(final_task)
    assert "checkpoint_match" in key_schema["properties"]
    assert "checkpoint_match" not in final_schema["properties"]
    assert (
        "repair_scope"
        in (key_schema["properties"]["step_assessments"]["items"]["properties"])
    )
    assert (
        "repair_scope"
        not in (final_schema["properties"]["step_assessments"]["items"]["properties"])
    )
    key_wire_schema = codex_worker_module._worker_model_output_json_schema(key_task)
    final_wire_schema = codex_worker_module._worker_model_output_json_schema(final_task)
    assert "checkpoint_match" in key_wire_schema["properties"]
    assert "checkpoint_match" not in final_wire_schema["properties"]
    assert set(key_wire_schema["properties"]) == {
        "checkpoint_match",
        "verdict",
        "blocking_type",
        "repair_scope",
        "reasons",
        "suggested_revision",
    }
    assert key_wire_schema["properties"]["repair_scope"]["enum"] == [
        "none",
        "focus_edge",
        "route_span",
        "strategy_horizon",
    ]
    assert "route_level_risks" not in key_wire_schema["properties"]


def test_key_event_repair_scope_contract_rejects_inconsistent_dispatch() -> None:
    valid = {
        "step_assessments": [
            {"step_id": "focus", "verdict": "reject", "repair_scope": "route_span"}
        ]
    }
    assert (
        codex_worker_module._paper_matched_key_event_critic_contract_reasons(valid)
        == []
    )
    valid["step_assessments"][0]["repair_scope"] = "strategy_horizon"
    assert (
        codex_worker_module._paper_matched_key_event_critic_contract_reasons(valid)
        == []
    )

    invalid = {
        "step_assessments": [
            {"step_id": "focus", "verdict": "reject", "repair_scope": "none"}
        ]
    }
    assert codex_worker_module._paper_matched_key_event_critic_contract_reasons(
        invalid
    ) == ["paper_key_critic_repair_scope_inconsistent"]


def test_key_event_repair_scope_survives_wire_materialization() -> None:
    task = sequential_module._critic_task(
        _spec(_context()),
        prompt="audit one key event",
        branch_index=0,
        iteration=1,
        timeout_s=10.0,
        paper_matched=True,
        target_smiles="CCO",
        audit_kind="key_event",
        focus_step_id="step:key",
    )
    wire = {
        "checkpoint_match": True,
        "verdict": "reject",
        "blocking_type": "stereochemistry",
        "repair_scope": "route_span",
        "reasons": ["the mapped product lacks required alkene geometry"],
        "suggested_revision": "rewrite the accepted mapped product stereo",
    }
    record = codex_worker_module.run_codex_worker(
        task,
        runner=lambda _task: codex_worker_module.WorkerProcessResult(
            stdout=json.dumps(wire),
            exit_code=0,
            backend="test",
        ),
    )

    assert record.output_validation["accepted"] is True
    assessment = record.output_artifact["payload"]["step_assessments"][0]
    assert assessment["repair_scope"] == "route_span"


def test_key_event_detector_distinguishes_execution_from_deferred_setup() -> None:
    strategy = {
        "strategy_query": ("Construct the decalin by an intramolecular Diels-Alder (IMDA)."),
        "critical_assumption": "The IMDA controls the ring-junction stereochemistry.",
        "critic_checkpoint": "Audit the graph edit that disconnects the IMDA-formed ring bonds.",
    }

    assert sequential_module._step_claims_strategy_key_event(
        {
            "transformation_hypothesis": "Retro-IMDA decalin disconnection",
            "checkpoint_relation": "executes_checkpoint",
            "reaction_operations": [
                {"op": "break_bond", "map_a": 1, "map_b": 2},
            ],
        },
        strategy,
    )
    assert not sequential_module._step_claims_strategy_key_event(
        {
            "transformation_hypothesis": (
                "Expose the diene needed for an eventual Diels-Alder cyclization"
            ),
            "checkpoint_relation": "preparatory",
            "reaction_operations": [
                {"op": "break_bond", "map_a": 1, "map_b": 2},
            ],
        },
        strategy,
    )
    assert not sequential_module._step_claims_strategy_key_event(
        {
            "transformation_hypothesis": "unnamed skeletal cascade",
            "checkpoint_relation": "preparatory",
            "reaction_operations": [
                {"op": "break_bond", "map_a": 1, "map_b": 2},
                {"op": "break_bond", "map_a": 3, "map_b": 4},
            ],
        },
        strategy,
    )
    assert sequential_module._step_claims_strategy_key_event(
        {
            "transformation_hypothesis": "checkpoint encoded as bond reorganization",
            "checkpoint_relation": "executes_checkpoint",
            "reaction_operations": [
                {"op": "change_bond_order", "map_a": 1, "map_b": 2, "delta": 1}
            ],
        },
        strategy,
    )
    assert sequential_module._step_claims_strategy_key_event(
        {
            "transformation_hypothesis": "checkpoint encoded as group replacement",
            "checkpoint_relation": "executes_checkpoint",
            "reaction_operations": [{"op": "add_group", "map_idx": 1, "fragment_smiles": "[OH]"}],
        },
        strategy,
    )


def test_default_budget_allows_one_key_event_critic_per_builder_candidate() -> None:
    context = _context()
    without_online_critic = DirectorConfig(
        planning_mode="sequential_branches",
        strategy_branch_count=3,
        max_node_expansions_per_branch=4,
        max_reactionjson_candidates_per_node=2,
        enable_key_event_critic=False,
    )
    with_online_critic = replace(
        without_online_critic,
        enable_key_event_critic=True,
    )

    base = sequential_module._node_call_budget(
        _spec(context),
        mode="initial_architecture",
        config=without_online_critic,
    )
    audited = sequential_module._node_call_budget(
        _spec(context),
        mode="initial_architecture",
        config=with_online_critic,
    )

    assert audited.model_invocations - base.model_invocations == 3 * 4 * 2


def test_passed_checkpoint_retires_only_current_strategy_horizon() -> None:
    root = normalize_strategy_card(_strategy_card(1))
    milestone = normalize_strategy_card(_strategy_card(2))
    milestone["host_lineage"] = {
        "root_mapped_smiles": "[CH3:1][CH2:2][OH:3]",
        "milestone_index": 2,
    }
    config = DirectorConfig(
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        enable_key_event_critic=True,
        max_strategic_milestones_per_branch=4,
    )

    active, refresh = sequential_module._strategy_horizon_for_leaf(
        config=config,
        branch={
            "strategy_milestone_cards": [root, milestone],
            "key_event_critic_history": [
                {
                    "strategy_digest": milestone["strategy_digest"],
                    "strategy_milestone_index": 2,
                    "focus_step_id": "step:checkpoint",
                    "status": "completed",
                }
            ],
        },
        root_strategy_card=root,
        steps=[
            {
                "step_id": "step:checkpoint",
                "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
            }
        ],
        selected_product_mapped="[CH3:1][CH2:2][OH:3]",
    )

    assert active["strategy_digest"] == milestone["strategy_digest"]
    assert refresh is True
    selected_branch = {
        "strategy_milestone_cards": [root, milestone],
        "key_event_critic_history": [
            {
                "strategy_digest": milestone["strategy_digest"],
                "strategy_milestone_index": 2,
                "focus_step_id": "step:checkpoint",
                "status": "completed",
            }
        ],
    }
    assert not sequential_module._selected_path_passed_strategy_checkpoint(
        selected_branch,
        strategy_card=milestone,
        steps=[{"step_id": "step:other"}],
    )
    assert sequential_module._selected_path_passed_strategy_checkpoint(
        selected_branch,
        strategy_card=milestone,
        steps=[{"step_id": "step:checkpoint"}],
    )


def test_strategy_horizon_rejection_replans_only_the_selected_leaf() -> None:
    root = normalize_strategy_card(_strategy_card(1))
    milestone = normalize_strategy_card(_strategy_card(2))
    selected_leaf = "[CH3:1][CH2:2][OH:3]"
    milestone["host_lineage"] = {
        "root_mapped_smiles": selected_leaf,
        "milestone_index": 2,
    }
    config = DirectorConfig(
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        enable_key_event_critic=True,
        max_strategic_milestones_per_branch=4,
    )
    branch = {
        "strategy_milestone_cards": [root, milestone],
        "key_event_critic_history": [
            {
                "strategy_digest": milestone["strategy_digest"],
                "strategy_milestone_index": 2,
                "lineage_root_mapped_smiles": selected_leaf,
                "focus_step_id": "step:rejected-checkpoint",
                "status": "rejected",
                "assessment": {
                    "verdict": "reject",
                    "blocking": True,
                    "blocking_type": "stereochemistry",
                    "repair_scope": "strategy_horizon",
                    "reasons": ["the checkpoint cannot deliver the required geometry"],
                },
            }
        ],
    }

    active, refresh = sequential_module._strategy_horizon_for_leaf(
        config=config,
        branch=branch,
        root_strategy_card=root,
        steps=[],
        selected_product_mapped=selected_leaf,
    )
    assert active["strategy_digest"] == milestone["strategy_digest"]
    assert refresh is True
    assert sequential_module._rejected_strategy_horizon_for_leaf(
        branch,
        strategy_card=active,
        steps=[],
        selected_product_mapped=selected_leaf,
    )["focus_step_id"] == "step:rejected-checkpoint"

    sibling, sibling_refresh = sequential_module._strategy_horizon_for_leaf(
        config=config,
        branch=branch,
        root_strategy_card=root,
        steps=[],
        selected_product_mapped="[OH2:3]",
    )
    assert sibling["strategy_digest"] == root["strategy_digest"]
    assert sibling_refresh is False


def test_key_event_milestone_projection_uses_selected_critic_pass_without_map_pairs() -> None:
    card = {
        "strategy_id": "strategy:prose-only",
        "strategy_digest": "d" * 64,
        "strategy_query": "Use a selective key cyclization.",
        "critical_assumption": "The cyclization controls the scaffold.",
        "critic_checkpoint": "Audit the actual scaffold-forming graph edit.",
    }
    branch = {
        "steps": [{"step_id": "step:checkpoint"}],
        "key_event_critic_history": [
            {
                "strategy_digest": card["strategy_digest"],
                "strategy_milestone_index": 1,
                "focus_step_id": "step:checkpoint",
                "status": "completed",
            }
        ],
    }

    sequential_module._refresh_strategy_milestone_projection(
        branch,
        strategy_cards=[card],
        use_key_event_critic=True,
    )

    assert branch["strategic_milestone_count"] == 1
    assert branch["strategy_anchor_diagnostics"] == [
        {
            "strategy_id": card["strategy_id"],
            "strategy_digest": card["strategy_digest"],
            "required_map_pairs": [],
            "realized_map_pairs": [],
            "remaining_map_pairs": [],
            "fulfilled": True,
            "authority": "selected_path_key_event_critic",
            "grants_strategy_adherence": False,
            "grants_strategy_completion": True,
            "completion_semantics": ("host_replayed_selected_step_with_key_event_critic_pass"),
            "mapped_edit_overlap": False,
            "checkpoint_critic_confirmed": True,
            "grants_route_admission": False,
        }
    ]


def test_key_event_prompt_audits_only_focus_step_not_route_completeness() -> None:
    prompt = sequential_module._critic_prompt(
        target="CCO",
        branch_index=0,
        strategy_card={
            "strategy_query": "Construct the scaffold by a key C-O cyclization.",
            "critical_assumption": "The cyclization is stereoselective.",
            "critic_checkpoint": "Audit the graph edit that forms the scaffold C-O bond.",
        },
        steps=[
            {
                "step_id": "step:key",
                "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                "mapped_precursor_smiles": ["[CH3:1][CH2:2].[OH2:3]"],
                "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                "transformation_hypothesis": "key C-O cyclization",
                "checkpoint_relation": "executes_checkpoint",
            }
        ],
        paper_matched=True,
        audit_kind="key_event",
        focus_step_id="step:key",
    )

    assert "audit only focus_step_id" in prompt
    assert "Do not reject them, demand a complete route" in prompt
    assert (
        "Return only checkpoint_match, verdict, blocking_type, repair_scope" in prompt
    )
    assert "checkpoint_match=false" in prompt
    assert "benign mislabeled preparatory move" in prompt
    assert "irreversibly cuts required topology" in prompt
    assert "directly tests critical_assumption" in prompt
    assert "inspect every chemically compatible reactive handle and site" in prompt
    assert "intramolecular pairings, ring sizes" in prompt
    assert "focus_step_topology is the Host's compact RDKit ring-path" in prompt
    assert "net structural/H/charge/redox plausibility" in prompt
    assert "stated conditions supply every required hydrogen transfer" in prompt
    assert "focus step's mapped product and every preceding row are immutable" in prompt
    assert "repair_scope=focus_edge" in prompt
    assert "repair_scope=route_span" in prompt
    assert "repair_scope=strategy_horizon" in prompt
    assert "changing the focus mapped product" in prompt
    assert "identifies the mutation owner independently of blocking_type" in prompt
    assert "extending a chemically coherent precursor farther upstream" in prompt
    assert "not executable as written: reject it" in prompt
    assert "suggested_revision would change the focus step's operations" in prompt
    assert "Reserve uncertain for missing evidence" in prompt
    assert "telescopes independent reactions" in prompt
    assert "must be an adjacent edge" in prompt
    assert "IMDA" not in prompt
    assert "RCM" not in prompt
    assert '"focus_step_id":"step:key"' in prompt
    context = json.loads(prompt.split("KeyEventCriticInput:\n", 1)[1])
    assert "checkpoint_relation" not in context["steps"][0]
    assert "conditions" not in context["steps"][0]
    assert "catalyst" not in context["steps"][0]
    assert context["focus_step_topology"] == {
        "step_id": "step:key",
        "product": {"ring_sizes": [], "ring_paths": []},
        "precursors": [{"precursor_index": 0, "ring_sizes": [], "ring_paths": []}],
    }


def test_key_event_failure_basin_is_derived_and_lineage_scoped() -> None:
    card = normalize_strategy_card(_strategy_card(1))
    other_card = normalize_strategy_card(_strategy_card(2))
    lineage_root = "[CH3:1][CH3:2]"
    steps = [
        {
            "step_id": "root:split",
            "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "mapped_precursor_smiles": [lineage_root, "[OH2:3]"],
        },
        {
            "step_id": "left:advance",
            "mapped_product_smiles": lineage_root,
            "mapped_precursor_smiles": ["[CH4:1]", "[CH4:2]"],
        },
    ]
    branch = {
        "strategy_milestone_cards": [card, other_card],
        "key_event_critic_history": [
            {
                "status": "rejected",
                "focus_step_id": "attempt:1",
                "fingerprint": "candidate-a",
                "strategy_digest": card["strategy_digest"],
                "strategy_milestone_index": 1,
                "lineage_root_mapped_smiles": lineage_root,
                "checkpoint_match": True,
                "assessment": {
                    "blocking": True,
                    "blocking_type": "mechanism",
                    "reasons": ["the first candidate lacks the initiating handle"],
                    "suggested_revision": "change the reactive topology",
                },
            },
            {
                "status": "rejected",
                "focus_step_id": "attempt:2",
                "fingerprint": "candidate-b",
                "strategy_digest": card["strategy_digest"],
                "strategy_milestone_index": 1,
                "lineage_root_mapped_smiles": lineage_root,
                "checkpoint_match": True,
                "assessment": {
                    "blocking": True,
                    "blocking_type": "mechanism",
                    "reasons": ["the second candidate still lacks a termination handle"],
                    "suggested_revision": "replace the route-defining event",
                },
            },
        ],
    }

    feedback = sequential_module._pending_key_event_feedback_for_leaf(
        branch,
        strategy_card=card,
        steps=steps,
        selected_product_mapped="[CH4:1]",
    )

    assert len(feedback["active_constraints"]) == 2
    assert feedback["failure_basin"] == {
        "distinct_rejected_attempt_count": 2,
        "blocking_type_counts": {"mechanism": 2},
        "recurring_blocking_types": ["mechanism"],
        "distinct_candidate_fingerprints": ["candidate-a", "candidate-b"],
        "checkpoint_match_count": 2,
        "recurrent_across_distinct_candidates": True,
        "authority": "derived_diagnostic_only",
    }
    assert (
        sequential_module._pending_key_event_feedback_for_leaf(
            branch,
            strategy_card=card,
            steps=steps,
            selected_product_mapped="[OH2:3]",
        )
        == {}
    )
    assert (
        sequential_module._pending_key_event_feedback_for_leaf(
            branch,
            strategy_card=other_card,
            steps=steps,
            selected_product_mapped="[CH4:1]",
        )
        == {}
    )

    critic_prompt = sequential_module._critic_prompt(
        target="CCO",
        branch_index=0,
        strategy_card=card,
        steps=[
            {
                "step_id": "attempt:3",
                "mapped_product_smiles": lineage_root,
                "mapped_precursor_smiles": ["[CH4:1]", "[CH4:2]"],
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 2}
                ],
            }
        ],
        paper_matched=True,
        audit_kind="key_event",
        focus_step_id="attempt:3",
        checkpoint_feedback=feedback,
    )
    critic_context = json.loads(critic_prompt.split("KeyEventCriticInput:\n", 1)[1])
    assert critic_context["failure_basin"] == feedback["failure_basin"]
    assert "do not mechanically request another focus edge" in critic_prompt
    assert "not from a fixed attempt count" in critic_prompt

    branch["key_event_critic_history"].append(
        {
            "status": "completed",
            "focus_step_id": "attempt:passed",
            "strategy_digest": card["strategy_digest"],
            "strategy_milestone_index": 1,
            "lineage_root_mapped_smiles": lineage_root,
            "required_selected_step_ids": ["attempt:passed"],
            "checkpoint_match": True,
            "assessment": {"verdict": "pass", "blocking": False},
        }
    )
    selected_steps = [
        *steps,
        {
            "step_id": "attempt:passed",
            "mapped_product_smiles": "[CH4:1]",
            "mapped_precursor_smiles": ["[CH4:1]"],
        },
    ]
    assert (
        sequential_module._pending_key_event_feedback_for_leaf(
            branch,
            strategy_card=card,
            steps=selected_steps,
            selected_product_mapped="[CH4:1]",
        )
        == {}
    )


def test_strategy_critic_preserves_unchallenged_generator_detail() -> None:
    prompt = sequential_module._paper_strategy_portfolio_critic_prompt(
        target="CCO",
        strategy_cards=[
            {
                "strategy_query": "Use RCM while masking the isopropenyl group.",
                "critical_assumption": "The tether geometry controls facial selectivity.",
                "critic_checkpoint": "Audit the first ring-closing graph edit.",
            },
            {
                "strategy_query": "Use a Pauson–Khand annulation after delayed alkene unveiling.",
                "critical_assumption": "The tether geometry controls facial selectivity.",
                "critic_checkpoint": "Audit the annulation graph edit.",
            },
            {
                "strategy_query": "Rearrange a cis-divinylcyclobutane with controlled precursor geometry.",
                "critical_assumption": "The precursor geometry controls stereospecificity.",
                "critic_checkpoint": "Audit the rearrangement graph edit.",
            },
        ],
    )

    assert "Copy every acceptable card verbatim" in prompt
    assert "preserve every unchallenged reactive-handle identity" in prompt
    assert "never paraphrase merely for brevity or style" in prompt
    assert "masking the isopropenyl group" in prompt
    assert "delayed alkene unveiling" in prompt
    assert "cis-divinylcyclobutane" in prompt


def test_paper_schema_preflight_rejects_editor_keyword_before_any_executor(
    monkeypatch,
) -> None:
    context = _context()
    original = codex_worker_module._worker_model_output_json_schema

    def schema_with_unsupported_editor_keyword(task: WorkerTask) -> dict:
        schema = original(task)
        if task.task_type == "paper_matched_route_editor":
            schema = json.loads(json.dumps(schema))
            schema["properties"]["replace_span"]["properties"]["remove_step_ids"]["uniqueItems"] = (
                True
            )
        return schema

    monkeypatch.setattr(
        codex_worker_module,
        "_worker_model_output_json_schema",
        schema_with_unsupported_editor_keyword,
    )
    executor_calls = 0

    def executor(task: WorkerTask) -> WorkerRunRecord:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError(f"executor must not run during preflight: {task.task_id}")

    runner = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        editor_executor=executor,
    )
    try:
        runner(
            _spec(context),
            context,
            "initial_architecture",
            DirectorConfig(
                paper_matched_reach_profile=True,
                allow_editor_route_mutations=True,
            ),
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("unsupported provider schema was not rejected")

    assert executor_calls == 0
    assert message == (
        "provider_response_schema_unsupported_keyword:"
        "paper_matched_route_editor/RetrosynthesisProposalReport:"
        "$.properties.replace_span.properties.remove_step_ids.uniqueItems:"
        "uniqueItems"
    )


def test_paper_editor_compacts_25_steps_without_losing_topology_or_replay() -> None:
    verbose = "non-structural mechanistic rationale " * 80
    steps = [
        {
            "step_id": f"route:{index}",
            "product_smiles": "C" * (26 - index),
            "precursor_smiles": ["C" * (25 - index), "O"],
            "reaction_family": "local handle installation",
            "transformation_rationale": verbose,
            "conditions": [verbose],
            "limitations": [verbose],
            "reaction_operations": [{"op": "break_bond", "map_a": index, "map_b": index + 1}],
            "strategy_anchor": index == 1,
        }
        for index in range(1, 26)
    ]
    prompt = _node_prompt(
        target="C" * 25,
        branch_index=0,
        lens="paper editor",
        selected_product="C" * 25,
        selected_product_mapped="[CH3:1]",
        steps=steps,
        open_leaves=("C", "O"),
        prior_rejections=(),
        repair=True,
        strategy_card=_strategy_card(1),
        forbidden_strategy_cards=(),
        host_failure_feedback={
            "blocking_step": {"step_id": "route:25"},
            "repair_actions": ["replace the local handle"],
        },
        complete_route_json=True,
        editor_route_mutations=True,
        paper_matched=True,
    )

    assert len(prompt.encode("utf-8")) <= 96_000
    context = json.loads(prompt.split("PaperMatchedRouteEditorContext:\n", 1)[1])
    assert context["schema_version"] == "paper_matched_route_editor_context.v4"
    assert len(context["route_json"]) == 25
    assert [row["step_id"] for row in context["route_json"]] == [
        f"route:{index}" for index in range(1, 26)
    ]
    assert all(row["product_smiles"] for row in context["route_json"])
    assert all(row["precursor_smiles"] for row in context["route_json"])
    assert all(row["reaction_operations"] for row in context["route_json"])
    assert all("mapped_product_smiles" in row for row in context["route_json"])
    assert all("conditions" in row for row in context["route_json"])
    assert all("parent_step_ids" in row for row in context["route_json"])
    assert "transformation_rationale" not in context["route_json"][0]
    assert "strategy_anchor" not in context["route_json"][0]
    assert context["repair_history"] == {}


def test_bounded_critic_prompt_keeps_all_route_structures_under_byte_cap() -> None:
    verbose = "mechanistic and selectivity discussion " * 80
    steps = [
        {
            "step_id": f"step:{index}",
            "product_smiles": "CC1=CC=CC=C1" + ".C" * (index % 3),
            "precursor_smiles": ["CC1CCCCC1", "O=C=O"],
            "transformation_hypothesis": verbose,
            "reaction_operations": [
                {"op": "change_bond_order", "map_a": 1, "map_b": 2, "delta": -1}
            ],
            "condition_predictions": [{"conditions": [verbose], "rationale": verbose}],
            "execution_domain": "whole_cell" if index == 7 else "chemical",
            "biocatalytic_step": {
                "mode": "whole_cell_transformation",
                "enzyme_classes": ["cytochrome P450 monooxygenase"],
                "selectivity_objective": verbose,
                "validation_plan": [verbose] * 4,
            }
            if index == 7
            else {},
            "strategy_anchor": index in {1, 8, 16},
            "strategy_milestone_index": 1 + (index // 8),
        }
        for index in range(1, 26)
    ]
    prompt = sequential_module._bounded_critic_prompt(
        target="CCO",
        branch_index=2,
        strategy_card={
            "execution_domain": "hybrid",
            "key_forward_transformation": verbose,
            "biocatalytic_intent": {
                "mode": "chemoenzymatic_cascade",
                "selectivity_objective": verbose,
                "validation_plan": [verbose] * 4,
            },
        },
        strategy_milestone_cards=[
            {"strategy_milestone_index": index, "key_forward_transformation": verbose}
            for index in range(1, 4)
        ],
        steps=steps,
        maximum_bytes=24_000,
    )

    assert prompt is not None
    assert len(prompt.encode("utf-8")) <= 24_000
    route = json.loads(prompt.split("BlindRouteCriticInput:\n", 1)[1])
    assert len(route["steps"]) == 25
    assert [row["step_id"] for row in route["steps"]] == [f"step:{index}" for index in range(1, 26)]
    assert all(row["product_smiles"] for row in route["steps"])
    assert all(row["precursor_smiles"] for row in route["steps"])
    assert all(row["reaction_operations"] for row in route["steps"])
    assert route["steps"][6]["execution_domain"] == "whole_cell"
    assert "Do not invent a hidden required stereoisomer" in prompt


def test_builder_and_key_critic_do_not_invent_unspecified_product_stereo() -> None:
    product = "CC(O)CC"
    mapped_product = "[CH3:1][CH:2]([OH:3])[CH2:4][CH3:5]"
    strategy = {
        "strategy_query": "Form the alcohol-bearing stereocenter selectively.",
        "critical_assumption": "Substrate control selects the desired alcohol face.",
        "critic_checkpoint": "Audit formation of the alcohol-bearing center.",
    }
    builder_prompt = _node_prompt(
        target=product,
        branch_index=0,
        lens="stereochemical control",
        selected_product=product,
        selected_product_mapped=mapped_product,
        steps=(),
        open_leaves=(product,),
        prior_rejections=(),
        repair=False,
        strategy_card=strategy,
        forbidden_strategy_cards=(),
        host_failure_feedback={},
        max_reactionjson_candidates=1,
        paper_matched=True,
    )
    assert "immutable Host product does not demand one R/S assignment" in builder_prompt
    builder_context = json.loads(
        builder_prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1]
    )
    assert builder_context["selected_leaf_stereo"]["unassigned_center_maps"] == [2]

    focus_step = {
        "step_id": "focus:1",
        "product_smiles": product,
        "mapped_product_smiles": mapped_product,
        "precursor_smiles": ["CC(=O)CC"],
        "mapped_precursor_smiles": ["[CH3:1][C:2](=[O:3])[CH2:4][CH3:5]"],
        "checkpoint_relation": "executes_checkpoint",
        "reaction_family": "stereoselective ketone reduction",
        "conditions": ["chiral reduction catalyst"],
        "reaction_operations": [
            {"op": "change_bond_order", "map_a": 2, "map_b": 3, "delta": 1}
        ],
    }
    critic_prompt = sequential_module._bounded_critic_prompt(
        target=product,
        branch_index=0,
        strategy_card=strategy,
        steps=[focus_step],
        maximum_bytes=96_000,
        paper_matched=True,
        audit_kind="key_event",
        focus_step_id="focus:1",
    )
    assert critic_prompt is not None
    assert "Do not invent a hidden required stereoisomer" in critic_prompt
    assert "never a focus_edge or route_span Builder obligation" in critic_prompt


def test_bounded_critic_prompt_returns_none_instead_of_raising_when_impossible() -> None:
    prompt = sequential_module._bounded_critic_prompt(
        target="CCO",
        branch_index=0,
        strategy_card={},
        steps=[
            {
                "step_id": "step:1",
                "product_smiles": "C" * 1_000,
                "precursor_smiles": ["C" * 1_000],
                "reaction_operations": [],
            }
        ],
        maximum_bytes=100,
    )

    assert prompt is None


def test_route_execution_profile_requires_both_materialized_domains_for_fusion() -> None:
    from cascade_planner.orchestration.sequential_strategy_director import (
        _route_execution_profile,
    )

    chemical_only = _route_execution_profile([{"execution_domain": "chemical"}])
    hybrid = _route_execution_profile(
        [
            {"execution_domain": "chemical"},
            {
                "execution_domain": "enzymatic",
                "enzyme": "ketoreductase",
                "biocatalytic_step": {"enzyme_class": "ketoreductase"},
            },
        ]
    )
    strategic_hybrid = _route_execution_profile(
        [
            {"execution_domain": "chemical", "strategy_anchor": True},
            {"execution_domain": "whole_cell", "enzyme": "P450"},
        ]
    )

    assert chemical_only["genuine_chemoenzymatic_fusion"] is False
    assert hybrid["genuine_chemoenzymatic_fusion"] is True
    assert hybrid["strategic_chemoenzymatic_fusion"] is False
    assert hybrid["step_execution_domains"] == ["chemical", "enzymatic"]
    assert strategic_hybrid["strategic_chemoenzymatic_fusion"] is True


def test_editor_route_mutation_rebuilds_stale_solved_or_tree() -> None:
    old_search = ChemEnzyReactionJsonOrSearch(
        target_smiles="CCO",
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        max_depth=6,
    )
    old_search.replay_route(
        (
            {
                "step_id": "old:1",
                "product_smiles": "CCO",
                "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                "precursor_smiles": ["C", "O"],
                "mapped_precursor_smiles": ["[CH4:1]", "[OH2:3]"],
            },
        ),
        stock_smiles={"C", "O"},
    )
    assert old_search.project().summary["root_solved"] is True

    branch = {
        "steps": [
            {
                "step_id": "edited:1",
                "product_smiles": "CCO",
                "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                "precursor_smiles": ["C", "CO"],
                "mapped_precursor_smiles": ["[CH4:1]", "[CH3:2][OH:3]"],
            }
        ],
        "target_mapped_smiles": "[CH3:1][CH2:2][OH:3]",
        "_reactionjson_or_search": old_search,
        "reactionjson_or_search": dict(old_search.project().summary),
    }
    runner = SequentialStrategyDirectorRunner(
        stock_membership=lambda values: {value: value in {"C", "O"} for value in values}
    )
    runner._rebuild_branch_or_search_after_editor(
        branch,
        target="CCO",
        max_depth=6,
    )

    assert branch["reactionjson_or_search"]["root_solved"] is False
    assert list(branch["open_leaves"]) == ["CO"]
    reset = branch["reactionjson_or_search_resets"][0]
    assert reset["previous_summary"]["root_solved"] is True
    assert reset["rebuilt_summary"]["root_solved"] is False


def _strategy_card(branch: int) -> dict:
    return {
        "strategy_query": f"Build branch-{branch} scaffold through its key construction.",
        "critical_assumption": (f"Branch {branch} key construction controls the target scaffold."),
        "critic_checkpoint": (
            f"Audit the graph transformation that executes branch {branch}'s key construction."
        ),
        "scaffold_motif": f"branch-{branch}-scaffold",
        "key_forward_transformation": f"branch-{branch}-key-construction",
        # The strategy contract now requires key bonds to be real bonds in the
        # immutable campaign target (the fixture target is mapped C-C-O).
        "key_bond_changes": ["map 1-map 2"],
        "functional_group_conflicts": [],
        "protection_policy": "avoid unless required for chemoselectivity",
        "stereochemical_plan": "preserve or construct assigned stereochemistry",
        "convergence_plan": "join independently accessible fragments",
        "strategic_step_count": 1,
        "skeleton_change_class": f"branch-{branch}-class",
        "expected_complexity_drop": "high",
        "orthogonality_basis": f"branch-{branch}-orthogonal",
        "strategy_signature": f"branch-{branch}-signature",
        "execution_domain": "chemical",
    }


def _strategy_record(task) -> WorkerRunRecord:
    branch = int(re.search(r'"branch_id":(\d+)', task.objective).group(1))
    card = _strategy_card(branch)
    artifact = {
        "schema_version": "strategy_card_report_artifact.v1",
        "artifact_id": task.task_id,
        "artifact_type": "StrategyCardReport",
        "case_id": task.case_id,
        "source": "test",
        "input_refs": task.input_refs,
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": "strategy only",
        "payload": {
            "schema_version": "strategy_card_report.v1",
            "case_id": task.case_id,
            "target_smiles": "CCO",
            "strategy_card": card,
            "alternatives_considered": [
                {
                    "candidate_label": f"branch-{branch}-option-{index}",
                    "key_forward_transformation": (
                        card["key_forward_transformation"]
                        if index == 1
                        else f"branch-{branch}-alternative-{index}"
                    ),
                    "key_bond_changes": list(card["key_bond_changes"]),
                    "advantages": ["structural simplification"],
                    "risks": ["host validation required"],
                    "decision": "selected" if index == 1 else "rejected",
                }
                for index in range(1, 4)
            ],
            "selection_rationale": "best route-defining construction",
            "limitations": ["hypothesis only"],
            "no_route_or_solved_claim": True,
        },
    }
    return WorkerRunRecord(
        run_id=f"{task.task_id}:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="accepted_draft",
        output_artifact=artifact,
        output_validation={"accepted": True, "reasons": []},
        usage={"input_tokens": 200, "output_tokens": 100},
    )


def _strategy_portfolio_record(task) -> WorkerRunRecord:
    cards = [
        {
            "strategy_query": _strategy_card(branch)["strategy_query"],
            "critical_assumption": (
                f"Branch {branch} key construction controls the target scaffold."
            ),
            "critic_checkpoint": (
                f"Audit the graph transformation that executes branch {branch}'s key construction."
            ),
        }
        for branch in range(1, 4)
    ]
    return WorkerRunRecord(
        run_id=f"{task.task_id}:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="accepted_draft",
        output_artifact={
            "schema_version": "strategy_portfolio_report_artifact.v1",
            "artifact_id": task.task_id,
            "artifact_type": "StrategyPortfolioReport",
            "case_id": task.case_id,
            "source": "test",
            "input_refs": task.input_refs,
            "evidence_refs": [],
            "validation_status": "draft",
            "summary": "three strategies in one call",
            "payload": {
                "schema_version": "strategy_portfolio_report.v1",
                "case_id": task.case_id,
                "target_smiles": "CCO",
                "strategy_cards": cards,
                "no_route_or_solved_claim": True,
            },
        },
        output_validation={"accepted": True, "reasons": []},
        usage={"input_tokens": 400, "output_tokens": 300},
    )


def _fake_executor(task):
    if task.required_artifact_type == "StrategyPortfolioReport":
        return _strategy_portfolio_record(task)
    if task.required_artifact_type == "StrategyCardReport":
        return _strategy_record(task)
    match = re.search(r'"selected_open_leaf":"([^"]+)"', task.objective)
    assert match
    product = match.group(1)
    branch = int(re.search(r'"branch_id":(\d+)', task.objective).group(1))
    roots = {
        1: ["CC", "O"],
        2: ["CO", "C"],
        3: ["C=C", "O"],
    }
    continuations = {
        "CC": ["C=C"],
        "O": ["[OH-]"],
        "CO": ["C=O"],
        "C": ["[CH3]"],
        "C=C": ["C#C"],
    }
    precursors = roots[branch] if product == "CCO" else continuations[product]
    candidate = {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": task.task_id,
        "product_smiles": product,
        "precursor_smiles": precursors,
        "reaction_family": f"branch-{branch}-reaction",
        "product_retron_type": "local",
        "transformation_rationale": "one node only",
        "source_channel": "codex_strategy",
        "source_refs": [],
        "evidence_refs": [],
        "evidence_level": "model_only",
        "confidence": "medium",
        "conditions": ["screen"],
        "catalyst": "",
        "enzyme": "",
        "limitations": ["host validation required"],
        "required_validation": ["structure"],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "strategy_card": _strategy_card(branch),
        "reaction_operations": [],
    }
    artifact = {
        "schema_version": "retrosynthesis_proposal_report_artifact.v1",
        "artifact_id": task.task_id,
        "artifact_type": "RetrosynthesisProposalReport",
        "case_id": task.case_id,
        "source": "test",
        "input_refs": task.input_refs,
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": "one node",
        "payload": {
            "schema_version": "retrosynthesis_proposal_report.v1",
            "case_id": task.case_id,
            "agent_role": "sequential policy",
            "target_smiles": product,
            "candidates": [candidate],
            "evidence_refs": [],
            "limitations": [],
            "no_solved_claim": True,
        },
    }
    return WorkerRunRecord(
        run_id=f"{task.task_id}:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="accepted_draft",
        output_artifact=artifact,
        output_validation={"accepted": True, "reasons": []},
        usage={"input_tokens": 500, "output_tokens": 100},
    )


def _critic_record(task, *, assessment: str = "viable") -> WorkerRunRecord:
    payload = {
        "schema_version": "chemical_strategy_critique.v1",
        "case_id": task.case_id,
        "strategy_id": "strategy:test",
        "strategy_digest": "d" * 64,
        "route_family_id": "family:test",
        "overall_assessment": assessment,
        "strategy_adherence": True,
        "step_assessments": [
            {
                "step_id": "root",
                "mechanistic_analysis": "plausible bond construction",
                "atom_provenance": "covered",
                "functional_group_compatibility": "screenable",
                "chemoselectivity": "substrate controlled",
                "stereochemistry": "preserved",
                "sequence_ordering": "reasonable",
                "competing_pathways": [],
                "enzyme_assessment": "not applicable",
                "verdict": "pass" if assessment == "viable" else "uncertain",
                "reasons": [],
            }
        ],
        "route_level_risks": [],
        "repair_actions": [],
        "experimental_variables": ["substrate scope"],
        "limitations": [],
        "no_reaction_proof": True,
        "no_source_authority": True,
        "no_solved_claim": True,
    }
    return WorkerRunRecord(
        run_id=f"{task.task_id}:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="accepted_draft",
        output_artifact={
            "schema_version": "chemical_strategy_critique_artifact.v1",
            "artifact_id": task.task_id,
            "artifact_type": "ChemicalStrategyCritique",
            "case_id": task.case_id,
            "source": "test",
            "input_refs": [],
            "evidence_refs": [],
            "validation_status": "draft",
            "summary": "independent forward critique",
            "payload": payload,
        },
        output_validation={"accepted": True, "reasons": []},
        usage={"input_tokens": 300, "output_tokens": 100},
    )


def _blocking_critic_record(task, *, step_id: str) -> WorkerRunRecord:
    record = _critic_record(task, assessment="reject")
    artifact = dict(record.output_artifact or {})
    payload = dict(artifact.get("payload") or {})
    assessments = [dict(row) for row in payload.get("step_assessments") or []]
    assessments[0]["step_id"] = step_id
    assessments[0]["verdict"] = "reject"
    assessments[0]["reasons"] = ["blocking functional-group incompatibility"]
    payload["step_assessments"] = assessments
    artifact["payload"] = payload
    return replace(record, output_artifact=artifact)


def _fake_critic_executor(task):
    return _critic_record(task)


def _proposal_record(candidate: dict, *, target: str = "CCO") -> WorkerRunRecord:
    artifact = {
        "schema_version": "retrosynthesis_proposal_report_artifact.v1",
        "artifact_id": "proposal:test",
        "artifact_type": "RetrosynthesisProposalReport",
        "case_id": "strategy-case:test",
        "source": "test",
        "input_refs": [],
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": "test route",
        "payload": {
            "schema_version": "retrosynthesis_proposal_report.v1",
            "case_id": "strategy-case:test",
            "agent_role": "test",
            "target_smiles": target,
            "candidates": [candidate],
            "evidence_refs": [],
            "limitations": [],
            "no_solved_claim": True,
        },
    }
    return WorkerRunRecord(
        run_id="proposal:test:run",
        task_id="proposal:test",
        case_id="strategy-case:test",
        status="accepted_draft",
        output_artifact=artifact,
        output_validation={"accepted": True, "reasons": []},
        usage={"input_tokens": 1, "output_tokens": 1},
    )


def _complete_route_candidate() -> dict:
    return {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": "route:test",
        "product_smiles": "CCO",
        "precursor_smiles": [],
        "reaction_family": "fragmentation",
        "product_retron_type": "C-O",
        "transformation_rationale": "split the terminal C-O bond",
        "conditions": [],
        "catalyst": "",
        "enzyme": "",
        "limitations": [],
        "required_validation": ["structure"],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        "route_json": [
            {
                "step_id": "route:1",
                "product_smiles": "CCO",
                "precursor_smiles": [],
                "reaction_family": "fragmentation",
                "product_retron_type": "C-O",
                "transformation_rationale": "split the terminal C-O bond",
                "conditions": [],
                "catalyst": "",
                "enzyme": "",
                "limitations": [],
                "required_validation": ["structure"],
                "no_solved_claim": True,
                "not_parent_route_proof": True,
                "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
            },
            {
                "step_id": "route:2",
                "product_smiles": "CC",
                "precursor_smiles": [],
                "reaction_family": "fragmentation",
                "product_retron_type": "C-C",
                "transformation_rationale": "split the ethyl fragment",
                "conditions": ["aqueous workup"],
                "catalyst": "",
                "enzyme": "",
                "limitations": [],
                "required_validation": ["structure"],
                "no_solved_claim": True,
                "not_parent_route_proof": True,
                "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
            },
        ],
    }


def test_complete_route_json_replays_as_a_contiguous_linear_chain() -> None:
    expansions = _expansions_from_record(
        _proposal_record(_complete_route_candidate()),
        expected_product="CCO",
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        require_reaction_operations=True,
        require_complete_route_json=True,
    )
    assert expansions is not None
    assert [row.product_smiles for row in expansions] == ["CCO", "CC"]
    assert expansions[-1].precursor_smiles == ("C", "C")
    assert expansions[1].conditions == ("aqueous workup",)


def test_step_role_survives_replay_but_never_drives_strategy_completion() -> None:
    candidate = _complete_route_candidate()
    candidate["route_json"][0]["step_role"] = "enabling"
    candidate["route_json"][1]["step_role"] = "key"
    expansions = _expansions_from_record(
        _proposal_record(candidate),
        expected_product="CCO",
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        require_reaction_operations=True,
        require_complete_route_json=True,
    )

    assert expansions is not None
    assert [row.step_role for row in expansions] == ["enabling", "key"]
    paper_card = {
        "strategy_query": "Construct the key scaffold through cycloaddition.",
        "strategy_signature": "key-cycloaddition",
    }
    assert (
        sequential_module._expansion_executes_strategy_anchor(
            expansions[0], paper_card, fallback=True
        )
        is False
    )
    assert (
        sequential_module._expansion_executes_strategy_anchor(
            expansions[1], paper_card, fallback=False
        )
        is False
    )
    row = sequential_module._step_row(
        replace(expansions[0], strategy_card=paper_card),
        step_id="route:1",
        strategy_anchor=False,
    )
    assert row["step_role"] == "enabling"
    corrupted_legacy_first_row = {
        **row,
        "step_role": "",
        "strategy_anchor": True,
    }
    assert (
        sequential_module._strategy_anchor_fulfilled_for_card(
            [corrupted_legacy_first_row], paper_card
        )
        is False
    )
    key_row = sequential_module._step_row(
        replace(expansions[1], strategy_card=paper_card),
        step_id="route:2",
        strategy_anchor=True,
    )
    assert (
        sequential_module._strategy_anchor_fulfilled_for_card([row, key_row], paper_card) is False
    )
    assert (
        sequential_module._strategy_anchor_progress([row, key_row], paper_card)["fulfilled"]
        is False
    )


def test_rejected_worker_envelope_with_safe_routejson_still_reaches_host_compiler() -> None:
    record = replace(
        _proposal_record(_complete_route_candidate()),
        status="rejected_output",
        output_validation={
            "accepted": False,
            "reasons": ["worker_exit_code_nonzero", "missing_evidence_or_input_refs"],
        },
    )

    expansions = _expansions_from_record(
        record,
        expected_product="CCO",
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        require_reaction_operations=True,
        require_complete_route_json=True,
    )

    assert expansions is not None
    assert [row.product_smiles for row in expansions] == ["CCO", "CC"]


def test_runtime_failure_without_safe_routejson_remains_fail_closed() -> None:
    record = replace(
        _proposal_record(_complete_route_candidate()),
        status="timeout",
        output_artifact=None,
        output_validation={"accepted": False, "reasons": ["timeout"]},
    )

    assert (
        _expansions_from_record(
            record,
            expected_product="CCO",
            mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
            require_reaction_operations=True,
            require_complete_route_json=True,
        )
        is None
    )


def test_complete_route_json_preserves_replay_atom_maps_across_steps() -> None:
    candidate = _complete_route_candidate()
    candidate["route_json"][0]["reaction_operations"] = [
        {"op": "break_bond", "map_a": 20, "map_b": 30}
    ]
    candidate["route_json"][1]["reaction_operations"] = [
        {"op": "break_bond", "map_a": 10, "map_b": 20}
    ]
    expansions = _expansions_from_record(
        _proposal_record(candidate),
        expected_product="CCO",
        mapped_product_smiles="[CH3:10][CH2:20][OH:30]",
        require_reaction_operations=True,
        require_complete_route_json=True,
    )
    assert expansions is not None
    assert expansions[0].precursor_smiles == ("CC", "O")
    assert expansions[1].precursor_smiles == ("C", "C")


def test_complete_route_json_rejects_terminal_leaf_pseudo_step() -> None:
    candidate = _complete_route_candidate()
    candidate["route_json"].append(
        {
            "step_id": "terminal-leaf",
            "product_smiles": "C",
            "precursor_smiles": [],
            "reaction_operations": [],
        }
    )
    expansions = _expansions_from_record(
        _proposal_record(candidate),
        expected_product="CCO",
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        require_reaction_operations=True,
        require_complete_route_json=True,
    )
    assert expansions is None


def test_complete_route_json_allows_one_step_stock_closure_from_target_root() -> None:
    candidate = _complete_route_candidate()
    candidate["route_json"] = [candidate["route_json"][0]]
    expansions = _expansions_from_record(
        _proposal_record(candidate),
        expected_product="CCO",
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        require_reaction_operations=True,
        require_complete_route_json=True,
    )
    assert expansions is not None
    assert len(expansions) == 1


def test_complete_route_json_rejects_internal_cycle() -> None:
    candidate = _complete_route_candidate()
    candidate["route_json"][1]["reaction_operations"] = [
        {"op": "add_group", "map_idx": 2, "fragment_smiles": "[*][OH:3]"}
    ]
    expansions = _expansions_from_record(
        _proposal_record(candidate),
        expected_product="CCO",
        require_reaction_operations=True,
        require_complete_route_json=True,
    )
    assert expansions is None


def _three_step_editor_route() -> list[dict]:
    return [
        {
            "step_id": "route:1",
            "product_smiles": "CCCO",
            "reaction_family": "C-O disconnection",
            "conditions": [],
            "catalyst": "",
            "reaction_operations": [{"op": "break_bond", "map_a": 3, "map_b": 4}],
        },
        {
            "step_id": "route:2",
            "product_smiles": "CCC",
            "reaction_family": "C-C disconnection",
            "conditions": [],
            "catalyst": "",
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        },
        {
            "step_id": "route:3",
            "product_smiles": "CC",
            "reaction_family": "C-C disconnection",
            "conditions": [],
            "catalyst": "",
            "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
        },
    ]


def _branching_path_repair_route() -> list[dict]:
    return [
        {
            "step_id": "route:1",
            "product_smiles": "CCOC",
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        },
        {
            "step_id": "route:2",
            "product_smiles": "CC",
            "mapped_product_smiles": "[CH3:1][CH3:2]",
            "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
        },
        {
            "step_id": "route:3",
            "product_smiles": "C",
            "mapped_product_smiles": "[CH4:1]",
            "reaction_operations": [
                {
                    "op": "add_group",
                    "map_idx": 1,
                    "fragment_smiles": "[*][Cl:5]",
                }
            ],
        },
        {
            "step_id": "route:4",
            "product_smiles": "CO",
            "mapped_product_smiles": "[OH:3][CH3:4]",
            "reaction_operations": [{"op": "break_bond", "map_a": 3, "map_b": 4}],
        },
    ]


def _linear_path_repair_route_with_suffix() -> list[dict]:
    return [
        {
            "step_id": "linear:1",
            "product_smiles": "CCO",
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        },
        {
            "step_id": "linear:2",
            "product_smiles": "CC",
            "reaction_operations": [
                {
                    "op": "add_group",
                    "map_idx": 2,
                    "fragment_smiles": "[*][Cl:4]",
                }
            ],
        },
        {
            "step_id": "linear:3",
            "product_smiles": "CCCl",
            "reaction_operations": [
                {
                    "op": "add_group",
                    "map_idx": 1,
                    "fragment_smiles": "[*][Br:5]",
                }
            ],
        },
        {
            "step_id": "linear:4",
            "product_smiles": "ClCCBr",
            "reaction_operations": [
                {
                    "op": "add_group",
                    "map_idx": 1,
                    "fragment_smiles": "[*][F:6]",
                }
            ],
        },
    ]


def _branched_path_repair_span_with_suffix() -> list[dict]:
    return [
        *_branching_path_repair_route(),
        {
            "step_id": "route:5",
            "product_smiles": "CCl",
            "mapped_product_smiles": "[CH3:1][Cl:5]",
            "reaction_operations": [
                {
                    "op": "add_group",
                    "map_idx": 1,
                    "fragment_smiles": "[*][F:6]",
                }
            ],
        },
    ]


def test_path_repair_span_removes_only_dependency_subtree() -> None:
    rollback, diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=_branching_path_repair_route(),
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        directive={
            "rollback_start_step_id": "route:2",
            "rebuild_through_step_id": "route:3",
            "repair_goal": "replace the blocked left-branch chemistry",
            "active_constraints": ["preserve the independent CO branch"],
        },
        blocking_step_ids=["route:3"],
    )

    assert diagnostic == {}
    assert rollback is not None
    assert [row["step_id"] for row in rollback.durable_steps] == [
        "route:1",
        "route:4",
    ]
    assert [row["step_id"] for row in rollback.removed_steps] == [
        "route:2",
        "route:3",
    ]
    assert rollback.repair_frontier_product_smiles == "CC"
    assert rollback.repair_frontier_mapped_product_smiles == "[CH3:1][CH3:2]"
    assert any(row["mapped_smiles"] == "[CH3:1][CH3:2]" for row in rollback.open_leaf_states)
    assert 5 not in rollback.reserved_atom_maps
    assert {1, 2, 3, 4}.issubset(rollback.reserved_atom_maps)


def test_path_repair_span_ignores_interleaved_independent_sibling() -> None:
    rows = _branching_path_repair_route()
    interleaved = [rows[0], rows[1], rows[3], rows[2]]

    rollback, diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=interleaved,
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        directive={
            "rollback_start_step_id": "route:2",
            "rebuild_through_step_id": "route:3",
            "repair_goal": "replace the blocked left-branch chemistry",
            "active_constraints": ["preserve the independent CO branch"],
        },
        blocking_step_ids=["route:3"],
    )

    assert diagnostic == {}
    assert rollback is not None
    assert [row["step_id"] for row in rollback.removed_steps] == [
        "route:2",
        "route:3",
    ]
    assert [row["step_id"] for row in rollback.durable_steps] == [
        "route:1",
        "route:4",
    ]


def test_path_repair_selects_sibling_blockers_as_separate_transactions() -> None:
    scope, diagnostic = sequential_module._select_path_repair_blocker_scope(
        current_steps=_branching_path_repair_route(),
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        blocking_steps=[
            {"step_id": "route:3"},
            {"step_id": "route:4"},
        ],
    )

    assert diagnostic == {}
    assert scope is not None
    assert scope.component_step_ids == (("route:3",), ("route:4",))
    assert scope.selected_step_ids == ("route:3",)
    assert scope.deferred_step_ids == ("route:4",)


def test_path_repair_joins_critic_declared_chemical_sibling_dependency() -> None:
    route = _branching_path_repair_route()
    blockers = sequential_module._blocking_critic_steps(
        {
            "step_assessments": [
                {
                    "step_id": "route:3",
                    "verdict": "reject",
                    "blocking": True,
                },
                {
                    "step_id": "route:4",
                    "verdict": "reject",
                    "blocking": True,
                },
            ],
            "coupled_blocker_groups": [["route:3", "route:4"]],
        },
        route,
    )
    scope, diagnostic = sequential_module._select_path_repair_blocker_scope(
        current_steps=route,
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        blocking_steps=blockers,
    )

    assert diagnostic == {}
    assert scope is not None
    assert scope.component_step_ids == (("route:3", "route:4"),)
    assert scope.selected_step_ids == ("route:3", "route:4")
    assert scope.deferred_step_ids == ()


def test_path_repair_span_does_not_consume_deferred_sibling_blocker() -> None:
    rollback, diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=_branching_path_repair_route(),
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        directive={
            "rollback_start_step_id": "route:1",
            "rebuild_through_step_id": "route:4",
            "repair_goal": "replace both branches despite local scope",
            "active_constraints": [],
        },
        blocking_step_ids=["route:3"],
        deferred_blocking_step_ids=["route:4"],
    )

    assert rollback is None
    assert diagnostic["reason"] == ("path_repair_span_crosses_deferred_blocker_component")


def test_path_repair_boundary_preflight_stops_incompatible_suffix_before_builder() -> None:
    rollback, diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=_linear_path_repair_route_with_suffix(),
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        directive={
            "rollback_start_step_id": "linear:2",
            "rebuild_through_step_id": "linear:3",
            "repair_goal": "change the state required by the retained suffix",
            "active_constraints": [],
        },
        blocking_step_ids=["linear:3"],
    )

    assert diagnostic == {}
    assert rollback is not None
    preflight = sequential_module._path_repair_boundary_preflight(
        rollback,
        preserved_suffix_compatible=False,
    )
    assert preflight["reason"] == ("path_repair_preserved_suffix_declared_incompatible")
    assert preflight["builder_calls_avoided"] is True


def test_path_repair_recritic_accepts_only_predeclared_deferred_blockers() -> None:
    pending = {
        "selected_blocker_step_ids": ["route:3"],
        "deferred_blocker_step_ids": ["route:4"],
    }

    resolved, accepted = sequential_module._path_repair_component_recritic_result(
        pending,
        [{"step_id": "route:4"}],
    )
    rejected, diagnostic = sequential_module._path_repair_component_recritic_result(
        pending,
        [{"step_id": "repair:1"}, {"step_id": "route:4"}],
    )

    assert resolved is True
    assert accepted["unexpected_blocker_step_ids"] == []
    assert rejected is False
    assert diagnostic["unexpected_blocker_step_ids"] == ["repair:1"]


def test_path_repair_span_can_include_sibling_rows_and_preserve_later_suffix() -> None:
    rollback, diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=_branched_path_repair_span_with_suffix(),
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        directive={
            "rollback_start_step_id": "route:1",
            "rebuild_through_step_id": "route:4",
            "repair_goal": "reorder chemistry across both precursor branches",
            "active_constraints": [],
        },
        blocking_step_ids=["route:1"],
    )

    assert diagnostic == {}
    assert rollback is not None
    assert [row["step_id"] for row in rollback.removed_steps] == [
        "route:1",
        "route:2",
        "route:3",
        "route:4",
    ]
    assert [row["step_id"] for row in rollback.preserved_suffix_steps] == ["route:5"]
    assert rollback.reconnect_boundaries[0]["step_id"] == "route:5"


def test_path_repair_span_rejects_missing_start_and_end_ids() -> None:
    rows = _branching_path_repair_route()
    missing_start, start_diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=rows,
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        directive={
            "rollback_start_step_id": "missing:start",
            "rebuild_through_step_id": "route:3",
            "repair_goal": "repair",
            "active_constraints": [],
        },
        blocking_step_ids=["route:3"],
    )
    missing_end, end_diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=rows,
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        directive={
            "rollback_start_step_id": "route:2",
            "rebuild_through_step_id": "missing:end",
            "repair_goal": "repair",
            "active_constraints": [],
        },
        blocking_step_ids=["route:3"],
    )

    assert missing_start is None
    assert start_diagnostic["reason"] == ("path_repair_rollback_start_step_not_found")
    assert missing_end is None
    assert end_diagnostic["reason"] == ("path_repair_rebuild_through_step_not_found")


def test_path_repair_span_rejects_reverse_or_unrelated_boundaries() -> None:
    rows = _branching_path_repair_route()
    reverse, reverse_diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=rows,
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        directive={
            "rollback_start_step_id": "route:3",
            "rebuild_through_step_id": "route:2",
            "repair_goal": "repair",
            "active_constraints": [],
        },
        blocking_step_ids=["route:3"],
    )
    unrelated, unrelated_diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=rows,
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        directive={
            "rollback_start_step_id": "route:2",
            "rebuild_through_step_id": "route:4",
            "repair_goal": "repair",
            "active_constraints": [],
        },
        blocking_step_ids=["route:3"],
    )

    assert reverse is None
    assert reverse_diagnostic["reason"] == ("path_repair_span_direction_or_dependency_invalid")
    assert unrelated is None
    assert unrelated_diagnostic["reason"] == ("path_repair_span_direction_or_dependency_invalid")


def test_path_repair_removes_only_declared_span_and_preserves_suffix() -> None:
    rollback, diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=_linear_path_repair_route_with_suffix(),
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        directive={
            "rollback_start_step_id": "linear:2",
            "rebuild_through_step_id": "linear:3",
            "repair_goal": "repair the two-step halogen relay",
            "active_constraints": [],
        },
        blocking_step_ids=["linear:3"],
    )

    assert diagnostic == {}
    assert rollback is not None
    assert [row["step_id"] for row in rollback.durable_steps] == ["linear:1"]
    assert [row["step_id"] for row in rollback.removed_steps] == [
        "linear:2",
        "linear:3",
    ]
    assert [row["step_id"] for row in rollback.preserved_suffix_steps] == ["linear:4"]
    assert rollback.reconnect_boundaries == (
        {
            "step_id": "linear:4",
            "product_smiles": "ClCCBr",
            "mapped_product_smiles": "[CH2:1]([CH2:2][Cl:4])[Br:5]",
        },
    )
    assert 6 in rollback.reserved_atom_maps
    assert 4 not in rollback.reserved_atom_maps
    assert 5 not in rollback.reserved_atom_maps


def test_path_repair_stitches_suffix_after_deterministic_boundary_map_remap() -> None:
    rollback, diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=_linear_path_repair_route_with_suffix(),
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        directive={
            "rollback_start_step_id": "linear:2",
            "rebuild_through_step_id": "linear:3",
            "repair_goal": "repair the two-step halogen relay",
            "active_constraints": [],
        },
        blocking_step_ids=["linear:3"],
    )
    assert diagnostic == {}
    assert rollback is not None
    rebuilt = [
        *rollback.durable_steps,
        {
            "step_id": "repair:1",
            "product_smiles": "CC",
            "mapped_product_smiles": "[CH3:1][CH3:2]",
            "reaction_operations": [
                {
                    "op": "add_group",
                    "map_idx": 2,
                    "fragment_smiles": "[*][Cl:7]",
                }
            ],
        },
        {
            "step_id": "repair:2",
            "product_smiles": "CCCl",
            "mapped_product_smiles": "[CH3:1][CH2:2][Cl:7]",
            "reaction_operations": [
                {
                    "op": "add_group",
                    "map_idx": 1,
                    "fragment_smiles": "[*][Br:8]",
                }
            ],
        },
    ]

    stitched, stitch_diagnostic = sequential_module._stitch_path_repair_suffix(
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        rebuilt_steps=rebuilt,
        preserved_suffix_steps=rollback.preserved_suffix_steps,
        reconnect_boundaries=rollback.reconnect_boundaries,
    )

    assert stitch_diagnostic == {
        "suffix_stitched": True,
        "boundary_count": 1,
        "preserved_suffix_step_count": 1,
        "remapped_boundary_atom_count": 2,
    }
    assert stitched is not None
    assert [row["step_id"] for row in stitched] == [
        "linear:1",
        "repair:1",
        "repair:2",
        "linear:4",
    ]
    assert stitched[-1]["mapped_product_smiles"] == ("[CH2:1]([CH2:2][Cl:7])[Br:8]")


def test_path_repair_suffix_replays_host_materialized_add_group_maps() -> None:
    rebuilt = [
        {
            "step_id": "durable:1",
            "product_smiles": "CBr",
            "reaction_operations": [
                {"op": "remove_group", "map_indices": [2]},
                {
                    "op": "add_group",
                    "map_idx": 1,
                    "fragment_smiles": "[*][OH:3]",
                },
            ],
        }
    ]
    suffix = [
        {
            "step_id": "suffix:1",
            "product_smiles": "CO",
            "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 3}],
        }
    ]

    stitched, diagnostic = sequential_module._stitch_path_repair_suffix(
        mapped_target_smiles="[CH3:1][Br:2]",
        rebuilt_steps=rebuilt,
        preserved_suffix_steps=suffix,
        reconnect_boundaries=[
            {
                "step_id": "suffix:1",
                "product_smiles": "CO",
                "mapped_product_smiles": "[CH3:1][OH:3]",
            }
        ],
    )

    assert diagnostic["suffix_stitched"] is True
    assert stitched is not None
    assert [row["step_id"] for row in stitched] == ["durable:1", "suffix:1"]


def test_path_repair_boundary_mapping_pins_shared_provenance_through_symmetry() -> None:
    translation = sequential_module._deterministic_boundary_atom_map_translation(
        "[Cl:4][C:1]([CH3:2])([CH3:3])[OH:5]",
        "[Cl:4][C:1]([CH3:2])([CH3:3])[OH:6]",
    )

    assert translation == {1: 1, 2: 2, 3: 3, 4: 4, 5: 6}


def test_path_repair_chooses_stable_symmetric_boundary_remap() -> None:
    translations = [
        sequential_module._deterministic_boundary_atom_map_translation(
            "[CH3:1][CH3:2]",
            "[CH3:7][CH3:8]",
        )
        for _ in range(5)
    ]

    assert translations == [{1: 7, 2: 8}] * 5


def test_path_repair_stitches_one_symmetric_molecular_occurrence() -> None:
    stitched, diagnostic = sequential_module._stitch_path_repair_suffix(
        mapped_target_smiles="[CH3:1][CH2:2][Cl:3]",
        rebuilt_steps=[
            {
                "step_id": "repair:1",
                "product_smiles": "CCCl",
                "reaction_operations": [{"op": "remove_group", "map_indices": [3]}],
            }
        ],
        preserved_suffix_steps=[
            {
                "step_id": "suffix:1",
                "product_smiles": "CC",
                "mapped_product_smiles": "[CH3:7][CH3:8]",
                "reaction_operations": [{"op": "break_bond", "map_a": 7, "map_b": 8}],
            }
        ],
        reconnect_boundaries=[
            {
                "step_id": "suffix:1",
                "product_smiles": "CC",
                "mapped_product_smiles": "[CH3:7][CH3:8]",
            }
        ],
    )

    assert diagnostic == {
        "suffix_stitched": True,
        "boundary_count": 1,
        "preserved_suffix_step_count": 1,
        "remapped_boundary_atom_count": 2,
    }
    assert stitched is not None
    assert [row["step_id"] for row in stitched] == ["repair:1", "suffix:1"]


def test_path_repair_refuses_single_occurrence_with_wrong_stereo() -> None:
    stitched, diagnostic = sequential_module._stitch_path_repair_suffix(
        mapped_target_smiles=("[CH3:1][C@H:2]([OH:3])[CH2:4][CH2:5][Cl:6]"),
        rebuilt_steps=[
            {
                "step_id": "repair:wrong-stereo",
                "product_smiles": "C[C@H](O)CCCl",
                "reaction_operations": [{"op": "remove_group", "map_indices": [6]}],
            }
        ],
        preserved_suffix_steps=[
            {
                "step_id": "suffix:1",
                "product_smiles": "CC[C@@H](C)O",
                "mapped_product_smiles": ("[CH3:7][C@@H:8]([OH:9])[CH2:10][CH3:11]"),
                "reaction_operations": [{"op": "break_bond", "map_a": 10, "map_b": 11}],
            }
        ],
        reconnect_boundaries=[
            {
                "step_id": "suffix:1",
                "product_smiles": "CC[C@@H](C)O",
                "mapped_product_smiles": ("[CH3:7][C@@H:8]([OH:9])[CH2:10][CH3:11]"),
            }
        ],
    )

    assert stitched is None
    assert diagnostic == {
        "reason": "path_repair_reconnect_boundary_stereo_mismatch",
        "boundary_step_id": "suffix:1",
        "boundary_product_smiles": "CC[C@@H](C)O",
        "candidate_count": 1,
        "stereo_mismatch_atom_maps": [8],
        "stereo_mismatch_bond_maps": [],
    }


def test_path_repair_retries_wrong_stereo_boundary_at_builder_parent(
    monkeypatch,
) -> None:
    builder_contexts: list[dict] = []

    def builder_executor(task: WorkerTask) -> WorkerRunRecord:
        context = json.loads(task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
        builder_contexts.append(context)
        operations = [{"op": "remove_group", "map_indices": [6]}]
        if len(builder_contexts) == 2:
            operations.append({"op": "invert_stereocenter", "map_idx": 2})
        product = str(task.host_context.get("selected_product") or "")
        return replace(
            _proposal_record(
                {
                    "candidate_id": task.task_id,
                    "product_smiles": product,
                    "precursor_smiles": [],
                    "checkpoint_relation": "preparatory",
                    "reaction_family": "stereo boundary retry canary",
                    "conditions": ["test conditions"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "reaction_operations": operations,
                },
                target=product,
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        request = {
            "expandable_smiles": ["C[C@H](O)CCCl"],
            "expandable_mapped_smiles": ["[CH3:1][C@H:2]([OH:3])[CH2:4][CH2:5][Cl:6]"],
            "route_steps": [],
        }
        assert request_handler(request)["candidates"] == []
        corrected = request_handler(request)["candidates"][0]
        return {
            "route_steps": [corrected["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    corrected["precursor_smiles"],
                    corrected["mapped_precursor_smiles"],
                )
            ],
            "solved": False,
            "policy_calls": 2,
            "mcts_iterations": 2,
            "diagnostics": {
                "engine": "AiZynthFinder.MctsSearchTree",
                "provider_callback_count": 2,
            },
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    runner = SequentialStrategyDirectorRunner(
        node_executor=builder_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )
    strategy = _strategy_card(1)
    branch = {
        "branch_index": 0,
        "lens": "repair the current mapped frontier",
        "steps": [],
        "target_mapped_smiles": ("[CH3:1][C@H:2]([OH:3])[CH2:4][CH2:5][Cl:6]"),
        "strategy_card": strategy,
        "root_strategy_card": strategy,
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 4,
        "path_repair_builder_call_count": 0,
        "call_count": 4,
        "open_leaf_states": [
            {
                "smiles": "C[C@H](O)CCCl",
                "mapped_smiles": ("[CH3:1][C@H:2]([OH:3])[CH2:4][CH2:5][Cl:6]"),
            }
        ],
        "open_leaves": ["C[C@H](O)CCCl"],
        "expanded_products": set(),
        "_path_repair_resume": {
            "repair_frontier_mapped_product_smiles": ("[CH3:1][C@H:2]([OH:3])[CH2:4][CH2:5][Cl:6]"),
            "repair_goal": "reconnect the suffix with the required configuration",
            "active_constraints": [],
            "durable_steps": [],
            "reconnect_boundaries": [
                {
                    "step_id": "suffix:1",
                    "product_smiles": "CC[C@@H](C)O",
                    "mapped_product_smiles": ("[CH3:7][C@@H:8]([OH:9])[CH2:10][CH3:11]"),
                }
            ],
            "reserved_atom_maps": [],
        },
    }

    records = runner._expand_seeded_branches_aizynthfinder(
        _spec(_context()),
        target="C[C@H](O)CCCl",
        seeded=[branch],
        existing_records=[],
        route_quota=sequential_module._NodeCallBudget(
            model_invocations=10,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        critic_editor_call_reserve=0,
        critic_input_reserve=0,
        critic_output_reserve=0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            enable_key_event_critic=False,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            strategy_branch_workers=1,
            max_node_expansions_per_branch=5,
            max_reactionjson_candidates_per_node=1,
        ),
        started=time.monotonic(),
    )

    assert len(records) == 2
    assert branch["path_repair_builder_call_count"] == 2
    feedback = builder_contexts[1]["last_rejection_for_this_leaf"]
    assert feedback["reason"] == ("path_repair_reconnect_boundary_stereo_mismatch")
    assert feedback["replay_diagnostic"]["stereo_mismatch_atom_maps"] == [8]
    assert branch["steps"][0]["reaction_operations"][-1] == {
        "op": "invert_stereocenter",
        "map_idx": 2,
    }
    assert any(
        row.get("reason") == "path_repair_reconnect_boundary_stereo_mismatch"
        for row in branch["materialization_diagnostics"]
    )


def test_path_repair_rejects_sideways_edit_then_accepts_exact_boundary(
    monkeypatch,
) -> None:
    builder_contexts: list[dict] = []

    def builder_executor(task: WorkerTask) -> WorkerRunRecord:
        context = json.loads(task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
        builder_contexts.append(context)
        fragment = "[*]Cl" if len(builder_contexts) == 1 else "[*]I"
        product = str(task.host_context.get("selected_product") or "")
        return replace(
            _proposal_record(
                {
                    "candidate_id": task.task_id,
                    "product_smiles": product,
                    "precursor_smiles": [],
                    "checkpoint_relation": "preparatory",
                    "reaction_family": "mapped boundary progress canary",
                    "conditions": ["test conditions"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "reaction_operations": [
                        {"op": "remove_group", "map_indices": [2]},
                        {"op": "add_group", "map_idx": 1, "fragment_smiles": fragment},
                    ],
                },
                target=product,
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        request = {
            "expandable_smiles": ["CBr"],
            "expandable_mapped_smiles": ["[CH3:1][Br:2]"],
            "route_steps": [],
        }
        assert request_handler(request)["candidates"] == []
        corrected = request_handler(request)["candidates"][0]
        return {
            "route_steps": [corrected["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    corrected["precursor_smiles"],
                    corrected["mapped_precursor_smiles"],
                )
            ],
            "solved": False,
            "policy_calls": 2,
            "mcts_iterations": 2,
            "diagnostics": {
                "engine": "AiZynthFinder.MctsSearchTree",
                "provider_callback_count": 2,
            },
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    runner = SequentialStrategyDirectorRunner(
        node_executor=builder_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )
    strategy = _strategy_card(1)
    branch = {
        "branch_index": 0,
        "lens": "repair the current mapped frontier",
        "steps": [],
        "target_mapped_smiles": "[CH3:1][Br:2]",
        "strategy_card": strategy,
        "root_strategy_card": strategy,
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 4,
        "path_repair_builder_call_count": 0,
        "call_count": 4,
        "open_leaf_states": [
            {"smiles": "CBr", "mapped_smiles": "[CH3:1][Br:2]"}
        ],
        "open_leaves": ["CBr"],
        "expanded_products": set(),
        "_path_repair_resume": {
            "repair_frontier_mapped_product_smiles": "[CH3:1][Br:2]",
            "repair_goal": "reach the preserved iodide boundary",
            "active_constraints": [],
            "durable_steps": [],
            "reconnect_boundaries": [
                {
                    "step_id": "suffix:iodide",
                    "product_smiles": "CI",
                    "mapped_product_smiles": "[CH3:1][I:9]",
                }
            ],
            "reserved_atom_maps": [9],
        },
    }

    records = runner._expand_seeded_branches_aizynthfinder(
        _spec(_context()),
        target="CBr",
        seeded=[branch],
        existing_records=[],
        route_quota=sequential_module._NodeCallBudget(
            model_invocations=10,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        critic_editor_call_reserve=0,
        critic_input_reserve=0,
        critic_output_reserve=0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            enable_key_event_critic=False,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            strategy_branch_workers=1,
            max_node_expansions_per_branch=5,
            max_reactionjson_candidates_per_node=1,
        ),
        started=time.monotonic(),
    )

    assert len(records) == 2
    assert branch["path_repair_builder_call_count"] == 2
    feedback = builder_contexts[1]["last_rejection_for_this_leaf"]
    assert feedback["reason"] == "path_repair_candidate_not_toward_reconnect_boundary"
    assert feedback["replay_diagnostic"]["candidate_boundary_distance"] == (
        feedback["replay_diagnostic"]["selected_boundary_distance"]
    )
    assert branch["steps"][0]["precursor_smiles"] == ["CI"]


def test_path_repair_replay_failure_survives_into_descendant_leaf_prompt(
    monkeypatch,
) -> None:
    builder_contexts: list[dict] = []

    def builder_executor(task: WorkerTask) -> WorkerRunRecord:
        context = json.loads(task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
        builder_contexts.append(context)
        product = str(task.host_context.get("selected_product") or "")
        if len(builder_contexts) == 1:
            operations = [
                {"op": "set_bond_stereo", "map_a": 1, "map_b": 2, "stereo": "E"}
            ]
        elif product == "CBr":
            operations = [
                {"op": "remove_group", "map_indices": [2]},
                {"op": "add_group", "map_idx": 1, "fragment_smiles": "[*]Cl"},
            ]
        else:
            match = re.search(r"\[Cl:(\d+)\]", context["selected_leaf_mapped"])
            assert match is not None
            chlorine_map = int(match.group(1))
            operations = [
                {"op": "remove_group", "map_indices": [chlorine_map]},
                {
                    "op": "add_group",
                    "map_idx": 1,
                    "fragment_smiles": "[*]I",
                },
            ]
        return replace(
            _proposal_record(
                {
                    "candidate_id": task.task_id,
                    "product_smiles": product,
                    "precursor_smiles": [],
                    "checkpoint_relation": "preparatory",
                    "reaction_family": "transaction replay memory canary",
                    "conditions": ["test conditions"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "reaction_operations": operations,
                },
                target=product,
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        root_request = {
            "expandable_smiles": ["CBr"],
            "expandable_mapped_smiles": ["[CH3:1][Br:2]"],
            "route_steps": [],
        }
        assert request_handler(root_request)["candidates"] == []
        first = request_handler(root_request)["candidates"][0]
        child_request = {
            "expandable_smiles": list(first["precursor_smiles"]),
            "expandable_mapped_smiles": list(first["mapped_precursor_smiles"]),
            "route_steps": [first["route_step"]],
        }
        second = request_handler(child_request)["candidates"][0]
        return {
            "route_steps": [first["route_step"], second["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    second["precursor_smiles"],
                    second["mapped_precursor_smiles"],
                )
            ],
            "solved": False,
            "policy_calls": 3,
            "mcts_iterations": 3,
            "diagnostics": {
                "engine": "AiZynthFinder.MctsSearchTree",
                "provider_callback_count": 3,
            },
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    runner = SequentialStrategyDirectorRunner(
        node_executor=builder_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )
    strategy = _strategy_card(1)
    branch = {
        "branch_index": 0,
        "lens": "repair the current mapped frontier",
        "steps": [],
        "target_mapped_smiles": "[CH3:1][Br:2]",
        "strategy_card": strategy,
        "root_strategy_card": strategy,
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 4,
        "path_repair_builder_call_count": 0,
        "call_count": 4,
        "open_leaf_states": [
            {"smiles": "CBr", "mapped_smiles": "[CH3:1][Br:2]"}
        ],
        "open_leaves": ["CBr"],
        "expanded_products": set(),
        "_path_repair_resume": {
            "repair_frontier_mapped_product_smiles": "[CH3:1][Br:2]",
            "repair_goal": "replace the rejected edge",
            "active_constraints": [],
            "durable_steps": [],
            "reconnect_boundaries": [],
            "reserved_atom_maps": [],
        },
    }

    records = runner._expand_seeded_branches_aizynthfinder(
        _spec(_context()),
        target="CBr",
        seeded=[branch],
        existing_records=[],
        route_quota=sequential_module._NodeCallBudget(
            model_invocations=10,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        critic_editor_call_reserve=0,
        critic_input_reserve=0,
        critic_output_reserve=0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            enable_key_event_critic=False,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            strategy_branch_workers=1,
            max_node_expansions_per_branch=5,
            max_reactionjson_candidates_per_node=1,
        ),
        started=time.monotonic(),
    )

    assert len(records) == 3
    assert builder_contexts[2]["selected_leaf_mapped"] != (
        builder_contexts[0]["selected_leaf_mapped"]
    )
    assert "last_rejection_for_this_leaf" not in builder_contexts[2]
    assert builder_contexts[2]["path_repair"]["replay_failures"] == [
        {
            "replay_error": "reactionjson_stereo_reference_neighbor_missing",
            "failed_operation": {
                "op": "set_bond_stereo",
                "map_a": 1,
                "map_b": 2,
                "stereo": "E",
            },
            "occurrence_count": 1,
        }
    ]


def test_path_repair_merges_new_key_critic_rejection_into_next_builder(
    monkeypatch,
) -> None:
    builder_contexts: list[dict] = []
    key_critic_prompts: list[str] = []
    builder_calls = 0
    key_critic_calls = 0

    def executor(task: WorkerTask) -> WorkerRunRecord:
        nonlocal builder_calls, key_critic_calls
        if task.task_type == "paper_matched_key_event_critic":
            key_critic_calls += 1
            key_critic_prompts.append(task.objective)
            reject = key_critic_calls == 1
            record = _critic_record(task, assessment="reject" if reject else "pass")
            artifact = copy.deepcopy(record.output_artifact)
            payload = artifact["payload"]
            payload["checkpoint_match"] = True
            payload["overall_assessment"] = "reject" if reject else "viable"
            payload["step_assessments"] = [
                {
                    "step_id": "",
                    "verdict": "reject" if reject else "pass",
                    "blocking": reject,
                    "blocking_type": "stereochemistry" if reject else "none",
                    "repair_scope": "route_span" if reject else "none",
                    "reasons": (
                        ["C13 and C18 configurations remain unspecified"]
                        if reject
                        else []
                    ),
                    "suggested_revision": (
                        "assign both new junction configurations after the graph edits"
                        if reject
                        else ""
                    ),
                }
            ]
            artifact["payload"] = payload
            return replace(record, output_artifact=artifact)

        builder_calls += 1
        context = json.loads(
            task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1]
        )
        builder_contexts.append(context)
        product = str(task.host_context.get("selected_product") or "")
        operation = (
            {"op": "break_bond", "map_a": 2, "map_b": 3}
            if builder_calls == 1
            else {"op": "break_bond", "map_a": 1, "map_b": 2}
        )
        return replace(
            _proposal_record(
                {
                    "candidate_id": task.task_id,
                    "product_smiles": product,
                    "precursor_smiles": [],
                    "checkpoint_relation": "executes_checkpoint",
                    "reaction_family": "repair checkpoint sibling",
                    "conditions": ["test conditions"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "reaction_operations": [operation],
                },
                target=product,
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        request = {
            "expandable_smiles": ["CCO"],
            "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
            "route_steps": [],
        }
        first = request_handler(request)
        assert first["candidates"] == []
        corrected = request_handler(request)["candidates"][0]
        return {
            "route_steps": [corrected["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    corrected["precursor_smiles"],
                    corrected["mapped_precursor_smiles"],
                )
            ],
            "solved": False,
            "policy_calls": 2,
            "mcts_iterations": 2,
            "diagnostics": {"engine": "AiZynthFinder.MctsSearchTree"},
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    strategy = _strategy_card(1)
    branch = {
        "branch_index": 0,
        "lens": "repair the current mapped frontier",
        "steps": [],
        "target_mapped_smiles": "[CH3:1][CH2:2][OH:3]",
        "strategy_card": strategy,
        "root_strategy_card": strategy,
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 1,
        "path_repair_builder_call_count": 0,
        "call_count": 1,
        "key_event_critic_completed": False,
        "key_event_critic_history": [],
        "open_leaf_states": [
            {"smiles": "CCO", "mapped_smiles": "[CH3:1][CH2:2][OH:3]"}
        ],
        "open_leaves": ["CCO"],
        "expanded_products": set(),
        "_path_repair_resume": {
            "repair_frontier_mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "repair_goal": "replace the rejected checkpoint without losing the route",
            "active_constraints": ["preserve the original transaction boundary"],
            "durable_steps": [],
            "reconnect_boundaries": [],
            "reserved_atom_maps": [],
            "completion_mode": "strategy_checkpoint",
        },
    }
    runner = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )

    records = runner._expand_seeded_branches_aizynthfinder(
        _spec(_context()),
        target="CCO",
        seeded=[branch],
        existing_records=[],
        route_quota=sequential_module._NodeCallBudget(
            model_invocations=12,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        critic_editor_call_reserve=0,
        critic_input_reserve=0,
        critic_output_reserve=0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            enable_key_event_critic=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            strategy_branch_workers=1,
            max_node_expansions_per_branch=5,
            max_reactionjson_candidates_per_node=1,
        ),
        started=time.monotonic(),
    )

    assert len(records) == 4
    assert builder_calls == 2
    assert key_critic_calls == 2
    first_constraints = builder_contexts[0]["pending_checkpoint_feedback"][
        "active_constraints"
    ]
    assert [row["blocking_type"] for row in first_constraints] == [
        "route_span_repair"
    ]
    retry_constraints = builder_contexts[1]["pending_checkpoint_feedback"][
        "active_constraints"
    ]
    assert [row["blocking_type"] for row in retry_constraints] == [
        "route_span_repair",
        "stereochemistry",
    ]
    assert retry_constraints[-1]["reasons"] == [
        "C13 and C18 configurations remain unspecified"
    ]
    assert retry_constraints[-1]["suggested_revision"] == (
        "assign both new junction configurations after the graph edits"
    )
    assert "last_rejection_for_this_leaf" not in builder_contexts[1]
    assert "C13 and C18 configurations remain unspecified" in key_critic_prompts[1]


def test_path_repair_suffix_reports_reconnect_boundary_not_reached() -> None:
    stitched, diagnostic = sequential_module._stitch_path_repair_suffix(
        mapped_target_smiles="[CH3:1][Br:2]",
        rebuilt_steps=[
            {
                "step_id": "repair:1",
                "product_smiles": "CBr",
                "reaction_operations": [
                    {"op": "remove_group", "map_indices": [2]},
                    {
                        "op": "add_group",
                        "map_idx": 1,
                        "fragment_smiles": "[*][OH:3]",
                    },
                ],
            }
        ],
        preserved_suffix_steps=[{"step_id": "suffix:1"}],
        reconnect_boundaries=[
            {
                "step_id": "suffix:1",
                "product_smiles": "CN",
                "mapped_product_smiles": "[CH3:1][NH2:4]",
            }
        ],
    )

    assert stitched is None
    assert diagnostic == {
        "reason": "path_repair_reconnect_boundary_not_reached",
        "boundary_step_id": "suffix:1",
        "boundary_product_smiles": "CN",
        "candidate_count": 0,
    }


def test_path_repair_suffix_reports_ambiguous_reconnect_boundary() -> None:
    stitched, diagnostic = sequential_module._stitch_path_repair_suffix(
        mapped_target_smiles="[CH3:1][CH2:2][CH2:3][CH3:4]",
        rebuilt_steps=[
            {
                "step_id": "repair:1",
                "product_smiles": "CCCC",
                "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
            }
        ],
        preserved_suffix_steps=[{"step_id": "suffix:1"}],
        reconnect_boundaries=[
            {
                "step_id": "suffix:1",
                "product_smiles": "CC",
                "mapped_product_smiles": "[CH3:7][CH3:8]",
            }
        ],
    )

    assert stitched is None
    assert diagnostic == {
        "reason": "path_repair_reconnect_boundary_ambiguous",
        "boundary_step_id": "suffix:1",
        "boundary_product_smiles": "CC",
        "candidate_count": 2,
    }


def test_path_repair_boundary_progress_requires_strict_structural_improvement() -> None:
    boundary = {
        "step_id": "suffix:iodide",
        "product_smiles": "CI",
        "mapped_product_smiles": "[CH3:1][I:9]",
    }

    sideways = sequential_module._path_repair_boundary_progress_failure(
        selected_leaf_mapped="[CH3:1][Br:2]",
        mapped_precursor_smiles=["[CH3:1][Cl:3]"],
        reconnect_boundaries=[boundary],
    )
    assert sideways is not None
    assert sideways["reason"] == "path_repair_candidate_not_toward_reconnect_boundary"
    assert sideways["candidate_boundary_distance"] == sideways["selected_boundary_distance"]

    assert (
        sequential_module._path_repair_boundary_progress_failure(
            selected_leaf_mapped="[CH3:1][Br:2]",
            mapped_precursor_smiles=["[CH3:1][I:37]"],
            reconnect_boundaries=[boundary],
        )
        is None
    )
    assert sequential_module._mapped_boundary_distance(
        "[CH3:1][I:37]",
        boundary["mapped_product_smiles"],
    ) == 0


def test_path_repair_replay_memory_deduplicates_across_descendant_leaves() -> None:
    invalid_stereo = {
        "phase": "route_builder_candidate",
        "product_smiles": "discarded product payload",
        "reason": "strategy_graph_edit_replay_failed",
        "replay_error": "reactionjson_stereo_reference_neighbor_missing",
        "operation_index": 2,
        "failed_operation": {
            "op": "set_bond_stereo",
            "map_a": 12,
            "map_b": 13,
            "stereo": "E",
        },
    }
    once = sequential_module._merge_path_repair_replay_failure((), invalid_stereo)
    twice = sequential_module._merge_path_repair_replay_failure(once, invalid_stereo)

    assert twice == [
        {
            "replay_error": "reactionjson_stereo_reference_neighbor_missing",
            "failed_operation": {
                "op": "set_bond_stereo",
                "map_a": 12,
                "map_b": 13,
                "stereo": "E",
            },
            "occurrence_count": 2,
        }
    ]
    assert "product_smiles" not in twice[0]
    assert "operation_index" not in twice[0]


def test_path_repair_stops_before_another_builder_call_at_suffix_boundary() -> None:
    boundary = {
        "step_id": "linear:4",
        "product_smiles": "ClCCBr",
        "mapped_product_smiles": "[CH2:1]([CH2:2][Cl:4])[Br:5]",
    }

    assert (
        sequential_module._path_repair_frontier_reaches_boundaries(
            product_smiles=["ClCCBr"],
            mapped_product_smiles=["[CH2:1]([CH2:2][Cl:7])[Br:8]"],
            reconnect_boundaries=[boundary],
        )
        is True
    )
    assert (
        sequential_module._path_repair_frontier_reaches_boundaries(
            product_smiles=["ClCCBr", "CC"],
            mapped_product_smiles=[
                "[CH2:1]([CH2:2][Cl:7])[Br:8]",
                "[CH3:9][CH3:10]",
            ],
            reconnect_boundaries=[boundary],
        )
        is False
    )

    assert sequential_module._path_repair_boundary_leaf_indices(
        product_smiles=["ClCCBr", "CC"],
        mapped_product_smiles=[
            "[CH2:1]([CH2:2][Cl:7])[Br:8]",
            "[CH3:9][CH3:10]",
        ],
        reconnect_boundaries=[boundary],
    ) == frozenset({0})

    preparatory = [{"checkpoint_relation": "preparatory"}]
    checkpoint = [
        *preparatory,
        {"checkpoint_relation": "executes_checkpoint"},
    ]
    assert (
        sequential_module._path_repair_completion_reached(
            preparatory,
            completion_mode="replacement_edge",
        )
        is True
    )
    assert (
        sequential_module._path_repair_completion_reached(
            preparatory,
            completion_mode="strategy_checkpoint",
        )
        is False
    )
    assert (
        sequential_module._path_repair_completion_reached(
            checkpoint,
            completion_mode="strategy_checkpoint",
        )
        is True
    )
    checkpoint_with_id = [
        *preparatory,
        {"step_id": "repair:key", "checkpoint_relation": "executes_checkpoint"},
    ]
    assert (
        sequential_module._path_repair_completion_reached(
            checkpoint_with_id,
            completion_mode="strategy_checkpoint",
            selected_critic_pass_step_ids=(),
        )
        is False
    )
    assert (
        sequential_module._path_repair_completion_reached(
            checkpoint_with_id,
            completion_mode="strategy_checkpoint",
            selected_critic_pass_step_ids=("repair:key",),
        )
        is True
    )
    pending = {
        "completion_mode": "strategy_checkpoint",
        "required_checkpoint_step_id": "repair:key",
    }
    assert (
        sequential_module._path_repair_recritic_completion_failure(
            pending,
            {"strategy_adherence": False},
        )
        == "path_repair_recritic_strategy_checkpoint_missing"
    )
    assert (
        sequential_module._path_repair_recritic_completion_failure(
            pending,
            {"strategy_adherence": True, "step_assessments": []},
        )
        == "path_repair_recritic_checkpoint_assessment_missing"
    )
    assert (
        sequential_module._path_repair_recritic_completion_failure(
            pending,
            {
                "strategy_adherence": True,
                "step_assessments": [{"step_id": "repair:key", "verdict": "uncertain"}],
            },
        )
        == ""
    )


def test_path_repair_span_rejects_uncovered_blocker() -> None:
    rollback, diagnostic = sequential_module._prepare_path_repair_span(
        current_steps=_branching_path_repair_route(),
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        directive={
            "rollback_start_step_id": "route:4",
            "rebuild_through_step_id": "route:4",
            "repair_goal": "change the wrong sibling",
            "active_constraints": [],
        },
        blocking_step_ids=["route:3"],
    )

    assert rollback is None
    assert diagnostic["reason"] == ("path_repair_span_does_not_cover_critic_blocker")


def test_online_route_span_repair_can_rebuild_from_provisional_frontier(
    monkeypatch,
) -> None:
    route = _linear_path_repair_route_with_suffix()
    authoritative_steps = [dict(route[0])]
    provisional_focus = dict(route[1])

    def editor_executor(task: WorkerTask) -> WorkerRunRecord:
        assert "may start at one of those rows" in task.objective
        editor_context = json.loads(task.objective.rsplit("\n", 1)[1])
        assert editor_context["schema_version"] == "path_repair_editor_context.v2"
        assert editor_context["provisional_rejected_step_ids"] == ["linear:2"]
        assert editor_context["critic_annotations"]["active_checkpoint_constraints"] == [
            {
                "blocking_type": "mechanism",
                "reasons": ["do not restore the previously rejected double capture"],
                "suggested_revision": "use one propagating closure per alkene",
            },
            {
                "blocking_type": "stereochemistry",
                "reasons": ["the initiating epoxide configuration must remain defined"],
                "suggested_revision": "retain the configured chiral epoxide",
            },
        ]
        assert "remain binding across this route-span transaction" in task.objective
        return replace(
            _proposal_record(
                {
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "repair_directive": {
                        "rollback_start_step_id": "linear:2",
                        "rebuild_through_step_id": "linear:2",
                        "additional_coupled_blocker_step_ids": [],
                        "preserved_suffix_compatible": True,
                        "repair_goal": "replace the rejected event with a coherent local sequence",
                        "active_constraints": ["reach the checkpoint before re-Critic"],
                    },
                },
                target="CCO",
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    runner = SequentialStrategyDirectorRunner(
        editor_executor=editor_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )

    def checkpoint_rebuild(_self, *, seeded, **_kwargs):
        repair_branch = seeded[0]
        reference_span = repair_branch["_path_repair_resume"]["repair_reference_span"]
        assert [row["step_id"] for row in reference_span] == ["linear:2"]
        assert reference_span[0]["mapped_product_smiles"] == "[CH3:1][CH3:2]"
        assert reference_span[0]["reaction_operations"] == [
            {
                "op": "add_group",
                "map_idx": 2,
                "fragment_smiles": "[*][Cl:4]",
            }
        ]
        assert reference_span[0]["prior_key_critic"] == {
            "status": "rejected",
            "checkpoint_match": True,
            "verdict": "reject",
            "blocking_type": "sequence_dependency",
        }
        repair_branch["steps"].append(
            {
                "step_id": "repair:checkpoint",
                "product_smiles": "CC",
                "mapped_product_smiles": "[CH3:1][CH3:2]",
                "precursor_smiles": ["C", "C"],
                "mapped_precursor_smiles": ["[CH4:1]", "[CH4:2]"],
                "checkpoint_relation": "executes_checkpoint",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 2}
                ],
            }
        )
        repair_branch["path_repair_builder_call_count"] = 1
        repair_branch["open_leaf_states"] = [
            {"smiles": "C", "mapped_smiles": "[CH4:1]"},
            {"smiles": "C", "mapped_smiles": "[CH4:2]"},
        ]
        repair_branch["open_leaves"] = ["C", "C"]
        return []

    monkeypatch.setattr(
        runner,
        "_expand_seeded_branches_aizynthfinder",
        checkpoint_rebuild,
    )
    branch = {
        "branch_index": 0,
        "steps": authoritative_steps,
        "target_mapped_smiles": "[CH3:1][CH2:2][OH:3]",
        "strategy_card": {
            "strategy_query": "repair the local route span",
            "critic_checkpoint": "audit the rebuilt key construction",
        },
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 1,
        "path_repair_builder_call_count": 0,
        "call_count": 1,
        "editor_attempt_count": 0,
        "editor_call_count": 0,
        "open_leaf_states": [],
        "open_leaves": [],
        "expanded_products": set(),
        "complete_in_bound_stock": False,
        "key_event_critic_history": [
            {
                "focus_step_id": "linear:2",
                "status": "rejected",
                "checkpoint_match": True,
                "assessment": {
                    "verdict": "reject",
                    "blocking_type": "sequence_dependency",
                },
            }
        ],
    }
    edited = runner._repair_branch_transactionally(
        _spec(_context()),
        target="CCO",
        branch=branch,
        blocking_steps=[provisional_focus],
        critique={
            "status": "reject",
            "step_assessments": [
                {
                    "step_id": "linear:2",
                    "blocking": True,
                    "verdict": "reject",
                    "blocking_type": "sequence_dependency",
                    "repair_scope": "route_span",
                    "reasons": ["one edge cannot encode the required local sequence"],
                }
            ],
        },
        iteration=-1,
        records=[],
        max_prompt_bytes=100_000,
        max_node_call_timeout_s=60.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=20,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        started=time.monotonic(),
        reserve_model_invocations=0,
        reserve_input_tokens=0,
        reserve_output_tokens=0,
        reserve_wall_time_s=0.0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            max_node_expansions_per_branch=5,
            max_route_local_repair_rounds=1,
        ),
        repair_context_steps=[*authoritative_steps, provisional_focus],
        checkpoint_feedback={
            "active_constraints": [
                {
                    "blocking_type": "mechanism",
                    "reasons": ["do not restore the previously rejected double capture"],
                    "suggested_revision": "use one propagating closure per alkene",
                },
                {
                    "blocking_type": "stereochemistry",
                    "reasons": ["the initiating epoxide configuration must remain defined"],
                    "suggested_revision": "retain the configured chiral epoxide",
                },
            ]
        },
    )

    assert edited is True
    assert [row["step_id"] for row in branch["steps"]] == [
        "linear:1",
        "repair:checkpoint",
    ]
    transaction = branch["path_repair_transactions"][-1]
    assert transaction["status"] == "rebuilt_pending_recritic"
    assert transaction["completion_mode"] == "strategy_checkpoint"
    assert transaction["boundary_rebuilt"] is True
    assert transaction["replacement_step_replayed"] is True
    assert transaction["active_constraints"][0].startswith(
        "Preserve unresolved Key-event Critic findings across this repair: mechanism:"
    )
    assert "previously rejected double capture" in transaction["active_constraints"][0]
    assert "initiating epoxide configuration" in transaction["active_constraints"][0]
    assert "reach the checkpoint before re-Critic" in transaction["active_constraints"]
    assert branch["_pending_path_repair_transaction"]["required_checkpoint_step_id"] == (
        "repair:checkpoint"
    )


def test_online_key_event_repair_does_not_commit_preparatory_prefix(
    monkeypatch,
) -> None:
    route = _linear_path_repair_route_with_suffix()[:3]
    authoritative_steps = [dict(row) for row in route[:2]]
    provisional_focus = dict(route[2])

    def editor_executor(task: WorkerTask) -> WorkerRunRecord:
        return replace(
            _proposal_record(
                {
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "repair_directive": {
                        "rollback_start_step_id": "linear:2",
                        "rebuild_through_step_id": "linear:3",
                        "additional_coupled_blocker_step_ids": [],
                        "preserved_suffix_compatible": True,
                        "repair_goal": "rebuild the missing key construction",
                        "active_constraints": ["reach the strategy checkpoint before re-Critic"],
                    },
                },
                target="CCO",
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    runner = SequentialStrategyDirectorRunner(
        editor_executor=editor_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )

    def preparatory_search(_self, *, seeded, **_kwargs):
        repair_branch = seeded[0]
        repair_branch["steps"].append(
            {
                "step_id": "repair:preparatory",
                "product_smiles": "CC",
                "mapped_product_smiles": "[CH3:1][CH3:2]",
                "precursor_smiles": ["CCCl"],
                "mapped_precursor_smiles": ["[CH3:1][CH2:2][Cl:7]"],
                "checkpoint_relation": "preparatory",
                "reaction_operations": [
                    {
                        "op": "add_group",
                        "map_idx": 2,
                        "fragment_smiles": "[*][Cl:7]",
                    }
                ],
            }
        )
        repair_branch["path_repair_builder_call_count"] = 1
        repair_branch["open_leaf_states"] = [
            {"smiles": "CCCl", "mapped_smiles": "[CH3:1][CH2:2][Cl:7]"}
        ]
        repair_branch["open_leaves"] = ["CCCl"]
        return []

    monkeypatch.setattr(
        runner,
        "_expand_seeded_branches_aizynthfinder",
        preparatory_search,
    )
    branch = {
        "branch_index": 0,
        "steps": authoritative_steps,
        "target_mapped_smiles": "[CH3:1][CH2:2][OH:3]",
        "strategy_card": {
            "strategy_query": "execute the key construction",
            "critical_assumption": "the key construction is selective",
            "critic_checkpoint": "audit the key construction",
        },
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 2,
        "path_repair_builder_call_count": 0,
        "call_count": 2,
        "editor_attempt_count": 0,
        "editor_call_count": 0,
        "open_leaf_states": [],
        "open_leaves": [],
        "expanded_products": set(),
        "complete_in_bound_stock": False,
    }

    edited = runner._repair_branch_transactionally(
        _spec(_context()),
        target="CCO",
        branch=branch,
        blocking_steps=[provisional_focus],
        critique={
            "status": "reject",
            "step_assessments": [
                {
                    "step_id": "linear:3",
                    "blocking": True,
                    "verdict": "reject",
                    "blocking_type": "sequence_dependency",
                    "repair_scope": "route_span",
                    "reasons": ["the accepted sequence must change"],
                }
            ],
        },
        iteration=-1,
        records=[],
        max_prompt_bytes=100_000,
        max_node_call_timeout_s=60.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=20,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        started=time.monotonic(),
        reserve_model_invocations=0,
        reserve_input_tokens=0,
        reserve_output_tokens=0,
        reserve_wall_time_s=0.0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            max_node_expansions_per_branch=5,
            max_route_local_repair_rounds=1,
        ),
        repair_context_steps=[*authoritative_steps, provisional_focus],
    )

    assert edited is False
    assert branch["steps"] == authoritative_steps
    transaction = branch["path_repair_transactions"][-1]
    assert transaction["status"] == "rolled_back_uncommitted"
    assert transaction["completion_mode"] == "strategy_checkpoint"
    assert transaction["replacement_step_replayed"] is False
    assert transaction["reason"] == ("path_repair_replacement_step_not_replayed")


def test_transactional_path_repair_does_not_commit_deletion_only(
    monkeypatch,
) -> None:
    def editor_executor(task: WorkerTask) -> WorkerRunRecord:
        return replace(
            _proposal_record(
                {
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "repair_directive": {
                        "rollback_start_step_id": "route:2",
                        "rebuild_through_step_id": "route:3",
                        "additional_coupled_blocker_step_ids": [],
                        "preserved_suffix_compatible": True,
                        "repair_goal": "replace the blocked left branch",
                        "active_constraints": ["preserve the CO sibling"],
                    },
                },
                target="CCOC",
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    runner = SequentialStrategyDirectorRunner(
        editor_executor=editor_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )

    def seed_only_search(_self, **_kwargs):
        return []

    monkeypatch.setattr(
        runner,
        "_expand_seeded_branches_aizynthfinder",
        seed_only_search,
    )
    original_steps = _branching_path_repair_route()
    branch = {
        "branch_index": 0,
        "steps": [dict(row) for row in original_steps],
        "target_mapped_smiles": "[CH3:1][CH2:2][O:3][CH3:4]",
        "strategy_card": {"strategy_query": "preserve a convergent split"},
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 5,
        "path_repair_builder_call_count": 0,
        "call_count": 5,
        "editor_attempt_count": 0,
        "editor_call_count": 0,
        "open_leaf_states": [],
        "open_leaves": [],
        "expanded_products": set(),
        "complete_in_bound_stock": False,
    }
    records: list[WorkerRunRecord] = []
    edited = runner._repair_branch_transactionally(
        _spec(_context()),
        target="CCOC",
        branch=branch,
        blocking_steps=[original_steps[2]],
        critique={
            "status": "reject",
            "step_assessments": [
                {
                    "step_id": "route:3",
                    "blocking": True,
                    "verdict": "reject",
                    "reasons": ["blocked chemistry"],
                }
            ],
        },
        iteration=0,
        records=records,
        max_prompt_bytes=100_000,
        max_node_call_timeout_s=60.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=20,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        started=time.monotonic(),
        reserve_model_invocations=0,
        reserve_input_tokens=0,
        reserve_output_tokens=0,
        reserve_wall_time_s=0.0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            max_node_expansions_per_branch=5,
            max_route_local_repair_rounds=6,
        ),
    )

    assert edited is False
    assert [row["step_id"] for row in branch["steps"]] == [row["step_id"] for row in original_steps]
    assert branch["editor_call_count"] == 0
    assert branch["path_repair_transactions"][-1]["status"] == ("rolled_back_uncommitted")


def test_transactional_path_repair_preflight_avoids_builder_for_incompatible_suffix(
    monkeypatch,
) -> None:
    def editor_executor(task: WorkerTask) -> WorkerRunRecord:
        return replace(
            _proposal_record(
                {
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "repair_directive": {
                        "rollback_start_step_id": "linear:2",
                        "rebuild_through_step_id": "linear:3",
                        "additional_coupled_blocker_step_ids": [],
                        "preserved_suffix_compatible": False,
                        "repair_goal": "change the state required by the retained suffix",
                        "active_constraints": [],
                    },
                },
                target="CCO",
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    runner = SequentialStrategyDirectorRunner(
        editor_executor=editor_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )
    builder_called = False

    def forbidden_search(_self, **_kwargs):
        nonlocal builder_called
        builder_called = True
        raise AssertionError("boundary preflight must stop before Builder/AiZ")

    monkeypatch.setattr(
        runner,
        "_expand_seeded_branches_aizynthfinder",
        forbidden_search,
    )
    original_steps = _linear_path_repair_route_with_suffix()
    branch = {
        "branch_index": 0,
        "steps": [dict(row) for row in original_steps],
        "target_mapped_smiles": "[CH3:1][CH2:2][OH:3]",
        "strategy_card": {"strategy_query": "preserve the local relay"},
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 3,
        "path_repair_builder_call_count": 0,
        "call_count": 4,
        "editor_attempt_count": 0,
        "editor_call_count": 0,
        "open_leaf_states": [],
        "open_leaves": [],
        "expanded_products": set(),
        "complete_in_bound_stock": False,
    }
    records: list[WorkerRunRecord] = []

    edited = runner._repair_branch_transactionally(
        _spec(_context()),
        target="CCO",
        branch=branch,
        blocking_steps=[original_steps[2]],
        critique={
            "status": "reject",
            "step_assessments": [
                {
                    "step_id": "linear:3",
                    "blocking": True,
                    "verdict": "reject",
                    "reasons": ["blocked chemistry"],
                }
            ],
        },
        iteration=0,
        records=records,
        max_prompt_bytes=100_000,
        max_node_call_timeout_s=60.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=20,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        started=time.monotonic(),
        reserve_model_invocations=0,
        reserve_input_tokens=0,
        reserve_output_tokens=0,
        reserve_wall_time_s=0.0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            max_node_expansions_per_branch=5,
            max_route_local_repair_rounds=6,
        ),
    )

    assert edited is False
    assert builder_called is False
    assert len(records) == 1
    assert branch["editor_attempt_count"] == 1
    assert branch["path_repair_builder_call_count"] == 0
    assert "path_repair_transactions" not in branch
    diagnostic = branch["editor_rejection_diagnostics"][-1]
    assert diagnostic["reason"] == ("path_repair_preserved_suffix_declared_incompatible")
    assert diagnostic["builder_calls_avoided"] is True


def test_transactional_path_repair_accepts_editor_declared_coupled_blocker(
    monkeypatch,
) -> None:
    def editor_executor(task: WorkerTask) -> WorkerRunRecord:
        return replace(
            _proposal_record(
                {
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "repair_directive": {
                        "rollback_start_step_id": "route:1",
                        "rebuild_through_step_id": "route:4",
                        "additional_coupled_blocker_step_ids": ["route:4"],
                        "preserved_suffix_compatible": True,
                        "repair_goal": "rebuild both chemically coupled branches",
                        "active_constraints": [],
                    },
                },
                target="CCOC",
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    runner = SequentialStrategyDirectorRunner(
        editor_executor=editor_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )
    builder_called = False

    def incomplete_search(_self, **_kwargs):
        nonlocal builder_called
        builder_called = True
        return []

    monkeypatch.setattr(
        runner,
        "_expand_seeded_branches_aizynthfinder",
        incomplete_search,
    )
    original_steps = _branching_path_repair_route()
    branch = {
        "branch_index": 0,
        "steps": [dict(row) for row in original_steps],
        "target_mapped_smiles": "[CH3:1][CH2:2][O:3][CH3:4]",
        "strategy_card": {"strategy_query": "preserve a convergent split"},
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 3,
        "path_repair_builder_call_count": 0,
        "call_count": 4,
        "editor_attempt_count": 0,
        "editor_call_count": 0,
        "open_leaf_states": [],
        "open_leaves": [],
        "expanded_products": set(),
        "complete_in_bound_stock": False,
    }

    edited = runner._repair_branch_transactionally(
        _spec(_context()),
        target="CCOC",
        branch=branch,
        blocking_steps=[original_steps[2], original_steps[3]],
        critique={
            "status": "reject",
            "step_assessments": [
                {
                    "step_id": "route:3",
                    "blocking": True,
                    "verdict": "reject",
                    "reasons": ["left branch is blocked"],
                },
                {
                    "step_id": "route:4",
                    "blocking": True,
                    "verdict": "reject",
                    "reasons": ["right branch is chemically coupled"],
                },
            ],
        },
        iteration=0,
        records=[],
        max_prompt_bytes=100_000,
        max_node_call_timeout_s=60.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=20,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        started=time.monotonic(),
        reserve_model_invocations=0,
        reserve_input_tokens=0,
        reserve_output_tokens=0,
        reserve_wall_time_s=0.0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            max_node_expansions_per_branch=5,
            max_route_local_repair_rounds=6,
        ),
    )

    assert edited is False
    assert builder_called is True
    transaction = branch["path_repair_transactions"][-1]
    assert transaction["selected_blocker_step_ids"] == ["route:3", "route:4"]
    assert transaction["deferred_blocker_step_ids"] == []
    assert transaction["removed_step_ids"] == [
        "route:1",
        "route:2",
        "route:3",
        "route:4",
    ]
    assert all(
        row.get("reason") != "path_repair_span_crosses_deferred_blocker_component"
        for row in branch.get("editor_rejection_diagnostics") or []
    )


def test_transactional_path_repair_does_not_call_editor_without_builder_budget() -> None:
    editor_called = False

    def editor_executor(_task: WorkerTask) -> WorkerRunRecord:
        nonlocal editor_called
        editor_called = True
        raise AssertionError("Editor must not run after Builder budget exhaustion")

    runner = SequentialStrategyDirectorRunner(
        editor_executor=editor_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )
    original_steps = _branching_path_repair_route()
    branch = {
        "branch_index": 0,
        "steps": [dict(row) for row in original_steps],
        "target_mapped_smiles": "[CH3:1][CH2:2][O:3][CH3:4]",
        "strategy_card": {"strategy_query": "preserve a convergent split"},
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 5,
        "path_repair_builder_call_count": 5,
        "call_count": 5,
        "editor_attempt_count": 0,
        "editor_call_count": 0,
        "open_leaf_states": [],
        "open_leaves": [],
        "expanded_products": set(),
        "complete_in_bound_stock": False,
    }

    edited = runner._repair_branch_transactionally(
        _spec(_context()),
        target="CCOC",
        branch=branch,
        blocking_steps=[original_steps[2]],
        critique={"status": "reject"},
        iteration=0,
        records=[],
        max_prompt_bytes=100_000,
        max_node_call_timeout_s=60.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=20,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        started=time.monotonic(),
        reserve_model_invocations=0,
        reserve_input_tokens=0,
        reserve_output_tokens=0,
        reserve_wall_time_s=0.0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            max_node_expansions_per_branch=5,
            max_route_local_repair_rounds=6,
        ),
    )

    assert edited is False
    assert editor_called is False
    assert branch["editor_attempt_count"] == 0
    assert branch["editor_rejection_diagnostics"][-1]["reason"] == (
        "path_repair_builder_budget_exhausted_before_boundary"
    )


def test_transactional_path_repair_budget_is_cumulative_across_transactions(
    monkeypatch,
) -> None:
    editor_calls = 0
    search_calls = 0

    def editor_executor(task: WorkerTask) -> WorkerRunRecord:
        nonlocal editor_calls
        editor_calls += 1
        return replace(
            _proposal_record(
                {
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "repair_directive": {
                        "rollback_start_step_id": "route:2",
                        "rebuild_through_step_id": "route:3",
                        "additional_coupled_blocker_step_ids": [],
                        "preserved_suffix_compatible": True,
                        "repair_goal": "replace the blocked left branch",
                        "active_constraints": [],
                    },
                },
                target="CCOC",
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    runner = SequentialStrategyDirectorRunner(
        editor_executor=editor_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )

    def incomplete_search(_self, *, seeded, **_kwargs):
        nonlocal search_calls
        repair_branch = seeded[0]
        increment = 3 if search_calls == 0 else 2
        search_calls += 1
        repair_branch["path_repair_builder_call_count"] = (
            int(repair_branch.get("path_repair_builder_call_count") or 0) + increment
        )
        return []

    monkeypatch.setattr(
        runner,
        "_expand_seeded_branches_aizynthfinder",
        incomplete_search,
    )
    original_steps = _branching_path_repair_route()
    branch = {
        "branch_index": 0,
        "steps": [dict(row) for row in original_steps],
        "target_mapped_smiles": "[CH3:1][CH2:2][O:3][CH3:4]",
        "strategy_card": {"strategy_query": "preserve a convergent split"},
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 5,
        "path_repair_builder_call_count": 0,
        "call_count": 5,
        "editor_attempt_count": 0,
        "editor_call_count": 0,
        "open_leaf_states": [],
        "open_leaves": [],
        "expanded_products": set(),
        "complete_in_bound_stock": False,
    }
    config = DirectorConfig(
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        enable_transactional_path_repair=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_branch_count=1,
        max_node_expansions_per_branch=5,
        max_route_local_repair_rounds=6,
    )
    quota = sequential_module._NodeCallBudget(
        model_invocations=20,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        wall_time_s=600.0,
    )

    for iteration in range(2):
        assert (
            runner._repair_branch_transactionally(
                _spec(_context()),
                target="CCOC",
                branch=branch,
                blocking_steps=[original_steps[2]],
                critique={"status": "reject"},
                iteration=iteration,
                records=[],
                max_prompt_bytes=100_000,
                max_node_call_timeout_s=60.0,
                quota=quota,
                started=time.monotonic(),
                reserve_model_invocations=0,
                reserve_input_tokens=0,
                reserve_output_tokens=0,
                reserve_wall_time_s=0.0,
                config=config,
            )
            is False
        )

    # A third transaction is ineligible before the Editor runs because the
    # two prior transactions consumed the branch's normal expansion budget.
    assert (
        runner._repair_branch_transactionally(
            _spec(_context()),
            target="CCOC",
            branch=branch,
            blocking_steps=[original_steps[2]],
            critique={"status": "reject"},
            iteration=2,
            records=[],
            max_prompt_bytes=100_000,
            max_node_call_timeout_s=60.0,
            quota=quota,
            started=time.monotonic(),
            reserve_model_invocations=0,
            reserve_input_tokens=0,
            reserve_output_tokens=0,
            reserve_wall_time_s=0.0,
            config=config,
        )
        is False
    )

    assert editor_calls == 2
    assert search_calls == 2
    assert branch["path_repair_builder_call_count"] == 5
    assert branch["route_call_count"] == 5
    assert [row["step_id"] for row in branch["steps"]] == [row["step_id"] for row in original_steps]
    assert [row["builder_calls"] for row in branch["path_repair_transactions"]] == [3, 2]


def test_path_repair_builder_worker_exception_does_not_consume_repair_axis(
    monkeypatch,
) -> None:
    def failing_builder(_task: WorkerTask) -> WorkerRunRecord:
        raise RuntimeError("provider failed before a durable worker record")

    def fake_sidecar(*, request_handler, max_policy_calls, **_kwargs):
        assert max_policy_calls == 6
        request_handler(
            {
                "expandable_smiles": ["CCO"],
                "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
                "route_steps": [],
            }
        )
        raise AssertionError("the failing Builder callback must escape")

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    runner = SequentialStrategyDirectorRunner(node_executor=failing_builder)
    branch = {
        "branch_index": 0,
        "lens": "repair the current mapped frontier",
        "steps": [],
        "target_mapped_smiles": "[CH3:1][CH2:2][OH:3]",
        "strategy_card": _strategy_card(1),
        "root_strategy_card": _strategy_card(1),
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 5,
        "path_repair_builder_call_count": 0,
        "call_count": 5,
        "open_leaf_states": [{"smiles": "CCO", "mapped_smiles": "[CH3:1][CH2:2][OH:3]"}],
        "open_leaves": ["CCO"],
        "expanded_products": set(),
        "_path_repair_resume": {
            "repair_frontier_mapped_product_smiles": ("[CH3:1][CH2:2][OH:3]"),
            "repair_goal": "replace the rejected first disconnection",
            "active_constraints": [],
            "durable_steps": [],
            "reconnect_boundaries": [],
            "reserved_atom_maps": [],
        },
    }
    records = runner._expand_seeded_branches_aizynthfinder(
        _spec(_context()),
        target="CCO",
        seeded=[branch],
        existing_records=[],
        route_quota=sequential_module._NodeCallBudget(
            model_invocations=20,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        critic_editor_call_reserve=0,
        critic_input_reserve=0,
        critic_output_reserve=0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            max_node_expansions_per_branch=5,
        ),
        started=time.monotonic(),
    )

    assert records == []
    assert branch["path_repair_builder_call_count"] == 0
    assert branch["route_call_count"] == 5
    assert branch["path_repair_aizynthfinder_search"]["failed"] is True


def test_transactional_path_repair_stages_rebuilt_mapped_boundary_until_recritic(
    monkeypatch,
) -> None:
    def editor_executor(task: WorkerTask) -> WorkerRunRecord:
        return replace(
            _proposal_record(
                {
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "repair_directive": {
                        "rollback_start_step_id": "route:2",
                        "rebuild_through_step_id": "route:3",
                        "additional_coupled_blocker_step_ids": [],
                        "preserved_suffix_compatible": True,
                        "repair_goal": "rebuild the left dependency subtree",
                        "active_constraints": [],
                    },
                },
                target="CCOC",
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    runner = SequentialStrategyDirectorRunner(
        editor_executor=editor_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )

    def rebuilt_search(_self, *, seeded, **_kwargs):
        branch = seeded[0]
        branch["steps"].extend(
            [
                {
                    "step_id": "repair:1",
                    "product_smiles": "CC",
                    "mapped_product_smiles": "[CH3:1][CH3:2]",
                    "precursor_smiles": ["C", "C"],
                    "mapped_precursor_smiles": ["[CH4:1]", "[CH4:2]"],
                    "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
                },
                {
                    "step_id": "repair:2",
                    "product_smiles": "C",
                    "mapped_product_smiles": "[CH4:1]",
                    "precursor_smiles": ["CCl"],
                    "mapped_precursor_smiles": ["[CH3:1][Cl:6]"],
                    "checkpoint_relation": "preparatory",
                    "reaction_operations": [
                        {
                            "op": "add_group",
                            "map_idx": 1,
                            "fragment_smiles": "[*][Cl:6]",
                        }
                    ],
                },
            ]
        )
        branch["path_repair_builder_call_count"] = (
            int(branch.get("path_repair_builder_call_count") or 0) + 2
        )
        branch["open_leaf_states"] = [{"smiles": "CCl", "mapped_smiles": "[CH3:1][Cl:6]"}]
        branch["open_leaves"] = ["CCl"]
        return []

    monkeypatch.setattr(
        runner,
        "_expand_seeded_branches_aizynthfinder",
        rebuilt_search,
    )
    original_steps = _branching_path_repair_route()
    branch = {
        "branch_index": 0,
        "steps": [dict(row) for row in original_steps],
        "target_mapped_smiles": "[CH3:1][CH2:2][O:3][CH3:4]",
        "strategy_card": {"strategy_query": "preserve a convergent split"},
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 3,
        "path_repair_builder_call_count": 0,
        "call_count": 4,
        "editor_attempt_count": 0,
        "editor_call_count": 0,
        "open_leaf_states": [],
        "open_leaves": [],
        "expanded_products": set(),
        "complete_in_bound_stock": False,
    }
    edited = runner._repair_branch_transactionally(
        _spec(_context()),
        target="CCOC",
        branch=branch,
        blocking_steps=[original_steps[2]],
        critique={
            "status": "reject",
            "step_assessments": [
                {
                    "step_id": "route:3",
                    "blocking": True,
                    "verdict": "reject",
                    "reasons": ["blocked chemistry"],
                }
            ],
        },
        iteration=0,
        records=[],
        max_prompt_bytes=100_000,
        max_node_call_timeout_s=60.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=20,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        started=time.monotonic(),
        reserve_model_invocations=0,
        reserve_input_tokens=0,
        reserve_output_tokens=0,
        reserve_wall_time_s=0.0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            max_node_expansions_per_branch=5,
            max_route_local_repair_rounds=6,
        ),
    )

    assert edited is True, (
        branch.get("path_repair_transactions"),
        branch.get("editor_rejection_diagnostics"),
    )
    assert [row["step_id"] for row in branch["steps"]] == [
        "route:1",
        "route:4",
        "repair:1",
        "repair:2",
    ]
    assert branch["editor_call_count"] == 1
    assert branch["path_repair_transactions"][-1]["status"] == ("rebuilt_pending_recritic")
    assert branch["path_repair_transactions"][-1]["replacement_step_replayed"] is True
    assert branch["complete_in_bound_stock"] is False
    assert branch.get("route_alternatives") in (None, [])

    rejected_branch = copy.deepcopy(branch)
    rolled_back = runner._rollback_pending_path_repair(
        rejected_branch,
        reason="path_repair_recritic_iteration_limit_reached",
        candidate_critique={
            "status": "reject",
            "overall_assessment": "reject",
            "critic_task_id": "critic:recheck",
            "step_assessments": [
                {
                    "step_id": "repair:1",
                    "blocking": True,
                    "verdict": "reject",
                    "reasons": ["the original blocker remains"],
                }
            ],
        },
    )

    assert rolled_back is True
    assert [row["step_id"] for row in rejected_branch["steps"]] == [
        "route:1",
        "route:2",
        "route:3",
        "route:4",
    ]
    assert rejected_branch["path_repair_transactions"][-1]["status"] == (
        "rolled_back_after_recritic"
    )
    # Rolling back the route state must not refund either policy-call axis.
    assert rejected_branch["route_call_count"] == 3
    assert rejected_branch["path_repair_builder_call_count"] == 2
    assert rejected_branch.get("route_alternatives") in (None, [])

    committed = runner._finalize_pending_path_repair(
        branch,
        {
            "status": "viable",
            "overall_assessment": "viable",
            "critic_task_id": "critic:recheck",
            "step_assessments": [],
        },
    )

    assert committed is True
    assert branch["path_repair_transactions"][-1]["status"] == ("committed_after_recritic")
    assert [row["step_id"] for row in branch["route_alternatives"][-1]["steps"]] == [
        "route:1",
        "route:2",
        "route:3",
        "route:4",
    ]


def test_transactional_path_repair_reuses_preserved_suffix_after_local_rebuild(
    monkeypatch,
) -> None:
    repair_call_ceilings: list[int] = []

    def editor_executor(task: WorkerTask) -> WorkerRunRecord:
        return replace(
            _proposal_record(
                {
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "repair_directive": {
                        "rollback_start_step_id": "linear:2",
                        "rebuild_through_step_id": "linear:3",
                        "additional_coupled_blocker_step_ids": [],
                        "preserved_suffix_compatible": True,
                        "repair_goal": "repair only the blocked two-step relay",
                        "active_constraints": [],
                    },
                },
                target="CCO",
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    runner = SequentialStrategyDirectorRunner(
        editor_executor=editor_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )

    def rebuilt_search(_self, *, seeded, config, **_kwargs):
        repair_call_ceilings.append(config.max_node_expansions_per_branch)
        repair_branch = seeded[0]
        repair_branch["steps"].extend(
            [
                {
                    "step_id": "repair:1",
                    "product_smiles": "CC",
                    "mapped_product_smiles": "[CH3:1][CH3:2]",
                    "reaction_operations": [
                        {
                            "op": "add_group",
                            "map_idx": 2,
                            "fragment_smiles": "[*][Cl:7]",
                        }
                    ],
                },
                {
                    "step_id": "repair:2",
                    "product_smiles": "CCCl",
                    "mapped_product_smiles": "[CH3:1][CH2:2][Cl:7]",
                    "reaction_operations": [
                        {
                            "op": "add_group",
                            "map_idx": 1,
                            "fragment_smiles": "[*][Br:8]",
                        }
                    ],
                },
            ]
        )
        repair_branch["path_repair_builder_call_count"] = (
            int(repair_branch.get("path_repair_builder_call_count") or 0) + 2
        )
        return []

    monkeypatch.setattr(
        runner,
        "_expand_seeded_branches_aizynthfinder",
        rebuilt_search,
    )
    original_steps = _linear_path_repair_route_with_suffix()
    branch = {
        "branch_index": 0,
        "steps": [dict(row) for row in original_steps],
        "target_mapped_smiles": "[CH3:1][CH2:2][OH:3]",
        "strategy_card": {"strategy_query": "preserve the local relay"},
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 3,
        "path_repair_builder_call_count": 0,
        "call_count": 4,
        "editor_attempt_count": 0,
        "editor_call_count": 0,
        "open_leaf_states": [],
        "open_leaves": [],
        "expanded_products": set(),
        "complete_in_bound_stock": False,
    }

    edited = runner._repair_branch_transactionally(
        _spec(_context()),
        target="CCO",
        branch=branch,
        blocking_steps=[original_steps[2]],
        critique={
            "status": "reject",
            "step_assessments": [
                {
                    "step_id": "linear:3",
                    "blocking": True,
                    "verdict": "reject",
                    "reasons": ["blocked chemistry"],
                }
            ],
        },
        iteration=0,
        records=[],
        max_prompt_bytes=100_000,
        max_node_call_timeout_s=60.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=20,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        started=time.monotonic(),
        reserve_model_invocations=0,
        reserve_input_tokens=0,
        reserve_output_tokens=0,
        reserve_wall_time_s=0.0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            max_node_expansions_per_branch=5,
            max_route_local_repair_rounds=6,
        ),
    )

    assert edited is True, (
        branch.get("path_repair_transactions"),
        branch.get("editor_rejection_diagnostics"),
    )
    assert [row["step_id"] for row in branch["steps"]] == [
        "linear:1",
        "repair:1",
        "repair:2",
        "linear:4",
    ]
    transaction = branch["path_repair_transactions"][-1]
    assert transaction["status"] == "rebuilt_pending_recritic"
    assert transaction["preserved_suffix_step_ids"] == ["linear:4"]
    assert transaction["suffix_stitch"]["suffix_stitched"] is True
    assert transaction["suffix_stitch"]["remapped_boundary_atom_count"] == 2
    # Initial and repair calls retain separate attribution counters while
    # reading the same configured per-branch expansion ceiling.
    assert branch["route_call_count"] == 3
    assert branch["path_repair_builder_call_count"] == 2
    assert repair_call_ceilings == [5]


def test_transactional_path_repair_sends_replayed_preparatory_step_to_recritic(
    monkeypatch,
) -> None:
    def editor_executor(task: WorkerTask) -> WorkerRunRecord:
        return replace(
            _proposal_record(
                {
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "repair_directive": {
                        "rollback_start_step_id": "route:2",
                        "rebuild_through_step_id": "route:3",
                        "additional_coupled_blocker_step_ids": [],
                        "preserved_suffix_compatible": True,
                        "repair_goal": "replace the blocked left branch",
                        "active_constraints": [],
                    },
                },
                target="CCOC",
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    runner = SequentialStrategyDirectorRunner(
        editor_executor=editor_executor,
        stock_membership=lambda values: {str(value): False for value in values},
    )

    def preparatory_search(_self, *, seeded, **_kwargs):
        repair_branch = seeded[0]
        repair_branch["steps"].append(
            {
                "step_id": "repair:1",
                "product_smiles": "CC",
                "mapped_product_smiles": "[CH3:1][CH3:2]",
                "precursor_smiles": ["C", "C"],
                "mapped_precursor_smiles": ["[CH4:1]", "[CH4:2]"],
                "checkpoint_relation": "preparatory",
                "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
            }
        )
        repair_branch["path_repair_builder_call_count"] = (
            int(repair_branch.get("path_repair_builder_call_count") or 0) + 1
        )
        repair_branch["open_leaf_states"] = [
            {"smiles": "C", "mapped_smiles": "[CH4:1]"},
            {"smiles": "C", "mapped_smiles": "[CH4:2]"},
        ]
        repair_branch["open_leaves"] = ["C", "C"]
        return []

    monkeypatch.setattr(
        runner,
        "_expand_seeded_branches_aizynthfinder",
        preparatory_search,
    )
    original_steps = _branching_path_repair_route()
    branch = {
        "branch_index": 0,
        "steps": [dict(row) for row in original_steps],
        "target_mapped_smiles": "[CH3:1][CH2:2][O:3][CH3:4]",
        "strategy_card": {"strategy_query": "preserve a convergent split"},
        "strategy_tree_engine": "aizynthfinder_mcts",
        "route_call_count": 4,
        "path_repair_builder_call_count": 0,
        "call_count": 4,
        "editor_attempt_count": 0,
        "editor_call_count": 0,
        "open_leaf_states": [],
        "open_leaves": [],
        "expanded_products": set(),
        "complete_in_bound_stock": False,
    }

    edited = runner._repair_branch_transactionally(
        _spec(_context()),
        target="CCOC",
        branch=branch,
        blocking_steps=[original_steps[2]],
        critique={
            "status": "reject",
            "step_assessments": [
                {
                    "step_id": "route:3",
                    "blocking": True,
                    "verdict": "reject",
                    "reasons": ["blocked chemistry"],
                }
            ],
        },
        iteration=0,
        records=[],
        max_prompt_bytes=100_000,
        max_node_call_timeout_s=60.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=20,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        started=time.monotonic(),
        reserve_model_invocations=0,
        reserve_input_tokens=0,
        reserve_output_tokens=0,
        reserve_wall_time_s=0.0,
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            strategy_tree_engine="aizynthfinder_mcts",
            strategy_branch_count=1,
            max_node_expansions_per_branch=5,
            max_route_local_repair_rounds=6,
        ),
    )

    assert edited is True
    assert [row["step_id"] for row in branch["steps"]] == [
        "route:1",
        "route:4",
        "repair:1",
    ]
    transaction = branch["path_repair_transactions"][-1]
    assert transaction["status"] == "rebuilt_pending_recritic"
    assert transaction["replacement_step_replayed"] is True
    assert "reason" not in transaction
    assert branch.get("_pending_path_repair_transaction")
    assert branch.get("editor_rejection_diagnostics") in (None, [])


def test_pending_path_repair_recritic_reject_restores_authoritative_route() -> None:
    editor_calls = 0

    def unexpected_editor(_task: WorkerTask) -> WorkerRunRecord:
        nonlocal editor_calls
        editor_calls += 1
        raise AssertionError("rejected provisional route must not reach Editor")

    original_steps = _branching_path_repair_route()
    candidate_steps = [
        {
            "step_id": "repair:1",
            "product_smiles": "CCOC",
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        }
    ]
    original_critique = {
        "status": "reject",
        "overall_assessment": "reject",
        "critic_task_id": "critic:original",
        "step_assessments": [
            {
                "step_id": "route:3",
                "blocking": True,
                "verdict": "reject",
                "reasons": ["original blocker"],
            }
        ],
    }
    branch = {
        "branch_index": 0,
        "steps": candidate_steps,
        "target_mapped_smiles": "[CH3:1][CH2:2][O:3][CH3:4]",
        "strategy_card": {"strategy_query": "preserve the route"},
        "chemical_critic": dict(original_critique),
        "path_repair_transactions": [
            {"status": "rebuilt_pending_recritic", "editor_task_id": "editor:1"}
        ],
        "_pending_path_repair_transaction": {
            "route_snapshot": {"steps": copy.deepcopy(original_steps)},
            "original_critique": dict(original_critique),
            "transaction_indices": [0],
            "editor_task_ids": ["editor:1"],
        },
    }
    runner = SequentialStrategyDirectorRunner(
        critic_executor=lambda task: _blocking_critic_record(
            task,
            step_id="repair:1",
        ),
        editor_executor=unexpected_editor,
    )

    runner._run_codex_critics(
        _spec(_context()),
        _context(),
        [branch],
        [],
        quota=sequential_module._NodeCallBudget(
            model_invocations=10,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_time_s=600.0,
        ),
        started=time.monotonic(),
        config=DirectorConfig(
            planning_mode="sequential_branches",
            paper_matched_reach_profile=True,
            enable_transactional_path_repair=True,
            require_complete_route_json=False,
            max_route_local_repair_rounds=6,
        ),
    )

    assert [row["step_id"] for row in branch["steps"]] == [
        "route:1",
        "route:2",
        "route:3",
        "route:4",
    ]
    assert branch["path_repair_transactions"][0]["status"] == ("rolled_back_after_recritic")
    assert branch["chemical_critic"]["critic_task_id"] == "critic:original"
    assert branch["chemical_critic"]["candidate_recritic"]["status"] == "reject"
    assert branch["chemical_critic"]["path_repair_failure_reason"] == (
        "path_repair_recritic_rejected"
    )
    assert editor_calls == 0


def test_editor_replace_span_preserves_prefix_and_reconnects_suffix() -> None:
    current = _three_step_editor_route()
    revised = {
        **current[1],
        "reaction_family": "revised C-C disconnection",
    }
    candidate = {
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "replace_span": {
            "remove_step_ids": ["route:2"],
            "revised_steps": [revised],
        },
    }

    expansions, diagnostic, mode = _editor_route_expansions_from_record(
        _proposal_record(candidate, target="CCCO"),
        current_steps=current,
        mapped_target_smiles="[CH3:1][CH2:2][CH2:3][OH:4]",
        expected_target_smiles="CCCO",
    )

    assert diagnostic == {}
    assert mode == "replace_span"
    assert expansions is not None
    assert [row.step_id for row in expansions] == [
        "route:1",
        "route:2",
        "route:3",
    ]
    assert expansions[1].reaction_family == "revised C-C disconnection"
    assert expansions[2].product_smiles == "CC"


def test_editor_replace_span_rejects_unreconnected_suffix() -> None:
    current = _three_step_editor_route()
    revised = {
        **current[1],
        "reaction_operations": [{"op": "change_bond_order", "map_a": 1, "map_b": 2, "delta": 1}],
    }
    candidate = {
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "replace_span": {
            "remove_step_ids": ["route:2"],
            "revised_steps": [revised],
        },
    }

    expansions, diagnostic, mode = _editor_route_expansions_from_record(
        _proposal_record(candidate, target="CCCO"),
        current_steps=current,
        mapped_target_smiles="[CH3:1][CH2:2][CH2:3][OH:4]",
        expected_target_smiles="CCCO",
    )

    assert expansions is None
    assert mode == "replace_span"
    assert diagnostic["reason"] == "route_json_chain_invalid"
    assert diagnostic["step_index"] == 2
    assert diagnostic["failed_step_id"] == "route:3"
    assert diagnostic["editor_mutation_mode"] == "replace_span"


def test_editor_replace_span_can_cover_multiple_rows_or_whole_route() -> None:
    current = _three_step_editor_route()
    for remove_ids, revised in (
        (["route:2", "route:3"], current[1:]),
        (["route:1", "route:2", "route:3"], current),
    ):
        merged, reason = sequential_module._apply_replace_span(
            current,
            {
                "remove_step_ids": remove_ids,
                "revised_steps": revised,
            },
        )

        assert reason == ""
        assert merged is not None
        assert [row["step_id"] for row in merged] == [
            "route:1",
            "route:2",
            "route:3",
        ]


def test_editor_replace_span_host_rejects_duplicate_remove_step_ids() -> None:
    current = _three_step_editor_route()

    merged, reason = sequential_module._apply_replace_span(
        current,
        {
            "remove_step_ids": ["route:2", "route:2"],
            "revised_steps": [current[1]],
        },
    )

    assert merged is None
    assert reason == "editor_replace_span_remove_step_ids_duplicate"


def test_editor_route_patch_is_recompiled_from_target_and_preserves_host_maps() -> None:
    current = _complete_route_candidate()["route_json"]
    candidate = {
        **_complete_route_candidate(),
        "route_json": None,
        "route_patch": [
            {
                "op": "replace_step",
                "step_id": "route:2",
                "after_step_id": "",
                "step_ids": [],
                "step": {
                    **current[1],
                    "conditions": ["aqueous workup"],
                },
            }
        ],
    }
    expansions, diagnostic, mode = _editor_route_expansions_from_record(
        _proposal_record(candidate),
        current_steps=current,
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        expected_target_smiles="CCO",
    )

    assert diagnostic == {}
    assert mode == "route_patch"
    assert expansions is not None
    assert expansions[1].mapped_product_smiles == "[CH3:1][CH3:2]"
    assert expansions[1].conditions == ("aqueous workup",)


def test_editor_replace_patch_preserves_omitted_reaction_operations() -> None:
    current = _complete_route_candidate()["route_json"]
    replacement = dict(current[1])
    replacement.pop("reaction_operations", None)
    replacement["conditions"] = ["updated workup"]
    candidate = {
        **_complete_route_candidate(),
        "route_json": None,
        "route_patch": [
            {
                "op": "replace_step",
                "step_id": "route:2",
                "step": replacement,
            }
        ],
    }

    expansions, diagnostic, mode = _editor_route_expansions_from_record(
        _proposal_record(candidate),
        current_steps=current,
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        expected_target_smiles="CCO",
    )

    assert diagnostic == {}
    assert mode == "route_patch"
    assert expansions is not None
    assert expansions[1].conditions == ("updated workup",)
    assert expansions[1].reaction_operations


def test_editor_set_conditions_patch_preserves_structure_and_atom_maps() -> None:
    current = _complete_route_candidate()["route_json"]
    original_operations = list(current[1]["reaction_operations"])
    candidate = {
        **_complete_route_candidate(),
        "route_json": None,
        "route_patch": [
            {
                "op": "set_conditions",
                "step_id": "route:2",
                "after_step_id": "",
                "step_ids": [],
                "step": None,
                "conditions": ["aqueous base", "0-20 C"],
                "catalyst": "phase-transfer catalyst",
            }
        ],
    }

    expansions, diagnostic, mode = _editor_route_expansions_from_record(
        _proposal_record(candidate),
        current_steps=current,
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        expected_target_smiles="CCO",
    )

    assert diagnostic == {}
    assert mode == "route_patch"
    assert expansions is not None
    assert expansions[1].conditions == ("aqueous base", "0-20 C")
    assert expansions[1].catalyst == "phase-transfer catalyst"
    assert list(expansions[1].reaction_operations) == original_operations
    assert expansions[1].mapped_product_smiles == "[CH3:1][CH3:2]"


def test_editor_complete_route_compiles_a_target_rooted_dag_not_a_linear_chain() -> None:
    route = [
        {
            "step_id": "root",
            "product_smiles": "CCOC",
            "mapped_product_smiles": "[CH3:1][CH2:2][O:3][CH3:4]",
            "precursor_smiles": [],
            "reaction_family": "ether disconnection",
            "transformation_rationale": "expose two sibling branches",
            "conditions": ["base"],
            "catalyst": "",
            "limitations": [],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        },
        {
            "step_id": "left",
            "product_smiles": "CC",
            "mapped_product_smiles": "[CH3:1][CH3:2]",
            "precursor_smiles": [],
            "reaction_family": "left branch",
            "transformation_rationale": "expand the carbon sibling",
            "conditions": [],
            "catalyst": "",
            "limitations": [],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
        },
        {
            "step_id": "right",
            "product_smiles": "CO",
            "mapped_product_smiles": "[OH:3][CH3:4]",
            "precursor_smiles": [],
            "reaction_family": "right branch",
            "transformation_rationale": "expand the oxygen sibling",
            "conditions": [],
            "catalyst": "",
            "limitations": [],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": [{"op": "break_bond", "map_a": 3, "map_b": 4}],
        },
    ]
    candidate = {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": "dag-editor",
        "product_smiles": "CCOC",
        "precursor_smiles": [],
        "reaction_family": "dependency-closed edit",
        "transformation_rationale": "preserve both sibling branches",
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "reaction_operations": [],
        "route_json": route,
        "route_patch": [],
    }

    expansions, diagnostic, mode = _editor_route_expansions_from_record(
        _proposal_record(candidate, target="CCOC"),
        current_steps=route,
        mapped_target_smiles="[CH3:1][CH2:2][O:3][CH3:4]",
        expected_target_smiles="CCOC",
    )

    assert diagnostic == {}
    assert mode == "full_route_json"
    assert expansions is not None
    assert len(expansions) == 3
    assert expansions[2].mapped_product_smiles == "[OH:3][CH3:4]"


def test_editor_patch_uses_host_map_fragment_when_replay_arrays_are_reordered() -> None:
    target = "CCOC(=O)C1(O)c2cc3c(cc2C(=O)C1(C)O)C1(CC(=O)C[C@@H](C)O1)O[C@H](C)C3"
    current = [
        {
            "step_id": "step-1",
            "product_smiles": target,
            "precursor_smiles": [],
            "reaction_operations": [
                {"op": "break_bond", "map_a": 19, "map_b": 20},
                {"op": "break_bond", "map_a": 19, "map_b": 26},
            ],
        },
        {
            "step_id": "step-2",
            "product_smiles": "CC(=O)C[C@@H](C)O",
            "precursor_smiles": [],
            "reaction_operations": [{"op": "break_bond", "map_a": 21, "map_b": 23}],
        },
    ]
    replacement = dict(current[1])
    replacement["product_smiles"] = "[CH3:20][C:21](=[O:22])[CH2:23][C@@H:24]([CH3:25])[OH:26]"
    candidate = {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": "map-order-repair",
        "product_smiles": target,
        "precursor_smiles": [],
        "reaction_family": "host replay repair",
        "transformation_rationale": "preserve mapped host fragment",
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "reaction_operations": current[0]["reaction_operations"],
        "route_json": None,
        "route_patch": [
            {
                "op": "replace_step",
                "step_id": "step-2",
                "step": replacement,
            }
        ],
    }

    expansions, diagnostic, mode = _editor_route_expansions_from_record(
        _proposal_record(candidate, target=target),
        current_steps=current,
        mapped_target_smiles=sequential_module._mapped_smiles(target),
        expected_target_smiles=target,
    )

    assert diagnostic == {}
    assert mode == "route_patch"
    assert expansions is not None
    assert len(expansions) == 2
    assert expansions[1].reaction_operations


def test_editor_route_failure_returns_exact_host_chain_diagnostic() -> None:
    candidate = _complete_route_candidate()
    candidate["route_json"][1]["product_smiles"] = "CO"

    expansions, diagnostic, mode = _editor_route_expansions_from_record(
        _proposal_record(candidate),
        current_steps=_complete_route_candidate()["route_json"],
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        expected_target_smiles="CCO",
    )

    assert expansions is None
    assert mode == "full_route_json"
    assert diagnostic["reason"] == "route_json_chain_invalid"
    assert diagnostic["step_index"] == 1
    assert diagnostic["failed_step_id"] == "route:2"
    assert diagnostic["detail"] == "product_not_in_open_precursors"
    assert diagnostic["compiler_mode"] == "target_rooted_route_dag"
    assert diagnostic["editor_mutation_mode"] == "full_route_json"


def test_editor_map_failure_returns_real_mapped_open_precursor_and_retry_draft() -> None:
    candidate = _complete_route_candidate()
    candidate["route_json"][1]["mapped_product_smiles"] = "[CH3:18][CH3:19]"
    candidate["route_json"][1]["reaction_operations"] = [
        {"op": "break_bond", "map_a": 18, "map_b": 19}
    ]
    record = _proposal_record(candidate)

    expansions, diagnostic, mode = _editor_route_expansions_from_record(
        record,
        current_steps=_complete_route_candidate()["route_json"],
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        expected_target_smiles="CCO",
    )

    assert expansions is None
    assert mode == "full_route_json"
    assert diagnostic["step_index"] == 1
    assert "reactionjson_map_not_found" in diagnostic["compiler_error"]
    assert diagnostic["host_replayed_prefix_step_count"] == 1
    assert diagnostic["host_selected_open_precursor"] == {
        "product_smiles": "CC",
        "mapped_product_smiles": "[CH3:1][CH3:2]",
    }
    assert diagnostic["mapped_open_precursor_authority"] == ("host_routejson_dag_compiler")

    retry = sequential_module._editor_retry_route_rows(
        record,
        diagnostic=diagnostic,
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
    )
    assert retry is not None
    assert retry[0]["mapped_precursor_smiles"] == [
        "[CH3:1][CH3:2]",
        "[OH2:3]",
    ]
    assert retry[1]["mapped_product_smiles"] == "[CH3:1][CH3:2]"
    assert retry[1]["precursor_smiles"] == []
    assert retry[1]["mapped_precursor_smiles"] == []


def test_routejson_admission_rejects_cross_step_atom_map_namespace_break() -> None:
    steps = [
        {
            "product_smiles": "CCO",
            "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        },
        {
            "product_smiles": "CC",
            "mapped_product_smiles": "[CH3:18][CH3:19]",
            "reaction_operations": [{"op": "break_bond", "map_a": 18, "map_b": 19}],
        },
    ]

    invalid = sequential_module._route_steps_host_replay_validation(
        steps,
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
    )
    valid = sequential_module._route_steps_host_replay_validation(
        [
            steps[0],
            {
                **steps[1],
                "mapped_product_smiles": "[CH3:1][CH3:2]",
                "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
            },
        ],
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
    )

    assert invalid["complete"] is False
    assert invalid["step_index"] == 1
    assert "reactionjson_map_not_found" in invalid["compiler_error"]
    assert valid["complete"] is True
    assert valid["compiled_step_count"] == 2


def test_routejson_failure_reports_nonaromatic_aromatic_bond_exactly() -> None:
    invalid = sequential_module._route_steps_host_replay_validation(
        [
            {
                "product_smiles": "CCC",
                "reaction_operations": [
                    {
                        "op": "add_bond",
                        "map_a": 1,
                        "map_b": 3,
                        "order": 1.5,
                    }
                ],
            }
        ],
        mapped_target_smiles="[CH3:1][CH2:2][CH3:3]",
    )

    assert invalid["complete"] is False
    assert invalid["step_index"] == 0
    assert invalid["compiler_error"] == ("reactionjson_aromatic_bond_requires_aromatic_atoms")
    assert invalid["operation_index"] == 0
    assert invalid["failed_operation"] == {
        "op": "add_bond",
        "map_a": 1,
        "map_b": 3,
        "order": 1.5,
    }
    assert invalid["endpoint_aromaticity"] == {"map_a": False, "map_b": False}
    assert invalid["allowed_orders"] == [1, 2, 3]


def test_rejected_routejson_metadata_draft_still_reaches_host_diagnostic() -> None:
    candidate = _complete_route_candidate()
    candidate["route_json"][1]["product_smiles"] = "not-a-smiles"
    record = replace(
        _proposal_record(candidate),
        status="rejected_output",
        output_validation={
            "accepted": False,
            "reasons": [
                "missing_evidence_or_input_refs",
                "proposal_report_candidate:0:invalid_precursor_smiles",
            ],
        },
    )

    expansions, diagnostic, mode = _editor_route_expansions_from_record(
        record,
        current_steps=_complete_route_candidate()["route_json"],
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        expected_target_smiles="CCO",
    )

    assert expansions is None
    assert mode == "full_route_json"
    assert diagnostic["reason"] == "route_json_step_invalid"
    assert diagnostic["step_index"] == 1
    assert diagnostic["detail"] == "product_smiles_invalid"


def test_editor_scaffold_replaces_redrawn_stereo_with_host_precursor() -> None:
    mapped_target = "[CH3:1][C@:2]([F:3])([Cl:4])[CH2:5][OH:6]"
    rows = [
        {
            "step_id": "step-1",
            "product_smiles": "CC(F)(Cl)CO",
            "reaction_operations": [
                {"op": "clear_stereocenter", "map_idx": 2},
                {"op": "break_bond", "map_a": 2, "map_b": 5},
            ],
        },
        {
            "step_id": "step-2",
            "product_smiles": "C[C@H](F)Cl",
            "reaction_operations": [],
        },
    ]

    scaffold = _editor_route_scaffold(
        rows,
        mapped_target_smiles=mapped_target,
    )

    assert scaffold[1]["product_smiles"] == "CC(F)Cl"


def test_editor_patch_supports_insert_delete_and_reorder_operations() -> None:
    rows = [
        {"step_id": "a", "product_smiles": "CCO", "reaction_operations": []},
        {"step_id": "b", "product_smiles": "CC", "reaction_operations": []},
    ]
    patched, reason = _apply_route_patch(
        rows,
        [
            {
                "op": "insert_after",
                "step_id": "",
                "after_step_id": "a",
                "step_ids": [],
                "step": {"step_id": "c", "product_smiles": "C", "reaction_operations": []},
            },
            {
                "op": "delete_step",
                "step_id": "b",
                "after_step_id": "",
                "step_ids": [],
                "step": None,
            },
            {
                "op": "reorder",
                "step_id": "",
                "after_step_id": "",
                "step_ids": ["c", "a"],
                "step": None,
            },
        ],
    )

    assert reason == ""
    assert [row["step_id"] for row in patched or []] == ["c", "a"]


def test_independent_codex_critic_runs_before_plan_delivery() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=3,
        max_route_families=6,
        max_skeletons=6,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=3,
        max_node_expansions_per_branch=1,
    )
    observed = []

    def executor(task):
        observed.append(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)
        return _fake_executor(task)

    result = SequentialStrategyDirectorRunner(node_executor=executor)(
        _spec(context), context, "initial_architecture", config
    )

    assert result.state is AgentState.SUCCEEDED
    assert result.usage["model_invocations"] == 9
    critic_tasks = [
        task for task in observed if task.required_artifact_type == "ChemicalStrategyCritique"
    ]
    assert len(critic_tasks) == 3
    assert all(task.allowed_tools == [] and task.input_refs == [] for task in critic_tasks)
    assert all(context.run_id not in task.case_id for task in critic_tasks)
    assert all("independent_chemical_critic" in task.objective for task in critic_tasks)
    assert all(task.budget.reasoning_effort == "medium" for task in critic_tasks)
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert all(
        dict(family.get("chemical_critic") or {}).get("status") == "viable"
        for family in plan.route_families
    )


def test_aiz_selected_solution_requires_complete_host_projection_and_leaf_closure() -> None:
    step = {
        "step_id": "route:1",
        "product_smiles": "CCO",
        "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
        "precursor_smiles": ["CC", "O"],
        "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
        "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
    }

    complete = sequential_module._materialize_aizynthfinder_projection(
        steps=[step],
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        search_diagnostics={
            "path_action_count": 1,
            "path_route_step_count": 1,
            "path_route_projection_complete": True,
        },
        stock_membership=lambda values: {str(value): str(value) in {"CC", "O"} for value in values},
    )
    missing_action = sequential_module._materialize_aizynthfinder_projection(
        steps=[step],
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        search_diagnostics={
            "path_action_count": 2,
            "path_route_step_count": 1,
            "path_route_projection_complete": False,
        },
        stock_membership=lambda values: {str(value): True for value in values},
    )

    assert complete["route_projection_complete"] is True
    assert complete["leaf_closure_complete"] is True
    assert complete["terminal_leaf_count"] == 2
    assert missing_action["routejson_replay_validation"]["complete"] is True
    assert missing_action["route_projection_complete"] is False
    assert missing_action["leaf_closure_complete"] is False
    assert (
        sequential_module._branch_stock_closed(
            {
                "steps": missing_action["steps"],
                "open_leaves": [],
                "aizynthfinder_strategy_search": {
                    "canonical_route_projection_complete": False,
                    "canonical_leaf_closure_complete": False,
                },
            }
        )
        is False
    )


def test_paper_strategy_branch_is_executed_inside_aiz_mcts_sidecar() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=1,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=1,
        require_strategy_graph_edits=True,
    )
    observed_route_calls = []

    def executor(task):
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        observed_route_calls.append(task)
        candidate = {
            "schema_version": "retrosynthesis_candidate.v1",
            "candidate_id": task.task_id,
            "product_smiles": "CCO",
            "precursor_smiles": [],
            "reaction_family": "C-O disconnection",
            "product_retron_type": "C-O",
            "transformation_rationale": "host replay canary",
            "conditions": [],
            "catalyst": "",
            "enzyme": "",
            "limitations": [],
            "required_validation": ["structure"],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        }
        return replace(
            _proposal_record(candidate),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=_fake_critic_executor,
        aizynthfinder_strategy_inline_stock_smiles=("CC", "O"),
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert len(observed_route_calls) == 1
    plan = GlobalCampaignPlan.from_dict(result.output)
    family = dict(plan.route_families[0])
    assert family["strategy_tree_engine"] == "aizynthfinder_mcts"
    search = dict(family["aizynthfinder_strategy_search"])
    assert search["engine"] == "AiZynthFinder.MctsSearchTree"
    assert search["selected_solved"] is True
    assert search["provider_callback_count"] == 1
    assert "reported_policy_calls" not in search
    assert plan.multi_step_skeletons[0]["routejson_replay_complete"] is True


def test_aiz_branch_adds_next_strategy_on_exact_upstream_leaf(monkeypatch) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=2,
        max_strategic_milestones_per_branch=2,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=1,
        require_strategy_graph_edits=True,
    )
    observed = []

    def executor(task):
        observed.append(task)
        if task.required_artifact_type == "StrategyCardReport":
            local = '"selected_upstream_leaf":"CO"' in task.objective
            card = _strategy_card(1)
            target = "CO" if local else "CCO"
            if local:
                card = {
                    **card,
                    "key_forward_transformation": "upstream C-C fragment union",
                    "key_bond_changes": ["map 2-map 3"],
                    "skeleton_change_class": "upstream convergent fragment union",
                    "strategy_signature": "upstream-cc-fragment-union",
                }
            record = _strategy_record(task)
            artifact = dict(record.output_artifact)
            payload = dict(artifact["payload"])
            payload["target_smiles"] = target
            payload["strategy_card"] = card
            artifact["payload"] = payload
            return replace(record, output_artifact=artifact)
        product = json.loads(
            re.search(r"CompactBranchContext:\n(\{.*\})", task.objective).group(1)
        )["selected_open_leaf"]
        operations = (
            [{"op": "break_bond", "map_a": 2, "map_b": 3}]
            if product == "CO"
            else [{"op": "break_bond", "map_a": 1, "map_b": 2}]
        )
        candidate = {
            "schema_version": "retrosynthesis_candidate.v1",
            "candidate_id": task.task_id,
            "product_smiles": product,
            "precursor_smiles": [],
            "reaction_family": "host-replayed strategic cleavage",
            "product_retron_type": "skeletal bond",
            "transformation_rationale": "two-milestone canary",
            "conditions": [],
            "catalyst": "",
            "enzyme": "",
            "limitations": [],
            "required_validation": ["structure"],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": operations,
        }
        return replace(
            _proposal_record(candidate, target=product),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        first = request_handler(
            {
                "expandable_smiles": ["CCO"],
                "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
                "route_steps": [],
            }
        )["candidates"][0]
        upstream_index = first["precursor_smiles"].index("CO")
        second = request_handler(
            {
                "expandable_smiles": ["CO"],
                "expandable_mapped_smiles": [first["mapped_precursor_smiles"][upstream_index]],
                "route_steps": [first["route_step"]],
            }
        )["candidates"][0]
        return {
            "route_steps": [first["route_step"], second["route_step"]],
            "open_leaf_states": [
                {"smiles": "C", "mapped_smiles": "[CH4:1]"},
                {"smiles": "C", "mapped_smiles": "[CH4:2]"},
                {"smiles": "O", "mapped_smiles": "[OH2:3]"},
            ],
            "solved": True,
            "policy_calls": 2,
            "mcts_iterations": 2,
            "diagnostics": {"engine": "AiZynthFinder.MctsSearchTree"},
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=_fake_critic_executor,
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert (
        len([task for task in observed if task.required_artifact_type == "StrategyCardReport"]) == 2
    )
    plan = GlobalCampaignPlan.from_dict(result.output)
    family = dict(plan.route_families[0])
    skeleton = dict(plan.multi_step_skeletons[0])
    assert family["strategic_milestone_count"] == 2
    assert len(family["strategy_milestone_cards"]) == 2
    assert [step["strategy_milestone_index"] for step in skeleton["steps"]] == [1, 2]
    assert all(step["strategy_anchor"] is True for step in skeleton["steps"])
    assert result.usage["upstream_strategy_milestone_calls"] == 1


def test_paper_matched_call_ceiling_does_not_require_budget_exhaustion(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=1,
        max_skeletons=1,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=25,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=6,
        require_strategy_graph_edits=True,
        require_complete_route_json=True,
        allow_editor_route_mutations=True,
    )

    def fake_sidecar(*, max_policy_calls, **_kwargs):
        assert max_policy_calls == 25
        # Reproduce the v5 failure: the provider loop increments its callback
        # counter although the host never entered a paid policy worker.
        return {
            "route_steps": [],
            "open_leaf_states": [
                {
                    "smiles": "CCO",
                    "mapped_smiles": "[CH3:1][CH2:2][OH:3]",
                }
            ],
            "solved": False,
            "policy_calls": 0,
            "mcts_iterations": 25,
            "diagnostics": {
                "engine": "AiZynthFinder.MctsSearchTree",
                "provider_callback_count": 25,
                "calls_exhausted": False,
                "host_stop_requested": True,
                "host_stop_reason": "route_builder_output_token_allocation_exhausted",
                "selected_open_leaves": 1,
            },
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=_fake_executor,
        critic_executor=_fake_critic_executor,
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.FAILED
    assert "paper_policy_call_budget_not_exhausted" not in result.error
    budget = result.usage["paper_policy_call_budget"]
    assert budget["actual_calls"] == [0]
    assert budget["branch_summaries"][0]["provider_callback_count"] == 25
    assert budget["branch_summaries"][0]["calls_exhausted"] is False
    assert budget["hard_failures"] == []


def test_paper_portfolio_preserves_usable_branch_when_one_sidecar_fails(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=5,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=3,
        strategy_branch_workers=3,
        max_node_expansions_per_branch=1,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=0,
        require_strategy_graph_edits=True,
        require_complete_route_json=True,
    )

    def executor(task):
        if task.required_artifact_type == "StrategyPortfolioReport":
            return _strategy_portfolio_record(task)
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        product = str(task.host_context.get("selected_product") or "")
        assert product
        candidate = {
            "candidate_id": task.task_id,
            "product_smiles": product,
            "precursor_smiles": [],
            "reaction_family": "C-O bond disconnection",
            "conditions": ["test conditions"],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        }
        return replace(
            _proposal_record(candidate, target=product),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, strategy_text, request_handler, **_kwargs):
        if "branch-3" in strategy_text:
            raise RuntimeError("synthetic isolated sidecar failure")
        candidate = request_handler(
            {
                "expandable_smiles": ["CCO"],
                "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
                "route_steps": [],
            }
        )["candidates"][0]
        return {
            "route_steps": [candidate["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    candidate["precursor_smiles"],
                    candidate["mapped_precursor_smiles"],
                )
            ],
            "solved": False,
            "policy_calls": 1,
            "mcts_iterations": 1,
            "diagnostics": {"engine": "AiZynthFinder.MctsSearchTree"},
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=_fake_critic_executor,
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED, (
        result.error,
        result.usage.get("paper_policy_call_budget"),
    )
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert len(plan.route_families) == 3
    assert len(plan.multi_step_skeletons) == 2
    assert all(
        int(
            dict(family["shared_model_budget_ledger"].get("quota") or {}).get("model_invocations")
            or 0
        )
        > 0
        for family in plan.route_families
    )
    advisory = next(
        row for row in plan.route_families if row["route_family_id"] == "codex:sequential:family:3"
    )
    assert advisory["route_status"] == "hypothesis_only_materialization_pending"
    assert result.usage["paper_policy_partial_branch_failure"] is True
    assert result.usage["branch_route_retention"][2]["retained_as_replayable_route"] is False
    failures = result.usage["paper_policy_call_budget"]["hard_failures"]
    assert len(failures) == 1
    assert failures[0]["reason"] == "paper_strategy_sidecar_failed"
    assert "synthetic isolated sidecar failure" in failures[0]["detail"]


def test_sidecar_failure_recovers_deepest_host_replayed_prefix(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=1,
        max_skeletons=1,
        max_steps_per_skeleton=2,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=2,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=0,
        require_complete_route_json=True,
    )

    def executor(task):
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        product = str(task.host_context.get("selected_product") or "")
        operation = (
            {"op": "break_bond", "map_a": 2, "map_b": 3}
            if product == "CCO"
            else {"op": "break_bond", "map_a": 1, "map_b": 2}
        )
        return replace(
            _proposal_record(
                {
                    "candidate_id": task.task_id,
                    "product_smiles": product,
                    "precursor_smiles": [],
                    "reaction_family": "host replay recovery canary",
                    "conditions": ["test conditions"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "reaction_operations": [operation],
                },
                target=product,
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def failing_sidecar(*, request_handler, **_kwargs):
        first = request_handler(
            {
                "expandable_smiles": ["CCO"],
                "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
                "route_steps": [],
            }
        )["candidates"][0]
        upstream_index = first["precursor_smiles"].index("CC")
        request_handler(
            {
                "expandable_smiles": ["CC"],
                "expandable_mapped_smiles": [first["mapped_precursor_smiles"][upstream_index]],
                "route_steps": [first["route_step"]],
            }
        )
        raise RuntimeError("synthetic failure after durable prefix")

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        failing_sidecar,
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=_fake_critic_executor,
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED, result.error
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert len(plan.multi_step_skeletons[0]["steps"]) == 1
    family = dict(plan.route_families[0])
    assert family["sidecar_recovered_prefix"] is True
    assert family["paper_policy_budget_failure"]["reason"] == ("paper_strategy_sidecar_failed")
    assert (
        "synthetic failure after durable prefix" in family["paper_policy_budget_failure"]["detail"]
    )
    assert result.usage["paper_policy_partial_branch_failure"] is True


def test_rejected_key_event_keeps_critic_until_third_candidate_passes(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=1,
        max_skeletons=1,
        max_steps_per_skeleton=3,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=3,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=0,
        require_strategy_graph_edits=True,
        require_complete_route_json=True,
        enable_key_event_critic=True,
    )
    builder_calls = 0
    builder_contexts: list[dict] = []
    key_critic_verdicts: list[str] = []
    key_critic_prompts: list[str] = []

    def key_critic_record(task, *, reject: bool, rejection_index: int) -> WorkerRunRecord:
        verdict = "reject" if reject else "pass"
        record = _critic_record(task, assessment=verdict)
        artifact = dict(record.output_artifact or {})
        payload = dict(artifact.get("payload") or {})
        assessment = dict((payload.get("step_assessments") or [{}])[0])
        assessment.update(
            {
                # Real compact provider responses do not author the opaque
                # Host step identity; the Host binds this unique assessment.
                "step_id": "",
                "verdict": verdict,
                "blocking": reject,
                "blocking_type": (
                    (
                        "functional_group_compatibility"
                        if rejection_index == 1
                        else "chemoselectivity"
                    )
                    if reject
                    else "none"
                ),
                "repair_scope": "focus_edge" if reject else "none",
                "reasons": (
                    [
                        (
                            "protect the free alcohol before organometallic generation"
                            if rejection_index == 1
                            else "remove the competing dienophile before cyclization"
                        )
                    ]
                    if reject
                    else []
                ),
                "suggested_revision": (
                    (
                        "install the compatible protecting group"
                        if rejection_index == 1
                        else "mask the competing alkene"
                    )
                    if reject
                    else ""
                ),
            }
        )
        payload.update(
            {
                "checkpoint_match": True,
                "overall_assessment": verdict,
                "step_assessments": [assessment],
            }
        )
        artifact["payload"] = payload
        return replace(record, output_artifact=artifact)

    def executor(task):
        nonlocal builder_calls
        if task.required_artifact_type == "StrategyPortfolioReport":
            return _strategy_portfolio_record(task)
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.task_type == "paper_matched_key_event_critic":
            key_critic_prompts.append(task.objective)
            reject = len(key_critic_verdicts) < 2
            key_critic_verdicts.append("reject" if reject else "pass")
            return key_critic_record(
                task,
                reject=reject,
                rejection_index=len(key_critic_verdicts),
            )
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)

        builder_calls += 1
        builder_contexts.append(
            json.loads(task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
        )
        product = str(task.host_context.get("selected_product") or "")
        operation = [
            {"op": "break_bond", "map_a": 2, "map_b": 3},
            {"op": "break_bond", "map_a": 1, "map_b": 2},
            {"op": "change_bond_order", "map_a": 1, "map_b": 2, "delta": 1},
        ][builder_calls - 1]
        candidate = {
            "candidate_id": task.task_id,
            "product_smiles": product,
            "precursor_smiles": [],
            "checkpoint_relation": "executes_checkpoint",
            "reaction_family": "correctable key construction",
            "conditions": ["test conditions"],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": [operation],
        }
        return replace(
            _proposal_record(candidate, target=product),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        request = {
            "expandable_smiles": ["CCO"],
            "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
            "route_steps": [],
        }
        first = request_handler(request)
        assert first["candidates"] == []
        second = request_handler(request)
        assert second["candidates"] == []
        third = request_handler(request)["candidates"][0]
        return {
            "route_steps": [third["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    third["precursor_smiles"],
                    third["mapped_precursor_smiles"],
                )
            ],
            "solved": False,
            "policy_calls": 3,
            "mcts_iterations": 3,
            "diagnostics": {"engine": "AiZynthFinder.MctsSearchTree"},
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    base_spec = _spec(context)
    spec = replace(
        base_spec,
        metadata={
            **dict(base_spec.metadata),
            "remaining_model_budget": {"model_invocations": 10},
        },
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED, result.error
    assert builder_calls == 3
    assert key_critic_verdicts == ["reject", "reject", "pass"]
    first_constraints = builder_contexts[1]["pending_checkpoint_feedback"]["active_constraints"]
    assert [row["blocking_type"] for row in first_constraints] == ["functional_group_compatibility"]
    accumulated = builder_contexts[2]["pending_checkpoint_feedback"]["active_constraints"]
    assert [row["blocking_type"] for row in accumulated] == [
        "functional_group_compatibility",
        "chemoselectivity",
    ]
    assert '"active_checkpoint_constraints":' not in key_critic_prompts[0]
    assert "protect the free alcohol" in key_critic_prompts[1]
    assert "protect the free alcohol" in key_critic_prompts[2]
    assert "remove the competing dienophile" in key_critic_prompts[2]
    assert result.usage["actual_key_event_critic_calls"] == 3
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert len(plan.multi_step_skeletons[0]["steps"]) == 1
    family = plan.route_families[0]
    assert family["key_event_critic_call_count"] == 3
    assert family["key_event_critic_completed"] is True
    assert [row["status"] for row in family["key_event_critic_history"]] == [
        "rejected",
        "rejected",
        "completed",
    ]
    assert family["pending_key_event_feedback"] == {}


def test_strategy_horizon_scope_replans_same_leaf_before_next_builder(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=1,
        max_skeletons=1,
        max_steps_per_skeleton=2,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=2,
        max_strategic_milestones_per_branch=2,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=0,
        require_strategy_graph_edits=True,
        require_complete_route_json=True,
        enable_key_event_critic=True,
    )
    builder_contexts: list[dict] = []
    strategy_tasks: list[WorkerTask] = []
    key_critic_calls = 0

    def replacement_strategy_record(task) -> WorkerRunRecord:
        record = _strategy_record(task)
        artifact = copy.deepcopy(record.output_artifact)
        artifact["payload"]["strategy_card"] = _strategy_card(3)
        return replace(record, output_artifact=artifact)

    def executor(task):
        nonlocal key_critic_calls
        if task.required_artifact_type == "StrategyPortfolioReport":
            return _strategy_portfolio_record(task)
        if task.required_artifact_type == "StrategyCardReport":
            if '"retired_strategy":' in task.objective:
                strategy_tasks.append(task)
                return replacement_strategy_record(task)
            return _strategy_record(task)
        if task.task_type == "paper_matched_key_event_critic":
            key_critic_calls += 1
            record = _critic_record(task, assessment="reject")
            artifact = copy.deepcopy(record.output_artifact)
            payload = artifact["payload"]
            payload["checkpoint_match"] = True
            payload["overall_assessment"] = "reject"
            payload["step_assessments"] = [
                {
                    "step_id": "",
                    "verdict": "reject",
                    "blocking": True,
                    "blocking_type": "stereochemistry",
                    "repair_scope": "strategy_horizon",
                    "reasons": ["the checkpoint cannot deliver the required geometry"],
                    "suggested_revision": "replace the route-defining checkpoint",
                }
            ]
            return replace(record, output_artifact=artifact)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)

        builder_contexts.append(
            json.loads(task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
        )
        product = str(task.host_context.get("selected_product") or "")
        first_builder = len(builder_contexts) == 1
        candidate = {
            "candidate_id": task.task_id,
            "product_smiles": product,
            "precursor_smiles": [],
            "checkpoint_relation": (
                "executes_checkpoint" if first_builder else "preparatory"
            ),
            "reaction_family": (
                "retired checkpoint" if first_builder else "new-horizon preparation"
            ),
            "conditions": ["test conditions"],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": [
                {"op": "break_bond", "map_a": 2, "map_b": 3}
            ],
        }
        return replace(
            _proposal_record(candidate, target=product),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        request = {
            "expandable_smiles": ["CCO"],
            "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
            "route_steps": [],
        }
        assert request_handler(request)["candidates"] == []
        replacement = request_handler(request)["candidates"][0]
        return {
            "route_steps": [replacement["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    replacement["precursor_smiles"],
                    replacement["mapped_precursor_smiles"],
                )
            ],
            "solved": False,
            "policy_calls": 2,
            "mcts_iterations": 2,
            "diagnostics": {"engine": "AiZynthFinder.MctsSearchTree"},
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    base_spec = _spec(context)
    spec = replace(
        base_spec,
        metadata={
            **dict(base_spec.metadata),
            "remaining_model_budget": {"model_invocations": 12},
        },
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED, result.error
    assert key_critic_calls == 1
    assert len(builder_contexts) == 2
    assert builder_contexts[0]["strategy"]["strategy_query"] != (
        builder_contexts[1]["strategy"]["strategy_query"]
    )
    assert len(strategy_tasks) == 2
    generator_context = json.loads(strategy_tasks[0].objective.rsplit("\n", 1)[1])
    assert generator_context["retired_strategy"]["blocking_type"] == "stereochemistry"
    assert builder_contexts[1].get("connected_path_reactions", []) == []


def test_route_span_scope_stops_same_parent_and_dispatches_path_repair(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=1,
        max_skeletons=1,
        max_steps_per_skeleton=2,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=2,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=0,
        require_complete_route_json=True,
        enable_key_event_critic=True,
        enable_transactional_path_repair=True,
    )
    builder_calls = 0
    repair_calls: list[dict] = []

    def executor(task):
        nonlocal builder_calls
        if task.required_artifact_type == "StrategyPortfolioReport":
            return _strategy_portfolio_record(task)
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.task_type == "paper_matched_key_event_critic":
            record = _critic_record(task, assessment="reject")
            artifact = dict(record.output_artifact or {})
            payload = dict(artifact.get("payload") or {})
            assessment = dict((payload.get("step_assessments") or [{}])[0])
            assessment.update(
                {
                    # Exercise the real compact-provider wire shape.  Prefix
                    # rewrite dispatch must use the Host-owned focus identity.
                    "step_id": "",
                    "verdict": "reject",
                    "blocking": True,
                    "blocking_type": "stereochemistry",
                    "repair_scope": "route_span",
                    "reasons": ["the accepted mapped product lacks required alkene geometry"],
                    "suggested_revision": (
                        "rewrite the accepted product stereo before retrying the checkpoint"
                    ),
                }
            )
            payload.update(
                {
                    "checkpoint_match": True,
                    "overall_assessment": "reject",
                    "step_assessments": [assessment],
                }
            )
            artifact["payload"] = payload
            return replace(record, output_artifact=artifact)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)

        builder_calls += 1
        product = str(task.host_context.get("selected_product") or "")
        candidate = {
            "candidate_id": task.task_id,
            "product_smiles": product,
            "precursor_smiles": [],
            "checkpoint_relation": ("preparatory" if builder_calls == 1 else "executes_checkpoint"),
            "reaction_family": (
                "single preparatory disconnection"
                if builder_calls == 1
                else "key construction requiring prefix rewrite"
            ),
            "conditions": ["test conditions"],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": [
                (
                    {"op": "break_bond", "map_a": 2, "map_b": 3}
                    if builder_calls == 1
                    else {"op": "break_bond", "map_a": 1, "map_b": 2}
                )
            ],
        }
        return replace(
            _proposal_record(candidate, target=product),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        first = request_handler(
            {
                "expandable_smiles": ["CCO"],
                "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
                "route_steps": [],
            }
        )["candidates"][0]
        upstream_index = first["precursor_smiles"].index("CC")
        rejected = request_handler(
            {
                "expandable_smiles": ["CC"],
                "expandable_mapped_smiles": [first["mapped_precursor_smiles"][upstream_index]],
                "route_steps": [first["route_step"]],
            }
        )
        assert rejected["candidates"] == []
        assert rejected["stop_search"] is True
        assert rejected["stop_reason"] == ("key_event_route_span_repair_required")
        # Deliberately return the sidecar's empty terminal projection.  The
        # Host request prefix remains authoritative after the repair stop.
        return {
            "route_steps": [],
            "open_leaf_states": [],
            "solved": False,
            "policy_calls": 2,
            "mcts_iterations": 2,
            "diagnostics": {"engine": "AiZynthFinder.MctsSearchTree"},
        }

    def fake_transactional_repair(
        self,
        _spec_value,
        *,
        branch,
        blocking_steps,
        repair_context_steps=None,
        checkpoint_feedback=None,
        **_kwargs,
    ):
        repair_calls.append(
            {
                "authoritative_steps": [dict(row) for row in branch.get("steps") or []],
                "blocking_steps": [dict(row) for row in blocking_steps],
                "repair_context_steps": [dict(row) for row in repair_context_steps or []],
                "checkpoint_feedback": dict(checkpoint_feedback or {}),
            }
        )
        return False

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    monkeypatch.setattr(
        SequentialStrategyDirectorRunner,
        "_repair_branch_transactionally",
        fake_transactional_repair,
    )
    base_spec = _spec(context)
    spec = replace(
        base_spec,
        metadata={
            **dict(base_spec.metadata),
            "remaining_model_budget": {"model_invocations": 10},
        },
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED, result.error
    assert builder_calls == 2
    assert len(repair_calls) == 1
    dispatched = repair_calls[0]
    assert len(dispatched["authoritative_steps"]) == 1
    assert len(dispatched["repair_context_steps"]) == 2
    rejected_focus = dispatched["repair_context_steps"][-1]
    assert rejected_focus["step_id"] == dispatched["blocking_steps"][0]["step_id"]
    assert rejected_focus["step_id"] not in {
        row["step_id"] for row in dispatched["authoritative_steps"]
    }
    assert dispatched["checkpoint_feedback"]["active_constraints"][0][
        "blocking_type"
    ] == "stereochemistry"
    assert dispatched["checkpoint_feedback"]["active_constraints"][0]["reasons"] == [
        "the accepted mapped product lacks required alkene geometry"
    ]
    plan = GlobalCampaignPlan.from_dict(result.output)
    family = plan.route_families[0]
    assert len(plan.multi_step_skeletons[0]["steps"]) == 1
    assert (
        family["aizynthfinder_strategy_search"]["online_path_repair_retained_host_prefix"] is True
    )


def test_uncertain_key_event_is_rechecked_after_selected_direct_precursor_evidence(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=1,
        max_skeletons=1,
        max_steps_per_skeleton=2,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=2,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=0,
        require_complete_route_json=True,
        enable_key_event_critic=True,
    )
    builder_contexts: list[dict] = []
    key_critic_calls = 0

    def executor(task):
        nonlocal key_critic_calls
        if task.required_artifact_type == "StrategyPortfolioReport":
            return _strategy_portfolio_record(task)
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.task_type == "paper_matched_key_event_critic":
            key_critic_calls += 1
            record = _critic_record(task, assessment="uncertain")
            artifact = dict(record.output_artifact or {})
            payload = dict(artifact.get("payload") or {})
            assessment = dict((payload.get("step_assessments") or [{}])[0])
            assessment.update(
                {
                    "step_id": str(task.host_context.get("focus_step_id") or ""),
                    "verdict": "uncertain",
                    "blocking": False,
                    "blocking_type": "stereochemistry",
                    "reasons": ["facial selectivity remains unresolved"],
                    "suggested_revision": "preserve the stereochemical relay in the next disconnection",
                }
            )
            payload.update(
                {
                    "checkpoint_match": True,
                    "overall_assessment": "uncertain",
                    "step_assessments": [assessment],
                }
            )
            artifact["payload"] = payload
            return replace(record, output_artifact=artifact)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)

        builder_contexts.append(
            json.loads(task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
        )
        product = str(task.host_context.get("selected_product") or "")
        operation = (
            {"op": "break_bond", "map_a": 2, "map_b": 3}
            if len(builder_contexts) == 1
            else {"op": "break_bond", "map_a": 1, "map_b": 2}
        )
        return replace(
            _proposal_record(
                {
                    "candidate_id": task.task_id,
                    "product_smiles": product,
                    "precursor_smiles": [],
                    "checkpoint_relation": (
                        "executes_checkpoint" if len(builder_contexts) == 1 else "preparatory"
                    ),
                    "reaction_family": "uncertain checkpoint canary",
                    "conditions": ["test conditions"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "reaction_operations": [operation],
                },
                target=product,
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        first = request_handler(
            {
                "expandable_smiles": ["CCO"],
                "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
                "route_steps": [],
            }
        )["candidates"][0]
        upstream_index = first["precursor_smiles"].index("CC")
        second = request_handler(
            {
                "expandable_smiles": ["CC"],
                "expandable_mapped_smiles": [first["mapped_precursor_smiles"][upstream_index]],
                "route_steps": [first["route_step"]],
            }
        )["candidates"][0]
        return {
            "route_steps": [first["route_step"], second["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    second["precursor_smiles"],
                    second["mapped_precursor_smiles"],
                )
            ],
            "solved": False,
            "policy_calls": 2,
            "mcts_iterations": 2,
            "diagnostics": {"engine": "AiZynthFinder.MctsSearchTree"},
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED, result.error
    assert key_critic_calls == 2
    assert "pending_checkpoint_feedback" not in builder_contexts[1]
    family = GlobalCampaignPlan.from_dict(result.output).route_families[0]
    assert family["key_event_critic_completed"] is False
    assert [row["status"] for row in family["key_event_critic_history"]] == [
        "uncertain",
        "uncertain",
    ]
    assert family["key_event_critic_history"][-1]["review_kind"] == (
        "selected_direct_precursor_evidence"
    )
    assert family["pending_key_event_feedback"] == {}


def test_followup_key_event_reject_rolls_back_host_path_before_retry(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=1,
        max_skeletons=1,
        max_steps_per_skeleton=3,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=3,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=0,
        require_complete_route_json=True,
        enable_key_event_critic=True,
    )
    builder_calls = 0
    builder_contexts: list[dict] = []
    key_critic_verdicts: list[str] = []

    def key_critic_record(task, verdict: str) -> WorkerRunRecord:
        record = _critic_record(task, assessment=verdict)
        artifact = dict(record.output_artifact or {})
        payload = dict(artifact.get("payload") or {})
        assessment = dict((payload.get("step_assessments") or [{}])[0])
        assessment.update(
            {
                "step_id": str(task.host_context.get("focus_step_id") or ""),
                "verdict": verdict,
                "blocking": verdict == "reject",
                "blocking_type": (
                    "sequence_dependency" if verdict == "reject" else "stereochemistry"
                ),
                "reasons": (
                    ["the selected evidence preserves the incompatible sequence"]
                    if verdict == "reject"
                    else (
                        ["direct precursor evidence is still needed"]
                        if verdict == "uncertain"
                        else []
                    )
                ),
                "suggested_revision": (
                    "retry from before the rejected key event" if verdict == "reject" else ""
                ),
            }
        )
        payload.update(
            {
                "checkpoint_match": True,
                "overall_assessment": verdict,
                "step_assessments": [assessment],
            }
        )
        artifact["payload"] = payload
        return replace(record, output_artifact=artifact)

    def executor(task):
        nonlocal builder_calls
        if task.required_artifact_type == "StrategyPortfolioReport":
            return _strategy_portfolio_record(task)
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.task_type == "paper_matched_key_event_critic":
            verdict = ["uncertain", "reject", "pass"][len(key_critic_verdicts)]
            key_critic_verdicts.append(verdict)
            return key_critic_record(task, verdict)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)

        builder_calls += 1
        builder_contexts.append(
            json.loads(task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
        )
        product = str(task.host_context.get("selected_product") or "")
        operation = [
            {"op": "break_bond", "map_a": 2, "map_b": 3},
            {"op": "break_bond", "map_a": 1, "map_b": 2},
            {"op": "break_bond", "map_a": 1, "map_b": 2},
        ][builder_calls - 1]
        return replace(
            _proposal_record(
                {
                    "candidate_id": task.task_id,
                    "product_smiles": product,
                    "precursor_smiles": [],
                    "checkpoint_relation": (
                        "preparatory" if builder_calls == 2 else "executes_checkpoint"
                    ),
                    "reaction_family": "followup rejection transaction canary",
                    "conditions": ["test conditions"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "reaction_operations": [operation],
                },
                target=product,
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        first = request_handler(
            {
                "expandable_smiles": ["CCO"],
                "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
                "route_steps": [],
            }
        )["candidates"][0]
        cc_index = first["precursor_smiles"].index("CC")
        rejected = request_handler(
            {
                "expandable_smiles": ["CC"],
                "expandable_mapped_smiles": [first["mapped_precursor_smiles"][cc_index]],
                "route_steps": [first["route_step"]],
            }
        )
        assert rejected["candidates"] == []
        assert rejected["model_call_consumed"] is True
        assert first["route_step"]["step_id"] in rejected["rejected_path_step_ids"]
        alternate = request_handler(
            {
                "expandable_smiles": ["CCO"],
                "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
                "route_steps": [],
            }
        )["candidates"][0]
        return {
            "route_steps": [alternate["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    alternate["precursor_smiles"],
                    alternate["mapped_precursor_smiles"],
                )
            ],
            "solved": False,
            "policy_calls": 3,
            "mcts_iterations": 3,
            "diagnostics": {
                "engine": "AiZynthFinder.MctsSearchTree",
                "path_rejection_count": 1,
            },
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED, result.error
    assert builder_calls == 3
    assert key_critic_verdicts == ["uncertain", "reject", "pass"]
    retry_constraints = builder_contexts[2]["pending_checkpoint_feedback"]["active_constraints"]
    assert len(retry_constraints) == 1
    assert retry_constraints[0]["severity"] == "blocking"
    assert retry_constraints[0]["blocking_type"] == "sequence_dependency"
    assert retry_constraints[0]["suggested_revision"] == (
        "retry from before the rejected key event"
    )
    plan = GlobalCampaignPlan.from_dict(result.output)
    steps = plan.multi_step_skeletons[0]["steps"]
    assert len(steps) == 1
    assert steps[0]["step_id"].endswith("node:3:candidate:1")
    family = plan.route_families[0]
    assert family["aizynthfinder_strategy_search"]["path_rejection_count"] == 1


def test_callback_start_path_reject_does_not_consume_builder_call(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=1,
        max_skeletons=1,
        max_steps_per_skeleton=2,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=2,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=0,
        require_complete_route_json=True,
        enable_key_event_critic=True,
    )
    builder_calls = 0
    review_calls = 0

    def executor(task):
        nonlocal builder_calls
        if task.required_artifact_type == "StrategyPortfolioReport":
            return _strategy_portfolio_record(task)
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)
        builder_calls += 1
        raise AssertionError("Builder must not run before applying a pending reject")

    rejected_step = {
        "step_id": "existing:rejected:key-event",
        "product_smiles": "CCO",
        "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
        "precursor_smiles": ["CC", "O"],
        "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
        "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
        "checkpoint_relation": "executes_checkpoint",
        "reaction_family": "rejected existing key event",
        "conditions": ["test conditions"],
    }
    observed_response = {}

    def fake_sidecar(*, request_handler, **_kwargs):
        observed_response.update(
            request_handler(
                {
                    "expandable_smiles": ["CC", "O"],
                    "expandable_mapped_smiles": [
                        "[CH3:1][CH3:2]",
                        "[OH2:3]",
                    ],
                    "route_steps": [rejected_step],
                }
            )
        )
        return {
            "route_steps": [],
            "open_leaf_states": [
                {
                    "smiles": "CCO",
                    "mapped_smiles": "[CH3:1][CH2:2][OH:3]",
                }
            ],
            "solved": False,
            "policy_calls": 0,
            "mcts_iterations": 1,
            "diagnostics": {
                "engine": "AiZynthFinder.MctsSearchTree",
                "path_rejection_count": 1,
            },
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )

    def review_disposition(self, *args, **kwargs):
        nonlocal review_calls
        del self, args, kwargs
        review_calls += 1
        if review_calls == 1:
            return sequential_module._KeyEventReviewDisposition(
                status="rejected",
                rejected_path_step_ids=(rejected_step["step_id"],),
                rejection_reason="pending followup critic rejection",
            )
        return sequential_module._KeyEventReviewDisposition()

    monkeypatch.setattr(
        SequentialStrategyDirectorRunner,
        "_review_selected_uncertain_key_event",
        review_disposition,
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.FAILED
    assert "selected_path_key_event_rejected" in result.error
    assert builder_calls == 0
    assert review_calls >= 1
    assert observed_response["candidates"] == []
    assert observed_response["model_call_consumed"] is False
    assert observed_response["rejected_path_step_ids"] == [rejected_step["step_id"]]


def test_same_aiz_state_reports_no_progress_and_filters_exact_duplicate(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=1,
        max_skeletons=1,
        max_steps_per_skeleton=3,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=4,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=0,
        require_complete_route_json=True,
        enable_key_event_critic=False,
    )
    builder_contexts: list[dict] = []

    def executor(task):
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)

        builder_context = json.loads(
            task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1]
        )
        builder_contexts.append(builder_context)
        operation = (
            {"op": "break_bond", "map_a": 2, "map_b": 3}
            if len(builder_contexts) < 3
            else {"op": "break_bond", "map_a": 1, "map_b": 2}
        )
        product = str(task.host_context.get("selected_product") or "")
        return replace(
            _proposal_record(
                {
                    "candidate_id": task.task_id,
                    "product_smiles": product,
                    "precursor_smiles": [],
                    "checkpoint_relation": "preparatory",
                    "reaction_family": "same-state feedback canary",
                    "conditions": ["test conditions"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "reaction_operations": [operation],
                },
                target=product,
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        request = {
            "expandable_smiles": ["CCO"],
            "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
            "route_steps": [],
        }
        request_handler(request)["candidates"][0]
        assert request_handler(request)["candidates"] == []
        third = request_handler(request)["candidates"][0]
        advanced_request = {
            **request,
            "route_steps": [third["route_step"]],
        }
        assert request_handler(advanced_request)["candidates"]
        return {
            "route_steps": [third["route_step"]],
            "open_leaf_states": [
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in zip(
                    third["precursor_smiles"],
                    third["mapped_precursor_smiles"],
                )
            ],
            "solved": False,
            "policy_calls": 4,
            "mcts_iterations": 4,
            "diagnostics": {"engine": "AiZynthFinder.MctsSearchTree"},
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED, result.error
    assert len(builder_contexts) == 4
    first_feedback = builder_contexts[1]["last_rejection_for_this_leaf"]
    assert first_feedback["reason"] == ("candidate_did_not_advance_selected_mcts_path")
    assert first_feedback["attempted_net_edits"] == [{"map_a": 2, "map_b": 3, "op": "break_bond"}]
    duplicate_feedback = builder_contexts[2]["last_rejection_for_this_leaf"]
    assert duplicate_feedback["reason"] == ("candidate_repeats_same_mcts_state_edit")
    assert "last_rejection_for_this_leaf" not in builder_contexts[3]
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert len(plan.multi_step_skeletons[0]["steps"]) == 1


def test_key_event_feedback_survives_preparatory_move_and_leaf_change(
    monkeypatch,
) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=1,
        max_skeletons=1,
        max_steps_per_skeleton=3,
        planning_mode="sequential_branches",
        paper_matched_reach_profile=True,
        strategy_tree_engine="aizynthfinder_mcts",
        strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1,
        strategy_branch_workers=1,
        max_node_expansions_per_branch=3,
        max_reactionjson_candidates_per_node=1,
        max_route_local_repair_rounds=0,
        require_complete_route_json=True,
        enable_key_event_critic=True,
    )
    builder_contexts: list[dict] = []
    key_critic_calls = 0

    def key_critic_record(task, *, corrected: bool) -> WorkerRunRecord:
        verdict = "pass" if corrected else "reject"
        record = _critic_record(task, assessment=verdict)
        artifact = dict(record.output_artifact or {})
        payload = dict(artifact.get("payload") or {})
        assessment = dict((payload.get("step_assessments") or [{}])[0])
        assessment.update(
            {
                "step_id": str(task.host_context.get("focus_step_id") or ""),
                "verdict": verdict,
                "blocking": not corrected,
                "blocking_type": "none" if corrected else "sequence_dependency",
                "reasons": (
                    [] if corrected else ["preserve one tethered intramolecular precursor"]
                ),
                "suggested_revision": (
                    "" if corrected else "retain the tether before the checkpoint"
                ),
            }
        )
        payload.update(
            {
                "checkpoint_match": corrected,
                "overall_assessment": verdict,
                "step_assessments": [assessment],
            }
        )
        artifact["payload"] = payload
        return replace(record, output_artifact=artifact)

    def executor(task):
        nonlocal key_critic_calls
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.task_type == "paper_matched_key_event_critic":
            key_critic_calls += 1
            return key_critic_record(task, corrected=key_critic_calls == 2)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)

        builder_context = json.loads(
            task.objective.split("PaperMatchedRouteBuilderContext:\n", 1)[1]
        )
        builder_contexts.append(builder_context)
        call_index = len(builder_contexts)
        product = str(task.host_context.get("selected_product") or "")
        if call_index == 1:
            relation = "executes_checkpoint"
            operation = {"op": "break_bond", "map_a": 2, "map_b": 3}
        elif call_index == 2:
            relation = "preparatory"
            operation = {"op": "break_bond", "map_a": 1, "map_b": 2}
        else:
            relation = "executes_checkpoint"
            operation = {"op": "break_bond", "map_a": 2, "map_b": 3}
        return replace(
            _proposal_record(
                {
                    "candidate_id": task.task_id,
                    "product_smiles": product,
                    "precursor_smiles": [],
                    "checkpoint_relation": relation,
                    "reaction_family": "checkpoint feedback canary",
                    "conditions": ["test conditions"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "reaction_operations": [operation],
                },
                target=product,
            ),
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
        )

    def fake_sidecar(*, request_handler, **_kwargs):
        root_request = {
            "expandable_smiles": ["CCO"],
            "expandable_mapped_smiles": ["[CH3:1][CH2:2][OH:3]"],
            "route_steps": [],
        }
        assert request_handler(root_request)["candidates"] == []
        preparatory = request_handler(root_request)["candidates"][0]
        upstream_index = preparatory["precursor_smiles"].index("CO")
        corrected = request_handler(
            {
                "expandable_smiles": ["CO"],
                "expandable_mapped_smiles": [
                    preparatory["mapped_precursor_smiles"][upstream_index]
                ],
                "route_steps": [preparatory["route_step"]],
            }
        )["candidates"][0]
        open_states = [
            {"smiles": value, "mapped_smiles": mapped}
            for value, mapped in zip(
                corrected["precursor_smiles"],
                corrected["mapped_precursor_smiles"],
            )
        ]
        return {
            "route_steps": [preparatory["route_step"], corrected["route_step"]],
            "open_leaf_states": open_states,
            "solved": False,
            "policy_calls": 3,
            "mcts_iterations": 3,
            "diagnostics": {"engine": "AiZynthFinder.MctsSearchTree"},
        }

    monkeypatch.setattr(
        sequential_module,
        "run_aizynthfinder_strategy_branch_sidecar",
        fake_sidecar,
    )
    base_spec = _spec(context)
    spec = replace(
        base_spec,
        metadata={
            **dict(base_spec.metadata),
            "remaining_model_budget": {"model_invocations": 10},
        },
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED, result.error
    assert len(builder_contexts) == 3
    expected_reason = "preserve one tethered intramolecular precursor"
    assert builder_contexts[1]["pending_checkpoint_feedback"]["active_constraints"][0][
        "reasons"
    ] == [expected_reason]
    assert "last_rejection_for_this_leaf" not in builder_contexts[1]
    assert builder_contexts[2]["pending_checkpoint_feedback"]["active_constraints"][0][
        "reasons"
    ] == [expected_reason]
    assert "last_rejection_for_this_leaf" not in builder_contexts[2]
    assert key_critic_calls == 2
    assert result.usage["actual_key_event_critic_calls"] == 2
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert len(plan.multi_step_skeletons[0]["steps"]) == 2


def test_unavailable_critic_is_reported_without_erasing_route_topology() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
    )

    def unavailable_critic(_task):
        raise TimeoutError("critic cutoff")

    result = SequentialStrategyDirectorRunner(
        node_executor=_fake_executor,
        critic_executor=unavailable_critic,
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert result.usage["critic_unavailable_branch_count"] == 1
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert len(plan.route_families) == 1
    assert dict(plan.route_families[0].get("chemical_critic") or {}).get("status") == "unavailable"


def test_codex_critic_editor_loop_repairs_a_blocking_step() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
        max_route_local_repair_rounds=2,
    )
    observed = []
    critic_calls = 0

    def executor(task):
        nonlocal critic_calls
        observed.append(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            critic_calls += 1
            if critic_calls == 1:
                return _blocking_critic_record(
                    task,
                    step_id="codex:branch:1:1",
                )
            return _critic_record(task)
        return _fake_executor(task)

    base_spec = _spec(context)
    spec = replace(
        base_spec,
        metadata={**dict(base_spec.metadata), "remaining_model_budget": {"model_invocations": 10}},
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        editor_executor=executor,
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert critic_calls == 2
    assert sum(task.task_type == "route_chemistry_edit" for task in observed) == 1
    plan = GlobalCampaignPlan.from_dict(result.output)
    family = plan.route_families[0]
    assert family["chemical_critic"]["status"] == "viable"
    assert len(family["critic_editor_history"]) == 2
    assert len(family["editor_repairs"]) == 1
    assert family["editor_attempt_count"] == 1
    assert family["editor_applied_count"] == 1
    assert family["editor_call_count"] == 1
    assert family["route_call_count"] == 1
    builder_tasks = [task for task in observed if task.task_type == "route_step_materialization"]
    assert builder_tasks[0].task_id.endswith(":branch:1:node:1")


def test_surgical_editor_retries_after_reactionjson_replay_failure() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
        max_route_local_repair_rounds=2,
    )
    editor_attempts = 0
    critic_attempts = 0
    observed = []

    def executor(task):
        nonlocal editor_attempts, critic_attempts
        observed.append(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            critic_attempts += 1
            if critic_attempts == 1:
                return _blocking_critic_record(
                    task,
                    step_id="codex:branch:1:1",
                )
            return _critic_record(task)
        record = _fake_executor(task)
        if task.task_type == "route_chemistry_edit":
            editor_attempts += 1
            if editor_attempts == 1:
                artifact = dict(record.output_artifact or {})
                payload = dict(artifact.get("payload") or {})
                candidates = [dict(row) for row in payload.get("candidates") or []]
                candidates[0]["product_smiles"] = "CC"
                payload["candidates"] = candidates
                record = replace(
                    record,
                    output_artifact={**artifact, "payload": payload},
                )
        return record

    base_spec = _spec(context)
    spec = replace(
        base_spec,
        metadata={
            **dict(base_spec.metadata),
            "remaining_model_budget": {"model_invocations": 12},
        },
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        editor_executor=executor,
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert editor_attempts == 2
    assert critic_attempts == 2
    editor_tasks = [task for task in observed if task.task_type == "route_chemistry_edit"]
    assert len(editor_tasks) == 2
    assert "product_mismatch" in editor_tasks[1].objective
    family = GlobalCampaignPlan.from_dict(result.output).route_families[0]
    assert family["chemical_critic"]["status"] == "viable"
    assert len(family["editor_repairs"]) == 1
    assert family["editor_attempt_count"] == 2
    assert family["editor_applied_count"] == 1


def test_editor_retries_after_routejson_materialization_failure() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
        max_route_local_repair_rounds=2,
        require_strategy_graph_edits=True,
        require_complete_route_json=True,
        allow_editor_route_mutations=True,
        paper_matched_reach_profile=True,
        critic_call_timeout_s=1.0,
        max_node_call_timeout_s=1.0,
    )
    editor_attempts = 0
    critic_attempts = 0
    observed = []

    def executor(task):
        nonlocal editor_attempts, critic_attempts
        observed.append(task)
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            critic_attempts += 1
            if critic_attempts == 1:
                return _blocking_critic_record(task, step_id="route:1")
            return _critic_record(task)
        candidate = _complete_route_candidate()
        if task.task_type in {"route_chemistry_edit", "paper_matched_route_editor"}:
            editor_attempts += 1
            if editor_attempts == 1:
                candidate["route_json"][1]["product_smiles"] = "CO"
        return _proposal_record(candidate)

    base_spec = _spec(context)
    spec = replace(
        base_spec,
        metadata={**dict(base_spec.metadata), "remaining_model_budget": {"model_invocations": 10}},
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        editor_executor=executor,
        stock_membership=lambda values: {value: value in {"C", "CC", "O"} for value in values},
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert editor_attempts == 2
    assert critic_attempts == 2
    assert (
        sum(
            task.task_type in {"route_chemistry_edit", "paper_matched_route_editor"}
            for task in observed
        )
        == 2
    )
    family = GlobalCampaignPlan.from_dict(result.output).route_families[0]
    assert family["editor_attempt_count"] == 2
    assert family["editor_applied_count"] == 1


def test_editor_retry_prompt_keeps_only_failure_focus_and_host_boundary() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
        max_route_local_repair_rounds=2,
        require_strategy_graph_edits=True,
        require_complete_route_json=True,
        allow_editor_route_mutations=True,
        paper_matched_reach_profile=True,
        critic_call_timeout_s=1.0,
        max_node_call_timeout_s=1.0,
    )
    editor_attempts = 0
    critic_attempts = 0
    retry_context = None

    def executor(task):
        nonlocal editor_attempts, critic_attempts, retry_context
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            critic_attempts += 1
            if critic_attempts == 1:
                return _blocking_critic_record(task, step_id="route:2")
            return _critic_record(task)
        candidate = _complete_route_candidate()
        if task.task_type in {
            "route_chemistry_edit",
            "paper_matched_route_editor",
        }:
            editor_attempts += 1
            revised_step = dict(candidate["route_json"][1])
            candidate = {
                "repair_summary": "replace the blocked upstream disconnection",
                "no_solved_claim": True,
                "not_parent_route_proof": True,
                "replace_span": {
                    "remove_step_ids": ["route:2"],
                    "revised_steps": [revised_step],
                },
            }
            if editor_attempts == 1:
                revised_step["reaction_operations"] = [
                    {"op": "break_bond", "map_a": 18, "map_b": 19}
                ]
            else:
                retry_context = json.loads(
                    task.objective.split("PaperMatchedRouteEditorContext:\n", 1)[1]
                )
        return _proposal_record(candidate)

    base_spec = _spec(context)
    spec = replace(
        base_spec,
        metadata={
            **dict(base_spec.metadata),
            "remaining_model_budget": {"model_invocations": 10},
        },
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        editor_executor=executor,
        stock_membership=lambda values: {value: value in {"C", "CC", "O"} for value in values},
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert editor_attempts == 2
    assert retry_context is not None
    assert retry_context["schema_version"] == ("paper_matched_route_editor_context.v4")
    assert retry_context["route_replay"] == {
        "route_json_source": "host_checked_editor_working_draft",
        "host_replayed_prefix_step_count": 1,
        "mapped_open_precursor_authority": "host_routejson_dag_compiler",
    }
    assert retry_context["route_json"][1]["mapped_product_smiles"] == ("[CH3:1][CH3:2]")
    assert "previous_failed_replace_span" not in retry_context["repair_history"]
    failure = retry_context["repair_history"]["last_host_replay_failure"]
    assert "replace_span" not in failure
    assert failure["operation_index"] == 0
    assert failure["failed_step_id"] == "route:2"
    assert failure["failed_operation"] == {
        "op": "break_bond",
        "map_a": 18,
        "map_b": 19,
    }
    assert failure["host_selected_open_precursor"] == {
        "product_smiles": "CC",
        "mapped_product_smiles": "[CH3:1][CH3:2]",
    }


def test_initial_routejson_failure_enters_editor_before_branch_blocking() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
        max_route_local_repair_rounds=2,
        require_strategy_graph_edits=True,
        require_complete_route_json=True,
        allow_editor_route_mutations=True,
    )
    observed = []

    def executor(task):
        observed.append(task)
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)
        candidate = _complete_route_candidate()
        if task.task_type != "route_chemistry_edit":
            # The Route Builder has produced a complete-looking draft, but
            # its second product is not the exact host-replayed precursor.
            candidate["route_json"][1]["product_smiles"] = "CO"
        return _proposal_record(candidate)

    base_spec = _spec(context)
    spec = replace(
        base_spec,
        metadata={
            **dict(base_spec.metadata),
            "remaining_model_budget": {"model_invocations": 10},
        },
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        editor_executor=executor,
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    editor_tasks = [task for task in observed if task.task_type == "route_chemistry_edit"]
    assert len(editor_tasks) == 1
    assert "reaction_operations" in editor_tasks[0].objective
    assert '"map_a":2' in editor_tasks[0].objective
    plan = GlobalCampaignPlan.from_dict(result.output)
    family = plan.route_families[0]
    assert family["materialization_failures"] == {}
    assert family["materialization_editor_history"][0]["outcome"] == ("host_recompiled")
    assert family["editor_repairs"][0]["phase"] == ("route_builder_materialization")
    assert len(family["critic_editor_history"]) == 1


def test_critic_wall_reservation_does_not_starve_route_nodes(monkeypatch) -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
    )
    clock = {"now": 0.0}

    def monotonic() -> float:
        return clock["now"]

    def slow_executor(task):
        record = (
            _critic_record(task)
            if task.required_artifact_type == "ChemicalStrategyCritique"
            else _fake_executor(task)
        )
        clock["now"] += 0.6 if task.required_artifact_type == "StrategyCardReport" else 0.1
        return record

    monkeypatch.setattr(sequential_module.time, "monotonic", monotonic)
    base = _spec(context)
    spec = replace(base, budget=Budget(max_wall_time_s=1.0, max_tokens=20_000))
    observed = []

    def recording_executor(task):
        observed.append(task)
        return slow_executor(task)

    result = SequentialStrategyDirectorRunner(
        node_executor=recording_executor,
        critic_executor=recording_executor,
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert any(task.required_artifact_type == "RetrosynthesisProposalReport" for task in observed)


def test_critic_reservation_is_applied_before_strategy_seeds(monkeypatch) -> None:
    """Strategy calls cannot consume the wall slice reserved for Route/Critic."""

    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=3,
        max_node_expansions_per_branch=1,
        max_route_local_repair_rounds=1,
    )
    clock = {"now": 0.0}
    observed = []

    def monotonic() -> float:
        return clock["now"]

    def executor(task):
        observed.append(task)
        if task.required_artifact_type == "StrategyCardReport":
            record = _strategy_record(task)
            clock["now"] += 0.2
            return record
        if task.required_artifact_type == "ChemicalStrategyCritique":
            clock["now"] += 0.01
            return _critic_record(task)
        clock["now"] += 0.01
        return _fake_executor(task)

    monkeypatch.setattr(sequential_module.time, "monotonic", monotonic)
    base = _spec(context)
    spec = replace(base, budget=Budget(max_wall_time_s=1.0, max_tokens=20_000))
    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        editor_executor=executor,
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    strategy_tasks = [
        task for task in observed if task.required_artifact_type == "StrategyCardReport"
    ]
    route_tasks = [
        task for task in observed if task.required_artifact_type == "RetrosynthesisProposalReport"
    ]
    assert len(strategy_tasks) == 3
    assert route_tasks, "Route Builder must retain a call after strategy seeding"


def test_paper_critic_reserve_keeps_only_untouched_routes_first_critic() -> None:
    branches = [
        {"steps": [{"step_id": "current"}], "editor_attempt_count": 6},
        {"steps": [{"step_id": "future"}], "editor_attempt_count": 0},
        {"steps": []},
    ]

    critics, editors = sequential_module._paper_critic_editor_reserve_after_current_critic(
        branches,
        current_index=0,
        iteration=2,
        max_rounds=6,
    )

    assert (critics, editors) == (1, 0)
    assert sequential_module._paper_critic_editor_reserve_after_current_critic(
        branches[:1],
        current_index=0,
        iteration=2,
        max_rounds=6,
    ) == (0, 0)
    assert sequential_module._paper_critic_editor_reserve_after_current_critic(
        [{"steps": [{"step_id": "current"}], "editor_attempt_count": 1}],
        current_index=0,
        iteration=1,
        max_rounds=6,
    ) == (0, 0)


def test_three_independent_branches_expand_one_node_per_call() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=3,
        max_route_families=6,
        max_skeletons=6,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=3,
        max_node_expansions_per_branch=2,
        max_route_local_repair_rounds=6,
        max_provider_requests=64,
        critic_call_timeout_s=1.0,
    )
    observed_tasks = []

    def recording_executor(task):
        observed_tasks.append(task)
        return _fake_executor(task)

    runner = SequentialStrategyDirectorRunner(
        node_executor=recording_executor,
        critic_executor=_fake_critic_executor,
    )
    result = runner(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert result.usage["model_invocations"] == 12
    assert result.usage["accepted_expansions"] == 6
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert len(plan.route_families) == 3
    assert [len(row["steps"]) for row in plan.multi_step_skeletons] == [2, 2, 2]
    assert all(
        row["routejson_authority"] == "legacy_declared_route_projection"
        and row["routejson_replay_complete"] is False
        for row in plan.multi_step_skeletons
    )
    assert [row["strategy"] for row in plan.route_families] == [
        "Codex-authored strategy - branch-1-key-construction: branch-1-orthogonal",
        "Codex-authored strategy - branch-2-key-construction: branch-2-orthogonal",
        "Codex-authored strategy - branch-3-key-construction: branch-3-orthogonal",
    ]
    audits = validate_global_campaign_plan(plan, context, config)
    assert all(row["accepted"] is True for row in audits)
    assert len(json.dumps(runner.prompt_for(context, "initial_architecture", config))) < 1000
    assert all(
        len(task.objective.encode("utf-8")) <= config.max_node_prompt_bytes
        for task in observed_tasks
    )
    assert all(
        task.required_artifact_type == "StrategyCardReport"
        and task.task_type == "strategic_disconnection_mining"
        and '"phase":"strategy_generator"' in task.objective
        for task in observed_tasks[:3]
    )
    assert all(
        "Compare at least three materially distinct strategies" in task.objective
        for task in observed_tasks[:3]
    )
    assert all("campaign_target_profile" in task.objective for task in observed_tasks[:3])
    assert all(task.task_type == "route_step_materialization" for task in observed_tasks[3:])
    for skeleton in plan.multi_step_skeletons:
        strategy_digests = {step["strategy_digest"] for step in skeleton["steps"]}
        assert len(strategy_digests) == 1
        assert next(iter(strategy_digests)) == next(
            family["strategy_card"]["strategy_digest"]
            for family in plan.route_families
            if family["route_family_id"] == skeleton["route_family_id"]
        )


def test_compiler_first_builder_carries_real_mapped_precursor_between_calls() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=2,
        require_strategy_graph_edits=True,
        # Matched mode: consume one ReactionJSON edit at a time and carry the
        # exact host-derived mapped precursor into the next call.
        require_complete_route_json=False,
    )
    observed = []

    def executor(task):
        observed.append(task)
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)
        product = re.search(r'"selected_open_leaf":"([^"]+)"', task.objective).group(1)
        operations = (
            [{"op": "break_bond", "map_a": 1, "map_b": 2}]
            if product == "CCO"
            else [{"op": "break_bond", "map_a": 2, "map_b": 3}]
        )
        return _proposal_record(
            {
                "schema_version": "retrosynthesis_candidate.v1",
                "candidate_id": task.task_id,
                "product_smiles": product,
                # Deliberately wrong advisory text: compiler output must win.
                "precursor_smiles": ["N"],
                "reaction_family": "compiler-first fragmentation",
                "product_retron_type": "bond disconnection",
                "transformation_rationale": "exercise host graph edits",
                "conditions": [],
                "catalyst": "",
                "enzyme": "",
                "limitations": [],
                "required_validation": ["structure"],
                "no_solved_claim": True,
                "not_parent_route_proof": True,
                "reaction_operations": operations,
                "route_json": None,
            },
            target=product,
        )

    runner = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        stock_membership=lambda values: {value: value in {"C", "O"} for value in values},
    )
    result = runner(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    node_tasks = [task for task in observed if task.task_type == "route_step_materialization"]
    assert len(node_tasks) == 2
    assert node_tasks[0].objective.startswith("Expand exactly one retrosynthetic node")
    assert '"selected_open_leaf":"CO"' in node_tasks[1].objective
    assert '"selected_open_leaf_mapped":"[CH3:2][OH:3]"' in node_tasks[1].objective
    plan = GlobalCampaignPlan.from_dict(result.output)
    skeleton = plan.multi_step_skeletons[0]
    assert skeleton["steps"][0]["precursor_smiles"] == ["C", "CO"]
    assert skeleton["steps"][1]["mapped_product_smiles"] == "[CH3:2][OH:3]"
    assert skeleton["routejson_authority"] == "host_routejson_dag_compiler"
    assert skeleton["routejson_replay_complete"] is True
    assert len(skeleton["route_json"]) == 2


def test_compiler_first_or_tree_survives_into_critic_editor_and_is_rebuilt() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
        max_route_local_repair_rounds=1,
        require_strategy_graph_edits=True,
        require_complete_route_json=False,
    )
    critic_calls = 0

    def executor(task):
        nonlocal critic_calls
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            critic_calls += 1
            if critic_calls == 1:
                step_id = re.search(r'"step_id":"([^"]+)"', task.objective).group(1)
                return _blocking_critic_record(task, step_id=step_id)
            return _critic_record(task)
        product = re.search(r'"selected_open_leaf":"([^"]+)"', task.objective).group(1)
        operations = (
            [{"op": "break_bond", "map_a": 1, "map_b": 2}]
            if task.task_type == "route_chemistry_edit"
            else [
                {"op": "break_bond", "map_a": 1, "map_b": 2},
                {"op": "break_bond", "map_a": 2, "map_b": 3},
            ]
        )
        return _proposal_record(
            {
                "schema_version": "retrosynthesis_candidate.v1",
                "candidate_id": task.task_id,
                "product_smiles": product,
                "precursor_smiles": [],
                "reaction_family": "compiler-first editor probe",
                "product_retron_type": "bond disconnection",
                "transformation_rationale": "exercise retained OR state",
                "conditions": [],
                "catalyst": "",
                "enzyme": "",
                "limitations": [],
                "required_validation": ["structure"],
                "no_solved_claim": True,
                "not_parent_route_proof": True,
                "reaction_operations": operations,
                "route_json": None,
            },
            target=product,
        )

    runner = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        editor_executor=executor,
        stock_membership=lambda values: {value: value in {"C", "O"} for value in values},
    )
    result = runner(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    plan = GlobalCampaignPlan.from_dict(result.output)
    family = plan.route_families[0]
    assert family["route_call_count"] == 1
    assert family["editor_call_count"] == 1
    assert family["reactionjson_or_search"]["root_solved"] is False
    assert family["reactionjson_or_search_resets"][0]["previous_summary"]["root_solved"] is True
    assert family["reactionjson_or_search_resets"][0]["rebuilt_summary"]["root_solved"] is False
    assert family["reactionjson_or_search_resets"][-1]["rebuilt_summary"]["root_solved"] is False
    skeleton = plan.multi_step_skeletons[0]
    assert skeleton["steps"][0]["precursor_smiles"] == ["C", "CO"]


def test_explicit_legacy_complete_route_contract_overrides_compiler_first_mode() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=2,
        require_strategy_graph_edits=True,
        require_complete_route_json=True,
    )
    observed = []

    def executor(task):
        observed.append(task)
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)
        assert "one complete linear RouteJSON route" in task.objective
        return _proposal_record(_complete_route_candidate())

    runner = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        stock_membership=lambda values: {value: True for value in values},
    )
    result = runner(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    node_tasks = [task for task in observed if task.task_type == "route_step_materialization"]
    assert node_tasks
    assert "one complete linear RouteJSON route" in node_tasks[0].objective
    assert "Expand exactly one retrosynthetic node" not in node_tasks[0].objective


def test_legacy_complete_route_contract_accepts_one_step_stock_closure() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
        require_strategy_graph_edits=True,
        require_complete_route_json=True,
    )

    def executor(task):
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)
        candidate = _complete_route_candidate()
        candidate["route_json"] = [candidate["route_json"][0]]
        return _proposal_record(candidate)

    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        stock_membership=lambda values: {value: True for value in values},
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert len(plan.multi_step_skeletons[0]["steps"]) == 1
    assert not plan.frontier_priorities


def test_every_distinct_open_leaf_is_delegated() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=3,
        max_route_families=6,
        max_skeletons=6,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=3,
        max_node_expansions_per_branch=1,
        max_provider_requests=64,
    )
    result = SequentialStrategyDirectorRunner(
        node_executor=_fake_executor,
        critic_executor=_fake_critic_executor,
    )(_spec(context), context, "initial_architecture", config)
    plan = GlobalCampaignPlan.from_dict(result.output)
    leaves = {row["smiles"] for row in plan.shared_intermediates}
    assert leaves == {"C", "C=C", "CC", "CO", "O"}
    assert {row["target_smiles"] for row in plan.frontier_priorities} == leaves


def test_internal_ledger_budget_is_checked_between_round_robin_calls() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=3,
        max_route_families=6,
        max_skeletons=6,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=3,
        max_node_expansions_per_branch=25,
    )
    tasks = []

    def recording_executor(task):
        tasks.append(task)
        return _fake_executor(task)

    base = _spec(context)
    spec = replace(
        base,
        metadata={
            **base.metadata,
            "remaining_model_budget": {
                "model_invocations": 7,
                "input_tokens": 100_000,
                "output_tokens": 100_000,
                "wall_time_s": 60.0,
            },
        },
    )
    result = SequentialStrategyDirectorRunner(node_executor=recording_executor)(
        spec, context, "initial_architecture", config
    )

    assert result.usage["model_invocations"] == 5
    expansion_tasks = [
        task for task in tasks if task.required_artifact_type == "RetrosynthesisProposalReport"
    ]
    critic_tasks = [
        task for task in tasks if task.required_artifact_type == "ChemicalStrategyCritique"
    ]
    branch_ids = [
        int(re.search(r'"branch_id":(\d+)', task.objective).group(1)) for task in expansion_tasks
    ]
    assert branch_ids == [1]
    assert len(critic_tasks) == 1


def test_shared_model_ledger_returns_unused_tokens_to_other_branches() -> None:
    ledger = sequential_module._SharedModelCallLedger(
        sequential_module._NodeCallBudget(
            model_invocations=5,
            input_tokens=100,
            output_tokens=100,
            wall_time_s=60.0,
        ),
        (),
        protected_model_invocations=1,
        protected_input_tokens=10,
        protected_output_tokens=20,
    )

    first, first_reason = ledger.reserve(input_tokens=20, output_tokens=30)
    second, second_reason = ledger.reserve(input_tokens=20, output_tokens=30)
    blocked, blocked_reason = ledger.reserve(input_tokens=20, output_tokens=30)

    assert first is not None and first_reason == ""
    assert second is not None and second_reason == ""
    assert blocked is None
    assert blocked_reason == "output_token_allocation_exhausted"

    ledger.settle(
        first,
        WorkerRunRecord(
            run_id="ledger:first",
            task_id="ledger:first",
            case_id="ledger",
            status="accepted_draft",
            backend="test",
            usage={"input_tokens": 5, "output_tokens": 5},
        ),
    )
    third, third_reason = ledger.reserve(input_tokens=20, output_tokens=30)

    assert third is not None and third_reason == ""
    snapshot = ledger.snapshot()
    assert snapshot["committed"]["output_tokens"] == 5
    assert snapshot["inflight"]["output_tokens"] == 60
    assert snapshot["protected_final_critics"]["output_tokens"] == 20


def test_stock_closed_branch_waits_until_all_root_strategies_exist() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=6,
        max_skeletons=6,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=3,
        max_node_expansions_per_branch=25,
    )
    runner = SequentialStrategyDirectorRunner(
        node_executor=_fake_executor,
        critic_executor=_fake_critic_executor,
        stock_membership=lambda values: {value: True for value in values},
    )
    result = runner(_spec(context), context, "initial_architecture", config)

    assert result.usage["model_invocations"] == 9
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert len(plan.route_families) == 3
    assert plan.shared_intermediates == ()


def test_parallel_branches_stop_after_first_stock_closed_route() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=6,
        max_skeletons=6,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=3,
        strategy_branch_workers=3,
        stop_on_first_stock_closed_branch=True,
        max_node_expansions_per_branch=25,
    )
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def concurrent_executor(task):
        nonlocal active, maximum_active
        if task.required_artifact_type != "RetrosynthesisProposalReport":
            return _fake_executor(task)
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.03)
            return _fake_executor(task)
        finally:
            with lock:
                active -= 1

    runner = SequentialStrategyDirectorRunner(
        node_executor=concurrent_executor,
        critic_executor=_fake_critic_executor,
        # Branch 1 yields CC + O and closes. Branches 2/3 retain at least one
        # open leaf, but may finish their already in-flight first call.
        stock_membership=lambda values: {value: value in {"CC", "O"} for value in values},
    )
    result = runner(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert maximum_active == 3
    assert result.usage["stock_closed_early_stop_triggered"] is True
    assert result.usage["stock_closed_branch_count"] == 1
    # Once all three first calls are in flight, a losing branch may settle and
    # begin one further call before the winning branch publishes closure.
    # The bounded overshoot is therefore at most one call per losing branch.
    assert 9 <= result.usage["model_invocations"] <= 11
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert len(plan.multi_step_skeletons) == 3
    # One strategy call plus one Route Builder call gives ``2 compact calls``.
    # A losing branch is allowed to start one more call before the winning
    # branch publishes the concurrent early-stop event, giving ``3``.  Assert
    # that bounded scheduling invariant rather than one thread interleaving.
    call_counts = [
        int(row["summary"].split(" compact calls", 1)[0].rsplit(" ", 1)[-1])
        for row in plan.multi_step_skeletons
    ]
    assert all(value in {2, 3} for value in call_counts)
    assert 2 in call_counts


def test_node_expansion_rejects_more_than_four_atom_contributing_precursors() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
    )

    def overbranched_executor(task):
        record = _fake_executor(task)
        if task.required_artifact_type == "StrategyCardReport":
            return record
        artifact = dict(record.output_artifact or {})
        payload = dict(artifact.get("payload") or {})
        candidates = [dict(row) for row in payload.get("candidates") or []]
        candidates[0]["precursor_smiles"] = ["C", "CC", "CCC", "CCCC", "CCCCC"]
        payload["candidates"] = candidates
        artifact["payload"] = payload
        return replace(record, output_artifact=artifact)

    result = SequentialStrategyDirectorRunner(node_executor=overbranched_executor)(
        _spec(context), context, "initial_architecture", config
    )

    assert result.state is AgentState.FAILED
    assert result.usage["model_invocations"] == 2


def test_node_expansion_rejects_missing_product_atom_provenance() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=1,
    )

    def atom_deficient_executor(task):
        record = _fake_executor(task)
        if task.required_artifact_type == "StrategyCardReport":
            return record
        artifact = dict(record.output_artifact or {})
        payload = dict(artifact.get("payload") or {})
        candidates = [dict(row) for row in payload.get("candidates") or []]
        candidates[0]["precursor_smiles"] = ["CO"]
        payload["candidates"] = candidates
        artifact["payload"] = payload
        return replace(record, output_artifact=artifact)

    result = SequentialStrategyDirectorRunner(node_executor=atom_deficient_executor)(
        _spec(context), context, "initial_architecture", config
    )

    assert result.state is AgentState.FAILED
    assert result.usage["model_invocations"] == 2


def test_reactionjson_is_the_only_precursor_structure_authority() -> None:
    task = type(
        "RouteTask",
        (),
        {
            "required_artifact_type": "RetrosynthesisProposalReport",
            "objective": ('CompactBranchContext:{"branch_id":1,"selected_open_leaf":"CCO"}'),
            "task_id": "reactionjson-authority",
            "case_id": "case",
            "input_refs": [],
        },
    )()
    record = _fake_executor(task)
    artifact = dict(record.output_artifact or {})
    payload = dict(artifact.get("payload") or {})
    candidate = dict(payload["candidates"][0])
    candidate["precursor_smiles"] = ["CCO"]
    candidate["reaction_operations"] = [{"op": "break_bond", "map_a": 2, "map_b": 3}]
    payload["candidates"] = [candidate]
    artifact["payload"] = payload
    record = replace(record, output_artifact=artifact)

    expansion = _expansion_from_record(
        record,
        expected_product="CCO",
        require_strategy_card=True,
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        require_reaction_operations=True,
    )

    assert expansion is not None
    assert expansion.precursor_smiles == ("CC", "O")
    assert expansion.reactionjson_audit["precursor_smiles"] == ["CC", "O"]
    assert expansion.reactionjson_audit["implicit_valence_completion_maps"] == [
        2,
        3,
    ]
    assert expansion.precursor_smiles != ("CCO",)


def test_bound_host_replay_is_serialized_without_executing_graph_edits_twice(
    monkeypatch,
) -> None:
    compiler = sequential_module.RouteJSONCompiler()
    operations = [{"op": "break_bond", "map_a": 2, "map_b": 3}]
    materialized = compiler.compile_step(
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        operations=operations,
        expected_product_smiles="CCO",
    )
    row = {
        "step_id": "bound:1",
        "product_smiles": materialized.product_smiles,
        "mapped_product_smiles": materialized.mapped_product_smiles,
        "precursor_smiles": list(materialized.precursor_smiles),
        "mapped_precursor_smiles": list(materialized.mapped_precursor_smiles),
        "reaction_operations": operations,
        "reactionjson_audit": dict(materialized.audit),
    }

    def fail_if_replayed(*_args, **_kwargs):
        raise AssertionError("a bound host replay must not execute twice")

    monkeypatch.setattr(
        sequential_module.RouteJSONCompiler,
        "compile_route_graph",
        fail_if_replayed,
    )
    route = sequential_module._host_route_json_from_steps([row])

    assert route[0]["product_smiles"] == "CCO"
    assert route[0]["precursor_smiles"] == ["CC", "O"]
    assert route[0]["reactionjson_audit"]["accepted"] is True


def test_paper_node_prompt_does_not_expose_anchor_progress_as_authority() -> None:
    prompt = _node_prompt(
        target="CCO",
        branch_index=0,
        lens="test",
        selected_product="CO",
        selected_product_mapped="[CH3:1][OH:2]",
        steps=[
            {
                "product_smiles": "CCO",
                "precursor_smiles": ["C", "CO"],
                "transformation_hypothesis": "route-defining cleavage",
                "strategy_anchor": True,
            }
        ],
        open_leaves=["CO"],
        prior_rejections=[],
        repair=False,
        strategy_card=_strategy_card(1),
        forbidden_strategy_cards=[],
        host_failure_feedback={},
        paper_matched=True,
    )

    assert '"strategy_anchor_progress"' not in prompt
    assert '"remaining_map_pairs"' not in prompt
    assert "steering hypothesis" in prompt
    assert "guides the whole pathway" in prompt
    assert "work out a complete chemically coherent pathway" in prompt
    assert "accepted_strategy_spine" not in prompt
    assert "key_step_seen" not in prompt
    assert "skip an unreasonable planned step" not in prompt


def test_strategy_card_requires_all_replayed_anchor_pairs() -> None:
    card = normalize_strategy_card(
        {
            "scaffold_motif": "two-ring scaffold",
            "key_forward_transformation": "two ordered closures",
            "key_bond_changes": ["map 1-map 2", "map 3-map 4"],
            "functional_group_conflicts": [],
            "protection_policy": "none",
            "stereochemical_plan": "substrate controlled",
            "strategic_step_count": 2,
            "strategy_signature": "two anchors",
        }
    )
    first = {
        "strategy_card": card,
        "strategy_anchor": False,
        "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
    }
    second = {
        "strategy_card": card,
        "strategy_anchor": False,
        "reaction_operations": [{"op": "break_bond", "map_a": 3, "map_b": 4}],
    }

    assert sequential_module._strategy_anchor_fulfilled_for_card([first], card) is False
    progress = sequential_module._strategy_anchor_progress([first], card)
    assert progress["realized_map_pairs"] == ["map_pair:1:2"]
    assert progress["remaining_map_pairs"] == ["map_pair:3:4"]
    assert progress["authority"] == "report_only_diagnostic"
    assert progress["grants_strategy_adherence"] is False
    assert progress["grants_strategy_completion"] is False
    assert sequential_module._strategy_anchor_fulfilled_for_card([first, second], card) is True


def test_paper_route_builder_has_no_terminal_decision_surface() -> None:
    assert not hasattr(sequential_module, "_route_builder_decision")
    assert not hasattr(sequential_module, "_route_builder_handoff_admission")


def _builder_history_path() -> list[dict]:
    return [
        {
            "step_id": "step-1",
            "product_smiles": "CCO",
            "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "precursor_smiles": ["CC", "O"],
            "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
            "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
            "transformation_hypothesis": "C-O disconnection",
        }
    ]


def test_connected_builder_history_omits_non_authoritative_move_role() -> None:
    steps = _builder_history_path()
    steps[0]["step_role"] = "enabling"
    lineage = sequential_module._route_lineage_context(
        steps,
        selected_product="CC",
        selected_product_mapped="[CH3:1][CH3:2]",
    )
    assert [dict(row) for row in lineage.reaction_spine] == [
        {
            "step_id": "step-1",
            "reaction_family": "C-O disconnection",
            "checkpoint_relation": "",
            "edit_summary": "break bond maps 2-3",
        }
    ]


def test_builder_lineage_uses_host_mapped_sibling_when_aiz_stereo_projection_differs() -> None:
    organozinc_mapped = (
        "[C@H:10]1([Zn:24][Br:25])[C@H:11]([CH3:12])[CH2:13]"
        "[CH2:14][C@@H:15]1[CH:16]([CH3:17])[OH:21]"
    )
    organozinc_host = "CC(O)[C@H]1CC[C@@H](C)[C@H]1[Zn]Br"
    organozinc_aiz = "CC(O)[C@H]1CC[C@@H](C)[C@@H]1[Zn]Br"
    sibling_mapped = "[CH3:1][CH2:9][Br:23]"
    prompt = _node_prompt(
        target="CC",
        branch_index=0,
        lens="neutral",
        selected_product=organozinc_aiz,
        selected_product_mapped=organozinc_mapped,
        steps=(
            {
                "step_id": "negishi",
                "product_smiles": "CC",
                "mapped_product_smiles": "[CH3:9][CH3:10]",
                "precursor_smiles": ["CCBr", organozinc_host],
                "mapped_precursor_smiles": [sibling_mapped, organozinc_mapped],
                "transformation_hypothesis": "Negishi coupling split",
                "reaction_operations": [{"op": "break_bond", "map_a": 9, "map_b": 10}],
            },
        ),
        open_leaves=(organozinc_aiz,),
        prior_rejections=(),
        repair=False,
        strategy_card={
            "strategy_query": "Join two sectors by Negishi coupling.",
            "critical_assumption": "The two handles are compatible.",
            "critic_checkpoint": "Audit C-C bond formation.",
        },
        forbidden_strategy_cards=(),
        host_failure_feedback={},
        paper_matched=True,
    )

    context = json.loads(prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
    assert [row["reaction_family"] for row in context["connected_path_reactions"]] == [
        "Negishi coupling split"
    ]
    assert context["current_split_context"] == {
        "parent_reaction": "Negishi coupling split",
        "parent_step_id": "negishi",
        "co_precursors": [
            {
                "mapped_smiles": sibling_mapped,
                "path_status": "not_expanded_on_current_path",
            }
        ],
    }


def test_split_context_tracks_duplicate_precursors_by_mapped_occurrence() -> None:
    context = sequential_module._current_split_context(
        (
            {
                "step_id": "three-way-split",
                "product_smiles": "CCOCC",
                "mapped_product_smiles": "[CH3:1][CH2:2][O:5][CH2:3][CH3:4]",
                "precursor_smiles": ["CC", "CC", "O"],
                "mapped_precursor_smiles": [
                    "[CH3:1][CH3:2]",
                    "[CH3:3][CH3:4]",
                    "[OH2:5]",
                ],
                "reaction_family": "three-way split",
            },
            {
                "step_id": "expand-first-ethane",
                "product_smiles": "CC",
                "mapped_product_smiles": "[CH3:1][CH3:2]",
                "precursor_smiles": ["C", "C"],
                "mapped_precursor_smiles": ["[CH4:1]", "[CH4:2]"],
            },
        ),
        selected_product="O",
        selected_product_mapped="[OH2:5]",
    )

    assert context["co_precursors"] == [
        {
            "mapped_smiles": "[CH3:1][CH3:2]",
            "path_status": "expanded_on_current_path",
        },
        {
            "mapped_smiles": "[CH3:3][CH3:4]",
            "path_status": "not_expanded_on_current_path",
        },
    ]


def test_durable_worker_journal_replays_only_identical_task_contracts(tmp_path) -> None:
    context = _context()
    spec = replace(
        _spec(context),
        metadata={
            **dict(_spec(context).metadata),
            "allowed_workdir": str(tmp_path),
            "durable_worker_journal": True,
        },
    )
    task = WorkerTask(
        task_id="journaled-node",
        case_id="case",
        task_type="route_step_materialization",
        required_artifact_type="RetrosynthesisProposalReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(max_tool_calls=0),
        objective="stable prompt",
    )
    calls = []

    def executor(value):
        calls.append(value.task_id)
        return WorkerRunRecord(
            run_id=f"{value.task_id}:run",
            task_id=value.task_id,
            case_id=value.case_id,
            status="accepted_draft",
            output_artifact={"artifact_type": "RetrosynthesisProposalReport"},
        )

    first = SequentialStrategyDirectorRunner(node_executor=executor)
    first._prepare_worker_record_journal(spec)
    original = first._run_journaled_worker(executor, task)
    cached_in_process = first._run_journaled_worker(executor, task)

    assert original.to_dict() == cached_in_process.to_dict()
    assert calls == ["journaled-node"]
    assert first._replayed_worker_record_count == 1

    resumed = SequentialStrategyDirectorRunner(node_executor=executor)
    resumed._prepare_worker_record_journal(spec)
    cached_after_restart = resumed._run_journaled_worker(executor, task)

    assert cached_after_restart.to_dict() == original.to_dict()
    assert calls == ["journaled-node"]
    assert resumed._replayed_worker_record_count == 1

    changed = replace(task, objective="changed prompt")
    resumed._run_journaled_worker(executor, changed)
    assert calls == ["journaled-node", "journaled-node"]


def test_model_io_journal_streams_input_and_output_to_one_jsonl(tmp_path) -> None:
    context = _context()
    spec = replace(
        _spec(context),
        metadata={
            **dict(_spec(context).metadata),
            "allowed_workdir": str(tmp_path),
            "durable_worker_journal": True,
        },
    )
    task = WorkerTask(
        task_id="model-io-node",
        case_id="case",
        task_type="route_step_materialization",
        required_artifact_type="RetrosynthesisProposalReport",
        input_refs=["context:one"],
        allowed_tools=[],
        budget=WorkerBudget(max_tool_calls=0, reasoning_effort="medium"),
        objective="complete model prompt",
        model="gpt-test",
    )
    observed_input_lines: list[dict] = []

    def executor(value):
        log_path = tmp_path / "model-io.jsonl"
        observed_input_lines.extend(
            json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
        )
        return WorkerRunRecord(
            run_id=f"{value.task_id}:run",
            task_id=value.task_id,
            case_id=value.case_id,
            status="accepted_draft",
            stdout="complete model output",
            output_artifact={"artifact_type": "RetrosynthesisProposalReport"},
            usage={"input_tokens": 10, "output_tokens": 5},
        )

    runner = SequentialStrategyDirectorRunner(node_executor=executor)
    runner._prepare_worker_record_journal(spec)
    runner._run_journaled_worker(executor, task)

    rows = [
        json.loads(line)
        for line in (tmp_path / "model-io.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in observed_input_lines] == ["model_input"]
    assert [row["event"] for row in rows] == ["model_input", "model_output"]
    assert rows[0]["prompt"] == "complete model prompt"
    assert rows[1]["stdout"] == "complete model output"
    assert rows[1]["status"] == "schema_accepted"
    assert rows[1]["status_scope"] == "worker_output_schema_validation"
    assert rows[1]["worker_record_status"] == "accepted_draft"
    assert rows[1]["output_artifact"]["artifact_type"] == ("RetrosynthesisProposalReport")


def test_worker_journal_seed_replays_into_a_fresh_run_journal(tmp_path) -> None:
    context = _context()
    source_dir = tmp_path / "source"
    fresh_dir = tmp_path / "fresh"
    source_dir.mkdir()
    fresh_dir.mkdir()
    base = _spec(context)
    source_spec = replace(
        base,
        metadata={
            **dict(base.metadata),
            "allowed_workdir": str(source_dir),
            "durable_worker_journal": True,
        },
    )
    task = WorkerTask(
        task_id="director:old:branch:1:strategy:1",
        case_id="case",
        task_type="strategic_disconnection_mining",
        required_artifact_type="StrategyCardReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(
            timeout_s=10,
            max_worker_runs=1,
            reasoning_effort="medium",
        ),
        objective="same prompt",
        allowed_workdir=str(source_dir),
    )
    calls: list[str] = []

    def executor(value: WorkerTask) -> WorkerRunRecord:
        calls.append(value.task_id)
        return WorkerRunRecord(
            run_id=f"{value.task_id}:run",
            task_id=value.task_id,
            case_id=value.case_id,
            status="accepted_draft",
            metadata={
                "model": value.model,
                "model_reasoning_effort": value.budget.reasoning_effort,
            },
            output_artifact={
                "schema_version": "strategycardreport.draft.v1",
                "artifact_id": f"{value.task_id}:StrategyCardReport",
                "artifact_type": "StrategyCardReport",
                "case_id": value.case_id,
                "source": "test",
                "input_refs": [],
                "evidence_refs": [],
                "validation_status": "draft",
            },
            output_validation={"accepted": True},
        )

    source = SequentialStrategyDirectorRunner(node_executor=executor)
    source._prepare_worker_record_journal(source_spec)
    original = source._run_journaled_worker(executor, task)
    seed_path = source_dir / "sequential-director-worker-records.jsonl"

    fresh_spec = replace(
        base,
        metadata={
            **dict(base.metadata),
            "allowed_workdir": str(fresh_dir),
            "durable_worker_journal": True,
        },
    )
    recovered = SequentialStrategyDirectorRunner(
        node_executor=executor,
        worker_record_seed_path=str(seed_path),
        worker_record_seed_recovery_mode="exact_model_io_v1",
    )
    recovered._prepare_worker_record_journal(fresh_spec)
    replayed = recovered._run_journaled_worker(
        executor,
        replace(
            task,
            task_id="director:new:branch:1:strategy:1",
            allowed_workdir=str(fresh_dir),
        ),
    )

    # The seed recovery path permits only a no-tool task's operational cwd to
    # relocate; prompt/model/schema identity remains digest-bound.
    assert replayed.to_dict() == original.to_dict()
    assert calls == ["director:old:branch:1:strategy:1"]
    assert recovered._seeded_worker_record_count == 1
    assert recovered._replayed_worker_record_count == 1
    assert recovered._exact_seed_replay_count == 1
    fresh_journal = fresh_dir / "sequential-director-worker-records.jsonl"
    assert fresh_journal.is_file()
    fresh_row = json.loads(fresh_journal.read_text(encoding="utf-8").splitlines()[0])
    assert fresh_row["portable_model_input_sha256"] == (
        sequential_module._portable_model_input_sha256(
            replace(
                task,
                task_id="director:new:branch:1:strategy:1",
                allowed_workdir=str(fresh_dir),
            )
        )
    )

    # A replayed alias is a complete next-generation seed even though no
    # provider call (and therefore no new model-io event) occurred in it.
    second_dir = tmp_path / "second-recovery"
    second_dir.mkdir()
    second_spec = replace(
        base,
        metadata={
            **dict(base.metadata),
            "allowed_workdir": str(second_dir),
            "durable_worker_journal": True,
        },
    )
    second = SequentialStrategyDirectorRunner(
        node_executor=executor,
        worker_record_seed_path=str(fresh_journal),
        worker_record_seed_recovery_mode="exact_model_io_v1",
    )
    second._prepare_worker_record_journal(second_spec)
    transitive = second._run_journaled_worker(
        executor,
        replace(
            task,
            task_id="director:second:branch:1:strategy:1",
            allowed_workdir=str(second_dir),
        ),
    )
    assert transitive.to_dict() == original.to_dict()
    assert calls == ["director:old:branch:1:strategy:1"]
    assert second._exact_seed_replay_count == 1

    changed = recovered._run_journaled_worker(
        executor,
        replace(
            task,
            task_id="director:changed:branch:1:strategy:1",
            objective="changed prompt",
            allowed_workdir=str(fresh_dir),
        ),
    )
    assert changed.task_id == "director:changed:branch:1:strategy:1"
    assert calls == [
        "director:old:branch:1:strategy:1",
        "director:changed:branch:1:strategy:1",
    ]


def test_legacy_alias_seed_recovers_input_from_record_provenance(tmp_path) -> None:
    context = _context()
    source_dir = tmp_path / "source"
    alias_dir = tmp_path / "legacy-alias"
    fresh_dir = tmp_path / "fresh"
    source_dir.mkdir()
    alias_dir.mkdir()
    fresh_dir.mkdir()
    base = _spec(context)
    source_spec = replace(
        base,
        metadata={
            **dict(base.metadata),
            "allowed_workdir": str(source_dir),
            "durable_worker_journal": True,
        },
    )
    task = WorkerTask(
        task_id="director:origin:branch:1:strategy:1",
        case_id="case",
        task_type="strategic_disconnection_mining",
        required_artifact_type="StrategyCardReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(reasoning_effort="medium"),
        objective="same prompt",
        allowed_workdir=str(source_dir),
    )
    calls: list[str] = []

    def executor(value: WorkerTask) -> WorkerRunRecord:
        calls.append(value.task_id)
        return WorkerRunRecord(
            run_id=f"{value.task_id}:run",
            task_id=value.task_id,
            case_id=value.case_id,
            status="accepted_draft",
            metadata={
                "model": value.model,
                "model_reasoning_effort": value.budget.reasoning_effort,
            },
            output_artifact={
                "schema_version": "strategycardreport.draft.v1",
                "artifact_id": f"{value.task_id}:StrategyCardReport",
                "artifact_type": "StrategyCardReport",
                "case_id": value.case_id,
                "source": "test",
                "input_refs": [],
                "evidence_refs": [],
                "validation_status": "draft",
            },
            output_validation={"accepted": True},
        )

    source = SequentialStrategyDirectorRunner(node_executor=executor)
    source._prepare_worker_record_journal(source_spec)
    original = source._run_journaled_worker(executor, task)
    source_row = json.loads(
        (source_dir / "sequential-director-worker-records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    source_row.pop("portable_model_input_sha256")
    source_row["task_id"] = "director:alias:branch:1:strategy:1"
    source_row["task_contract_sha256"] = "legacy-alias-contract"
    source_row["record"]["metadata"]["event_log_path"] = str(
        source_dir / "codex_worker_events" / "source.jsonl"
    )
    (alias_dir / "sequential-director-worker-records.jsonl").write_text(
        json.dumps(source_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fresh_spec = replace(
        base,
        metadata={
            **dict(base.metadata),
            "allowed_workdir": str(fresh_dir),
            "durable_worker_journal": True,
        },
    )
    recovered = SequentialStrategyDirectorRunner(
        node_executor=executor,
        worker_record_seed_path=str(alias_dir / "sequential-director-worker-records.jsonl"),
        worker_record_seed_recovery_mode="exact_model_io_v1",
    )
    recovered._prepare_worker_record_journal(fresh_spec)
    replayed = recovered._run_journaled_worker(
        executor,
        replace(
            task,
            task_id="director:new:branch:1:strategy:1",
            allowed_workdir=str(fresh_dir),
        ),
    )

    assert replayed.task_id == original.task_id
    assert replayed.output_artifact == original.output_artifact
    assert replayed.output_validation == original.output_validation
    assert calls == ["director:origin:branch:1:strategy:1"]
    assert recovered._exact_seed_replay_count == 1


def test_worker_journal_resume_reruns_cancelled_worker_record(tmp_path) -> None:
    context = _context()
    spec = replace(
        _spec(context),
        metadata={
            **dict(_spec(context).metadata),
            "allowed_workdir": str(tmp_path),
            "durable_worker_journal": True,
        },
    )
    task = WorkerTask(
        task_id="director:resume:branch:2:editor:1",
        case_id="case",
        task_type="paper_matched_route_editor",
        required_artifact_type="RouteJSONDraft",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(reasoning_effort="medium"),
        objective="repair the interrupted route",
        allowed_workdir=str(tmp_path),
    )
    calls: list[str] = []

    def cancelled_executor(value: WorkerTask) -> WorkerRunRecord:
        calls.append("cancelled")
        return WorkerRunRecord(
            run_id=f"{value.task_id}:cancelled",
            task_id=value.task_id,
            case_id=value.case_id,
            status="cancelled",
        )

    interrupted = SequentialStrategyDirectorRunner(node_executor=cancelled_executor)
    interrupted._prepare_worker_record_journal(spec)
    interrupted._run_journaled_worker(cancelled_executor, task)

    def completed_executor(value: WorkerTask) -> WorkerRunRecord:
        calls.append("completed")
        return WorkerRunRecord(
            run_id=f"{value.task_id}:completed",
            task_id=value.task_id,
            case_id=value.case_id,
            status="accepted_draft",
            output_artifact={"artifact_type": "RouteJSONDraft"},
        )

    resumed = SequentialStrategyDirectorRunner(node_executor=completed_executor)
    resumed._prepare_worker_record_journal(spec)
    result = resumed._run_journaled_worker(completed_executor, task)

    assert calls == ["cancelled", "completed"]
    assert result.status == "accepted_draft"
    assert resumed._replayed_worker_record_count == 0
    assert (
        len(
            (tmp_path / "sequential-director-worker-records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 2
    )


def test_worker_journal_resume_reruns_provider_error_record(tmp_path) -> None:
    context = _context()
    spec = replace(
        _spec(context),
        metadata={
            **dict(_spec(context).metadata),
            "allowed_workdir": str(tmp_path),
            "durable_worker_journal": True,
        },
    )
    task = WorkerTask(
        task_id="director:resume:branch:1:builder:4",
        case_id="case",
        task_type="paper_matched_route_step",
        required_artifact_type="RetrosynthesisProposalReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(reasoning_effort="medium"),
        objective="continue the interrupted route",
        allowed_workdir=str(tmp_path),
    )
    calls: list[str] = []

    def unavailable_executor(value: WorkerTask) -> WorkerRunRecord:
        calls.append("provider_error")
        return WorkerRunRecord(
            run_id=f"{value.task_id}:provider-error",
            task_id=value.task_id,
            case_id=value.case_id,
            status="provider_error",
            output_validation={
                "accepted": False,
                "reasons": ["provider_auth_unavailable"],
            },
        )

    interrupted = SequentialStrategyDirectorRunner(
        node_executor=unavailable_executor
    )
    interrupted._prepare_worker_record_journal(spec)
    first = interrupted._run_journaled_worker(unavailable_executor, task)
    assert first.status == "provider_error"

    def completed_executor(value: WorkerTask) -> WorkerRunRecord:
        calls.append("completed")
        return WorkerRunRecord(
            run_id=f"{value.task_id}:completed",
            task_id=value.task_id,
            case_id=value.case_id,
            status="accepted_draft",
            output_artifact={
                "artifact_type": "RetrosynthesisProposalReport"
            },
            output_validation={"accepted": True},
        )

    resumed = SequentialStrategyDirectorRunner(node_executor=completed_executor)
    resumed._prepare_worker_record_journal(spec)
    result = resumed._run_journaled_worker(completed_executor, task)

    assert calls == ["provider_error", "completed"]
    assert result.status == "accepted_draft"
    assert resumed._replayed_worker_record_count == 0


def test_worker_journal_resume_reruns_legacy_capacity_failure_record(tmp_path) -> None:
    context = _context()
    spec = replace(
        _spec(context),
        metadata={
            **dict(_spec(context).metadata),
            "allowed_workdir": str(tmp_path),
            "durable_worker_journal": True,
        },
    )
    task = WorkerTask(
        task_id="director:resume:branch:1:strategy:critic",
        case_id="case",
        task_type="paper_matched_strategy_critic",
        required_artifact_type="StrategyPortfolioReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(reasoning_effort="medium"),
        objective="review the interrupted strategy",
        allowed_workdir=str(tmp_path),
    )
    calls: list[str] = []

    def legacy_capacity_executor(value: WorkerTask) -> WorkerRunRecord:
        calls.append("capacity")
        message = "Selected model is at capacity. Please try a different model."
        return WorkerRunRecord(
            run_id=f"{value.task_id}:capacity",
            task_id=value.task_id,
            case_id=value.case_id,
            status="rejected_output",
            stdout=(
                '{"type":"error","message":"' + message + '"}\n'
                '{"type":"turn.failed","error":{"message":"' + message + '"}}'
            ),
            exit_code=1,
            output_validation={"accepted": False},
            metadata={
                "event_summary": {
                    "turn_completed": False,
                    "turn_failed": True,
                    "last_terminal_event_type": "turn.failed",
                    "fatal_error": message,
                }
            },
        )

    interrupted = SequentialStrategyDirectorRunner(
        node_executor=legacy_capacity_executor
    )
    interrupted._prepare_worker_record_journal(spec)
    first = interrupted._run_journaled_worker(legacy_capacity_executor, task)
    assert codex_worker_module.worker_provider_failure_reason(first) == (
        "provider_service_unavailable"
    )

    def completed_executor(value: WorkerTask) -> WorkerRunRecord:
        calls.append("completed")
        return WorkerRunRecord(
            run_id=f"{value.task_id}:completed",
            task_id=value.task_id,
            case_id=value.case_id,
            status="accepted_draft",
            output_artifact={"artifact_type": "StrategyPortfolioReport"},
            output_validation={"accepted": True},
        )

    resumed = SequentialStrategyDirectorRunner(node_executor=completed_executor)
    resumed._prepare_worker_record_journal(spec)
    result = resumed._run_journaled_worker(completed_executor, task)

    assert calls == ["capacity", "completed"]
    assert result.status == "accepted_draft"
    assert resumed._replayed_worker_record_count == 0


def test_provider_error_is_not_schema_rejection_or_semantic_model_usage() -> None:
    provider_error = WorkerRunRecord(
        run_id="provider-error:run",
        task_id="provider-error",
        case_id="case",
        status="provider_error",
        output_validation={
            "accepted": False,
            "reasons": ["provider_auth_unavailable"],
        },
        usage={"input_tokens": 900, "output_tokens": 100},
    )
    completed = WorkerRunRecord(
        run_id="completed:run",
        task_id="completed",
        case_id="case",
        status="accepted_draft",
        output_validation={"accepted": True},
        usage={"input_tokens": 30, "output_tokens": 10},
    )

    usage = sequential_module._aggregate_usage(
        (provider_error, completed), elapsed_s=1.0
    )
    assert usage["attempt_runs"] == 2
    assert usage["provider_failure_count"] == 1
    assert usage["model_invocations"] == 1
    assert usage["input_tokens"] == 30
    assert usage["output_tokens"] == 10
    assert sequential_module._model_output_validation_status(provider_error) == (
        "provider_error"
    )

    quota = sequential_module._NodeCallBudget(
        model_invocations=2,
        input_tokens=10_000,
        output_tokens=10_000,
        wall_time_s=30.0,
    )
    ledger = sequential_module._SharedModelCallLedger(quota, ())
    reservation, reason = ledger.reserve(input_tokens=500, output_tokens=100)
    assert reservation is not None and reason == ""
    ledger.settle(reservation, provider_error)
    assert ledger.snapshot()["committed"] == {
        "model_invocations": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_codex_worker_classifies_unfinished_401_as_provider_error() -> None:
    task = WorkerTask(
        task_id="provider-auth-worker",
        case_id="case",
        task_type="paper_matched_route_step",
        required_artifact_type="RetrosynthesisProposalReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(reasoning_effort="medium"),
        objective="propose one route step",
    )

    record = codex_worker_module.run_codex_worker(
        task,
        runner=lambda _task: codex_worker_module.WorkerProcessResult(
            stderr="401 Unauthorized: failed to refresh token",
            exit_code=1,
            backend="codex_cli",
            metadata={
                "event_summary": {
                    "turn_completed": False,
                    "fatal_error": "refresh_token_reused",
                }
            },
        ),
    )

    assert record.status == "provider_error"
    assert codex_worker_module.worker_provider_failure_reason(record) == (
        "provider_auth_unavailable"
    )


def test_codex_worker_classifies_unfinished_capacity_as_provider_error() -> None:
    task = WorkerTask(
        task_id="provider-capacity-worker",
        case_id="case",
        task_type="paper_matched_strategy_critic",
        required_artifact_type="StrategyPortfolioReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(reasoning_effort="medium"),
        objective="review one strategy",
    )
    message = "Selected model is at capacity. Please try a different model."

    record = codex_worker_module.run_codex_worker(
        task,
        runner=lambda _task: codex_worker_module.WorkerProcessResult(
            stdout=(
                '{"type":"error","message":"' + message + '"}\n'
                '{"type":"turn.failed","error":{"message":"' + message + '"}}'
            ),
            exit_code=1,
            backend="codex_cli",
            metadata={
                "event_summary": {
                    "turn_completed": False,
                    "turn_failed": True,
                    "last_terminal_event_type": "turn.failed",
                    "fatal_error": message,
                }
            },
        ),
    )

    assert record.status == "provider_error"
    assert codex_worker_module.worker_provider_failure_reason(record) == (
        "provider_service_unavailable"
    )


def test_reactionjson_failure_returns_causal_replay_diagnostic() -> None:
    task = type(
        "RouteTask",
        (),
        {
            "required_artifact_type": "RetrosynthesisProposalReport",
            "objective": ('CompactBranchContext:{"branch_id":1,"selected_open_leaf":"CCO"}'),
            "task_id": "reactionjson-diagnostic",
            "case_id": "case",
            "input_refs": [],
        },
    )()
    record = _fake_executor(task)
    artifact = dict(record.output_artifact or {})
    payload = dict(artifact.get("payload") or {})
    candidate = dict(payload["candidates"][0])
    candidate["precursor_smiles"] = []
    candidate["reaction_operations"] = [{"op": "break_bond", "map_a": 1, "map_b": 3}]
    payload["candidates"] = [candidate]
    artifact["payload"] = payload
    record = replace(record, output_artifact=artifact)

    diagnostic = _expansion_rejection_diagnostic(
        record,
        expected_product="CCO",
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        require_reaction_operations=True,
    )

    assert diagnostic["reason"] == "strategy_graph_edit_replay_failed"
    assert "bond_missing" in diagnostic["replay_error"]
    assert diagnostic["operation_index"] == 0
    assert diagnostic["failed_operation"] == {
        "op": "break_bond",
        "map_a": 1,
        "map_b": 3,
    }
    assert "attempted_operations" not in diagnostic


def test_graph_finalization_failure_reaches_builder_with_root_cause() -> None:
    candidate = {
        "candidate_id": "finalization-diagnostic",
        "product_smiles": "C",
        "precursor_smiles": [],
        "reaction_family": "invalid hydrogen edit used only as a diagnostic",
        "conditions": [],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "reaction_operations": [{"op": "set_explicit_h", "map_idx": 1, "count": 5}],
    }
    diagnostic = _expansion_rejection_diagnostic(
        _proposal_record(candidate, target="C"),
        expected_product="C",
        mapped_product_smiles="[CH4:1]",
        require_reaction_operations=True,
    )

    assert diagnostic["reason"] == "strategy_graph_edit_replay_failed"
    assert diagnostic["failure_stage"] == "graph_finalization"
    assert "AtomValenceException" in diagnostic["failure_detail"]

    prompt = _node_prompt(
        target="C",
        branch_index=0,
        lens="paper builder retry",
        selected_product="C",
        selected_product_mapped="[CH4:1]",
        steps=(),
        open_leaves=("C",),
        prior_rejections=({"product_smiles": "C", **diagnostic},),
        repair=False,
        strategy_card=_strategy_card(1),
        forbidden_strategy_cards=(),
        host_failure_feedback={},
        paper_matched=True,
    )
    context = json.loads(prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
    replay = context["last_rejection_for_this_leaf"]["replay_diagnostic"]
    assert replay["failure_stage"] == "graph_finalization"
    assert "AtomValenceException" in replay["failure_detail"]


def test_invalidated_alkene_stereo_maps_reach_builder_retry() -> None:
    candidate = {
        "candidate_id": "stereo-reference-diagnostic",
        "product_smiles": "C/C=C/C",
        "precursor_smiles": [],
        "reaction_family": "replace one alkene substituent",
        "conditions": [],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "reaction_operations": [
            {"op": "break_bond", "map_a": 1, "map_b": 2},
            {"op": "add_group", "map_idx": 2, "fragment_smiles": "[*]I"},
        ],
    }
    diagnostic = _expansion_rejection_diagnostic(
        _proposal_record(candidate, target="C/C=C/C"),
        expected_product="C/C=C/C",
        mapped_product_smiles="[CH3:1]/[CH:2]=[CH:3]/[CH3:4]",
        require_reaction_operations=True,
    )

    assert diagnostic["invalidated_bond_stereo"] == [
        {"map_a": 2, "map_b": 3, "previous_stereo": "E"}
    ]
    assert diagnostic["required_repair"] == (
        "add set_bond_stereo for each affected retained double bond"
    )

    prompt = _node_prompt(
        target="C/C=C/C",
        branch_index=0,
        lens="paper builder retry",
        selected_product="C/C=C/C",
        selected_product_mapped="[CH3:1]/[CH:2]=[CH:3]/[CH3:4]",
        steps=(),
        open_leaves=("C/C=C/C",),
        prior_rejections=(
            {"product_smiles": "C/C=C/C", **diagnostic},
        ),
        repair=False,
        strategy_card=_strategy_card(1),
        forbidden_strategy_cards=(),
        host_failure_feedback={},
        paper_matched=True,
    )
    context = json.loads(prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
    replay = context["last_rejection_for_this_leaf"]["replay_diagnostic"]
    assert replay["invalidated_bond_stereo"] == [
        {"map_a": 2, "map_b": 3, "previous_stereo": "E"}
    ]
    assert replay["required_repair"] == diagnostic["required_repair"]


def test_add_group_aromatic_failure_reaches_builder_as_typed_error() -> None:
    task = type(
        "RouteTask",
        (),
        {
            "required_artifact_type": "RetrosynthesisProposalReport",
            "objective": ('CompactBranchContext:{"branch_id":1,"selected_open_leaf":"CCO"}'),
            "task_id": "add-group-diagnostic",
            "case_id": "case",
            "input_refs": [],
        },
    )()
    record = _fake_executor(task)
    artifact = dict(record.output_artifact or {})
    payload = dict(artifact.get("payload") or {})
    candidate = dict(payload["candidates"][0])
    candidate["precursor_smiles"] = []
    candidate["reaction_operations"] = [
        {
            "op": "add_group",
            "map_idx": 2,
            "fragment_smiles": "[*]O",
            "order": 1.5,
        }
    ]
    payload["candidates"] = [candidate]
    artifact["payload"] = payload
    record = replace(record, output_artifact=artifact)

    diagnostic = _expansion_rejection_diagnostic(
        record,
        expected_product="CCO",
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        require_reaction_operations=True,
    )

    assert diagnostic["reason"] == "strategy_graph_edit_replay_failed"
    assert diagnostic["replay_error"] == ("reactionjson_aromatic_bond_requires_aromatic_atoms")
    assert diagnostic["operation_index"] == 0
    assert diagnostic["failed_operation"] == {
        "op": "add_group",
        "map_idx": 2,
        "fragment_smiles": "[*]O",
        "order": 1.5,
    }
    assert diagnostic["endpoint_aromaticity"] == {
        "anchor": False,
        "fragment_attachment": False,
    }
    assert diagnostic["allowed_orders"] == [1, 2, 3]


def test_paper_builder_receives_top_level_replay_failure_causally() -> None:
    prompt = _node_prompt(
        target="CCO",
        branch_index=0,
        lens="paper builder retry",
        selected_product="CCO",
        selected_product_mapped="[CH3:1][CH2:2][OH:3]",
        steps=(),
        open_leaves=("CCO",),
        prior_rejections=(
            {
                "reason": "strategy_graph_edit_replay_failed",
                "product_smiles": "CCO",
                "replay_error": "bond_missing: map 1-map 3",
                "operation_index": 0,
                "failed_operation": {
                    "op": "break_bond",
                    "map_a": 1,
                    "map_b": 3,
                },
            },
        ),
        repair=False,
        strategy_card=_strategy_card(1),
        forbidden_strategy_cards=(),
        host_failure_feedback={},
        paper_matched=True,
    )

    context = json.loads(prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
    rejection = context["last_rejection_for_this_leaf"]
    assert rejection["reason"] == "strategy_graph_edit_replay_failed"
    assert rejection["replay_diagnostic"]["replay_error"] == ("bond_missing: map 1-map 3")
    assert rejection["replay_diagnostic"]["operation_index"] == 0
    assert rejection["replay_diagnostic"]["failed_operation"] == {
        "op": "break_bond",
        "map_a": 1,
        "map_b": 3,
    }
    assert "attempted_operations" not in rejection["replay_diagnostic"]


def test_paper_builder_receives_nested_route_replay_failure_causally() -> None:
    prompt = _node_prompt(
        target="CBr",
        branch_index=0,
        lens="paper repair retry",
        selected_product="CBr",
        selected_product_mapped="[CH3:1][Br:2]",
        steps=(),
        open_leaves=("CBr",),
        prior_rejections=(
            {
                "reason": "candidate_does_not_extend_target_rooted_route",
                "product_smiles": "CBr",
                "routejson_replay_validation": {
                    "complete": False,
                    "reason": "routejson_target_rooted_dag_replay_failed",
                    "compiler_error": "reactionjson_fragment_map_collision",
                    "step_index": 3,
                    "operation_index": 1,
                    "failed_operation": {
                        "op": "add_group",
                        "map_idx": 1,
                        "fragment_smiles": "*[OH:32]",
                    },
                },
            },
        ),
        repair=True,
        strategy_card=_strategy_card(1),
        forbidden_strategy_cards=(),
        host_failure_feedback={},
        paper_matched=True,
    )

    context = json.loads(prompt.split("PaperMatchedRouteBuilderContext:\n", 1)[1])
    rejection = context["last_rejection_for_this_leaf"]
    assert rejection["reason"] == "candidate_does_not_extend_target_rooted_route"
    assert rejection["replay_diagnostic"] == {
        "reason": "routejson_target_rooted_dag_replay_failed",
        "compiler_error": "reactionjson_fragment_map_collision",
        "step_index": 3,
        "operation_index": 1,
        "failed_operation": {
            "op": "add_group",
            "map_idx": 1,
            "fragment_smiles": "*[OH:32]",
        },
    }


def test_strategy_card_survives_three_route_materialization_failures() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=3,
        require_strategy_graph_edits=True,
    )

    def invalid_materialization_executor(task):
        record = _fake_executor(task)
        if task.required_artifact_type == "StrategyCardReport":
            return record
        artifact = dict(record.output_artifact or {})
        payload = dict(artifact.get("payload") or {})
        candidate = dict(payload["candidates"][0])
        candidate["precursor_smiles"] = []
        candidate["reaction_operations"] = [{"op": "break_bond", "map_a": 1, "map_b": 3}]
        payload["candidates"] = [candidate]
        artifact["payload"] = payload
        return replace(record, output_artifact=artifact)

    result = SequentialStrategyDirectorRunner(node_executor=invalid_materialization_executor)(
        _spec(context), context, "initial_architecture", config
    )

    assert result.state is AgentState.FAILED
    assert result.usage["model_invocations"] == 4
    assert result.usage["materialization_retry_limit"] == 3
    retained = result.usage["retained_strategy_hypotheses"]
    assert len(retained) == 1
    assert retained[0]["strategy_signature"] == "branch-1-signature"
    assert retained[0]["key_forward_transformation"] == ("branch-1-key-construction")
    assert "strategy_graph_edit_replay_failed" in result.usage["rejection_reasons"]
    assert "materialization_retry_limit_reached" in result.usage["rejection_reasons"]


def test_strategy_card_without_any_materialized_step_cannot_report_success() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=3,
        require_strategy_graph_edits=True,
        require_complete_route_json=True,
    )

    result = SequentialStrategyDirectorRunner(node_executor=_fake_executor)(
        _spec(context), context, "initial_architecture", config
    )

    assert result.state is AgentState.FAILED
    assert result.output is None
    assert result.usage["accepted_expansions"] == 0
    assert len(result.usage["retained_strategy_hypotheses"]) == 1
    assert "route_json_missing" in result.usage["rejection_reasons"]


def test_later_materialization_failure_preserves_the_valid_route_prefix() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=2,
        require_strategy_graph_edits=True,
        require_complete_route_json=False,
    )

    def executor(task):
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)
        product = re.search(r'"selected_open_leaf":"([^"]+)"', task.objective).group(1)
        operations = (
            [{"op": "break_bond", "map_a": 1, "map_b": 2}]
            if product == "CCO"
            else [{"op": "break_bond", "map_a": 1, "map_b": 99}]
        )
        return _proposal_record(
            {
                "schema_version": "retrosynthesis_candidate.v1",
                "candidate_id": task.task_id,
                "product_smiles": product,
                "precursor_smiles": [],
                "reaction_family": "prefix preservation probe",
                "product_retron_type": "bond disconnection",
                "transformation_rationale": "exercise host graph edits",
                "conditions": [],
                "catalyst": "",
                "enzyme": "",
                "limitations": [],
                "required_validation": ["structure"],
                "no_solved_claim": True,
                "not_parent_route_proof": True,
                "reaction_operations": operations,
                "route_json": None,
            },
            target=product,
        )

    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        stock_membership=lambda values: {value: value in {"C", "O"} for value in values},
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert result.usage["accepted_expansions"] == 1
    plan = GlobalCampaignPlan.from_dict(result.output)
    steps = plan.multi_step_skeletons[0]["steps"]
    assert len(steps) == 1
    assert steps[0]["precursor_smiles"] == ["C", "CO"]
    assert {row["target_smiles"] for row in plan.frontier_priorities} == {"CO"}


def test_exhausted_materialization_leaf_is_deferred_to_short_tail_not_stock_closed() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=3,
        max_skeletons=3,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=1,
        max_node_expansions_per_branch=4,
        require_strategy_graph_edits=True,
        require_complete_route_json=False,
    )

    def executor(task):
        if task.required_artifact_type == "StrategyCardReport":
            return _strategy_record(task)
        if task.required_artifact_type == "ChemicalStrategyCritique":
            return _critic_record(task)
        product = re.search(r'"selected_open_leaf":"([^"]+)"', task.objective).group(1)
        operations = (
            [{"op": "break_bond", "map_a": 1, "map_b": 2}]
            if product == "CCO"
            else [{"op": "break_bond", "map_a": 1, "map_b": 99}]
        )
        return _proposal_record(
            {
                "schema_version": "retrosynthesis_candidate.v1",
                "candidate_id": task.task_id,
                "product_smiles": product,
                "precursor_smiles": [],
                "reaction_family": "short-tail deferral probe",
                "product_retron_type": "bond disconnection",
                "transformation_rationale": "retain unresolved route leaf",
                "conditions": [],
                "catalyst": "",
                "enzyme": "",
                "limitations": [],
                "required_validation": ["structure"],
                "no_solved_claim": True,
                "not_parent_route_proof": True,
                "reaction_operations": operations,
                "route_json": None,
            },
            target=product,
        )

    result = SequentialStrategyDirectorRunner(
        node_executor=executor,
        critic_executor=executor,
        stock_membership=lambda values: {value: value in {"C", "O"} for value in values},
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert result.usage["stock_closed_branch_count"] == 0
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert {row["target_smiles"] for row in plan.frontier_priorities} == {"CO"}
    assert plan.route_families[0]["blocked_materializations"] == ["CO"]


def test_duplicate_root_strategy_is_retried_before_route_expansion() -> None:
    context = _context()
    config = DirectorConfig(
        minimum_route_families=3,
        max_route_families=6,
        max_skeletons=6,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        strategy_branch_count=3,
        max_node_expansions_per_branch=2,
        critic_call_timeout_s=1.0,
    )
    branch_two_calls = 0
    observed = []

    def duplicate_once_executor(task):
        nonlocal branch_two_calls
        observed.append(task)
        record = _fake_executor(task)
        branch = int(re.search(r'"branch_id":(\d+)', task.objective).group(1))
        if branch != 2 or task.required_artifact_type != "StrategyCardReport":
            return record
        branch_two_calls += 1
        if branch_two_calls > 1:
            return record
        artifact = dict(record.output_artifact or {})
        payload = dict(artifact.get("payload") or {})
        payload["strategy_card"] = _strategy_card(1)
        artifact["payload"] = payload
        return replace(record, output_artifact=artifact)

    result = SequentialStrategyDirectorRunner(
        node_executor=duplicate_once_executor,
        critic_executor=_fake_critic_executor,
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert result.usage["model_invocations"] == 13
    assert result.usage["accepted_expansions"] == 6
    branch_ids = [
        int(re.search(r'"branch_id":(\d+)', task.objective).group(1)) for task in observed
    ]
    assert branch_ids[:4] == [1, 2, 3, 2]


def test_traversiadiene_fpp_shortcut_fails_atom_provenance_gate() -> None:
    traversiadiene = "C=C(C)[C@H]1CC[C@]2(C)C[C@@H]3[C@H](C)CC[C@@H]3/C(C)=C\\CC12"
    farnesyl_diphosphate = "C/C=C(\\C)CC/C=C(\\C)CC/C=C(\\C)COP(=O)(O)OP(=O)(O)O"

    assert _has_atom_provenance_deficit(traversiadiene, [farnesyl_diphosphate]) is True


def test_explicit_enzymatic_hydroxylation_allows_audited_external_oxygen() -> None:
    product = "CCO"
    record = WorkerRunRecord(
        run_id="p450-hydroxylation:run",
        task_id="p450-hydroxylation",
        case_id="p450-hydroxylation:case",
        status="accepted_draft",
        output_artifact={
            "artifact_type": "RetrosynthesisProposalReport",
            "payload": {
                "schema_version": "retrosynthesis_proposal_report.v1",
                "no_solved_claim": True,
                "candidates": [
                    {
                        "schema_version": "retrosynthesis_candidate.v1",
                        "candidate_id": "p450-c-h-hydroxylation",
                        "product_smiles": product,
                        "precursor_smiles": [],
                        "reaction_family": "P450 C-H hydroxylation",
                        "transformation_rationale": "remove the installed oxygen retrosynthetically",
                        "conditions": ["O2; NADPH"],
                        "catalyst": "P450/reductase",
                        "enzyme": "P450",
                        "execution_domain": "enzymatic",
                        "biocatalytic_step": {
                            "mode": "enzyme_reaction",
                            "enzyme_label": "P450",
                            "cosubstrates": ["molecular oxygen"],
                            "cofactor_requirements": ["NADPH"],
                        },
                        "limitations": [],
                        "no_solved_claim": True,
                        "not_parent_route_proof": True,
                        "reaction_operations": [{"op": "remove_group", "map_indices": [3]}],
                        "route_json": None,
                    }
                ],
            },
        },
        output_validation={"accepted": True, "reasons": []},
        usage={"input_tokens": 1, "output_tokens": 1},
    )

    expansions = _expansions_from_record(
        record,
        expected_product=product,
        mapped_product_smiles=sequential_module._mapped_smiles(product),
        require_reaction_operations=True,
        single_step_only=True,
    )

    assert expansions is not None
    assert expansions[0].precursor_smiles == ("CC",)
    assert expansions[0].execution_domain == "enzymatic"
    assert expansions[0].reactionjson_audit["external_atom_source_required"] is True
    assert expansions[0].reactionjson_audit["external_atom_source_grants_reaction_proof"] is False


def test_cyclopiamine_parallel_nitration_cards_are_not_orthogonal() -> None:
    first = {
        "scaffold_motif": "polycyclic alkaloid",
        "key_forward_transformation": "late-stage tertiary C-H nitration",
        "key_bond_changes": ["form C-N bond at tertiary carbon"],
        "functional_group_conflicts": ["oxidation-sensitive amine"],
        "protection_policy": "protect amine if required",
        "stereochemical_plan": "retain existing stereocentres",
        "convergence_plan": "late functionalization of complete skeleton",
        "strategic_step_count": 1,
        "skeleton_change_class": "functionalization",
        "expected_complexity_drop": "low",
        "orthogonality_basis": "nitration",
        "strategy_signature": "late-stage tertiary C-H nitration",
    }
    renamed_duplicate = {
        **first,
        "key_forward_transformation": "radical C-H nitro-functionalization",
        "orthogonality_basis": "radical nitro installation",
        "strategy_signature": "radical nitro functionalization",
    }

    assert _strategy_conflicts(renamed_duplicate, [first]) is True


def test_structural_edit_signature_overrides_renamed_strategy_labels() -> None:
    operations = [{"op": "break_bond", "map_a": 2, "map_b": 7}]
    first = normalize_strategy_card(
        {
            "scaffold_motif": "bridged ring",
            "key_forward_transformation": "annulation",
            "key_bond_changes": ["map 2-map 7"],
            "functional_group_conflicts": [],
            "protection_policy": "minimal",
            "stereochemical_plan": "substrate controlled",
            "convergence_plan": "two-fragment union",
            "strategic_step_count": 1,
            "skeleton_change_class": "ring formation",
            "expected_complexity_drop": "high",
            "orthogonality_basis": "annulation",
            "strategy_signature": "route alpha",
        },
        reaction_operations=operations,
    )
    renamed = normalize_strategy_card(
        {
            **first,
            "key_forward_transformation": "cascade closure",
            "orthogonality_basis": "different prose label",
            "strategy_signature": "route beta",
        },
        reaction_operations=operations,
    )

    assert _strategy_conflicts(renamed, [first]) is True


def test_same_target_bonds_and_mechanism_family_are_not_portfolio_orthogonal() -> None:
    first = normalize_strategy_card(
        {
            **_strategy_card(1),
            "scaffold_motif": "transannular closure of a medium-ring precursor",
            "key_forward_transformation": "radical transannular cascade cyclization",
            "skeleton_change_class": "fused-ring construction",
            "strategy_signature": "transannular radical closure",
        }
    )
    renamed_topology = normalize_strategy_card(
        {
            **_strategy_card(2),
            "scaffold_motif": "folded polyene precursor",
            "key_forward_transformation": "cationic polyene cascade cyclization",
            "skeleton_change_class": "polycyclic annulation",
            "strategy_signature": "cationic polyene annulation",
        }
    )

    assert first["topology_signature"] != renamed_topology["topology_signature"]
    assert _strategy_conflicts(renamed_topology, [first]) is True


def test_paper_strategy_portfolio_rejects_duplicate_cards() -> None:
    context = _context()
    spec = _spec(context)
    task = sequential_module._strategy_portfolio_task(
        spec,
        prompt="paper portfolio",
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        timeout_s=10.0,
    )
    record = _strategy_portfolio_record(task)
    artifact = dict(record.output_artifact or {})
    payload = dict(artifact.get("payload") or {})
    cards = list(payload["strategy_cards"])
    cards[1] = dict(cards[0])
    payload["strategy_cards"] = cards
    artifact["payload"] = payload

    assert (
        _strategy_cards_from_portfolio_record(
            replace(record, output_artifact=artifact),
            expected_target="CCO",
        )
        is None
    )


def test_paper_strategy_portfolio_seeds_three_branches_with_one_worker_call() -> None:
    context = _context()
    spec = _spec(context)
    branches = [
        {
            "branch_index": index,
            "lens": "neutral paper strategy",
            "strategy_seed": "",
            "strategy_card": {},
            "root_strategy_card": {},
            "strategy_milestone_cards": [],
            "strategy_call_count": 0,
            "call_count": 0,
            "rejections": [],
        }
        for index in range(3)
    ]
    observed = []

    def portfolio_executor(task):
        observed.append(task)
        return _strategy_portfolio_record(task)

    runner = SequentialStrategyDirectorRunner(node_executor=portfolio_executor)
    runner._prepare_worker_record_journal(spec)
    records = []
    runner._seed_paper_strategy_portfolio(
        spec,
        target="CCO",
        branches=branches,
        records=records,
        max_prompt_bytes=32_000,
        max_node_call_timeout_s=10.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=3,
            input_tokens=20_000,
            output_tokens=10_000,
            wall_time_s=30.0,
        ),
        started=time.monotonic(),
    )

    assert len(observed) == 1
    assert observed[0].required_artifact_type == "StrategyPortfolioReport"
    assert observed[0].budget.reasoning_effort == "medium"
    assert observed[0].budget.max_output_bytes == 6_000
    assert len(records) == 1
    assert all(branch["strategy_card"] for branch in branches)


def test_new_strategy_milestone_resets_only_current_checkpoint_state() -> None:
    context = _context()
    spec = _spec(context)
    observed = []

    def executor(task):
        observed.append(task)
        return _strategy_record(task)

    root = normalize_strategy_card(_strategy_card(1))
    branch = {
        "branch_index": 0,
        "strategy_card": root,
        "root_strategy_card": root,
        "strategy_milestone_cards": [root],
        "strategy_milestone_attempts": [],
        "strategy_call_count": 1,
        "strategy_milestone_generation_count": 0,
        "call_count": 1,
        "key_event_critic_completed": True,
        "pending_key_event_feedback": {"severity": "warning"},
        "strategy_mandate": "continue scaffold decomplexification",
    }
    runner = SequentialStrategyDirectorRunner(node_executor=executor)
    runner._prepare_worker_record_journal(spec)
    records = []
    card = runner._generate_upstream_strategy_milestone(
        spec,
        campaign_target="CCO",
        selected_product="CCO",
        selected_product_mapped="[CH3:1][CH2:2][OH:3]",
        branch=branch,
        route_steps=(),
        records=records,
        max_prompt_bytes=32_000,
        max_node_call_timeout_s=10.0,
        quota=sequential_module._NodeCallBudget(
            model_invocations=3,
            input_tokens=100_000,
            output_tokens=100_000,
            wall_time_s=30.0,
        ),
        started=time.monotonic(),
        budget_ledger=sequential_module._SharedModelCallLedger(
            sequential_module._NodeCallBudget(
                model_invocations=3,
                input_tokens=100_000,
                output_tokens=100_000,
                wall_time_s=30.0,
            ),
            (),
        ),
        paper_matched=True,
    )

    assert card is not None
    assert len(observed) == 2
    assert observed[0].task_type == "paper_matched_strategy_generator"
    assert "one concise strategy_query" in observed[0].objective
    assert observed[1].task_type == "paper_matched_strategy_critic"
    assert "earliest fact observable immediately after one reaction" in (observed[1].objective)
    assert "not required to be the next Builder reaction" in (observed[1].objective)
    assert "every revision or replacement must retain route-defining" in (observed[1].objective)
    assert branch["key_event_critic_completed"] is False
    assert branch["pending_key_event_feedback"] == {}
    assert len(branch["strategy_milestone_cards"]) == 2
    assert branch["strategy_milestone_cards"][-1]["host_lineage"] == {
        "root_mapped_smiles": "[CH3:1][CH2:2][OH:3]",
        "milestone_index": 2,
    }


def test_replacement_strategy_receives_only_compact_retired_horizon_feedback() -> None:
    retired = {
        "strategy_card": {
            "strategy_query": "Forge a trans-cyclooctene by RCM.",
            "critical_assumption": "Terminal-diene RCM can select the E cyclooctene.",
            "critic_checkpoint": "Formation of the E cyclooctene.",
        },
        "assessment": {
            "blocking_type": "stereochemistry",
            "reasons": ["Three catalyst variants left the same geometry uncontrolled."],
            "suggested_revision": "Choose a different route-defining graph transformation.",
        },
    }
    generator_prompt = sequential_module._milestone_strategy_prompt(
        campaign_target="CCO",
        selected_product="CCO",
        selected_product_mapped="[CH3:1][CH2:2][OH:3]",
        branch_index=0,
        milestone_index=2,
        strategy_mandate="decomplexify the scaffold",
        completed_strategy_cards=(),
        route_steps=(),
        retired_strategy_feedback=retired,
    )
    critic_prompt = sequential_module._upstream_strategy_critic_prompt(
        campaign_target="CCO",
        selected_product="CCO",
        selected_product_mapped="[CH3:1][CH2:2][OH:3]",
        branch_index=0,
        milestone_index=2,
        generated_card=_strategy_card(3),
        completed_strategy_cards=(),
        accepted_route_steps=(),
        retired_strategy_feedback=retired,
    )
    generator_context = json.loads(generator_prompt.rsplit("\n", 1)[1])
    critic_context = json.loads(critic_prompt.rsplit("\n", 1)[1])

    assert generator_context["retired_strategy"] == critic_context["retired_strategy"]
    assert generator_context["retired_strategy"] == {
        "strategy_query": "Forge a trans-cyclooctene by RCM.",
        "critical_assumption": "Terminal-diene RCM can select the E cyclooctene.",
        "critic_checkpoint": "Formation of the E cyclooctene.",
        "blocking_type": "stereochemistry",
        "reasons": ["Three catalyst variants left the same geometry uncontrolled."],
        "suggested_revision": "Choose a different route-defining graph transformation.",
    }
    assert "do not relabel the same checkpoint or merely swap reagents" in generator_prompt
    assert "repeats or paraphrases its route-defining checkpoint" in critic_prompt


def test_strategy_horizon_context_exposes_compact_selected_leaf_stereo() -> None:
    context = sequential_module._strategy_horizon_context(
        campaign_target="CC(O)CC",
        selected_product="CC(O)CC",
        selected_product_mapped="[CH3:1][C@H:2]([OH:3])[CH2:4][CH3:5]",
        branch_index=0,
        milestone_index=2,
        completed_strategy_cards=(),
        route_steps=(),
        phase="strategy_horizon_review",
    )

    stereo = context["selected_upstream_leaf_stereo"]
    assert any(row["map_idx"] == 2 for row in stereo["centers"])
    assert "selected_upstream_leaf_stereo" in sequential_module._upstream_strategy_critic_prompt(
        campaign_target="CC(O)CC",
        selected_product="CC(O)CC",
        selected_product_mapped="[CH3:1][C@H:2]([OH:3])[CH2:4][CH3:5]",
        branch_index=0,
        milestone_index=2,
        generated_card=_strategy_card(2),
        completed_strategy_cards=(),
        accepted_route_steps=(),
    )


def test_final_route_critic_uses_last_strategy_bound_to_selected_steps() -> None:
    root = normalize_strategy_card(_strategy_card(1))
    milestone = normalize_strategy_card(_strategy_card(2))
    branch = {
        "strategy_card": root,
        "root_strategy_card": root,
        "strategy_milestone_cards": [root, milestone],
        "steps": [
            {
                "step_id": "root",
                "strategy_card": root,
            },
            {
                "step_id": "upstream",
                "strategy_card": milestone,
            },
        ],
    }

    assert sequential_module._final_route_strategy_card(branch)[
        "strategy_digest"
    ] == milestone["strategy_digest"]


def test_new_sibling_strategy_reports_only_selected_path_critic_passes() -> None:
    context = _context()
    spec = _spec(context)
    observed = []

    def executor(task):
        observed.append(task)
        record = _strategy_record(task)
        record.output_artifact["payload"]["target_smiles"] = "O"
        return record

    root = normalize_strategy_card(_strategy_card(1))
    left = normalize_strategy_card(_strategy_card(2))
    left["host_lineage"] = {
        "root_mapped_smiles": "[CH3:1][CH3:2]",
        "milestone_index": 2,
    }
    route_steps = [
        {
            "step_id": "root-checkpoint",
            "product_smiles": "CCO",
            "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "precursor_smiles": ["CC", "O"],
            "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
            "strategy_card": root,
        },
        {
            "step_id": "left-preparation",
            "product_smiles": "CC",
            "mapped_product_smiles": "[CH3:1][CH3:2]",
            "precursor_smiles": ["C", "C"],
            "mapped_precursor_smiles": ["[CH4:1]", "[CH4:2]"],
            "strategy_card": left,
        },
    ]
    branch = {
        "branch_index": 2,
        "strategy_card": root,
        "root_strategy_card": root,
        "strategy_milestone_cards": [root, left],
        "strategy_milestone_attempts": [],
        "strategy_call_count": 2,
        "strategy_milestone_generation_count": 1,
        "call_count": 2,
        "strategy_mandate": "continue scaffold decomplexification",
        "key_event_critic_history": [
            {
                "status": "completed",
                "focus_step_id": "root-checkpoint",
                "strategy_digest": root["strategy_digest"],
                "strategy_milestone_index": 1,
            }
        ],
    }
    quota = sequential_module._NodeCallBudget(
        model_invocations=3,
        input_tokens=100_000,
        output_tokens=100_000,
        wall_time_s=30.0,
    )
    runner = SequentialStrategyDirectorRunner(node_executor=executor)
    runner._prepare_worker_record_journal(spec)

    generated = runner._generate_upstream_strategy_milestone(
        spec,
        campaign_target="CCO",
        selected_product="O",
        selected_product_mapped="[OH2:3]",
        branch=branch,
        route_steps=route_steps,
        records=[],
        max_prompt_bytes=64_000,
        max_node_call_timeout_s=10.0,
        quota=quota,
        started=time.monotonic(),
        budget_ledger=sequential_module._SharedModelCallLedger(quota, ()),
        paper_matched=True,
    )

    assert generated is not None
    assert len(observed) == 2
    generator_context = json.loads(observed[0].objective.rsplit("\n", 1)[1])
    critic_context = json.loads(observed[1].objective.rsplit("\n", 1)[1])
    expected_completed = [
        {
            "strategy_query": root["strategy_query"],
            "critical_assumption": root["critical_assumption"],
            "critic_checkpoint": root["critic_checkpoint"],
        }
    ]
    assert generator_context["completed_milestones"] == expected_completed
    assert critic_context["completed_milestones"] == [
        {
            **expected_completed[0],
            "critical_assumption": root["critical_assumption"],
        }
    ]
    assert left in branch["strategy_milestone_cards"]

    left_active, refresh = sequential_module._strategy_horizon_for_leaf(
        config=DirectorConfig(
            paper_matched_reach_profile=True,
            enable_key_event_critic=True,
            max_strategic_milestones_per_branch=4,
        ),
        branch=branch,
        root_strategy_card=root,
        steps=route_steps,
        selected_product_mapped="[CH4:1]",
    )
    assert left_active["strategy_digest"] == left["strategy_digest"]
    assert refresh is False


def test_dynamic_strategy_and_feedback_are_scoped_to_mapped_leaf_lineage() -> None:
    config = DirectorConfig(
        paper_matched_reach_profile=True,
        enable_key_event_critic=True,
        max_strategic_milestones_per_branch=3,
    )
    root = normalize_strategy_card(_strategy_card(1))
    left = normalize_strategy_card(_strategy_card(2))
    left["host_lineage"] = {
        "root_mapped_smiles": "[CH3:1][CH3:2]",
        "milestone_index": 2,
    }
    steps = [
        {
            "step_id": "root-event",
            "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
            "strategy_card": root,
        },
        {
            "step_id": "left-preparation",
            "mapped_product_smiles": "[CH3:1][CH3:2]",
            "mapped_precursor_smiles": ["[CH4:1]", "[CH4:2]"],
            "strategy_card": left,
        },
    ]
    branch = {
        "strategy_milestone_cards": [root, left],
        "key_event_critic_history": [
            {
                "status": "completed",
                "focus_step_id": "root-event",
                "strategy_digest": root["strategy_digest"],
                "strategy_milestone_index": 1,
            },
            {
                "status": "rejected",
                "focus_step_id": "left-checkpoint-attempt",
                "strategy_digest": left["strategy_digest"],
                "strategy_milestone_index": 2,
                "lineage_root_mapped_smiles": "[CH3:1][CH3:2]",
                "checkpoint_match": True,
                "assessment": {
                    "blocking": True,
                    "blocking_type": "chemoselectivity",
                    "reasons": ["left-lineage selectivity remains unresolved"],
                    "suggested_revision": "revise only the left lineage",
                },
            },
        ],
    }

    left_active, left_refresh = sequential_module._strategy_horizon_for_leaf(
        config=config,
        branch=branch,
        root_strategy_card=root,
        steps=steps,
        selected_product_mapped="[CH4:1]",
    )
    sibling_active, sibling_refresh = sequential_module._strategy_horizon_for_leaf(
        config=config,
        branch=branch,
        root_strategy_card=root,
        steps=steps,
        selected_product_mapped="[OH2:3]",
    )

    assert left_active["strategy_digest"] == left["strategy_digest"]
    assert left_refresh is False
    assert sibling_active["strategy_digest"] == root["strategy_digest"]
    assert sibling_refresh is True
    left_feedback = sequential_module._pending_key_event_feedback_for_leaf(
        branch,
        strategy_card=left_active,
        steps=steps,
        selected_product_mapped="[CH4:1]",
    )
    assert left_feedback["active_constraints"][0]["severity"] == "blocking"
    assert left_feedback["active_constraints"][0]["reasons"] == [
        "left-lineage selectivity remains unresolved"
    ]
    assert (
        sequential_module._pending_key_event_feedback_for_leaf(
            branch,
            strategy_card=sibling_active,
            steps=steps,
            selected_product_mapped="[OH2:3]",
        )
        == {}
    )


def test_unselected_checkpoint_pass_does_not_retire_active_feedback() -> None:
    card = normalize_strategy_card(_strategy_card(1))
    root = "[CH3:1][CH2:2][OH:3]"
    branch = {
        "strategy_milestone_cards": [card],
        "key_event_critic_history": [
            {
                "status": "rejected",
                "focus_step_id": "step:rejected",
                "strategy_digest": card["strategy_digest"],
                "strategy_milestone_index": 1,
                "lineage_root_mapped_smiles": root,
                "checkpoint_match": True,
                "assessment": {
                    "blocking": True,
                    "blocking_type": "mechanism",
                    "reasons": ["the reactive handles are incompatible"],
                    "suggested_revision": "revise the checkpoint handles",
                },
            },
            {
                "status": "completed",
                "focus_step_id": "step:passed-candidate",
                "strategy_digest": card["strategy_digest"],
                "strategy_milestone_index": 1,
                "lineage_root_mapped_smiles": root,
                "checkpoint_match": True,
                "assessment": {"verdict": "pass", "blocking": False},
            },
        ],
    }

    still_active = sequential_module._pending_key_event_feedback_for_leaf(
        branch,
        strategy_card=card,
        steps=[],
        selected_product_mapped=root,
    )
    assert still_active["active_constraints"][0]["blocking_type"] == "mechanism"

    selected_steps = [
        {
            "step_id": "step:passed-candidate",
            "mapped_product_smiles": root,
            "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
        }
    ]
    assert (
        sequential_module._pending_key_event_feedback_for_leaf(
            branch,
            strategy_card=card,
            steps=selected_steps,
            selected_product_mapped="[CH3:1][CH3:2]",
        )
        == {}
    )


def test_selected_direct_precursor_evidence_retires_uncertain_obligation_on_pass() -> None:
    card = normalize_strategy_card(_strategy_card(1))
    root = "[CH3:1][CH2:2][OH:3]"
    steps = [
        {
            "step_id": "focus",
            "mapped_product_smiles": root,
            "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
        },
        {
            "step_id": "evidence",
            "mapped_product_smiles": "[CH3:1][CH3:2]",
            "mapped_precursor_smiles": ["[CH4:1]", "[CH4:2]"],
        },
    ]
    uncertain = {
        "status": "uncertain",
        "focus_step_id": "focus",
        "strategy_digest": card["strategy_digest"],
        "strategy_milestone_index": 1,
        "lineage_root_mapped_smiles": root,
        "checkpoint_match": True,
        "assessment": {
            "blocking": False,
            "blocking_type": "stereochemistry",
            "reasons": ["selectivity remains unresolved"],
            "suggested_revision": "show direct precursor control",
        },
    }
    uncertain["obligation_id"] = sequential_module._key_event_obligation_id(uncertain)
    branch = {
        "strategy_milestone_cards": [card],
        "key_event_critic_history": [uncertain],
    }

    review = sequential_module._pending_uncertain_key_event_evidence_review(
        branch,
        strategy_card=card,
        steps=steps,
    )
    assert review["focus_step_id"] == "focus"
    assert review["evidence_step_id"] == "evidence"
    assert (
        sequential_module._pending_key_event_feedback_for_leaf(
            branch,
            strategy_card=card,
            steps=steps,
            selected_product_mapped="[CH4:1]",
        )
        == {}
    )
    critic_feedback = sequential_module._pending_key_event_feedback_for_leaf(
        branch,
        strategy_card=card,
        steps=steps,
        selected_product_mapped="[CH4:1]",
        include_uncertain=True,
    )
    assert critic_feedback["active_constraints"][0]["severity"] == "warning"
    assert critic_feedback["active_constraints"][0]["blocking_type"] == ("stereochemistry")

    branch["key_event_critic_history"].append(
        {
            "status": "completed",
            "focus_step_id": "focus",
            "strategy_digest": card["strategy_digest"],
            "strategy_milestone_index": 1,
            "lineage_root_mapped_smiles": root,
            "review_of_obligation_id": uncertain["obligation_id"],
            "review_evidence_step_id": "evidence",
            "required_selected_step_ids": ["focus", "evidence"],
            "checkpoint_match": True,
            "assessment": {"verdict": "pass", "blocking": False},
        }
    )
    assert (
        sequential_module._pending_key_event_feedback_for_leaf(
            branch,
            strategy_card=card,
            steps=steps,
            selected_product_mapped="[CH4:1]",
        )
        == {}
    )


def test_nullable_reaction_operation_schema_filler_is_pruned_before_replay() -> None:
    normalized = normalize_reaction_operations(
        [
            {
                "op": "break_bond",
                "map_a": 4,
                "map_b": 20,
                "order": 1,
                "atomic_num": None,
            },
            {
                "op": "set_explicit_h",
                "map_idx": 4,
                "map_a": 4,
                "count": 2,
            },
        ]
    )

    assert normalized == (
        {"op": "break_bond", "map_a": 4, "map_b": 20},
        {"op": "set_explicit_h", "map_idx": 4, "count": 2},
    )


def test_builder_candidate_persists_host_resolved_add_group_atom_maps() -> None:
    record = _proposal_record(
        {
            "candidate_id": "fresh-map",
            "product_smiles": "CBr",
            "precursor_smiles": [],
            "reaction_family": "oxygen substitution disconnection",
            "conditions": [],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": [
                {"op": "remove_group", "map_indices": [2]},
                {"op": "add_group", "map_idx": 1, "fragment_smiles": "[*]O"},
            ],
        },
        target="CBr",
    )

    compiled, rejected = sequential_module._reactionjson_candidates_from_record(
        record,
        expected_product="CBr",
        mapped_product_smiles="[CH3:1][Br:2]",
        require_reaction_operations=True,
        max_candidates=1,
        reserved_atom_maps=(31,),
    )

    assert rejected == []
    assert len(compiled) == 1
    assert compiled[0].expansion.reaction_operations[-1] == {
        "op": "add_group",
        "map_idx": 1,
        "fragment_smiles": "*[OH:32]",
    }
    resolved_step = sequential_module._step_row(
        compiled[0].expansion,
        step_id="repair:resolved-map",
    )
    assert (
        sequential_module._route_steps_host_replay_validation(
            [resolved_step],
            mapped_target_smiles="[CH3:1][Br:2]",
        )["complete"]
        is True
    )
    double_reserved = sequential_module._route_steps_host_replay_validation(
        [resolved_step],
        mapped_target_smiles="[CH3:1][Br:2]",
        reserved_atom_maps=(32,),
    )
    assert double_reserved["compiler_error"] == "reactionjson_fragment_map_collision"


def test_each_event_repairs_one_local_neighborhood_before_host_validation() -> None:
    base = _context()
    context = replace(
        base,
        topology={
            "edges": {
                "failed": {
                    "product_smiles": "CCO",
                    "precursor_smiles": ["CC", "O"],
                    "reaction_validation": {"accepted": False},
                }
            }
        },
        content_sha256="",
        byte_count=0,
    )
    config = DirectorConfig(
        minimum_route_families=1,
        max_route_families=6,
        max_skeletons=6,
        max_steps_per_skeleton=25,
        planning_mode="sequential_branches",
        max_route_local_repair_rounds=6,
    )
    observed = []

    def repair_executor(task):
        observed.append(task)
        aliased = replace(
            task,
            objective=re.sub(r'"branch_id":\d+', '"branch_id":1', task.objective),
        )
        return _fake_executor(aliased)

    result = SequentialStrategyDirectorRunner(
        node_executor=repair_executor,
        critic_executor=_fake_critic_executor,
    )(_spec(context), context, "event_replan", config)

    assert result.state is AgentState.SUCCEEDED
    assert result.usage["model_invocations"] == 2
    assert all("route-local repair" in task.objective for task in observed)
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert plan.mode == "event_replan"
    assert len(plan.multi_step_skeletons) == 1

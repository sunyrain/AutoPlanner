from __future__ import annotations

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
from cascade_planner.application.strategy_contract import normalize_strategy_card
from cascade_planner.application.strategy_contract import normalize_reaction_operations
from cascade_planner.orchestration.global_campaign_director import (
    DirectorConfig,
    GlobalCampaignPlan,
    validate_global_campaign_plan,
)
from cascade_planner.orchestration.sequential_strategy_director import (
    SequentialStrategyDirectorRunner,
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
        row.split("strategy_v2_slot=", 1)[1].split(":", 1)[0]
        for row in v2
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
    prompt = _paper_strategy_portfolio_prompt(target="CCO")

    assert "exactly three independent high-level strategies" in prompt
    assert "single call" in prompt
    assert "paper's four dimensions" in prompt
    assert "strategy_query is the one-sentence steering query" in prompt
    assert "one short sentence or a short list" in prompt
    assert "skeletal construction or reorganization" in prompt
    assert "reactive-handle logic" in prompt
    assert "suggested target-rooted anchors only" in prompt
    assert "Do not draw precursors" in prompt
    assert "mechanistic essay" in prompt
    assert "execution_domain=chemical" not in prompt
    assert "ketyl" not in prompt.lower()
    assert "grob" not in prompt.lower()


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
            "strategy_signature": "co-formation",
        },
        forbidden_strategy_cards=(_strategy_card(2),),
        host_failure_feedback={},
        max_reactionjson_candidates=1,
        paper_matched=True,
    )

    assert "Route Builder for one MCTS node" in prompt
    assert "strategy.strategy_query as the primary steering prior" in prompt
    assert "not a graph-completion checklist" in prompt
    assert "step as key, enabling, or supporting" in prompt
    assert "atom/H/charge/redox accounting" in prompt
    assert "feasibility_check" in prompt
    assert "mentally replay it" in prompt
    assert "Return no complete RouteJSON" in prompt
    assert '"strategy_anchor_progress"' not in prompt
    assert '"remaining_map_pairs"' not in prompt
    assert "simple_for_explorative_search" in prompt
    assert "A stop is never a stock or solved claim" in prompt
    assert "Expand exactly one retrosynthetic node" not in prompt
    assert '"schema_version":"paper_matched_route_builder_context.v1"' in prompt
    assert "forbidden_root_strategies" not in prompt
    assert "campaign_target_profile" not in prompt
    assert "biocatalytic_intent" not in prompt


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
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 2, "map_b": 3}
                ],
            },
            {
                "step_id": "route:2",
                "product_smiles": "CC",
                "precursor_smiles": ["C", "C"],
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 2}
                ],
            },
        ),
        open_leaves=("C", "O"),
        prior_rejections=(),
        repair=True,
        strategy_card=_strategy_card(1),
        forbidden_strategy_cards=(),
        host_failure_feedback={
            "blocking_step": {"step_id": "route:2", "product_smiles": "CC"}
        },
        complete_route_json=True,
        editor_route_mutations=True,
        paper_matched=True,
    )

    assert "Repair every concrete Critic blocker" in prompt
    assert "one coordinated route_patch" in prompt
    assert "compare at least two repair architectures" in prompt
    assert "output only the chosen patch" in prompt
    assert "impossible named mechanism or exact bond cut is not sacred" in prompt
    assert "update every affected dependency in the same patch" in prompt
    assert "set_conditions changes only conditions/catalyst" in prompt
    assert "Uncertainty alone is not grounds for abstention" in prompt
    assert "repair_status=unrepairable" in prompt
    assert "Do not output route_json" in prompt
    assert "Never truncate an unresolved suffix" in prompt


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
                "condition_predictions": [
                    {"reagents": ["base"], "catalyst": "Pd catalyst"}
                ],
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 2, "map_b": 3}
                ],
            },
        ),
        paper_matched=True,
    )

    assert "pass means executable as written" in prompt
    assert "merely underspecified conditions are not blockers" in prompt
    assert "exact host-derived mapped products" in prompt
    assert "at most two concrete reasons" in prompt
    assert "strategy_adherence is advisory" in prompt
    assert "without complementary handles" in prompt
    assert "never improve the score by truncating" in prompt
    assert "long mechanistic analysis" in prompt
    context = json.loads(prompt.split("PaperMatchedRouteCriticInput:\n", 1)[1])
    assert context["steps"][0]["mapped_product_smiles"] == (
        "[CH3:1][CH2:2][OH:3]"
    )
    assert context["steps"][0]["mapped_precursor_smiles"] == [
        "[CH3:1][CH3:2]",
        "[OH2:3]",
    ]
    assert context["steps"][0]["condition_predictions"][0]["catalyst"] == (
        "Pd catalyst"
    )


def test_editor_feedback_includes_every_concrete_critic_blocker() -> None:
    steps = [
        {
            "step_id": "route:1",
            "product_smiles": "CCO",
            "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "reaction_operations": [
                {"op": "break_bond", "map_a": 2, "map_b": 3}
            ],
        },
        {
            "step_id": "route:2",
            "product_smiles": "CC",
            "mapped_product_smiles": "[CH3:1][CH3:2]",
            "reaction_operations": [
                {"op": "break_bond", "map_a": 1, "map_b": 2}
            ],
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
    assert [
        row["route_step"]["step_id"] for row in feedback["blocking_steps"]
    ] == ["route:1", "route:2"]
    assert feedback["repair_actions"] == critique["repair_actions"]
    assert feedback["failure_reasons"] == [
        "nucleophile handle is absent",
        "protection must precede coupling",
    ]


def test_worker_output_contract_uses_one_compact_editor_patch_channel() -> None:
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

    assert "coordinated route_patch repair" in instruction
    assert "host applies and replays the patch" in instruction
    assert "complete revised route_json" not in instruction

    schema = codex_worker_module._retrosynthesis_proposal_report_payload_json_schema(
        task
    )
    candidate = schema["properties"]["candidates"]["items"]
    assert "route_json" not in candidate["properties"]
    assert candidate["properties"]["route_patch"]["type"] == "array"
    assert "repair_status" in candidate["properties"]
    assert "repair_summary" in candidate["properties"]
    assert "unrepairable_reason" in candidate["properties"]
    assert "minItems" not in candidate["properties"]["route_patch"]
    assert "product_smiles" not in candidate["properties"]
    patch_item = candidate["properties"]["route_patch"]["items"]
    route_step = patch_item["properties"]["step"]
    assert "mapped_product_smiles" in route_step["properties"]
    assert "conditions" in route_step["properties"]
    assert "set_conditions" in patch_item["properties"]["op"]["enum"]


def test_paper_critic_editor_output_windows_fit_complete_route_documents() -> None:
    context = _context()
    spec = _spec(context)
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

    assert editor_task.budget.max_output_bytes == 40_000
    assert critic_task.budget.max_output_bytes == 32_000


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
            "reaction_operations": [
                {"op": "break_bond", "map_a": index, "map_b": index + 1}
            ],
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
    assert context["schema_version"] == "paper_matched_route_editor_context.v2"
    assert len(context["frozen_route"]) == 25
    assert [row["step_id"] for row in context["frozen_route"]] == [
        f"route:{index}" for index in range(1, 26)
    ]
    assert all(row["product_smiles"] for row in context["frozen_route"])
    assert all(row["precursor_smiles"] for row in context["frozen_route"])
    assert all(row["reaction_operations"] for row in context["frozen_route"])
    assert all("mapped_product_smiles" in row for row in context["frozen_route"])
    assert all("conditions" in row for row in context["frozen_route"])
    assert all("parent_step_ids" in row for row in context["frozen_route"])
    assert "transformation_rationale" not in context["frozen_route"][0]
    assert context["frozen_route"][0]["strategy_anchor"] is True


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
            "condition_predictions": [
                {"conditions": [verbose], "rationale": verbose}
            ],
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
    assert [row["step_id"] for row in route["steps"]] == [
        f"step:{index}" for index in range(1, 26)
    ]
    assert all(row["product_smiles"] for row in route["steps"])
    assert all(row["precursor_smiles"] for row in route["steps"])
    assert all(row["reaction_operations"] for row in route["steps"])
    assert route["steps"][6]["execution_domain"] == "whole_cell"


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

    chemical_only = _route_execution_profile(
        [{"execution_domain": "chemical"}]
    )
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
        stock_membership=lambda values: {
            value: value in {"C", "O"} for value in values
        }
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
    paper_fields = {
        "strategy_query",
        "scaffold_motif",
        "key_forward_transformation",
        "key_bond_changes",
        "functional_group_conflicts",
        "protection_policy",
        "stereochemical_plan",
        "strategic_step_count",
        "strategy_signature",
    }
    cards = [
        {
            key: value
            for key, value in _strategy_card(branch).items()
            if key in paper_fields
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
                "selection_rationale": "pairwise distinct strategic constructions",
                "limitations": ["hypotheses only"],
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
        "reaction_operations": [
            {"op": "break_bond", "map_a": 2, "map_b": 3}
        ],
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
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 2, "map_b": 3}
                ],
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
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 2}
                ],
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
            "reaction_operations": [
                {"op": "break_bond", "map_a": 2, "map_b": 3}
            ],
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
            "reaction_operations": [
                {"op": "break_bond", "map_a": 1, "map_b": 2}
            ],
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
            "reaction_operations": [
                {"op": "break_bond", "map_a": 3, "map_b": 4}
            ],
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
    target = (
        "CCOC(=O)C1(O)c2cc3c(cc2C(=O)C1(C)O)C1(CC(=O)C[C@@H](C)O1)"
        "O[C@H](C)C3"
    )
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
            "reaction_operations": [
                {"op": "break_bond", "map_a": 21, "map_b": 23}
            ],
        },
    ]
    replacement = dict(current[1])
    replacement["product_smiles"] = (
        "[CH3:20][C:21](=[O:22])[CH2:23][C@@H:24]"
        "([CH3:25])[OH:26]"
    )
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
    assert diagnostic["detail"] == "product_not_in_open_precursors"
    assert diagnostic["compiler_mode"] == "target_rooted_route_dag"
    assert diagnostic["editor_mutation_mode"] == "full_route_json"


def test_routejson_admission_rejects_cross_step_atom_map_namespace_break() -> None:
    steps = [
        {
            "product_smiles": "CCO",
            "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "reaction_operations": [
                {"op": "break_bond", "map_a": 2, "map_b": 3}
            ],
        },
        {
            "product_smiles": "CC",
            "mapped_product_smiles": "[CH3:18][CH3:19]",
            "reaction_operations": [
                {"op": "break_bond", "map_a": 18, "map_b": 19}
            ],
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
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 2}
                ],
            },
        ],
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
    )

    assert invalid["complete"] is False
    assert invalid["step_index"] == 1
    assert "reactionjson_map_not_found" in invalid["compiler_error"]
    assert valid["complete"] is True
    assert valid["compiled_step_count"] == 2


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
        task
        for task in observed
        if task.required_artifact_type == "ChemicalStrategyCritique"
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
            "reaction_operations": [
                {"op": "break_bond", "map_a": 2, "map_b": 3}
            ],
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
    assert search["reported_policy_calls"] == 1
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
                "expandable_mapped_smiles": [
                    first["mapped_precursor_smiles"][upstream_index]
                ],
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
    assert len(
        [task for task in observed if task.required_artifact_type == "StrategyCardReport"]
    ) == 2
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
            "policy_calls": 25,
            "mcts_iterations": 25,
            "diagnostics": {
                "engine": "AiZynthFinder.MctsSearchTree",
                "calls_exhausted": True,
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
    assert budget["hard_failures"] == []


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
    assert dict(plan.route_families[0].get("chemical_critic") or {}).get(
        "status"
    ) == "unavailable"


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
    assert sum(
        task.task_type == "route_chemistry_edit" for task in observed
    ) == 1
    plan = GlobalCampaignPlan.from_dict(result.output)
    family = plan.route_families[0]
    assert family["chemical_critic"]["status"] == "viable"
    assert len(family["critic_editor_history"]) == 2
    assert len(family["editor_repairs"]) == 1
    assert family["editor_attempt_count"] == 1
    assert family["editor_applied_count"] == 1
    assert family["editor_call_count"] == 1
    assert family["route_call_count"] == 1


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
    editor_tasks = [
        task for task in observed if task.task_type == "route_chemistry_edit"
    ]
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
        stock_membership=lambda values: {
            value: value in {"C", "CC", "O"} for value in values
        },
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert editor_attempts == 2
    assert critic_attempts == 2
    assert sum(
        task.task_type in {"route_chemistry_edit", "paper_matched_route_editor"}
        for task in observed
    ) == 2
    family = GlobalCampaignPlan.from_dict(result.output).route_families[0]
    assert family["editor_attempt_count"] == 2
    assert family["editor_applied_count"] == 1


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
    editor_tasks = [
        task for task in observed if task.task_type == "route_chemistry_edit"
    ]
    assert len(editor_tasks) == 1
    assert "reaction_operations" in editor_tasks[0].objective
    assert '"map_a":2' in editor_tasks[0].objective
    plan = GlobalCampaignPlan.from_dict(result.output)
    family = plan.route_families[0]
    assert family["materialization_failures"] == {}
    assert family["materialization_editor_history"][0]["outcome"] == (
        "host_recompiled"
    )
    assert family["editor_repairs"][0]["phase"] == (
        "route_builder_materialization"
    )
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
    assert any(
        task.required_artifact_type == "RetrosynthesisProposalReport"
        for task in observed
    )


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
        task
        for task in observed
        if task.required_artifact_type == "RetrosynthesisProposalReport"
    ]
    assert len(strategy_tasks) == 3
    assert route_tasks, "Route Builder must retain a call after strategy seeding"


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
    assert all(
        task.task_type == "route_step_materialization"
        for task in observed_tasks[3:]
    )
    for skeleton in plan.multi_step_skeletons:
        strategy_digests = {
            step["strategy_digest"] for step in skeleton["steps"]
        }
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
        stock_membership=lambda values: {
            value: value in {"C", "O"} for value in values
        },
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
                step_id = re.search(
                    r'"step_id":"([^"]+)"', task.objective
                ).group(1)
                return _blocking_critic_record(task, step_id=step_id)
            return _critic_record(task)
        product = re.search(
            r'"selected_open_leaf":"([^"]+)"', task.objective
        ).group(1)
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
        stock_membership=lambda values: {
            value: value in {"C", "O"} for value in values
        },
    )
    result = runner(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    plan = GlobalCampaignPlan.from_dict(result.output)
    family = plan.route_families[0]
    assert family["route_call_count"] == 1
    assert family["editor_call_count"] == 1
    assert family["reactionjson_or_search"]["root_solved"] is False
    assert family["reactionjson_or_search_resets"][0]["previous_summary"][
        "root_solved"
    ] is True
    assert family["reactionjson_or_search_resets"][0]["rebuilt_summary"][
        "root_solved"
    ] is False
    assert family["reactionjson_or_search_resets"][-1]["rebuilt_summary"][
        "root_solved"
    ] is False
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
    )(
        _spec(context), context, "initial_architecture", config
    )
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

    assert result.usage["model_invocations"] == 3
    expansion_tasks = [
        task
        for task in tasks
        if task.required_artifact_type == "RetrosynthesisProposalReport"
    ]
    critic_tasks = [
        task for task in tasks if task.required_artifact_type == "ChemicalStrategyCritique"
    ]
    branch_ids = [
        int(re.search(r'"branch_id":(\d+)', task.objective).group(1))
        for task in expansion_tasks
    ]
    assert branch_ids == []
    assert len(critic_tasks) == 0


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
        stock_membership=lambda values: {
            value: value in {"CC", "O"} for value in values
        },
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
            "objective": (
                'CompactBranchContext:{"branch_id":1,'
                '"selected_open_leaf":"CCO"}'
            ),
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
    candidate["reaction_operations"] = [
        {"op": "break_bond", "map_a": 2, "map_b": 3}
    ]
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
    assert "strategy.strategy_query as the primary steering prior" in prompt
    assert "graph-completion checklist" in prompt
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
        "strategy_anchor": True,
        "reaction_operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
    }
    second = {
        "strategy_card": card,
        "strategy_anchor": True,
        "reaction_operations": [{"op": "break_bond", "map_a": 3, "map_b": 4}],
    }

    assert sequential_module._strategy_anchor_fulfilled_for_card([first], card) is False
    progress = sequential_module._strategy_anchor_progress([first], card)
    assert progress["realized_map_pairs"] == ["map_pair:1:2"]
    assert progress["remaining_map_pairs"] == ["map_pair:3:4"]
    assert progress["authority"] == "report_only_diagnostic"
    assert progress["grants_strategy_adherence"] is False
    assert progress["grants_strategy_completion"] is False
    assert (
        sequential_module._strategy_anchor_fulfilled_for_card([first, second], card)
        is True
    )


def test_paper_route_builder_stop_signal_is_not_a_solved_claim() -> None:
    record = WorkerRunRecord(
        run_id="paper-stop:run",
        task_id="paper-stop",
        case_id="paper-stop:case",
        status="accepted_draft",
        output_artifact={
            "artifact_type": "RetrosynthesisProposalReport",
            "payload": {
                "stop_signal": True,
                "stop_reason": "simple_for_explorative_search",
                "candidates": [],
            },
        },
        output_validation={"accepted": True, "reasons": []},
    )

    stop = sequential_module._route_builder_stop_signal(record)

    assert stop == {
        "stop_signal": True,
        "stop_reason": "simple_for_explorative_search",
        "grants_stock_closure": False,
        "grants_solved": False,
    }


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
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
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
        for line in (tmp_path / "model-io.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["event"] for row in observed_input_lines] == ["model_input"]
    assert [row["event"] for row in rows] == ["model_input", "model_output"]
    assert rows[0]["prompt"] == "complete model prompt"
    assert rows[1]["stdout"] == "complete model output"
    assert rows[1]["output_artifact"]["artifact_type"] == (
        "RetrosynthesisProposalReport"
    )


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
            output_artifact={"artifact_type": "StrategyCardReport"},
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
        worker_record_seed_recovery_mode="critic_prompt_compaction_v1",
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
    assert recovered._logical_seed_replay_count == 1
    assert (fresh_dir / "sequential-director-worker-records.jsonl").is_file()


def test_reactionjson_failure_returns_causal_replay_diagnostic() -> None:
    task = type(
        "RouteTask",
        (),
        {
            "required_artifact_type": "RetrosynthesisProposalReport",
            "objective": (
                'CompactBranchContext:{"branch_id":1,'
                '"selected_open_leaf":"CCO"}'
            ),
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
    candidate["reaction_operations"] = [
        {"op": "break_bond", "map_a": 1, "map_b": 3}
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
    assert "bond_missing" in diagnostic["replay_error"]
    assert diagnostic["attempted_operations"] == [
        {"op": "break_bond", "map_a": 1, "map_b": 3}
    ]


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
        candidate["reaction_operations"] = [
            {"op": "break_bond", "map_a": 1, "map_b": 3}
        ]
        payload["candidates"] = [candidate]
        artifact["payload"] = payload
        return replace(record, output_artifact=artifact)

    result = SequentialStrategyDirectorRunner(
        node_executor=invalid_materialization_executor
    )(_spec(context), context, "initial_architecture", config)

    assert result.state is AgentState.FAILED
    assert result.usage["model_invocations"] == 4
    assert result.usage["materialization_retry_limit"] == 3
    retained = result.usage["retained_strategy_hypotheses"]
    assert len(retained) == 1
    assert retained[0]["strategy_signature"] == "branch-1-signature"
    assert retained[0]["key_forward_transformation"] == (
        "branch-1-key-construction"
    )
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
        stock_membership=lambda values: {
            value: value in {"C", "O"} for value in values
        },
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
        stock_membership=lambda values: {
            value: value in {"C", "O"} for value in values
        },
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
    )(
        _spec(context), context, "initial_architecture", config
    )

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
                        "reaction_operations": [
                            {"op": "remove_group", "map_indices": [3]}
                        ],
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
    assert (
        expansions[0].reactionjson_audit["external_atom_source_grants_reaction_proof"]
        is False
    )


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

    assert _strategy_cards_from_portfolio_record(
        replace(record, output_artifact=artifact),
        expected_target="CCO",
    ) is None


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
    assert len(records) == 1
    assert all(branch["strategy_card"] for branch in branches)


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
    )(
        _spec(context), context, "event_replan", config
    )

    assert result.state is AgentState.SUCCEEDED
    assert result.usage["model_invocations"] == 2
    assert all("route-local repair" in task.objective for task in observed)
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert plan.mode == "event_replan"
    assert len(plan.multi_step_skeletons) == 1

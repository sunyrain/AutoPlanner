"""Saved September 15 IO through the real replay and repair consumers.

These are deterministic regressions, not new model or chemical-validation runs.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from cascade_planner.agent.codex_worker import WorkerRunRecord, _materialize_paper_matched_artifact
from cascade_planner.application.reactionjson_replay import (
    ReactionJsonReplayError, replay_reactionjson,
)
from cascade_planner.application.routejson_compiler import RouteJSONCompiler
from cascade_planner.orchestration import sequential_strategy_director as director
from cascade_planner.orchestration.chemical_reasoning import CHEMICAL_REVIEW_SCOPE
from cascade_planner.orchestration.global_campaign_director import DirectorConfig
from cascade_planner.runtime import AgentSpec, Budget


SAVED = json.loads((Path(__file__).parent / "fixtures/statin_io_regressions.json")
                   .read_text(encoding="utf-8"))


def test_saved_clear_then_break_wittig_replays_without_removing_model_operation():
    saved = SAVED["stereo_clear_wittig"]
    result = replay_reactionjson(
        mapped_product_smiles=saved["mapped_product_smiles"], operations=saved["operations"],
        expected_precursor_smiles=saved["expected_precursor_smiles"],
    )
    assert result["expected_precursors_match"]
    assert result["operation_count"] == 4


def test_cleared_alkene_stays_unspecified_without_erasing_neighbour_stereo():
    result = replay_reactionjson(
        mapped_product_smiles="[CH3:1]/[CH:2]=[CH:3]/[CH:4]=[CH:5]/[CH3:6]",
        operations=[{"op": "set_bond_stereo", "map_a": 2, "map_b": 3, "stereo": "NONE"}],
        expected_precursor_smiles=["CC=C/C=C/C"],
    )
    assert result["expected_precursors_match"]
    replay_reactionjson(mapped_product_smiles="[CH3:1]/[CH:2]=[CH:3]/[CH3:4]", operations=[
        {"op": "set_bond_stereo", "map_a": 2, "map_b": 3, "stereo": "Z"},
        {"op": "set_bond_stereo", "map_a": 2, "map_b": 3, "stereo": "NONE"},
    ], expected_precursor_smiles=["CC=CC"])
    with pytest.raises(ReactionJsonReplayError, match="stereo_clear_not_representable"):
        replay_reactionjson(
            mapped_product_smiles="[CH3:1]/[CH:2]=[CH:3]/[CH:4]=[CH:5]/[CH:6]=[CH:7]/[CH3:8]",
            operations=[{"op": "set_bond_stereo", "map_a": 4, "map_b": 5, "stereo": "NONE"}],
        )
    with pytest.raises(ReactionJsonReplayError, match="reactionjson_bond_missing"):
        replay_reactionjson(mapped_product_smiles="[CH3:1][CH3:2]", operations=[
            {"op": "break_bond", "map_a": 1, "map_b": 2},
            {"op": "set_bond_stereo", "map_a": 1, "map_b": 2, "stereo": "NONE"},
        ])


def test_saved_reviews_differ_only_by_consistent_atom_numbering():
    from cascade_planner.application.route_review_context import equivalent_review_atom_numbering

    saved = SAVED["renumbered_review"]
    old, new = saved["old_prompt"], saved["new_prompt"]
    mapping = equivalent_review_atom_numbering(old, new)
    assert mapping is not None and mapping[45] == 72 and mapping[46] == 73
    marker = "PaperMatchedRouteCriticInput:\n"
    prefix, raw = new.split(marker)
    context = json.loads(raw)
    for field, value in [("conditions", ["different condition"]),
                         ("reaction_family", "different intention")]:
        changed = copy.deepcopy(context)
        changed["steps"][0][field] = value
        assert equivalent_review_atom_numbering(old, prefix + marker + json.dumps(changed)) is None
    for before, after in [("[C@H:8]", "[C@@H:8]"), ("[O:72]", "[18O:72]"),
                          ('"map_b":73', '"map_b":74')]:
        assert before in new
        assert equivalent_review_atom_numbering(old, new.replace(before, after, 1)) is None
    # Unchanged prose referring to a renumbered atom also invalidates reuse.
    for prompt in (old, new):
        assert "Atom 45" not in prompt
    assert equivalent_review_atom_numbering(
        old.replace('"phase":', '"note":"Atom 45 is sensitive","phase":'),
        new.replace('"phase":', '"note":"Atom 45 is sensitive","phase":'),
    ) is None


def test_complete_reaction_intent_survives_path_projection():
    intent = "Specific precursor and transformation; " * 12
    rows = director._compact_path_reaction_rows([{
        "step_id": "step", "reaction_family": intent, "reaction_operations": [],
    }])
    assert rows[0]["reaction_family"] == intent


@pytest.mark.parametrize("role", ["paper_matched_strategy_generator", "paper_matched_strategy_critic",
                                  "paper_matched_route_step", "paper_matched_route_critic",
                                  "paper_matched_path_repair_editor"])
def test_process_requirements_reach_executor_and_actual_io_without_new_output_contract(tmp_path, role):
    from cascade_planner.application.unified_campaign_spec import TargetConstraints

    brief = "Avoid -78 C; preserve E alkene and syn diol; do not transfer the burden upstream."
    runner = director.SequentialStrategyDirectorRunner(target_constraints=TargetConstraints(
        safety_limits={"process_brief": brief}, forbidden_reagents=("POCl3",),
    ).to_dict())
    workspace = tmp_path / ".autoplanner/director-workspace"
    spec = AgentSpec.from_context(
        run_id="requirements", agent_id="requirements", role="global_campaign_director",
        objective="fixture", context={}, idempotency_key="requirements",
        metadata={"durable_worker_journal": True, "allowed_workdir": str(workspace)},
    )
    runner._prepare_worker_record_journal(spec)
    task = director._strategy_portfolio_task(spec, prompt="Original role prompt", model="test",
                                             reasoning_effort="medium", timeout_s=60)
    task = replace(task, task_type=role)
    seen = []

    def executor(actual):
        seen.append(actual)
        return WorkerRunRecord(run_id="fixture", task_id=actual.task_id, case_id=actual.case_id,
                               status="timeout", usage={"input_tokens": 1, "output_tokens": 0})

    runner._run_journaled_worker(executor, task)
    assert seen[0].objective.count(brief) == 1
    assert "POCl3" in seen[0].objective
    assert seen[0].host_context == task.host_context
    assert seen[0].required_artifact_type == task.required_artifact_type
    lines = [json.loads(line) for line in (workspace / "model-io.jsonl").read_text(
        encoding="utf-8").split("\n") if line.strip()]
    assert lines[0]["prompt"] == seen[0].objective
    assert director.SequentialStrategyDirectorRunner(
        target_constraints=TargetConstraints().to_dict())._with_target_constraints(task) == task


@pytest.mark.parametrize("candidate", SAVED["aromatic_candidates"],
                         ids=lambda row: row["target_name"])
def test_saved_aromatic_core_disconnections_reach_expected_precursors(candidate):
    audit = replay_reactionjson(
        mapped_product_smiles=candidate["mapped_product_smiles"],
        operations=candidate["operations"],
        expected_precursor_smiles=candidate["expected_precursor_smiles"],
    )
    assert audit["expected_precursors_match"] is True
    assert audit["semantics"]["replay_grants_no_reaction_proof"] is True
    reaction = RouteJSONCompiler().compile_step(
        mapped_product_smiles=candidate["mapped_product_smiles"],
        operations=candidate["operations"],
    )
    assert sorted(reaction.reaction_input_smiles) == sorted(candidate["expected_precursor_smiles"])


def test_aromatic_cleanup_does_not_invent_missing_bond_edits_or_fix_invalid_valence():
    candidate = SAVED["aromatic_candidates"][1]  # Paal-Knorr retrosynthesis
    for operations in (
        [row for row in candidate["operations"] if row["op"] != "change_bond_order"],
        [*candidate["operations"], {"op": "add_group", "map_idx": 4, "fragment_smiles": "[*]C"}],
    ):
        with pytest.raises(ReactionJsonReplayError):
            replay_reactionjson(mapped_product_smiles=candidate["mapped_product_smiles"],
                                operations=operations)


def test_editor_receives_saved_full_conditions_and_all_condition_items():
    steps = copy.deepcopy(SAVED["fluvastatin"]["original_steps"])
    assert len(steps[2]["conditions"][0]) == 637
    assert len(steps[1]["conditions"][0]) == 337
    steps[0]["conditions"].extend(["Keep condition " + str(i) for i in range(6)])
    steps[0]["catalyst"] = "Specified catalyst: " + "complete loading and ligation; " * 10
    prompt = director._path_repair_editor_prompt(
        target=SAVED["fluvastatin"]["target"], strategy_card={}, repair_mode="cut_frontier",
        steps=steps, critic_feedback={},
    )
    context = json.loads(prompt.split("PathRepairEditorContext:\n", 1)[1])
    for original, projected in zip(steps, context["route_json"], strict=True):
        assert projected["conditions"] == original["conditions"]
        assert projected["reaction_family"] == original["reaction_family"]
    assert context["route_json"][0]["catalyst"] == steps[0]["catalyst"]


@pytest.mark.parametrize("paper,kind", [(True, "final_route"), (True, "key_event"),
                                      (True, "key_event_followup"), (False, "final_route")])
def test_chemical_review_scope_is_consistent_across_critic_paths(paper, kind):
    prompt = director._critic_prompt(
        target="CCO", branch_index=0, strategy_card={}, steps=[],
        paper_matched=paper, audit_kind=kind,
    )
    assert CHEMICAL_REVIEW_SCOPE in prompt


@pytest.mark.parametrize("closed", [False, True])
def test_critic_prose_cannot_overwrite_host_stock_closure(closed):
    branch = {
        "steps": [{"step_id": "existing"}],
        "open_leaves": [] if closed else ["CCO"],
        "aizynthfinder_strategy_search": {
            "canonical_route_projection_complete": True,
            "canonical_leaf_closure_complete": closed,
        },
        "chemical_critic": {
            "status": "reject" if closed else "viable",
            "route_overall_evaluation": "Not stock-closed." if closed else "Stock-closed.",
        },
    }
    assert director._branch_stock_closed(branch) is closed


def saved_fluv_repair(*, retire_supply=True):
    saved = SAVED["fluvastatin"]
    original = copy.deepcopy(saved["original_steps"])
    directive = copy.deepcopy(saved["original_directive"])
    if retire_supply:
        # The existing scope field explicitly includes the obsolete peroxide
        # supply subtree. The alcohol preparation and its upstream route stay.
        directive["change_step_ids"].extend(
            f"codex:branch:1:node:{node}:candidate:1" for node in (8, 10, 13, 15, 17, 22)
        )
    target = original[0]["mapped_product_smiles"]
    span, diagnostic = director._prepare_path_repair_span(
        current_steps=original, mapped_target_smiles=target, directive=directive,
        blocking_step_ids=[original[3]["step_id"]],
    )
    assert diagnostic == {} and span is not None
    compiler = RouteJSONCompiler()
    # Correct atom-retaining alcohol oxidation, not the model's remove O9 /
    # add O37 workaround from the failed run. No new model call is involved.
    reaction = compiler.compile_step(
        mapped_product_smiles=span.repair_frontier_mapped_product_smiles,
        operations=[{"op": "change_bond_order", "map_a": 8, "map_b": 9, "delta": -1}],
        reserved_atom_maps=span.reserved_atom_maps,
    )
    replacement = compiler.assemble_route([reaction], metadata=[{
        "step_id": "repair:oxidation", "reaction_family": "Selective allylic alcohol oxidation",
        "conditions": ["Activated MnO2; isolate the E-enal before dianion addition."],
    }])
    return original, directive, span, replacement


@pytest.mark.parametrize("retire_supply", [False, True])
def test_saved_fluv_repair_requires_explicit_scope_and_preserves_suffix(retire_supply):
    original, _, span, replacement = saved_fluv_repair(retire_supply=retire_supply)
    target = original[0]["mapped_product_smiles"]
    rebuilt = [*span.durable_steps, *replacement]
    compiler = RouteJSONCompiler()
    state = compiler.compile_route_graph_state(mapped_target_smiles=target, steps=rebuilt)
    reached = director._path_repair_frontier_reaches_boundaries(
        product_smiles=[row.product_smiles for row in state.open_precursors],
        mapped_product_smiles=[row.mapped_product_smiles for row in state.open_precursors],
        reconnect_boundaries=span.completion_boundaries,
    )
    assert reached is retire_supply
    if not retire_supply:
        return  # Still-required supply branches cannot disappear implicitly.
    stitched, diagnostic = director._stitch_path_repair_suffix(
        mapped_target_smiles=target, rebuilt_steps=rebuilt,
        preserved_suffix_steps=span.preserved_suffix_steps,
        reconnect_boundaries=span.suffix_reconnect_boundaries,
    )
    assert stitched is not None and diagnostic["remapped_boundary_atom_count"] == 1
    assert len(stitched) == 10  # 3 target-side + 1 oxidation + 6 retained preparations
    final_state = compiler.compile_route_graph_state(mapped_target_smiles=target, steps=stitched)
    assert director._path_repair_frontier_reaches_boundaries(
        product_smiles=[row.product_smiles for row in final_state.open_precursors],
        mapped_product_smiles=[row.mapped_product_smiles for row in final_state.open_precursors],
        reconnect_boundaries=span.final_open_boundaries,
    )
    for old, new in zip(span.preserved_suffix_steps, stitched[-6:], strict=True):
        assert new["step_id"] == old["step_id"]
        assert new["product_smiles"] == old["product_smiles"]
        assert new["conditions"] == old["conditions"]
    assert "[OH:9]" in stitched[4]["mapped_product_smiles"]
    assert "[OH:37]" not in stitched[4]["mapped_product_smiles"]


def test_replacement_terminal_requires_host_stock_and_retained_occurrences_remain_mandatory():
    boundary = {"boundary_kind": "removed_terminal_open_precursor",
                "product_smiles": "CCCl", "mapped_product_smiles": "[CH3:1][CH2:2][Cl:3]"}
    arguments = dict(product_smiles=["CCO"], mapped_product_smiles=["[CH3:1][CH2:2][OH:4]"],
                     reconnect_boundaries=[boundary])
    assert not director._path_repair_frontier_reaches_boundaries(**arguments)
    assert director._path_repair_frontier_reaches_boundaries(**arguments, stock_membership={"CCO": True})
    for kind in ("preserved_suffix_entry", "preserved_durable_open_precursor"):
        arguments["reconnect_boundaries"] = [{**boundary, "boundary_kind": kind}]
        assert not director._path_repair_frontier_reaches_boundaries(
            **arguments, stock_membership={"CCO": True})


def test_optional_duplicate_does_not_consume_a_required_boundary_occurrence():
    molecule = {"product_smiles": "C", "mapped_product_smiles": "[CH4:1]"}
    boundaries = [{**molecule, "boundary_kind": "removed_terminal_open_precursor"},
                  {**molecule, "boundary_kind": "preserved_suffix_entry"}]
    assert director._path_repair_frontier_reaches_boundaries(
        product_smiles=["C"], mapped_product_smiles=["[CH4:1]"], reconnect_boundaries=boundaries)
    assert not director._path_repair_frontier_reaches_boundaries(
        product_smiles=[], mapped_product_smiles=[], reconnect_boundaries=boundaries)


def test_saved_fluv_replacement_crosses_the_transaction_and_awaits_chemical_review(monkeypatch):
    original, directive, _, replacement = saved_fluv_repair()
    target = SAVED["fluvastatin"]["target"]
    compiler = RouteJSONCompiler()
    original_state = compiler.compile_route_graph_state(
        mapped_target_smiles=original[0]["mapped_product_smiles"], steps=original)
    # A frozen membership set, not a blanket True callback: no new advanced
    # intermediate can silently become a stock starting material in this test.
    stocked = {row.product_smiles for row in original_state.open_precursors}

    def editor(task):
        assert CHEMICAL_REVIEW_SCOPE in task.objective
        return WorkerRunRecord(
            run_id=task.task_id + ":run", task_id=task.task_id, case_id=task.case_id,
            status="accepted_draft", output_validation={"accepted": True, "reasons": []},
            output_artifact=_materialize_paper_matched_artifact(task, directive, backend="test"),
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    runner = director.SequentialStrategyDirectorRunner(
        editor_executor=editor,
        stock_membership=lambda values: {value: value in stocked for value in values},
    )

    def local_builder(_spec, *, seeded, **_kwargs):
        seeded[0]["steps"].extend(copy.deepcopy(replacement))
        seeded[0]["path_repair_builder_call_count"] += 1
        return []

    monkeypatch.setattr(runner, "_expand_seeded_branches_aizynthfinder", local_builder)
    spec = AgentSpec.from_context(
        run_id="saved-fluv", agent_id="director:saved-fluv", parent_agent_id="kernel:test",
        role="global_campaign_director", objective="offline saved IO repair",
        context={}, idempotency_key="saved-fluv", budget=Budget(max_wall_time_s=60, max_tokens=20000),
    )
    branch = {
        "branch_index": 0, "steps": copy.deepcopy(original),
        "target_mapped_smiles": original[0]["mapped_product_smiles"],
        "strategy_tree_engine": "aizynthfinder_mcts", "route_call_count": 23,
        "path_repair_builder_call_count": 0, "call_count": 23,
        "editor_attempt_count": 0, "editor_call_count": 0,
        "open_leaf_states": [], "open_leaves": [], "expanded_products": set(),
        "complete_in_bound_stock": True,
    }
    assert runner._repair_branch_transactionally(
        spec, target=target, branch=branch, completion_mode="cut_frontier",
        blocking_steps=[original[3]],
        critique={"status": "reject", "step_assessments": [{
            "step_id": original[3]["step_id"], "verdict": "reject", "blocking": True,
            "reasons": ["The encoded peroxide cannot undergo acetal hydrolysis."],
        }]}, iteration=0, records=[], max_prompt_bytes=100000, max_node_call_timeout_s=60,
        quota=director._NodeCallBudget(model_invocations=20, input_tokens=1000000,
                                      output_tokens=1000000, wall_time_s=600),
        started=time.monotonic(), reserve_model_invocations=0, reserve_input_tokens=0,
        reserve_output_tokens=0, reserve_wall_time_s=0,
        config=DirectorConfig(planning_mode="sequential_branches", paper_matched_reach_profile=True,
                              enable_transactional_path_repair=True,
                              strategy_tree_engine="aizynthfinder_mcts", strategy_branch_count=1,
                              max_node_expansions_per_branch=25, max_route_local_repair_rounds=6),
    ), (branch.get("path_repair_transactions"), branch.get("editor_rejection_diagnostics"))
    transaction = branch["path_repair_transactions"][-1]
    assert transaction["status"] == "rebuilt_pending_recritic"
    assert transaction["completion_boundary_reached"] is True
    assert transaction["final_frontier_restored"] is True
    assert transaction["suffix_stitch"]["remapped_boundary_atom_count"] == 1
    assert len(branch["steps"]) == 10
    assert branch["complete_in_bound_stock"] is True
    assert branch["_pending_path_repair_transaction"]["route_snapshot"]["steps"] == original
    # A chemically rejected re-review must still restore the old authoritative
    # route; the new boundary semantics must not force commitment.
    assert runner._rollback_pending_path_repair(
        branch, reason="path_repair_recritic_reject",
        candidate_critique={"status": "reject", "step_assessments": [{
            "step_id": "repair:oxidation", "verdict": "reject", "blocking": True,
        }]},
    )
    assert branch["steps"] == original

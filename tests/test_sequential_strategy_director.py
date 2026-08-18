from __future__ import annotations

import json
import re
from dataclasses import replace

import cascade_planner.orchestration.sequential_strategy_director as sequential_module

from cascade_planner.agent.codex_worker import WorkerRunRecord
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
    _has_atom_provenance_deficit,
    _strategy_conflicts,
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
        metadata={"model": "gpt-5.6-terra", "reasoning_effort": "medium"},
    )


def _strategy_card(branch: int) -> dict:
    return {
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


def _fake_executor(task):
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


def test_complete_route_json_allows_one_step_suffix_for_non_root_leaf() -> None:
    candidate = _complete_route_candidate()
    candidate["route_json"] = [candidate["route_json"][0]]
    expansions = _expansions_from_record(
        _proposal_record(candidate),
        expected_product="CCO",
        mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
        require_reaction_operations=True,
        require_complete_route_json=True,
        minimum_route_depth=1,
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
    plan = GlobalCampaignPlan.from_dict(result.output)
    assert all(
        dict(family.get("chemical_critic") or {}).get("status") == "viable"
        for family in plan.route_families
    )


def test_unavailable_critic_fails_closed_before_plan_delivery() -> None:
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

    assert result.state is AgentState.FAILED
    assert result.usage["critic_unavailable_branch_count"] == 1
    assert result.output is None


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
        if task.task_type == "route_chemistry_edit":
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
    )(spec, context, "initial_architecture", config)

    assert result.state is AgentState.SUCCEEDED
    assert editor_attempts == 2
    assert critic_attempts == 2
    assert sum(task.task_type == "route_chemistry_edit" for task in observed) == 2


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
        # Compatibility mode: without the paper RouteJSON contract, the host
        # may consume one ReactionJSON edit at a time.
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
    assert skeleton["routejson_authority"] == "host_routejson_compiler"
    assert len(skeleton["route_json"]) == 2


def test_paper_route_contract_overrides_compiler_first_compatibility_mode() -> None:
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

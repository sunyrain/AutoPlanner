from __future__ import annotations

import json
import re
from dataclasses import replace

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


def _fake_executor(task):
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
        "strategy_card": {
            "scaffold_motif": f"branch-{branch}-scaffold",
            "key_forward_transformation": f"branch-{branch}-key-construction",
            "key_bond_changes": [f"branch-{branch}-key-bond"],
            "functional_group_conflicts": [],
            "protection_policy": "avoid unless required for chemoselectivity",
            "stereochemical_plan": "preserve or construct assigned stereochemistry",
            "convergence_plan": "join independently accessible fragments",
            "strategic_step_count": 1,
            "skeleton_change_class": f"branch-{branch}-class",
            "expected_complexity_drop": "high",
            "orthogonality_basis": f"branch-{branch}-orthogonal",
            "strategy_signature": f"branch-{branch}-signature",
        },
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


def _fake_critic_executor(task):
    return _critic_record(task)


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
    assert result.usage["model_invocations"] == 6
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
    assert result.usage["model_invocations"] == 9
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
    assert all('"phase":"root_strategy"' in task.objective for task in observed_tasks[:3])
    assert all(
        "compare at least three plausible high-level strategies" in task.objective
        for task in observed_tasks[:3]
    )
    assert all("campaign_target_profile" in task.objective for task in observed_tasks[:3])


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
                "model_invocations": 4,
                "input_tokens": 100_000,
                "output_tokens": 100_000,
                "wall_time_s": 60.0,
            },
        },
    )
    result = SequentialStrategyDirectorRunner(node_executor=recording_executor)(
        spec, context, "initial_architecture", config
    )

    assert result.usage["model_invocations"] == 4
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
    assert branch_ids == [1, 2, 3]
    assert len(critic_tasks) == 1


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

    assert result.usage["model_invocations"] == 6
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
    assert result.usage["model_invocations"] == 1


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
    assert result.usage["model_invocations"] == 1


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
        if branch != 2:
            return record
        branch_two_calls += 1
        if branch_two_calls > 1:
            return record
        artifact = dict(record.output_artifact or {})
        payload = dict(artifact.get("payload") or {})
        candidates = [dict(row) for row in payload.get("candidates") or []]
        card = dict(candidates[0]["strategy_card"])
        card.update(
            {
                "key_forward_transformation": "branch-1-key-construction",
                "skeleton_change_class": "branch-1-class",
                "strategy_signature": "branch-1-signature",
            }
        )
        candidates[0]["strategy_card"] = card
        payload["candidates"] = candidates
        artifact["payload"] = payload
        return replace(record, output_artifact=artifact)

    result = SequentialStrategyDirectorRunner(
        node_executor=duplicate_once_executor,
        critic_executor=_fake_critic_executor,
    )(
        _spec(context), context, "initial_architecture", config
    )

    assert result.state is AgentState.SUCCEEDED
    assert result.usage["model_invocations"] == 9
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

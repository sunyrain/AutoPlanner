from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import pytest

from cascade_planner.application.campaign_context import (
    CampaignContext,
    CampaignContextCompiler,
    CampaignContextTooLargeError,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.run_kernel import RunKernel, RunLimits, RunSpec
from cascade_planner.orchestration.global_campaign_director import (
    DirectorConfig,
    GlobalCampaignDirector,
    GlobalCampaignPlan,
    GlobalCampaignPlanValidationError,
    ReplayDirectorRunner,
    director_trigger_reasons,
    director_prompt,
    director_web_search_enabled,
    repair_global_campaign_plan_contract,
    validate_global_campaign_plan,
)
from cascade_planner.runtime import AgentResult, AgentSpec, AgentState


def _kernel(tmp_path: Path, *, calls: int = 3) -> RunKernel:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="campaign-1",
            target_name="example target",
            target_smiles="CCOC(=O)N",
            created_at="2026-07-13T00:00:00Z",
            limits=RunLimits(
                model=RetrosynthesisRunBudget(
                    max_model_invocations=calls,
                    max_total_input_tokens=100_000,
                    max_total_output_tokens=20_000,
                    max_total_wall_time_s=900.0,
                    max_accepted_expansions=8,
                    max_attempt_runs=12,
                    max_prompt_context_bytes=96_000,
                )
            ),
        ),
    )
    kernel.start()
    return kernel


def _context(
    kernel: RunKernel,
    *,
    previous: CampaignContext | None = None,
    material_events: tuple[str, ...] = (),
) -> CampaignContext:
    return CampaignContextCompiler().compile(
        kernel=kernel,
        hypergraph={
            "molecules": {
                "mol:target": {"molecule_id": "mol:target", "smiles": "CCOC(=O)N"},
                "mol:shared": {"molecule_id": "mol:shared", "smiles": "CCO"},
            },
            "hyperedges": [
                {
                    "edge_id": "edge:1",
                    "product_id": "mol:target",
                    "reactant_ids": ["mol:shared"],
                }
            ],
            "route_families": [
                {"route_family_id": "existing:1", "edge_ids": ["edge:1"]}
            ],
        },
        route_portfolio={
            "routes": [
                {"route_id": "route:1", "edge_ids": ["edge:1"]},
                {"route_id": "route:duplicate", "edge_ids": ["edge:1"]},
            ]
        },
        evidence_ledger={
            "records": [
                {
                    "evidence_id": "ev:1",
                    "source_ref": "patent:1",
                    "raw_text": "confidential procedure text " * 100,
                }
            ]
        },
        stock_ledger={"records": [{"stock_id": "stock:1", "smiles": "CCO"}]},
        failure_history=[
            {"failure_id": f"failure:{index}", "reason": "no_match"}
            for index in range(60)
        ],
        previous=previous,
        material_events=material_events,
    )


def _plan(context: CampaignContext, *, invalid_smiles: bool = False) -> dict[str, Any]:
    target = "not-a-smiles" if invalid_smiles else "CCOC(=O)N"
    return {
        "schema_version": "global_campaign_plan.v1",
        "plan_id": f"plan:{context.content_sha256[:12]}",
        "run_id": context.run_id,
        "mode": "initial_architecture",
        "context_sha256": context.content_sha256,
        "graph_revision": context.revision.graph_revision,
        "route_families": [
            {
                "route_family_id": "family:amide",
                "title": "amide-late",
                "strategy": "disconnect the terminal amide",
                "target_smiles": "CCOC(=O)N",
                "advantages": ["short"],
                "risks": ["selectivity"],
                "diversity_basis": "polar disconnection",
            },
            {
                "route_family_id": "family:ester",
                "title": "ester-late",
                "strategy": "change convergence order",
                "target_smiles": "CCOC(=O)N",
                "advantages": ["shared feedstock"],
                "risks": ["chemoselectivity"],
                "diversity_basis": "different strategic bond",
            },
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": "skeleton:amide",
                "route_family_id": "family:amide",
                "summary": "two-step family",
                "steps": [
                    {
                        "step_id": "proposal:amide:1",
                        "product_smiles": target,
                        "precursor_smiles": ["CCOC(=O)O", "N"],
                        "transformation_hypothesis": "amide formation",
                        "strategic_role": "terminal convergence",
                        "source_hints": ["amide coupling"],
                        "required_validation": ["identity", "element_balance"],
                        "hypothesis_only": True,
                    },
                    {
                        "step_id": "proposal:amide:2",
                        "product_smiles": "CCOC(=O)O",
                        "precursor_smiles": ["CCO", "O=C=O"],
                        "transformation_hypothesis": "carboxylation",
                        "strategic_role": "shared intermediate formation",
                        "source_hints": [],
                        "required_validation": ["identity", "precedent"],
                        "hypothesis_only": True,
                    },
                ],
            },
            {
                "skeleton_id": "skeleton:ester",
                "route_family_id": "family:ester",
                "summary": "orthogonal family",
                "steps": [
                    {
                        "step_id": "proposal:ester:1",
                        "product_smiles": "CCOC(=O)N",
                        "precursor_smiles": ["CCO", "NC=O"],
                        "transformation_hypothesis": "esterification-like assembly",
                        "strategic_role": "alternative convergence",
                        "source_hints": [],
                        "required_validation": ["identity", "precedent"],
                        "hypothesis_only": True,
                    }
                ],
            },
        ],
        "strategic_disconnections": [
            {
                "disconnection_id": "disc:amide",
                "target_smiles": "CCOC(=O)N",
                "bond_or_retron": "amide C-N",
                "rationale": "late convergent assembly",
                "route_family_ids": ["family:amide"],
                "required_validation": ["precedent"],
            }
        ],
        "shared_intermediates": [
            {
                "intermediate_id": "shared:ethanol",
                "smiles": "CCO",
                "route_family_ids": ["family:amide", "family:ester"],
                "strategic_role": "common feedstock",
                "risk": "low",
            }
        ],
        "critical_unknowns": [
            {
                "unknown_id": "unknown:selectivity",
                "description": "chemoselectivity unknown",
                "affected_proposal_ids": ["proposal:ester:1"],
                "resolution_task": "find exact precedent",
                "priority": 9,
            }
        ],
        "source_plan": [
            {
                "source_task_id": "source:amide",
                "query": "target amide synthesis patent",
                "source_types": ["patent", "paper_si"],
                "target_claims": ["exact procedure"],
                "affected_proposal_ids": ["proposal:amide:1"],
                "priority": 10,
            }
        ],
        "fallback_strategies": [
            {
                "fallback_id": "fallback:ester",
                "trigger": "amide family rejected",
                "action": "prioritize ester family",
                "route_family_ids": ["family:ester"],
            }
        ],
        "frontier_priorities": [
            {
                "priority_id": "priority:1",
                "proposal_id": "proposal:amide:1",
                "priority": 10,
                "rationale": "closest to closure",
                "expected_portfolio_gain": "validates one whole family",
            }
        ],
        "pivot_conditions": [
            {
                "pivot_id": "pivot:1",
                "condition": "critical edge rejected",
                "action": "replan both families around bottleneck",
            }
        ],
        "stop_conditions": [
            {
                "stop_id": "stop:1",
                "condition": "all actionable deficits exhausted",
                "disposition": "unresolved",
            }
        ],
        "portfolio_rationale": "Preserve two strategically distinct families sharing one audited feedstock.",
        "limitations": ["all steps require deterministic validation"],
    }


def _runner(plan: dict[str, Any]):
    calls: list[AgentSpec] = []

    def run(
        spec: AgentSpec,
        _context: CampaignContext,
        _mode: str,
        _config: DirectorConfig,
    ) -> AgentResult:
        calls.append(spec)
        return AgentResult(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            attempt=spec.attempt,
            idempotency_key=f"{spec.idempotency_key}:result",
            context_hash=spec.context_hash,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=AgentState.SUCCEEDED,
            output=plan,
            usage={
                "model_invocations": 1,
                "input_tokens": 2_000,
                "output_tokens": 1_000,
                "wall_time_s": 2.5,
            },
            metadata={"backend": "deterministic_fake", "direct_child": True},
        )

    return calls, run


def test_context_preserves_topology_but_compacts_raw_and_duplicate_data(
    tmp_path: Path,
) -> None:
    context = _context(_kernel(tmp_path))

    assert set(context.topology["molecules"]) == {"mol:target", "mol:shared"}
    assert len(context.topology["hyperedges"]) == 1
    assert len(context.route_portfolio["routes"]) == 1
    assert context.route_portfolio["duplicate_route_ids_by_representative"] == {
        "route:1": ["route:duplicate"]
    }
    evidence = context.evidence["records"][0]
    assert "raw_text" not in evidence
    assert evidence["omitted_raw_content"][0]["sha256"]
    assert len(context.failure_history) == 48
    assert len(json.dumps(context.to_dict(), sort_keys=True, separators=(",", ":"))) == context.byte_count


def test_context_delta_and_byte_budget_are_host_enforced(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    first = _context(kernel)
    second = _context(
        kernel,
        previous=first,
        material_events=("critical_edge_rejected",),
    )

    assert second.delta.changed_sections == ()
    assert second.delta.material_events == ("critical_edge_rejected",)
    assert director_trigger_reasons(second, mode="event_replan") == [
        "critical_edge_rejected"
    ]
    with pytest.raises(CampaignContextTooLargeError):
        CampaignContextCompiler(max_context_bytes=100).compile(
            kernel=kernel,
            hypergraph={"nodes": [{"id": "n:1"}]},
        )


def test_director_coordinates_global_families_through_one_kernel_call_and_cache(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    calls, runner = _runner(_plan(context))
    director = GlobalCampaignDirector(kernel, runner=runner)

    first = director.run(context, mode="initial_architecture")
    second = director.run(context, mode="initial_architecture")

    assert first.invoked is True and first.cache_hit is False
    assert second.invoked is False and second.cache_hit is True
    assert len(calls) == 1
    assert calls[0].role == "global_campaign_director"
    assert calls[0].parent_agent_id == "run-kernel:campaign-1"
    assert len(first.plan.route_families) == 2
    assert len(first.plan.multi_step_skeletons) == 2
    assert len(first.plan.shared_intermediates) == 1
    assert all(row["accepted"] is True for row in first.proposal_audits)
    assert kernel.state.attempt_count == 0
    assert kernel.state.model_totals["model_invocations"] == 1
    assert kernel.state.accepted_expansion_count == 0


def test_event_replan_without_material_change_is_ignored_without_model(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    calls, runner = _runner(_plan(context))

    outcome = GlobalCampaignDirector(kernel, runner=runner).run(
        context,
        mode="event_replan",
    )

    assert outcome.status == "ignored"
    assert outcome.invoked is False
    assert calls == []
    assert kernel.state.attempt_count == 0


def test_director_enforces_campaign_level_call_caps_per_global_mode(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path, calls=6)
    calls: list[str] = []

    def dynamic_runner(
        spec: AgentSpec,
        context: CampaignContext,
        mode: str,
        _config: DirectorConfig,
    ) -> AgentResult:
        raw = _plan(context)
        raw["mode"] = mode
        calls.append(mode)
        return AgentResult(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            attempt=spec.attempt,
            idempotency_key=f"{spec.idempotency_key}:result",
            context_hash=spec.context_hash,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=AgentState.SUCCEEDED,
            output=raw,
            usage={"model_invocations": 1, "wall_time_s": 0.1},
        )

    director = GlobalCampaignDirector(kernel, runner=dynamic_runner)
    initial = _context(kernel)
    changed_initial = _context(
        kernel,
        previous=initial,
        material_events=("new_route_family",),
    )

    assert director.run(initial, mode="initial_architecture").status == "accepted"
    blocked_initial = director.run(
        changed_initial,
        mode="initial_architecture",
    )
    assert blocked_initial.status == "budget_exhausted"
    assert blocked_initial.reasons == ("director_mode_call_budget_exhausted",)

    previous = initial
    for event in ("critical_edge_rejected", "exact_rows_added"):
        context = _context(kernel, previous=previous, material_events=(event,))
        assert director.run(context, mode="event_replan").status == "accepted"
        previous = context
    third_replan = _context(
        kernel,
        previous=previous,
        material_events=("stock_boundary_changed",),
    )
    assert director.run(third_replan, mode="event_replan").status == (
        "budget_exhausted"
    )

    final = _context(kernel, previous=third_replan)
    assert director.run(final, mode="final_portfolio_synthesis").status == "accepted"
    another_final = _context(
        kernel,
        previous=final,
        material_events=("portfolio_stagnation",),
    )
    assert director.run(
        another_final,
        mode="final_portfolio_synthesis",
    ).status == "budget_exhausted"
    assert calls == [
        "initial_architecture",
        "event_replan",
        "event_replan",
        "final_portfolio_synthesis",
    ]


def test_critical_edge_rejection_triggers_targeted_global_replan(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    initial = _context(kernel)
    replan_context = _context(
        kernel,
        previous=initial,
        material_events=("critical_edge_rejected",),
    )
    raw = _plan(replan_context)
    raw["mode"] = "event_replan"
    raw["pivot_conditions"][0]["condition"] = "proposal:amide:1 rejected"
    calls, runner = _runner(raw)

    outcome = GlobalCampaignDirector(kernel, runner=runner).run(
        replan_context,
        mode="event_replan",
    )

    assert outcome.invoked is True
    assert len(calls) == 1
    assert outcome.plan.mode == "event_replan"
    assert outcome.plan.pivot_conditions[0]["condition"] == (
        "proposal:amide:1 rejected"
    )


def test_invalid_molecule_is_rejected_as_candidate_not_promoted_to_fact(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    calls, runner = _runner(_plan(context, invalid_smiles=True))

    outcome = GlobalCampaignDirector(kernel, runner=runner).run(
        context,
        mode="initial_architecture",
    )

    invalid = next(
        row
        for row in outcome.proposal_audits
        if row["proposal_id"] == "proposal:amide:1"
    )
    assert invalid["accepted"] is False
    assert "product_identity_invalid" in invalid["reasons"]
    assert kernel.state.accepted_expansion_count == 0
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda plan: plan["multi_step_skeletons"][0]["steps"][0].update(
                product_smiles="CCO"
            ),
            "skeleton_requires_exactly_one_target_root",
        ),
        (
            lambda plan: plan["multi_step_skeletons"][0]["steps"][1].update(
                product_smiles="CCC"
            ),
            "skeleton_contains_disconnected_steps",
        ),
        (
            lambda plan: plan["multi_step_skeletons"][0]["steps"][1].update(
                product_smiles="CCOC(=O)O",
                precursor_smiles=["CCOC(=O)N"],
            ),
            "skeleton_ancestor_cycle",
        ),
    ],
)
def test_director_rejects_non_target_rooted_disconnected_or_cyclic_skeletons(
    tmp_path: Path,
    mutate: Any,
    reason: str,
) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    raw = _plan(context)
    mutate(raw)
    audits = validate_global_campaign_plan(GlobalCampaignPlan.from_dict(raw), context)
    affected = [row for row in audits if row["skeleton_id"] == "skeleton:amide"]
    assert affected
    assert all(row["accepted"] is False for row in affected)
    assert all(reason in row["reasons"] for row in affected)


def test_director_rejects_duplicate_target_level_route_family_chemistry(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    raw = _plan(context)
    raw["multi_step_skeletons"][1]["steps"][0]["precursor_smiles"] = [
        "CCOC(=O)O",
        "N",
    ]
    audits = validate_global_campaign_plan(GlobalCampaignPlan.from_dict(raw), context)
    duplicate = [row for row in audits if row["skeleton_id"] == "skeleton:ester"]
    assert duplicate
    assert all("route_family_root_not_distinct" in row["reasons"] for row in duplicate)


def test_director_allows_shared_target_edge_when_upstream_program_diverges(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    raw = _plan(context)
    second = raw["multi_step_skeletons"][1]
    second["steps"][0]["precursor_smiles"] = ["CCOC(=O)O", "N"]
    second["steps"].append(
        {
            "step_id": "proposal:ester:2",
            "product_smiles": "CCOC(=O)O",
            "precursor_smiles": ["CCOC(=O)Cl"],
            "transformation_hypothesis": "hydrolysis of an alternative acid feed",
            "strategic_role": "upstream divergent supply",
            "source_hints": [],
            "required_validation": ["identity", "precedent"],
            "hypothesis_only": True,
        }
    )

    audits = validate_global_campaign_plan(GlobalCampaignPlan.from_dict(raw), context)

    shared_root_family = [
        row for row in audits if row["skeleton_id"] == "skeleton:ester"
    ]
    assert shared_root_family
    assert all(row["accepted"] is True for row in shared_root_family)
    assert all(
        "route_family_root_not_distinct" not in row["reasons"]
        for row in shared_root_family
    )


def test_director_output_cannot_grant_scientific_authority(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    raw = _plan(context)
    raw["route_families"][0]["solved"] = True
    plan = GlobalCampaignPlan.from_dict(raw)

    with pytest.raises(
        GlobalCampaignPlanValidationError,
        match="claimed_scientific_authority",
    ):
        validate_global_campaign_plan(plan, context)


def test_director_repairs_only_redundant_family_target_metadata(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    raw = _plan(context)
    raw["route_families"][0]["target_smiles"] = "CCOC(=O)O"

    repaired, repairs = repair_global_campaign_plan_contract(
        GlobalCampaignPlan.from_dict(raw),
        context,
    )

    assert repaired.route_families[0]["target_smiles"] == "CCOC(N)=O"
    assert repairs[0]["route_family_id"] == raw["route_families"][0]["route_family_id"]
    assert repairs[0]["semantics"]["chemistry_unchanged"] is True
    assert validate_global_campaign_plan(repaired, context)


def test_director_does_not_repair_family_without_exact_target_root(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    raw = _plan(context)
    family_id = raw["route_families"][0]["route_family_id"]
    raw["route_families"][0]["target_smiles"] = "CCOC(=O)O"
    for skeleton in raw["multi_step_skeletons"]:
        if skeleton["route_family_id"] == family_id:
            skeleton["steps"][0]["product_smiles"] = "CCO"

    unrepaired, repairs = repair_global_campaign_plan_contract(
        GlobalCampaignPlan.from_dict(raw),
        context,
    )

    assert repairs == ()
    assert unrepaired.route_families[0]["target_smiles"] == "CCOC(=O)O"


def test_director_retains_route_family_without_skeleton_as_advisory(tmp_path: Path) -> None:
    context = _context(_kernel(tmp_path))
    raw = _plan(context)
    raw["route_families"].append(
        {
            **raw["route_families"][0],
            "route_family_id": "family:orphan-metadata",
            "diversity_basis": "declared but never expanded",
        }
    )

    repaired, repairs = repair_global_campaign_plan_contract(
        GlobalCampaignPlan.from_dict(raw),
        context,
    )

    assert "family:orphan-metadata" in {
        row["route_family_id"] for row in repaired.route_families
    }
    assert any(
        row["reason"] == "route_family_without_skeleton_retained_as_advisory"
        and row["semantics"]["retained_as_advisory_only"] is True
        for row in repairs
    )
    assert validate_global_campaign_plan(repaired, context)


def test_director_removes_identity_leaf_marker_without_rewriting_chemistry(
    tmp_path: Path,
) -> None:
    context = _context(_kernel(tmp_path))
    raw = _plan(context)
    raw["multi_step_skeletons"][0]["steps"].append(
        {
            "step_id": "proposal:amide:stock-leaf",
            "product_smiles": "CCO",
            "precursor_smiles": ["CCO"],
            "transformation_hypothesis": "commercially available",
            "strategic_role": "terminal leaf marker",
            "source_hints": [],
            "required_validation": ["stock audit"],
            "hypothesis_only": True,
        }
    )
    raw["frontier_priorities"].append(
        {
            "priority_id": "priority:identity-leaf",
            "proposal_id": "proposal:amide:stock-leaf",
            "priority": 7,
            "rationale": "leaf is commercially available",
            "expected_portfolio_gain": "none",
        }
    )

    repaired, repairs = repair_global_campaign_plan_contract(
        GlobalCampaignPlan.from_dict(raw),
        context,
    )

    repaired_ids = {
        step["step_id"]
        for skeleton in repaired.multi_step_skeletons
        for step in skeleton["steps"]
    }
    assert "proposal:amide:stock-leaf" not in repaired_ids
    assert all(
        row.get("proposal_id") != "proposal:amide:stock-leaf"
        for row in repaired.frontier_priorities
    )
    assert any(row["reason"] == "identity_leaf_marker_removed" for row in repairs)
    assert validate_global_campaign_plan(repaired, context)


def test_director_completes_chemenzy_metadata_for_codex_selected_non_root_step(
    tmp_path: Path,
) -> None:
    context = _context(_kernel(tmp_path))
    raw = _plan(context)
    raw["frontier_priorities"].append(
        {
            "priority_id": "priority:shared-acid",
            "proposal_id": "proposal:amide:2",
            "priority": 9,
            "rationale": "expand the selected shared intermediate locally",
            "expected_portfolio_gain": "closes one upstream branch",
        }
    )

    repaired, repairs = repair_global_campaign_plan_contract(
        GlobalCampaignPlan.from_dict(raw),
        context,
    )

    priority = next(
        row
        for row in repaired.frontier_priorities
        if row["priority_id"] == "priority:shared-acid"
    )
    assert priority["target_smiles"] == "CCOC(=O)O"
    assert priority["provider_preferences"] == ["chemenzy"]
    assert priority["route_family_ids"] == ["family:amide"]
    assert any(
        row["field"] == "frontier_priorities.provider_delegation"
        for row in repairs
    )
    assert validate_global_campaign_plan(repaired, context)


def test_director_resolves_chemenzy_request_to_shared_intermediate(
    tmp_path: Path,
) -> None:
    context = _context(_kernel(tmp_path))
    raw = _plan(context)
    raw["frontier_priorities"].append(
        {
            "priority_id": "priority:shared-ethanol",
            "proposal_id": "shared:ethanol",
            "priority": 9.5,
            "rationale": "delegate the Codex-selected shared node",
            "expected_portfolio_gain": "adds local alternatives",
        }
    )

    repaired, repairs = repair_global_campaign_plan_contract(
        GlobalCampaignPlan.from_dict(raw),
        context,
    )

    priority = next(
        row
        for row in repaired.frontier_priorities
        if row["priority_id"] == "priority:shared-ethanol"
    )
    assert priority["target_smiles"] == "CCO"
    assert priority["provider_preferences"] == ["chemenzy"]
    assert priority["route_family_ids"] == ["family:amide", "family:ester"]
    assert any(
        row["reason"].endswith("shared_intermediate") for row in repairs
    )
    assert validate_global_campaign_plan(repaired, context)


def test_concurrent_identical_context_invokes_runner_once(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    calls, runner = _runner(_plan(context))
    director = GlobalCampaignDirector(kernel, runner=runner)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda _index: director.run(
                    context,
                    mode="initial_architecture",
                ),
                range(2),
            )
        )

    assert len(calls) == 1
    assert sum(outcome.invoked for outcome in outcomes) == 1
    assert sum(outcome.cache_hit for outcome in outcomes) == 1
    assert kernel.state.model_totals["model_invocations"] == 1


def test_replay_runner_is_model_free_and_schema_identical(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path, calls=0)
    context = _context(kernel)
    replay = ReplayDirectorRunner({"initial_architecture": _plan(context)})

    outcome = GlobalCampaignDirector(kernel, runner=replay).run(
        context,
        mode="initial_architecture",
    )

    assert outcome.plan.to_dict()["schema_version"] == "global_campaign_plan.v1"
    assert len(replay.calls) == 1
    assert kernel.state.model_totals["model_invocations"] == 0


def test_director_defers_web_tools_until_evidence_informed_replan(
    tmp_path: Path,
) -> None:
    context = _context(_kernel(tmp_path))
    config = DirectorConfig(enable_web_search=True)

    assert director_web_search_enabled(config, mode="initial_architecture") is False
    assert director_web_search_enabled(config, mode="event_replan") is True
    assert "Live search is deferred from this first-route pass" in director_prompt(
        context,
        mode="initial_architecture",
        config=config,
    )
    assert "Live search is enabled" in director_prompt(
        context,
        mode="event_replan",
        config=config,
    )

    opted_in = DirectorConfig(
        enable_web_search=True,
        enable_initial_web_search=True,
    )
    assert (
        director_web_search_enabled(opted_in, mode="initial_architecture") is True
    )


def test_director_long_route_profile_requests_explicit_uncompressed_steps(
    tmp_path: Path,
) -> None:
    context = _context(_kernel(tmp_path))
    prompt = director_prompt(
        context,
        mode="initial_architecture",
        config=DirectorConfig(max_steps_per_skeleton=24),
    )

    assert "long-route-capable proof run" in prompt
    assert "include 20+ explicit steps when chemistry requires them" in prompt
    assert "Do not compress a multistep chemical sequence" in prompt
    assert "may share a target-forming edge" in prompt
    assert "The host applies RDKit canonicalization" in prompt


def test_director_enforces_configured_planning_depth_without_claiming_proof(
    tmp_path: Path,
) -> None:
    context = _context(
        _kernel(tmp_path),
        material_events=("director_depth_deficit",),
    )
    config = DirectorConfig(
        minimum_planning_route_steps=20,
        max_steps_per_skeleton=24,
    )

    prompt = director_prompt(context, mode="event_replan", config=config)

    assert "MUST itself contain at least 20 explicit single-reaction steps" in prompt
    assert "cannot be split across a main skeleton" in prompt
    assert "at least 20 unique step products" in prompt
    assert "no precursor chain may point back to an ancestor" in prompt
    assert "planning/display requirement, not proof" in prompt
    assert "prior plan missed the configured 20-step planning depth" in prompt
    assert "do not pad or fabricate steps" in prompt
    with pytest.raises(ValueError, match="planning route depth"):
        DirectorConfig(
            minimum_planning_route_steps=20,
            max_steps_per_skeleton=12,
        )


def test_director_safely_merges_unique_same_family_leaf_continuation(
    tmp_path: Path,
) -> None:
    context = _context(_kernel(tmp_path))
    raw = _plan(context)
    parent = raw["multi_step_skeletons"][0]
    continuation_step = parent["steps"].pop()
    raw["multi_step_skeletons"].append(
        {
            "skeleton_id": "skeleton:amide:continuation",
            "route_family_id": "family:amide",
            "summary": "split upstream continuation",
            "steps": [continuation_step],
        }
    )

    repaired, repairs = repair_global_campaign_plan_contract(
        GlobalCampaignPlan.from_dict(raw),
        context,
        config=DirectorConfig(max_steps_per_skeleton=12),
    )

    skeletons = {
        row["skeleton_id"]: row for row in repaired.multi_step_skeletons
    }
    assert "skeleton:amide:continuation" not in skeletons
    assert len(skeletons["skeleton:amide"]["steps"]) == 2
    assert any(
        row["reason"] == "unique_leaf_continuation_skeleton_merged"
        and row["combined_step_count"] == 2
        for row in repairs
    )
    assert validate_global_campaign_plan(
        repaired,
        context,
        DirectorConfig(max_steps_per_skeleton=12),
    )


def test_director_does_not_merge_unrelated_or_over_depth_continuation(
    tmp_path: Path,
) -> None:
    context = _context(_kernel(tmp_path))
    raw = _plan(context)
    raw["multi_step_skeletons"].append(
        {
            "skeleton_id": "skeleton:unrelated-continuation",
            "route_family_id": "family:amide",
            "summary": "unrelated chemistry",
            "steps": [
                {
                    "step_id": "proposal:unrelated",
                    "product_smiles": "CCC",
                    "precursor_smiles": ["CC", "C"],
                    "transformation_hypothesis": "unrelated split",
                    "required_validation": ["identity"],
                    "hypothesis_only": True,
                }
            ],
        }
    )

    repaired, repairs = repair_global_campaign_plan_contract(
        GlobalCampaignPlan.from_dict(raw),
        context,
        config=DirectorConfig(max_steps_per_skeleton=12),
    )

    assert any(
        row["skeleton_id"] == "skeleton:unrelated-continuation"
        for row in repaired.multi_step_skeletons
    )
    assert not any(
        row["reason"] == "unique_leaf_continuation_skeleton_merged"
        for row in repairs
    )


def test_director_topology_replan_requests_connected_alternative_skeletons(
    tmp_path: Path,
) -> None:
    context = _context(
        _kernel(tmp_path),
        material_events=("director_topology_rejected",),
    )

    prompt = director_prompt(
        context,
        mode="event_replan",
        config=DirectorConfig(max_steps_per_skeleton=24),
    )

    assert "A prior skeleton failed host topology" in prompt
    assert "never append a disconnected backup chain" in prompt


def test_proposal_dispositions_preserve_superseded_and_ignored_history(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    context = _context(kernel)
    calls, runner = _runner(_plan(context))
    director = GlobalCampaignDirector(kernel, runner=runner)
    outcome = director.run(context, mode="initial_architecture")

    superseded = director.record_dispositions(
        outcome.plan,
        {
            "proposal:amide:1": "superseded",
            "proposal:ester:1": "ignored",
        },
        reasons={"proposal:amide:1": ["better_shared_intermediate_found"]},
    )

    assert {row["disposition"] for row in superseded} == {
        "superseded",
        "ignored",
    }
    audits = [
        row
        for row in kernel.index.artifacts_for_run(kernel.spec.run_id)
        if row["artifact_id"].startswith("director_proposal_audit:")
    ]
    assert len(audits) == 2
    assert len(calls) == 1

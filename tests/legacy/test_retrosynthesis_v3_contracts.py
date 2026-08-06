from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade_planner.application.retrosynthesis_run_contract import (
    ModelCostEvent,
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisCostLedger,
    RetrosynthesisRunBudget,
    model_cost_event_from_worker_record,
)
from cascade_planner.legacy.application_runtime.route_deficit_queue import (
    RouteDeficitKind,
    compile_route_deficit_queue,
    next_route_deficit,
)
from cascade_planner.legacy.application_runtime.retrosynthesis_acceptance import (
    evaluate_retrosynthesis_acceptance,
)
from cascade_planner.legacy.harness_runtime.route_forest import compile_explored_route_forest


def test_run_cost_ledger_is_shared_and_fails_closed_on_unobserved_usage() -> None:
    ledger = RetrosynthesisCostLedger(
        budget=RetrosynthesisRunBudget(
            max_model_invocations=2,
            max_total_input_tokens=1_000,
            max_total_output_tokens=100,
            max_total_wall_time_s=60,
            max_visual_invocations=1,
        )
    )
    ledger.reserve("campaign:root", prompt_context_bytes=512)
    ledger.settle(
        ModelCostEvent(
            invocation_id="campaign:root",
            worker_kind="route_portfolio_proposal",
            elapsed_s=4.5,
            input_tokens=200,
            output_tokens=25,
        )
    )
    ledger.reserve("visual:edge-1", visual=True, prompt_context_bytes=128)
    ledger.settle(
        ModelCostEvent(
            invocation_id="visual:edge-1",
            worker_kind="visual_gap_repair",
            elapsed_s=1.5,
            visual=True,
            usage_observed=False,
        )
    )

    assert ledger.totals() == {
        "model_invocations": 2,
        "visual_invocations": 1,
        "input_tokens": 200,
        "cached_input_tokens": 0,
        "output_tokens": 25,
        "reasoning_output_tokens": 0,
        "accepted_expansions": 0,
        "attempt_runs": 0,
        "wall_time_s": 6.0,
    }
    assert ledger.gate_reasons() == [
        "prior_model_usage_unobserved",
        "run_model_invocation_budget_exhausted",
    ]
    with pytest.raises(RuntimeError, match="run_model_invocation_budget_exhausted"):
        ledger.reserve("campaign:retry")

    replayed = RetrosynthesisCostLedger.from_dict(ledger.to_dict())
    assert replayed.to_dict()["totals"] == ledger.to_dict()["totals"]


def test_worker_record_usage_is_host_normalized() -> None:
    event = model_cost_event_from_worker_record(
        SimpleNamespace(
            elapsed_s=12.25,
            status="rejected_output",
            usage={
                "input_tokens": "123",
                "cached_input_tokens": 100,
                "output_tokens": 8,
                "reasoning_output_tokens": 2,
            },
        ),
        invocation_id="campaign:one",
        worker_kind="route_portfolio_proposal",
    )
    assert event.input_tokens == 123
    assert event.cached_input_tokens == 100
    assert event.output_tokens == 8
    assert event.usage_observed is True
    assert event.status == "rejected_output"


def test_route_deficit_queue_prioritizes_selected_exact_evidence_before_model() -> None:
    ledger = {
        "schema_version": "frontier_ledger.v1",
        "edges": {
            "edge:selected": {
                "proposal": {"source_refs": ["doi:10.1/example"]},
                "reaction_proof": {"achieved_proof_level": 2},
            },
            "edge:advisory": {
                "proposal": {"source_refs": []},
                "reaction_proof": {"achieved_proof_level": 0},
            },
        },
        "molecules": {
            "CCO": {
                "proposal": {"outgoing_edge_signatures": []},
                "work": {"proposal_expansion_allowed": True},
                "stock": {
                    "closed": False,
                    "procurement_boundary_closed": False,
                },
            }
        },
    }
    portfolio = {
        "routes": [
            {
                "route_id": "route:1",
                "selected": True,
                "edge_signatures": ["edge:selected"],
                "leaf_smiles": ["CCO"],
                "complete": False,
            }
        ]
    }
    queue = compile_route_deficit_queue(
        frontier_ledger=ledger,
        route_portfolio=portfolio,
        acceptance_spec=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=2,
            minimum_edge_proof_level=3,
        ),
    )

    assert queue["summary"]["next_kind"] == RouteDeficitKind.EXACT_EVIDENCE.value
    assert queue["deficits"][0]["object_id"] == "edge:selected"
    assert queue["summary"]["by_kind"] == {
        "exact_evidence": 1,
        "reaction_validation": 1,
        "stock_audit": 1,
        "structure_materialization": 0,
        "proposal_expansion": 1,
        "route_diversity": 1,
    }
    assert next_route_deficit(queue, allow_model=False)["kind"] == (
        "exact_evidence"
    )


def test_source_capability_is_folded_into_same_deficit_queue() -> None:
    queue = compile_route_deficit_queue(
        frontier_ledger={"edges": {}, "molecules": {}},
        source_capability_queue={
            "capabilities": [
                {
                    "capability_id": "compile:edge-1",
                    "action_type": "compile_exact_literature_rows",
                    "eligible": True,
                    "source_ref": "patent:WO1",
                }
            ]
        },
        acceptance_spec=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=1,
        ),
    )
    assert queue["deficits"][0]["kind"] == "exact_evidence"
    assert queue["deficits"][0]["source_refs"] == ["patent:WO1"]
    assert queue["semantics"]["single_cross_subsystem_work_projection"] is True


def test_acceptance_requires_two_distinct_verified_stock_closed_routes() -> None:
    route = {
        "complete": True,
        "reaction_validated": True,
        "procurement_ready": True,
        "weakest_proof_level": 3,
    }
    report = evaluate_retrosynthesis_acceptance(
        route_portfolio={
            "routes": [
                {
                    **route,
                    "route_id": "science",
                    "hyperedge_ids": ["s1", "s2"],
                    "independent_support_groups": ["doi:science"],
                },
                {
                    **route,
                    "route_id": "patent",
                    "hyperedge_ids": ["p1", "p2"],
                    "independent_support_groups": ["patent:wo"],
                },
            ]
        },
        acceptance_spec=RetrosynthesisAcceptanceSpec(),
    )
    assert report["accepted"] is True
    assert report["selected_route_ids"] == ["science", "patent"]


def test_acceptance_rejects_duplicate_edge_set_disguised_as_alternative() -> None:
    route = {
        "complete": True,
        "reaction_validated": True,
        "procurement_ready": True,
        "weakest_proof_level": 3,
        "hyperedge_ids": ["same-edge"],
    }
    report = evaluate_retrosynthesis_acceptance(
        route_portfolio={
            "routes": [
                {**route, "route_id": "a", "independent_support_groups": ["a"]},
                {**route, "route_id": "b", "independent_support_groups": ["b"]},
            ]
        },
    )
    assert report["accepted"] is False
    assert report["selected_route_count"] == 1


def test_route_forest_exposes_digest_checked_acceptance_deficit_and_cost() -> None:
    acceptance = evaluate_retrosynthesis_acceptance(
        route_portfolio={"routes": []},
    )
    queue = compile_route_deficit_queue(
        frontier_ledger={"edges": {}, "molecules": {}},
    )
    cost = RetrosynthesisCostLedger().to_dict()
    blackboard = {
        "case_id": "v3-control-projection",
        "target_profile": {"target_name": "fixture", "target_smiles": "CCO"},
        "retrosynthesis_run_contract": {
            "schema_version": "retrosynthesis_run_contract.v1",
            "acceptance_spec": RetrosynthesisAcceptanceSpec().to_dict(),
            "cost_ledger": cost,
        },
        "retrosynthesis_acceptance": acceptance,
        "route_deficit_queue": queue,
    }

    forest = compile_explored_route_forest(blackboard)
    control = forest["retrosynthesis_control"]

    assert control["authoritative"] is True
    assert control["acceptance"]["accepted"] is False
    assert control["next_deficit"]["kind"] == "route_diversity"
    assert control["cost_totals"]["model_invocations"] == 0

    blackboard["retrosynthesis_acceptance"] = {
        **acceptance,
        "accepted": True,
    }
    tampered = compile_explored_route_forest(blackboard)[
        "retrosynthesis_control"
    ]
    assert tampered["authoritative"] is False
    assert tampered["acceptance"] == {}

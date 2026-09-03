from __future__ import annotations

from copy import deepcopy
from typing import Any

from cascade_planner.application.action_convergence import (
    compile_action_convergence_ledger,
)
from cascade_planner.application.replan_pressure import (
    DURABLE_STAGNATION_STREAK,
    compile_replan_pressure,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.interfaces.target_solver import (
    TargetSolveConfig,
    _replan_budget_guard,
    _replan_reasons,
    _replan_signal_gate,
)


def _no_gain_records(count: int) -> list[dict[str, Any]]:
    return [
        {
            "reservation_sequence": index,
            "action_execution_id": f"execution:{index}",
            "action_id": f"action:{index}",
            "opportunity_sha256": f"{index:064x}",
            "input_revision": 7,
            "output_revision": 7,
            "same_revision_cohort": False,
            "settled": True,
            "gained": False,
        }
        for index in range(1, count + 1)
    ]


def _ledger(count: int = DURABLE_STAGNATION_STREAK) -> dict[str, Any]:
    return compile_action_convergence_ledger(
        _no_gain_records(count),
        current_graph_revision=7,
    )


def _gates(*, b1: bool) -> dict[str, Any]:
    return {
        "gates": {
            "B1_global_multi_route": b1,
            "B2_host_validated_routes": True,
            "B4_stock_boundary": True,
        }
    }


def test_text_stagnation_without_verified_history_stays_non_actionable() -> None:
    pressure = compile_replan_pressure(
        _gates(b1=False),
        material_events=("portfolio_stagnation",),
    )
    gate = _replan_signal_gate(
        _gates(b1=False),
        material_events=("portfolio_stagnation",),
        trigger_reasons=("host_validated_route_deficit",),
    )

    assert pressure["convergence_ledger_verified"] is False
    assert pressure["derived_material_events"] == []
    assert gate["accepted"] is False
    assert gate["actionable_material_events"] == []
    assert gate["ignored_material_events"] == ["portfolio_stagnation"]


def test_durable_stagnation_requires_route_diversity_deficit() -> None:
    ledger = _ledger()
    open_pressure = compile_replan_pressure(
        _gates(b1=False),
        convergence_ledger=ledger,
    )
    closed_pressure = compile_replan_pressure(
        _gates(b1=True),
        convergence_ledger=ledger,
    )
    reasons = _replan_reasons(
        _gates(b1=False),
        material_events=(),
        convergence_ledger=ledger,
    )
    gate = _replan_signal_gate(
        _gates(b1=False),
        material_events=(),
        trigger_reasons=reasons,
        convergence_ledger=ledger,
    )

    assert open_pressure["convergence_ledger_verified"] is True
    assert open_pressure["derived_material_events"] == [
        "portfolio_stagnation"
    ]
    assert closed_pressure["derived_material_events"] == []
    assert open_pressure["pressure_total"] > closed_pressure["pressure_total"]
    assert open_pressure["score"]["priority"] > closed_pressure["score"][
        "priority"
    ]
    assert reasons == ("search_stagnation_with_route_diversity_deficit",)
    assert gate["accepted"] is True
    assert gate["durable_state_events"] == ["portfolio_stagnation"]


def test_tampered_convergence_ledger_fails_closed() -> None:
    tampered = deepcopy(_ledger())
    tampered["consecutive_no_gain"] += 1

    pressure = compile_replan_pressure(
        _gates(b1=False),
        material_events=("portfolio_stagnation",),
        convergence_ledger=tampered,
    )

    assert pressure["convergence_ledger_verified"] is False
    assert pressure["consecutive_no_gain"] == 0
    assert pressure["derived_material_events"] == []


def test_material_state_events_contribute_distinct_pressure_components() -> None:
    events = (
        "critical_edge_rejected",
        "new_route_family",
        "shared_bottleneck_changed",
        "source_conflict_added",
    )
    pressure = compile_replan_pressure(
        _gates(b1=True),
        material_events=events,
    )
    reasons = _replan_reasons(_gates(b1=True), material_events=events)
    gate = _replan_signal_gate(
        _gates(b1=True),
        material_events=events,
        trigger_reasons=reasons,
    )

    assert pressure["components"] == {
        "route_diversity_deficit": 0.0,
        "durable_stagnation": 0.0,
        "stagnation_route_interaction": 0.0,
        "critical_edge_failure": 1.0,
        "shared_bottleneck_change": 1.0,
        "source_conflict": 1.0,
        "new_route_family": 1.0,
    }
    assert reasons == (
        "critical_edge_failure_pressure",
        "new_route_family_pressure",
        "shared_bottleneck_pressure",
        "source_conflict_pressure",
    )
    assert gate["accepted"] is True
    assert gate["actionable_material_events"] == list(events)


def test_pressure_ignores_unrelated_labels_and_replays_stably() -> None:
    records = _no_gain_records(DURABLE_STAGNATION_STREAK)
    first_ledger = compile_action_convergence_ledger(
        records,
        current_graph_revision=7,
    )
    replayed_ledger = compile_action_convergence_ledger(
        reversed(records),
        current_graph_revision=7,
    )
    baseline = compile_replan_pressure(
        _gates(b1=False),
        convergence_ledger=first_ledger,
    )
    labelled = compile_replan_pressure(
        {
            **_gates(b1=False),
            "target_name": "opaque",
            "dataset_id": "held-out",
            "objective_mode": "compatibility-only",
        },
        convergence_ledger=replayed_ledger,
    )

    assert first_ledger["content_sha256"] == replayed_ledger["content_sha256"]
    assert baseline == labelled


def test_accepted_pressure_cannot_bypass_exhausted_model_budget() -> None:
    reasons = _replan_reasons(
        _gates(b1=True),
        material_events=("critical_edge_rejected",),
    )
    signal_gate = _replan_signal_gate(
        _gates(b1=True),
        material_events=("critical_edge_rejected",),
        trigger_reasons=reasons,
    )
    budget_gate = _replan_budget_guard(
        model_cost={"model_invocations": 1},
        budget=RetrosynthesisRunBudget(max_model_invocations=1),
        config=TargetSolveConfig(),
    )

    assert signal_gate["accepted"] is True
    assert budget_gate["accepted"] is False
    assert "insufficient_model_invocations_for_bounded_replan" in budget_gate[
        "reasons"
    ]

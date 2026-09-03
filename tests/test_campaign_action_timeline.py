from __future__ import annotations

from cascade_planner.application.campaign_action_status import (
    compile_active_campaign_actions,
)
from cascade_planner.application.run_kernel import RunState
from cascade_planner.interfaces.campaign_action_timeline import (
    compile_campaign_action_timeline,
)


def test_active_action_projection_ignores_child_tasks_and_reads_legacy_kind() -> None:
    active = compile_active_campaign_actions(
        RunState(
            run_id="active-action-projection",
            status="running",
            in_flight_tasks={
                "campaign-action:wrapper": {
                    "input_revision": 4,
                    "resource_class": "model",
                    "metadata": {
                        "campaign_action_id": "action:codex_global_replan:abc",
                        "campaign_action_execution_id": "campaign-action:abc",
                        "producer": "codex_global_director",
                    },
                },
                "model-child": {
                    "input_revision": 4,
                    "metadata": {
                        "campaign_action_execution_id": "campaign-action:abc",
                    },
                },
            },
        )
    )

    assert len(active) == 1
    assert active[0]["kind"] == "codex_global_replan"
    assert active[0]["status"] == "running"
    assert active[0]["semantics"]["not_a_second_queue"] is True


def test_timeline_unifies_all_action_actors_and_active_kernel_state() -> None:
    kinds = (
        "chemenzy_target_expand",
        "codex_global_architecture",
        "acquire_exact_evidence",
        "reaction_validate",
        "program_review",
    )
    stages = [
        {
            "stage": f"campaign_action_unified_core_{index:02d}",
            "status": "completed",
            "completed_at": f"2026-08-10T00:00:0{index}Z",
            "detail": {
                "action": {
                    "execution_id": f"campaign-action:{index}",
                    "action_id": f"action:{kind}:{index}",
                    "kind": kind,
                    "producer": kind,
                    "resource_class": "other",
                    "input_revision": index,
                },
                "outcome": {
                    "status": "completed",
                    "output_revision": index + 1,
                    "elapsed_s": 0.25,
                },
            },
        }
        for index, kind in enumerate(kinds, start=1)
    ]
    timeline = compile_campaign_action_timeline(
        stages,
        active_actions=(
            {
                "execution_id": "campaign-action:condition",
                "action_id": "action:condition_enrich:condition",
                "kind": "condition_enrich",
                "producer": "condition_provider",
                "resource_class": "condition",
                "input_revision": 9,
            },
        ),
    )

    assert timeline["schema_version"] == "campaign_action_timeline.v1"
    assert timeline["record_count"] == 6
    assert {record["actor"] for record in timeline["records"]} >= {
        "ChemEnzy",
        "Codex",
        "Evidence",
        "Validation",
        "Program",
        "Conditions",
    }
    assert timeline["state_counts"] == {"running": 1, "succeeded": 5}
    assert len(timeline["content_sha256"]) == 64

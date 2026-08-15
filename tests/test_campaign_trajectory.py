from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from cascade_planner.application.campaign_trajectory import (
    compile_action_counts,
    compile_campaign_snapshot,
    compile_campaign_trajectory,
    compile_route_snapshot,
    compile_trajectory_bindings,
    project_campaign_trajectory_at_cutoff,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _bindings(label: str = "a") -> dict:
    return compile_trajectory_bindings(
        code={"source_bundle_sha256": label * 64},
        config={"config_sha256": label * 64},
        input_summary={"campaign_spec_sha256": "1" * 64},
        stock_oracle={"reference_sha256": "2" * 64},
        providers={"model": f"model-{label}"},
    )


def _snapshot(
    *,
    phase: str,
    event_sequence: int,
    graph_revision: int,
    wall_time_s: float,
    b1: bool,
    b4: bool,
    target_routes: int,
    host_routes: int,
    bindings: dict | None = None,
    program: bool = False,
    model_invocations: int = 0,
    native_search_invocations: int = 0,
) -> dict:
    return compile_campaign_snapshot(
        phase=phase,
        observed_at=f"2026-08-10T00:00:{event_sequence:02d}Z",
        event_sequence=event_sequence,
        graph_revision=graph_revision,
        wall_time_s=wall_time_s,
        gates={
            "gates": {
                "B0_blind_input": True,
                "B1_global_multi_route": b1,
                "B2_host_validated_routes": False,
                "B3_exact_multi_source": False,
                "B4_stock_boundary": b4,
                "B5_configured_portfolio_acceptance": False,
            },
            "counts": {"target_rooted_distinct_skeletons": target_routes},
        },
        resource_usage={
            "model": {
                "model_invocations": model_invocations,
                "visual_invocations": 0,
                "input_tokens": model_invocations * 100,
                "output_tokens": model_invocations * 10,
                "wall_time_s": float(model_invocations),
            },
            "native_search": {"committed_total": native_search_invocations},
            "tasks": {
                "dimensions": {"total": {"settled": event_sequence}},
            },
            "attempt_count": event_sequence,
            "accepted_expansion_count": target_routes,
            "settled_task_count": event_sequence,
        },
        action_counts={"total": event_sequence, "by_kind": {}},
        route_counts={
            "target_rooted_route_count": target_routes,
            "host_validated_route_count": host_routes,
        },
        pareto_archive=[{"route_id": f"route:{graph_revision}"}],
        bindings=bindings or _bindings(),
        program_milestones={"program:validation_accepted": program},
    )


def test_snapshot_v2_records_complete_anytime_state() -> None:
    snapshot = _snapshot(
        phase="action:1",
        event_sequence=7,
        graph_revision=3,
        wall_time_s=12.5,
        b1=True,
        b4=False,
        target_routes=2,
        host_routes=1,
    )

    assert snapshot["schema_version"] == "campaign_anytime_snapshot.v2"
    assert snapshot["event_sequence"] == 7
    assert snapshot["wall_time_s"] == 12.5
    assert snapshot["bindings"]["complete"] is True
    assert snapshot["milestones"]["route:first_target_rooted"] is True
    assert snapshot["milestones"]["route:first_host_validated"] is True
    assert snapshot["action_counts"]["total"] == 7
    assert snapshot["pareto_archive"][0]["route_id"] == "route:3"


def test_trajectory_reconstructs_first_times_and_binding_epochs() -> None:
    first = _snapshot(
        phase="seed",
        event_sequence=10,
        graph_revision=2,
        wall_time_s=8.25,
        b1=False,
        b4=False,
        target_routes=1,
        host_routes=0,
    )
    second = _snapshot(
        phase="resume:action",
        event_sequence=20,
        graph_revision=5,
        wall_time_s=14.5,
        b1=True,
        b4=True,
        target_routes=2,
        host_routes=1,
        bindings=_bindings("b"),
        program=True,
    )

    trajectory = compile_campaign_trajectory([second, first])

    assert [row["phase"] for row in trajectory["snapshots"]] == [
        "seed",
        "resume:action",
    ]
    assert trajectory["time_to_first"]["first_route"]["elapsed_wall_time_s"] == 8.25
    assert trajectory["time_to_first"]["B1"]["elapsed_wall_time_s"] == 14.5
    assert trajectory["time_to_first"]["B4"]["elapsed_wall_time_s"] == 14.5
    assert (
        trajectory["time_to_first"]["first_host_valid_route"]["elapsed_wall_time_s"]
        == 14.5
    )
    assert (
        trajectory["time_to_first"]["program"]["program:validation_accepted"][
            "elapsed_wall_time_s"
        ]
        == 14.5
    )
    assert len(trajectory["binding_epochs"]) == 2
    assert trajectory["continuity"]["resume_baseline_preserved"] is True


def test_trajectory_exposes_resume_wall_time_reset() -> None:
    before = _snapshot(
        phase="before",
        event_sequence=10,
        graph_revision=2,
        wall_time_s=10.0,
        b1=False,
        b4=False,
        target_routes=0,
        host_routes=0,
    )
    reset = _snapshot(
        phase="after-resume",
        event_sequence=20,
        graph_revision=3,
        wall_time_s=1.0,
        b1=False,
        b4=False,
        target_routes=0,
        host_routes=0,
    )

    trajectory = compile_campaign_trajectory([reset, before])

    assert trajectory["continuity"]["event_sequence_monotonic"] is True
    assert trajectory["continuity"]["wall_time_monotonic"] is False
    assert trajectory["continuity"]["resume_baseline_preserved"] is False


def test_trajectory_rejects_nested_binding_tampering() -> None:
    snapshot = _snapshot(
        phase="seed",
        event_sequence=1,
        graph_revision=1,
        wall_time_s=1.0,
        b1=False,
        b4=False,
        target_routes=0,
        host_routes=0,
    )
    tampered = deepcopy(snapshot)
    tampered["bindings"]["code"]["value"]["extra"] = "tampered"
    unsigned = {
        key: value for key, value in tampered.items() if key != "content_sha256"
    }
    tampered["content_sha256"] = _digest(unsigned)

    with pytest.raises(ValueError, match="bindings are invalid"):
        compile_campaign_trajectory([tampered])


def test_trajectory_reads_legacy_v1_snapshot_without_fabricating_time() -> None:
    legacy = {
        "schema_version": "campaign_anytime_snapshot.v1",
        "phase": "legacy",
        "observed_at": "2026-08-06T00:00:00Z",
        "graph_revision": 2,
        "milestones": {"B1_global_multi_route": True},
        "highest_contiguous_gate": "B1",
        "counts": {"target_rooted_distinct_skeletons": 2},
        "resource_usage": {"attempt_count": 2},
        "next_action": {},
        "semantics": {
            "one_trajectory_for_all_result_views": True,
            "milestones_do_not_select_solver_control_flow": True,
            "snapshot_grants_no_additional_scientific_authority": True,
        },
    }
    legacy["content_sha256"] = _digest(legacy)

    trajectory = compile_campaign_trajectory([legacy])

    assert trajectory["continuity"]["legacy_snapshot_count"] == 1
    assert trajectory["time_to_first"]["B1"]["elapsed_wall_time_s"] is None


def test_action_counts_deduplicate_execution_identity() -> None:
    execution = {
        "status": "completed",
        "action": {"execution_id": "exec:1", "kind": "stock_audit"},
        "outcome": {"status": "completed", "action_execution_id": "exec:1"},
    }

    counts = compile_action_counts([execution, deepcopy(execution)])

    assert counts == {
        "total": 1,
        "by_kind": {"stock_audit": 1},
        "by_status": {"completed": 1},
    }


def test_route_snapshot_keeps_compact_pareto_history() -> None:
    result = compile_route_snapshot(
        graph={
            "scientific_sha256": "3" * 64,
            "route_families": {"family:1": {}, "family:2": {}},
        },
        portfolio={
            "content_sha256": "4" * 64,
            "route_candidates": [
                {
                    "route_id": "route:1",
                    "route_family_id": "family:1",
                    "edge_ids": ["edge:1"],
                    "pareto_optimal": True,
                    "minimum_edge_proof_level": 1,
                    "stock_closure_rate": 0.0,
                    "length": 1,
                    "complete": False,
                    "selected": True,
                },
                {
                    "route_id": "route:2",
                    "route_family_id": "family:2",
                    "edge_ids": ["edge:2"],
                    "pareto_optimal": False,
                    "complete": True,
                    "selected": False,
                },
            ],
            "selected_routes": [{"route_id": "route:1"}],
        },
        gates={
            "counts": {
                "target_rooted_distinct_skeletons": 2,
                "reaction_validated_skeletons": 1,
                "evidence_closed_skeletons": 0,
                "stock_closed_skeletons": 1,
            }
        },
    )

    assert result["counts"]["route_family_count"] == 2
    assert result["counts"]["canonical_materialized_route_count"] == 0
    assert result["counts"]["strict_host_validated_route_count"] == 0
    assert result["counts"]["exact_procedure_route_count"] == 0
    assert result["counts"]["condition_complete_route_count"] == 0
    assert result["counts"]["strict_stock_closed_route_count"] == 0
    assert result["counts"]["configured_complete_route_count"] == 1
    assert result["counts"]["pareto_route_count"] == 1
    assert [row["route_id"] for row in result["pareto_archive"]] == ["route:1"]


def test_route_snapshot_requires_exact_complete_procedure_for_c4() -> None:
    procedure = {
        "procedure_record_id": "procedure:1",
        "conditions": {
            "reagents": ["base"],
            "solvent": "THF",
            "temperature_c": 20,
            "time": "2 h",
        },
        "condition_completeness": {"complete": True, "missing_required_groups": []},
    }
    graph = {
        "edges": {
            "edge:1": {
                "product_molecule_id": "mol:target",
                "precursor_molecule_ids": ["mol:leaf"],
            }
        },
        "exact_records": {"record:1": {"record_id": "record:1"}},
        "procedure_records": {"procedure:1": procedure},
    }
    route = {
        "route_id": "route:1",
        "route_family_id": "family:1",
        "edge_ids": ["edge:1"],
        "all_leaves_stock_closed": True,
        "independent_source_groups": ["publication:1"],
    }
    portfolio = {
        "proof_policy": {"minimum_independent_source_groups": 1},
        "route_candidates": [route],
        "selected_routes": [route],
        "edge_proofs": {
            "edge:1": {
                "reaction_validated": True,
                "exact_record_ids": ["record:1"],
                "procedure_record_ids": ["procedure:1"],
            }
        },
    }

    result = compile_route_snapshot(graph=graph, portfolio=portfolio, gates={})

    assert result["counts"]["strict_host_validated_route_count"] == 1
    assert result["counts"]["exact_procedure_route_count"] == 1
    assert result["counts"]["condition_complete_route_count"] == 1
    assert result["counts"]["strict_stock_closed_route_count"] == 1

    graph["procedure_records"]["procedure:1"]["condition_completeness"]["complete"] = (
        False
    )
    incomplete = compile_route_snapshot(graph=graph, portfolio=portfolio, gates={})
    assert incomplete["counts"]["exact_procedure_route_count"] == 1
    assert incomplete["counts"]["condition_complete_route_count"] == 0


def test_fixed_cutoff_projection_censors_later_milestones_and_resources() -> None:
    before = _snapshot(
        phase="before-cutoff",
        event_sequence=4,
        graph_revision=1,
        wall_time_s=8.0,
        b1=True,
        b4=False,
        target_routes=1,
        host_routes=0,
        model_invocations=1,
    )
    after = _snapshot(
        phase="after-cutoff",
        event_sequence=8,
        graph_revision=2,
        wall_time_s=12.0,
        b1=True,
        b4=True,
        target_routes=2,
        host_routes=1,
        model_invocations=2,
        native_search_invocations=1,
    )
    trajectory = compile_campaign_trajectory([after, before])

    projection = project_campaign_trajectory_at_cutoff(
        trajectory,
        cutoff={"wall_time_s": 10.0, "model_invocations": 1},
    )

    assert projection["available"] is True
    assert projection["selected_snapshot_sha256"] == before["content_sha256"]
    assert projection["gate_summary"]["B1"] is True
    assert projection["gate_summary"]["B4"] is False
    assert projection["time_to_first"]["B4"] is None
    assert projection["route_counts"]["target_rooted_route_count"] == 1
    assert projection["observed_resources"]["model_invocations"] == 1


def test_fixed_cutoff_projection_rejects_a_resume_resource_regression() -> None:
    before = _snapshot(
        phase="before-resume",
        event_sequence=4,
        graph_revision=1,
        wall_time_s=8.0,
        b1=False,
        b4=False,
        target_routes=0,
        host_routes=0,
    )
    reset = _snapshot(
        phase="after-resume",
        event_sequence=8,
        graph_revision=2,
        wall_time_s=2.0,
        b1=True,
        b4=False,
        target_routes=1,
        host_routes=0,
    )
    trajectory = compile_campaign_trajectory([reset, before])

    projection = project_campaign_trajectory_at_cutoff(
        trajectory,
        cutoff={"wall_time_s": 10.0},
    )

    assert projection["available"] is False
    assert projection["unavailable_reason"] == (
        "trajectory_resource_continuity_invalid"
    )
    assert projection["resource_regressions"] == ["wall_time_s"]

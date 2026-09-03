from __future__ import annotations

from cascade_planner.eval.result_first_strategy_projection import (
    project_launch_strategies,
)


def _row(
    case_id: str,
    *,
    b4: bool,
    native_host: bool,
    b4_s: float | None,
    terminal_s: float,
    provider_s: float,
    codex_s: float,
    phase: str = "",
) -> dict:
    return {
        "case_id": case_id,
        "gate_summary": {"B4": b4},
        "time_to_first_s": {"B4": b4_s},
        "elapsed_s": terminal_s,
        "b4_phase": phase,
        "model_cost": {"model_invocations": 1},
        "provider_search_attempts": [
            {"kind": "native", "host_admitted_solved": native_host}
        ],
        "start_cohort_latency_audit": {
            "cohort_elapsed_s": max(provider_s, codex_s),
            "chemenzy_first_proposal": {
                "elapsed_from_start_cohort_s": provider_s,
            },
            "actions": [
                {
                    "action_kind": "codex_global_architecture",
                    "completed_offset_s": codex_s,
                }
            ],
        },
    }


def test_progressive_delivery_recovers_peer_wait_without_changing_recall() -> None:
    projection = project_launch_strategies(
        [
            _row(
                "native",
                b4=True,
                native_host=True,
                b4_s=110.0,
                terminal_s=110.0,
                provider_s=40.0,
                codex_s=100.0,
                phase="chemenzy_seed",
            ),
            _row(
                "codex",
                b4=True,
                native_host=False,
                b4_s=120.0,
                terminal_s=120.0,
                provider_s=90.0,
                codex_s=100.0,
                phase="unified_core:10",
            ),
            _row(
                "failure",
                b4=False,
                native_host=False,
                b4_s=None,
                terminal_s=130.0,
                provider_s=80.0,
                codex_s=100.0,
            ),
        ]
    )

    native = projection["targets"][0]
    assert native["observed"]["post_cohort_delivery_s"] == 10.0
    assert native["concurrent_progressive"]["b4_time_s"] == 50.0
    assert native["native_first"]["initial_codex_dispatched"] is False
    assert projection["strategies"]["concurrent_progressive"]["b4_count"] == 2
    assert projection["strategies"]["native_first"]["b4_count"] == 2
    assert projection["comparison"][
        "native_first_minus_concurrent_progressive"
    ]["initial_codex_dispatch_count_delta"] == -1


def test_native_first_serializes_every_non_native_and_failed_target() -> None:
    projection = project_launch_strategies(
        [
            _row(
                "codex",
                b4=True,
                native_host=False,
                b4_s=120.0,
                terminal_s=120.0,
                provider_s=90.0,
                codex_s=100.0,
            ),
            _row(
                "failure",
                b4=False,
                native_host=False,
                b4_s=None,
                terminal_s=130.0,
                provider_s=80.0,
                codex_s=100.0,
            ),
        ]
    )

    codex, failure = projection["targets"]
    assert codex["native_first"]["b4_time_s"] == 210.0
    assert failure["native_first"]["terminal_time_s"] == 210.0
    failed_wait = projection["comparison"][
        "native_first_minus_concurrent_progressive"
    ]["failed_target_added_wait_s"]
    assert failed_wait["count"] == 1
    assert failed_wait["mean"] == 80.0
    assert projection["decision"]["recommended_strategy"] == (
        "concurrent_progressive"
    )

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from cascade_planner.application.campaign_trajectory import (
    compile_campaign_snapshot,
    compile_trajectory_bindings,
)
from cascade_planner.application.milestone_subscription import (
    acknowledge_milestone_notification,
    observe_milestone_subscription,
)
from cascade_planner.application.run_kernel import RunKernel, RunSpec


def _kernel(
    tmp_path: Path,
    run_id: str = "milestone-run",
    target_smiles: str = "CCO",
) -> RunKernel:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id=run_id,
            target_name="target",
            target_smiles=target_smiles,
            created_at="2026-08-13T00:00:00Z",
        ),
    )
    kernel.start()
    return kernel


def _snapshot(kernel: RunKernel, *, b4: bool, sequence: int = 2) -> dict:
    bindings = compile_trajectory_bindings(
        code={"source_bundle_sha256": "a" * 64},
        config={"config_sha256": "b" * 64},
        input_summary={
            "campaign_spec_sha256": kernel.spec.campaign_spec.to_dict()[
                "content_sha256"
            ]
        },
        stock_oracle={"reference_sha256": "c" * 64},
        providers={"model": "fixture"},
    )
    return compile_campaign_snapshot(
        phase="action:settled",
        observed_at=f"2026-08-13T00:00:{sequence:02d}Z",
        event_sequence=sequence,
        graph_revision=1,
        wall_time_s=2.0,
        gates={
            "gates": {
                "B0_blind_input": True,
                "B1_global_multi_route": True,
                "B2_host_validated_routes": False,
                "B3_exact_multi_source": False,
                "B4_stock_boundary": b4,
                "B5_configured_portfolio_acceptance": False,
            },
            "counts": {},
        },
        resource_usage={},
        bindings=bindings,
    )


def test_notify_only_requires_first_digest_valid_b4_and_is_idempotent(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    before = observe_milestone_subscription(
        kernel,
        snapshots=[_snapshot(kernel, b4=False)],
        policy="notify-only",
    )
    assert before["observed"] is False
    assert not (kernel.run_dir / ".autoplanner/milestone-subscriptions").exists()

    snapshot = _snapshot(kernel, b4=True, sequence=3)
    first = observe_milestone_subscription(
        kernel,
        snapshots=[snapshot],
        policy="notify-only",
    )
    repeated = observe_milestone_subscription(
        kernel,
        snapshots=[snapshot],
        policy="notify-only",
    )

    assert first == repeated
    assert first["receipt"]["notification"]["status"] == "pending"
    assert first["receipt"]["first_observation"]["snapshot_sha256"] == snapshot[
        "content_sha256"
    ]
    assert kernel.state.status == "running"
    receipts = list(
        (kernel.run_dir / ".autoplanner/milestone-subscriptions").glob("[0-9a-f]*.json")
    )
    assert len(receipts) == 1


def test_notify_and_cancel_is_concurrent_idempotent_and_kernel_owned(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    snapshot = _snapshot(kernel, b4=True, sequence=3)

    def observe(_: int) -> dict:
        return observe_milestone_subscription(
            kernel,
            snapshots=[snapshot],
            policy="notify-and-cancel",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(observe, range(4)))

    assert len({row["receipt"]["content_sha256"] for row in results}) == 1
    assert kernel.state.status == "cancelled"
    cancellation = results[0]["receipt"]["cancellation"]
    assert cancellation["requested"] is True
    assert cancellation["event_id"]
    cancel_events = [
        json.loads(line)
        for line in kernel.events_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "state_transition"
        and json.loads(line)["payload"].get("status") == "cancelled"
    ]
    assert len(cancel_events) == 1


def test_invalid_snapshot_and_wrong_campaign_binding_fail_closed(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    tampered = _snapshot(kernel, b4=True)
    tampered["milestones"]["B5_configured_portfolio_acceptance"] = True
    with pytest.raises(ValueError, match="digest"):
        observe_milestone_subscription(
            kernel,
            snapshots=[tampered],
            policy="notify-only",
        )

    other = _kernel(tmp_path / "other", run_id="other-run", target_smiles="CCN")
    with pytest.raises(ValueError, match="binding"):
        observe_milestone_subscription(
            kernel,
            snapshots=[_snapshot(other, b4=True)],
            policy="notify-only",
        )


def test_provider_solved_or_legacy_snapshot_cannot_trigger(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    result = observe_milestone_subscription(
        kernel,
        snapshots=[
            {
                "schema_version": "campaign_anytime_snapshot.v1",
                "provider": {"solved": True},
                "milestones": {"B4_stock_boundary": True},
            }
        ],
        policy="notify-only",
    )
    assert result["observed"] is False
    assert kernel.state.status == "running"


def test_failed_notification_stays_pending_and_success_is_idempotent(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    observe_milestone_subscription(
        kernel,
        snapshots=[_snapshot(kernel, b4=True)],
        policy="notify-and-cancel",
    )
    failed = acknowledge_milestone_notification(
        kernel,
        channel="webhook",
        status="failed",
        channel_receipt={"error_code": "timeout"},
    )
    delivered = acknowledge_milestone_notification(
        kernel,
        channel="webhook",
        status="delivered",
        channel_receipt={"delivery_id": "delivery-1"},
    )
    repeated = acknowledge_milestone_notification(
        kernel,
        channel="webhook",
        status="delivered",
        channel_receipt={"delivery_id": "delivery-1"},
    )

    assert failed["receipt"]["notification"]["status"] == "pending"
    assert failed["receipt"]["notification"]["attempt_count"] == 1
    assert delivered == repeated
    assert delivered["receipt"]["notification"]["status"] == "delivered"
    assert delivered["receipt"]["notification"]["attempt_count"] == 2
    assert kernel.state.status == "cancelled"
    cancel_events = [
        json.loads(line)
        for line in kernel.events_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "state_transition"
        and json.loads(line)["payload"].get("status") == "cancelled"
    ]
    assert len(cancel_events) == 1

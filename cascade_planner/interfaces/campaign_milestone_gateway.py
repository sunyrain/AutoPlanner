"""Product milestone subscription operations for CampaignGateway."""
from __future__ import annotations

from typing import Any

from cascade_planner.application.milestone_subscription import (
    acknowledge_milestone_notification,
    observe_milestone_subscription,
    snapshots_from_target_checkpoint,
)


def observe_campaign_milestone(
    service: Any,
    *,
    policy: str,
    milestone: str,
) -> dict[str, Any]:
    return observe_milestone_subscription(
        service.kernel,
        snapshots=snapshots_from_target_checkpoint(
            service.kernel.run_dir,
            expected_run_id=service.kernel.spec.run_id,
        ),
        policy=policy,
        milestone=milestone,
    )


class CampaignMilestoneGatewayMixin:
    """Keep product subscription endpoints off the bounded core facade."""

    def observe_milestone(
        self,
        run_id: str,
        *,
        policy: str,
        milestone: str = "B4_stock_boundary",
        run_dir: Any = None,
    ) -> dict[str, Any]:
        return observe_campaign_milestone(
            self._open(run_id, run_dir=run_dir),
            policy=policy,
            milestone=milestone,
        )

    def acknowledge_milestone_notification(
        self,
        run_id: str,
        *,
        channel: str,
        status: str,
        channel_receipt: Any = None,
        run_dir: Any = None,
    ) -> dict[str, Any]:
        return acknowledge_milestone_notification(
            self._open(run_id, run_dir=run_dir).kernel,
            channel=channel,
            status=status,
            channel_receipt=channel_receipt,
        )


__all__ = ["CampaignMilestoneGatewayMixin", "observe_campaign_milestone"]

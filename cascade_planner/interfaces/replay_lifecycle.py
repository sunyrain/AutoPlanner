"""Apply optional digest-bound fact lifecycle events during replay."""
from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.canonical_hypergraph import CanonicalIngestionBatch
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)


def apply_replay_lifecycle_events(
    service: RetrosynthesisCampaignService,
    pack: Mapping[str, Any],
) -> dict[str, Any] | None:
    requested = [
        dict(value)
        for value in pack.get("fact_lifecycle_events") or []
        if isinstance(value, Mapping)
    ]
    if not requested:
        return None
    graph = service.graph_store.load()
    present = set(graph.get("fact_lifecycle_events") or {})
    pending = tuple(
        value for value in requested if str(value.get("event_id") or "") not in present
    )
    if not pending:
        return {"status": "reused", "work_count": len(requested)}
    result = service.graph_store.apply(
        CanonicalIngestionBatch(fact_lifecycle_events=pending),
        idempotency_key=f"replay:lifecycle:{service.kernel.state.graph_revision}",
    )
    if any(
        value.get("kind") == "fact_lifecycle_event"
        for value in result.get("rejected") or []
    ):
        raise ValueError("replay_fact_lifecycle_event_rejected")
    return {"status": "executed", "work_count": len(pending)}


__all__ = ["apply_replay_lifecycle_events"]

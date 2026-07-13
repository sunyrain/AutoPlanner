"""Project the single V4 deficit schema into :class:`RunKernel`."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.run_kernel import Deficit, RunKernel


def publish_frontier_items(
    kernel: RunKernel,
    items: Iterable[Mapping[str, Any]],
    *,
    source_revision: int,
    idempotency_key: str,
    projection_sha256: str = "",
) -> None:
    """Replace operational work from canonical ``DeficitItem`` rows.

    Both graph exploration and proof-portfolio closeout call this boundary;
    neither owns a second queue or an independent completion counter.
    """
    deficits = []
    for raw in items:
        row = dict(raw)
        metadata = dict(row.get("metadata") or {})
        reasons = metadata.get("reasons") or [row.get("reason")]
        deficits.append(
            Deficit(
                deficit_id=str(row.get("deficit_id") or ""),
                kind=str(row.get("kind") or ""),
                source_revision=int(source_revision),
                priority=float(row.get("priority") or 0.0),
                deterministic=row.get("deterministic") is True,
                model_allowed=row.get("model_allowed") is True,
                entity_refs=tuple(
                    str(value) for value in row.get("entity_ids") or []
                ),
                reasons=tuple(
                    str(value) for value in reasons if str(value or "")
                ),
                metadata={
                    **metadata,
                    "route_family_ids": list(row.get("route_family_ids") or []),
                    "projection_sha256": str(projection_sha256 or ""),
                },
            )
        )
    kernel.replace_deficits(
        deficits,
        source_revision=int(source_revision),
        idempotency_key=idempotency_key,
    )


__all__ = ["publish_frontier_items"]

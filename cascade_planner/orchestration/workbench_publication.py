"""Publish one workbench snapshot without owning scientific state."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from cascade_planner.application.run_kernel import RunKernel


def publish_workbench_snapshot(
    kernel: RunKernel,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    ref = kernel.artifacts.put_json(
        snapshot,
        logical_name="retrosynthesis_route_workbench.json",
        producer="autoplanner.route_workbench",
    )
    run_digest = hashlib.sha256(kernel.spec.run_id.encode("utf-8")).hexdigest()
    kernel.artifacts.write_pointer(
        f"u/{run_digest[:24]}/latest",
        ref,
        metadata={
            "run_id": kernel.spec.run_id,
            "graph_revision": kernel.state.graph_revision,
            "portfolio_route_count": snapshot["portfolio"]["route_count"],
        },
    )
    kernel.index.index_artifact(
        run_id=kernel.spec.run_id,
        artifact_id="retrosynthesis_route_workbench",
        ref=ref,
        revision=kernel.state.graph_revision,
        authority_scope="display_projection_only",
    )
    return ref.to_dict()


def published_workbench_campaign_summary(
    kernel: RunKernel,
    graph: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Read final reporting metadata only when it binds the current graph."""

    run_digest = hashlib.sha256(kernel.spec.run_id.encode("utf-8")).hexdigest()
    try:
        ref, pointer = kernel.artifacts.load_pointer(f"u/{run_digest[:24]}/latest")
        if int(dict(pointer.get("metadata") or {}).get("graph_revision") or -1) != int(
            graph.get("revision") or 0
        ):
            return None
        snapshot = kernel.artifacts.read_json(ref)
    except Exception:
        return None
    if not isinstance(snapshot, Mapping):
        return None
    revision = dict(snapshot.get("revision") or {})
    if str(revision.get("graph_scientific_sha256") or "") != str(
        graph.get("scientific_sha256") or ""
    ):
        return None
    summary = dict(snapshot.get("campaign_summary") or {})
    return summary if summary.get("available") is True else None


__all__ = [
    "publish_workbench_snapshot",
    "published_workbench_campaign_summary",
]

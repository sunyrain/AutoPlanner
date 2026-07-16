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


__all__ = ["publish_workbench_snapshot"]

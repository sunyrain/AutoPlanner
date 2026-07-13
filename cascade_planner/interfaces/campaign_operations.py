"""Model-free operational projections used by the shared campaign gateway."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import time
import tracemalloc
from typing import Any

from cascade_planner.harness.v4_route_workbench import (
    render_v4_route_workbench_html,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)
from cascade_planner.runtime.artifact_store import ArtifactStore
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.runtime.run_index import RunIndex


CAMPAIGN_GATEWAY_RESULT_SCHEMA = "autoplanner_campaign_gateway_result.v1"


def benchmark_campaign(
    service: RetrosynthesisCampaignService,
    *,
    iterations: int,
) -> dict[str, Any]:
    count = max(1, min(25, int(iterations)))
    wall_samples: list[float] = []
    cpu_samples: list[float] = []
    tracemalloc.start()
    try:
        for _ in range(count):
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            service.status()
            service.graph_store.full_recompute_oracle()
            service.workbench()
            cpu_samples.append(time.process_time() - cpu_start)
            wall_samples.append(time.perf_counter() - wall_start)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": "benchmark",
        "run_id": service.kernel.spec.run_id,
        "iterations": count,
        "wall_time_s": _sample_summary(wall_samples),
        "cpu_time_s": _sample_summary(cpu_samples),
        "python_peak_bytes": peak,
        "model_invocations": 0,
        "semantics": {
            "model_free": True,
            "network_free": True,
            "measures_status_oracle_and_projection": True,
        },
    }


def export_campaign(
    service: RetrosynthesisCampaignService,
    *,
    output_dir: str | Path | None,
) -> dict[str, Any]:
    published = service.publish_workbench()
    destination = Path(
        output_dir or service.kernel.run_dir / "exports"
    ).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_path = destination / "route_workbench.json"
    delta_path = destination / "route_workbench.delta.json"
    html_path = destination / "route_workbench.html"
    _write_text(snapshot_path, _pretty_json(published["snapshot"]))
    _write_text(delta_path, _pretty_json(published["delta"]))
    _write_text(
        html_path,
        render_v4_route_workbench_html(published["snapshot"]),
    )
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": "export",
        "run_id": service.kernel.spec.run_id,
        "snapshot_ref": published["snapshot_ref"],
        "files": {
            "snapshot": str(snapshot_path),
            "delta": str(delta_path),
            "html": str(html_path),
        },
    }


def plan_artifact_gc(
    paths: RuntimePaths,
    index: RunIndex,
    *,
    minimum_age_s: float,
) -> dict[str, Any]:
    pinned: set[str] = set()
    for manifest in index.list_runs(limit=10_000):
        for row in index.artifacts_for_run(str(manifest["run_id"])):
            digest = str(dict(row.get("ref") or {}).get("sha256") or "")
            if digest:
                pinned.add(digest)
    plan = ArtifactStore(paths.artifact_store_root).garbage_collection_plan(
        pinned_digests=pinned,
        minimum_age_s=max(0.0, float(minimum_age_s)),
    )
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": "gc",
        "dry_run": True,
        "indexed_artifact_pin_count": len(pinned),
        "plan": plan,
    }


def _sample_summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": round(min(values), 6),
        "median": round(statistics.median(values), 6),
        "maximum": round(max(values), 6),
    }


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


__all__ = ["benchmark_campaign", "export_campaign", "plan_artifact_gc"]

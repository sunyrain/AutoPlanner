"""Model-free operational projections used by the shared campaign gateway."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cascade_planner.application.campaign_review_bundle import (
    compile_campaign_review_bundle,
)
from cascade_planner.harness.v4_route_workbench import (
    render_v4_route_workbench_html,
)
from cascade_planner.interfaces.campaign_benchmark import (
    benchmark_campaign,
)
from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
)
from cascade_planner.interfaces.biocatalytic_program_gc import (
    biocatalytic_program_pinned_digests,
)
from cascade_planner.interfaces.experimental_claim_gc import (
    experimental_claim_pinned_digests,
)
from cascade_planner.interfaces.mechanism_program_gc import (
    mechanism_program_pinned_digests,
)
from cascade_planner.interfaces.program_gc import program_store_pinned_digests
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)
from cascade_planner.runtime.artifact_store import ArtifactStore
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.runtime.run_index import RunIndex


def export_campaign(
    service: RetrosynthesisCampaignService,
    *,
    output_dir: str | Path | None,
) -> dict[str, Any]:
    published = service.publish_workbench()
    destination = Path(output_dir or service.kernel.run_dir / "exports").expanduser().resolve()
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
    report_path = service.kernel.run_dir / "target-only-solve-report.json"
    report = {}
    if report_path.is_file():
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
            report = dict(value) if isinstance(value, dict) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            report = {}
    review_bundle = compile_campaign_review_bundle(report)
    review_paths = {
        "review_bundle": destination / "campaign_review_bundle.json",
        "action_trace": destination / "campaign_action_trace.json",
        "failure_trace": destination / "campaign_failure_trace.json",
        "route_lineage": destination / "campaign_route_lineage.json",
        "resource_curve": destination / "campaign_resource_curve.json",
    }
    _write_text(review_paths["review_bundle"], _pretty_json(review_bundle))
    for name in ("action_trace", "failure_trace", "route_lineage", "resource_curve"):
        _write_text(
            review_paths[name],
            _pretty_json(dict(review_bundle["components"])[name]),
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
            **{name: str(path) for name, path in review_paths.items()},
        },
        "review_bundle_sha256": str(review_bundle.get("content_sha256") or ""),
    }


def plan_artifact_gc(
    paths: RuntimePaths,
    index: RunIndex,
    *,
    minimum_age_s: float,
) -> dict[str, Any]:
    indexed_pins: set[str] = set()
    for manifest in index.list_runs(limit=10_000):
        for row in index.artifacts_for_run(str(manifest["run_id"])):
            digest = str(dict(row.get("ref") or {}).get("sha256") or "")
            if digest:
                indexed_pins.add(digest)
    program_pins = program_store_pinned_digests(paths, index)
    program_pins |= biocatalytic_program_pinned_digests(paths, index)
    program_pins |= mechanism_program_pinned_digests(paths, index)
    program_pins |= experimental_claim_pinned_digests(paths, index)
    pinned = indexed_pins | program_pins
    plan = ArtifactStore(paths.artifact_store_root).garbage_collection_plan(
        pinned_digests=pinned,
        minimum_age_s=max(0.0, float(minimum_age_s)),
    )
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": "gc",
        "dry_run": True,
        "indexed_artifact_pin_count": len(indexed_pins),
        "program_store_pin_count": len(program_pins),
        "total_artifact_pin_count": len(pinned),
        "plan": plan,
    }


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


__all__ = ["benchmark_campaign", "export_campaign", "plan_artifact_gc"]

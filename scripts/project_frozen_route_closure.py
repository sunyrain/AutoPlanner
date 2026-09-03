#!/usr/bin/env python3
"""Project route-closure recomputation on frozen graphs without publishing it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.canonical_hypergraph import (
    CanonicalIngestionBatch,
    compile_canonical_hypergraph_revision,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", required=True)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    root = Path(args.panel_root).expanduser().resolve()
    rows = [project_target(root, target) for target in args.target]
    result = {
        "schema_version": "frozen_route_closure_projection.v1",
        "panel_root": str(root),
        "target_count": len(rows),
        "changed_target_count": sum(row["projection_changed"] for row in rows),
        "root_b4_gain_target_count": sum(row["root_b4_gain"] for row in rows),
        "targets": rows,
        "semantics": {
            "source_graphs_are_read_only": True,
            "projection_uses_canonical_full_recompute_oracle": True,
            "no_provider_or_model_is_invoked": True,
            "no_graph_revision_is_published": True,
            "no_change_means_followup_search_is_required_for_new_result": True,
        },
    }
    result["content_sha256"] = _digest(result)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def project_target(panel_root: Path, target_name: str) -> dict[str, Any]:
    run_dir = panel_root / "runs" / target_name
    report_path = run_dir / "target-only-solve-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    service = RetrosynthesisCampaignService.open(
        panel_root / "runtime",
        run_dir,
        artifact_store_root=panel_root / "artifacts",
        run_index_path=panel_root / "runtime" / "run_index.sqlite3",
    )
    graph = service.graph_store.load()
    projected, compile_report = compile_canonical_hypergraph_revision(
        graph,
        batch=CanonicalIngestionBatch(recompute_derived=True),
        acceptance_spec=service.kernel.spec.acceptance,
    )
    before = _route_result_counts(graph)
    after = _route_result_counts(projected)
    old_b4 = dict(report.get("gates") or {}).get("B4_stock_boundary") is True
    new_b4 = after["selected_closed_route_count"] > 0
    return {
        "target_name": target_name,
        "case_id": str(report.get("run_id") or ""),
        "source_revision": int(graph.get("revision") or 0),
        "source_scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "projected_scientific_sha256": str(
            projected.get("scientific_sha256") or ""
        ),
        "projection_changed": compile_report.get("changed") is True,
        "before": before,
        "after": after,
        "root_b4_before": old_b4,
        "root_b4_after": new_b4,
        "root_b4_gain": not old_b4 and new_b4,
        "report_path": str(report_path),
    }


def _route_result_counts(graph: Mapping[str, Any]) -> dict[str, Any]:
    routes = [
        dict(route)
        for route in dict(graph.get("route_families") or {}).values()
        if isinstance(route, Mapping)
    ]
    selected = [route for route in routes if route.get("selected") is True]
    return {
        "route_count": len(routes),
        "selected_route_count": len(selected),
        "selected_closed_route_count": sum(
            route.get("closed") is True for route in selected
        ),
        "selected_unmaterialized_hypothesis_count": sum(
            len(route.get("unmaterialized_hypothesis_ids") or [])
            for route in selected
        ),
        "selected_stock_closure_rate_max": max(
            (float(route.get("stock_closure_rate") or 0.0) for route in selected),
            default=0.0,
        ),
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

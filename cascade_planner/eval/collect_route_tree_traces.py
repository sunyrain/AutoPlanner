"""Collect route-tree search traces for planner policy/value training.

This is the data bridge for the final architecture: each row is one expansion
state with its open leaves, candidate actions, selected action, and final route
outcome. It intentionally records search behavior, not just final routes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
logging.disable(logging.CRITICAL)

from cascade_planner.cascadeboard.live_benchmark import _build_stock_checker, _load_benchmark_entries
from cascade_planner.cascadeboard.live_retro import build_live_retro_engine
from cascade_planner.cascadeboard.route_export import route_results_payload
from cascade_planner.route_tree.runtime import default_route_tree_runtime
from cascade_planner.route_tree.search import plan_with_route_tree
from cascade_planner.route_tree.trace import RouteTreeTraceCollector


TRACE_SCHEMA_VERSION = "route_tree_trace.v1"


def collect_route_tree_traces(
    *,
    bench_path: str,
    output_path: str,
    limit: int | None = None,
    max_depth: int = 6,
    n_results: int = 3,
    branch_factor: int = 12,
    expansion_budget: int = 200,
    check_stock: bool = False,
    use_route_model: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    entries = _load_benchmark_entries(bench_path, limit, shard_index=shard_index, num_shards=num_shards)
    stock_checker = _build_stock_checker(check_stock)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        retro_engine = build_live_retro_engine()
    controller = default_route_tree_runtime() if use_route_model else None
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_events = 0
    n_routes = 0
    t_all = time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, entry in enumerate(entries):
            target = entry.get("target_smiles") or entry.get("target") or ""
            if not target:
                continue
            trace = RouteTreeTraceCollector()
            t0 = time.time()
            error = None
            results = []
            try:
                results = plan_with_route_tree(
                    target=target,
                    retro_engine=retro_engine,
                    stock_checker=stock_checker,
                    max_depth=max_depth,
                    n_results=n_results,
                    branch_factor=branch_factor,
                    expansion_budget=expansion_budget,
                    constraints=None,
                    controller=controller,
                    trace_collector=trace,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.time() - t0
            payload = route_results_payload(
                target,
                results,
                objective="balanced",
                constraints=None,
                elapsed_s=elapsed,
                stock_checker=stock_checker,
            )
            route_rows = payload.get("routes") or []
            n_routes += len(route_rows)
            target_context = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "benchmark": bench_path,
                "benchmark_index": entry.get("_benchmark_index", idx),
                "target_smiles": target,
                "doi": entry.get("doi"),
                "cascade_id": entry.get("cascade_id"),
                "route_domain": entry.get("route_domain"),
                "gt_route": entry.get("gt_route", []),
                "planner_error": error,
                "elapsed_s": round(elapsed, 3),
                "n_routes": len(route_rows),
                "route_metrics": [route.get("metrics") or {} for route in route_rows],
                "route_model_active": controller is not None,
            }
            rows = trace.to_rows()
            if not rows:
                fh.write(json.dumps({**target_context, "event": None}, ensure_ascii=False) + "\n")
                n_events += 1
                continue
            for event in rows:
                fh.write(json.dumps({**target_context, "event": event}, ensure_ascii=False) + "\n")
                n_events += 1

    manifest = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "benchmark": bench_path,
        "output_path": str(out_path),
        "n_targets": len(entries),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "n_events": n_events,
        "n_routes": n_routes,
        "max_depth": max_depth,
        "n_results": n_results,
        "branch_factor": branch_factor,
        "expansion_budget": expansion_budget,
        "check_stock": check_stock,
        "route_model_active": controller is not None,
        "elapsed_s": round(time.time() - t_all, 3),
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect route-tree search traces for planner training")
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--output", default="results/shared/route_tree_traces/traces.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-results", type=int, default=3)
    ap.add_argument("--branch-factor", type=int, default=12)
    ap.add_argument("--expansion-budget", type=int, default=200)
    ap.add_argument("--check-stock", action="store_true")
    ap.add_argument("--use-route-model", action="store_true", help="Use AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER runtime if available")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    manifest = collect_route_tree_traces(
        bench_path=args.bench,
        output_path=args.output,
        limit=args.limit,
        max_depth=args.max_depth,
        n_results=args.n_results,
        branch_factor=args.branch_factor,
        expansion_budget=args.expansion_budget,
        check_stock=args.check_stock,
        use_route_model=args.use_route_model,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

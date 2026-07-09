"""Rule-post baseline for native route pools using product-audit evidence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cascade_planner.eval.product_route_feasibility_audit import build_product_route_feasibility_audit, product_audit_guard_key
from cascade_planner.eval.rerank_native_routes_with_v4_value import (
    _audit_delta,
    _audit_summary,
    _cap_native_run_for_audit,
    _gt_recovery,
    _ranked_product_metrics,
    _read_rows,
    _routes_for_target,
)


def rerank_native_routes_with_product_audit(
    *,
    native_pool: Path,
    output: Path,
    report: Path,
    benchmark: Path | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    run = json.loads(Path(native_pool).read_text(encoding="utf-8"))
    benchmark_rows = _read_rows(benchmark) if benchmark else None
    capped_run = _cap_native_run_for_audit(run, top_k=top_k)
    native_audit = build_product_route_feasibility_audit(capped_run, benchmark_rows=benchmark_rows)
    rule_targets = []
    for target, audit_target in zip(capped_run.get("targets") or [], native_audit.get("targets") or []):
        route_audits = sorted(
            audit_target.get("routes") or [],
            key=lambda row: (*product_audit_guard_key(row), int(row.get("rank") or 10**9)),
        )
        original_routes = _routes_for_target(target)
        ordered_routes = []
        for audit_route in route_audits:
            index = int(audit_route.get("rank") or 0) - 1
            if 0 <= index < len(original_routes):
                payload = dict(original_routes[index])
                payload["native_rank"] = index
                payload["rule_post_rank_metadata"] = {
                    "route_class": audit_route.get("route_class"),
                    "issues": audit_route.get("issues") or [],
                    "tags": audit_route.get("tags") or [],
                    "route_plausibility": audit_route.get("route_plausibility") or {},
                }
                ordered_routes.append(payload)
        target_out = dict(target)
        if isinstance((target.get("planner_output") or {}).get("routes"), list):
            planner = dict(target.get("planner_output") or {})
            planner["routes"] = ordered_routes
            planner["n_results"] = len(ordered_routes)
            target_out["planner_output"] = planner
        target_out["routes"] = ordered_routes
        target_out["route_count"] = len(ordered_routes)
        rule_targets.append(target_out)
    rule_run = {
        "metadata": {
            **(run.get("metadata") or {}),
            "reranker": "product_audit_rule_post",
            "source_native_pool": str(native_pool),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": {
            "targets": len(rule_targets),
            "total_routes": sum(len(_routes_for_target(target)) for target in rule_targets),
        },
        "targets": rule_targets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rule_run, indent=2, ensure_ascii=False), encoding="utf-8")
    rule_audit = build_product_route_feasibility_audit(rule_run, benchmark_rows=benchmark_rows)
    result = {
        "schema_version": "product_audit_rule_post_report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "native_pool": str(native_pool),
            "output": str(output),
            "benchmark": str(benchmark) if benchmark else None,
            "top_k": top_k,
        },
        "summary": rule_run["summary"],
        "native_audit_summary": _audit_summary(native_audit),
        "rule_audit_summary": _audit_summary(rule_audit),
        "native_ranked_product_metrics": _ranked_product_metrics(native_audit),
        "rule_ranked_product_metrics": _ranked_product_metrics(rule_audit),
        "native_gt_recovery": _gt_recovery(capped_run, benchmark_rows),
        "rule_gt_recovery": _gt_recovery(rule_run, benchmark_rows),
        "delta_rule_minus_native": _audit_delta(native_audit, rule_audit),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    return result


def _markdown(report: dict[str, Any]) -> str:
    delta = report.get("delta_rule_minus_native") or {}
    lines = [
        "# Product-Audit Rule Post-Rank Report",
        "",
        f"- Targets: `{report['summary']['targets']}`",
        f"- Routes: `{report['summary']['total_routes']}`",
        f"- Delta top3 triage: `{delta.get('top3_triage_signal_rate')}`",
        f"- Delta triage any: `{delta.get('triage_signal_rate')}`",
        "",
        "## Ranked Metrics",
        "",
        "```json",
        json.dumps(
            {
                "native": report.get("native_ranked_product_metrics"),
                "rule": report.get("rule_ranked_product_metrics"),
                "native_gt_recovery": report.get("native_gt_recovery"),
                "rule_gt_recovery": report.get("rule_gt_recovery"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Rule-post rerank native routes with product audit classes")
    ap.add_argument("--native-pool", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--benchmark")
    ap.add_argument("--top-k", type=int)
    args = ap.parse_args()
    result = rerank_native_routes_with_product_audit(
        native_pool=Path(args.native_pool),
        output=Path(args.output),
        report=Path(args.report),
        benchmark=Path(args.benchmark) if args.benchmark else None,
        top_k=args.top_k,
    )
    print(json.dumps(result["delta_rule_minus_native"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

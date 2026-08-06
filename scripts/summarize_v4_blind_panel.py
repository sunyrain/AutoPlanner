#!/usr/bin/env python3
"""Summarize a V4 blind panel without conflating route and proof metrics."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    root = Path(args.panel_root).expanduser().resolve()
    status_path = root / "panel-status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    summary = summarize_panel(status)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "panel-summary.json"
    )
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def summarize_panel(status: Mapping[str, Any]) -> dict[str, Any]:
    target_rows = {
        str(name): dict(value)
        for name, value in dict(status.get("targets") or {}).items()
        if isinstance(value, Mapping)
    }
    completed = {
        name: row for name, row in target_rows.items() if row.get("status") == "completed"
    }
    total = int(status.get("target_count") or len(target_rows))
    metric_fields = {
        "structural_route_present": "target_rooted_distinct_skeletons",
        "materialized_route_present": "materialized_skeletons",
        "host_reaction_validated": "reaction_validated_skeletons",
        "official_benchmark_stock_closed": "stock_closed_skeletons",
        "exact_source_grade": "evidence_closed_skeletons",
    }
    metric_counts = {
        metric: sum(
            int(dict(row.get("route_counts") or {}).get(field) or 0) > 0
            for row in completed.values()
        )
        for metric, field in metric_fields.items()
    }
    metric_counts["configured_proof_policy_accepted"] = sum(
        row.get("accepted_under_configured_policy") is True
        for row in completed.values()
    )
    metric_counts["within_resource_budget"] = sum(
        row.get("within_resource_budget") is True for row in completed.values()
    )
    denominator = total if total > 0 else 1
    completed_denominator = len(completed) if completed else 1
    rates = {
        metric: {
            "count": count,
            "rate_over_full_panel": round(count / denominator, 6),
            "rate_over_completed": round(count / completed_denominator, 6),
        }
        for metric, count in metric_counts.items()
    }
    costs = Counter()
    elapsed: list[float] = []
    for row in completed.values():
        for key, value in dict(row.get("model_cost") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                costs[key] += value
        value = row.get("elapsed_s")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            elapsed.append(float(value))
    failures = Counter(
        _failure_key(row)
        for row in target_rows.values()
        if row.get("status") not in {"completed", "queued", "running"}
    )
    per_target = [
        {
            "target_name": name,
            "case_id": str(row.get("case_id") or ""),
            "status": str(row.get("status") or ""),
            "retrostar_solved": (
                int(
                    dict(row.get("route_counts") or {}).get(
                        "stock_closed_skeletons"
                    )
                    or 0
                )
                > 0
            ),
            "route_counts": dict(row.get("route_counts") or {}),
            "accepted_under_configured_policy": (
                row.get("accepted_under_configured_policy") is True
            ),
            "within_resource_budget": row.get("within_resource_budget") is True,
            "elapsed_s": row.get("elapsed_s"),
            "model_cost": dict(row.get("model_cost") or {}),
            "report_path": str(row.get("report_path") or ""),
            "error": str(row.get("error") or "")[:1000],
        }
        for name, row in sorted(target_rows.items())
    ]
    body = {
        "schema_version": "v4_blind_panel_summary.v1",
        "panel": {
            "manifest_path": str(status.get("manifest_path") or ""),
            "output_root": str(status.get("output_root") or ""),
            "model": str(status.get("model") or ""),
            "execution_profile": str(status.get("execution_profile") or ""),
            "ablation": str(status.get("ablation") or ""),
            "worker_count": int(status.get("worker_count") or 0),
            "started_at": str(status.get("started_at") or ""),
            "finished_at": str(status.get("finished_at") or ""),
            "complete": status.get("complete") is True,
        },
        "counts": {
            "targets": total,
            "completed": len(completed),
            "running": sum(row.get("status") == "running" for row in target_rows.values()),
            "queued": sum(row.get("status") == "queued" for row in target_rows.values()),
            "failed_or_incomplete": total - len(completed),
        },
        "metrics": rates,
        "resource_totals": dict(sorted(costs.items())),
        "elapsed_s": _distribution(elapsed),
        "failure_categories": dict(sorted(failures.items())),
        "per_target": per_target,
        "semantics": {
            "retrostar_comparable_solved_metric": (
                "official_benchmark_stock_closed"
            ),
            "retrostar_solved_requires_target_rooted_host_admitted_structure": True,
            "proof_and_condition_metrics_are_reported_separately": True,
            "full_panel_denominator_includes_failed_and_incomplete_targets": True,
        },
    }
    body["content_sha256"] = _digest(body)
    return body


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    rows = sorted(float(value) for value in values)
    if not rows:
        return {"count": 0, "sum": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "count": len(rows),
        "sum": round(sum(rows), 3),
        "mean": round(statistics.fmean(rows), 3),
        "median": round(statistics.median(rows), 3),
        "p95": round(rows[min(len(rows) - 1, math.ceil(0.95 * len(rows)) - 1)], 3),
    }


def _failure_key(row: Mapping[str, Any]) -> str:
    status = str(row.get("status") or "unknown")
    error = str(row.get("error") or "").strip().splitlines()
    first = error[0][:160] if error else ""
    return f"{status}:{first}" if first else status


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


def _markdown(summary: Mapping[str, Any]) -> str:
    counts = dict(summary.get("counts") or {})
    lines = [
        "# V4 Blind Panel Summary",
        "",
        f"- Targets: {counts.get('targets', 0)}",
        f"- Completed: {counts.get('completed', 0)}",
        f"- Failed or incomplete: {counts.get('failed_or_incomplete', 0)}",
        "",
        "| Metric | Count | Full-panel rate | Completed rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, raw in dict(summary.get("metrics") or {}).items():
        row = dict(raw)
        lines.append(
            f"| {metric} | {row.get('count', 0)} | "
            f"{100 * float(row.get('rate_over_full_panel') or 0):.2f}% | "
            f"{100 * float(row.get('rate_over_completed') or 0):.2f}% |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

"""Aggregate external smoke summaries across benchmark shards."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


WEIGHTED_METRICS = (
    "plan_rate",
    "strict_stock_solve_any",
    "candidate_exact_reaction_in_pool",
    "candidate_gt_reactant_in_pool",
    "exact_reaction_in_route_pool",
    "gt_reactant_in_route_pool",
    "avg_time_per_target_s",
    "avg_route_count",
)

COUNT_METRICS = (
    "broad_reservoir_targets",
    "broad_reservoir_routes",
    "broad_reservoir_stock_routes",
    "broad_reservoir_metadata_stock_routes",
    "broad_reservoir_runtime_stock_routes",
    "native_payload_routes",
    "native_payload_metadata_stock_routes",
    "native_payload_runtime_stock_routes",
)


def aggregate_external_smoke_summaries(
    *,
    summaries: list[Path],
    output: Path,
    markdown: Path | None = None,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    loaded = []
    for summary_path in summaries:
        summary_path = Path(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        loaded.append(str(summary_path))
        for row in summary.get("rows") or []:
            dataset = str(row.get("dataset_label") or row.get("label") or "")
            config = str(row.get("config") or "")
            if not dataset or not config:
                continue
            n = int(row.get("n_run_targets") or row.get("n_benchmark_rows") or 0)
            if n <= 0:
                continue
            key = (dataset, config)
            bucket = groups.setdefault(
                key,
                {
                    "dataset_label": dataset,
                    "config": config,
                    "n_run_targets": 0,
                    "source_labels": [],
                    "_weighted": {metric: {"sum": 0.0, "weight": 0} for metric in WEIGHTED_METRICS},
                    "_counts": {metric: 0 for metric in COUNT_METRICS},
                },
            )
            bucket["n_run_targets"] += n
            bucket["source_labels"].append(str(row.get("label") or f"{config}_{dataset}"))
            for metric in WEIGHTED_METRICS:
                value = _safe_float(row.get(metric))
                if value is None:
                    continue
                bucket["_weighted"][metric]["sum"] += value * n
                bucket["_weighted"][metric]["weight"] += n
            for metric in COUNT_METRICS:
                bucket["_counts"][metric] += int(row.get(metric) or 0)

    rows = []
    for (_dataset, _config), bucket in sorted(groups.items()):
        row = {
            "dataset_label": bucket["dataset_label"],
            "config": bucket["config"],
            "n_run_targets": bucket["n_run_targets"],
            "source_labels": bucket["source_labels"],
        }
        for metric, acc in bucket["_weighted"].items():
            row[metric] = round(acc["sum"] / acc["weight"], 6) if acc["weight"] else None
        for metric, value in bucket["_counts"].items():
            row[metric] = value
        rows.append(row)

    report = {
        "schema_version": "external_smoke_aggregate.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summaries": loaded,
        "rows": rows,
        "paired_config_deltas": _paired_config_deltas(rows),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if markdown is not None:
        markdown = Path(markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def _paired_config_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset_label"]), {})[str(row["config"])] = row
    out = []
    for dataset, configs in sorted(by_dataset.items()):
        baseline = configs.get("C")
        if not baseline:
            continue
        for config, candidate in sorted(configs.items()):
            if config == "C":
                continue
            out.append(
                {
                    "dataset_label": dataset,
                    "baseline_config": "C",
                    "candidate_config": config,
                    "n_run_targets": min(int(baseline.get("n_run_targets") or 0), int(candidate.get("n_run_targets") or 0)),
                    "metric_deltas": {
                        metric: _delta(candidate.get(metric), baseline.get(metric))
                        for metric in WEIGHTED_METRICS
                    },
                }
            )
    return out


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# External Smoke Aggregate",
        "",
        "| Dataset | Config | n | plan | stock | route exact | route GT | avg seconds | avg routes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| {dataset} | {config} | {n} | {plan} | {stock} | {exact} | {gt} | {seconds} | {routes} |".format(
                dataset=row.get("dataset_label"),
                config=row.get("config"),
                n=row.get("n_run_targets"),
                plan=_fmt(row.get("plan_rate")),
                stock=_fmt(row.get("strict_stock_solve_any")),
                exact=_fmt(row.get("exact_reaction_in_route_pool")),
                gt=_fmt(row.get("gt_reactant_in_route_pool")),
                seconds=_fmt(row.get("avg_time_per_target_s")),
                routes=_fmt(row.get("avg_route_count")),
            )
        )
    return "\n".join(lines) + "\n"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(candidate: Any, baseline: Any) -> float | None:
    cand = _safe_float(candidate)
    base = _safe_float(baseline)
    if cand is None or base is None:
        return None
    return round(cand - base, 6)


def _fmt(value: Any) -> str:
    number = _safe_float(value)
    return "" if number is None else f"{number:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate external smoke summaries across shards")
    ap.add_argument("--summary", action="append", required=True, help="Path to an external_smoke_summary.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", default="")
    args = ap.parse_args()
    aggregate_external_smoke_summaries(
        summaries=[Path(item) for item in args.summary],
        output=Path(args.output),
        markdown=Path(args.markdown) if args.markdown else None,
    )


if __name__ == "__main__":
    main()

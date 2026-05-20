"""Controller-v2 acceptance reports for route-tree benchmark runs."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GATES: dict[str, float] = {
    "plan_rate": 0.95,
    "candidate_exact_reaction_in_pool": 0.40,
    "candidate_gt_reactant_in_pool": 0.58,
    "exact_reaction_in_route_pool": 0.26,
    "gt_reactant_in_route_pool": 0.44,
    "strict_stock_solve_any": 0.45,
    "condition_window_success_any": 0.76,
    "cascade_compatibility_success_any": 0.76,
}
MAX_GATES: dict[str, float] = {
    "avg_time_per_target_s": 30.0,
}
RESERVOIR_GATES: dict[str, float] = {
    "plan_rate": 0.95,
    "strict_stock_solve_any": 0.60,
    "candidate_gt_reactant_in_pool": 0.58,
    "exact_reaction_in_route_pool": 0.40,
    "gt_reactant_in_route_pool": 0.63,
}
RESERVOIR_MAX_GATES: dict[str, float] = {
    "avg_time_per_target_s": 30.0,
}


def build_reports(
    *,
    run_path: str,
    output_dir: str,
    label: str,
    trace_path: str | None = None,
    candidate_audit_path: str | None = None,
    baseline_path: str | None = None,
) -> dict[str, Any]:
    run = _load_json(run_path)
    baseline = _load_json(baseline_path) if baseline_path else None
    audit = _load_json(candidate_audit_path) if candidate_audit_path else None
    trace_rows = _load_trace(trace_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    closure = _closure_report(run, run_path=run_path, label=label, audit=audit)
    source = _source_policy_report(run, trace_rows=trace_rows, trace_path=trace_path, label=label)
    comparison = _comparison_report(run, label=label, baseline=baseline, baseline_path=baseline_path)
    reservoir = _reservoir_distill_report(run, trace_rows=trace_rows, label=label)

    closure_path = out_dir / "closure_report.json"
    source_path = out_dir / "source_policy_report.json"
    reservoir_path = out_dir / "reservoir_distill_report.json"
    comparison_path = out_dir / "comparison.md"
    closure_path.write_text(json.dumps(closure, indent=2, ensure_ascii=False), encoding="utf-8")
    source_path.write_text(json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8")
    reservoir_path.write_text(json.dumps(reservoir, indent=2, ensure_ascii=False), encoding="utf-8")
    comparison_path.write_text(_comparison_markdown(comparison), encoding="utf-8")
    return {
        "closure_report": str(closure_path),
        "source_policy_report": str(source_path),
        "reservoir_distill_report": str(reservoir_path),
        "comparison": str(comparison_path),
        "gate_pass": comparison["gate_pass"],
    }


def _closure_report(run: dict[str, Any], *, run_path: str, label: str, audit: dict[str, Any] | None) -> dict[str, Any]:
    targets = run.get("targets") or []
    summary = run.get("summary") or {}
    stock_counts = Counter()
    first_stock_ranks = Counter()
    depth_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in targets:
        metrics = row.get("metrics") or {}
        stock = metrics.get("strict_stock_solve_any")
        stock_counts[str(stock)] += 1
        if metrics.get("strict_stock_first_rank") is not None:
            first_stock_ranks[str(metrics.get("strict_stock_first_rank"))] += 1
        depth_key = str(row.get("depth") if row.get("depth") is not None else "unknown")
        depth_counts[depth_key]["n"] += 1
        depth_counts[depth_key]["plan"] += int(bool(metrics.get("plan")))
        depth_counts[depth_key]["strict_stock_solve_any"] += int(stock is True)
        depth_counts[depth_key]["condition_window_success_any"] += int(bool(metrics.get("condition_window_success_any")))
        depth_counts[depth_key]["cascade_compatibility_success_any"] += int(bool(metrics.get("cascade_compatibility_success_any")))

    return {
        "schema_version": "controller_v2_closure_report.v1",
        "label": label,
        "run_path": run_path,
        "n_targets": len(targets),
        "metrics": _metric_subset(summary),
        "gates": _gate_status(summary),
        "stock_counts": dict(stock_counts),
        "strict_stock_first_rank_counts": dict(first_stock_ranks),
        "runtime_time_bucket_counts": summary.get("runtime_time_bucket_counts") or {},
        "route_tree_stop_reason_counts": summary.get("route_tree_stop_reason_counts") or {},
        "route_tree_runtime_bottleneck_counts": summary.get("route_tree_runtime_bottleneck_counts") or {},
        "slow_targets_top20": summary.get("route_tree_slow_targets_top20") or [],
        "by_domain": summary.get("per_domain") or {},
        "by_depth": {
            depth: {
                key: (value / max(counts["n"], 1) if key != "n" else value)
                for key, value in counts.items()
            }
            for depth, counts in sorted(depth_counts.items())
        },
        "candidate_audit_overall": (audit or {}).get("overall") or {},
    }


def _source_policy_report(
    run: dict[str, Any],
    *,
    trace_rows: list[dict[str, Any]],
    trace_path: str | None,
    label: str,
) -> dict[str, Any]:
    summary = run.get("summary") or {}
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    source_latency = defaultdict(float)
    source_groups: dict[str, Counter[str]] = defaultdict(Counter)
    decisions = Counter()
    policy_reasons = Counter()
    fallback_reasons = Counter()
    failure_labels = Counter()
    confidence_values: list[float] = []

    for row in trace_rows:
        event = row.get("event") if isinstance(row.get("event"), dict) else row
        if not isinstance(event, dict):
            continue
        for diag in event.get("proposal_diagnostics") or []:
            allocation = diag.get("allocation") or {}
            decision = allocation.get("decision")
            if decision:
                decisions[str(decision)] += 1
            reason = allocation.get("policy_reason")
            if reason:
                policy_reasons[str(reason)] += 1
                if "fallback" in str(reason).lower() or str(reason) == "heuristic":
                    fallback_reasons[str(reason)] += 1
            fallback_reason = allocation.get("fallback_reason")
            if fallback_reason:
                fallback_reasons[str(fallback_reason)] += 1
            confidence = _safe_float(allocation.get("policy_confidence"))
            if confidence is not None:
                confidence_values.append(confidence)
            for source, stats in (diag.get("sources") or {}).items():
                key = str(source)
                source_stats[key]["calls"] += int(stats.get("calls") or 0)
                source_stats[key]["queried"] += int(bool(stats.get("queried")))
                source_stats[key]["allocated_budget"] += int(stats.get("allocated_budget") or 0)
                source_stats[key]["requested_k"] += int(stats.get("requested_k_total") or 0)
                source_stats[key]["raw_returned"] += int(stats.get("raw_returned") or 0)
                source_stats[key]["kept_returned"] += int(stats.get("kept_returned") or 0)
                source_stats[key]["final_returned"] += int(stats.get("final_returned") or 0)
                source_stats[key]["invalid_dropped"] += int(stats.get("invalid_dropped") or 0)
                source_stats[key]["ranker_dropped"] += int(stats.get("ranker_dropped") or 0)
                source_latency[key] += float(stats.get("latency_ms_total") or 0.0)
                group = str(stats.get("source_policy_group") or stats.get("source_group") or "")
                if group:
                    source_groups[group][key] += 1
            for label_value in _event_failure_labels(event):
                failure_labels[label_value] += 1

    return {
        "schema_version": "controller_v2_source_policy_report.v1",
        "label": label,
        "trace_path": trace_path,
        "run_summary_source_call_counts": summary.get("route_tree_source_call_counts") or {},
        "run_summary_source_latency_ms": summary.get("route_tree_source_latency_ms") or {},
        "trace_source_stats": {
            source: {
                **dict(counts),
                "latency_ms_total": round(source_latency[source], 3),
                "latency_ms_per_call": _safe_div(source_latency[source], counts.get("calls")),
                "final_per_requested": _safe_div(counts.get("final_returned"), counts.get("requested_k")),
            }
            for source, counts in sorted(source_stats.items())
        },
        "trace_source_groups": {
            group: dict(counts)
            for group, counts in sorted(source_groups.items())
        },
        "policy_decision_counts": dict(decisions),
        "policy_reason_counts": dict(policy_reasons),
        "fallback_reason_counts": dict(fallback_reasons),
        "avg_policy_confidence": _avg(confidence_values),
        "trace_failure_label_counts": dict(failure_labels),
    }


def _comparison_report(
    run: dict[str, Any],
    *,
    label: str,
    baseline: dict[str, Any] | None,
    baseline_path: str | None,
) -> dict[str, Any]:
    summary = run.get("summary") or {}
    baseline_summary = (baseline or {}).get("summary") or {}
    rows = []
    gate_pass = True
    for metric in list(GATES) + list(MAX_GATES):
        value = summary.get(metric)
        baseline_value = baseline_summary.get(metric)
        threshold = GATES.get(metric, MAX_GATES.get(metric))
        direction = ">=" if metric in GATES else "<="
        passed = _passes(metric, value)
        if passed is not True:
            gate_pass = False
        rows.append({
            "metric": metric,
            "value": value,
            "baseline": baseline_value,
            "delta": _delta(value, baseline_value),
            "gate": f"{direction} {threshold}",
            "pass": passed,
        })
    return {
        "schema_version": "controller_v2_comparison.v1",
        "label": label,
        "baseline_path": baseline_path,
        "rows": rows,
        "gate_pass": gate_pass,
        "promotion_notes": _promotion_notes(rows),
    }


def _reservoir_distill_report(
    run: dict[str, Any],
    *,
    trace_rows: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    summary = run.get("summary") or {}
    targets = run.get("targets") or []
    broad_counts = Counter()
    broad_routes = 0
    hybrid_required = 0
    for target in targets:
        metrics = target.get("metrics") or {}
        payload = target.get("planner_output") or {}
        broad = payload.get("broad_reservoir") or {}
        count = int(broad.get("native_route_count") or metrics.get("broad_reservoir_route_count") or 0)
        broad_routes += count
        if count > 0:
            broad_counts["targets_with_broad_reservoir"] += 1
        if count > 0 and metrics.get("strict_stock_solve_any") is True:
            hybrid_required += 1
    fallback_reasons = Counter()
    controller_confidence = []
    for row in trace_rows:
        event = row.get("event") if isinstance(row.get("event"), dict) else row
        if not isinstance(event, dict):
            continue
        for diag in event.get("proposal_diagnostics") or []:
            allocation = diag.get("allocation") or {}
            reason = str(allocation.get("policy_reason") or "")
            if "fallback" in reason:
                fallback_reasons[reason] += 1
            confidence = _safe_float(allocation.get("policy_confidence"))
            if confidence is not None:
                controller_confidence.append(confidence)
    gates = {
        metric: {
            "value": summary.get(metric),
            "gate": f">= {threshold}",
            "pass": _safe_float(summary.get(metric)) is not None and _safe_float(summary.get(metric)) >= threshold,
        }
        for metric, threshold in RESERVOIR_GATES.items()
    } | {
        metric: {
            "value": summary.get(metric),
            "gate": f"<= {threshold}",
            "pass": _safe_float(summary.get(metric)) is not None and _safe_float(summary.get(metric)) <= threshold,
        }
        for metric, threshold in RESERVOIR_MAX_GATES.items()
    }
    gate_pass = all(item["pass"] is True for item in gates.values())
    coverage_regression_note = ""
    if gate_pass and broad_routes and broad_routes >= max(len(targets), 1) * 5:
        coverage_regression_note = "Hybrid promoted: gates pass with bounded native reservoir contributing at cap-scale."
    elif gate_pass:
        coverage_regression_note = "Distilled-controller gates pass under configured reservoir acceptance checks."
    else:
        coverage_regression_note = "Do not promote: one or more reservoir acceptance gates failed or were unmeasured."
    return {
        "schema_version": "reservoir_distill_report.v1",
        "label": label,
        "metrics": {key: summary.get(key) for key in list(RESERVOIR_GATES) + list(RESERVOIR_MAX_GATES)},
        "gates": gates,
        "gate_pass": gate_pass,
        "broad_reservoir": {
            "total_native_routes": broad_routes,
            "targets_with_broad_reservoir": broad_counts.get("targets_with_broad_reservoir", 0),
            "hybrid_stock_targets": hybrid_required,
        },
        "controller": {
            "avg_confidence": _avg(controller_confidence),
            "fallback_reason_counts": dict(fallback_reasons),
        },
        "latency_breakdown": summary.get("route_tree_source_latency_ms") or {},
        "promotion_note": coverage_regression_note,
    }


def _comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Controller v2 comparison: {report.get('label')}",
        "",
        f"Gate pass: `{report.get('gate_pass')}`",
        "",
        "| Metric | Value | Baseline | Delta | Gate | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| {metric} | {value} | {baseline} | {delta} | {gate} | {passed} |".format(
                metric=row.get("metric"),
                value=_fmt(row.get("value")),
                baseline=_fmt(row.get("baseline")),
                delta=_fmt(row.get("delta")),
                gate=row.get("gate"),
                passed=row.get("pass"),
            )
        )
    notes = report.get("promotion_notes") or []
    if notes:
        lines.extend(["", "## Promotion Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def _metric_subset(summary: dict[str, Any]) -> dict[str, Any]:
    keys = list(GATES) + list(MAX_GATES) + [
        "filled_route_any",
        "avg_strict_stock_first_rank",
        "total_time_s",
        "check_stock",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def _gate_status(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        metric: {
            "value": summary.get(metric),
            "gate": f">= {threshold}",
            "pass": _passes(metric, summary.get(metric)),
        }
        for metric, threshold in GATES.items()
    } | {
        metric: {
            "value": summary.get(metric),
            "gate": f"<= {threshold}",
            "pass": _passes(metric, summary.get(metric)),
        }
        for metric, threshold in MAX_GATES.items()
    }


def _passes(metric: str, value: Any) -> bool | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    if metric in GATES:
        return numeric >= GATES[metric]
    if metric in MAX_GATES:
        return numeric <= MAX_GATES[metric]
    return None


def _promotion_notes(rows: list[dict[str, Any]]) -> list[str]:
    by_metric = {row["metric"]: row for row in rows}
    notes = []
    stock = by_metric.get("strict_stock_solve_any") or {}
    cand = by_metric.get("candidate_gt_reactant_in_pool") or {}
    runtime = by_metric.get("avg_time_per_target_s") or {}
    if stock.get("pass") is False:
        notes.append("Do not promote: strict_stock_solve_any is below the first promotion gate.")
    elif stock.get("pass") is None:
        notes.append("Do not promote: strict_stock_solve_any was not measured.")
    if cand.get("pass") is False:
        notes.append("Do not promote: candidate_gt_reactant_in_pool is below the coverage gate.")
    elif cand.get("pass") is None:
        notes.append("Do not promote: candidate_gt_reactant_in_pool was not measured.")
    if runtime.get("pass") is False:
        notes.append("Do not promote until source latency breakdown and budget caps are adjusted.")
    elif runtime.get("pass") is None:
        notes.append("Do not promote: avg_time_per_target_s was not measured.")
    if not notes:
        notes.append("All configured promotion gates pass for this run.")
    return notes


def _event_failure_labels(event: dict[str, Any]) -> list[str]:
    labels = []
    for action in event.get("candidate_actions") or []:
        labels.extend(str(value) for value in (action.get("failure_labels") or []) if value)
    for label in event.get("failure_categories") or []:
        if label:
            labels.append(str(label))
    return labels


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_trace(path: str | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(num: Any, den: Any) -> float | None:
    n = _safe_float(num)
    d = _safe_float(den)
    if n is None or d is None or d == 0:
        return None
    return round(n / d, 6)


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _delta(value: Any, baseline: Any) -> float | None:
    v = _safe_float(value)
    b = _safe_float(baseline)
    if v is None or b is None:
        return None
    return round(v - b, 6)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    numeric = _safe_float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build controller-v2 acceptance reports")
    ap.add_argument("--run", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--candidate-audit", default=None)
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()
    result = build_reports(
        run_path=args.run,
        output_dir=args.output_dir,
        label=args.label,
        trace_path=args.trace,
        candidate_audit_path=args.candidate_audit,
        baseline_path=args.baseline,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

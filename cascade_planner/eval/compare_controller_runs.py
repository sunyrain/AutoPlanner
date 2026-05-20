"""Compare two live benchmark controller runs target by target."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOOL_METRICS = (
    "plan",
    "strict_stock_solve_any",
    "condition_window_success_any",
    "cascade_compatibility_success_any",
    "skeleton_type_GT@1",
    "filled_type_GT@1",
)
RECOVERY_METRICS = (
    "candidate_exact_reaction_in_pool",
    "candidate_gt_reactant_in_pool",
    "exact_reaction_in_route_pool",
    "gt_reactant_in_route_pool",
)


def compare_controller_runs(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_targets = _target_index(baseline)
    candidate_targets = _target_index(candidate)
    indices = sorted(set(baseline_targets) | set(candidate_targets))
    rows = []
    summary = Counter()
    domain_summary: dict[str, Counter[str]] = defaultdict(Counter)
    rescue_by_category: dict[str, list[float]] = defaultdict(list)

    for idx in indices:
        base = baseline_targets.get(idx) or {}
        cand = candidate_targets.get(idx) or {}
        row = _compare_target(idx, base, cand)
        rows.append(row)
        domain = row["route_domain"]
        for category in row["change_labels"]:
            summary[category] += 1
            domain_summary[domain][category] += 1
            rescue_by_category[category].append(float(row.get("candidate_stock_rescue_retries") or 0))

    changed = [row for row in rows if row["change_labels"]]
    return {
        "schema_version": "controller_run_comparison.v1",
        "baseline_summary": _metric_summary(baseline),
        "candidate_summary": _metric_summary(candidate),
        "delta_summary": _summary_delta(baseline, candidate),
        "n_targets": len(indices),
        "change_counts": dict(summary),
        "domain_change_counts": {domain: dict(counts) for domain, counts in sorted(domain_summary.items())},
        "avg_candidate_stock_rescue_retries_by_change": {
            category: round(sum(values) / max(len(values), 1), 6)
            for category, values in sorted(rescue_by_category.items())
        },
        "changed_targets": changed,
    }


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    lines = [
        "# Controller Run Delta",
        "",
        "## Summary",
        "",
        "| metric | baseline | candidate | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    baseline = report.get("baseline_summary") or {}
    candidate = report.get("candidate_summary") or {}
    delta = report.get("delta_summary") or {}
    for metric in _summary_metrics():
        lines.append(f"| {metric} | {_fmt(baseline.get(metric))} | {_fmt(candidate.get(metric))} | {_fmt(delta.get(metric))} |")

    lines.extend(["", "## Change Counts", "", "| change | count |", "| --- | ---: |"])
    for label, count in sorted((report.get("change_counts") or {}).items(), key=lambda item: (-int(item[1]), item[0])):
        lines.append(f"| {label} | {count} |")

    lines.extend(["", "## Changed Targets", "", "| idx | domain | changes | stock | route_GT | exact_route | cand_GT | type@1 | rescue | elapsed_delta |", "| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |"])
    for row in report.get("changed_targets") or []:
        lines.append(
            "| {idx} | {domain} | {changes} | {stock} | {route_gt} | {exact_route} | {cand_gt} | {type1} | {rescue} | {elapsed} |".format(
                idx=row.get("index"),
                domain=row.get("route_domain"),
                changes=", ".join(row.get("change_labels") or []),
                stock=_arrow(row, "strict_stock_solve_any"),
                route_gt=_arrow(row, "gt_reactant_in_route_pool"),
                exact_route=_arrow(row, "exact_reaction_in_route_pool"),
                cand_gt=_arrow(row, "candidate_gt_reactant_in_pool"),
                type1=_arrow(row, "skeleton_type_GT@1"),
                rescue=int(row.get("candidate_stock_rescue_retries") or 0),
                elapsed=_fmt(row.get("elapsed_delta_s")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compare_target(idx: int, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_metrics = baseline.get("metrics") or {}
    cand_metrics = candidate.get("metrics") or {}
    base_recovery = baseline.get("route_recovery") or {}
    cand_recovery = candidate.get("route_recovery") or {}
    values: dict[str, dict[str, Any]] = {}
    labels = []
    for metric in BOOL_METRICS:
        values[metric] = {
            "baseline": _bool_or_none(base_metrics.get(metric)),
            "candidate": _bool_or_none(cand_metrics.get(metric)),
        }
    for metric in RECOVERY_METRICS:
        values[metric] = {
            "baseline": _bool_or_none(base_recovery.get(metric)),
            "candidate": _bool_or_none(cand_recovery.get(metric)),
        }
    for metric, pair in values.items():
        label = _change_label(metric, pair.get("baseline"), pair.get("candidate"))
        if label:
            labels.append(label)
    elapsed_base = _target_elapsed_s(baseline)
    elapsed_candidate = _target_elapsed_s(candidate)
    return {
        "index": idx,
        "target_smiles": candidate.get("target_smiles") or baseline.get("target_smiles") or "",
        "route_domain": candidate.get("route_domain") or baseline.get("route_domain") or "unknown",
        "values": values,
        "change_labels": labels,
        "baseline_elapsed_s": elapsed_base,
        "candidate_elapsed_s": elapsed_candidate,
        "elapsed_delta_s": _delta(elapsed_candidate, elapsed_base),
        "candidate_stock_rescue_retries": _max_key_recursive(candidate, "stock_rescue_retries"),
        "baseline_stock_rescue_retries": _max_key_recursive(baseline, "stock_rescue_retries"),
    }


def _change_label(metric: str, baseline: bool | None, candidate: bool | None) -> str:
    if baseline is candidate:
        return ""
    if baseline is False and candidate is True:
        return f"{metric}_gained"
    if baseline is True and candidate is False:
        return f"{metric}_lost"
    if baseline is None and candidate is not None:
        return f"{metric}_appeared"
    if baseline is not None and candidate is None:
        return f"{metric}_missing"
    return ""


def _target_index(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for row in run.get("targets") or []:
        try:
            out[int(row.get("index"))] = row
        except (TypeError, ValueError):
            continue
    return out


def _metric_summary(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary") or {}
    return {metric: summary.get(metric) for metric in _summary_metrics() if metric in summary}


def _summary_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base = (baseline.get("summary") or {})
    cand = (candidate.get("summary") or {})
    return {metric: _delta(cand.get(metric), base.get(metric)) for metric in _summary_metrics()}


def _summary_metrics() -> tuple[str, ...]:
    return (
        "plan_rate",
        "candidate_exact_reaction_in_pool",
        "candidate_gt_reactant_in_pool",
        "exact_reaction_in_route_pool",
        "gt_reactant_in_route_pool",
        "strict_stock_solve_any",
        "condition_window_success_any",
        "cascade_compatibility_success_any",
        "avg_time_per_target_s",
        "skeleton_type_GT@1",
        "filled_type_GT@1",
    )


def _target_elapsed_s(target: dict[str, Any]) -> float | None:
    value = target.get("elapsed_s")
    if value is None:
        value = _max_key_recursive(target, "elapsed_s")
    return _safe_float(value)


def _max_key_recursive(value: Any, key: str) -> float | int:
    hits: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if key in item:
                found = _safe_float(item.get(key))
                if found is not None:
                    hits.append(found)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    if not hits:
        return 0
    best = max(hits)
    return int(best) if float(best).is_integer() else round(best, 6)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


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
    if value is None:
        return "n/a"
    number = _safe_float(value)
    if number is None:
        return str(value)
    return f"{number:.3f}"


def _arrow(row: dict[str, Any], metric: str) -> str:
    pair = ((row.get("values") or {}).get(metric) or {})
    return f"{_short_bool(pair.get('baseline'))}->{_short_bool(pair.get('candidate'))}"


def _short_bool(value: Any) -> str:
    if value is True:
        return "1"
    if value is False:
        return "0"
    return "-"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two controller benchmark runs target by target")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    report = compare_controller_runs(baseline, candidate)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, args.output_md)
    print(json.dumps({
        "n_targets": report["n_targets"],
        "change_counts": report["change_counts"],
        "output_json": str(output_json),
        "output_md": args.output_md,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

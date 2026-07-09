"""Compare CascadeProgramSearch benchmark outputs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "chem_enzy_solved_rate",
    "cascade_solved_rate",
    "stock_closed_rate",
    "condition_conflict_free_rate",
    "enzyme_evidence_sufficient_rate",
    "top_result_exact_reaction_in_pool",
    "top_result_gt_reactant_in_pool",
    "result_exact_reaction_in_pool",
    "result_gt_reactant_in_pool",
    "candidate_exact_reaction_in_pool",
    "candidate_gt_reactant_in_pool",
    "avg_cascade_search_time_s",
]


def compare_cascade_search_runs(
    *,
    baseline_path: Path,
    run_paths: dict[str, Path],
    trace_paths: dict[str, Path] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    baseline_path = Path(baseline_path)
    trace_paths = {str(k): Path(v) for k, v in (trace_paths or {}).items()}
    metric_names = list(metrics or DEFAULT_METRICS)
    baseline = _read_json(baseline_path)
    baseline_targets = baseline.get("targets") or []
    runs: list[dict[str, Any]] = []
    changed_details: dict[str, list[dict[str, Any]]] = {}

    all_paths = {"baseline": baseline_path, **{str(k): Path(v) for k, v in run_paths.items()}}
    for name, path in all_paths.items():
        payload = _read_json(path)
        summary = _summary_with_top_result_metrics(payload)
        changed = [] if name == "baseline" else _changed_targets(baseline_targets, payload.get("targets") or [])
        changed_details[name] = changed
        runs.append(
            {
                "name": name,
                "path": str(path),
                "summary": summary,
                "selected_metrics": {
                    metric: summary.get(metric)
                    for metric in metric_names
                },
                "pair_diagnostics": _pair_trace_counts(trace_paths.get(name)),
                "route_block_value_final_rerank": _final_rerank_counts(payload.get("targets") or []),
                "product_audit_final_rerank": _final_rerank_counts(
                    payload.get("targets") or [],
                    key="product_audit_final_rerank",
                    original_rank_field="product_audit_original_rank",
                ),
                "top_route_changed_vs_baseline": len(changed),
            }
        )

    return {
        "baseline_path": str(baseline_path),
        "metrics": metric_names,
        "runs": runs,
        "changed_details": changed_details,
    }


def _summary_with_top_result_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload.get("summary") or {})
    targets = payload.get("targets") or []
    if not targets:
        return summary
    summary.setdefault(
        "top_result_exact_reaction_in_pool",
        _top_result_rate(targets, "exact_reaction_hit_count", ("recovery", "exact_reaction_in_route_pool")),
    )
    summary.setdefault(
        "top_result_gt_reactant_in_pool",
        _top_result_rate(targets, "gt_reactant_hit_count", ("recovery", "gt_reactant_in_route_pool")),
    )
    return summary


def _top_result_rate(targets: list[dict[str, Any]], program_field: str, fallback_path: tuple[str, ...]) -> float:
    total = 0
    for target in targets:
        programs = (target.get("cascade_search") or {}).get("result_programs") or []
        if programs:
            total += int(bool((programs[0] or {}).get(program_field)))
            continue
        total += int(bool(_nested_get(target, fallback_path)))
    return total / len(targets)


def write_comparison_report(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json = Path(output_json)
    output_md = Path(output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(_format_markdown(report), encoding="utf-8")


def _changed_targets(baseline_targets: list[dict[str, Any]], run_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for index, (baseline, run) in enumerate(zip(baseline_targets, run_targets)):
        baseline_route = _top_route_signature(baseline)
        run_route = _top_route_signature(run)
        if baseline_route == run_route:
            continue
        out.append(
            {
                "index": index,
                "target_smiles": run.get("target_smiles") or baseline.get("target_smiles"),
                "baseline_top_route": list(baseline_route),
                "run_top_route": list(run_route),
                "baseline_stock_closed": _nested_get(baseline, ("cascade_search", "stock_closed")),
                "run_stock_closed": _nested_get(run, ("cascade_search", "stock_closed")),
                "baseline_failures": _nested_get(baseline, ("cascade_search", "failure_categories")) or [],
                "run_failures": _nested_get(run, ("cascade_search", "failure_categories")) or [],
            }
        )
    return out


def _top_route_signature(target: dict[str, Any]) -> tuple[str, ...]:
    cascade = target.get("cascade_search") or {}
    programs = cascade.get("result_programs") or []
    if programs:
        return tuple(str(item) for item in (programs[0].get("route_rxns") or []))
    return tuple(str(item) for item in (cascade.get("route_rxns") or []))


def _pair_trace_counts(trace_path: Path | None) -> dict[str, int]:
    if trace_path is None or not Path(trace_path).exists():
        return {}
    text = Path(trace_path).read_text(encoding="utf-8", errors="replace")
    out: dict[str, int] = {}
    for key in ("cascade_pair_applicable", "cascade_pair_reward_applied"):
        for value in ("true", "false"):
            out[f"{key}_{value}"] = len(re.findall(fr'{key}": {value}', text))
    for reason in sorted(set(re.findall(r'cascade_pair_guard_reason": "([^"]+)"', text))):
        out[f"guard_reason_{reason}"] = len(
            re.findall(fr'cascade_pair_guard_reason": "{re.escape(reason)}"', text)
        )
    return out


def _final_rerank_counts(
    targets: list[dict[str, Any]],
    *,
    key: str = "route_block_value_final_rerank",
    original_rank_field: str = "original_rank",
) -> dict[str, int]:
    enabled = 0
    changed = 0
    promoted = 0
    for target in targets:
        info = (target.get("cascade_search") or {}).get(key) or {}
        if not info:
            continue
        enabled += 1
        if info.get("changed_top_route"):
            changed += 1
        programs = (target.get("cascade_search") or {}).get("result_programs") or []
        if programs and int((programs[0] or {}).get(original_rank_field) or 1) != 1:
            promoted += 1
    return {
        "enabled_targets": enabled,
        "top_route_changed": changed,
        "promoted_non_native_top": promoted,
    }


def _format_markdown(report: dict[str, Any]) -> str:
    metrics = list(report.get("metrics") or [])
    runs = list(report.get("runs") or [])
    lines = ["# Cascade Search Run Comparison", ""]
    lines.append("## Metrics")
    lines.append("")
    header = [
        "Run",
        *metrics,
        "Pair applicable",
        "Pair applied",
        "Final rerank targets",
        "Final rerank changed",
        "Product audit targets",
        "Product audit changed",
        "Top route changed",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---", *["---:" for _ in header[1:]]]) + "|")
    for run in runs:
        selected = run.get("selected_metrics") or {}
        pair = run.get("pair_diagnostics") or {}
        final_rerank = run.get("route_block_value_final_rerank") or {}
        product_audit = run.get("product_audit_final_rerank") or {}
        row = [
            str(run.get("name")),
            *[_format_value(selected.get(metric)) for metric in metrics],
            str(pair.get("cascade_pair_applicable_true", 0)),
            str(pair.get("cascade_pair_reward_applied_true", 0)),
            str(final_rerank.get("enabled_targets", 0)),
            str(final_rerank.get("top_route_changed", 0)),
            str(product_audit.get("enabled_targets", 0)),
            str(product_audit.get("top_route_changed", 0)),
            str(run.get("top_route_changed_vs_baseline", 0)),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Changed Targets")
    for name, changed in (report.get("changed_details") or {}).items():
        if name == "baseline":
            continue
        lines.append("")
        lines.append(f"### {name}")
        if not changed:
            lines.append("")
            lines.append("No top route changes vs baseline.")
            continue
        for item in changed:
            lines.append("")
            lines.append(f"- index {item['index']}: `{item.get('target_smiles')}`")
            lines.append(f"  baseline: `{'; '.join(item.get('baseline_top_route') or [])}`")
            lines.append(f"  run: `{'; '.join(item.get('run_top_route') or [])}`")
    lines.append("")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return ""
    return str(value)


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    return name.strip(), Path(path.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare CascadeProgramSearch benchmark JSON outputs")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--run", action="append", default=[], help="Named run as NAME=PATH")
    ap.add_argument("--trace", action="append", default=[], help="Named trace as NAME=PATH")
    ap.add_argument("--metric", action="append", default=[], help="Metric key from the run summary")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    run_paths = dict(_parse_named_path(value) for value in args.run)
    trace_paths = dict(_parse_named_path(value) for value in args.trace)
    report = compare_cascade_search_runs(
        baseline_path=Path(args.baseline),
        run_paths=run_paths,
        trace_paths=trace_paths,
        metrics=args.metric or None,
    )
    write_comparison_report(report, output_json=Path(args.output_json), output_md=Path(args.output_md))
    print(json.dumps({"runs": [row["name"] for row in report["runs"]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

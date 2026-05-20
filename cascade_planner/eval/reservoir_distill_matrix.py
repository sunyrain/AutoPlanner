"""Assemble reservoir-distillation A-E evaluation matrix reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cascade_planner.eval.compare_controller_runs import compare_controller_runs, write_markdown as write_delta_markdown
from cascade_planner.eval.controller_v2_reports import build_reports
from cascade_planner.eval.candidate_miss_audit import audit_candidate_misses
from cascade_planner.eval.stock_failure_audit import build_stock_failure_audit
from cascade_planner.eval.audit_stock_closed_alternatives import (
    build_stock_closed_alternative_audit,
    write_markdown as write_alternative_audit_markdown,
)


MATRIX_LABELS = {
    "A": "AutoPlanner D baseline",
    "B": "offline rank_plus_stock top-5 reservoir teacher",
    "C": "distilled controller only",
    "D": "distilled controller + bounded reservoir fallback",
    "D_FILTER": "distilled controller + bounded reservoir fallback + quality filter",
    "D_TOP10_FILTER": "distilled controller + bounded reservoir top-10 + quality filter",
    "D_APPEND": "distilled controller + append-only native reservoir diagnostic",
    "E": "optional top-10 reservoir ablation",
}


def build_reservoir_distill_matrix(
    *,
    runs: dict[str, Path],
    output_dir: Path,
    benchmark_path: Path | None = None,
    traces: dict[str, Path] | None = None,
    baseline_label: str = "A",
) -> dict[str, Any]:
    traces = traces or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    run_reports = {}
    baseline = _load_json(runs[baseline_label]) if baseline_label in runs else None
    for label, path in sorted(runs.items()):
        run_dir = output_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        run_payload = _load_json(path)
        (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        trace_output = run_dir / "run_trace.jsonl"
        trace_path = traces.get(label)
        if trace_path and trace_path.exists():
            trace_output.write_text(trace_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            trace_output.write_text("", encoding="utf-8")
        audit_path = _write_candidate_audit(
            run_payload,
            output_path=run_dir / "candidate_miss_audit.json",
            benchmark_path=benchmark_path,
        )
        stock_path = _write_stock_audit(
            run_payload,
            output_path=run_dir / "stock_failure_audit.json",
        )
        alternative_audit_path = _write_alternative_audit(
            run_payload,
            output_json=run_dir / "stock_closed_alternative_audit.json",
            output_md=run_dir / "stock_closed_alternative_audit.md",
        )
        reports = build_reports(
            run_path=str(path),
            output_dir=str(run_dir),
            label=f"{label} {MATRIX_LABELS.get(label, label)}",
            trace_path=str(trace_path) if trace_path else None,
            candidate_audit_path=str(audit_path) if audit_path else None,
            baseline_path=str(runs[baseline_label]) if baseline is not None and label != baseline_label else None,
        )
        run_reports[label] = {
            "label": MATRIX_LABELS.get(label, label),
            "run": str(run_dir / "run.json"),
            "trace": str(trace_output),
            "candidate_miss_audit": str(audit_path) if audit_path else None,
            "stock_failure_audit": str(stock_path) if stock_path else None,
            "stock_closed_alternative_audit": str(alternative_audit_path) if alternative_audit_path else None,
            **reports,
        }
    comparison_path = output_dir / "comparison.md"
    comparison = _write_matrix_comparison(
        runs=runs,
        baseline_label=baseline_label,
        output_path=comparison_path,
    )
    manifest = {
        "schema_version": "reservoir_distill_matrix.v1",
        "matrix": {label: MATRIX_LABELS.get(label, label) for label in sorted(runs)},
        "runs": {label: str(path) for label, path in sorted(runs.items())},
        "traces": {label: str(path) for label, path in sorted(traces.items())},
        "benchmark_path": str(benchmark_path) if benchmark_path else None,
        "reports": run_reports,
        "comparison": str(comparison_path),
        "comparison_summary": comparison,
        "promotion_rules": {
            "D": [
                "plan_rate, strict_stock_solve_any, avg_time_per_target_s, and route-quality review gates must pass",
                "exact/GT coverage is a reference-route recall diagnostic, not a hard route-usability gate",
                "inspect stock-closed non-GT routes with audit_stock_closed_alternatives before strong usability claims",
                "do not promote if avg_time_per_target_s exceeds the configured runtime gate",
                "relaxed effect-first runs may use a 20-30 second runtime gate instead of the original strict 16 seconds",
                "mark hybrid promoted if gates require always-on bounded native reservoir",
            ]
        },
    }
    manifest_path = output_dir / "reservoir_distill_matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _write_candidate_audit(run: dict[str, Any], *, output_path: Path, benchmark_path: Path | None) -> Path | None:
    del benchmark_path
    try:
        audit = audit_candidate_misses(run)
    except Exception as exc:
        audit = {"skipped": f"candidate_miss_audit_failed:{type(exc).__name__}"}
    output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _write_stock_audit(run: dict[str, Any], *, output_path: Path) -> Path:
    try:
        audit = build_stock_failure_audit(run)
    except Exception:
        audit = _fallback_stock_audit(run)
    output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _write_alternative_audit(run: dict[str, Any], *, output_json: Path, output_md: Path) -> Path:
    try:
        audit = build_stock_closed_alternative_audit(run, sample_size=30, non_gt_only=True)
    except Exception as exc:
        audit = {"skipped": f"stock_closed_alternative_audit_failed:{type(exc).__name__}"}
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if not audit.get("skipped"):
        write_alternative_audit_markdown(audit, output_md)
    else:
        output_md.write_text("# Stock-Closed Alternative Route Audit\n\nAudit skipped.\n", encoding="utf-8")
    return output_json


def _write_matrix_comparison(
    *,
    runs: dict[str, Path],
    baseline_label: str,
    output_path: Path,
) -> dict[str, Any]:
    baseline = _load_json(runs[baseline_label]) if baseline_label in runs else None
    if baseline is None:
        output_path.write_text("# Reservoir Distill Matrix\n\nNo baseline run was provided.\n", encoding="utf-8")
        return {"skipped": "missing_baseline"}
    lines = [
        "# Reservoir Distill Matrix",
        "",
        "| Label | Name | plan | stock | cand GT | route exact | route GT | avg seconds | avg routes | latency breakdown |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    deltas = {}
    for label, path in sorted(runs.items()):
        payload = _load_json(path)
        summary = payload.get("summary") or {}
        lines.append(
            "| {label} | {name} | {plan} | {stock} | {cand_gt} | {exact} | {gt} | {runtime} | {routes} | {latency} |".format(
                label=label,
                name=MATRIX_LABELS.get(label, label),
                plan=_fmt(summary.get("plan_rate")),
                stock=_fmt(summary.get("strict_stock_solve_any")),
                cand_gt=_fmt(summary.get("candidate_gt_reactant_in_pool")),
                exact=_fmt(summary.get("exact_reaction_in_route_pool")),
                gt=_fmt(summary.get("gt_reactant_in_route_pool")),
                runtime=_fmt(summary.get("avg_time_per_target_s")),
                routes=_fmt(summary.get("avg_route_count") or summary.get("avg_routes_per_target")),
                latency=_latency_summary(summary),
            )
        )
        if label != baseline_label:
            deltas[label] = compare_controller_runs(baseline, payload).get("delta_summary") or {}
    if deltas:
        lines.extend(["", "## Deltas vs Baseline", "", "| Label | stock | cand GT | route exact | route GT | avg seconds |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for label, delta in sorted(deltas.items()):
            lines.append(
                "| {label} | {stock} | {cand_gt} | {exact} | {gt} | {runtime} |".format(
                    label=label,
                    stock=_fmt(delta.get("strict_stock_solve_any")),
                    cand_gt=_fmt(delta.get("candidate_gt_reactant_in_pool")),
                    exact=_fmt(delta.get("exact_reaction_in_route_pool")),
                    gt=_fmt(delta.get("gt_reactant_in_route_pool")),
                    runtime=_fmt(delta.get("avg_time_per_target_s")),
                )
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for label, path in sorted(runs.items()):
        if label == baseline_label:
            continue
        write_delta_markdown(compare_controller_runs(baseline, _load_json(path)), output_path.parent / f"{label}_vs_{baseline_label}.md")
    return {"baseline_label": baseline_label, "deltas": deltas}


def _fallback_stock_audit(run: dict[str, Any]) -> dict[str, Any]:
    targets = run.get("targets") or []
    failures = []
    for row in targets:
        metrics = row.get("metrics") or {}
        if metrics.get("strict_stock_solve_any") is False:
            failures.append({
                "index": row.get("index"),
                "target_smiles": row.get("target_smiles"),
                "candidate_gt_reactant_in_pool": (row.get("route_recovery") or {}).get("candidate_gt_reactant_in_pool"),
                "exact_reaction_in_route_pool": (row.get("route_recovery") or {}).get("exact_reaction_in_route_pool"),
            })
    return {"schema_version": "stock_failure_audit.fallback.v1", "n_failures": len(failures), "failures": failures}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _latency_summary(summary: dict[str, Any]) -> str:
    latency = summary.get("route_tree_source_latency_ms") or summary.get("source_latency_ms") or {}
    if not isinstance(latency, dict) or not latency:
        return "n/a"
    return ", ".join(f"{source}:{_fmt(value)}" for source, value in sorted(latency.items()))


def _parse_labeled_paths(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        out[label.strip()] = Path(path.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble reservoir-distillation A-E matrix reports")
    ap.add_argument("--run", action="append", required=True, help="LABEL=run.json, e.g. A=baseline.json")
    ap.add_argument("--trace", action="append", default=[], help="LABEL=run_trace.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--baseline-label", default="A")
    args = ap.parse_args()
    build_reservoir_distill_matrix(
        runs=_parse_labeled_paths(args.run),
        traces=_parse_labeled_paths(args.trace),
        output_dir=Path(args.output_dir),
        benchmark_path=Path(args.benchmark) if args.benchmark else None,
        baseline_label=args.baseline_label,
    )


if __name__ == "__main__":
    main()

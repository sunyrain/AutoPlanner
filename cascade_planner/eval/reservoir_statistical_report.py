"""Bootstrap confidence intervals for reservoir-distilled controller reports."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from cascade_planner.cascadeboard.live_benchmark import summarize_target_results


METRICS = (
    "plan_rate",
    "strict_stock_solve_any",
    "candidate_gt_reactant_in_pool",
    "candidate_exact_reaction_in_pool",
    "exact_reaction_in_route_pool",
    "gt_reactant_in_route_pool",
    "avg_time_per_target_s",
    "avg_route_count",
)
BOOL_METRIC_SOURCES = {
    "plan_rate": ("metrics", "plan"),
    "strict_stock_solve_any": ("metrics", "strict_stock_solve_any"),
    "candidate_gt_reactant_in_pool": ("route_recovery", "candidate_gt_reactant_in_pool"),
    "candidate_exact_reaction_in_pool": ("route_recovery", "candidate_exact_reaction_in_pool"),
    "exact_reaction_in_route_pool": ("route_recovery", "exact_reaction_in_route_pool"),
    "gt_reactant_in_route_pool": ("route_recovery", "gt_reactant_in_route_pool"),
}
FULL100_LABELS = ("A", "B", "C", "D", "D_APPEND")
EXTERNAL_LABELS = ("C", "D", "D_APPEND")


def build_statistical_report(
    *,
    acceptance_dir: Path,
    external_summary: Path,
    output_json: Path,
    output_md: Path,
    iterations: int = 1000,
    seed: int = 13,
) -> dict[str, Any]:
    acceptance_dir = Path(acceptance_dir)
    external_summary = Path(external_summary)
    full100 = {
        label: _run_bootstrap(acceptance_dir / label / "run.json", iterations=iterations, seed=seed + idx)
        for idx, label in enumerate(FULL100_LABELS)
    }
    external_payload = _load_json(external_summary)
    external_output_dir = Path(external_payload.get("output_dir") or external_summary.parent)
    external_rows = {}
    for row in external_payload.get("rows") or []:
        label = str(row.get("label") or "")
        if not label:
            continue
        external_rows[label] = _run_bootstrap(external_output_dir / label / "run.json", iterations=iterations, seed=seed + len(external_rows) + 100)
    deltas = {
        "full100": _paired_delta_group(acceptance_dir, labels=FULL100_LABELS, iterations=iterations, seed=seed + 1000),
        "external": _external_delta_groups(external_payload, external_output_dir, iterations=iterations, seed=seed + 2000),
    }
    coverage = {
        "full100": all(row.get("exists") and row.get("n_targets", 0) >= 100 for row in full100.values()),
        "external": _external_coverage(external_payload, external_rows),
    }
    report = {
        "schema_version": "reservoir_statistical_report.v1",
        "acceptance_dir": str(acceptance_dir),
        "external_summary": str(external_summary),
        "bootstrap_iterations": int(iterations),
        "seed": int(seed),
        "metrics": list(METRICS),
        "coverage": coverage,
        "ready": bool(coverage["full100"] and coverage["external"] and int(iterations) >= 1000),
        "full100": full100,
        "external": external_rows,
        "paired_deltas": deltas,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(_markdown(report), encoding="utf-8")
    return report


def _run_bootstrap(path: Path, *, iterations: int, seed: int) -> dict[str, Any]:
    payload = _load_json(path)
    targets = payload.get("targets") or []
    if not targets:
        return {"path": str(path), "exists": path.exists(), "n_targets": 0, "metrics": {}}
    rng = random.Random(seed)
    sampled = {metric: [] for metric in METRICS}
    for _ in range(int(iterations)):
        rows = [targets[rng.randrange(len(targets))] for _ in targets]
        summary = summarize_target_results(rows, check_stock=True)
        for metric in METRICS:
            value = summary.get(metric)
            if isinstance(value, (int, float)):
                sampled[metric].append(float(value))
    point = summarize_target_results(targets, check_stock=True)
    return {
        "path": str(path),
        "exists": path.exists(),
        "n_targets": len(targets),
        "metrics": {
            metric: {
                "point": _round(point.get(metric)),
                "ci95_low": _round(_percentile(values, 0.025)),
                "ci95_high": _round(_percentile(values, 0.975)),
            }
            for metric, values in sampled.items()
            if values
        },
    }


def _paired_delta_group(base_dir: Path, *, labels: tuple[str, ...], iterations: int, seed: int) -> dict[str, Any]:
    out = {}
    if "C" not in labels:
        return out
    for label in labels:
        if label == "C":
            continue
        row = _paired_delta_bootstrap(
            base_dir / "C" / "run.json",
            base_dir / label / "run.json",
            iterations=iterations,
            seed=seed + len(out),
        )
        if row.get("ready"):
            out[f"C_vs_{label}"] = row
    return out


def _external_delta_groups(external_payload: dict[str, Any], output_dir: Path, *, iterations: int, seed: int) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, str]] = {}
    for row in external_payload.get("rows") or []:
        dataset = str(row.get("dataset_label") or "")
        config = str(row.get("config") or "")
        label = str(row.get("label") or "")
        if dataset and config and label:
            by_dataset.setdefault(dataset, {})[config] = label
    out = {}
    for dataset, configs in sorted(by_dataset.items()):
        baseline = configs.get("C")
        if not baseline:
            continue
        for candidate in ("D", "D_APPEND"):
            candidate_label = configs.get(candidate)
            if not candidate_label:
                continue
            row = _paired_delta_bootstrap(
                output_dir / baseline / "run.json",
                output_dir / candidate_label / "run.json",
                iterations=iterations,
                seed=seed + len(out),
            )
            if row.get("ready"):
                out[f"{dataset}:C_vs_{candidate}"] = row
    return out


def _paired_delta_bootstrap(baseline_path: Path, candidate_path: Path, *, iterations: int, seed: int) -> dict[str, Any]:
    baseline = _load_json(baseline_path).get("targets") or []
    candidate = _load_json(candidate_path).get("targets") or []
    n = min(len(baseline), len(candidate))
    if n <= 0:
        return {"ready": False, "baseline": str(baseline_path), "candidate": str(candidate_path), "n_targets": n}
    baseline = baseline[:n]
    candidate = candidate[:n]
    rng = random.Random(seed)
    out = {}
    for metric in METRICS:
        base_values = _target_metric_values(baseline, metric)
        cand_values = _target_metric_values(candidate, metric)
        if len(base_values) < n or len(cand_values) < n:
            continue
        point = mean(cand_values) - mean(base_values)
        samples = []
        for _ in range(int(iterations)):
            indices = [rng.randrange(n) for _ in range(n)]
            samples.append(mean(cand_values[i] for i in indices) - mean(base_values[i] for i in indices))
        out[metric] = {
            "point_delta": _round(point),
            "ci95_low": _round(_percentile(samples, 0.025)),
            "ci95_high": _round(_percentile(samples, 0.975)),
        }
    return {
        "ready": bool(out),
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "n_targets": n,
        "metrics": out,
    }


def _target_metric_values(targets: list[dict[str, Any]], metric: str) -> list[float]:
    values = []
    for target in targets:
        if metric in BOOL_METRIC_SOURCES:
            section, key = BOOL_METRIC_SOURCES[metric]
            value = (target.get(section) or {}).get(key)
            if value is None:
                values.append(0.0)
            else:
                values.append(float(bool(value)))
        elif metric == "avg_time_per_target_s":
            values.append(float((target.get("planner_output") or {}).get("time_s") or 0.0))
        elif metric == "avg_route_count":
            values.append(float(len((target.get("planner_output") or {}).get("routes") or [])))
    return values


def _external_coverage(external_payload: dict[str, Any], external_rows: dict[str, dict[str, Any]]) -> bool:
    datasets = {}
    for row in external_payload.get("rows") or []:
        dataset = str(row.get("dataset_label") or "")
        label = str(row.get("label") or "")
        if not dataset or not label:
            continue
        n = int((external_rows.get(label) or {}).get("n_targets") or 0)
        datasets[dataset] = max(datasets.get(dataset, 0), n)
    required = {"paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like"}
    return required.issubset(datasets) and all(datasets[name] >= 10 for name in required)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reservoir Statistical Report",
        "",
        f"Ready: `{report.get('ready')}`",
        f"Bootstrap iterations: `{report.get('bootstrap_iterations')}`",
        "",
        "## Coverage",
        "",
        "| Scope | Pass |",
        "| --- | ---: |",
    ]
    for key, value in (report.get("coverage") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Full100 Deltas", "", "| Pair | Metric | Delta | 95% CI |", "| --- | --- | ---: | --- |"])
    for pair, row in ((report.get("paired_deltas") or {}).get("full100") or {}).items():
        for metric in ("strict_stock_solve_any", "exact_reaction_in_route_pool", "gt_reactant_in_route_pool", "avg_time_per_target_s"):
            stats = (row.get("metrics") or {}).get(metric) or {}
            lines.append(f"| `{pair}` | `{metric}` | {_fmt(stats.get('point_delta'))} | [{_fmt(stats.get('ci95_low'))}, {_fmt(stats.get('ci95_high'))}] |")
    lines.extend(["", "## External Deltas", "", "| Pair | Metric | Delta | 95% CI |", "| --- | --- | ---: | --- |"])
    for pair, row in ((report.get("paired_deltas") or {}).get("external") or {}).items():
        for metric in ("strict_stock_solve_any", "exact_reaction_in_route_pool", "gt_reactant_in_route_pool", "avg_time_per_target_s"):
            stats = (row.get("metrics") or {}).get(metric) or {}
            lines.append(f"| `{pair}` | `{metric}` | {_fmt(stats.get('point_delta'))} | [{_fmt(stats.get('ci95_low'))}, {_fmt(stats.get('ci95_high'))}] |")
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[idx]


def _round(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), 6)


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build bootstrap CIs for reservoir-distilled controller reports")
    ap.add_argument("--acceptance-dir", default="results/shared/reservoir_distill_20260513/full100_acceptance_real_v2")
    ap.add_argument(
        "--external-summary",
        default="results/shared/reservoir_distill_20260513/external_publication_matrix_limit10_ua_20260514/external_smoke_summary.json",
    )
    ap.add_argument("--output-json", default="results/shared/reservoir_distill_20260513/reservoir_statistical_report_20260514.json")
    ap.add_argument("--output-md", default="results/shared/reservoir_distill_20260513/reservoir_statistical_report_20260514.md")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    report = build_statistical_report(
        acceptance_dir=Path(args.acceptance_dir),
        external_summary=Path(args.external_summary),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        iterations=args.iterations,
        seed=args.seed,
    )
    print(json.dumps({"ready": report.get("ready"), "output_json": args.output_json, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()

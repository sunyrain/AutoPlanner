"""Compare ChemEnzy external-baseline JSON against AutoPlanner route-tree output."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import canonical_reaction


def compare_baselines(
    *,
    benchmark_path: Path,
    chem_enzy_path: Path,
    route_tree_path: Path | None = None,
) -> dict[str, Any]:
    benchmark = _rows(json.loads(benchmark_path.read_text(encoding="utf-8")))
    chem = json.loads(chem_enzy_path.read_text(encoding="utf-8"))
    route_tree = json.loads(route_tree_path.read_text(encoding="utf-8")) if route_tree_path else None
    benchmark_by_target = {row.get("target_smiles"): row for row in benchmark if row.get("target_smiles")}
    benchmark_targets = set(benchmark_by_target)
    report = {
        "benchmark": {
            "path": str(benchmark_path),
            "n_targets": len(benchmark),
        },
        "chem_enzy": _summarize_chem_enzy(chem, benchmark_by_target, benchmark_targets),
        "route_tree": _summarize_route_tree(route_tree, benchmark_by_target, benchmark_targets) if route_tree else None,
    }
    return report


def _summarize_chem_enzy(
    payload: dict[str, Any],
    benchmark_by_target: dict[str, dict[str, Any]],
    benchmark_targets: set[str],
) -> dict[str, Any]:
    rows = [row for row in payload.get("targets") or [] if row.get("target_smiles") in benchmark_targets]
    solved = 0
    route_counts = []
    enzymatic = 0
    times = []
    exact_overlap = 0
    partial_overlap = 0
    condition_recovery = 0
    failures = Counter()
    for row in rows:
        target = row.get("target_smiles")
        routes = row.get("routes") or []
        route_counts.append(len(routes))
        solved += int(bool(row.get("solved")))
        failures.update(failure.get("category") for failure in row.get("failures") or [] if failure.get("category"))
        gt_rxns = _gt_reactions(benchmark_by_target.get(target) or {})
        target_exact = False
        target_partial = False
        target_condition = False
        target_enzymatic = False
        for route in routes:
            if route.get("search_time_s") is not None:
                times.append(float(route["search_time_s"]))
            steps = route.get("steps") or []
            route_rxns = {canonical_reaction(step.get("rxn_smiles") or "") or step.get("rxn_smiles") for step in steps}
            if gt_rxns and gt_rxns.issubset(route_rxns):
                target_exact = True
            if gt_rxns and route_rxns & gt_rxns:
                target_partial = True
            if any(step.get("condition_predictions") for step in steps):
                target_condition = True
            if route.get("enzymatic_step_present") or any(step.get("enzyme_ec_annotations") for step in steps):
                target_enzymatic = True
        exact_overlap += int(target_exact)
        partial_overlap += int(target_partial)
        condition_recovery += int(target_condition)
        enzymatic += int(target_enzymatic)
    n = len(rows)
    return {
        "n_targets": n,
        "solved_rate": solved / n if n else None,
        "avg_route_count": sum(route_counts) / n if n else None,
        "enzymatic_step_presence_rate": enzymatic / n if n else None,
        "exact_gt_step_overlap_rate": exact_overlap / n if n else None,
        "partial_gt_step_overlap_rate": partial_overlap / n if n else None,
        "condition_field_recovery_rate": condition_recovery / n if n else None,
        "avg_search_time_s": sum(times) / len(times) if times else None,
        "failure_categories": dict(failures),
    }


def _summarize_route_tree(
    payload: dict[str, Any],
    benchmark_by_target: dict[str, dict[str, Any]],
    benchmark_targets: set[str],
) -> dict[str, Any]:
    rows = payload.get("targets") if isinstance(payload, dict) else None
    if rows is None and isinstance(payload, list):
        rows = payload
    rows = [row for row in rows or [] if row.get("target_smiles") in benchmark_targets]
    solved = 0
    route_counts = []
    exact_overlap = 0
    partial_overlap = 0
    condition_success = 0
    times = []
    failures = Counter()
    for row in rows:
        target = row.get("target_smiles")
        metrics = row.get("metrics") or {}
        planner_output = row.get("planner_output") or {}
        routes = planner_output.get("routes") or []
        route_counts.append(len(routes))
        solved += int(bool(metrics.get("plan") or planner_output.get("n_results")))
        condition_success += int(bool(metrics.get("condition_window_success_any")))
        if planner_output.get("time_s") is not None:
            times.append(float(planner_output["time_s"]))
        recovery = row.get("route_recovery") or {}
        if recovery.get("exact_route_reaction_match_any") or recovery.get("exact_reaction_in_route_pool"):
            exact_overlap += 1
        if recovery.get("gt_reactant_in_route_pool") or recovery.get("candidate_gt_reactant_in_pool"):
            partial_overlap += 1
        failures.update(recovery.get("recovery_bottleneck_labels") or [])
        if not recovery and routes:
            gt_rxns = _gt_reactions(benchmark_by_target.get(target) or {})
            route_rxns = {
                canonical_reaction(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
                or step.get("reaction_smiles")
                or step.get("rxn_smiles")
                for route in routes
                for step in route.get("steps") or []
            }
            exact_overlap += int(bool(gt_rxns and gt_rxns.issubset(route_rxns)))
            partial_overlap += int(bool(gt_rxns and route_rxns & gt_rxns))
    n = len(rows)
    return {
        "n_targets": n,
        "solved_rate": solved / n if n else None,
        "avg_route_count": sum(route_counts) / n if n else None,
        "exact_or_pool_gt_overlap_rate": exact_overlap / n if n else None,
        "partial_or_reactant_overlap_rate": partial_overlap / n if n else None,
        "condition_window_success_rate": condition_success / n if n else None,
        "avg_search_time_s": sum(times) / len(times) if times else None,
        "failure_categories": dict(failures),
    }


def _gt_reactions(row: dict[str, Any]) -> set[str]:
    out = set()
    for step in row.get("gt_route") or []:
        rxn = step.get("rxn_smiles")
        if rxn:
            out.add(canonical_reaction(rxn) or rxn)
    return out


def _rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    raise ValueError("unsupported row format")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare ChemEnzy baseline with route-tree baseline")
    ap.add_argument("--benchmark", default="data/benchmark_cascade_gold_smoke_v1.json")
    ap.add_argument("--chem-enzy", default="results/shared/chem_enzy_baseline/smoke.json")
    ap.add_argument("--route-tree", default=None)
    ap.add_argument("--output", default="results/shared/chem_enzy_baseline/comparison.json")
    args = ap.parse_args()
    report = compare_baselines(
        benchmark_path=Path(args.benchmark),
        chem_enzy_path=Path(args.chem_enzy),
        route_tree_path=Path(args.route_tree) if args.route_tree else None,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out)}, indent=2))


if __name__ == "__main__":
    main()

"""PaRoutes-style top-k proxy metrics for ordered route pools.

This is not the official ChemEnzy/PaRoutes evaluator.  It uses the benchmark
GT reactions available in our local JSON rows and reports exact reaction-set,
exact sequence, leaf overlap, and shorter-route proxy metrics for top-k routes.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles
from cascade_planner.eval.rerank_native_routes_with_v4_value import _read_rows, _routes_for_target


TOP_KS = (1, 5, 10)


def evaluate_paroutes_topk_proxy(
    *,
    run_path: Path,
    benchmark: Path,
    output_json: Path,
    output_md: Path | None = None,
) -> dict[str, Any]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    bench_rows = _read_rows(benchmark)
    bench_by_target = {str(row.get("target_smiles") or ""): row for row in bench_rows}
    rows = []
    for idx, target in enumerate(run.get("targets") or []):
        target_smiles = str(target.get("target_smiles") or "")
        bench = bench_by_target.get(target_smiles) or (bench_rows[idx] if idx < len(bench_rows) else {})
        rows.append(_target_metrics(target, bench))

    summary = _summary(rows)
    report = {
        "schema_version": "paroutes_topk_proxy.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "run": str(run_path),
            "benchmark": str(benchmark),
            "note": "Proxy metrics from flattened GT reactions; not the official PaRoutes tree evaluator.",
        },
        "summary": summary,
        "targets": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown(report), encoding="utf-8")
    return report


def _target_metrics(target: dict[str, Any], bench: dict[str, Any]) -> dict[str, Any]:
    gt_rxns = _gt_reactions(bench)
    gt_counter = Counter(gt_rxns)
    gt_leaves = _route_leaves_from_reactions(gt_rxns)
    routes = _routes_for_target(target)
    route_rows = []
    for rank, route in enumerate(routes, start=1):
        pred_rxns = _route_reactions(route)
        pred_counter = Counter(pred_rxns)
        pred_leaves = _route_leaves(route, pred_rxns)
        exact_set = bool(gt_counter and pred_counter == gt_counter)
        exact_sequence = bool(gt_rxns and (pred_rxns == gt_rxns or pred_rxns == list(reversed(gt_rxns))))
        overlap = _leaf_overlap(pred_leaves, gt_leaves)
        route_rows.append(
            {
                "rank": rank,
                "n_steps": len(pred_rxns),
                "exact_reaction_set": exact_set,
                "exact_reaction_sequence": exact_sequence,
                "leaf_overlap": overlap,
                "shorter_or_equal": bool(gt_rxns and len(pred_rxns) <= len(gt_rxns)),
            }
        )
    topk = {}
    for k in TOP_KS:
        subset = route_rows[:k]
        topk[str(k)] = {
            "route_present": bool(subset),
            "exact_reaction_set_hit": any(row["exact_reaction_set"] for row in subset),
            "exact_reaction_sequence_hit": any(row["exact_reaction_sequence"] for row in subset),
            "shorter_or_equal_hit": any(row["shorter_or_equal"] for row in subset),
            "best_leaf_overlap": max((float(row["leaf_overlap"] or 0.0) for row in subset), default=0.0),
        }
    return {
        "target_id": target.get("cascade_id") or target.get("target_id") or bench.get("cascade_id") or target.get("index"),
        "target_smiles": target.get("target_smiles") or bench.get("target_smiles"),
        "n_routes": len(routes),
        "gt_n_steps": len(gt_rxns),
        "gt_n_leaves": len(gt_leaves),
        "topk": topk,
        "routes": route_rows[:10],
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {"n_targets": n, "targets_with_routes": sum(1 for row in rows if row["n_routes"] > 0)}
    for k in TOP_KS:
        key = str(k)
        out[f"top{k}_route_present_rate"] = _rate(rows, lambda row: row["topk"][key]["route_present"])
        out[f"top{k}_exact_reaction_set_rate"] = _rate(rows, lambda row: row["topk"][key]["exact_reaction_set_hit"])
        out[f"top{k}_exact_reaction_sequence_rate"] = _rate(rows, lambda row: row["topk"][key]["exact_reaction_sequence_hit"])
        out[f"top{k}_shorter_or_equal_rate"] = _rate(rows, lambda row: row["topk"][key]["shorter_or_equal_hit"])
        out[f"top{k}_avg_best_leaf_overlap"] = round(
            sum(float(row["topk"][key]["best_leaf_overlap"] or 0.0) for row in rows) / max(n, 1),
            6,
        )
    return out


def _gt_reactions(bench: dict[str, Any]) -> list[str]:
    return [
        canonical_reaction(step.get("rxn_smiles"))
        for step in bench.get("gt_route") or []
        if canonical_reaction(step.get("rxn_smiles"))
    ]


def _route_reactions(route: dict[str, Any]) -> list[str]:
    out = []
    for step in route.get("steps") or []:
        rxn = step.get("reaction_smiles") or step.get("rxn_smiles")
        key = canonical_reaction(rxn)
        if key:
            out.append(key)
    return out


def _route_leaves(route: dict[str, Any], pred_rxns: list[str]) -> set[str]:
    metrics = route.get("metrics") or {}
    terminals = {canonical_smiles(smi) for smi in metrics.get("terminal_reactants") or [] if smi}
    if terminals:
        return {smi for smi in terminals if smi}
    products = set()
    reactants = set()
    for step in route.get("steps") or []:
        product = step.get("product") or step.get("product_smiles")
        if product:
            products.add(canonical_smiles(product))
        if step.get("main_reactant"):
            reactants.add(canonical_smiles(step.get("main_reactant")))
        for smi in step.get("aux_reactants") or []:
            reactants.add(canonical_smiles(smi))
        for smi in step.get("reactant_smiles") or []:
            reactants.add(canonical_smiles(smi))
    if reactants:
        return {smi for smi in reactants - products if smi}
    return _route_leaves_from_reactions(pred_rxns)


def _route_leaves_from_reactions(rxns: list[str]) -> set[str]:
    reactants = set()
    products = set()
    for rxn in rxns:
        if ">>" not in rxn:
            continue
        lhs, rhs = rxn.split(">>", 1)
        reactants.update(canonical_side(lhs))
        products.update(canonical_side(rhs))
    return {smi for smi in reactants - products if smi}


def _leaf_overlap(pred_leaves: set[str], gt_leaves: set[str]) -> float:
    if not gt_leaves:
        return 0.0
    return round(len(pred_leaves & gt_leaves) / len(gt_leaves), 6)


def _rate(rows: list[dict[str, Any]], fn) -> float:
    return round(sum(1 for row in rows if fn(row)) / max(len(rows), 1), 6)


def _markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# PaRoutes Top-k Proxy Report",
        "",
        f"- Targets: `{s['n_targets']}`",
        f"- Targets with routes: `{s['targets_with_routes']}`",
        "- Note: proxy metrics from flattened GT reactions; not the official PaRoutes tree evaluator.",
        "",
        "| k | route present | exact set | exact sequence | shorter/equal | avg best leaf overlap |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in TOP_KS:
        lines.append(
            f"| {k} | {s[f'top{k}_route_present_rate']:.4f} | {s[f'top{k}_exact_reaction_set_rate']:.4f} | "
            f"{s[f'top{k}_exact_reaction_sequence_rate']:.4f} | {s[f'top{k}_shorter_or_equal_rate']:.4f} | "
            f"{s[f'top{k}_avg_best_leaf_overlap']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PaRoutes-style top-k proxy metrics")
    ap.add_argument("--run", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md")
    args = ap.parse_args()
    report = evaluate_paroutes_topk_proxy(
        run_path=Path(args.run),
        benchmark=Path(args.benchmark),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

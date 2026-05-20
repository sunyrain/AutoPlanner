"""Depth-scaling benchmark for CascadeBoard CC-A* retrosynthesis.

This runner measures the current skeleton-guided CC-A* stack directly, grouped
by requested route depth. It complements AiZ/MCTS solvebench scripts by using
the CascadeBoard skeleton model plus live candidate generators.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.cascadeboard.cc_aostar import plan_with_cc_aostar
from cascade_planner.cascadeboard.live_benchmark import _to_route_skeleton
from cascade_planner.cascadeboard.live_retro import build_live_retro_engine
from cascade_planner.cascadeboard.route_export import route_results_payload
from cascade_planner.cascadeboard.route_recovery import target_recovery_metrics
from cascade_planner.cascadeboard.skeleton_inpainter import generate_multiple_skeletons, load_model


RDLogger.DisableLog("rdApp.*")


def _load_entries(path: str | Path) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text())
    out = []
    for idx, row in enumerate(rows):
        target = row.get("target_smiles")
        if not target or Chem.MolFromSmiles(target) is None:
            continue
        item = dict(row)
        item["_benchmark_index"] = idx
        out.append(item)
    return out


def _pick_depth_cases(
    entries: list[dict[str, Any]],
    *,
    depths: list[int],
    n_per_depth: int,
    ultra_depth: int | None,
    ultra_targets: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        depth = int(entry.get("depth") or len(entry.get("gt_route", [])) or 0)
        if depth > 0:
            by_depth[depth].append(entry)

    cases: list[dict[str, Any]] = []
    for depth in depths:
        pool = list(by_depth.get(depth, []))
        rng.shuffle(pool)
        for entry in pool[:n_per_depth]:
            cases.append({
                "mode": f"gt_depth_{depth}",
                "requested_depth": depth,
                "entry": entry,
            })

    if ultra_depth and ultra_targets > 0:
        pool = sorted(
            entries,
            key=lambda e: (
                int(e.get("depth") or len(e.get("gt_route", [])) or 0),
                _heavy_atoms(e.get("target_smiles")),
            ),
            reverse=True,
        )
        for entry in pool[:ultra_targets]:
            cases.append({
                "mode": f"ultra_depth_{ultra_depth}",
                "requested_depth": ultra_depth,
                "entry": entry,
            })
    return cases


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _route_summary(route: dict[str, Any] | None) -> dict[str, Any]:
    if not route:
        return {}
    metrics = route.get("metrics") or {}
    compat = metrics.get("cascade_compatibility") or {}
    cond = metrics.get("condition") or {}
    natural = metrics.get("route_naturalness") or {}
    progress = metrics.get("retrosynthesis_progress") or {}
    enz = metrics.get("enzyme_evidence") or {}
    uncertainty = ((route.get("explanation") or {}).get("uncertainty_table") or {})
    return {
        "score": route.get("score"),
        "confidence": route.get("confidence"),
        "filled_route": metrics.get("filled_route"),
        "progressive_route": metrics.get("progressive_route"),
        "route_solved": metrics.get("route_solved"),
        "compatibility_success": compat.get("cascade_compatibility_success"),
        "compatibility_issues": compat.get("issues") or [],
        "condition_success": cond.get("condition_window_success"),
        "naturalness_score": natural.get("naturalness_score"),
        "main_chain_reduction": progress.get("main_chain_reduction"),
        "retrosynthesis_progress_success": progress.get("retrosynthesis_progress_success"),
        "terminal_simplified": progress.get("terminal_simplified"),
        "naturalness_issues": natural.get("issues_by_step") or [],
        "enzyme_evidence_coverage": enz.get("enzyme_evidence_coverage"),
        "candidate_sources": metrics.get("candidate_source_counts") or {},
        "expansions": uncertainty.get("expansions"),
        "generated_reactions": uncertainty.get("generated_reactions"),
        "candidate_cache_hits": uncertainty.get("candidate_cache_hits"),
        "pruned_by_route_quality": uncertainty.get("pruned_by_route_quality"),
        "types": [s.get("reaction_type") for s in route.get("steps", [])],
        "sources": [s.get("source") for s in route.get("steps", [])],
    }


def _mean(values: list[Any]) -> float | None:
    nums = []
    for value in values:
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if num == num:
            nums.append(num)
    return round(sum(nums) / len(nums), 3) if nums else None


def _run_case(
    case: dict[str, Any],
    *,
    skeleton_model,
    retro_engine: dict,
    device: str,
    skeleton_samples: int,
    n_results: int,
    candidate_budget: int,
    expansion_budget: int | None,
    expansion_multiplier: int,
) -> dict[str, Any]:
    entry = case["entry"]
    target = entry["target_smiles"]
    requested_depth = int(case["requested_depth"])
    domain = entry.get("route_domain") or "chemoenzymatic"
    t0 = time.time()
    error = None
    skeleton_summaries: list[dict[str, Any]] = []
    payload = route_results_payload(target, [], elapsed_s=0.0)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            skeletons = generate_multiple_skeletons(
                skeleton_model,
                target,
                n_steps=requested_depth,
                k=skeleton_samples,
                domain=domain,
                objective="balanced",
                temperature=0.8,
                device=device,
            )
            skeleton_summaries = [
                {
                    "types": list(getattr(s, "types", [])),
                    "ec1s": list(getattr(s, "ec1s", [])),
                    "compatibility": getattr(s, "compat_pred", ""),
                    "operation_mode": getattr(s, "opmode_pred", ""),
                    "log_prob": float(getattr(s, "log_prob", 0.0) or 0.0),
                }
                for s in skeletons
            ]
            budget = expansion_budget or max(
                requested_depth * candidate_budget * max(skeleton_samples, 1) * expansion_multiplier,
                requested_depth,
                n_results,
            )
            results = plan_with_cc_aostar(
                target=target,
                skeletons=[_to_route_skeleton(s) for s in skeletons],
                retro_engine=retro_engine,
                n_results=n_results,
                candidate_budget=candidate_budget,
                expansion_budget=budget,
                stock_checker=None,
                constraints=None,
            )
        elapsed = time.time() - t0
        payload = route_results_payload(target, results, objective="balanced", elapsed_s=elapsed)
    except Exception as exc:
        elapsed = time.time() - t0
        error = f"{type(exc).__name__}: {exc}"
        payload = route_results_payload(target, [], objective="balanced", elapsed_s=elapsed)

    routes = payload.get("routes") or []
    top_route = routes[0] if routes else None
    recovery = target_recovery_metrics(routes[:5], entry) if entry.get("gt_route") else {}
    return {
        "mode": case["mode"],
        "index": entry.get("_benchmark_index"),
        "doi": entry.get("doi"),
        "cascade_id": entry.get("cascade_id"),
        "target_smiles": target,
        "heavy_atoms": _heavy_atoms(target),
        "route_domain": domain,
        "gt_depth": int(entry.get("depth") or len(entry.get("gt_route", [])) or 0),
        "requested_depth": requested_depth,
        "elapsed_s": round(elapsed, 3),
        "error": error,
        "n_routes": len(routes),
        "skeletons": skeleton_summaries,
        "top_route": _route_summary(top_route),
        "recovery": recovery,
        "planner_output": payload,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    summary: dict[str, Any] = {}
    for mode, items in sorted(grouped.items()):
        n = len(items)
        top = [r.get("top_route") or {} for r in items]
        planned = [r for r in items if r.get("n_routes")]
        natural = [float(t.get("naturalness_score")) for t in top if t.get("naturalness_score") is not None]
        expansions = [float(t.get("expansions")) for t in top if t.get("expansions") is not None]
        sources = Counter()
        issues = Counter()
        for t in top:
            sources.update(t.get("candidate_sources") or {})
            issues.update(t.get("compatibility_issues") or [])
        summary[mode] = {
            "n": n,
            "plan_rate": round(len(planned) / max(n, 1), 3),
            "filled_rate": round(sum(bool(t.get("filled_route")) for t in top) / max(n, 1), 3),
            "progressive_rate": round(sum(bool(t.get("progressive_route")) for t in top) / max(n, 1), 3),
            "solve_rate": round(sum(bool(t.get("route_solved")) for t in top) / max(n, 1), 3),
            "compatibility_rate": round(sum(bool(t.get("compatibility_success")) for t in top) / max(n, 1), 3),
            "condition_rate": round(sum(bool(t.get("condition_success")) for t in top) / max(n, 1), 3),
            "avg_naturalness": round(sum(natural) / len(natural), 3) if natural else None,
            "avg_main_chain_reduction": _mean([t.get("main_chain_reduction") for t in top]),
            "avg_expansions": round(sum(expansions) / len(expansions), 3) if expansions else None,
            "avg_time_s": round(sum(float(r.get("elapsed_s") or 0.0) for r in items) / max(n, 1), 3),
            "candidate_sources": dict(sources),
            "compatibility_issues": dict(issues),
        }
    return summary


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    fields = [
        "mode",
        "index",
        "cascade_id",
        "route_domain",
        "gt_depth",
        "requested_depth",
        "heavy_atoms",
        "elapsed_s",
        "error",
        "n_routes",
        "filled_route",
        "progressive_route",
        "route_solved",
        "compatibility_success",
        "condition_success",
        "naturalness_score",
        "main_chain_reduction",
        "expansions",
        "generated_reactions",
        "pruned_by_route_quality",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            top = row.get("top_route") or {}
            writer.writerow({
                **{k: row.get(k) for k in fields if k in row},
                "filled_route": top.get("filled_route"),
                "progressive_route": top.get("progressive_route"),
                "route_solved": top.get("route_solved"),
                "compatibility_success": top.get("compatibility_success"),
                "condition_success": top.get("condition_success"),
                "naturalness_score": top.get("naturalness_score"),
                "main_chain_reduction": top.get("main_chain_reduction"),
                "expansions": top.get("expansions"),
                "generated_reactions": top.get("generated_reactions"),
                "pruned_by_route_quality": top.get("pruned_by_route_quality"),
            })


def _fmt_optional(value: Any, precision: int = 3) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "NA"
    if num != num:
        return "NA"
    return f"{num:.{precision}f}"


def _write_md(summary: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# CC-A* Depth Benchmark",
        "",
        "| mode | n | plan | filled | progressive | solved | compat | condition | naturalness | main-chain reduction | expansions | avg time s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, row in summary.items():
        nat = "NA" if row["avg_naturalness"] is None else f"{row['avg_naturalness']:.3f}"
        exp = "NA" if row["avg_expansions"] is None else f"{row['avg_expansions']:.1f}"
        lines.append(
            f"| {mode} | {row['n']} | {row['plan_rate']:.3f} | {row['filled_rate']:.3f} | "
            f"{row['progressive_rate']:.3f} | {row['solve_rate']:.3f} | "
            f"{row['compatibility_rate']:.3f} | {row['condition_rate']:.3f} | "
            f"{nat} | {_fmt_optional(row['avg_main_chain_reduction'])} | {exp} | {row['avg_time_s']:.3f} |"
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CC-A* depth scaling benchmark")
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--output-json", default="results/v2/cc_aostar_depth_benchmark.json")
    ap.add_argument("--output-csv", default="results/v2/cc_aostar_depth_benchmark.csv")
    ap.add_argument("--output-md", default="results/v2/cc_aostar_depth_benchmark.md")
    ap.add_argument("--model", default="results/shared/skeleton_inpainter/best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depths", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--n-per-depth", type=int, default=3)
    ap.add_argument("--ultra-depth", type=int, default=6)
    ap.add_argument("--ultra-targets", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skeleton-samples", type=int, default=1)
    ap.add_argument("--n-results", type=int, default=1)
    ap.add_argument("--candidate-budget", type=int, default=2)
    ap.add_argument("--expansion-budget", type=int, default=None)
    ap.add_argument("--expansion-multiplier", type=int, default=4)
    args = ap.parse_args()

    entries = _load_entries(args.bench)
    cases = _pick_depth_cases(
        entries,
        depths=args.depths,
        n_per_depth=args.n_per_depth,
        ultra_depth=args.ultra_depth,
        ultra_targets=args.ultra_targets,
        seed=args.seed,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        retro_engine = build_live_retro_engine()
        skeleton_model = load_model(args.model, device=args.device)

    rows = []
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    for idx, case in enumerate(cases, 1):
        entry = case["entry"]
        print(
            f"[{idx}/{len(cases)}] {case['mode']} depth={case['requested_depth']} "
            f"target={entry['target_smiles'][:48]}",
            flush=True,
        )
        row = _run_case(
            case,
            skeleton_model=skeleton_model,
            retro_engine=retro_engine,
            device=args.device,
            skeleton_samples=args.skeleton_samples,
            n_results=args.n_results,
            candidate_budget=args.candidate_budget,
            expansion_budget=args.expansion_budget,
            expansion_multiplier=args.expansion_multiplier,
        )
        rows.append(row)
        partial = {
            "metadata": vars(args),
            "summary": _summarize(rows),
            "targets": rows,
        }
        out_json.write_text(json.dumps(partial, indent=2))
        top = row.get("top_route") or {}
        print(
            f"  routes={row['n_routes']} filled={top.get('filled_route')} "
            f"progressive={top.get('progressive_route')} solved={top.get('route_solved')} "
            f"compat={top.get('compatibility_success')} nat={top.get('naturalness_score')} "
            f"time={row['elapsed_s']}s error={row['error']}",
            flush=True,
        )

    summary = _summarize(rows)
    output = {
        "metadata": vars(args),
        "summary": summary,
        "targets": rows,
    }
    out_json.write_text(json.dumps(output, indent=2))
    _write_csv(rows, args.output_csv)
    _write_md(summary, args.output_md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

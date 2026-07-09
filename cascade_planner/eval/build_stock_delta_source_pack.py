"""Build source-only stock-closure supervision rows for the reservoir controller.

The builder intentionally refuses eval-only packs by default. Its output is an
augmented JSONL pack that can be passed to ``train_reservoir_distilled_controller``.
Synthetic rows carry ``source_only=true`` so they affect source/budget heads
without training action/value/stock heads on fabricated candidates.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.route_tree.source_gate import SOURCE_GROUPS, source_policy_group


def build_stock_delta_source_pack(
    *,
    pack_path: Path,
    run_path: Path,
    output_pack: Path,
    report_path: Path,
    max_rows: int | None = None,
    allow_eval_only: bool = False,
) -> dict[str, Any]:
    rows = _read_jsonl(pack_path)
    if not rows:
        raise ValueError(f"empty pack: {pack_path}")
    eval_rows = [row for row in rows if bool(row.get("eval_only"))]
    if eval_rows and not allow_eval_only:
        raise ValueError(f"refusing to augment eval-only pack rows: {pack_path}")

    prototypes: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = str(row.get("target_smiles") or row.get("target_id") or "")
        if target and target not in prototypes:
            prototypes[target] = row

    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    targets = run.get("targets") or run.get("results") or []
    synthetic: list[dict[str, Any]] = []
    skipped = Counter()
    group_totals = Counter()
    for target in targets:
        if max_rows is not None and len(synthetic) >= max_rows:
            break
        metrics = target.get("metrics") or {}
        if not bool(metrics.get("strict_stock_solve_any")):
            skipped["not_stock_closed"] += 1
            continue
        target_smiles = str(target.get("target_smiles") or "")
        prototype = prototypes.get(target_smiles)
        if prototype is None:
            skipped["missing_pack_prototype"] += 1
            continue
        routes = (target.get("planner_output") or {}).get("routes") or []
        dist, first_rank = _stock_closed_source_distribution(routes)
        if not dist:
            skipped["missing_stock_source_distribution"] += 1
            continue
        group_totals.update(dist)
        synthetic.append(_synthetic_row(prototype, target, dist, first_rank))

    output_pack.parent.mkdir(parents=True, exist_ok=True)
    with output_pack.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in synthetic:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "reservoir_stock_delta_source_pack.v1",
        "pack_path": str(pack_path),
        "run_path": str(run_path),
        "output_pack": str(output_pack),
        "input_rows": len(rows),
        "synthetic_rows": len(synthetic),
        "output_rows": len(rows) + len(synthetic),
        "eval_only_rows_in_input": len(eval_rows),
        "skipped": dict(skipped),
        "group_totals": dict(group_totals),
        "source_only": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
    return out


def _stock_closed_source_distribution(routes: list[dict[str, Any]]) -> tuple[dict[str, float], int | None]:
    counts: dict[str, float] = defaultdict(float)
    first_rank: int | None = None
    for rank, route in enumerate(routes, start=1):
        if not _route_stock_closed(route):
            continue
        if first_rank is None:
            first_rank = rank
        route_sources = _route_source_groups(route)
        if not route_sources:
            continue
        weight = 1.0 / float(rank)
        for group in route_sources:
            counts[group] += weight / float(len(route_sources))
    total = sum(counts.values())
    if total <= 0:
        return {}, first_rank
    return {group: counts.get(group, 0.0) / total for group in SOURCE_GROUPS}, first_rank


def _route_stock_closed(route: dict[str, Any]) -> bool:
    metrics = route.get("metrics") or {}
    if "strict_stock_solve" in metrics:
        return bool(metrics.get("strict_stock_solve"))
    steps = route.get("steps") or []
    if not steps:
        return False
    for step in steps:
        status = step.get("stock_status") or {}
        if not status or not all(bool(value) for value in status.values()):
            return False
    return True


def _route_source_groups(route: dict[str, Any]) -> list[str]:
    groups: list[str] = []
    for step in route.get("steps") or []:
        source = (
            step.get("source")
            or step.get("proposal_source")
            or step.get("model_source")
            or step.get("enzyme_source")
            or ""
        )
        group = source_policy_group(str(source or ""))
        if group not in SOURCE_GROUPS:
            group = "fallback"
        groups.append(group)
    return groups


def _synthetic_row(
    prototype: dict[str, Any],
    target: dict[str, Any],
    dist: dict[str, float],
    first_rank: int | None,
) -> dict[str, Any]:
    top_group = max(dist, key=dist.get)
    route_recovery = target.get("route_recovery") or {}
    target_index = target.get("index", prototype.get("benchmark_index"))
    return {
        "state_id": f"stock_delta:{target_index}:{prototype.get('state_id')}",
        "target_id": prototype.get("target_id") or target.get("cascade_id") or str(target_index),
        "target_smiles": target.get("target_smiles") or prototype.get("target_smiles"),
        "benchmark_index": prototype.get("benchmark_index", target_index),
        "depth": 0,
        "remaining_depth": prototype.get("remaining_depth", target.get("depth", 0)),
        "leaf": target.get("target_smiles") or prototype.get("leaf") or prototype.get("target_smiles"),
        "source": _representative_source(top_group),
        "source_group": top_group,
        "source_policy_group": top_group,
        "candidate_reaction": "",
        "reactants": [],
        "route_context_features": {
            "stock_delta_source_supervision": True,
            "stock_closed_first_rank": first_rank,
            "route_domain": target.get("route_domain"),
        },
        "source_diagnostics": {},
        "reservoir_rank": 0,
        "teacher_selected": True,
        "teacher_route_rank": first_rank,
        "teacher_stock_closed": True,
        "teacher_exact_hit": bool(route_recovery.get("exact_reaction_in_route_pool")),
        "teacher_gt_reactant_hit": bool(route_recovery.get("gt_reactant_in_route_pool")),
        "teacher_route_value": 1.0,
        "teacher_action_value": 0.0,
        "budget_label": "1x",
        "teacher_source_group_distribution": {group: float(dist.get(group, 0.0)) for group in SOURCE_GROUPS},
        "failure_labels": [],
        "latency_ms": 0.0,
        "eval_only": False,
        "source_only": True,
    }


def _representative_source(group: str) -> str:
    return {
        "chemical": "retrochimera",
        "enzymatic": "enzyformer",
        "rhea_retrorules": "retrorules",
        "retrieval": "v3_retrieval",
        "template": "chemtemplates",
        "fallback": "uspto_template",
    }.get(group, "uspto_template")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build source-only stock-delta reservoir training rows")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--output-pack", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--allow-eval-only", action="store_true")
    args = ap.parse_args()
    report = build_stock_delta_source_pack(
        pack_path=Path(args.pack),
        run_path=Path(args.run),
        output_pack=Path(args.output_pack),
        report_path=Path(args.report),
        max_rows=args.max_rows,
        allow_eval_only=args.allow_eval_only,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

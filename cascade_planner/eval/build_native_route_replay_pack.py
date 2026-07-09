"""Build native route-composition replay rows from benchmark run outputs."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.route_tree.source_gate import source_policy_group


EVAL_BENCHMARK_NAMES = {"benchmark_v2_100.json", "full100"}


def build_native_route_replay_pack(
    *,
    run_path: Path,
    output_pack: Path,
    report_path: Path,
    split: str,
    indices: set[int] | None = None,
    allow_eval_benchmark_train: bool = False,
) -> dict[str, Any]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    eval_only = split == "eval"
    if not eval_only and _looks_like_eval_benchmark(run, run_path) and not allow_eval_benchmark_train:
        raise ValueError(
            "refusing to build train/val native replay rows from an eval-looking full100 run; "
            "use split='eval' for diagnostic rows"
        )
    rows = []
    for pos, target in enumerate(run.get("targets") or run.get("results") or []):
        index = _target_index(target, pos)
        if indices is not None and index not in indices:
            continue
        rows.extend(
            _rows_for_target(
                target,
                index=index,
                split=split,
                eval_only=eval_only,
            )
        )
    output_pack.parent.mkdir(parents=True, exist_ok=True)
    output_pack.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "schema_version": "native_route_replay_pack.v1",
        "run_path": str(run_path),
        "output_pack": str(output_pack),
        "split": split,
        "eval_only": eval_only,
        "rows": len(rows),
        "target_count": len({row["benchmark_index"] for row in rows}),
        "source_counts": dict(Counter(str(row.get("source") or "") for row in rows)),
        "teacher_stock_closed": sum(1 for row in rows if row.get("teacher_stock_closed")),
        "teacher_exact_hit": sum(1 for row in rows if row.get("teacher_exact_hit")),
        "teacher_gt_reactant_hit": sum(1 for row in rows if row.get("teacher_gt_reactant_hit")),
        "guard": {
            "eval_benchmark_detected": _looks_like_eval_benchmark(run, run_path),
            "allow_eval_benchmark_train": bool(allow_eval_benchmark_train),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _rows_for_target(
    target: dict[str, Any],
    *,
    index: int,
    split: str,
    eval_only: bool,
) -> list[dict[str, Any]]:
    recovery = target.get("route_recovery") or {}
    per_route = recovery.get("per_route") or []
    broad_ranks = _broad_route_ranks(target)
    rows = []
    for rank, route in enumerate(_routes(target), start=1):
        metrics = route.get("metrics") or {}
        route_recovery = per_route[rank - 1] if rank - 1 < len(per_route) and isinstance(per_route[rank - 1], dict) else {}
        stock_closed = bool(metrics.get("strict_stock_solve")) or rank in broad_ranks
        exact_hit = bool(route_recovery.get("exact_reaction_hit") or route_recovery.get("exact_reaction_hits"))
        gt_hit = bool(route_recovery.get("gt_reactant_hit"))
        native_route = rank in broad_ranks or _route_has_native_source(route)
        if not native_route and not stock_closed and not exact_hit and not gt_hit:
            continue
        steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
        route_cost, step_costs, terminal_gap_cost = _route_cost(route=route, steps=steps, stock_closed=stock_closed)
        route_value = _cost_to_value(route_cost)
        for step in steps:
            reaction = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
            if not reaction:
                continue
            source = _replay_source(step.get("source"), native_route=native_route)
            group = source_policy_group(source)
            step_index = int(step.get("index") or 0)
            reactants = [str(step.get("main_reactant") or ""), *[str(item) for item in step.get("aux_reactants") or [] if item]]
            reactants = [item for item in reactants if item]
            action_cost = step_costs[step_index] if 0 <= step_index < len(step_costs) else _step_cost(step, reactants)
            action_value = _cost_to_value(action_cost)
            rows.append(
                {
                    "state_id": f"native_replay:{split}:{index}:{rank}:{step_index}",
                    "target_id": str(target.get("cascade_id") or index),
                    "target_smiles": str(target.get("target_smiles") or ""),
                    "benchmark_index": index,
                    "depth": step_index,
                    "remaining_depth": max(len(steps) - step_index - 1, 0),
                    "leaf": str(step.get("product") or target.get("target_smiles") or ""),
                    "source": source,
                    "source_group": group,
                    "source_policy_group": group,
                    "candidate_reaction": reaction,
                    "reactants": reactants,
                    "route_context_features": {
                        "native_route_replay": True,
                        "native_route": bool(native_route),
                        "route_rank": rank,
                        "step_index": step_index,
                        "route_stock_closed": bool(stock_closed),
                        "route_exact_hit": bool(exact_hit),
                        "route_gt_reactant_hit": bool(gt_hit),
                        "cost_model": "reaction_cost_and_or.v1",
                        "route_cost": round(float(route_cost), 6),
                        "step_cost": round(float(action_cost), 6),
                        "terminal_gap_cost": round(float(terminal_gap_cost), 6),
                    },
                    "source_diagnostics": {},
                    "reservoir_rank": rank,
                    "teacher_selected": True,
                    "teacher_route_rank": rank,
                    "teacher_stock_closed": bool(stock_closed),
                    "teacher_exact_hit": bool(exact_hit),
                    "teacher_gt_reactant_hit": bool(gt_hit),
                    "teacher_route_value": route_value,
                    "teacher_action_value": action_value,
                    "teacher_route_cost": route_cost,
                    "teacher_action_cost": action_cost,
                    "teacher_value_policy": "reaction_cost_and_or.v1",
                    "budget_label": "1x",
                    "failure_labels": [],
                    "latency_ms": 0.0,
                    "eval_only": bool(eval_only),
                }
            )
    return rows


def _route_cost(*, route: dict[str, Any], steps: list[dict[str, Any]], stock_closed: bool) -> tuple[float, list[float], float]:
    step_costs = [_step_cost(step, _step_reactants(step)) for step in steps]
    terminal_gap_cost = 0.0 if stock_closed else math.log1p(float(max(1, _nonstock_terminal_count(route))))
    return float(sum(step_costs) + terminal_gap_cost), step_costs, float(terminal_gap_cost)


def _step_cost(step: dict[str, Any], reactants: list[str]) -> float:
    probability = _probability_from_score(
        step.get("score")
        if step.get("score") is not None
        else step.get("probability")
        if step.get("probability") is not None
        else step.get("model_score")
    )
    if probability > 0.0:
        return -math.log(probability)
    return math.log1p(float(max(1, len(reactants))))


def _step_reactants(step: dict[str, Any]) -> list[str]:
    reactants = [str(step.get("main_reactant") or ""), *[str(item) for item in step.get("aux_reactants") or [] if item]]
    return [item for item in reactants if item]


def _nonstock_terminal_count(route: dict[str, Any]) -> int:
    count = 0
    for step in route.get("steps") or []:
        if not isinstance(step, dict):
            continue
        stock_status = step.get("stock_status")
        if isinstance(stock_status, dict):
            count += sum(1 for value in stock_status.values() if not bool(value))
    return count


def _probability_from_score(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    if probability <= 0.0:
        return 0.0
    return min(1.0, max(1e-6, probability))


def _cost_to_value(cost: float) -> float:
    return 1.0 / (1.0 + max(0.0, float(cost)))


def _target_index(target: dict[str, Any], pos: int) -> int:
    value = target.get("benchmark_index", target.get("index", pos))
    try:
        return int(value)
    except (TypeError, ValueError):
        return pos


def _routes(target: dict[str, Any]) -> list[dict[str, Any]]:
    routes = (target.get("planner_output") or {}).get("routes") or []
    return routes if isinstance(routes, list) else []


def _broad_route_ranks(target: dict[str, Any]) -> set[int]:
    broad = (target.get("planner_output") or {}).get("broad_reservoir") or {}
    ranks = set()
    for route in broad.get("routes") or []:
        try:
            ranks.add(int(route.get("route_rank")))
        except (TypeError, ValueError):
            continue
    return ranks


def _route_has_native_source(route: dict[str, Any]) -> bool:
    metrics = route.get("metrics") or {}
    source_counts = metrics.get("candidate_source_counts") or {}
    if any(_is_native_source(source) for source in source_counts):
        return True
    for step in route.get("steps") or []:
        if isinstance(step, dict) and _is_native_source(step.get("source")):
            return True
    return False


def _is_native_source(source: Any) -> bool:
    text = str(source or "")
    return text == "ChemEnzyRetroPlanner" or text == "native_chem_enzy" or ">>" in text or text.startswith("(")


def _replay_source(source: Any, *, native_route: bool) -> str:
    text = str(source or "")
    if text == "ChemEnzyRetroPlanner":
        return "v3_retrieval"
    if text in {"native_chem_enzy", "native_template"}:
        return "chemtemplates"
    if ">>" in text or text.startswith("("):
        return "chemtemplates"
    if not text and native_route:
        return "chemtemplates"
    return text or "template"


def _looks_like_eval_benchmark(run: dict[str, Any], run_path: Path) -> bool:
    path_text = str(run_path).lower()
    if "full100" in path_text:
        return True
    metadata = run.get("metadata") or {}
    candidates = [metadata.get("benchmark")]
    for item in metadata.get("source_metadata") or []:
        if isinstance(item, dict):
            candidates.append(item.get("benchmark"))
    for candidate in candidates:
        text = str(candidate or "").lower()
        if any(name in text for name in EVAL_BENCHMARK_NAMES):
            return True
    return False


def _parse_indices(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build native route-composition replay rows")
    ap.add_argument("--run", required=True)
    ap.add_argument("--output-pack", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--split", choices=["train", "val", "eval"], required=True)
    ap.add_argument("--indices", default=None)
    ap.add_argument("--allow-eval-benchmark-train", action="store_true")
    args = ap.parse_args()
    report = build_native_route_replay_pack(
        run_path=Path(args.run),
        output_pack=Path(args.output_pack),
        report_path=Path(args.report),
        split=args.split,
        indices=_parse_indices(args.indices),
        allow_eval_benchmark_train=args.allow_eval_benchmark_train,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

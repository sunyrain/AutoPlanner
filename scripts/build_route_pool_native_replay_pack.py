#!/usr/bin/env python3
"""Build native_replay proposal rows from route-pool JSONL files.

The output schema is the same lightweight row format consumed by
``cascade_planner.route_tree.proposals`` via AUTOPLANNER_NATIVE_REPLAY_PROPOSALS.
By default this builder keeps chemical steps only, because native_replay is a
template/chemical source in the route-tree source gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.route_tree.source_gate import source_policy_group  # noqa: E402


SCHEMA_VERSION = "route_pool_native_replay_pack.v1"
EVAL_NAME_MARKERS = ("test", "full100", "uspto190", "p4n4", "eval")


def build_route_pool_native_replay_pack(
    *,
    input_paths: list[Path],
    output_pack: Path,
    report_path: Path,
    split: str,
    include_enzymatic: bool = False,
    allow_eval_benchmark_train: bool = False,
    max_routes: int | None = None,
) -> dict[str, Any]:
    eval_only = split == "eval"
    if not eval_only and not allow_eval_benchmark_train:
        eval_inputs = [str(path) for path in input_paths if _looks_like_eval_path(path)]
        if eval_inputs:
            raise ValueError(
                "refusing to build non-eval native replay rows from eval-looking input(s): "
                + ", ".join(eval_inputs)
            )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counters: Counter[str] = Counter()
    transform_counts: Counter[str] = Counter()
    routes_seen = 0
    for path in input_paths:
        for route_index, route in _iter_jsonl(path):
            if max_routes is not None and routes_seen >= max_routes:
                break
            routes_seen += 1
            route_rows = _rows_for_route_pool_entry(
                route,
                input_path=path,
                route_position=route_index,
                split=split,
                eval_only=eval_only,
                include_enzymatic=include_enzymatic,
                counters=counters,
                transform_counts=transform_counts,
            )
            for row in route_rows:
                key = (str(row.get("leaf") or ""), str(row.get("candidate_reaction") or ""))
                if key in seen:
                    counters["dedupe_dropped"] += 1
                    continue
                seen.add(key)
                rows.append(row)
        if max_routes is not None and routes_seen >= max_routes:
            break

    output_pack.parent.mkdir(parents=True, exist_ok=True)
    output_pack.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "input_paths": [str(path) for path in input_paths],
        "output_pack": str(output_pack),
        "split": split,
        "eval_only": bool(eval_only),
        "include_enzymatic": bool(include_enzymatic),
        "routes_seen": routes_seen,
        "rows": len(rows),
        "counters": dict(counters),
        "top_transformation_superclasses": dict(transform_counts.most_common(20)),
        "guard": {
            "allow_eval_benchmark_train": bool(allow_eval_benchmark_train),
            "eval_name_markers": list(EVAL_NAME_MARKERS),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield index, row


def _rows_for_route_pool_entry(
    route: dict[str, Any],
    *,
    input_path: Path,
    route_position: int,
    split: str,
    eval_only: bool,
    include_enzymatic: bool,
    counters: Counter[str],
    transform_counts: Counter[str],
) -> list[dict[str, Any]]:
    target = str(route.get("target_smiles") or route.get("target") or "")
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    if not target or not steps:
        counters["route_missing_target_or_steps"] += 1
        return []
    route_rank = _safe_int(route.get("native_rank"), route_position + 1)
    route_value = _safe_float(route.get("value_target"), None)
    if route_value is None:
        route_value = _safe_float(route.get("native_score"), 0.0)
    rows: list[dict[str, Any]] = []
    for step_position, step in enumerate(steps):
        counters["steps_seen"] += 1
        if _is_enzymatic_step(step) and not include_enzymatic:
            counters["steps_skipped_enzymatic"] += 1
            continue
        reaction = str(step.get("rxn_smiles") or step.get("reaction_smiles") or "")
        product = str(step.get("product_smiles") or _first(step.get("products")) or _reaction_product(reaction))
        reactants = [str(item) for item in step.get("reactants") or [] if item]
        if not reactants:
            reactants = _reaction_reactants(reaction)
        if not reaction or not product or not reactants:
            counters["steps_skipped_missing_fields"] += 1
            continue
        transform = str(step.get("transformation_superclass") or step.get("transformation_name") or "unknown")
        transform_counts[transform] += 1
        source = str(step.get("source_model") or "ChemEnzyRetroPlanner")
        policy_group = source_policy_group("native_replay")
        action_value = _safe_float(step.get("native_step_score"), None)
        if action_value is None:
            action_value = _safe_float(step.get("score"), None)
        if action_value is None:
            action_value = route_value
        rows.append(
            {
                "state_id": f"route_pool_native_replay:{split}:{route.get('route_id') or route_position}:{step_position}",
                "target_id": str(route.get("target_id") or route.get("route_id") or route_position),
                "target_smiles": target,
                "benchmark_index": route_position,
                "depth": int(step.get("step_index") or step_position),
                "remaining_depth": max(0, len(steps) - step_position - 1),
                "leaf": product,
                "source": source,
                "source_group": "chemical",
                "source_policy_group": policy_group,
                "candidate_reaction": reaction,
                "reactants": reactants,
                "route_context_features": {
                    "route_pool_native_replay": True,
                    "input_path": str(input_path),
                    "route_id": route.get("route_id"),
                    "route_source": route.get("route_source"),
                    "route_rank": route_rank,
                    "step_index": step_position,
                    "transformation_name": step.get("transformation_name"),
                    "transformation_superclass": step.get("transformation_superclass"),
                    "stock_closed": bool(route.get("stock_closed")),
                },
                "reservoir_rank": route_rank,
                "teacher_selected": True,
                "teacher_route_rank": route_rank,
                "teacher_stock_closed": bool(route.get("stock_closed")),
                "teacher_exact_hit": False,
                "teacher_gt_reactant_hit": False,
                "teacher_route_value": float(route_value or 0.0),
                "teacher_action_value": float(action_value or 0.0),
                "teacher_value_policy": SCHEMA_VERSION,
                "budget_label": "1x",
                "failure_labels": [],
                "latency_ms": 0.0,
                "eval_only": bool(eval_only),
            }
        )
        counters["steps_kept"] += 1
    return rows


def _is_enzymatic_step(step: dict[str, Any]) -> bool:
    catalyst_classes = {str(item).strip().lower() for item in step.get("catalyst_classes") or []}
    if "enzyme" in catalyst_classes or "enzymatic" in catalyst_classes:
        return True
    if any(str(item).strip() for item in step.get("ec1_values") or []):
        return True
    conditions = step.get("step_conditions") or {}
    if isinstance(conditions, dict) and any(str(key).lower().startswith("ec") for key in conditions):
        return True
    return False


def _reaction_product(reaction: str) -> str:
    return reaction.split(">>", 1)[1] if ">>" in reaction else ""


def _reaction_reactants(reaction: str) -> list[str]:
    if ">>" not in reaction:
        return []
    return [part for part in reaction.split(">>", 1)[0].split(".") if part]


def _first(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0] or "")
    return ""


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _looks_like_eval_path(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in EVAL_NAME_MARKERS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-pack", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "eval"), required=True)
    parser.add_argument("--include-enzymatic", action="store_true")
    parser.add_argument("--allow-eval-benchmark-train", action="store_true")
    parser.add_argument("--max-routes", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_route_pool_native_replay_pack(
        input_paths=list(args.input),
        output_pack=args.output_pack,
        report_path=args.report,
        split=args.split,
        include_enzymatic=bool(args.include_enzymatic),
        allow_eval_benchmark_train=bool(args.allow_eval_benchmark_train),
        max_routes=args.max_routes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

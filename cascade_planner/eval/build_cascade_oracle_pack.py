"""Build cascade-native oracle labels from AutoPlanner traces and native routes."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.cascade_oracle import (
    CascadeOracleRuntime,
    build_cascade_oracle_payload_from_native,
)
from cascade_planner.route_tree.schema import CandidateAction
from cascade_planner.route_tree.source_gate import source_policy_group


EVAL_BENCHMARK_NAMES = {"benchmark_v2_100.json", "full100"}


def build_cascade_oracle_pack(
    *,
    trace_path: Path,
    native_payload_path: Path,
    output_pack: Path,
    output_payload: Path,
    report_path: Path,
    split: str,
    topk: int = 5,
    selection: str = "rank_plus_stock",
    allow_eval_benchmark_train: bool = False,
) -> dict[str, Any]:
    eval_only = split == "eval"
    if not eval_only and _looks_like_eval_input(trace_path) and not allow_eval_benchmark_train:
        raise ValueError(
            "refusing to build train/val cascade oracle rows from an eval-looking full100 trace; "
            "use split='eval' for diagnostic rows"
        )
    payload = build_cascade_oracle_payload_from_native(
        native_payload_path=native_payload_path,
        output_path=output_payload,
        topk=topk,
        selection=selection,
    )
    oracle = CascadeOracleRuntime(output_payload)
    rows: list[dict[str, Any]] = []
    stats = Counter()
    for trace_row in _load_trace_rows(trace_path):
        for row in _rows_for_trace(trace_row, oracle=oracle, split=split, eval_only=eval_only):
            rows.append(row)
            stats["rows"] += 1
            if row.get("oracle_match"):
                stats["oracle_matches"] += 1
            if row.get("teacher_stock_closed"):
                stats["teacher_stock_closed"] += 1
            stats[f"source_group:{row.get('source_group')}"] += 1
    output_pack.parent.mkdir(parents=True, exist_ok=True)
    output_pack.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "schema_version": "cascade_oracle_pack.v1",
        "trace_path": str(trace_path),
        "native_payload_path": str(native_payload_path),
        "oracle_payload_path": str(output_payload),
        "output_pack": str(output_pack),
        "split": split,
        "eval_only": eval_only,
        "topk": int(topk),
        "selection": selection,
        "targets_in_oracle_payload": len(payload.get("targets") or []),
        "stats": dict(stats),
        "guard": {
            "eval_input_detected": _looks_like_eval_input(trace_path),
            "allow_eval_benchmark_train": bool(allow_eval_benchmark_train),
        },
        "label_policy": {
            "uses_exact_or_gt": False,
            "teacher": "ChemEnzy native route pool scored by AutoPlanner cascade rubric",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _rows_for_trace(
    trace_row: dict[str, Any],
    *,
    oracle: CascadeOracleRuntime,
    split: str,
    eval_only: bool,
) -> list[dict[str, Any]]:
    event = trace_row.get("event") if isinstance(trace_row.get("event"), dict) else trace_row
    if not isinstance(event, dict):
        return []
    target = str(trace_row.get("target_smiles") or (event.get("state") or {}).get("target") or "")
    leaf = str(event.get("expanded_leaf") or "")
    candidates = [row for row in event.get("candidate_actions") or [] if isinstance(row, dict)]
    if not target or not leaf or not candidates:
        return []
    rows = []
    allocation = _extract_allocation(event)
    for rank, action_dict in enumerate(candidates, start=1):
        action = CandidateAction.from_candidate(leaf, action_dict, rank=rank, source=action_dict.get("source"))
        match = oracle.action_value(target=target, leaf=leaf, action=action)
        selected = action.canonical_key == str(event.get("selected_action_key") or "")
        candidate_cost = _candidate_action_cost(action, action_dict)
        action_value = _cost_to_value(candidate_cost)
        route_value = action_value
        route_rank = 0
        teacher_stock_closed = False
        oracle_match = False
        oracle_reason = ""
        if match is not None:
            oracle_match = True
            oracle_reason = match.reason
            route_rank = int(match.route_rank or 0)
            teacher_stock_closed = bool(match.stock_closed)
            route_value = max(route_value, float(match.value))
            action_value = max(action_value, float(match.value))
        row = {
            "state_id": str(event.get("state_id") or (event.get("state") or {}).get("state_id") or ""),
            "target_id": str(trace_row.get("target_id") or trace_row.get("benchmark_index") or target),
            "target_smiles": target,
            "benchmark_index": trace_row.get("benchmark_index"),
            "depth": int(event.get("depth") or 0),
            "remaining_depth": max(0, int(((event.get("outcome") or {}).get("max_depth") or 0)) - int(event.get("depth") or 0)),
            "leaf": leaf,
            "source": str(action.source or action_dict.get("source") or ""),
            "source_group": source_policy_group(str(action.source or action_dict.get("source") or "")),
            "source_policy_group": source_policy_group(str(action.source or action_dict.get("source") or "")),
            "candidate_reaction": action.rxn_smiles,
            "reactants": list(action.reactants),
            "route_context_features": {
                "cascade_oracle": True,
                "state_depth": int(event.get("depth") or 0),
                "open_leaf_count": len(event.get("open_leaves") or []),
                "proposal_budget": _proposal_budget(event),
                "oracle_match": bool(oracle_match),
                "oracle_confidence": float(match.confidence) if match is not None else 0.0,
                "oracle_reason_bucket": _stable_bucket(oracle_reason, 32),
                "candidate_cost": round(float(candidate_cost), 6),
            },
            "source_diagnostics": {},
            "reservoir_rank": route_rank,
            "teacher_selected": bool(oracle_match or selected),
            "teacher_route_rank": route_rank,
            "teacher_stock_closed": bool(teacher_stock_closed),
            "teacher_exact_hit": False,
            "teacher_gt_reactant_hit": False,
            "teacher_route_value": round(float(max(0.0, min(1.0, route_value))), 6),
            "teacher_action_value": round(float(max(0.0, min(1.0, action_value))), 6),
            "teacher_source_group_distribution": _source_distribution(source_policy_group(str(action.source or "")), oracle_match=oracle_match),
            "budget_label": str(allocation.get("budget_multiplier_label") or "1x"),
            "failure_labels": [] if teacher_stock_closed or oracle_match else ["stock_dead_end"],
            "latency_ms": 0.0,
            "eval_only": bool(eval_only),
            "split": split,
            "oracle_match": bool(oracle_match),
            "oracle_reason": oracle_reason,
        }
        rows.append(row)
    return rows


def _load_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _extract_allocation(event: dict[str, Any]) -> dict[str, Any]:
    for row in reversed(event.get("source_budgets") or []):
        if isinstance(row, dict):
            gate = row.get("proposal_gate")
            if isinstance(gate, dict):
                return gate
    return {}


def _proposal_budget(event: dict[str, Any]) -> int:
    for row in reversed(event.get("proposal_diagnostics") or []):
        if isinstance(row, dict) and row.get("proposal_budget") is not None:
            try:
                return int(row.get("proposal_budget"))
            except (TypeError, ValueError):
                return 0
    return 0


def _source_distribution(group: str, *, oracle_match: bool) -> dict[str, float]:
    return {group or "fallback": 1.0}


def _candidate_action_cost(action: CandidateAction, action_dict: dict[str, Any]) -> float:
    probability = _probability_from_score(action.raw_score)
    if probability <= 0.0:
        probability = _probability_from_score(action_dict.get("score"))
    reaction_cost = _negative_log_probability(probability)
    reactant_count_cost = math.log1p(float(len(action.reactants) or 1))
    return reaction_cost + reactant_count_cost


def _probability_from_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score <= 0.0:
        return 0.0
    return min(1.0, score)


def _negative_log_probability(probability: Any) -> float:
    probability = max(1e-6, _probability_from_score(probability))
    return -math.log(probability)


def _cost_to_value(cost: Any) -> float:
    try:
        value = float(cost)
    except (TypeError, ValueError):
        value = math.inf
    if not math.isfinite(value):
        return 0.0
    return 1.0 / (1.0 + max(0.0, value))


def _stable_bucket(value: Any, buckets: int) -> int:
    text = str(value or "")
    total = 0
    for char in text:
        total = (total * 131 + ord(char)) % max(1, buckets)
    return total


def _looks_like_eval_input(path: Path) -> bool:
    text = str(path).lower()
    return any(name in text for name in EVAL_BENCHMARK_NAMES)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cascade-native oracle pack from traces and native route payload")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--native-payload", required=True)
    ap.add_argument("--output-pack", required=True)
    ap.add_argument("--output-payload", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--split", choices=["train", "val", "eval"], default="train")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--selection", default="rank_plus_stock")
    ap.add_argument("--allow-eval-benchmark-train", action="store_true")
    args = ap.parse_args()
    report = build_cascade_oracle_pack(
        trace_path=Path(args.trace),
        native_payload_path=Path(args.native_payload),
        output_pack=Path(args.output_pack),
        output_payload=Path(args.output_payload),
        report_path=Path(args.report),
        split=args.split,
        topk=args.topk,
        selection=args.selection,
        allow_eval_benchmark_train=bool(args.allow_eval_benchmark_train),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

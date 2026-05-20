"""Diagnose state/action ranking inside ChemEnzy MolStar expansion pools.

This is a search-control diagnostic: it asks whether useful actions that are
already in an expansion pool are ranked highly enough by the scalar cost used
by MolStar. It is not a cascade-record gold classifier.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def analyze_cascade_action_ranking(
    *,
    pack_dir: Path,
    output_path: Path | None = None,
    label_name: str = "action_value",
) -> dict[str, Any]:
    rows = _read_jsonl(pack_dir / "action_value.jsonl")
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row.get("state_id") or f"row_{len(by_state)}")].append(row)

    report = {
        "metadata": {
            "pack_dir": str(pack_dir),
            "n_action_rows": len(rows),
            "n_states": len(by_state),
            "positive_label": label_name,
            "diagnostic_contract": "internal_search_action_ranking_not_record_gold.v1",
        },
        "cost_rankings": {
            "base_cost": _cost_ranking_metrics(by_state, "base_cost", label_name),
            "total_cost": _cost_ranking_metrics(by_state, "total_cost", label_name),
        },
        "score_rankings": _available_score_rankings(by_state, rows, label_name),
        "breakdowns": {
            "route_domain": _breakdown(rows, "route_domain", label_name),
            "reaction_domain": _breakdown(rows, "reaction_domain", label_name),
            "adjacent_reaction_domain": _breakdown(
                rows,
                lambda row: (row.get("context_features") or {}).get("adjacent_reaction_domain") or "unknown",
                label_name,
            ),
            "source_model": _breakdown(rows, "source_model", label_name),
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _available_score_rankings(
    by_state: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    label_name: str,
) -> dict[str, Any]:
    out = {}
    for field in ("action_value_score",):
        if any(row.get(field) is not None for row in rows):
            out[field] = _score_ranking_metrics(by_state, field, label_name)
    return out


def _score_ranking_metrics(
    by_state: dict[str, list[dict[str, Any]]],
    score_field: str,
    label_name: str,
) -> dict[str, Any]:
    positive_ranks = []
    exact_ranks = []
    positive_states = 0
    exact_states = 0
    top1_positive = 0
    top5_positive = 0
    top1_exact = 0
    top5_exact = 0
    positive_pair_correct = 0.0
    positive_pair_total = 0
    exact_pair_correct = 0.0
    exact_pair_total = 0

    for items in by_state.values():
        ordered = sorted(items, key=lambda row: _score_key(row, score_field), reverse=True)
        positive_ids = {id(row) for row in items if _positive_label_value(row, label_name) > 0.0}
        exact_ids = {
            id(row)
            for row in items
            if int((row.get("labels") or {}).get("exact_gt_reaction") or 0) > 0
        }
        if positive_ids:
            positive_states += 1
            best_rank = _best_rank(ordered, positive_ids)
            positive_ranks.append(best_rank)
            top1_positive += int(best_rank <= 1)
            top5_positive += int(best_rank <= 5)
            correct, total = _pairwise_score_accuracy(items, positive_ids, score_field)
            positive_pair_correct += correct
            positive_pair_total += total
        if exact_ids:
            exact_states += 1
            best_rank = _best_rank(ordered, exact_ids)
            exact_ranks.append(best_rank)
            top1_exact += int(best_rank <= 1)
            top5_exact += int(best_rank <= 5)
            correct, total = _pairwise_score_accuracy(items, exact_ids, score_field)
            exact_pair_correct += correct
            exact_pair_total += total

    return {
        "positive_states": positive_states,
        "top1_positive_state_hit_rate": _rate(top1_positive, positive_states),
        "top5_positive_state_hit_rate": _rate(top5_positive, positive_states),
        "mean_best_positive_rank": _mean(positive_ranks),
        "median_best_positive_rank": _median(positive_ranks),
        "pairwise_positive_score_accuracy": _rate_float(positive_pair_correct, positive_pair_total),
        "exact_states": exact_states,
        "top1_exact_state_hit_rate": _rate(top1_exact, exact_states),
        "top5_exact_state_hit_rate": _rate(top5_exact, exact_states),
        "mean_best_exact_rank": _mean(exact_ranks),
        "median_best_exact_rank": _median(exact_ranks),
        "pairwise_exact_score_accuracy": _rate_float(exact_pair_correct, exact_pair_total),
    }


def _cost_ranking_metrics(
    by_state: dict[str, list[dict[str, Any]]],
    cost_field: str,
    label_name: str,
) -> dict[str, Any]:
    positive_ranks = []
    exact_ranks = []
    positive_margins = []
    exact_margins = []
    positive_states = 0
    exact_states = 0
    top1_positive = 0
    top5_positive = 0
    top1_exact = 0
    top5_exact = 0
    positive_pair_correct = 0.0
    positive_pair_total = 0
    exact_pair_correct = 0.0
    exact_pair_total = 0

    for items in by_state.values():
        ordered = sorted(items, key=lambda row: _cost_key(row, cost_field))
        positive_ids = {
            id(row)
            for row in items
            if _positive_label_value(row, label_name) > 0.0
        }
        exact_ids = {
            id(row)
            for row in items
            if int((row.get("labels") or {}).get("exact_gt_reaction") or 0) > 0
        }
        if positive_ids:
            positive_states += 1
            best_rank = _best_rank(ordered, positive_ids)
            positive_ranks.append(best_rank)
            top1_positive += int(best_rank <= 1)
            top5_positive += int(best_rank <= 5)
            margin = _best_positive_minus_best_negative(items, positive_ids, cost_field)
            if margin is not None:
                positive_margins.append(margin)
            correct, total = _pairwise_cost_accuracy(items, positive_ids, cost_field)
            positive_pair_correct += correct
            positive_pair_total += total
        if exact_ids:
            exact_states += 1
            best_rank = _best_rank(ordered, exact_ids)
            exact_ranks.append(best_rank)
            top1_exact += int(best_rank <= 1)
            top5_exact += int(best_rank <= 5)
            margin = _best_positive_minus_best_negative(items, exact_ids, cost_field)
            if margin is not None:
                exact_margins.append(margin)
            correct, total = _pairwise_cost_accuracy(items, exact_ids, cost_field)
            exact_pair_correct += correct
            exact_pair_total += total

    return {
        "positive_states": positive_states,
        "top1_positive_state_hit_rate": _rate(top1_positive, positive_states),
        "top5_positive_state_hit_rate": _rate(top5_positive, positive_states),
        "mean_best_positive_rank": _mean(positive_ranks),
        "median_best_positive_rank": _median(positive_ranks),
        "mean_positive_minus_best_negative_cost_margin": _mean(positive_margins),
        "pairwise_positive_cost_accuracy": _rate_float(positive_pair_correct, positive_pair_total),
        "exact_states": exact_states,
        "top1_exact_state_hit_rate": _rate(top1_exact, exact_states),
        "top5_exact_state_hit_rate": _rate(top5_exact, exact_states),
        "mean_best_exact_rank": _mean(exact_ranks),
        "median_best_exact_rank": _median(exact_ranks),
        "mean_exact_minus_best_negative_cost_margin": _mean(exact_margins),
        "pairwise_exact_cost_accuracy": _rate_float(exact_pair_correct, exact_pair_total),
    }


def _breakdown(
    rows: list[dict[str, Any]],
    field: str | Callable[[dict[str, Any]], Any],
    label_name: str,
) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if callable(field):
            key = str(field(row))
        else:
            key = str(row.get(field) or "unknown")
        labels = row.get("labels") or {}
        counters[key]["action_rows"] += 1
        counters[key]["positive_actions"] += int(_positive_label_value(row, label_name) > 0.0)
        counters[key]["exact_actions"] += int(labels.get("exact_gt_reaction") or 0)
        counters[key]["gt_reactant_actions"] += int(labels.get("gt_reactant_hit") or 0)
    out = {}
    for key, counter in sorted(counters.items()):
        rows_n = int(counter["action_rows"])
        out[key] = {
            "action_rows": rows_n,
            "positive_actions": int(counter["positive_actions"]),
            "positive_action_rate": _rate(counter["positive_actions"], rows_n),
            "exact_actions": int(counter["exact_actions"]),
            "exact_action_rate": _rate(counter["exact_actions"], rows_n),
            "gt_reactant_actions": int(counter["gt_reactant_actions"]),
            "gt_reactant_action_rate": _rate(counter["gt_reactant_actions"], rows_n),
        }
    return out


def _cost_key(row: dict[str, Any], cost_field: str) -> tuple[int, float]:
    value = _float_or_none(row.get(cost_field))
    return (1, 1e12) if value is None else (0, value)


def _score_key(row: dict[str, Any], score_field: str) -> float:
    value = _float_or_none(row.get(score_field))
    return -1e12 if value is None else value


def _positive_label_value(row: dict[str, Any], label_name: str) -> float:
    return float((row.get("labels") or {}).get(label_name) or 0.0)


def _best_rank(ordered: list[dict[str, Any]], positive_ids: set[int]) -> int:
    for idx, row in enumerate(ordered, start=1):
        if id(row) in positive_ids:
            return idx
    return len(ordered) + 1


def _best_positive_minus_best_negative(
    items: list[dict[str, Any]],
    positive_ids: set[int],
    cost_field: str,
) -> float | None:
    pos_costs = [_float_or_none(row.get(cost_field)) for row in items if id(row) in positive_ids]
    neg_costs = [_float_or_none(row.get(cost_field)) for row in items if id(row) not in positive_ids]
    pos_costs = [value for value in pos_costs if value is not None]
    neg_costs = [value for value in neg_costs if value is not None]
    if not pos_costs or not neg_costs:
        return None
    return round(min(pos_costs) - min(neg_costs), 6)


def _pairwise_cost_accuracy(
    items: list[dict[str, Any]],
    positive_ids: set[int],
    cost_field: str,
) -> tuple[float, int]:
    pos_costs = [_float_or_none(row.get(cost_field)) for row in items if id(row) in positive_ids]
    neg_costs = [_float_or_none(row.get(cost_field)) for row in items if id(row) not in positive_ids]
    pos_costs = [value for value in pos_costs if value is not None]
    neg_costs = [value for value in neg_costs if value is not None]
    correct = 0.0
    total = 0
    for pos in pos_costs:
        for neg in neg_costs:
            total += 1
            correct += float(pos < neg) + 0.5 * float(pos == neg)
    return correct, total


def _pairwise_score_accuracy(
    items: list[dict[str, Any]],
    positive_ids: set[int],
    score_field: str,
) -> tuple[float, int]:
    pos_scores = [_float_or_none(row.get(score_field)) for row in items if id(row) in positive_ids]
    neg_scores = [_float_or_none(row.get(score_field)) for row in items if id(row) not in positive_ids]
    pos_scores = [value for value in pos_scores if value is not None]
    neg_scores = [value for value in neg_scores if value is not None]
    correct = 0.0
    total = 0
    for pos in pos_scores:
        for neg in neg_scores:
            total += 1
            correct += float(pos > neg) + 0.5 * float(pos == neg)
    return correct, total


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _rate(num: int | float, den: int | float) -> float | None:
    if not den:
        return None
    return round(float(num) / float(den), 6)


def _rate_float(num: float, den: int) -> float | None:
    return _rate(num, den)


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(float(sum(values)) / len(values), 6)


def _median(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 6)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze action ranking inside a cascade action-value pack")
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--label-name", default="action_value")
    args = ap.parse_args()
    report = analyze_cascade_action_ranking(
        pack_dir=Path(args.pack_dir),
        output_path=Path(args.output) if args.output else None,
        label_name=args.label_name,
    )
    print(
        json.dumps(
            {
                "metadata": report["metadata"],
                "cost_rankings": report["cost_rankings"],
                "score_rankings": report["score_rankings"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

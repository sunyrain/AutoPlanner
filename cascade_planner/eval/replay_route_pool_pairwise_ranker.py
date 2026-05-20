"""Apply a trained route-pool pairwise ranker to a route-pool pack."""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.eval.train_route_pool_ranker import _evaluate_rankings, _float


SCHEMA_VERSION = "route_pool_pairwise_ranker_replay.v1"


def replay_route_pool_pairwise_ranker(
    *,
    pack_jsonl: Path,
    ranker_pickle: Path,
    output_jsonl: Path,
    report_json: Path,
    score_field: str = "route_pool_ranker_score",
) -> dict[str, Any]:
    rows = _read_jsonl(pack_jsonl)
    with ranker_pickle.open("rb") as fh:
        payload = pickle.load(fh)
    feature_names = [str(name) for name in payload.get("feature_names") or []]
    if not feature_names:
        raise ValueError(f"ranker pickle has no feature_names: {ranker_pickle}")
    model = payload.get("model")
    mean = np.asarray(payload.get("mean"), dtype=np.float32)
    std = np.asarray(payload.get("std"), dtype=np.float32)
    x = np.asarray([[_float((row.get("feature") or {}).get(name)) for name in feature_names] for row in rows], dtype=np.float32)
    scores = model.decision_function((x - mean) / std).astype(np.float32)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    group_ids = [str(row.get("target_id") or "") for row in rows]
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, group_id in enumerate(group_ids):
        by_group[group_id].append(idx)
    ranked_rows: list[dict[str, Any]] = []
    top1_changes = 0
    for group_id, indices in sorted(by_group.items()):
        native_top = min(indices, key=lambda idx: (int(rows[idx].get("native_rank") or 10**9), str(rows[idx].get("route_id") or "")))
        order = sorted(indices, key=lambda idx: (-float(scores[idx]), int(rows[idx].get("native_rank") or 10**9), str(rows[idx].get("route_id") or "")))
        if order and order[0] != native_top:
            top1_changes += 1
        for rank, idx in enumerate(order, start=1):
            row = dict(rows[idx])
            row[score_field] = float(scores[idx])
            row["route_pool_ranker_rank"] = int(rank)
            ranked_rows.append(row)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in ranked_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    dataset = {
        "rows": rows,
        "y": np.asarray([int(row.get("route_label") or 0) for row in rows], dtype=np.int32),
        "group_ids": group_ids,
    }
    metrics = _evaluate_rankings(dataset, scores)
    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "pack_jsonl": str(pack_jsonl),
            "ranker_pickle": str(ranker_pickle),
            "feature_names": feature_names,
            "score_field": score_field,
        },
        "counts": {
            "rows": len(rows),
            "targets": len(by_group),
            "positive_targets": metrics.get("positive_groups"),
            "top1_changed_targets": int(top1_changes),
        },
        "metrics": metrics,
        "outputs": {
            "reranked_jsonl": str(output_jsonl),
            "report_json": str(report_json),
        },
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a trained route-pool pairwise ranker")
    ap.add_argument("--pack-jsonl", required=True)
    ap.add_argument("--ranker-pickle", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--score-field", default="route_pool_ranker_score")
    args = ap.parse_args()
    report = replay_route_pool_pairwise_ranker(
        pack_jsonl=Path(args.pack_jsonl),
        ranker_pickle=Path(args.ranker_pickle),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
        score_field=args.score_field,
    )
    print(json.dumps({"counts": report["counts"], "metrics": report["metrics"], "outputs": report["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

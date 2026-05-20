"""Replay a trained route/block value model as final reranker.

This is an offline replay on a fixed route_block_value_pack. It verifies the
route/block scorer as a final reranker before any search-time promotion.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.eval.train_route_block_value_model import (
    _audit_scores,
    _dataset,
    _evaluate_rankings,
    _group_indices,
    _learned_ccts_scores,
    _native_scores,
    _retrieval_scores,
    _with_training_labels,
)


SCHEMA_VERSION = "route_block_value_final_rerank_replay.v1"


def replay_route_block_value_model(
    *,
    pack_jsonl: Path,
    model_pickle: Path,
    output_json: Path,
    output_md: Path | None = None,
    split: str = "test",
    positive_task: str | None = None,
    negative_task: str | None = None,
    min_mrr_delta_vs_retrieval: float = 0.03,
) -> dict[str, Any]:
    model_payload = _read_pickle(model_pickle)
    metadata = model_payload.get("metadata") if isinstance(model_payload.get("metadata"), dict) else {}
    positive_task = positive_task or str(metadata.get("positive_task") or "")
    negative_task = negative_task if negative_task is not None else metadata.get("negative_task")
    if not positive_task:
        raise ValueError("positive_task must be supplied or present in the model metadata")

    rows = [row for row in _read_jsonl(pack_jsonl) if str(row.get("split") or "") == str(split)]
    if not rows:
        raise ValueError(f"no rows for split {split!r}: {pack_jsonl}")
    labeled_rows = _with_training_labels(rows, positive_task=positive_task, negative_task=negative_task)
    feature_names = list(model_payload.get("feature_names") or [])
    if not feature_names:
        raise ValueError(f"model has no feature_names: {model_pickle}")
    dataset = _dataset(labeled_rows, feature_names)

    model = model_payload["model"]
    mean = np.asarray(model_payload["mean"], dtype=np.float32)
    std = np.asarray(model_payload["std"], dtype=np.float32)
    methods = {
        "native_rank": _native_scores(dataset),
        "retrieval_only": _retrieval_scores(dataset),
        "audit_guard": _audit_scores(dataset),
        "learned_ccts_only": _learned_ccts_scores(dataset),
        "route_block_value_model": model.decision_function((dataset["x"] - mean) / std),
    }
    metrics = {name: _method_report(dataset, scores, native_scores=methods["native_rank"]) for name, scores in methods.items()}
    deltas = {
        "model_minus_native_mrr": _delta(metrics, "route_block_value_model", "native_rank"),
        "model_minus_retrieval_mrr": _delta(metrics, "route_block_value_model", "retrieval_only"),
        "model_minus_audit_mrr": _delta(metrics, "route_block_value_model", "audit_guard"),
    }
    gate = {
        "beats_retrieval_by_margin": {
            "ok": deltas["model_minus_retrieval_mrr"] >= float(min_mrr_delta_vs_retrieval),
            "actual": deltas["model_minus_retrieval_mrr"],
            "required_min": float(min_mrr_delta_vs_retrieval),
        },
        "beats_audit_guard": {
            "ok": deltas["model_minus_audit_mrr"] > 0.0,
            "actual": deltas["model_minus_audit_mrr"],
            "required_min": 0.0,
        },
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "pack_jsonl": str(pack_jsonl),
            "model_pickle": str(model_pickle),
            "split": split,
            "positive_task": positive_task,
            "negative_task": negative_task,
            "feature_count": len(feature_names),
            "min_mrr_delta_vs_retrieval": float(min_mrr_delta_vs_retrieval),
            "replay_contract": "fixed-pool final rerank only; does not prove search-time live-search promotion",
        },
        "counts": {
            "rows": len(dataset["rows"]),
            "groups": len(set(dataset["group_ids"])),
            "positive_rows": int(np.sum(dataset["y"] > 0)),
            "negative_rows": int(np.sum(dataset["y"] == 0)),
        },
        "metrics": metrics,
        "deltas": deltas,
        "gates": gate,
        "decision": {
            "fixed_pool_final_rerank_passed": bool(gate["beats_retrieval_by_margin"]["ok"]),
            "promote_search_time": False,
            "reason": (
                "final rerank replay is positive"
                if gate["beats_retrieval_by_margin"]["ok"]
                else "final rerank replay does not beat retrieval-only by the configured margin"
            )
            + "; live-search promotion still requires runtime hard-negative and guarded search quality lift",
        },
    }
    output_json = Path(output_json)
    output_md = Path(output_md) if output_md else output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(result), encoding="utf-8")
    return result


def _method_report(dataset: dict[str, Any], scores: np.ndarray, *, native_scores: np.ndarray) -> dict[str, Any]:
    report = _evaluate_rankings(dataset, scores)
    tops = _top_indices(dataset, scores)
    native_tops = _top_indices(dataset, native_scores)
    positive_tops = sum(1 for idx in tops.values() if int(dataset["y"][idx]) > 0)
    changed = sum(1 for group, idx in tops.items() if native_tops.get(group) != idx)
    return {
        **report,
        "top_positive_rate": round(positive_tops / len(tops), 6) if tops else 0.0,
        "top_positive_count": int(positive_tops),
        "top_route_changed_vs_native": int(changed),
    }


def _top_indices(dataset: dict[str, Any], scores: np.ndarray) -> dict[str, int]:
    tops: dict[str, int] = {}
    for group_id, indices in _group_indices(dataset).items():
        tops[group_id] = sorted(
            indices,
            key=lambda idx: (-float(scores[idx]), int(dataset["rows"][idx].get("native_rank") or 10**9)),
        )[0]
    return tops


def _delta(metrics: dict[str, Any], left: str, right: str) -> float:
    return round(
        float((metrics.get(left) or {}).get("mrr_covered") or 0.0)
        - float((metrics.get(right) or {}).get("mrr_covered") or 0.0),
        6,
    )


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Route/Block Value Final Rerank Replay",
        "",
        f"Decision: `{'pass' if result['decision']['fixed_pool_final_rerank_passed'] else 'fail'}`",
        "",
        result["decision"]["reason"],
        "",
        "## Metrics",
        "",
        "| method | MRR | R@1 all | R@3 all | top positive | changed vs native |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, report in (result.get("metrics") or {}).items():
        recall = report.get("recall_at_k_all") or {}
        lines.append(
            f"| `{name}` | {_fmt(report.get('mrr_covered'))} | {_fmt(recall.get('1'))} | "
            f"{_fmt(recall.get('3'))} | {_fmt(report.get('top_positive_rate'))} | "
            f"{report.get('top_route_changed_vs_native')} |"
        )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| item | value |",
            "|---|---:|",
            f"| model - native MRR | {_fmt(result['deltas'].get('model_minus_native_mrr'))} |",
            f"| model - retrieval MRR | {_fmt(result['deltas'].get('model_minus_retrieval_mrr'))} |",
            f"| model - audit MRR | {_fmt(result['deltas'].get('model_minus_audit_mrr'))} |",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _read_pickle(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"expected pickle payload dict: {path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay route/block value model as fixed-pool final reranker")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--model-pickle", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md")
    ap.add_argument("--split", default="test")
    ap.add_argument("--positive-task")
    ap.add_argument("--negative-task")
    ap.add_argument("--min-mrr-delta-vs-retrieval", type=float, default=0.03)
    args = ap.parse_args()
    report = replay_route_block_value_model(
        pack_jsonl=Path(args.pack),
        model_pickle=Path(args.model_pickle),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        split=args.split,
        positive_task=args.positive_task,
        negative_task=args.negative_task,
        min_mrr_delta_vs_retrieval=args.min_mrr_delta_vs_retrieval,
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a lightweight ranker for GraphFP + dual-tower fused candidates."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RDLogger.DisableLog("rdApp.*")
SCHEMA_VERSION = "fusion_lightgbm_reranker.v1"
EVAL_KS = (1, 3, 5, 10, 20, 50, 75, 100)
NUMERIC_FEATURES = [
    "rank",
    "fusion_score",
    "has_graphfp",
    "has_dualtower",
    "has_both_sources",
    "graphfp_rank",
    "dualtower_rank",
    "min_rank",
    "max_rank",
    "graphfp_inverse_rank",
    "dualtower_inverse_rank",
    "rrf_score",
    "graphfp_missing",
    "dualtower_missing",
    "rank_gap",
]


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    train_rows = _load_rows(args.train_jsonl)
    valid_rows = _load_rows(args.valid_jsonl)
    test_rows = _load_rows(args.test_jsonl) if args.test_jsonl else []

    x_train, y_train, group_train = _matrix(train_rows, n_bits=args.n_bits)
    x_valid, y_valid, group_valid = _matrix(valid_rows, n_bits=args.n_bits)
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )
    ranker.fit(
        x_train,
        y_train,
        group=group_train,
        eval_set=[(x_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[1, 5, 10, 20, 50],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    valid_pred = ranker.predict(x_valid)
    blend_grid = _select_blend_alpha(valid_rows, valid_pred)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "train_jsonl": str(args.train_jsonl),
            "valid_jsonl": str(args.valid_jsonl),
            "test_jsonl": str(args.test_jsonl) if args.test_jsonl else None,
            "n_bits": args.n_bits,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_child_samples": args.min_child_samples,
        },
        "data": {
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "test_rows": len(test_rows),
            "train_groups": len(group_train),
            "valid_groups": len(group_valid),
            "train_positives": int(y_train.sum()),
            "valid_positives": int(y_valid.sum()),
        },
        "valid": {
            "model_only": _evaluate_rows(valid_rows, valid_pred),
            "blend_grid": blend_grid,
        },
        "feature_importance": _feature_importance(ranker, args.n_bits),
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    if test_rows:
        x_test, _y_test, _group_test = _matrix(test_rows, n_bits=args.n_bits)
        test_pred = ranker.predict(x_test)
        report["test"] = {
            "model_only": _evaluate_rows(test_rows, test_pred),
            "best_valid_blend": _evaluate_rows(test_rows, test_pred, blend_alpha=blend_grid["best_alpha"]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "fusion_lightgbm_reranker.pkl").open("wb") as handle:
        pickle.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "model": ranker,
                "n_bits": args.n_bits,
                "numeric_features": NUMERIC_FEATURES,
                "best_blend_alpha": blend_grid["best_alpha"],
            },
            handle,
        )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "valid": report["valid"], "test": report.get("test")}, indent=2, ensure_ascii=False))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _matrix(rows: list[dict[str, Any]], *, n_bits: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    x_rows = []
    y_rows = []
    group_sizes = []
    current = None
    for row in rows:
        group_id = str(row.get("group_id") or row.get("idx"))
        if group_id != current:
            group_sizes.append(0)
            current = group_id
        group_sizes[-1] += 1
        x_rows.append(_row_features(row, n_bits=n_bits))
        y_rows.append(1.0 if row.get("exact") else 0.0)
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32), group_sizes


def _row_features(row: dict[str, Any], *, n_bits: int) -> np.ndarray:
    product_fp = _morgan_fp(row.get("product"), n_bits=n_bits)
    reactant_fp = _morgan_fp(row.get("reactants_text"), n_bits=n_bits)
    shared = np.logical_and(product_fp > 0, reactant_fp > 0).astype(np.float32)
    graphfp_rank = _rank_value(row.get("graphfp_rank"))
    dualtower_rank = _rank_value(row.get("dualtower_rank"))
    has_graphfp = float(row.get("graphfp_rank") is not None)
    has_dualtower = float(row.get("dualtower_rank") is not None)
    min_rank = min(graphfp_rank, dualtower_rank)
    max_rank = max(graphfp_rank if has_graphfp else 0.0, dualtower_rank if has_dualtower else 0.0)
    numeric = np.asarray(
        [
            float(row.get("rank") or 0.0),
            float(row.get("fusion_score") or 0.0),
            has_graphfp,
            has_dualtower,
            float(has_graphfp and has_dualtower),
            graphfp_rank,
            dualtower_rank,
            min_rank,
            max_rank,
            1.0 / graphfp_rank if has_graphfp else 0.0,
            1.0 / dualtower_rank if has_dualtower else 0.0,
            (1.0 / (60.0 + graphfp_rank) if has_graphfp else 0.0)
            + (1.0 / (60.0 + dualtower_rank) if has_dualtower else 0.0),
            float(not has_graphfp),
            float(not has_dualtower),
            abs(graphfp_rank - dualtower_rank) if has_graphfp and has_dualtower else 100.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([product_fp, reactant_fp, shared, numeric])


def _rank_value(value: Any) -> float:
    if value is None:
        return 1000.0
    try:
        return max(float(value), 1.0)
    except (TypeError, ValueError):
        return 1000.0


def _morgan_fp(smiles: str | None, *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits, useChirality=True)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _evaluate_rows(rows: list[dict[str, Any]], pred: np.ndarray, *, blend_alpha: float | None = None) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, pred):
        grouped[str(row.get("group_id") or row.get("idx"))].append((row, float(score)))
    baseline_hits = {k: 0 for k in EVAL_KS}
    rerank_hits = {k: 0 for k in EVAL_KS}
    reciprocal_sum = 0.0
    oracle = 0
    for group_rows in grouped.values():
        baseline = sorted(group_rows, key=lambda item: int(item[0].get("rank") or 10**9))
        oracle += int(any(bool(row.get("exact")) for row, _ in baseline))
        reranked = sorted(group_rows, key=lambda item: _combined_score(item[0], item[1], blend_alpha), reverse=True)
        for k in EVAL_KS:
            baseline_hits[k] += int(any(bool(row.get("exact")) for row, _ in baseline[:k]))
            rerank_hits[k] += int(any(bool(row.get("exact")) for row, _ in reranked[:k]))
        for rank, (row, _score) in enumerate(reranked, 1):
            if row.get("exact"):
                reciprocal_sum += 1.0 / rank
                break
    n = len(grouped)
    return {
        "n_groups": n,
        "oracle": _rate(oracle, n),
        "baseline": {f"exact@{k}": _rate(baseline_hits[k], n) for k in EVAL_KS},
        "reranked": {f"exact@{k}": _rate(rerank_hits[k], n) for k in EVAL_KS},
        "reranked_mrr": round(reciprocal_sum / max(n, 1), 6),
        "blend_alpha": blend_alpha,
    }


def _combined_score(row: dict[str, Any], model_score: float, blend_alpha: float | None) -> float:
    if blend_alpha is None:
        return float(model_score)
    return float(model_score) + float(blend_alpha) * float(row.get("fusion_score") or 0.0)


def _select_blend_alpha(rows: list[dict[str, Any]], pred: np.ndarray) -> dict[str, Any]:
    alphas = [None, 0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0]
    evaluated = [_evaluate_rows(rows, pred, blend_alpha=alpha) for alpha in alphas]
    best = max(
        evaluated,
        key=lambda row: (
            row["reranked"]["exact@20"]["count"],
            row["reranked"]["exact@10"]["count"],
            row["reranked"]["exact@5"]["count"],
            row["reranked_mrr"],
        ),
    )
    return {"best_alpha": best["blend_alpha"], "best_metrics": best, "all": evaluated}


def _feature_importance(model: lgb.LGBMRanker, n_bits: int) -> list[dict[str, Any]]:
    names = (
        [f"product_fp_{idx}" for idx in range(n_bits)]
        + [f"reactant_fp_{idx}" for idx in range(n_bits)]
        + [f"shared_fp_{idx}" for idx in range(n_bits)]
        + list(NUMERIC_FEATURES)
    )
    importances = model.booster_.feature_importance(importance_type="gain")
    rows = [
        {"feature": name, "gain": round(float(value), 6)}
        for name, value in zip(names, importances)
        if float(value) > 0
    ]
    rows.sort(key=lambda row: row["gain"], reverse=True)
    return rows[:60]


def _rate(value: int, total: int) -> dict[str, Any]:
    return {"count": int(value), "rate": round(float(value) / max(int(total), 1), 6)}


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fusion LightGBM Reranker",
        "",
        "## Data",
        "",
        "| split | rows | groups | positives |",
        "| --- | ---: | ---: | ---: |",
        f"| train | {report['data']['train_rows']} | {report['data']['train_groups']} | {report['data']['train_positives']} |",
        f"| valid | {report['data']['valid_rows']} | {report['data']['valid_groups']} | {report['data']['valid_positives']} |",
    ]
    if report["data"].get("test_rows"):
        lines.append(f"| test | {report['data']['test_rows']} | - | - |")
    lines.extend(["", "## Valid Best Blend", "", _metrics_table(report["valid"]["blend_grid"]["best_metrics"])])
    if report.get("test"):
        lines.extend(["", "## Test Best Valid Blend", "", _metrics_table(report["test"]["best_valid_blend"])])
    lines.extend(["", "## Top Features", "", "| feature | gain |", "| --- | ---: |"])
    for row in report.get("feature_importance") or []:
        lines.append(f"| `{row['feature']}` | {row['gain']} |")
    return "\n".join(lines)


def _metrics_table(metrics: dict[str, Any]) -> str:
    lines = ["| metric | baseline | reranked |", "| --- | ---: | ---: |"]
    for k in EVAL_KS:
        key = f"exact@{k}"
        lines.append(
            f"| {key} | {metrics['baseline'][key]['count']} ({metrics['baseline'][key]['rate']}) | "
            f"{metrics['reranked'][key]['count']} ({metrics['reranked'][key]['rate']}) |"
        )
    lines.append(f"| MRR | - | {metrics['reranked_mrr']} |")
    lines.append(f"| blend_alpha | - | {metrics['blend_alpha']} |")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bits", type=int, default=256)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()

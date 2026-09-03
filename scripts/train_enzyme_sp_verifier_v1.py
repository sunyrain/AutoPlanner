"""Train enzyme substrate-product-EC verifier v1.

The first production verifier is a small LightGBM model.  It is deliberately
fast enough to run during route-search filtering while still seeing the three
pieces that matter for enzyme-step plausibility: substrate side, product side,
and EC evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


from cascade_planner.cascade_search.enzyme_sp_features import (
    EC1_VALUES,
    FEATURE_NAMES,
    FP_BITS,
    NUMERIC_FEATURES,
    SideFeatureCache,
    build_matrix,
)


def load_split(path: Path, *, limit: int | None, seed: int) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    if limit is not None and len(rows) > limit:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(rows)), limit))
        rows = [rows[idx] for idx in indices]
    return rows


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, score))


def select_threshold(y_true: np.ndarray, score: np.ndarray, *, target_precision: float) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    candidates = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= target_precision:
            candidates.append((float(r), float(p), float(t)))
    if candidates:
        recall_value, precision_value, threshold = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
        return {
            "mode": "target_precision",
            "target_precision": target_precision,
            "threshold": threshold,
            "precision": precision_value,
            "recall": recall_value,
        }
    best = {"f1": -1.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "threshold": float(t), "precision": float(p), "recall": float(r)}
    return {"mode": "best_f1_fallback", "target_precision": target_precision, **best}


def binary_metrics(y_true: np.ndarray, score: np.ndarray, *, threshold: float) -> dict[str, Any]:
    pred = (score >= threshold).astype(np.int8)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, labels=[1], zero_division=0)
    return {
        "roc_auc": safe_auc(y_true, score),
        "pr_auc": safe_ap(y_true, score),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision[0]) if len(precision) else 0.0,
        "recall": float(recall[0]) if len(recall) else 0.0,
        "f1": float(f1[0]) if len(f1) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "positive_rate": float(np.mean(y_true)),
        "score_mean": float(np.mean(score)),
    }


def by_label_type(y_true: np.ndarray, score: np.ndarray, label_types: list[str], *, threshold: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, label_type in enumerate(label_types):
        grouped[label_type].append(idx)
    out: dict[str, dict[str, Any]] = {}
    for label_type, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int32)
        y = y_true[idx]
        s = score[idx]
        pred = (s >= threshold).astype(np.int8)
        row = {
            "rows": int(len(idx)),
            "positives": int(np.sum(y == 1)),
            "negatives": int(np.sum(y == 0)),
            "mean_score": float(np.mean(s)),
            "median_score": float(np.median(s)),
            "accepted_at_threshold": int(np.sum(pred == 1)),
        }
        if row["negatives"]:
            row["negative_rejection_rate"] = float(np.mean(s < threshold))
            row["false_positive_rate"] = float(np.mean(s >= threshold))
        if row["positives"]:
            row["positive_recall"] = float(np.mean(s >= threshold))
        out[label_type] = row
    return out


def feature_importance(model: lgb.LGBMClassifier, top_n: int = 50) -> list[dict[str, Any]]:
    importances = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(importances)[::-1][:top_n]
    return [
        {"feature": FEATURE_NAMES[int(idx)], "gain": float(importances[int(idx)])}
        for idx in order
        if importances[int(idx)] > 0
    ]


def write_scores(path: Path, rows: list[dict[str, Any]], scores: np.ndarray, *, threshold: float) -> None:
    out = []
    for row, score in zip(rows, scores):
        out.append(
            {
                "row_id": row.get("row_id") or "",
                "reaction_id": row.get("reaction_id") or "",
                "source_reaction_id": row.get("source_reaction_id") or "",
                "label": int(row.get("label") or 0),
                "label_type": row.get("label_type") or "",
                "ec1": row.get("ec1") or "",
                "ec2": row.get("ec2") or "",
                "verifier_score": float(score),
                "accepted_at_threshold": bool(float(score) >= threshold),
            }
        )
    pq.write_table(pa.Table.from_pylist(out), path)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Enzyme Substrate-Product Verifier v1 Training Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- data_dir: `{report['data_dir']}`",
        f"- output_dir: `{report['output_dir']}`",
        f"- selected_threshold: `{report['threshold']['threshold']:.6f}` ({report['threshold']['mode']})",
        "",
        "## Data",
        "",
        "| Split | Rows | Positives | Negatives |",
        "|---|---:|---:|---:|",
    ]
    for split in ["train", "valid", "test"]:
        row = report["data"][split]
        lines.append(f"| `{split}` | {row['rows']} | {row['positives']} | {row['negatives']} |")
    lines.extend(["", "## Metrics", "", "| Split | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |", "|---|---:|---:|---:|---:|---:|---:|"])
    for split in ["valid", "test"]:
        for metric_name, label in [("at_0_5", "@0.5"), ("at_selected_threshold", "@selected")]:
            metric = report["metrics"][split][metric_name]
            lines.append(
                f"| {split} {label} | {fmt(metric.get('roc_auc'))} | {fmt(metric.get('pr_auc'))} | "
                f"{fmt(metric.get('precision'))} | {fmt(metric.get('recall'))} | {fmt(metric.get('f1'))} | {fmt(metric.get('accuracy'))} |"
            )
    lines.extend(["", "## Test Label Type Behavior", "", "| Label Type | Rows | Mean Score | Recall/Rejection | Accepted |", "|---|---:|---:|---:|---:|"])
    for label_type, row in sorted(report["metrics"]["test"]["by_label_type"].items()):
        if row.get("positives", 0):
            behavior = row.get("positive_recall")
        else:
            behavior = row.get("negative_rejection_rate")
        lines.append(
            f"| `{label_type}` | {row['rows']} | {fmt(row['mean_score'])} | "
            f"{fmt(behavior)} | {row['accepted_at_threshold']} |"
        )
    lines.extend(["", "## Top Feature Importance", "", "| Feature | Gain |", "|---|---:|"])
    for row in report["feature_importance_top"]:
        lines.append(f"| `{row['feature']}` | {row['gain']:.3f} |")
    lines.extend(["", "## Files", "", "| Artifact | Path |", "|---|---|"])
    for key, value in report["files"].items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train enzyme substrate-product verifier v1")
    parser.add_argument("--data-dir", default="data/enzyme_sp_verifier_v1")
    parser.add_argument("--output-dir", default="results/shared/enzyme_sp_verifier_v1_20260528")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=450)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_split(data_dir / "train.parquet", limit=args.train_limit, seed=args.seed)
    valid_rows = load_split(data_dir / "valid.parquet", limit=args.valid_limit, seed=args.seed + 1)
    test_rows = load_split(data_dir / "test.parquet", limit=args.test_limit, seed=args.seed + 2)

    cache = SideFeatureCache()
    print(f"[enzyme_sp_verifier_v1] build train matrix rows={len(train_rows)}")
    x_train, y_train, w_train, train_types, train_ids = build_matrix(train_rows, cache)
    print(f"[enzyme_sp_verifier_v1] build valid matrix rows={len(valid_rows)}")
    x_valid, y_valid, w_valid, valid_types, valid_ids = build_matrix(valid_rows, cache)
    print(f"[enzyme_sp_verifier_v1] build test matrix rows={len(test_rows)}")
    x_test, y_test, w_test, test_types, test_ids = build_matrix(test_rows, cache)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1,
    )
    print("[enzyme_sp_verifier_v1] train LightGBM")
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(x_valid, y_valid)],
        eval_sample_weight=[w_valid],
        eval_metric=["auc", "average_precision", "binary_logloss"],
        callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(period=25)],
    )

    valid_score = model.predict_proba(x_valid)[:, 1]
    test_score = model.predict_proba(x_test)[:, 1]
    threshold = select_threshold(y_valid, valid_score, target_precision=args.target_precision)
    selected = float(threshold["threshold"])

    metrics = {
        "valid": {
            "at_0_5": binary_metrics(y_valid, valid_score, threshold=0.5),
            "at_selected_threshold": binary_metrics(y_valid, valid_score, threshold=selected),
            "by_label_type": by_label_type(y_valid, valid_score, valid_types, threshold=selected),
        },
        "test": {
            "at_0_5": binary_metrics(y_test, test_score, threshold=0.5),
            "at_selected_threshold": binary_metrics(y_test, test_score, threshold=selected),
            "by_label_type": by_label_type(y_test, test_score, test_types, threshold=selected),
        },
    }

    files = {
        "model": str(out_dir / "enzyme_sp_verifier_v1_lgbm.joblib"),
        "report_json": str(out_dir / "enzyme_sp_verifier_v1_report.json"),
        "report_md": str(out_dir / "enzyme_sp_verifier_v1_report.md"),
        "feature_schema": str(out_dir / "feature_schema.json"),
        "valid_scores": str(out_dir / "valid_scores.parquet"),
        "test_scores": str(out_dir / "test_scores.parquet"),
    }
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "numeric_features": NUMERIC_FEATURES,
            "ec1_values": EC1_VALUES,
            "fp_bits": FP_BITS,
            "threshold": threshold,
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
            "schema_version": "enzyme_sp_verifier_v1.lgbm.v1",
        },
        files["model"],
    )
    Path(files["feature_schema"]).write_text(
        json.dumps(
            {
                "schema_version": "enzyme_sp_verifier_v1.features.v1",
                "feature_names": FEATURE_NAMES,
                "numeric_features": NUMERIC_FEATURES,
                "ec1_values": EC1_VALUES,
                "fp_bits": FP_BITS,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_scores(Path(files["valid_scores"]), valid_rows, valid_score, threshold=selected)
    write_scores(Path(files["test_scores"]), test_rows, test_score, threshold=selected)

    report = {
        "schema_version": "enzyme_sp_verifier_v1.training.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "data": {
            "train": {"rows": len(train_rows), "positives": int(np.sum(y_train == 1)), "negatives": int(np.sum(y_train == 0))},
            "valid": {"rows": len(valid_rows), "positives": int(np.sum(y_valid == 1)), "negatives": int(np.sum(y_valid == 0))},
            "test": {"rows": len(test_rows), "positives": int(np.sum(y_test == 1)), "negatives": int(np.sum(y_test == 0))},
            "feature_count": len(FEATURE_NAMES),
            "molecule_cache_size": len(cache.cache),
            "invalid_molecules": dict(cache.invalid),
        },
        "training": {
            "n_estimators_requested": args.n_estimators,
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "early_stopping_rounds": args.early_stopping_rounds,
            "target_precision": args.target_precision,
            "seed": args.seed,
            "train_limit": args.train_limit,
            "valid_limit": args.valid_limit,
            "test_limit": args.test_limit,
        },
        "threshold": threshold,
        "metrics": metrics,
        "feature_importance_top": feature_importance(model),
        "files": files,
    }
    Path(files["report_json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(files["report_md"]).write_text(render_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "best_iteration": report["training"]["best_iteration"],
                "threshold": threshold,
                "valid_selected": report["metrics"]["valid"]["at_selected_threshold"],
                "test_selected": report["metrics"]["test"]["at_selected_threshold"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

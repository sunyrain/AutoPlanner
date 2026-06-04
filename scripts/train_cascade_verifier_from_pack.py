#!/usr/bin/env python3
"""Train a lightweight learned verifier from a cascade perturbation pack."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support

from cascade_planner.cascade_verifier import cascade_verifier_features


REASON_LABELS = [
    "atom_balance_violation",
    "temperature_conflict",
    "ph_conflict",
    "solvent_conflict",
    "enzyme_toxicity",
    "cofactor_ledger_gap",
    "route_order_mismatch",
]


def main() -> None:
    args = _parse_args()
    result = train_from_pack(args.input, model_output=args.model_output, report_output=args.report_output)
    if args.markdown:
        _write_markdown(result, args.markdown)
    print(json.dumps(result["summary"], indent=2))


def train_from_pack(pack_path: Path, *, model_output: Path, report_output: Path) -> dict[str, Any]:
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    examples = [row for row in pack.get("examples") or [] if isinstance(row, dict)]
    if not examples:
        raise ValueError(f"no examples in {pack_path}")

    splits = _split_indices(examples)
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(cascade_verifier_features(examples[idx]) for idx in splits["train"])
    x_val = vectorizer.transform(cascade_verifier_features(examples[idx]) for idx in splits["val"])
    x_test = vectorizer.transform(cascade_verifier_features(examples[idx]) for idx in splits["test"])

    y_feasible_train = np.asarray([int(examples[idx].get("label") == 1) for idx in splits["train"]])
    y_feasible_val = np.asarray([int(examples[idx].get("label") == 1) for idx in splits["val"]])
    y_feasible_test = np.asarray([int(examples[idx].get("label") == 1) for idx in splits["test"]])
    feasible_model = _fit_binary(x_train, y_feasible_train)

    reason_models: dict[str, Any] = {}
    reason_metrics: dict[str, Any] = {}
    y_reason_test_rows: list[list[int]] = []
    y_reason_pred_rows: list[list[int]] = []
    for reason in REASON_LABELS:
        y_train = np.asarray([int(reason in (examples[idx].get("expected_failure_reasons") or [])) for idx in splits["train"]])
        y_test = np.asarray([int(reason in (examples[idx].get("expected_failure_reasons") or [])) for idx in splits["test"]])
        model = _fit_binary(x_train, y_train)
        pred = model.predict(x_test)
        reason_models[reason] = model
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            pred,
            average="binary",
            zero_division=0,
        )
        reason_metrics[reason] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(y_test.sum()),
        }
        y_reason_test_rows.append(y_test.tolist())
        y_reason_pred_rows.append(pred.tolist())

    feasible_val_scores = _positive_scores(feasible_model, x_val)
    feasible_test_scores = _positive_scores(feasible_model, x_test)
    feasible_val_pred = (feasible_val_scores >= 0.5).astype(int)
    feasible_test_pred = (feasible_test_scores >= 0.5).astype(int)
    calibration = _feasibility_threshold_calibration(
        y_feasible_val,
        feasible_val_scores,
        y_feasible_test,
        feasible_test_scores,
    )
    y_reason_test = np.asarray(y_reason_test_rows, dtype=int).T if y_reason_test_rows else np.zeros((len(splits["test"]), 0))
    y_reason_pred = np.asarray(y_reason_pred_rows, dtype=int).T if y_reason_pred_rows else np.zeros((len(splits["test"]), 0))

    micro_f1 = f1_score(y_reason_test, y_reason_pred, average="micro", zero_division=0) if y_reason_test.size else 0.0
    macro_f1 = f1_score(y_reason_test, y_reason_pred, average="macro", zero_division=0) if y_reason_test.size else 0.0
    summary = {
        "schema_version": "learned_cascade_verifier_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(pack_path),
        "n_examples": len(examples),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "feature_dim": int(len(vectorizer.feature_names_)),
        "feasibility": {
            "val_accuracy": round(float(accuracy_score(y_feasible_val, feasible_val_pred)), 4) if len(y_feasible_val) else None,
            "test_accuracy": round(float(accuracy_score(y_feasible_test, feasible_test_pred)), 4) if len(y_feasible_test) else None,
            "test_report": classification_report(
                y_feasible_test,
                feasible_test_pred,
                target_names=["infeasible", "feasible"],
                output_dict=True,
                zero_division=0,
            ),
            "threshold_calibration": calibration,
        },
        "reasons": {
            "micro_f1": round(float(micro_f1), 4),
            "macro_f1": round(float(macro_f1), 4),
            "per_reason": reason_metrics,
        },
        "contract": "Learned verifier trained on rule-derived perturbation labels; not an expert feasibility model.",
    }
    artifact = {
        "vectorizer": vectorizer,
        "feasible_model": feasible_model,
        "reason_models": reason_models,
        "reason_labels": REASON_LABELS,
        "summary": summary,
        "recommended_feasible_threshold": calibration.get("recommended_threshold"),
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output)
    result = {"summary": summary}
    report_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _positive_scores(model: Any, x: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return np.asarray(proba[:, classes.index(1)], dtype=float)
        if classes:
            constant = 1.0 if int(classes[0]) == 1 else 0.0
            return np.full(proba.shape[0], constant, dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def _feasibility_threshold_calibration(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, Any]:
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    rows = []
    for threshold in thresholds:
        rows.append(_threshold_row(threshold, y_val, val_scores, y_test, test_scores))

    target_precision = 0.9
    eligible = [
        row for row in rows
        if row["val_precision"] is not None
        and row["val_precision"] >= target_precision
        and row["val_predicted_positive"] > 0
    ]
    if eligible:
        chosen = max(eligible, key=lambda row: (row["val_recall"] or 0.0, row["test_precision"] or 0.0, -row["threshold"]))
    else:
        nonempty = [row for row in rows if row["val_predicted_positive"] > 0]
        chosen = max(nonempty or rows, key=lambda row: (row["val_precision"] or 0.0, row["val_recall"] or 0.0))

    return {
        "target_val_precision": target_precision,
        "recommended_threshold": chosen["threshold"],
        "recommended_policy": (
            "Use learned verifier as a conservative promotion/gate signal only "
            "when feasible_probability >= recommended_threshold; below that, keep it experimental."
        ),
        "operating_points": rows,
    }


def _threshold_row(
    threshold: float,
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, Any]:
    val = _precision_recall_at_threshold(y_val, val_scores, threshold)
    test = _precision_recall_at_threshold(y_test, test_scores, threshold)
    return {
        "threshold": float(threshold),
        "val_precision": val["precision"],
        "val_recall": val["recall"],
        "val_predicted_positive": val["predicted_positive"],
        "val_true_positive": val["true_positive"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_predicted_positive": test["predicted_positive"],
        "test_true_positive": test["true_positive"],
    }


def _precision_recall_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = scores >= float(threshold)
    predicted_positive = int(pred.sum())
    true_positive = int(((y_true == 1) & pred).sum())
    actual_positive = int((y_true == 1).sum())
    precision = round(float(true_positive / predicted_positive), 4) if predicted_positive else None
    recall = round(float(true_positive / actual_positive), 4) if actual_positive else None
    return {
        "precision": precision,
        "recall": recall,
        "predicted_positive": predicted_positive,
        "true_positive": true_positive,
    }


def _fit_binary(x_train: Any, y_train: np.ndarray) -> Any:
    if len(set(int(v) for v in y_train.tolist())) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y_train[0]) if len(y_train) else 0)
        model.fit(x_train, y_train)
        return model
    model = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
    model.fit(x_train, y_train)
    return model


def _split_indices(examples: list[dict[str, Any]]) -> dict[str, list[int]]:
    by_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, row in enumerate(examples):
        split = str(((row.get("cascade") or {}).get("metadata") or {}).get("split") or "").lower()
        if split in by_split:
            by_split[split].append(idx)
    if all(by_split.values()):
        return by_split

    groups = sorted({str(row.get("source_target_index")) for row in examples})
    train_groups = set(groups[: int(len(groups) * 0.7)])
    val_groups = set(groups[int(len(groups) * 0.7): int(len(groups) * 0.85)])
    out = {"train": [], "val": [], "test": []}
    for idx, row in enumerate(examples):
        group = str(row.get("source_target_index"))
        if group in train_groups:
            out["train"].append(idx)
        elif group in val_groups:
            out["val"].append(idx)
        else:
            out["test"].append(idx)
    return out


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Learned Cascade Verifier Report",
        "",
        f"- Examples: `{summary['n_examples']}`",
        f"- Feature dim: `{summary['feature_dim']}`",
        f"- Split counts: `{summary['split_counts']}`",
        f"- Feasibility test accuracy: `{summary['feasibility']['test_accuracy']}`",
        f"- Recommended feasible threshold: `{summary['feasibility']['threshold_calibration']['recommended_threshold']}`",
        f"- Reason micro F1: `{summary['reasons']['micro_f1']}`",
        f"- Reason macro F1: `{summary['reasons']['macro_f1']}`",
        "",
        "## Feasibility Thresholds",
        "",
        "| Threshold | Val precision | Val recall | Val positives | Test precision | Test recall | Test positives |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["feasibility"]["threshold_calibration"]["operating_points"]:
        lines.append(
            "| {threshold} | {val_precision} | {val_recall} | {val_predicted_positive} | {test_precision} | {test_recall} | {test_predicted_positive} |".format(
                threshold=row["threshold"],
                val_precision=row["val_precision"],
                val_recall=row["val_recall"],
                val_predicted_positive=row["val_predicted_positive"],
                test_precision=row["test_precision"],
                test_recall=row["test_recall"],
                test_predicted_positive=row["test_predicted_positive"],
            )
        )
    lines.extend([
        "",
        "## Per Reason",
        "",
        "| Reason | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for reason, row in summary["reasons"]["per_reason"].items():
        lines.append(f"| `{reason}` | {row['precision']} | {row['recall']} | {row['f1']} | {row['support']} |")
    lines.extend(["", "## Contract", "", summary["contract"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learned cascade verifier from perturbation pack")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()

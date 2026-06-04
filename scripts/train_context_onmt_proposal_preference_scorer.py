#!/usr/bin/env python
"""Train a lightweight no-expert scorer from context-ONMT proposal preferences."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascade_search.proposal_preference import (  # noqa: E402
    ELEMENTS,
    candidate_vector,
    feature_names,
    is_tie,
)

SCHEMA_VERSION = "context_onmt_proposal_preference_scorer.v1"


def train_preference_scorer(
    *,
    train_jsonl: Path,
    valid_jsonl: Path,
    test_jsonl: Path | None,
    output_dir: Path,
    n_bits: int = 128,
    max_iter: int = 1000,
    c_value: float = 1.0,
) -> dict[str, Any]:
    train_pairs = _read_jsonl(train_jsonl)
    valid_pairs = _read_jsonl(valid_jsonl)
    test_pairs = _read_jsonl(test_jsonl) if test_jsonl else []
    if not train_pairs:
        raise ValueError(f"no train preference pairs in {train_jsonl}")
    if not valid_pairs:
        raise ValueError(f"no valid preference pairs in {valid_jsonl}")

    x_train, y_train, train_meta = _pointwise_matrix(train_pairs, n_bits=n_bits)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, C=c_value, solver="liblinear", random_state=0),
    )
    model.fit(x_train, y_train)

    split_metrics = {
        "train": _evaluate_split(model, train_pairs, n_bits=n_bits),
        "valid": _evaluate_split(model, valid_pairs, n_bits=n_bits),
    }
    if test_pairs:
        split_metrics["test"] = _evaluate_split(model, test_pairs, n_bits=n_bits)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_output = output_dir / "context_onmt_proposal_preference_scorer.joblib"
    report_output = output_dir / "context_onmt_proposal_preference_scorer_report.json"
    markdown_output = output_dir / "context_onmt_proposal_preference_scorer_report.md"

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "n_bits": int(n_bits),
        "elements": list(ELEMENTS),
        "feature_names": feature_names(n_bits),
        "created_at": _utc_now(),
        "contract": (
            "Pointwise scorer trained from rule-generated proposal preference pairs. "
            "Chosen sides are clean seed reactants; rejected sides are hard negatives, not expert labels."
        ),
    }
    joblib.dump(artifact, model_output)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": artifact["created_at"],
        "train_jsonl": str(train_jsonl),
        "valid_jsonl": str(valid_jsonl),
        "test_jsonl": str(test_jsonl) if test_jsonl else None,
        "output_dir": str(output_dir),
        "model_output": str(model_output),
        "n_bits": int(n_bits),
        "feature_dim": int(x_train.shape[1]),
        "train_pointwise_examples": int(len(y_train)),
        "train_pair_negative_type_counts": dict(Counter(row.get("negative_type") for row in train_pairs)),
        "split_metrics": split_metrics,
        "decision": _decision(split_metrics),
        "contract": artifact["contract"],
    }
    report_output.write_text(json.dumps({"summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_output.write_text(_markdown(summary), encoding="utf-8")
    return {"summary": summary}


def score_candidate(artifact: dict[str, Any], *, product: str, reactants: str) -> float:
    from cascade_planner.cascade_search.proposal_preference import score_candidate as _score_candidate

    return _score_candidate(artifact, product=product, reactants=reactants)


def _pointwise_matrix(pairs: list[dict[str, Any]], *, n_bits: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    meta: list[dict[str, Any]] = []
    for pair in pairs:
        product = str(pair.get("product") or "")
        chosen = str(pair.get("chosen_reactants") or "")
        rejected = str(pair.get("rejected_reactants") or "")
        if not product or not chosen or not rejected:
            continue
        x_rows.append(candidate_vector(product, chosen, n_bits=n_bits))
        y_rows.append(1)
        meta.append({"pair_id": pair.get("pair_id"), "side": "chosen"})
        x_rows.append(candidate_vector(product, rejected, n_bits=n_bits))
        y_rows.append(0)
        meta.append({"pair_id": pair.get("pair_id"), "side": "rejected"})
    if not x_rows:
        raise ValueError("no usable pointwise examples")
    return np.vstack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.int64), meta


def _evaluate_split(model: Any, pairs: list[dict[str, Any]], *, n_bits: int) -> dict[str, Any]:
    x, y, meta = _pointwise_matrix(pairs, n_bits=n_bits)
    scores = _positive_scores(model, x)
    pred = (scores >= 0.5).astype(int)
    by_pair: dict[str, dict[str, Any]] = defaultdict(dict)
    for row, score, label, meta_row in zip(_pointwise_rows(pairs), scores.tolist(), y.tolist(), meta):
        item = by_pair[str(row.get("pair_id"))]
        item["negative_type"] = row.get("negative_type")
        item[meta_row["side"]] = float(score)
        item[f"{meta_row['side']}_label"] = int(label)
    margins: list[float] = []
    by_type: dict[str, list[float]] = defaultdict(list)
    for item in by_pair.values():
        if "chosen" not in item or "rejected" not in item:
            continue
        margin = float(item["chosen"] - item["rejected"])
        margins.append(margin)
        by_type[str(item.get("negative_type") or "unknown")].append(margin)
    auc = None
    if len(set(y.tolist())) > 1:
        auc = float(roc_auc_score(y, scores))
    out = {
        "n_pairs": int(len(margins)),
        "n_pointwise_examples": int(len(y)),
        "pointwise_accuracy": round(float(accuracy_score(y, pred)), 6),
        "pointwise_auc": round(float(auc), 6) if auc is not None else None,
        "pointwise_log_loss": round(float(log_loss(y, scores, labels=[0, 1])), 6),
        "pairwise_accuracy": round(float(np.mean([margin > 0 for margin in margins])), 6) if margins else 0.0,
        "pairwise_tie_rate": round(float(np.mean([is_tie(margin) for margin in margins])), 6) if margins else 0.0,
        "mean_pairwise_margin": round(float(np.mean(margins)), 6) if margins else 0.0,
        "median_pairwise_margin": round(float(np.median(margins)), 6) if margins else 0.0,
        "by_negative_type": {},
    }
    for neg_type, type_margins in sorted(by_type.items()):
        out["by_negative_type"][neg_type] = {
            "n_pairs": int(len(type_margins)),
            "pairwise_accuracy": round(float(np.mean([margin > 0 for margin in type_margins])), 6),
            "mean_margin": round(float(np.mean(type_margins)), 6),
        }
    return out


def _pointwise_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pair in pairs:
        out.append(pair)
        out.append(pair)
    return out


def _positive_scores(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(getattr(model[-1], "classes_", []))
    if 1 in classes:
        return np.asarray(proba[:, classes.index(1)], dtype=float)
    return np.asarray(proba[:, -1], dtype=float)


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _decision(split_metrics: dict[str, Any]) -> str:
    valid = split_metrics.get("valid") or {}
    test = split_metrics.get("test") or {}
    valid_pair = float(valid.get("pairwise_accuracy") or 0.0)
    test_pair = float(test.get("pairwise_accuracy") or valid_pair)
    valid_auc = float(valid.get("pointwise_auc") or 0.0)
    if valid_pair >= 0.75 and test_pair >= 0.75 and valid_auc >= 0.75:
        return "preference_signal_learned_offline_not_generator_promotion"
    if valid_pair >= 0.6:
        return "weak_preference_signal_needs_better_negatives_or_features"
    return "hold_no_reliable_preference_signal"


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Context ONMT Proposal Preference Scorer",
        "",
        f"created_at: `{summary['created_at']}`",
        f"decision: `{summary['decision']}`",
        "",
        "## Inputs",
        "",
        f"- train: `{summary['train_jsonl']}`",
        f"- valid: `{summary['valid_jsonl']}`",
        f"- test: `{summary.get('test_jsonl')}`",
        f"- model: `{summary['model_output']}`",
        "",
        "## Metrics",
        "",
        "| split | pairs | pointwise acc | AUC | pairwise acc | mean margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, metrics in (summary.get("split_metrics") or {}).items():
        lines.append(
            "| {split} | {pairs} | {pacc} | {auc} | {pairacc} | {margin} |".format(
                split=split,
                pairs=metrics.get("n_pairs"),
                pacc=metrics.get("pointwise_accuracy"),
                auc=metrics.get("pointwise_auc"),
                pairacc=metrics.get("pairwise_accuracy"),
                margin=metrics.get("mean_pairwise_margin"),
            )
        )
    lines.extend(["", "## By Negative Type", ""])
    for split, metrics in (summary.get("split_metrics") or {}).items():
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| type | pairs | pairwise acc | mean margin |")
        lines.append("| --- | ---: | ---: | ---: |")
        for neg_type, row in (metrics.get("by_negative_type") or {}).items():
            lines.append(f"| `{neg_type}` | {row.get('n_pairs')} | {row.get('pairwise_accuracy')} | {row.get('mean_margin')} |")
        lines.append("")
    lines.extend(["## Contract", "", summary["contract"], ""])
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--valid-jsonl", type=Path, required=True)
    ap.add_argument("--test-jsonl", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-bits", type=int, default=128)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--c-value", type=float, default=1.0)
    args = ap.parse_args()
    result = train_preference_scorer(
        train_jsonl=args.train_jsonl,
        valid_jsonl=args.valid_jsonl,
        test_jsonl=args.test_jsonl,
        output_dir=args.output_dir,
        n_bits=args.n_bits,
        max_iter=args.max_iter,
        c_value=args.c_value,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

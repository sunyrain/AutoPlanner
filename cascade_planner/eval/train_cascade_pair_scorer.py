"""Train a Stage-1 adjacent-step cascade pair scorer.

The model predicts local cascade compatibility between two adjacent process
steps.  It is trained on the pair pack produced from dataset_v4_release and is
intended to be used as a soft search reward, not as a route-level gold judge.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascade_search.pair_scorer import PAIR_LABEL_NAMES, build_pair_feature_schema, pair_feature_vector


@dataclass
class PairDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    groups: list[str]
    schema: dict[str, Any]


class PairScorerNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, output_dim: int = len(PAIR_LABEL_NAMES)):
        super().__init__()
        h2 = max(32, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, h2),
            nn.GELU(),
            nn.Linear(h2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_cascade_pair_scorer(
    *,
    pack_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 128,
    n_bits: int = 128,
    seed: int = 42,
    device: str | None = None,
    selection_metric: str = "pairwise_group_accuracy",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = _read_jsonl(pack_dir / "pair_value.jsonl")
    dataset = build_dataset(rows, n_bits=n_bits)
    if len(dataset.rows) < 8:
        raise ValueError(f"not enough pair rows to train: {len(dataset.rows)}")
    train_idx, val_idx, test_idx = _split_indices(dataset.rows)
    train_rows = [dataset.rows[idx] for idx in train_idx]
    val_rows = [dataset.rows[idx] for idx in val_idx]
    test_rows = [dataset.rows[idx] for idx in test_idx]
    x_train = torch.tensor(dataset.x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(dataset.y[train_idx], dtype=torch.float32)
    w_train = torch.tensor(dataset.weights[train_idx], dtype=torch.float32)
    x_val = torch.tensor(dataset.x[val_idx], dtype=torch.float32)
    y_val = torch.tensor(dataset.y[val_idx], dtype=torch.float32)
    w_val = torch.tensor(dataset.weights[val_idx], dtype=torch.float32)
    x_test = torch.tensor(dataset.x[test_idx], dtype=torch.float32)
    y_test = torch.tensor(dataset.y[test_idx], dtype=torch.float32)
    w_test = torch.tensor(dataset.weights[test_idx], dtype=torch.float32)

    model = PairScorerNetwork(dataset.x.shape[1], hidden=hidden, output_dim=dataset.y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_train, y_train, w_train), batch_size=batch_size, shuffle=True)
    best_state = None
    best_selection_score = None
    best_epoch = None
    best_metrics = None
    best_loss = float("inf")
    history = []
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            logits = model(xb)
            loss_mat = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            sample_weight = wb.unsqueeze(-1) * (1.0 + 2.0 * yb[:, :1])
            loss = (loss_mat * sample_weight).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n_seen += len(xb)
        val_loss, val_metrics = evaluate(model, x_val.to(device), y_val.to(device), val_rows, w_val.to(device))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(total / max(n_seen, 1), 6),
                "val_loss": round(val_loss, 6),
                **val_metrics,
            }
        )
        selection_score = _selection_score(selection_metric, val_loss, val_metrics)
        if _is_better_selection(selection_metric, selection_score, best_selection_score):
            best_selection_score = selection_score
            best_epoch = epoch + 1
            best_loss = val_loss
            best_metrics = dict(val_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_loss, final_metrics = evaluate(model, x_val.to(device), y_val.to(device), val_rows, w_val.to(device))
    test_loss, test_metrics = evaluate(model, x_test.to(device), y_test.to(device), test_rows, w_test.to(device))
    rule_baseline = evaluate_rule_baseline(val_rows)
    rule_test_baseline = evaluate_rule_baseline(test_rows)
    report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "pack_dir": str(pack_dir),
            "model_output": str(model_output),
            "n_rows": len(dataset.rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "hidden": hidden,
            "n_bits": n_bits,
            "device": device,
            "feature_schema": dataset.schema,
            "label_names": list(PAIR_LABEL_NAMES),
            "selection_metric": selection_metric,
            "training_caution": (
                "This model learns adjacent-step cascade compatibility. "
                "It is a local soft reward, not a route-level gold classifier."
            ),
        },
        "best_val_loss": round(float(best_loss), 6),
        "best_checkpoint": {
            "epoch": best_epoch,
            "selection_metric": selection_metric,
            "selection_score": round(float(best_selection_score), 6) if best_selection_score is not None else None,
            "metrics": best_metrics or {},
        },
        "final_val_loss": round(float(final_loss), 6),
        "final_metrics": final_metrics,
        "final_test_loss": round(float(test_loss), 6),
        "final_test_metrics": test_metrics,
        "rule_baseline": rule_baseline,
        "rule_test_baseline": rule_test_baseline,
        "history": history,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_class": "PairScorerNetwork",
            "feature_schema": dataset.schema,
            "hidden": hidden,
            "pack_dir": str(pack_dir),
            "label_names": list(PAIR_LABEL_NAMES),
        },
        model_output,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_output is not None:
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(report_markdown(report), encoding="utf-8")
    return report


def build_dataset(rows: list[dict[str, Any]], *, n_bits: int = 128) -> PairDataset:
    kept = [row for row in rows if row.get("left_step") and row.get("right_step")]
    schema = build_pair_feature_schema(kept, n_bits=n_bits)
    features = [pair_feature_vector(row, schema) for row in kept]
    labels = []
    weights = []
    groups = []
    for row in kept:
        row_labels = row.get("labels") or {}
        labels.append([float(row_labels.get(name) or 0.0) for name in PAIR_LABEL_NAMES])
        weights.append(float(row_labels.get("label_weight") or 1.0))
        groups.append(str(row.get("split_group_id") or row.get("pair_id") or ""))
    schema["feature_dim"] = len(features[0]) if features else 0
    return PairDataset(
        rows=kept,
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        groups=groups,
        schema=schema,
    )


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    rows: list[dict[str, Any]],
    weights: torch.Tensor,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss_mat = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        sample_weight = weights.unsqueeze(-1) * (1.0 + 2.0 * y[:, :1])
        loss = float((loss_mat * sample_weight).mean().item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    targets = y.detach().cpu().numpy()
    return loss, _metrics(rows, probs, targets)


def evaluate_rule_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([float((row.get("rule_features") or {}).get("rule_compatibility") or 0.0) for row in rows], dtype=np.float32)
    targets = np.asarray([float((row.get("labels") or {}).get("compatibility") or 0.0) for row in rows], dtype=np.float32)
    return _single_label_metrics(rows, scores, targets, label_name="compatibility")


def _metrics(rows: list[dict[str, Any]], probs: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    out = {}
    for idx, label in enumerate(PAIR_LABEL_NAMES):
        out[f"{label}_auc"] = round(_binary_auc(probs[:, idx], targets[:, idx]), 6)
    comp_metrics = _single_label_metrics(
        rows,
        probs[:, 0],
        targets[:, 0],
        label_name="compatibility",
    )
    out.update({
        "pairwise_group_accuracy": comp_metrics["pairwise_group_accuracy"],
        "top1_compatibility_hit_rate": comp_metrics["top1_positive_state_hit_rate"],
        "mean_best_compatibility_rank": comp_metrics["mean_best_positive_rank"],
        "median_best_compatibility_rank": comp_metrics["median_best_positive_rank"],
    })
    return out


def _single_label_metrics(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    label_name: str,
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_group[str(row.get("split_group_id") or row.get("pair_id") or idx)].append(idx)
    positive_groups = 0
    top1 = 0
    pair_correct = 0.0
    pair_total = 0
    best_ranks = []
    for indices in by_group.values():
        labels = [float(targets[idx]) for idx in indices]
        positives = [idx for idx in indices if float(targets[idx]) > 0.5]
        if not positives:
            continue
        positive_groups += 1
        ordered = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)
        rank = min(ordered.index(idx) + 1 for idx in positives)
        best_ranks.append(rank)
        top1 += int(rank <= 1)
        pos_scores = [scores[idx] for idx in positives]
        neg_scores = [scores[idx] for idx in indices if idx not in positives]
        for ps in pos_scores:
            for ns in neg_scores:
                pair_total += 1
                pair_correct += float(ps > ns) + 0.5 * float(ps == ns)
    return {
        "label_name": label_name,
        "positive_groups": positive_groups,
        "top1_positive_state_hit_rate": round(top1 / positive_groups, 6) if positive_groups else 0.0,
        "pairwise_group_accuracy": round(pair_correct / pair_total, 6) if pair_total else 0.0,
        "mean_best_positive_rank": round(float(np.mean(best_ranks)), 6) if best_ranks else None,
        "median_best_positive_rank": round(float(np.median(best_ranks)), 6) if best_ranks else None,
    }


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    total = 0.0
    for score in positives:
        total += float(np.sum(score > negatives))
        total += 0.5 * float(np.sum(score == negatives))
    return total / float(len(positives) * len(negatives))


def _split_indices(rows: list[dict[str, Any]]) -> tuple[list[int], list[int], list[int]]:
    explicit_train = [idx for idx, row in enumerate(rows) if row.get("split") == "train"]
    explicit_val = [idx for idx, row in enumerate(rows) if row.get("split") == "val"]
    explicit_test = [idx for idx, row in enumerate(rows) if row.get("split") == "test"]
    if explicit_train and explicit_val:
        return explicit_train, explicit_val, explicit_test or explicit_val

    train_idx = [idx for idx, row in enumerate(rows) if row.get("split") != "val"]
    val_idx = [idx for idx, row in enumerate(rows) if row.get("split") == "val"]
    if train_idx and val_idx:
        return train_idx, val_idx, val_idx
    groups = sorted({str(row.get("split_group_id") or row.get("pair_id") or idx) for idx, row in enumerate(rows)})
    if len(groups) <= 1:
        pivot = max(1, int(len(rows) * 0.8))
        return list(range(pivot)), list(range(pivot, len(rows))), list(range(pivot, len(rows)))
    val_groups = set(groups[:: max(1, len(groups) // max(1, len(groups) // 5 or 1))][: max(1, len(groups) // 5)])
    train_idx = []
    val_idx = []
    for idx, row in enumerate(rows):
        group = str(row.get("split_group_id") or row.get("pair_id") or idx)
        (val_idx if group in val_groups else train_idx).append(idx)
    if train_idx and val_idx:
        return train_idx, val_idx, val_idx
    pivot = max(1, int(len(rows) * 0.8))
    return list(range(pivot)), list(range(pivot, len(rows))), list(range(pivot, len(rows)))


def _selection_score(metric: str, val_loss: float, val_metrics: dict[str, Any]) -> float:
    if metric == "val_loss":
        return float(val_loss)
    if metric not in val_metrics:
        return float("-inf")
    value = val_metrics.get(metric)
    return float(value if value is not None else float("-inf"))


def _is_better_selection(metric: str, score: float, best_score: float | None) -> bool:
    if best_score is None:
        return True
    if metric == "val_loss":
        return score < best_score
    return score > best_score


def report_markdown(report: dict[str, Any]) -> str:
    meta = report.get("metadata") or {}
    final = report.get("final_metrics") or {}
    baseline = report.get("rule_baseline") or {}
    lines = [
        "# Cascade Pair Scorer",
        "",
        f"- pack: `{meta.get('pack_dir')}`",
        f"- model: `{meta.get('model_output')}`",
        f"- rows: `{meta.get('n_rows')}`",
        f"- train/val/test: `{meta.get('n_train')}` / `{meta.get('n_val')}` / `{meta.get('n_test')}`",
        f"- final val pairwise group accuracy: `{final.get('pairwise_group_accuracy')}`",
        f"- baseline val pairwise group accuracy: `{baseline.get('pairwise_group_accuracy')}`",
        f"- final test pairwise group accuracy: `{(report.get('final_test_metrics') or {}).get('pairwise_group_accuracy')}`",
        f"- baseline test pairwise group accuracy: `{(report.get('rule_test_baseline') or {}).get('pairwise_group_accuracy')}`",
        "",
        "## Contract",
        "",
        str(meta.get("training_caution") or ""),
    ]
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Train cascade pair compatibility scorer")
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--model-output", required=True)
    ap.add_argument("--report-output", required=True)
    ap.add_argument("--md-output", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-bits", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--selection-metric",
        default="pairwise_group_accuracy",
        choices=[
            "val_loss",
            "pairwise_group_accuracy",
            "top1_compatibility_hit_rate",
        ],
    )
    args = ap.parse_args()
    report = train_cascade_pair_scorer(
        pack_dir=Path(args.pack_dir),
        model_output=Path(args.model_output),
        report_output=Path(args.report_output),
        md_output=Path(args.md_output) if args.md_output else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        n_bits=args.n_bits,
        seed=args.seed,
        device=args.device,
        selection_metric=args.selection_metric,
    )
    print(json.dumps(report["final_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

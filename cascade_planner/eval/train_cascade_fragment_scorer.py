"""Train a fast Stage-2 cascade fragment preference scorer."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascade_search.pair_scorer import build_pair_feature_schema, pair_feature_vector


FRAGMENT_LABEL_NAMES = [
    "fragment_preference",
    "cascade_compatible",
    "one_pot",
    "telescoped",
    "condition_compatible",
    "cofactor_compatible",
    "isolation_required",
    "biocascade",
]


@dataclass
class FragmentDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    groups: list[str]
    schema: dict[str, Any]


class FragmentScorerNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 160, output_dim: int = len(FRAGMENT_LABEL_NAMES)):
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


def train_cascade_fragment_scorer(
    *,
    pack_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 160,
    n_bits: int = 128,
    seed: int = 42,
    device: str | None = None,
    selection_metric: str = "pairwise_group_accuracy",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = _read_jsonl(pack_dir / "fragment_value.jsonl")
    dataset = build_dataset(rows, n_bits=n_bits)
    if len(dataset.rows) < 8:
        raise ValueError(f"not enough fragment rows to train: {len(dataset.rows)}")
    train_idx, val_idx = _split_indices(dataset.rows)
    x_train = torch.tensor(dataset.x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(dataset.y[train_idx], dtype=torch.float32)
    w_train = torch.tensor(dataset.weights[train_idx], dtype=torch.float32)
    x_val = torch.tensor(dataset.x[val_idx], dtype=torch.float32)
    y_val = torch.tensor(dataset.y[val_idx], dtype=torch.float32)
    w_val = torch.tensor(dataset.weights[val_idx], dtype=torch.float32)

    model = FragmentScorerNetwork(dataset.x.shape[1], hidden=hidden, output_dim=dataset.y.shape[1]).to(device)
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
            sample_weight = wb.unsqueeze(-1) * (1.0 + 1.5 * yb[:, :1])
            loss = (loss_mat * sample_weight).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n_seen += len(xb)
        val_loss, val_metrics = evaluate(model, x_val.to(device), y_val.to(device), val_rows=[dataset.rows[idx] for idx in val_idx], weights=w_val.to(device))
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
    final_loss, final_metrics = evaluate(model, x_val.to(device), y_val.to(device), val_rows=[dataset.rows[idx] for idx in val_idx], weights=w_val.to(device))
    report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "pack_dir": str(pack_dir),
            "model_output": str(model_output),
            "n_rows": len(dataset.rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_bits": n_bits,
            "hidden": hidden,
            "device": device,
            "feature_schema": dataset.schema,
            "label_names": list(FRAGMENT_LABEL_NAMES),
            "selection_metric": selection_metric,
            "training_caution": (
                "This model learns 2/3-step fragment preference. "
                "It is a local process scorer, not a route-level gold classifier."
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
        "history": history,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_class": "FragmentScorerNetwork",
            "feature_schema": dataset.schema,
            "hidden": hidden,
            "pack_dir": str(pack_dir),
            "label_names": list(FRAGMENT_LABEL_NAMES),
        },
        model_output,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_output is not None:
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(report_markdown(report), encoding="utf-8")
    return report


def build_dataset(rows: list[dict[str, Any]], *, n_bits: int = 128) -> FragmentDataset:
    kept = [row for row in rows if row.get("pair_rows")]
    schema = _build_feature_schema(kept, n_bits=n_bits)
    features = [fragment_feature_vector(row, schema) for row in kept]
    labels = []
    weights = []
    groups = []
    for row in kept:
        row_labels = row.get("labels") or {}
        labels.append([float(row_labels.get(name) or 0.0) for name in FRAGMENT_LABEL_NAMES])
        weights.append(float(row_labels.get("label_weight") or 1.0))
        groups.append(str(row.get("split_group_id") or row.get("fragment_id") or ""))
    schema["feature_dim"] = len(features[0]) if features else 0
    return FragmentDataset(
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
    *,
    val_rows: list[dict[str, Any]],
    weights: torch.Tensor,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss_mat = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        sample_weight = weights.unsqueeze(-1) * (1.0 + 1.5 * y[:, :1])
        loss = float((loss_mat * sample_weight).mean().item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    targets = y.detach().cpu().numpy()
    return loss, _metrics(val_rows, probs, targets)


def _metrics(rows: list[dict[str, Any]], probs: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    out = {}
    for idx, label in enumerate(FRAGMENT_LABEL_NAMES):
        out[f"{label}_auc"] = round(_binary_auc(probs[:, idx], targets[:, idx]), 6)
    comp_metrics = _single_label_metrics(rows, probs[:, 0], targets[:, 0], label_name="fragment_preference")
    out.update(
        {
            "pairwise_group_accuracy": comp_metrics["pairwise_group_accuracy"],
            "top1_fragment_hit_rate": comp_metrics["top1_positive_state_hit_rate"],
            "mean_best_fragment_rank": comp_metrics["mean_best_positive_rank"],
            "median_best_fragment_rank": comp_metrics["median_best_positive_rank"],
        }
    )
    return out


def _single_label_metrics(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    label_name: str,
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_group.setdefault(str(row.get("split_group_id") or row.get("fragment_id") or idx), []).append(idx)
    positive_groups = 0
    top1 = 0
    pair_correct = 0.0
    pair_total = 0
    best_ranks = []
    for indices in by_group.values():
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


def _split_indices(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    train_idx = [idx for idx, row in enumerate(rows) if row.get("split") != "val"]
    val_idx = [idx for idx, row in enumerate(rows) if row.get("split") == "val"]
    if train_idx and val_idx:
        return train_idx, val_idx
    pivot = max(1, int(len(rows) * 0.8))
    return list(range(pivot)), list(range(pivot, len(rows)))


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
    lines = [
        "# Cascade Fragment Scorer",
        "",
        f"- pack: `{meta.get('pack_dir')}`",
        f"- model: `{meta.get('model_output')}`",
        f"- rows: `{meta.get('n_rows')}`",
        f"- train/val: `{meta.get('n_train')}` / `{meta.get('n_val')}`",
        f"- final pairwise group accuracy: `{final.get('pairwise_group_accuracy')}`",
        "",
        "## Contract",
        "",
        str(meta.get("training_caution") or ""),
    ]
    return "\n".join(lines) + "\n"


def _build_feature_schema(rows: list[dict[str, Any]], *, n_bits: int = 128) -> dict[str, Any]:
    pair_rows = [pair for row in rows for pair in (row.get("pair_rows") or [])]
    if not pair_rows:
        pair_rows = [{}]
    schema = build_pair_feature_schema(pair_rows, n_bits=n_bits)
    schema["feature_contract"] = "cascade_fragment_preference_fragment_window.v1"
    schema["window_features"] = [
        "window_size",
        "pair_count",
        "mean_pair_rule_compatibility",
        "min_pair_rule_compatibility",
        "max_pair_rule_compatibility",
        "max_pair_isolation_need",
        "mean_pair_condition_compatibility",
        "mean_pair_cofactor_compatible",
        "mean_pair_biocascade",
    ]
    return schema


def fragment_feature_vector(row: dict[str, Any], schema: dict[str, Any]) -> np.ndarray:
    pair_rows = list(row.get("pair_rows") or [])
    if not pair_rows:
        return np.zeros(int(schema.get("feature_dim") or 0), dtype=np.float32)
    pair_vectors = [pair_feature_vector(pair, schema) for pair in pair_rows]
    pair_matrix = np.asarray(pair_vectors, dtype=np.float32)
    summary = np.asarray(
        [
            float(row.get("window_size") or len(pair_rows) + 1),
            float(len(pair_rows)),
            float(np.mean([float((pair.get("rule_features") or {}).get("rule_compatibility") or 0.0) for pair in pair_rows])),
            float(np.min([float((pair.get("rule_features") or {}).get("rule_compatibility") or 0.0) for pair in pair_rows])),
            float(np.max([float((pair.get("rule_features") or {}).get("rule_compatibility") or 0.0) for pair in pair_rows])),
            float(np.max([float((pair.get("rule_features") or {}).get("rule_isolation_need") or 0.0) for pair in pair_rows])),
            float(np.mean([_condition_proxy(pair.get("rule_features") or {}) for pair in pair_rows])),
            float(np.mean([1.0 - float((pair.get("rule_features") or {}).get("cofactor_conflict") or 0.0) for pair in pair_rows])),
            float(np.mean([float((pair.get("rule_features") or {}).get("both_enzymatic") or 0.0) for pair in pair_rows])),
        ],
        dtype=np.float32,
    )
    return np.concatenate([pair_matrix.mean(axis=0), summary], axis=0)


def _condition_proxy(features: dict[str, Any]) -> float:
    return float(
        max(
            0.0,
            min(
                1.0,
                float(features.get("temp_overlap", 0.55)) * 0.45
                + float(features.get("ph_overlap", 0.55)) * 0.35
                + float(features.get("solvent_match", 0.55)) * 0.20,
            ),
        )
    )


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train cascade fragment preference scorer")
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--model-output", required=True)
    ap.add_argument("--report-output", required=True)
    ap.add_argument("--md-output", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--n-bits", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--selection-metric",
        default="pairwise_group_accuracy",
        choices=[
            "val_loss",
            "pairwise_group_accuracy",
            "top1_fragment_hit_rate",
        ],
    )
    args = ap.parse_args()
    report = train_cascade_fragment_scorer(
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

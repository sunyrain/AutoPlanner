"""Train a source-value model from cascade source-value packs.

The model predicts which proposal source is useful for the current expansion
state before querying that source.  It uses only pre-expansion state/source
features, not candidate costs or hit labels as inputs.
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


CATEGORICAL_FIELDS = ["route_domain", "source_model", "reaction_domain", "adjacent_reaction_domain"]
NUMERIC_FIELDS = [
    "parent_depth",
    "parent_heavy_atoms",
    "parent_hetero_atoms",
    "parent_ring_count",
    "parent_mol_wt",
]


@dataclass
class SourceValueDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    groups: list[str]
    schema: dict[str, Any]


class SourceValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, max(16, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_cascade_source_value(
    *,
    pack_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden: int = 64,
    seed: int = 42,
    device: str | None = None,
    loss_mode: str = "bce",
    selection_metric: str = "top1_positive_state_hit_rate",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = _read_jsonl(pack_dir / "source_value.jsonl")
    dataset = build_dataset(rows)
    if len(dataset.rows) < 4:
        raise ValueError(f"not enough source-value rows to train: {len(dataset.rows)}")
    train_idx, val_idx = _split_indices(dataset.rows)
    x_train = torch.tensor(dataset.x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(dataset.y[train_idx], dtype=torch.float32)
    x_val = torch.tensor(dataset.x[val_idx], dtype=torch.float32)
    y_val = torch.tensor(dataset.y[val_idx], dtype=torch.float32)
    val_rows = [dataset.rows[idx] for idx in val_idx]
    train_rows = [dataset.rows[idx] for idx in train_idx]

    model = SourceValueNetwork(dataset.x.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if loss_mode not in {"bce", "pairwise"}:
        raise ValueError(f"unsupported loss_mode: {loss_mode}")
    _validate_selection_metric(selection_metric)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    pair_loader = None
    n_train_pairs = 0
    if loss_mode == "pairwise":
        pair_idx = _state_pair_indices(train_rows)
        n_train_pairs = len(pair_idx)
        if pair_idx:
            pos_x = torch.tensor(dataset.x[[train_idx[pos] for pos, _ in pair_idx]], dtype=torch.float32)
            neg_x = torch.tensor(dataset.x[[train_idx[neg] for _, neg in pair_idx]], dtype=torch.float32)
            pair_loader = DataLoader(TensorDataset(pos_x, neg_x), batch_size=batch_size, shuffle=True)
    best_state = None
    best_loss = float("inf")
    best_selection_score = None
    best_epoch = None
    best_metrics = None
    history = []
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        active_loader = pair_loader if pair_loader is not None else loader
        for batch in active_loader:
            if pair_loader is not None:
                pos_x, neg_x = batch
                pos_x = pos_x.to(device)
                neg_x = neg_x.to(device)
                loss = nn.functional.softplus(-(model(pos_x) - model(neg_x))).mean()
                batch_size_seen = len(pos_x)
            else:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                weights = 1.0 + 3.0 * yb
                loss = (
                    nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                    * weights
                ).mean()
                batch_size_seen = len(xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch_size_seen
            n_seen += batch_size_seen
        val_loss, val_metrics = evaluate(model, x_val.to(device), y_val.to(device), val_rows)
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
            best_loss = val_loss
            best_selection_score = selection_score
            best_epoch = epoch + 1
            best_metrics = dict(val_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    final_loss, final_metrics = evaluate(model, x_val.to(device), y_val.to(device), val_rows)
    prior_metrics = evaluate_source_prior(train_rows=train_rows, val_rows=val_rows)
    report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "pack_dir": str(pack_dir),
            "model_output": str(model_output),
            "n_rows": len(dataset.rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "hidden": hidden,
            "device": device,
            "feature_schema": dataset.schema,
            "label_contract": "internal_search_source_value_not_record_gold.v1",
            "loss_mode": loss_mode,
            "selection_metric": selection_metric,
            "n_train_pairs": n_train_pairs,
            "training_caution": (
                "This is a source-budget value model, not a cascade-record gold classifier. "
                "Benchmark-derived reports are diagnostic unless the pack came from a training split."
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
        "source_prior_baseline": prior_metrics,
        "history": history,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_class": "SourceValueNetwork",
            "feature_schema": dataset.schema,
            "hidden": hidden,
            "pack_dir": str(pack_dir),
            "selection_metric": selection_metric,
        },
        model_output,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_output is not None:
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(report_markdown(report), encoding="utf-8")
    return report


def build_dataset(rows: list[dict[str, Any]]) -> SourceValueDataset:
    categories = {
        field: sorted({_feature_field(row, field) for row in rows})
        for field in CATEGORICAL_FIELDS
    }
    schema = {
        "schema_version": "cascade_source_value_features.v2",
        "categorical_fields": CATEGORICAL_FIELDS,
        "categories": categories,
        "numeric_fields": NUMERIC_FIELDS,
        "feature_contract": "pre_expansion_state_source_with_node_context.v2",
    }
    features = []
    labels = []
    kept = []
    groups = []
    for row in rows:
        labels_dict = row.get("labels") or {}
        label = float(labels_dict.get("source_value") or 0.0)
        features.append(_feature_vector(row, schema))
        labels.append(label)
        groups.append(str(row.get("state_id") or ""))
        kept.append(row)
    schema["feature_dim"] = len(features[0]) if features else 0
    return SourceValueDataset(
        rows=kept,
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels, dtype=np.float32),
        groups=groups,
        schema=schema,
    )


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    rows: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = float(nn.functional.binary_cross_entropy_with_logits(logits, y).item())
        scores = torch.sigmoid(logits).detach().cpu().numpy()
    labels = y.detach().cpu().numpy()
    binary = (labels > 0).astype(np.float32)
    return loss, _ranking_metrics(rows, scores, binary)


def evaluate_source_prior(*, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, list[float]] = defaultdict(list)
    for row in train_rows:
        totals[str(row.get("source_model") or "unknown")].append(float((row.get("labels") or {}).get("source_value") or 0.0))
    priors = {source: sum(values) / len(values) for source, values in totals.items() if values}
    fallback = sum(priors.values()) / len(priors) if priors else 0.0
    scores = np.asarray([priors.get(str(row.get("source_model") or "unknown"), fallback) for row in val_rows], dtype=np.float32)
    labels = np.asarray([float((row.get("labels") or {}).get("source_value") or 0.0) > 0 for row in val_rows], dtype=np.float32)
    metrics = _ranking_metrics(val_rows, scores, labels)
    metrics["source_priors"] = {key: round(value, 6) for key, value in sorted(priors.items())}
    return metrics


def _state_pair_indices(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    by_state: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_state[str(row.get("state_id") or idx)].append(idx)
    pairs = []
    for indices in by_state.values():
        positives = [
            idx for idx in indices
            if float((rows[idx].get("labels") or {}).get("source_value") or 0.0) > 0.0
        ]
        negatives = [
            idx for idx in indices
            if float((rows[idx].get("labels") or {}).get("source_value") or 0.0) <= 0.0
        ]
        for pos in positives:
            for neg in negatives:
                pairs.append((pos, neg))
    return pairs


def _validate_selection_metric(metric: str) -> None:
    allowed = {
        "val_loss",
        "auc",
        "top1_positive_state_hit_rate",
        "pairwise_state_accuracy",
    }
    if metric not in allowed:
        raise ValueError(f"unsupported selection_metric: {metric}")


def _selection_score(metric: str, val_loss: float, val_metrics: dict[str, Any]) -> float:
    if metric == "val_loss":
        return float(val_loss)
    value = val_metrics.get(metric)
    if value is None:
        return float("-inf")
    return float(value)


def _is_better_selection(metric: str, score: float, best_score: float | None) -> bool:
    if best_score is None:
        return True
    if metric == "val_loss":
        return score < best_score
    return score > best_score


def _ranking_metrics(rows: list[dict[str, Any]], scores: np.ndarray, binary_labels: np.ndarray) -> dict[str, Any]:
    by_state: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_state[str(row.get("state_id") or idx)].append(idx)
    positive_states = 0
    top1_hits = 0
    pair_correct = 0
    pair_total = 0
    for indices in by_state.values():
        labels = binary_labels[indices]
        if labels.max() <= 0:
            continue
        positive_states += 1
        top_idx = indices[int(np.argmax(scores[indices]))]
        top1_hits += int(binary_labels[top_idx] > 0)
        pos_scores = [scores[idx] for idx in indices if binary_labels[idx] > 0]
        neg_scores = [scores[idx] for idx in indices if binary_labels[idx] <= 0]
        for ps in pos_scores:
            for ns in neg_scores:
                pair_total += 1
                pair_correct += int(ps > ns) + 0.5 * int(ps == ns)
    return {
        "auc": round(_binary_auc(scores, binary_labels), 6),
        "positive_source_rate": round(float(binary_labels.mean()), 6) if len(binary_labels) else 0.0,
        "positive_states": positive_states,
        "top1_positive_state_hit_rate": round(top1_hits / positive_states, 6) if positive_states else 0.0,
        "pairwise_state_accuracy": round(pair_correct / pair_total, 6) if pair_total else None,
    }


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels > 0]
    negatives = scores[labels <= 0]
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
    if not train_idx or not val_idx:
        pivot = max(1, int(round(len(rows) * 0.8)))
        train_idx = list(range(pivot))
        val_idx = list(range(pivot, len(rows)))
    return train_idx, val_idx


def _feature_vector(row: dict[str, Any], schema: dict[str, Any]) -> list[float]:
    out = []
    for field in schema["categorical_fields"]:
        value = _feature_field(row, field)
        out.extend(1.0 if value == category else 0.0 for category in schema["categories"][field])
    mol_features = _parent_mol_features(str(row.get("parent_mol") or ""))
    numeric = {
        "parent_depth": _scaled_float(row.get("parent_depth"), scale=8.0),
        **mol_features,
    }
    out.extend(float(numeric[name]) for name in schema["numeric_fields"])
    return out


def _feature_field(row: dict[str, Any], field: str) -> str:
    if field in row and row.get(field) not in (None, ""):
        return str(row.get(field))
    context = row.get("context_features") or {}
    if isinstance(context, dict) and context.get(field) not in (None, ""):
        return str(context.get(field))
    return "unknown"


def _parent_mol_features(smiles: str) -> dict[str, float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except Exception:
        return {name: 0.0 for name in NUMERIC_FIELDS if name != "parent_depth"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: 0.0 for name in NUMERIC_FIELDS if name != "parent_depth"}
    heavy = float(mol.GetNumHeavyAtoms())
    hetero = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))
    rings = float(mol.GetRingInfo().NumRings())
    mw = float(Descriptors.MolWt(mol))
    return {
        "parent_heavy_atoms": min(1.0, heavy / 80.0),
        "parent_hetero_atoms": min(1.0, hetero / 30.0),
        "parent_ring_count": min(1.0, rings / 10.0),
        "parent_mol_wt": min(1.0, mw / 1000.0),
    }


def _scaled_float(value: Any, *, scale: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, min(1.0, out / scale))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def report_markdown(report: dict[str, Any]) -> str:
    final = report["final_metrics"]
    baseline = report["source_prior_baseline"]
    best = report.get("best_checkpoint") or {}
    lines = [
        "# Cascade Source Value Training",
        "",
        "This model predicts source utility for internal ChemEnzy expansion states.",
        "It is not a cascade-record gold classifier.",
        "",
        "## Checkpoint Selection",
        "",
        f"- metric: {best.get('selection_metric')}",
        f"- epoch: {best.get('epoch')}",
        f"- score: {best.get('selection_score')}",
        "",
        "## Final Metrics",
        "",
        f"- auc: {final['auc']}",
        f"- top1_positive_state_hit_rate: {final['top1_positive_state_hit_rate']}",
        f"- pairwise_state_accuracy: {final['pairwise_state_accuracy']}",
        f"- positive_source_rate: {final['positive_source_rate']}",
        "",
        "## Source Prior Baseline",
        "",
        f"- auc: {baseline['auc']}",
        f"- top1_positive_state_hit_rate: {baseline['top1_positive_state_hit_rate']}",
        f"- pairwise_state_accuracy: {baseline['pairwise_state_accuracy']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train cascade source-value model")
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--model-output", required=True)
    ap.add_argument("--report-output", required=True)
    ap.add_argument("--md-output")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device")
    ap.add_argument("--loss-mode", choices=["bce", "pairwise"], default="bce")
    ap.add_argument(
        "--selection-metric",
        choices=["val_loss", "auc", "top1_positive_state_hit_rate", "pairwise_state_accuracy"],
        default="top1_positive_state_hit_rate",
    )
    args = ap.parse_args()
    report = train_cascade_source_value(
        pack_dir=Path(args.pack_dir),
        model_output=Path(args.model_output),
        report_output=Path(args.report_output),
        md_output=Path(args.md_output) if args.md_output else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        device=args.device,
        loss_mode=args.loss_mode,
        selection_metric=args.selection_metric,
    )
    print(json.dumps(report["final_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

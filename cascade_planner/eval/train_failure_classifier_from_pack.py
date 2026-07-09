"""Train a multi-label failure-risk classifier from a training pack.

The model predicts common planner bottlenecks such as generator misses,
stock dead-ends, selector misses, and condition/cascade failures. It is meant
to support search-control agents, not to replace route validation.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


RDLogger.DisableLog("rdApp.*")

DOMAIN_VALUES = ["all_chemical", "all_enzymatic", "chemoenzymatic", "hybrid_mimetic", "whole_cell_biocatalytic"]
METRIC_KEYS = [
    "plan",
    "filled_route_any",
    "strict_stock_solve_any",
    "condition_window_success_any",
    "cascade_compatibility_success_any",
    "terminal_GT_reactant_in_top5",
    "filled_type_GT@1",
    "filled_type_GT@5",
    "skeleton_type_GT@1",
    "skeleton_type_GT@5",
]
DEFAULT_EXCLUDED_LABELS = ("no_professional_solved_route",)


@dataclass
class FailureDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    labels: list[str]
    feature_schema: dict[str, Any]


class FailureClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, max(32, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(32, hidden // 2), out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_failure_rows(pack_dir: Path) -> list[dict[str, Any]]:
    path = pack_dir / "failure_diagnosis.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing failure_diagnosis.jsonl in {pack_dir}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("target_smiles"):
                rows.append(row)
    return rows


def build_dataset(
    rows: list[dict[str, Any]],
    *,
    n_bits: int = 128,
    min_label_count: int = 10,
    exclude_labels: Iterable[str] | None = DEFAULT_EXCLUDED_LABELS,
) -> FailureDataset:
    excluded = set(exclude_labels or [])
    counts = Counter(
        label
        for row in rows
        for label in row.get("labels") or []
        if label not in excluded
    )
    labels = sorted(label for label, count in counts.items() if count >= min_label_count)
    if not labels:
        labels = sorted(counts)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    x_rows = []
    y_rows = []
    for row in rows:
        x_rows.append(row_features(row, n_bits=n_bits))
        y = np.zeros(len(labels), dtype=np.float32)
        for label in row.get("labels") or []:
            if label in label_to_idx:
                y[label_to_idx[label]] = 1.0
        y_rows.append(y)
    schema = {
        "n_bits": n_bits,
        "domain_values": DOMAIN_VALUES,
        "metric_keys": METRIC_KEYS,
        "labels": labels,
        "feature_dim": len(x_rows[0]) if x_rows else 0,
        "label_counts": {label: counts[label] for label in labels},
        "excluded_labels": sorted(excluded),
    }
    return FailureDataset(
        rows=rows,
        x=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.float32),
        labels=labels,
        feature_schema=schema,
    )


def row_features(row: dict[str, Any], *, n_bits: int) -> np.ndarray:
    target_fp = fp(row.get("target_smiles"), n_bits=n_bits)
    mol = Chem.MolFromSmiles(row.get("target_smiles") or "")
    heavy = float(mol.GetNumHeavyAtoms()) if mol is not None else 0.0
    domain = row.get("route_domain") or ""
    domain_vec = [1.0 if domain == value else 0.0 for value in DOMAIN_VALUES]
    metrics = row.get("metrics") or {}
    metric_vec = [bool_metric(metrics.get(key)) for key in METRIC_KEYS]
    scalar = [
        heavy / 80.0,
        float(row.get("depth") or 0.0) / 10.0,
        float(row.get("n_routes") or 0.0) / 10.0,
        float(bool(row.get("has_failure_label"))),
    ]
    return np.concatenate([
        target_fp,
        np.asarray(domain_vec + metric_vec + scalar, dtype=np.float32),
    ])


def fp(smiles: str | None, *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def bool_metric(value: Any) -> float:
    if value is True:
        return 1.0
    if value is False:
        return -1.0
    return 0.0


def split_by_target(rows: list[dict[str, Any]], val_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    targets = sorted({canonical_smiles(row.get("target_smiles") or "") for row in rows})
    n_val = max(1, int(round(len(targets) * val_fraction))) if targets else 0
    val_targets = set(targets[::max(1, len(targets) // n_val)][:n_val]) if n_val else set()
    train_idx = []
    val_idx = []
    for idx, row in enumerate(rows):
        target = canonical_smiles(row.get("target_smiles") or "")
        (val_idx if target in val_targets else train_idx).append(idx)
    if not train_idx or not val_idx:
        pivot = max(1, int(len(rows) * (1.0 - val_fraction)))
        train_idx = list(range(pivot))
        val_idx = list(range(pivot, len(rows)))
    return train_idx, val_idx


def split_summary(rows: list[dict[str, Any]], train_idx: list[int], val_idx: list[int]) -> dict[str, Any]:
    train_targets = {
        canonical_smiles(rows[idx].get("target_smiles") or "")
        for idx in train_idx
        if 0 <= idx < len(rows)
    }
    val_targets = {
        canonical_smiles(rows[idx].get("target_smiles") or "")
        for idx in val_idx
        if 0 <= idx < len(rows)
    }
    overlap = sorted(t for t in train_targets & val_targets if t)
    return {
        "mode": "target_smiles_grouped",
        "train_targets": len([t for t in train_targets if t]),
        "val_targets": len([t for t in val_targets if t]),
        "overlap_targets": overlap,
        "has_target_overlap": bool(overlap),
    }


def train_failure_classifier_from_pack(
    *,
    pack_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    n_bits: int = 128,
    hidden: int = 160,
    min_label_count: int = 10,
    exclude_labels: Iterable[str] | None = DEFAULT_EXCLUDED_LABELS,
    seed: int = 42,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rows = load_failure_rows(pack_dir)
    dataset = build_dataset(
        rows,
        n_bits=n_bits,
        min_label_count=min_label_count,
        exclude_labels=exclude_labels,
    )
    train_idx, val_idx = split_by_target(rows)
    x_train = torch.tensor(dataset.x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(dataset.y[train_idx], dtype=torch.float32)
    x_val = torch.tensor(dataset.x[val_idx], dtype=torch.float32)
    y_val = torch.tensor(dataset.y[val_idx], dtype=torch.float32)
    model = FailureClassifier(dataset.x.shape[1], len(dataset.labels), hidden=hidden)

    positives = y_train.sum(dim=0)
    negatives = y_train.shape[0] - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), min=1.0, max=30.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    history = []
    best_state = None
    best_val = float("inf")
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n_seen += len(xb)
        val_loss, val_metrics = evaluate(model, x_val, y_val, dataset.labels)
        row = {
            "epoch": epoch + 1,
            "train_loss": round(total / max(n_seen, 1), 6),
            "val_loss": round(val_loss, 6),
            "macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, val_metrics = evaluate(model, x_val, y_val, dataset.labels)
    report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "pack_dir": str(pack_dir),
            "model_output": str(model_output),
            "n_rows": len(rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "split": split_summary(rows, train_idx, val_idx),
            "feature_schema": dataset.feature_schema,
            "pos_weight": [round(float(x), 4) for x in pos_weight],
        },
        "best_val_loss": round(float(best_val), 6),
        "val_metrics": val_metrics,
        "history": history,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_class": "FailureClassifier",
        "feature_schema": dataset.feature_schema,
        "hidden": hidden,
        "pack_dir": str(pack_dir),
    }, model_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_output:
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(report_markdown(report), encoding="utf-8")
    return report


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, labels: list[str]) -> tuple[float, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = float(nn.functional.binary_cross_entropy_with_logits(logits, y).item())
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
    tp = (pred * y).sum(dim=0)
    fp = (pred * (1 - y)).sum(dim=0)
    fn = ((1 - pred) * y).sum(dim=0)
    per_label = {}
    f1s = []
    for idx, label in enumerate(labels):
        precision = float(tp[idx] / torch.clamp(tp[idx] + fp[idx], min=1.0))
        recall = float(tp[idx] / torch.clamp(tp[idx] + fn[idx], min=1.0))
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": int(y[:, idx].sum().item()),
        }
    return loss, {
        "macro_f1": round(sum(f1s) / max(len(f1s), 1), 6),
        "per_label": per_label,
    }


def report_markdown(report: dict[str, Any]) -> str:
    meta = report.get("metadata") or {}
    metrics = report.get("val_metrics") or {}
    lines = [
        "# Failure Risk Classifier",
        "",
        f"Pack: `{meta.get('pack_dir')}`",
        f"Model: `{meta.get('model_output')}`",
        "",
        "## Samples",
        "",
        f"- rows: `{meta.get('n_rows')}`",
        f"- train: `{meta.get('n_train')}`",
        f"- validation: `{meta.get('n_val')}`",
        f"- split: `{(meta.get('split') or {}).get('mode')}`",
        f"- train targets: `{(meta.get('split') or {}).get('train_targets')}`",
        f"- validation targets: `{(meta.get('split') or {}).get('val_targets')}`",
        f"- target overlap: `{(meta.get('split') or {}).get('has_target_overlap')}`",
        f"- feature dim: `{(meta.get('feature_schema') or {}).get('feature_dim')}`",
        f"- excluded labels: `{', '.join((meta.get('feature_schema') or {}).get('excluded_labels') or []) or 'none'}`",
        "",
        "## Validation",
        "",
        f"- best val loss: `{report.get('best_val_loss')}`",
        f"- macro F1: `{metrics.get('macro_f1')}`",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in sorted((metrics.get("per_label") or {}).items()):
        lines.append(
            f"| `{label}` | {row.get('precision')} | {row.get('recall')} | {row.get('f1')} | {row.get('support')} |"
        )
    lines.extend([
        "",
        "## Caveat",
        "",
        "This model predicts likely failure modes for search control. It is not evidence that a reaction or route is chemically valid.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a failure-risk classifier from a training pack")
    ap.add_argument("--pack-dir", default="results/shared/training_pack/broad_20260507")
    ap.add_argument("--model-output", default="results/shared/failure_classifier/pack_failure_classifier_20260507.pt")
    ap.add_argument("--report-output", default="results/shared/failure_classifier/pack_failure_classifier_20260507.json")
    ap.add_argument("--md-output", default="results/shared/failure_classifier/pack_failure_classifier_20260507.md")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-bits", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--min-label-count", type=int, default=10)
    ap.add_argument(
        "--include-outcome-labels",
        action="store_true",
        help="Keep outcome-style labels such as no_professional_solved_route.",
    )
    args = ap.parse_args()
    report = train_failure_classifier_from_pack(
        pack_dir=Path(args.pack_dir),
        model_output=Path(args.model_output),
        report_output=Path(args.report_output),
        md_output=Path(args.md_output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_bits=args.n_bits,
        hidden=args.hidden,
        min_label_count=args.min_label_count,
        exclude_labels=() if args.include_outcome_labels else DEFAULT_EXCLUDED_LABELS,
    )
    print(json.dumps({
        "model_output": args.model_output,
        "best_val_loss": report["best_val_loss"],
        "macro_f1": report["val_metrics"]["macro_f1"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Train a route-level cascade condition/compatibility classifier.

The model learns from route_value.jsonl and predicts whether a route is likely
to pass condition-window and cascade-compatibility checks from target and route
sequence features. It is intended as an early reranking signal; deterministic
route metrics remain the source of truth after a route is built.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


RDLogger.DisableLog("rdApp.*")

LABELS = ["condition_failure", "compatibility_failure"]


@dataclass
class CascadeQualityDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    feature_schema: dict[str, Any]


class CascadeQualityClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2, hidden: int = 160):
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


def load_route_rows(pack_dir: Path) -> list[dict[str, Any]]:
    path = pack_dir / "route_value.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing route_value.jsonl in {pack_dir}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_vocab(rows: list[dict[str, Any]], key: str, *, min_count: int = 5, max_size: int = 64) -> list[str]:
    counts = Counter(token for row in rows for token in row.get(key) or [])
    values = [token for token, count in counts.most_common(max_size) if token and count >= min_count]
    return values


def build_dataset(
    rows: list[dict[str, Any]],
    *,
    n_bits: int = 128,
    min_vocab_count: int = 5,
    max_vocab_size: int = 64,
    include_bigrams: bool = False,
) -> CascadeQualityDataset:
    schema = {
        "n_bits": n_bits,
        "labels": LABELS,
        "type_vocab": build_vocab(rows, "type_sequence", min_count=min_vocab_count, max_size=max_vocab_size),
        "source_vocab": build_vocab(rows, "source_sequence", min_count=min_vocab_count, max_size=max_vocab_size),
        "ec1_vocab": build_vocab(rows, "ec1_sequence", min_count=min_vocab_count, max_size=max_vocab_size),
        "type_bigram_vocab": build_bigram_vocab(rows, "type_sequence", min_count=min_vocab_count, max_size=max_vocab_size) if include_bigrams else [],
        "source_bigram_vocab": build_bigram_vocab(rows, "source_sequence", min_count=min_vocab_count, max_size=max_vocab_size) if include_bigrams else [],
        "ec1_bigram_vocab": build_bigram_vocab(rows, "ec1_sequence", min_count=min_vocab_count, max_size=max_vocab_size) if include_bigrams else [],
        "include_bigrams": include_bigrams,
    }
    x_rows = []
    y_rows = []
    for row in rows:
        x_rows.append(row_features(row, schema))
        features = row.get("features") or {}
        y_rows.append([
            1.0 if float(features.get("condition_success") or 0.0) < 0.5 else 0.0,
            1.0 if float(features.get("compatibility_success") or 0.0) < 0.5 else 0.0,
        ])
    schema["feature_dim"] = len(x_rows[0]) if x_rows else 0
    schema["label_counts"] = {
        label: int(sum(y[idx] for y in y_rows))
        for idx, label in enumerate(LABELS)
    }
    return CascadeQualityDataset(
        rows=rows,
        x=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.float32),
        feature_schema=schema,
    )


def row_features(row: dict[str, Any], schema: dict[str, Any]) -> np.ndarray:
    n_bits = int(schema.get("n_bits") or 128)
    target_fp = fp(row.get("target_smiles"), n_bits=n_bits)
    n_steps = max(1, int(row.get("n_steps") or len(row.get("type_sequence") or []) or 1))
    mol = Chem.MolFromSmiles(row.get("target_smiles") or "")
    heavy = float(mol.GetNumHeavyAtoms()) if mol is not None else 0.0
    type_vec = bag(row.get("type_sequence") or [], schema.get("type_vocab") or [], denom=n_steps)
    source_vec = bag(row.get("source_sequence") or [], schema.get("source_vocab") or [], denom=n_steps)
    ec_vec = bag(row.get("ec1_sequence") or [], schema.get("ec1_vocab") or [], denom=n_steps)
    type_bigram_vec = bag(bigrams(row.get("type_sequence") or []), schema.get("type_bigram_vocab") or [], denom=max(n_steps - 1, 1))
    source_bigram_vec = bag(bigrams(row.get("source_sequence") or []), schema.get("source_bigram_vocab") or [], denom=max(n_steps - 1, 1))
    ec_bigram_vec = bag(bigrams(row.get("ec1_sequence") or []), schema.get("ec1_bigram_vocab") or [], denom=max(n_steps - 1, 1))
    ec_sequence = row.get("ec1_sequence") or []
    type_sequence = row.get("type_sequence") or []
    source_sequence = row.get("source_sequence") or []
    scalar = np.asarray([
        float(n_steps) / 8.0,
        heavy / 100.0,
        float(len(set(type_sequence))) / 8.0,
        float(len(set(source_sequence))) / 8.0,
        float(len([x for x in ec_sequence if x])) / max(n_steps, 1),
    ], dtype=np.float32)
    return np.concatenate([target_fp, type_vec, source_vec, ec_vec, type_bigram_vec, source_bigram_vec, ec_bigram_vec, scalar])


def fp(smiles: str | None, *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def bag(values: list[str], vocab: list[str], *, denom: int) -> np.ndarray:
    counts = Counter(values)
    return np.asarray([counts.get(token, 0) / max(denom, 1) for token in vocab], dtype=np.float32)


def bigrams(values: list[str]) -> list[str]:
    clean = [str(v or "NONE") for v in values]
    return [f"{a}->{b}" for a, b in zip(clean, clean[1:])]


def build_bigram_vocab(rows: list[dict[str, Any]], key: str, *, min_count: int = 5, max_size: int = 64) -> list[str]:
    counts = Counter(token for row in rows for token in bigrams(row.get(key) or []))
    return [token for token, count in counts.most_common(max_size) if token and count >= min_count]


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


def train_cascade_quality_from_pack(
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
    min_vocab_count: int = 5,
    include_bigrams: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rows = load_route_rows(pack_dir)
    dataset = build_dataset(rows, n_bits=n_bits, min_vocab_count=min_vocab_count, include_bigrams=include_bigrams)
    train_idx, val_idx = split_by_target(rows)
    x_train = torch.tensor(dataset.x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(dataset.y[train_idx], dtype=torch.float32)
    x_val = torch.tensor(dataset.x[val_idx], dtype=torch.float32)
    y_val = torch.tensor(dataset.y[val_idx], dtype=torch.float32)

    model = CascadeQualityClassifier(dataset.x.shape[1], len(LABELS), hidden=hidden)
    positives = y_train.sum(dim=0)
    negatives = y_train.shape[0] - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), min=0.05, max=30.0)
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
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n_seen += len(xb)
        val_loss, val_metrics = evaluate(model, x_val, y_val)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(total / max(n_seen, 1), 6),
            "val_loss": round(val_loss, 6),
            "macro_f1": val_metrics["macro_f1"],
        })
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, val_metrics = evaluate(model, x_val, y_val)
    dataset.feature_schema["decision_thresholds"] = {
        label: row.get("threshold")
        for label, row in (val_metrics.get("per_label") or {}).items()
    }
    report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "pack_dir": str(pack_dir),
            "model_output": str(model_output),
            "n_rows": len(rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
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
        "model_class": "CascadeQualityClassifier",
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


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = float(nn.functional.binary_cross_entropy_with_logits(logits, y).item())
        probs = torch.sigmoid(logits)
    per_label = {}
    f1s = []
    for idx, label in enumerate(LABELS):
        best = None
        for threshold in [x / 100.0 for x in range(5, 96, 5)]:
            metrics = binary_metrics(probs[:, idx], y[:, idx], threshold)
            if best is None or metrics["f1"] > best["f1"]:
                best = metrics
                best["threshold"] = threshold
        assert best is not None
        f1 = best["f1"]
        f1s.append(f1)
        per_label[label] = {
            "threshold": round(float(best["threshold"]), 3),
            "precision": round(float(best["precision"]), 6),
            "recall": round(float(best["recall"]), 6),
            "specificity": round(float(best["specificity"]), 6),
            "f1": round(f1, 6),
            "support": int(y[:, idx].sum().item()),
            "negative_support": int((1 - y[:, idx]).sum().item()),
        }
    return loss, {
        "macro_f1": round(sum(f1s) / max(len(f1s), 1), 6),
        "per_label": per_label,
    }


def binary_metrics(probs: torch.Tensor, y: torch.Tensor, threshold: float) -> dict[str, float]:
    pred = (probs >= threshold).float()
    tp = (pred * y).sum()
    fp_ = (pred * (1 - y)).sum()
    fn = ((1 - pred) * y).sum()
    tn = ((1 - pred) * (1 - y)).sum()
    precision = float(tp / torch.clamp(tp + fp_, min=1.0))
    recall = float(tp / torch.clamp(tp + fn, min=1.0))
    specificity = float(tn / torch.clamp(tn + fp_, min=1.0))
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def report_markdown(report: dict[str, Any]) -> str:
    meta = report.get("metadata") or {}
    schema = meta.get("feature_schema") or {}
    metrics = report.get("val_metrics") or {}
    lines = [
        "# Cascade Quality Classifier",
        "",
        f"Pack: `{meta.get('pack_dir')}`",
        f"Model: `{meta.get('model_output')}`",
        "",
        "## Samples",
        "",
        f"- rows: `{meta.get('n_rows')}`",
        f"- train: `{meta.get('n_train')}`",
        f"- validation: `{meta.get('n_val')}`",
        f"- feature dim: `{schema.get('feature_dim')}`",
        f"- type vocab: `{len(schema.get('type_vocab') or [])}`",
        f"- source vocab: `{len(schema.get('source_vocab') or [])}`",
        f"- EC1 vocab: `{len(schema.get('ec1_vocab') or [])}`",
        f"- type bigram vocab: `{len(schema.get('type_bigram_vocab') or [])}`",
        f"- source bigram vocab: `{len(schema.get('source_bigram_vocab') or [])}`",
        f"- EC1 bigram vocab: `{len(schema.get('ec1_bigram_vocab') or [])}`",
        "",
        "## Validation",
        "",
        f"- best val loss: `{report.get('best_val_loss')}`",
        f"- macro F1: `{metrics.get('macro_f1')}`",
        "",
        "| Label | Threshold | Precision | Recall | Specificity | F1 | Support | Neg support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in sorted((metrics.get("per_label") or {}).items()):
        lines.append(
            f"| `{label}` | {row.get('threshold')} | {row.get('precision')} | {row.get('recall')} | "
            f"{row.get('specificity')} | {row.get('f1')} | {row.get('support')} | {row.get('negative_support')} |"
        )
    lines.extend([
        "",
        "## Caveat",
        "",
        "This model is an early route-quality prior. Deterministic condition and compatibility metrics remain authoritative after route construction.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train cascade condition/compatibility classifier from a training pack")
    ap.add_argument("--pack-dir", default="results/shared/training_pack/broad_20260507")
    ap.add_argument("--model-output", default="results/shared/cascade_quality/pack_cascade_quality_20260507.pt")
    ap.add_argument("--report-output", default="results/shared/cascade_quality/pack_cascade_quality_20260507.json")
    ap.add_argument("--md-output", default="results/shared/cascade_quality/pack_cascade_quality_20260507.md")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-bits", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--min-vocab-count", type=int, default=5)
    ap.add_argument("--include-bigrams", action="store_true")
    args = ap.parse_args()
    report = train_cascade_quality_from_pack(
        pack_dir=Path(args.pack_dir),
        model_output=Path(args.model_output),
        report_output=Path(args.report_output),
        md_output=Path(args.md_output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_bits=args.n_bits,
        hidden=args.hidden,
        min_vocab_count=args.min_vocab_count,
        include_bigrams=args.include_bigrams,
    )
    print(json.dumps({
        "model_output": args.model_output,
        "best_val_loss": report["best_val_loss"],
        "macro_f1": report["val_metrics"]["macro_f1"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

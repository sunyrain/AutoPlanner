"""Train a scaffold-split skeleton reranker."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
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


@dataclass
class SkeletonRerankerDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    feature_schema: dict[str, Any]


class SkeletonReranker(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden, max(48, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(48, hidden // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_skeleton_reranker(
    *,
    split_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 12,
    batch_size: int = 512,
    lr: float = 1e-3,
    n_bits: int = 128,
    hidden: int = 192,
    min_vocab_count: int = 2,
    positive_threshold: float = 1.0,
    synthetic_negatives_per_positive: int = 0,
    loss_mode: str = "bce",
    seed: int = 42,
) -> dict[str, Any]:
    if loss_mode not in {"bce", "pairwise"}:
        raise ValueError("loss_mode must be bce or pairwise")
    torch.manual_seed(seed)
    np.random.seed(seed)
    split_rows = load_split_rows(split_dir)
    train_rows = augment_synthetic_negatives(
        split_rows["train"],
        n_per_positive=synthetic_negatives_per_positive,
        positive_threshold=positive_threshold,
        seed=seed,
    )
    active_rows = {**split_rows, "train": train_rows}
    schema = build_feature_schema(
        train_rows,
        n_bits=n_bits,
        min_vocab_count=min_vocab_count,
        positive_threshold=positive_threshold,
    )
    datasets = {
        split: build_dataset(rows, schema)
        for split, rows in active_rows.items()
    }
    train = datasets["train"]
    model = SkeletonReranker(train.x.shape[1], hidden=hidden)
    x_train = torch.tensor(train.x, dtype=torch.float32)
    y_train = torch.tensor(train.y, dtype=torch.float32)
    w_train = torch.tensor(train.weights, dtype=torch.float32)
    positives = torch.clamp(y_train.sum(), min=1.0)
    negatives = torch.clamp(torch.tensor(float(len(y_train))) - positives, min=1.0)
    pos_weight = torch.clamp(negatives / positives, min=0.1, max=50.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(x_train, y_train, w_train), batch_size=batch_size, shuffle=True)
    pairwise_groups = pairwise_group_indices(train.rows, train.y)

    history = []
    best_state = None
    best_val = float("-inf") if loss_mode == "pairwise" else float("inf")
    for epoch in range(max(1, epochs)):
        model.train()
        if loss_mode == "pairwise":
            train_loss, n_seen = train_pairwise_epoch(model, x_train, pairwise_groups, opt)
        else:
            total = 0.0
            n_seen = 0
            for xb, yb, wb in dl:
                logits = model(xb)
                loss_vec = nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    yb,
                    pos_weight=pos_weight,
                    reduction="none",
                )
                loss = (loss_vec * wb).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(xb)
                n_seen += len(xb)
            train_loss = total / max(n_seen, 1)
        val_metrics = evaluate_model(model, datasets["val"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": val_metrics["loss"],
            "val_group_hit1": val_metrics["ranking"]["group_hit1_rate"],
            "val_auc_like": val_metrics["classification"]["auc_like"],
        })
        selection_score = (
            float(val_metrics["ranking"].get("mrr") or 0.0)
            if loss_mode == "pairwise"
            else -float(val_metrics["loss"])
        )
        if selection_score > best_val:
            best_val = selection_score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = {
        split: evaluate_model(model, dataset)
        for split, dataset in datasets.items()
    }
    report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "split_dir": str(split_dir),
            "model_output": str(model_output),
            "n_rows": {split: len(rows) for split, rows in split_rows.items()},
            "n_train_after_synthetic": len(train_rows),
            "synthetic_negatives_per_positive": synthetic_negatives_per_positive,
            "loss_mode": loss_mode,
            "feature_schema": schema,
            "positive_threshold": positive_threshold,
            "pos_weight": round(float(pos_weight), 4),
        },
        "best_selection_score": round(float(best_val), 6),
        "metrics": final_metrics,
        "history": history,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_class": "SkeletonReranker",
        "feature_schema": schema,
        "hidden": hidden,
    }, model_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if md_output:
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(render_report(report), encoding="utf-8")
    return report


def load_split_rows(split_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"skeleton_prior_{split}.jsonl"
        if not path.exists():
            path = split_dir / split / "skeleton_pairwise_training.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        out[split] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return out


def augment_synthetic_negatives(
    rows: list[dict[str, Any]],
    *,
    n_per_positive: int,
    positive_threshold: float,
    seed: int,
) -> list[dict[str, Any]]:
    if n_per_positive <= 0:
        return list(rows)
    type_pool = sorted({
        str(value or "NONE")
        for row in rows
        for value in row.get("type_sequence") or []
        if str(value or "NONE")
    })
    if len(type_pool) < 2:
        return list(rows)
    out = list(rows)
    rng = random.Random(seed)
    for row_idx, row in enumerate(rows):
        if float(row.get("label") or 0.0) < positive_threshold:
            continue
        original = [str(value or "NONE") for value in row.get("type_sequence") or []]
        if not original:
            continue
        for neg_idx in range(n_per_positive):
            corrupted = corrupt_type_sequence(original, type_pool, rng=rng, variant=neg_idx)
            if corrupted == original:
                continue
            neg = dict(row)
            neg["type_sequence"] = corrupted
            neg["label"] = 0.0
            neg["label_type"] = "synthetic_skeleton_negative"
            neg["source"] = "synthetic_contrastive"
            neg["source_path"] = "synthetic:skeleton_type_corruption"
            neg["skeleton_id"] = f"synthetic_{row_idx}_{neg_idx}"
            out.append(neg)
    return out


def corrupt_type_sequence(
    original: list[str],
    type_pool: list[str],
    *,
    rng: random.Random,
    variant: int,
) -> list[str]:
    out = list(original)
    if len(out) > 1 and variant % 3 == 1:
        shift = 1 + (variant % (len(out) - 1))
        return out[shift:] + out[:shift]
    if len(out) > 1 and variant % 3 == 2:
        return list(reversed(out))
    pos = variant % len(out)
    choices = [value for value in type_pool if value != out[pos]]
    if not choices:
        return out
    out[pos] = choices[rng.randrange(len(choices))]
    return out


def build_feature_schema(
    rows: list[dict[str, Any]],
    *,
    n_bits: int,
    min_vocab_count: int,
    positive_threshold: float,
) -> dict[str, Any]:
    type_vocab = vocab(rows, "type_sequence", min_count=min_vocab_count, max_size=96)
    ec1_vocab = vocab(rows, "ec1_sequence", min_count=min_vocab_count, max_size=32)
    schema = {
        "n_bits": n_bits,
        "max_steps": 8,
        "type_vocab": type_vocab,
        "ec1_vocab": ec1_vocab,
        "positive_threshold": positive_threshold,
        "feature_contract": "target_fp + sequence-position type/ec + scalar route-shape features; source/DOI/domain/opmode excluded to avoid metadata leakage",
    }
    schema["feature_dim"] = len(row_features(rows[0], schema)) if rows else 0
    return schema


def vocab(rows: list[dict[str, Any]], key: str, *, min_count: int, max_size: int) -> list[str]:
    counts = Counter(str(value or "NONE") for row in rows for value in row.get(key) or [])
    return [token for token, count in counts.most_common(max_size) if count >= min_count and token]


def build_dataset(rows: list[dict[str, Any]], schema: dict[str, Any]) -> SkeletonRerankerDataset:
    x_rows = [row_features(row, schema) for row in rows]
    threshold = float(schema.get("positive_threshold") or 1.0)
    labels = np.asarray([1.0 if float(row.get("label") or 0.0) >= threshold else 0.0 for row in rows], dtype=np.float32)
    weights = np.asarray([row_weight(row, positive=bool(label)) for row, label in zip(rows, labels)], dtype=np.float32)
    return SkeletonRerankerDataset(
        rows=rows,
        x=np.asarray(x_rows, dtype=np.float32),
        y=labels,
        weights=weights,
        feature_schema=schema,
    )


def pairwise_group_indices(rows: list[dict[str, Any]], labels: np.ndarray) -> list[tuple[list[int], list[int]]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (
            canonical_smiles(row.get("target_smiles") or ""),
            int(row.get("depth") or len(row.get("type_sequence") or []) or 0),
        )
        groups[key].append(idx)
    out = []
    for indices in groups.values():
        pos = [idx for idx in indices if labels[idx] >= 0.5]
        neg = [idx for idx in indices if labels[idx] < 0.5]
        if pos and neg:
            out.append((pos, neg))
    return out


def train_pairwise_epoch(
    model: SkeletonReranker,
    x_train: torch.Tensor,
    groups: list[tuple[list[int], list[int]]],
    opt: torch.optim.Optimizer,
) -> tuple[float, int]:
    if not groups:
        return 0.0, 0
    total = 0.0
    n_seen = 0
    for pos_idx, neg_idx in groups:
        pos_logits = model(x_train[pos_idx])
        neg_logits = model(x_train[neg_idx])
        loss = nn.functional.softplus(neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        n_pairs = len(pos_idx) * len(neg_idx)
        total += float(loss.item()) * n_pairs
        n_seen += n_pairs
    return total / max(n_seen, 1), n_seen


def row_features(row: dict[str, Any], schema: dict[str, Any]) -> np.ndarray:
    n_bits = int(schema.get("n_bits") or 128)
    target_fp = fp(row.get("target_smiles"), n_bits=n_bits)
    max_steps = int(schema.get("max_steps") or 8)
    types = [str(value or "NONE") for value in row.get("type_sequence") or []]
    ec1s = [str(value or "NONE") for value in row.get("ec1_sequence") or []]
    type_pos = position_one_hot(types, schema.get("type_vocab") or [], max_steps=max_steps)
    ec_pos = position_one_hot(ec1s, schema.get("ec1_vocab") or [], max_steps=max_steps)
    mol = Chem.MolFromSmiles(row.get("target_smiles") or "")
    heavy = float(mol.GetNumHeavyAtoms()) if mol is not None else 0.0
    depth = int(row.get("depth") or len(types) or 0)
    ec_known = sum(1 for value in ec1s if value not in {"", "NONE", "0"})
    scalar = np.asarray([
        depth / max(max_steps, 1),
        heavy / 100.0,
        len(set(types)) / max(max_steps, 1),
        ec_known / max(depth, 1),
    ], dtype=np.float32)
    return np.concatenate([target_fp, type_pos, ec_pos, scalar])


def row_weight(row: dict[str, Any], *, positive: bool) -> float:
    base = 3.0 if positive else 1.0
    if row.get("source") == "benchmark_gt":
        base *= 1.5
    return base


def fp(smiles: str | None, *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def position_one_hot(values: list[str], vocab_values: list[str], *, max_steps: int) -> np.ndarray:
    index = {token: idx for idx, token in enumerate(vocab_values)}
    arr = np.zeros(max_steps * len(vocab_values), dtype=np.float32)
    for pos, value in enumerate(values[:max_steps]):
        vocab_idx = index.get(value)
        if vocab_idx is None:
            continue
        arr[pos * len(vocab_values) + vocab_idx] = 1.0
    return arr


def evaluate_model(model: SkeletonReranker, dataset: SkeletonRerankerDataset) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(dataset.x, dtype=torch.float32)
        y = torch.tensor(dataset.y, dtype=torch.float32)
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y).item() if len(y) else 0.0
        probs = torch.sigmoid(logits).cpu().numpy()
    return {
        "loss": round(float(loss), 6),
        "classification": classification_metrics(dataset.y, probs),
        "ranking": ranking_metrics(dataset.rows, dataset.y, probs),
    }


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if len(labels) == 0:
        return {"accuracy": None, "auc_like": None}
    preds = scores >= 0.5
    accuracy = float((preds == (labels >= 0.5)).mean())
    pos_scores = [float(score) for label, score in zip(labels, scores) if label >= 0.5]
    neg_scores = [float(score) for label, score in zip(labels, scores) if label < 0.5]
    auc_like = pairwise_auc(pos_scores, neg_scores)
    return {
        "accuracy": round(accuracy, 4),
        "auc_like": round(auc_like, 4) if auc_like is not None else None,
        "positive_count": len(pos_scores),
        "negative_count": len(neg_scores),
        "mean_positive_score": round(mean_safe(pos_scores), 4) if pos_scores else None,
        "mean_negative_score": round(mean_safe(neg_scores), 4) if neg_scores else None,
    }


def ranking_metrics(rows: list[dict[str, Any]], labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for row, label, score in zip(rows, labels, scores):
        key = (
            canonical_smiles(row.get("target_smiles") or ""),
            int(row.get("depth") or len(row.get("type_sequence") or []) or 0),
        )
        groups[key].append((float(score), float(label)))
    hit1 = 0
    mrr_values = []
    usable = 0
    for values in groups.values():
        if not any(label >= 0.5 for _, label in values) or len(values) < 2:
            continue
        usable += 1
        ranked = sorted(values, key=lambda item: item[0], reverse=True)
        hit1 += int(ranked[0][1] >= 0.5)
        first_pos = next((idx for idx, (_, label) in enumerate(ranked, 1) if label >= 0.5), None)
        if first_pos:
            mrr_values.append(1.0 / first_pos)
    denom = usable or 1
    return {
        "groups": usable,
        "group_hit1": hit1,
        "group_hit1_rate": hit1 / denom,
        "mrr": round(mean_safe(mrr_values), 4) if mrr_values else None,
    }


def pairwise_auc(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    if not pos_scores or not neg_scores:
        return None
    total = 0.0
    count = 0
    for pos in pos_scores:
        for neg in neg_scores:
            if pos > neg:
                total += 1.0
            elif math.isclose(pos, neg):
                total += 0.5
            count += 1
    return total / max(count, 1)


def mean_safe(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Skeleton Reranker Training",
        "",
        f"- split dir: `{report['metadata']['split_dir']}`",
        f"- model: `{report['metadata']['model_output']}`",
        f"- rows: `{report['metadata']['n_rows']}`",
        f"- train rows after synthetic negatives: `{report['metadata']['n_train_after_synthetic']}`",
        f"- synthetic negatives per positive: `{report['metadata']['synthetic_negatives_per_positive']}`",
        f"- positive threshold: `{report['metadata']['positive_threshold']}`",
        f"- loss mode: `{report['metadata']['loss_mode']}`",
        f"- best selection score: `{report['best_selection_score']}`",
        "",
        "## Metrics",
        "",
    ]
    for split, metrics in report["metrics"].items():
        cls = metrics["classification"]
        rank = metrics["ranking"]
        lines.extend([
            f"### {split}",
            f"- loss: `{metrics['loss']}`",
            f"- auc_like: `{cls['auc_like']}`",
            f"- positives / negatives: `{cls['positive_count']}` / `{cls['negative_count']}`",
            f"- group hit@1: `{rank['group_hit1']}` / `{rank['groups']}` (`{rank['group_hit1_rate']:.3f}`)",
            f"- MRR: `{rank['mrr']}`",
            "",
        ])
    lines.append("This reranker is trained on scaffold-split skeleton priors and intentionally excludes candidate source as a feature to avoid benchmark-vs-planner leakage.")
    lines.append("It is a research artifact only; do not enable it as a default runtime ranker until holdout group hit@1 improves materially.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", default="results/shared/skeleton_prior_split/scaffold_20260507")
    parser.add_argument("--model-output", default="results/shared/skeleton_reranker/scaffold_20260507.pt")
    parser.add_argument("--report-output", default="results/shared/skeleton_reranker/scaffold_20260507.json")
    parser.add_argument("--md-output", default="results/shared/skeleton_reranker/scaffold_20260507.md")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-bits", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--min-vocab-count", type=int, default=2)
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--synthetic-negatives-per-positive", type=int, default=0)
    parser.add_argument("--loss-mode", default="bce", choices=["bce", "pairwise"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train_skeleton_reranker(
        split_dir=Path(args.split_dir),
        model_output=Path(args.model_output),
        report_output=Path(args.report_output),
        md_output=Path(args.md_output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_bits=args.n_bits,
        hidden=args.hidden,
        min_vocab_count=args.min_vocab_count,
        positive_threshold=args.positive_threshold,
        synthetic_negatives_per_positive=args.synthetic_negatives_per_positive,
        loss_mode=args.loss_mode,
        seed=args.seed,
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()

"""Train a candidate reranker from a consolidated training pack.

This trainer targets the largest observed bottleneck after one-step proposal:
ranking the candidates that are already in the pool under route context. It
uses product/reactant fingerprints, source/rank metadata, and value-function
features exported in ``candidate_ranking.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
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

NUMERIC_FEATURES = [
    "candidate_score",
    "stock_fraction",
    "main_reduction",
    "has_ec",
    "has_evidence",
    "large_aux_penalty",
    "self_loop",
]
SOURCE_VALUES = ["retrochimera", "enzyformer", "v3_retrieval", "enzexpand", "fake", "candidate"]
METADATA_FEATURES = [
    "has_ec",
    "has_type",
    "has_doi",
    "has_uniprot",
    "has_T",
    "has_pH",
    "has_T_and_pH",
    "T_scaled",
    "pH_scaled",
    "has_solvent",
    "has_catalyst",
    "has_enzyme_uid",
    "has_cofactor",
    "has_condition_match",
]


@dataclass
class RankerDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    groups: list[tuple[str, int]]
    feature_schema: dict[str, Any]


class PackCandidateRanker(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden, max(32, hidden // 3)),
            nn.GELU(),
            nn.Linear(max(32, hidden // 3), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_candidate_rows(pack_dir: Path) -> list[dict[str, Any]]:
    path = pack_dir / "candidate_ranking.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing candidate_ranking.jsonl in {pack_dir}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("product") and row.get("candidate"):
            rows.append(row)
    return rows


def build_dataset(rows: list[dict[str, Any]], *, n_bits: int = 128) -> RankerDataset:
    x_rows = []
    labels = []
    weights = []
    groups = []
    for row in rows:
        x_rows.append(row_features(row, n_bits=n_bits))
        labels.append(float(row.get("label") or 0.0))
        weights.append(float(row.get("weight") or 1.0))
        groups.append((str(row.get("route_id") or ""), int(row.get("step_index") or 0)))
    schema = {
        "n_bits": n_bits,
        "numeric_features": NUMERIC_FEATURES,
        "source_values": SOURCE_VALUES,
        "metadata_features": METADATA_FEATURES,
        "feature_dim": len(x_rows[0]) if x_rows else 0,
        "label_contract": {
            "benchmark_exact": 1.0,
            "planner_selected_positive": 0.75,
            "planner_selected_weak": 0.5,
            "negative": 0.0,
        },
    }
    return RankerDataset(
        rows=rows,
        x=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(labels, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        groups=groups,
        feature_schema=schema,
    )


def row_features(row: dict[str, Any], *, n_bits: int) -> np.ndarray:
    cand = row.get("candidate") or {}
    product_fp = fp(row.get("product"), n_bits=n_bits)
    reactants = [cand.get("main_reactant") or ""]
    reactants.extend(cand.get("aux_reactants") or [])
    reactant_fp = fp(".".join(x for x in reactants if x), n_bits=n_bits)
    exported = row.get("features") or {}
    numeric = [float(exported.get(name) or 0.0) for name in NUMERIC_FEATURES]
    rank = float(row.get("rank") or 0.0)
    rank_features = [
        1.0 / max(rank, 1.0),
        math.log1p(rank) / 5.0,
        float(bool(row.get("gt_available"))),
    ]
    source = str(cand.get("source") or "").lower()
    source_features = [1.0 if source == value else 0.0 for value in SOURCE_VALUES]
    ec = str(cand.get("ec") or "")
    type_text = str(cand.get("type") or cand.get("reaction_type") or "")
    metadata_features = candidate_metadata_features(cand)
    return np.concatenate([
        product_fp,
        reactant_fp,
        np.asarray(numeric + rank_features + source_features + metadata_features, dtype=np.float32),
    ])


def candidate_metadata_features(cand: dict[str, Any]) -> list[float]:
    evidence = cand.get("evidence") or {}
    t_value = safe_float(cand.get("T"))
    ph_value = safe_float(cand.get("pH"))
    values = {
        "has_ec": float(bool(cand.get("ec"))),
        "has_type": float(bool(cand.get("type") or cand.get("reaction_type"))),
        "has_doi": float(bool(cand.get("doi") or evidence.get("doi"))),
        "has_uniprot": float(bool(cand.get("uniprot_accession") or evidence.get("uniprot_accession"))),
        "has_T": float(t_value is not None),
        "has_pH": float(ph_value is not None),
        "has_T_and_pH": float(t_value is not None and ph_value is not None),
        "T_scaled": float(t_value or 0.0) / 100.0,
        "pH_scaled": float(ph_value or 0.0) / 14.0,
        "has_solvent": float(bool(cand.get("solvent"))),
        "has_catalyst": float(bool(cand.get("catalyst"))),
        "has_enzyme_uid": float(bool(cand.get("enzyme_uid"))),
        "has_cofactor": float(bool(cand.get("cofactor") or evidence.get("cofactor"))),
        "has_condition_match": float(bool(cand.get("condition_match") or evidence.get("condition_match"))),
    }
    return [values[name] for name in METADATA_FEATURES]


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fp(smiles: str | None, *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def split_by_target(rows: list[dict[str, Any]], val_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    targets = sorted({canonical_smiles(row.get("target_smiles") or row.get("product") or "") for row in rows})
    n_val = max(1, int(round(len(targets) * val_fraction))) if targets else 0
    val_targets = set(targets[::max(1, len(targets) // n_val)][:n_val]) if n_val else set()
    train_idx = []
    val_idx = []
    for idx, row in enumerate(rows):
        target = canonical_smiles(row.get("target_smiles") or row.get("product") or "")
        (val_idx if target in val_targets else train_idx).append(idx)
    if not train_idx or not val_idx:
        pivot = max(1, int(len(rows) * (1.0 - val_fraction)))
        train_idx = list(range(pivot))
        val_idx = list(range(pivot, len(rows)))
    return train_idx, val_idx


def train_candidate_ranker_from_pack(
    *,
    pack_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 8,
    batch_size: int = 1024,
    lr: float = 1e-3,
    n_bits: int = 128,
    hidden: int = 192,
    seed: int = 42,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    rows = load_candidate_rows(pack_dir)
    dataset = build_dataset(rows, n_bits=n_bits)
    train_idx, val_idx = split_by_target(rows)
    x_train = torch.tensor(dataset.x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(dataset.y[train_idx], dtype=torch.float32)
    w_train = torch.tensor(dataset.weights[train_idx], dtype=torch.float32)
    x_val = torch.tensor(dataset.x[val_idx], dtype=torch.float32)
    y_val = torch.tensor(dataset.y[val_idx], dtype=torch.float32)
    model = PackCandidateRanker(dataset.x.shape[1], hidden=hidden)

    positives = float((y_train >= 0.75).sum().item())
    negatives = max(float((y_train < 0.75).sum().item()), 1.0)
    pos_boost = negatives / max(positives, 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(x_train, y_train, w_train), batch_size=batch_size, shuffle=True)

    history = []
    best_state = None
    best_val = float("inf")
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb, wb in dl:
            logits = model(xb)
            loss_vec = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            class_weight = torch.where(yb >= 0.75, torch.tensor(pos_boost, dtype=torch.float32), torch.tensor(1.0))
            loss = (loss_vec * wb * class_weight).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n_seen += len(xb)
        val_loss, val_auc_like = evaluate_loss(model, x_val, y_val)
        row = {
            "epoch": epoch + 1,
            "train_loss": round(total / max(n_seen, 1), 6),
            "val_loss": round(val_loss, 6),
            "val_pair_auc": round(val_auc_like, 6),
        }
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    rerank_metrics = evaluate_reranking(model, dataset, val_idx)
    report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "pack_dir": str(pack_dir),
            "model_output": str(model_output),
            "n_rows": len(rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_bits": n_bits,
            "hidden": hidden,
            "feature_schema": dataset.feature_schema,
            "positive_threshold": 0.75,
            "pos_boost": round(float(pos_boost), 6),
        },
        "best_val_loss": round(float(best_val), 6),
        "history": history,
        "reranking": rerank_metrics,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_class": "PackCandidateRanker",
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


def evaluate_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = float(nn.functional.binary_cross_entropy_with_logits(logits, y).item())
        scores = torch.sigmoid(logits).cpu().numpy()
    labels = y.cpu().numpy()
    pos = scores[labels >= 0.75]
    neg = scores[labels < 0.75]
    if len(pos) == 0 or len(neg) == 0:
        return loss, 0.0
    # Lightweight pairwise AUC approximation.
    sample_pos = pos[: min(len(pos), 500)]
    sample_neg = neg[: min(len(neg), 500)]
    auc = float((sample_pos[:, None] > sample_neg[None, :]).mean())
    return loss, auc


def evaluate_reranking(model: nn.Module, dataset: RankerDataset, val_idx: list[int]) -> dict[str, Any]:
    model.eval()
    x_val = torch.tensor(dataset.x[val_idx], dtype=torch.float32)
    with torch.no_grad():
        scores = torch.sigmoid(model(x_val)).cpu().numpy()
    by_group: dict[tuple[str, int], list[tuple[int, float, dict[str, Any]]]] = {}
    for local_idx, score in zip(val_idx, scores):
        group = dataset.groups[local_idx]
        by_group.setdefault(group, []).append((local_idx, float(score), dataset.rows[local_idx]))

    groups_with_exact = []
    for items in by_group.values():
        if any(row.get("label_type") == "benchmark_exact" for _, _, row in items):
            groups_with_exact.append(items)

    def topk_hit(items, key, k: int) -> bool:
        ordered = sorted(items, key=key)
        return any(row.get("label_type") == "benchmark_exact" for _, _, row in ordered[:k])

    n = len(groups_with_exact)
    base_top1 = sum(topk_hit(items, lambda item: int(item[2].get("rank") or 999999), 1) for items in groups_with_exact)
    base_top5 = sum(topk_hit(items, lambda item: int(item[2].get("rank") or 999999), 5) for items in groups_with_exact)
    model_top1 = sum(topk_hit(items, lambda item: -item[1], 1) for items in groups_with_exact)
    model_top5 = sum(topk_hit(items, lambda item: -item[1], 5) for items in groups_with_exact)
    return {
        "groups_with_exact_gt": n,
        "base_top1_exact": round(base_top1 / max(n, 1), 6),
        "model_top1_exact": round(model_top1 / max(n, 1), 6),
        "base_top5_exact": round(base_top5 / max(n, 1), 6),
        "model_top5_exact": round(model_top5 / max(n, 1), 6),
    }


def report_markdown(report: dict[str, Any]) -> str:
    meta = report.get("metadata") or {}
    rr = report.get("reranking") or {}
    lines = [
        "# Pack Candidate Ranker",
        "",
        f"Pack: `{meta.get('pack_dir')}`",
        f"Model: `{meta.get('model_output')}`",
        "",
        "## Samples",
        "",
        f"- rows: `{meta.get('n_rows')}`",
        f"- train: `{meta.get('n_train')}`",
        f"- validation: `{meta.get('n_val')}`",
        f"- feature dim: `{(meta.get('feature_schema') or {}).get('feature_dim')}`",
        "",
        "## Validation Reranking",
        "",
        f"- exact-GT groups: `{rr.get('groups_with_exact_gt')}`",
        f"- base top-1 exact: `{rr.get('base_top1_exact')}`",
        f"- model top-1 exact: `{rr.get('model_top1_exact')}`",
        f"- base top-5 exact: `{rr.get('base_top5_exact')}`",
        f"- model top-5 exact: `{rr.get('model_top5_exact')}`",
        "",
        "## Caveat",
        "",
        "This is a candidate-pool reranker. It can improve selection when a useful candidate is already present; it cannot fix generator exact misses by itself.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a candidate reranker from training_pack candidate_ranking.jsonl")
    ap.add_argument("--pack-dir", default="results/shared/training_pack/broad_20260507")
    ap.add_argument("--model-output", default="results/shared/candidate_ranker/pack_candidate_ranker_20260507.pt")
    ap.add_argument("--report-output", default="results/shared/candidate_ranker/pack_candidate_ranker_20260507.json")
    ap.add_argument("--md-output", default="results/shared/candidate_ranker/pack_candidate_ranker_20260507.md")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-bits", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=192)
    args = ap.parse_args()

    report = train_candidate_ranker_from_pack(
        pack_dir=Path(args.pack_dir),
        model_output=Path(args.model_output),
        report_output=Path(args.report_output),
        md_output=Path(args.md_output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_bits=args.n_bits,
        hidden=args.hidden,
    )
    print(json.dumps({
        "model_output": args.model_output,
        "best_val_loss": report["best_val_loss"],
        "reranking": report["reranking"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

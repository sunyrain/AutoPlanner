"""Train source-specific proposal rankers and a lightweight source gate."""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.route_tree.source_gate import molecule_class_flags
from cascade_planner.vnext.features import (
    candidate_feature_dim,
    candidate_feature_vector,
    morgan_fp,
    stable_bucket,
)
from cascade_planner.vnext.models import CandidatePoolCrossAttentionRanker
from cascade_planner.vnext.schema import SOURCE_BUDGET_GROUPS


SOURCE_GROUP_TO_OUTPUT = {
    "chemical": "chemical_proposal_ranker.pt",
    "enzymatic": "enzymatic_proposal_ranker.pt",
    "rhea_retrorules": "rhea_retrorules_ranker.pt",
}


class SourceGateNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_classes: int = len(SOURCE_BUDGET_GROUPS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_all_from_proposal_pack(
    *,
    proposal_pack: Path,
    output_dir: Path,
    groups: list[str],
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    n_bits: int = 128,
    max_candidates: int = 8,
    d_model: int = 128,
    device: str = "auto",
    max_rows_per_group: int | None = None,
    feature_workers: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_t = _select_device(device)
    reports: dict[str, Any] = {}
    for group in groups:
        if group == "source_gate":
            reports[group] = train_source_gate(
                proposal_pack=proposal_pack,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                n_bits=n_bits,
                device=device_t,
                max_rows=max_rows_per_group,
                feature_workers=feature_workers,
            )
        else:
            reports[group] = train_ranker_for_group(
                proposal_pack=proposal_pack,
                group=group,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                n_bits=n_bits,
                max_candidates=max_candidates,
                d_model=d_model,
                device=device_t,
                max_rows=max_rows_per_group,
                feature_workers=feature_workers,
            )
    summary = {
        "schema_version": "proposal_rankers.v1",
        "proposal_pack": str(proposal_pack),
        "output_dir": str(output_dir),
        "reports": reports,
    }
    (output_dir / "proposal_rankers.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "proposal_rankers.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def train_ranker_for_group(
    *,
    proposal_pack: Path,
    group: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    n_bits: int,
    max_candidates: int,
    d_model: int,
    device: torch.device,
    max_rows: int | None,
    feature_workers: int,
) -> dict[str, Any]:
    rows = _load_rows_for_group(proposal_pack, group=group, max_rows=max_rows)
    if not rows:
        raise ValueError(f"no proposal rows for group {group}")
    x, mask, labels = _ranker_arrays(rows, n_bits=n_bits, max_candidates=max_candidates, feature_workers=feature_workers)
    train_idx, val_idx = _split_by_product(rows)
    model = CandidatePoolCrossAttentionRanker(candidate_feature_dim=x.shape[-1], d_model=d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(
        TensorDataset(
            torch.tensor(x[train_idx]),
            torch.tensor(mask[train_idx]),
            torch.tensor(labels[train_idx]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    history = []
    best = float("inf")
    best_state = None
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for bx, bm, by in dl:
            bx = bx.to(device)
            bm = bm.to(device)
            by = by.to(device)
            out = model(bx, bm.bool())
            loss = _masked_bce(out["candidate_logits"], by, bm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(bx)
            n_seen += len(bx)
        metrics = _eval_ranker(model, x, mask, labels, val_idx, device=device)
        row = {"epoch": epoch + 1, "train_loss": round(total / max(n_seen, 1), 6), **_round(metrics)}
        history.append(row)
        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model_path = output_dir / SOURCE_GROUP_TO_OUTPUT[group]
    metadata = {
        "model_kind": "proposal_ranker",
        "source_group": group,
        "proposal_pack": str(proposal_pack),
        "n_rows": len(rows),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_bits": n_bits,
        "max_candidates": max_candidates,
        "candidate_feature_dim": int(x.shape[-1]),
        "d_model": d_model,
        "device": str(device),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "feature_schema": {
                "model_kind": "proposal_ranker",
                "source_group": group,
                "n_bits": n_bits,
                "max_candidates": max_candidates,
                "candidate_feature_dim": int(x.shape[-1]),
            },
            "model_config": {"d_model": d_model},
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        model_path,
    )
    report = {"metadata": {**metadata, "model_output": str(model_path)}, "best_val_loss": round(best, 6), "history": history}
    (output_dir / f"{group}_proposal_ranker.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def train_source_gate(
    *,
    proposal_pack: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    n_bits: int,
    device: torch.device,
    max_rows: int | None,
    feature_workers: int,
) -> dict[str, Any]:
    rows = _load_rows_for_group(proposal_pack, group=None, max_rows=max_rows)
    if int(feature_workers or 1) > 1:
        with ProcessPoolExecutor(max_workers=int(feature_workers)) as pool:
            x = np.asarray(
                list(pool.map(_source_gate_features_from_args, ((row, n_bits) for row in rows), chunksize=256)),
                dtype=np.float32,
            )
    else:
        x = np.asarray([_source_gate_features(row, n_bits=n_bits) for row in rows], dtype=np.float32)
    y = np.asarray([SOURCE_BUDGET_GROUPS.index(_row_group(row)) for row in rows], dtype=np.int64)
    train_idx, val_idx = _split_by_product(rows)
    model = SourceGateNetwork(x.shape[-1], n_classes=len(SOURCE_BUDGET_GROUPS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(
        TensorDataset(torch.tensor(x[train_idx]), torch.tensor(y[train_idx])),
        batch_size=batch_size,
        shuffle=True,
    )
    history = []
    best = float("inf")
    best_state = None
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for bx, by in dl:
            bx = bx.to(device)
            by = by.to(device)
            logits = model(bx)
            loss = nn.functional.cross_entropy(logits, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(bx)
            n_seen += len(bx)
        metrics = _eval_source_gate(model, x, y, val_idx, device=device)
        row = {"epoch": epoch + 1, "train_loss": round(total / max(n_seen, 1), 6), **_round(metrics)}
        history.append(row)
        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model_path = output_dir / "source_gate.pt"
    metadata = {
        "model_kind": "source_gate",
        "proposal_pack": str(proposal_pack),
        "n_rows": len(rows),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_bits": n_bits,
        "input_dim": int(x.shape[-1]),
        "source_budget_groups": list(SOURCE_BUDGET_GROUPS),
        "class_counts": dict(Counter(SOURCE_BUDGET_GROUPS[int(i)] for i in y)),
        "device": str(device),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "feature_schema": {"model_kind": "source_gate", "n_bits": n_bits, "input_dim": int(x.shape[-1])},
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        model_path,
    )
    report = {"metadata": {**metadata, "model_output": str(model_path)}, "best_val_loss": round(best, 6), "history": history}
    (output_dir / "source_gate.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _ranker_arrays(
    rows: list[dict[str, Any]],
    *,
    n_bits: int,
    max_candidates: int,
    feature_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_dim = candidate_feature_dim(n_bits)
    x = np.zeros((len(rows), max_candidates, feat_dim), dtype=np.float32)
    mask = np.zeros((len(rows), max_candidates), dtype=np.float32)
    labels = np.zeros((len(rows), max_candidates), dtype=np.float32)
    if int(feature_workers or 1) > 1:
        tasks = ((row, n_bits, max_candidates, feat_dim) for row in rows)
        with ProcessPoolExecutor(max_workers=int(feature_workers)) as pool:
            for i, (row_x, row_mask, row_labels) in enumerate(pool.map(_ranker_arrays_for_row, tasks, chunksize=128)):
                x[i] = row_x
                mask[i] = row_mask
                labels[i] = row_labels
        return x, mask, labels
    for i, row in enumerate(rows):
        x[i], mask[i], labels[i] = _ranker_arrays_for_row((row, n_bits, max_candidates, feat_dim))
    return x, mask, labels


def _ranker_arrays_for_row(args: tuple[dict[str, Any], int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row, n_bits, max_candidates, feat_dim = args
    row_x = np.zeros((max_candidates, feat_dim), dtype=np.float32)
    row_mask = np.zeros(max_candidates, dtype=np.float32)
    row_labels = np.zeros(max_candidates, dtype=np.float32)
    positives = set(int(idx) for idx in row.get("positive_candidate_indices") or [])
    for j, candidate in enumerate((row.get("candidate_pool") or [])[:max_candidates]):
        row_x[j] = candidate_feature_vector(
            row.get("product") or "",
            candidate,
            rank=candidate.get("rank") or j + 1,
            gt_available=True,
            n_bits=n_bits,
        )
        row_mask[j] = 1.0
        row_labels[j] = 1.0 if j in positives else 0.0
    return row_x, row_mask, row_labels


def _source_gate_features(row: dict[str, Any], *, n_bits: int) -> np.ndarray:
    flags = molecule_class_flags(row.get("product"))
    conditions = row.get("conditions") or {}
    ec1 = _ec1(row.get("ec"))
    values = [
        ec1 / 7.0,
        float(bool(row.get("ec"))),
        stable_bucket(str(row.get("reaction_type") or ""), 32) / 31.0,
        float(conditions.get("T") or 0.0) / 100.0,
        float(conditions.get("pH") or 0.0) / 14.0,
        float(bool(conditions.get("T"))),
        float(bool(conditions.get("pH"))),
        float(flags.get("nucleotide")),
        float(flags.get("carbohydrate")),
        float(flags.get("peptide_like")),
        float(flags.get("small_organic")),
        float(flags.get("aromatic_chemical")),
        float(flags.get("large_molecule")),
    ]
    return np.concatenate([morgan_fp(row.get("product"), n_bits=n_bits), np.asarray(values, dtype=np.float32)]).astype(np.float32)


def _source_gate_features_from_args(args: tuple[dict[str, Any], int]) -> np.ndarray:
    row, n_bits = args
    return _source_gate_features(row, n_bits=n_bits)


def _load_rows_for_group(proposal_pack: Path, *, group: str | None, max_rows: int | None) -> list[dict[str, Any]]:
    rows = []
    with Path(proposal_pack).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if group is None or _row_group(row) == group:
                rows.append(row)
                if max_rows is not None and len(rows) >= max_rows:
                    break
    return rows


def _row_group(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "").lower()
    if source in {
        "uspto50k",
        "retrochimera",
        "chemical",
        "chemtemplates",
        "chem_enzy_onestep",
        "chem_enzy_graphfp",
        "chem_enzy_onmt",
    } or (not row.get("ec") and source != "rhea"):
        return "chemical"
    if source in {"rhea", "retrorules"}:
        return "rhea_retrorules"
    if row.get("ec") or source in {"ecreact", "enzymatic_retro_data", "enzyformer", "enzexpand", "v3_retrieval"}:
        return "enzymatic"
    return "fallback"


def _eval_ranker(
    model: CandidatePoolCrossAttentionRanker,
    x: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    idx: list[int],
    *,
    device: torch.device,
) -> dict[str, float]:
    if not idx:
        return {"val_loss": 0.0, "val_top1_positive": 0.0, "val_top5_positive": 0.0}
    model.eval()
    with torch.no_grad():
        bx = torch.tensor(x[idx], device=device)
        bm = torch.tensor(mask[idx], device=device)
        by = torch.tensor(labels[idx], device=device)
        logits = model(bx, bm.bool())["candidate_logits"]
        loss = _masked_bce(logits, by, bm)
        top = torch.argmax(logits, dim=1)
        top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
        hits = []
        hits5 = []
        for i, j in enumerate(top.tolist()):
            positives = (by[i] >= 0.75) & (bm[i] > 0)
            if positives.any():
                hits.append(float(bool(positives[j])))
                hits5.append(float(any(bool(positives[k]) for k in top5[i].tolist())))
    return {
        "val_loss": float(loss.item()),
        "val_top1_positive": float(np.mean(hits)) if hits else 0.0,
        "val_top5_positive": float(np.mean(hits5)) if hits5 else 0.0,
    }


def _eval_source_gate(model: SourceGateNetwork, x: np.ndarray, y: np.ndarray, idx: list[int], *, device: torch.device) -> dict[str, float]:
    if not idx:
        return {"val_loss": 0.0, "val_acc": 0.0}
    model.eval()
    with torch.no_grad():
        bx = torch.tensor(x[idx], device=device)
        by = torch.tensor(y[idx], device=device)
        logits = model(bx)
        loss = nn.functional.cross_entropy(logits, by)
        pred = torch.argmax(logits, dim=1)
        acc = float((pred == by).float().mean().item())
    return {"val_loss": float(loss.item()), "val_acc": acc}


def _masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def _split_by_product(rows: list[dict[str, Any]], val_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    products = sorted({str(row.get("product") or "") for row in rows})
    if len(products) >= 2:
        n_val = max(1, int(round(len(products) * val_fraction)))
        stride = max(1, len(products) // n_val)
        val_products = set(products[::stride][:n_val])
        train_idx = [idx for idx, row in enumerate(rows) if str(row.get("product") or "") not in val_products]
        val_idx = [idx for idx, row in enumerate(rows) if idx not in train_idx]
    else:
        pivot = max(1, int(len(rows) * (1.0 - val_fraction)))
        train_idx = list(range(pivot))
        val_idx = list(range(pivot, len(rows)))
    if not train_idx or not val_idx:
        pivot = max(1, int(len(rows) * (1.0 - val_fraction)))
        train_idx = list(range(pivot))
        val_idx = list(range(pivot, len(rows)))
    return train_idx, val_idx


def _ec1(value: Any) -> int:
    try:
        ec1 = int(str(value).split(".", 1)[0])
    except (TypeError, ValueError):
        return 0
    return ec1 if 1 <= ec1 <= 7 else 0


def _select_device(device: str) -> torch.device:
    requested = str(device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def _round(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in metrics.items()}


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Proposal Rankers", "", f"Pack: `{summary.get('proposal_pack')}`", ""]
    for name, report in (summary.get("reports") or {}).items():
        meta = report.get("metadata") or {}
        hist = report.get("history") or []
        lines.append(f"- {name}: rows `{meta.get('n_rows')}`, best val loss `{report.get('best_val_loss')}`, latest `{hist[-1] if hist else {}}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train source-specific proposal rankers and source gate")
    parser.add_argument("--proposal-pack", default="results/shared/proposal_recall/full_20260508_parallel/proposal_recall_pack.jsonl")
    parser.add_argument("--output-dir", default="results/shared/proposal_rankers/current")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["chemical", "enzymatic", "rhea_retrorules", "source_gate"],
        choices=["chemical", "enzymatic", "rhea_retrorules", "source_gate"],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-bits", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-rows-per-group", type=int, default=None)
    parser.add_argument("--feature-workers", type=int, default=1)
    args = parser.parse_args()
    summary = train_all_from_proposal_pack(
        proposal_pack=Path(args.proposal_pack),
        output_dir=Path(args.output_dir),
        groups=list(args.groups),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_bits=args.n_bits,
        max_candidates=args.max_candidates,
        d_model=args.d_model,
        device=args.device,
        max_rows_per_group=args.max_rows_per_group,
        feature_workers=args.feature_workers,
    )
    print(json.dumps({name: report.get("best_val_loss") for name, report in summary["reports"].items()}, indent=2))


if __name__ == "__main__":
    main()

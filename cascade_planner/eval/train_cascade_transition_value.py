"""Train CascadeTransitionValueModel from process-transition packs."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascade_search.transition_value import (
    CascadeTransitionValueNetwork,
    transition_feature_dim,
    transition_feature_vector,
)
from cascade_planner.vnext.features import read_jsonl


AUX_LABELS = [
    "stock_closed",
    "condition_compatible",
    "cofactor_closed",
    "evidence_sufficient",
]


@dataclass
class TransitionDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y_value: np.ndarray
    y_aux: np.ndarray
    groups: list[str]
    feature_schema: dict[str, Any]


class CascadeTransitionTrainingNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 192, n_aux: int = len(AUX_LABELS)):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, max(32, hidden // 2)),
            nn.GELU(),
        )
        width = max(32, hidden // 2)
        self.value_head = nn.Linear(width, 1)
        self.aux_head = nn.Linear(width, n_aux)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(x)
        return {
            "value_logit": self.value_head(hidden).squeeze(-1),
            "aux_logits": self.aux_head(hidden),
        }


def load_transition_rows(pack_dir: Path) -> list[dict[str, Any]]:
    path = pack_dir / "transition_value.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing transition_value.jsonl in {pack_dir}")
    return read_jsonl(path)


def build_dataset(rows: list[dict[str, Any]], *, n_bits: int = 128, label_name: str = "transition_value") -> TransitionDataset:
    x_rows = []
    y_value = []
    y_aux = []
    kept = []
    groups = []
    for row in rows:
        parent = row.get("parent_state") or {}
        action = row.get("candidate_action") or {}
        child = row.get("child_summary") or {}
        labels = row.get("labels") or {}
        if not parent or not action or not child:
            continue
        x_rows.append(
            transition_feature_vector(
                parent,
                action,
                child,
                expanded_leaf=row.get("expanded_leaf"),
                n_bits=n_bits,
            )
        )
        y_value.append(float(labels.get(label_name) or 0.0))
        y_aux.append([float(labels.get(name) or 0.0) for name in AUX_LABELS])
        groups.append(str(row.get("pool_id") or row.get("state_id") or ""))
        kept.append(row)
    schema = {
        "schema_version": "cascade_transition_value.v1",
        "n_bits": n_bits,
        "input_dim": transition_feature_dim(n_bits),
        "feature_dim": transition_feature_dim(n_bits),
        "aux_labels": AUX_LABELS,
        "label_contract": "process_transition_delta.v1",
        "label_name": label_name,
        "model_kind": "cascade_transition_value",
    }
    return TransitionDataset(
        rows=kept,
        x=np.asarray(x_rows, dtype=np.float32),
        y_value=np.asarray(y_value, dtype=np.float32),
        y_aux=np.asarray(y_aux, dtype=np.float32),
        groups=groups,
        feature_schema=schema,
    )


def split_by_target(rows: list[dict[str, Any]], val_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    targets = sorted({str(row.get("target_smiles") or "") for row in rows if row.get("target_smiles")})
    if not targets:
        pivot = max(1, int(len(rows) * (1.0 - val_fraction)))
        return list(range(pivot)), list(range(pivot, len(rows)))
    n_val = max(1, int(round(len(targets) * val_fraction)))
    val_targets = set(targets[:: max(1, len(targets) // n_val)][:n_val])
    train_idx = []
    val_idx = []
    for idx, row in enumerate(rows):
        (val_idx if str(row.get("target_smiles") or "") in val_targets else train_idx).append(idx)
    if not train_idx or not val_idx:
        pivot = max(1, int(len(rows) * (1.0 - val_fraction)))
        train_idx = list(range(pivot))
        val_idx = list(range(pivot, len(rows)))
    return train_idx, val_idx


def train_cascade_transition_value(
    *,
    pack_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 8,
    batch_size: int = 512,
    lr: float = 1e-3,
    n_bits: int = 128,
    hidden: int = 192,
    seed: int = 42,
    device: str | None = None,
    label_name: str = "transition_value",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_transition_rows(pack_dir)
    dataset = build_dataset(rows, n_bits=n_bits, label_name=label_name)
    if len(dataset.rows) < 2:
        raise ValueError(f"not enough transition rows to train: {len(dataset.rows)}")
    train_idx, val_idx = split_by_target(dataset.rows)
    x_train = torch.tensor(dataset.x[train_idx], dtype=torch.float32)
    yv_train = torch.tensor(dataset.y_value[train_idx], dtype=torch.float32)
    ya_train = torch.tensor(dataset.y_aux[train_idx], dtype=torch.float32)
    x_val = torch.tensor(dataset.x[val_idx], dtype=torch.float32)
    yv_val = torch.tensor(dataset.y_value[val_idx], dtype=torch.float32)
    ya_val = torch.tensor(dataset.y_aux[val_idx], dtype=torch.float32)
    groups_val = [dataset.groups[idx] for idx in val_idx]

    model = CascadeTransitionTrainingNetwork(dataset.x.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_train, yv_train, ya_train), batch_size=batch_size, shuffle=True)
    best_state = None
    best_loss = float("inf")
    history = []
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yvb, yab in loader:
            xb = xb.to(device)
            yvb = yvb.to(device)
            yab = yab.to(device)
            out = model(xb)
            value_loss = nn.functional.binary_cross_entropy_with_logits(out["value_logit"], yvb)
            aux_loss = nn.functional.binary_cross_entropy_with_logits(out["aux_logits"], yab)
            loss = value_loss + 0.35 * aux_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n_seen += len(xb)
        val_loss, metrics = evaluate(model, x_val.to(device), yv_val.to(device), ya_val.to(device), groups_val)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(total / max(n_seen, 1), 6),
                "val_loss": round(val_loss, 6),
                **metrics,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    final_loss, final_metrics = evaluate(model, x_val.to(device), yv_val.to(device), ya_val.to(device), groups_val)
    runtime_model = CascadeTransitionValueNetwork(dataset.x.shape[1], hidden=hidden)
    # Copy the trained backbone layers into the runtime value-only network.
    runtime_model.net[0].load_state_dict(model.backbone[0].state_dict())
    runtime_model.net[3].load_state_dict(model.backbone[3].state_dict())
    runtime_model.net[5].load_state_dict(model.value_head.state_dict())
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
            "feature_schema": dataset.feature_schema,
            "aux_labels": AUX_LABELS,
            "label_name": label_name,
        },
        "best_val_loss": round(float(best_loss), 6),
        "final_val_loss": round(float(final_loss), 6),
        "final_metrics": final_metrics,
        "history": history,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": runtime_model.state_dict(),
            "model_class": "CascadeTransitionValueNetwork",
            "feature_schema": dataset.feature_schema,
            "hidden": hidden,
            "pack_dir": str(pack_dir),
        },
        model_output,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_output:
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(report_markdown(report), encoding="utf-8")
    return report


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y_value: torch.Tensor,
    y_aux: torch.Tensor,
    groups: list[str],
) -> tuple[float, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        out = model(x)
        value_loss = nn.functional.binary_cross_entropy_with_logits(out["value_logit"], y_value)
        aux_loss = nn.functional.binary_cross_entropy_with_logits(out["aux_logits"], y_aux)
        loss = float((value_loss + 0.35 * aux_loss).item())
        scores = torch.sigmoid(out["value_logit"]).detach().cpu().numpy()
    labels = y_value.detach().cpu().numpy()
    group_map: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        group_map[str(group)].append(idx)
    top1_total = 0.0
    regret_total = 0.0
    n_groups = 0
    for indices in group_map.values():
        if not indices:
            continue
        ordered = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)
        best_label = max(float(labels[idx]) for idx in indices)
        chosen = float(labels[ordered[0]])
        top1_total += float(chosen >= best_label - 1e-6)
        regret_total += max(0.0, best_label - chosen)
        n_groups += 1
    mae = float(np.mean(np.abs(scores - labels))) if len(labels) else 0.0
    return loss, {
        "top1_best_transition_rate": round(top1_total / max(n_groups, 1), 6),
        "mean_top1_regret": round(regret_total / max(n_groups, 1), 6),
        "value_mae": round(mae, 6),
        "n_val_pools": n_groups,
    }


def report_markdown(report: dict[str, Any]) -> str:
    meta = report.get("metadata") or {}
    metrics = report.get("final_metrics") or {}
    lines = [
        "# Cascade Transition Value Model",
        "",
        f"- model: `{meta.get('model_output')}`",
        f"- rows: `{meta.get('n_rows')}`",
        f"- train/val: `{meta.get('n_train')}` / `{meta.get('n_val')}`",
        f"- best val loss: `{report.get('best_val_loss')}`",
        f"- final val loss: `{report.get('final_val_loss')}`",
        f"- top1 best transition rate: `{metrics.get('top1_best_transition_rate')}`",
        f"- mean top1 regret: `{metrics.get('mean_top1_regret')}`",
        f"- value MAE: `{metrics.get('value_mae')}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CascadeTransitionValueModel from transition pack")
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--model-output", required=True)
    ap.add_argument("--report-output", required=True)
    ap.add_argument("--md-output", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-bits", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--label-name", default="transition_value")
    args = ap.parse_args()
    report = train_cascade_transition_value(
        pack_dir=Path(args.pack_dir),
        model_output=Path(args.model_output),
        report_output=Path(args.report_output),
        md_output=Path(args.md_output) if args.md_output else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_bits=args.n_bits,
        hidden=args.hidden,
        seed=args.seed,
        device=args.device,
        label_name=args.label_name,
    )
    print(json.dumps(report["final_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

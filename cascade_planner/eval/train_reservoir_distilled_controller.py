"""Train the reservoir-distilled controller from a distillation pack."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.route_tree.reservoir_distilled import (
    RESERVOIR_CONTROLLER_SCHEMA_VERSION,
    ReservoirDistilledController,
    reservoir_controller_feature_vector,
)
from cascade_planner.route_tree.source_gate import SOURCE_GROUPS, SOURCE_POLICY_BUDGET_LABELS, source_policy_group


DEFAULT_LOSS_WEIGHTS = {
    "source_ce": 0.8,
    "budget_ce": 0.4,
    "action_rank_regression": 1.0,
    "route_value_regression": 0.8,
    "route_rank_regression": 0.8,
    "stock_bce": 0.8,
    "latency_penalty": 0.3,
    "teacher_source_kl": 0.2,
    "leaf_value_regression": 0.4,
}


@dataclass
class ReservoirDistillDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray
    budget_y: np.ndarray
    source_dist: np.ndarray
    source_weight: np.ndarray
    head_weight: np.ndarray
    pair_group: np.ndarray
    leaf_value: np.ndarray
    action_value: np.ndarray
    route_value: np.ndarray
    stock_dead_end: np.ndarray
    latency_cost: np.ndarray
    feature_schema: dict[str, Any]


def train_reservoir_distilled_controller(
    *,
    pack_path: Path,
    val_pack_path: Path,
    output_path: Path,
    report_path: Path,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_bits: int = 256,
    hidden_dim: int = 256,
    dropout: float = 0.10,
    seed: int = 42,
    device: str = "auto",
    source_ce_weight: float = DEFAULT_LOSS_WEIGHTS["source_ce"],
    teacher_source_kl_weight: float = DEFAULT_LOSS_WEIGHTS["teacher_source_kl"],
    balance_source_by_state: bool = False,
    stock_closed_head_weight: float = 1.0,
    pairwise_group_key: str = "state_id",
    include_teacher_label_features: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = build_reservoir_distill_dataset(
        pack_path,
        n_bits=n_bits,
        balance_source_by_state=balance_source_by_state,
        stock_closed_head_weight=stock_closed_head_weight,
        pairwise_group_key=pairwise_group_key,
        include_teacher_label_features=include_teacher_label_features,
    )
    val = build_reservoir_distill_dataset(
        val_pack_path,
        n_bits=n_bits,
        input_dim=train.feature_schema["input_dim"],
        balance_source_by_state=balance_source_by_state,
        stock_closed_head_weight=stock_closed_head_weight,
        pairwise_group_key=pairwise_group_key,
        include_teacher_label_features=include_teacher_label_features,
    )
    device_t = _select_device(device)
    model = ReservoirDistilledController(
        train.feature_schema["input_dim"],
        hidden_dim=hidden_dim,
        dropout=dropout,
        n_source_groups=len(SOURCE_GROUPS),
        n_budget_labels=len(SOURCE_POLICY_BUDGET_LABELS),
    ).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(_tensor_dataset(train), batch_size=batch_size, shuffle=True)
    loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
    loss_weights["source_ce"] = float(source_ce_weight)
    loss_weights["teacher_source_kl"] = float(teacher_source_kl_weight)
    history = []
    best_state = None
    best_val = float("inf")
    best_metrics: dict[str, Any] | None = None
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for batch in dl:
            batch = [item.to(device_t) for item in batch]
            out = model(batch[0])
            out["_source_view"] = model(batch[1])
            loss = _loss(out, batch, weights=loss_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(batch[0])
            n_seen += len(batch[0])
        val_metrics = _eval(model, val, device_t, weights=loss_weights)
        row = {
            "epoch": epoch + 1,
            "train_loss": round(total / max(n_seen, 1), 6),
            **{key: round(float(value), 6) for key, value in val_metrics.items()},
        }
        history.append(row)
        if val_metrics["val_loss"] < best_val:
            best_val = float(val_metrics["val_loss"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_metrics = dict(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "model_kind": "reservoir_distilled_controller",
        "schema_version": RESERVOIR_CONTROLLER_SCHEMA_VERSION,
        "pack_path": str(pack_path),
        "val_pack_path": str(val_pack_path),
        "output_path": str(output_path),
        "n_rows": len(train.rows),
        "n_val_rows": len(val.rows),
        "n_bits": n_bits,
        "input_dim": train.feature_schema["input_dim"],
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "source_groups": list(SOURCE_GROUPS),
        "budget_labels": list(SOURCE_POLICY_BUDGET_LABELS),
        "feature_schema": train.feature_schema,
        "balance_source_by_state": bool(balance_source_by_state),
        "stock_closed_head_weight": float(stock_closed_head_weight),
        "pairwise_group_key": str(pairwise_group_key),
        "include_teacher_label_features": bool(include_teacher_label_features),
        "loss_weights": loss_weights,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "feature_schema": train.feature_schema,
        },
        output_path,
    )
    report = {
        "metadata": metadata,
        "best_val_loss": round(best_val, 6),
        "best_metrics": best_metrics or {},
        "history": history,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_reservoir_distill_dataset(
    path: Path,
    *,
    n_bits: int = 256,
    input_dim: int | None = None,
    balance_source_by_state: bool = False,
    stock_closed_head_weight: float = 1.0,
    pairwise_group_key: str = "state_id",
    include_teacher_label_features: bool = False,
) -> ReservoirDistillDataset:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"no reservoir distillation rows in {path}")
    first = _feature_for_row(
        rows[0],
        n_bits=n_bits,
        input_dim=None,
        include_teacher_label_features=include_teacher_label_features,
    )
    feature_dim = int(input_dim or len(first))
    x = np.zeros((len(rows), feature_dim), dtype=np.float32)
    source_x = np.zeros((len(rows), feature_dim), dtype=np.float32)
    source_y = np.zeros(len(rows), dtype=np.int64)
    budget_y = np.zeros(len(rows), dtype=np.int64)
    source_dist = np.zeros((len(rows), len(SOURCE_GROUPS)), dtype=np.float32)
    source_weight = np.ones(len(rows), dtype=np.float32)
    head_weight = np.ones(len(rows), dtype=np.float32)
    pair_group = np.zeros(len(rows), dtype=np.int64)
    leaf_value = np.zeros(len(rows), dtype=np.float32)
    action_value = np.zeros(len(rows), dtype=np.float32)
    route_value = np.zeros(len(rows), dtype=np.float32)
    stock_dead_end = np.zeros(len(rows), dtype=np.float32)
    latency_cost = np.zeros(len(rows), dtype=np.float32)
    state_counts: dict[str, int] = {}
    if balance_source_by_state:
        for row in rows:
            key = str(row.get("state_id") or row.get("target_id") or row.get("benchmark_index") or "")
            state_counts[key] = state_counts.get(key, 0) + 1
        n_states = max(len(state_counts), 1)
        mean_rows_per_state = len(rows) / float(n_states)
    pair_group_ids: dict[str, int] = {}
    for idx, row in enumerate(rows):
        group_key = _pairwise_group(row, pairwise_group_key)
        if group_key not in pair_group_ids:
            pair_group_ids[group_key] = len(pair_group_ids)
        pair_group[idx] = pair_group_ids[group_key]
        x[idx] = _feature_for_row(
            row,
            n_bits=n_bits,
            input_dim=feature_dim,
            include_teacher_label_features=include_teacher_label_features,
        )
        source_x[idx] = _source_feature_for_row(row, n_bits=n_bits, input_dim=feature_dim)
        budget_y[idx] = _label_index(SOURCE_POLICY_BUDGET_LABELS, str(row.get("budget_label") or "1x"), default="1x")
        source_dist[idx] = _source_dist(row)
        source_y[idx] = int(np.argmax(source_dist[idx]))
        route_label = float(row.get("teacher_route_value") or 0.0)
        action_label = float(row.get("teacher_action_value") or 0.0)
        leaf_value[idx] = float(np.clip(route_label, 0.0, 1.0))
        action_value[idx] = float(np.clip(action_label, 0.0, 1.0))
        route_value[idx] = float(np.clip(route_label, 0.0, 1.0))
        stock_dead_end[idx] = _stock_dead_end_target(row)
        latency_cost[idx] = float(np.clip(float(row.get("latency_ms") or 0.0) / 1000.0, 0.0, 1.0))
        if bool(row.get("source_only") or row.get("source_loss_only")):
            head_weight[idx] = 0.0
        elif bool(row.get("teacher_stock_closed")):
            head_weight[idx] *= max(0.0, float(stock_closed_head_weight))
        if balance_source_by_state:
            key = str(row.get("state_id") or row.get("target_id") or row.get("benchmark_index") or "")
            source_weight[idx] = float(mean_rows_per_state / max(state_counts.get(key, 1), 1))
    schema = {
        "schema_version": RESERVOIR_CONTROLLER_SCHEMA_VERSION,
        "n_bits": n_bits,
        "input_dim": feature_dim,
        "source_groups": list(SOURCE_GROUPS),
        "budget_labels": list(SOURCE_POLICY_BUDGET_LABELS),
        "model_kind": "reservoir_distilled_controller",
        "pack_path": str(path),
        "balance_source_by_state": bool(balance_source_by_state),
        "stock_closed_head_weight": float(stock_closed_head_weight),
        "pairwise_group_key": str(pairwise_group_key),
        "include_teacher_label_features": bool(include_teacher_label_features),
    }
    return ReservoirDistillDataset(
        rows=rows,
        x=x,
        source_x=source_x,
        source_y=source_y,
        budget_y=budget_y,
        source_dist=source_dist,
        source_weight=source_weight,
        head_weight=head_weight,
        pair_group=pair_group,
        leaf_value=leaf_value,
        action_value=action_value,
        route_value=route_value,
        stock_dead_end=stock_dead_end,
        latency_cost=latency_cost,
        feature_schema=schema,
    )


def _feature_for_row(
    row: dict[str, Any],
    *,
    n_bits: int,
    input_dim: int | None,
    include_teacher_label_features: bool = False,
) -> np.ndarray:
    reactants = [str(smi) for smi in row.get("reactants") or [] if smi]
    candidate = {
        "main_reactant": reactants[0] if reactants else "",
        "aux_reactants": reactants[1:],
        "rxn_smiles": row.get("candidate_reaction") or "",
        "reaction_smiles": row.get("candidate_reaction") or "",
        "source": row.get("source") or "",
        "score": row.get("candidate_score") or 0.0,
        "rank": row.get("candidate_rank") or row.get("reservoir_rank") or 0,
    }
    return reservoir_controller_feature_vector(
        product=str(row.get("target_smiles") or row.get("target_id") or ""),
        leaf=str(row.get("leaf") or row.get("target_smiles") or ""),
        candidate=candidate,
        source=str(row.get("source") or ""),
        source_diagnostics=row.get("source_diagnostics") or {},
        route_context_features=row.get("route_context_features") or {},
        reservoir_fields=row if include_teacher_label_features else _runtime_reservoir_fields(row),
        n_bits=n_bits,
        input_dim=input_dim,
        source_groups=tuple(SOURCE_GROUPS),
    )


def _runtime_reservoir_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Keep training features aligned with fields available during inference."""

    return {
        "reservoir_rank": row.get("reservoir_rank"),
    }


def _stock_dead_end_target(row: dict[str, Any]) -> float:
    if bool(row.get("teacher_stock_closed")):
        return 0.0
    labels = {str(label) for label in row.get("failure_labels") or []}
    if "stock_dead_end" in labels:
        return 1.0
    if _has_teacher_route_match(row):
        return 1.0
    return 0.0


def _has_teacher_route_match(row: dict[str, Any]) -> bool:
    value = row.get("teacher_route_rank")
    if value in {None, ""}:
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return True


def _pairwise_group(row: dict[str, Any], key: str) -> str:
    key = str(key or "").strip()
    if not key:
        return "__global__"
    if key == "state_id":
        return str(row.get("state_id") or row.get("target_id") or row.get("benchmark_index") or "__missing__")
    return str(row.get(key) or "__missing__")


def _source_feature_for_row(row: dict[str, Any], *, n_bits: int, input_dim: int | None) -> np.ndarray:
    """State-level view for source/budget distillation.

    Candidate-level features include a source one-hot. That is useful for the
    action/value heads, but it leaks the answer for source allocation and is
    absent at runtime. The source head is therefore trained on the same blank
    source view used by ``ReservoirDistilledControllerRuntime.allocate``.
    """

    return reservoir_controller_feature_vector(
        product=str(row.get("target_smiles") or row.get("target_id") or ""),
        leaf=str(row.get("leaf") or row.get("target_smiles") or ""),
        candidate={},
        source="",
        source_diagnostics=row.get("source_diagnostics") or {},
        route_context_features=row.get("route_context_features") or {},
        reservoir_fields={},
        n_bits=n_bits,
        input_dim=input_dim,
        source_groups=tuple(SOURCE_GROUPS),
    )


def _tensor_dataset(dataset: ReservoirDistillDataset) -> TensorDataset:
    return TensorDataset(
        torch.tensor(dataset.x, dtype=torch.float32),
        torch.tensor(dataset.source_x, dtype=torch.float32),
        torch.tensor(dataset.source_y, dtype=torch.long),
        torch.tensor(dataset.budget_y, dtype=torch.long),
        torch.tensor(dataset.source_dist, dtype=torch.float32),
        torch.tensor(dataset.source_weight, dtype=torch.float32),
        torch.tensor(dataset.head_weight, dtype=torch.float32),
        torch.tensor(dataset.pair_group, dtype=torch.long),
        torch.tensor(dataset.leaf_value, dtype=torch.float32),
        torch.tensor(dataset.action_value, dtype=torch.float32),
        torch.tensor(dataset.route_value, dtype=torch.float32),
        torch.tensor(dataset.stock_dead_end, dtype=torch.float32),
        torch.tensor(dataset.latency_cost, dtype=torch.float32),
    )


def _loss(
    out: dict[str, torch.Tensor],
    batch: list[torch.Tensor],
    *,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    weights = weights or DEFAULT_LOSS_WEIGHTS
    _, source_x, source_y, budget_y, source_dist, source_weight, head_weight, pair_group, leaf_y, action_y, route_y, stock_y, latency_y = batch
    source_out = out.get("_source_view")
    if source_out is None:
        source_out = out
    source_weight = source_weight / torch.clamp(source_weight.mean(), min=1e-6)
    source_ce_rows = nn.functional.cross_entropy(
        source_out["source_group_logits"],
        source_y,
        reduction="none",
    )
    source_ce = (source_ce_rows * source_weight).mean()
    budget_ce_rows = nn.functional.cross_entropy(
        source_out["budget_logits"],
        budget_y,
        reduction="none",
    )
    budget_ce = (budget_ce_rows * source_weight).mean()
    action_reg = _weighted_mean(
        nn.functional.smooth_l1_loss(out["action_value"], action_y, reduction="none"),
        head_weight,
    )
    action_rank = _pairwise_ranking_loss(out["action_value"], action_y, weights=head_weight, group_ids=pair_group)
    route_reg = _weighted_mean(
        nn.functional.smooth_l1_loss(out["route_rerank_value"], route_y, reduction="none"),
        head_weight,
    )
    route_rank = _pairwise_ranking_loss(out["route_rerank_value"], route_y, weights=head_weight, group_ids=pair_group)
    stock_bce = _weighted_mean(
        nn.functional.binary_cross_entropy_with_logits(out["stock_dead_end_logit"], stock_y, reduction="none"),
        head_weight,
    )
    latency_reg = _weighted_mean(
        nn.functional.smooth_l1_loss(torch.sigmoid(out["latency_cost"]), latency_y, reduction="none"),
        head_weight,
    )
    source_kl_rows = nn.functional.kl_div(
        nn.functional.log_softmax(source_out["source_group_logits"], dim=-1),
        source_dist,
        reduction="none",
    ).sum(dim=-1)
    source_kl = (source_kl_rows * source_weight).mean()
    leaf_reg = _weighted_mean(
        nn.functional.smooth_l1_loss(out["leaf_value"], leaf_y, reduction="none"),
        head_weight,
    )
    return (
        float(weights.get("source_ce", 0.8)) * source_ce
        + float(weights.get("budget_ce", 0.4)) * budget_ce
        + float(weights.get("action_rank_regression", 1.0)) * (action_reg + action_rank)
        + float(weights.get("route_value_regression", 0.8)) * route_reg
        + float(weights.get("route_rank_regression", 0.8)) * route_rank
        + float(weights.get("stock_bce", 0.8)) * stock_bce
        + float(weights.get("latency_penalty", 0.3)) * latency_reg
        + float(weights.get("teacher_source_kl", 0.2)) * source_kl
        + float(weights.get("leaf_value_regression", 0.4)) * leaf_reg
    )


def _pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights is not None:
        keep = weights > 0
        scores = scores[keep]
        labels = labels[keep]
        if group_ids is not None:
            group_ids = group_ids[keep]
    if scores.numel() < 2:
        return scores.sum() * 0.0
    diff = labels[:, None] - labels[None, :]
    mask = diff > 1e-5
    if group_ids is not None:
        mask = mask & (group_ids[:, None] == group_ids[None, :])
    if not torch.any(mask):
        return scores.sum() * 0.0
    score_diff = scores[:, None] - scores[None, :]
    return nn.functional.softplus(-score_diff[mask]).mean()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    total = weights.sum()
    if float(total.detach().cpu()) <= 1e-6:
        return values.sum() * 0.0
    return (values * weights).sum() / total


def _eval(
    model: ReservoirDistilledController,
    dataset: ReservoirDistillDataset,
    device: torch.device,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        batch = [item.to(device) for item in _tensor_dataset(dataset).tensors]
        out = model(batch[0])
        out["_source_view"] = model(batch[1])
        loss = _loss(out, batch, weights=weights)
        source_out = out["_source_view"]
        source_acc = (torch.argmax(source_out["source_group_logits"], dim=-1) == batch[2]).float().mean()
        budget_acc = (torch.argmax(source_out["budget_logits"], dim=-1) == batch[3]).float().mean()
        head_weight = batch[6]
        action_mae = _weighted_mean(torch.abs(out["action_value"] - batch[9]), head_weight)
        route_mae = _weighted_mean(torch.abs(out["route_rerank_value"] - batch[10]), head_weight)
        route_rank = _pairwise_ranking_loss(out["route_rerank_value"], batch[10], weights=head_weight, group_ids=batch[7])
        stock_pred = torch.sigmoid(out["stock_dead_end_logit"])
        stock_bce = _weighted_mean(
            nn.functional.binary_cross_entropy(stock_pred, batch[11], reduction="none"),
            head_weight,
        )
    return {
        "val_loss": float(loss.item()),
        "val_source_acc": float(source_acc.item()),
        "val_budget_acc": float(budget_acc.item()),
        "val_action_mae": float(action_mae.item()),
        "val_route_mae": float(route_mae.item()),
        "val_route_rank_loss": float(route_rank.item()),
        "val_stock_bce": float(stock_bce.item()),
    }


def _source_dist(row: dict[str, Any]) -> np.ndarray:
    raw = row.get("teacher_source_group_distribution") or {}
    values = np.zeros(len(SOURCE_GROUPS), dtype=np.float32)
    for idx, group in enumerate(SOURCE_GROUPS):
        try:
            values[idx] = max(0.0, float(raw.get(group) or 0.0))
        except (TypeError, ValueError):
            values[idx] = 0.0
    total = float(values.sum())
    if total <= 0:
        group = str(row.get("source_policy_group") or source_policy_group(str(row.get("source") or "")))
        values[_label_index(SOURCE_GROUPS, group, default="fallback")] = 1.0
        total = 1.0
    return values / total


def _label_index(labels: tuple[str, ...], value: str, *, default: str) -> int:
    try:
        return list(labels).index(str(value))
    except ValueError:
        return list(labels).index(default)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict) and not row.get("eval_only"):
            rows.append(row)
    return rows


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train reservoir-distilled controller")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--val-pack", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-bits", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--source-ce-weight", type=float, default=DEFAULT_LOSS_WEIGHTS["source_ce"])
    ap.add_argument("--teacher-source-kl-weight", type=float, default=DEFAULT_LOSS_WEIGHTS["teacher_source_kl"])
    ap.add_argument("--balance-source-by-state", action="store_true")
    ap.add_argument("--stock-closed-head-weight", type=float, default=1.0)
    ap.add_argument(
        "--pairwise-group-key",
        default="state_id",
        help="Group key for action pairwise ranking; empty string restores global batch ranking.",
    )
    ap.add_argument(
        "--include-teacher-label-features",
        action="store_true",
        help="Diagnostic compatibility mode; default training hides teacher labels that are unavailable at inference.",
    )
    args = ap.parse_args()
    train_reservoir_distilled_controller(
        pack_path=Path(args.pack),
        val_pack_path=Path(args.val_pack),
        output_path=Path(args.output),
        report_path=Path(args.report),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_bits=args.n_bits,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        source_ce_weight=args.source_ce_weight,
        teacher_source_kl_weight=args.teacher_source_kl_weight,
        balance_source_by_state=args.balance_source_by_state,
        stock_closed_head_weight=args.stock_closed_head_weight,
        pairwise_group_key=args.pairwise_group_key,
        include_teacher_label_features=args.include_teacher_label_features,
    )


if __name__ == "__main__":
    main()

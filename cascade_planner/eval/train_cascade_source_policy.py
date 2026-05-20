"""Train the route-tree CascadeSourcePolicy gate from source-policy packs."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.eval.build_route_tree_source_policy_pack import FAILURE_LABELS
from cascade_planner.route_tree.source_gate import (
    SOURCE_GROUPS,
    SOURCE_POLICY_BUDGET_LABELS,
    SOURCE_POLICY_DECISIONS,
    _CascadeSourcePolicyMLP,
    source_policy_feature_vector,
    source_policy_group,
)
from cascade_planner.vnext.features import read_jsonl


@dataclass
class CascadeSourcePolicyDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    group_y: np.ndarray
    budget_y: np.ndarray
    decision_y: np.ndarray
    failure_y: np.ndarray
    utility_y: np.ndarray
    state_ids: list[str]
    schema: dict[str, Any]


def train_cascade_source_policy(
    *,
    pack: Path,
    output: Path,
    report: Path,
    md_output: Path | None = None,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    n_bits: int = 64,
    hidden: int = 128,
    seed: int = 42,
    device: str = "auto",
    max_rows: int | None = None,
) -> dict[str, Any]:
    del hidden  # Runtime architecture is fixed by _CascadeSourcePolicyMLP.
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = _select_device(device)
    dataset = build_cascade_source_policy_dataset(pack, n_bits=n_bits, max_rows=max_rows)
    if len(dataset.rows) < 2:
        raise ValueError(f"not enough source-policy rows to train: {len(dataset.rows)}")
    train_idx, val_idx = _split_by_target(dataset.rows)
    model = _CascadeSourcePolicyMLP(
        dataset.x.shape[-1],
        n_groups=len(dataset.schema["source_groups"]),
        n_budgets=len(dataset.schema["budget_labels"]),
        n_decisions=len(dataset.schema["decision_labels"]),
        n_failures=len(dataset.schema["failure_labels"]),
    ).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(
        TensorDataset(
            torch.tensor(dataset.x[train_idx]),
            torch.tensor(dataset.group_y[train_idx]),
            torch.tensor(dataset.budget_y[train_idx]),
            torch.tensor(dataset.decision_y[train_idx]),
            torch.tensor(dataset.failure_y[train_idx]),
            torch.tensor(dataset.utility_y[train_idx]),
            torch.tensor(np.asarray(train_idx, dtype=np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for bx, group_y, budget_y, decision_y, failure_y, utility_y, row_idx in dl:
            bx = bx.to(device_t)
            group_y = group_y.to(device_t)
            budget_y = budget_y.to(device_t)
            decision_y = decision_y.to(device_t)
            failure_y = failure_y.to(device_t)
            utility_y = utility_y.to(device_t)
            out = model(bx)
            utility_weight = 1.0 + 2.0 * utility_y
            group_loss = nn.functional.cross_entropy(out["group_logits"], group_y, reduction="none")
            budget_loss = nn.functional.cross_entropy(out["budget_logits"], budget_y, reduction="none")
            decision_loss = nn.functional.cross_entropy(out["decision_logits"], decision_y, reduction="none")
            failure_loss = nn.functional.binary_cross_entropy_with_logits(out["failure_logits"], failure_y, reduction="none").mean(dim=1)
            utility_loss = nn.functional.binary_cross_entropy_with_logits(out["utility_logit"], utility_y, reduction="none")
            rank_loss = _batch_pairwise_ranking_loss(
                out["utility_logit"],
                utility_y,
                [dataset.state_ids[int(i)] for i in row_idx.tolist()],
            )
            loss = (
                (group_loss * utility_weight).mean()
                + 0.45 * budget_loss.mean()
                + 0.20 * decision_loss.mean()
                + 0.55 * failure_loss.mean()
                + utility_loss.mean()
                + 0.25 * rank_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(bx)
            n_seen += len(bx)
        metrics = _eval_source_policy(model, dataset, val_idx, device=device_t)
        history.append({"epoch": epoch + 1, "train_loss": round(total / max(n_seen, 1), 6), **_round(metrics)})
        if metrics["val_loss"] < best_val:
            best_val = float(metrics["val_loss"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "model_kind": "cascade_source_policy",
        "pack": str(pack),
        "model_output": str(output),
        "n_rows": len(dataset.rows),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_bits": n_bits,
        "input_dim": int(dataset.x.shape[-1]),
        "source_budget_groups": list(dataset.schema["source_groups"]),
        "budget_labels": list(dataset.schema["budget_labels"]),
        "decision_labels": list(dataset.schema["decision_labels"]),
        "failure_labels": list(dataset.schema["failure_labels"]),
        "min_confidence": 0.30,
        "device": str(device_t),
        "feature_schema": dataset.schema,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "feature_schema": dataset.schema,
            "model_config": {"hidden": 128},
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        output,
    )
    final_report = {
        "metadata": metadata,
        "data_summary": {
            "trace_count": _pack_trace_count(pack),
            "supervision_row_count": len(dataset.rows),
            "target_count": _target_count(dataset.rows),
            "train_target_count": _target_count(dataset.rows, train_idx),
            "val_target_count": _target_count(dataset.rows, val_idx),
        },
        "best_val_loss": round(best_val, 6),
        "history": history,
        "final_metrics": history[-1] if history else {},
        "label_contract": "source_group/budget/failure/utility rows from route_tree_source_policy_pack.v1",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_output is not None:
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(_report_markdown(final_report), encoding="utf-8")
    return final_report


def build_cascade_source_policy_dataset(
    pack: Path,
    *,
    n_bits: int = 64,
    max_rows: int | None = None,
) -> CascadeSourcePolicyDataset:
    raw_rows = read_jsonl(Path(pack))
    rows = [row for row in raw_rows if not bool(row.get("eval_only"))]
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        raise ValueError(f"no trainable source-policy rows in {pack}")
    groups = list(SOURCE_GROUPS)
    budget_labels = list(SOURCE_POLICY_BUDGET_LABELS)
    decision_labels = list(SOURCE_POLICY_DECISIONS)
    failure_labels = list(FAILURE_LABELS)
    features: list[np.ndarray] = []
    group_y: list[int] = []
    budget_y: list[int] = []
    decision_y: list[int] = []
    failure_y: list[list[float]] = []
    utility_y: list[float] = []
    state_ids: list[str] = []
    source_history: dict[str, dict[str, float]] = defaultdict(_history_template)
    for row in rows:
        source = str(row.get("source") or row.get("source_name") or "")
        ctx = _context_from_row(row)
        features.append(
            source_policy_feature_vector(
                str(row.get("leaf") or row.get("target_smiles") or ""),
                context=ctx,
                source=source,
                source_stats=source_history.get(source),
                n_bits=n_bits,
                total_budget=int(row.get("proposal_budget") or row.get("allocated_budget") or 1),
            )
        )
        group_y.append(_index(groups, str(row.get("source_group") or source_policy_group(source))))
        budget_y.append(_index(budget_labels, str(row.get("budget_multiplier_label") or "1x")))
        decision_y.append(_index(decision_labels, str(row.get("decision") or "query")))
        failure_set = set(row.get("failure_labels") or [])
        failure_y.append([1.0 if label in failure_set else 0.0 for label in failure_labels])
        utility_y.append(float(row.get("source_utility") or 0.0))
        state_ids.append(str(row.get("state_id") or row.get("target_id") or ""))
        _update_history(source_history[source], row)
    schema = {
        "schema_version": "cascade_source_policy_features.v1",
        "model_kind": "cascade_source_policy",
        "n_bits": n_bits,
        "input_dim": int(features[0].shape[-1]) if features else 0,
        "source_groups": groups,
        "source_budget_groups": groups,
        "budget_labels": budget_labels,
        "decision_labels": decision_labels,
        "failure_labels": failure_labels,
        "feature_contract": "route_state_leaf_source_history_without_current_outcome.v1",
    }
    return CascadeSourcePolicyDataset(
        rows=rows,
        x=np.asarray(features, dtype=np.float32),
        group_y=np.asarray(group_y, dtype=np.int64),
        budget_y=np.asarray(budget_y, dtype=np.int64),
        decision_y=np.asarray(decision_y, dtype=np.int64),
        failure_y=np.asarray(failure_y, dtype=np.float32),
        utility_y=np.asarray(utility_y, dtype=np.float32),
        state_ids=state_ids,
        schema=schema,
    )


def _context_from_row(row: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        depth=int(row.get("depth") or 0),
        ec1=_ec1(row.get("ec1")),
        reaction_type=str(row.get("reaction_type") or ""),
        T=_safe_float(row.get("T")),
        pH=_safe_float(row.get("pH")),
        route_metadata=dict(row.get("route_metadata") or {}),
    )


def _history_template() -> dict[str, float]:
    return {
        "calls": 0.0,
        "queried": 0.0,
        "requested_k_total": 0.0,
        "raw_returned": 0.0,
        "final_returned": 0.0,
        "latency_ms_total": 0.0,
        "allocated_budget": 0.0,
        "useful_hits": 0.0,
    }


def _update_history(history: dict[str, float], row: dict[str, Any]) -> None:
    history["calls"] += 1.0
    history["queried"] += float(bool(row.get("source_called")))
    history["requested_k_total"] += float(row.get("requested_k") or 0)
    history["raw_returned"] += float(row.get("raw_returned") or 0)
    history["final_returned"] += float(row.get("final_returned") or 0)
    history["latency_ms_total"] += float(row.get("latency_ms") or 0.0)
    history["allocated_budget"] += float(row.get("allocated_budget") or 0)
    history["useful_hits"] += float(bool(row.get("useful_candidate_hit")))


def _batch_pairwise_ranking_loss(logits: torch.Tensor, utility: torch.Tensor, state_ids: list[str]) -> torch.Tensor:
    losses = []
    for state_id in sorted(set(state_ids)):
        indices = [idx for idx, value in enumerate(state_ids) if value == state_id]
        if len(indices) < 2:
            continue
        scores = logits[indices]
        labels = utility[indices]
        pos = torch.nonzero(labels >= 0.5, as_tuple=False).reshape(-1)
        neg = torch.nonzero(labels < 0.5, as_tuple=False).reshape(-1)
        if len(pos) == 0 or len(neg) == 0:
            continue
        diffs = scores[pos][:, None] - scores[neg][None, :]
        losses.append(nn.functional.softplus(-diffs).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _eval_source_policy(
    model: nn.Module,
    dataset: CascadeSourcePolicyDataset,
    idx: list[int],
    *,
    device: torch.device,
) -> dict[str, Any]:
    if not idx:
        return {"val_loss": 0.0, "val_group_acc": 0.0, "val_top1_useful_source_hit": 0.0}
    model.eval()
    with torch.no_grad():
        x = torch.tensor(dataset.x[idx], device=device)
        group_y = torch.tensor(dataset.group_y[idx], device=device)
        budget_y = torch.tensor(dataset.budget_y[idx], device=device)
        decision_y = torch.tensor(dataset.decision_y[idx], device=device)
        failure_y = torch.tensor(dataset.failure_y[idx], device=device)
        utility_y = torch.tensor(dataset.utility_y[idx], device=device)
        out = model(x)
        loss = (
            nn.functional.cross_entropy(out["group_logits"], group_y)
            + 0.45 * nn.functional.cross_entropy(out["budget_logits"], budget_y)
            + 0.20 * nn.functional.cross_entropy(out["decision_logits"], decision_y)
            + 0.55 * nn.functional.binary_cross_entropy_with_logits(out["failure_logits"], failure_y)
            + nn.functional.binary_cross_entropy_with_logits(out["utility_logit"], utility_y)
        )
        group_pred = torch.argmax(out["group_logits"], dim=1)
        budget_pred = torch.argmax(out["budget_logits"], dim=1)
        decision_pred = torch.argmax(out["decision_logits"], dim=1)
        utility_scores = torch.sigmoid(out["utility_logit"]).detach().cpu().numpy()
        failure_scores = torch.sigmoid(out["failure_logits"]).detach().cpu().numpy()
    rows = [dataset.rows[i] for i in idx]
    utility_labels = dataset.utility_y[idx]
    return {
        "val_loss": float(loss.item()),
        "val_group_acc": float((group_pred == group_y).float().mean().item()),
        "val_budget_acc": float((budget_pred == budget_y).float().mean().item()),
        "val_decision_acc": float((decision_pred == decision_y).float().mean().item()),
        "val_utility_auc": _binary_auc(utility_labels, utility_scores),
        "val_top1_useful_source_hit": _top1_useful_source_hit(rows, utility_scores),
        "val_failure_auc": _failure_auc(dataset.failure_y[idx], failure_scores, dataset.schema["failure_labels"]),
    }


def _top1_useful_source_hit(rows: list[dict[str, Any]], scores: np.ndarray) -> float:
    by_state: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_state[str(row.get("state_id") or row.get("target_id") or "")].append((float(score), row))
    hits = []
    for items in by_state.values():
        if not any(float((row or {}).get("source_utility") or 0.0) >= 0.5 for _score, row in items):
            continue
        _score, top_row = max(items, key=lambda item: item[0])
        hits.append(float(float(top_row.get("source_utility") or 0.0) >= 0.5))
    return float(np.mean(hits)) if hits else 0.0


def _failure_auc(labels: np.ndarray, scores: np.ndarray, failure_labels: list[str]) -> dict[str, float | None]:
    return {
        label: _binary_auc(labels[:, idx], scores[:, idx])
        for idx, label in enumerate(failure_labels)
    }


def _split_by_target(rows: list[dict[str, Any]], val_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    targets = sorted({canonical_smiles(row.get("target_smiles") or "") or str(row.get("target_id") or "") for row in rows})
    if len(targets) >= 2:
        n_val = max(1, int(round(len(targets) * val_fraction)))
        stride = max(1, len(targets) // n_val)
        val_targets = set(targets[::stride][:n_val])
        train_idx = [idx for idx, row in enumerate(rows) if (canonical_smiles(row.get("target_smiles") or "") or str(row.get("target_id") or "")) not in val_targets]
        val_idx = [idx for idx in range(len(rows)) if idx not in train_idx]
    else:
        pivot = max(1, int(len(rows) * (1.0 - val_fraction)))
        train_idx = list(range(pivot))
        val_idx = list(range(pivot, len(rows)))
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1] or val_idx
    return train_idx, val_idx


def _target_count(rows: list[dict[str, Any]], indices: list[int] | None = None) -> int:
    selected = rows if indices is None else [rows[idx] for idx in indices]
    targets = {
        canonical_smiles(row.get("target_smiles") or row.get("target_id") or "")
        or str(row.get("target_id") or "")
        for row in selected
    }
    return len({target for target in targets if target})


def _pack_trace_count(pack: Path) -> int | None:
    manifest = pack.parent / "source_policy_manifest.json"
    if not manifest.exists():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    paths = payload.get("trace_paths") or []
    return len(paths) if isinstance(paths, list) else None


def _binary_auc(labels: np.ndarray, probs: np.ndarray) -> float | None:
    y = (np.asarray(labels, dtype=np.float32).reshape(-1) >= 0.5).astype(np.int32)
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / max(pos * neg, 1))


def _round(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            out[key] = None
        elif isinstance(value, dict):
            out[key] = _round(value)
        else:
            out[key] = round(float(value), 6)
    return out


def _index(values: list[str], value: str) -> int:
    try:
        return values.index(value)
    except ValueError:
        return values.index("fallback") if "fallback" in values else 0


def _ec1(value: Any) -> int:
    try:
        ec1 = int(str(value).split(".", 1)[0])
    except (TypeError, ValueError):
        return 0
    return ec1 if 1 <= ec1 <= 7 else 0


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _select_device(device: str) -> torch.device:
    requested = str(device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for source-policy training but torch.cuda.is_available() is false")
    return torch.device(requested)


def _report_markdown(report: dict[str, Any]) -> str:
    meta = report.get("metadata") or {}
    return "\n".join(
        [
            "# Cascade Source Policy",
            "",
            f"Pack: `{meta.get('pack')}`",
            f"Model: `{meta.get('model_output')}`",
            f"Rows: `{meta.get('n_rows')}`",
            f"Train/val: `{meta.get('n_train')}` / `{meta.get('n_val')}`",
            f"Best val loss: `{report.get('best_val_loss')}`",
            "",
            "## Final Metrics",
            "",
            "```json",
            json.dumps(report.get("final_metrics") or {}, indent=2),
            "```",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CascadeSourcePolicy from source_policy_pack.jsonl")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--md-output", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-bits", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    result = train_cascade_source_policy(
        pack=Path(args.pack),
        output=Path(args.output),
        report=Path(args.report),
        md_output=Path(args.md_output) if args.md_output else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_bits=args.n_bits,
        seed=args.seed,
        device=args.device,
        max_rows=args.max_rows,
    )
    print(json.dumps({"best_val_loss": result.get("best_val_loss"), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()

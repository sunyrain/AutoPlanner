"""Train the v4 route-level cascade product-value reranker."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascade_search.v4_product_value import (
    ROUTE_LABEL_NAMES,
    V4CascadeProductValueNetwork,
    build_route_feature_schema,
    route_feature_vector,
    route_label_vector,
)


@dataclass
class ProductValueDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    value_target: np.ndarray
    schema: dict[str, Any]


def train_v4_cascade_product_value(
    *,
    train_pack: Path,
    val_pack: Path,
    preference_pack: Path | None,
    listwise_pack: Path | None = None,
    extra_train_pack: list[Path] | None = None,
    extra_preference_pack: list[Path] | None = None,
    extra_listwise_pack: list[Path] | None = None,
    output: Path,
    report: Path,
    md_output: Path | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 192,
    n_bits: int = 128,
    seed: int = 42,
    device: str | None = None,
    preference_loss_weight: float = 0.4,
    listwise_loss_weight: float = 0.0,
    label_loss_weight: float = 1.0,
    value_loss_weight: float = 0.6,
    selection_metric: str = "value_auc",
    preference_sampling: str = "random",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = _read_jsonl(train_pack)
    for path in extra_train_pack or []:
        train_rows.extend(_read_jsonl(path))
    val_rows = _read_jsonl(val_pack)
    if not val_rows:
        split = max(1, int(len(train_rows) * 0.85))
        val_rows = train_rows[split:]
        train_rows = train_rows[:split]
    if len(train_rows) < 4:
        raise ValueError(f"not enough v4 train rows: {len(train_rows)}")
    dataset = build_dataset(train_rows, val_rows, n_bits=n_bits)
    n_train = len(train_rows)
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, len(dataset.rows)))

    x_train = torch.tensor(dataset.x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(dataset.y[train_idx], dtype=torch.float32)
    value_train = torch.tensor(dataset.value_target[train_idx], dtype=torch.float32)
    x_val = torch.tensor(dataset.x[val_idx], dtype=torch.float32)
    y_val = torch.tensor(dataset.y[val_idx], dtype=torch.float32)
    value_val = torch.tensor(dataset.value_target[val_idx], dtype=torch.float32)

    model = V4CascadeProductValueNetwork(dataset.x.shape[1], hidden=hidden, output_dim=len(ROUTE_LABEL_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_train, y_train, value_train), batch_size=batch_size, shuffle=True)
    preferences = _read_preferences(preference_pack)
    for path in extra_preference_pack or []:
        preferences.extend(_read_preferences(path))
    listwise_groups = _read_listwise_groups(listwise_pack)
    for path in extra_listwise_pack or []:
        listwise_groups.extend(_read_listwise_groups(path))
    train_by_id = {str(row.get("route_id")): idx for idx, row in enumerate(train_rows)}
    pref_pairs: list[tuple[int, int]] = []
    pref_pairs_by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for pref in preferences:
        better_id = str(pref.get("better_route_id") or "")
        worse_id = str(pref.get("worse_route_id") or "")
        if better_id not in train_by_id or worse_id not in train_by_id:
            continue
        pair = (train_by_id[better_id], train_by_id[worse_id])
        source = str(pref.get("preference_source") or "unknown")
        pref_pairs.append(pair)
        pref_pairs_by_source[source].append(pair)
    listwise_train_groups = _listwise_train_groups(listwise_groups, train_by_id)
    listwise_train_groups_by_source: dict[str, list[list[int]]] = defaultdict(list)
    for group in listwise_groups:
        source = str(group.get("preference_source") or group.get("listwise_source") or "unknown")
        indices = _listwise_group_indices(group, train_by_id)
        if len(indices) >= 2:
            listwise_train_groups_by_source[source].append(indices)

    best_state = None
    best_metric = -1.0
    best_epoch = 0
    history = []
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb, vb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            vb = vb.to(device)
            label_logits, value_logits = model(xb)
            label_loss = F.binary_cross_entropy_with_logits(label_logits, yb)
            value_loss = F.binary_cross_entropy_with_logits(value_logits, vb)
            loss = float(label_loss_weight) * label_loss + float(value_loss_weight) * value_loss
            if pref_pairs:
                if preference_sampling == "balanced_by_source":
                    pref_loss = _preference_loss_balanced_by_source(
                        model,
                        x_train,
                        pref_pairs_by_source,
                        device=device,
                        batch_size=batch_size,
                    )
                else:
                    pref_loss = _preference_loss(model, x_train, pref_pairs, device=device, batch_size=batch_size)
                loss = loss + float(preference_loss_weight) * pref_loss
            if listwise_train_groups and listwise_loss_weight:
                listwise_loss = _listwise_loss(
                    model,
                    x_train,
                    listwise_train_groups,
                    device=device,
                    batch_size=batch_size,
                )
                loss = loss + float(listwise_loss_weight) * listwise_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n_seen += len(xb)
        val_metrics = evaluate(model, x_val.to(device), y_val.to(device), value_val.to(device), val_rows)
        pref_acc = evaluate_preference_accuracy(model, x_train, pref_pairs, device=device)
        pref_acc_by_source = evaluate_preference_accuracy_by_source(model, x_train, pref_pairs_by_source, device=device)
        balanced_pref_acc = _mean_metric(pref_acc_by_source.values())
        listwise_acc = evaluate_listwise_top1_accuracy(model, x_train, listwise_train_groups, device=device)
        listwise_acc_by_source = evaluate_listwise_top1_accuracy_by_source(
            model,
            x_train,
            listwise_train_groups_by_source,
            device=device,
        )
        balanced_listwise_acc = _mean_metric(listwise_acc_by_source.values())
        epoch_report = {
            "epoch": epoch + 1,
            "train_loss": round(total / max(n_seen, 1), 6),
            "train_preference_accuracy": round(pref_acc, 6) if pref_acc is not None else None,
            "balanced_train_preference_accuracy": round(balanced_pref_acc, 6) if balanced_pref_acc is not None else None,
            "train_listwise_top1_accuracy": round(listwise_acc, 6) if listwise_acc is not None else None,
            "balanced_train_listwise_top1_accuracy": (
                round(balanced_listwise_acc, 6) if balanced_listwise_acc is not None else None
            ),
            "train_preference_accuracy_by_source": {
                key: round(value, 6)
                for key, value in sorted(pref_acc_by_source.items())
            },
            "train_listwise_top1_accuracy_by_source": {
                key: round(value, 6)
                for key, value in sorted(listwise_acc_by_source.items())
            },
            **val_metrics,
        }
        history.append(epoch_report)
        selection = _selection_score(
            selection_metric,
            val_metrics=val_metrics,
            train_preference_accuracy=pref_acc,
            balanced_train_preference_accuracy=balanced_pref_acc,
            train_listwise_top1_accuracy=listwise_acc,
            balanced_train_listwise_top1_accuracy=balanced_listwise_acc,
            train_loss=total / max(n_seen, 1),
        )
        if selection > best_metric:
            best_metric = selection
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate(model, x_val.to(device), y_val.to(device), value_val.to(device), val_rows)
    final_pref_acc = evaluate_preference_accuracy(model, x_train, pref_pairs, device=device)
    final_pref_acc_by_source = evaluate_preference_accuracy_by_source(model, x_train, pref_pairs_by_source, device=device)
    final_balanced_pref_acc = _mean_metric(final_pref_acc_by_source.values())
    final_listwise_acc = evaluate_listwise_top1_accuracy(model, x_train, listwise_train_groups, device=device)
    final_listwise_acc_by_source = evaluate_listwise_top1_accuracy_by_source(
        model,
        x_train,
        listwise_train_groups_by_source,
        device=device,
    )
    final_balanced_listwise_acc = _mean_metric(final_listwise_acc_by_source.values())
    result = {
        "schema_version": "v4_cascade_product_value_training.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_pack": str(train_pack),
            "val_pack": str(val_pack),
            "preference_pack": str(preference_pack) if preference_pack else None,
            "listwise_pack": str(listwise_pack) if listwise_pack else None,
            "extra_train_pack": [str(path) for path in extra_train_pack or []],
            "extra_preference_pack": [str(path) for path in extra_preference_pack or []],
            "extra_listwise_pack": [str(path) for path in extra_listwise_pack or []],
            "output": str(output),
            "report": str(report),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_preferences": len(preferences),
            "n_train_preferences": len(pref_pairs),
            "n_listwise_groups": len(listwise_groups),
            "n_train_listwise_groups": len(listwise_train_groups),
            "n_train_preferences_by_source": {
                key: len(value)
                for key, value in sorted(pref_pairs_by_source.items())
            },
            "n_train_listwise_groups_by_source": {
                key: len(value)
                for key, value in sorted(listwise_train_groups_by_source.items())
            },
            "preference_source_counts": dict(Counter(str(pref.get("preference_source") or "unknown") for pref in preferences)),
            "listwise_source_counts": dict(
                Counter(str(row.get("preference_source") or row.get("listwise_source") or "unknown") for row in listwise_groups)
            ),
            "hidden": hidden,
            "n_bits": n_bits,
            "device": device,
            "preference_loss_weight": preference_loss_weight,
            "listwise_loss_weight": listwise_loss_weight,
            "label_loss_weight": label_loss_weight,
            "value_loss_weight": value_loss_weight,
            "preference_sampling": preference_sampling,
            "selection_metric": selection_metric,
            "label_names": list(ROUTE_LABEL_NAMES),
            "feature_schema": dataset.schema,
            "training_contract": "v4_route_product_value_not_autoplanner_trace.v1",
        },
        "best_checkpoint": {
            "epoch": best_epoch,
            "selection_metric": selection_metric,
            "selection_score": round(best_metric, 6),
        },
        "final_metrics": final_metrics,
        "final_train_preference_accuracy": round(final_pref_acc, 6) if final_pref_acc is not None else None,
        "final_balanced_train_preference_accuracy": round(final_balanced_pref_acc, 6) if final_balanced_pref_acc is not None else None,
        "final_train_listwise_top1_accuracy": round(final_listwise_acc, 6) if final_listwise_acc is not None else None,
        "final_balanced_train_listwise_top1_accuracy": (
            round(final_balanced_listwise_acc, 6) if final_balanced_listwise_acc is not None else None
        ),
        "final_train_preference_accuracy_by_source": {
            key: round(value, 6)
            for key, value in sorted(final_pref_acc_by_source.items())
        },
        "final_train_listwise_top1_accuracy_by_source": {
            key: round(value, 6)
            for key, value in sorted(final_listwise_acc_by_source.items())
        },
        "history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_class": "V4CascadeProductValueNetwork",
            "feature_schema": dataset.schema,
            "hidden": hidden,
            "label_names": list(ROUTE_LABEL_NAMES),
            "training_contract": "v4_route_product_value_not_autoplanner_trace.v1",
        },
        output,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_output is not None:
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(_markdown(result), encoding="utf-8")
    return result


def build_dataset(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], *, n_bits: int) -> ProductValueDataset:
    rows = list(train_rows) + list(val_rows)
    schema = build_route_feature_schema(train_rows, n_bits=n_bits)
    x = np.asarray([route_feature_vector(row, schema) for row in rows], dtype=np.float32)
    y = np.asarray([route_label_vector(row) for row in rows], dtype=np.float32)
    value_target = np.asarray([float(row.get("value_target") or np.mean(route_label_vector(row))) for row in rows], dtype=np.float32)
    schema["feature_dim"] = int(x.shape[1]) if len(x) else 0
    return ProductValueDataset(rows=rows, x=x, y=y, value_target=value_target, schema=schema)


def evaluate(
    model: V4CascadeProductValueNetwork,
    x: torch.Tensor,
    y: torch.Tensor,
    value_target: torch.Tensor,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        label_logits, value_logits = model(x)
        label_loss = F.binary_cross_entropy_with_logits(label_logits, y)
        value_loss = F.binary_cross_entropy_with_logits(value_logits, value_target)
        label_probs = torch.sigmoid(label_logits).detach().cpu().numpy()
        value_probs = torch.sigmoid(value_logits).detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    value_np = value_target.detach().cpu().numpy()
    out = {
        "val_label_loss": round(float(label_loss.item()), 6),
        "val_value_loss": round(float(value_loss.item()), 6),
        "value_auc": round(_binary_auc(value_probs, (value_np >= np.median(value_np)).astype(np.float32)), 6),
        "value_spearman": round(_spearman(value_probs, value_np), 6),
    }
    for idx, label in enumerate(ROUTE_LABEL_NAMES):
        out[f"{label}_auc"] = round(_binary_auc(label_probs[:, idx], y_np[:, idx]), 6)
    out["val_rows"] = len(rows)
    return out


def evaluate_preference_accuracy(
    model: V4CascadeProductValueNetwork,
    x_train: torch.Tensor,
    pref_pairs: list[tuple[int, int]],
    *,
    device: str,
) -> float | None:
    if not pref_pairs:
        return None
    model.eval()
    with torch.no_grad():
        _, logits = model(x_train.to(device))
        scores = torch.sigmoid(logits).detach().cpu().numpy()
    correct = sum(1 for better, worse in pref_pairs if scores[better] > scores[worse])
    return correct / max(len(pref_pairs), 1)


def evaluate_preference_accuracy_by_source(
    model: V4CascadeProductValueNetwork,
    x_train: torch.Tensor,
    pref_pairs_by_source: dict[str, list[tuple[int, int]]],
    *,
    device: str,
) -> dict[str, float]:
    if not pref_pairs_by_source:
        return {}
    model.eval()
    with torch.no_grad():
        _, logits = model(x_train.to(device))
        scores = torch.sigmoid(logits).detach().cpu().numpy()
    out = {}
    for source, pairs in pref_pairs_by_source.items():
        if not pairs:
            continue
        correct = sum(1 for better, worse in pairs if scores[better] > scores[worse])
        out[source] = correct / max(len(pairs), 1)
    return out


def evaluate_listwise_top1_accuracy(
    model: V4CascadeProductValueNetwork,
    x_train: torch.Tensor,
    groups: list[list[int]],
    *,
    device: str,
) -> float | None:
    if not groups:
        return None
    model.eval()
    with torch.no_grad():
        _, logits = model(x_train.to(device))
        scores = torch.sigmoid(logits).detach().cpu().numpy()
    correct = 0
    total = 0
    for indices in groups:
        if len(indices) < 2:
            continue
        best_idx = max(indices, key=lambda idx: float(scores[idx]))
        correct += int(best_idx == indices[0])
        total += 1
    return correct / max(total, 1) if total else None


def evaluate_listwise_top1_accuracy_by_source(
    model: V4CascadeProductValueNetwork,
    x_train: torch.Tensor,
    groups_by_source: dict[str, list[list[int]]],
    *,
    device: str,
) -> dict[str, float]:
    out = {}
    for source, groups in groups_by_source.items():
        value = evaluate_listwise_top1_accuracy(model, x_train, groups, device=device)
        if value is not None:
            out[source] = value
    return out


def _preference_loss(
    model: V4CascadeProductValueNetwork,
    x_train: torch.Tensor,
    pref_pairs: list[tuple[int, int]],
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    if not pref_pairs:
        return torch.tensor(0.0, device=device)
    idx = np.random.choice(len(pref_pairs), size=min(len(pref_pairs), max(8, batch_size)), replace=len(pref_pairs) < batch_size)
    better_idx = torch.tensor([pref_pairs[int(i)][0] for i in idx], dtype=torch.long, device=device)
    worse_idx = torch.tensor([pref_pairs[int(i)][1] for i in idx], dtype=torch.long, device=device)
    xb = x_train.to(device)
    _, better_logits = model(xb[better_idx])
    _, worse_logits = model(xb[worse_idx])
    return F.softplus(-(better_logits - worse_logits)).mean()


def _preference_loss_balanced_by_source(
    model: V4CascadeProductValueNetwork,
    x_train: torch.Tensor,
    pref_pairs_by_source: dict[str, list[tuple[int, int]]],
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    losses = []
    sources = [source for source, pairs in sorted(pref_pairs_by_source.items()) if pairs]
    if not sources:
        return torch.tensor(0.0, device=device)
    per_source = max(4, batch_size // max(len(sources), 1))
    for source in sources:
        pairs = pref_pairs_by_source[source]
        losses.append(_preference_loss(model, x_train, pairs, device=device, batch_size=per_source))
    return torch.stack(losses).mean()


def _listwise_loss(
    model: V4CascadeProductValueNetwork,
    x_train: torch.Tensor,
    groups: list[list[int]],
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    if not groups:
        return torch.tensor(0.0, device=device)
    group_count = min(len(groups), max(2, batch_size // 16))
    sampled = np.random.choice(len(groups), size=group_count, replace=len(groups) < group_count)
    losses = []
    xb = x_train.to(device)
    for group_idx in sampled:
        indices = groups[int(group_idx)]
        if len(indices) < 2:
            continue
        tensor_idx = torch.tensor(indices, dtype=torch.long, device=device)
        _, logits = model(xb[tensor_idx])
        # Plackett-Luce loss: ordered_route_ids are best-to-worst, and each
        # position should beat all candidates that remain below it.
        parts = []
        for pos in range(len(indices) - 1):
            remaining = logits[pos:]
            parts.append(torch.logsumexp(remaining, dim=0) - logits[pos])
        if parts:
            losses.append(torch.stack(parts).mean())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _mean_metric(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _read_preferences(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return _read_jsonl(path)


def _read_listwise_groups(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return _read_jsonl(path)


def _listwise_train_groups(rows: list[dict[str, Any]], train_by_id: dict[str, int]) -> list[list[int]]:
    out = []
    for row in rows:
        indices = _listwise_group_indices(row, train_by_id)
        if len(indices) >= 2:
            out.append(indices)
    return out


def _listwise_group_indices(row: dict[str, Any], train_by_id: dict[str, int]) -> list[int]:
    ids = row.get("ordered_route_ids") or row.get("route_ids") or []
    if not isinstance(ids, list):
        return []
    seen = set()
    indices = []
    for route_id in ids:
        key = str(route_id or "")
        if not key or key in seen or key not in train_by_id:
            continue
        seen.add(key)
        indices.append(train_by_id[key])
    return indices


def _selection_score(
    selection_metric: str,
    *,
    val_metrics: dict[str, Any],
    train_preference_accuracy: float | None,
    balanced_train_preference_accuracy: float | None,
    train_listwise_top1_accuracy: float | None,
    balanced_train_listwise_top1_accuracy: float | None,
    train_loss: float,
) -> float:
    if selection_metric == "train_preference_accuracy":
        return float(train_preference_accuracy or 0.0)
    if selection_metric == "balanced_train_preference_accuracy":
        return float(balanced_train_preference_accuracy or 0.0)
    if selection_metric == "train_listwise_top1_accuracy":
        return float(train_listwise_top1_accuracy or 0.0)
    if selection_metric == "balanced_train_listwise_top1_accuracy":
        return float(balanced_train_listwise_top1_accuracy or 0.0)
    if selection_metric == "negative_train_loss":
        return -float(train_loss)
    return float(val_metrics.get("value_auc") or 0.0)


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    pos = scores[labels >= 0.5]
    neg = scores[labels < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    total = 0.0
    for value in pos:
        total += float(np.sum(value > neg)) + 0.5 * float(np.sum(value == neg))
    return total / float(len(pos) * len(neg))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) < 2:
        return 0.0
    ra = _rank(a)
    rb = _rank(b)
    denom = float(np.std(ra) * np.std(rb))
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return float(np.mean((ra - np.mean(ra)) * (rb - np.mean(rb))) / denom)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v4 Cascade Product-Value Training",
        "",
        f"- Train rows: `{report['metadata']['n_train']}`",
        f"- Val rows: `{report['metadata']['n_val']}`",
        f"- Preferences: `{report['metadata']['n_preferences']}`",
        f"- Best epoch: `{report['best_checkpoint']['epoch']}`",
        f"- Best value AUC: `{report['best_checkpoint']['selection_score']}`",
        "",
        "## Final Metrics",
        "",
        "```json",
        json.dumps(report.get("final_metrics") or {}, indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train v4 cascade product-value route reranker")
    ap.add_argument("--train-pack", required=True)
    ap.add_argument("--val-pack", required=True)
    ap.add_argument("--preference-pack")
    ap.add_argument("--listwise-pack")
    ap.add_argument("--extra-train-pack", action="append", default=[])
    ap.add_argument("--extra-preference-pack", action="append", default=[])
    ap.add_argument("--extra-listwise-pack", action="append", default=[])
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--md-output")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--n-bits", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device")
    ap.add_argument("--preference-loss-weight", type=float, default=0.4)
    ap.add_argument("--listwise-loss-weight", type=float, default=0.0)
    ap.add_argument("--label-loss-weight", type=float, default=1.0)
    ap.add_argument("--value-loss-weight", type=float, default=0.6)
    ap.add_argument("--preference-sampling", default="random", choices=["random", "balanced_by_source"])
    ap.add_argument(
        "--selection-metric",
        default="value_auc",
        choices=[
            "value_auc",
            "train_preference_accuracy",
            "balanced_train_preference_accuracy",
            "train_listwise_top1_accuracy",
            "balanced_train_listwise_top1_accuracy",
            "negative_train_loss",
        ],
    )
    args = ap.parse_args()
    result = train_v4_cascade_product_value(
        train_pack=Path(args.train_pack),
        val_pack=Path(args.val_pack),
        preference_pack=Path(args.preference_pack) if args.preference_pack else None,
        listwise_pack=Path(args.listwise_pack) if args.listwise_pack else None,
        extra_train_pack=[Path(path) for path in args.extra_train_pack],
        extra_preference_pack=[Path(path) for path in args.extra_preference_pack],
        extra_listwise_pack=[Path(path) for path in args.extra_listwise_pack],
        output=Path(args.output),
        report=Path(args.report),
        md_output=Path(args.md_output) if args.md_output else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        n_bits=args.n_bits,
        seed=args.seed,
        device=args.device,
        preference_loss_weight=args.preference_loss_weight,
        listwise_loss_weight=args.listwise_loss_weight,
        label_loss_weight=args.label_loss_weight,
        value_loss_weight=args.value_loss_weight,
        preference_sampling=args.preference_sampling,
        selection_metric=args.selection_metric,
    )
    print(json.dumps(result["final_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

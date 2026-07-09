"""Calibrate and freeze route-tree value backup on a locked validation split."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cascade_planner.eval.train_vnext_from_pack import build_route_tree_search_policy_dataset
from cascade_planner.route_tree.runtime import RouteTreeRuntime


def calibrate_route_tree_value_from_traces(
    *,
    checkpoint_path: Path,
    trace_paths: list[Path],
    output_json: Path,
    output_md: Path | None = None,
    frozen_checkpoint_path: Path | None = None,
    validation_set_id: str = "",
    source_policy_version: str = "",
    device: str = "cpu",
    max_ece: float = 0.25,
    min_rows: int = 10,
) -> dict[str, Any]:
    runtime = RouteTreeRuntime(checkpoint_path)
    dataset = build_route_tree_search_policy_dataset(
        trace_paths,
        n_bits=runtime.n_bits,
        max_candidates=runtime.max_candidates,
        max_steps=runtime.max_steps,
        max_open_leaves=runtime.max_open_leaves,
    )
    logits = _policy_logits(runtime, dataset, device=device)
    value_labels = np.asarray(dataset.value, dtype=np.float32)
    value_binary = (value_labels >= 0.5).astype(np.float32)
    temperature = _fit_temperature(logits["value"], value_labels)
    calibrated_value = _sigmoid(logits["value"] / max(temperature, 1e-6))
    ece, bins = _binary_ece_bins(value_binary, calibrated_value, n_bins=10)
    thresholds = _threshold_selection(value_binary, calibrated_value)
    source_caps = _source_budget_cap_selection(logits["budget_probs"], runtime.source_budget_groups)
    outcomes = {
        "value_auroc": _binary_auc(value_binary, calibrated_value),
        "solved_auroc": _binary_auc(dataset.solved, _sigmoid(logits["solved"])),
        "stock_auroc": _binary_auc(dataset.stock_closed, _sigmoid(logits["stock"])),
        "progressive_auroc": _binary_auc(dataset.progressive, _sigmoid(logits["progressive"])),
        "compatibility_auroc": _binary_auc(dataset.compatibility, _sigmoid(logits["compatibility"])),
    }
    accepted = bool(len(value_labels) >= min_rows and ece <= max_ece)
    calibration = {
        "calibrated": accepted,
        "method": "temperature_scaling_locked_validation",
        "temperature": round(float(temperature), 6),
        "ece": round(float(ece), 6),
        "max_ece": float(max_ece),
        "n_rows": int(len(value_labels)),
        "min_rows": int(min_rows),
        "validation_set_id": validation_set_id,
        "validation_trace_paths": [str(path) for path in trace_paths],
        "source_policy_version": source_policy_version or str(checkpoint_path),
        "value_target": "solved_stock_progressive_compatibility_utility",
        "bins": bins,
        "outcomes": outcomes,
        "value_threshold_selection": thresholds,
        "source_budget_cap_selection": source_caps,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "Accepted checkpoints may enable route_value backup. Rejected "
            "calibration reports must not be promoted to frozen runtime."
        ),
    }
    report = {
        "schema_version": "route_tree_value_calibration.v1",
        "checkpoint": str(checkpoint_path),
        "frozen_checkpoint": str(frozen_checkpoint_path) if frozen_checkpoint_path else None,
        "calibration_accepted": accepted,
        "calibration": calibration,
    }
    if accepted and frozen_checkpoint_path is not None:
        _write_frozen_checkpoint(
            checkpoint_path=checkpoint_path,
            frozen_checkpoint_path=frozen_checkpoint_path,
            calibration=calibration,
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_report_markdown(report), encoding="utf-8")
    return report


def _policy_logits(runtime: RouteTreeRuntime, dataset, *, device: str) -> dict[str, np.ndarray]:
    device_t = torch.device(device)
    runtime.model.to(device_t)
    runtime.model.eval()
    rows = len(dataset.rows)
    batch_size = 256
    outputs: dict[str, list[np.ndarray]] = {
        "value": [],
        "solved": [],
        "stock": [],
        "progressive": [],
        "compatibility": [],
        "budget_probs": [],
    }
    with torch.no_grad():
        for start in range(0, rows, batch_size):
            end = min(start + batch_size, rows)
            out = runtime.model(
                torch.tensor(dataset.step_tokens[start:end], dtype=torch.float32, device=device_t),
                torch.tensor(dataset.step_mask[start:end], dtype=torch.bool, device=device_t),
                torch.tensor(dataset.route_features[start:end], dtype=torch.float32, device=device_t),
                torch.tensor(dataset.action_features[start:end], dtype=torch.float32, device=device_t),
                torch.tensor(dataset.action_mask[start:end], dtype=torch.bool, device=device_t),
                torch.tensor(dataset.node_features[start:end], dtype=torch.float32, device=device_t),
                torch.tensor(dataset.node_mask[start:end], dtype=torch.bool, device=device_t),
            )
            outputs["value"].append(out["value_logit"].detach().cpu().numpy())
            outputs["solved"].append(out["solved_logit"].detach().cpu().numpy())
            outputs["stock"].append(out["stock_logit"].detach().cpu().numpy())
            outputs["progressive"].append(out["progressive_logit"].detach().cpu().numpy())
            outputs["compatibility"].append(out["compatibility_logit"].detach().cpu().numpy())
            outputs["budget_probs"].append(torch.softmax(out["budget_logits"], dim=-1).detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if logits.size == 0:
        return 1.0
    grid = np.concatenate([
        np.linspace(0.05, 1.0, 80),
        np.linspace(1.05, 10.0, 180),
    ])
    losses = [_bce_with_logits(logits / temp, labels) for temp in grid]
    return float(grid[int(np.argmin(losses))])


def _bce_with_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    labels = np.clip(labels, 0.0, 1.0)
    return float(np.mean(np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))))


def _binary_ece_bins(labels: np.ndarray, probs: np.ndarray, *, n_bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    y = (np.asarray(labels, dtype=np.float32).reshape(-1) >= 0.5).astype(np.float32)
    p = np.clip(np.asarray(probs, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if y.size == 0:
        return 0.0, []
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        mask = (p >= lo) & (p <= hi) if idx == n_bins - 1 else (p >= lo) & (p < hi)
        count = int(mask.sum())
        confidence = float(p[mask].mean()) if count else 0.0
        accuracy = float(y[mask].mean()) if count else 0.0
        if count:
            ece += (count / max(len(y), 1)) * abs(confidence - accuracy)
        bins.append({
            "bin": idx,
            "lo": round(lo, 3),
            "hi": round(hi, 3),
            "count": count,
            "confidence": round(confidence, 6),
            "accuracy": round(accuracy, 6),
        })
    return float(ece), bins


def _threshold_selection(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y = (np.asarray(labels, dtype=np.float32).reshape(-1) >= 0.5).astype(np.int32)
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    best = {"threshold": 0.5, "f1": 0.0, "balanced_accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (p >= threshold).astype(np.int32)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        balanced = 0.5 * (recall + specificity)
        if (balanced, f1) > (best["balanced_accuracy"], best["f1"]):
            best = {
                "threshold": round(float(threshold), 3),
                "f1": round(float(f1), 6),
                "balanced_accuracy": round(float(balanced), 6),
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
            }
    return best


def _source_budget_cap_selection(probs: np.ndarray, groups: list[str]) -> dict[str, Any]:
    if probs.size == 0:
        return {}
    out: dict[str, Any] = {}
    for idx, group in enumerate(groups[: probs.shape[1]]):
        values = probs[:, idx]
        out[group] = {
            "mean_share": round(float(np.mean(values)), 6),
            "p90_share": round(float(np.quantile(values, 0.90)), 6),
            "recommended_max_share": round(float(min(0.90, max(0.10, np.quantile(values, 0.90)))), 6),
        }
    return out


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
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / max(pos * neg, 1)
    return round(float(auc), 6)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-value))


def _write_frozen_checkpoint(*, checkpoint_path: Path, frozen_checkpoint_path: Path, calibration: dict[str, Any]) -> None:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    metadata = dict(ckpt.get("metadata") or {})
    feature_schema = dict(ckpt.get("feature_schema") or {})
    metadata["value_calibrated"] = True
    metadata["value_calibration"] = calibration
    feature_schema["value_calibrated"] = True
    feature_schema["value_calibration"] = calibration
    ckpt["metadata"] = metadata
    ckpt["feature_schema"] = feature_schema
    ckpt["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    frozen_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(frozen_checkpoint_path))


def _report_markdown(report: dict[str, Any]) -> str:
    calibration = report.get("calibration") or {}
    outcomes = calibration.get("outcomes") or {}
    lines = [
        "# Route-Tree Value Calibration",
        "",
        f"Checkpoint: `{report.get('checkpoint')}`",
        f"Frozen checkpoint: `{report.get('frozen_checkpoint')}`",
        f"Accepted: `{report.get('calibration_accepted')}`",
        "",
        "## Summary",
        "",
        f"- rows: `{calibration.get('n_rows')}`",
        f"- temperature: `{calibration.get('temperature')}`",
        f"- ECE: `{calibration.get('ece')}`",
        f"- validation_set_id: `{calibration.get('validation_set_id')}`",
        "",
        "## AUROC",
        "",
    ]
    for key, value in outcomes.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Bins",
        "",
        "| Bin | Range | Count | Confidence | Accuracy |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in calibration.get("bins") or []:
        lines.append(
            f"| {row.get('bin')} | {row.get('lo')}-{row.get('hi')} | "
            f"{row.get('count')} | {row.get('confidence')} | {row.get('accuracy')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate route-tree value backup on locked validation traces")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--trace", action="append", required=True, help="Locked validation trace JSONL; can be repeated")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--frozen-checkpoint", default=None)
    ap.add_argument("--validation-set-id", default="")
    ap.add_argument("--source-policy-version", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-ece", type=float, default=0.25)
    ap.add_argument("--min-rows", type=int, default=10)
    args = ap.parse_args()
    report = calibrate_route_tree_value_from_traces(
        checkpoint_path=Path(args.checkpoint),
        trace_paths=[Path(path) for path in args.trace],
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        frozen_checkpoint_path=Path(args.frozen_checkpoint) if args.frozen_checkpoint else None,
        validation_set_id=args.validation_set_id,
        source_policy_version=args.source_policy_version,
        device=args.device,
        max_ece=args.max_ece,
        min_rows=args.min_rows,
    )
    print(json.dumps({
        "calibration_accepted": report["calibration_accepted"],
        "temperature": report["calibration"]["temperature"],
        "ece": report["calibration"]["ece"],
        "frozen_checkpoint": report.get("frozen_checkpoint"),
    }, indent=2))


if __name__ == "__main__":
    main()

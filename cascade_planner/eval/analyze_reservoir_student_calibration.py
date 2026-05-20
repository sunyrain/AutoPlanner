"""Analyze reservoir student controller confidence against run outcomes."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cascade_planner.route_tree.reservoir_distilled import (
    ReservoirDistilledController,
    reservoir_controller_feature_vector,
)

RECOVERY_KEYS = [
    "candidate_gt_reactant_in_pool",
    "exact_reaction_in_route_pool",
    "gt_reactant_in_route_pool",
]


def analyze_reservoir_student_calibration(
    *,
    baseline_run: Path,
    student_run: Path,
    controller_path: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    baseline = _load_targets(baseline_run)
    student = _load_targets(student_run)
    controller = _load_controller(controller_path)
    rows = []
    for idx, (base_target, student_target) in enumerate(zip(baseline, student)):
        target_smiles = str(student_target.get("target_smiles") or base_target.get("target_smiles") or "")
        pred = _predict_root(controller, target_smiles)
        base_stock = _metric_bool(base_target, "strict_stock_solve_any")
        student_stock = _metric_bool(student_target, "strict_stock_solve_any")
        recovery = _recovery_comparison(base_target, student_target)
        rows.append(
            {
                "index": int(student_target.get("index", base_target.get("index", idx))),
                "target_smiles": target_smiles,
                "route_domain": student_target.get("route_domain") or base_target.get("route_domain") or "",
                "baseline_stock": base_stock,
                "student_stock": student_stock,
                "stock_delta": int(student_stock) - int(base_stock),
                "student_plan": _metric_bool(student_target, "plan"),
                "baseline_plan": _metric_bool(base_target, "plan"),
                **recovery,
                **pred,
            }
        )

    report = {
        "schema_version": "reservoir_student_calibration.v1",
        "baseline_run": str(baseline_run),
        "student_run": str(student_run),
        "controller_path": str(controller_path),
        "n": len(rows),
        "summary": _summary(rows),
        "by_confidence_bucket": _grouped(rows, key="confidence_bucket"),
        "by_selected_group": _grouped(rows, key="selected_group"),
        "by_domain": _grouped(rows, key="route_domain"),
        "stock_losses": [row for row in rows if row["baseline_stock"] and not row["student_stock"]],
        "stock_gains": [row for row in rows if row["student_stock"] and not row["baseline_stock"]],
        "recovery_losses": _recovery_loss_rows(rows),
        "student_priority_misses": _priority_misses(rows),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(report), encoding="utf-8")
    return report


def _load_controller(path: Path) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu")
    meta = ckpt.get("metadata") or {}
    groups = list(meta.get("source_groups") or [])
    budget_labels = list(meta.get("budget_labels") or [])
    model = ReservoirDistilledController(
        int(meta["input_dim"]),
        hidden_dim=int(meta.get("hidden_dim") or 256),
        dropout=float(meta.get("dropout") or 0.10),
        n_source_groups=len(groups),
        n_budget_labels=len(budget_labels),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "model": model,
        "input_dim": int(meta["input_dim"]),
        "n_bits": int(meta.get("n_bits") or 256),
        "groups": groups,
        "budget_labels": budget_labels,
    }


def _predict_root(controller: dict[str, Any], target_smiles: str) -> dict[str, Any]:
    x = reservoir_controller_feature_vector(
        product=target_smiles,
        leaf=target_smiles,
        source="",
        n_bits=int(controller["n_bits"]),
        input_dim=int(controller["input_dim"]),
        source_groups=controller["groups"],
    )
    with torch.no_grad():
        out = controller["model"](torch.tensor(x[None, :], dtype=torch.float32))
        probs = torch.softmax(out["source_group_logits"], dim=-1)[0].cpu().numpy()
        budget_probs = torch.softmax(out["budget_logits"], dim=-1)[0].cpu().numpy()
    groups = list(controller["groups"])
    selected_idx = int(np.argmax(probs)) if len(probs) else 0
    confidence = float(probs[selected_idx]) if len(probs) else 0.0
    budget_idx = int(np.argmax(budget_probs)) if len(budget_probs) else 0
    return {
        "selected_group": groups[selected_idx] if selected_idx < len(groups) else "",
        "confidence": confidence,
        "confidence_bucket": f"{int(confidence * 10) / 10:.1f}",
        "group_probs": {group: float(probs[idx]) for idx, group in enumerate(groups)},
        "budget_label": controller["budget_labels"][budget_idx] if budget_idx < len(controller["budget_labels"]) else "",
    }


def _load_targets(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = data.get("targets") or data.get("results") or []
    if not isinstance(targets, list):
        raise ValueError(f"run has no target list: {path}")
    return targets


def _metric_bool(target: dict[str, Any], key: str) -> bool:
    metrics = target.get("metrics") or {}
    return bool(metrics.get(key))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(rows), 1)
    losses = [row for row in rows if row["baseline_stock"] and not row["student_stock"]]
    gains = [row for row in rows if row["student_stock"] and not row["baseline_stock"]]
    summary = {
        "n": len(rows),
        "baseline_stock_rate": sum(bool(row["baseline_stock"]) for row in rows) / n,
        "student_stock_rate": sum(bool(row["student_stock"]) for row in rows) / n,
        "stock_loss_count": len(losses),
        "stock_gain_count": len(gains),
        "mean_confidence": sum(float(row["confidence"]) for row in rows) / n,
        "loss_mean_confidence": _mean([float(row["confidence"]) for row in losses]),
        "gain_mean_confidence": _mean([float(row["confidence"]) for row in gains]),
    }
    for key in RECOVERY_KEYS:
        baseline_key = f"baseline_{key}"
        student_key = f"student_{key}"
        loss_key = f"{key}_loss_count"
        gain_key = f"{key}_gain_count"
        summary[f"baseline_{key}_rate"] = sum(bool(row.get(baseline_key)) for row in rows) / n
        summary[f"student_{key}_rate"] = sum(bool(row.get(student_key)) for row in rows) / n
        summary[loss_key] = sum(1 for row in rows if row.get(baseline_key) and not row.get(student_key))
        summary[gain_key] = sum(1 for row in rows if row.get(student_key) and not row.get(baseline_key))
    return summary


def _grouped(rows: list[dict[str, Any]], *, key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "")].append(row)
    return {name: _summary(items) for name, items in sorted(grouped.items())}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _recovery_bool(target: dict[str, Any], key: str) -> bool:
    recovery = target.get("route_recovery") or {}
    return bool(recovery.get(key))


def _recovery_comparison(base_target: dict[str, Any], student_target: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in RECOVERY_KEYS:
        base_value = _recovery_bool(base_target, key)
        student_value = _recovery_bool(student_target, key)
        out[f"baseline_{key}"] = base_value
        out[f"student_{key}"] = student_value
        out[f"{key}_delta"] = int(student_value) - int(base_value)
    return out


def _recovery_loss_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [row for row in rows if row.get(f"baseline_{key}") and not row.get(f"student_{key}")]
        for key in RECOVERY_KEYS
    }


def _priority_misses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        miss_score = int(row["baseline_stock"] and not row["student_stock"])
        miss_score += sum(
            int(row.get(f"baseline_{key}") and not row.get(f"student_{key}"))
            for key in RECOVERY_KEYS
        )
        if miss_score:
            item = dict(row)
            item["miss_score"] = miss_score
            scored.append(item)
    return sorted(scored, key=lambda row: (-int(row["miss_score"]), str(row.get("route_domain") or ""), int(row.get("index") or 0)))


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reservoir Student Calibration",
        "",
        f"Baseline: `{report['baseline_run']}`",
        f"Student: `{report['student_run']}`",
        f"Controller: `{report['controller_path']}`",
        "",
        "## Summary",
        "",
        _summary_table(report["summary"]),
        "",
        "## By Selected Group",
        "",
        _group_table(report["by_selected_group"]),
        "",
        "## By Confidence Bucket",
        "",
        _group_table(report["by_confidence_bucket"]),
        "",
        "## Stock Losses",
        "",
        _target_table(report["stock_losses"]),
        "",
        "## Stock Gains",
        "",
        _target_table(report["stock_gains"]),
        "",
        "## Recovery Losses",
        "",
        _recovery_loss_markdown(report["recovery_losses"]),
        "",
        "## Student Priority Misses",
        "",
        _priority_miss_table(report["student_priority_misses"][:30]),
        "",
    ]
    return "\n".join(lines)


def _summary_table(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "| metric | value |",
            "| --- | ---: |",
            *[f"| {key} | {_fmt(value)} |" for key, value in summary.items()],
        ]
    )


def _group_table(groups: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| group | n | baseline stock | student stock | losses | gains | mean conf | loss conf | gain conf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, row in groups.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(row["n"]),
                    _fmt(row["baseline_stock_rate"]),
                    _fmt(row["student_stock_rate"]),
                    str(row["stock_loss_count"]),
                    str(row["stock_gain_count"]),
                    _fmt(row["mean_confidence"]),
                    _fmt(row["loss_mean_confidence"]),
                    _fmt(row["gain_mean_confidence"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _target_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "None."
    lines = [
        "| idx | domain | selected | conf | chemical | enzymatic | retrieval | target |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        probs = row.get("group_probs") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["index"]),
                    str(row["route_domain"]),
                    str(row["selected_group"]),
                    _fmt(row["confidence"]),
                    _fmt(probs.get("chemical")),
                    _fmt(probs.get("enzymatic")),
                    _fmt(probs.get("retrieval")),
                    str(row["target_smiles"])[:80],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _recovery_loss_markdown(groups: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "| metric | losses | top indices |",
        "| --- | ---: | --- |",
    ]
    for key, rows in groups.items():
        indices = ", ".join(str(row.get("index")) for row in rows[:20])
        lines.append(f"| {key} | {len(rows)} | {indices} |")
    return "\n".join(lines)


def _priority_miss_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "None."
    lines = [
        "| idx | score | domain | stock loss | cand GT loss | exact loss | route GT loss | selected | conf | target |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("index")),
                    str(row.get("miss_score")),
                    str(row.get("route_domain")),
                    str(bool(row.get("baseline_stock") and not row.get("student_stock"))),
                    str(bool(row.get("baseline_candidate_gt_reactant_in_pool") and not row.get("student_candidate_gt_reactant_in_pool"))),
                    str(bool(row.get("baseline_exact_reaction_in_route_pool") and not row.get("student_exact_reaction_in_route_pool"))),
                    str(bool(row.get("baseline_gt_reactant_in_route_pool") and not row.get("student_gt_reactant_in_route_pool"))),
                    str(row.get("selected_group")),
                    _fmt(row.get("confidence")),
                    str(row.get("target_smiles"))[:80],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze reservoir student controller calibration")
    ap.add_argument("--baseline-run", required=True)
    ap.add_argument("--student-run", required=True)
    ap.add_argument("--controller", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    report = analyze_reservoir_student_calibration(
        baseline_run=Path(args.baseline_run),
        student_run=Path(args.student_run),
        controller_path=Path(args.controller),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

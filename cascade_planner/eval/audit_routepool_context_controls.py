"""Negative controls for CCTS route-pool ranking.

The goal is to verify whether cascade-context features carry row-level signal
inside a fixed ChemEnzy route pool.  The controls keep the same strict
train/val/test split and same target groups, then break either feature-label
alignment or train/val labels.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.eval.train_route_pool_ranker import train_route_pool_ranker


SCHEMA_VERSION = "routepool_context_control_audit.v1"
CASCADE_PREFIXES = ("ccts_", "block_", "v4_step_")
CASCADE_EXTRA_FIELDS = {"n_blocks"}


def run_routepool_context_controls(
    *,
    train_jsonl: Path,
    val_jsonl: Path,
    test_jsonl: Path,
    output_dir: Path,
    seed: int = 42,
    max_pos_per_group: int = 8,
    max_neg_per_pos: int = 24,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = output_dir / "control_packs"
    original = {"train": Path(train_jsonl), "val": Path(val_jsonl), "test": Path(test_jsonl)}
    rows_by_split = {split: _read_jsonl(path) for split, path in original.items()}
    cascade_keys = _cascade_feature_keys(rows_by_split["train"])
    feature_shuffle = _write_feature_shuffle_pack(
        rows_by_split,
        cascade_keys=cascade_keys,
        output_dir=pack_dir / "cascade_features_shuffled_within_target",
        seed=seed,
    )
    label_shuffle = _write_label_shuffle_pack(
        rows_by_split,
        output_dir=pack_dir / "train_val_labels_shuffled_within_target",
        seed=seed + 1000,
    )
    scenarios = [
        {
            "name": "native_only_original",
            "feature_set": "native_only",
            "paths": original,
            "control": "baseline_native_features_only",
        },
        {
            "name": "cascade_only_original",
            "feature_set": "cascade_only",
            "paths": original,
            "control": "true_cascade_features",
        },
        {
            "name": "cascade_only_feature_shuffle",
            "feature_set": "cascade_only",
            "paths": feature_shuffle,
            "control": "cascade_features_shuffled_within_target_all_splits",
        },
        {
            "name": "cascade_only_label_shuffle",
            "feature_set": "cascade_only",
            "paths": label_shuffle,
            "control": "train_val_labels_shuffled_within_target_test_real",
        },
        {
            "name": "all_original",
            "feature_set": "all",
            "paths": original,
            "control": "all_features_true",
        },
    ]
    reports = {}
    rows = []
    for scenario in scenarios:
        scenario_dir = output_dir / scenario["name"]
        result = train_route_pool_ranker(
            train_jsonl=scenario["paths"]["train"],
            val_jsonl=scenario["paths"]["val"],
            test_jsonl=scenario["paths"]["test"],
            output_dir=scenario_dir,
            feature_set=scenario["feature_set"],
            max_pos_per_group=max_pos_per_group,
            max_neg_per_pos=max_neg_per_pos,
            seed=seed,
        )
        reports[scenario["name"]] = {
            "report_json": str(scenario_dir / "route_pool_pairwise_ranker_report.json"),
            "report_md": str(scenario_dir / "route_pool_pairwise_ranker_report.md"),
            "model": str(scenario_dir / "route_pool_pairwise_ranker.pkl"),
        }
        rows.append(_scenario_summary_row(scenario, result))
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "test_jsonl": str(test_jsonl),
            "output_dir": str(output_dir),
            "seed": int(seed),
            "max_pos_per_group": int(max_pos_per_group),
            "max_neg_per_pos": int(max_neg_per_pos),
            "cascade_feature_keys": cascade_keys,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "negative controls for strict same-target ChemEnzy route-pool ranking",
        },
        "control_pack_outputs": {
            "feature_shuffle": {key: str(value) for key, value in feature_shuffle.items()},
            "label_shuffle": {key: str(value) for key, value in label_shuffle.items()},
        },
        "scenario_reports": reports,
        "scenario_metrics": rows,
        "diagnostics": _diagnostics(rows),
    }
    summary_json = output_dir / "routepool_context_control_summary.json"
    summary_md = output_dir / "routepool_context_control_summary.md"
    summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md.write_text(_markdown(result), encoding="utf-8")
    result["outputs"] = {"summary_json": str(summary_json), "summary_md": str(summary_md)}
    summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _scenario_summary_row(scenario: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    model_test = ((result.get("model") or {}).get("test") or {})
    selection = result.get("selection") or {}
    native_test = (((result.get("baselines") or {}).get("native_rank") or {}).get("test") or {})
    ccts_mean_test = (((result.get("baselines") or {}).get("ccts_model_mean") or {}).get("test") or {})
    return {
        "name": scenario["name"],
        "feature_set": scenario["feature_set"],
        "control": scenario["control"],
        "model_test_mrr_covered": model_test.get("mrr_covered"),
        "model_test_recall_at1_all": (model_test.get("recall_at_k_all") or {}).get("1"),
        "model_test_recall_at3_all": (model_test.get("recall_at_k_all") or {}).get("3"),
        "model_test_recall_at5_all": (model_test.get("recall_at_k_all") or {}).get("5"),
        "model_test_recall_at1_covered": (model_test.get("recall_at_k_covered") or {}).get("1"),
        "model_test_recall_at3_covered": (model_test.get("recall_at_k_covered") or {}).get("3"),
        "selected_method": selection.get("selected_method"),
        "selected_test_mrr_covered": selection.get("selected_test_mrr_covered"),
        "selected_test_recall_at3_all": selection.get("selected_test_recall_at3_all"),
        "native_test_mrr_covered": native_test.get("mrr_covered"),
        "native_test_recall_at3_all": (native_test.get("recall_at_k_all") or {}).get("3"),
        "ccts_model_mean_test_mrr_covered": ccts_mean_test.get("mrr_covered"),
        "ccts_model_mean_test_recall_at3_all": (ccts_mean_test.get("recall_at_k_all") or {}).get("3"),
    }


def _diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(row.get("name")): row for row in rows}

    def metric(name: str, key: str) -> float | None:
        value = by_name.get(name, {}).get(key)
        return float(value) if value is not None else None

    original_mrr = metric("cascade_only_original", "model_test_mrr_covered")
    shuffled_mrr = metric("cascade_only_feature_shuffle", "model_test_mrr_covered")
    label_mrr = metric("cascade_only_label_shuffle", "model_test_mrr_covered")
    native_mrr = metric("native_only_original", "native_test_mrr_covered")
    return {
        "cascade_original_minus_feature_shuffle_mrr": _delta(original_mrr, shuffled_mrr),
        "cascade_original_minus_label_shuffle_mrr": _delta(original_mrr, label_mrr),
        "cascade_original_minus_native_rank_mrr": _delta(original_mrr, native_mrr),
        "interpretation": _interpretation(original_mrr, shuffled_mrr, label_mrr, native_mrr),
    }


def _interpretation(
    original_mrr: float | None,
    shuffled_mrr: float | None,
    label_mrr: float | None,
    native_mrr: float | None,
) -> str:
    if original_mrr is None or shuffled_mrr is None or label_mrr is None or native_mrr is None:
        return "insufficient metrics"
    if original_mrr > shuffled_mrr + 0.05 and original_mrr > label_mrr + 0.05 and original_mrr > native_mrr + 0.05:
        return "cascade features pass negative controls at route-pool ranking level"
    if original_mrr <= shuffled_mrr + 0.05:
        return "feature-shuffle control remains too strong; cascade signal may be target-level or non-causal"
    if original_mrr <= label_mrr + 0.05:
        return "label-shuffle control remains too strong; training/evaluation may be unstable"
    return "cascade signal is weak or marginal under current controls"


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 6)


def _cascade_feature_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        for key in (row.get("feature") or {}):
            if key.startswith(CASCADE_PREFIXES) or key in CASCADE_EXTRA_FIELDS:
                keys.add(str(key))
    return sorted(keys)


def _write_feature_shuffle_pack(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    cascade_keys: list[str],
    output_dir: Path,
    seed: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for offset, split in enumerate(["train", "val", "test"]):
        rows = _shuffle_features_within_target(rows_by_split[split], cascade_keys, seed=seed + offset)
        path = output_dir / f"route_pool_ranker_{split}.jsonl"
        _write_jsonl(path, rows)
        out[split] = path
    return out


def _write_label_shuffle_pack(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
    seed: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for offset, split in enumerate(["train", "val", "test"]):
        if split == "test":
            rows = _deepcopy_rows(rows_by_split[split])
        else:
            rows = _shuffle_labels_within_target(rows_by_split[split], seed=seed + offset)
        path = output_dir / f"route_pool_ranker_{split}.jsonl"
        _write_jsonl(path, rows)
        out[split] = path
    return out


def _shuffle_features_within_target(rows: list[dict[str, Any]], keys: list[str], *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    out = _deepcopy_rows(rows)
    by_target = _indices_by_target(out)
    for indices in by_target.values():
        if len(indices) <= 1:
            continue
        vectors = [
            {key: (out[idx].get("feature") or {}).get(key) for key in keys}
            for idx in indices
        ]
        perm = rng.permutation(len(indices))
        for dst_pos, src_pos in enumerate(perm):
            feature = out[indices[dst_pos]].setdefault("feature", {})
            for key, value in vectors[int(src_pos)].items():
                feature[key] = value
            out[indices[dst_pos]]["control_metadata"] = {
                **(out[indices[dst_pos]].get("control_metadata") or {}),
                "cascade_features_shuffled_within_target": True,
            }
    return out


def _shuffle_labels_within_target(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    out = _deepcopy_rows(rows)
    by_target = _indices_by_target(out)
    for indices in by_target.values():
        if len(indices) <= 1:
            continue
        labels = [out[idx].get("route_label") for idx in indices]
        perm = rng.permutation(len(indices))
        for dst_pos, src_pos in enumerate(perm):
            out[indices[dst_pos]]["route_label"] = labels[int(src_pos)]
            out[indices[dst_pos]]["control_metadata"] = {
                **(out[indices[dst_pos]].get("control_metadata") or {}),
                "route_label_shuffled_within_target": True,
            }
    return out


def _indices_by_target(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        out[str(row.get("target_id") or "")].append(idx)
    return out


def _deepcopy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(rows, ensure_ascii=False))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Route-Pool Context Control Audit",
        "",
        f"- Contract: `{(result.get('metadata') or {}).get('contract')}`",
        f"- Cascade feature keys: `{len((result.get('metadata') or {}).get('cascade_feature_keys') or [])}`",
        f"- Interpretation: `{(result.get('diagnostics') or {}).get('interpretation')}`",
        "",
        "## Scenario Metrics",
        "",
        "| scenario | control | model MRR | model R@1 all | model R@3 all | native MRR | selected method | selected MRR |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in result.get("scenario_metrics") or []:
        lines.append(
            "| {name} | {control} | {model_test_mrr_covered} | {model_test_recall_at1_all} | {model_test_recall_at3_all} | {native_test_mrr_covered} | {selected_method} | {selected_test_mrr_covered} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "```json",
            json.dumps(result.get("diagnostics") or {}, indent=2, ensure_ascii=False),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run strict route-pool context negative controls")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pos-per-group", type=int, default=8)
    ap.add_argument("--max-neg-per-pos", type=int, default=24)
    args = ap.parse_args()
    result = run_routepool_context_controls(
        train_jsonl=Path(args.train_jsonl),
        val_jsonl=Path(args.val_jsonl),
        test_jsonl=Path(args.test_jsonl),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        max_pos_per_group=args.max_pos_per_group,
        max_neg_per_pos=args.max_neg_per_pos,
    )
    print(
        json.dumps(
            {
                "diagnostics": result.get("diagnostics"),
                "outputs": result.get("outputs"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

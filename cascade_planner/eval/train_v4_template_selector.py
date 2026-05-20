"""Train a lightweight selector over v4 template upstream proposals.

Input is the JSON produced by audit_v4_template_upstream_bridge.py.  The goal is
not to prove a final model, but to test whether simple non-leaky features can
move analog / transform-consistent template proposals upward.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMRanker


SCHEMA_VERSION = "v4_template_selector.v1"


def train_v4_template_selector(
    *,
    train_json: Path,
    eval_json: Path,
    output_json: Path,
    label: str = "analog_hit",
    seed: int = 42,
) -> dict[str, Any]:
    started = time.monotonic()
    train_payload = _read_json(train_json)
    eval_payload = _read_json(eval_json)
    feature_names = _feature_names(train_payload)
    train_data = _dataset(train_payload, feature_names=feature_names, label=label)
    eval_data = _dataset(eval_payload, feature_names=feature_names, label=label)
    if not train_data["rows"] or not eval_data["rows"]:
        raise ValueError("empty train/eval selector dataset")
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=160,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=int(seed),
        verbose=-1,
    )
    model.fit(train_data["x"], train_data["y"], group=train_data["groups"])
    train_scores = model.predict(train_data["x"])
    eval_scores = model.predict(eval_data["x"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_json": str(train_json),
            "eval_json": str(eval_json),
            "output_json": str(output_json),
            "label": label,
            "seed": seed,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "Uses only proposal metadata and template/provider scores; similarity labels are excluded from features.",
        },
        "feature_names": feature_names,
        "counts": {
            "train_rows": len(train_data["rows"]),
            "eval_rows": len(eval_data["rows"]),
            "train_groups": len(train_data["groups"]),
            "eval_groups": len(eval_data["groups"]),
            "train_positive_rows": int(train_data["y"].sum()),
            "eval_positive_rows": int(eval_data["y"].sum()),
        },
        "baseline_proposal_score": {
            "train": _evaluate(train_data["rows"], np.array([float(row.get("proposal_score") or 0.0) for row in train_data["rows"]])),
            "eval": _evaluate(eval_data["rows"], np.array([float(row.get("proposal_score") or 0.0) for row in eval_data["rows"]])),
        },
        "selector": {
            "train": _evaluate(train_data["rows"], train_scores),
            "eval": _evaluate(eval_data["rows"], eval_scores),
            "feature_importances": _feature_importances(model, feature_names),
        },
        "eval_examples": _ranked_examples(eval_data["rows"], eval_scores),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _dataset(payload: dict[str, Any], *, feature_names: list[str], label: str) -> dict[str, Any]:
    rows = []
    x_rows = []
    y_rows = []
    groups = []
    for target in payload.get("targets") or []:
        proposals = _proposal_rows(target)
        if not proposals:
            continue
        group_rows = []
        for proposal in proposals:
            row = dict(proposal)
            row["target_smiles"] = target.get("target_smiles")
            row["positive_label"] = bool(row.get(label))
            group_rows.append(row)
        if not group_rows:
            continue
        groups.append(len(group_rows))
        for row in group_rows:
            rows.append(row)
            x_rows.append([_feature_value(row, name) for name in feature_names])
            y_rows.append(1 if row.get("positive_label") else 0)
    return {
        "rows": rows,
        "x": np.asarray(x_rows, dtype=np.float32),
        "y": np.asarray(y_rows, dtype=np.int32),
        "groups": groups,
    }


def _feature_names(payload: dict[str, Any]) -> list[str]:
    transform_pairs = Counter()
    upstream_transforms = Counter()
    downstream_transforms = Counter()
    for target in payload.get("targets") or []:
        for row in _proposal_rows(target):
            pair = str(row.get("template_transform_pair") or "")
            upstream, downstream = _split_pair(pair)
            transform_pairs[pair] += 1
            upstream_transforms[upstream] += 1
            downstream_transforms[downstream] += 1
    top_pairs = [pair for pair, _ in transform_pairs.most_common(20)]
    top_upstream = [value for value, _ in upstream_transforms.most_common(12)]
    top_downstream = [value for value, _ in downstream_transforms.most_common(12)]
    return [
        "proposal_score",
        "downstream_rank_inv_sqrt",
        "template_rank_inv_sqrt",
        "outcome_rank_inv",
        "template_count_log",
        "connector_heavy_atoms_norm",
        "connector_is_main_reactant",
        "pair_has_downstream",
        "app_connector_main_similarity",
        "app_main_heavy_atoms_norm",
        "app_heavy_atom_delta_norm",
        "app_abs_heavy_atom_delta_norm",
        "app_total_reactant_atoms_norm",
        "app_reactant_count_norm",
        "app_largest_reactant_fraction",
        "app_template_example_best_transition_sim",
        "app_template_example_mean_top3_transition_sim",
        "app_template_example_best_product_sim",
        "app_template_example_best_main_sim",
        "rc_mapped_atom_count_norm",
        "rc_inherited_atom_count_norm",
        "rc_new_atom_count_norm",
        "rc_new_atom_fraction",
        "rc_inherited_atom_fraction",
        "rc_template_matched_atom_count_norm",
        "rc_template_matched_fraction",
    ] + [f"pair={pair}" for pair in top_pairs] + [f"upstream={value}" for value in top_upstream] + [f"downstream={value}" for value in top_downstream]


def _feature_value(row: dict[str, Any], name: str) -> float:
    if name == "proposal_score":
        return float(row.get("proposal_score") or 0.0)
    if name == "downstream_rank_inv_sqrt":
        return 1.0 / math.sqrt(max(1, int(row.get("downstream_rank") or 10**6)))
    if name == "template_rank_inv_sqrt":
        return 1.0 / math.sqrt(max(1, int(row.get("template_rank") or 10**6)))
    if name == "outcome_rank_inv":
        return 1.0 / max(1, int(row.get("outcome_rank") or 10**6))
    if name == "template_count_log":
        return math.log1p(float(row.get("template_count") or 0.0))
    if name == "connector_heavy_atoms_norm":
        return min(1.0, float(row.get("connector_heavy_atoms") or 0.0) / 40.0)
    if name == "connector_is_main_reactant":
        return 1.0 if row.get("connector_is_main_reactant") else 0.0
    if name == "pair_has_downstream":
        _, downstream = _split_pair(str(row.get("template_transform_pair") or ""))
        return 1.0 if downstream and downstream != "unknown" else 0.0
    if name == "app_reactant_count_norm":
        return min(1.0, float(row.get("app_reactant_count") or 0.0) / 4.0)
    if name == "rc_mapped_atom_count_norm":
        return min(1.0, float(row.get("rc_mapped_atom_count") or 0.0) / 80.0)
    if name == "rc_inherited_atom_count_norm":
        return min(1.0, float(row.get("rc_inherited_atom_count") or 0.0) / 80.0)
    if name == "rc_new_atom_count_norm":
        return min(1.0, float(row.get("rc_new_atom_count") or 0.0) / 20.0)
    if name == "rc_template_matched_atom_count_norm":
        return min(1.0, float(row.get("rc_template_matched_atom_count") or 0.0) / 20.0)
    if name.startswith("rc_"):
        try:
            return float(row.get(name) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if name.startswith("app_"):
        try:
            return float(row.get(name) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if name.startswith("pair="):
        return 1.0 if str(row.get("template_transform_pair") or "") == name.split("=", 1)[1] else 0.0
    if name.startswith("upstream="):
        upstream, _ = _split_pair(str(row.get("template_transform_pair") or ""))
        return 1.0 if upstream == name.split("=", 1)[1] else 0.0
    if name.startswith("downstream="):
        _, downstream = _split_pair(str(row.get("template_transform_pair") or ""))
        return 1.0 if downstream == name.split("=", 1)[1] else 0.0
    return 0.0


def _proposal_rows(target: dict[str, Any]) -> list[dict[str, Any]]:
    proposals = target.get("proposals")
    if isinstance(proposals, list):
        return [row for row in proposals if isinstance(row, dict)]
    return [row for row in target.get("examples") or [] if isinstance(row, dict)]


def _split_pair(pair: str) -> tuple[str, str]:
    left, sep, right = str(pair or "").partition("->")
    return (left.strip().lower() or "unknown", right.strip().lower() if sep else "")


def _evaluate(rows: list[dict[str, Any]], scores: np.ndarray, top_ks: tuple[int, ...] = (1, 3, 5, 10, 20, 50)) -> dict[str, Any]:
    by_target: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_target[str(row.get("target_smiles") or "")].append((row, float(score)))
    positive_targets = 0
    ranks = []
    hit_at = {k: 0 for k in top_ks}
    analog_at = {k: 0 for k in top_ks}
    pair_at = {k: 0 for k in top_ks}
    for target, items in by_target.items():
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        pos_ranks = [idx for idx, (row, _) in enumerate(ranked, start=1) if row.get("positive_label")]
        if pos_ranks:
            positive_targets += 1
            ranks.append(min(pos_ranks))
        for k in top_ks:
            top = [row for row, _ in ranked[:k]]
            hit_at[k] += int(any(row.get("positive_label") for row in top))
            analog_at[k] += int(any(row.get("analog_hit") for row in top))
            pair_at[k] += int(any(row.get("pair_and_analog") for row in top))
    denom = max(len(by_target), 1)
    return {
        "targets": len(by_target),
        "positive_targets": positive_targets,
        "mrr": round(sum(1.0 / rank for rank in ranks) / max(len(by_target), 1), 6),
        "mean_best_rank": round(sum(ranks) / max(len(ranks), 1), 6) if ranks else None,
        **{f"hit_at_{k}": round(hit_at[k] / denom, 6) for k in top_ks},
        **{f"analog_at_{k}": round(analog_at[k] / denom, 6) for k in top_ks},
        **{f"pair_and_analog_at_{k}": round(pair_at[k] / denom, 6) for k in top_ks},
    }


def _feature_importances(model: Any, names: list[str]) -> list[dict[str, Any]]:
    values = getattr(model, "feature_importances_", [])
    rows = [{"feature": name, "importance": int(value)} for name, value in zip(names, values)]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows[:30]


def _ranked_examples(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    by_target: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_target[str(row.get("target_smiles") or "")].append((row, float(score)))
    examples = []
    for target, items in sorted(by_target.items())[:10]:
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        examples.append(
            {
                "target_smiles": target,
                "top": [
                    {
                        "score": round(score, 6),
                        "proposal_score": row.get("proposal_score"),
                        "template_pair": row.get("template_transform_pair"),
                        "downstream_rank": row.get("downstream_rank"),
                        "template_rank": row.get("template_rank"),
                        "analog_hit": row.get("analog_hit"),
                        "pair_and_analog": row.get("pair_and_analog"),
                    }
                    for row, score in ranked[:5]
                ],
            }
        )
    return examples


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object: {path}")
    return payload


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v4 Template Selector Report",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(report.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Baseline",
        "",
        "```json",
        json.dumps((report.get("baseline_proposal_score") or {}).get("eval") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Selector",
        "",
        "```json",
        json.dumps((report.get("selector") or {}).get("eval") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight selector over v4 template proposal audit rows")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--label", choices=("analog_hit", "pair_and_analog"), default="analog_hit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train_v4_template_selector(
        train_json=Path(args.train_json),
        eval_json=Path(args.eval_json),
        output_json=Path(args.output_json),
        label=args.label,
        seed=args.seed,
    )
    print(json.dumps({"counts": report["counts"], "baseline_eval": report["baseline_proposal_score"]["eval"], "selector_eval": report["selector"]["eval"], "output_json": args.output_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

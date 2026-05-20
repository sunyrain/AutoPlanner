"""Train a lightweight transform-pair selector from v4 selector packs."""
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

from cascade_planner.eval.audit_provider_routepool_oracle import _fp, _fp_similarity, _transition_fp


SCHEMA_VERSION = "v4_transform_pair_selector.v1"


def train_v4_transform_pair_selector(
    *,
    train_jsonl: Path,
    eval_jsonl: Path,
    output_json: Path,
    eval_scores_jsonl: Path | None = None,
    evidence_features: bool = True,
    evidence_max_per_pair: int = 512,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.monotonic()
    train_rows = _read_jsonl(train_jsonl)
    eval_rows = _read_jsonl(eval_jsonl)
    evidence_report = None
    if evidence_features:
        evidence_report = _add_train_evidence_features(
            train_rows=train_rows,
            eval_rows=eval_rows,
            max_per_pair=evidence_max_per_pair,
        )
    feature_names = _feature_names(train_rows)
    train_data = _dataset(train_rows, feature_names=feature_names)
    eval_data = _dataset(eval_rows, feature_names=feature_names)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=16,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=int(seed),
        verbose=-1,
    )
    model.fit(train_data["x"], train_data["y"], group=train_data["groups"])
    train_scores = model.predict(train_data["x"])
    eval_scores = model.predict(eval_data["x"])
    baseline_scores_eval = np.asarray([_baseline_score(row) for row in eval_data["rows"]], dtype=np.float32)
    baseline_scores_train = np.asarray([_baseline_score(row) for row in train_data["rows"]], dtype=np.float32)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_jsonl": str(train_jsonl),
            "eval_jsonl": str(eval_jsonl),
            "output_json": str(output_json),
            "eval_scores_jsonl": str(eval_scores_jsonl) if eval_scores_jsonl else None,
            "evidence_features": bool(evidence_features),
            "evidence_max_per_pair": evidence_max_per_pair,
            "seed": seed,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": (
                "Transform-pair labels come from v4 split references; no "
                "template outcome similarity is used as feature. Optional "
                "evidence features are computed from train rows only."
            ),
        },
        "evidence_report": evidence_report,
        "feature_names": feature_names,
        "counts": {
            "train_rows": len(train_data["rows"]),
            "eval_rows": len(eval_data["rows"]),
            "train_groups": len(train_data["groups"]),
            "eval_groups": len(eval_data["groups"]),
            "train_positive_rows": int(train_data["y"].sum()),
            "eval_positive_rows": int(eval_data["y"].sum()),
        },
        "baseline_frequency": {
            "train": _evaluate(train_data["rows"], baseline_scores_train),
            "eval": _evaluate(eval_data["rows"], baseline_scores_eval),
        },
        "selector": {
            "train": _evaluate(train_data["rows"], train_scores),
            "eval": _evaluate(eval_data["rows"], eval_scores),
            "feature_importances": _feature_importances(model, feature_names),
        },
        "eval_examples": _examples(eval_data["rows"], eval_scores),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    if eval_scores_jsonl is not None:
        _write_scored_rows(eval_data["rows"], eval_scores, eval_scores_jsonl)
    return report


def _dataset(rows: list[dict[str, Any]], *, feature_names: list[str]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # Group by target + connector so selector learns which transform pair
        # to try for a specific ChemEnzy connector.
        key = f"{row.get('target_smiles')}::{row.get('connector')}::{row.get('downstream_rank')}"
        by_group[key].append(row)
    out_rows = []
    x = []
    y = []
    groups = []
    for _, group_rows in by_group.items():
        if not group_rows:
            continue
        groups.append(len(group_rows))
        for row in group_rows:
            out_rows.append(row)
            x.append([_feature(row, name) for name in feature_names])
            y.append(1 if row.get("label") else 0)
    return {"rows": out_rows, "x": np.asarray(x, dtype=np.float32), "y": np.asarray(y, dtype=np.int32), "groups": groups}


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    pairs = Counter(str(row.get("transform_pair") or "") for row in rows)
    upstream = Counter(str(row.get("upstream_transform") or "") for row in rows)
    downstream = Counter(str(row.get("downstream_transform") or "") for row in rows)
    return [
        "connector_rank_inv_sqrt",
        "downstream_rank_inv_sqrt",
        "pair_freq_rank_inv_sqrt",
        "connector_heavy_atoms_norm",
        "connector_is_main_reactant",
        "candidate_score",
        "has_downstream_transform",
        "evidence_pair_count_log",
        "evidence_best_transition_sim",
        "evidence_mean_top3_transition_sim",
        "evidence_best_connector_sim",
        "evidence_best_target_sim",
        "evidence_best_joint_sim",
    ] + [f"pair={pair}" for pair, _ in pairs.most_common(30)] + [f"upstream={name}" for name, _ in upstream.most_common(12)] + [f"downstream={name}" for name, _ in downstream.most_common(12)]


def _feature(row: dict[str, Any], name: str) -> float:
    if name == "connector_rank_inv_sqrt":
        return 1.0 / math.sqrt(max(1, int(row.get("connector_rank") or 10**6)))
    if name == "downstream_rank_inv_sqrt":
        return 1.0 / math.sqrt(max(1, int(row.get("downstream_rank") or 10**6)))
    if name == "pair_freq_rank_inv_sqrt":
        return 1.0 / math.sqrt(max(1, int(row.get("transform_pair_rank_by_train_frequency") or 10**6)))
    if name == "connector_heavy_atoms_norm":
        return min(1.0, float(row.get("connector_heavy_atoms") or 0.0) / 40.0)
    if name == "connector_is_main_reactant":
        return 1.0 if row.get("connector_is_main_reactant") else 0.0
    if name == "candidate_score":
        try:
            return float(row.get("candidate_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if name == "has_downstream_transform":
        return 1.0 if row.get("downstream_transform") else 0.0
    if name in {
        "evidence_pair_count_log",
        "evidence_best_transition_sim",
        "evidence_mean_top3_transition_sim",
        "evidence_best_connector_sim",
        "evidence_best_target_sim",
        "evidence_best_joint_sim",
    }:
        try:
            return float(row.get(name) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if name.startswith("pair="):
        return 1.0 if str(row.get("transform_pair") or "") == name.split("=", 1)[1] else 0.0
    if name.startswith("upstream="):
        return 1.0 if str(row.get("upstream_transform") or "") == name.split("=", 1)[1] else 0.0
    if name.startswith("downstream="):
        return 1.0 if str(row.get("downstream_transform") or "") == name.split("=", 1)[1] else 0.0
    return 0.0


def _baseline_score(row: dict[str, Any]) -> float:
    return 1.0 / math.sqrt(max(1, int(row.get("transform_pair_rank_by_train_frequency") or 10**6)))


def _evaluate(rows: list[dict[str, Any]], scores: np.ndarray, top_ks: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict[str, Any]:
    by_group: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        key = f"{row.get('target_smiles')}::{row.get('connector')}::{row.get('downstream_rank')}"
        by_group[key].append((row, float(score)))
    ranks = []
    hit_at = {k: 0 for k in top_ks}
    positive_groups = 0
    for _, items in by_group.items():
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        pos = [idx for idx, (row, _) in enumerate(ranked, start=1) if row.get("label")]
        if pos:
            positive_groups += 1
            ranks.append(min(pos))
        for k in top_ks:
            hit_at[k] += int(any(row.get("label") for row, _ in ranked[:k]))
    denom = max(len(by_group), 1)
    return {
        "groups": len(by_group),
        "positive_groups": positive_groups,
        "mrr_all_groups": round(sum(1.0 / rank for rank in ranks) / denom, 6),
        "mrr_positive_groups": round(sum(1.0 / rank for rank in ranks) / max(len(ranks), 1), 6) if ranks else 0.0,
        "mean_best_rank": round(sum(ranks) / max(len(ranks), 1), 6) if ranks else None,
        **{f"hit_at_{k}": round(hit_at[k] / denom, 6) for k in top_ks},
        **{f"hit_at_{k}_positive_groups": round(hit_at[k] / max(positive_groups, 1), 6) for k in top_ks},
    }


def _feature_importances(model: Any, names: list[str]) -> list[dict[str, Any]]:
    rows = [{"feature": name, "importance": int(value)} for name, value in zip(names, getattr(model, "feature_importances_", []))]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows[:30]


def _examples(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    by_group: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        key = f"{row.get('target_smiles')}::{row.get('connector')}::{row.get('downstream_rank')}"
        by_group[key].append((row, float(score)))
    out = []
    for key, items in list(by_group.items())[:10]:
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        out.append(
            {
                "group": key,
                "top": [
                    {
                        "score": round(score, 6),
                        "transform_pair": row.get("transform_pair"),
                        "label": row.get("label"),
                        "freq_rank": row.get("transform_pair_rank_by_train_frequency"),
                    }
                    for row, score in ranked[:5]
                ],
            }
        )
    return out


def _write_scored_rows(rows: list[dict[str, Any]], scores: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row, score in zip(rows, scores):
            out = {
                "target_smiles": row.get("target_smiles"),
                "connector": row.get("connector"),
                "downstream_rank": row.get("downstream_rank"),
                "transform_pair": row.get("transform_pair"),
                "label": bool(row.get("label")),
                "selector_score": float(score),
                "frequency_rank": row.get("transform_pair_rank_by_train_frequency"),
                "baseline_frequency_score": _baseline_score(row),
                "connector_matched_reference_similarity": row.get("connector_matched_reference_similarity"),
                "connector_true_pairs": row.get("connector_true_pairs"),
                "evidence_pair_count_log": row.get("evidence_pair_count_log"),
                "evidence_best_transition_sim": row.get("evidence_best_transition_sim"),
                "evidence_mean_top3_transition_sim": row.get("evidence_mean_top3_transition_sim"),
                "evidence_best_connector_sim": row.get("evidence_best_connector_sim"),
                "evidence_best_target_sim": row.get("evidence_best_target_sim"),
                "evidence_best_joint_sim": row.get("evidence_best_joint_sim"),
            }
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")


def _add_train_evidence_features(
    *,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    max_per_pair: int,
) -> dict[str, Any]:
    evidence_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        if not row.get("label"):
            continue
        pair = str(row.get("transform_pair") or "")
        if not pair:
            continue
        evidence_by_pair[pair].append(
            {
                "group_key": _group_key(row),
                "transition_fp": _transition_fp(row.get("target_smiles"), row.get("connector")),
                "target_fp": _fp(row.get("target_smiles")),
                "connector_fp": _fp(row.get("connector")),
            }
        )
    for pair, rows in list(evidence_by_pair.items()):
        evidence_by_pair[pair] = rows[: max(0, int(max_per_pair))]

    for row in train_rows:
        _set_evidence_features(row, evidence_by_pair)
    for row in eval_rows:
        _set_evidence_features(row, evidence_by_pair)

    counts = {pair: len(rows) for pair, rows in evidence_by_pair.items()}
    return {
        "train_positive_evidence_rows": sum(counts.values()),
        "transform_pairs_with_evidence": len(counts),
        "max_per_pair": max_per_pair,
        "top_evidence_pairs": dict(Counter(counts).most_common(20)),
        "feature_contract": "For each candidate pair, compare target->connector against train positives of the same transform pair; train rows exclude the identical group key.",
    }


def _set_evidence_features(row: dict[str, Any], evidence_by_pair: dict[str, list[dict[str, Any]]]) -> None:
    pair = str(row.get("transform_pair") or "")
    evidences = evidence_by_pair.get(pair) or []
    row_key = _group_key(row)
    transition_fp = _transition_fp(row.get("target_smiles"), row.get("connector"))
    target_fp = _fp(row.get("target_smiles"))
    connector_fp = _fp(row.get("connector"))
    transition_sims = []
    connector_sims = []
    target_sims = []
    joint_sims = []
    for evidence in evidences:
        if evidence.get("group_key") == row_key:
            continue
        transition_sim = _fp_similarity(transition_fp, evidence.get("transition_fp"))
        connector_sim = _fp_similarity(connector_fp, evidence.get("connector_fp"))
        target_sim = _fp_similarity(target_fp, evidence.get("target_fp"))
        joint_sim = 0.60 * transition_sim + 0.30 * connector_sim + 0.10 * target_sim
        transition_sims.append(transition_sim)
        connector_sims.append(connector_sim)
        target_sims.append(target_sim)
        joint_sims.append(joint_sim)
    transition_sims.sort(reverse=True)
    connector_sims.sort(reverse=True)
    target_sims.sort(reverse=True)
    joint_sims.sort(reverse=True)
    row["evidence_pair_count_log"] = round(math.log1p(len(transition_sims)), 6)
    row["evidence_best_transition_sim"] = round(float(transition_sims[0]), 6) if transition_sims else 0.0
    row["evidence_mean_top3_transition_sim"] = round(float(sum(transition_sims[:3]) / min(3, len(transition_sims))), 6) if transition_sims else 0.0
    row["evidence_best_connector_sim"] = round(float(connector_sims[0]), 6) if connector_sims else 0.0
    row["evidence_best_target_sim"] = round(float(target_sims[0]), 6) if target_sims else 0.0
    row["evidence_best_joint_sim"] = round(float(joint_sims[0]), 6) if joint_sims else 0.0


def _group_key(row: dict[str, Any]) -> str:
    return f"{row.get('target_smiles')}::{row.get('connector')}::{row.get('downstream_rank')}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# v4 Transform-Pair Selector", "", "## Counts", "", "```json", json.dumps(report.get("counts") or {}, indent=2, ensure_ascii=False), "```", "", "## Baseline Eval", "", "```json", json.dumps((report.get("baseline_frequency") or {}).get("eval") or {}, indent=2, ensure_ascii=False), "```", "", "## Selector Eval", "", "```json", json.dumps((report.get("selector") or {}).get("eval") or {}, indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v4 transform-pair selector")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--eval-scores-jsonl")
    parser.add_argument("--disable-evidence-features", action="store_true")
    parser.add_argument("--evidence-max-per-pair", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train_v4_transform_pair_selector(
        train_jsonl=Path(args.train_jsonl),
        eval_jsonl=Path(args.eval_jsonl),
        output_json=Path(args.output_json),
        eval_scores_jsonl=Path(args.eval_scores_jsonl) if args.eval_scores_jsonl else None,
        evidence_features=not args.disable_evidence_features,
        evidence_max_per_pair=args.evidence_max_per_pair,
        seed=args.seed,
    )
    print(json.dumps({"counts": report["counts"], "baseline_eval": report["baseline_frequency"]["eval"], "selector_eval": report["selector"]["eval"], "output_json": args.output_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

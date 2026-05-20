"""Train a lightweight CCTS-v3 ranker from cached candidate-evidence rows."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMRanker, early_stopping

from cascade_planner.eval.train_ccts_v0_transition_ranker import CandidateDataset, _baseline_scores, _standardize
from cascade_planner.eval.train_ccts_v2_sparse_labels import _evaluate_sparse_dataset, _rank_delta


SCHEMA_VERSION = "ccts_v3_cached_ranker.v1"


def train_ccts_v3_cached_ranker(
    *,
    train_jsonl: Path,
    val_jsonl: Path,
    test_jsonl: Path,
    output_dir: Path,
    n_estimators: int = 240,
    learning_rate: float = 0.035,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl)
    test_rows = _read_jsonl(test_jsonl)
    feature_names = _feature_names()
    train = _dataset(train_rows, feature_names)
    val = _dataset(val_rows, feature_names)
    test = _dataset(test_rows, feature_names)

    model_specs = _model_specs(feature_names)
    models = {}
    reports = {}
    for name, indices in model_specs.items():
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=15,
            min_child_samples=24,
            subsample=0.80,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            reg_alpha=0.1,
            random_state=int(seed),
            verbose=-1,
        )
        model.fit(
            train.x[:, indices],
            train.y,
            group=train.group_sizes,
            eval_set=[(val.x[:, indices], val.y)],
            eval_group=[val.group_sizes],
            eval_at=[1, 3, 5, 10],
            callbacks=[early_stopping(30, verbose=False)],
        )
        models[name] = model
        reports[name] = {
            "train": _evaluate_sparse_dataset(train, model.predict(train.x[:, indices], num_iteration=model.best_iteration_)),
            "val": _evaluate_sparse_dataset(val, model.predict(val.x[:, indices], num_iteration=model.best_iteration_)),
            "test": _evaluate_sparse_dataset(test, model.predict(test.x[:, indices], num_iteration=model.best_iteration_)),
            "feature_names": [feature_names[idx] for idx in indices],
            "best_iteration": int(model.best_iteration_ or 0),
        }

    baselines = _baseline_reports(train, val, test)
    blends = _blend_reports(val, test, baselines)
    score_columns = {
        "chem_rank": _baseline_scores(test.rows),
        **{f"raw_{name}": _row_score(test.rows, name) for name in _raw_score_names()},
        **{
            name: model.predict(test.x[:, model_specs[name]], num_iteration=model.best_iteration_)
            for name, model in models.items()
        },
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "test_jsonl": str(test_jsonl),
            "output_dir": str(output_dir),
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "seed": seed,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            "train_rows": len(train.rows),
            "val_rows": len(val.rows),
            "test_rows": len(test.rows),
            "train_groups": len(train.group_sizes),
            "val_groups": len(val.group_sizes),
            "test_groups": len(test.group_sizes),
            "train_relevance_rows": int(np.sum(train.y > 0)),
            "val_relevance_rows": int(np.sum(val.y > 0)),
            "test_relevance_rows": int(np.sum(test.y > 0)),
        },
        "baselines": baselines,
        "blends": blends,
        "models": reports,
        "rank_delta": _rank_delta(test, score_columns),
        "decision": _decision(baselines, blends, reports),
        "feature_schema": {
            "feature_names": feature_names,
            "model_specs": {name: [feature_names[idx] for idx in indices] for name, indices in model_specs.items()},
        },
    }
    (output_dir / "ccts_v3_cached_ranker_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "ccts_v3_cached_ranker_report.md").write_text(_markdown(result), encoding="utf-8")
    with (output_dir / "ccts_v3_cached_ranker_models.pkl").open("wb") as fh:
        pickle.dump({"schema_version": SCHEMA_VERSION, "models": models, "feature_schema": result["feature_schema"], "metadata": result["metadata"]}, fh)
    return result


def _feature_names() -> list[str]:
    return [
        "chem__rank",
        "chem__inv_rank",
        "chem__inv_log_rank",
        "chem__score",
        "chem__has_score",
        "chem__n_reactants",
        "candev__nearest_any_transition_sim",
        "candev__nearest_any_product_sim",
        "candev__nearest_any_main_sim",
        "candev__nearest_context_transform_sim",
        "candev__nearest_pair_compatible_sim",
        "candev__inferred_transform_matches_context",
        "candev__inferred_transform_match_score",
        "candev__context_minus_any_transition_sim",
        "candev__pair_minus_context_transform_sim",
        "candev__context_ge_070",
        "candev__context_ge_080",
        "candev__pair_ge_070",
        "candev__inv_rank_x_context_sim",
        "candev__inv_log_rank_x_context_sim",
        "candev__inv_rank_x_inferred_match_score",
    ]


def _feature_row(row: dict[str, Any]) -> list[float]:
    rank = max(1, int(row.get("candidate_rank") or 1))
    score = _float(row.get("candidate_score"))
    any_sim = _float(row.get("candidate_nearest_any_transition_sim"))
    any_product = _float(row.get("candidate_nearest_any_product_sim"))
    any_main = _float(row.get("candidate_nearest_any_main_sim"))
    context_sim = _float(row.get("candidate_nearest_context_transform_sim"))
    pair_sim = _float(row.get("candidate_nearest_pair_compatible_sim"))
    match_score = _float(row.get("candidate_inferred_transform_match_score"))
    inv_rank = 1.0 / rank
    inv_log_rank = 1.0 / np.log2(rank + 1.0)
    return [
        float(rank),
        inv_rank,
        float(inv_log_rank),
        score,
        float(score != 0.0),
        float(len(row.get("candidate_reactants") or [])),
        any_sim,
        any_product,
        any_main,
        context_sim,
        pair_sim,
        float(bool(row.get("candidate_inferred_transform_matches_context"))),
        match_score,
        context_sim - any_sim,
        pair_sim - context_sim,
        float(context_sim >= 0.70),
        float(context_sim >= 0.80),
        float(pair_sim >= 0.70),
        inv_rank * context_sim,
        float(inv_log_rank) * context_sim,
        inv_rank * match_score,
    ]


def _model_specs(feature_names: list[str]) -> dict[str, list[int]]:
    chem = [idx for idx, name in enumerate(feature_names) if name.startswith("chem__")]
    cand = [idx for idx, name in enumerate(feature_names) if name.startswith("candev__")]
    return {
        "chem_only": chem,
        "candidate_evidence_only": cand,
        "chem_plus_candidate_evidence": chem + cand,
    }


def _dataset(rows: list[dict[str, Any]], feature_names: list[str]) -> CandidateDataset:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("transition_id") or ""), []).append(row)
    ordered = []
    group_ids = []
    group_sizes = []
    for group_id, group_rows in by_group.items():
        group_rows = sorted(group_rows, key=lambda row: int(row.get("candidate_rank") or 10**9))
        ordered.extend(group_rows)
        group_ids.append(group_id)
        group_sizes.append(len(group_rows))
    return CandidateDataset(
        rows=ordered,
        x=np.asarray([_feature_row(row) for row in ordered], dtype=np.float32),
        y=np.asarray([int(row.get("training_relevance") or 0) for row in ordered], dtype=np.int32),
        group_sizes=group_sizes,
        group_ids=group_ids,
        feature_names=feature_names,
        chem_feature_indices=[idx for idx, name in enumerate(feature_names) if name.startswith("chem__")],
    )


def _baseline_reports(train: CandidateDataset, val: CandidateDataset, test: CandidateDataset) -> dict[str, Any]:
    out = {
        "chem_rank": {
            "train": _evaluate_sparse_dataset(train, _baseline_scores(train.rows)),
            "val": _evaluate_sparse_dataset(val, _baseline_scores(val.rows)),
            "test": _evaluate_sparse_dataset(test, _baseline_scores(test.rows)),
        }
    }
    for name in _raw_score_names():
        out[name] = {
            "train": _evaluate_sparse_dataset(train, _row_score(train.rows, name)),
            "val": _evaluate_sparse_dataset(val, _row_score(val.rows, name)),
            "test": _evaluate_sparse_dataset(test, _row_score(test.rows, name)),
        }
    return out


def _blend_reports(val: CandidateDataset, test: CandidateDataset, baselines: dict[str, Any]) -> dict[str, Any]:
    chem_val = _baseline_scores(val.rows)
    chem_test = _baseline_scores(test.rows)
    out = {}
    for name in _raw_score_names():
        aux_val = _row_score(val.rows, name)
        aux_test = _row_score(test.rows, name)
        best_alpha = 0.0
        best_score = _selection_score(_evaluate_sparse_dataset(val, chem_val))
        for alpha in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0]:
            score = _selection_score(_evaluate_sparse_dataset(val, _standardize(chem_val) + alpha * _standardize(aux_val)))
            if score > best_score:
                best_alpha = float(alpha)
                best_score = score
        out[f"chem_rank_plus_{name}"] = {
            "alpha_selected_on_val": best_alpha,
            "val_selection_score": round(float(best_score), 6),
            "test": _evaluate_sparse_dataset(test, _standardize(chem_test) + best_alpha * _standardize(aux_test)),
        }
    return out


def _selection_score(report: dict[str, Any]) -> float:
    block = report.get("block_supported_positive_label") or {}
    exact = report.get("exact_label") or {}
    block_k = block.get("recall_at_k_all") or {}
    exact_k = exact.get("recall_at_k_all") or {}
    return float(block.get("mrr_covered") or 0.0) + 0.8 * float(block_k.get("5") or 0.0) + 0.4 * float(exact.get("mrr_covered") or 0.0) + 0.3 * float(exact_k.get("5") or 0.0)


def _decision(baselines: dict[str, Any], blends: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    label = "block_supported_positive_label"
    chem = float((((baselines.get("chem_rank") or {}).get("test") or {}).get(label) or {}).get("mrr_covered") or 0.0)
    candidates = {}
    for source in (baselines, blends, models):
        for name, payload in source.items():
            if name == "chem_rank":
                continue
            test = payload.get("test") or {}
            candidates[name] = float(((test.get(label) or {}).get("mrr_covered")) or 0.0)
    best_name, best = max(candidates.items(), key=lambda kv: kv[1]) if candidates else ("", 0.0)
    return {
        "primary_label": label,
        "chem_rank_mrr": round(chem, 6),
        "best_method": best_name,
        "best_method_mrr": round(best, 6),
        "delta_vs_chem_rank": round(best - chem, 6),
        "candidate_evidence_promotable": bool(best - chem >= 0.03),
    }


def _raw_score_names() -> list[str]:
    return [
        "candidate_nearest_context_transform_sim",
        "candidate_inferred_transform_match_score",
        "candidate_nearest_pair_compatible_sim",
    ]


def _row_score(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_float(row.get(key)) for row in rows], dtype=np.float32)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CCTS-v3 Cached Ranker",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(result.get("decision") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Metrics",
        "",
        "| Method | Label | Coverage | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for group_name in ("baselines", "blends", "models"):
        for name, payload in (result.get(group_name) or {}).items():
            rows.append((name, payload.get("test") or {}))
    for name, metrics in rows:
        for label in ("training_label", "block_supported_positive_label", "block_supported_exact_label", "exact_label", "positive_label"):
            metric = metrics.get(label) or {}
            at = metric.get("recall_at_k_all") or {}
            lines.append("| " + " | ".join([name, label, str(metric.get("coverage")), str(metric.get("mrr_covered")), str(at.get("1")), str(at.get("3")), str(at.get("5")), str(at.get("10"))]) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CCTS-v3 cached candidate-evidence ranker")
    ap.add_argument("--train-jsonl", default="results/shared/cascadebench_strict_20260516/ccts_v3_candidate_cache/train_candidates.jsonl")
    ap.add_argument("--val-jsonl", default="results/shared/cascadebench_strict_20260516/ccts_v3_candidate_cache/val_candidates.jsonl")
    ap.add_argument("--test-jsonl", default="results/shared/cascadebench_strict_20260516/ccts_v3_candidate_cache/test_candidates.jsonl")
    ap.add_argument("--output-dir", default="results/shared/cascadebench_strict_20260516/ccts_v3_cached_ranker")
    ap.add_argument("--n-estimators", type=int, default=240)
    ap.add_argument("--learning-rate", type=float, default=0.035)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    result = train_ccts_v3_cached_ranker(
        train_jsonl=Path(args.train_jsonl),
        val_jsonl=Path(args.val_jsonl),
        test_jsonl=Path(args.test_jsonl),
        output_dir=Path(args.output_dir),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps({"decision": result["decision"], "outputs": {"report": str(Path(args.output_dir) / "ccts_v3_cached_ranker_report.json"), "markdown": str(Path(args.output_dir) / "ccts_v3_cached_ranker_report.md")}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Train CCTS-v3 as a conservative pairwise residual ranker.

This experiment keeps ChemEnzy as the fixed generator and trains only the
candidate ordering layer.  It intentionally excludes direct GT/reactant
similarity features and uses validation-only model/blend selection.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from cascade_planner.eval.train_ccts_v0_transition_ranker import CandidateDataset, _baseline_scores, _standardize
from cascade_planner.eval.train_ccts_v2_sparse_labels import _evaluate_sparse_dataset, _rank_delta


SCHEMA_VERSION = "ccts_v3_pairwise_residual_ranker.v1"
DEFAULT_CACHE_DIR = Path("results/shared/cascadebench_strict_20260516/ccts_v3_candidate_cache")


@dataclass
class FittedPairwiseModel:
    model: LogisticRegression
    feature_indices: list[int]
    mean: np.ndarray
    std: np.ndarray
    train_label_key: str
    c_value: float


def train_ccts_v3_pairwise_residual_ranker(
    *,
    train_jsonl: Path,
    val_jsonl: Path,
    test_jsonl: Path,
    output_dir: Path,
    c_values: list[float] | None = None,
    max_pos_per_group: int = 4,
    max_neg_per_pos: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    c_values = c_values or [0.03, 0.1, 0.3, 1.0]

    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl)
    test_rows = _read_jsonl(test_jsonl)
    feature_names = _feature_names()
    train = _dataset(train_rows, feature_names)
    val = _dataset(val_rows, feature_names)
    test = _dataset(test_rows, feature_names)

    baselines = _baseline_reports(train, val, test)
    nonlearned_blends = _nonlearned_blend_reports(val, test)
    model_specs = _model_specs(feature_names)
    label_keys = ["training_relevance", "block_supported_positive_label"]

    fitted: dict[str, FittedPairwiseModel] = {}
    reports: dict[str, Any] = {}
    residual_blends: dict[str, Any] = {}
    rng = np.random.default_rng(int(seed))

    for label_key in label_keys:
        for spec_name, feature_indices in model_specs.items():
            train_x, train_y = _pairwise_training_matrix(
                train,
                feature_indices=feature_indices,
                label_key=label_key,
                max_pos_per_group=max_pos_per_group,
                max_neg_per_pos=max_neg_per_pos,
                rng=rng,
            )
            if train_x.shape[0] == 0:
                continue
            mean, std = _candidate_scaler(train.x[:, feature_indices])
            train_pairs = _scale_pairwise_matrix(train_x, std)
            best = None
            best_val_score = -1e9
            best_payload: dict[str, Any] | None = None
            for c_value in c_values:
                model = LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    solver="liblinear",
                    max_iter=500,
                    random_state=int(seed),
                )
                model.fit(train_pairs, train_y)
                tmp = FittedPairwiseModel(
                    model=model,
                    feature_indices=feature_indices,
                    mean=mean,
                    std=std,
                    train_label_key=label_key,
                    c_value=float(c_value),
                )
                val_scores = _model_scores(tmp, val)
                val_report = _evaluate_sparse_dataset(val, val_scores)
                score = _selection_score(val_report)
                if score > best_val_score:
                    best_val_score = score
                    best = tmp
                    best_payload = {
                        "c_value": float(c_value),
                        "val_selection_score": round(float(score), 6),
                        "pair_rows": int(train_pairs.shape[0]),
                    }
            if best is None or best_payload is None:
                continue
            name = f"pairwise_{label_key}__{spec_name}"
            fitted[name] = best
            train_scores = _model_scores(best, train)
            val_scores = _model_scores(best, val)
            test_scores = _model_scores(best, test)
            reports[name] = {
                **best_payload,
                "train": _evaluate_sparse_dataset(train, train_scores),
                "val": _evaluate_sparse_dataset(val, val_scores),
                "test": _evaluate_sparse_dataset(test, test_scores),
                "feature_names": [feature_names[idx] for idx in feature_indices],
                "coef": [round(float(x), 6) for x in best.model.coef_[0].tolist()],
            }
            residual_blends.update(_residual_blend_reports(name, val, test, val_scores, test_scores))

    method_scores = _all_test_scores(test, fitted)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "test_jsonl": str(test_jsonl),
            "output_dir": str(output_dir),
            "c_values": [float(x) for x in c_values],
            "max_pos_per_group": int(max_pos_per_group),
            "max_neg_per_pos": int(max_neg_per_pos),
            "seed": int(seed),
            "elapsed_s": round(time.monotonic() - started, 3),
            "leakage_excluded_features": ["reactant_similarity"],
            "selection_rule": "select method/blend by validation selection score; test is held out",
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
        "nonlearned_blends": nonlearned_blends,
        "models": reports,
        "residual_blends": residual_blends,
        "selection": _select_by_val(baselines, nonlearned_blends, reports, residual_blends),
        "rank_delta": _rank_delta(test, method_scores),
        "feature_schema": {
            "feature_names": feature_names,
            "model_specs": {name: [feature_names[idx] for idx in indices] for name, indices in model_specs.items()},
        },
    }
    (output_dir / "ccts_v3_pairwise_residual_ranker_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "ccts_v3_pairwise_residual_ranker_report.md").write_text(_markdown(result), encoding="utf-8")
    with (output_dir / "ccts_v3_pairwise_residual_ranker_models.pkl").open("wb") as fh:
        pickle.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "models": fitted,
                "feature_schema": result["feature_schema"],
                "selection": result["selection"],
                "metadata": result["metadata"],
            },
            fh,
        )
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
        "candev__score_x_context_sim",
        "candev__score_x_inferred_match_score",
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
        score * context_sim,
        score * match_score,
    ]


def _model_specs(feature_names: list[str]) -> dict[str, list[int]]:
    chem = [idx for idx, name in enumerate(feature_names) if name.startswith("chem__")]
    cand = [idx for idx, name in enumerate(feature_names) if name.startswith("candev__")]
    return {
        "evidence_only": cand,
        "chem_plus_evidence": chem + cand,
    }


def _dataset(rows: list[dict[str, Any]], feature_names: list[str]) -> CandidateDataset:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("transition_id") or ""), []).append(row)
    ordered: list[dict[str, Any]] = []
    group_ids: list[str] = []
    group_sizes: list[int] = []
    for group_id in sorted(by_group):
        group_rows = sorted(by_group[group_id], key=lambda row: int(row.get("candidate_rank") or 10**9))
        ordered.extend(group_rows)
        group_ids.append(group_id)
        group_sizes.append(len(group_rows))
    return CandidateDataset(
        rows=ordered,
        x=np.asarray([_feature_row(row) for row in ordered], dtype=np.float32),
        y=np.asarray([int(bool(row.get("training_relevance"))) for row in ordered], dtype=np.int32),
        group_sizes=group_sizes,
        group_ids=group_ids,
        feature_names=feature_names,
        chem_feature_indices=[idx for idx, name in enumerate(feature_names) if name.startswith("chem__")],
    )


def _pairwise_training_matrix(
    dataset: CandidateDataset,
    *,
    feature_indices: list[int],
    label_key: str,
    max_pos_per_group: int,
    max_neg_per_pos: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    diffs: list[np.ndarray] = []
    labels: list[int] = []
    offset = 0
    for group_size in dataset.group_sizes:
        rows = dataset.rows[offset : offset + group_size]
        x = dataset.x[offset : offset + group_size, feature_indices]
        positives = [idx for idx, row in enumerate(rows) if _is_positive(row, label_key)]
        negatives = [idx for idx, row in enumerate(rows) if not _is_positive(row, label_key)]
        if positives and negatives:
            positives = sorted(
                positives,
                key=lambda idx: (
                    -float(rows[idx].get("block_supported_exact_label") is True),
                    -float(rows[idx].get("exact_label") is True),
                    int(rows[idx].get("candidate_rank") or 10**9),
                ),
            )[: max(1, int(max_pos_per_group))]
            hard_negatives = _hard_negative_indices(rows, negatives, max_neg_per_pos=max_neg_per_pos)
            for pos_idx in positives:
                for neg_idx in hard_negatives:
                    diff = x[pos_idx] - x[neg_idx]
                    diffs.append(diff)
                    labels.append(1)
                    diffs.append(-diff)
                    labels.append(0)
        offset += group_size
    if not diffs:
        return np.zeros((0, len(feature_indices)), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    order = rng.permutation(len(diffs))
    return np.asarray(diffs, dtype=np.float32)[order], np.asarray(labels, dtype=np.int32)[order]


def _hard_negative_indices(rows: list[dict[str, Any]], negatives: list[int], *, max_neg_per_pos: int) -> list[int]:
    by_rank = sorted(negatives, key=lambda idx: int(rows[idx].get("candidate_rank") or 10**9))[: max_neg_per_pos]
    by_context = sorted(
        negatives,
        key=lambda idx: (
            -_float(rows[idx].get("candidate_nearest_context_transform_sim")),
            -_float(rows[idx].get("candidate_inferred_transform_match_score")),
            int(rows[idx].get("candidate_rank") or 10**9),
        ),
    )[: max_neg_per_pos]
    chosen: list[int] = []
    for idx in by_rank + by_context:
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= max_neg_per_pos:
            break
    return chosen


def _candidate_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _scale_pairwise_matrix(x_diff: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x_diff / std


def _model_scores(fitted: FittedPairwiseModel, dataset: CandidateDataset) -> np.ndarray:
    x = dataset.x[:, fitted.feature_indices]
    scaled = (x - fitted.mean) / fitted.std
    return fitted.model.decision_function(scaled).astype(np.float32)


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


def _nonlearned_blend_reports(val: CandidateDataset, test: CandidateDataset) -> dict[str, Any]:
    out: dict[str, Any] = {}
    bases = {"chem_rank": _baseline_scores(val.rows)}
    test_bases = {"chem_rank": _baseline_scores(test.rows)}
    for raw_name in _raw_score_names():
        aux_val = _row_score(val.rows, raw_name)
        aux_test = _row_score(test.rows, raw_name)
        name, payload = _select_blend(
            val,
            test,
            base_val=bases["chem_rank"],
            base_test=test_bases["chem_rank"],
            aux_val=aux_val,
            aux_test=aux_test,
            name=f"chem_rank_plus_{raw_name}",
        )
        out[name] = payload
    return out


def _residual_blend_reports(
    model_name: str,
    val: CandidateDataset,
    test: CandidateDataset,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base_scores = {
        "chem_rank": (_baseline_scores(val.rows), _baseline_scores(test.rows)),
        "raw_context": (
            _row_score(val.rows, "candidate_nearest_context_transform_sim"),
            _row_score(test.rows, "candidate_nearest_context_transform_sim"),
        ),
        "chem_plus_raw_context": _best_two_signal_blend(
            val,
            test,
            _baseline_scores(val.rows),
            _baseline_scores(test.rows),
            _row_score(val.rows, "candidate_nearest_context_transform_sim"),
            _row_score(test.rows, "candidate_nearest_context_transform_sim"),
        ),
    }
    for base_name, (base_val, base_test) in base_scores.items():
        name, payload = _select_blend(
            val,
            test,
            base_val=base_val,
            base_test=base_test,
            aux_val=val_scores,
            aux_test=test_scores,
            name=f"{base_name}_plus_{model_name}",
        )
        out[name] = payload
    return out


def _best_two_signal_blend(
    val: CandidateDataset,
    test: CandidateDataset,
    base_val: np.ndarray,
    base_test: np.ndarray,
    aux_val: np.ndarray,
    aux_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    best_alpha = 0.0
    best_score = _selection_score(_evaluate_sparse_dataset(val, _standardize(base_val)))
    for alpha in _alpha_grid():
        scores = _standardize(base_val) + alpha * _standardize(aux_val)
        score = _selection_score(_evaluate_sparse_dataset(val, scores))
        if score > best_score:
            best_alpha = float(alpha)
            best_score = score
    return _standardize(base_val) + best_alpha * _standardize(aux_val), _standardize(base_test) + best_alpha * _standardize(aux_test)


def _select_blend(
    val: CandidateDataset,
    test: CandidateDataset,
    *,
    base_val: np.ndarray,
    base_test: np.ndarray,
    aux_val: np.ndarray,
    aux_test: np.ndarray,
    name: str,
) -> tuple[str, dict[str, Any]]:
    best_alpha = 0.0
    best_val_scores = _standardize(base_val)
    best_score = _selection_score(_evaluate_sparse_dataset(val, best_val_scores))
    for alpha in _alpha_grid():
        candidate_val_scores = _standardize(base_val) + alpha * _standardize(aux_val)
        score = _selection_score(_evaluate_sparse_dataset(val, candidate_val_scores))
        if score > best_score:
            best_alpha = float(alpha)
            best_score = score
            best_val_scores = candidate_val_scores
    test_scores = _standardize(base_test) + best_alpha * _standardize(aux_test)
    return name, {
        "alpha_selected_on_val": float(best_alpha),
        "val_selection_score": round(float(best_score), 6),
        "val": _evaluate_sparse_dataset(val, best_val_scores),
        "test": _evaluate_sparse_dataset(test, test_scores),
    }


def _all_test_scores(test: CandidateDataset, fitted: dict[str, FittedPairwiseModel]) -> dict[str, np.ndarray]:
    scores = {
        "chem_rank": _baseline_scores(test.rows),
        **{f"raw_{name}": _row_score(test.rows, name) for name in _raw_score_names()},
    }
    for name, model in fitted.items():
        scores[name] = _model_scores(model, test)
    return scores


def _select_by_val(
    baselines: dict[str, Any],
    nonlearned_blends: dict[str, Any],
    models: dict[str, Any],
    residual_blends: dict[str, Any],
) -> dict[str, Any]:
    candidates: list[tuple[str, str, float, dict[str, Any]]] = []
    for family, mapping in (
        ("baseline", baselines),
        ("nonlearned_blend", nonlearned_blends),
        ("model", models),
        ("residual_blend", residual_blends),
    ):
        for name, payload in mapping.items():
            val_report = payload.get("val") or {}
            score = _selection_score(val_report)
            candidates.append((family, name, score, payload))
    candidates.sort(key=lambda row: row[2], reverse=True)
    best_family, best_name, best_score, best_payload = candidates[0]
    chem_test = ((baselines.get("chem_rank") or {}).get("test") or {}).get("block_supported_positive_label") or {}
    best_test = (best_payload.get("test") or {}).get("block_supported_positive_label") or {}
    return {
        "selected_family": best_family,
        "selected_method": best_name,
        "selected_val_score": round(float(best_score), 6),
        "selected_test_block_mrr": best_test.get("mrr_covered"),
        "chem_test_block_mrr": chem_test.get("mrr_covered"),
        "selected_delta_vs_chem_block_mrr": round(float(best_test.get("mrr_covered") or 0.0) - float(chem_test.get("mrr_covered") or 0.0), 6),
        "top_val_methods": [
            {
                "family": family,
                "method": name,
                "val_selection_score": round(float(score), 6),
                "test_block_mrr": ((payload.get("test") or {}).get("block_supported_positive_label") or {}).get("mrr_covered"),
                "test_exact_mrr": ((payload.get("test") or {}).get("exact_label") or {}).get("mrr_covered"),
            }
            for family, name, score, payload in candidates[:12]
        ],
    }


def _selection_score(report: dict[str, Any]) -> float:
    block = report.get("block_supported_positive_label") or {}
    exact = report.get("exact_label") or {}
    block_k = block.get("recall_at_k_all") or {}
    exact_k = exact.get("recall_at_k_all") or {}
    return (
        float(block.get("mrr_covered") or 0.0)
        + 0.8 * float(block_k.get("5") or 0.0)
        + 0.4 * float(exact.get("mrr_covered") or 0.0)
        + 0.3 * float(exact_k.get("5") or 0.0)
    )


def _raw_score_names() -> list[str]:
    return [
        "candidate_nearest_context_transform_sim",
        "candidate_inferred_transform_match_score",
        "candidate_nearest_pair_compatible_sim",
        "candidate_nearest_any_transition_sim",
    ]


def _row_score(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_float(row.get(key)) for row in rows], dtype=np.float32)


def _is_positive(row: dict[str, Any], label_key: str) -> bool:
    return bool(row.get(label_key))


def _alpha_grid() -> list[float]:
    return [-2.0, -1.5, -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CCTS-v3 Pairwise Residual Ranker",
        "",
        "## Selection",
        "",
        "```json",
        json.dumps(result.get("selection") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Metrics",
        "",
        "| Family | Method | Label | Coverage | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for family in ("baselines", "nonlearned_blends", "models", "residual_blends"):
        for name, payload in (result.get(family) or {}).items():
            rows.append((family, name, payload.get("test") or {}))
    for family, name, metrics in rows:
        for label in ("training_label", "block_supported_positive_label", "block_supported_exact_label", "exact_label", "positive_label"):
            metric = metrics.get(label) or {}
            at = metric.get("recall_at_k_all") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        family,
                        name,
                        label,
                        str(metric.get("coverage")),
                        str(metric.get("mrr_covered")),
                        str(at.get("1")),
                        str(at.get("3")),
                        str(at.get("5")),
                        str(at.get("10")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CCTS-v3 pairwise residual ranker from cached ChemEnzy candidates")
    ap.add_argument("--train-jsonl", default=str(DEFAULT_CACHE_DIR / "train_candidates.jsonl"))
    ap.add_argument("--val-jsonl", default=str(DEFAULT_CACHE_DIR / "val_candidates.jsonl"))
    ap.add_argument("--test-jsonl", default=str(DEFAULT_CACHE_DIR / "test_candidates.jsonl"))
    ap.add_argument("--output-dir", default="results/shared/cascadebench_strict_20260516/ccts_v3_pairwise_residual_ranker")
    ap.add_argument("--c-values", default="0.03,0.1,0.3,1.0")
    ap.add_argument("--max-pos-per-group", type=int, default=4)
    ap.add_argument("--max-neg-per-pos", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    result = train_ccts_v3_pairwise_residual_ranker(
        train_jsonl=Path(args.train_jsonl),
        val_jsonl=Path(args.val_jsonl),
        test_jsonl=Path(args.test_jsonl),
        output_dir=Path(args.output_dir),
        c_values=[float(item) for item in str(args.c_values).split(",") if item.strip()],
        max_pos_per_group=args.max_pos_per_group,
        max_neg_per_pos=args.max_neg_per_pos,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "outputs": {
                    "report": str(Path(args.output_dir) / "ccts_v3_pairwise_residual_ranker_report.json"),
                    "markdown": str(Path(args.output_dir) / "ccts_v3_pairwise_residual_ranker_report.md"),
                    "model": str(Path(args.output_dir) / "ccts_v3_pairwise_residual_ranker_models.pkl"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

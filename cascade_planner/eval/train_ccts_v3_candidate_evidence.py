"""Train CCTS-v3 with candidate-specific cascade evidence features.

CCTS-v2 showed that generic route/program context is too weak: it often does
not look closely enough at the actual ChemEnzy candidate transition.  This
variant keeps the same strict CascadeBench split and fixed ChemEnzy top-100
candidate pool, but adds train-split evidence retrieval features for each
candidate reactant-product transform.

This is still a same-pool transition ranker.  It does not change the planner or
search trajectory; the gate is whether candidate-specific cascade evidence can
beat ChemEnzy ranking on held-out v4 transitions without leakage.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import RDLogger

from cascade_planner.eval.audit_candidate_specific_evidence import (
    _candidate_evidence_scores,
    _train_bank,
)
from cascade_planner.eval.train_ccts_v0_transition_ranker import (
    CandidateDataset,
    _baseline_scores,
    _candidate_rows_from_cache,
    _metric_for_selection,
    _read_json,
    _standardize,
    _write_jsonl,
)
from cascade_planner.eval.train_ccts_v1_transition_ranker import _coverage_leakage_report
from cascade_planner.eval.train_ccts_v2_sparse_labels import (
    _add_block_support,
    _evaluate_sparse_dataset,
    _model_report,
    _neighbor_context,
    _rank_delta,
    _training_relevance,
)
from cascade_planner.eval.train_ccts_v2_transition_ranker import (
    _build_schema,
    _candidate_label_row,
    _feature_names,
    _feature_vector,
    _fit_ranker,
    _load_program_evidence,
    _transition_context,
)


SCHEMA_VERSION = "ccts_v3_candidate_evidence.v1"
LABEL_MODES = {"binary_block_exact", "graded_evidence"}


def train_ccts_v3_candidate_evidence(
    *,
    train_coverage: Path,
    train_cache: Path,
    val_coverage: Path,
    val_cache: Path,
    test_coverage: Path,
    test_cache: Path,
    program_manifest: Path,
    output_dir: Path,
    label_mode: str = "binary_block_exact",
    similarity_threshold: float = 0.70,
    adjacency_similarity_threshold: float = 0.70,
    max_candidates_per_transition: int = 100,
    n_estimators: int = 360,
    learning_rate: float = 0.035,
    seed: int = 42,
    model_set: str = "fast",
) -> dict[str, Any]:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {sorted(LABEL_MODES)}")
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_payload = _read_json(train_coverage)
    val_payload = _read_json(val_coverage)
    test_payload = _read_json(test_coverage)
    train_transitions = [row for row in train_payload.get("transitions") or [] if isinstance(row, dict)]
    val_transitions = [row for row in val_payload.get("transitions") or [] if isinstance(row, dict)]
    test_transitions = [row for row in test_payload.get("transitions") or [] if isinstance(row, dict)]
    train_cache_rows = _read_json(train_cache)
    val_cache_rows = _read_json(val_cache)
    test_cache_rows = _read_json(test_cache)

    program_evidence = _load_program_evidence(program_manifest)
    neighbor_context = _neighbor_context(program_manifest)
    train_evidence_bank = _train_bank(program_manifest)
    if model_set == "raw":
        base_schema: dict[str, Any] = {}
        feature_names = _candidate_evidence_feature_names()
    else:
        base_schema = _build_schema(train_transitions, train_cache_rows, program_evidence)
        feature_names = _feature_names(base_schema) + _candidate_evidence_feature_names()

    train_data = _build_v3_dataset(
        train_transitions,
        train_cache_rows,
        evidence=program_evidence,
        neighbor_context=neighbor_context,
        train_evidence_bank=train_evidence_bank,
        base_schema=base_schema,
        feature_names=feature_names,
        include_full_features=model_set != "raw",
        label_mode=label_mode,
        similarity_threshold=similarity_threshold,
        adjacency_similarity_threshold=adjacency_similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        require_trainable_group=True,
    )
    val_data = _build_v3_dataset(
        val_transitions,
        val_cache_rows,
        evidence=program_evidence,
        neighbor_context=neighbor_context,
        train_evidence_bank=train_evidence_bank,
        base_schema=base_schema,
        feature_names=feature_names,
        include_full_features=model_set != "raw",
        label_mode=label_mode,
        similarity_threshold=similarity_threshold,
        adjacency_similarity_threshold=adjacency_similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        require_trainable_group=False,
    )
    test_data = _build_v3_dataset(
        test_transitions,
        test_cache_rows,
        evidence=program_evidence,
        neighbor_context=neighbor_context,
        train_evidence_bank=train_evidence_bank,
        base_schema=base_schema,
        feature_names=feature_names,
        include_full_features=model_set != "raw",
        label_mode=label_mode,
        similarity_threshold=similarity_threshold,
        adjacency_similarity_threshold=adjacency_similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        require_trainable_group=False,
    )
    if not train_data.rows or not val_data.rows or not test_data.rows:
        raise ValueError("CCTS-v3 requires non-empty train/val/test candidate datasets")

    model_specs = _model_specs_v3(train_data.feature_names, model_set=model_set)
    models: dict[str, Any] = {}
    model_reports: dict[str, Any] = {}

    baseline_scores = {
        "chem_rank": {
            "train": _baseline_scores(train_data.rows),
            "val": _baseline_scores(val_data.rows),
            "test": _baseline_scores(test_data.rows),
        },
        "candidate_nearest_context_transform_sim": {
            "train": _row_score(train_data.rows, "candidate_nearest_context_transform_sim"),
            "val": _row_score(val_data.rows, "candidate_nearest_context_transform_sim"),
            "test": _row_score(test_data.rows, "candidate_nearest_context_transform_sim"),
        },
        "candidate_inferred_transform_match_score": {
            "train": _row_score(train_data.rows, "candidate_inferred_transform_match_score"),
            "val": _row_score(val_data.rows, "candidate_inferred_transform_match_score"),
            "test": _row_score(test_data.rows, "candidate_inferred_transform_match_score"),
        },
        "candidate_nearest_pair_compatible_sim": {
            "train": _row_score(train_data.rows, "candidate_nearest_pair_compatible_sim"),
            "val": _row_score(val_data.rows, "candidate_nearest_pair_compatible_sim"),
            "test": _row_score(test_data.rows, "candidate_nearest_pair_compatible_sim"),
        },
    }
    baseline_reports = {
        name: {
            "train": _evaluate_sparse_dataset(train_data, scores["train"]),
            "val": _evaluate_sparse_dataset(val_data, scores["val"]),
            "test": _evaluate_sparse_dataset(test_data, scores["test"]),
        }
        for name, scores in baseline_scores.items()
    }
    raw_blends = _raw_blend_reports(train_data, val_data, test_data, baseline_scores)
    if model_set != "raw":
        for name, indices in model_specs.items():
            if not indices:
                continue
            model = _fit_ranker(
                train_data,
                val_data,
                feature_indices=indices,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                seed=seed,
            )
            models[name] = model
            model_reports[name] = _model_report(
                model,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                feature_indices=indices,
            )
    model_score_columns = {
        "chem_rank": baseline_scores["chem_rank"]["test"],
        **{f"raw_{name}": payload["test"] for name, payload in baseline_scores.items() if name != "chem_rank"},
    }
    if models:
        model_score_columns.update(
            {
                name: model.predict(test_data.x[:, model_specs[name]], num_iteration=getattr(model, "best_iteration_", None))
                for name, model in models.items()
            }
        )
    leakage = _coverage_leakage_report({"train": train_transitions, "val": val_transitions, "test": test_transitions})
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_coverage": str(train_coverage),
            "val_coverage": str(val_coverage),
            "test_coverage": str(test_coverage),
            "program_manifest": str(program_manifest),
            "output_dir": str(output_dir),
            "label_mode": label_mode,
            "similarity_threshold": similarity_threshold,
            "adjacency_similarity_threshold": adjacency_similarity_threshold,
            "max_candidates_per_transition": max_candidates_per_transition,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "seed": seed,
            "model_set": model_set,
            "elapsed_s": round(time.monotonic() - started, 3),
            "train_evidence_bank_count": train_evidence_bank.get("count"),
        },
        "counts": {
            "train_transitions": len(train_transitions),
            "val_transitions": len(val_transitions),
            "test_transitions": len(test_transitions),
            "train_candidate_rows": len(train_data.rows),
            "val_candidate_rows": len(val_data.rows),
            "test_candidate_rows": len(test_data.rows),
            "train_groups": len(train_data.group_sizes),
            "val_groups": len(val_data.group_sizes),
            "test_groups": len(test_data.group_sizes),
            "train_relevance_rows": int(np.sum(train_data.y > 0)),
            "val_relevance_rows": int(np.sum(val_data.y > 0)),
            "test_relevance_rows": int(np.sum(test_data.y > 0)),
            "train_block_supported_rows": sum(1 for row in train_data.rows if row.get("block_supported_positive_label")),
            "val_block_supported_rows": sum(1 for row in val_data.rows if row.get("block_supported_positive_label")),
            "test_block_supported_rows": sum(1 for row in test_data.rows if row.get("block_supported_positive_label")),
        },
        "leakage_checks": leakage,
        "baselines": baseline_reports,
        "raw_blends": raw_blends,
        "models": model_reports,
        "rank_delta": _rank_delta(test_data, model_score_columns),
        "decision": _decision(baseline_reports, raw_blends, model_reports),
        "feature_schema": {
            "feature_names": train_data.feature_names,
            "model_specs": {name: [train_data.feature_names[idx] for idx in indices] for name, indices in model_specs.items()},
            **base_schema,
        },
    }
    if models:
        with (output_dir / "ccts_v3_candidate_evidence_models.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "models": models,
                    "feature_schema": result["feature_schema"],
                    "metadata": result["metadata"],
                },
                fh,
            )
    (output_dir / "ccts_v3_candidate_evidence_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "ccts_v3_candidate_evidence_report.md").write_text(_markdown(result), encoding="utf-8")
    _write_jsonl(output_dir / "ccts_v3_candidate_evidence_test_candidates.jsonl", _compact_rows(test_data.rows))
    return result


def _build_v3_dataset(
    transitions: list[dict[str, Any]],
    cache: dict[str, Any],
    *,
    evidence: Any,
    neighbor_context: dict[str, dict[str, Any]],
    train_evidence_bank: dict[str, Any],
    base_schema: dict[str, Any],
    feature_names: list[str],
    include_full_features: bool,
    label_mode: str,
    similarity_threshold: float,
    adjacency_similarity_threshold: float,
    max_candidates_per_transition: int,
    require_trainable_group: bool,
) -> CandidateDataset:
    rows: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    group_sizes: list[int] = []
    group_ids: list[str] = []
    product_sim_cache: dict[tuple[str, str], list[float]] = {}
    for transition in transitions:
        product = str(transition.get("product_smiles") or "")
        candidates = _candidate_rows_from_cache(cache, product)[: int(max_candidates_per_transition)]
        context = _transition_context(transition, evidence)
        neighbor = neighbor_context.get(str(transition.get("transition_id") or "")) or {}
        group_rows = []
        group_x = []
        group_y = []
        for rank, candidate in enumerate(candidates, start=1):
            row = _candidate_label_row(transition, candidate, context=context, rank=rank, similarity_threshold=similarity_threshold)
            row.update(
                {
                    "transform": context.get("transform") or "",
                    "previous_transform": context.get("previous_transform") or "",
                    "next_transform": context.get("next_transform") or "",
                    "step_mode": context.get("step_mode") or "",
                    "pairwise_mode": context.get("pairwise_mode") or "",
                    "compatibility_label": context.get("compatibility_label") or "",
                }
            )
            _add_block_support(row, neighbor=neighbor, adjacency_similarity_threshold=adjacency_similarity_threshold)
            row.update(
                _candidate_evidence_scores(
                    product=product,
                    candidate_main=str(row.get("candidate_main_reactant") or ""),
                    context_transform=str(context.get("transform") or ""),
                    previous_transform=str(context.get("previous_transform") or ""),
                    next_transform=str(context.get("next_transform") or ""),
                    train_bank=train_evidence_bank,
                    product_sim_cache=product_sim_cache,
                )
            )
            row["training_relevance"] = _training_relevance(row, label_mode=label_mode)
            x = []
            if include_full_features:
                x = _feature_vector(row, context=context, evidence=evidence, schema=base_schema)
            x.extend(_candidate_evidence_feature_vector(row))
            group_rows.append(row)
            group_x.append(x)
            group_y.append(int(row["training_relevance"]))
        if not group_rows:
            continue
        if require_trainable_group and len(set(group_y)) <= 1:
            continue
        rows.extend(group_rows)
        x_rows.extend(group_x)
        y_rows.extend(group_y)
        group_sizes.append(len(group_rows))
        group_ids.append(str(transition.get("transition_id") or ""))
    chem_indices = [idx for idx, name in enumerate(feature_names) if name.startswith("chem__")]
    return CandidateDataset(
        rows=rows,
        x=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.int32),
        group_sizes=group_sizes,
        group_ids=group_ids,
        feature_names=feature_names,
        chem_feature_indices=chem_indices,
    )


def _candidate_evidence_feature_names() -> list[str]:
    return [
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


def _candidate_evidence_feature_vector(row: dict[str, Any]) -> list[float]:
    any_sim = float(row.get("candidate_nearest_any_transition_sim") or 0.0)
    any_product = float(row.get("candidate_nearest_any_product_sim") or 0.0)
    any_main = float(row.get("candidate_nearest_any_main_sim") or 0.0)
    context_sim = float(row.get("candidate_nearest_context_transform_sim") or 0.0)
    pair_sim = float(row.get("candidate_nearest_pair_compatible_sim") or 0.0)
    match_score = float(row.get("candidate_inferred_transform_match_score") or 0.0)
    rank = max(1, int(row.get("candidate_rank") or 1))
    inv_rank = 1.0 / float(rank)
    inv_log_rank = 1.0 / float(np.log2(rank + 1.0))
    return [
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
        inv_log_rank * context_sim,
        inv_rank * match_score,
    ]


def _model_specs_v3(feature_names: list[str], *, model_set: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(feature_names):
        if name.startswith("chem__"):
            groups["chem_only"].append(idx)
        if name.startswith("context__"):
            groups["context_only"].append(idx)
        if name.startswith("compat__"):
            groups["compatibility_only"].append(idx)
        if name.startswith("evidence__"):
            groups["program_evidence_only"].append(idx)
        if name.startswith("candev__"):
            groups["candidate_evidence_only"].append(idx)
    groups["chem_plus_candidate_evidence"] = groups["chem_only"] + groups["candidate_evidence_only"]
    groups["candidate_evidence_plus_context"] = (
        groups["candidate_evidence_only"]
        + groups["context_only"]
        + groups["compatibility_only"]
        + groups["program_evidence_only"]
    )
    groups["ccts_v3_no_context"] = groups["chem_only"] + groups["candidate_evidence_only"] + groups["compatibility_only"] + groups["program_evidence_only"]
    groups["ccts_v3_full"] = list(range(len(feature_names)))
    if model_set == "raw":
        return {}
    if model_set == "fast":
        keep = {
            "candidate_evidence_only",
            "chem_plus_candidate_evidence",
            "ccts_v3_no_context",
        }
        return {key: value for key, value in groups.items() if key in keep}
    if model_set == "full":
        return dict(groups)
    raise ValueError("model_set must be 'raw', 'fast', or 'full'")


def _row_score(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key) or 0.0) for row in rows], dtype=np.float32)


def _raw_blend_reports(
    train_data: CandidateDataset,
    val_data: CandidateDataset,
    test_data: CandidateDataset,
    baseline_scores: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    del train_data
    chem_val = baseline_scores["chem_rank"]["val"]
    chem_test = baseline_scores["chem_rank"]["test"]
    best_start = _v3_selection_metric(_evaluate_sparse_dataset(val_data, chem_val))
    out: dict[str, Any] = {}
    for aux_name, payload in baseline_scores.items():
        if aux_name == "chem_rank":
            continue
        aux_val = payload["val"]
        aux_test = payload["test"]
        best_alpha = 0.0
        best_score = best_start
        for alpha in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0]:
            blend_val = _standardize(chem_val) + float(alpha) * _standardize(aux_val)
            score = _v3_selection_metric(_evaluate_sparse_dataset(val_data, blend_val))
            if score > best_score:
                best_alpha = float(alpha)
                best_score = score
        blend_test = _standardize(chem_test) + best_alpha * _standardize(aux_test)
        out[f"chem_rank_plus_{aux_name}"] = {
            "alpha_selected_on_val": best_alpha,
            "val_selection_score": round(float(best_score), 6),
            "test": _evaluate_sparse_dataset(test_data, blend_test),
        }
    return out


def _v3_selection_metric(report: dict[str, Any]) -> float:
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


def _decision(
    baseline_reports: dict[str, Any],
    raw_blends: dict[str, Any],
    model_reports: dict[str, Any],
) -> dict[str, Any]:
    label = "block_supported_positive_label"
    chem_mrr = float(((baseline_reports.get("chem_rank") or {}).get("test") or {}).get(label, {}).get("mrr_covered") or 0.0)
    candidates: dict[str, float] = {}
    for name, payload in baseline_reports.items():
        if name == "chem_rank":
            continue
        candidates[f"raw:{name}"] = float(((payload.get("test") or {}).get(label) or {}).get("mrr_covered") or 0.0)
    for name, payload in raw_blends.items():
        candidates[f"blend:{name}"] = float((((payload.get("test") or {}).get(label) or {}).get("mrr_covered")) or 0.0)
    for name, payload in model_reports.items():
        candidates[f"model:{name}"] = float((((payload.get("test") or {}).get(label) or {}).get("mrr_covered")) or 0.0)
    best_name, best_mrr = max(candidates.items(), key=lambda kv: kv[1]) if candidates else ("", 0.0)
    delta = best_mrr - chem_mrr
    return {
        "primary_label": label,
        "chem_rank_mrr": round(chem_mrr, 6),
        "best_method": best_name,
        "best_method_mrr": round(best_mrr, 6),
        "delta_vs_chem_rank": round(delta, 6),
        "candidate_evidence_promotable": bool(delta >= 0.03),
        "search_time_gate": "hold" if delta < 0.03 else "candidate_features_ready_for_replay_or_in_search_ablation",
    }


def _compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "transition_id",
        "product_smiles",
        "candidate_rank",
        "candidate_score",
        "candidate_source",
        "candidate_model",
        "candidate_type",
        "transform",
        "previous_transform",
        "next_transform",
        "exact_label",
        "similar_label",
        "similar_only_label",
        "positive_label",
        "block_supported_positive_label",
        "block_supported_exact_label",
        "training_relevance",
        "reactant_similarity",
        "previous_support_similarity",
        "next_support_similarity",
        "candidate_nearest_any_transition_sim",
        "candidate_nearest_context_transform_sim",
        "candidate_nearest_pair_compatible_sim",
        "candidate_inferred_transform",
        "candidate_inferred_transform_matches_context",
        "candidate_inferred_transform_match_score",
    ]
    return [{key: row.get(key) for key in keep} for row in rows]


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CCTS-v3 Candidate Evidence",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(result.get("decision") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Metrics",
        "",
        "| Method | Label | Coverage | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows: list[tuple[str, dict[str, Any]]] = []
    for name, payload in (result.get("baselines") or {}).items():
        rows.append((name, (payload.get("test") or {})))
    for name, payload in (result.get("raw_blends") or {}).items():
        rows.append((name, (payload.get("test") or {})))
    for name, payload in (result.get("models") or {}).items():
        rows.append((name, (payload.get("test") or {})))
    for name, metrics in rows:
        for label in ("training_label", "block_supported_positive_label", "block_supported_exact_label", "exact_label", "positive_label"):
            metric = metrics.get(label) or {}
            at = metric.get("recall_at_k_all") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
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
    lines.extend(
        [
            "",
            "## Rank Delta Vs ChemEnzy Rank",
            "",
            "```json",
            json.dumps(result.get("rank_delta") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Train CCTS-v3 candidate-specific evidence ranker")
    ap.add_argument("--train-coverage", default="results/shared/cascadebench_strict_20260516/coverage/coverage_train_top100.json")
    ap.add_argument("--train-cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    ap.add_argument("--val-coverage", default="results/shared/cascadebench_strict_20260516/coverage/coverage_val_top100.json")
    ap.add_argument("--val-cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    ap.add_argument("--test-coverage", default="results/shared/cascadebench_strict_20260516/coverage/coverage_test_top100.json")
    ap.add_argument("--test-cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--output-dir", default="results/shared/cascadebench_strict_20260516/ccts_v3_candidate_evidence")
    ap.add_argument("--label-mode", choices=sorted(LABEL_MODES), default="binary_block_exact")
    ap.add_argument("--similarity-threshold", type=float, default=0.70)
    ap.add_argument("--adjacency-similarity-threshold", type=float, default=0.70)
    ap.add_argument("--max-candidates-per-transition", type=int, default=100)
    ap.add_argument("--n-estimators", type=int, default=360)
    ap.add_argument("--learning-rate", type=float, default=0.035)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-set", choices=("raw", "fast", "full"), default="raw")
    args = ap.parse_args()
    result = train_ccts_v3_candidate_evidence(
        train_coverage=Path(args.train_coverage),
        train_cache=Path(args.train_cache),
        val_coverage=Path(args.val_coverage),
        val_cache=Path(args.val_cache),
        test_coverage=Path(args.test_coverage),
        test_cache=Path(args.test_cache),
        program_manifest=Path(args.program_manifest),
        output_dir=Path(args.output_dir),
        label_mode=args.label_mode,
        similarity_threshold=args.similarity_threshold,
        adjacency_similarity_threshold=args.adjacency_similarity_threshold,
        max_candidates_per_transition=args.max_candidates_per_transition,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
        model_set=args.model_set,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "counts": result["counts"],
                "outputs": {
                    "report": str(Path(args.output_dir) / "ccts_v3_candidate_evidence_report.json"),
                    "markdown": str(Path(args.output_dir) / "ccts_v3_candidate_evidence_report.md"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

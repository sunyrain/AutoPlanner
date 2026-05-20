"""Train CCTS-v2 with sparse block-supported labels.

This is a corrective experiment for the Phase II guardrail audit.  The prior
CCTS target treated exact and similar-reactant candidates as equal positives,
which made the objective mostly single-step similarity.  This variant keeps the
same fixed ChemEnzy candidate pool and feature space, but trains on a stricter
evidence hierarchy:

- `binary_block_exact`: positive iff exact OR block-supported positive.
- `graded_evidence`: exact+block > exact-only > block-supported-similar >
  similar-only > other.

The default is `binary_block_exact` because it is the cleanest test of whether
block-level cascade supervision helps beyond ChemEnzy/molecule-only ranking.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.eval.train_ccts_v0_transition_ranker import (
    CandidateDataset,
    _baseline_scores,
    _candidate_rows_from_cache,
    _read_json,
    _write_jsonl,
)
from cascade_planner.eval.train_ccts_v1_transition_ranker import _coverage_leakage_report
from cascade_planner.eval.train_ccts_v2_transition_ranker import (
    _build_schema,
    _candidate_label_row,
    _feature_vector,
    _fit_ranker,
    _load_program_evidence,
    _model_specs,
    _transition_context,
)


SCHEMA_VERSION = "ccts_v2_sparse_labels.v1"
LABEL_MODES = {"binary_block_exact", "graded_evidence"}


def train_ccts_v2_sparse_labels(
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
    n_estimators: int = 300,
    learning_rate: float = 0.035,
    seed: int = 42,
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
    schema = _build_schema(train_transitions, train_cache_rows, program_evidence)

    train_data = _build_sparse_dataset(
        train_transitions,
        train_cache_rows,
        evidence=program_evidence,
        neighbor_context=neighbor_context,
        schema=schema,
        label_mode=label_mode,
        similarity_threshold=similarity_threshold,
        adjacency_similarity_threshold=adjacency_similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        require_trainable_group=True,
    )
    val_data = _build_sparse_dataset(
        val_transitions,
        val_cache_rows,
        evidence=program_evidence,
        neighbor_context=neighbor_context,
        schema=schema,
        label_mode=label_mode,
        similarity_threshold=similarity_threshold,
        adjacency_similarity_threshold=adjacency_similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        require_trainable_group=False,
    )
    test_data = _build_sparse_dataset(
        test_transitions,
        test_cache_rows,
        evidence=program_evidence,
        neighbor_context=neighbor_context,
        schema=schema,
        label_mode=label_mode,
        similarity_threshold=similarity_threshold,
        adjacency_similarity_threshold=adjacency_similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        require_trainable_group=False,
    )
    if not train_data.rows or not val_data.rows or not test_data.rows:
        raise ValueError("sparse CCTS requires non-empty train/val/test datasets")

    model_specs = _model_specs(train_data.feature_names)
    models: dict[str, Any] = {}
    model_reports: dict[str, Any] = {}
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

    baseline = {
        "train": _evaluate_sparse_dataset(train_data, _baseline_scores(train_data.rows)),
        "val": _evaluate_sparse_dataset(val_data, _baseline_scores(val_data.rows)),
        "test": _evaluate_sparse_dataset(test_data, _baseline_scores(test_data.rows)),
    }
    score_columns = {
        "chem_rank": _baseline_scores(test_data.rows),
        **{
            name: model.predict(test_data.x[:, model_specs[name]], num_iteration=getattr(model, "best_iteration_", None))
            for name, model in models.items()
        },
    }
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
            "elapsed_s": round(time.monotonic() - started, 3),
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
        "baseline_chem_rank": baseline,
        "models": model_reports,
        "rank_delta": _rank_delta(test_data, score_columns),
        "feature_schema": {
            "feature_names": train_data.feature_names,
            "model_specs": {name: [train_data.feature_names[idx] for idx in indices] for name, indices in model_specs.items()},
            **schema,
        },
    }
    with (output_dir / "ccts_v2_sparse_models.pkl").open("wb") as fh:
        pickle.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "models": models,
                "feature_schema": result["feature_schema"],
                "metadata": result["metadata"],
            },
            fh,
        )
    (output_dir / "ccts_v2_sparse_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "ccts_v2_sparse_report.md").write_text(_markdown(result), encoding="utf-8")
    _write_jsonl(output_dir / "ccts_v2_sparse_test_candidates.jsonl", _compact_rows(test_data.rows))
    return result


def _build_sparse_dataset(
    transitions: list[dict[str, Any]],
    cache: dict[str, Any],
    *,
    evidence: Any,
    neighbor_context: dict[str, dict[str, Any]],
    schema: dict[str, Any],
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
    feature_names = _feature_names_from_schema(schema)
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
            _add_block_support(row, neighbor=neighbor, adjacency_similarity_threshold=adjacency_similarity_threshold)
            row["training_relevance"] = _training_relevance(row, label_mode=label_mode)
            group_rows.append(row)
            group_x.append(_feature_vector(row, context=context, evidence=evidence, schema=schema))
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


def _feature_names_from_schema(schema: dict[str, Any]) -> list[str]:
    # Reuse the canonical feature vector by asking _model_specs later; the names
    # must match train_ccts_v2_transition_ranker._feature_names.
    from cascade_planner.eval.train_ccts_v2_transition_ranker import _feature_names

    return _feature_names(schema)


def _neighbor_context(program_manifest: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    out = {}
    for split in ("train", "val", "test"):
        path = outputs.get(split)
        if not path:
            continue
        for program in _read_jsonl(Path(path)):
            steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
            for idx, step in enumerate(steps):
                prev_step = steps[idx - 1] if idx > 0 else None
                next_step = steps[idx + 1] if idx + 1 < len(steps) else None
                out[str(step.get("transition_id") or "")] = {
                    "previous_product": canonical_smiles(str((prev_step or {}).get("product_smiles") or "")),
                    "next_main_reactant": canonical_smiles(str((next_step or {}).get("main_reactant") or "")),
                }
    return out


def _add_block_support(row: dict[str, Any], *, neighbor: dict[str, Any], adjacency_similarity_threshold: float) -> None:
    product = canonical_smiles(str(row.get("product_smiles") or ""))
    reactants = [canonical_smiles(str(smi or "")) for smi in row.get("candidate_reactants") or []]
    reactants = [smi for smi in reactants if smi]
    previous_product = neighbor.get("previous_product") or ""
    next_main = neighbor.get("next_main_reactant") or ""
    prev_sim = _best_similarity(previous_product, reactants) if previous_product else None
    next_sim = _similarity(product, next_main) if next_main else None
    has_prev = bool(previous_product)
    has_next = bool(next_main)
    prev_supported = (prev_sim is not None and prev_sim >= adjacency_similarity_threshold) if has_prev else None
    next_supported = (next_sim is not None and next_sim >= adjacency_similarity_threshold) if has_next else None
    if has_prev and has_next:
        block_supported = bool(prev_supported and next_supported)
    elif has_prev:
        block_supported = bool(prev_supported)
    elif has_next:
        block_supported = bool(next_supported)
    else:
        block_supported = False
    similar_only = bool(row.get("similar_label") and not row.get("exact_label"))
    row.update(
        {
            "previous_support_similarity": prev_sim,
            "next_support_similarity": next_sim,
            "block_supported_positive_label": bool(row.get("positive_label") and block_supported),
            "block_supported_exact_label": bool(row.get("exact_label") and block_supported),
            "block_supported_similar_only_label": bool(similar_only and block_supported),
            "similar_only_label": similar_only,
        }
    )


def _training_relevance(row: dict[str, Any], *, label_mode: str) -> int:
    exact = bool(row.get("exact_label"))
    block = bool(row.get("block_supported_positive_label"))
    similar_only = bool(row.get("similar_only_label"))
    if label_mode == "binary_block_exact":
        return int(exact or block)
    if exact and block:
        return 3
    if exact:
        return 3
    if block:
        return 2
    if similar_only:
        return 1
    return 0


def _model_report(model: Any, *, train_data: CandidateDataset, val_data: CandidateDataset, test_data: CandidateDataset, feature_indices: list[int]) -> dict[str, Any]:
    return {
        "train": _evaluate_sparse_dataset(train_data, model.predict(train_data.x[:, feature_indices], num_iteration=model.best_iteration_)),
        "val": _evaluate_sparse_dataset(val_data, model.predict(val_data.x[:, feature_indices], num_iteration=model.best_iteration_)),
        "test": _evaluate_sparse_dataset(test_data, model.predict(test_data.x[:, feature_indices], num_iteration=model.best_iteration_)),
        "feature_count": len(feature_indices),
        "feature_names": [train_data.feature_names[idx] for idx in feature_indices],
        "best_iteration": int(model.best_iteration_ or 0),
    }


def _evaluate_sparse_dataset(dataset: CandidateDataset, scores: np.ndarray) -> dict[str, Any]:
    grouped = []
    offset = 0
    for group_size, group_id in zip(dataset.group_sizes, dataset.group_ids):
        rows = dataset.rows[offset : offset + group_size]
        group_scores = scores[offset : offset + group_size]
        grouped.append((group_id, rows, group_scores))
        offset += group_size
    return {
        "groups": len(grouped),
        "candidate_rows": len(dataset.rows),
        "relevance_rows": int(np.sum(dataset.y > 0)),
        "training_label": _ranking_metrics(grouped, lambda row: int(row.get("training_relevance") or 0) > 0),
        "block_supported_positive_label": _ranking_metrics(grouped, lambda row: bool(row.get("block_supported_positive_label"))),
        "block_supported_exact_label": _ranking_metrics(grouped, lambda row: bool(row.get("block_supported_exact_label"))),
        "exact_label": _ranking_metrics(grouped, lambda row: bool(row.get("exact_label"))),
        "similar_only_label": _ranking_metrics(grouped, lambda row: bool(row.get("similar_only_label"))),
        "positive_label": _ranking_metrics(grouped, lambda row: bool(row.get("positive_label"))),
    }


def _ranking_metrics(grouped: list[tuple[str, list[dict[str, Any]], np.ndarray]], label_getter: Any) -> dict[str, Any]:
    covered = 0
    reciprocal = []
    recalls = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    buckets = Counter()
    for _, rows, scores in grouped:
        order = sorted(range(len(rows)), key=lambda idx: (-float(scores[idx]), int(rows[idx].get("candidate_rank") or 10**9)))
        ranks = [pos + 1 for pos, idx in enumerate(order) if label_getter(rows[idx])]
        if ranks:
            covered += 1
            rank = min(ranks)
            reciprocal.append(1.0 / rank)
            buckets[_rank_bucket(rank)] += 1
            for k in recalls:
                if rank <= k:
                    recalls[k] += 1
        else:
            buckets["missing"] += 1
    total = max(len(grouped), 1)
    covered_den = max(covered, 1)
    return {
        "covered_groups": covered,
        "coverage": round(covered / total, 6),
        "mrr_covered": round(sum(reciprocal) / covered_den, 6) if reciprocal else 0.0,
        "recall_at_k_all": {str(k): round(value / total, 6) for k, value in recalls.items()},
        "recall_at_k_covered": {str(k): round(value / covered_den, 6) for k, value in recalls.items()},
        "first_rank_buckets": dict(sorted(buckets.items())),
    }


def _rank_delta(dataset: CandidateDataset, score_columns: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for score_name, scores in score_columns.items():
        if score_name == "chem_rank":
            continue
        out[score_name] = _label_delta(dataset, base_scores=score_columns["chem_rank"], candidate_scores=scores)
    return out


def _label_delta(dataset: CandidateDataset, *, base_scores: np.ndarray, candidate_scores: np.ndarray) -> dict[str, Any]:
    labels = {
        "training_label": lambda row: int(row.get("training_relevance") or 0) > 0,
        "block_supported_positive_label": lambda row: bool(row.get("block_supported_positive_label")),
        "exact_label": lambda row: bool(row.get("exact_label")),
    }
    out = {}
    offset = 0
    for label_name, getter in labels.items():
        covered = []
        for group_size in dataset.group_sizes:
            rows = dataset.rows[offset : offset + group_size]
            base_group = base_scores[offset : offset + group_size]
            cand_group = candidate_scores[offset : offset + group_size]
            base_rank = _best_rank(rows, base_group, getter)
            cand_rank = _best_rank(rows, cand_group, getter)
            if base_rank is not None and cand_rank is not None:
                covered.append((base_rank, cand_rank))
            offset += group_size
        offset = 0
        out[label_name] = {
            "covered": len(covered),
            "improved": sum(1 for base, cand in covered if cand < base),
            "same": sum(1 for base, cand in covered if cand == base),
            "worsened": sum(1 for base, cand in covered if cand > base),
            "mean_delta": round(sum(cand - base for base, cand in covered) / max(len(covered), 1), 6) if covered else None,
        }
    return out


def _best_rank(rows: list[dict[str, Any]], scores: np.ndarray, label_getter: Any) -> int | None:
    order = sorted(range(len(rows)), key=lambda idx: (-float(scores[idx]), int(rows[idx].get("candidate_rank") or 10**9)))
    ranks = [pos + 1 for pos, idx in enumerate(order) if label_getter(rows[idx])]
    return min(ranks) if ranks else None


def _compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "transition_id",
        "product_smiles",
        "candidate_rank",
        "candidate_score",
        "candidate_source",
        "candidate_model",
        "candidate_type",
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
    ]
    return [{key: row.get(key) for key in keep} for row in rows]


def _best_similarity(smiles: str, candidates: list[str]) -> float:
    if not smiles or not candidates:
        return 0.0
    return max((_similarity(smiles, candidate) for candidate in candidates), default=0.0)


def _similarity(left: str, right: str) -> float:
    left_fp = _fp(left)
    right_fp = _fp(right)
    if left_fp is None or right_fp is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def _rank_bucket(rank: int) -> str:
    if rank <= 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return "51plus"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CCTS-v2 Sparse Labels",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Metrics",
        "",
        "| Model | Label | Coverage | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [("chem_rank", (result.get("baseline_chem_rank") or {}).get("test") or {})]
    rows.extend((name, (payload.get("test") or {})) for name, payload in (result.get("models") or {}).items())
    for name, metrics in rows:
        for label in ("training_label", "block_supported_positive_label", "exact_label", "positive_label"):
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
    ap = argparse.ArgumentParser(description="Train CCTS-v2 sparse/block-supported label ranker")
    ap.add_argument("--train-coverage", default="results/shared/cascadebench_strict_20260516/coverage/coverage_train_top100.json")
    ap.add_argument("--train-cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    ap.add_argument("--val-coverage", default="results/shared/cascadebench_strict_20260516/coverage/coverage_val_top100.json")
    ap.add_argument("--val-cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    ap.add_argument("--test-coverage", default="results/shared/cascadebench_strict_20260516/coverage/coverage_test_top100.json")
    ap.add_argument("--test-cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--output-dir", default="results/shared/cascadebench_strict_20260516/ccts_v2_sparse_block")
    ap.add_argument("--label-mode", choices=sorted(LABEL_MODES), default="binary_block_exact")
    ap.add_argument("--similarity-threshold", type=float, default=0.70)
    ap.add_argument("--adjacency-similarity-threshold", type=float, default=0.70)
    ap.add_argument("--max-candidates-per-transition", type=int, default=100)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.035)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    result = train_ccts_v2_sparse_labels(
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
    )
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "leakage_checks": result["leakage_checks"],
                "test": {
                    "baseline": result["baseline_chem_rank"]["test"],
                    "models": {name: row["test"] for name, row in result["models"].items()},
                },
                "outputs": {
                    "report": str(Path(args.output_dir) / "ccts_v2_sparse_report.json"),
                    "markdown": str(Path(args.output_dir) / "ccts_v2_sparse_report.md"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

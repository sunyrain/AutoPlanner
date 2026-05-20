"""Train CBA-v0: a cascade block transform-pair applicability ranker.

CBA-v0 is intentionally narrower than a route reranker.  It ranks transform-pair
prototypes learned from the v4 train split for a held-out cascade context.  The
held-out upstream step is used only as the evaluation label; its structure is
not used as an input feature.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMRanker, early_stopping
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

from cascade_planner.cascade_search.v4_product_value import canonical_smiles


SCHEMA_VERSION = "cba_v0_block_applicability.transform_pair_ranker.v1"
TOP_KS = (1, 3, 5, 10, 20, 50)


@dataclass
class BlockRecord:
    block_id: str
    program_id: str
    doi: str
    split: str
    target_smiles: str
    downstream_product: str
    downstream_main_reactant: str
    transform_pair: str
    upstream_transform: str
    downstream_transform: str
    cascade_type: str
    compatibility_label: str
    hidden_or_nonisolated: bool
    right_catalyst_classes: tuple[str, ...]
    right_condition_tokens: tuple[str, ...]


@dataclass
class PairPrototype:
    transform_pair: str
    upstream_transform: str
    downstream_transform: str
    count: int
    frequency_rank: int
    hidden_rate: float
    catalyst_counts: dict[str, int]
    condition_counts: dict[str, int]
    cascade_type_counts: dict[str, int]
    target_fps: list[Any]
    downstream_fps: list[Any]
    downstream_main_fps: list[Any]
    target_heavy_mean: float
    downstream_heavy_mean: float


@dataclass
class CandidateDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    group_sizes: list[int]
    group_ids: list[str]
    feature_names: list[str]


def train_cba_v0_block_applicability(
    *,
    program_manifest: Path,
    output_dir: Path,
    n_estimators: int = 260,
    learning_rate: float = 0.04,
    seed: int = 42,
    condition_token_limit: int = 64,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_blocks = _load_split_blocks(program_manifest)
    train_blocks = split_blocks["train"]
    val_blocks = split_blocks["val"]
    test_blocks = split_blocks["test"]
    prototypes = _build_pair_prototypes(train_blocks)
    schema = _build_schema(train_blocks, prototypes, condition_token_limit=condition_token_limit)
    train_data = _build_dataset(train_blocks, prototypes, schema=schema, require_positive=True)
    val_data = _build_dataset(val_blocks, prototypes, schema=schema, require_positive=False)
    test_data = _build_dataset(test_blocks, prototypes, schema=schema, require_positive=False)
    if not train_data.rows or not val_data.rows or not test_data.rows:
        raise ValueError("CBA-v0 requires non-empty train/val/test datasets")

    model_specs = _model_specs(train_data.feature_names)
    models: dict[str, Any] = {}
    reports: dict[str, Any] = {}
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
        reports[name] = _model_report(model, train_data=train_data, val_data=val_data, test_data=test_data, feature_indices=indices)

    baselines = {
        "frequency": {
            "train": _ranking_metrics(train_data, _baseline_scores(train_data.rows, "frequency")),
            "val": _ranking_metrics(val_data, _baseline_scores(val_data.rows, "frequency")),
            "test": _ranking_metrics(test_data, _baseline_scores(test_data.rows, "frequency")),
        },
        "target_retrieval": {
            "train": _ranking_metrics(train_data, _baseline_scores(train_data.rows, "target_retrieval")),
            "val": _ranking_metrics(val_data, _baseline_scores(val_data.rows, "target_retrieval")),
            "test": _ranking_metrics(test_data, _baseline_scores(test_data.rows, "target_retrieval")),
        },
        "downstream_retrieval": {
            "train": _ranking_metrics(train_data, _baseline_scores(train_data.rows, "downstream_retrieval")),
            "val": _ranking_metrics(val_data, _baseline_scores(val_data.rows, "downstream_retrieval")),
            "test": _ranking_metrics(test_data, _baseline_scores(test_data.rows, "downstream_retrieval")),
        },
    }
    scorecards = {
        "val": _scorecard(val_data, baselines=baselines, models=models, model_specs=model_specs),
        "test": _scorecard(test_data, baselines=baselines, models=models, model_specs=model_specs),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "output_dir": str(output_dir),
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "seed": seed,
            "condition_token_limit": condition_token_limit,
            "elapsed_s": round(time.monotonic() - started, 3),
            "leakage_guard": "candidate transform-pair prototypes are built from train split only; held-out upstream block structure is label-only",
        },
        "counts": {
            "train_blocks": len(train_blocks),
            "val_blocks": len(val_blocks),
            "test_blocks": len(test_blocks),
            "train_transform_pair_prototypes": len(prototypes),
            "train_candidate_rows": len(train_data.rows),
            "val_candidate_rows": len(val_data.rows),
            "test_candidate_rows": len(test_data.rows),
            "val_positive_groups": _positive_group_count(val_data),
            "test_positive_groups": _positive_group_count(test_data),
            "val_transform_pair_seen_rate": _positive_group_count(val_data) / max(len(val_data.group_sizes), 1),
            "test_transform_pair_seen_rate": _positive_group_count(test_data) / max(len(test_data.group_sizes), 1),
        },
        "baselines": baselines,
        "models": reports,
        "scorecards": scorecards,
        "feature_schema": {
            "feature_names": train_data.feature_names,
            "model_specs": {name: [train_data.feature_names[idx] for idx in indices] for name, indices in model_specs.items()},
            **schema,
        },
    }
    with (output_dir / "cba_v0_models.pkl").open("wb") as fh:
        pickle.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "models": models,
                "feature_schema": result["feature_schema"],
                "metadata": result["metadata"],
            },
            fh,
        )
    (output_dir / "cba_v0_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "cba_v0_report.md").write_text(_markdown(result), encoding="utf-8")
    _write_jsonl(output_dir / "cba_v0_test_rankings.jsonl", _ranked_rows(test_data, _all_test_scores(test_data, models, model_specs), top_n=20))
    return result


def _load_split_blocks(program_manifest: Path) -> dict[str, list[BlockRecord]]:
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    return {
        split: _blocks_from_programs(_read_jsonl(Path(outputs[split])), split=split)
        for split in ("train", "val", "test")
    }


def _blocks_from_programs(programs: list[dict[str, Any]], *, split: str) -> list[BlockRecord]:
    rows = []
    for program in programs:
        steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
        compatibility = program.get("compatibility") or {}
        for idx, (left, right) in enumerate(zip(steps, steps[1:])):
            up = _norm_transform(left.get("transformation_superclass"))
            down = _norm_transform(right.get("transformation_superclass"))
            rows.append(
                BlockRecord(
                    block_id=f"{program.get('program_id')}::{idx}",
                    program_id=str(program.get("program_id") or ""),
                    doi=str(program.get("doi") or ""),
                    split=split,
                    target_smiles=_canon(program.get("target_smiles")),
                    downstream_product=_canon(right.get("product_smiles")),
                    downstream_main_reactant=_canon(right.get("main_reactant")),
                    transform_pair=f"{up}->{down}",
                    upstream_transform=up,
                    downstream_transform=down,
                    cascade_type=_norm(program.get("cascade_type")),
                    compatibility_label=_norm(compatibility.get("compatibility_label")),
                    hidden_or_nonisolated=bool(left.get("intermediate_isolated") is False),
                    right_catalyst_classes=tuple(_norm_list(right.get("catalyst_classes"))),
                    right_condition_tokens=tuple(_norm_list(right.get("condition_tokens"))),
                )
            )
    return rows


def _build_pair_prototypes(train_blocks: list[BlockRecord]) -> list[PairPrototype]:
    grouped: dict[str, list[BlockRecord]] = defaultdict(list)
    for block in train_blocks:
        grouped[block.transform_pair].append(block)
    counts = Counter({pair: len(rows) for pair, rows in grouped.items()})
    ranked_pairs = {pair: rank for rank, (pair, _count) in enumerate(counts.most_common(), start=1)}
    out = []
    for pair, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        upstream, downstream = pair.split("->", 1)
        target_fps = [_fp(row.target_smiles) for row in rows]
        downstream_fps = [_fp(row.downstream_product) for row in rows]
        downstream_main_fps = [_fp(row.downstream_main_reactant) for row in rows]
        target_heavy = [_heavy_atoms(row.target_smiles) for row in rows]
        downstream_heavy = [_heavy_atoms(row.downstream_product) for row in rows]
        out.append(
            PairPrototype(
                transform_pair=pair,
                upstream_transform=upstream,
                downstream_transform=downstream,
                count=len(rows),
                frequency_rank=ranked_pairs[pair],
                hidden_rate=sum(1 for row in rows if row.hidden_or_nonisolated) / max(len(rows), 1),
                catalyst_counts=Counter(cls for row in rows for cls in row.right_catalyst_classes),
                condition_counts=Counter(token for row in rows for token in row.right_condition_tokens),
                cascade_type_counts=Counter(row.cascade_type for row in rows),
                target_fps=[fp for fp in target_fps if fp is not None],
                downstream_fps=[fp for fp in downstream_fps if fp is not None],
                downstream_main_fps=[fp for fp in downstream_main_fps if fp is not None],
                target_heavy_mean=float(np.mean(target_heavy)) if target_heavy else 0.0,
                downstream_heavy_mean=float(np.mean(downstream_heavy)) if downstream_heavy else 0.0,
            )
        )
    return out


def _build_schema(train_blocks: list[BlockRecord], prototypes: list[PairPrototype], *, condition_token_limit: int) -> dict[str, Any]:
    condition_counts = Counter(token for row in train_blocks for token in row.right_condition_tokens)
    catalyst_classes = sorted({cls for row in train_blocks for cls in row.right_catalyst_classes})
    transforms = sorted({proto.upstream_transform for proto in prototypes} | {proto.downstream_transform for proto in prototypes})
    return {
        "transforms": transforms,
        "transform_pairs": [proto.transform_pair for proto in prototypes],
        "cascade_types": sorted({row.cascade_type for row in train_blocks}),
        "compatibility_labels": sorted({row.compatibility_label for row in train_blocks}),
        "catalyst_classes": catalyst_classes,
        "condition_tokens": [token for token, _count in condition_counts.most_common(int(condition_token_limit))],
    }


def _build_dataset(
    blocks: list[BlockRecord],
    prototypes: list[PairPrototype],
    *,
    schema: dict[str, Any],
    require_positive: bool,
) -> CandidateDataset:
    feature_names = _feature_names(schema)
    rows = []
    x_rows = []
    y_rows = []
    group_sizes = []
    group_ids = []
    for block in blocks:
        group_rows = []
        group_x = []
        group_y = []
        for proto in prototypes:
            row = _candidate_row(block, proto)
            group_rows.append(row)
            group_x.append(_feature_vector(block, proto, schema=schema))
            group_y.append(int(row["positive_label"]))
        if require_positive and sum(group_y) == 0:
            continue
        rows.extend(group_rows)
        x_rows.extend(group_x)
        y_rows.extend(group_y)
        group_sizes.append(len(group_rows))
        group_ids.append(block.block_id)
    return CandidateDataset(
        rows=rows,
        x=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.int32),
        group_sizes=group_sizes,
        group_ids=group_ids,
        feature_names=feature_names,
    )


def _candidate_row(block: BlockRecord, proto: PairPrototype) -> dict[str, Any]:
    right_match = proto.downstream_transform == block.downstream_transform
    return {
        "block_id": block.block_id,
        "program_id": block.program_id,
        "doi": block.doi,
        "split": block.split,
        "target_smiles": block.target_smiles,
        "true_transform_pair": block.transform_pair,
        "candidate_transform_pair": proto.transform_pair,
        "candidate_upstream_transform": proto.upstream_transform,
        "candidate_downstream_transform": proto.downstream_transform,
        "candidate_count": proto.count,
        "candidate_frequency_rank": proto.frequency_rank,
        "candidate_hidden_rate": round(float(proto.hidden_rate), 6),
        "right_transform_match": right_match,
        "positive_label": proto.transform_pair == block.transform_pair,
    }


def _feature_vector(block: BlockRecord, proto: PairPrototype, *, schema: dict[str, Any]) -> list[float]:
    target_fp = _fp(block.target_smiles)
    down_fp = _fp(block.downstream_product)
    down_main_fp = _fp(block.downstream_main_reactant)
    target_sims = _bulk_similarity(target_fp, proto.target_fps)
    down_sims = _bulk_similarity(down_fp, proto.downstream_fps)
    down_main_sims = _bulk_similarity(down_main_fp, proto.downstream_main_fps)
    block_target_heavy = _heavy_atoms(block.target_smiles)
    block_down_heavy = _heavy_atoms(block.downstream_product)
    catalyst_overlap = len(set(block.right_catalyst_classes) & set(proto.catalyst_counts))
    condition_overlap = len(set(block.right_condition_tokens) & set(proto.condition_counts))
    values = [
        math.log1p(float(proto.count)),
        1.0 / float(max(1, proto.frequency_rank)),
        float(proto.hidden_rate),
        float(proto.downstream_transform == block.downstream_transform),
        float(proto.upstream_transform == block.upstream_transform),
        float(catalyst_overlap),
        float(condition_overlap),
        float(catalyst_overlap / max(len(block.right_catalyst_classes), 1)),
        float(condition_overlap / max(len(block.right_condition_tokens), 1)),
        max(target_sims) if target_sims else 0.0,
        float(np.mean(target_sims)) if target_sims else 0.0,
        max(down_sims) if down_sims else 0.0,
        float(np.mean(down_sims)) if down_sims else 0.0,
        max(down_main_sims) if down_main_sims else 0.0,
        float(np.mean(down_main_sims)) if down_main_sims else 0.0,
        block_target_heavy / 100.0,
        block_down_heavy / 100.0,
        proto.target_heavy_mean / 100.0,
        proto.downstream_heavy_mean / 100.0,
        abs(block_target_heavy - proto.target_heavy_mean) / 100.0,
        abs(block_down_heavy - proto.downstream_heavy_mean) / 100.0,
        float(len(block.right_catalyst_classes)),
        float(len(block.right_condition_tokens)),
    ]
    values.extend(_one_hot(proto.upstream_transform, schema["transforms"]))
    values.extend(_one_hot(proto.downstream_transform, schema["transforms"]))
    values.extend(_one_hot(block.downstream_transform, schema["transforms"]))
    values.extend(_one_hot(proto.transform_pair, schema["transform_pairs"]))
    values.extend(_one_hot(block.cascade_type, schema["cascade_types"]))
    values.extend(_one_hot(block.compatibility_label, schema["compatibility_labels"]))
    values.extend(_multi_hot(block.right_catalyst_classes, schema["catalyst_classes"]))
    values.extend(_multi_hot(block.right_condition_tokens, schema["condition_tokens"]))
    return values


def _feature_names(schema: dict[str, Any]) -> list[str]:
    names = [
        "evidence__log_pair_count",
        "evidence__inverse_frequency_rank",
        "evidence__candidate_hidden_rate",
        "downstream__right_transform_match",
        "oracle_check__upstream_transform_match",
        "downstream__right_catalyst_overlap",
        "downstream__right_condition_overlap",
        "downstream__right_catalyst_overlap_rate",
        "downstream__right_condition_overlap_rate",
        "target__target_to_pair_max_sim",
        "target__target_to_pair_mean_sim",
        "downstream__product_to_pair_max_sim",
        "downstream__product_to_pair_mean_sim",
        "downstream__main_to_pair_max_sim",
        "downstream__main_to_pair_mean_sim",
        "target__target_heavy_atoms_scaled",
        "downstream__product_heavy_atoms_scaled",
        "evidence__pair_target_heavy_mean_scaled",
        "evidence__pair_downstream_heavy_mean_scaled",
        "target__target_heavy_delta_to_pair",
        "downstream__product_heavy_delta_to_pair",
        "downstream__n_right_catalyst_classes",
        "downstream__n_right_condition_tokens",
    ]
    names.extend([f"candidate__upstream_transform={value}" for value in schema["transforms"]])
    names.extend([f"candidate__downstream_transform={value}" for value in schema["transforms"]])
    names.extend([f"downstream__true_downstream_transform={value}" for value in schema["transforms"]])
    names.extend([f"candidate__transform_pair={value}" for value in schema["transform_pairs"]])
    names.extend([f"context__cascade_type={value}" for value in schema["cascade_types"]])
    names.extend([f"context__compatibility_label={value}" for value in schema["compatibility_labels"]])
    names.extend([f"downstream__right_catalyst_class={value}" for value in schema["catalyst_classes"]])
    names.extend([f"downstream__right_condition_token={value}" for value in schema["condition_tokens"]])
    return names


def _model_specs(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(feature_names):
        if name.startswith("evidence__"):
            groups["frequency_evidence"].append(idx)
        if name.startswith("target__") or name.startswith("candidate__") or name.startswith("evidence__") or name.startswith("context__"):
            groups["cba_target_only"].append(idx)
        if (
            name.startswith("target__")
            or name.startswith("candidate__")
            or name.startswith("evidence__")
            or name.startswith("context__")
            or name.startswith("downstream__")
        ):
            groups["cba_downstream_context"].append(idx)
    groups["oracle_upper_check"] = list(range(len(feature_names)))
    groups["cba_target_only"] = [idx for idx in groups["cba_target_only"] if not feature_names[idx].startswith("oracle_check__")]
    groups["cba_downstream_context"] = [idx for idx in groups["cba_downstream_context"] if not feature_names[idx].startswith("oracle_check__")]
    return dict(groups)


def _fit_ranker(
    train_data: CandidateDataset,
    val_data: CandidateDataset,
    *,
    feature_indices: list[int],
    n_estimators: int,
    learning_rate: float,
    seed: int,
) -> LGBMRanker:
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.05,
        random_state=int(seed),
        verbose=-1,
    )
    model.fit(
        train_data.x[:, feature_indices],
        train_data.y,
        group=train_data.group_sizes,
        eval_set=[(val_data.x[:, feature_indices], val_data.y)],
        eval_group=[val_data.group_sizes],
        eval_at=[1, 3, 5, 10, 20],
        callbacks=[early_stopping(30, verbose=False)],
    )
    return model


def _model_report(
    model: Any,
    *,
    train_data: CandidateDataset,
    val_data: CandidateDataset,
    test_data: CandidateDataset,
    feature_indices: list[int],
) -> dict[str, Any]:
    return {
        "train": _ranking_metrics(train_data, _predict(model, train_data, feature_indices)),
        "val": _ranking_metrics(val_data, _predict(model, val_data, feature_indices)),
        "test": _ranking_metrics(test_data, _predict(model, test_data, feature_indices)),
        "feature_count": len(feature_indices),
        "best_iteration": int(model.best_iteration_ or 0),
    }


def _predict(model: Any, data: CandidateDataset, indices: list[int]) -> np.ndarray:
    return model.predict(data.x[:, indices], num_iteration=model.best_iteration_)


def _baseline_scores(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    scores = []
    for row in rows:
        count_score = math.log1p(float(row.get("candidate_count") or 0))
        freq_score = 1.0 / max(1.0, float(row.get("candidate_frequency_rank") or 1))
        if name == "frequency":
            scores.append(count_score + freq_score)
        elif name == "target_retrieval":
            scores.append(count_score + freq_score)
        elif name == "downstream_retrieval":
            scores.append(count_score + freq_score + (5.0 if row.get("right_transform_match") else 0.0))
        else:
            scores.append(0.0)
    return np.asarray(scores, dtype=np.float32)


def _ranking_metrics(data: CandidateDataset, scores: np.ndarray) -> dict[str, Any]:
    offset = 0
    ranks = []
    top1_right_match = 0
    top1_pair_counts = Counter()
    examples = []
    for group_size, group_id in zip(data.group_sizes, data.group_ids):
        rows = data.rows[offset : offset + group_size]
        group_scores = scores[offset : offset + group_size]
        order = sorted(range(group_size), key=lambda idx: (-float(group_scores[idx]), str(rows[idx].get("candidate_transform_pair") or "")))
        top = rows[order[0]] if order else {}
        top1_right_match += int(bool(top.get("right_transform_match")))
        top1_pair_counts[str(top.get("candidate_transform_pair") or "")] += 1
        positive_ranks = [rank + 1 for rank, idx in enumerate(order) if rows[idx].get("positive_label")]
        ranks.append(min(positive_ranks) if positive_ranks else None)
        if len(examples) < 20:
            examples.append(
                {
                    "block_id": group_id,
                    "true_transform_pair": rows[0].get("true_transform_pair") if rows else None,
                    "top_transform_pair": top.get("candidate_transform_pair"),
                    "top_right_transform_match": top.get("right_transform_match"),
                    "positive_rank": min(positive_ranks) if positive_ranks else None,
                }
            )
        offset += group_size
    covered = [int(rank) for rank in ranks if rank is not None]
    total = len(ranks)
    return {
        "groups": total,
        "covered_groups": len(covered),
        "coverage": round(len(covered) / max(total, 1), 6),
        "mrr_all": round(sum((1.0 / rank) if rank else 0.0 for rank in ranks) / max(total, 1), 6),
        "mrr_covered": round(sum(1.0 / rank for rank in covered) / max(len(covered), 1), 6) if covered else 0.0,
        "recall_at_k_all": {
            str(k): round(sum(1 for rank in covered if rank <= k) / max(total, 1), 6)
            for k in TOP_KS
        },
        "recall_at_k_covered": {
            str(k): round(sum(1 for rank in covered if rank <= k) / max(len(covered), 1), 6)
            for k in TOP_KS
        },
        "top1_right_transform_match_rate": round(top1_right_match / max(total, 1), 6),
        "top1_pair_counts": dict(top1_pair_counts.most_common(20)),
        "examples": examples,
    }


def _scorecard(
    data: CandidateDataset,
    *,
    baselines: dict[str, Any],
    models: dict[str, Any],
    model_specs: dict[str, list[int]],
) -> dict[str, Any]:
    scores = {
        "frequency": _baseline_scores(data.rows, "frequency"),
        "target_retrieval": _baseline_scores(data.rows, "target_retrieval"),
        "downstream_retrieval": _baseline_scores(data.rows, "downstream_retrieval"),
    }
    for name, model in models.items():
        scores[name] = _predict(model, data, model_specs[name])
    return {name: _ranking_metrics(data, values) for name, values in scores.items()}


def _all_test_scores(data: CandidateDataset, models: dict[str, Any], model_specs: dict[str, list[int]]) -> dict[str, np.ndarray]:
    scores = {
        "frequency": _baseline_scores(data.rows, "frequency"),
        "target_retrieval": _baseline_scores(data.rows, "target_retrieval"),
        "downstream_retrieval": _baseline_scores(data.rows, "downstream_retrieval"),
    }
    for name, model in models.items():
        scores[name] = _predict(model, data, model_specs[name])
    return scores


def _ranked_rows(data: CandidateDataset, score_columns: dict[str, np.ndarray], *, top_n: int) -> list[dict[str, Any]]:
    selected = "cba_downstream_context" if "cba_downstream_context" in score_columns else sorted(score_columns)[0]
    out = []
    offset = 0
    for group_size, group_id in zip(data.group_sizes, data.group_ids):
        rows = data.rows[offset : offset + group_size]
        group_scores = {name: values[offset : offset + group_size] for name, values in score_columns.items()}
        orders = {
            name: sorted(range(group_size), key=lambda idx: (-float(scores[idx]), str(rows[idx].get("candidate_transform_pair") or "")))
            for name, scores in group_scores.items()
        }
        rank_maps = {name: {idx: rank + 1 for rank, idx in enumerate(order)} for name, order in orders.items()}
        positive_rank = {
            name: min((rank_maps[name][idx] for idx, row in enumerate(rows) if row.get("positive_label")), default=None)
            for name in orders
        }
        out.append(
            {
                "block_id": group_id,
                "true_transform_pair": rows[0].get("true_transform_pair") if rows else None,
                "positive_rank": positive_rank,
                "top_candidates": [
                    {
                        "candidate_transform_pair": rows[idx].get("candidate_transform_pair"),
                        "candidate_count": rows[idx].get("candidate_count"),
                        "right_transform_match": rows[idx].get("right_transform_match"),
                        "positive_label": rows[idx].get("positive_label"),
                        "scores": {name: round(float(scores[idx]), 6) for name, scores in group_scores.items()},
                        "ranks": {name: rank_maps[name][idx] for name in rank_maps},
                    }
                    for idx in orders[selected][: max(1, int(top_n))]
                ],
            }
        )
        offset += group_size
    return out


def _positive_group_count(data: CandidateDataset) -> int:
    offset = 0
    total = 0
    for group_size in data.group_sizes:
        total += int(np.any(data.y[offset : offset + group_size] > 0))
        offset += group_size
    return total


def _bulk_similarity(query_fp: Any, fps: list[Any]) -> list[float]:
    if query_fp is None or not fps:
        return []
    return [float(value) for value in DataStructs.BulkTanimotoSimilarity(query_fp, fps)]


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def _heavy_atoms(smiles: str) -> float:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return float(mol.GetNumHeavyAtoms()) if mol is not None else 0.0


def _canon(value: Any) -> str:
    return canonical_smiles(str(value or "")) or str(value or "")


def _norm(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _norm_transform(value: Any) -> str:
    return _norm(value)


def _norm_list(values: Any) -> list[str]:
    return sorted({_norm(value) for value in (values or []) if _norm(value) != "unknown"})


def _one_hot(value: str, values: list[str]) -> list[float]:
    return [float(value == item) for item in values]


def _multi_hot(values: Any, schema_values: list[str]) -> list[float]:
    present = {str(value) for value in (values or [])}
    return [float(value in present) for value in schema_values]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CBA-v0 Block Applicability",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Transform-Pair Ranking",
        "",
        "| Model | Coverage | MRR all | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all | R@20 all | Right-transform@1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    test_cards = (result.get("scorecards") or {}).get("test") or {}
    for name, metric in test_cards.items():
        at = metric.get("recall_at_k_all") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    str(metric.get("coverage")),
                    str(metric.get("mrr_all")),
                    str(metric.get("mrr_covered")),
                    str(at.get("1")),
                    str(at.get("3")),
                    str(at.get("5")),
                    str(at.get("10")),
                    str(at.get("20")),
                    str(metric.get("top1_right_transform_match_rate")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "CBA-v0 ranks train-split transform-pair prototypes for held-out cascade blocks. "
            "The held-out upstream step is label-only, so this evaluates whether cascade context can select block types before any reaction-level injection.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Train CBA-v0 transform-pair block applicability ranker")
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-estimators", type=int, default=260)
    ap.add_argument("--learning-rate", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--condition-token-limit", type=int, default=64)
    args = ap.parse_args()
    result = train_cba_v0_block_applicability(
        program_manifest=Path(args.program_manifest),
        output_dir=Path(args.output_dir),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
        condition_token_limit=args.condition_token_limit,
    )
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "test": result["scorecards"]["test"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

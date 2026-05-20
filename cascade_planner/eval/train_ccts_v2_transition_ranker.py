"""Train CCTS-v2 with explicit cascade program context and hard negatives."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMRanker, early_stopping
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles
from cascade_planner.eval.train_ccts_v0_transition_ranker import (
    _baseline_scores,
    _best_set_similarity,
    _bucket_int,
    _candidate_rows_from_cache,
    _evaluate_dataset,
    _float_or_zero,
    _metric_for_selection,
    _read_json,
    _standardize,
    _write_jsonl,
    CandidateDataset,
)
from cascade_planner.eval.train_ccts_v1_transition_ranker import _coverage_leakage_report


CCTS_V2_SCHEMA_VERSION = "ccts_v2_transition_ranker.program_context.v1"
HARD_NEGATIVE_MAX_RANK = 10


@dataclass
class ProgramEvidence:
    by_transition_id: dict[str, dict[str, Any]]
    transition_items: list[dict[str, Any]]
    transition_fp_index: dict[str, dict[str, Any]]
    graph: dict[str, Any]


def train_ccts_v2_transition_ranker(
    *,
    train_coverage: Path,
    train_cache: Path,
    val_coverage: Path,
    val_cache: Path,
    test_coverage: Path,
    test_cache: Path,
    program_manifest: Path,
    output_dir: Path,
    similarity_threshold: float = 0.7,
    max_candidates_per_transition: int = 100,
    n_estimators: int = 360,
    learning_rate: float = 0.035,
    seed: int = 42,
) -> dict[str, Any]:
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
    schema = _build_schema(train_transitions, train_cache_rows, program_evidence)

    train_data = _build_v2_dataset(
        train_transitions,
        train_cache_rows,
        evidence=program_evidence,
        schema=schema,
        similarity_threshold=similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        require_trainable_group=True,
    )
    val_data = _build_v2_dataset(
        val_transitions,
        val_cache_rows,
        evidence=program_evidence,
        schema=schema,
        similarity_threshold=similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        require_trainable_group=False,
    )
    test_data = _build_v2_dataset(
        test_transitions,
        test_cache_rows,
        evidence=program_evidence,
        schema=schema,
        similarity_threshold=similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        require_trainable_group=False,
    )
    if not train_data.rows or not val_data.rows or not test_data.rows:
        raise ValueError("CCTS-v2 requires non-empty train/val/test candidate datasets")

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
        "train": _evaluate_dataset(train_data, _baseline_scores(train_data.rows)),
        "val": _evaluate_dataset(val_data, _baseline_scores(val_data.rows)),
        "test": _evaluate_dataset(test_data, _baseline_scores(test_data.rows)),
    }
    hard_negative_reports = _hard_negative_reports(
        test_data,
        {"chem_rank": _baseline_scores(test_data.rows)}
        | {
            name: model.predict(test_data.x[:, indices], num_iteration=model.best_iteration_)
            for name, model in models.items()
            for indices in [model_specs[name]]
        },
    )
    blends = _blend_reports(train_data=train_data, val_data=val_data, test_data=test_data, models=models, model_specs=model_specs)
    leakage = _coverage_leakage_report({"train": train_transitions, "val": val_transitions, "test": test_transitions})
    result = {
        "schema_version": CCTS_V2_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_coverage": str(train_coverage),
            "val_coverage": str(val_coverage),
            "test_coverage": str(test_coverage),
            "program_manifest": str(program_manifest),
            "output_dir": str(output_dir),
            "similarity_threshold": similarity_threshold,
            "max_candidates_per_transition": max_candidates_per_transition,
            "hard_negative_max_rank": HARD_NEGATIVE_MAX_RANK,
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
            "train_positive_rows": int(train_data.y.sum()),
            "val_positive_rows": int(val_data.y.sum()),
            "test_positive_rows": int(test_data.y.sum()),
            "test_hard_negative_rows": sum(1 for row in test_data.rows if row.get("hard_negative_label")),
        },
        "leakage_checks": leakage,
        "baseline_chem_rank": baselines,
        "models": reports,
        "blends": blends,
        "hard_negative_reports": hard_negative_reports,
        "feature_schema": {
            "feature_names": train_data.feature_names,
            "model_specs": {name: [train_data.feature_names[idx] for idx in indices] for name, indices in model_specs.items()},
            **schema,
        },
    }
    with (output_dir / "ccts_v2_models.pkl").open("wb") as fh:
        pickle.dump(
            {
                "schema_version": CCTS_V2_SCHEMA_VERSION,
                "models": models,
                "feature_schema": result["feature_schema"],
                "metadata": result["metadata"],
            },
            fh,
        )
    (output_dir / "ccts_v2_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "ccts_v2_report.md").write_text(_markdown(result), encoding="utf-8")
    _write_jsonl(output_dir / "ccts_v2_test_candidates.jsonl", _compact_rows(test_data.rows))
    return result


def _build_v2_dataset(
    transitions: list[dict[str, Any]],
    cache: dict[str, Any],
    *,
    evidence: ProgramEvidence,
    schema: dict[str, Any],
    similarity_threshold: float,
    max_candidates_per_transition: int,
    require_trainable_group: bool,
) -> CandidateDataset:
    rows = []
    x_rows = []
    y_rows = []
    groups = []
    group_ids = []
    feature_names = _feature_names(schema)
    for transition in transitions:
        product = str(transition.get("product_smiles") or "")
        candidates = _candidate_rows_from_cache(cache, product)[: int(max_candidates_per_transition)]
        context = _transition_context(transition, evidence)
        group_rows = []
        group_x = []
        group_y = []
        for rank, candidate in enumerate(candidates, start=1):
            row = _candidate_label_row(transition, candidate, context=context, rank=rank, similarity_threshold=similarity_threshold)
            group_rows.append(row)
            group_x.append(_feature_vector(row, context=context, evidence=evidence, schema=schema))
            group_y.append(int(row["positive_label"]))
        if not group_rows:
            continue
        if require_trainable_group and (sum(group_y) == 0 or sum(group_y) == len(group_y)):
            continue
        rows.extend(group_rows)
        x_rows.extend(group_x)
        y_rows.extend(group_y)
        groups.append(len(group_rows))
        group_ids.append(str(transition.get("transition_id") or ""))
    chem_indices = [idx for idx, name in enumerate(feature_names) if name.startswith("chem__")]
    return CandidateDataset(
        rows=rows,
        x=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.int32),
        group_sizes=groups,
        group_ids=group_ids,
        feature_names=feature_names,
        chem_feature_indices=chem_indices,
    )


def _candidate_label_row(
    transition: dict[str, Any],
    candidate: dict[str, Any],
    *,
    context: dict[str, Any],
    rank: int,
    similarity_threshold: float,
) -> dict[str, Any]:
    gt_rxn = canonical_reaction(str(transition.get("rxn_smiles") or ""))
    gt_reactants = set(str(smi) for smi in transition.get("reactants") or [])
    rxn = canonical_reaction(candidate.get("reaction_smiles") or candidate.get("rxn_smiles") or "")
    cand_reactants = set(canonical_side((rxn.split(">>", 1)[0] if ">>" in rxn else "")))
    cand_main = canonical_smiles(candidate.get("main_reactant")) or str(candidate.get("main_reactant") or "")
    reactant_similarity = _best_set_similarity(gt_reactants, cand_reactants)
    exact = bool(gt_rxn and rxn == gt_rxn)
    similar = bool(reactant_similarity >= similarity_threshold)
    positive = exact or similar
    likely_shortcut = bool(context.get("intermediate_isolated") is False and rank <= HARD_NEGATIVE_MAX_RANK and not positive)
    hard_negative = bool(rank <= HARD_NEGATIVE_MAX_RANK and not positive)
    return {
        "transition_id": transition.get("transition_id"),
        "product_smiles": transition.get("product_smiles"),
        "target_smiles": transition.get("target_smiles"),
        "route_domain": transition.get("route_domain"),
        "candidate_rank": rank,
        "candidate_score": _float_or_zero(candidate.get("score")),
        "candidate_source": str(candidate.get("source") or ""),
        "candidate_model": str(candidate.get("model_full_name") or candidate.get("teacher_source") or ""),
        "candidate_type": str(candidate.get("type") or candidate.get("proposal_type") or ""),
        "candidate_reaction_smiles": rxn,
        "candidate_reactants": sorted(cand_reactants),
        "candidate_main_reactant": cand_main,
        "reactant_similarity": round(reactant_similarity, 6),
        "exact_label": exact,
        "similar_label": similar,
        "positive_label": positive,
        "hard_negative_label": hard_negative,
        "hidden_shortcut_negative_label": likely_shortcut,
    }


def _feature_vector(row: dict[str, Any], *, context: dict[str, Any], evidence: ProgramEvidence, schema: dict[str, Any]) -> list[float]:
    product = str(row.get("product_smiles") or "")
    target = str(row.get("target_smiles") or "")
    main = str(row.get("candidate_main_reactant") or "")
    reactants = [str(smi) for smi in row.get("candidate_reactants") or []]
    product_props = _mol_props(product)
    main_props = _mol_props(main)
    target_props = _mol_props(target)
    rank = max(1, int(row.get("candidate_rank") or 1))
    raw_score = _float_or_zero(row.get("candidate_score"))
    evidence_features = _program_evidence_features(row, context=context, evidence=evidence)
    values = [
        float(rank),
        1.0 / float(rank),
        1.0 / math.log2(rank + 1.0),
        raw_score,
        float(raw_score != 0.0),
        float(len(reactants)),
        float(max(0, len(reactants) - 1)),
        _safe_div(main_props["heavy_atoms"], product_props["heavy_atoms"]),
        product_props["heavy_atoms"] - main_props["heavy_atoms"],
        product_props["rings"] - main_props["rings"],
        product_props["hetero_atoms"] - main_props["hetero_atoms"],
        product_props["heavy_atoms"],
        main_props["heavy_atoms"],
        target_props["heavy_atoms"],
        product_props["mw"] / 500.0,
        main_props["mw"] / 500.0,
        _tanimoto(product, target),
        float(context.get("step_pos") or 0),
        float(context.get("remaining_steps") or 0),
        float(bool(context.get("previous_transform"))),
        float(context.get("intermediate_isolated") is False),
        float(context.get("intermediate_isolated") is True),
        float(len(context.get("catalyst_classes") or [])),
        float(len(context.get("condition_tokens") or [])),
        *evidence_features,
    ]
    for key in ("sources", "models", "candidate_types"):
        values.extend(_one_hot(str(row.get(_row_schema_key(key)) or ""), schema[key]))
    for key, ctx_key in (
        ("route_domains", "route_domain"),
        ("transforms", "transform"),
        ("previous_transforms", "previous_transform"),
        ("next_transforms", "next_transform"),
        ("step_modes", "step_mode"),
        ("pairwise_modes", "pairwise_mode"),
        ("compatibility_labels", "compatibility_label"),
    ):
        values.extend(_one_hot(str(context.get(ctx_key) or ""), schema[key]))
    values.extend(_multi_hot(context.get("catalyst_classes") or [], schema["catalyst_classes"]))
    values.extend(_multi_hot(context.get("ec1_values") or [], schema["ec1_values"]))
    values.extend(_multi_hot(context.get("condition_tokens") or [], schema["condition_tokens"]))
    values.extend(_fp_bits(main, int(schema["n_bits"])))
    values.extend(_fp_bits(".".join(sorted(reactants)), int(schema["n_bits"])))
    return values


def _program_evidence_features(row: dict[str, Any], *, context: dict[str, Any], evidence: ProgramEvidence) -> list[float]:
    graph = evidence.graph
    transform = str(context.get("transform") or "unknown")
    prev = str(context.get("previous_transform") or "")
    nxt = str(context.get("next_transform") or "")
    pair_prev = f"{prev}->{transform}" if prev else ""
    pair_next = f"{transform}->{nxt}" if nxt else ""
    transform_count = float((graph.get("transform_keys") or {}).get(transform, 0))
    prev_pair_count = float((graph.get("adjacency_keys") or {}).get(pair_prev, 0)) if pair_prev else 0.0
    next_pair_count = float((graph.get("adjacency_keys") or {}).get(pair_next, 0)) if pair_next else 0.0
    catalyst_match_count = sum(float((graph.get("catalyst_transform_keys") or {}).get(f"{transform}|{cls}", 0)) for cls in context.get("catalyst_classes") or [])
    condition_match_count = sum(float((graph.get("condition_transform_keys") or {}).get(f"{transform}|{token}", 0)) for token in context.get("condition_tokens") or [])
    hidden_count = float((graph.get("hidden_intermediate_transform_keys") or {}).get(transform, 0))
    product_sim, main_sim, trans_sim = _nearest_program_transition_similarity(row, context=context, evidence=evidence)
    return [
        math.log1p(transform_count),
        math.log1p(prev_pair_count),
        math.log1p(next_pair_count),
        math.log1p(catalyst_match_count),
        math.log1p(condition_match_count),
        math.log1p(hidden_count),
        float(prev_pair_count > 0),
        float(next_pair_count > 0),
        float(catalyst_match_count > 0),
        float(condition_match_count > 0),
        product_sim,
        main_sim,
        trans_sim,
    ]


def _nearest_program_transition_similarity(row: dict[str, Any], *, context: dict[str, Any], evidence: ProgramEvidence) -> tuple[float, float, float]:
    transform = str(context.get("transform") or "")
    product = str(row.get("product_smiles") or "")
    main = str(row.get("candidate_main_reactant") or "")
    bucket = evidence.transition_fp_index.get(transform) or evidence.transition_fp_index.get("")
    if not bucket:
        return (0.0, 0.0, 0.0)
    product_fp = _similarity_fp(product)
    main_fp = _similarity_fp(main)
    product_sims = _bulk_similarity(product_fp, bucket.get("product_fps") or [])
    main_sims = _bulk_similarity(main_fp, bucket.get("main_fps") or [])
    best = (0.0, 0.0, 0.0)
    for prod_sim, react_sim in zip(product_sims, main_sims):
        trans_sim = 0.5 * prod_sim + 0.5 * react_sim
        if trans_sim > best[2]:
            best = (float(prod_sim), float(react_sim), float(trans_sim))
    return best


def _transition_context(transition: dict[str, Any], evidence: ProgramEvidence) -> dict[str, Any]:
    program_row = evidence.by_transition_id.get(str(transition.get("transition_id") or "")) or {}
    return {
        "transition_id": transition.get("transition_id"),
        "route_domain": transition.get("route_domain"),
        "step_pos": transition.get("step_pos"),
        "remaining_steps": transition.get("remaining_steps"),
        "transform": transition.get("transformation_superclass") or program_row.get("transformation_superclass") or "unknown",
        "previous_transform": transition.get("previous_transformation_superclass") or program_row.get("previous_transformation_superclass") or "",
        "next_transform": program_row.get("next_transformation_superclass") or "",
        "step_mode": transition.get("step_mode") or program_row.get("step_mode") or "unknown",
        "pairwise_mode": transition.get("pairwise_mode") or program_row.get("pairwise_mode") or "unknown",
        "intermediate_isolated": transition.get("intermediate_isolated") if transition.get("intermediate_isolated") is not None else program_row.get("intermediate_isolated"),
        "catalyst_classes": transition.get("catalyst_classes") or program_row.get("catalyst_classes") or [],
        "ec1_values": transition.get("ec1_values") or program_row.get("ec1_values") or [],
        "condition_tokens": program_row.get("condition_tokens") or [],
        "compatibility_label": program_row.get("compatibility_label") or "",
    }


def _load_program_evidence(program_manifest: Path) -> ProgramEvidence:
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    rows = []
    for split in ("train", "val", "test"):
        path = outputs.get(split)
        if path:
            rows.extend(_read_jsonl(Path(path)))
    graph = json.loads(Path(outputs["train_evidence_graph"]).read_text(encoding="utf-8"))
    by_transition = {}
    train_transition_items = []
    train_program_ids = {row.get("program_id") for row in _read_jsonl(Path(outputs["train"]))}
    for program in rows:
        compatibility_label = (program.get("compatibility") or {}).get("compatibility_label")
        for step in program.get("steps") or []:
            item = dict(step)
            item["program_id"] = program.get("program_id")
            item["doi"] = program.get("doi")
            item["cascade_id"] = program.get("cascade_id")
            item["route_domain"] = program.get("cascade_type")
            item["compatibility_label"] = compatibility_label
            by_transition[str(step.get("transition_id") or "")] = item
            if program.get("program_id") in train_program_ids:
                train_transition_items.append(
                    {
                        "transition_id": step.get("transition_id"),
                        "transform": step.get("transformation_superclass"),
                        "product_smiles": step.get("product_smiles"),
                        "main_reactant": step.get("main_reactant"),
                    }
                )
    return ProgramEvidence(
        by_transition_id=by_transition,
        transition_items=train_transition_items,
        transition_fp_index=_transition_fp_index(train_transition_items),
        graph=graph,
    )


def _transition_fp_index(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"product_fps": [], "main_fps": []})
    for item in items:
        transform = str(item.get("transform") or "")
        product_fp = _similarity_fp(str(item.get("product_smiles") or ""))
        main_fp = _similarity_fp(str(item.get("main_reactant") or ""))
        if product_fp is None or main_fp is None:
            continue
        grouped[transform]["product_fps"].append(product_fp)
        grouped[transform]["main_fps"].append(main_fp)
        grouped[""]["product_fps"].append(product_fp)
        grouped[""]["main_fps"].append(main_fp)
    return dict(grouped)


def _build_schema(transitions: list[dict[str, Any]], cache: dict[str, Any], evidence: ProgramEvidence, *, n_bits: int = 128) -> dict[str, Any]:
    sources, models, candidate_types = set(), set(), set()
    for transition in transitions:
        for cand in _candidate_rows_from_cache(cache, str(transition.get("product_smiles") or "")):
            sources.add(str(cand.get("source") or ""))
            models.add(str(cand.get("model_full_name") or cand.get("teacher_source") or ""))
            candidate_types.add(str(cand.get("type") or cand.get("proposal_type") or ""))
    contexts = [_transition_context(row, evidence) for row in transitions]
    return {
        "n_bits": int(n_bits),
        "sources": sorted(sources),
        "models": sorted(models),
        "candidate_types": sorted(candidate_types),
        "route_domains": sorted({str(row.get("route_domain") or "") for row in contexts}),
        "transforms": sorted({str(row.get("transform") or "") for row in contexts}),
        "previous_transforms": sorted({str(row.get("previous_transform") or "") for row in contexts}),
        "next_transforms": sorted({str(row.get("next_transform") or "") for row in contexts}),
        "step_modes": sorted({str(row.get("step_mode") or "") for row in contexts}),
        "pairwise_modes": sorted({str(row.get("pairwise_mode") or "") for row in contexts}),
        "compatibility_labels": sorted({str(row.get("compatibility_label") or "") for row in contexts}),
        "catalyst_classes": sorted({str(value) for row in contexts for value in row.get("catalyst_classes") or []}),
        "ec1_values": sorted({str(value) for row in contexts for value in row.get("ec1_values") or []}),
        "condition_tokens": sorted({str(value) for row in contexts for value in row.get("condition_tokens") or []})[:200],
    }


def _feature_names(schema: dict[str, Any]) -> list[str]:
    names = [
        "chem__rank",
        "chem__inv_rank",
        "chem__inv_log_rank",
        "chem__score",
        "chem__has_score",
        "chem__n_reactants",
        "chem__n_aux_reactants",
        "chem__main_to_product_heavy_ratio",
        "chem__product_minus_main_heavy",
        "chem__product_minus_main_rings",
        "chem__product_minus_main_hetero",
        "chem__product_heavy_atoms",
        "chem__main_heavy_atoms",
        "chem__target_heavy_atoms",
        "chem__product_mw_scaled",
        "chem__main_mw_scaled",
        "chem__product_target_similarity",
        "context__step_pos",
        "context__remaining_steps",
        "context__has_previous_transform",
        "context__intermediate_nonisolated",
        "context__intermediate_isolated",
        "context__n_catalyst_classes",
        "context__n_condition_tokens",
        "evidence__log_transform_count",
        "evidence__log_prev_pair_count",
        "evidence__log_next_pair_count",
        "evidence__log_catalyst_match_count",
        "evidence__log_condition_match_count",
        "evidence__log_hidden_transform_count",
        "evidence__has_prev_pair",
        "evidence__has_next_pair",
        "evidence__has_catalyst_match",
        "evidence__has_condition_match",
        "evidence__nearest_product_sim",
        "evidence__nearest_main_sim",
        "evidence__nearest_transition_sim",
    ]
    for key in ("sources", "models", "candidate_types"):
        names.extend([f"chem__{key}={value}" for value in schema[key]])
    for key in ("route_domains", "transforms", "previous_transforms", "next_transforms", "step_modes", "pairwise_modes", "compatibility_labels"):
        names.extend([f"context__{key}={value}" for value in schema[key]])
    for key in ("catalyst_classes", "ec1_values", "condition_tokens"):
        prefix = "compat" if key != "condition_tokens" else "evidence"
        names.extend([f"{prefix}__{key}={value}" for value in schema[key]])
    names.extend([f"chem__main_fp_{idx}" for idx in range(int(schema["n_bits"]))])
    names.extend([f"chem__reactants_fp_{idx}" for idx in range(int(schema["n_bits"]))])
    return names


def _model_specs(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(feature_names):
        if name.startswith("chem__"):
            groups["chem_only"].append(idx)
        if name.startswith("context__"):
            groups["context_only"].append(idx)
        if name.startswith("compat__"):
            groups["compatibility_only"].append(idx)
        if name.startswith("evidence__"):
            groups["evidence_only"].append(idx)
    groups["ccts_v2_no_context"] = groups["chem_only"] + groups["compatibility_only"] + groups["evidence_only"]
    groups["ccts_v2_full"] = list(range(len(feature_names)))
    return dict(groups)


def _fit_ranker(train_data: CandidateDataset, val_data: CandidateDataset, *, feature_indices: list[int], n_estimators: int, learning_rate: float, seed: int) -> LGBMRanker:
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=15,
        min_child_samples=24,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=3.0,
        reg_alpha=0.1,
        random_state=int(seed),
        verbose=-1,
    )
    model.fit(
        train_data.x[:, feature_indices],
        train_data.y,
        group=train_data.group_sizes,
        eval_set=[(val_data.x[:, feature_indices], val_data.y)],
        eval_group=[val_data.group_sizes],
        eval_at=[1, 3, 5, 10],
        callbacks=[early_stopping(30, verbose=False)],
    )
    return model


def _model_report(model: Any, *, train_data: CandidateDataset, val_data: CandidateDataset, test_data: CandidateDataset, feature_indices: list[int]) -> dict[str, Any]:
    return {
        "train": _evaluate_dataset(train_data, model.predict(train_data.x[:, feature_indices], num_iteration=model.best_iteration_)),
        "val": _evaluate_dataset(val_data, model.predict(val_data.x[:, feature_indices], num_iteration=model.best_iteration_)),
        "test": _evaluate_dataset(test_data, model.predict(test_data.x[:, feature_indices], num_iteration=model.best_iteration_)),
        "feature_count": len(feature_indices),
        "feature_names": [train_data.feature_names[idx] for idx in feature_indices],
        "best_iteration": int(model.best_iteration_ or 0),
    }


def _blend_reports(*, train_data: CandidateDataset, val_data: CandidateDataset, test_data: CandidateDataset, models: dict[str, Any], model_specs: dict[str, list[int]]) -> dict[str, Any]:
    del train_data
    if "chem_only" not in models:
        return {}
    base = models["chem_only"]
    base_idx = model_specs["chem_only"]
    base_val = base.predict(val_data.x[:, base_idx], num_iteration=base.best_iteration_)
    base_test = base.predict(test_data.x[:, base_idx], num_iteration=base.best_iteration_)
    out = {}
    for aux in ("context_only", "compatibility_only", "evidence_only", "ccts_v2_full"):
        if aux not in models:
            continue
        aux_model = models[aux]
        aux_idx = model_specs[aux]
        aux_val = aux_model.predict(val_data.x[:, aux_idx], num_iteration=aux_model.best_iteration_)
        aux_test = aux_model.predict(test_data.x[:, aux_idx], num_iteration=aux_model.best_iteration_)
        best_alpha = 0.0
        best_score = _metric_for_selection(_evaluate_dataset(val_data, base_val))
        for alpha in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]:
            score = _metric_for_selection(_evaluate_dataset(val_data, _standardize(base_val) + alpha * _standardize(aux_val)))
            if score > best_score:
                best_alpha = float(alpha)
                best_score = score
        out[f"chem_only_plus_{aux}"] = {
            "base_model": "chem_only",
            "aux_model": aux,
            "alpha_selected_on_val": best_alpha,
            "val_selection_score": round(best_score, 6),
            "test": _evaluate_dataset(test_data, _standardize(base_test) + best_alpha * _standardize(aux_test)),
        }
    return out


def _hard_negative_reports(dataset: CandidateDataset, score_columns: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {name: {"groups": 0, "wins": 0, "ties": 0, "losses": 0} for name in score_columns}
    offset = 0
    for group_size in dataset.group_sizes:
        rows = dataset.rows[offset : offset + group_size]
        positives = [idx for idx, row in enumerate(rows) if row.get("positive_label")]
        hard_negs = [idx for idx, row in enumerate(rows) if row.get("hard_negative_label")]
        if positives and hard_negs:
            for name, scores in score_columns.items():
                g_scores = scores[offset : offset + group_size]
                best_pos = max(float(g_scores[idx]) for idx in positives)
                best_neg = max(float(g_scores[idx]) for idx in hard_negs)
                out[name]["groups"] += 1
                if best_pos > best_neg:
                    out[name]["wins"] += 1
                elif best_pos == best_neg:
                    out[name]["ties"] += 1
                else:
                    out[name]["losses"] += 1
        offset += group_size
    for row in out.values():
        row["win_rate"] = round(row["wins"] / max(row["groups"], 1), 6)
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_schema_key(key: str) -> str:
    return {"sources": "candidate_source", "models": "candidate_model", "candidate_types": "candidate_type"}[key]


def _one_hot(value: str, values: list[str]) -> list[float]:
    return [float(value == item) for item in values]


def _multi_hot(values: list[str], schema_values: list[str]) -> list[float]:
    present = {str(value) for value in values}
    return [float(value in present) for value in schema_values]


def _fp_bits(smiles: str, n_bits: int) -> list[float]:
    cached = _fp_bits_cached(str(smiles or ""), int(n_bits))
    return [float(v) for v in cached]


@lru_cache(maxsize=200000)
def _fp_bits_cached(smiles: str, n_bits: int) -> tuple[int, ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tuple(0 for _ in range(n_bits))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return tuple(int(v) for v in arr)


@lru_cache(maxsize=200000)
def _mol_props(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {"heavy_atoms": 0.0, "rings": 0.0, "hetero_atoms": 0.0, "mw": 0.0}
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    return {
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "rings": float(mol.GetRingInfo().NumRings()),
        "hetero_atoms": float(hetero),
        "mw": float(Descriptors.MolWt(mol)),
    }


def _tanimoto(a: str, b: str) -> float:
    fp_a = _similarity_fp(str(a or ""))
    fp_b = _similarity_fp(str(b or ""))
    if fp_a is None or fp_b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _bulk_similarity(fp: Any, fps: list[Any]) -> list[float]:
    if fp is None or not fps:
        return [0.0 for _ in fps]
    return [float(value) for value in DataStructs.BulkTanimotoSimilarity(fp, fps)]


@lru_cache(maxsize=200000)
def _similarity_fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep_keys = [
        "transition_id",
        "product_smiles",
        "candidate_rank",
        "candidate_score",
        "candidate_source",
        "candidate_model",
        "reactant_similarity",
        "exact_label",
        "similar_label",
        "positive_label",
        "hard_negative_label",
        "hidden_shortcut_negative_label",
    ]
    return [{key: row.get(key) for key in keep_keys} for row in rows]


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CCTS-v2 Transition Ranker",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        f"- strict leakage pass: `{(result.get('leakage_checks') or {}).get('strict_pass')}`",
        "",
        "## Test Metrics",
        "",
        "| Model | Label | Coverage | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [("chem_rank", (result.get("baseline_chem_rank") or {}).get("test") or {})]
    rows.extend((name, report.get("test") or {}) for name, report in (result.get("models") or {}).items())
    rows.extend((name, report.get("test") or {}) for name, report in (result.get("blends") or {}).items())
    for name, report in rows:
        for label in ("exact_label", "similar_label", "positive_label"):
            metric = report.get(label) or {}
            at = metric.get("recall_at_k_all") or {}
            lines.append(
                "| " + " | ".join([name, label, str(metric.get("coverage")), str(metric.get("mrr_covered")), str(at.get("1")), str(at.get("3")), str(at.get("5")), str(at.get("10"))]) + " |"
            )
    lines.extend(["", "## Hard Negative Win Rate", "", "| Model | Groups | Win Rate |", "|---|---:|---:|"])
    for name, row in (result.get("hard_negative_reports") or {}).items():
        lines.append(f"| {name} | {row.get('groups')} | {row.get('win_rate')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Train CCTS-v2 with program context and hard-negative reports")
    ap.add_argument("--train-coverage", required=True)
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-coverage", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--test-coverage", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--similarity-threshold", type=float, default=0.7)
    ap.add_argument("--max-candidates-per-transition", type=int, default=100)
    ap.add_argument("--n-estimators", type=int, default=360)
    ap.add_argument("--learning-rate", type=float, default=0.035)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    result = train_ccts_v2_transition_ranker(
        train_coverage=Path(args.train_coverage),
        train_cache=Path(args.train_cache),
        val_coverage=Path(args.val_coverage),
        val_cache=Path(args.val_cache),
        test_coverage=Path(args.test_coverage),
        test_cache=Path(args.test_cache),
        program_manifest=Path(args.program_manifest),
        output_dir=Path(args.output_dir),
        similarity_threshold=args.similarity_threshold,
        max_candidates_per_transition=args.max_candidates_per_transition,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps({"counts": result["counts"], "leakage_checks": result["leakage_checks"], "hard_negative_reports": result["hard_negative_reports"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Train CCTS-v0: a cascade-conditioned transition ranker.

This first version intentionally does not generate reactions.  It reranks a
fixed ChemEnzy one-step candidate pool for each v4 cascade transition and
tests whether v4 cascade evidence can move observed/similar transitions upward.
"""
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


CCTS_SCHEMA_VERSION = "ccts_v0_transition_ranker.v1"


@dataclass
class CandidateDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    group_sizes: list[int]
    group_ids: list[str]
    feature_names: list[str]
    chem_feature_indices: list[int]


@dataclass
class EvidenceItem:
    transition_id: str
    product: str
    main_reactant: str
    reactants: list[str]
    previous_transform: str
    remaining_bucket: str
    step_pos_bucket: str
    quality_tier: str


def train_ccts_v0_transition_ranker(
    *,
    train_coverage: Path,
    train_cache: Path,
    val_coverage: Path,
    val_cache: Path,
    test_coverage: Path,
    test_cache: Path,
    output_dir: Path,
    similarity_threshold: float = 0.7,
    max_candidates_per_transition: int = 100,
    evidence_pool_size: int = 80,
    n_estimators: int = 240,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_payload = _read_json(train_coverage)
    val_payload = _read_json(val_coverage)
    test_payload = _read_json(test_coverage)
    train_cache_rows = _read_json(train_cache)
    val_cache_rows = _read_json(val_cache)
    test_cache_rows = _read_json(test_cache)

    train_transitions = [row for row in train_payload.get("transitions") or [] if isinstance(row, dict)]
    val_transitions = [row for row in val_payload.get("transitions") or [] if isinstance(row, dict)]
    test_transitions = [row for row in test_payload.get("transitions") or [] if isinstance(row, dict)]
    evidence_bank = _build_evidence_bank(train_transitions)
    schema = _build_feature_schema(train_transitions, train_cache_rows)

    train_data = _build_candidate_dataset(
        train_transitions,
        train_cache_rows,
        evidence_bank=evidence_bank,
        schema=schema,
        similarity_threshold=similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        evidence_pool_size=evidence_pool_size,
        require_trainable_group=True,
    )
    val_data = _build_candidate_dataset(
        val_transitions,
        val_cache_rows,
        evidence_bank=evidence_bank,
        schema=schema,
        similarity_threshold=similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        evidence_pool_size=evidence_pool_size,
        require_trainable_group=False,
    )
    test_data = _build_candidate_dataset(
        test_transitions,
        test_cache_rows,
        evidence_bank=evidence_bank,
        schema=schema,
        similarity_threshold=similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        evidence_pool_size=evidence_pool_size,
        require_trainable_group=False,
    )
    if not train_data.rows:
        raise ValueError("no trainable CCTS candidate rows")
    if not val_data.rows:
        raise ValueError("no validation candidate rows")

    models: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    feature_groups = _feature_groups(train_data.feature_names)
    model_specs = [
        ("chem_scalar", feature_groups["chem_scalar"]),
        ("chem_only", train_data.chem_feature_indices),
        ("context_evidence_only", feature_groups["ccts_scalar"]),
        ("chem_scalar_plus_context_evidence", feature_groups["chem_scalar"] + feature_groups["ccts_scalar"]),
        ("ccts_evidence", list(range(train_data.x.shape[1]))),
    ]
    for model_name, feature_indices in model_specs:
        if not feature_indices:
            continue
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
        x_train = train_data.x[:, feature_indices]
        x_val = val_data.x[:, feature_indices]
        model.fit(
            x_train,
            train_data.y,
            group=train_data.group_sizes,
            eval_set=[(x_val, val_data.y)],
            eval_group=[val_data.group_sizes],
            eval_at=[1, 3, 5, 10],
            callbacks=[early_stopping(30, verbose=False)],
        )
        models[model_name] = model
        train_scores = model.predict(x_train, num_iteration=model.best_iteration_)
        val_scores = model.predict(x_val, num_iteration=model.best_iteration_)
        test_scores = model.predict(test_data.x[:, feature_indices], num_iteration=model.best_iteration_)
        reports[model_name] = {
            "train": _evaluate_dataset(train_data, train_scores),
            "val": _evaluate_dataset(val_data, val_scores),
            "test": _evaluate_dataset(test_data, test_scores),
            "feature_count": len(feature_indices),
            "feature_names": [train_data.feature_names[idx] for idx in feature_indices],
            "best_iteration": int(model.best_iteration_ or 0),
        }
    blend_reports = _build_blend_reports(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        reports=reports,
        models=models,
        model_specs={name: indices for name, indices in model_specs},
    )

    baseline_reports = {
        "train": _evaluate_dataset(train_data, _baseline_scores(train_data.rows)),
        "val": _evaluate_dataset(val_data, _baseline_scores(val_data.rows)),
        "test": _evaluate_dataset(test_data, _baseline_scores(test_data.rows)),
    }
    result = {
        "schema_version": CCTS_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_coverage": str(train_coverage),
            "val_coverage": str(val_coverage),
            "test_coverage": str(test_coverage),
            "train_cache": str(train_cache),
            "val_cache": str(val_cache),
            "test_cache": str(test_cache),
            "output_dir": str(output_dir),
            "similarity_threshold": similarity_threshold,
            "max_candidates_per_transition": max_candidates_per_transition,
            "evidence_pool_size": evidence_pool_size,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "seed": seed,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            "train_transitions": len(train_transitions),
            "val_transitions": len(val_transitions),
            "test_transitions": len(test_transitions),
            "evidence_bank": len(evidence_bank),
            "train_candidate_rows": len(train_data.rows),
            "val_candidate_rows": len(val_data.rows),
            "test_candidate_rows": len(test_data.rows),
            "train_groups": len(train_data.group_sizes),
            "val_groups": len(val_data.group_sizes),
            "test_groups": len(test_data.group_sizes),
            "train_positive_rows": int(train_data.y.sum()),
            "val_positive_rows": int(val_data.y.sum()),
            "test_positive_rows": int(test_data.y.sum()),
        },
        "baseline_chem_rank": baseline_reports,
        "models": reports,
        "blends": blend_reports,
        "feature_schema": {
            "feature_names": train_data.feature_names,
            "chem_feature_names": [train_data.feature_names[idx] for idx in train_data.chem_feature_indices],
            **schema,
        },
    }

    with (output_dir / "ccts_v0_models.pkl").open("wb") as fh:
        pickle.dump(
            {
                "schema_version": CCTS_SCHEMA_VERSION,
                "models": models,
                "feature_schema": result["feature_schema"],
                "metadata": result["metadata"],
            },
            fh,
        )
    (output_dir / "ccts_v0_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "ccts_v0_report.md").write_text(_markdown(result), encoding="utf-8")
    _write_jsonl(output_dir / "ccts_v0_test_candidates.jsonl", _compact_candidate_rows(test_data.rows))
    return result


def _build_candidate_dataset(
    transitions: list[dict[str, Any]],
    cache: dict[str, Any],
    *,
    evidence_bank: list[EvidenceItem],
    schema: dict[str, Any],
    similarity_threshold: float,
    max_candidates_per_transition: int,
    evidence_pool_size: int,
    require_trainable_group: bool,
) -> CandidateDataset:
    rows: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    group_sizes: list[int] = []
    group_ids: list[str] = []
    evidence_lookup = {
        product: _product_evidence_pool(product, evidence_bank, limit=evidence_pool_size)
        for product in sorted({str(row.get("product_smiles") or "") for row in transitions if row.get("product_smiles")})
    }
    for transition in transitions:
        product = str(transition.get("product_smiles") or "")
        candidates = _candidate_rows_from_cache(cache, product)[: int(max_candidates_per_transition)]
        group_feature_rows = []
        group_labels = []
        group_meta = []
        for idx, candidate in enumerate(candidates, 1):
            row = _candidate_label_row(
                transition,
                candidate,
                rank=idx,
                similarity_threshold=similarity_threshold,
            )
            features = _feature_vector(
                transition,
                row,
                evidence_pool=evidence_lookup.get(product, []),
                schema=schema,
            )
            group_feature_rows.append(features)
            group_labels.append(int(row["positive_label"]))
            group_meta.append(row)
        if not group_meta:
            continue
        if require_trainable_group and (sum(group_labels) == 0 or sum(group_labels) == len(group_labels)):
            continue
        rows.extend(group_meta)
        x_rows.extend(group_feature_rows)
        y_rows.extend(group_labels)
        group_sizes.append(len(group_meta))
        group_ids.append(str(transition.get("transition_id") or ""))
    feature_names = _feature_names(schema)
    chem_feature_indices = [idx for idx, name in enumerate(feature_names) if name.startswith("chem__")]
    return CandidateDataset(
        rows=rows,
        x=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.int32),
        group_sizes=group_sizes,
        group_ids=group_ids,
        feature_names=feature_names,
        chem_feature_indices=chem_feature_indices,
    )


def _candidate_label_row(
    transition: dict[str, Any],
    candidate: dict[str, Any],
    *,
    rank: int,
    similarity_threshold: float,
) -> dict[str, Any]:
    gt_rxn = canonical_reaction(str(transition.get("rxn_smiles") or ""))
    gt_reactants = set(str(smi) for smi in transition.get("reactants") or [])
    gt_main = str(transition.get("main_reactant") or "")
    rxn = canonical_reaction(candidate.get("reaction_smiles") or candidate.get("rxn_smiles") or "")
    cand_reactants = set(canonical_side((rxn.split(">>", 1)[0] if ">>" in rxn else "")))
    cand_main = canonical_smiles(candidate.get("main_reactant")) or str(candidate.get("main_reactant") or "")
    reactant_similarity = _best_set_similarity(gt_reactants, cand_reactants)
    exact = bool(gt_rxn and rxn == gt_rxn)
    reactant_set = bool(gt_reactants and cand_reactants == gt_reactants)
    main_hit = bool(gt_main and cand_main == gt_main)
    any_hit = bool(gt_reactants & cand_reactants)
    similar = bool(reactant_similarity >= similarity_threshold)
    return {
        "transition_id": transition.get("transition_id"),
        "target_smiles": transition.get("target_smiles"),
        "product_smiles": transition.get("product_smiles"),
        "route_domain": transition.get("route_domain"),
        "step_pos": transition.get("step_pos"),
        "remaining_steps": transition.get("remaining_steps"),
        "previous_transformation_superclass": transition.get("previous_transformation_superclass") or "",
        "candidate_rank": rank,
        "candidate_score": _float_or_zero(candidate.get("score")),
        "candidate_source": str(candidate.get("source") or ""),
        "candidate_model": str(candidate.get("model_full_name") or candidate.get("teacher_source") or ""),
        "candidate_type": str(candidate.get("type") or candidate.get("proposal_type") or ""),
        "candidate_reaction_smiles": rxn,
        "candidate_reactants": sorted(cand_reactants),
        "candidate_main_reactant": cand_main,
        "exact_label": exact,
        "reactant_set_label": reactant_set,
        "main_reactant_label": main_hit,
        "any_reactant_label": any_hit,
        "similar_label": similar,
        "reactant_similarity": round(reactant_similarity, 6),
        "positive_label": bool(exact or similar),
    }


def _feature_vector(
    transition: dict[str, Any],
    candidate_row: dict[str, Any],
    *,
    evidence_pool: list[tuple[float, EvidenceItem]],
    schema: dict[str, Any],
) -> list[float]:
    product = str(candidate_row.get("product_smiles") or transition.get("product_smiles") or "")
    target = str(candidate_row.get("target_smiles") or transition.get("target_smiles") or "")
    main = str(candidate_row.get("candidate_main_reactant") or "")
    reactants = [str(smi) for smi in candidate_row.get("candidate_reactants") or []]
    product_props = _mol_props(product)
    target_props = _mol_props(target)
    main_props = _mol_props(main)
    rank = max(1, int(candidate_row.get("candidate_rank") or 1))
    raw_score = _float_or_zero(candidate_row.get("candidate_score"))
    prev_transform = str(candidate_row.get("previous_transformation_superclass") or "")
    evidence = _evidence_features(
        product=product,
        main_reactant=main,
        reactants=reactants,
        previous_transform=prev_transform,
        remaining_bucket=_bucket_int(candidate_row.get("remaining_steps")),
        step_pos_bucket=_bucket_int(candidate_row.get("step_pos")),
        evidence_pool=evidence_pool,
    )
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
        target_props["heavy_atoms"],
        product_props["heavy_atoms"],
        main_props["heavy_atoms"],
        product_props["rings"],
        main_props["rings"],
        product_props["hetero_atoms"],
        main_props["hetero_atoms"],
        product_props["mw"] / 500.0,
        main_props["mw"] / 500.0,
        _tanimoto(product, target),
        float(candidate_row.get("step_pos") or 0),
        float(candidate_row.get("remaining_steps") or 0),
        float(bool(prev_transform)),
        *evidence,
    ]
    values.extend(_one_hot(str(candidate_row.get("candidate_source") or ""), schema["sources"]))
    values.extend(_one_hot(str(candidate_row.get("candidate_model") or ""), schema["models"]))
    values.extend(_one_hot(str(candidate_row.get("candidate_type") or ""), schema["candidate_types"]))
    values.extend(_one_hot(prev_transform, schema["previous_transforms"]))
    values.extend(_one_hot(_bucket_int(candidate_row.get("remaining_steps")), schema["remaining_buckets"]))
    values.extend(_one_hot(_bucket_int(candidate_row.get("step_pos")), schema["step_pos_buckets"]))
    values.extend(_fp_bits(main, int(schema["n_bits"])))
    values.extend(_fp_bits(".".join(sorted(reactants)), int(schema["n_bits"])))
    return values


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
        "chem__target_heavy_atoms",
        "chem__product_heavy_atoms",
        "chem__main_heavy_atoms",
        "chem__product_rings",
        "chem__main_rings",
        "chem__product_hetero_atoms",
        "chem__main_hetero_atoms",
        "chem__product_mw_scaled",
        "chem__main_mw_scaled",
        "chem__product_target_similarity",
        "chem__step_pos",
        "chem__remaining_steps",
        "chem__has_previous_transform",
        "ccts__evidence_max_transition_sim",
        "ccts__evidence_mean_top3_transition_sim",
        "ccts__evidence_max_product_sim",
        "ccts__evidence_max_main_sim",
        "ccts__evidence_count_transition_sim_ge_070",
        "ccts__same_prev_max_transition_sim",
        "ccts__same_remaining_max_transition_sim",
        "ccts__same_step_pos_max_transition_sim",
        "ccts__top_evidence_gold",
        "ccts__top_evidence_silver",
    ]
    for key in ("sources", "models", "candidate_types", "previous_transforms", "remaining_buckets", "step_pos_buckets"):
        for value in schema[key]:
            prefix = "chem" if key in {"sources", "models", "candidate_types"} else "ccts"
            names.append(f"{prefix}__{key}={value}")
    names.extend([f"chem__main_fp_{idx}" for idx in range(int(schema["n_bits"]))])
    names.extend([f"chem__reactants_fp_{idx}" for idx in range(int(schema["n_bits"]))])
    return names


def _feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(feature_names):
        if name.startswith("chem__"):
            groups["chem_all"].append(idx)
            if "_fp_" not in name:
                groups["chem_scalar"].append(idx)
        if name.startswith("ccts__"):
            groups["ccts_scalar"].append(idx)
    return groups


def _build_blend_reports(
    *,
    train_data: CandidateDataset,
    val_data: CandidateDataset,
    test_data: CandidateDataset,
    reports: dict[str, Any],
    models: dict[str, Any],
    model_specs: dict[str, list[int]],
) -> dict[str, Any]:
    out = {}
    if "chem_only" not in models:
        return out
    base_model = models["chem_only"]
    base_indices = model_specs["chem_only"]
    base_val = base_model.predict(val_data.x[:, base_indices], num_iteration=base_model.best_iteration_)
    base_test = base_model.predict(test_data.x[:, base_indices], num_iteration=base_model.best_iteration_)
    for aux_name in ("context_evidence_only", "chem_scalar_plus_context_evidence", "ccts_evidence"):
        if aux_name not in models:
            continue
        aux_model = models[aux_name]
        aux_indices = model_specs[aux_name]
        aux_val = aux_model.predict(val_data.x[:, aux_indices], num_iteration=aux_model.best_iteration_)
        aux_test = aux_model.predict(test_data.x[:, aux_indices], num_iteration=aux_model.best_iteration_)
        best_alpha = 0.0
        best_score = _metric_for_selection(_evaluate_dataset(val_data, base_val))
        candidates = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
        for alpha in candidates:
            blended_val = _standardize(base_val) + float(alpha) * _standardize(aux_val)
            score = _metric_for_selection(_evaluate_dataset(val_data, blended_val))
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
        blended_test = _standardize(base_test) + best_alpha * _standardize(aux_test)
        out[f"chem_only_plus_{aux_name}"] = {
            "aux_model": aux_name,
            "alpha_selected_on_val": best_alpha,
            "val_selection_score": round(best_score, 6),
            "test": _evaluate_dataset(test_data, blended_test),
        }
    return out


def _metric_for_selection(report: dict[str, Any]) -> float:
    pos = report.get("positive_label") or {}
    exact = report.get("exact_label") or {}
    pos_k = pos.get("recall_at_k_all") or {}
    exact_k = exact.get("recall_at_k_all") or {}
    return (
        float(pos.get("mrr_covered") or 0.0)
        + 0.8 * float(pos_k.get("5") or 0.0)
        + 0.4 * float(exact.get("mrr_covered") or 0.0)
        + 0.3 * float(exact_k.get("5") or 0.0)
    )


def _standardize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    std = float(scores.std())
    if std < 1e-9:
        return scores * 0.0
    return (scores - float(scores.mean())) / std


def _build_feature_schema(transitions: list[dict[str, Any]], cache: dict[str, Any], *, n_bits: int = 128) -> dict[str, Any]:
    sources = set()
    models = set()
    candidate_types = set()
    for transition in transitions:
        for cand in _candidate_rows_from_cache(cache, str(transition.get("product_smiles") or "")):
            sources.add(str(cand.get("source") or ""))
            models.add(str(cand.get("model_full_name") or cand.get("teacher_source") or ""))
            candidate_types.add(str(cand.get("type") or cand.get("proposal_type") or ""))
    previous_transforms = {str(row.get("previous_transformation_superclass") or "") for row in transitions}
    remaining_buckets = {_bucket_int(row.get("remaining_steps")) for row in transitions}
    step_pos_buckets = {_bucket_int(row.get("step_pos")) for row in transitions}
    return {
        "n_bits": int(n_bits),
        "sources": sorted(sources),
        "models": sorted(models),
        "candidate_types": sorted(candidate_types),
        "previous_transforms": sorted(previous_transforms),
        "remaining_buckets": sorted(remaining_buckets),
        "step_pos_buckets": sorted(step_pos_buckets),
    }


def _build_evidence_bank(transitions: list[dict[str, Any]]) -> list[EvidenceItem]:
    bank = []
    for row in transitions:
        bank.append(
            EvidenceItem(
                transition_id=str(row.get("transition_id") or ""),
                product=str(row.get("product_smiles") or ""),
                main_reactant=str(row.get("main_reactant") or ""),
                reactants=[str(smi) for smi in row.get("reactants") or []],
                previous_transform=str(row.get("previous_transformation_superclass") or ""),
                remaining_bucket=_bucket_int(row.get("remaining_steps")),
                step_pos_bucket=_bucket_int(row.get("step_pos")),
                quality_tier=str(row.get("quality_tier") or ""),
            )
        )
    return bank


def _evidence_features(
    *,
    product: str,
    main_reactant: str,
    reactants: list[str],
    previous_transform: str,
    remaining_bucket: str,
    step_pos_bucket: str,
    evidence_pool: list[tuple[float, EvidenceItem]],
) -> list[float]:
    scored = []
    for prod_sim, item in evidence_pool:
        main_sim = max(_tanimoto(main_reactant, item.main_reactant), _best_set_similarity(set(reactants), set(item.reactants)))
        transition_sim = 0.5 * prod_sim + 0.5 * main_sim
        scored.append((transition_sim, prod_sim, main_sim, item))
    if not scored:
        return [0.0] * 10
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[0]
    top3 = scored[:3]
    same_prev = [row[0] for row in scored if row[3].previous_transform == previous_transform]
    same_remaining = [row[0] for row in scored if row[3].remaining_bucket == remaining_bucket]
    same_step = [row[0] for row in scored if row[3].step_pos_bucket == step_pos_bucket]
    return [
        float(top[0]),
        float(sum(row[0] for row in top3) / len(top3)),
        float(max(row[1] for row in scored)),
        float(max(row[2] for row in scored)),
        float(sum(1 for row in scored if row[0] >= 0.70)),
        float(max(same_prev) if same_prev else 0.0),
        float(max(same_remaining) if same_remaining else 0.0),
        float(max(same_step) if same_step else 0.0),
        float(top[3].quality_tier == "gold"),
        float(top[3].quality_tier == "silver"),
    ]


def _product_evidence_pool(product: str, evidence_bank: list[EvidenceItem], *, limit: int) -> list[tuple[float, EvidenceItem]]:
    scored = [(_tanimoto(product, item.product), item) for item in evidence_bank]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[: max(1, int(limit))]


def _evaluate_dataset(dataset: CandidateDataset, scores: np.ndarray) -> dict[str, Any]:
    grouped = []
    offset = 0
    for group_size, group_id in zip(dataset.group_sizes, dataset.group_ids):
        rows = dataset.rows[offset : offset + group_size]
        group_scores = scores[offset : offset + group_size]
        grouped.append((group_id, rows, group_scores))
        offset += group_size
    metrics = {
        "groups": len(grouped),
        "candidate_rows": len(dataset.rows),
        "positive_rows": int(dataset.y.sum()),
    }
    for label_name in ("positive_label", "exact_label", "similar_label"):
        metrics[label_name] = _ranking_metrics(grouped, label_name)
    return metrics


def _ranking_metrics(grouped: list[tuple[str, list[dict[str, Any]], np.ndarray]], label_name: str) -> dict[str, Any]:
    covered = 0
    reciprocal = []
    recalls = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    all_recalls = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    rank_counts = Counter()
    for _, rows, scores in grouped:
        order = sorted(range(len(rows)), key=lambda idx: (-float(scores[idx]), int(rows[idx].get("candidate_rank") or 10**9)))
        positive_positions = [pos + 1 for pos, idx in enumerate(order) if rows[idx].get(label_name)]
        if positive_positions:
            covered += 1
            rank = min(positive_positions)
            reciprocal.append(1.0 / rank)
            rank_counts[_rank_bucket(rank)] += 1
            for k in recalls:
                if rank <= k:
                    recalls[k] += 1
                    all_recalls[k] += 1
        else:
            rank_counts["missing"] += 1
    total = max(1, len(grouped))
    covered_den = max(1, covered)
    return {
        "covered_groups": covered,
        "coverage": round(covered / total, 6),
        "mrr_covered": round(sum(reciprocal) / covered_den, 6) if reciprocal else 0.0,
        "recall_at_k_all": {str(k): round(v / total, 6) for k, v in all_recalls.items()},
        "recall_at_k_covered": {str(k): round(v / covered_den, 6) for k, v in recalls.items()},
        "first_positive_rank_buckets": dict(sorted(rank_counts.items())),
    }


def _baseline_scores(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([-float(row.get("candidate_rank") or 10**9) for row in rows], dtype=np.float32)


def _candidate_rows_from_cache(cache: dict[str, Any], product: str) -> list[dict[str, Any]]:
    if not product:
        return []
    for raw_key, rows in cache.items():
        try:
            key = json.loads(raw_key)
        except Exception:
            continue
        if key.get("product") == product:
            return [row for row in rows if isinstance(row, dict)]
    return []


def _compact_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = []
    for row in rows:
        keep.append(
            {
                key: row.get(key)
                for key in (
                    "transition_id",
                    "product_smiles",
                    "candidate_rank",
                    "candidate_score",
                    "candidate_source",
                    "candidate_model",
                    "candidate_type",
                    "exact_label",
                    "similar_label",
                    "reactant_similarity",
                    "positive_label",
                )
            }
        )
    return keep


def _one_hot(value: str, values: list[str]) -> list[float]:
    return [float(value == item) for item in values]


def _bucket_int(value: Any) -> str:
    try:
        raw = int(value or 0)
    except (TypeError, ValueError):
        raw = 0
    if raw <= 0:
        return "0"
    if raw == 1:
        return "1"
    if raw == 2:
        return "2"
    return "3plus"


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


def _fp_bits(smiles: str, n_bits: int) -> list[float]:
    cached = _fp_bits_cached(str(smiles or ""), int(n_bits))
    return [float(v) for v in cached]


@lru_cache(maxsize=200000)
def _fp_bits_cached(smiles: str, n_bits: int) -> tuple[int, ...]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
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


def _best_set_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return max((_tanimoto(a, b) for a in left for b in right), default=0.0)


def _tanimoto(a: str, b: str) -> float:
    fp_a = _similarity_fp(str(a or ""))
    fp_b = _similarity_fp(str(b or ""))
    if fp_a is None or fp_b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


@lru_cache(maxsize=200000)
def _similarity_fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _float_or_zero(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CCTS-v0 Transition Ranker",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Metrics",
        "",
        "| Model | Label | Coverage | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all | R@5 covered |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [("chem_rank", (result.get("baseline_chem_rank") or {}).get("test") or {})]
    for model_name, report in (result.get("models") or {}).items():
        rows.append((model_name, (report.get("test") or {})))
    for model_name, report in (result.get("blends") or {}).items():
        rows.append((model_name, (report.get("test") or {})))
    for model_name, report in rows:
        for label in ("exact_label", "similar_label", "positive_label"):
            metric = report.get(label) or {}
            all_k = metric.get("recall_at_k_all") or {}
            cov_k = metric.get("recall_at_k_covered") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        label,
                        str(metric.get("coverage")),
                        str(metric.get("mrr_covered")),
                        str(all_k.get("1")),
                        str(all_k.get("3")),
                        str(all_k.get("5")),
                        str(all_k.get("10")),
                        str(cov_k.get("5")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Train CCTS-v0 transition ranker from ChemEnzy candidate coverage audits")
    ap.add_argument("--train-coverage", required=True)
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-coverage", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--test-coverage", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--similarity-threshold", type=float, default=0.7)
    ap.add_argument("--max-candidates-per-transition", type=int, default=100)
    ap.add_argument("--evidence-pool-size", type=int, default=80)
    ap.add_argument("--n-estimators", type=int, default=240)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    report = train_ccts_v0_transition_ranker(
        train_coverage=Path(args.train_coverage),
        train_cache=Path(args.train_cache),
        val_coverage=Path(args.val_coverage),
        val_cache=Path(args.val_cache),
        test_coverage=Path(args.test_coverage),
        test_cache=Path(args.test_cache),
        output_dir=Path(args.output_dir),
        similarity_threshold=args.similarity_threshold,
        max_candidates_per_transition=args.max_candidates_per_transition,
        evidence_pool_size=args.evidence_pool_size,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "test": {
                    "baseline": report["baseline_chem_rank"]["test"],
                    "models": {k: v["test"] for k, v in report["models"].items()},
                    "blends": {k: v["test"] for k, v in report.get("blends", {}).items()},
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

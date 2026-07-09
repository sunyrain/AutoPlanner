"""Probe no-human controls on the runtime hard-negative candidate cache.

This is a negative-evidence audit for the route/block promotion gate.  It asks
whether cheap no-human additions, such as material-sanity/product-shape
features or a non-linear ranker, can beat the existing retrieval-only runtime
control on the fixed ChemEnzy candidate cache.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from cascade_planner.eval.train_ccts_v0_transition_ranker import CandidateDataset, _baseline_scores, _standardize
from cascade_planner.eval.train_ccts_v2_sparse_labels import _evaluate_sparse_dataset
from cascade_planner.eval.train_ccts_v3_runtime_pairwise_ranker import _dataset, _feature_names, _read_jsonl


SCHEMA_VERSION = "runtime_hardneg_nohuman_probe.v1"
DEFAULT_CACHE_DIR = Path("results/shared/cascadebench_strict_20260516/ccts_v3_runtime_candidate_cache")


def probe_runtime_hardneg_nohuman_controls(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_json: Path,
    output_md: Path | None = None,
    min_delta_vs_retrieval: float = 0.03,
    c_values: list[float] | None = None,
    hgb_configs: list[tuple[str, float, float]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.monotonic()
    RDLogger.DisableLog("rdApp.*")
    c_values = c_values or [0.01, 0.03, 0.1, 0.3, 1.0]
    hgb_configs = hgb_configs or [
        ("block_cls", 0.1, 0.1),
        ("block_cls", 0.06, 0.01),
        ("exact_or_block_cls", 0.06, 0.0),
    ]
    feature_names = _feature_names()
    datasets = {
        split: _dataset(_read_jsonl(Path(cache_dir) / f"{split}_candidates.jsonl"), feature_names)
        for split in ("train", "val", "test")
    }

    retrieval = _retrieval_control(datasets)
    material = _material_sanity_probe(datasets, retrieval, c_values=c_values, seed=seed)
    hgb = _hgb_probe(datasets, retrieval, hgb_configs=hgb_configs, seed=seed)

    best_material = material.get("selected_by_val") or {}
    best_hgb = hgb.get("selected_by_val") or {}
    best_learned = max(
        [row for row in [best_material, best_hgb] if row],
        key=lambda row: _float(row.get("blend_test_block_mrr")),
        default={},
    )
    retrieval_mrr = _float((retrieval.get("test") or {}).get("block_mrr"))
    best_delta = round(_float(best_learned.get("blend_test_block_mrr")) - retrieval_mrr, 6) if best_learned else None
    gate_passed = bool(best_delta is not None and best_delta >= float(min_delta_vs_retrieval))
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "cache_dir": str(cache_dir),
            "output_json": str(output_json),
            "output_md": str(output_md or output_json.with_suffix(".md")),
            "min_delta_vs_retrieval": float(min_delta_vs_retrieval),
            "seed": int(seed),
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "no expert labels; fixed runtime hard-negative ChemEnzy candidate cache",
        },
        "counts": {
            split: {
                "rows": len(ds.rows),
                "groups": len(ds.group_sizes),
                "block_supported_rows": sum(1 for row in ds.rows if row.get("block_supported_positive_label")),
                "exact_rows": sum(1 for row in ds.rows if row.get("exact_label")),
            }
            for split, ds in datasets.items()
        },
        "retrieval_control": retrieval,
        "material_sanity_pairwise_probe": material,
        "hgb_runtime_probe": hgb,
        "gate": {
            "passed": gate_passed,
            "best_learned_probe": best_learned.get("name"),
            "retrieval_test_block_mrr": retrieval_mrr,
            "best_blend_test_block_mrr": _float(best_learned.get("blend_test_block_mrr")),
            "best_delta_vs_retrieval": best_delta,
            "required_delta_vs_retrieval": float(min_delta_vs_retrieval),
        },
        "decision": {
            "status": "pass" if gate_passed else "fail",
            "reason": (
                "no-human material/product sanity or HGB runtime probes cleared retrieval-only"
                if gate_passed
                else "no-human material/product sanity and HGB runtime probes did not clear retrieval-only"
            ),
            "next_action": (
                "Do not tune this fixed feature set further; change the no-human label/candidate construction "
                "or train a scorer that targets retrieval-control residual errors."
            ),
        },
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md = Path(output_md) if output_md else output_json.with_suffix(".md")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(result), encoding="utf-8")
    return result


def _retrieval_control(datasets: dict[str, CandidateDataset]) -> dict[str, Any]:
    val = datasets["val"]
    test = datasets["test"]
    base_val = _baseline_scores(val.rows)
    base_test = _baseline_scores(test.rows)
    any_val = _row_score(val.rows, "runtime_nearest_any_transition_sim")
    any_test = _row_score(test.rows, "runtime_nearest_any_transition_sim")
    pair_val = _row_score(val.rows, "runtime_nearest_pair_compatible_sim")
    pair_test = _row_score(test.rows, "runtime_nearest_pair_compatible_sim")
    best: dict[str, Any] | None = None
    for alpha_any in _alpha_grid():
        for alpha_pair in _alpha_grid():
            val_scores = (
                _standardize(base_val)
                + float(alpha_any) * _standardize(any_val)
                + float(alpha_pair) * _standardize(pair_val)
            )
            val_report = _evaluate_sparse_dataset(val, val_scores)
            score = _selection_score(val_report)
            if best is None or score > _float(best.get("val_selection_score")):
                test_scores = (
                    _standardize(base_test)
                    + float(alpha_any) * _standardize(any_test)
                    + float(alpha_pair) * _standardize(pair_test)
                )
                best = {
                    "name": "chem_rank_plus_runtime_any_pair_grid",
                    "alpha_any": float(alpha_any),
                    "alpha_pair": float(alpha_pair),
                    "val_selection_score": round(float(score), 6),
                    "val": _compact_metrics(val_report),
                    "test": _compact_metrics(_evaluate_sparse_dataset(test, test_scores)),
                    "_val_scores": val_scores,
                    "_test_scores": test_scores,
                }
    assert best is not None
    return {key: value for key, value in best.items() if not key.startswith("_")}


def _material_sanity_probe(
    datasets: dict[str, CandidateDataset],
    retrieval: dict[str, Any],
    *,
    c_values: list[float],
    seed: int,
) -> dict[str, Any]:
    train, val, test = datasets["train"], datasets["val"], datasets["test"]
    x_train = _material_feature_matrix(train.rows)
    x_val = _material_feature_matrix(val.rows)
    x_test = _material_feature_matrix(test.rows)
    pair_x, pair_y = _pairwise_matrix(
        train,
        x_train,
        label_key="block_supported_positive_label",
        max_pos_per_group=4,
        max_neg_per_pos=20,
        seed=seed,
    )
    if pair_x.shape[0] == 0:
        return {"present": False, "reason": "no pairwise rows"}
    mean, std = _scaler(x_train)
    retrieval_val = _retrieval_scores(datasets["val"], retrieval)
    retrieval_test = _retrieval_scores(datasets["test"], retrieval)
    rows = []
    for c_value in c_values:
        model = LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            max_iter=500,
            random_state=int(seed),
        )
        model.fit(pair_x / std, pair_y)
        val_scores = model.decision_function((x_val - mean) / std)
        test_scores = model.decision_function((x_test - mean) / std)
        val_report = _evaluate_sparse_dataset(val, val_scores)
        test_report = _evaluate_sparse_dataset(test, test_scores)
        blend = _best_blend(
            val,
            test,
            base_val=retrieval_val,
            base_test=retrieval_test,
            aux_val=val_scores,
            aux_test=test_scores,
        )
        rows.append(
            {
                "name": f"material_sanity_pairwise_C{c_value}",
                "c_value": float(c_value),
                "pair_rows": int(pair_x.shape[0]),
                "val_selection_score": round(float(_selection_score(val_report)), 6),
                "val_block_mrr": _metric(val_report, "block_supported_positive_label", "mrr_covered"),
                "test_block_mrr": _metric(test_report, "block_supported_positive_label", "mrr_covered"),
                "test_exact_mrr": _metric(test_report, "exact_label", "mrr_covered"),
                **blend,
            }
        )
    return {
        "present": True,
        "feature_contract": "RDKit material/product-shape features; no expert labels; no GT labels as features",
        "rows": rows,
        "selected_by_val": max(rows, key=lambda row: _float(row.get("val_selection_score")), default={}),
        "selected_by_blend_val": max(rows, key=lambda row: _float(row.get("blend_val_selection_score")), default={}),
    }


def _hgb_probe(
    datasets: dict[str, CandidateDataset],
    retrieval: dict[str, Any],
    *,
    hgb_configs: list[tuple[str, float, float]],
    seed: int,
) -> dict[str, Any]:
    train, val, test = datasets["train"], datasets["val"], datasets["test"]
    x_train = _augmented_runtime_matrix(train)
    x_val = _augmented_runtime_matrix(val)
    x_test = _augmented_runtime_matrix(test)
    retrieval_val = _retrieval_scores(val, retrieval)
    retrieval_test = _retrieval_scores(test, retrieval)
    rows = []
    for label_name, learning_rate, l2_regularization in hgb_configs:
        labels = _hgb_labels(train.rows, label_name)
        pos = max(1, int(np.sum(labels > 0)))
        neg = max(1, int(labels.shape[0] - pos))
        weights = np.where(labels > 0, float(neg) / float(pos), 1.0)
        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=float(learning_rate),
            l2_regularization=float(l2_regularization),
            max_leaf_nodes=31,
            random_state=int(seed),
        )
        model.fit(x_train, labels, sample_weight=weights)
        val_scores = model.predict_proba(x_val)[:, 1]
        test_scores = model.predict_proba(x_test)[:, 1]
        val_report = _evaluate_sparse_dataset(val, val_scores)
        test_report = _evaluate_sparse_dataset(test, test_scores)
        blend = _best_blend(
            val,
            test,
            base_val=retrieval_val,
            base_test=retrieval_test,
            aux_val=val_scores,
            aux_test=test_scores,
        )
        rows.append(
            {
                "name": f"hgb_{label_name}_lr{learning_rate}_l2{l2_regularization}",
                "label_name": label_name,
                "learning_rate": float(learning_rate),
                "l2_regularization": float(l2_regularization),
                "val_selection_score": round(float(_selection_score(val_report)), 6),
                "val_block_mrr": _metric(val_report, "block_supported_positive_label", "mrr_covered"),
                "test_block_mrr": _metric(test_report, "block_supported_positive_label", "mrr_covered"),
                "test_exact_mrr": _metric(test_report, "exact_label", "mrr_covered"),
                **blend,
            }
        )
    return {
        "present": True,
        "feature_contract": "runtime-safe CCTS features plus simple interactions; no expert labels",
        "rows": rows,
        "selected_by_val": max(rows, key=lambda row: _float(row.get("val_selection_score")), default={}),
        "selected_by_blend_val": max(rows, key=lambda row: _float(row.get("blend_val_selection_score")), default={}),
    }


def _retrieval_scores(dataset: CandidateDataset, retrieval: dict[str, Any]) -> np.ndarray:
    base = _baseline_scores(dataset.rows)
    any_scores = _row_score(dataset.rows, "runtime_nearest_any_transition_sim")
    pair_scores = _row_score(dataset.rows, "runtime_nearest_pair_compatible_sim")
    return (
        _standardize(base)
        + _float(retrieval.get("alpha_any")) * _standardize(any_scores)
        + _float(retrieval.get("alpha_pair")) * _standardize(pair_scores)
    )


def _best_blend(
    val: CandidateDataset,
    test: CandidateDataset,
    *,
    base_val: np.ndarray,
    base_test: np.ndarray,
    aux_val: np.ndarray,
    aux_test: np.ndarray,
) -> dict[str, Any]:
    best: tuple[float, float] | None = None
    for alpha in _alpha_grid(extra=(3.0, 4.0)):
        scores = _standardize(base_val) + float(alpha) * _standardize(aux_val)
        score = _selection_score(_evaluate_sparse_dataset(val, scores))
        if best is None or score > best[0]:
            best = (float(score), float(alpha))
    assert best is not None
    test_scores = _standardize(base_test) + best[1] * _standardize(aux_test)
    test_report = _evaluate_sparse_dataset(test, test_scores)
    return {
        "blend_alpha_selected_on_val": best[1],
        "blend_val_selection_score": round(best[0], 6),
        "blend_test_block_mrr": _metric(test_report, "block_supported_positive_label", "mrr_covered"),
        "blend_test_exact_mrr": _metric(test_report, "exact_label", "mrr_covered"),
    }


def _pairwise_matrix(
    dataset: CandidateDataset,
    x: np.ndarray,
    *,
    label_key: str,
    max_pos_per_group: int,
    max_neg_per_pos: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    diffs = []
    labels = []
    offset = 0
    for group_size in dataset.group_sizes:
        rows = dataset.rows[offset : offset + group_size]
        group_x = x[offset : offset + group_size]
        positives = [idx for idx, row in enumerate(rows) if bool(row.get(label_key))]
        negatives = [idx for idx, row in enumerate(rows) if not bool(row.get(label_key))]
        positives = sorted(
            positives,
            key=lambda idx: (
                -float(rows[idx].get("block_supported_exact_label") is True),
                -float(rows[idx].get("exact_label") is True),
                int(rows[idx].get("candidate_rank") or 10**9),
            ),
        )[: max(1, int(max_pos_per_group))]
        hard_negatives = _hard_negatives(rows, negatives, max_neg_per_pos=max_neg_per_pos)
        for pos_idx in positives:
            for neg_idx in hard_negatives:
                diff = group_x[pos_idx] - group_x[neg_idx]
                diffs.append(diff)
                labels.append(1)
                diffs.append(-diff)
                labels.append(0)
        offset += group_size
    if not diffs:
        return np.zeros((0, x.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    order = np.random.default_rng(int(seed)).permutation(len(diffs))
    return np.asarray(diffs, dtype=np.float32)[order], np.asarray(labels, dtype=np.int32)[order]


def _hard_negatives(rows: list[dict[str, Any]], negatives: list[int], *, max_neg_per_pos: int) -> list[int]:
    by_rank = sorted(negatives, key=lambda idx: int(rows[idx].get("candidate_rank") or 10**9))[:max_neg_per_pos]
    by_evidence = sorted(
        negatives,
        key=lambda idx: (
            -_float(rows[idx].get("runtime_nearest_any_transition_sim")),
            -_float(rows[idx].get("runtime_nearest_pair_compatible_sim")),
            int(rows[idx].get("candidate_rank") or 10**9),
        ),
    )[:max_neg_per_pos]
    chosen = []
    for idx in [*by_rank, *by_evidence]:
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= max_neg_per_pos:
            break
    return chosen


def _material_feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    out = []
    for row in rows:
        product = str(row.get("product_smiles") or "")
        reactants = [str(smi) for smi in row.get("candidate_reactants") or []]
        product_props = _mol_props(product)
        reactant_props = [_mol_props(smi) for smi in reactants]
        product_heavy = product_props["heavy"]
        reactant_heavy = sum(prop["heavy"] for prop in reactant_props)
        max_reactant_heavy = max([prop["heavy"] for prop in reactant_props] or [0])
        product_rings = product_props["rings"]
        reactant_rings = sum(prop["rings"] for prop in reactant_props)
        max_reactant_rings = max([prop["rings"] for prop in reactant_props] or [0])
        heavy_delta = product_heavy - reactant_heavy
        rank = max(1, int(row.get("candidate_rank") or 1))
        score = _float(row.get("candidate_score"))
        any_sim = _float(row.get("runtime_nearest_any_transition_sim"))
        pair_sim = _float(row.get("runtime_nearest_pair_compatible_sim"))
        prior = _float(row.get("runtime_inferred_transform_prior"))
        out.append(
            [
                -float(rank),
                1.0 / float(rank),
                1.0 / math.log2(float(rank) + 1.0),
                score,
                math.log1p(max(score, 0.0)),
                float(len(reactants)),
                any_sim,
                pair_sim,
                prior,
                float(bool(row.get("runtime_prev_pair_supported"))),
                float(bool(row.get("runtime_next_pair_supported"))),
                math.log1p(_float(row.get("runtime_pair_bucket_size"))),
                float(product_heavy),
                float(reactant_heavy),
                float(max_reactant_heavy),
                float(heavy_delta),
                float(abs(heavy_delta)),
                float(heavy_delta >= 10),
                float(heavy_delta <= -10),
                float(abs(heavy_delta) <= 3),
                float(product_rings),
                float(reactant_rings),
                float(max_reactant_rings),
                float(max_reactant_rings >= product_rings),
                float(sum(1 for smi in [product, *reactants] if not _mol_props(smi)["valid"])),
                float(bool(reactants) and max_reactant_heavy <= 8 and product_heavy >= 20),
                float(bool(product_heavy) and max_reactant_heavy >= max(12, int(product_heavy * 0.65))),
                float(sum(1 for smi in reactants[1:] if _mol_props(smi)["heavy"] <= 3)),
                (1.0 / float(rank)) * any_sim,
                (1.0 / float(rank)) * pair_sim,
                score * any_sim,
                score * pair_sim,
            ]
        )
    return np.asarray(out, dtype=np.float32)


def _augmented_runtime_matrix(dataset: CandidateDataset) -> np.ndarray:
    base = dataset.x
    extra = []
    for row in dataset.rows:
        rank = max(1, int(row.get("candidate_rank") or 1))
        inv_rank = 1.0 / float(rank)
        score = _float(row.get("candidate_score"))
        any_sim = _float(row.get("runtime_nearest_any_transition_sim"))
        pair_sim = _float(row.get("runtime_nearest_pair_compatible_sim"))
        prior = _float(row.get("runtime_inferred_transform_prior"))
        n_reactants = len(row.get("candidate_reactants") or [])
        extra.append(
            [
                any_sim - pair_sim,
                pair_sim - any_sim,
                max(any_sim, pair_sim),
                min(any_sim, pair_sim),
                any_sim * pair_sim,
                inv_rank * max(any_sim, pair_sim),
                score * max(any_sim, pair_sim),
                prior * inv_rank,
                float(n_reactants == 1),
                float(n_reactants == 2),
                float(n_reactants >= 3),
            ]
        )
    return np.hstack([base, np.asarray(extra, dtype=np.float32)])


def _hgb_labels(rows: list[dict[str, Any]], label_name: str) -> np.ndarray:
    if label_name == "block_cls":
        return np.asarray([int(bool(row.get("block_supported_positive_label"))) for row in rows], dtype=np.int32)
    if label_name == "exact_or_block_cls":
        return np.asarray(
            [int(bool(row.get("block_supported_positive_label") or row.get("exact_label"))) for row in rows],
            dtype=np.int32,
        )
    raise ValueError(f"unsupported hgb label: {label_name}")


def _selection_score(report: dict[str, Any]) -> float:
    block = report.get("block_supported_positive_label") or {}
    exact = report.get("exact_label") or {}
    block_k = block.get("recall_at_k_all") or {}
    exact_k = exact.get("recall_at_k_all") or {}
    return (
        _float(block.get("mrr_covered"))
        + 0.8 * _float(block_k.get("5"))
        + 0.4 * _float(exact.get("mrr_covered"))
        + 0.3 * _float(exact_k.get("5"))
    )


def _compact_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_score": round(float(_selection_score(report)), 6),
        "block_mrr": _metric(report, "block_supported_positive_label", "mrr_covered"),
        "exact_mrr": _metric(report, "exact_label", "mrr_covered"),
        "block_recall_at5_all": ((report.get("block_supported_positive_label") or {}).get("recall_at_k_all") or {}).get("5"),
        "exact_recall_at5_all": ((report.get("exact_label") or {}).get("recall_at_k_all") or {}).get("5"),
    }


def _metric(report: dict[str, Any], label: str, key: str) -> float:
    return _float((report.get(label) or {}).get(key))


def _row_score(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_float(row.get(key)) for row in rows], dtype=np.float32)


def _scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0).astype(np.float32)
    std = np.std(x, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


@lru_cache(maxsize=300000)
def _mol_props(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {"valid": False, "heavy": 0, "rings": 0}
    return {"valid": True, "heavy": int(mol.GetNumHeavyAtoms()), "rings": int(mol.GetRingInfo().NumRings())}


def _alpha_grid(extra: tuple[float, ...] = ()) -> list[float]:
    return sorted({
        -2.0,
        -1.5,
        -1.0,
        -0.75,
        -0.5,
        -0.3,
        -0.2,
        -0.1,
        -0.05,
        0.0,
        0.05,
        0.1,
        0.2,
        0.3,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        *[float(value) for value in extra],
    })


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _markdown(result: dict[str, Any]) -> str:
    retrieval = result.get("retrieval_control") or {}
    gate = result.get("gate") or {}
    lines = [
        "# Runtime Hard-Negative No-Human Probe",
        "",
        f"Decision: `{(result.get('decision') or {}).get('status')}`",
        "",
        (result.get("decision") or {}).get("reason", ""),
        "",
        "## Retrieval Control",
        "",
        "| alpha_any | alpha_pair | test block MRR | test exact MRR |",
        "|---:|---:|---:|---:|",
        (
            f"| {retrieval.get('alpha_any')} | {retrieval.get('alpha_pair')} | "
            f"{(retrieval.get('test') or {}).get('block_mrr')} | {(retrieval.get('test') or {}).get('exact_mrr')} |"
        ),
        "",
        "## Probe Gate",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| retrieval test block MRR | {gate.get('retrieval_test_block_mrr')} |",
        f"| best blend test block MRR | {gate.get('best_blend_test_block_mrr')} |",
        f"| best delta vs retrieval | {gate.get('best_delta_vs_retrieval')} |",
        f"| required delta | {gate.get('required_delta_vs_retrieval')} |",
        f"| passed | `{gate.get('passed')}` |",
        "",
        "## Material Sanity Pairwise",
        "",
        "| Model | val score | test block | test exact | blend alpha | blend test block |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ((result.get("material_sanity_pairwise_probe") or {}).get("rows") or []):
        lines.append(
            f"| `{row.get('name')}` | {row.get('val_selection_score')} | {row.get('test_block_mrr')} | "
            f"{row.get('test_exact_mrr')} | {row.get('blend_alpha_selected_on_val')} | "
            f"{row.get('blend_test_block_mrr')} |"
        )
    lines.extend(
        [
            "",
            "## HGB Runtime Probe",
            "",
            "| Model | val score | test block | test exact | blend alpha | blend test block |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ((result.get("hgb_runtime_probe") or {}).get("rows") or []):
        lines.append(
            f"| `{row.get('name')}` | {row.get('val_selection_score')} | {row.get('test_block_mrr')} | "
            f"{row.get('test_exact_mrr')} | {row.get('blend_alpha_selected_on_val')} | "
            f"{row.get('blend_test_block_mrr')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe no-human controls on runtime hard-negative cache")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md")
    ap.add_argument("--min-delta-vs-retrieval", type=float, default=0.03)
    ap.add_argument("--c-values", default="0.01,0.03,0.1,0.3,1.0")
    args = ap.parse_args()
    result = probe_runtime_hardneg_nohuman_controls(
        cache_dir=Path(args.cache_dir),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        min_delta_vs_retrieval=args.min_delta_vs_retrieval,
        c_values=[float(item) for item in str(args.c_values).split(",") if item.strip()],
    )
    print(json.dumps({"decision": result["decision"], "gate": result["gate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

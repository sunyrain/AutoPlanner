"""Build cached CCTS-v3 candidate-evidence rows.

The train_ccts_v3 script intentionally works from enriched candidate rows, but
candidate-specific nearest-evidence retrieval is expensive enough that it
should be cached per split.  This builder writes those rows and raw ranking
metrics without training any model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rdkit import RDLogger

from cascade_planner.eval.audit_candidate_specific_evidence import _candidate_evidence_scores, _train_bank
from cascade_planner.eval.train_ccts_v0_transition_ranker import _best_set_similarity, _candidate_rows_from_cache, _read_json, _write_jsonl
from cascade_planner.eval.train_ccts_v2_sparse_labels import (
    _add_block_support,
    _evaluate_sparse_dataset,
    _neighbor_context,
    _training_relevance,
)
from cascade_planner.eval.train_ccts_v2_transition_ranker import _load_program_evidence, _transition_context
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles

import numpy as np
from cascade_planner.eval.train_ccts_v0_transition_ranker import CandidateDataset, _baseline_scores


SCHEMA_VERSION = "ccts_v3_candidate_cache.v1"


def build_ccts_v3_candidate_cache(
    *,
    coverage: Path,
    cache: Path,
    program_manifest: Path,
    output_jsonl: Path,
    output_report: Path,
    split_name: str,
    label_mode: str = "binary_block_exact",
    similarity_threshold: float = 0.70,
    adjacency_similarity_threshold: float = 0.70,
    max_candidates_per_transition: int = 100,
) -> dict[str, Any]:
    started = time.monotonic()
    payload = _read_json(coverage)
    transitions = [row for row in payload.get("transitions") or [] if isinstance(row, dict)]
    cache_rows = _read_json(cache)
    evidence = _load_program_evidence(program_manifest)
    neighbors = _neighbor_context(program_manifest)
    train_bank = _train_bank(program_manifest)
    rows: list[dict[str, Any]] = []
    product_sim_cache: dict[tuple[str, str], list[float]] = {}
    for transition in transitions:
        product = str(transition.get("product_smiles") or "")
        context = _transition_context(transition, evidence)
        neighbor = neighbors.get(str(transition.get("transition_id") or "")) or {}
        for rank, candidate in enumerate(_candidate_rows_from_cache(cache_rows, product)[: int(max_candidates_per_transition)], start=1):
            row = _label_row(transition, candidate, rank=rank, similarity_threshold=similarity_threshold)
            row.update(
                {
                    "split": split_name,
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
                    train_bank=train_bank,
                    product_sim_cache=product_sim_cache,
                )
            )
            row["training_relevance"] = _training_relevance(row, label_mode=label_mode)
            rows.append(row)

    dataset = _dataset_from_rows(rows)
    scores = {
        "chem_rank": _baseline_scores(rows),
        "candidate_nearest_context_transform_sim": _row_score(rows, "candidate_nearest_context_transform_sim"),
        "candidate_inferred_transform_match_score": _row_score(rows, "candidate_inferred_transform_match_score"),
        "candidate_nearest_pair_compatible_sim": _row_score(rows, "candidate_nearest_pair_compatible_sim"),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "split_name": split_name,
            "coverage": str(coverage),
            "cache": str(cache),
            "program_manifest": str(program_manifest),
            "output_jsonl": str(output_jsonl),
            "label_mode": label_mode,
            "similarity_threshold": similarity_threshold,
            "adjacency_similarity_threshold": adjacency_similarity_threshold,
            "max_candidates_per_transition": max_candidates_per_transition,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            "transitions": len(transitions),
            "candidate_rows": len(rows),
            "groups": len(dataset.group_sizes),
            "relevance_rows": int(np.sum(dataset.y > 0)),
            "block_supported_rows": sum(1 for row in rows if row.get("block_supported_positive_label")),
            "exact_rows": sum(1 for row in rows if row.get("exact_label")),
        },
        "metrics": {name: _evaluate_sparse_dataset(dataset, values) for name, values in scores.items()},
    }
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, rows)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _label_row(transition: dict[str, Any], candidate: dict[str, Any], *, rank: int, similarity_threshold: float) -> dict[str, Any]:
    gt_rxn = canonical_reaction(str(transition.get("rxn_smiles") or ""))
    gt_reactants = set(str(smi) for smi in transition.get("reactants") or [])
    rxn = canonical_reaction(candidate.get("reaction_smiles") or candidate.get("rxn_smiles") or "")
    cand_reactants = set(canonical_side((rxn.split(">>", 1)[0] if ">>" in rxn else "")))
    cand_main = canonical_smiles(candidate.get("main_reactant")) or str(candidate.get("main_reactant") or "")
    reactant_similarity = _best_set_similarity(gt_reactants, cand_reactants)
    exact = bool(gt_rxn and rxn == gt_rxn)
    similar = bool(reactant_similarity >= similarity_threshold)
    return {
        "transition_id": transition.get("transition_id"),
        "target_smiles": transition.get("target_smiles"),
        "product_smiles": transition.get("product_smiles"),
        "route_domain": transition.get("route_domain"),
        "step_pos": transition.get("step_pos"),
        "remaining_steps": transition.get("remaining_steps"),
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
        "similar_only_label": bool(similar and not exact),
        "positive_label": bool(exact or similar),
    }


def _dataset_from_rows(rows: list[dict[str, Any]]) -> CandidateDataset:
    group_ids: list[str] = []
    group_sizes: list[int] = []
    ordered_rows: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("transition_id") or ""), []).append(row)
    for group_id, group_rows in by_group.items():
        group_rows = sorted(group_rows, key=lambda row: int(row.get("candidate_rank") or 10**9))
        group_ids.append(group_id)
        group_sizes.append(len(group_rows))
        ordered_rows.extend(group_rows)
    return CandidateDataset(
        rows=ordered_rows,
        x=np.zeros((len(ordered_rows), 1), dtype=np.float32),
        y=np.asarray([int(row.get("training_relevance") or 0) for row in ordered_rows], dtype=np.int32),
        group_sizes=group_sizes,
        group_ids=group_ids,
        feature_names=["dummy"],
        chem_feature_indices=[],
    )


def _row_score(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key) or 0.0) for row in rows], dtype=np.float32)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Build cached CCTS-v3 candidate-evidence rows")
    ap.add_argument("--coverage", default="results/shared/cascadebench_strict_20260516/coverage/coverage_test_top100.json")
    ap.add_argument("--cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--output-jsonl", default="results/shared/cascadebench_strict_20260516/ccts_v3_candidate_cache/test_candidates.jsonl")
    ap.add_argument("--output-report", default="results/shared/cascadebench_strict_20260516/ccts_v3_candidate_cache/test_report.json")
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--label-mode", choices=("binary_block_exact", "graded_evidence"), default="binary_block_exact")
    ap.add_argument("--similarity-threshold", type=float, default=0.70)
    ap.add_argument("--adjacency-similarity-threshold", type=float, default=0.70)
    ap.add_argument("--max-candidates-per-transition", type=int, default=100)
    args = ap.parse_args()
    report = build_ccts_v3_candidate_cache(
        coverage=Path(args.coverage),
        cache=Path(args.cache),
        program_manifest=Path(args.program_manifest),
        output_jsonl=Path(args.output_jsonl),
        output_report=Path(args.output_report),
        split_name=args.split_name,
        label_mode=args.label_mode,
        similarity_threshold=args.similarity_threshold,
        adjacency_similarity_threshold=args.adjacency_similarity_threshold,
        max_candidates_per_transition=args.max_candidates_per_transition,
    )
    print(json.dumps({"counts": report["counts"], "outputs": {"jsonl": args.output_jsonl, "report": args.output_report}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

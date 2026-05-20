"""Build runtime-safe CCTS-v3 candidate evidence rows.

The earlier CCTS-v3 cache stored evidence scores that were conditioned on the
held-out transition's true transformation class.  That is useful as an oracle
diagnostic, but it is not available during search.  This builder keeps the same
fixed ChemEnzy candidate pool and labels, then recomputes candidate evidence
without using the current transition's true transform.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import RDLogger

from cascade_planner.eval.audit_candidate_specific_evidence import _best_in_bucket, _fp, _train_bank
from cascade_planner.eval.train_ccts_v0_transition_ranker import CandidateDataset, _baseline_scores, _write_jsonl
from cascade_planner.eval.train_ccts_v2_sparse_labels import _evaluate_sparse_dataset


SCHEMA_VERSION = "ccts_v3_runtime_evidence_cache.v1"
DEFAULT_CACHE_DIR = Path("results/shared/cascadebench_strict_20260516/ccts_v3_candidate_cache")
LEAKY_EVIDENCE_FIELDS = {
    "reactant_similarity",
    "candidate_nearest_context_transform_sim",
    "candidate_nearest_pair_compatible_sim",
    "candidate_nearest_context_support_id",
    "candidate_inferred_transform",
    "candidate_inferred_transform_matches_context",
    "candidate_inferred_transform_match_score",
}


def build_ccts_v3_runtime_evidence_cache(
    *,
    input_jsonl: Path,
    program_manifest: Path,
    output_jsonl: Path,
    output_report: Path,
    split_name: str,
) -> dict[str, Any]:
    started = time.monotonic()
    rows = _read_jsonl(input_jsonl)
    train_bank = _train_bank(program_manifest)
    product_sim_cache: dict[tuple[str, str], list[float]] = {}
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        clean = {key: value for key, value in row.items() if key not in LEAKY_EVIDENCE_FIELDS}
        clean.update(
            _runtime_evidence_scores(
                product=str(row.get("product_smiles") or ""),
                candidate_main=str(row.get("candidate_main_reactant") or ""),
                previous_transform=str(row.get("previous_transform") or ""),
                next_transform=str(row.get("next_transform") or ""),
                train_bank=train_bank,
                product_sim_cache=product_sim_cache,
            )
        )
        out_rows.append(clean)
    dataset = _dataset(out_rows)
    ordered_rows = dataset.rows
    score_columns = {
        "chem_rank": _baseline_scores(ordered_rows),
        "runtime_nearest_any_transition_sim": _row_score(ordered_rows, "runtime_nearest_any_transition_sim"),
        "runtime_nearest_pair_compatible_sim": _row_score(ordered_rows, "runtime_nearest_pair_compatible_sim"),
        "runtime_inferred_transform_prior": _row_score(ordered_rows, "runtime_inferred_transform_prior"),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "input_jsonl": str(input_jsonl),
            "program_manifest": str(program_manifest),
            "output_jsonl": str(output_jsonl),
            "output_report": str(output_report),
            "split_name": split_name,
            "elapsed_s": round(time.monotonic() - started, 3),
            "runtime_safe_contract": "does not use current held-out transform; candidate transform is inferred from nearest train evidence",
            "removed_fields": sorted(LEAKY_EVIDENCE_FIELDS),
        },
        "counts": {
            "candidate_rows": len(out_rows),
            "groups": len(dataset.group_sizes),
            "relevance_rows": int(np.sum(dataset.y > 0)),
            "block_supported_rows": sum(1 for row in out_rows if row.get("block_supported_positive_label")),
            "exact_rows": sum(1 for row in out_rows if row.get("exact_label")),
            "train_bank_items": train_bank.get("count"),
            "unique_runtime_inferred_transforms": len({row.get("runtime_inferred_transform") for row in out_rows}),
        },
        "metrics": {name: _evaluate_sparse_dataset(dataset, scores) for name, scores in score_columns.items()},
        "inferred_transform_summary": _inferred_transform_summary(out_rows),
        "leakage_check": {
            "leaky_evidence_fields_present_in_output": {
                key: sum(1 for row in out_rows if key in row)
                for key in sorted(LEAKY_EVIDENCE_FIELDS)
            }
        },
    }
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, out_rows)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _runtime_evidence_scores(
    *,
    product: str,
    candidate_main: str,
    previous_transform: str,
    next_transform: str,
    train_bank: dict[str, Any],
    product_sim_cache: dict[tuple[str, str], list[float]],
) -> dict[str, Any]:
    product_fp = _fp(product)
    main_fp = _fp(candidate_main)
    best_any = _best_in_bucket(product, product_fp, main_fp, train_bank["all"], product_sim_cache, "all")
    inferred_transform = str((best_any.get("item") or {}).get("transform") or "")
    prev_bucket = train_bank["by_next_pair"].get(f"{previous_transform}->{inferred_transform}") if previous_transform and inferred_transform else []
    next_bucket = train_bank["by_prev_pair"].get(f"{inferred_transform}->{next_transform}") if next_transform and inferred_transform else []
    pair_bucket = list(prev_bucket or [])
    seen = {item.get("transition_id") for item in pair_bucket}
    for item in next_bucket or []:
        if item.get("transition_id") not in seen:
            pair_bucket.append(item)
            seen.add(item.get("transition_id"))
    best_pair = _best_in_bucket(
        product,
        product_fp,
        main_fp,
        pair_bucket,
        product_sim_cache,
        f"rtpair:{previous_transform}->{inferred_transform}->{next_transform}",
    )
    transform_counts = train_bank.get("transform_counts") or {}
    prior = float(transform_counts.get(inferred_transform) or 0.0) / max(1.0, float(train_bank.get("count") or 0.0))
    return {
        "runtime_nearest_any_transition_sim": best_any.get("transition_sim", 0.0),
        "runtime_nearest_any_product_sim": best_any.get("product_sim", 0.0),
        "runtime_nearest_any_main_sim": best_any.get("main_sim", 0.0),
        "runtime_nearest_any_support_id": (best_any.get("item") or {}).get("transition_id"),
        "runtime_inferred_transform": inferred_transform,
        "runtime_inferred_transform_prior": round(prior, 6),
        "runtime_prev_pair_supported": bool(prev_bucket),
        "runtime_next_pair_supported": bool(next_bucket),
        "runtime_pair_bucket_size": len(pair_bucket),
        "runtime_nearest_pair_compatible_sim": best_pair.get("transition_sim", 0.0),
        "runtime_nearest_pair_support_id": (best_pair.get("item") or {}).get("transition_id"),
    }


def _dataset(rows: list[dict[str, Any]]) -> CandidateDataset:
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
        x=np.zeros((len(ordered), 1), dtype=np.float32),
        y=np.asarray([int(bool(row.get("training_relevance"))) for row in ordered], dtype=np.int32),
        group_sizes=group_sizes,
        group_ids=group_ids,
        feature_names=["dummy"],
        chem_feature_indices=[],
    )


def _row_score(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_float(row.get(key)) for row in rows], dtype=np.float32)


def _inferred_transform_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, subset in {
        "all": rows,
        "block_supported_positive": [row for row in rows if row.get("block_supported_positive_label")],
        "exact": [row for row in rows if row.get("exact_label")],
        "negative": [row for row in rows if not row.get("positive_label")],
    }.items():
        out[name] = {
            "rows": len(subset),
            "avg_any_sim": round(sum(_float(row.get("runtime_nearest_any_transition_sim")) for row in subset) / max(1, len(subset)), 6),
            "avg_pair_sim": round(sum(_float(row.get("runtime_nearest_pair_compatible_sim")) for row in subset) / max(1, len(subset)), 6),
            "pair_supported_rate": round(sum(1 for row in subset if row.get("runtime_pair_bucket_size")) / max(1, len(subset)), 6),
            "top_runtime_inferred_transforms": dict(Counter(str(row.get("runtime_inferred_transform") or "") for row in subset).most_common(20)),
        }
    return out


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


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Build runtime-safe CCTS-v3 candidate evidence cache")
    ap.add_argument("--input-jsonl", default=str(DEFAULT_CACHE_DIR / "test_candidates.jsonl"))
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--output-jsonl", default="results/shared/cascadebench_strict_20260516/ccts_v3_runtime_candidate_cache/test_candidates.jsonl")
    ap.add_argument("--output-report", default="results/shared/cascadebench_strict_20260516/ccts_v3_runtime_candidate_cache/test_report.json")
    ap.add_argument("--split-name", default="test")
    args = ap.parse_args()
    report = build_ccts_v3_runtime_evidence_cache(
        input_jsonl=Path(args.input_jsonl),
        program_manifest=Path(args.program_manifest),
        output_jsonl=Path(args.output_jsonl),
        output_report=Path(args.output_report),
        split_name=args.split_name,
    )
    print(json.dumps({"counts": report["counts"], "outputs": {"jsonl": args.output_jsonl, "report": args.output_report}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

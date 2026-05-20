"""Audit CCTS positive-label semantics.

CCTS-v0/v1 labels a candidate positive when it is an exact reaction hit or a
similar-reactant hit.  This script quantifies how much of the training and test
signal is exact, similar-only, or very dense within a candidate group.  Dense or
similar-only positives are useful for proposal coverage, but weak evidence for
cascade block coherence.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import RDLogger

from cascade_planner.eval.train_ccts_v0_transition_ranker import (
    _best_set_similarity,
    _read_json,
)
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles


SCHEMA_VERSION = "ccts_label_semantics_audit.v1"


def audit_ccts_label_semantics(
    *,
    train_coverage: Path,
    val_coverage: Path,
    test_coverage: Path,
    cache: Path,
    test_candidates_jsonl: Path | None,
    output_json: Path,
    output_md: Path,
    similarity_threshold: float = 0.7,
    max_candidates_per_transition: int = 100,
    evidence_pool_size: int = 80,
) -> dict[str, Any]:
    started = time.monotonic()
    if test_candidates_jsonl and test_candidates_jsonl.exists():
        split_reports = {
            "test": _split_report_from_rows(_read_jsonl(test_candidates_jsonl)),
        }
        source_mode = "compact_test_candidates_jsonl"
    else:
        cache_rows = _read_json(cache)
        cache_index = _cache_index(cache_rows)
        split_payloads = {
            "train": _read_json(train_coverage),
            "val": _read_json(val_coverage),
            "test": _read_json(test_coverage),
        }
        split_reports = {}
        for split, payload in split_payloads.items():
            transitions = [row for row in payload.get("transitions") or [] if isinstance(row, dict)]
            rows, group_sizes, group_ids = _label_rows(
                transitions,
                cache_index,
                similarity_threshold=similarity_threshold,
                max_candidates_per_transition=max_candidates_per_transition,
            )
            split_reports[split] = _split_report(rows, group_sizes, group_ids)
        source_mode = "recomputed_from_coverage_and_cache"

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_coverage": str(train_coverage),
            "val_coverage": str(val_coverage),
            "test_coverage": str(test_coverage),
            "cache": str(cache),
            "test_candidates_jsonl": str(test_candidates_jsonl) if test_candidates_jsonl else None,
            "source_mode": source_mode,
            "similarity_threshold": similarity_threshold,
            "max_candidates_per_transition": max_candidates_per_transition,
            "evidence_pool_size": evidence_pool_size,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "splits": split_reports,
        "interpretation": _interpretation(split_reports),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(result), encoding="utf-8")
    return result


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


def _split_report_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_sizes = []
    group_ids = []
    last = None
    size = 0
    for row in rows:
        group_id = str(row.get("transition_id") or "")
        if last is None:
            last = group_id
        if group_id != last:
            group_ids.append(last)
            group_sizes.append(size)
            last = group_id
            size = 0
        size += 1
    if last is not None:
        group_ids.append(last)
        group_sizes.append(size)
    return _split_report(rows, group_sizes, group_ids)


def _label_rows(
    transitions: list[dict[str, Any]],
    cache_index: dict[str, list[dict[str, Any]]],
    *,
    similarity_threshold: float,
    max_candidates_per_transition: int,
) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    rows: list[dict[str, Any]] = []
    group_sizes: list[int] = []
    group_ids: list[str] = []
    for transition in transitions:
        product = str(transition.get("product_smiles") or "")
        candidates = cache_index.get(product, [])[: int(max_candidates_per_transition)]
        group = [
            _candidate_label_row_fast(
                transition,
                candidate,
                rank=idx,
                similarity_threshold=similarity_threshold,
            )
            for idx, candidate in enumerate(candidates, 1)
        ]
        if not group:
            continue
        rows.extend(group)
        group_sizes.append(len(group))
        group_ids.append(str(transition.get("transition_id") or ""))
    return rows, group_sizes, group_ids


def _cache_index(cache: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for raw_key, rows in cache.items():
        try:
            key = json.loads(raw_key)
        except Exception:
            continue
        product = str(key.get("product") or "")
        if product and isinstance(rows, list):
            out[product] = [row for row in rows if isinstance(row, dict)]
    return out


def _candidate_label_row_fast(
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
        "candidate_rank": rank,
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


def _split_report(rows: list[dict[str, Any]], group_sizes: list[int], group_ids: list[str]) -> dict[str, Any]:
    group_reports = []
    offset = 0
    for group_id, size in zip(group_ids, group_sizes):
        group = rows[offset : offset + size]
        offset += size
        exact_rows = [row for row in group if row.get("exact_label")]
        similar_rows = [row for row in group if row.get("similar_label")]
        similar_only_rows = [row for row in group if row.get("similar_label") and not row.get("exact_label")]
        positive_rows = [row for row in group if row.get("positive_label")]
        exact_best_rank = min((int(row.get("candidate_rank") or 10**9) for row in exact_rows), default=None)
        similar_best_rank = min((int(row.get("candidate_rank") or 10**9) for row in similar_rows), default=None)
        positive_best_rank = min((int(row.get("candidate_rank") or 10**9) for row in positive_rows), default=None)
        group_reports.append(
            {
                "group_id": group_id,
                "candidate_count": len(group),
                "positive_count": len(positive_rows),
                "exact_count": len(exact_rows),
                "similar_count": len(similar_rows),
                "similar_only_count": len(similar_only_rows),
                "positive_fraction": len(positive_rows) / max(len(group), 1),
                "has_positive": bool(positive_rows),
                "has_exact": bool(exact_rows),
                "has_similar_only": bool(similar_only_rows),
                "positive_best_rank": positive_best_rank,
                "exact_best_rank": exact_best_rank,
                "similar_best_rank": similar_best_rank,
                "route_domain": (group[0].get("route_domain") if group else None),
                "product_smiles": (group[0].get("product_smiles") if group else None),
            }
        )
    positive_groups = [row for row in group_reports if row["has_positive"]]
    exact_groups = [row for row in group_reports if row["has_exact"]]
    similar_only_positive_groups = [row for row in positive_groups if row["similar_only_count"] > 0 and row["exact_count"] == 0]
    exact_plus_similar_groups = [row for row in positive_groups if row["similar_only_count"] > 0 and row["exact_count"] > 0]
    dense_groups = [row for row in positive_groups if row["positive_fraction"] >= 0.10 or row["positive_count"] >= 10]
    rows_positive = [row for row in rows if row.get("positive_label")]
    exact_positive = [row for row in rows_positive if row.get("exact_label")]
    similar_only_positive = [row for row in rows_positive if row.get("similar_label") and not row.get("exact_label")]
    return {
        "groups": len(group_reports),
        "candidate_rows": len(rows),
        "positive_rows": len(rows_positive),
        "exact_positive_rows": len(exact_positive),
        "similar_only_positive_rows": len(similar_only_positive),
        "positive_group_coverage": _rate(len(positive_groups), len(group_reports)),
        "exact_group_coverage": _rate(len(exact_groups), len(group_reports)),
        "similar_only_positive_group_rate_all": _rate(len(similar_only_positive_groups), len(group_reports)),
        "similar_only_positive_group_rate_covered": _rate(len(similar_only_positive_groups), len(positive_groups)),
        "exact_plus_similar_group_rate_covered": _rate(len(exact_plus_similar_groups), len(positive_groups)),
        "dense_positive_group_rate_covered": _rate(len(dense_groups), len(positive_groups)),
        "positive_rows_similar_only_fraction": _rate(len(similar_only_positive), len(rows_positive)),
        "positive_rows_exact_fraction": _rate(len(exact_positive), len(rows_positive)),
        "positive_count_distribution": dict(Counter(_count_bucket(row["positive_count"]) for row in group_reports)),
        "positive_fraction_distribution": dict(Counter(_fraction_bucket(row["positive_fraction"]) for row in group_reports)),
        "positive_best_rank_distribution": dict(Counter(_rank_bucket(row["positive_best_rank"]) for row in group_reports)),
        "exact_best_rank_distribution": dict(Counter(_rank_bucket(row["exact_best_rank"]) for row in group_reports)),
        "route_domain_positive_group_rate": _by_route_domain(group_reports),
        "examples_similar_only_positive_groups": _examples(similar_only_positive_groups),
        "examples_dense_positive_groups": _examples(dense_groups),
    }


def _by_route_domain(group_reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    domains = sorted({str(row.get("route_domain") or "unknown") for row in group_reports})
    out = {}
    for domain in domains:
        rows = [row for row in group_reports if str(row.get("route_domain") or "unknown") == domain]
        positives = [row for row in rows if row.get("has_positive")]
        exact = [row for row in rows if row.get("has_exact")]
        out[domain] = {
            "groups": len(rows),
            "positive_group_coverage": _rate(len(positives), len(rows)),
            "exact_group_coverage": _rate(len(exact), len(rows)),
        }
    return out


def _examples(rows: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-int(row.get("positive_count") or 0), str(row.get("group_id") or "")))
    return [
        {
            "group_id": row.get("group_id"),
            "product_smiles": row.get("product_smiles"),
            "route_domain": row.get("route_domain"),
            "candidate_count": row.get("candidate_count"),
            "positive_count": row.get("positive_count"),
            "exact_count": row.get("exact_count"),
            "similar_only_count": row.get("similar_only_count"),
            "positive_fraction": round(float(row.get("positive_fraction") or 0.0), 6),
            "positive_best_rank": row.get("positive_best_rank"),
            "exact_best_rank": row.get("exact_best_rank"),
        }
        for row in ordered[:limit]
    ]


def _interpretation(split_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    test = split_reports.get("test") or {}
    warnings = []
    if (test.get("positive_rows_similar_only_fraction") or 0.0) > 0.5:
        warnings.append("Most positive candidate rows are similar-only rather than exact reaction hits.")
    if (test.get("similar_only_positive_group_rate_covered") or 0.0) > 0.5:
        warnings.append("Most covered positive groups can be satisfied by similar-only candidates.")
    if (test.get("dense_positive_group_rate_covered") or 0.0) > 0.25:
        warnings.append("Many groups have dense positives; ranking can become a broad similarity task.")
    return {
        "warnings": warnings,
        "recommended_fix": [
            "Keep exact/similar labels as coverage diagnostics, not the sole training objective.",
            "Add block-consistency labels that require adjacent transition compatibility, not just single-step reactant similarity.",
            "Downweight similar-only positives unless supported by train-only structural analogues in the same transform-pair/context.",
            "Report exact, similar-only, and block-supported metrics separately.",
        ],
    }


def _count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 5:
        return "4-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    return "21plus"


def _fraction_bucket(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 0.02:
        return "(0,0.02)"
    if value < 0.05:
        return "[0.02,0.05)"
    if value < 0.10:
        return "[0.05,0.10)"
    if value < 0.20:
        return "[0.10,0.20)"
    return "[0.20,1]"


def _rank_bucket(rank: Any) -> str:
    if rank is None:
        return "missing"
    rank = int(rank)
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


def _rate(num: int | float, denom: int | float) -> float:
    return round(float(num) / float(denom), 6) if denom else 0.0


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CCTS Label Semantics Audit",
        "",
        "## Split Summary",
        "",
        "| Split | Groups | Candidate Rows | Positive Rows | Pos Group Coverage | Exact Group Coverage | Similar-Only Group Rate Covered | Similar-Only Positive Row Fraction | Dense Positive Group Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in (result.get("splits") or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(row.get("groups")),
                    str(row.get("candidate_rows")),
                    str(row.get("positive_rows")),
                    _fmt(row.get("positive_group_coverage")),
                    _fmt(row.get("exact_group_coverage")),
                    _fmt(row.get("similar_only_positive_group_rate_covered")),
                    _fmt(row.get("positive_rows_similar_only_fraction")),
                    _fmt(row.get("dense_positive_group_rate_covered")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Test Distributions", ""])
    test = (result.get("splits") or {}).get("test") or {}
    for key in (
        "positive_count_distribution",
        "positive_fraction_distribution",
        "positive_best_rank_distribution",
        "exact_best_rank_distribution",
        "route_domain_positive_group_rate",
    ):
        lines.extend(["", f"### {key}", "", "```json", json.dumps(test.get(key) or {}, indent=2, ensure_ascii=False), "```"])
    lines.extend(
        [
            "",
            "## Similar-Only Positive Examples",
            "",
            "```json",
            json.dumps(test.get("examples_similar_only_positive_groups") or [], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Dense Positive Examples",
            "",
            "```json",
            json.dumps(test.get("examples_dense_positive_groups") or [], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Interpretation",
            "",
            "```json",
            json.dumps(result.get("interpretation") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit CCTS label semantics")
    ap.add_argument("--train-coverage", default="results/shared/cascadebench_strict_20260516/coverage/coverage_train_top100.json")
    ap.add_argument("--val-coverage", default="results/shared/cascadebench_strict_20260516/coverage/coverage_val_top100.json")
    ap.add_argument("--test-coverage", default="results/shared/cascadebench_strict_20260516/coverage/coverage_test_top100.json")
    ap.add_argument("--cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    ap.add_argument("--test-candidates-jsonl", default="results/shared/cascadebench_strict_20260516/ccts_v1_top100/ccts_v1_test_candidates.jsonl")
    ap.add_argument("--output-json", default="results/shared/cascadebench_strict_20260516/ccts_v1_top100/ccts_label_semantics_audit.json")
    ap.add_argument("--output-md", default="results/shared/cascadebench_strict_20260516/ccts_v1_top100/CCTS_LABEL_SEMANTICS_AUDIT.zh.md")
    ap.add_argument("--similarity-threshold", type=float, default=0.7)
    ap.add_argument("--max-candidates-per-transition", type=int, default=100)
    ap.add_argument("--evidence-pool-size", type=int, default=80)
    args = ap.parse_args()
    result = audit_ccts_label_semantics(
        train_coverage=Path(args.train_coverage),
        val_coverage=Path(args.val_coverage),
        test_coverage=Path(args.test_coverage),
        cache=Path(args.cache),
        test_candidates_jsonl=Path(args.test_candidates_jsonl) if args.test_candidates_jsonl else None,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        similarity_threshold=args.similarity_threshold,
        max_candidates_per_transition=args.max_candidates_per_transition,
        evidence_pool_size=args.evidence_pool_size,
    )
    print(
        json.dumps(
            {
                "test": (result.get("splits") or {}).get("test"),
                "interpretation": result.get("interpretation"),
                "outputs": {"json": args.output_json, "md": args.output_md},
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

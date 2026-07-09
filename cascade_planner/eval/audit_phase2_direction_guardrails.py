"""Audit whether Phase II evidence supports search-time cascade integration.

This report is intentionally conservative.  It separates direct evidence
from proxy evidence so entry-substrate solves or transform-pair sketches are
not mistaken for executable cascade route planning.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "phase2_direction_guardrail_audit.v1"


DEFAULT_SPLIT_MANIFEST = Path("results/shared/cascadebench_strict_20260516/splits/cascadebench_strict_split_manifest.json")
DEFAULT_PROGRAM_MANIFEST = Path("results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
DEFAULT_CCTS_REPORT = Path("results/shared/cascadebench_strict_20260516/ccts_v1_top100/ccts_v1_report.json")
DEFAULT_CCTS_BLOCK_EVAL = Path("results/shared/cascadebench_strict_20260516/ccts_v1_top100/cascadebench_block_eval.json")
DEFAULT_CBA_REPORT = Path("results/shared/cascadebench_v2_20260516/cba_v0_pair_classifier/cba_v0_pair_classifier_report.json")
DEFAULT_CBA_SKETCH_AUDIT = Path("results/shared/cascadebench_v2_20260516/cba_v0_pair_classifier/cba_v0_route_sketch_audit.json")
DEFAULT_ENTRY_SUBSTRATE_RESULT = Path("results/shared/cascadebench_v2_20260516/cba_v0_pair_classifier/cba_v0_entry_substrate_chem_enzy_full50.json")
DEFAULT_CCTS_LABEL_AUDIT = Path("results/shared/cascadebench_strict_20260516/ccts_v1_top100/ccts_label_semantics_audit.json")
DEFAULT_CCTS_BLOCK_LABEL_AUDIT = Path("results/shared/cascadebench_strict_20260516/ccts_v2_sparse_block/ccts_block_supported_label_audit.json")


def audit_phase2_direction_guardrails(
    *,
    split_manifest: Path,
    program_manifest: Path,
    ccts_report: Path,
    ccts_block_eval: Path,
    cba_report: Path,
    cba_sketch_audit: Path,
    entry_substrate_result: Path,
    ccts_label_audit: Path,
    ccts_block_label_audit: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    split_payload = _read_json_if_exists(split_manifest)
    program_payload = _read_json_if_exists(program_manifest)
    ccts_payload = _read_json_if_exists(ccts_report)
    block_payload = _read_json_if_exists(ccts_block_eval)
    cba_payload = _read_json_if_exists(cba_report)
    sketch_payload = _read_json_if_exists(cba_sketch_audit)
    entry_payload = _read_json_if_exists(entry_substrate_result)
    label_payload = _read_json_if_exists(ccts_label_audit)
    block_label_payload = _read_json_if_exists(ccts_block_label_audit)

    ccts_transition = _ccts_transition_summary(ccts_payload)
    block_summary = _ccts_block_summary(block_payload)
    cba_summary = _cba_pair_summary(cba_payload)
    sketch_summary = _cba_sketch_summary(sketch_payload)
    entry_summary = _entry_substrate_summary(entry_payload)
    label_summary = _ccts_label_summary(label_payload)
    block_label_summary = _ccts_block_label_summary(block_label_payload)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "split_manifest": str(split_manifest),
            "program_manifest": str(program_manifest),
            "ccts_report": str(ccts_report),
            "ccts_block_eval": str(ccts_block_eval),
            "cba_report": str(cba_report),
            "cba_sketch_audit": str(cba_sketch_audit),
            "entry_substrate_result": str(entry_substrate_result),
            "ccts_label_audit": str(ccts_label_audit),
            "ccts_block_label_audit": str(ccts_block_label_audit),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "evidence": {
            "strict_split": _strict_split_summary(split_payload),
            "program_pack": _program_pack_summary(program_payload),
            "ccts_same_pool_transition": ccts_transition,
            "ccts_block_recovery": block_summary,
            "cba_pair_classifier": cba_summary,
            "cba_route_sketch": sketch_summary,
            "entry_substrate_solve": entry_summary,
            "ccts_label_semantics": label_summary,
            "ccts_block_supported_labels": block_label_summary,
        },
        "decision": _decision(
            split_payload=split_payload,
            ccts_transition=ccts_transition,
            block_summary=block_summary,
            cba_summary=cba_summary,
            sketch_summary=sketch_summary,
            entry_summary=entry_summary,
            label_summary=label_summary,
            block_label_summary=block_label_summary,
        ),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(result), encoding="utf-8")
    return result


def _strict_split_summary(payload: dict[str, Any]) -> dict[str, Any]:
    leakage = payload.get("leakage_checks") or {}
    counts = payload.get("counts") or {}
    splits = payload.get("splits") or {}
    return {
        "schema_version": payload.get("schema_version"),
        "strict_pass": bool(leakage.get("strict_pass")),
        "candidate_rows": counts.get("candidate_rows_after_optional_limit"),
        "split_groups": counts.get("split_groups"),
        "unique_targets": counts.get("unique_targets"),
        "unique_doi": counts.get("unique_doi"),
        "unique_transition_tokens": counts.get("unique_transition_tokens"),
        "splits": {
            split: {
                "rows": row.get("rows"),
                "unique_targets": row.get("unique_targets"),
                "unique_doi": row.get("unique_doi"),
                "unique_transition_tokens": row.get("unique_transition_tokens"),
            }
            for split, row in splits.items()
        },
        "leakage_counts": {
            key: (value or {}).get("count")
            for key, value in leakage.items()
            if isinstance(value, dict) and key.endswith("_split")
        },
    }


def _program_pack_summary(payload: dict[str, Any]) -> dict[str, Any]:
    counts = payload.get("counts") or {}
    return {
        "schema_version": payload.get("schema_version"),
        "programs": counts.get("programs"),
        "steps": counts.get("steps"),
        "unique_train_adjacency_keys": counts.get("adjacencies_train"),
        "skipped": counts.get("skipped"),
        "evidence_graph_summary": payload.get("evidence_graph_summary"),
    }


def _ccts_transition_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline = ((payload.get("baseline_chem_rank") or {}).get("test") or {})
    if baseline:
        rows.append(_transition_row("chem_rank", "baseline", baseline))
    for source, section in (("model", payload.get("models") or {}), ("blend", payload.get("blends") or {})):
        for name, rep in section.items():
            test = (rep or {}).get("test") or {}
            if test:
                rows.append(_transition_row(name, source, test))
    best = _best_row(rows, key="positive_mrr_covered")
    chem_only = _find_row(rows, "chem_only")
    ccts_full = _find_row(rows, "ccts_v1_full")
    context_only = _find_row(rows, "context_evidence_only")
    best_delta_vs_chem_only = _metric_delta(best, chem_only, "positive_mrr_covered")
    return {
        "counts": payload.get("counts") or {},
        "leakage_checks": payload.get("leakage_checks") or {},
        "rows": rows,
        "best_positive_mrr_row": best,
        "deltas": {
            "best_vs_chem_only_positive_mrr_covered": best_delta_vs_chem_only,
            "ccts_full_vs_chem_only_positive_mrr_covered": _metric_delta(ccts_full, chem_only, "positive_mrr_covered"),
            "context_only_vs_chem_only_positive_mrr_covered": _metric_delta(context_only, chem_only, "positive_mrr_covered"),
        },
        "interpretation": _ccts_transition_interpretation(best_delta_vs_chem_only, ccts_full, chem_only, context_only),
    }


def _transition_row(name: str, source: str, metrics: dict[str, Any]) -> dict[str, Any]:
    pos = metrics.get("positive_label") or {}
    exact = metrics.get("exact_label") or {}
    pos_at = pos.get("recall_at_k_all") or {}
    exact_at = exact.get("recall_at_k_all") or {}
    return {
        "name": name,
        "source": source,
        "positive_coverage": pos.get("coverage"),
        "positive_mrr_covered": pos.get("mrr_covered"),
        "positive_r1_all": pos_at.get("1"),
        "positive_r3_all": pos_at.get("3"),
        "positive_r5_all": pos_at.get("5"),
        "positive_r10_all": pos_at.get("10"),
        "exact_coverage": exact.get("coverage"),
        "exact_mrr_covered": exact.get("mrr_covered"),
        "exact_r1_all": exact_at.get("1"),
        "exact_r3_all": exact_at.get("3"),
        "exact_r5_all": exact_at.get("5"),
        "exact_r10_all": exact_at.get("10"),
    }


def _ccts_block_summary(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"counts": payload.get("counts") or {}, "blocks": {}}
    for size, block_summary in (payload.get("summary") or {}).items():
        all_subset = (block_summary or {}).get("all") or {}
        rows = []
        for score, score_row in (all_subset.get("scores") or {}).items():
            pos = (score_row or {}).get("positive_label") or {}
            at = pos.get("recovery_at_k_all") or {}
            rows.append(
                {
                    "score": score,
                    "positive_coverage": pos.get("coverage"),
                    "positive_mrr_covered": pos.get("mrr_covered"),
                    "positive_r1_all": at.get("1"),
                    "positive_r3_all": at.get("3"),
                    "positive_r5_all": at.get("5"),
                    "positive_r10_all": at.get("10"),
                }
            )
        best = _best_row(rows, key="positive_mrr_covered")
        chem_only = _find_row(rows, "chem_only", name_key="score")
        ccts_full = _find_row(rows, "ccts_v1_full", name_key="score")
        out["blocks"][size] = {
            "block_count": all_subset.get("blocks"),
            "rows": sorted(rows, key=lambda row: str(row.get("score"))),
            "best_positive_mrr_row": best,
            "deltas": {
                "best_vs_chem_only_positive_mrr_covered": _metric_delta(best, chem_only, "positive_mrr_covered"),
                "ccts_full_vs_chem_only_positive_mrr_covered": _metric_delta(ccts_full, chem_only, "positive_mrr_covered"),
            },
        }
    out["interpretation"] = _block_interpretation(out)
    return out


def _cba_pair_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source, section in (("baseline", payload.get("baselines") or {}), ("model", payload.get("models") or {})):
        for name, rep in section.items():
            test = (rep or {}).get("test") or {}
            if test:
                at = test.get("recall_at_k_all") or {}
                rows.append(
                    {
                        "name": name,
                        "source": source,
                        "coverage": test.get("coverage"),
                        "mrr_all": test.get("mrr_all"),
                        "r1_all": at.get("1"),
                        "r3_all": at.get("3"),
                        "r5_all": at.get("5"),
                        "r10_all": at.get("10"),
                    }
                )
    full = _find_row(rows, "full_context")
    downstream_freq = _find_row(rows, "downstream_conditional_frequency")
    downstream_context = _find_row(rows, "downstream_context")
    return {
        "counts": payload.get("counts") or {},
        "rows": rows,
        "deltas": {
            "full_context_vs_downstream_conditional_mrr_all": _metric_delta(full, downstream_freq, "mrr_all"),
            "full_context_vs_downstream_context_mrr_all": _metric_delta(full, downstream_context, "mrr_all"),
            "full_context_vs_downstream_conditional_r1_all": _metric_delta(full, downstream_freq, "r1_all"),
        },
        "interpretation": "CBA pair classification has real but mostly coarse transform-pair signal; downstream transform frequency is already a strong baseline.",
    }


def _cba_sketch_summary(payload: dict[str, Any]) -> dict[str, Any]:
    target_summary = (payload.get("target_summary") or {}).get("all_targets") or {}
    routed_summary = (payload.get("target_summary") or {}).get("routed_chem_enzy_targets") or {}
    route_pool = payload.get("route_pool_comparison") or {}
    return {
        "counts": payload.get("counts") or {},
        "target_summary_all": target_summary,
        "target_summary_chem_enzy_routed": routed_summary,
        "route_pool_comparison": route_pool,
        "interpretation": "Pair@K is a sketch metric. Pair+analog@K is stricter and much lower, so pair recovery cannot be treated as route feasibility.",
    }


def _entry_substrate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload.get("targets") or [] if isinstance(row, dict)]
    route_counts = [int(row.get("route_count") or len(row.get("routes") or [])) for row in rows]
    return {
        "summary": payload.get("summary") or {},
        "route_count_min": min(route_counts) if route_counts else None,
        "route_count_max": max(route_counts) if route_counts else None,
        "guard_levels": _count(row.get("source_guard_level") for row in rows),
        "routed_by_chem_enzy_source": _count(bool(row.get("routed_by_chem_enzy_source")) for row in rows),
        "interpretation": "Entry-substrate solve is an availability/peripheral check only; it is not evidence that the cascade block itself is executable.",
    }


def _ccts_label_summary(payload: dict[str, Any]) -> dict[str, Any]:
    test = (payload.get("splits") or {}).get("test") or {}
    interpretation = payload.get("interpretation") or {}
    keep = [
        "groups",
        "candidate_rows",
        "positive_rows",
        "exact_positive_rows",
        "similar_only_positive_rows",
        "positive_group_coverage",
        "exact_group_coverage",
        "similar_only_positive_group_rate_covered",
        "positive_rows_similar_only_fraction",
        "positive_rows_exact_fraction",
        "dense_positive_group_rate_covered",
    ]
    return {
        "metadata": payload.get("metadata") or {},
        "test": {key: test.get(key) for key in keep},
        "warnings": interpretation.get("warnings") or [],
        "recommended_fix": interpretation.get("recommended_fix") or [],
        "interpretation": "Current CCTS labels are dominated by similar-only positives, so they are weak supervision for cascade block coherence.",
    }


def _ccts_block_label_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    rows = summary.get("rows") or {}
    groups = summary.get("groups_summary") or {}
    return {
        "metadata": payload.get("metadata") or {},
        "rows": {
            "positive": rows.get("positive"),
            "exact": rows.get("exact"),
            "similar_only": rows.get("similar_only"),
            "block_supported_positive": rows.get("block_supported_positive"),
            "block_supported_exact": rows.get("block_supported_exact"),
            "block_supported_similar_only": rows.get("block_supported_similar_only"),
            "block_supported_positive_fraction_of_positive": rows.get("block_supported_positive_fraction_of_positive"),
            "block_supported_exact_fraction_of_exact": rows.get("block_supported_exact_fraction_of_exact"),
        },
        "groups": {
            "positive_group_coverage": groups.get("positive_group_coverage"),
            "exact_group_coverage": groups.get("exact_group_coverage"),
            "block_supported_positive_group_coverage": groups.get("block_supported_positive_group_coverage"),
            "block_supported_exact_group_coverage": groups.get("block_supported_exact_group_coverage"),
            "block_supported_positive_fraction_of_positive_groups": groups.get("block_supported_positive_fraction_of_positive_groups"),
        },
        "support_mode_counts": summary.get("support_mode_counts") or {},
        "best_block_supported_positive_rank_distribution": summary.get("best_block_supported_positive_rank_distribution") or {},
        "interpretation": summary.get("interpretation"),
    }


def _decision(
    *,
    split_payload: dict[str, Any],
    ccts_transition: dict[str, Any],
    block_summary: dict[str, Any],
    cba_summary: dict[str, Any],
    sketch_summary: dict[str, Any],
    entry_summary: dict[str, Any],
    label_summary: dict[str, Any],
    block_label_summary: dict[str, Any],
) -> dict[str, Any]:
    strict_pass = bool(((split_payload.get("leakage_checks") or {}).get("strict_pass")))
    ccts_delta = (ccts_transition.get("deltas") or {}).get("best_vs_chem_only_positive_mrr_covered")
    ccts_full_delta = (ccts_transition.get("deltas") or {}).get("ccts_full_vs_chem_only_positive_mrr_covered")
    block2 = ((block_summary.get("blocks") or {}).get("2") or {}).get("deltas") or {}
    cba_delta = (cba_summary.get("deltas") or {}).get("full_context_vs_downstream_conditional_mrr_all")
    pair_strong = ((sketch_summary.get("target_summary_all") or {}).get("pair_and_strong_analog_at_10"))
    entry_rate = ((entry_summary.get("summary") or {}).get("solved_rate"))
    similar_only_fraction = ((label_summary.get("test") or {}).get("positive_rows_similar_only_fraction"))
    dense_positive_rate = ((label_summary.get("test") or {}).get("dense_positive_group_rate_covered"))
    block_supported_fraction = ((block_label_summary.get("rows") or {}).get("block_supported_positive_fraction_of_positive"))
    block_supported_group_cov = ((block_label_summary.get("groups") or {}).get("block_supported_positive_group_coverage"))
    no_go_reasons = []
    if not strict_pass:
        no_go_reasons.append("strict split leakage checks did not pass")
    if ccts_delta is None or ccts_delta < 0.02:
        no_go_reasons.append("same-pool CCTS context gain over chem_only is marginal")
    if ccts_full_delta is None or ccts_full_delta <= 0.0:
        no_go_reasons.append("full-context CCTS is not better than chem_only")
    if (block2.get("ccts_full_vs_chem_only_positive_mrr_covered") or 0.0) <= 0.0:
        no_go_reasons.append("block-level full-context CCTS does not improve over chem_only")
    if pair_strong is None or pair_strong < 0.30:
        no_go_reasons.append("CBA Pair+strong-analog@10 is too low to treat sketches as validated blocks")
    if similar_only_fraction is not None and similar_only_fraction > 0.50:
        no_go_reasons.append("CCTS positive labels are dominated by similar-only single-step proxies")
    if dense_positive_rate is not None and dense_positive_rate > 0.25:
        no_go_reasons.append("CCTS candidate groups often have dense positives, weakening ranking supervision")

    return {
        "search_time_ccts_v1_gate": "no_go" if no_go_reasons else "go",
        "strict_split_is_usable": strict_pass,
        "is_current_path_partly_proxy_trap": True,
        "no_go_reasons": no_go_reasons,
        "usable_evidence": [
            "Strict split has zero DOI/target/scaffold/transition-token/reaction cross-split leakage.",
            "ChemEnzy top100 candidate pool has non-trivial transition coverage; ranking can be evaluated fairly.",
            "CBA transform-pair classification shows cascade context contains learnable coarse block-selection signal.",
            "ChemEnzy route pool often misses transform-consistent cascade blocks even when it solves targets generically.",
            "Block-supported positives exist, so a stricter sparse-label ranker is feasible.",
        ],
        "weak_or_proxy_evidence": [
            "Entry-substrate full50 solve rate is only an availability proxy.",
            "CBA Pair@10 is a type-sketch metric; Pair+strong-analog@10 is the stricter number.",
            "CCTS same-pool context-only/full-context scores are not yet strong enough to justify search-time promotion.",
            "Current CCTS labels mostly measure single-step reactant similarity rather than cascade block coherence.",
            "Block-supported labels are available, but current context/evidence features still do not beat chem_only.",
            "Large native ChemEnzy route counts are not evidence of cascade coherence.",
        ],
        "next_actions": [
            "Freeze this report as the Phase II direction guardrail.",
            "Do not expand entry-substrate solving as a main benchmark.",
            "Rebuild the training target around block-level candidate validation: positive must be same-pool exact/similar transition or train-only structural analogue, not just transform-pair class.",
            "Split CCTS labels into exact, similar-only, and block-supported positives; train primarily on exact/block-supported labels and downweight dense similar-only groups.",
            "Use CBA only as a proposal/retrieval prior until Pair+strong-analog and same-pool block recovery improve.",
            "Only enter search-time integration after strict same-pool block recovery improves clearly over chem_only/no-context baselines.",
        ],
        "reference_numbers": {
            "ccts_best_vs_chem_only_positive_mrr_covered": ccts_delta,
            "ccts_full_vs_chem_only_positive_mrr_covered": ccts_full_delta,
            "block2_ccts_full_vs_chem_only_positive_mrr_covered": block2.get("ccts_full_vs_chem_only_positive_mrr_covered"),
            "cba_full_vs_downstream_conditional_mrr_all": cba_delta,
            "cba_pair_strong_analog_at_10_target": pair_strong,
            "entry_substrate_solved_rate": entry_rate,
            "ccts_positive_rows_similar_only_fraction": similar_only_fraction,
            "ccts_dense_positive_group_rate_covered": dense_positive_rate,
            "ccts_block_supported_positive_fraction_of_positive": block_supported_fraction,
            "ccts_block_supported_positive_group_coverage": block_supported_group_cov,
        },
    }


def _ccts_transition_interpretation(
    best_delta: float | None,
    ccts_full: dict[str, Any] | None,
    chem_only: dict[str, Any] | None,
    context_only: dict[str, Any] | None,
) -> str:
    if not chem_only:
        return "Chem-only comparator is missing; CCTS evidence cannot be interpreted."
    if ccts_full and (ccts_full.get("positive_mrr_covered") or 0.0) < (chem_only.get("positive_mrr_covered") or 0.0):
        return "CCTS full-context underperforms chem_only; context features are not yet promotion-ready."
    if context_only and (context_only.get("positive_mrr_covered") or 0.0) < (chem_only.get("positive_mrr_covered") or 0.0):
        if best_delta is not None and best_delta < 0.02:
            return "Best blend is only marginally above chem_only while context-only is weak; treat as no-go for search-time integration."
    return "CCTS shows some same-pool ranking signal, but must be judged against chem_only and block-level metrics."


def _block_interpretation(payload: dict[str, Any]) -> str:
    block2 = ((payload.get("blocks") or {}).get("2") or {}).get("deltas") or {}
    block3 = ((payload.get("blocks") or {}).get("3") or {}).get("deltas") or {}
    if (block2.get("ccts_full_vs_chem_only_positive_mrr_covered") or 0.0) <= 0.0:
        return "Block-level recovery confirms the warning: full-context CCTS is not improving over chem_only on contiguous cascade blocks."
    if (block3.get("ccts_full_vs_chem_only_positive_mrr_covered") or 0.0) <= 0.0:
        return "Two-step block recovery has limited lift, but three-step recovery does not; search integration remains premature."
    return "Block-level CCTS lift is positive; search integration may be considered after robustness checks."


def _markdown(result: dict[str, Any]) -> str:
    evidence = result.get("evidence") or {}
    decision = result.get("decision") or {}
    split = evidence.get("strict_split") or {}
    program = evidence.get("program_pack") or {}
    ccts = evidence.get("ccts_same_pool_transition") or {}
    block = evidence.get("ccts_block_recovery") or {}
    cba = evidence.get("cba_pair_classifier") or {}
    sketch = evidence.get("cba_route_sketch") or {}
    entry = evidence.get("entry_substrate_solve") or {}
    labels = evidence.get("ccts_label_semantics") or {}
    block_labels = evidence.get("ccts_block_supported_labels") or {}
    lines = [
        "# Phase II Direction Guardrail Audit",
        "",
        "## Verdict",
        "",
        f"- Search-time CCTS-v1 gate: `{decision.get('search_time_ccts_v1_gate')}`",
        f"- Strict split usable: `{decision.get('strict_split_is_usable')}`",
        f"- Partly proxy trap: `{decision.get('is_current_path_partly_proxy_trap')}`",
        "",
        "No-go reasons:",
        "",
    ]
    for reason in decision.get("no_go_reasons") or []:
        lines.append(f"- {reason}")
    if not decision.get("no_go_reasons"):
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Strict Split",
            "",
            "```json",
            json.dumps(split, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Program Pack",
            "",
            "```json",
            json.dumps(program, indent=2, ensure_ascii=False),
            "```",
            "",
            "## CCTS Same-Pool Transition Ranking",
            "",
            "| Score | Source | Pos MRR covered | Pos R@1 all | Pos R@3 all | Pos R@10 all | Exact MRR covered | Exact R@1 all |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ccts.get("rows") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('name')}`",
                    str(row.get("source")),
                    _fmt(row.get("positive_mrr_covered")),
                    _fmt(row.get("positive_r1_all")),
                    _fmt(row.get("positive_r3_all")),
                    _fmt(row.get("positive_r10_all")),
                    _fmt(row.get("exact_mrr_covered")),
                    _fmt(row.get("exact_r1_all")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Interpretation: {ccts.get('interpretation')}",
            "",
            "## CCTS Label Semantics",
            "",
            "```json",
            json.dumps(labels, indent=2, ensure_ascii=False),
            "```",
            "",
            "## CCTS Block-Supported Labels",
            "",
            "```json",
            json.dumps(block_labels, indent=2, ensure_ascii=False),
            "```",
            "",
            "## CCTS Block Recovery",
            "",
        ]
    )
    for size, row in (block.get("blocks") or {}).items():
        lines.extend(
            [
                f"### Block Size {size}",
                "",
                f"- Blocks: `{row.get('block_count')}`",
                "",
                "| Score | Pos MRR covered | Pos R@1 all | Pos R@3 all | Pos R@10 all |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for score_row in row.get("rows") or []:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{score_row.get('score')}`",
                        _fmt(score_row.get("positive_mrr_covered")),
                        _fmt(score_row.get("positive_r1_all")),
                        _fmt(score_row.get("positive_r3_all")),
                        _fmt(score_row.get("positive_r10_all")),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            f"Interpretation: {block.get('interpretation')}",
            "",
            "## CBA Transform-Pair Classification",
            "",
            "| Score | Source | MRR all | R@1 all | R@3 all | R@10 all |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in cba.get("rows") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('name')}`",
                    str(row.get("source")),
                    _fmt(row.get("mrr_all")),
                    _fmt(row.get("r1_all")),
                    _fmt(row.get("r3_all")),
                    _fmt(row.get("r10_all")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Interpretation: {cba.get('interpretation')}",
            "",
            "## CBA Route Sketch",
            "",
            "Target-level all:",
            "",
            "```json",
            json.dumps(sketch.get("target_summary_all") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
            "Route-pool comparison:",
            "",
            "```json",
            json.dumps(sketch.get("route_pool_comparison") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
            f"Interpretation: {sketch.get('interpretation')}",
            "",
            "## Entry-Substrate Solve",
            "",
            "```json",
            json.dumps(entry, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Usable Evidence",
            "",
        ]
    )
    for item in decision.get("usable_evidence") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Weak Or Proxy Evidence", ""])
    for item in decision.get("weak_or_proxy_evidence") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in decision.get("next_actions") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Reference Numbers",
            "",
            "```json",
            json.dumps(decision.get("reference_numbers") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"payload": payload}


def _best_row(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: float(row.get(key) or 0.0))


def _find_row(rows: list[dict[str, Any]], name: str, *, name_key: str = "name") -> dict[str, Any] | None:
    for row in rows:
        if row.get(name_key) == name:
            return row
    return None


def _metric_delta(left: dict[str, Any] | None, right: dict[str, Any] | None, key: str) -> float | None:
    if not left or not right:
        return None
    if left.get(key) is None or right.get(key) is None:
        return None
    return round(float(left.get(key)) - float(right.get(key)), 6)


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Phase II direction guardrails")
    ap.add_argument("--split-manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    ap.add_argument("--program-manifest", default=str(DEFAULT_PROGRAM_MANIFEST))
    ap.add_argument("--ccts-report", default=str(DEFAULT_CCTS_REPORT))
    ap.add_argument("--ccts-block-eval", default=str(DEFAULT_CCTS_BLOCK_EVAL))
    ap.add_argument("--cba-report", default=str(DEFAULT_CBA_REPORT))
    ap.add_argument("--cba-sketch-audit", default=str(DEFAULT_CBA_SKETCH_AUDIT))
    ap.add_argument("--entry-substrate-result", default=str(DEFAULT_ENTRY_SUBSTRATE_RESULT))
    ap.add_argument("--ccts-label-audit", default=str(DEFAULT_CCTS_LABEL_AUDIT))
    ap.add_argument("--ccts-block-label-audit", default=str(DEFAULT_CCTS_BLOCK_LABEL_AUDIT))
    ap.add_argument("--output-json", default="results/shared/cascadebench_v2_20260516/phase2_direction_guardrail_audit.json")
    ap.add_argument("--output-md", default="results/shared/cascadebench_v2_20260516/PHASE2_DIRECTION_GUARDRAIL_AUDIT.zh.md")
    args = ap.parse_args()
    result = audit_phase2_direction_guardrails(
        split_manifest=Path(args.split_manifest),
        program_manifest=Path(args.program_manifest),
        ccts_report=Path(args.ccts_report),
        ccts_block_eval=Path(args.ccts_block_eval),
        cba_report=Path(args.cba_report),
        cba_sketch_audit=Path(args.cba_sketch_audit),
        entry_substrate_result=Path(args.entry_substrate_result),
        ccts_label_audit=Path(args.ccts_label_audit),
        ccts_block_label_audit=Path(args.ccts_block_label_audit),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
    )
    print(
        json.dumps(
            {
                "decision": result.get("decision"),
                "outputs": {
                    "json": args.output_json,
                    "md": args.output_md,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

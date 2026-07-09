"""Summarize route/block model evidence and promotion gates.

This report is meant to prevent the current phase from drifting back into
weak proxy wins.  It consolidates the strongest route/block artifacts and asks
whether the learned scorer beats both ChemEnzy native rank and the simpler
retrieval-only evidence controls.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_block_strengthening_summary.v1"


def summarize_route_block_strengthening(
    *,
    route_pool_report: Path,
    route_block_value_report: Path | None = None,
    no_human_ablation_summary: Path | None = None,
    final_rerank_replay: Path | None = None,
    ablation_summary: Path,
    bootstrap_stability: Path,
    transition_hardneg_summary: Path,
    runtime_nohuman_probe: Path | None = None,
    guarded_search_comparison: Path,
    output_json: Path,
    output_md: Path | None = None,
    min_mrr_delta_vs_retrieval: float = 0.03,
    min_live_quality_delta: float = 0.01,
) -> dict[str, Any]:
    route_pool = _read_json(route_pool_report)
    route_block_value = _read_json(route_block_value_report) if route_block_value_report else {}
    no_human = _read_json(no_human_ablation_summary) if no_human_ablation_summary else {}
    final_rerank = _read_json(final_rerank_replay) if final_rerank_replay else {}
    ablation = _read_json(ablation_summary)
    bootstrap = _read_json(bootstrap_stability)
    hardneg = _read_json(transition_hardneg_summary)
    nohuman_probe = _read_json(runtime_nohuman_probe) if runtime_nohuman_probe else {}
    guarded = _read_json(guarded_search_comparison)

    route_pool_summary = _route_pool_summary(route_pool, ablation, bootstrap)
    route_block_value_summary = _route_block_value_summary(route_block_value)
    no_human_summary = _no_human_summary(no_human)
    hardneg_summary = _hardneg_summary(hardneg)
    nohuman_probe_summary = _runtime_nohuman_probe_summary(nohuman_probe)
    guarded_summary = _guarded_summary(guarded)
    gates = {
        "fixed_pool": _fixed_pool_gates(
            no_human_summary,
            route_pool_summary,
            min_mrr_delta_vs_retrieval=min_mrr_delta_vs_retrieval,
        ),
        "route_pool": _route_pool_gates(
            route_pool_summary,
            min_mrr_delta_vs_retrieval=min_mrr_delta_vs_retrieval,
        ),
        "runtime_hard_negative": _hardneg_gates(
            hardneg_summary,
            min_mrr_delta_vs_retrieval=min_mrr_delta_vs_retrieval,
        ),
        "guarded_live_search": _guarded_gates(
            guarded_summary,
            min_live_quality_delta=min_live_quality_delta,
        ),
    }
    promote = all(
        [
            gates["fixed_pool"]["learned_beats_retrieval"]["ok"],
            gates["runtime_hard_negative"]["learned_beats_retrieval"]["ok"],
            gates["guarded_live_search"]["quality_lift"]["ok"],
        ]
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "route_pool_report": str(route_pool_report),
            "route_block_value_report": str(route_block_value_report) if route_block_value_report else None,
            "no_human_ablation_summary": str(no_human_ablation_summary) if no_human_ablation_summary else None,
            "final_rerank_replay": str(final_rerank_replay) if final_rerank_replay else None,
            "ablation_summary": str(ablation_summary),
            "bootstrap_stability": str(bootstrap_stability),
            "transition_hardneg_summary": str(transition_hardneg_summary),
            "runtime_nohuman_probe": str(runtime_nohuman_probe) if runtime_nohuman_probe else None,
            "guarded_search_comparison": str(guarded_search_comparison),
            "thresholds": {
                "min_mrr_delta_vs_retrieval": float(min_mrr_delta_vs_retrieval),
                "min_live_quality_delta": float(min_live_quality_delta),
            },
        },
        "route_pool": route_pool_summary,
        "route_block_value": route_block_value_summary,
        "no_human_route_block_value": no_human_summary,
        "final_rerank_replay": _final_rerank_summary(final_rerank),
        "runtime_hard_negative": hardneg_summary,
        "runtime_hard_negative_nohuman_probe": nohuman_probe_summary,
        "guarded_live_search": guarded_summary,
        "gates": gates,
        "decision": {
            "promote_route_block_scorer": bool(promote),
            "status": "promote" if promote else "do_not_promote_yet",
            "reason": _decision_reason(gates),
            "next_actions": [
                "Use the strict runtime train-provenance value pack for retrieval-conditioned claims.",
                "Continue no-human route/block weak-supervision training; do not depend on expert CSV labels.",
                "Require learned scorer to beat retrieval-only and audit controls before search-time promotion.",
                "Use statin panel as qualitative application review, not training-specific optimization.",
            ],
        },
    }
    output_json = Path(output_json)
    output_md = Path(output_md) if output_md else output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(result), encoding="utf-8")
    return result


def _route_pool_summary(route_pool: dict[str, Any], ablation: dict[str, Any], bootstrap: dict[str, Any]) -> dict[str, Any]:
    counts = route_pool.get("counts") or {}
    baselines = route_pool.get("baselines") or {}
    native_test = ((baselines.get("native_rank") or {}).get("test") or {})
    ccts_mean_test = ((baselines.get("ccts_model_mean") or {}).get("test") or {})
    ccts_best_test = ((baselines.get("ccts_best_route_evidence") or {}).get("test") or {})
    block_test = ((baselines.get("block_rerank_score") or {}).get("test") or {})
    model_test = ((route_pool.get("model") or {}).get("test") or {})
    selection = route_pool.get("selection") or {}
    feature_rows = list(ablation.get("feature_sets") or [])
    feature_by_name = {str(row.get("feature_set")): row for row in feature_rows}
    observed = bootstrap.get("observed_mrr") or {}
    deltas = bootstrap.get("deltas") or {}
    best_feature = max(feature_rows, key=lambda row: _float(row.get("model_mrr_covered")), default={})
    return {
        "counts": counts,
        "native_mrr": _float(native_test.get("mrr_covered")),
        "native_recall_at3_all": _recall(native_test, 3),
        "learned_all_mrr": _float(model_test.get("mrr_covered")),
        "selected_method": selection.get("selected_method"),
        "selected_mrr": _float(selection.get("selected_test_mrr_covered")),
        "ccts_model_mean_mrr": _float(ccts_mean_test.get("mrr_covered")),
        "ccts_best_route_evidence_mrr": _float(ccts_best_test.get("mrr_covered")),
        "block_rerank_mrr": _float(block_test.get("mrr_covered")),
        "best_feature_set": {
            "feature_set": best_feature.get("feature_set"),
            "model_mrr": _float(best_feature.get("model_mrr_covered")),
            "model_recall_at3_all": _float(best_feature.get("model_recall_at3_all")),
        },
        "cascade_only": _feature_summary(feature_by_name.get("cascade_only") or {}),
        "no_cascade": _feature_summary(feature_by_name.get("no_cascade") or {}),
        "no_v4": _feature_summary(feature_by_name.get("no_v4") or {}),
        "ccts_only": _feature_summary(feature_by_name.get("ccts_only") or {}),
        "bootstrap": {
            "n_positive_groups": int(bootstrap.get("n_positive_groups") or 0),
            "samples": int(bootstrap.get("bootstrap_samples") or 0),
            "observed_mrr": observed,
            "cascade_only_minus_native": deltas.get("model_cascade_only_minus_native_rank") or {},
            "cascade_only_minus_retrieval_proxy": deltas.get("model_cascade_only_minus_ccts_model_mean") or {},
            "cascade_only_minus_no_cascade": deltas.get("model_cascade_only_minus_model_no_cascade") or {},
            "no_v4_minus_retrieval_proxy": deltas.get("model_no_v4_minus_ccts_model_mean") or {},
        },
    }


def _route_block_value_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {"present": False}
    return {
        "present": True,
        "counts": report.get("counts") or {},
        "split_counts": report.get("split_counts") or {},
        "evidence_provenance_audit": report.get("evidence_provenance_audit") or {},
        "training_contract": report.get("training_contract") or {},
    }


def _no_human_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {"present": False}
    models = list(report.get("models") or [])
    best_no_audit_no_retrieval = max(
        [row for row in models if "no_audit_no_retrieval" in str(row.get("model") or "")],
        key=lambda row: _float(row.get("model_minus_retrieval_mrr")),
        default={},
    )
    best_ablation = max(models, key=lambda row: _float(row.get("model_mrr")), default={})
    return {
        "present": True,
        "decision": report.get("decision") or {},
        "models": models,
        "best_ablation": best_ablation,
        "best_no_audit_no_retrieval": best_no_audit_no_retrieval,
        "control_model_minus_retrieval_mrr": _float(best_no_audit_no_retrieval.get("model_minus_retrieval_mrr")),
    }


def _final_rerank_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {"present": False}
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    model = metrics.get("route_block_value_model") if isinstance(metrics.get("route_block_value_model"), dict) else {}
    native = metrics.get("native_rank") if isinstance(metrics.get("native_rank"), dict) else {}
    retrieval = metrics.get("retrieval_only") if isinstance(metrics.get("retrieval_only"), dict) else {}
    audit = metrics.get("audit_guard") if isinstance(metrics.get("audit_guard"), dict) else {}
    return {
        "present": True,
        "decision": report.get("decision") or {},
        "counts": report.get("counts") or {},
        "model_mrr": _float(model.get("mrr_covered")),
        "native_mrr": _float(native.get("mrr_covered")),
        "retrieval_mrr": _float(retrieval.get("mrr_covered")),
        "audit_mrr": _float(audit.get("mrr_covered")),
        "model_minus_native_mrr": _float((report.get("deltas") or {}).get("model_minus_native_mrr")),
        "model_minus_retrieval_mrr": _float((report.get("deltas") or {}).get("model_minus_retrieval_mrr")),
        "model_minus_audit_mrr": _float((report.get("deltas") or {}).get("model_minus_audit_mrr")),
        "top_route_changed_vs_native": int(model.get("top_route_changed_vs_native") or 0),
    }


def _hardneg_summary(hardneg: dict[str, Any]) -> dict[str, Any]:
    selection = hardneg.get("selection") or {}
    rows = list(hardneg.get("test_metric_rows") or [])
    chem_block = _find_metric(rows, method="chem_rank", label="block_supported_positive_label")
    selected_block = _find_metric(rows, method=selection.get("selected_method"), label="block_supported_positive_label")
    retrieval_block = _best_row(
        [
            row
            for row in rows
            if row.get("family") == "nonlearned_blends"
            and row.get("label") == "block_supported_positive_label"
        ],
        key="mrr_covered",
    )
    chem_exact = _find_metric(rows, method="chem_rank", label="exact_label")
    selected_exact = _find_metric(rows, method=selection.get("selected_method"), label="exact_label")
    return {
        "selected_method": selection.get("selected_method"),
        "chem_block_mrr": _float(chem_block.get("mrr_covered")),
        "selected_block_mrr": _float(selected_block.get("mrr_covered") or selection.get("selected_test_block_mrr")),
        "retrieval_only_block_mrr": _float(retrieval_block.get("mrr_covered")),
        "selected_delta_vs_chem_block_mrr": _float(selection.get("selected_delta_vs_chem_block_mrr")),
        "selected_delta_vs_retrieval_block_mrr": round(
            _float(selected_block.get("mrr_covered") or selection.get("selected_test_block_mrr"))
            - _float(retrieval_block.get("mrr_covered")),
            6,
        ),
        "block_coverage": _float(chem_block.get("coverage")),
        "chem_exact_mrr": _float(chem_exact.get("mrr_covered")),
        "selected_exact_mrr": _float(selected_exact.get("mrr_covered")),
        "exact_coverage": _float(chem_exact.get("coverage")),
    }


def _runtime_nohuman_probe_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {"present": False}
    gate = report.get("gate") if isinstance(report.get("gate"), dict) else {}
    material = report.get("material_sanity_pairwise_probe") if isinstance(report.get("material_sanity_pairwise_probe"), dict) else {}
    hgb = report.get("hgb_runtime_probe") if isinstance(report.get("hgb_runtime_probe"), dict) else {}
    return {
        "present": True,
        "status": (report.get("decision") or {}).get("status"),
        "reason": (report.get("decision") or {}).get("reason"),
        "retrieval_test_block_mrr": _float(gate.get("retrieval_test_block_mrr")),
        "best_blend_test_block_mrr": _float(gate.get("best_blend_test_block_mrr")),
        "best_delta_vs_retrieval": _float(gate.get("best_delta_vs_retrieval")),
        "required_delta_vs_retrieval": _float(gate.get("required_delta_vs_retrieval")),
        "passed": bool(gate.get("passed")),
        "best_learned_probe": gate.get("best_learned_probe"),
        "material_selected": material.get("selected_by_val") or {},
        "hgb_selected": hgb.get("selected_by_val") or {},
    }


def _guarded_summary(guarded: dict[str, Any]) -> dict[str, Any]:
    runs = list(guarded.get("runs") or [])
    baseline = next((run for run in runs if run.get("name") == "baseline"), {})
    candidates = [run for run in runs if run.get("name") != "baseline"]
    best_changed = max(candidates, key=lambda run: int(run.get("top_route_changed_vs_baseline") or 0), default={})
    best_applied = max(
        candidates,
        key=lambda run: int(((run.get("pair_diagnostics") or {}).get("cascade_pair_reward_applied_true") or 0)),
        default={},
    )
    return {
        "baseline": _run_summary(baseline),
        "best_changed": _run_summary(best_changed),
        "best_applied": _run_summary(best_applied),
        "best_quality": _run_summary(_best_quality_run(candidates, baseline)),
        "n_candidate_runs": len(candidates),
    }


def _route_pool_gates(summary: dict[str, Any], *, min_mrr_delta_vs_retrieval: float) -> dict[str, Any]:
    bootstrap = summary.get("bootstrap") or {}
    native_delta = bootstrap.get("cascade_only_minus_native") or {}
    retrieval_delta = bootstrap.get("cascade_only_minus_retrieval_proxy") or {}
    no_cascade_delta = bootstrap.get("cascade_only_minus_no_cascade") or {}
    return {
        "learned_beats_native": _ci_gate(native_delta, min_delta=0.0),
        "learned_beats_retrieval": _ci_gate(retrieval_delta, min_delta=min_mrr_delta_vs_retrieval),
        "cascade_context_beats_no_cascade": _ci_gate(no_cascade_delta, min_delta=0.0),
        "positive_group_count": int(bootstrap.get("n_positive_groups") or 0),
    }


def _fixed_pool_gates(
    no_human: dict[str, Any],
    route_pool: dict[str, Any],
    *,
    min_mrr_delta_vs_retrieval: float,
) -> dict[str, Any]:
    decision = no_human.get("decision") if isinstance(no_human.get("decision"), dict) else {}
    if no_human.get("present") and decision:
        actual = _float(decision.get("control_model_minus_retrieval_mrr"))
        required = _float(decision.get("required_model_minus_retrieval_mrr")) or float(min_mrr_delta_vs_retrieval)
        return {
            "source": "no_human_route_block_value",
            "learned_beats_retrieval": {
                "ok": bool(decision.get("strict_fixed_pool_gate_passed")),
                "actual": actual,
                "required_min": required,
                "control_model": decision.get("control_model"),
                "control_positive_task": decision.get("control_positive_task"),
            },
        }
    legacy = _route_pool_gates(route_pool, min_mrr_delta_vs_retrieval=min_mrr_delta_vs_retrieval)
    return {
        "source": "legacy_route_pool_scorer",
        "learned_beats_retrieval": legacy["learned_beats_retrieval"],
    }


def _hardneg_gates(summary: dict[str, Any], *, min_mrr_delta_vs_retrieval: float) -> dict[str, Any]:
    return {
        "learned_beats_chem_rank": {
            "ok": summary["selected_delta_vs_chem_block_mrr"] > 0,
            "actual": summary["selected_delta_vs_chem_block_mrr"],
            "required_min": 0.0,
        },
        "learned_beats_retrieval": {
            "ok": summary["selected_delta_vs_retrieval_block_mrr"] >= float(min_mrr_delta_vs_retrieval),
            "actual": summary["selected_delta_vs_retrieval_block_mrr"],
            "required_min": float(min_mrr_delta_vs_retrieval),
        },
        "candidate_coverage": {
            "block_supported_coverage": summary["block_coverage"],
            "exact_coverage": summary["exact_coverage"],
        },
    }


def _guarded_gates(summary: dict[str, Any], *, min_live_quality_delta: float) -> dict[str, Any]:
    baseline = summary.get("baseline") or {}
    best = summary.get("best_quality") or summary.get("best_applied") or {}
    deltas = {
        "cascade_solved_rate": _float(best.get("cascade_solved_rate")) - _float(baseline.get("cascade_solved_rate")),
        "stock_closed_rate": _float(best.get("stock_closed_rate")) - _float(baseline.get("stock_closed_rate")),
        "top_result_exact_reaction_in_pool": (
            _float(best.get("top_result_exact_reaction_in_pool"))
            - _float(baseline.get("top_result_exact_reaction_in_pool"))
        ),
        "top_result_gt_reactant_in_pool": (
            _float(best.get("top_result_gt_reactant_in_pool"))
            - _float(baseline.get("top_result_gt_reactant_in_pool"))
        ),
        "result_exact_reaction_in_pool": (
            _float(best.get("result_exact_reaction_in_pool")) - _float(baseline.get("result_exact_reaction_in_pool"))
        ),
        "result_gt_reactant_in_pool": (
            _float(best.get("result_gt_reactant_in_pool")) - _float(baseline.get("result_gt_reactant_in_pool"))
        ),
    }
    quality_delta = max(deltas["top_result_exact_reaction_in_pool"], deltas["top_result_gt_reactant_in_pool"])
    return {
        "no_guardrail_regression": {
            "ok": deltas["cascade_solved_rate"] >= 0 and deltas["stock_closed_rate"] >= 0,
            "deltas": deltas,
        },
        "quality_lift": {
            "ok": quality_delta >= float(min_live_quality_delta),
            "actual": round(quality_delta, 6),
            "required_min": float(min_live_quality_delta),
        },
        "search_actually_changed": {
            "ok": int(best.get("top_route_changed_vs_baseline") or 0) > 0,
            "top_route_changed_vs_baseline": int(best.get("top_route_changed_vs_baseline") or 0),
            "pair_rewards_applied": int(best.get("pair_rewards_applied") or 0),
            "final_rerank_changed": int(best.get("final_rerank_changed") or 0),
            "product_audit_final_rerank_changed": int(best.get("product_audit_final_rerank_changed") or 0),
            "product_audit_final_rerank_enabled_targets": int(
                best.get("product_audit_final_rerank_enabled_targets") or 0
            ),
        },
    }


def _ci_gate(delta: dict[str, Any], *, min_delta: float) -> dict[str, Any]:
    observed = _float(delta.get("observed_delta"))
    ci_low = _float(delta.get("ci95_low"))
    return {
        "ok": observed >= float(min_delta) and ci_low > 0,
        "observed_delta": observed,
        "ci95_low": ci_low,
        "ci95_high": _float(delta.get("ci95_high")),
        "p_delta_le_0": _float(delta.get("p_delta_le_0")),
        "required_min_observed_delta": float(min_delta),
        "requires_ci_low_gt_0": True,
    }


def _decision_reason(gates: dict[str, Any]) -> str:
    missing = []
    if not gates["fixed_pool"]["learned_beats_retrieval"]["ok"]:
        missing.append("fixed-pool learned scorer does not clear retrieval-only control")
    if not gates["runtime_hard_negative"]["learned_beats_retrieval"]["ok"]:
        missing.append("runtime hard-negative learned scorer does not clear retrieval-only control")
    if not gates["guarded_live_search"]["quality_lift"]["ok"]:
        missing.append("guarded live search has no aggregate quality lift")
    if not missing:
        return "all route/block promotion gates passed"
    return "; ".join(missing)


def _markdown(result: dict[str, Any]) -> str:
    route = result["route_pool"]
    route_block = result.get("route_block_value") or {}
    no_human = result.get("no_human_route_block_value") or {}
    final_rerank = result.get("final_rerank_replay") or {}
    hard = result["runtime_hard_negative"]
    nohuman_probe = result.get("runtime_hard_negative_nohuman_probe") or {}
    live = result["guarded_live_search"]
    decision = result["decision"]
    lines = [
        "# Route/Block Strengthening Summary",
        "",
        f"Decision: `{decision['status']}`",
        "",
        decision["reason"],
        "",
        "## Route-Pool Evidence",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| positive test groups | {route['counts'].get('test_positive_groups', '')} |",
        f"| native MRR | {_fmt(route['native_mrr'])} |",
        f"| selected MRR | {_fmt(route['selected_mrr'])} |",
        f"| CCTS model-mean MRR | {_fmt(route['ccts_model_mean_mrr'])} |",
        f"| cascade-only model MRR | {_fmt((route['cascade_only'] or {}).get('model_mrr'))} |",
        f"| best feature set | `{(route['best_feature_set'] or {}).get('feature_set')}` |",
        "",
        "## Route/Block Value Pack",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| present | `{bool(route_block.get('present'))}` |",
        f"| rows | {(route_block.get('counts') or {}).get('rows', '')} |",
        f"| evidence provenance | `{((route_block.get('evidence_provenance_audit') or {}).get('status'))}` |",
        f"| missing provenance rows | {(route_block.get('evidence_provenance_audit') or {}).get('missing_retrieval_provenance_rows', '')} |",
        "",
        "## No-Human Route/Block Value",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| present | `{bool(no_human.get('present'))}` |",
        f"| expert labels required | `{((no_human.get('decision') or {}).get('expert_labels_required'))}` |",
        f"| fixed-pool signal present | `{((no_human.get('decision') or {}).get('fixed_pool_signal_present'))}` |",
        f"| fixed-pool gate passed | `{((no_human.get('decision') or {}).get('fixed_pool_training_gate_passed'))}` |",
        f"| strict fixed-pool gate passed | `{((no_human.get('decision') or {}).get('strict_fixed_pool_gate_passed'))}` |",
        f"| search-time promotion | `{((no_human.get('decision') or {}).get('promote_search_time'))}` |",
        f"| best no-audit/no-retrieval MRR | {_fmt((no_human.get('best_no_audit_no_retrieval') or {}).get('model_mrr'))} |",
        f"| retrieval baseline MRR | {_fmt((no_human.get('best_no_audit_no_retrieval') or {}).get('retrieval_mrr'))} |",
        f"| no-audit/no-retrieval minus retrieval | {_fmt(no_human.get('control_model_minus_retrieval_mrr'))} |",
        "",
        "## Final Rerank Replay",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| present | `{bool(final_rerank.get('present'))}` |",
        f"| model MRR | {_fmt(final_rerank.get('model_mrr'))} |",
        f"| native MRR | {_fmt(final_rerank.get('native_mrr'))} |",
        f"| retrieval MRR | {_fmt(final_rerank.get('retrieval_mrr'))} |",
        f"| audit MRR | {_fmt(final_rerank.get('audit_mrr'))} |",
        f"| model - retrieval | {_fmt(final_rerank.get('model_minus_retrieval_mrr'))} |",
        f"| model - audit | {_fmt(final_rerank.get('model_minus_audit_mrr'))} |",
        f"| changed vs native | {final_rerank.get('top_route_changed_vs_native', 0)} |",
        "",
        "## Runtime Hard-Negative Evidence",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| ChemEnzy block MRR | {_fmt(hard['chem_block_mrr'])} |",
        f"| retrieval-only block MRR | {_fmt(hard['retrieval_only_block_mrr'])} |",
        f"| learned selected block MRR | {_fmt(hard['selected_block_mrr'])} |",
        f"| learned - retrieval | {_fmt(hard['selected_delta_vs_retrieval_block_mrr'])} |",
        f"| block-supported coverage | {_fmt(hard['block_coverage'])} |",
        "",
        "## Runtime No-Human Probe",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| present | `{bool(nohuman_probe.get('present'))}` |",
        f"| status | `{nohuman_probe.get('status')}` |",
        f"| retrieval test block MRR | {_fmt(nohuman_probe.get('retrieval_test_block_mrr'))} |",
        f"| best blend test block MRR | {_fmt(nohuman_probe.get('best_blend_test_block_mrr'))} |",
        f"| best delta vs retrieval | {_fmt(nohuman_probe.get('best_delta_vs_retrieval'))} |",
        f"| required delta | {_fmt(nohuman_probe.get('required_delta_vs_retrieval'))} |",
        f"| best learned probe | `{nohuman_probe.get('best_learned_probe')}` |",
        "",
        "## Guarded Live Search",
        "",
        "| Run | solved | stock | top exact | top GT | any exact | any GT | pair applied | final rerank changed | product audit changed | changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _live_row("baseline", live.get("baseline") or {}),
        _live_row("best_quality", live.get("best_quality") or live.get("best_applied") or {}),
        "",
        "## Gates",
        "",
    ]
    for section, checks in result["gates"].items():
        lines.append(f"### {section}")
        lines.append("")
        if isinstance(checks, dict) and checks.get("source"):
            lines.append(f"- `source`: `{checks.get('source')}`")
        for name, payload in checks.items():
            if isinstance(payload, dict) and "ok" in payload:
                lines.append(f"- `{name}`: `{payload['ok']}` ({_gate_detail(payload)})")
        lines.append("")
    lines.extend(["## Next Actions", ""])
    lines.extend(f"- {item}" for item in decision["next_actions"])
    lines.append("")
    return "\n".join(lines)


def _feature_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_set": row.get("feature_set"),
        "model_mrr": _float(row.get("model_mrr_covered")),
        "model_recall_at1_all": _float(row.get("model_recall_at1_all")),
        "model_recall_at3_all": _float(row.get("model_recall_at3_all")),
        "model_recall_at5_all": _float(row.get("model_recall_at5_all")),
        "selected_method": row.get("selected_method"),
        "selected_mrr": _float(row.get("selected_test_mrr_covered")),
    }


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary") or {}
    pair = run.get("pair_diagnostics") or {}
    final_rerank = run.get("route_block_value_final_rerank") or {}
    product_audit = run.get("product_audit_final_rerank") or {}
    return {
        "name": run.get("name"),
        "n_targets": int(summary.get("n_targets") or 0),
        "cascade_solved_rate": _float(summary.get("cascade_solved_rate")),
        "stock_closed_rate": _float(summary.get("stock_closed_rate")),
        "top_result_exact_reaction_in_pool": _float(summary.get("top_result_exact_reaction_in_pool")),
        "top_result_gt_reactant_in_pool": _float(summary.get("top_result_gt_reactant_in_pool")),
        "result_exact_reaction_in_pool": _float(summary.get("result_exact_reaction_in_pool")),
        "result_gt_reactant_in_pool": _float(summary.get("result_gt_reactant_in_pool")),
        "pair_rewards_applied": int(pair.get("cascade_pair_reward_applied_true") or 0),
        "pair_applicable": int(pair.get("cascade_pair_applicable_true") or 0),
        "final_rerank_enabled_targets": int(final_rerank.get("enabled_targets") or 0),
        "final_rerank_changed": int(final_rerank.get("top_route_changed") or 0),
        "product_audit_final_rerank_enabled_targets": int(product_audit.get("enabled_targets") or 0),
        "product_audit_final_rerank_changed": int(product_audit.get("top_route_changed") or 0),
        "top_route_changed_vs_baseline": int(run.get("top_route_changed_vs_baseline") or 0),
    }


def _best_quality_run(candidates: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    if not candidates:
        return {}
    baseline_summary = baseline.get("summary") or {}

    def key(run: dict[str, Any]) -> tuple[float, float, int, int]:
        summary = run.get("summary") or {}
        exact_delta = _float(summary.get("top_result_exact_reaction_in_pool")) - _float(
            baseline_summary.get("top_result_exact_reaction_in_pool")
        )
        gt_delta = _float(summary.get("top_result_gt_reactant_in_pool")) - _float(
            baseline_summary.get("top_result_gt_reactant_in_pool")
        )
        final_rerank = run.get("route_block_value_final_rerank") or {}
        product_audit = run.get("product_audit_final_rerank") or {}
        pair = run.get("pair_diagnostics") or {}
        return (
            max(exact_delta, gt_delta),
            gt_delta,
            int(run.get("top_route_changed_vs_baseline") or 0)
            + int(final_rerank.get("top_route_changed") or 0)
            + int(product_audit.get("top_route_changed") or 0),
            int(pair.get("cascade_pair_reward_applied_true") or 0)
            + int(final_rerank.get("enabled_targets") or 0)
            + int(product_audit.get("enabled_targets") or 0),
        )

    return max(candidates, key=key, default={})


def _find_metric(rows: list[dict[str, Any]], *, method: Any, label: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == method and row.get("label") == label:
            return row
    return {}


def _best_row(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    return max(rows, key=lambda row: _float(row.get(key)), default={})


def _recall(report: dict[str, Any], k: int) -> float:
    return _float((report.get("recall_at_k_all") or {}).get(str(k)))


def _float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _gate_detail(payload: dict[str, Any]) -> str:
    if "observed_delta" in payload:
        return f"delta={_fmt(payload.get('observed_delta'))}, ci_low={_fmt(payload.get('ci95_low'))}"
    if "actual" in payload:
        detail = f"actual={_fmt(payload.get('actual'))}, required={_fmt(payload.get('required_min'))}"
        if payload.get("control_model"):
            detail += f", control={payload.get('control_model')}"
        return detail
    if "top_route_changed_vs_baseline" in payload:
        return (
            f"changed={payload.get('top_route_changed_vs_baseline')}, "
            f"pair_applied={payload.get('pair_rewards_applied')}, "
            f"route_block_final_changed={payload.get('final_rerank_changed')}, "
            f"product_audit_changed={payload.get('product_audit_final_rerank_changed')}"
        )
    return json.dumps(payload, ensure_ascii=False)


def _live_row(name: str, row: dict[str, Any]) -> str:
    return (
        f"| `{name}` | {_fmt(row.get('cascade_solved_rate'))} | {_fmt(row.get('stock_closed_rate'))} | "
        f"{_fmt(row.get('top_result_exact_reaction_in_pool'))} | {_fmt(row.get('top_result_gt_reactant_in_pool'))} | "
        f"{_fmt(row.get('result_exact_reaction_in_pool'))} | {_fmt(row.get('result_gt_reactant_in_pool'))} | "
        f"{row.get('pair_rewards_applied', 0)} | {row.get('final_rerank_changed', 0)} | "
        f"{row.get('product_audit_final_rerank_changed', 0)} | "
        f"{row.get('top_route_changed_vs_baseline', 0)} |"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize route/block model strengthening evidence")
    ap.add_argument("--route-pool-report", required=True)
    ap.add_argument("--route-block-value-report")
    ap.add_argument("--no-human-ablation-summary")
    ap.add_argument("--final-rerank-replay")
    ap.add_argument("--ablation-summary", required=True)
    ap.add_argument("--bootstrap-stability", required=True)
    ap.add_argument("--transition-hardneg-summary", required=True)
    ap.add_argument("--runtime-nohuman-probe")
    ap.add_argument("--guarded-search-comparison", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md")
    ap.add_argument("--min-mrr-delta-vs-retrieval", type=float, default=0.03)
    ap.add_argument("--min-live-quality-delta", type=float, default=0.01)
    args = ap.parse_args()
    report = summarize_route_block_strengthening(
        route_pool_report=Path(args.route_pool_report),
        route_block_value_report=Path(args.route_block_value_report) if args.route_block_value_report else None,
        no_human_ablation_summary=Path(args.no_human_ablation_summary) if args.no_human_ablation_summary else None,
        final_rerank_replay=Path(args.final_rerank_replay) if args.final_rerank_replay else None,
        ablation_summary=Path(args.ablation_summary),
        bootstrap_stability=Path(args.bootstrap_stability),
        transition_hardneg_summary=Path(args.transition_hardneg_summary),
        runtime_nohuman_probe=Path(args.runtime_nohuman_probe) if args.runtime_nohuman_probe else None,
        guarded_search_comparison=Path(args.guarded_search_comparison),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        min_mrr_delta_vs_retrieval=args.min_mrr_delta_vs_retrieval,
        min_live_quality_delta=args.min_live_quality_delta,
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

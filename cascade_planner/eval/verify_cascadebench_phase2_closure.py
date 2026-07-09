"""Verify CascadeBench Phase II closure artifacts.

This is a lightweight guardrail for the Phase II no-go decision.  It checks
that the generated decision summary, reports, and supervision candidate pack are
present and internally consistent before the work is treated as closed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "cascadebench_phase2_closure_verifier.v1"


REQUIRED_FILES = (
    "phase2_decision_summary.json",
    "phase2_decision_summary.md",
    "CCTS_V3_CANDIDATE_EVIDENCE_REPORT.zh.md",
    "PHASE2_REPRODUCIBILITY_MANIFEST.md",
    "route_pool_cascade_evidence_summary.json",
    "route_pool_cascade_evidence_summary.md",
    "route_pool_cascade_evidence_20/route_pool_cascade_evidence.json",
    "route_pool_cascade_evidence_20/route_pool_cascade_evidence.md",
    "route_pool_cascade_evidence_statin/route_pool_cascade_evidence.json",
    "route_pool_cascade_evidence_statin/route_pool_cascade_evidence.md",
    "route_pool_cascade_evidence_full100/route_pool_cascade_evidence.json",
    "route_pool_cascade_evidence_full100/route_pool_cascade_evidence.md",
    "route_pool_evidence_review_batch/route_pool_evidence_review_batch.jsonl",
    "route_pool_evidence_review_batch/route_pool_evidence_review_batch.csv",
    "route_pool_evidence_review_batch/route_pool_evidence_review_batch_report.json",
    "route_pool_evidence_review_batch/route_pool_evidence_review_batch_report.md",
    "route_pool_evidence_review_batch/route_pool_transform_sanity_audit.json",
    "route_pool_evidence_review_batch/route_pool_transform_sanity_audit.md",
    "route_pool_evidence_review_batch/route_pool_evidence_review_worklist.jsonl",
    "route_pool_evidence_review_batch/route_pool_evidence_review_worklist.csv",
    "route_pool_evidence_review_batch/route_pool_evidence_review_worklist_report.json",
    "route_pool_evidence_review_batch/route_pool_evidence_review_worklist_report.md",
    "route_pool_evidence_review_batch/route_pool_evidence_review_calibration_subset.jsonl",
    "route_pool_evidence_review_batch/route_pool_evidence_review_calibration_subset.csv",
    "route_pool_evidence_review_batch/route_pool_evidence_review_calibration_subset_report.json",
    "route_pool_evidence_review_batch/route_pool_evidence_review_calibration_subset_report.md",
    "route_pool_evidence_review_batch/human_route_pool_evidence_review_labels.jsonl",
    "route_pool_evidence_review_batch/human_route_pool_evidence_review_labels_report.json",
    "route_pool_evidence_review_batch/human_route_pool_evidence_review_labels_report.md",
    "route_pool_evidence_review_batch/human_route_pool_evidence_review_labels_invalid.jsonl",
    "route_pool_evidence_review_batch/human_route_pool_evidence_review_labels_unreviewed.jsonl",
    "route_pool_evidence_review_prompts/route_pool_evidence_review_prompts.jsonl",
    "route_pool_evidence_review_prompts/route_pool_evidence_review_prompts_report.json",
    "route_pool_evidence_review_prompts/route_pool_evidence_review_prompts_report.md",
    "route_pool_evidence_review_prompts/route_pool_evidence_review_calibration_prompts.jsonl",
    "route_pool_evidence_review_prompts/route_pool_evidence_review_calibration_prompts_report.json",
    "route_pool_evidence_review_prompts/route_pool_evidence_review_calibration_prompts_report.md",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_responses.jsonl",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_run_report.json",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_run_report.md",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_labels.jsonl",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_labels_report.json",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_labels_report.md",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_label_summary.json",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_label_summary.md",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_signal_calibration.json",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_signal_calibration.md",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_promotion_gate.json",
    "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_promotion_gate.md",
    "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_pipeline_manifest.json",
    "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_pipeline_manifest.md",
    "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_responses.jsonl",
    "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_labels.jsonl",
    "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_signal_calibration.json",
    "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_promotion_gate.json",
    "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_pipeline_manifest.json",
    "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_pipeline_manifest.md",
    "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_responses.jsonl",
    "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_labels.jsonl",
    "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_signal_calibration.json",
    "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_promotion_gate.json",
    "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_csv_pipeline_manifest.json",
    "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_csv_pipeline_manifest.md",
    "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_labels.jsonl",
    "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_labels_report.json",
    "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_labels_unreviewed.jsonl",
    "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_signal_calibration.json",
    "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_promotion_gate.json",
    "route_pool_evidence_review_calibration_packet/route_pool_evidence_review_calibration_subset_TO_FILL.csv",
    "route_pool_evidence_review_calibration_packet/route_pool_review_calibration_packet.json",
    "route_pool_evidence_review_calibration_packet/README.md",
    "template_outcome_supervision_pack/template_outcome_supervision.jsonl",
    "template_outcome_supervision_pack/template_outcome_supervision_report.json",
    "template_outcome_supervision_pack/template_outcome_supervision_report.md",
    "template_outcome_review_batch/template_outcome_review_batch.jsonl",
    "template_outcome_review_batch/template_outcome_review_batch.csv",
    "template_outcome_review_batch/template_outcome_review_batch_report.json",
    "template_outcome_review_batch/template_outcome_review_batch_report.md",
)


def verify_cascadebench_phase2_closure(*, root: Path, output_json: Path | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for rel in REQUIRED_FILES:
        path = root / rel
        checks.append(_check(bool(path.exists()), f"required file exists: {rel}", {"path": str(path)}))

    summary = _read_json(root / "phase2_decision_summary.json")
    supervision = _read_json(root / "template_outcome_supervision_pack" / "template_outcome_supervision_report.json")
    review = _read_json(root / "template_outcome_review_batch" / "template_outcome_review_batch_report.json")
    route_pool_evidence = _read_json(root / "route_pool_cascade_evidence_summary.json")
    route_pool_review = _read_json(root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_batch_report.json")
    route_pool_transform_sanity = _read_json(root / "route_pool_evidence_review_batch" / "route_pool_transform_sanity_audit.json")
    route_pool_worklist = _read_json(root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_worklist_report.json")
    route_pool_calibration_subset = _read_json(root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_calibration_subset_report.json")
    route_pool_human_csv_ingest = _read_json(root / "route_pool_evidence_review_batch" / "human_route_pool_evidence_review_labels_report.json")
    route_pool_prompts = _read_json(root / "route_pool_evidence_review_prompts" / "route_pool_evidence_review_prompts_report.json")
    route_pool_calibration_prompts = _read_json(root / "route_pool_evidence_review_prompts" / "route_pool_evidence_review_calibration_prompts_report.json")
    route_pool_dryrun = _read_json(root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_run_report.json")
    route_pool_dryrun_labels = _read_json(root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_labels_report.json")
    route_pool_dryrun_label_summary = _read_json(root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_label_summary.json")
    route_pool_dryrun_signal_calibration = _read_json(root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_signal_calibration.json")
    route_pool_dryrun_gate = _read_json(root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_promotion_gate.json")
    route_pool_pipeline = _read_json(root / "route_pool_evidence_review_pipeline_dryrun" / "dryrun_route_pool_evidence_review_pipeline_manifest.json")
    route_pool_calibration_pipeline = _read_json(root / "route_pool_evidence_review_calibration_pipeline_dryrun" / "calibration_route_pool_evidence_review_pipeline_manifest.json")
    route_pool_calibration_csv_pipeline = _read_json(root / "route_pool_evidence_review_calibration_csv_pipeline_blank" / "calibration_human_route_pool_evidence_review_csv_pipeline_manifest.json")
    route_pool_calibration_packet = _read_json(root / "route_pool_evidence_review_calibration_packet" / "route_pool_review_calibration_packet.json")

    decision = summary.get("decision") or {}
    gates = summary.get("decision_gates") or {}
    checks.extend(
        [
            _check(decision.get("search_integration_ready") is False, "decision keeps search integration disabled"),
            _check(
                str(decision.get("recommendation") or "").startswith("do_not_integrate_search"),
                "decision recommendation is no search integration",
            ),
            _check(float(gates.get("min_meaningful_delta") or 0.0) >= 0.01, "meaningful-delta threshold is present"),
            _check(gates.get("two_stage_selector_pair_beats_frequency") is False, "two-stage selector does not beat frequency on pair+analog"),
            _check(gates.get("template_twostage_pair_selector_improves_mrr") is False, "template pair selector gate is false"),
            _check(gates.get("rc_pair_selector_improves_mrr") is False, "rc pair selector gate is false"),
            _check(gates.get("oracle_pair_upper_bound_exists") is True, "oracle upper-bound gap is recorded"),
        ]
    )

    selector_reports = summary.get("selector_reports") or {}
    for name in (
        "transform_pair_train_vocab",
        "template_twostage_freq_top3_pair",
        "template_twostage_freq_top3_rc_pair",
    ):
        report = selector_reports.get(name) or {}
        checks.append(_check(report.get("status") == "ok", f"selector report present: {name}"))
        delta = (report.get("delta") or {}).get("mrr", (report.get("delta") or {}).get("mrr_all_groups"))
        if delta is not None:
            checks.append(_check(float(delta) <= 0.0, f"selector report is not promoted: {name}", {"delta_mrr": delta}))

    supervision_summary = supervision.get("summary") or {}
    classes = supervision_summary.get("classes") or {}
    checks.extend(
        [
            _check(int(supervision_summary.get("rows") or 0) > 0, "supervision pack has rows"),
            _check(int(supervision_summary.get("targets") or 0) > 0, "supervision pack has targets"),
            _check(int(classes.get("pair_and_analog_positive") or 0) > 0, "supervision pack includes pair+analog positives"),
            _check(int(classes.get("high_score_hard_negative") or 0) > 0, "supervision pack includes hard negatives"),
            _check(int(classes.get("pair_only_near_miss") or 0) > 0, "supervision pack includes pair-only near misses"),
        ]
    )

    review_summary = review.get("summary") or {}
    review_classes = review_summary.get("sampled_classes") or {}
    checks.extend(
        [
            _check(int(review_summary.get("sampled_rows") or 0) > 0, "review batch has sampled rows"),
            _check(int(review_classes.get("pair_and_analog_positive") or 0) > 0, "review batch includes pair+analog positives"),
            _check(int(review_classes.get("high_score_hard_negative") or 0) > 0, "review batch includes hard negatives"),
            _check(int(review_classes.get("analog_only_positive") or 0) > 0, "review batch includes analog-only positives"),
            _check(_review_batch_has_expert_fields(root / "template_outcome_review_batch" / "template_outcome_review_batch.jsonl"), "review batch rows include expert fields"),
        ]
    )

    evidence_interp = route_pool_evidence.get("interpretation") or {}
    evidence_audits = route_pool_evidence.get("audits") or {}
    checks.extend(
        [
            _check(int(evidence_interp.get("audits_present") or 0) >= 3, "route-pool evidence summary includes three pools"),
            _check(
                bool(evidence_interp.get("strict_same_pair_analog_is_sparse")) is True,
                "route-pool evidence records sparse strict same-pair analog support",
            ),
            _check(
                "diagnostic" in str(evidence_interp.get("recommended_use") or ""),
                "route-pool evidence is diagnostic, not promoted training signal",
            ),
        ]
    )
    pipeline_summaries = route_pool_pipeline.get("summaries") or {}
    checks.extend(
        [
            _check(int(((pipeline_summaries.get("run") or {}).get("written_rows")) or 0) > 0, "route-pool review pipeline dry-run wrote rows"),
            _check(int(((pipeline_summaries.get("labels") or {}).get("accepted_rows")) or 0) > 0, "route-pool review pipeline dry-run labels ingest"),
            _check(
                ((pipeline_summaries.get("promotion_gate") or {}).get("ready_for_training")) is False,
                "route-pool review pipeline dry-run promotion gate blocks training",
            ),
            _check(
                ((pipeline_summaries.get("signal_calibration") or {}).get("ready_for_proxy_training")) is False,
                "route-pool review pipeline dry-run signal calibration blocks proxy training",
            ),
        ]
    )
    checks.extend(
        [
            _check(route_pool_dryrun_gate.get("ready_for_training") is False, "dry-run review promotion gate rejects training"),
            _check(
                "do not train" in str(route_pool_dryrun_gate.get("recommendation") or ""),
                "dry-run review promotion gate recommendation blocks training",
            ),
        ]
    )
    for name in ("v4_test20", "statin", "full100"):
        row = evidence_audits.get(name) or {}
        checks.append(_check(int(row.get("routes") or 0) > 0, f"route-pool evidence audit has routes: {name}"))
        checks.append(
            _check(
                float(row.get("same_pair_analog_route_rate") or 0.0) <= float(row.get("any_analog_route_rate") or 0.0),
                f"strict same-pair analog is not broader than any-analog: {name}",
            )
        )

    pool_review_summary = route_pool_review.get("summary") or {}
    pool_review_classes = pool_review_summary.get("sampled_classes") or {}
    checks.extend(
        [
            _check(int(pool_review_summary.get("sampled_rows") or 0) > 0, "route-pool evidence review batch has sampled rows"),
            _check(int(pool_review_classes.get("any_analog_supported") or 0) > 0, "route-pool review includes any-analog examples"),
            _check(
                int(pool_review_classes.get("multistep_without_observed_pair") or 0) > 0,
                "route-pool review includes no-observed-pair examples",
            ),
            _check(_route_pool_review_has_expert_fields(root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_batch.jsonl"), "route-pool review rows include expert fields"),
            _check(_route_pool_review_class_blocks_are_consistent(root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_batch.jsonl"), "route-pool review class-specific blocks are consistent"),
        ]
    )
    transform_summary = route_pool_transform_sanity.get("summary") or {}
    transform_interp = route_pool_transform_sanity.get("interpretation") or {}
    worklist_summary = route_pool_worklist.get("summary") or {}
    calibration_subset_summary = route_pool_calibration_subset.get("summary") or {}
    human_csv_summary = route_pool_human_csv_ingest.get("summary") or {}
    checks.extend(
        [
            _check(int(transform_summary.get("rows") or 0) == int(pool_review_summary.get("sampled_rows") or -1), "transform sanity audit covers the route-pool review batch"),
            _check(
                int(transform_summary.get("steps") or 0) >= 2 * int(transform_summary.get("rows") or 0),
                "transform sanity audit inspects upstream and downstream steps",
            ),
            _check(
                "review_triage_only" in str(transform_interp.get("recommended_use") or ""),
                "transform sanity audit blocks direct supervised use of noisy transform labels",
            ),
            _check(
                transform_interp.get("transform_labels_safe_for_training_without_review") is False,
                "transform sanity audit does not mark transform labels training-safe without review",
            ),
            _check(
                int(worklist_summary.get("rows") or 0) == int(pool_review_summary.get("sampled_rows") or -1),
                "route-pool worklist covers the route-pool review batch",
            ),
            _check(
                int(worklist_summary.get("rows_with_transform_label_warning") or 0)
                == int(transform_summary.get("rows_with_label_mismatch") or -2),
                "route-pool worklist carries transform label warnings",
            ),
            _check(
                _route_pool_worklist_is_flat(root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_worklist.jsonl"),
                "route-pool worklist is flat and reviewer friendly",
            ),
            _check(
                int(calibration_subset_summary.get("selected_rows") or 0) >= 30,
                "route-pool calibration subset has enough review rows",
            ),
            _check(
                len(calibration_subset_summary.get("pools") or {}) >= 3,
                "route-pool calibration subset covers three source pools",
            ),
            _check(
                int((calibration_subset_summary.get("transform_label_warning") or {}).get("True") or 0) > 0
                and int((calibration_subset_summary.get("transform_label_warning") or {}).get("False") or 0) > 0,
                "route-pool calibration subset covers warning and non-warning cases",
            ),
            _check(
                int((calibration_subset_summary.get("classes") or {}).get("same_pair_analog_supported") or 0) >= 1,
                "route-pool calibration subset preserves rare same-pair example",
            ),
            _check(
                int(human_csv_summary.get("csv_rows") or 0) == int(worklist_summary.get("rows") or -1),
                "human CSV ingest covers the route-pool worklist",
            ),
            _check(
                int(human_csv_summary.get("accepted_rows") or 0) == 0
                and int(human_csv_summary.get("unreviewed_rows") or 0) == int(worklist_summary.get("rows") or -1),
                "blank human CSV ingest does not create labels",
            ),
            _check(
                int(human_csv_summary.get("invalid_rows", -1)) == 0,
                "blank human CSV ingest has no invalid rows",
            ),
        ]
    )

    prompt_summary = route_pool_prompts.get("summary") or {}
    calibration_prompt_summary = route_pool_calibration_prompts.get("summary") or {}
    prompt_schema = route_pool_prompts.get("expected_output_schema") or {}
    checks.extend(
        [
            _check(int(prompt_summary.get("prompt_rows") or 0) > 0, "route-pool review prompts have rows"),
            _check(int(prompt_summary.get("prompt_rows") or 0) == int(pool_review_summary.get("sampled_rows") or -1), "prompt rows match review batch rows"),
            _check("route_plausible" in prompt_schema and "cascade_coherent" in prompt_schema, "route-pool prompt output schema includes core fields"),
            _check(_route_pool_prompts_are_structured(root / "route_pool_evidence_review_prompts" / "route_pool_evidence_review_prompts.jsonl"), "route-pool prompt rows include JSON-only instruction and route block chemistry"),
            _check(
                _route_pool_prompts_include_transform_sanity(root / "route_pool_evidence_review_prompts" / "route_pool_evidence_review_prompts.jsonl"),
                "route-pool prompt rows include transform sanity triage",
            ),
            _check(
                int(calibration_prompt_summary.get("selected_prompts") or 0) == int(calibration_subset_summary.get("selected_rows") or -1),
                "route-pool calibration prompt subset matches calibration subset",
            ),
            _check(
                int(calibration_prompt_summary.get("missing_ids", -1)) == 0
                and int(calibration_prompt_summary.get("duplicate_subset_ids", -1)) == 0,
                "route-pool calibration prompt subset has no missing or duplicate ids",
            ),
            _check(
                int(calibration_prompt_summary.get("rows_with_transform_label_warning") or 0)
                == int((calibration_subset_summary.get("transform_label_warning") or {}).get("True") or -1),
                "route-pool calibration prompts preserve transform label warnings",
            ),
        ]
    )

    dryrun_summary = route_pool_dryrun.get("summary") or {}
    dryrun_label_summary = route_pool_dryrun_labels.get("summary") or {}
    checks.extend(
        [
            _check(bool(dryrun_summary.get("dry_run")) is True, "route-pool LLM review dry-run was dry-run"),
            _check(int(dryrun_summary.get("written_rows") or 0) > 0, "route-pool LLM review dry-run wrote responses"),
            _check(int(dryrun_label_summary.get("accepted_rows") or 0) > 0, "route-pool LLM review dry-run responses ingest"),
            _check(int(dryrun_label_summary.get("invalid_rows") or 0) == 0, "route-pool LLM review dry-run has no invalid rows"),
            _check(
                _route_pool_labels_include_transform_sanity(root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_labels.jsonl"),
                "route-pool ingested labels retain transform sanity triage",
            ),
        ]
    )
    label_summary_decision = route_pool_dryrun_label_summary.get("decision") or {}
    signal_calibration_decision = route_pool_dryrun_signal_calibration.get("decision") or {}
    calibration_pipeline_summaries = route_pool_calibration_pipeline.get("summaries") or {}
    calibration_csv_pipeline_summaries = route_pool_calibration_csv_pipeline.get("summaries") or {}
    calibration_packet_contract = route_pool_calibration_packet.get("contract") or {}
    checks.extend(
        [
            _check(
                label_summary_decision.get("reviewed_enough_for_proxy_calibration") is False,
                "dry-run review labels are not treated as enough for proxy calibration",
            ),
            _check(
                "insufficient" in str(label_summary_decision.get("recommendation") or ""),
                "dry-run review label summary recommends real review before training",
            ),
            _check(
                signal_calibration_decision.get("ready_for_proxy_training") is False,
                "dry-run signal calibration rejects proxy training",
            ),
            _check(
                "do not train" in str(signal_calibration_decision.get("recommendation") or ""),
                "dry-run signal calibration recommendation blocks training",
            ),
            _check(
                int(((calibration_pipeline_summaries.get("run") or {}).get("prompt_rows_total")) or 0)
                == int(calibration_subset_summary.get("selected_rows") or -1),
                "calibration dry-run pipeline runs against calibration prompts",
            ),
            _check(
                ((calibration_pipeline_summaries.get("signal_calibration") or {}).get("ready_for_proxy_training")) is False,
                "calibration dry-run pipeline signal calibration blocks proxy training",
            ),
            _check(
                ((calibration_pipeline_summaries.get("promotion_gate") or {}).get("ready_for_training")) is False,
                "calibration dry-run pipeline promotion gate blocks training",
            ),
            _check(
                int(((calibration_csv_pipeline_summaries.get("labels") or {}).get("csv_rows")) or 0)
                == int(calibration_subset_summary.get("selected_rows") or -1),
                "calibration CSV pipeline covers calibration subset",
            ),
            _check(
                int((calibration_csv_pipeline_summaries.get("labels") or {}).get("accepted_rows", -1)) == 0
                and int(((calibration_csv_pipeline_summaries.get("labels") or {}).get("unreviewed_rows")) or 0)
                == int(calibration_subset_summary.get("selected_rows") or -1),
                "blank calibration CSV pipeline does not create labels",
            ),
            _check(
                ((calibration_csv_pipeline_summaries.get("signal_calibration") or {}).get("ready_for_proxy_training")) is False,
                "blank calibration CSV pipeline signal calibration blocks proxy training",
            ),
            _check(
                ((calibration_csv_pipeline_summaries.get("promotion_gate") or {}).get("ready_for_training")) is False,
                "blank calibration CSV pipeline promotion gate blocks training",
            ),
            _check(
                calibration_packet_contract.get("does_not_create_labels") is True,
                "calibration reviewer packet does not create labels",
            ),
            _check(
                _csv_header_has(root / "route_pool_evidence_review_calibration_packet" / "route_pool_evidence_review_calibration_subset_TO_FILL.csv", "expert_risk_tags"),
                "calibration reviewer packet CSV includes expert_risk_tags",
            ),
            _check(
                _file_contains(root / "route_pool_evidence_review_calibration_packet" / "README.md", "run_route_pool_evidence_review_csv_pipeline"),
                "calibration reviewer packet README includes validation command",
            ),
        ]
    )

    ok = all(row["ok"] for row in checks)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "root": str(root),
            "output_json": str(output_json) if output_json else None,
        },
        "ok": ok,
        "summary": {
            "checks": len(checks),
            "passed": sum(1 for row in checks if row["ok"]),
            "failed": sum(1 for row in checks if not row["ok"]),
        },
        "checks": checks,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        output_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _check(ok: bool, name: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": bool(ok), "name": name, "details": details or {}}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _review_batch_has_expert_fields(path: Path) -> bool:
    if not path.exists():
        return False
    required = {
        "expert_template_applicable",
        "expert_outcome_plausible",
        "expert_cascade_coherent",
        "expert_priority",
        "expert_reject_reason",
        "expert_comments",
    }
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            return isinstance(row, dict) and required.issubset(row)
    return False


def _route_pool_review_has_expert_fields(path: Path) -> bool:
    if not path.exists():
        return False
    required = {
        "expert_route_plausible",
        "expert_block_transform_correct",
        "expert_support_precedent_relevant",
        "expert_cascade_coherent",
        "expert_priority",
        "expert_reject_reason",
        "expert_comments",
    }
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            return isinstance(row, dict) and required.issubset(row)
    return False


def _route_pool_review_class_blocks_are_consistent(path: Path) -> bool:
    if not path.exists():
        return False
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            seen += 1
            row = json.loads(line)
            block = row.get("review_block") or {}
            cls = row.get("evidence_class")
            if cls == "any_analog_supported" and not block.get("any_analog_supported"):
                return False
            if cls == "same_pair_analog_supported" and not block.get("same_pair_analog_supported"):
                return False
            if cls == "multistep_without_observed_pair" and block.get("pair_observed_in_evidence"):
                return False
    return seen > 0


def _route_pool_prompts_are_structured(path: Path) -> bool:
    if not path.exists():
        return False
    required_block = {
        "upstream_rxn",
        "downstream_rxn",
        "upstream_product",
        "downstream_product",
    }
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            seen += 1
            row = json.loads(line)
            prompt = str(row.get("prompt") or "")
            block = row.get("route_block") or {}
            if "Return JSON only" not in prompt:
                return False
            if not required_block.issubset(block):
                return False
            if "expected_output_schema" not in row:
                return False
    return seen > 0


def _route_pool_prompts_include_transform_sanity(path: Path) -> bool:
    if not path.exists():
        return False
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            seen += 1
            row = json.loads(line)
            sanity = row.get("transform_sanity")
            if not isinstance(sanity, dict) or "block_has_label_mismatch" not in sanity:
                return False
            if "transform_sanity" not in str(row.get("prompt") or ""):
                return False
    return seen > 0


def _route_pool_labels_include_transform_sanity(path: Path) -> bool:
    if not path.exists():
        return False
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            seen += 1
            row = json.loads(line)
            sanity = row.get("transform_sanity")
            if not isinstance(sanity, dict) or "block_has_label_mismatch" not in sanity:
                return False
    return seen > 0


def _route_pool_worklist_is_flat(path: Path) -> bool:
    if not path.exists():
        return False
    required = {
        "review_id",
        "upstream_rxn",
        "downstream_rxn",
        "transform_label_warning",
        "transform_label_warning_reasons",
        "expert_route_plausible",
        "expert_block_transform_correct",
    }
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            seen += 1
            row = json.loads(line)
            if not required.issubset(row):
                return False
            if isinstance(row.get("review_block"), dict):
                return False
    return seen > 0


def _csv_header_has(path: Path, column: str) -> bool:
    if not path.exists():
        return False
    first = path.read_text(encoding="utf-8").splitlines()
    if not first:
        return False
    return column in [part.strip() for part in first[0].split(",")]


def _file_contains(path: Path, pattern: str) -> bool:
    if not path.exists():
        return False
    return pattern in path.read_text(encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CascadeBench Phase II Closure Verification",
        "",
        f"- ok: `{report.get('ok')}`",
        f"- checks: `{(report.get('summary') or {}).get('checks')}`",
        f"- passed: `{(report.get('summary') or {}).get('passed')}`",
        f"- failed: `{(report.get('summary') or {}).get('failed')}`",
        "",
        "## Checks",
        "",
        "| Check | OK | Details |",
        "|---|---:|---|",
    ]
    for row in report.get("checks") or []:
        lines.append(f"| `{row.get('name')}` | `{row.get('ok')}` | `{json.dumps(row.get('details') or {}, ensure_ascii=False)}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CascadeBench Phase II closure artifacts")
    parser.add_argument("--root", default="results/shared/cascadebench_strict_20260516")
    parser.add_argument("--output-json", default="results/shared/cascadebench_strict_20260516/phase2_closure_verification.json")
    args = parser.parse_args()
    report = verify_cascadebench_phase2_closure(root=Path(args.root), output_json=Path(args.output_json))
    print(json.dumps({"ok": report["ok"], "summary": report["summary"], "output_json": args.output_json}, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

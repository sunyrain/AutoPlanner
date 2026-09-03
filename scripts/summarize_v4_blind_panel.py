#!/usr/bin/env python3
"""Summarize a V4 blind panel without conflating route and proof metrics."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    root = Path(args.panel_root).expanduser().resolve()
    status_path = root / "panel-status.json"
    status = _hydrate_report_diagnostics(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    summary = summarize_panel(status)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "panel-summary.json"
    )
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def summarize_panel(status: Mapping[str, Any]) -> dict[str, Any]:
    target_rows = {
        str(name): dict(value)
        for name, value in dict(status.get("targets") or {}).items()
        if isinstance(value, Mapping)
    }
    completed = {
        name: row for name, row in target_rows.items() if row.get("status") == "completed"
    }
    observed = {
        name: row
        for name, row in target_rows.items()
        if row.get("status") == "completed"
        or dict(row.get("fixed_cutoff_projection") or {}).get("available") is True
    }
    status_counts = Counter(
        str(row.get("status") or "unknown") for row in target_rows.values()
    )
    total = int(status.get("target_count") or len(target_rows))
    metric_fields = {
        "structural_route_present": "target_rooted_distinct_skeletons",
        "materialized_route_present": "materialized_skeletons",
        "host_reaction_validated": "reaction_validated_skeletons",
        "official_benchmark_stock_closed": "stock_closed_skeletons",
        "exact_source_grade": "evidence_closed_skeletons",
    }
    # Fixed-cutoff scientific outcomes belong to the frozen target row, not
    # to the kernel lifecycle status. A bounded run may finish with
    # ``status=paused`` because proof/condition work remains resumable while
    # already containing a target-rooted stock-closed route. Suppressing that
    # route from the full-panel metric would contradict the same row's
    # paper-equivalent projection.
    metric_rows = tuple(observed.values())
    completed_metric_rows = tuple(completed.values())
    metric_counts = {
        metric: sum(
            int(dict(row.get("route_counts") or {}).get(field) or 0) > 0
            for row in metric_rows
        )
        for metric, field in metric_fields.items()
    }
    completed_metric_counts = {
        metric: sum(
            int(dict(row.get("route_counts") or {}).get(field) or 0) > 0
            for row in completed_metric_rows
        )
        for metric, field in metric_fields.items()
    }
    metric_counts["configured_proof_policy_accepted"] = sum(
        row.get("accepted_under_configured_policy") is True
        for row in metric_rows
    )
    completed_metric_counts["configured_proof_policy_accepted"] = sum(
        row.get("accepted_under_configured_policy") is True
        for row in completed_metric_rows
    )
    metric_counts["paper_equivalent_solved"] = sum(
        dict(row.get("paper_equivalent") or {}).get("paper_equivalent_solved")
        is True
        for row in metric_rows
    )
    completed_metric_counts["paper_equivalent_solved"] = sum(
        dict(row.get("paper_equivalent") or {}).get("paper_equivalent_solved")
        is True
        for row in completed_metric_rows
    )
    metric_counts["paper_stock_comparable"] = sum(
        dict(row.get("paper_equivalent") or {}).get("stock_comparable_to_synthex")
        is True
        for row in metric_rows
    )
    completed_metric_counts["paper_stock_comparable"] = sum(
        dict(row.get("paper_equivalent") or {}).get("stock_comparable_to_synthex")
        is True
        for row in completed_metric_rows
    )
    metric_counts["within_resource_budget"] = sum(
        row.get("within_resource_budget") is True for row in metric_rows
    )
    completed_metric_counts["within_resource_budget"] = sum(
        row.get("within_resource_budget") is True
        for row in completed_metric_rows
    )
    denominator = total if total > 0 else 1
    completed_denominator = len(completed) if completed else 1
    rates = {
        metric: {
            "count": count,
            "rate_over_full_panel": round(count / denominator, 6),
            "rate_over_completed": (
                round(completed_metric_counts[metric] / len(completed), 6)
                if completed
                else 0.0
            ),
        }
        for metric, count in metric_counts.items()
    }
    costs = Counter()
    elapsed: list[float] = []
    milestone_counts = Counter()
    completed_milestone_counts = Counter()
    time_to_first: dict[str, list[float]] = {
        "first_route": [],
        "first_host_valid_route": [],
        "B4": [],
    }
    action_kinds = Counter()
    action_statuses = Counter()
    chemenzy = Counter()
    result_route_totals = Counter()
    b4_outcomes = Counter()
    independent_axis_outcomes = Counter()
    reaction_rejection_reasons = Counter()
    rejection_taxonomy_counts = Counter()
    rejection_taxonomy_reasons: dict[str, Counter[str]] = {}
    stock_diagnostics = Counter()
    provider_lineage_dispositions = Counter()
    provider_search = Counter()
    provider_attempt_dispositions = Counter()
    provider_route_provenance_first_loss_counts = Counter()
    selected_provider_route_b4_dispositions = Counter()
    provider_route_b4_funnel = Counter()
    provider_non_success_attempt_records: list[dict[str, Any]] = []
    selected_provider_route_b4_open_records: list[dict[str, Any]] = []
    provider_search_stop_reasons = Counter()
    provider_search_time_s: list[float] = []
    provider_first_success_time_s: list[float] = []
    provider_post_first_success_time_s: list[float] = []
    start_cohort = Counter()
    start_cohort_elapsed_s: list[float] = []
    start_chemenzy_completed_s: list[float] = []
    start_codex_completed_s: list[float] = []
    start_chemenzy_peer_wait_s: list[float] = []
    provider_integration_loss_targets: list[str] = []
    provider_topology_loss_targets: list[str] = []
    provider_partial_lineage_targets: list[str] = []
    atom_balance_soft_gate_counterfactual_targets: list[str] = []
    atom_balance_frontier_signal_targets: list[str] = []
    guided_frontier_solved_root_open_targets: list[str] = []
    pre_b4_credibility_action_targets: list[str] = []
    resource_observed = Counter()
    runtime_recovery = Counter()
    credibility_action_kinds = {
        "acquire_exact_evidence",
        "bind_exact_evidence",
        "condition_enrich",
        "program_admit",
        "program_discover",
        "program_review",
        "experiment_feedback",
        "resolve_conflict",
    }
    for row in completed.values():
        completed_gates = _gate_summary(row)
        for gate in ("B1", "B2", "B4", "B5"):
            completed_milestone_counts[gate] += completed_gates.get(gate) is True
    for target_name, row in observed.items():
        case_id = str(row.get("case_id") or target_name)
        for key, value in dict(row.get("model_cost") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                costs[key] += value
        value = row.get("elapsed_s")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            elapsed.append(float(value))
        gates = _gate_summary(row)
        b4_outcomes[_b4_outcome(row, gates=gates)] += 1
        independent_axis_outcomes[_independent_axis_outcome(row, gates=gates)] += 1
        reaction_rejection_reasons.update(
            {
                str(key): int(value or 0)
                for key, value in dict(
                    row.get("reaction_rejection_reason_counts") or {}
                ).items()
            }
        )
        rejection_taxonomy = dict(row.get("rejection_taxonomy") or {})
        rejection_taxonomy_counts.update(
            {
                str(key): int(value or 0)
                for key, value in dict(
                    rejection_taxonomy.get("counts") or {}
                ).items()
            }
        )
        for category, raw_counts in dict(
            rejection_taxonomy.get("reason_counts") or {}
        ).items():
            if not isinstance(raw_counts, Mapping):
                continue
            category_counter = rejection_taxonomy_reasons.setdefault(
                str(category), Counter()
            )
            category_counter.update(
                {
                    str(reason): int(count or 0)
                    for reason, count in dict(raw_counts).items()
                }
            )
        stock_diagnostics.update(
            {
                str(key): int(value or 0)
                for key, value in dict(row.get("stock_audit_diagnostics") or {}).items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
        )
        lineage_dispositions = {
            str(key): int(value or 0)
            for key, value in dict(
                row.get("provider_lineage_disposition_counts") or {}
            ).items()
        }
        provider_lineage_dispositions.update(lineage_dispositions)
        provider_route_provenance_first_loss_counts.update(
            {
                str(key): int(value or 0)
                for key, value in dict(
                    row.get("candidate_provenance_first_loss_counts") or {}
                ).items()
                if str(key)
            }
        )
        selected_provider_route_b4_dispositions.update(
            {
                str(key): int(value or 0)
                for key, value in dict(
                    row.get("selected_provider_route_b4_disposition_counts") or {}
                ).items()
                if str(key)
            }
        )
        for route in row.get("selected_provider_route_b4_records") or []:
            if not isinstance(route, Mapping):
                continue
            disposition = str(route.get("disposition") or "")
            if disposition == "stock_closed":
                continue
            selected_provider_route_b4_open_records.append(
                {
                        "case_id": case_id,
                        "target_name": target_name,
                        "report_path": str(row.get("report_path") or ""),
                        "root_b4_open": gates.get("B4") is not True,
                        **dict(route),
                }
            )
        for key in (
            "selected_provider_route_count",
            "canonical_bound_selected_provider_route_count",
            "canonical_materialized_selected_provider_route_count",
            "stock_closed_selected_provider_route_count",
        ):
            provider_route_b4_funnel[key] += int(row.get(key) or 0)
        resource_observed.update(
            {
                str(key): value
                for key, value in dict(row.get("resource_observed") or {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        runtime_recovery.update(
            {
                str(key): int(value or 0)
                for key, value in dict(row.get("runtime_recovery") or {}).items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
        )
        for attempt in row.get("provider_search_attempts") or []:
            if not isinstance(attempt, Mapping):
                continue
            kind = str(attempt.get("kind") or "unknown")
            provider_search["attempt_count"] += 1
            provider_search[f"{kind}_attempt_count"] += 1
            attempt_disposition = _provider_attempt_disposition(attempt)
            provider_attempt_dispositions[attempt_disposition] += 1
            if attempt_disposition != "host_admitted_success":
                provider_non_success_attempt_records.append(
                    {
                        "case_id": case_id,
                        "target_name": target_name,
                        "report_path": str(row.get("report_path") or ""),
                        "root_b4_open": gates.get("B4") is not True,
                        "artifact": str(attempt.get("artifact") or ""),
                        "kind": kind,
                        "disposition": attempt_disposition,
                        "search_status": str(
                            attempt.get("search_status") or ""
                        ),
                        "stop_reason": str(attempt.get("stop_reason") or ""),
                        "backend_failure_categories": list(
                            attempt.get("backend_failure_categories") or []
                        ),
                        "raw_route_count": int(
                            attempt.get("native_raw_route_count") or 0
                        ),
                        "normalized_route_count": int(
                            attempt.get("normalized_route_count") or 0
                        ),
                        "rule_gate_kept_route_count": int(
                            attempt.get("rule_gate_kept_route_count") or 0
                        ),
                        "output_route_count": int(
                            attempt.get("output_route_count") or 0
                        ),
                    }
                )
            provider_route_b4_funnel["attempt_count"] += 1
            if attempt.get("raw_solved") is True:
                provider_search["raw_solved_attempt_count"] += 1
                provider_search[f"{kind}_raw_solved_attempt_count"] += 1
                provider_route_b4_funnel["raw_nonempty_attempt_count"] += 1
            if attempt.get("host_admitted_solved") is True:
                provider_search["host_admitted_solved_attempt_count"] += 1
                provider_search[f"{kind}_host_admitted_solved_attempt_count"] += 1
                provider_route_b4_funnel[
                    "host_admitted_nonempty_attempt_count"
                ] += 1
                first_output_index = attempt.get("first_output_raw_route_index")
                if isinstance(first_output_index, int) and not isinstance(
                    first_output_index, bool
                ):
                    for prefix in (4, 8, 16):
                        if first_output_index < prefix:
                            provider_search[
                                f"host_solution_within_raw_prefix_{prefix}_attempt_count"
                            ] += 1
            for key in (
                "native_raw_route_count",
                "native_found_route_count",
                "output_route_count",
                "rule_gate_input_route_count",
                "rule_gate_kept_route_count",
                "rule_gate_dropped_route_count",
                "executed_iterations",
            ):
                value = attempt.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    provider_search[key] += value
            provider_route_b4_funnel["provider_raw_route_count"] += int(
                attempt.get("native_raw_route_count") or 0
            )
            provider_route_b4_funnel["provider_normalized_route_count"] += int(
                attempt.get("normalized_route_count") or 0
            )
            provider_route_b4_funnel["host_search_admitted_route_count"] += int(
                attempt.get("host_search_admitted_route_count") or 0
            )
            provider_route_b4_funnel[
                "provider_rule_gate_kept_route_count"
            ] += int(
                attempt.get("rule_gate_kept_route_count") or 0
            )
            provider_route_b4_funnel["provider_output_route_count"] += int(
                attempt.get("output_route_count") or 0
            )
            for key in (
                "quarantined_route_count",
                "atom_balance_only_quarantine_count",
                "atom_balance_only_stock_closed_quarantine_count",
                "hard_structure_quarantine_count",
            ):
                value = attempt.get(key)
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                ):
                    provider_search[key] += value
            stop_reason = str(attempt.get("stop_reason") or "").strip()
            if stop_reason:
                provider_search_stop_reasons[stop_reason] += 1
            attempt_time = attempt.get("provider_time_s")
            if isinstance(attempt_time, (int, float)) and math.isfinite(
                float(attempt_time)
            ):
                provider_search_time_s.append(float(attempt_time))
            first_success = attempt.get("first_success_time_s")
            if isinstance(first_success, (int, float)) and math.isfinite(
                float(first_success)
            ):
                provider_first_success_time_s.append(float(first_success))
                if (
                    isinstance(attempt_time, (int, float))
                    and math.isfinite(float(attempt_time))
                    and float(attempt_time) >= float(first_success)
                ):
                    provider_post_first_success_time_s.append(
                        float(attempt_time) - float(first_success)
                    )
        latency = dict(row.get("start_cohort_latency_audit") or {})
        if latency.get("applicable") is True:
            start_cohort["applicable_target_count"] += 1
            first_proposal = dict(latency.get("chemenzy_first_proposal") or {})
            if first_proposal.get("nonempty_raw_proposal_observed") is True:
                start_cohort["nonempty_chemenzy_proposal_target_count"] += 1
            if first_proposal.get("codex_peer_in_flight_at_chemenzy_completion") is True:
                start_cohort["chemenzy_completed_before_codex_target_count"] += 1
            _append_finite(start_cohort_elapsed_s, latency.get("cohort_elapsed_s"))
            _append_finite(
                start_chemenzy_completed_s,
                first_proposal.get("elapsed_from_start_cohort_s"),
            )
            _append_finite(
                start_chemenzy_peer_wait_s,
                first_proposal.get("peer_wait_excluded_s"),
            )
            for action in latency.get("actions") or []:
                if (
                    isinstance(action, Mapping)
                    and action.get("action_kind") == "codex_global_architecture"
                ):
                    _append_finite(
                        start_codex_completed_s,
                        action.get("completed_offset_s"),
                    )
                    break
        if (
            gates.get("B4") is not True
            and any(
                int(attempt.get("atom_balance_only_stock_closed_quarantine_count") or 0)
                > 0
                and attempt.get("search_target_is_root") is True
                for attempt in row.get("provider_search_attempts") or []
                if isinstance(attempt, Mapping)
            )
        ):
            atom_balance_soft_gate_counterfactual_targets.append(
                str(row.get("case_id") or "unknown")
            )
        if (
            gates.get("B4") is not True
            and any(
                attempt.get("kind") == "guided"
                and attempt.get("host_admitted_solved") is True
                and attempt.get("search_target_is_root") is not True
                for attempt in row.get("provider_search_attempts") or []
                if isinstance(attempt, Mapping)
            )
        ):
            guided_frontier_solved_root_open_targets.append(
                str(row.get("case_id") or "unknown")
            )
        if (
            gates.get("B4") is not True
            and any(
                int(attempt.get("atom_balance_only_stock_closed_quarantine_count") or 0)
                > 0
                and attempt.get("search_target_is_root") is not True
                for attempt in row.get("provider_search_attempts") or []
                if isinstance(attempt, Mapping)
            )
        ):
            atom_balance_frontier_signal_targets.append(
                str(row.get("case_id") or "unknown")
            )
        has_outside_route_edges = lineage_dispositions.get(
            "canonical_edges_present_outside_complete_measured_route", 0
        ) > 0
        if has_outside_route_edges:
            case_id = str(row.get("case_id") or "unknown")
            provider_partial_lineage_targets.append(case_id)
            if (
                gates.get("B4") is not True
                and int(row.get("provider_eligible_incomplete_lineage_count") or 0) > 0
            ):
                provider_integration_loss_targets.append(case_id)
        topology_loss_count = int(
            row.get("provider_topology_conservation_failure_count") or 0
        )
        if topology_loss_count > 0:
            case_id = str(row.get("case_id") or "unknown")
            provider_topology_loss_targets.append(case_id)
            provider_integration_loss_targets.append(case_id)
        for gate in ("B1", "B2", "B4", "B5"):
            milestone_counts[gate] += gates.get(gate) is True
        projection = dict(row.get("fixed_cutoff_projection") or {})
        first = dict(projection.get("time_to_first") or {})
        for milestone in time_to_first:
            observed_value = dict(first.get(milestone) or {}).get(
                "elapsed_wall_time_s"
            )
            if isinstance(observed_value, (int, float)) and math.isfinite(
                float(observed_value)
            ):
                time_to_first[milestone].append(float(observed_value))
        actions = dict(projection.get("action_counts") or {})
        kinds = Counter(
            {
                str(key): int(value or 0)
                for key, value in dict(actions.get("by_kind") or {}).items()
            }
        )
        action_kinds.update(kinds)
        action_statuses.update(
            {
                str(key): int(value or 0)
                for key, value in dict(actions.get("by_status") or {}).items()
            }
        )
        if gates.get("B4") is not True and any(
            kinds.get(kind, 0) > 0 for kind in credibility_action_kinds
        ):
            pre_b4_credibility_action_targets.append(
                str(row.get("case_id") or "unknown")
            )
        for key, value in dict(row.get("anytime_route_counts") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                result_route_totals[str(key)] += value
        for key, value in dict(row.get("chemenzy") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                chemenzy[str(key)] += value
    failures = Counter(
        _failure_key(row)
        for row in target_rows.values()
        if row.get("status") not in {"completed", "queued", "running"}
    )
    terminal_dispositions = Counter(
        _terminal_disposition(row) for row in target_rows.values()
    )
    missing_target_rows = max(0, total - len(target_rows))
    if missing_target_rows:
        terminal_dispositions["missing_target_row"] += missing_target_rows
    accounted_target_count = sum(terminal_dispositions.values())
    per_target = [
        {
            "target_name": name,
            "case_id": str(row.get("case_id") or ""),
            "status": str(row.get("status") or ""),
            "scientific_status": str(row.get("scientific_status") or ""),
            "scientific_disposition": str(
                row.get("scientific_disposition") or ""
            ),
            "terminal_disposition": _terminal_disposition(row),
            "retrostar_solved": (
                int(
                    dict(row.get("route_counts") or {}).get(
                        "stock_closed_skeletons"
                    )
                    or 0
                )
                > 0
            ),
            "route_counts": dict(row.get("route_counts") or {}),
            "anytime_route_counts": dict(row.get("anytime_route_counts") or {}),
            "gate_summary": _gate_summary(row) if name in observed else {},
            "result_first_outcome": (
                _b4_outcome(row)
                if name in observed
                else str(row.get("status") or "unknown")
            ),
            "independent_axis_outcome": (
                _independent_axis_outcome(row)
                if name in observed
                else str(row.get("status") or "unknown")
            ),
            "open_result_axes": (
                _open_result_axes(_gate_summary(row)) if name in observed else []
            ),
            "time_to_first_s": {
                milestone: dict(raw or {}).get("elapsed_wall_time_s")
                for milestone, raw in dict(
                    dict(row.get("fixed_cutoff_projection") or {}).get(
                        "time_to_first"
                    )
                    or {}
                ).items()
                if milestone in {"first_route", "first_host_valid_route", "B4"}
            },
            "reaction_rejection_reason_counts": dict(
                row.get("reaction_rejection_reason_counts") or {}
            ),
            "rejection_taxonomy": dict(
                row.get("rejection_taxonomy") or {}
            ),
            "stock_audit_diagnostics": dict(
                row.get("stock_audit_diagnostics") or {}
            ),
            "provider_lineage_disposition_counts": dict(
                row.get("provider_lineage_disposition_counts") or {}
            ),
            "provider_search_attempts": list(
                row.get("provider_search_attempts") or []
            ),
            "candidate_provenance_first_loss_counts": dict(
                row.get("candidate_provenance_first_loss_counts") or {}
            ),
            "resource_observed": dict(row.get("resource_observed") or {}),
            "runtime_recovery": dict(row.get("runtime_recovery") or {}),
            "result_action_trace": dict(row.get("result_action_trace") or {}),
            "chemenzy": dict(row.get("chemenzy") or {}),
            "accepted_under_configured_policy": (
                row.get("accepted_under_configured_policy") is True
            ),
            "paper_equivalent": dict(row.get("paper_equivalent") or {}),
            "within_resource_budget": row.get("within_resource_budget") is True,
            "elapsed_s": row.get("elapsed_s"),
            "runner_elapsed_s": row.get("runner_elapsed_s"),
            "model_cost": dict(row.get("model_cost") or {}),
            "fixed_cutoff_projection_sha256": str(
                dict(row.get("fixed_cutoff_projection") or {}).get(
                    "content_sha256"
                )
                or ""
            ),
            "report_path": str(row.get("report_path") or ""),
            "error": str(row.get("error") or "")[:1000],
        }
        for name, row in sorted(target_rows.items())
    ]
    body = {
        "schema_version": "v4_blind_panel_summary.v3",
        "panel": {
            "manifest_path": str(status.get("manifest_path") or ""),
            "output_root": str(status.get("output_root") or ""),
            "model": str(status.get("model") or ""),
            "execution_profile": str(status.get("execution_profile") or ""),
            "ablation": str(status.get("ablation") or ""),
            "worker_count": int(status.get("worker_count") or 0),
            "fixed_cutoff_policy": dict(
                status.get("fixed_cutoff_policy") or {}
            ),
            "started_at": str(status.get("started_at") or ""),
            "finished_at": str(status.get("finished_at") or ""),
            "complete": status.get("complete") is True,
        },
        "counts": {
            "targets": total,
            "completed": len(completed),
            "running": status_counts["running"],
            "queued": status_counts["queued"],
            "pending": status_counts["running"] + status_counts["queued"],
            "failed": status_counts["failed"],
            "terminal_failed_or_incomplete": sum(
                count
                for state, count in status_counts.items()
                if state not in {"completed", "queued", "running"}
            ),
            "failed_or_incomplete": total - len(completed),
        },
        "metrics": rates,
        "result_first": {
            "milestone_counts": {
                gate: int(milestone_counts[gate])
                for gate in ("B1", "B2", "B4", "B5")
            },
            "milestone_rates": {
                    gate: {
                        "over_full_panel": round(milestone_counts[gate] / denominator, 6),
                        "over_completed": round(
                            completed_milestone_counts[gate]
                            / completed_denominator,
                            6,
                        ),
                }
                for gate in ("B1", "B2", "B4", "B5")
            },
            "time_to_first_s": {
                milestone: _distribution(values)
                for milestone, values in time_to_first.items()
            },
            "route_totals": dict(sorted(result_route_totals.items())),
            "b4_outcome_taxonomy": dict(sorted(b4_outcomes.items())),
            "independent_axis_taxonomy": dict(
                sorted(independent_axis_outcomes.items())
            ),
            "reaction_rejection_reason_counts": dict(
                sorted(reaction_rejection_reasons.items())
            ),
            "rejection_taxonomy": {
                "counts": dict(sorted(rejection_taxonomy_counts.items())),
                "reason_counts": {
                    category: dict(
                        sorted(
                            counts.items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                    )
                    for category, counts in sorted(
                        rejection_taxonomy_reasons.items()
                    )
                },
                "semantics": {
                    "report_only": True,
                    "no_execution_or_admission_authority": True,
                },
            },
            "stock_audit_totals": dict(sorted(stock_diagnostics.items())),
            "provider_lineage_disposition_counts": dict(
                sorted(provider_lineage_dispositions.items())
            ),
            "provider_search": {
                "counts": dict(sorted(provider_search.items())),
                "attempt_dispositions": dict(
                    sorted(provider_attempt_dispositions.items())
                ),
                "stop_reasons": dict(sorted(provider_search_stop_reasons.items())),
                "provider_time_s": _distribution(provider_search_time_s),
                "first_success_time_s": _distribution(
                    provider_first_success_time_s
                ),
                "post_first_success_time_s": _distribution(
                    provider_post_first_success_time_s
                ),
            },
            "provider_route_b4_funnel": {
                "attempt_counts": {
                    "attempted": int(provider_route_b4_funnel["attempt_count"]),
                    "raw_nonempty": int(
                        provider_route_b4_funnel["raw_nonempty_attempt_count"]
                    ),
                    "host_admitted_nonempty": int(
                        provider_route_b4_funnel[
                            "host_admitted_nonempty_attempt_count"
                        ]
                    ),
                },
                "route_counts": _provider_route_b4_funnel_counts(
                    provider_route_b4_funnel
                ),
                "route_counts_monotonic_nonincreasing": (
                    _is_nonincreasing(
                        _provider_route_b4_funnel_counts(
                            provider_route_b4_funnel
                        ).values()
                    )
                ),
                "target_endpoint": {
                    "completed": len(completed),
                    "root_b4": int(milestone_counts["B4"]),
                },
            },
            "provider_route_provenance_first_loss_counts": dict(
                sorted(provider_route_provenance_first_loss_counts.items())
            ),
            "selected_provider_route_b4_disposition_counts": dict(
                sorted(selected_provider_route_b4_dispositions.items())
            ),
            "result_loss_records": {
                "provider_non_success_attempt_count": len(
                    provider_non_success_attempt_records
                ),
                "provider_non_success_attempts": sorted(
                    provider_non_success_attempt_records,
                    key=lambda value: (
                        str(value.get("case_id") or ""),
                        str(value.get("artifact") or ""),
                    ),
                ),
                "selected_provider_route_b4_open_count": len(
                    selected_provider_route_b4_open_records
                ),
                "selected_provider_route_b4_open": sorted(
                    selected_provider_route_b4_open_records,
                    key=lambda value: (
                        str(value.get("case_id") or ""),
                        str(value.get("disposition") or ""),
                        str(value.get("route_trace_id") or ""),
                    ),
                ),
                "root_b4_open_selected_provider_route_count": sum(
                    value.get("root_b4_open") is True
                    for value in selected_provider_route_b4_open_records
                ),
                "root_b4_open_selected_provider_routes": [
                    value
                    for value in sorted(
                        selected_provider_route_b4_open_records,
                        key=lambda value: (
                            str(value.get("case_id") or ""),
                            str(value.get("disposition") or ""),
                            str(value.get("route_trace_id") or ""),
                        ),
                    )
                    if value.get("root_b4_open") is True
                ],
            },
            "start_cohort_latency": {
                "counts": dict(sorted(start_cohort.items())),
                "cohort_elapsed_s": _distribution(start_cohort_elapsed_s),
                "chemenzy_completed_s": _distribution(
                    start_chemenzy_completed_s
                ),
                "codex_completed_s": _distribution(start_codex_completed_s),
                "chemenzy_peer_wait_s": _distribution(
                    start_chemenzy_peer_wait_s
                ),
            },
            "provider_integration_loss_target_count": len(
                set(provider_integration_loss_targets)
            ),
            "provider_integration_loss_case_ids": sorted(
                set(provider_integration_loss_targets)
            ),
            "provider_topology_loss_target_count": len(
                set(provider_topology_loss_targets)
            ),
            "provider_topology_loss_case_ids": sorted(
                set(provider_topology_loss_targets)
            ),
            "provider_partial_lineage_target_count": len(
                provider_partial_lineage_targets
            ),
            "provider_partial_lineage_case_ids": sorted(
                provider_partial_lineage_targets
            ),
            "atom_balance_soft_gate_counterfactual_target_count": len(
                atom_balance_soft_gate_counterfactual_targets
            ),
            "atom_balance_soft_gate_counterfactual_case_ids": sorted(
                atom_balance_soft_gate_counterfactual_targets
            ),
            "atom_balance_frontier_signal_target_count": len(
                atom_balance_frontier_signal_targets
            ),
            "atom_balance_frontier_signal_case_ids": sorted(
                atom_balance_frontier_signal_targets
            ),
            "guided_frontier_solved_root_open_target_count": len(
                guided_frontier_solved_root_open_targets
            ),
            "guided_frontier_solved_root_open_case_ids": sorted(
                guided_frontier_solved_root_open_targets
            ),
            "action_counts": {
                "by_kind": dict(sorted(action_kinds.items())),
                "by_status": dict(sorted(action_statuses.items())),
                "total": sum(action_kinds.values()),
            },
            "chemenzy_totals": dict(sorted(chemenzy.items())),
            "pre_b4_credibility_action_target_count": len(
                pre_b4_credibility_action_targets
            ),
            "pre_b4_credibility_action_case_ids": sorted(
                pre_b4_credibility_action_targets
            ),
        },
        "resource_totals": dict(sorted(costs.items())),
        "resource_accounting": {
            "observed_totals": dict(sorted(resource_observed.items())),
            "recovery_totals": dict(sorted(runtime_recovery.items())),
        },
        "elapsed_s": _distribution(elapsed),
        "failure_categories": dict(sorted(failures.items())),
        "outcome_accounting": {
            "terminal_disposition_counts": dict(
                sorted(terminal_dispositions.items())
            ),
            "accounted_target_count": accounted_target_count,
            "unaccounted_target_count": max(0, total - accounted_target_count),
        },
        "per_target": per_target,
        "semantics": {
            "retrostar_comparable_solved_metric": (
                "official_benchmark_stock_closed"
            ),
            "retrostar_solved_requires_target_rooted_host_admitted_structure": True,
            "proof_and_condition_metrics_are_reported_separately": True,
            "full_panel_denominator_includes_failed_and_incomplete_targets": True,
            "score_fields_are_fixed_cutoff_trajectory_projections": True,
            "final_solver_state_is_not_used_for_scoring": True,
            "queued_and_running_targets_remain_in_full_panel_denominator": True,
            "credibility_work_is_audited_only_before_unreached_B4": True,
            "legacy_partial_lineage_loss_requires_unreached_B4": True,
            "explicit_route_topology_loss_is_reported_even_after_B4": True,
            "partial_provider_lineage_is_reported_separately": True,
            "B2_reaction_validation_and_B4_stock_are_independent_axes": True,
            "result_axis_taxonomy_does_not_imply_serial_gate_causality": True,
            "provider_post_first_success_time_is_diagnostic_not_a_new_cutoff": True,
            "start_cohort_peer_wait_is_operational_not_scientific_authority": True,
            "atom_balance_soft_gate_counterfactual_is_not_scored_as_B4": True,
            "frontier_stock_closure_alone_does_not_imply_root_B4": True,
            "guided_frontier_success_is_reported_separately_from_root_B4": True,
            "every_target_has_one_mutually_exclusive_terminal_disposition": True,
            "provider_empty_filtered_failure_and_timeout_are_distinct": True,
            "provider_route_provenance_first_loss_uses_canonical_records": True,
            "b4_funnel_excludes_advisory_and_does_not_require_b2": True,
            "result_loss_records_are_traceable_to_case_and_report": True,
            "runtime_recovery_is_distinct_from_chemical_stock_recovery": True,
        },
    }
    body["content_sha256"] = _digest(body)
    return body


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    rows = sorted(float(value) for value in values)
    if not rows:
        return {"count": 0, "sum": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "count": len(rows),
        "sum": round(sum(rows), 3),
        "mean": round(statistics.fmean(rows), 3),
        "median": round(statistics.median(rows), 3),
        "p95": round(rows[min(len(rows) - 1, math.ceil(0.95 * len(rows)) - 1)], 3),
    }


def _append_finite(values: list[float], value: Any) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            values.append(number)


def _failure_key(row: Mapping[str, Any]) -> str:
    status = str(row.get("status") or "unknown")
    error = str(row.get("error") or "").strip().splitlines()
    first = error[0][:160] if error else ""
    return f"{status}:{first}" if first else status


def _terminal_disposition(row: Mapping[str, Any]) -> str:
    """Assign one target-level result disposition without censoring failures."""

    status = str(row.get("status") or "unknown").casefold()
    if status == "completed":
        gates = _gate_summary(row)
        return (
            "completed_b4_stock_closed"
            if gates.get("B4") is True
            else f"completed_b4_open__{_b4_outcome(row, gates=gates)}"
        )
    if status in {"queued", "running"}:
        return f"pending_{status}"
    if status in {"failed", "timeout", "timed_out", "cancelled", "canceled"}:
        normalized = "timeout" if status in {"timeout", "timed_out"} else status
        normalized = "cancelled" if normalized in {"cancelled", "canceled"} else normalized
        return f"terminal_{normalized}"
    if status.startswith("preflight_"):
        return status
    return f"terminal_incomplete_{status or 'unknown'}"


def _provider_attempt_disposition(attempt: Mapping[str, Any]) -> str:
    if attempt.get("host_admitted_solved") is True:
        return "host_admitted_success"
    if attempt.get("raw_solved") is True:
        return "raw_nonempty_host_filtered"
    categories = {
        str(value).casefold()
        for value in attempt.get("backend_failure_categories") or []
        if str(value)
    }
    status = str(attempt.get("search_status") or "").casefold()
    stop_reason = str(attempt.get("stop_reason") or "").casefold()
    if "timeout" in status or "timeout" in stop_reason or any(
        "timeout" in value for value in categories
    ):
        return "provider_timeout"
    if categories and categories <= {"no_route_found", "no_routes_found"}:
        return "raw_empty_no_route"
    if categories or status in {"failed", "error"}:
        return "provider_failure"
    if int(attempt.get("native_raw_route_count") or 0) == 0:
        return "raw_empty_no_route"
    return "unresolved_without_host_admission"


def _selected_provider_route_b4_disposition(route: Mapping[str, Any]) -> str:
    if not route.get("candidate_ids"):
        return "canonical_ingestion_open"
    if not route.get("canonical_route_ids"):
        return "canonical_materialization_open"
    if not route.get("stock_closed_route_ids"):
        return "stock_closure_open"
    return "stock_closed"


def _provider_route_b4_funnel_counts(
    counts: Mapping[str, Any],
) -> dict[str, int]:
    return {
        "raw": int(counts.get("provider_raw_route_count") or 0),
        "normalized": int(counts.get("provider_normalized_route_count") or 0),
        "host_search_admitted": int(
            counts.get("host_search_admitted_route_count") or 0
        ),
        "provider_rule_gate_kept": int(
            counts.get("provider_rule_gate_kept_route_count") or 0
        ),
        "provider_output": int(counts.get("provider_output_route_count") or 0),
        "host_portfolio_selected": int(
            counts.get("selected_provider_route_count") or 0
        ),
        "canonical_bound_selected": int(
            counts.get("canonical_bound_selected_provider_route_count") or 0
        ),
        "canonical_materialized_selected": int(
            counts.get("canonical_materialized_selected_provider_route_count") or 0
        ),
        "stock_closed_selected": int(
            counts.get("stock_closed_selected_provider_route_count") or 0
        ),
    }


def _is_nonincreasing(values: Iterable[int]) -> bool:
    rows = list(values)
    return all(left >= right for left, right in zip(rows, rows[1:]))


def _gate_summary(row: Mapping[str, Any]) -> dict[str, bool]:
    direct = dict(row.get("gate_summary") or {})
    if direct:
        return {str(key): value is True for key, value in direct.items()}
    projection = dict(row.get("fixed_cutoff_projection") or {})
    milestones = dict(projection.get("milestones") or {})
    return {
        gate: milestones.get(source) is True
        for gate, source in {
            "B1": "B1_global_multi_route",
            "B2": "B2_host_validated_routes",
            "B4": "B4_stock_boundary",
            "B5": "B5_configured_portfolio_acceptance",
        }.items()
    }


def _b4_outcome(
    row: Mapping[str, Any], *, gates: Mapping[str, bool] | None = None
) -> str:
    resolved_gates = dict(gates or _gate_summary(row))
    if resolved_gates.get("B4") is True:
        return "stock_closed"
    route_counts = {
        **dict(row.get("route_counts") or {}),
        **dict(row.get("anytime_route_counts") or {}),
    }
    target_rooted = max(
        int(route_counts.get("target_rooted_route_count") or 0),
        int(route_counts.get("target_rooted_distinct_skeletons") or 0),
    )
    host_validated = max(
        int(route_counts.get("host_validated_route_count") or 0),
        int(route_counts.get("reaction_validated_skeletons") or 0),
    )
    chemenzy = dict(row.get("chemenzy") or {})
    chemenzy_status = str(chemenzy.get("status") or "").lower()
    initial_status = str(chemenzy.get("initial_delegation_status") or "").lower()
    if target_rooted == 0:
        if "timeout" in {chemenzy_status, initial_status}:
            return "no_route_provider_timeout"
        if chemenzy_status in {"failed", "error"}:
            return "no_route_provider_failure"
        return "no_target_rooted_route"
    if host_validated == 0 or resolved_gates.get("B2") is not True:
        return "route_present_host_validation_open"
    return "host_validated_stock_open"


def _independent_axis_outcome(
    row: Mapping[str, Any], *, gates: Mapping[str, bool] | None = None
) -> str:
    resolved_gates = dict(gates or _gate_summary(row))
    if resolved_gates.get("B1") is not True:
        chemenzy = dict(row.get("chemenzy") or {})
        statuses = {
            str(chemenzy.get("status") or "").lower(),
            str(chemenzy.get("initial_delegation_status") or "").lower(),
        }
        if "timeout" in statuses:
            return "route_absent__provider_timeout"
        if statuses & {"failed", "error"}:
            return "route_absent__provider_failure"
        return "route_absent"
    validation = "validation_closed" if resolved_gates.get("B2") is True else "validation_open"
    stock = "stock_closed" if resolved_gates.get("B4") is True else "stock_open"
    return f"route_present__{validation}__{stock}"


def _open_result_axes(gates: Mapping[str, bool]) -> list[str]:
    axes: list[str] = []
    if gates.get("B1") is not True:
        axes.append("route_generation")
    if gates.get("B2") is not True:
        axes.append("reaction_validation")
    if gates.get("B4") is not True:
        axes.append("stock_boundary")
    return axes


def _hydrate_report_diagnostics(status: Mapping[str, Any]) -> dict[str, Any]:
    """Attach read-only diagnostics omitted from compact panel rows."""

    hydrated = dict(status)
    targets: dict[str, Any] = {}
    for name, raw in dict(status.get("targets") or {}).items():
        row = dict(raw) if isinstance(raw, Mapping) else raw
        if not isinstance(row, dict):
            targets[str(name)] = row
            continue
        report_path = Path(str(row.get("report_path") or ""))
        if not report_path.is_file():
            targets[str(name)] = row
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            targets[str(name)] = row
            continue
        projection_available = (
            dict(row.get("fixed_cutoff_projection") or {}).get("available") is True
        )
        claim = dict(report.get("claim") or {})
        current_disposition = dict(report.get("current_disposition") or {})
        stop_decision = dict(report.get("stop_decision") or {})
        scientifically_terminal = bool(
            claim.get("accepted_under_configured_policy") is True
            or current_disposition.get("state") == "accepted"
            or (
                stop_decision.get("terminal") is True
                and str(stop_decision.get("decision") or "")
                in {"completed", "accepted"}
            )
        )
        row.setdefault(
            "scientific_status",
            "accepted" if scientifically_terminal else "unresolved",
        )
        row.setdefault(
            "scientific_disposition",
            str(current_disposition.get("state") or ""),
        )
        row["rejection_taxonomy"] = dict(
            report.get("rejection_taxonomy") or {}
        )
        # Migrate compact rows emitted before operational completion and
        # scientific acceptance were split.  Preserve the legacy value for
        # auditability; the frozen report and cutoff projection are unchanged.
        if row.get("status") == "incomplete" and projection_available:
            row["legacy_target_status"] = "incomplete"
            row["status"] = "completed"
        reasons = Counter()
        latest_stock: dict[str, int] = {}
        lineage_dispositions = Counter()
        eligible_incomplete_lineage_count = 0
        topology_conservation_failure_count = 0
        runtime_recovery = Counter()
        result_action_kinds: list[str] = []
        campaign_termination = ""
        for stage in report.get("stages") or []:
            if not isinstance(stage, Mapping):
                continue
            detail = dict(stage.get("detail") or {})
            if str(stage.get("stage") or "") == "chemenzy_route_lineage":
                lineage_dispositions.update(
                    {
                        str(key): int(value or 0)
                        for key, value in dict(
                            detail.get("disposition_counts") or {}
                        ).items()
                    }
                )
                eligible_incomplete_lineage_count += sum(
                    1
                    for route in detail.get("routes") or []
                    if isinstance(route, Mapping)
                    and route.get("final_disposition")
                    == "canonical_edges_present_outside_complete_measured_route"
                    and (
                        route.get("proposal_eligible") is True
                        or route.get("host_portfolio_selected") is True
                    )
                    and route.get("quarantined") is not True
                )
                topology_conservation_failure_count += sum(
                    1
                    for route in detail.get("routes") or []
                    if isinstance(route, Mapping)
                    and route.get("topology_conservation_applicable") is True
                    and route.get("topology_conservation_accepted") is not True
                )
            action = dict(detail.get("action") or {})
            kind = str(action.get("kind") or "")
            handler = dict(dict(detail.get("outcome") or {}).get("handler_result") or {})
            if kind:
                result_action_kinds.append(kind)
            if detail.get("cache_hit") is True:
                runtime_recovery["action_cache_hit_count"] += 1
            if detail.get("recovered_from_action_history") is True:
                runtime_recovery["action_history_recovery_count"] += 1
            if detail.get("outcome_pointer_recovered") is True:
                runtime_recovery["outcome_pointer_recovery_count"] += 1
            if handler.get("provider_result_replayed") is True:
                runtime_recovery["provider_result_replay_count"] += 1
            runtime_recovery["provider_result_replay_count"] += int(
                handler.get("provider_result_replay_count") or 0
            )
            if str(stage.get("stage") or "") == "campaign_anytime_core":
                campaign_termination = str(detail.get("termination") or "")
            if kind == "stock_audit":
                observed = {
                    key: int(handler.get(key) or 0)
                    for key in (
                        "selected_leaf_count",
                        "stock_closed_leaf_count",
                        "selected_stock_candidate_count",
                        "stock_closed_candidate_count",
                        "miss_count",
                        "remaining_pending_candidate_count",
                    )
                    if isinstance(handler.get(key), int)
                    and not isinstance(handler.get(key), bool)
                }
                if observed:
                    observed["stock_open_leaf_count"] = max(
                        0,
                        observed.get("selected_leaf_count", 0)
                        - observed.get("stock_closed_leaf_count", 0),
                    )
                    latest_stock = observed
            if kind != "reaction_validate":
                continue
            reasons.update(
                {
                    str(key): int(value or 0)
                    for key, value in dict(
                        handler.get("rejection_reason_counts") or {}
                    ).items()
                }
            )
        row["reaction_rejection_reason_counts"] = dict(sorted(reasons.items()))
        row["stock_audit_diagnostics"] = dict(sorted(latest_stock.items()))
        row["provider_lineage_disposition_counts"] = dict(
            sorted(lineage_dispositions.items())
        )
        row["provider_eligible_incomplete_lineage_count"] = (
            eligible_incomplete_lineage_count
        )
        row["provider_topology_conservation_failure_count"] = (
            topology_conservation_failure_count
        )
        provenance = dict(report.get("candidate_provenance") or {})
        row["candidate_provenance_first_loss_counts"] = {
            str(key): int(value or 0)
            for key, value in dict(provenance.get("first_loss_counts") or {}).items()
            if str(key)
        }
        row["canonical_bound_provider_route_count"] = int(
            provenance.get("bound_provider_route_count") or 0
        )
        provider_route_records = [
            dict(value)
            for value in provenance.get("provider_route_records") or []
            if isinstance(value, Mapping)
        ]
        selected_provider_routes = [
            value
            for value in provider_route_records
            if value.get("host_portfolio_selected") is True
        ]
        row["selected_provider_route_count"] = len(selected_provider_routes)
        row["canonical_bound_selected_provider_route_count"] = sum(
            bool(value.get("candidate_ids")) for value in selected_provider_routes
        )
        row["canonical_materialized_selected_provider_route_count"] = sum(
            bool(value.get("canonical_route_ids")) for value in selected_provider_routes
        )
        row["stock_closed_selected_provider_route_count"] = sum(
            bool(value.get("stock_closed_route_ids"))
            for value in selected_provider_routes
        )
        row["selected_provider_route_b4_disposition_counts"] = dict(
            sorted(
                Counter(
                    _selected_provider_route_b4_disposition(value)
                    for value in selected_provider_routes
                ).items()
            )
        )
        row["selected_provider_route_b4_records"] = [
            {
                "route_trace_id": str(value.get("route_trace_id") or ""),
                "disposition": _selected_provider_route_b4_disposition(value),
                "first_loss_boundary": str(
                    value.get("first_loss_boundary") or ""
                ),
                "final_disposition": str(value.get("final_disposition") or ""),
                "raw_route_sha256": str(value.get("raw_route_sha256") or ""),
                "normalized_route_sha256": str(
                    value.get("normalized_route_sha256") or ""
                ),
                "canonical_route_family_id": str(
                    value.get("canonical_route_family_id") or ""
                ),
                "candidate_count": len(value.get("candidate_ids") or []),
                "canonical_edge_count": len(
                    value.get("canonical_edge_ids") or []
                ),
                "canonical_route_count": len(
                    value.get("canonical_route_ids") or []
                ),
                "stock_closed_route_count": len(
                    value.get("stock_closed_route_ids") or []
                ),
            }
            for value in selected_provider_routes
        ]
        row["resource_observed"] = {
            str(key): value
            for key, value in dict(
                dict(report.get("resource_envelope") or {}).get("observed") or {}
            ).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        row["runtime_recovery"] = dict(sorted(runtime_recovery.items()))
        try:
            first_closure = result_action_kinds.index("recompute_route_closure")
        except ValueError:
            first_closure = -1
        row["result_action_trace"] = {
            "recompute_route_closure_count": result_action_kinds.count(
                "recompute_route_closure"
            ),
            "guided_before_first_route_closure": (
                result_action_kinds[:first_closure].count(
                    "native_short_tail_expand"
                )
                if first_closure >= 0
                else result_action_kinds.count("native_short_tail_expand")
            ),
            "guided_after_first_route_closure": (
                result_action_kinds[first_closure + 1 :].count(
                    "native_short_tail_expand"
                )
                if first_closure >= 0
                else 0
            ),
            "campaign_termination": campaign_termination,
        }
        report_target = report.get("target")
        campaign_target = dict(report.get("campaign_spec") or {}).get("target")
        root_target_smiles = str(
            (
                dict(report_target).get("canonical_smiles")
                if isinstance(report_target, Mapping)
                else ""
            )
            or (
                dict(campaign_target).get("canonical_smiles")
                if isinstance(campaign_target, Mapping)
                else ""
            )
            or ""
        )
        row["provider_search_attempts"] = _provider_search_attempts(
            report_path,
            root_target_smiles=root_target_smiles,
        )
        row["start_cohort_latency_audit"] = _start_cohort_latency_audit(report)
        targets[str(name)] = row
    hydrated["targets"] = targets
    return hydrated


def _start_cohort_latency_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one initial cohort audit without depending on stage position."""

    for stage in report.get("stages") or []:
        if not isinstance(stage, Mapping):
            continue
        detail = dict(stage.get("detail") or {})
        audit = dict(dict(detail.get("start_cohort") or {}).get("latency_audit") or {})
        if audit.get("schema_version") == "campaign_action_cohort_latency_audit.v1":
            return audit
    return {}


def _provider_search_attempts(
    report_path: Path,
    *,
    root_target_smiles: str = "",
) -> list[dict[str, Any]]:
    """Read compact search telemetry from native/guided provider artifacts."""

    attempts: list[dict[str, Any]] = []
    try:
        solve_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        solve_report = {}
    seen_native_attempts: set[tuple[Any, ...]] = set()
    for stage in solve_report.get("stages") or []:
        if not isinstance(stage, Mapping):
            continue
        detail = dict(stage.get("detail") or {})
        for result_index, raw_result in enumerate(detail.get("results") or []):
            if not isinstance(raw_result, Mapping):
                continue
            result = dict(raw_result)
            provider_id = str(result.get("provider_id") or "")
            if provider_id != "aizynthfinder":
                continue
            invocation_count = int(result.get("provider_invocation_count") or 0)
            if invocation_count <= 0:
                continue
            frontier_values = [
                str(value) for value in result.get("frontier_smiles") or [] if str(value)
            ]
            statistics = dict(result.get("statistics") or {})
            profiling = dict(statistics.get("profiling") or {})
            identity = (
                provider_id,
                tuple(frontier_values),
                str(statistics.get("target") or ""),
                _finite_number(statistics.get("search_time")),
            )
            if identity in seen_native_attempts:
                continue
            seen_native_attempts.add(identity)
            accepted_route_count = int(result.get("accepted_route_count") or 0)
            rejected_route_count = int(result.get("rejected_route_count") or 0)
            selected_route_count = int(
                result.get("selected_proposal_route_count") or 0
            )
            complete_route_count = int(
                result.get("complete_provider_route_count") or 0
            )
            provider_solved = bool(result.get("provider_solved"))
            search_time = _finite_number(statistics.get("search_time"))
            first_solution_time = _finite_number(
                statistics.get("first_solution_time")
            )
            attempts.append(
                {
                    "artifact": (
                        f"{stage.get('stage') or 'native_short_tail'}:"
                        f"{result_index + 1}"
                    ),
                    "provider_id": provider_id,
                    "kind": "guided",
                    "search_target_smiles": (
                        frontier_values[0] if frontier_values else ""
                    ),
                    "search_target_is_root": bool(
                        root_target_smiles
                        and frontier_values
                        and frontier_values[0] == root_target_smiles
                    ),
                    "ok": str(result.get("status") or "")
                    in {"completed", "unresolved"},
                    "raw_solved": provider_solved,
                    "host_admitted_solved": bool(
                        provider_solved
                        and complete_route_count > 0
                        and selected_route_count > 0
                    ),
                    "search_status": str(result.get("status") or ""),
                    "backend_failure_categories": (
                        [str(result.get("reason") or "native_search_failed")]
                        if str(result.get("status") or "") == "failed"
                        else []
                    ),
                    "provider_time_s": search_time,
                    "first_success_time_s": (
                        first_solution_time
                        if first_solution_time is not None
                        and first_solution_time > 0
                        else None
                    ),
                    "native_raw_route_count": _integer(
                        statistics.get("number_of_routes")
                    ),
                    "native_found_route_count": accepted_route_count,
                    "normalized_route_count": accepted_route_count,
                    "host_search_admitted_route_count": selected_route_count,
                    "host_search_rejected_route_count": rejected_route_count,
                    "output_route_count": selected_route_count,
                    "first_output_raw_route_index": (
                        0 if selected_route_count > 0 else None
                    ),
                    "rule_gate_input_route_count": (
                        accepted_route_count + rejected_route_count
                    ),
                    "rule_gate_kept_route_count": accepted_route_count,
                    "rule_gate_dropped_route_count": rejected_route_count,
                    "quarantined_route_count": rejected_route_count,
                    "atom_balance_only_quarantine_count": 0,
                    "atom_balance_only_stock_closed_quarantine_count": 0,
                    "hard_structure_quarantine_count": 0,
                    "executed_iterations": _integer(
                        profiling.get("iterations")
                    ),
                    "stop_reason": str(result.get("reason") or ""),
                }
            )
    for result_path in sorted(report_path.parent.glob("chemenzy-v4-*-result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        stem = result_path.name.removeprefix("chemenzy-v4-").removesuffix(
            "-result.json"
        )
        kind = "native" if stem == "seed" else "guided"
        backend = dict(result.get("raw_backend_metadata") or {})
        stop = dict(backend.get("search_stop") or {})
        search = dict(result.get("search_status") or {})
        route_metrics = dict(result.get("route_set_metrics") or {})
        rule_gate = dict(route_metrics.get("cascade_verifier_gate") or {})
        host_admission = dict(
            route_metrics.get("route_materialization_admission") or {}
        )
        quarantined_routes = [
            dict(route)
            for route in result.get("quarantined_routes") or []
            if isinstance(route, Mapping)
        ]
        atom_balance_only_quarantine_count = 0
        atom_balance_only_stock_closed_quarantine_count = 0
        hard_structure_quarantine_count = 0
        for route in quarantined_routes:
            verifier = dict(dict(route.get("metrics") or {}).get("cascade_verifier") or {})
            reasons = {
                str(finding.get("reason") or "")
                for finding in verifier.get("findings") or []
                if isinstance(finding, Mapping) and str(finding.get("reason") or "")
            }
            if reasons == {"atom_balance_violation"}:
                atom_balance_only_quarantine_count += 1
                if dict(route.get("metrics") or {}).get("strict_stock_solve") is True:
                    atom_balance_only_stock_closed_quarantine_count += 1
            if reasons & {
                "invalid_smiles",
                "product_mismatch",
                "route_order_mismatch",
            }:
                hard_structure_quarantine_count += 1
        output_raw_route_indices = sorted(
            int(index)
            for route in result.get("routes") or []
            if isinstance(route, Mapping)
            and isinstance(
                index := dict(route.get("raw_backend_metadata") or {}).get(
                    "route_index"
                ),
                int,
            )
            and not isinstance(index, bool)
            and index >= 0
        )
        attempts.append(
            {
                "artifact": result_path.name,
                "kind": kind,
                "search_target_smiles": str(result.get("target") or ""),
                "search_target_is_root": bool(
                    root_target_smiles
                    and str(result.get("target") or "") == root_target_smiles
                ),
                "ok": result.get("ok") is True,
                "raw_solved": result.get("raw_solved") is True,
                "host_admitted_solved": (
                    result.get("materialization_admission_solved") is True
                ),
                "search_status": str(search.get("status") or ""),
                "backend_failure_categories": sorted(
                    {
                        str(failure.get("category") or "")
                        for failure in result.get("backend_failures") or []
                        if isinstance(failure, Mapping)
                        and str(failure.get("category") or "")
                    }
                ),
                "provider_time_s": _finite_number(result.get("time_s")),
                "first_success_time_s": _finite_number(
                    backend.get("first_succ_time")
                ),
                "native_raw_route_count": _integer(
                    search.get("native_raw_n_routes")
                ),
                "native_found_route_count": _integer(
                    search.get("native_search_found_n_routes")
                ),
                "normalized_route_count": _integer(
                    host_admission.get("audited_route_count")
                ),
                "host_search_admitted_route_count": _integer(
                    host_admission.get("accepted_route_count")
                ),
                "host_search_rejected_route_count": _integer(
                    host_admission.get("rejected_route_count")
                ),
                "output_route_count": _integer(result.get("n_results")),
                "first_output_raw_route_index": (
                    output_raw_route_indices[0]
                    if output_raw_route_indices
                    else None
                ),
                "rule_gate_input_route_count": _integer(
                    rule_gate.get("input_routes")
                ),
                "rule_gate_kept_route_count": _integer(rule_gate.get("kept_routes")),
                "rule_gate_dropped_route_count": _integer(
                    rule_gate.get("dropped_routes")
                ),
                "quarantined_route_count": len(quarantined_routes),
                "atom_balance_only_quarantine_count": (
                    atom_balance_only_quarantine_count
                ),
                "atom_balance_only_stock_closed_quarantine_count": (
                    atom_balance_only_stock_closed_quarantine_count
                ),
                "hard_structure_quarantine_count": hard_structure_quarantine_count,
                "executed_iterations": _integer(
                    stop.get("executed_iterations", backend.get("iter"))
                ),
                "stop_reason": str(stop.get("reason") or ""),
            }
        )
    return attempts


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _markdown(summary: Mapping[str, Any]) -> str:
    counts = dict(summary.get("counts") or {})
    outcome_accounting = dict(summary.get("outcome_accounting") or {})
    resource_accounting = dict(summary.get("resource_accounting") or {})
    lines = [
        "# V4 Blind Panel Summary",
        "",
        f"- Targets: {counts.get('targets', 0)}",
        f"- Completed: {counts.get('completed', 0)}",
        f"- Running / queued: {counts.get('running', 0)} / "
        f"{counts.get('queued', 0)}",
        f"- Terminal failed or incomplete: "
        f"{counts.get('terminal_failed_or_incomplete', 0)}",
        "",
        "| Metric | Count | Full-panel rate | Completed rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, raw in dict(summary.get("metrics") or {}).items():
        row = dict(raw)
        lines.append(
            f"| {metric} | {row.get('count', 0)} | "
            f"{100 * float(row.get('rate_over_full_panel') or 0):.2f}% | "
            f"{100 * float(row.get('rate_over_completed') or 0):.2f}% |"
        )
    result_first = dict(summary.get("result_first") or {})
    lines.extend(
        [
            "",
            "## Result-first milestones",
            "",
            "| Milestone | Count | Full-panel rate | Completed rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    milestone_counts = dict(result_first.get("milestone_counts") or {})
    milestone_rates = dict(result_first.get("milestone_rates") or {})
    for milestone in ("B1", "B2", "B4", "B5"):
        rate = dict(milestone_rates.get(milestone) or {})
        lines.append(
            f"| {milestone} | {milestone_counts.get(milestone, 0)} | "
            f"{100 * float(rate.get('over_full_panel') or 0):.2f}% | "
            f"{100 * float(rate.get('over_completed') or 0):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Result bottlenecks",
            "",
            "B2 reaction validation and B4 stock closure are independent axes; "
            "the table does not imply a serial gate.",
            "",
            "| Independent-axis outcome | Completed targets |",
            "| --- | ---: |",
        ]
    )
    for outcome, count in dict(
        result_first.get("independent_axis_taxonomy") or {}
    ).items():
        lines.append(f"| {outcome} | {count} |")
    lines.extend(
        [
            "",
            "## Outcome accounting",
            "",
            f"- Accounted targets: "
            f"{outcome_accounting.get('accounted_target_count', 0)}",
            f"- Unaccounted targets: "
            f"{outcome_accounting.get('unaccounted_target_count', 0)}",
            "",
            "| Mutually exclusive terminal disposition | Targets |",
            "| --- | ---: |",
        ]
    )
    for disposition, count in dict(
        outcome_accounting.get("terminal_disposition_counts") or {}
    ).items():
        lines.append(f"| {disposition} | {count} |")
    lines.extend(
        [
            "",
            "| Time to first | Count | Median (s) | P95 (s) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for milestone in ("first_route", "first_host_valid_route", "B4"):
        distribution = dict(
            dict(result_first.get("time_to_first_s") or {}).get(milestone) or {}
        )
        lines.append(
            f"| {milestone} | {distribution.get('count', 0)} | "
            f"{distribution.get('median', 0)} | {distribution.get('p95', 0)} |"
        )
    provider = dict(result_first.get("provider_search") or {})
    provider_counts = dict(provider.get("counts") or {})
    provider_attempt_dispositions = dict(
        provider.get("attempt_dispositions") or {}
    )
    provider_funnel = dict(
        result_first.get("provider_route_b4_funnel") or {}
    )
    provider_funnel_attempts = dict(provider_funnel.get("attempt_counts") or {})
    provider_funnel_routes = dict(provider_funnel.get("route_counts") or {})
    provider_b4_dispositions = dict(
        result_first.get("selected_provider_route_b4_disposition_counts") or {}
    )
    provider_provenance_losses = dict(
        result_first.get("provider_route_provenance_first_loss_counts") or {}
    )
    result_loss_records = dict(result_first.get("result_loss_records") or {})
    cohort = dict(result_first.get("start_cohort_latency") or {})
    cohort_counts = dict(cohort.get("counts") or {})
    lines.extend(
        [
            "",
            "## Result-throughput diagnostics",
            "",
            "| Diagnostic | Count / median |",
            "| --- | ---: |",
            f"| Provider attempts | {provider_counts.get('attempt_count', 0)} |",
            f"| Host-admitted provider attempts | "
            f"{provider_counts.get('host_admitted_solved_attempt_count', 0)} |",
            f"| Host solution already within raw top-4 | "
            f"{provider_counts.get('host_solution_within_raw_prefix_4_attempt_count', 0)} |",
            f"| Host solution already within raw top-8 | "
            f"{provider_counts.get('host_solution_within_raw_prefix_8_attempt_count', 0)} |",
            f"| Atom-balance-only quarantined stock-closed routes | "
            f"{provider_counts.get('atom_balance_only_stock_closed_quarantine_count', 0)} |",
            f"| Provider runtime median (s) | "
            f"{dict(provider.get('provider_time_s') or {}).get('median', 0)} |",
            f"| First raw success median (s) | "
            f"{dict(provider.get('first_success_time_s') or {}).get('median', 0)} |",
            f"| Time retained after first raw success median (s) | "
            f"{dict(provider.get('post_first_success_time_s') or {}).get('median', 0)} |",
            f"| ChemEnzy completed before Codex | "
            f"{cohort_counts.get('chemenzy_completed_before_codex_target_count', 0)} |",
            f"| ChemEnzy peer-wait median (s) | "
            f"{dict(cohort.get('chemenzy_peer_wait_s') or {}).get('median', 0)} |",
        ]
    )
    lines.extend(
        [
            "",
            "## Provider attempt outcomes",
            "",
            "Every provider attempt is assigned exactly one result disposition.",
            "",
            "| Attempt disposition | Attempts |",
            "| --- | ---: |",
        ]
    )
    for disposition, count in provider_attempt_dispositions.items():
        lines.append(f"| {disposition} | {count} |")
    lines.extend(
        [
            "",
            "## Selected provider route to B4 funnel",
            "",
            "This is the serial result-delivery funnel. It excludes advisory routes "
            "and does not require B2 reaction validation.",
            "",
            f"- Attempts / raw nonempty / host-admitted nonempty: "
            f"{provider_funnel_attempts.get('attempted', 0)} / "
            f"{provider_funnel_attempts.get('raw_nonempty', 0)} / "
            f"{provider_funnel_attempts.get('host_admitted_nonempty', 0)}",
            f"- Route counts monotonic non-increasing: "
            f"{provider_funnel.get('route_counts_monotonic_nonincreasing', False)}",
            "",
            "| Funnel stage | Routes |",
            "| --- | ---: |",
        ]
    )
    for stage, count in provider_funnel_routes.items():
        lines.append(f"| {stage} | {count} |")
    lines.extend(
        [
            "",
            "| Selected-route B4 disposition | Routes |",
            "| --- | ---: |",
        ]
    )
    for disposition, count in provider_b4_dispositions.items():
        lines.append(f"| {disposition} | {count} |")
    lines.extend(
        [
            "",
            "## Actionable result-loss records",
            "",
            f"- Selected provider routes still open on a root-B4-open target: "
            f"{result_loss_records.get('root_b4_open_selected_provider_route_count', 0)}",
            "",
            "| Case | Provider artifact | Attempt kind | Attempt disposition | Raw | Normalized | Rule-kept | Output |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    non_success_attempts = list(
        result_loss_records.get("provider_non_success_attempts") or []
    )
    if non_success_attempts:
        for raw in non_success_attempts:
            row = dict(raw)
            lines.append(
                f"| {row.get('case_id', '')} | {row.get('artifact', '')} | "
                f"{row.get('kind', '')} | {row.get('disposition', '')} | "
                f"{row.get('raw_route_count', 0)} | "
                f"{row.get('normalized_route_count', 0)} | "
                f"{row.get('rule_gate_kept_route_count', 0)} | "
                f"{row.get('output_route_count', 0)} |"
            )
    else:
        lines.append("| none |  |  |  | 0 | 0 | 0 | 0 |")
    lines.extend(
        [
            "",
            "| Case | Root B4 open | Route trace | B4 disposition | Canonical edges | Canonical routes | Stock-closed routes | Report |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    b4_open_routes = list(
        result_loss_records.get("selected_provider_route_b4_open") or []
    )
    if b4_open_routes:
        for raw in b4_open_routes:
            row = dict(raw)
            lines.append(
                f"| {row.get('case_id', '')} | "
                f"{int(row.get('root_b4_open') is True)} | "
                f"{row.get('route_trace_id', '')} | "
                f"{row.get('disposition', '')} | "
                f"{row.get('canonical_edge_count', 0)} | "
                f"{row.get('canonical_route_count', 0)} | "
                f"{row.get('stock_closed_route_count', 0)} | "
                f"{row.get('report_path', '')} |"
            )
    else:
        lines.append("| none | 0 |  |  | 0 | 0 | 0 |  |")
    lines.extend(
        [
            "",
            "## All-route canonical provenance first loss",
            "",
            "This audit includes selected, advisory, and quarantined provider routes. "
            "It is not a serial B4 funnel and must not be read as B4 causal attrition.",
            "",
            "| Canonical provenance first loss | Routes |",
            "| --- | ---: |",
        ]
    )
    for disposition, count in provider_provenance_losses.items():
        lines.append(f"| {disposition} | {count} |")
    lines.extend(
        [
            "",
            "## Resource and recovery accounting",
            "",
            "| Observed resource | Total |",
            "| --- | ---: |",
        ]
    )
    for resource, value in dict(
        resource_accounting.get("observed_totals") or {}
    ).items():
        lines.append(f"| {resource} | {value} |")
    lines.extend(
        [
            "",
            "| Recovery path | Count |",
            "| --- | ---: |",
        ]
    )
    for recovery, count in dict(
        resource_accounting.get("recovery_totals") or {}
    ).items():
        lines.append(f"| {recovery} | {count} |")
    lines.extend(
        [
            "",
            "## Failures",
            "",
            "| Failure category | Targets |",
            "| --- | ---: |",
        ]
    )
    failure_categories = dict(summary.get("failure_categories") or {})
    if failure_categories:
        for category, count in failure_categories.items():
            lines.append(f"| {category} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## Per-target result view",
            "",
            "| Case | Status | Terminal disposition | Independent-axis outcome | B1 | B2 | B4 | Routes | Host-valid | "
            "Stock-closed | Elapsed (s) |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: |",
        ]
    )
    for raw in summary.get("per_target") or []:
        row = dict(raw)
        gates = dict(row.get("gate_summary") or {})
        routes = {
            **dict(row.get("route_counts") or {}),
            **dict(row.get("anytime_route_counts") or {}),
        }
        lines.append(
            f"| {row.get('case_id') or row.get('target_name') or ''} | "
            f"{row.get('status', '')} | {row.get('terminal_disposition', '')} | "
            f"{row.get('independent_axis_outcome', '')} | "
            f"{int(gates.get('B1') is True)} | {int(gates.get('B2') is True)} | "
            f"{int(gates.get('B4') is True)} | "
            f"{routes.get('target_rooted_route_count', 0)} | "
            f"{routes.get('host_validated_route_count', 0)} | "
            f"{routes.get('stock_closed_route_count', 0)} | "
            f"{row.get('elapsed_s') or ''} |"
        )
    lines.extend(
        [
            "",
            "## Priority audit",
            "",
            "- Credibility actions on targets that never reached B4: "
            f"{result_first.get('pre_b4_credibility_action_target_count', 0)}",
            "- Provider routes with canonical edges outside the measured parent route: "
            f"{result_first.get('provider_integration_loss_target_count', 0)} targets",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

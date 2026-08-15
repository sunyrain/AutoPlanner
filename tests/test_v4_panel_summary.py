from __future__ import annotations

import json

from scripts.summarize_v4_blind_panel import (
    _hydrate_report_diagnostics,
    _markdown,
    summarize_panel,
)


def test_panel_summary_keeps_stock_solved_separate_from_proof_acceptance() -> None:
    summary = summarize_panel(
        {
            "target_count": 2,
            "targets": {
                "target 1": {
                    "status": "completed",
                    "case_id": "one",
                    "accepted_under_configured_policy": False,
                    "within_resource_budget": True,
                    "route_counts": {
                        "target_rooted_distinct_skeletons": 1,
                        "materialized_skeletons": 1,
                        "reaction_validated_skeletons": 0,
                        "stock_closed_skeletons": 1,
                        "evidence_closed_skeletons": 0,
                    },
                    "model_cost": {"model_invocations": 2, "input_tokens": 100},
                    "elapsed_s": 10,
                    "gate_summary": {
                        "B1": True,
                        "B2": False,
                        "B4": True,
                        "B5": False,
                    },
                    "anytime_route_counts": {
                        "target_rooted_route_count": 2,
                        "stock_closed_route_count": 1,
                    },
                    "fixed_cutoff_projection": {
                        "time_to_first": {
                            "first_route": {"elapsed_wall_time_s": 2.0},
                            "B4": {"elapsed_wall_time_s": 8.0},
                        },
                        "action_counts": {
                            "by_kind": {
                                "chemenzy_target_expand": 1,
                                "bind_exact_evidence": 2,
                            },
                            "by_status": {"completed": 3},
                        },
                    },
                    "chemenzy": {"provider_invocation_count": 1},
                },
                "target 2": {
                    "status": "failed",
                    "case_id": "two",
                    "error": "provider timeout",
                },
            },
        }
    )

    assert summary["metrics"]["official_benchmark_stock_closed"] == {
        "count": 1,
        "rate_over_full_panel": 0.5,
        "rate_over_completed": 1.0,
    }
    assert summary["metrics"]["configured_proof_policy_accepted"]["count"] == 0
    assert summary["resource_totals"]["model_invocations"] == 2
    assert summary["counts"]["failed_or_incomplete"] == 1
    assert summary["counts"]["pending"] == 0
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["terminal_failed_or_incomplete"] == 1
    assert summary["result_first"]["milestone_counts"] == {
        "B1": 1,
        "B2": 0,
        "B4": 1,
        "B5": 0,
    }
    assert summary["result_first"]["time_to_first_s"]["B4"]["median"] == 8.0
    assert summary["result_first"]["route_totals"]["target_rooted_route_count"] == 2
    assert summary["result_first"]["action_counts"]["by_kind"] == {
        "bind_exact_evidence": 2,
        "chemenzy_target_expand": 1,
    }
    assert summary["result_first"]["pre_b4_credibility_action_target_count"] == 0
    assert summary["result_first"]["b4_outcome_taxonomy"] == {"stock_closed": 1}
    assert summary["result_first"]["independent_axis_taxonomy"] == {
        "route_present__validation_open__stock_closed": 1
    }
    assert summary["per_target"][0]["result_first_outcome"] == "stock_closed"
    assert summary["per_target"][0]["open_result_axes"] == ["reaction_validation"]
    assert summary["per_target"][0]["time_to_first_s"] == {
        "B4": 8.0,
        "first_route": 2.0,
    }


def test_panel_summary_flags_credibility_work_before_unreached_b4() -> None:
    summary = summarize_panel(
        {
            "target_count": 2,
            "targets": {
                "target 1": {
                    "status": "completed",
                    "case_id": "one",
                    "gate_summary": {"B1": True, "B2": True, "B4": False},
                    "fixed_cutoff_projection": {
                        "action_counts": {
                            "by_kind": {"program_discover": 3},
                            "by_status": {"completed": 3},
                        }
                    },
                },
                "target 2": {"status": "queued", "case_id": "two"},
            },
        }
    )

    assert summary["result_first"]["pre_b4_credibility_action_target_count"] == 1
    assert summary["result_first"]["pre_b4_credibility_action_case_ids"] == ["one"]
    assert summary["result_first"]["milestone_rates"]["B2"] == {
        "over_full_panel": 0.5,
        "over_completed": 1.0,
    }
    assert summary["result_first"]["b4_outcome_taxonomy"] == {
        "no_target_rooted_route": 1
    }
    assert summary["result_first"]["independent_axis_taxonomy"] == {
        "route_present__validation_closed__stock_open": 1
    }
    assert summary["counts"]["pending"] == 1
    assert summary["counts"]["terminal_failed_or_incomplete"] == 0


def test_panel_summary_classifies_result_first_b4_bottlenecks() -> None:
    summary = summarize_panel(
        {
            "target_count": 3,
            "targets": {
                "timeout": {
                    "status": "completed",
                    "case_id": "timeout",
                    "gate_summary": {"B1": False, "B2": False, "B4": False},
                    "chemenzy": {
                        "status": "timeout",
                        "initial_delegation_status": "timeout",
                    },
                },
                "validation": {
                    "status": "completed",
                    "case_id": "validation",
                    "gate_summary": {"B1": True, "B2": False, "B4": False},
                    "anytime_route_counts": {
                        "target_rooted_route_count": 2,
                        "host_validated_route_count": 0,
                    },
                },
                "stock": {
                    "status": "completed",
                    "case_id": "stock",
                    "gate_summary": {"B1": True, "B2": True, "B4": False},
                    "anytime_route_counts": {
                        "target_rooted_route_count": 2,
                        "host_validated_route_count": 1,
                    },
                },
            },
        }
    )

    assert summary["result_first"]["b4_outcome_taxonomy"] == {
        "host_validated_stock_open": 1,
        "no_route_provider_timeout": 1,
        "route_present_host_validation_open": 1,
    }
    assert summary["result_first"]["independent_axis_taxonomy"] == {
        "route_absent__provider_timeout": 1,
        "route_present__validation_closed__stock_open": 1,
        "route_present__validation_open__stock_open": 1,
    }


def test_panel_summary_accounts_for_every_target_and_provider_attempt() -> None:
    summary = summarize_panel(
        {
            "target_count": 5,
            "targets": {
                "solved": {
                    "status": "completed",
                    "case_id": "solved",
                    "gate_summary": {"B1": True, "B2": False, "B4": True},
                    "provider_search_attempts": [
                        {
                            "kind": "native",
                            "raw_solved": True,
                            "host_admitted_solved": True,
                            "native_raw_route_count": 6,
                            "normalized_route_count": 6,
                            "host_search_admitted_route_count": 5,
                            "rule_gate_kept_route_count": 2,
                            "output_route_count": 2,
                        }
                    ],
                    "candidate_provenance_first_loss_counts": {"none": 2},
                    "selected_provider_route_count": 2,
                    "canonical_bound_selected_provider_route_count": 2,
                    "canonical_materialized_selected_provider_route_count": 2,
                    "stock_closed_selected_provider_route_count": 2,
                    "selected_provider_route_b4_disposition_counts": {
                        "stock_closed": 2
                    },
                    "resource_observed": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "native_search_committed": 1,
                    },
                    "runtime_recovery": {
                        "action_cache_hit_count": 1,
                        "action_history_recovery_count": 1,
                        "provider_result_replay_count": 0,
                    },
                    "result_action_trace": {
                        "recompute_route_closure_count": 1,
                        "guided_before_first_route_closure": 0,
                        "guided_after_first_route_closure": 1,
                        "campaign_termination": "milestone_reached",
                    },
                },
                "filtered": {
                    "status": "completed",
                    "case_id": "filtered",
                    "gate_summary": {"B1": False, "B2": False, "B4": False},
                    "provider_search_attempts": [
                        {
                            "kind": "native",
                            "raw_solved": True,
                            "host_admitted_solved": False,
                            "native_raw_route_count": 3,
                            "normalized_route_count": 3,
                            "host_search_admitted_route_count": 3,
                            "rule_gate_kept_route_count": 0,
                            "output_route_count": 0,
                        }
                    ],
                    "candidate_provenance_first_loss_counts": {
                        "host_quarantine": 3
                    },
                },
                "empty": {
                    "status": "completed",
                    "case_id": "empty",
                    "gate_summary": {"B1": False, "B2": False, "B4": False},
                    "provider_search_attempts": [
                        {
                            "kind": "native",
                            "raw_solved": False,
                            "host_admitted_solved": False,
                            "native_raw_route_count": 0,
                            "backend_failure_categories": ["no_route_found"],
                        }
                    ],
                },
                "failed": {
                    "status": "failed",
                    "case_id": "failed",
                    "error": "provider crashed",
                },
                "queued": {"status": "queued", "case_id": "queued"},
            },
        }
    )

    assert summary["outcome_accounting"]["terminal_disposition_counts"] == {
        "completed_b4_open__no_target_rooted_route": 2,
        "completed_b4_stock_closed": 1,
        "pending_queued": 1,
        "terminal_failed": 1,
    }
    assert summary["outcome_accounting"]["accounted_target_count"] == 5
    assert summary["outcome_accounting"]["unaccounted_target_count"] == 0
    assert summary["result_first"]["provider_search"]["attempt_dispositions"] == {
        "host_admitted_success": 1,
        "raw_empty_no_route": 1,
        "raw_nonempty_host_filtered": 1,
    }
    assert summary["result_first"]["provider_route_b4_funnel"] == {
        "attempt_counts": {
            "attempted": 3,
            "host_admitted_nonempty": 1,
            "raw_nonempty": 2,
        },
        "route_counts": {
            "raw": 9,
            "normalized": 9,
            "host_search_admitted": 8,
            "provider_rule_gate_kept": 2,
            "provider_output": 2,
            "host_portfolio_selected": 2,
            "canonical_bound_selected": 2,
            "canonical_materialized_selected": 2,
            "stock_closed_selected": 2,
        },
        "route_counts_monotonic_nonincreasing": True,
        "target_endpoint": {"completed": 3, "root_b4": 1},
    }
    assert summary["result_first"][
        "provider_route_provenance_first_loss_counts"
    ] == {
        "host_quarantine": 3,
        "none": 2,
    }
    assert summary["result_first"][
        "selected_provider_route_b4_disposition_counts"
    ] == {"stock_closed": 2}
    loss_records = summary["result_first"]["result_loss_records"]
    assert loss_records["provider_non_success_attempt_count"] == 2
    assert [
        row["disposition"]
        for row in loss_records["provider_non_success_attempts"]
    ] == ["raw_empty_no_route", "raw_nonempty_host_filtered"]
    assert loss_records["selected_provider_route_b4_open_count"] == 0
    assert loss_records["root_b4_open_selected_provider_route_count"] == 0
    assert summary["resource_accounting"]["observed_totals"] == {
        "input_tokens": 100,
        "native_search_committed": 1,
        "output_tokens": 20,
    }
    assert summary["resource_accounting"]["recovery_totals"] == {
        "action_cache_hit_count": 1,
        "action_history_recovery_count": 1,
        "provider_result_replay_count": 0,
    }
    assert summary["per_target"][0]["terminal_disposition"] == (
        "completed_b4_open__no_target_rooted_route"
    )
    assert summary["per_target"][-1]["terminal_disposition"] == (
        "completed_b4_stock_closed"
    )

    markdown = _markdown(summary)
    assert "## Outcome accounting" in markdown
    assert "| Accounted targets" not in markdown
    assert "- Accounted targets: 5" in markdown
    assert "## Provider attempt outcomes" in markdown
    assert "| raw_nonempty_host_filtered | 1 |" in markdown
    assert "## Selected provider route to B4 funnel" in markdown
    assert "| stock_closed_selected | 2 |" in markdown
    assert "## Actionable result-loss records" in markdown
    assert "| raw_nonempty_host_filtered | 3 | 3 | 0 | 0 |" in markdown
    assert "It is not a serial B4 funnel" in markdown
    assert "## Resource and recovery accounting" in markdown
    assert "| provider_result_replay_count | 0 |" in markdown
    assert "| solved | completed | completed_b4_stock_closed |" in markdown


def test_panel_summary_hydrates_reaction_rejection_reasons(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "target": {"canonical_smiles": "CCO"},
                "stages": [
                    {
                        "stage": "campaign_action_unified_core_01",
                        "detail": {
                            "action": {"kind": "reaction_validate"},
                            "outcome": {
                                "handler_result": {
                                    "rejection_reason_counts": {
                                        "reaction_edit_budget_exceeded": 2
                                    }
                                }
                            },
                        },
                    }
                    ,
                    {
                        "stage": "campaign_action_unified_core_02",
                        "detail": {
                            "action": {"kind": "stock_audit"},
                            "outcome": {
                                "handler_result": {
                                    "selected_leaf_count": 5,
                                    "stock_closed_leaf_count": 2,
                                    "miss_count": 3,
                                    "remaining_pending_candidate_count": 0,
                                }
                            },
                        },
                    },
                    {
                        "stage": "chemenzy_route_lineage",
                        "detail": {
                            "disposition_counts": {
                                "canonical_edges_present_outside_complete_measured_route": 1
                            },
                            "routes": [
                                {
                                    "final_disposition": "canonical_edges_present_outside_complete_measured_route",
                                    "proposal_eligible": True,
                                    "host_portfolio_selected": True,
                                    "quarantined": False,
                                }
                            ],
                        },
                    },
                    {
                        "stage": "campaign_action_unified_core_03",
                        "detail": {
                            "action": {"kind": "recompute_route_closure"},
                            "cache_hit": True,
                            "recovered_from_action_history": True,
                            "outcome_pointer_recovered": True,
                            "outcome": {
                                "handler_result": {
                                    "provider_result_replayed": True
                                }
                            },
                        },
                    },
                    {
                        "stage": "campaign_action_unified_core_04",
                        "detail": {
                            "action": {"kind": "chemenzy_frontier_expand"},
                            "outcome": {"handler_result": {}},
                        },
                    },
                    {
                        "stage": "campaign_anytime_core",
                        "detail": {"termination": "no_action"},
                    },
                ]
                ,
                "candidate_provenance": {
                    "bound_provider_route_count": 2,
                    "first_loss_counts": {"stock_closure": 2},
                    "provider_route_records": [
                        {
                            "host_portfolio_selected": True,
                            "candidate_ids": ["candidate:1"],
                            "canonical_route_ids": ["route:1"],
                            "stock_closed_route_ids": ["route:1"],
                        },
                        {
                            "host_portfolio_selected": True,
                            "candidate_ids": ["candidate:2"],
                            "canonical_route_ids": ["route:2"],
                            "stock_closed_route_ids": ["route:2"],
                        },
                        {
                            "host_portfolio_selected": True,
                            "route_trace_id": "route-trace:open",
                            "raw_route_sha256": "a" * 64,
                            "normalized_route_sha256": "b" * 64,
                            "canonical_route_family_id": "route-family:open",
                            "candidate_ids": ["candidate:3"],
                            "canonical_edge_ids": ["edge:3"],
                            "canonical_route_ids": [],
                            "stock_closed_route_ids": [],
                            "first_loss_boundary": "canonical_materialization",
                            "final_disposition": (
                                "canonical_edges_present_outside_complete_measured_route"
                            ),
                        },
                    ],
                },
                "resource_envelope": {
                    "observed": {
                        "input_tokens": 40,
                        "output_tokens": 10,
                        "native_search_committed": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "chemenzy-v4-seed-result.json").write_text(
        json.dumps(
            {
                "ok": True,
                "target": "CCO",
                "raw_solved": True,
                "materialization_admission_solved": True,
                "n_results": 3,
                "time_s": 12.5,
                "routes": [
                    {"raw_backend_metadata": {"route_index": 5}},
                    {"raw_backend_metadata": {"route_index": 7}},
                    {"raw_backend_metadata": {"route_index": 12}},
                ],
                "search_status": {
                    "native_raw_n_routes": 8,
                    "native_search_found_n_routes": 9,
                },
                "raw_backend_metadata": {
                    "first_succ_time": 2.5,
                    "search_stop": {
                        "reason": "success_route_limit_reached",
                        "executed_iterations": 7,
                    },
                },
                "route_set_metrics": {
                    "cascade_verifier_gate": {
                        "input_routes": 8,
                        "kept_routes": 3,
                        "dropped_routes": 5,
                    },
                    "route_materialization_admission": {
                        "audited_route_count": 8,
                        "accepted_route_count": 7,
                        "rejected_route_count": 1,
                    },
                },
                "quarantined_routes": [
                    {
                        "metrics": {
                            "strict_stock_solve": True,
                            "cascade_verifier": {
                                "findings": [
                                    {"reason": "atom_balance_violation"}
                                ]
                            },
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    status = _hydrate_report_diagnostics(
        {
            "target_count": 1,
            "targets": {
                "target": {
                    "status": "completed",
                    "case_id": "case",
                    "report_path": str(report),
                }
            },
        }
    )

    summary = summarize_panel(status)

    assert summary["result_first"]["reaction_rejection_reason_counts"] == {
        "reaction_edit_budget_exceeded": 2
    }
    assert summary["result_first"]["stock_audit_totals"] == {
        "miss_count": 3,
        "remaining_pending_candidate_count": 0,
        "selected_leaf_count": 5,
        "stock_closed_leaf_count": 2,
        "stock_open_leaf_count": 3,
    }
    assert summary["result_first"]["provider_lineage_disposition_counts"] == {
        "canonical_edges_present_outside_complete_measured_route": 1
    }
    assert summary["result_first"]["provider_integration_loss_target_count"] == 1
    assert summary["result_first"]["provider_integration_loss_case_ids"] == ["case"]
    assert summary["result_first"]["provider_partial_lineage_target_count"] == 1
    assert summary["result_first"]["provider_partial_lineage_case_ids"] == ["case"]
    assert summary["result_first"]["provider_search"]["counts"] == {
        "attempt_count": 1,
        "executed_iterations": 7,
        "host_admitted_solved_attempt_count": 1,
        "host_solution_within_raw_prefix_16_attempt_count": 1,
        "host_solution_within_raw_prefix_8_attempt_count": 1,
        "native_attempt_count": 1,
        "native_found_route_count": 9,
        "native_host_admitted_solved_attempt_count": 1,
        "native_raw_route_count": 8,
        "native_raw_solved_attempt_count": 1,
        "output_route_count": 3,
        "raw_solved_attempt_count": 1,
        "rule_gate_dropped_route_count": 5,
        "rule_gate_input_route_count": 8,
        "rule_gate_kept_route_count": 3,
        "quarantined_route_count": 1,
        "atom_balance_only_quarantine_count": 1,
        "atom_balance_only_stock_closed_quarantine_count": 1,
    }
    assert summary["result_first"]["provider_search"]["stop_reasons"] == {
        "success_route_limit_reached": 1
    }
    assert summary["result_first"]["provider_search"]["provider_time_s"][
        "median"
    ] == 12.5
    assert summary["per_target"][0]["provider_search_attempts"][0][
        "first_output_raw_route_index"
    ] == 5
    assert summary["result_first"]["provider_search"][
        "post_first_success_time_s"
    ]["median"] == 10.0
    assert summary["result_first"][
        "provider_route_provenance_first_loss_counts"
    ] == {
        "stock_closure": 2
    }
    assert summary["result_first"][
        "selected_provider_route_b4_disposition_counts"
    ] == {"canonical_materialization_open": 1, "stock_closed": 2}
    loss_records = summary["result_first"]["result_loss_records"]
    assert loss_records["provider_non_success_attempt_count"] == 0
    assert loss_records["selected_provider_route_b4_open_count"] == 1
    assert loss_records["selected_provider_route_b4_open"][0] == {
        "candidate_count": 1,
        "canonical_edge_count": 1,
        "canonical_route_count": 0,
        "canonical_route_family_id": "route-family:open",
        "case_id": "case",
        "disposition": "canonical_materialization_open",
        "final_disposition": (
            "canonical_edges_present_outside_complete_measured_route"
        ),
        "first_loss_boundary": "canonical_materialization",
        "normalized_route_sha256": "b" * 64,
        "raw_route_sha256": "a" * 64,
        "report_path": str(report),
        "root_b4_open": True,
        "route_trace_id": "route-trace:open",
        "stock_closed_route_count": 0,
        "target_name": "target",
    }
    assert loss_records["root_b4_open_selected_provider_route_count"] == 1
    assert loss_records["root_b4_open_selected_provider_routes"] == [
        loss_records["selected_provider_route_b4_open"][0]
    ]
    assert summary["resource_accounting"]["observed_totals"] == {
        "input_tokens": 40,
        "native_search_committed": 2,
        "output_tokens": 10,
    }
    assert summary["resource_accounting"]["recovery_totals"] == {
        "action_cache_hit_count": 1,
        "action_history_recovery_count": 1,
        "outcome_pointer_recovery_count": 1,
        "provider_result_replay_count": 1,
    }
    assert summary["per_target"][0]["result_action_trace"] == {
        "campaign_termination": "no_action",
        "guided_after_first_route_closure": 1,
        "guided_before_first_route_closure": 0,
        "recompute_route_closure_count": 1,
    }


def test_b4_open_atom_balance_only_stock_route_is_counterfactual_not_scored() -> None:
    summary = summarize_panel(
        {
            "target_count": 1,
            "targets": {
                "target": {
                    "status": "completed",
                    "case_id": "soft-gate",
                    "gate_summary": {"B1": True, "B2": False, "B4": False},
                    "provider_search_attempts": [
                        {
                            "atom_balance_only_stock_closed_quarantine_count": 2,
                            "search_target_is_root": True,
                        }
                    ],
                }
            },
        }
    )

    assert summary["result_first"]["milestone_counts"]["B4"] == 0
    assert summary["result_first"][
        "atom_balance_soft_gate_counterfactual_case_ids"
    ] == ["soft-gate"]


def test_frontier_atom_balance_route_does_not_claim_root_counterfactual() -> None:
    summary = summarize_panel(
        {
            "target_count": 1,
            "targets": {
                "target": {
                    "status": "completed",
                    "case_id": "frontier-only",
                    "gate_summary": {"B1": True, "B2": False, "B4": False},
                    "provider_search_attempts": [
                        {
                            "atom_balance_only_stock_closed_quarantine_count": 1,
                            "search_target_is_root": False,
                            "kind": "guided",
                            "host_admitted_solved": True,
                        }
                    ],
                }
            },
        }
    )

    assert summary["result_first"][
        "atom_balance_soft_gate_counterfactual_case_ids"
    ] == []
    assert summary["result_first"][
        "atom_balance_frontier_signal_case_ids"
    ] == ["frontier-only"]
    assert summary["result_first"][
        "guided_frontier_solved_root_open_case_ids"
    ] == ["frontier-only"]


def test_panel_summary_hydrates_start_cohort_latency(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "stages": [
                    {
                        "stage": "global_campaign",
                        "detail": {
                            "start_cohort": {
                                "latency_audit": {
                                    "schema_version": (
                                        "campaign_action_cohort_latency_audit.v1"
                                    ),
                                    "applicable": True,
                                    "cohort_elapsed_s": 20.0,
                                    "chemenzy_first_proposal": {
                                        "nonempty_raw_proposal_observed": True,
                                        "elapsed_from_start_cohort_s": 8.0,
                                        "codex_peer_in_flight_at_chemenzy_completion": True,
                                        "peer_wait_excluded_s": 12.0,
                                    },
                                    "actions": [
                                        {
                                            "action_kind": "chemenzy_target_expand",
                                            "completed_offset_s": 8.0,
                                        },
                                        {
                                            "action_kind": "codex_global_architecture",
                                            "completed_offset_s": 20.0,
                                        },
                                    ],
                                }
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    status = _hydrate_report_diagnostics(
        {
            "target_count": 1,
            "targets": {
                "target": {
                    "status": "completed",
                    "case_id": "case",
                    "report_path": str(report),
                }
            },
        }
    )

    summary = summarize_panel(status)

    latency = summary["result_first"]["start_cohort_latency"]
    assert latency["counts"] == {
        "applicable_target_count": 1,
        "chemenzy_completed_before_codex_target_count": 1,
        "nonempty_chemenzy_proposal_target_count": 1,
    }
    assert latency["cohort_elapsed_s"]["median"] == 20.0
    assert latency["chemenzy_completed_s"]["median"] == 8.0
    assert latency["codex_completed_s"]["median"] == 20.0
    assert latency["chemenzy_peer_wait_s"]["median"] == 12.0


def test_partial_provider_lineage_is_not_result_loss_after_b4() -> None:
    summary = summarize_panel(
        {
            "target_count": 1,
            "targets": {
                "target": {
                    "status": "completed",
                    "case_id": "closed",
                    "gate_summary": {"B1": True, "B2": False, "B4": True},
                    "provider_lineage_disposition_counts": {
                        "canonical_edges_present_outside_complete_measured_route": 1,
                        "stock_closed": 3,
                    },
                    "provider_eligible_incomplete_lineage_count": 0,
                }
            },
        }
    )

    assert summary["result_first"]["provider_partial_lineage_case_ids"] == [
        "closed"
    ]
    assert summary["result_first"]["provider_integration_loss_case_ids"] == []


def test_explicit_provider_topology_loss_is_not_hidden_by_another_b4_route() -> None:
    summary = summarize_panel(
        {
            "target_count": 1,
            "targets": {
                "target": {
                    "status": "completed",
                    "case_id": "partially-imported",
                    "gate_summary": {"B1": True, "B2": False, "B4": True},
                    "provider_topology_conservation_failure_count": 1,
                }
            },
        }
    )

    assert summary["result_first"]["provider_topology_loss_case_ids"] == [
        "partially-imported"
    ]
    assert summary["result_first"]["provider_integration_loss_case_ids"] == [
        "partially-imported"
    ]


def test_quarantined_partial_lineage_is_not_integration_loss_without_b4() -> None:
    summary = summarize_panel(
        {
            "target_count": 1,
            "targets": {
                "target": {
                    "status": "completed",
                    "case_id": "rejected",
                    "gate_summary": {"B1": True, "B2": False, "B4": False},
                    "provider_lineage_disposition_counts": {
                        "canonical_edges_present_outside_complete_measured_route": 1,
                    },
                    "provider_eligible_incomplete_lineage_count": 0,
                }
            },
        }
    )

    assert summary["result_first"]["provider_partial_lineage_case_ids"] == [
        "rejected"
    ]
    assert summary["result_first"]["provider_integration_loss_case_ids"] == []

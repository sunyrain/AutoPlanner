import json
from datetime import datetime, timedelta, timezone

from cascade_planner.baselines.semisynthesis_rescue import ACETIC_ANHYDRIDE, DEACETYLBUFOTALIN, DMAP
from scripts.run_bufotalin_12h_iteration import BUFOTALIN_TARGET
from scripts.write_bufotalin_status_snapshot import build_status_snapshot, _is_relevant_runtime_process


def test_build_status_snapshot_reports_audit_and_final_candidates(tmp_path):
    _write_payload(tmp_path / "anchor")
    _write_final_candidates(tmp_path)
    started = datetime(2026, 5, 25, tzinfo=timezone.utc)
    finished = started + timedelta(hours=12, minutes=1)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"target": BUFOTALIN_TARGET, "running": False}),
        encoding="utf-8",
    )
    (tmp_path / "runner_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "start", "started_at": started.isoformat(), "target": BUFOTALIN_TARGET}),
                json.dumps({"event": "finish", "time": finished.isoformat()}),
            ]
        ),
        encoding="utf-8",
    )

    snapshot = build_status_snapshot(tmp_path)

    assert snapshot["headline"]["high_confidence_final_routes"] == 1
    assert snapshot["headline"]["selected_final_routes"] == 3
    assert snapshot["headline"]["failed_checks"] == ["has_native_search_success"]
    assert snapshot["headline"]["early_stop_review_ready"] is False
    assert snapshot["headline"]["status_label"] == "stopped_incomplete"


def test_build_status_snapshot_marks_review_ready_when_only_time_gate_failed(tmp_path):
    _write_payload(tmp_path / "anchor")
    _write_payload(tmp_path / "cycle_001_native", native_raw_n_routes=1, source_supported=False)
    _write_final_candidates(tmp_path)
    _write_proposal_gate_summary(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"target": BUFOTALIN_TARGET, "running": True}),
        encoding="utf-8",
    )
    (tmp_path / "runner_events.jsonl").write_text(
        json.dumps(
            {
                "event": "start",
                "started_at": datetime(2026, 5, 25, tzinfo=timezone.utc).isoformat(),
                "target": BUFOTALIN_TARGET,
            }
        ),
        encoding="utf-8",
    )

    snapshot = build_status_snapshot(tmp_path)

    assert snapshot["headline"]["complete"] is False
    assert snapshot["headline"]["failed_checks"] == ["finished", "ran_min_hours"]
    assert snapshot["headline"]["early_stop_review_ready"] is True
    assert snapshot["headline"]["status_label"] == "early_stop_review_ready"
    assert snapshot["early_stop"]["only_time_gate_failed"] is True
    assert snapshot["early_stop"]["no_active_workers"] is True
    assert snapshot["headline"]["proposal_gate_dropped_routes"] == 2
    assert snapshot["proposal_gate"]["reason_counts"]["unsupported_biosynthetic_prenyl_terminal"] == 2


def test_runtime_process_detection_ignores_pytest_command_mentions():
    assert _is_relevant_runtime_process("python scripts/run_bufotalin_12h_iteration.py --hours 12")
    assert _is_relevant_runtime_process(
        "/root/miniconda3/bin/python /root/autodl-tmp/AutoPlanner/scripts/run_bufotalin_12h_iteration.py --cycle-config x"
    )
    assert not _is_relevant_runtime_process(
        "python -m pytest tests/test_run_bufotalin_12h_iteration.py tests/test_write_bufotalin_status_snapshot.py -q"
    )


def _write_payload(cycle_dir, *, native_raw_n_routes=0, source_supported=True):
    cycle_dir.mkdir(parents=True)
    figures = cycle_dir / "figures"
    figures.mkdir()
    (figures / "scheme_route_01.svg").write_text("<svg/>", encoding="utf-8")
    (figures / "scheme_route_01.pdf").write_text("%PDF", encoding="utf-8")
    (figures / "manifest.json").write_text(
        json.dumps({"figures": [{"svg": "scheme_route_01.svg", "pdf": "scheme_route_01.pdf"}]}),
        encoding="utf-8",
    )
    (cycle_dir / "web_payload.json").write_text(
        json.dumps(
            {
                "target_smiles": BUFOTALIN_TARGET,
                "ok": True,
                "n_results": 1,
                "search_status": {
                    "status": "solved",
                    "native_raw_n_routes": native_raw_n_routes,
                    "semisynthesis_rescue_n_routes": 1 if source_supported else 0,
                },
                "route_set_metrics": {
                    "template_relevance_top_level_probe": {
                        "hit_expected_precursor": True,
                        "returned": 1,
                    }
                },
                "routes": [
                    {
                        "steps": [
                            {
                                "main_reactant": DEACETYLBUFOTALIN,
                                "aux_reactants": [ACETIC_ANHYDRIDE],
                                "condition_predictions": [
                                    {"Reagent": ACETIC_ANHYDRIDE, "Catalyst": DMAP}
                                ],
                            }
                        ],
                        "metrics": {
                            "route_solved": True,
                            "semisynthesis_anchor": source_supported,
                            "source_supported_semisynthesis": source_supported,
                            "cascade_verifier": {"feasible": True},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_final_candidates(root):
    final = root / "final_candidates"
    figures = final / "figures"
    figures.mkdir(parents=True)
    for idx in range(1, 4):
        (figures / f"scheme_route_{idx:02d}.svg").write_text("<svg/>", encoding="utf-8")
        (figures / f"scheme_route_{idx:02d}.pdf").write_text("%PDF", encoding="utf-8")
    (figures / "manifest.json").write_text(
        json.dumps(
            {
                "figures": [
                    {"svg": f"scheme_route_{idx:02d}.svg", "pdf": f"scheme_route_{idx:02d}.pdf"}
                    for idx in range(1, 4)
                ]
            }
        ),
        encoding="utf-8",
    )
    (final / "final_candidates.md").write_text("# final", encoding="utf-8")
    (final / "final_candidates.json").write_text(
        json.dumps(
            {
                "high_confidence_count": 1,
                "stitched_review_only_count": 1,
                "native_review_only_count": 1,
                "selected_count": 3,
                "excluded_route_count": 0,
                "selected": [
                    {
                        "confidence_tier": "high_confidence_source_supported",
                        "presentation_ready": True,
                    },
                    {
                        "confidence_tier": "stitched_semisynthesis_upstream_review_only",
                        "presentation_ready": False,
                        "n_steps": 4,
                        "warnings": ["rcr_condition_prediction_only"],
                    },
                    {
                        "confidence_tier": "native_model_candidate_review_only",
                        "presentation_ready": False,
                        "n_steps": 3,
                        "warnings": ["low_condition_prediction_score", "rcr_condition_prediction_only"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (final / "final_candidates_payload.json").write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "final_candidate": {
                            "confidence_tier": "high_confidence_source_supported",
                            "target_terminal": False,
                            "source_supported_semisynthesis": True,
                        },
                        "steps": [
                            {
                                "main_reactant": DEACETYLBUFOTALIN,
                                "aux_reactants": [ACETIC_ANHYDRIDE],
                                "condition_predictions": [
                                    {"Reagent": ACETIC_ANHYDRIDE, "Catalyst": DMAP}
                                ],
                            }
                        ],
                    },
                    _review_route("stitched_semisynthesis_upstream_review_only", n_steps=4),
                    _review_route("native_model_candidate_review_only", n_steps=3),
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_proposal_gate_summary(root):
    (root / "cycle_proposal_gate_retrofit_summary.json").write_text(
        json.dumps(
            {
                "mode": "hard_reject",
                "payload_count": 2,
                "input_routes": 5,
                "kept_routes": 3,
                "dropped_routes": 2,
                "reason_counts": {"unsupported_biosynthetic_prenyl_terminal": 2},
            }
        ),
        encoding="utf-8",
    )


def _review_route(confidence_tier: str, *, n_steps: int):
    return {
        "n_steps": n_steps,
        "steps": [
            {
                "main_reactant": f"C{idx}",
                "condition_predictions": [
                    {
                        "condition_label": "RCR model prediction",
                        "Score": 0.2,
                    }
                ],
            }
            for idx in range(n_steps)
        ],
        "final_candidate": {
            "confidence_tier": confidence_tier,
            "presentation_ready": False,
            "target_terminal": False,
            "exclusion_reasons": [],
            "warnings": ["rcr_condition_prediction_only"],
        },
    }

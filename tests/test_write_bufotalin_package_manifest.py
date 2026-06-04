import json
from datetime import datetime, timezone

from cascade_planner.baselines.semisynthesis_rescue import ACETIC_ANHYDRIDE, DEACETYLBUFOTALIN, DMAP
from scripts.run_bufotalin_12h_iteration import BUFOTALIN_TARGET
from scripts.write_bufotalin_package_manifest import build_package_manifest


def test_build_package_manifest_reports_core_files_and_review_ready(tmp_path):
    _write_core_docs(tmp_path)
    _write_payload(tmp_path / "anchor", source_supported=True, native_raw_n_routes=0)
    _write_payload(tmp_path / "cycle_001_native", source_supported=False, native_raw_n_routes=1)
    _write_final_candidates(tmp_path)
    _write_stopped_manifest(tmp_path)

    manifest = build_package_manifest(tmp_path)

    assert manifest["schema_version"] == "bufotalin_package_manifest.v1"
    assert manifest["target"] == BUFOTALIN_TARGET
    assert manifest["status"]["strict_12h_complete"] is False
    assert manifest["status"]["early_stop_review_ready"] is True
    assert manifest["status"]["stop_reason"] == "user_cancelled"
    assert manifest["conclusion"]["high_confidence_route_count"] == 1
    assert manifest["conclusion"]["review_only_route_count"] == 1
    assert manifest["conclusion"]["native_routes_position"] == "review_only"
    assert "Deacetylbufotalin" in manifest["conclusion"]["main_route_summary"]
    assert manifest["figures"]["figure_count"] == 3
    assert manifest["all_core_files_present"] is True
    assert manifest["audits"]["final_candidate_quality"]["passed"] is True
    assert manifest["proposal_gate"]["available"] is True
    assert manifest["proposal_gate"]["dropped_routes"] == 2
    assert manifest["proposal_gate"]["repaired_routes"] == 1
    assert manifest["proposal_gate"]["repair_reason_counts"]["late_stage_tbs_silylation"] == 1
    assert manifest["proposal_frontiers"]["available"] is True
    assert manifest["proposal_frontiers"]["unique_frontiers"] == 1
    assert manifest["frontier_proposal_probe"]["available"] is True
    assert manifest["frontier_proposal_probe"]["proposal_count"] == 4
    assert manifest["frontier_proposal_probe"]["gate_keep_count"] == 1


def _write_core_docs(root):
    for name in [
        "README.md",
        "completion_gap_report.md",
        "early_stop_result_report.md",
        "early_stop_review_audit.json",
        "final_candidate_quality_audit.json",
        "cycle_proposal_gate_retrofit_summary.json",
        "proposal_frontier_analysis.json",
        "frontier_proposal_probe.json",
        "status_snapshot.json",
    ]:
        (root / name).write_text("{}" if name.endswith(".json") else "# doc", encoding="utf-8")
    (root / "cycle_proposal_gate_retrofit_summary.json").write_text(
        json.dumps(
            {
                "mode": "hard_reject",
                "payload_count": 2,
                "input_routes": 5,
                "kept_routes": 3,
                "dropped_routes": 2,
                "repaired_routes": 1,
                "repair_reason_counts": {"late_stage_tbs_silylation": 1},
                "reason_counts": {"unsupported_biosynthetic_prenyl_terminal": 2},
            }
        ),
        encoding="utf-8",
    )
    (root / "frontier_proposal_probe.json").write_text(
        json.dumps(
            {
                "summary": {
                    "frontier_count": 2,
                    "proposal_count": 4,
                    "gate_keep_count": 1,
                    "gate_reject_count": 3,
                    "elapsed_s": 0.25,
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "proposal_frontier_analysis.json").write_text(
        json.dumps(
            {
                "summary": {
                    "dropped_rows_with_frontier": 2,
                    "unique_frontiers": 1,
                    "complex_core_frontier_count": 1,
                    "unsupported_prenyl_frontier_count": 0,
                },
                "top_frontiers": [
                    {
                        "smiles": "CCO",
                        "count": 2,
                        "profile": {"formula": "C2O1"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_stopped_manifest(root):
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "target": BUFOTALIN_TARGET,
                "running": False,
                "stopped": True,
                "stop_reason": "user_cancelled",
                "stopped_at": datetime(2026, 5, 25, 4, 30, tzinfo=timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    (root / "runner_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "start",
                        "started_at": datetime(2026, 5, 25, tzinfo=timezone.utc).isoformat(),
                        "target": BUFOTALIN_TARGET,
                    }
                ),
                json.dumps(
                    {
                        "event": "user_stop",
                        "time": datetime(2026, 5, 25, 4, 30, tzinfo=timezone.utc).isoformat(),
                        "reason": "user_cancelled",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )


def _write_payload(cycle_dir, *, source_supported: bool, native_raw_n_routes: int):
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
                                    {"Temperature": 25, "Reagent": ACETIC_ANHYDRIDE, "Catalyst": DMAP}
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
    final_dir = root / "final_candidates"
    figures = final_dir / "figures"
    figures.mkdir(parents=True)
    (figures / "index.html").write_text("<html></html>", encoding="utf-8")
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
    (final_dir / "final_candidates.md").write_text("# final", encoding="utf-8")
    (final_dir / "final_candidates.json").write_text(
        json.dumps(
            {
                "high_confidence_count": 1,
                "stitched_review_only_count": 1,
                "native_review_only_count": 1,
                "selected_count": 3,
                "excluded_route_count": 3,
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
    (final_dir / "final_candidates_payload.json").write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "steps": [
                            {
                                "main_reactant": DEACETYLBUFOTALIN,
                                "aux_reactants": [ACETIC_ANHYDRIDE],
                                "condition_predictions": [
                                    {"Temperature": 25, "Reagent": ACETIC_ANHYDRIDE, "Catalyst": DMAP}
                                ],
                            }
                        ],
                        "final_candidate": {
                            "confidence_tier": "high_confidence_source_supported",
                            "presentation_ready": True,
                            "target_terminal": False,
                            "exclusion_reasons": [],
                            "source_supported_semisynthesis": True,
                        },
                    },
                    _review_route("stitched_semisynthesis_upstream_review_only", n_steps=4),
                    _review_route("native_model_candidate_review_only", n_steps=3),
                ]
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

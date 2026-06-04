import json
from datetime import datetime, timedelta, timezone

from cascade_planner.baselines.semisynthesis_rescue import ACETIC_ANHYDRIDE, DEACETYLBUFOTALIN, DMAP
from scripts.audit_bufotalin_early_stop_review import audit_early_stop_review
from scripts.run_bufotalin_12h_iteration import BUFOTALIN_TARGET


def test_audit_early_stop_review_accepts_stopped_review_ready_package(tmp_path):
    _write_payload(tmp_path / "anchor", source_supported=True, native_raw_n_routes=0)
    _write_payload(tmp_path / "cycle_001_native", source_supported=False, native_raw_n_routes=1)
    _write_final_candidates(tmp_path)
    _write_started_manifest(tmp_path)

    report = audit_early_stop_review(tmp_path)

    assert report["review_ready"]
    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert checks["strict_12h_goal_not_claimed"]
    assert checks["only_time_gate_failed"]
    assert checks["has_high_confidence_final_route"]


def test_audit_early_stop_review_rejects_strictly_complete_run(tmp_path):
    _write_payload(tmp_path / "anchor", source_supported=True, native_raw_n_routes=0)
    _write_payload(tmp_path / "cycle_001_native", source_supported=False, native_raw_n_routes=1)
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

    report = audit_early_stop_review(tmp_path)

    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert not report["review_ready"]
    assert not checks["strict_12h_goal_not_claimed"]


def test_audit_early_stop_review_requires_native_search_evidence(tmp_path):
    _write_payload(tmp_path / "anchor", source_supported=True, native_raw_n_routes=0)
    _write_final_candidates(tmp_path)
    _write_started_manifest(tmp_path)

    report = audit_early_stop_review(tmp_path)

    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert not report["review_ready"]
    assert not checks["has_native_search_evidence"]


def test_audit_early_stop_review_requires_manifest_user_stop(tmp_path):
    _write_payload(tmp_path / "anchor", source_supported=True, native_raw_n_routes=0)
    _write_payload(tmp_path / "cycle_001_native", source_supported=False, native_raw_n_routes=1)
    _write_final_candidates(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"target": BUFOTALIN_TARGET, "running": False}),
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

    report = audit_early_stop_review(tmp_path)

    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert not report["review_ready"]
    assert not checks["manifest_records_user_stop"]


def _write_started_manifest(root):
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

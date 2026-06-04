import json
from datetime import datetime, timedelta, timezone

from scripts.audit_bufotalin_12h_completion import audit_completion
from scripts.run_bufotalin_12h_iteration import BUFOTALIN_TARGET
from cascade_planner.baselines.semisynthesis_rescue import ACETIC_ANHYDRIDE, DEACETYLBUFOTALIN, DMAP


def test_audit_completion_requires_finish_and_min_hours(tmp_path):
    started = datetime(2026, 5, 25, tzinfo=timezone.utc)
    finished = started + timedelta(hours=12, minutes=1)
    _write_cycle_payload(tmp_path / "anchor", source_supported=True, feasible=True)
    _write_cycle_payload(tmp_path / "cycle_001", source_supported=False, feasible=True)
    _write_final_candidates(tmp_path)
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

    report = audit_completion(tmp_path)

    assert report["complete"]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["ran_min_hours"]["passed"]
    assert "12.017 h >= required 12.000 h" in checks["ran_min_hours"]["evidence"]


def test_audit_completion_fails_running_short_run(tmp_path):
    _write_cycle_payload(tmp_path / "anchor", source_supported=True, feasible=True)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"target": BUFOTALIN_TARGET, "running": True}),
        encoding="utf-8",
    )
    (tmp_path / "runner_events.jsonl").write_text(
        json.dumps({"event": "start", "started_at": datetime(2026, 5, 25, tzinfo=timezone.utc).isoformat(), "target": BUFOTALIN_TARGET}),
        encoding="utf-8",
    )

    report = audit_completion(tmp_path)

    assert not report["complete"]
    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert not checks["finished"]
    assert not checks["ran_min_hours"]


def test_audit_completion_reports_elapsed_hours_for_user_stop(tmp_path):
    started = datetime(2026, 5, 25, tzinfo=timezone.utc)
    stopped = started + timedelta(hours=4, minutes=30)
    _write_cycle_payload(tmp_path / "anchor", source_supported=True, feasible=True)
    _write_final_candidates(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"target": BUFOTALIN_TARGET, "running": False, "stop_reason": "user_cancelled"}),
        encoding="utf-8",
    )
    (tmp_path / "runner_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "start", "started_at": started.isoformat(), "target": BUFOTALIN_TARGET}),
                json.dumps({"event": "user_stop", "time": stopped.isoformat(), "reason": "user_cancelled"}),
            ]
        ),
        encoding="utf-8",
    )

    report = audit_completion(tmp_path)

    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert not report["complete"]
    assert report["elapsed_hours"] == 4.5
    assert not checks["finished"]
    assert not checks["ran_min_hours"]
    evidence = {check["name"]: check["evidence"] for check in report["checks"]}
    assert "4.500 h < required 12.000 h" in evidence["ran_min_hours"]


def test_audit_completion_tolerates_exploratory_timeout_with_supported_fallback(tmp_path):
    started = datetime(2026, 5, 25, tzinfo=timezone.utc)
    finished = started + timedelta(hours=12, minutes=1)
    _write_cycle_payload(tmp_path / "anchor", source_supported=True, feasible=True)
    _write_cycle_payload(tmp_path / "cycle_001", source_supported=False, feasible=True)
    _write_cycle_payload(
        tmp_path / "cycle_002_timeout",
        source_supported=True,
        feasible=True,
        native_raw_n_routes=0,
        failures=["cycle_worker_timeout"],
    )
    _write_final_candidates(tmp_path)
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

    report = audit_completion(tmp_path)

    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert report["complete"]
    assert checks["no_completed_payload_failures"]


def test_audit_completion_rejects_native_timeout_failure(tmp_path):
    started = datetime(2026, 5, 25, tzinfo=timezone.utc)
    finished = started + timedelta(hours=12, minutes=1)
    _write_cycle_payload(tmp_path / "anchor", source_supported=True, feasible=True)
    _write_cycle_payload(
        tmp_path / "cycle_001_timeout",
        source_supported=False,
        feasible=True,
        native_raw_n_routes=1,
        failures=["cycle_worker_timeout"],
    )
    _write_final_candidates(tmp_path)
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

    report = audit_completion(tmp_path)

    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert not report["complete"]
    assert not checks["no_completed_payload_failures"]


def test_audit_completion_rejects_rendered_native_without_conditions(tmp_path):
    started = datetime(2026, 5, 25, tzinfo=timezone.utc)
    finished = started + timedelta(hours=12, minutes=1)
    _write_cycle_payload(
        tmp_path / "cycle_001",
        source_supported=False,
        feasible=True,
        with_conditions=False,
    )
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

    report = audit_completion(tmp_path)

    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert not report["complete"]
    assert not checks["has_conditioned_renderable_route"]
    assert not checks["figures_are_feasible_or_supported"]


def _write_cycle_payload(
    cycle_dir,
    *,
    source_supported: bool,
    feasible: bool,
    with_conditions: bool = True,
    native_raw_n_routes: int | None = None,
    failures: list[str] | None = None,
):
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
                    "native_raw_n_routes": native_raw_n_routes if native_raw_n_routes is not None else (1 if not source_supported else 0),
                    "semisynthesis_rescue_n_routes": 1 if source_supported else 0,
                },
                "backend_failures": [
                    {"category": failure, "message": failure}
                    for failure in failures or []
                ],
                "route_set_metrics": {
                    "template_relevance_top_level_probe": {
                        "hit_expected_precursor": True,
                        "returned": 1,
                    }
                },
                "routes": [
                    {
                        "steps": [{"reaction_smiles": "CCO>>CC=O"}],
                        "steps": [
                    {
                        "main_reactant": DEACETYLBUFOTALIN,
                        "aux_reactants": [ACETIC_ANHYDRIDE],
                        "reaction_smiles": "CCO>>CC=O",
                        "condition_predictions": [
                            {
                                "Temperature": 25,
                                "Reagent": ACETIC_ANHYDRIDE,
                                "Catalyst": DMAP,
                            }
                        ]
                        if with_conditions
                        else [],
                    }
                        ],
                        "metrics": {
                            "route_solved": True,
                            "semisynthesis_anchor": source_supported,
                            "source_supported_semisynthesis": source_supported,
                            "cascade_verifier": {"feasible": feasible},
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
                                    {
                                        "Temperature": 25,
                                        "Reagent": ACETIC_ANHYDRIDE,
                                        "Catalyst": DMAP,
                                    }
                                ],
                            }
                        ],
                        "final_candidate": {
                            "confidence_tier": "high_confidence_source_supported",
                            "presentation_ready": True,
                            "target_terminal": False,
                            "exclusion_reasons": [],
                            "source_supported_semisynthesis": True,
                        }
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

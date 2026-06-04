import json
from datetime import datetime, timedelta, timezone

from scripts.run_bufotalin_12h_iteration import BUFOTALIN_TARGET
from scripts.watch_bufotalin_run_completion import watch_and_finalize
from cascade_planner.baselines.semisynthesis_rescue import ACETIC_ANHYDRIDE, DEACETYLBUFOTALIN, DMAP


def test_watch_and_finalize_runs_export_and_audit_for_exited_pid(tmp_path):
    started = datetime(2026, 5, 25, tzinfo=timezone.utc)
    finished = started + timedelta(hours=12, minutes=1)
    _write_payload(tmp_path / "anchor", route_kind="source_supported")
    _write_payload(tmp_path / "cycle_001_native", route_kind="native")
    _write_payload(tmp_path / "cycle_002_stitched", route_kind="stitched")
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

    report = watch_and_finalize(tmp_path, pid=-1, poll_s=1, min_hours=12)

    assert report["final_export_returncode"] == 0
    assert report["final_quality_returncode"] == 0
    assert report["audit_returncode"] == 0
    assert report["completed"]
    assert (tmp_path / "final_candidates" / "final_candidates.json").exists()
    assert (tmp_path / "final_candidate_quality_audit.json").exists()


def _write_payload(cycle_dir, *, route_kind: str):
    cycle_dir.mkdir(parents=True)
    figures = cycle_dir / "figures"
    figures.mkdir()
    (figures / "scheme_route_01.svg").write_text("<svg/>", encoding="utf-8")
    (figures / "scheme_route_01.pdf").write_text("%PDF", encoding="utf-8")
    (figures / "manifest.json").write_text(
        json.dumps({"figures": [{"svg": "scheme_route_01.svg", "pdf": "scheme_route_01.pdf"}]}),
        encoding="utf-8",
    )
    source_supported = route_kind == "source_supported"
    native = route_kind == "native"
    stitched = route_kind == "stitched"
    steps = _route_steps(route_kind, n_steps=1 if source_supported else 4 if stitched else 3)
    (cycle_dir / "web_payload.json").write_text(
        json.dumps(
            {
                "target_smiles": BUFOTALIN_TARGET,
                "ok": True,
                "n_results": 1,
                "search_status": {
                    "status": "solved",
                    "native_raw_n_routes": 0 if source_supported else 1,
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
                        "n_steps": len(steps),
                        "score": 0.9,
                        "steps": steps,
                        "metrics": {
                            "route_solved": True,
                            "semisynthesis_anchor": source_supported,
                            "source_supported_semisynthesis": source_supported,
                            "stitched_semisynthesis": stitched,
                            "native_returned_route": native,
                            "terminal_reactants": ["CCO", ACETIC_ANHYDRIDE],
                            "cascade_verifier": {"feasible": True},
                        },
                        "raw_backend_metadata": {
                            **({"route_class_hint": "stitched_semisynthesis_upstream"} if stitched else {}),
                            "advanced_precursor_record": {
                                "name": "Deacetylbufotalin",
                                "cas": "465-19-0",
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _route_steps(route_kind: str, *, n_steps: int):
    if route_kind == "source_supported":
        return [
            {
                "reaction_smiles": f"{DEACETYLBUFOTALIN}.{ACETIC_ANHYDRIDE}>>{BUFOTALIN_TARGET}",
                "main_reactant": DEACETYLBUFOTALIN,
                "aux_reactants": [ACETIC_ANHYDRIDE],
                "condition_predictions": [
                    {
                        "condition_label": "Ac2O, DMAP",
                        "Reagent": ACETIC_ANHYDRIDE,
                        "Catalyst": DMAP,
                        "Temperature": 25,
                    }
                ],
            }
        ]
    return [
        {
            "reaction_smiles": f"{route_kind}_{idx}>>{route_kind}_{idx + 1}",
            "main_reactant": f"{route_kind}_{idx}",
            "aux_reactants": [],
            "condition_predictions": [
                {
                    "condition_label": "RCR model prediction",
                    "Score": 0.2,
                }
            ],
        }
        for idx in range(n_steps)
    ]

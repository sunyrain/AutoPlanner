import json

import scripts.summarize_bufotalin_iteration as summarize_module
from scripts.summarize_bufotalin_iteration import summarize_iteration_root


def test_summarize_iteration_reports_running_progress_and_quality_counts(tmp_path, monkeypatch):
    anchor = tmp_path / "anchor"
    figures = anchor / "figures"
    figures.mkdir(parents=True)
    (figures / "manifest.json").write_text(
        json.dumps({"figures": [{"svg": "scheme_route_01.svg", "pdf": "scheme_route_01.pdf"}]}),
        encoding="utf-8",
    )
    (anchor / "web_payload.json").write_text(
        json.dumps(
            {
                "ok": True,
                "n_results": 1,
                "search_status": {
                    "status": "solved",
                    "native_raw_n_routes": 0,
                    "semisynthesis_rescue_n_routes": 1,
                },
                "route_set_metrics": {
                    "template_relevance_top_level_probe": {
                        "hit_expected_precursor": True,
                        "returned": 9,
                    }
                },
                "routes": [
                    {
                        "steps": [{"condition_predictions": [{"Temperature": 25}]}],
                        "metrics": {
                            "route_solved": True,
                            "semisynthesis_anchor": True,
                            "source_supported_semisynthesis": True,
                            "cascade_verifier": {"feasible": True},
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cycle = tmp_path / "cycle_001_default"
    cycle.mkdir()
    (cycle / "cycle_config.json").write_text("{}", encoding="utf-8")
    (cycle / "worker.log").write_text(
        "\r 12%|#2        | 24/200 [01:09<04:46,  1.63s/it]",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        summarize_module,
        "_active_process_commands",
        lambda: [f"python scripts/run_bufotalin_12h_iteration.py --cycle-output {cycle}"],
    )

    summary = summarize_iteration_root(tmp_path)

    assert summary["completed_payload_count"] == 1
    assert summary["running_cycle_count"] == 1
    assert summary["cascade_verifier_feasible_payloads"] == 1
    assert summary["template_relevance_probe_hit_payloads"] == 1
    assert summary["condition_complete_payloads"] == 1
    assert summary["renderable_conditioned_payloads"] == 1
    assert summary["figure_svg_count"] == 1
    running = [row for row in summary["rows"] if row["cycle"] == "cycle_001_default"][0]
    assert running["worker_progress"]["current_iteration"] == 24
    assert running["worker_progress"]["total_iterations"] == 200


def test_summarize_iteration_reports_stopped_progress_without_active_process(tmp_path, monkeypatch):
    cycle = tmp_path / "cycle_001_default"
    cycle.mkdir()
    (cycle / "cycle_config.json").write_text("{}", encoding="utf-8")
    (cycle / "worker.log").write_text(
        "\r 12%|#2        | 24/200 [01:09<04:46,  1.63s/it]",
        encoding="utf-8",
    )
    monkeypatch.setattr(summarize_module, "_active_process_commands", lambda: [])

    summary = summarize_iteration_root(tmp_path)

    assert summary["running_cycle_count"] == 0
    stopped = [row for row in summary["rows"] if row["cycle"] == "cycle_001_default"][0]
    assert stopped["status"] == "stopped"
    assert stopped["worker_progress"]["current_iteration"] == 24

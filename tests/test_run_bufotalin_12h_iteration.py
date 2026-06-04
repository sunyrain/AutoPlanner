from scripts.run_bufotalin_12h_iteration import (
    BUFOTALIN_TARGET,
    BUFOTALIN_MAINLINE_ONE_STEP_MODELS,
    ENABLE_CONDITION_PREDICTION_FLAG,
    _attach_template_relevance_probe,
    _adaptive_cycle_config,
    _apply_cycle_proposal_gate,
    _backfill_display_route_conditions,
    _condition_prediction_enabled_for_worker,
    _cycle_configs,
    _finalize_result_package,
    _limit_result_routes,
    _prepare_runtime_root,
    _semisynthesis_anchor_precursors,
)
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteCandidate
import os
from pathlib import Path


def test_bufotalin_iteration_prioritizes_fast_default_cycles_before_diagnostics():
    cycles = _cycle_configs(BUFOTALIN_TARGET)

    assert cycles[0]["name"].startswith("upstream_first_d16_i200_k100_upstream_mainline")
    assert cycles[0]["config"].one_step_models == list(BUFOTALIN_MAINLINE_ONE_STEP_MODELS)
    assert cycles[1]["name"].startswith("upstream_first_d16_i200_k100_upstream_mainline")
    assert cycles[2]["name"].startswith("upstream_first_d16_i200_k100_upstream_mainline")
    assert cycles[3]["name"].startswith("d20_i200_k100_default")
    assert all("template_local_diagnostic" not in cycle["name"] for cycle in cycles)
    assert all("ensemble_local_diagnostic" not in cycle["name"] for cycle in cycles)
    assert all(
        cycle["config"].one_step_models == list(BUFOTALIN_MAINLINE_ONE_STEP_MODELS)
        for cycle in cycles
    )
    assert "template_relevance.bkms_metabolic" in cycles[0]["config"].one_step_models
    assert "template_relevance.pistachio_ringbreaker" in cycles[0]["config"].one_step_models


def test_template_relevance_probe_records_expected_precursor_hit():
    class FakeProvider:
        def __init__(self, **_):
            pass

        def predict(self, target, top_k=10):
            return [
                {
                    "rank": 1,
                    "score": 0.9,
                    "main_reactant": "CCO",
                    "source": "template_relevance",
                    "model_full_name": "template_relevance.bkms_metabolic",
                    "aux_reactants": [],
                }
            ]

    anchor_step = type("Step", (), {"stock_status": {"CCO": False}})()
    anchor_route = type("Route", (), {"raw_backend_metadata": {"rescue_type": "late_stage"}, "steps": [anchor_step]})()
    payload = {}

    import scripts.run_bufotalin_12h_iteration as runner

    original = runner.ChemEnzyOneStepProposalProvider
    try:
        runner.ChemEnzyOneStepProposalProvider = FakeProvider
        report = _attach_template_relevance_probe(
            payload,
            target="CCCO",
            anchor_routes=[anchor_route],
            vendor_root="vendor/ChemEnzyRetroPlanner",
            gpu=-1,
            top_k=1,
        )
    finally:
        runner.ChemEnzyOneStepProposalProvider = original

    assert report["hit_expected_precursor"]
    assert payload["route_set_metrics"]["template_relevance_top_level_probe"]["returned"] == 1


def test_semisynthesis_anchor_precursors_include_source_supported_reactants():
    metadata = {"semisynthesis_rescue": {"forward_reagent": "CC(=O)OC(C)=O"}}
    anchor_step = type(
        "Step",
        (),
        {
            "stock_status": {"C[C@H](O)C": True, "CC(=O)OC(C)=O": True},
            "reactant_smiles": ["C[C@H](O)C", "CC(=O)OC(C)=O"],
            "raw_backend_metadata": metadata,
        },
    )()
    anchor_route = type("Route", (), {"raw_backend_metadata": {"rescue_type": "late_stage"}, "steps": [anchor_step]})()

    assert _semisynthesis_anchor_precursors([anchor_route]) == ["CC(C)O"]


def test_limit_result_routes_preserves_original_count_metadata():
    result = BaselineRunResult(
        target_smiles="CCO",
        backend="unit",
        routes=[
            RouteCandidate(target_smiles="CCO", solved=True, route_rank=i)
            for i in range(5)
        ],
    )

    _limit_result_routes(result, max_routes=2)

    assert len(result.routes) == 2
    assert result.solved
    assert result.raw_backend_metadata["cycle_route_limit"]["applied"]
    assert result.raw_backend_metadata["cycle_route_limit"]["original_route_count"] == 5
    assert result.raw_backend_metadata["cycle_route_limit"]["kept_route_count"] == 2


def test_condition_prediction_worker_flag_is_read_from_output_root(tmp_path):
    cycle_dir = tmp_path / "cycle_001"
    cycle_dir.mkdir()

    assert not _condition_prediction_enabled_for_worker(cycle_dir)

    (tmp_path / ENABLE_CONDITION_PREDICTION_FLAG).write_text("1", encoding="utf-8")

    assert _condition_prediction_enabled_for_worker(cycle_dir)


def test_adaptive_cycle_config_caps_large_cpu_template_cycles():
    cycles = _cycle_configs(BUFOTALIN_TARGET)
    large_cycle = next(cycle for cycle in cycles if cycle["name"].startswith("d24_i300_k150"))

    capped = _adaptive_cycle_config(large_cycle["config"], large_cycle)

    assert capped.max_iterations == 200
    assert capped.max_depth == 20
    assert capped.expansion_topk == 100
    assert capped.search_flags["adaptive_budget"]["enabled"]
    assert capped.search_flags["adaptive_budget"]["original"]["max_iterations"] == 300
    assert capped.search_flags["adaptive_budget"]["applied"]["max_depth"] == 20


def test_adaptive_cycle_config_tightens_large_n5_cycles_more_aggressively():
    cycles = _cycle_configs(BUFOTALIN_TARGET)
    large_n5 = next(
        cycle
        for cycle in cycles
        if cycle["name"].startswith("d30_i500_k200") and cycle["stock_mode"] == "n5"
    )

    capped = _adaptive_cycle_config(large_n5["config"], large_n5)

    assert capped.max_iterations == 160
    assert capped.max_depth == 20
    assert capped.expansion_topk == 100
    assert capped.search_flags["adaptive_budget"]["applied"]["max_iterations"] == 160


def test_adaptive_cycle_config_keeps_stable_and_upstream_cycles_unchanged():
    cycles = _cycle_configs(BUFOTALIN_TARGET)
    upstream = cycles[0]
    stable = next(cycle for cycle in cycles if cycle["name"].startswith("d20_i200_k100"))

    assert _adaptive_cycle_config(upstream["config"], upstream) is upstream["config"]
    assert _adaptive_cycle_config(stable["config"], stable) is stable["config"]


def test_prepare_runtime_root_defaults_to_project_tmp_and_condition_prediction(tmp_path, monkeypatch):
    monkeypatch.delenv("TMPDIR", raising=False)

    _prepare_runtime_root(tmp_path)

    assert (tmp_path / "_tmp").is_dir()
    assert (tmp_path / ENABLE_CONDITION_PREDICTION_FLAG).read_text(encoding="utf-8") == "1\n"
    assert os.environ["TMPDIR"] == str((tmp_path / "_tmp").resolve())


def test_finalize_result_package_runs_postprocessors_and_allows_early_stop_audit_failure(tmp_path, monkeypatch):
    import scripts.run_bufotalin_12h_iteration as runner

    commands = []

    def fake_export(output_root):
        return {"enabled": True, "returncode": 0, "output_dir": str(output_root / "final_candidates")}

    def fake_run(command, *, output_root):
        commands.append(command[1])
        if command[1].endswith("audit_bufotalin_early_stop_review.py"):
            return {"command": command, "returncode": 1, "stdout": '{"review_ready": false}', "stderr": ""}
        return {"command": command, "returncode": 0, "stdout": "{}", "stderr": ""}

    monkeypatch.setattr(runner, "_export_final_candidates", fake_export)
    monkeypatch.setattr(runner, "_run_finalize_command", fake_run)

    report = _finalize_result_package(tmp_path, vendor_root=Path("vendor/ChemEnzyRetroPlanner"), gpu=-1)

    assert report["ok"] is True
    assert report["final_candidates_export"]["returncode"] == 0
    assert commands[0].endswith("summarize_bufotalin_proposal_gate.py")
    assert "scripts/probe_bufotalin_frontier_proposals.py" in commands
    assert (tmp_path / "early_stop_review_audit.json").read_text(encoding="utf-8") == '{"review_ready": false}'
    assert (tmp_path / "result_package_finalize.json").exists()


def test_backfill_display_route_conditions_only_updates_renderable_routes(monkeypatch):
    class FakePredictor:
        def get_n_conditions(self, rxn, n=1, return_scores=True):
            return ([[25.0, "O", "CCO", "", None, None]], [0.8])

    import scripts.run_bufotalin_12h_iteration as runner

    monkeypatch.setattr(runner, "_load_rcr_condition_predictor", lambda vendor_root: FakePredictor())
    payload = {
        "routes": [
            {
                "metrics": {"cascade_verifier": {"feasible": True}},
                "steps": [{"reaction_smiles": "CCO>>CC=O", "condition_predictions": []}],
            },
            {
                "metrics": {"cascade_verifier": {"feasible": False}},
                "steps": [{"reaction_smiles": "CCN>>CC=N", "condition_predictions": []}],
            },
        ]
    }

    report = _backfill_display_route_conditions(payload, vendor_root="vendor/ChemEnzyRetroPlanner", enabled=True)

    assert report["steps_attempted"] == 1
    assert report["steps_filled"] == 1
    assert payload["routes"][0]["steps"][0]["condition_predictions"][0]["Temperature"] == 25.0
    assert payload["routes"][1]["steps"][0]["condition_predictions"] == []


def test_cycle_proposal_gate_filters_unsupported_prenyl_terminal_before_payload_export():
    prenyl_terminal = (
        "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/"
        "CC/C(C)=C/CC/C(C)=C/CO"
    )
    payload = {
        "n_results": 1,
        "routes": [
            {
                "score": 0.1,
                "n_steps": 4,
                "route_rank": 0,
                "metrics": {"terminal_reactants": [prenyl_terminal]},
                "steps": [
                    {
                        "index": idx,
                        "product": "C" * (idx + 2),
                        "main_reactant": "C" * (idx + 1),
                        "reaction_smiles": f"{'C' * (idx + 1)}>>{'C' * (idx + 2)}",
                        "condition_predictions": [{"condition_label": "RCR model prediction"}],
                    }
                    for idx in range(4)
                ],
            }
        ],
        "route_set_metrics": {},
        "ui_metadata": {},
        "search_status": {},
        "failure_diagnosis": [],
        "failure_analysis": {"failure_categories": []},
    }

    report = _apply_cycle_proposal_gate(payload)

    assert report["input_routes"] == 1
    assert report["kept_routes"] == 0
    assert report["dropped_routes"] == 1
    assert payload["routes"] == []
    assert payload["n_results"] == 0
    assert payload["search_status"]["proposal_gate_removed_all"]
    assert "unsupported_biosynthetic_prenyl_terminal" in report["reason_counts"]
    assert "proposal_gate_filtered_all" in payload["failure_diagnosis"]


def test_cycle_proposal_gate_repairs_tbs_frontier_with_source_supported_rescue():
    tbs_frontier = "CC(C)(C)[Si](C)(C)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](O)C[C@]32O)C1"
    payload = {
        "n_results": 1,
        "routes": [
            {
                "score": 0.1,
                "n_steps": 1,
                "route_rank": 0,
                "metrics": {
                    "terminal_reactants": ["CC(C)(C)S"],
                    "terminal_stock_status": {"CC(C)(C)S": True},
                },
                "steps": [
                    {
                        "index": 0,
                        "product": tbs_frontier,
                        "main_reactant": "CC(C)(C)S",
                        "aux_reactants": [],
                        "reaction_smiles": f"CC(C)(C)S>>{tbs_frontier}",
                        "condition_predictions": [],
                        "stock_status": {"CC(C)(C)S": True},
                    }
                ],
            }
        ],
        "route_set_metrics": {},
        "ui_metadata": {},
        "search_status": {},
        "failure_diagnosis": [],
        "failure_analysis": {"failure_categories": []},
    }

    report = _apply_cycle_proposal_gate(payload)

    assert report["kept_routes"] == 1
    assert report["dropped_routes"] == 0
    assert report["repaired_routes"] == 1
    repaired = payload["routes"][0]
    assert repaired["frontier_repair"]["rescue_type"] == "late_stage_tbs_silylation"
    assert "CC(C)(C)[Si](C)(C)Cl" in repaired["steps"][0]["aux_reactants"]
    assert repaired["metrics"]["frontier_repaired_semisynthesis"]
    assert repaired["metrics"]["cascade_verifier"]["feasible"]
    assert repaired["proposal_gate"]["decision"] == "keep"


def test_cycle_proposal_gate_keeps_enzyme_supported_prenyl_terminal():
    prenyl_terminal = (
        "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/"
        "CC/C(C)=C/CC/C(C)=C/CO"
    )
    payload = {
        "n_results": 1,
        "routes": [
            {
                "score": 0.1,
                "n_steps": 1,
                "route_rank": 0,
                "metrics": {"terminal_reactants": [prenyl_terminal]},
                "steps": [
                    {
                        "index": 0,
                        "product": "CC",
                        "main_reactant": "C",
                        "reaction_smiles": "C>>CC",
                        "is_enzymatic": True,
                        "ec": "2.5.1.21",
                        "condition_predictions": [{"condition_label": "RCR model prediction"}],
                    }
                ],
            }
        ],
        "route_set_metrics": {},
        "ui_metadata": {},
        "search_status": {},
    }

    report = _apply_cycle_proposal_gate(payload)

    assert report["kept_routes"] == 1
    assert report["dropped_routes"] == 0
    assert payload["routes"][0]["proposal_gate"]["route_hard_reasons"] == []

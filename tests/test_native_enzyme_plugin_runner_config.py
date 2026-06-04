import scripts.run_native_enzyme_plugin_benchmark as generic_plugin
import scripts.run_statin_native_enzyme_plugin_comparison as statin_plugin


def test_generic_native_enzyme_plugin_runner_exposes_material_gate(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_enzyme_plugin_benchmark.py",
            "--quality-score-bonus",
            "0.18",
            "--min-quality-score",
            "0.7",
            "--material-max-heavy-gain",
            "4",
            "--material-max-carbon-gain",
            "3",
            "--material-max-hetero-gain",
            "2",
        ],
    )

    args = generic_plugin.parse_args()
    payload = generic_plugin.enzyme_plugin_payload(args)

    assert payload["enable_sp_v1"] is True
    assert payload["sp_v1_hard_gate"] is True
    assert payload["require_material_sanity"] is True
    assert payload["quality_score_bonus"] == 0.18
    assert payload["min_quality_score"] == 0.7
    assert payload["material_max_heavy_gain"] == 4
    assert payload["material_max_carbon_gain"] == 3
    assert payload["material_max_hetero_gain"] == 2


def test_generic_runner_exposes_chemical_plugin(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_enzyme_plugin_benchmark.py",
            "--enable-chemical-plugin",
            "--chemical-plugin-top-k",
            "12",
            "--chemical-plugin-max-added",
            "5",
            "--disable-chemical-proposal-gate",
        ],
    )

    args = generic_plugin.parse_args()
    payload = generic_plugin.chemical_plugin_payload(args)

    assert args.enable_chemical_plugin is True
    assert payload["enabled"] is True
    assert payload["top_k"] == 12
    assert payload["max_added"] == 5
    assert payload["fusion_mode"] == "graphfp_first"
    assert payload["require_proposal_gate"] is False


def test_statin_native_enzyme_plugin_runner_can_disable_material_gate(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_statin_native_enzyme_plugin_comparison.py",
            "--pack-dir",
            "custom_pack",
            "--disable-material-gate",
            "--disable-sp-v1-hard-gate",
        ],
    )

    args = statin_plugin.parse_args()
    payload = statin_plugin.enzyme_plugin_payload(args)

    assert payload["pack_dir"] == "custom_pack"
    assert payload["enable_sp_v1"] is True
    assert payload["sp_v1_hard_gate"] is False
    assert payload["require_material_sanity"] is False


def test_statin_runner_exposes_chemical_plugin(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_statin_native_enzyme_plugin_comparison.py",
            "--enable-chemical-plugin",
            "--chemical-plugin-dual-top-k",
            "80",
            "--chemical-plugin-score-scale",
            "0.5",
        ],
    )

    args = statin_plugin.parse_args()
    payload = statin_plugin.chemical_plugin_payload(args)

    assert args.enable_chemical_plugin is True
    assert payload["dual_top_k"] == 80
    assert payload["score_scale"] == 0.5
    assert payload["require_proposal_gate"] is True


def test_statin_summary_aggregates_material_gate_stats():
    rows = [
        {
            "native": {
                "solved": True,
                "route_count": 1,
                "native_classified_enzyme_route_count": 0,
                "any_enzyme_route_count": 0,
            },
            "plugin": {
                "solved": True,
                "route_count": 2,
                "native_classified_enzyme_route_count": 0,
                "plugin_injected_enzyme_route_count": 1,
                "any_enzyme_route_count": 1,
                "route_plausibility": {
                    "passed": 1,
                    "injected_enzyme_routes": {"passed": 1},
                },
            },
            "delta": {
                "plugin_selected_enzyme_route": True,
                "plugin_selected_injected_enzyme_route": True,
            },
            "plugin_stats": {
                "added_candidates": 3,
                "sp_v1_accepted": 2,
                "sp_v1_rejected": 1,
                "quality_passed": 2,
                "quality_warned": 0,
                "quality_rejected": 1,
                "material_rejected": 1,
                "error_count": 0,
            },
            "plugin_representative_audit": {
                "available": True,
                "route_plausibility_passed": False,
            },
        }
    ]

    summary = statin_plugin.summarize(rows)

    assert summary["plugin_added_candidates"] == 3
    assert summary["plugin_sp_v1_accepted"] == 2
    assert summary["plugin_sp_v1_rejected"] == 1
    assert summary["plugin_quality_passed"] == 2
    assert summary["plugin_quality_rejected"] == 1
    assert summary["plugin_material_rejected"] == 1
    assert summary["plugin_plausible_routes"] == 1
    assert summary["plugin_plausible_injected_enzyme_routes"] == 1


def test_statin_result_summary_includes_route_pool_plausibility():
    step = statin_plugin.RouteStepCandidate(
        product_smiles="CCO",
        reactant_smiles=["CCO"],
        rxn_smiles="CCO>>CCO",
    )
    route = statin_plugin.RouteCandidate(
        target_smiles="CCO",
        steps=[step],
        solved=True,
    )
    result = statin_plugin.BaselineRunResult(
        target_smiles="CCO",
        backend="test",
        routes=[route],
    )

    payload = statin_plugin.result_summary(result, route)

    assert payload["route_plausibility"]["routes"] == 1
    assert payload["route_plausibility"]["passed"] == 1
    assert payload["route_plausibility"]["failed"] == 0


def test_generic_summary_aggregates_chemical_plugin_stats():
    rows = [
        {
            "run": "native_enzyme_chemical_plugin",
            "solved": True,
            "route_count": 3,
            "enzyme_route_count": 1,
            "enzyme_step_count": 1,
            "elapsed_s": 1.0,
            "failure_categories": [],
            "plugin_stats": {"added_candidates": 2, "sp_v1_accepted": 1},
            "chemical_plugin_stats": {
                "calls": 4,
                "dual_candidates": 5,
                "added_candidates": 3,
                "proposal_gate_kept": 3,
                "proposal_gate_rejected": 1,
                "error_count": 0,
            },
        }
    ]

    summary = generic_plugin.summarize(rows)["native_enzyme_chemical_plugin"]

    assert summary["plugin_added_candidates"] == 2
    assert summary["chemical_plugin_calls"] == 4
    assert summary["chemical_plugin_dual_candidates"] == 5
    assert summary["chemical_plugin_added_candidates"] == 3
    assert summary["chemical_plugin_gate_rejected"] == 1

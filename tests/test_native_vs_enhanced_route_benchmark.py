from argparse import Namespace
import json

from cascade_planner.baselines.route_contract import BaselineRunResult, RouteCandidate, RouteStepCandidate
import scripts.run_native_vs_enhanced_route_benchmark as bench


def test_parse_args_enables_formal_stock_aware_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_native_vs_enhanced_route_benchmark.py"])

    args = bench.parse_args()

    assert args.stock_aware_action_rerank is True
    assert args.exact_stock_reactant_bonus == 2.0
    assert args.full_stock_action_bonus == 5.0
    assert args.disable_vendor_stock is False
    assert args.enable_template_relevance_source is False
    assert args.disable_retrochimera_source is False
    assert args.disable_chemtemplates_after_depth == -1
    assert args.enable_stock_closing_probe is False
    assert args.stock_closing_probe_sources == "chem_enzy_graphfp_fusion,template_relevance,chemtemplates"
    assert args.stock_closing_probe_topk == 75
    assert args.stock_closing_probe_remaining_depth == 2
    assert args.source_min_budgets == ""
    assert args.enable_enzyme_continuation_source_gate is False
    assert args.enable_selected_enzyme_evidence_enrichment is False
    assert args.selected_enzyme_evidence_topk == 3
    assert args.selected_enzyme_evidence_min_similarity == 0.35
    assert args.enable_sp_v1_enzyme_result_selector is False
    assert args.sp_v1_enzyme_result_pool_min == 5
    assert args.sp_v1_enzyme_selector_max_rank == 5
    assert args.sp_v1_enzyme_selector_max_extra_cost == 0.0
    assert args.enable_sp_v1_enzyme_selector_cost_exception is False
    assert args.sp_v1_enzyme_selector_cost_exception_max_extra_cost == 0.0
    assert args.enable_enzyme_sp_material_gate is False
    assert args.enzyme_sp_material_gate_sources == ""
    assert args.enable_semisynthesis_stock is False
    assert args.enable_semisynthesis_rescue_source is False
    assert args.semisynthesis_rescue_min_budget == 2
    assert args.enable_chemical_anchor_stock is False
    assert args.enable_chemical_anchor_rescue_source is False
    assert args.chemical_anchor_rescue_min_budget == 2
    assert args.preset_n_results_override == 0
    assert args.preset_expansion_budget_override == 0
    assert args.preset_max_depth_override == 0
    assert args.preset_route_tree_timeout_s_override == 0.0
    assert args.target_rows is None
    assert args.checkpoint_every == 0
    assert args.resume_rows is None
    assert args.template_relevance_models == "template_relevance.reaxys"
    assert args.template_relevance_topk == 20
    assert args.template_relevance_min_budget == 4
    assert args.enhancement_preset == ""


def test_parse_args_final_clean_fastclosure_preset_applies_audited_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_vs_enhanced_route_benchmark.py",
            "--enhancement-preset",
            bench.FINAL_CLEAN_FASTCLOSURE_PRESET,
        ],
    )

    args = bench.parse_args()
    env = bench.enhanced_env_values(args)

    assert args.enhancement_preset == bench.FINAL_CLEAN_FASTCLOSURE_PRESET
    assert args.max_depth == 6
    assert args.expansion_topk == 60
    assert args.branch_factor == 8
    assert args.expansion_budget == 90
    assert args.n_results == 1
    assert args.route_tree_timeout_s == 240.0
    assert args.bridge_enzyme_bonus == 2.0
    assert args.source_min_budgets == (
        "chem_enzy_bionav:8,enzyme_precedent:4,v3_retrieval:4,enzexpand:4,enzyformer:3"
    )
    assert args.enable_enzyme_continuation_source_gate is True
    assert args.enable_selected_enzyme_evidence_enrichment is True
    assert args.enable_sp_v1_enzyme_result_selector is True
    assert args.enable_sp_v1_enzyme_selector_cost_exception is True
    assert args.sp_v1_enzyme_selector_cost_exception_max_extra_cost == 2.0
    assert args.enable_enzyme_sp_material_gate is False
    assert args.enzyme_sp_accepted_bonus == 3.0
    assert args.enzyme_sp_score_bonus == 1.0
    assert args.enable_enhanced_chemical_fusion_source is True
    assert args.enhanced_fusion_mode == "graphfp_first"
    assert args.enhanced_chemical_fusion_topk == 60
    assert args.enable_template_relevance_source is True
    assert args.template_relevance_models == "template_relevance.reaxys"
    assert args.disable_retrochimera_source is True
    assert args.enable_enhanced_bionav_source is True
    assert args.disable_chemtemplates_after_depth == 0
    assert args.enable_stock_closing_probe is True
    assert bench.stock_closing_probe_sources(args) == ["chem_enzy_graphfp_fusion", "template_relevance"]
    assert args.stock_closing_probe_topk == 60
    assert args.stock_closing_probe_topk_cap == 60
    assert env["AUTOPLANNER_DISABLE_RETROCHIMERA"] == "1"
    assert env["AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH"] == "chemtemplates:0"
    assert env["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] == "1"
    assert env["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_ALLOW_COST_EXCEPTION"] == "1"
    assert env["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_COST_EXCEPTION_MAX_EXTRA_COST"] == "2.0"
    assert "AUTOPLANNER_ENZYME_SP_MATERIAL_GATE" not in env
    assert env["AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS"] == args.source_min_budgets
    assert env["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_SOURCES"] == (
        "chem_enzy_graphfp_fusion,template_relevance"
    )
    assert env["AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS"] == (
        "chem_enzy_graphfp_fusion:60,template_relevance:20,chem_enzy_bionav:10"
    )


def test_parse_args_preset_overrides_are_explicit_and_post_preset(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_vs_enhanced_route_benchmark.py",
            "--enhancement-preset",
            bench.FINAL_CLEAN_FASTCLOSURE_PRESET,
            "--preset-n-results-override",
            "10",
            "--preset-expansion-budget-override",
            "240",
            "--preset-max-depth-override",
            "7",
            "--preset-route-tree-timeout-s-override",
            "600",
        ],
    )

    args = bench.parse_args()

    assert args.enhancement_preset == bench.FINAL_CLEAN_FASTCLOSURE_PRESET
    assert args.n_results == 10
    assert args.expansion_budget == 240
    assert args.max_depth == 7
    assert args.route_tree_timeout_s == 600.0
    assert args.enable_stock_closing_probe is True
    assert args.disable_retrochimera_source is True


def test_parse_args_final_clean_material_gate_preset_extends_clean_preset(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_vs_enhanced_route_benchmark.py",
            "--enhancement-preset",
            bench.FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_PRESET,
        ],
    )

    args = bench.parse_args()
    env = bench.enhanced_env_values(args)

    assert args.enhancement_preset == bench.FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_PRESET
    assert args.max_depth == 6
    assert args.enable_sp_v1_enzyme_selector_cost_exception is True
    assert args.enable_enzyme_sp_material_gate is True
    assert args.enzyme_sp_material_gate_sources == "enzyme_precedent"
    assert env["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE"] == "1"
    assert env["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES"] == "enzyme_precedent"
    assert env["AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS"] == (
        "chem_enzy_graphfp_fusion:60,template_relevance:20,chem_enzy_bionav:10"
    )


def test_parse_args_final_clean_semisynthesis_preset_extends_material_gate(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_vs_enhanced_route_benchmark.py",
            "--enhancement-preset",
            bench.FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_PRESET,
        ],
    )

    args = bench.parse_args()
    env = bench.enhanced_env_values(args)

    assert args.enhancement_preset == bench.FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_PRESET
    assert args.enable_enzyme_sp_material_gate is True
    assert args.enzyme_sp_material_gate_sources == "enzyme_precedent"
    assert args.enable_semisynthesis_stock is True
    assert args.enable_semisynthesis_rescue_source is True
    assert env["AUTOPLANNER_ENABLE_SEMISYNTHESIS_RESCUE_PROPOSALS"] == "1"
    assert env["AUTOPLANNER_SEMISYNTHESIS_RESCUE_MIN_BUDGET"] == "2"
    assert env["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE"] == "1"
    assert env["AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS"] == (
        "chem_enzy_graphfp_fusion:60,template_relevance:20,chem_enzy_bionav:10,semisynthesis_rescue:8"
    )


def test_parse_args_final_clean_chemical_anchor_preset_extends_semisynthesis(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_vs_enhanced_route_benchmark.py",
            "--enhancement-preset",
            bench.FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_CHEMICAL_ANCHOR_PRESET,
        ],
    )

    args = bench.parse_args()
    env = bench.enhanced_env_values(args)

    assert (
        args.enhancement_preset
        == bench.FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_CHEMICAL_ANCHOR_PRESET
    )
    assert args.enable_enzyme_sp_material_gate is True
    assert args.enzyme_sp_material_gate_sources == "enzyme_precedent"
    assert args.enable_semisynthesis_stock is True
    assert args.enable_semisynthesis_rescue_source is True
    assert args.enable_chemical_anchor_stock is True
    assert args.enable_chemical_anchor_rescue_source is True
    assert env["AUTOPLANNER_ENABLE_CHEMICAL_ANCHOR_RESCUE_PROPOSALS"] == "1"
    assert env["AUTOPLANNER_CHEMICAL_ANCHOR_RESCUE_MIN_BUDGET"] == "2"
    assert env["AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS"] == (
        "chem_enzy_graphfp_fusion:60,template_relevance:20,chem_enzy_bionav:10,"
        "semisynthesis_rescue:8,chemical_anchor_rescue:4"
    )


def test_parse_args_can_enable_source_limited_enzyme_sp_material_gate(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_vs_enhanced_route_benchmark.py",
            "--enable-enzyme-sp-material-gate",
            "--enzyme-sp-material-gate-sources",
            "enzyme_precedent,chem_enzy_onmt",
        ],
    )

    args = bench.parse_args()
    env = bench.enhanced_env_values(args)

    assert args.enable_enzyme_sp_material_gate is True
    assert args.enzyme_sp_material_gate_sources == "enzyme_precedent,chem_enzy_onmt"
    assert env["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE"] == "1"
    assert env["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES"] == "enzyme_precedent,chem_enzy_onmt"


def test_parse_args_can_disable_formal_stock_aware_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_vs_enhanced_route_benchmark.py",
            "--disable-stock-aware-action-rerank",
        ],
    )

    args = bench.parse_args()

    assert args.stock_aware_action_rerank is False


def test_load_benchmark_targets_preserves_requested_positive_before_max(monkeypatch):
    monkeypatch.setattr(
        bench,
        "load_positive_targets",
        lambda path, *, count: [
            {"target_smiles": f"P{i}", "label": 1, "label_source": "positive"}
            for i in range(count)
        ],
    )
    monkeypatch.setattr(
        bench,
        "load_negative_targets",
        lambda pack_dir, *, count, seed: [
            {"target_smiles": f"N{i}", "label": 0, "label_source": "negative"}
            for i in range(count)
        ],
    )
    args = Namespace(
        probe_rows="probe.jsonl",
        pack_dir="pack",
        positives=1,
        negatives=1,
        seed=123,
        max_targets=1,
        shuffle_targets=False,
    )

    rows = bench.load_benchmark_targets(args)

    assert rows == [{"target_smiles": "P0", "label": 1, "label_source": "positive"}]


def test_load_benchmark_targets_allows_zero_positives(monkeypatch):
    monkeypatch.setattr(
        bench,
        "load_positive_targets",
        lambda path, *, count: [
            {"target_smiles": f"P{i}", "label": 1, "label_source": "positive"}
            for i in range(count)
        ],
    )
    monkeypatch.setattr(
        bench,
        "load_negative_targets",
        lambda pack_dir, *, count, seed: [
            {"target_smiles": f"N{i}", "label": 0, "label_source": "negative"}
            for i in range(count)
        ],
    )
    args = Namespace(
        probe_rows="probe.jsonl",
        pack_dir="pack",
        positives=0,
        negatives=2,
        seed=123,
        max_targets=2,
        shuffle_targets=False,
    )

    rows = bench.load_benchmark_targets(args)

    assert rows == [
        {"target_smiles": "N0", "label": 0, "label_source": "negative"},
        {"target_smiles": "N1", "label": 0, "label_source": "negative"},
    ]


def test_load_benchmark_targets_can_use_explicit_target_rows(tmp_path, monkeypatch):
    target_rows = tmp_path / "targets.jsonl"
    target_rows.write_text(
        "\n".join(
            [
                json.dumps({"target_smiles": "CCO", "label": 0, "label_source": "slow_negative"}),
                json.dumps({"target": "CCN", "label": 1}),
                json.dumps({"smiles": "CCO", "label": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bench,
        "load_positive_targets",
        lambda path, *, count: [{"target_smiles": "SHOULD_NOT_LOAD", "label": 1}],
    )
    monkeypatch.setattr(
        bench,
        "load_negative_targets",
        lambda pack_dir, *, count, seed: [{"target_smiles": "SHOULD_NOT_LOAD", "label": 0}],
    )
    args = Namespace(
        target_rows=target_rows,
        probe_rows="probe.jsonl",
        pack_dir="pack",
        positives=99,
        negatives=99,
        seed=123,
        max_targets=0,
        shuffle_targets=False,
    )

    rows = bench.load_benchmark_targets(args)

    assert rows == [
        {"target_smiles": "CCO", "label": 0, "label_source": "slow_negative"},
        {"target": "CCN", "label": 1, "target_smiles": "CCN", "label_source": "explicit_target_rows"},
    ]


def test_completed_resume_target_keys_requires_all_requested_runs():
    args = Namespace(skip_native=False, skip_enhanced=False)
    rows = [
        {"target_smiles": "CCO", "target_canonical": "CCO", "run": "native_chemenzy"},
        {"target_smiles": "CCO", "target_canonical": "CCO", "run": "enhanced_route_tree"},
        {"target_smiles": "CCN", "target_canonical": "CCN", "run": "native_chemenzy"},
    ]

    completed = bench.completed_resume_target_keys(rows, args=args)

    assert "CCO" in completed
    assert "CCN" not in completed


def test_source_min_budget_env_only_emits_positive_enzyme_precedent_budget():
    assert bench.source_min_budget_env(Namespace(enzyme_precedent_min_budget=0)) == ""
    assert bench.source_min_budget_env(Namespace(enzyme_precedent_min_budget=3)) == "enzyme_precedent:3"


def test_source_min_budget_env_merges_general_floors_and_legacy_alias():
    args = Namespace(
        source_min_budgets="chem_enzy_bionav:8,enzyme_precedent:2,v3_retrieval=4,bad,enzexpand:x",
        enzyme_precedent_min_budget=5,
    )

    assert bench.parse_source_budget_spec(args.source_min_budgets) == {
        "chem_enzy_bionav": 8,
        "enzyme_precedent": 2,
        "v3_retrieval": 4,
    }
    assert bench.source_min_budget_env(args) == "chem_enzy_bionav:8,enzyme_precedent:5,v3_retrieval:4"


def test_enhanced_env_values_sets_sp_v1_enzyme_result_selector():
    args = Namespace(
        bridge_enzyme_bonus=2.0,
        source_min_budgets="",
        enzyme_precedent_min_budget=0,
        route_tree_timeout_s=60.0,
        enzyme_sp_accepted_bonus=3.0,
        enzyme_sp_score_bonus=1.0,
        stock_aware_action_rerank=True,
        exact_stock_reactant_bonus=2.0,
        full_stock_action_bonus=5.0,
        enable_enzyme_continuation_source_gate=False,
        enable_selected_enzyme_evidence_enrichment=False,
        selected_enzyme_evidence_topk=3,
        selected_enzyme_evidence_min_similarity=0.35,
        enable_sp_v1_enzyme_result_selector=True,
        sp_v1_enzyme_result_pool_min=7,
        sp_v1_enzyme_selector_max_rank=4,
        sp_v1_enzyme_selector_max_extra_cost=1.5,
        enable_sp_v1_enzyme_selector_cost_exception=True,
        sp_v1_enzyme_selector_cost_exception_max_extra_cost=2.5,
        disable_retrochimera_source=False,
        disable_chemtemplates_after_depth=-1,
        enable_stock_closing_probe=False,
        chem_template_max_per_query=0,
        chem_template_max_templates=0,
        enable_enhanced_chemenzy_assembly=False,
        enable_enhanced_chemical_fusion_source=False,
        enable_template_relevance_source=False,
        enable_enhanced_bionav_source=False,
    )

    env = bench.enhanced_env_values(args)

    assert env["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] == "1"
    assert env["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] == "7"
    assert env["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK"] == "4"
    assert env["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_EXTRA_COST"] == "1.5"
    assert env["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_ALLOW_COST_EXCEPTION"] == "1"
    assert env["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_COST_EXCEPTION_MAX_EXTRA_COST"] == "2.5"


def test_enhanced_env_values_can_disable_chemtemplates_after_depth():
    args = Namespace(
        bridge_enzyme_bonus=0.0,
        source_min_budgets="",
        enzyme_precedent_min_budget=0,
        route_tree_timeout_s=60.0,
        enzyme_sp_accepted_bonus=0.0,
        enzyme_sp_score_bonus=0.0,
        stock_aware_action_rerank=True,
        exact_stock_reactant_bonus=2.0,
        full_stock_action_bonus=5.0,
        enable_enzyme_continuation_source_gate=False,
        enable_selected_enzyme_evidence_enrichment=False,
        selected_enzyme_evidence_topk=3,
        selected_enzyme_evidence_min_similarity=0.35,
        enable_sp_v1_enzyme_result_selector=False,
        sp_v1_enzyme_result_pool_min=5,
        sp_v1_enzyme_selector_max_rank=5,
        sp_v1_enzyme_selector_max_extra_cost=0.0,
        disable_retrochimera_source=False,
        disable_chemtemplates_after_depth=0,
        enable_stock_closing_probe=False,
        chem_template_max_per_query=0,
        chem_template_max_templates=0,
        enable_enhanced_chemenzy_assembly=False,
        enable_enhanced_chemical_fusion_source=False,
        enable_template_relevance_source=False,
        enable_enhanced_bionav_source=False,
    )

    env = bench.enhanced_env_values(args)

    assert env["AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH"] == "chemtemplates:0"
    assert bench._merge_disable_sources_after_depth("retrochimera:0", {"chemtemplates": 0}) == (
        "retrochimera:0,chemtemplates:0"
    )


def test_native_one_step_models_parses_strict_native_baseline_models():
    args = Namespace(
        native_one_step_models=" graphfp_models.USPTO-full_remapped, onmt_models.bionav_native_one_step "
    )

    assert bench.native_one_step_models(args) == [
        "graphfp_models.USPTO-full_remapped",
        "onmt_models.bionav_native_one_step",
    ]


def test_native_one_step_models_falls_back_to_default_models():
    assert bench.native_one_step_models(Namespace(native_one_step_models="")) == list(bench.DEFAULT_ONE_STEP_MODELS)


def test_native_chemical_rescue_models_parse_or_fallback():
    args = Namespace(native_chemical_rescue_one_step_models=" graphfp_models.USPTO-full_remapped ")

    assert bench.native_chemical_rescue_models(args) == ["graphfp_models.USPTO-full_remapped"]
    assert bench.native_chemical_rescue_models(Namespace(native_chemical_rescue_one_step_models="")) == [
        "graphfp_models.USPTO-full_remapped"
    ]


def test_template_relevance_models_parse_or_fallback():
    args = Namespace(template_relevance_models=" template_relevance.reaxys, template_relevance.pistachio ")

    assert bench.template_relevance_models(args) == [
        "template_relevance.reaxys",
        "template_relevance.pistachio",
    ]
    assert bench.template_relevance_models(Namespace(template_relevance_models="")) == ["template_relevance.reaxys"]


def test_stock_closing_probe_sources_parse_or_fallback():
    args = Namespace(stock_closing_probe_sources=" chem_enzy_graphfp_fusion, template_relevance ")

    assert bench.stock_closing_probe_sources(args) == ["chem_enzy_graphfp_fusion", "template_relevance"]
    assert bench.stock_closing_probe_sources(Namespace(stock_closing_probe_sources="")) == [
        "chem_enzy_graphfp_fusion",
        "template_relevance",
        "chemtemplates",
    ]


def test_native_chemical_rescue_only_attempts_unsolved_non_enzyme_routes():
    enabled = Namespace(enable_native_chemical_rescue=True)

    assert bench._should_attempt_native_chemical_rescue([], args=enabled) is True
    assert bench._should_attempt_native_chemical_rescue([{"route_solved": True}], args=enabled) is False
    assert bench._should_attempt_native_chemical_rescue([{"has_enzyme_step": True}], args=enabled) is False
    assert bench._should_attempt_native_chemical_rescue([], args=Namespace(enable_native_chemical_rescue=False)) is False


def test_mark_native_chemical_rescue_route_relabels_step_sources():
    route = {
        "route_solved": True,
        "has_enzyme_step": False,
        "steps": [
            {"source": "ChemEnzyRetroPlanner", "reaction_smiles": "CC>>CCC"},
            {"reaction_smiles": "C>>CC"},
        ],
    }

    marked = bench._mark_native_chemical_rescue_route(route)

    assert marked["native_chemical_rescue"] is True
    assert marked["route_tree_search_status"] == "native_chemical_rescue"
    assert marked["steps"][0]["source"] == "native_chemical_rescue:ChemEnzyRetroPlanner"
    assert marked["steps"][1]["source"] == "native_chemical_rescue:ChemEnzyRetroPlanner"


def test_enhanced_env_values_include_stock_bonus_and_template_caps():
    args = Namespace(
        bridge_enzyme_bonus=2.0,
        enzyme_precedent_min_budget=2,
        route_tree_timeout_s=20.0,
        stock_aware_action_rerank=True,
        exact_stock_reactant_bonus=1.25,
        full_stock_action_bonus=2.5,
        enzyme_sp_accepted_bonus=0.75,
        enzyme_sp_score_bonus=1.5,
        chem_template_max_per_query=40,
        chem_template_max_templates=1000,
        enable_enhanced_chemenzy_assembly=False,
        enable_enzyme_continuation_source_gate=True,
        enable_selected_enzyme_evidence_enrichment=True,
        selected_enzyme_evidence_topk=5,
        selected_enzyme_evidence_min_similarity=0.42,
    )

    values = bench.enhanced_env_values(args)

    assert values["AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS"] == "2.0"
    assert values["AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS"] == "enzyme_precedent:2"
    assert values["AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS"] == "0.75"
    assert values["AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS"] == "1.5"
    assert values["AUTOPLANNER_ROUTE_TREE_EXACT_STOCK_REACTANT_BONUS"] == "1.25"
    assert values["AUTOPLANNER_ROUTE_TREE_FULL_STOCK_ACTION_BONUS"] == "2.5"
    assert values["AUTOPLANNER_ROUTE_TREE_BRENDA_CONDITION_PRIOR"] == "1"
    assert values["AUTOPLANNER_BRIDGE_GATE_ALLOW_ENZYME_CONTINUATION"] == "1"
    assert values["AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_ENRICHMENT"] == "1"
    assert values["AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_TOPK"] == "5"
    assert values["AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_MIN_SIMILARITY"] == "0.42"
    assert values["AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_QUERY"] == "40"
    assert values["AUTOPLANNER_CHEM_TEMPLATES_MAX_TEMPLATES"] == "1000"


def test_enhanced_env_values_can_disable_retrochimera_source():
    args = Namespace(
        bridge_enzyme_bonus=0.0,
        enzyme_precedent_min_budget=0,
        route_tree_timeout_s=20.0,
        stock_aware_action_rerank=False,
        exact_stock_reactant_bonus=1.0,
        full_stock_action_bonus=2.0,
        enzyme_sp_accepted_bonus=0.0,
        enzyme_sp_score_bonus=0.0,
        chem_template_max_per_query=0,
        chem_template_max_templates=0,
        enable_enhanced_chemenzy_assembly=False,
        enable_enhanced_chemical_fusion_source=False,
        enable_template_relevance_source=False,
        enable_enhanced_bionav_source=False,
        enable_enzyme_continuation_source_gate=False,
        enable_selected_enzyme_evidence_enrichment=False,
        disable_retrochimera_source=True,
    )

    values = bench.enhanced_env_values(args)

    assert values["AUTOPLANNER_DISABLE_RETROCHIMERA"] == "1"


def test_stock_closing_probe_env_enables_late_closure_probe():
    args = Namespace(
        stock_closing_probe_sources="chem_enzy_graphfp_fusion,template_relevance",
        stock_closing_probe_topk=64,
        stock_closing_probe_topk_cap=80,
        stock_closing_probe_max_actions=3,
        stock_closing_probe_remaining_depth=2,
    )

    values = bench.stock_closing_probe_env(args)

    assert values["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE"] == "1"
    assert values["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_SOURCES"] == (
        "chem_enzy_graphfp_fusion,template_relevance"
    )
    assert values["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_TOPK"] == "64"
    assert values["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_TOPK_CAP"] == "80"
    assert values["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_MAX_ACTIONS"] == "3"
    assert values["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_REMAINING_DEPTH"] == "2"


def test_live_retro_disable_flag_controls_retrochimera(monkeypatch):
    from cascade_planner.cascadeboard import live_retro

    monkeypatch.delenv("AUTOPLANNER_DISABLE_RETROCHIMERA", raising=False)
    assert live_retro._retrochimera_enabled() is True

    monkeypatch.setenv("AUTOPLANNER_DISABLE_RETROCHIMERA", "1")
    assert live_retro._retrochimera_enabled() is False


def test_enhanced_chemenzy_assembly_env_enables_fusion_and_bionav_override():
    args = Namespace(
        enhanced_bionav_model="model.pt",
        enhanced_chemenzy_topk=50,
        enhanced_chemenzy_min_budget=6,
        enhanced_chemenzy_protected_topk=4,
        enhanced_chemenzy_protected_front=1,
        enhanced_chemenzy_route_mode="fallback",
        enhanced_fusion_mode="graphfp_first",
        enhanced_dualtower_device="cuda:1",
    )

    values = bench.enhanced_chemenzy_assembly_env(args)

    assert values["AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS"] == "1"
    assert values["AUTOPLANNER_CHEMENZY_ONMT_TOKENIZER"] == "pretokenized"
    assert values["AUTOPLANNER_ENABLE_GRAPHFP_DUALTOWER_FUSION"] == "1"
    assert values["AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_MODE"] == "graphfp_first"
    assert values["AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE"] == "fallback"
    assert values["AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_PROTECTED_TOPK"] == "4"
    assert values["AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_PROTECTED_FRONT"] == "1"
    assert values["AUTOPLANNER_DUALTOWER_TEMPLATE_DEVICE"] == "cuda:1"


def test_enhanced_bionav_source_env_enables_onmt_only_enzyme_source():
    args = Namespace(
        enhanced_bionav_model="model.pt",
        enhanced_bionav_source_topk=12,
        enable_enhanced_chemenzy_assembly=False,
        enhanced_chemenzy_topk=50,
    )

    values = bench.enhanced_bionav_source_env(args)

    assert values["AUTOPLANNER_ENABLE_CHEMENZY_BIONAV_PROPOSALS"] == "1"
    assert values["AUTOPLANNER_CHEMENZY_BIONAV_MODELS"] == "onmt_models.bionav_one_step"
    assert values["AUTOPLANNER_CHEMENZY_BIONAV_TOPK"] == "12"
    assert values["AUTOPLANNER_CHEMENZY_ONMT_TOKENIZER"] == "pretokenized"
    assert values["AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS"] == "chem_enzy_bionav:12"


def test_enhanced_bionav_source_env_preserves_assembly_request_cap():
    args = Namespace(
        enhanced_bionav_model="model.pt",
        enhanced_bionav_source_topk=12,
        enable_enhanced_chemenzy_assembly=True,
        enhanced_chemenzy_topk=50,
    )

    values = bench.enhanced_bionav_source_env(args)

    assert values["AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS"] == "chem_enzy_bionav:12,chem_enzy_onestep:50"


def test_enhanced_chemical_fusion_source_env_enables_graphfp_only_source():
    args = Namespace(
        enhanced_chemical_fusion_topk=40,
        enhanced_chemical_fusion_min_budget=5,
        enhanced_fusion_mode="graphfp_first",
        enhanced_dualtower_device="cuda:1",
    )

    values = bench.enhanced_chemical_fusion_source_env(args)

    assert values["AUTOPLANNER_ENABLE_CHEMENZY_GRAPHFP_FUSION_PROPOSALS"] == "1"
    assert values["AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_MODELS"] == "graphfp_models.USPTO-full_remapped"
    assert values["AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_TOPK"] == "40"
    assert values["AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_MIN_BUDGET"] == "5"
    assert values["AUTOPLANNER_ENABLE_GRAPHFP_DUALTOWER_FUSION"] == "1"
    assert values["AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_MODE"] == "graphfp_first"
    assert values["AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS"] == "chem_enzy_graphfp_fusion:40"
    assert values["AUTOPLANNER_DUALTOWER_TEMPLATE_DEVICE"] == "cuda:1"


def test_template_relevance_source_env_enables_independent_mar_source():
    args = Namespace(
        template_relevance_models="template_relevance.reaxys,template_relevance.pistachio",
        template_relevance_topk=24,
        template_relevance_min_budget=6,
        template_relevance_gpu=0,
        template_relevance_vendor_root="vendor/ChemEnzyRetroPlanner",
    )

    values = bench.template_relevance_source_env(args)

    assert values["AUTOPLANNER_ENABLE_TEMPLATE_RELEVANCE_PROPOSALS"] == "1"
    assert values["AUTOPLANNER_TEMPLATE_RELEVANCE_MODELS"] == (
        "template_relevance.reaxys,template_relevance.pistachio"
    )
    assert values["AUTOPLANNER_TEMPLATE_RELEVANCE_TOPK"] == "24"
    assert values["AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET"] == "6"
    assert values["AUTOPLANNER_TEMPLATE_RELEVANCE_GPU"] == "0"
    assert values["AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS"] == "template_relevance:24"


def test_enhanced_env_values_merges_independent_source_request_caps():
    args = Namespace(
        bridge_enzyme_bonus=0.0,
        enzyme_precedent_min_budget=0,
        route_tree_timeout_s=20.0,
        stock_aware_action_rerank=False,
        exact_stock_reactant_bonus=1.0,
        full_stock_action_bonus=2.0,
        enzyme_sp_accepted_bonus=0.0,
        enzyme_sp_score_bonus=0.0,
        chem_template_max_per_query=0,
        chem_template_max_templates=0,
        enable_enhanced_chemenzy_assembly=True,
        enable_enhanced_chemical_fusion_source=True,
        enable_template_relevance_source=True,
        template_relevance_models="template_relevance.reaxys",
        template_relevance_topk=24,
        template_relevance_min_budget=6,
        template_relevance_gpu=-1,
        template_relevance_vendor_root="vendor/ChemEnzyRetroPlanner",
        enable_enhanced_bionav_source=True,
        enhanced_bionav_model="model.pt",
        enhanced_chemenzy_topk=50,
        enhanced_chemical_fusion_topk=40,
        enhanced_chemical_fusion_min_budget=5,
        enhanced_fusion_mode="graphfp_first",
        enhanced_bionav_source_topk=12,
        enhanced_chemenzy_min_budget=6,
        enhanced_chemenzy_protected_topk=4,
        enhanced_chemenzy_protected_front=1,
        enhanced_chemenzy_route_mode="fallback",
        enhanced_dualtower_device=None,
        enable_enzyme_continuation_source_gate=False,
        enable_selected_enzyme_evidence_enrichment=False,
    )

    values = bench.enhanced_env_values(args)

    assert values["AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS"] == (
        "chem_enzy_graphfp_fusion:40,template_relevance:24,chem_enzy_bionav:12,chem_enzy_onestep:50"
    )


def test_build_enhanced_stock_checker_can_be_disabled():
    assert bench.build_enhanced_stock_checker(Namespace(enhanced_stock="none")) is None


def test_build_enhanced_stock_checker_adds_common_commodity_stock():
    checker = bench.build_enhanced_stock_checker(
        Namespace(enhanced_stock="zinc", disable_common_stock=False, disable_vendor_stock=True)
    )

    assert checker is not None
    assert checker("N") is True
    assert checker("O=O") is True


def test_build_enhanced_stock_checker_can_disable_common_commodity_stock(monkeypatch):
    import cascade_planner.cascadeboard.zinc_stock as zinc_stock

    monkeypatch.setattr(zinc_stock, "is_in_zinc_stock", lambda smi: False)
    checker = bench.build_enhanced_stock_checker(
        Namespace(enhanced_stock="zinc", disable_common_stock=True, disable_vendor_stock=True)
    )

    assert checker is not None
    assert checker("N") is False


def test_build_enhanced_stock_checker_adds_vendor_stock(tmp_path, monkeypatch):
    from cascade_planner.cascadeboard.vendor_stock import build_vendor_stock_index

    csv_path = tmp_path / "stock.csv"
    index_path = tmp_path / "stock.sqlite"
    csv_path.write_text("smiles\nCC(=O)O\n", encoding="utf-8")
    build_vendor_stock_index(csv_path=csv_path, sqlite_path=index_path)
    import cascade_planner.cascadeboard.zinc_stock as zinc_stock

    monkeypatch.setattr(zinc_stock, "is_in_zinc_stock", lambda smi: False)
    checker = bench.build_enhanced_stock_checker(
        Namespace(
            enhanced_stock="zinc",
            disable_common_stock=True,
            disable_vendor_stock=False,
            vendor_stock_index=index_path,
        )
    )

    assert checker is not None
    assert checker("CC(=O)[O-]") is True


def test_build_enhanced_stock_checker_can_add_semisynthesis_stock(monkeypatch):
    import cascade_planner.cascadeboard.zinc_stock as zinc_stock
    from cascade_planner.baselines.semisynthesis_rescue import TEN_DEACETYLBACCATIN_III

    monkeypatch.setattr(zinc_stock, "is_in_zinc_stock", lambda smi: False)
    checker = bench.build_enhanced_stock_checker(
        Namespace(
            enhanced_stock="zinc",
            disable_common_stock=True,
            disable_vendor_stock=True,
            enable_semisynthesis_stock=True,
        )
    )

    assert checker is not None
    assert checker(TEN_DEACETYLBACCATIN_III) is True
    assert checker("CCCCCCCCCCCCCCCC") is False


def test_build_enhanced_stock_checker_can_add_chemical_anchor_stock(monkeypatch):
    import cascade_planner.cascadeboard.zinc_stock as zinc_stock
    from cascade_planner.baselines.chemical_anchor_rescue import BENZOTHIAZOLE_DIOL_CORE

    monkeypatch.setattr(zinc_stock, "is_in_zinc_stock", lambda smi: False)
    checker = bench.build_enhanced_stock_checker(
        Namespace(
            enhanced_stock="zinc",
            disable_common_stock=True,
            disable_vendor_stock=True,
            enable_semisynthesis_stock=False,
            enable_chemical_anchor_stock=True,
        )
    )

    assert checker is not None
    assert checker(BENZOTHIAZOLE_DIOL_CORE) is True
    assert checker("CCCCCCCCCCCCCCCC") is False


def test_native_route_payload_adds_plausibility_audit():
    route = RouteCandidate(
        target_smiles="CCCCCCCCCCCC",
        solved=True,
        steps=[
            RouteStepCandidate(
                product_smiles="CCCCCCCCCCCC",
                reactant_smiles=["C"],
                rxn_smiles="C>>CCCCCCCCCCCC",
                source_model="ChemEnzyRetroPlanner",
            )
        ],
    )

    payload = bench.native_route_payload(route)

    assert payload["route_solved"] is True
    assert payload["route_plausibility_passed"] is False
    assert payload["route_plausibility"]["passed"] is False
    assert "large_unexplained_carbon_gain" in payload["route_plausibility"]["reasons"]


def test_summarize_counts_plausibility_audited_native_routes():
    rows = [
        {
            "run": "native_chemenzy",
            "label": 0,
            "ok": True,
            "route_count": 1,
            "solved_routes": 1,
            "progressive_routes": 1,
            "enzyme_routes": 0,
            "sp_v1_accepted_enzyme_routes": 0,
            "enzyme_proposal_calls": 0,
            "enzyme_proposal_candidates": 0,
            "mean_steps": 1,
            "elapsed_s": 1.0,
            "failure_categories": [],
            "routes": [{"route_solved": True, "route_plausibility_passed": False}],
            "stats": {"proposal_source_stats": {}},
        },
        {
            "run": "native_chemenzy",
            "label": 0,
            "ok": True,
            "route_count": 1,
            "solved_routes": 1,
            "progressive_routes": 1,
            "enzyme_routes": 0,
            "sp_v1_accepted_enzyme_routes": 0,
            "enzyme_proposal_calls": 0,
            "enzyme_proposal_candidates": 0,
            "mean_steps": 1,
            "elapsed_s": 1.0,
            "failure_categories": [],
            "routes": [{"route_solved": True, "route_plausibility": {"passed": True}}],
            "stats": {"proposal_source_stats": {}},
        },
    ]

    summary = bench.summarize(rows)["native_chemenzy"]

    assert summary["targets_with_solved_route"] == 2
    assert summary["targets_with_plausibility_audited_route"] == 2
    assert summary["targets_with_plausibility_passed_route"] == 1
    assert summary["targets_with_plausibility_passed_solved_route"] == 1


def test_summarize_counts_sp_v1_accepted_enzyme_routes():
    rows = [
        {
            "run": "enhanced_route_tree",
            "label": 1,
            "ok": True,
            "route_count": 1,
            "solved_routes": 0,
            "progressive_routes": 0,
            "enzyme_routes": 1,
            "sp_v1_accepted_enzyme_routes": 1,
            "enzyme_proposal_calls": 2,
            "enzyme_proposal_candidates": 3,
            "mean_steps": 1,
            "elapsed_s": 1.0,
            "failure_categories": [],
            "routes": [
                {
                    "route_tree_search_status": "timeout_frontier",
                    "has_enzyme_step": True,
                    "has_sp_v1_accepted_enzyme_step": True,
                }
            ],
            "stats": {"enzyme_sp_verifier_rejections": 1, "proposal_source_stats": {}},
        }
    ]

    summary = bench.summarize(rows)["enhanced_route_tree"]

    assert summary["targets_with_enzyme_route"] == 1
    assert summary["targets_with_sp_v1_accepted_enzyme_route"] == 1
    assert summary["timeout_frontier_routes"] == 1


def test_summarize_counts_native_chemical_rescue():
    rows = [
        {
            "run": "enhanced_route_tree",
            "label": 0,
            "ok": True,
            "route_count": 1,
            "solved_routes": 1,
            "progressive_routes": 1,
            "enzyme_routes": 0,
            "sp_v1_accepted_enzyme_routes": 0,
            "enzyme_proposal_calls": 0,
            "enzyme_proposal_candidates": 0,
            "mean_steps": 2,
            "elapsed_s": 3.0,
            "failure_categories": [],
            "routes": [{"route_tree_search_status": "native_chemical_rescue"}],
            "stats": {
                "enzyme_sp_verifier_rejections": 0,
                "native_chemical_rescue": {
                    "attempted": True,
                    "accepted_routes": 1,
                    "solved_routes": 1,
                },
                "proposal_source_stats": {},
            },
        }
    ]

    summary = bench.summarize(rows)["enhanced_route_tree"]

    assert summary["native_chemical_rescue_attempts"] == 1
    assert summary["native_chemical_rescue_solved_targets"] == 1
    assert summary["native_chemical_rescue_routes"] == 1


def test_native_chemical_rescue_marks_gate_rejections_without_filtering():
    adapter = _FakeNativeRescueAdapter(
        RouteCandidate(
            target_smiles="CCCCCCCCCCCC",
            solved=True,
            steps=[
                RouteStepCandidate(
                    product_smiles="CCCCCCCCCCCC",
                    reactant_smiles=["C"],
                    rxn_smiles="C>>CCCCCCCCCCCC",
                    source_model="ChemEnzyRetroPlanner",
                )
            ],
        )
    )

    routes, metadata = bench.run_native_chemical_rescue(
        {"target_smiles": "CCCCCCCCCCCC", "label": 0},
        adapter=adapter,
        args=_native_rescue_args(require_gate=False),
    )

    assert len(routes) == 1
    assert metadata["accepted_routes"] == 1
    assert metadata["proposal_gate_rejected_routes"] == 1
    assert routes[0]["native_chemical_rescue_proposal_gate"]["hard_reject"] is True


def test_native_chemical_rescue_can_require_gate_pass():
    adapter = _FakeNativeRescueAdapter(
        RouteCandidate(
            target_smiles="CCCCCCCCCCCC",
            solved=True,
            steps=[
                RouteStepCandidate(
                    product_smiles="CCCCCCCCCCCC",
                    reactant_smiles=["C"],
                    rxn_smiles="C>>CCCCCCCCCCCC",
                    source_model="ChemEnzyRetroPlanner",
                )
            ],
        )
    )

    routes, metadata = bench.run_native_chemical_rescue(
        {"target_smiles": "CCCCCCCCCCCC", "label": 0},
        adapter=adapter,
        args=_native_rescue_args(require_gate=True),
    )

    assert routes == []
    assert metadata["accepted_routes"] == 0
    assert metadata["proposal_gate_required"] is True
    assert metadata["proposal_gate_rejected_routes"] == 1
    assert "large_unexplained_carbon_gain" in metadata["proposal_gate_reason_counts"]


class _FakeNativeRescueAdapter:
    def __init__(self, route):
        self.route = route

    def run_target(self, config):
        return BaselineRunResult(
            target_smiles=config.target_smiles,
            backend="fake",
            routes=[self.route],
            raw_backend_metadata={"elapsed_s": 0.01},
        )


def _native_rescue_args(*, require_gate: bool):
    return Namespace(
        iterations=1,
        max_depth=1,
        expansion_topk=1,
        gpu=-1,
        native_chemical_rescue_timeout_s=1.0,
        n_results=3,
        native_chemical_rescue_one_step_models="graphfp_models.USPTO-full_remapped",
        native_chemical_rescue_require_proposal_gate=require_gate,
    )

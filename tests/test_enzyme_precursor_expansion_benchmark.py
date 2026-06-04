import scripts.run_enzyme_precursor_expansion_benchmark as bench


def test_extract_accepted_enzyme_precursors_filters_selected_sp_v1_enzyme_steps():
    rows = [
        {
            "target_smiles": "PRODUCT",
            "target_canonical": "PRODUCT",
            "label": 1,
            "label_source": "probe",
            "run": "enhanced_route_tree",
            "routes": [
                {
                    "route_tree_search_status": "timeout_frontier",
                    "steps": [
                        {
                            "source": "enzexpand",
                            "ec": "1.x",
                            "product": "PRODUCT",
                            "main_reactant": "CCO",
                            "reaction_smiles": "CCO>>PRODUCT",
                            "enzyme_sp_verifier_v1": {"accepted": True, "score": 0.8, "threshold": 0.3},
                        },
                        {
                            "source": "enzexpand",
                            "ec": "1.x",
                            "product": "PRODUCT",
                            "main_reactant": "CCN",
                            "reaction_smiles": "CCN>>PRODUCT",
                            "enzyme_sp_verifier_v1": {"accepted": False, "score": 0.1, "threshold": 0.3},
                        },
                        {
                            "source": "retrochimera",
                            "product": "PRODUCT",
                            "main_reactant": "CCC",
                            "reaction_smiles": "CCC>>PRODUCT",
                            "enzyme_sp_verifier_v1": {"accepted": True, "score": 0.9, "threshold": 0.3},
                        },
                    ],
                }
            ],
        }
    ]

    subgoals = bench.extract_accepted_enzyme_precursors(rows)

    assert len(subgoals) == 1
    assert subgoals[0]["subgoal_smiles"] == "CCO"
    assert subgoals[0]["enzyme_step_source"] == "enzexpand"
    assert subgoals[0]["enzyme_sp_v1_score"] == 0.8


def test_extract_accepted_enzyme_precursors_excludes_carrier_like_by_default():
    carrier = "CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O)NCCSC(=O)C"
    rows = [
        {
            "target_smiles": "PRODUCT",
            "label": 1,
            "run": "enhanced_route_tree",
            "routes": [
                {
                    "steps": [
                        {
                            "source": "enzyme_precedent",
                            "ec": "2.3.1.-",
                            "product": "PRODUCT",
                            "main_reactant": carrier,
                            "reaction_smiles": f"{carrier}>>PRODUCT",
                            "enzyme_sp_verifier_v1": {"accepted": True, "score": 0.9, "threshold": 0.3},
                        }
                    ],
                }
            ],
        }
    ]

    assert bench.extract_accepted_enzyme_precursors(rows) == []
    included = bench.extract_accepted_enzyme_precursors(rows, include_carrier_like=True)

    assert len(included) == 1
    assert included[0]["carrier_like"] is True
    assert "polyphosphate" in included[0]["carrier_like_reasons"]


def test_summarize_subgoal_rows_counts_closure_and_timeout_frontiers():
    rows = [
        {
            "subgoal_in_stock_initial": True,
            "route_count": 0,
            "solved_routes": 0,
            "progressive_routes": 0,
            "elapsed_s": 0.1,
            "stats": {"search_stop_reason": "initial_stock"},
            "routes": [],
        },
        {
            "subgoal_in_stock_initial": False,
            "route_count": 1,
            "solved_routes": 1,
            "progressive_routes": 1,
            "mean_steps": 2,
            "elapsed_s": 1.0,
            "stats": {
                "search_stop_reason": "stock_closed",
                "route_tree_runtime_bottlenecks": [],
                "proposal_source_stats": {"retrochimera": {"calls": 2, "final_returned": 3}},
            },
            "routes": [{"route_tree_search_status": "stock_closed", "source_counts": {"retrochimera": 1}}],
        },
        {
            "subgoal_in_stock_initial": False,
            "route_count": 1,
            "solved_routes": 0,
            "progressive_routes": 0,
            "mean_steps": 1,
            "elapsed_s": 2.0,
            "stats": {
                "search_stop_reason": "hard_timeout",
                "route_tree_runtime_bottlenecks": ["proposal_slow"],
                "proposal_source_stats": {
                    "chemtemplates": {"calls": 1, "final_returned": 0, "kept_returned": 4, "raw_returned": 4}
                },
            },
            "routes": [{"route_tree_search_status": "timeout_frontier", "source_counts": {"uspto_template": 1}}],
        },
    ]

    summary = bench.summarize_subgoal_rows(rows)

    assert summary["subgoals"] == 3
    assert summary["subgoals_initial_stock"] == 1
    assert summary["subgoals_with_routes"] == 2
    assert summary["subgoals_with_solved_route"] == 1
    assert summary["subgoals_closed_by_stock_or_route"] == 2
    assert summary["timeout_frontier_routes"] == 1
    assert summary["proposal_source_candidates"] == {"retrochimera": 3, "chemtemplates": 0}
    assert summary["proposal_source_outputs"] == {"retrochimera": 3, "chemtemplates": 4}
    assert summary["selected_step_source_counts"] == {"retrochimera": 1, "uspto_template": 1}
    assert summary["runtime_bottlenecks"] == {"proposal_slow": 1}


def test_summarize_native_and_hybrid_rows_counts_subplanner_closure():
    route_tree_rows = [
        {"subgoal_canonical": "A", "subgoal_in_stock_initial": False, "solved_routes": 0},
        {"subgoal_canonical": "B", "subgoal_in_stock_initial": True, "solved_routes": 0},
    ]
    native_rows = [
        {
            "subgoal_canonical": "A",
            "native_solved": True,
            "native_route_count": 2,
            "native_mean_steps": 3,
            "native_elapsed_s": 1.5,
            "native_failure_categories": [],
        },
        {
            "subgoal_canonical": "B",
            "native_solved": False,
            "native_route_count": 0,
            "native_mean_steps": 0,
            "native_elapsed_s": None,
            "native_failure_categories": ["no_route_found"],
        },
    ]

    native_summary = bench.summarize_native_subplanner_rows(native_rows)
    hybrid = bench.summarize_hybrid_rows(route_tree_rows, native_rows)

    assert native_summary["native_subgoals_solved"] == 1
    assert native_summary["native_total_routes"] == 2
    assert native_summary["native_failure_categories"] == {"no_route_found": 1}
    assert hybrid["hybrid_closed_by_route_tree_or_native"] == 2
    assert hybrid["route_tree_solved_or_initial_stock"] == 1
    assert hybrid["native_subplanner_solved"] == 1


def test_chemical_only_engine_keeps_only_chemical_sources():
    engine = {
        "retrochimera": object(),
        "chemtemplates": object(),
        "chem_enzy_onestep": object(),
        "enzyformer": object(),
        "enzexpand": object(),
        "enzyme_precedent": object(),
    }

    filtered = bench.chemical_only_engine(engine)

    assert set(filtered) == {"retrochimera", "chemtemplates", "chem_enzy_onestep"}


def test_load_route_rows_accepts_multiple_jsonl_files(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text('{"target_smiles":"A"}\n', encoding="utf-8")
    second.write_text('{"target_smiles":"B"}\n', encoding="utf-8")

    rows = bench.load_route_rows([first, second])

    assert [row["target_smiles"] for row in rows] == ["A", "B"]
    assert rows[0]["_route_rows_path"] == str(first)
    assert rows[1]["_route_rows_path"] == str(second)


def test_closure_env_values_can_cap_chemical_template_scan():
    class Args:
        enable_chem_enzy_onestep = True
        chem_enzy_onestep_topk = 12
        chem_enzy_onestep_min_budget = 2
        route_tree_timeout_s = 10.0
        stock_aware_action_rerank = True
        exact_stock_reactant_bonus = 1.5
        full_stock_action_bonus = 2.5
        normalized_stock_reactant_bonus = 0.75
        normalized_stock_full_action_bonus = 1.25
        no_progress_single_reactant_penalty = 0.6
        closure_stock_rescue = False
        closure_stock_rescue_remaining_depth = 0
        closure_stock_rescue_max_retries = 12
        closure_stock_rescue_budget_multiplier = 3.0
        closure_stock_rescue_budget_cap = 30
        closure_stock_rescue_min_actions = 12
        closure_stock_rescue_require_stock_gain = False
        chem_template_max_per_query = 40
        chem_template_max_templates = 1000

    values = bench.closure_env_values(Args())

    assert values["AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS"] == "1"
    assert values["AUTOPLANNER_CHEMENZY_ONESTEP_TOPK"] == "12"
    assert values["AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_QUERY"] == "40"
    assert values["AUTOPLANNER_CHEM_TEMPLATES_MAX_TEMPLATES"] == "1000"
    assert values["AUTOPLANNER_ROUTE_TREE_EXACT_STOCK_REACTANT_BONUS"] == "1.5"
    assert values["AUTOPLANNER_ROUTE_TREE_NORMALIZED_STOCK_REACTANT_BONUS"] == "0.75"
    assert values["AUTOPLANNER_ROUTE_TREE_NORMALIZED_STOCK_FULL_ACTION_BONUS"] == "1.25"
    assert values["AUTOPLANNER_ROUTE_TREE_NO_PROGRESS_SINGLE_REACTANT_PENALTY"] == "0.6"
    assert "AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE" not in values


def test_native_subplanner_row_normalizes_baseline_result():
    from cascade_planner.baselines.route_contract import BaselineRunResult, RouteCandidate, RouteStepCandidate

    result = BaselineRunResult(
        target_smiles="CCO",
        backend="ChemEnzyRetroPlanner",
        routes=[
            RouteCandidate(
                target_smiles="CCO",
                solved=True,
                steps=[RouteStepCandidate(product_smiles="CCO", reactant_smiles=["C"], rxn_smiles="C>>CCO")],
            )
        ],
        raw_backend_metadata={"elapsed_s": 1.2, "total_elapsed_s": 1.3},
    )

    row = bench.native_subplanner_row({"subgoal_canonical": "CCO", "subgoal_smiles": "CCO"}, result, elapsed_total_s=2.0)

    assert row["native_ok"] is True
    assert row["native_solved"] is True
    assert row["native_route_count"] == 1
    assert row["native_mean_steps"] == 1
    assert row["native_elapsed_s"] == 1.2

    outcomes = bench.native_subplanner_outcomes([row])
    assert outcomes[0]["native_solved"] is True
    assert outcomes[0]["native_route_count"] == 1


def test_closure_env_values_can_enable_root_stock_rescue():
    class Args:
        enable_chem_enzy_onestep = True
        chem_enzy_onestep_topk = 12
        chem_enzy_onestep_min_budget = 2
        route_tree_timeout_s = 10.0
        max_depth = 4
        stock_aware_action_rerank = True
        exact_stock_reactant_bonus = 1.5
        full_stock_action_bonus = 2.5
        closure_stock_rescue = True
        closure_stock_rescue_remaining_depth = 0
        closure_stock_rescue_max_retries = 9
        closure_stock_rescue_budget_multiplier = 2.5
        closure_stock_rescue_budget_cap = 28
        closure_stock_rescue_min_actions = 10
        closure_stock_rescue_require_stock_gain = True
        chem_template_max_per_query = 0
        chem_template_max_templates = 0

    values = bench.closure_env_values(Args())

    assert values["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE"] == "1"
    assert values["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REMAINING_DEPTH"] == "4"
    assert values["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_MAX_RETRIES"] == "9"
    assert values["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_BUDGET_MULTIPLIER"] == "2.5"
    assert values["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_BUDGET_CAP"] == "28"
    assert values["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_MIN_ACTIONS"] == "10"
    assert values["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REQUIRE_STOCK_GAIN"] == "1"

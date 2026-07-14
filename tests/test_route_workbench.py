from __future__ import annotations

from copy import deepcopy

import pytest

from cascade_planner.application.route_workbench import (
    MAX_VISIBLE_ROUTES,
    ROUTE_WORKBENCH_DELTA_SCHEMA,
    ROUTE_WORKBENCH_SCHEMA,
    RouteWorkbenchProjectionError,
    compile_route_workbench,
    compile_route_workbench_delta,
)
from cascade_planner.harness.route_forest_delivery import (
    build_route_forest_delivery_payload,
    route_forest_delivery_integrity_reasons,
)
from cascade_planner.harness.v4_route_workbench import (
    compile_v4_route_forest,
    render_v4_route_workbench_html,
)
from cascade_planner.harness.v4_route_display import compile_route_display_rows


def _graph() -> dict:
    return {
        "schema_version": "canonical_retrosynthesis_hypergraph.v1",
        "run_id": "ui-run",
        "target_name": "example target",
        "target_molecule_id": "m:target",
        "revision": 3,
        "scientific_sha256": "graph-3",
        "molecules": {
            "m:target": {
                "canonical_smiles": "CCOC(C)=O",
                "is_leaf": False,
                "stock_observation_ids": [],
            },
            "m:ethanol": {
                "canonical_smiles": "CCO",
                "is_leaf": True,
                "active_stock_observation_id": "stock:ethanol",
                "stock_observation_ids": ["stock:ethanol"],
                "stock_closed": True,
            },
            "m:acid": {
                "canonical_smiles": "CC(=O)O",
                "is_leaf": True,
                "active_stock_observation_id": "stock:acid",
                "stock_observation_ids": ["stock:acid"],
                "stock_closed": True,
            },
        },
        "edges": {
            "edge:ester": {
                "product_molecule_id": "m:target",
                "precursor_molecule_ids": ["m:acid", "m:ethanol"],
                "origin_records": [{"origin_kind": "codex_global_director"}],
                "reaction_proofs": [{"accepted": True, "proof_digest": "proof"}],
            }
        },
        "source_bindings": {
            "source:patent": {
                "source_kind": "patent",
                "source_ref": "US-example",
                "title": "Example patent",
            }
        },
        "exact_records": {"record:1": {"location_ref": "Example 1"}},
        "stock_observations": {
            "stock:ethanol": {"supplier": "fixture", "catalog_number": "E-1"},
            "stock:acid": {"supplier": "fixture", "catalog_number": "A-1"},
        },
        "conflicts": {
            "conflict:resolved": {"status": "resolved", "subject_id": "edge:ester"}
        },
        "hypotheses": {
            "hypothesis:open": {
                "status": "frontier_candidate",
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "route_family_ids": ["family:other"],
                "route_diversity_gain": 0.8,
                "origin_records": [{"origin_kind": "chemenzy"}],
            }
        },
        "delta": {
            "rejected": [
                {"kind": "reaction_edge", "reasons": ["element_balance_invalid"]}
            ]
        },
    }


def _route(index: int, *, complete: bool = True) -> dict:
    return {
        "route_id": f"route:{index}",
        "route_family_id": f"family:{index}",
        "strategy": f"late disconnection {index}",
        "edge_ids": ["edge:ester"],
        "leaf_molecule_ids": ["m:acid", "m:ethanol"],
        "root_edge_ids": ["edge:ester"],
        "module_selections": {},
        "minimum_edge_proof_level": 3,
        "all_edges_proven": True,
        "stock_closure_rate": 1.0,
        "independent_source_groups": ["patent:example", "paper:example"],
        "risk_score": 0.1 + index / 100,
        "convergence_score": 0.0,
        "complete": complete,
        "pareto_optimal": index < 2,
    }


def _portfolio(*, route_count: int = 2) -> dict:
    routes = [_route(index) for index in range(route_count)]
    return {
        "schema_version": "proof_stitched_route_portfolio.v1",
        "graph_revision": 3,
        "graph_scientific_sha256": "graph-3",
        "content_sha256": "portfolio-3",
        "proof_policy": {
            "stock_boundary": "benchmark_search",
            "minimum_edge_proof_level": 3,
        },
        "selected_routes": routes,
        "edge_proofs": {
            "edge:ester": {
                "achieved_level": 3,
                "accepted": True,
                "reaction_validated": True,
                "exact_source_bound": True,
                "source_binding_ids": ["source:patent"],
                "exact_record_ids": ["record:1"],
                "conflict_ids": [],
                "reasons": [],
            }
        },
        "leaf_proofs": {
            "m:acid": {"accepted": True},
            "m:ethanol": {"accepted": True},
        },
        "route_modules": [
            {
                "module_id": "module:ester",
                "route_family_id": "family:0",
                "product_molecule_id": "m:target",
                "alternatives": [{"edge_id": "edge:ester"}],
            }
        ],
        "deficits": [],
        "metrics": {
            "shared_intermediates": {
                "m:ethanol": ["route:0", "route:1"],
            }
        },
        "closeout": {"decision": "accepted", "complete_route_count": route_count},
        "accepted": True,
    }


def test_workbench_is_bounded_proof_aware_and_keeps_hypotheses_separate() -> None:
    projection = compile_route_workbench(_graph(), _portfolio(route_count=7))

    assert projection["schema_version"] == ROUTE_WORKBENCH_SCHEMA
    assert projection["portfolio"]["route_count"] == MAX_VISIBLE_ROUTES
    assert projection["portfolio"]["display_limit"] == MAX_VISIBLE_ROUTES
    assert projection["views"]["hypotheses"]["count"] == 1
    assert projection["views"]["stock_closed"]["count"] == MAX_VISIBLE_ROUTES
    assert set(projection["views"]["hypotheses"]["hypothesis_ids"]) == {
        "hypothesis:open"
    }
    assert projection["hypotheses"]["hypothesis:open"]["proof_level"] == 0
    assert projection["routes"]["route:0"]["proof_level"] == 3
    assert projection["routes"]["route:0"]["stage"] == "stock_closed"
    assert projection["routes"]["route:0"]["closure_profile"] == "exploration_closed"
    assert projection["routes"]["route:0"]["process_ready"] is False
    assert projection["routes"]["route:0"]["condition_complete"] is False
    assert projection["routes"]["route:0"]["proof_vector"]["conditions"] == "missing"
    assert projection["portfolio"]["closure_profile"] == "exploration_closed"
    assert projection["portfolio"]["process_ready"] is False
    assert "source:patent" in projection["edges"]["edge:ester"]["badges"]
    assert projection["shared_intermediates"]["m:ethanol"]["render_once"] is True
    assert projection["views"]["condition_complete"]["count"] == 0
    assert projection["edges"]["edge:ester"]["proof_vector"] == {
        "schema_version": "retrosynthesis_proof_vector.v1",
        "identity": "source_exact",
        "reaction": "host_validated",
        "conditions": "missing",
        "sources": "none",
        "stock": "not_applicable_to_edge",
        "process": "blocked",
        "condition_record_count": 0,
        "exact_procedure_record_count": 0,
        "complete_procedure_record_count": 0,
        "condition_completeness": "missing",
        "semantics": {
            "axes_are_independent": True,
            "exact_structure_does_not_imply_exact_conditions": True,
            "display_projection_grants_no_authority": True,
        },
    }
    assert projection["layout"]["stable_ids"] is True
    assert projection["semantics"]["canonical_graph_is_authority"] is True


def test_condition_complete_requires_replayable_and_complete_source_procedure() -> None:
    graph = deepcopy(_graph())
    graph["exact_records"]["record:1"] = {
        "location_ref": "Example 1",
        "conditions": {
            "reagents": ["base"],
            "solvent": "THF",
            "temperature_c": 20,
            "time": "2 h",
        },
        "authority_scope": "source_exact_structure_observation",
        "procedure_authority_scope": "source_exact_reaction_procedure",
        "condition_completeness": {
            "schema_version": "reaction_condition_completeness.v1",
            "complete": True,
            "missing_required_groups": [],
        },
    }

    projection = compile_route_workbench(graph, _portfolio())

    edge_vector = projection["edges"]["edge:ester"]["proof_vector"]
    route = projection["routes"]["route:0"]
    assert edge_vector["conditions"] == "source_exact"
    assert edge_vector["condition_completeness"] == "complete"
    assert edge_vector["process"] == "procedure_bound_candidate"
    assert route["condition_complete"] is True
    assert route["proof_vector"]["condition_completeness"] == "complete"
    assert route["process_ready"] is False


def test_workbench_projects_independent_campaign_gates_without_granting_proof() -> None:
    projection = compile_route_workbench(
        _graph(),
        _portfolio(),
        campaign_summary={
            "gates": {
                "B0_blind_input": True,
                "B1_global_multi_route": True,
                "B2_host_validated_routes": True,
                "B3_exact_multi_source": False,
                "B4_stock_boundary": True,
                "B5_configured_portfolio_acceptance": True,
            },
            "highest_contiguous_gate": "B2",
            "resource_envelope": {"within_budget": True},
            "model_cost": {"model_invocations": 1},
            "stop_decision": {"decision": "completed"},
            "claim": {"exact_multi_source_grade": False},
            "current_disposition": {"state": "accepted"},
        },
    )

    summary = projection["campaign_summary"]
    assert summary["available"] is True
    assert summary["highest_contiguous_gate"] == "B2"
    assert summary["gates"]["B3_exact_multi_source"] is False
    assert summary["gates"]["B4_stock_boundary"] is True
    assert summary["current_disposition"]["state"] == "accepted"
    assert summary["semantics"]["measurement_only"] is True

    forest = compile_v4_route_forest(projection)
    payload = build_route_forest_delivery_payload(forest)
    assert payload["campaign_summary"] == summary
    assert route_forest_delivery_integrity_reasons(payload, source_forest=forest) == []


def test_workbench_inspectors_expose_proof_sources_stock_rejections_and_conflicts() -> None:
    projection = compile_route_workbench(_graph(), _portfolio())

    edge = projection["inspectors"]["edges"]["edge:ester"]
    assert edge["proof"]["reaction_validated"] is True
    assert edge["sources"][0]["source_kind"] == "patent"
    assert edge["exact_records"][0]["location_ref"] == "Example 1"
    assert edge["condition_status"] == "missing"
    assert edge["condition_gap"] == "no_replayable_reaction_conditions_bound"
    molecule = projection["inspectors"]["molecules"]["m:ethanol"]
    assert molecule["stock_closed"] is True
    assert molecule["stock_observations"][0]["catalog_number"] == "E-1"
    assert projection["inspectors"]["rejections"][0]["reasons"] == [
        "element_balance_invalid"
    ]
    assert "conflict:resolved" in projection["inspectors"]["conflicts"]


def test_workbench_delta_upserts_entities_and_requires_matching_run() -> None:
    first = compile_route_workbench(_graph(), _portfolio())
    graph = _graph()
    graph["revision"] = 4
    graph["scientific_sha256"] = "graph-4"
    graph["molecules"]["m:ethanol"]["stock_closed"] = False
    portfolio = _portfolio()
    portfolio["graph_revision"] = 4
    portfolio["graph_scientific_sha256"] = "graph-4"
    portfolio["content_sha256"] = "portfolio-4"
    portfolio["leaf_proofs"]["m:ethanol"] = {"accepted": False}
    current = compile_route_workbench(graph, portfolio)

    delta = compile_route_workbench_delta(first, current)
    assert delta["schema_version"] == ROUTE_WORKBENCH_DELTA_SCHEMA
    assert delta["from_graph_revision"] == 3
    assert delta["to_graph_revision"] == 4
    assert "m:ethanol" in delta["upserts"]["molecules"]
    assert delta["base_sha256"] == first["content_sha256"]
    assert delta["result_sha256"] == current["content_sha256"]
    assert delta["empty"] is False

    wrong = deepcopy(first)
    wrong["run_id"] = "different-run"
    with pytest.raises(RouteWorkbenchProjectionError, match="delta_run_mismatch"):
        compile_route_workbench_delta(wrong, current)


def test_workbench_rejects_stale_or_mismatched_portfolio() -> None:
    portfolio = _portfolio()
    portfolio["graph_revision"] = 2
    with pytest.raises(RouteWorkbenchProjectionError, match="graph_revision_mismatch"):
        compile_route_workbench(_graph(), portfolio)

    portfolio = _portfolio()
    portfolio["graph_scientific_sha256"] = "other"
    with pytest.raises(RouteWorkbenchProjectionError, match="graph_digest_mismatch"):
        compile_route_workbench(_graph(), portfolio)


def test_v4_workbench_adapter_renders_bounded_routes_and_separate_hypotheses() -> None:
    projection = compile_route_workbench(_graph(), _portfolio())
    forest = compile_v4_route_forest(projection)
    payload = build_route_forest_delivery_payload(forest)

    assert route_forest_delivery_integrity_reasons(payload, source_forest=forest) == []
    portfolio_lanes = [
        value
        for value in payload["branch_lanes"]["lanes"]
        if value["kind"] == "proof_eligible_portfolio_route"
    ]
    hypothesis_lanes = [
        value
        for value in payload["branch_lanes"]["lanes"]
        if value["kind"] == "retrosynthetic_proposal"
    ]
    assert len(portfolio_lanes) == 2
    assert len(hypothesis_lanes) == 1
    assert all("expanded" in value["stage_memberships"] for value in portfolio_lanes)
    assert all("reaction" in value["stage_memberships"] for value in portfolio_lanes)
    assert all("stock" in value["stage_memberships"] for value in portfolio_lanes)
    assert all(value["solved"] is False for value in forest["branches"] if value["kind"] == "proof_eligible_portfolio_route")
    assert all(value["executable"] is False for value in forest["branches"] if value["kind"] == "proof_eligible_portfolio_route")
    assert all(value["completion_label"] == "搜索边界闭合" for value in forest["branches"] if value["kind"] == "proof_eligible_portfolio_route")
    route_steps = [
        value for value in forest["steps"] if value.get("branch_id") == "route:0"
    ]
    assert route_steps[0]["condition_status"] == "missing"
    assert "精确结构来源不等于实验条件" in route_steps[0]["condition_summary"]
    assert route_steps[0]["trust_vector"]["conditions"] == 0.0
    assert hypothesis_lanes[0]["stage_memberships"] == ["suggestion"]

    html = render_v4_route_workbench_html(projection)
    assert "autoplanner.route-forest-ui.v4" in html
    assert "MAX_PORTFOLIO_ROUTES = 5" in html
    assert "__AUTOPLANNER_ROUTE_PERF__" in html
    assert "translate3d" not in html


def test_v4_workbench_preserves_repeated_reagent_stoichiometry_without_duplicate_ids() -> None:
    graph = _graph()
    graph["edges"]["edge:ester"]["precursor_molecule_ids"] = [
        "m:acid",
        "m:ethanol",
        "m:ethanol",
    ]
    projection = compile_route_workbench(graph, _portfolio())
    forest = compile_v4_route_forest(projection)
    payload = build_route_forest_delivery_payload(forest)

    assert route_forest_delivery_integrity_reasons(payload, source_forest=forest) == []
    step = next(value for value in forest["steps"] if value["graph_step_id"] == "edge:ester")
    assert step["from_node_ids"] == ["m:acid", "m:ethanol"]
    assert step["stoichiometric_input_count"] == 3
    assert step["precursor_multiplicity"] == [
        {"molecule_node_id": "m:acid", "count": 1},
        {"molecule_node_id": "m:ethanol", "count": 2},
    ]
    assert {"label": "input multiplicity", "value": "C2H6O ×2"} in step[
        "conditions"
    ]


def test_v4_workbench_projects_only_full_restitched_replacement_routes() -> None:
    graph = _graph()
    graph["molecules"]["m:acetyl-chloride"] = {
        "canonical_smiles": "CC(=O)Cl",
        "is_leaf": True,
        "active_stock_observation_id": "stock:acetyl-chloride",
        "stock_observation_ids": ["stock:acetyl-chloride"],
        "stock_closed": True,
    }
    graph["stock_observations"]["stock:acetyl-chloride"] = {
        "supplier": "fixture",
        "catalog_number": "AC-1",
    }
    graph["edges"]["edge:ester-alt"] = {
        "product_molecule_id": "m:target",
        "precursor_molecule_ids": ["m:acetyl-chloride", "m:ethanol"],
        "origin_records": [{"origin_kind": "chemenzy"}],
        "reaction_proofs": [{"accepted": True, "proof_digest": "proof-alt"}],
    }

    portfolio = _portfolio(route_count=1)
    portfolio["selected_routes"][0]["module_selections"] = {
        "module:ester": "edge:ester"
    }
    portfolio["route_modules"][0]["alternatives"] = [
        {"edge_id": "edge:ester"},
        {"edge_id": "edge:ester-alt"},
    ]
    portfolio["edge_proofs"]["edge:ester-alt"] = {
        "achieved_level": 2,
        "accepted": True,
        "reaction_validated": True,
        "exact_source_bound": False,
        "source_binding_ids": [],
        "exact_record_ids": [],
        "conflict_ids": [],
        "reasons": [],
    }
    portfolio["leaf_proofs"]["m:acetyl-chloride"] = {"accepted": True}
    portfolio["route_candidates"] = [
        {
            **_route(10),
            "route_id": "route:0-alt",
            "route_family_id": "family:0",
            "strategy": "validated acyl chloride replacement",
            "edge_ids": ["edge:ester-alt"],
            "root_edge_ids": ["edge:ester-alt"],
            "leaf_molecule_ids": ["m:acetyl-chloride", "m:ethanol"],
            "module_selections": {"module:ester": "edge:ester-alt"},
            "minimum_edge_proof_level": 2,
            "all_edges_proven": True,
            "all_leaves_stock_closed": True,
            "complete": True,
        }
    ]

    projection = compile_route_workbench(graph, portfolio)

    assert projection["replacement_validation"]["candidate_count"] == 1
    assert projection["replacement_validation"]["validated_count"] == 1
    replacement_record = projection["replacement_validation"]["records"][0]
    replacement_route_id = replacement_record["replacement_route_id"]
    assert replacement_record["accepted"] is True
    assert replacement_record["replacement_edge_id"] == "edge:ester-alt"
    assert projection["replacement_routes"][replacement_route_id]["listed"] is False
    assert (
        projection["replacement_routes"][replacement_route_id]["underlying_route_id"]
        == "route:0-alt"
    )

    forest = compile_v4_route_forest(projection)
    payload = build_route_forest_delivery_payload(forest)

    assert route_forest_delivery_integrity_reasons(payload, source_forest=forest) == []
    replacement_branch = next(
        branch
        for branch in payload["branches"]
        if branch["branch_id"] == replacement_route_id
    )
    assert replacement_branch["kind"] == "validated_replacement_route"
    assert replacement_branch["listed"] is False
    assert replacement_branch["complete"] is True
    delivered_record = payload["replacement_validation"]["records"][0]
    assert delivered_record["accepted"] is True
    assert delivered_record["validated"] is True
    assert delivered_record["status"] == "route_revalidated"
    assert delivered_record["base_branch_id"] == "route:0"
    assert delivered_record["candidate_branch_id"] == replacement_route_id
    assert delivered_record["base_step_id"]
    assert delivered_record["candidate_step_id"]


def test_v4_display_uses_dag_stages_native_sources_and_auxiliary_roles() -> None:
    edges = {
        "edge:a": {
            "product_molecule_id": "m:i1",
            "precursor_molecule_ids": ["m:large-a", "m:small"],
            "origin_kinds": ["literature_replay"],
            "source_kinds": ["patent"],
        },
        "edge:b": {
            "product_molecule_id": "m:i2",
            "precursor_molecule_ids": ["m:large-b"],
            "origin_kinds": ["chemenzy"],
            "source_kinds": [],
        },
        "edge:c": {
            "product_molecule_id": "m:target",
            "precursor_molecule_ids": ["m:i1", "m:i2"],
            "origin_kinds": ["codex_global_director"],
            "source_kinds": ["paper_si"],
        },
    }
    inspectors = {
        "edge:a": {"exact_records": [{"claim_scope_id": "patent_c31"}]},
        "edge:b": {"exact_records": []},
        "edge:c": {"exact_records": [{"claim_scope_id": "T15_final"}]},
    }
    nodes = {
        "m:large-a": {"heavy_atom_count": 20},
        "m:large-b": {"heavy_atom_count": 18},
        "m:small": {"heavy_atom_count": 3},
        "m:i1": {"heavy_atom_count": 22},
        "m:i2": {"heavy_atom_count": 18},
        "m:target": {"heavy_atom_count": 40},
    }

    rows = compile_route_display_rows(
        ["edge:c", "edge:b", "edge:a"],
        edge_rows=edges,
        edge_inspectors=inspectors,
        nodes_by_id=nodes,
    )

    assert [row["edge_id"] for row in rows] == ["edge:a", "edge:b", "edge:c"]
    assert [row["stage_label"] for row in rows] == ["S1a", "S1b", "S2"]
    assert [row["retrosynthesis_label"] for row in rows] == ["R2a", "R2b", "R1"]
    assert rows[0]["display_label"] == "S1a · P-C31"
    assert rows[0]["producer_label"] == "文献重放"
    assert rows[0]["auxiliary_precursor_ids"] == ["m:small"]
    assert rows[2]["source_step_labels"] == ["S-T15"]

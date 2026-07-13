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
    assert "source:patent" in projection["edges"]["edge:ester"]["badges"]
    assert projection["shared_intermediates"]["m:ethanol"]["render_once"] is True
    assert projection["layout"]["stable_ids"] is True
    assert projection["semantics"]["canonical_graph_is_authority"] is True


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
        },
    )

    summary = projection["campaign_summary"]
    assert summary["available"] is True
    assert summary["highest_contiguous_gate"] == "B2"
    assert summary["gates"]["B3_exact_multi_source"] is False
    assert summary["gates"]["B4_stock_boundary"] is True
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

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
from cascade_planner.harness.v4_planned_route_branches import (
    append_planned_route_branches,
)
from cascade_planner.harness.v4_route_display import compile_route_display_rows
from cascade_planner.harness.v4_route_evidence_projection import PROOF_TIER


def test_v4_level_one_uses_the_canonical_structural_materialization_label() -> None:
    assert PROOF_TIER[1] == "L1_structural_materialized"


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
    assert projection["views"]["literature_grounded"]["count"] == MAX_VISIBLE_ROUTES
    assert projection["portfolio"]["achieved_profile"] == "literature_grounded"
    assert projection["portfolio"]["acceptance_profile_counts"] == {
        "exploration_closed": MAX_VISIBLE_ROUTES,
        "reaction_validated": MAX_VISIBLE_ROUTES,
        "literature_grounded": MAX_VISIBLE_ROUTES,
        "condition_complete": 0,
        "procurement_closed": 0,
        "process_ready": 0,
    }
    assert projection["routes"]["route:0"]["proof_vector"]["stock"] == "benchmark_hit"
    assert projection["edges"]["edge:ester"]["proof_vector"] == {
        "schema_version": "retrosynthesis_proof_vector.v1",
        "identity": "source_exact",
        "reaction": "host_validated",
        "conditions": "missing",
        "sources": "none",
        "stock": "not_applicable_to_edge",
        "process": "blocked",
        "condition_record_count": 0,
        "procedure_record_count": 0,
        "exact_procedure_record_count": 0,
        "complete_procedure_record_count": 0,
        "condition_missing_required_groups": [],
        "condition_completeness": "missing",
            "semantics": {
                "axes_are_independent": True,
                "exact_structure_does_not_imply_exact_conditions": True,
                "source_observed_conditions_do_not_grant_exact_identity": True,
                "display_projection_grants_no_authority": True,
            },
    }
    assert projection["layout"]["stable_ids"] is True
    assert projection["semantics"]["canonical_graph_is_authority"] is True


def test_workbench_exposes_enzyme_compression_and_mechanism_hypotheses() -> None:
    graph = _graph()
    graph["edges"]["edge:ester"]["route_innovations"] = [
        {
            "schema_version": "route_innovation.v1",
            "innovation_id": "innovation:enzyme",
            "kind": "biocatalytic_superstep",
            "chemical_step_equivalent_count": 4,
            "step_savings": 3,
            "enzyme": {"classes": ["acyltransferase"], "ec_numbers": []},
            "selectivity_objective": "chemoselective acylation",
            "authority_scope": "proposal_only",
            "not_reaction_proof": True,
        }
    ]
    graph["hypotheses"]["hypothesis:open"]["route_innovations"] = [
        {
            "schema_version": "route_innovation.v1",
            "innovation_id": "innovation:mechanism",
            "kind": "mechanism_extrapolation",
            "mechanistic_rationale": "one source-anchored oxidation step",
            "anchor": {"source_refs": ["doi:10.1000/anchor"]},
            "falsifiable_checks": ["LCMS"],
            "evidence_grade": "low_mechanistic_hypothesis",
        }
    ]
    portfolio = _portfolio(route_count=1)
    portfolio["edge_proofs"]["edge:ester"]["innovation_proof_gate"] = {
        "required": True,
        "accepted": False,
        "reasons": ["biocatalysis_validation_missing"],
    }
    portfolio["selected_routes"][0].update(
        {
            "physical_step_count": 1,
            "chemical_step_equivalent_count": 4,
            "net_step_savings": 3,
            "biocatalytic_superstep_count": 1,
            "mechanism_extrapolation_count": 0,
            "unvalidated_biocatalytic_edge_ids": ["edge:ester"],
        }
    )

    projection = compile_route_workbench(graph, portfolio)
    edge = projection["edges"]["edge:ester"]
    route = projection["routes"]["route:0"]
    hypothesis = projection["hypotheses"]["hypothesis:open"]

    assert "innovation:biocatalytic_superstep" in edge["badges"]
    assert "multi-step-compression" in edge["badges"]
    assert edge["innovation_proof_gate"]["accepted"] is False
    assert route["chemical_step_equivalent_count"] == 4
    assert route["net_step_savings"] == 3
    assert "biocatalytic-superstep" in route["badges"]
    assert hypothesis["innovation_kinds"] == ["mechanism_extrapolation"]

    forest = compile_v4_route_forest(projection)
    materialized_step = next(
        step for step in forest["steps"] if step.get("graph_step_id") == "edge:ester"
    )
    assert materialized_step["innovation_kinds"] == ["biocatalytic_superstep"]
    assert materialized_step["innovation_proof_gate"]["accepted"] is False
    route_branch = next(
        branch for branch in forest["branches"] if branch["branch_id"] == "route:0"
    )
    assert route_branch["chemical_step_equivalent_count"] == 4
    assert route_branch["net_step_savings"] == 3


def test_reported_route_remains_visible_when_edges_are_unresolved() -> None:
    graph = _graph()
    portfolio = _portfolio(route_count=1)
    route = portfolio["selected_routes"][0]
    route.update(
        {
            "complete": False,
            "all_edges_proven": False,
            "minimum_edge_proof_level": 0,
            "reported_in_source": True,
            "reported_source_refs": ["doi:10.1000/reported-route"],
        }
    )
    portfolio["edge_proofs"]["edge:ester"] = {
        "achieved_level": 0,
        "accepted": False,
        "reaction_validated": False,
        "exact_source_bound": False,
        "source_binding_ids": [],
        "exact_record_ids": [],
        "conflict_ids": [],
        "reasons": ["structure_translation_pending"],
    }
    portfolio["accepted"] = False

    projection = compile_route_workbench(graph, portfolio)
    displayed = projection["routes"]["route:0"]
    forest = compile_v4_route_forest(projection)
    branch = next(row for row in forest["branches"] if row["branch_id"] == "route:0")

    assert displayed["reported_in_source"] is True
    assert displayed["proof_level"] == 0
    assert "reported-candidate" in displayed["badges"]
    assert displayed["warning_codes"] == [
        "reported_route_contains_unresolved_edges"
    ]
    assert branch["kind"] == "reported_candidate_route"
    assert branch["advisory_only"] is True
    assert branch["solved"] is False
    assert branch["source_refs"] == ["doi:10.1000/reported-route"]
    assert forest["steps"][0]["proof_tier"] == "L0_advisory"


def test_closed_reported_route_keeps_closure_independent_from_proof() -> None:
    graph = _graph()
    for observation in graph["stock_observations"].values():
        observation.update(
            {
                "authority_scope": "benchmark_search_stock_observation",
                "accepted": True,
            }
        )
    portfolio = _portfolio(route_count=1)
    route = portfolio["selected_routes"][0]
    route.update(
        {
            "complete": True,
            "all_edges_proven": False,
            "minimum_edge_proof_level": 1,
            "reported_in_source": True,
            "reported_source_refs": ["doi:10.1000/reported-route"],
            "reported_step_count": 15,
            "planner_hypothesis_step_count": 5,
            "unproven_edge_ids": ["edge:ester"],
        }
    )
    portfolio["edge_proofs"]["edge:ester"].update(
        {
            "achieved_level": 1,
            "accepted": False,
            "reaction_validated": False,
            "exact_source_bound": False,
            "exact_record_ids": [],
            "independent_source_groups": [],
            "reasons": ["current_host_reaction_validation_missing"],
        }
    )
    portfolio["accepted"] = False

    projection = compile_route_workbench(graph, portfolio)
    displayed = projection["routes"]["route:0"]
    forest = compile_v4_route_forest(projection)
    branch = next(row for row in forest["branches"] if row["branch_id"] == "route:0")

    assert displayed["complete"] is True
    assert displayed["configured_boundary_closed"] is True
    assert displayed["closure_profile"] == "exploration_closed"
    assert displayed["search_closed"] is True
    assert displayed["process_ready"] is False
    assert displayed["proof_level_counts"] == {"1": 1}
    assert displayed["warning_codes"] == [
        "reported_route_contains_unresolved_edges"
    ]
    assert branch["kind"] == "reported_candidate_route"
    assert branch["complete"] is True
    assert branch["not_parent_route_proof"] is False
    assert branch["solved"] is False
    assert branch["route_state_label"] == (
        "路线已闭合 · 15 步文献报道 · 5 步规划待补证"
    )
    assert forest["frontier_ledger"]["counts"]["l0_break_suggestion_edges"] == 0
    assert forest["frontier_ledger"]["counts"]["l1_materialized_edges"] == 1
    assert forest["frontier_ledger"]["counts"]["benchmark_only_stock_leaves"] == 2
    assert forest["frontier_ledger"]["counts"]["procurement_boundary_leaves"] == 0
    assert forest["frontier_ledger"]["closure"]["any_benchmark_route_closed"] is True
    assert forest["frontier_ledger"]["closure"]["any_procurement_route_closed"] is False


def test_atom_balance_failure_is_a_red_step_with_structured_finding() -> None:
    graph = _graph()
    graph["edges"]["edge:ester"]["validation_findings"] = [
        {
            "finding_code": "atom_balance_violation",
            "severity": "blocker",
            "message": "Unexplained material gain.",
            "evidence": {
                "audit": {"unexplained_element_gains": {"C": 8}}
            },
            "required_action": "Add the missing atom-contributing reactant.",
        }
    ]
    portfolio = _portfolio(route_count=1)
    portfolio["selected_routes"][0].update(
        {
            "complete": True,
            "all_edges_proven": False,
            "minimum_edge_proof_level": 0,
        }
    )
    portfolio["edge_proofs"]["edge:ester"].update(
        {
            "achieved_level": 0,
            "accepted": False,
            "reaction_validated": False,
            "exact_source_bound": False,
            "source_binding_ids": [],
            "exact_record_ids": [],
            "independent_source_groups": [],
            "reasons": ["historical_atom_balance_violation"],
        }
    )
    portfolio["accepted"] = False

    forest = compile_v4_route_forest(compile_route_workbench(graph, portfolio))
    step = next(row for row in forest["steps"] if row["graph_step_id"] == "edge:ester")

    assert step["proof_tier"] == "L0_rejected"
    assert step["trust_vector"]["proof_tier"] == "L0_rejected"
    assert step["visual_encoding"]["color"] == "#be123c"
    assert step["validation_findings"][0]["evidence"]["audit"] == {
        "unexplained_element_gains": {"C": 8}
    }


def test_condition_complete_requires_replayable_and_complete_source_procedure() -> None:
    graph = deepcopy(_graph())
    graph["procedure_records"] = {
        "procedure:1": {
        "procedure_record_id": "procedure:1",
        "exact_record_id": "record:1",
        "location_refs": ["Example 1"],
        "conditions": {
            "reagents": ["base"],
            "solvent": "THF",
            "temperature_c": 20,
            "time": "2 h",
        },
        "procedure_authority_scope": "source_exact_reaction_procedure",
        "condition_completeness": {
            "schema_version": "reaction_condition_completeness.v1",
            "complete": True,
            "missing_required_groups": [],
        },
        }
    }
    graph["edges"]["edge:ester"]["procedure_record_ids"] = ["procedure:1"]
    portfolio = _portfolio()
    portfolio["edge_proofs"]["edge:ester"]["procedure_record_ids"] = ["procedure:1"]

    projection = compile_route_workbench(graph, portfolio)

    edge_vector = projection["edges"]["edge:ester"]["proof_vector"]
    route = projection["routes"]["route:0"]
    assert edge_vector["conditions"] == "source_exact"
    assert edge_vector["condition_completeness"] == "complete"
    assert edge_vector["process"] == "procedure_bound_candidate"
    assert route["condition_complete"] is True
    assert route["proof_vector"]["condition_completeness"] == "complete"
    assert route["process_ready"] is False
    assert route["acceptance_profiles"]["condition_complete"] is True
    assert projection["views"]["condition_complete"]["count"] == 2
    forest = compile_v4_route_forest(projection)
    step = next(value for value in forest["steps"] if value["procedure_records"])
    assert step["evidence_refs"] == ["Example 1"]
    assert step["condition_missing_required_groups"] == []
    assert len(step["procedure_records"]) == 1


def test_best_source_procedure_controls_displayed_condition_gap() -> None:
    graph = deepcopy(_graph())
    graph["procedure_records"] = {
        "procedure:unparsed": {
            "procedure_record_id": "procedure:unparsed",
            "exact_record_id": "record:1",
            "location_refs": ["SI page 3"],
            "conditions": {},
            "procedure_authority_scope": "source_exact_reaction_procedure",
            "condition_completeness": {
                "complete": False,
                "missing_required_groups": [
                    "agents",
                    "solvent",
                    "temperature",
                    "time",
                ],
            },
        },
        "procedure:partial": {
            "procedure_record_id": "procedure:partial",
            "exact_record_id": "record:1",
            "location_refs": ["Patent example 8"],
            "conditions": {"reagents": ["HATU"], "solvent": "DMF"},
            "procedure_authority_scope": "source_exact_reaction_procedure",
            "condition_completeness": {
                "complete": False,
                "missing_required_groups": ["temperature", "time"],
            },
        },
    }
    procedure_ids = ["procedure:unparsed", "procedure:partial"]
    graph["edges"]["edge:ester"]["procedure_record_ids"] = procedure_ids
    portfolio = _portfolio()
    portfolio["edge_proofs"]["edge:ester"]["procedure_record_ids"] = procedure_ids

    projection = compile_route_workbench(graph, portfolio)
    edge = projection["inspectors"]["edges"]["edge:ester"]

    assert edge["proof_vector"]["procedure_record_count"] == 2
    assert edge["proof_vector"]["conditions"] == "source_exact"
    assert edge["condition_missing_required_groups"] == ["temperature", "time"]
    forest = compile_v4_route_forest(projection)
    step = next(value for value in forest["steps"] if value["procedure_records"])
    assert {row["label"] for row in step["conditions"]} == {
        "reagents",
        "solvent",
    }


def test_source_observed_conditions_display_without_granting_exact_identity() -> None:
    graph = _graph()
    graph["source_observation_records"] = {
        "observation:24": {
            "record_id": "observation:24",
            "source_ref": "doi:10.1000/reported-route",
            "location_refs": ["Compound 24"],
            "conditions": {
                "reagents": ["p-TsOH"],
                "solvent": ["ethylene glycol"],
                "temperature": "room temperature",
                "time": "16 h",
                "yield_percent": 93.0,
            },
            "authority_scope": "source_reported_procedure_observation",
        }
    }
    graph["edges"]["edge:ester"]["source_observation_record_ids"] = [
        "observation:24"
    ]
    portfolio = _portfolio(route_count=1)
    portfolio["edge_proofs"]["edge:ester"] = {
        "achieved_level": 0,
        "accepted": False,
        "reaction_validated": False,
        "exact_source_bound": False,
        "source_binding_ids": [],
        "exact_record_ids": [],
        "source_observation_record_ids": ["observation:24"],
        "conflict_ids": [],
        "reasons": ["structure_translation_pending"],
    }

    projection = compile_route_workbench(graph, portfolio)
    vector = projection["edges"]["edge:ester"]["proof_vector"]
    forest = compile_v4_route_forest(projection)
    step = forest["steps"][0]

    assert vector["identity"] == "materialized"
    assert vector["reaction"] == "mapped"
    assert vector["conditions"] == "source_recorded_unverified"
    assert vector["exact_procedure_record_count"] == 0
    assert step["proof_tier"] == "L0_advisory"
    assert len(step["source_observation_records"]) == 1
    assert step["trusted_exact_source_bindings"] == []
    assert forest["run_trace"]["literature_counts"]["source_observation_records"] == 1
    assert {row["label"] for row in step["conditions"]} >= {
        "reagents",
        "solvent",
        "temperature",
        "time",
    }


def test_visual_source_conditions_are_not_displayed_as_model_predictions() -> None:
    graph = _graph()
    graph["edges"]["edge:ester"]["condition_predictions"] = [
        {
            "authority_scope": "model_extracted_source_condition_candidate",
            "source_ref": "doi:10.1000/visual-source",
            "source_locator": "page 3, scheme 1",
            "conditions": {
                "reagents": ["cyclopentadiene"],
                "catalyst": "pyrrolidine",
                "solvent": "methanol",
            },
            "condition_completeness": {
                "complete": False,
                "missing_required_groups": ["temperature", "time"],
            },
            "not_reaction_proof": True,
        }
    ]
    portfolio = _portfolio(route_count=1)

    projection = compile_route_workbench(graph, portfolio)
    vector = projection["edges"]["edge:ester"]["proof_vector"]
    forest = compile_v4_route_forest(projection)
    step = forest["steps"][0]

    assert vector["conditions"] == "source_recorded_unverified"
    assert vector["condition_record_count"] == 1
    assert vector["condition_completeness"] == "partial"
    assert step["condition_status"] == "source_recorded_unverified"
    assert step["condition_predictions"] == []
    assert "doi:10.1000/visual-source" in step["source_refs"]
    assert "page 3, scheme 1" in step["evidence_refs"]
    assert {row["label"] for row in step["conditions"]} >= {
        "reagents",
        "catalyst",
        "solvent",
    }


def test_process_ready_requires_procurement_exact_sources_and_complete_procedure() -> None:
    graph = deepcopy(_graph())
    graph["procedure_records"] = {
        "procedure:1": {
        "procedure_record_id": "procedure:1",
        "exact_record_id": "record:1",
        "location_refs": ["Example 1"],
        "conditions": {
            "reagents": ["base"],
            "solvent": "THF",
            "temperature_c": 20,
            "time": "2 h",
        },
        "procedure_authority_scope": "source_exact_reaction_procedure",
        "condition_completeness": {
            "schema_version": "reaction_condition_completeness.v1",
            "complete": True,
            "missing_required_groups": [],
        },
        }
    }
    graph["edges"]["edge:ester"]["procedure_record_ids"] = ["procedure:1"]
    portfolio = _portfolio()
    portfolio["edge_proofs"]["edge:ester"]["procedure_record_ids"] = ["procedure:1"]
    portfolio["proof_policy"]["stock_boundary"] = "procurement"

    projection = compile_route_workbench(graph, portfolio)
    route = projection["routes"]["route:0"]

    assert route["proof_vector"]["stock"] == "offer_verified"
    assert route["proof_vector"]["process"] == "executable_candidate"
    assert route["acceptance_profiles"] == {
        "exploration_closed": True,
        "reaction_validated": True,
        "literature_grounded": True,
        "condition_complete": True,
        "procurement_closed": True,
        "process_ready": True,
    }
    assert route["process_ready"] is True
    assert projection["portfolio"]["achieved_profile"] == "process_ready"
    assert projection["portfolio"]["process_ready"] is True
    assert projection["views"]["process_ready"]["count"] == 2


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
    assert edge["condition_gap"] == "no_hash_bound_source_procedure"
    molecule = projection["inspectors"]["molecules"]["m:ethanol"]
    assert molecule["stock_closed"] is True
    assert molecule["stock_observations"][0]["catalog_number"] == "E-1"
    assert projection["inspectors"]["rejections"][0]["reasons"] == [
        "element_balance_invalid"
    ]
    assert "conflict:resolved" in projection["inspectors"]["conflicts"]


def test_model_condition_predictions_reach_the_reaction_inspector() -> None:
    graph = _graph()
    graph["edges"]["edge:ester"]["condition_predictions"] = [
        {
            "Reagent": "O=S(=O)(O)O",
            "Solvent": "CCO",
            "Temperature": 52.7239,
            "Score": "0.3832",
            "authority_scope": "model_predicted_condition",
            "not_reaction_proof": True,
        },
        {
            "Reagent": "O=S(Cl)Cl",
            "Temperature": 30.2,
            "Score": "0.2631",
            "authority_scope": "model_predicted_condition",
            "not_reaction_proof": True,
        },
    ]
    portfolio = _portfolio(route_count=1)
    portfolio["edge_proofs"]["edge:ester"].update(
        {
            "exact_source_bound": False,
            "source_binding_ids": [],
            "exact_record_ids": [],
        }
    )

    projection = compile_route_workbench(graph, portfolio)
    inspector = projection["inspectors"]["edges"]["edge:ester"]
    forest = compile_v4_route_forest(projection)
    step = forest["steps"][0]

    assert inspector["condition_status"] == "model_predicted"
    assert inspector["condition_predictions"] == graph["edges"]["edge:ester"][
        "condition_predictions"
    ]
    assert step["condition_status"] == "model_predicted"
    assert step["condition_predictions"] == inspector["condition_predictions"]
    assert step["conditions"] == [
        {"label": "reagents", "value": "O=S(=O)(O)O"},
        {"label": "solvent", "value": "CCO"},
        {"label": "temperature", "value": "52.7 °C"},
    ]


def test_lifecycle_invalidations_are_visible_and_remove_source_authority() -> None:
    portfolio = _portfolio(route_count=1)
    proof = portfolio["edge_proofs"]["edge:ester"]
    proof.update(
        {
            "achieved_level": 2,
            "accepted": False,
            "exact_source_bound": False,
            "source_binding_ids": [],
            "exact_record_ids": [],
            "procedure_record_ids": [],
            "inactive_fact_count": 1,
            "inactive_facts": [
                {
                    "subject_kind": "source_binding",
                    "subject_id": "source:patent",
                    "status": "revoked",
                    "lifecycle_event_id": "lifecycle:source-revoked",
                    "effective_at": "2026-07-15T12:00:00Z",
                    "reason_codes": ["source_retracted"],
                }
            ],
            "reasons": ["source_binding_revoked:source:patent"],
        }
    )
    portfolio["selected_routes"][0].update(
        {
            "minimum_edge_proof_level": 2,
            "all_edges_proven": False,
            "complete": False,
            "independent_source_groups": [],
        }
    )

    projection = compile_route_workbench(_graph(), portfolio)
    edge = projection["edges"]["edge:ester"]
    route = projection["routes"]["route:0"]
    inspector = projection["inspectors"]["edges"]["edge:ester"]

    assert edge["proof_vector"]["identity"] == "materialized"
    assert edge["proof_vector"]["sources"] == "none"
    assert edge["inactive_fact_count"] == 1
    assert "fact-revoked" in edge["badges"]
    assert route["literature_grounded"] is False
    assert route["inactive_fact_count"] == 1
    assert inspector["sources"] == []
    assert inspector["exact_records"] == []
    assert inspector["inactive_facts"][0]["status"] == "revoked"

    forest = compile_v4_route_forest(projection)
    step = next(value for value in forest["steps"] if value["graph_step_id"] == "edge:ester")
    branch = next(
        value
        for value in forest["branches"]
        if value.get("kind") == "exploratory_canonical_route"
    )
    assert step["inactive_facts"][0]["lifecycle_event_id"] == (
        "lifecycle:source-revoked"
    )
    assert branch["route_state_label"] == "权威事实失效 · 路线已降级"
    html = render_v4_route_workbench_html(projection)
    assert "lifecycle:source-revoked" in html


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
    assert all(
        value["route_state_label"] == "搜索边界闭合 · 非采购路线"
        for value in forest["branches"]
        if value["kind"] == "proof_eligible_portfolio_route"
    )
    assert all(value["full_synthesis_claim"] is False for value in portfolio_lanes)
    assert all(value["condition_label"] == "条件缺失" for value in portfolio_lanes)
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
    assert "DECLARED ROUTE GRAPH" in html
    assert "any_declared_route_graph_closed" in html
    assert "translate3d" not in html


def test_incomplete_canonical_route_is_exploratory_and_condition_gap_is_system_owned() -> None:
    portfolio = _portfolio(route_count=1)
    portfolio["selected_routes"][0]["complete"] = False
    portfolio["selected_routes"][0]["leaf_molecule_ids"] = []
    portfolio["edge_proofs"]["edge:ester"].update(
        {
            "exact_source_bound": False,
            "source_binding_ids": [],
            "exact_record_ids": [],
        }
    )
    graph = _graph()
    graph["source_bindings"] = {}
    graph["exact_records"] = {}

    forest = compile_v4_route_forest(compile_route_workbench(graph, portfolio))
    branch = next(row for row in forest["branches"] if row["branch_id"] == "route:0")
    step = next(row for row in forest["steps"] if row["branch_id"] == "route:0")

    assert branch["kind"] == "exploratory_canonical_route"
    assert branch["proof_eligible"] is False
    assert step["condition_resolution"]["stage"] == "exact_reaction_source_search_pending"
    assert step["condition_resolution"]["acquisition_owner"] == "autoplanner"
    assert step["condition_resolution"]["user_input_required"] is False
    assert "不会编造条件" in step["condition_resolution"]["next_action"]


def test_workbench_keeps_full_planner_skeleton_with_rejected_step_advisory() -> None:
    graph = _graph()
    graph["route_families"] = {
        "family:plan": {
            "aliases": ["RF-plan"],
            "strategy": "two-step planner route with one omitted reagent",
        }
    }
    graph["hypotheses"] = {
        "hypothesis:materialized": {
            "hypothesis_id": "hypothesis:materialized",
            "edge_digest": "ester",
            "status": "materialized",
            "product_smiles": "CCOC(C)=O",
            "precursor_smiles": ["CCO", "CC(=O)O"],
            "route_family_ids": ["family:plan"],
            "origin_records": [
                {
                    "origin_kind": "codex_global_director",
                    "proposal_id": "SK1-S01",
                    "route_family_id": "RF-plan",
                    "skeleton_id": "SK1",
                    "transformation_hypothesis": "ester formation",
                }
            ],
        },
        "hypothesis:blocked": {
            "hypothesis_id": "hypothesis:blocked",
            "edge_digest": "blocked",
            "status": "admission_rejected",
            "admission_accepted": False,
            "admission_reasons": ["element_inventory_not_conserved"],
            "product_smiles": "CCO",
            "precursor_smiles": ["CC"],
            "route_family_ids": ["family:plan"],
            "origin_records": [
                {
                    "origin_kind": "codex_global_director",
                    "proposal_id": "SK1-S02",
                    "route_family_id": "RF-plan",
                    "skeleton_id": "SK1",
                    "transformation_hypothesis": "hydration",
                }
            ],
        },
    }

    projection = compile_route_workbench(graph, _portfolio())
    assert projection["portfolio"]["route_count"] == 2
    planned = next(iter(projection["planned_routes"].values()))
    assert planned["declared_step_count"] == 2
    assert planned["materialized_step_count"] == 1
    assert planned["admission_rejected_step_count"] == 1
    assert planned["complete"] is False
    assert "planner_route_contains_admission_rejected_steps" in planned[
        "warning_codes"
    ]
    closure = projection["route_closure"]
    assert closure["declared_program_count"] == 1
    assert closure["graph_closed_program_count"] == 0
    assert closure["any_declared_route_graph_closed"] is False
    assert closure["programs"][0]["gap_step_count"] == 1
    assert closure["programs"][0]["state"] == "admission_rejected_gap"
    assert closure["semantics"]["route_length_is_not_an_optimization_target"] is True

    forest = compile_v4_route_forest(projection)
    payload = build_route_forest_delivery_payload(forest)
    assert route_forest_delivery_integrity_reasons(
        payload,
        source_forest=forest,
    ) == []
    branch = next(
        row
        for row in forest["branches"]
        if row.get("kind") == "planner_route_hypothesis"
    )
    branch_steps = [
        row for row in forest["steps"] if row.get("branch_id") == branch["branch_id"]
    ]
    assert len(branch_steps) == 2
    assert any(row["proof_tier"] == "L0_rejected" for row in branch_steps)
    assert branch["complete"] is False
    assert branch["advisory_only"] is True
    assert forest["counts"]["portfolio_routes"] == 2
    assert forest["route_closure"] == closure
    assert payload["route_closure"] == closure


def test_workbench_marks_fully_materialized_declared_program_graph_closed() -> None:
    graph = _graph()
    graph["hypotheses"] = {
        "hypothesis:one": {
            "hypothesis_id": "hypothesis:one",
            "edge_digest": "ester",
            "status": "materialized",
            "product_smiles": "CCOC(C)=O",
            "precursor_smiles": ["CCO", "CC(=O)O"],
            "origin_records": [
                {
                    "origin_kind": "codex_global_director",
                    "origin_ref": "director:one",
                    "proposal_id": "SK1-S01",
                    "route_family_id": "RF1",
                    "skeleton_id": "SK1",
                }
            ],
        }
    }

    projection = compile_route_workbench(graph, _portfolio())
    closure = projection["route_closure"]

    assert closure["any_declared_route_graph_closed"] is True
    assert closure["graph_closed_program_count"] == 1
    assert closure["longest_graph_closed_step_count"] == 1
    assert closure["programs"][0]["state"] == "declared_route_graph_closed"
    assert closure["semantics"]["graph_closure_is_not_literature_grounding"] is True


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


def test_v4_workbench_deduplicates_planner_input_endpoints_but_preserves_equivalents() -> None:
    graph = _graph()
    graph["route_families"] = {
        "family:plan": {"aliases": ["RF-plan"], "strategy": "planner route"}
    }
    graph["hypotheses"] = {
        "hypothesis:repeated": {
            "hypothesis_id": "hypothesis:repeated",
            "status": "frontier_candidate",
            "product_smiles": "CCOC(C)=O",
            "precursor_smiles": ["CCO", "CCO"],
            "route_family_ids": ["family:plan"],
            "origin_records": [
                {
                    "origin_kind": "codex_global_director",
                    "proposal_id": "SK1-S01",
                    "route_family_id": "RF-plan",
                    "skeleton_id": "SK1",
                    "transformation_hypothesis": "duplicate equivalent input",
                }
            ],
        }
    }

    forest = compile_v4_route_forest(compile_route_workbench(graph, _portfolio()))
    payload = build_route_forest_delivery_payload(forest)

    assert route_forest_delivery_integrity_reasons(payload, source_forest=forest) == []
    step = next(
        value
        for value in forest["steps"]
        if value.get("stoichiometric_input_count") == 2
        and value.get("reaction_class") == "Disconnection hypothesis"
    )
    assert len(step["from_node_ids"]) == 1
    assert step["stoichiometric_input_count"] == 2
    assert step["precursor_multiplicity"] == [
        {"molecule_node_id": step["from_node_ids"][0], "count": 2}
    ]


def test_planned_branch_projection_deduplicates_graph_endpoints_and_keeps_equivalents() -> None:
    nodes: dict[str, dict] = {}
    steps: list[dict] = []
    branches: list[dict] = []
    graph_nodes: dict[str, dict] = {}
    graph_edges: list[dict] = []
    branch_views: list[dict] = []

    append_planned_route_branches(
        {
            "planned_routes": {
                "planned:fixture": {
                    "strategy": "fixture planner route",
                    "steps": [
                        {
                            "step_id": "SK1-S01",
                            "product_smiles": "CCOC(C)=O",
                            "precursor_smiles": ["CCO", "CCO"],
                            "transformation_hypothesis": "duplicate equivalent input",
                        }
                    ],
                }
            }
        },
        nodes_by_id=nodes,
        steps=steps,
        branches=branches,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        branch_views=branch_views,
        node_factory=lambda molecule_id, row: {
            "node_id": molecule_id,
            "label": str(row.get("canonical_smiles") or molecule_id),
            "canonical_isomeric_smiles": str(row.get("canonical_smiles") or ""),
            "role": str(row.get("role") or "intermediate"),
        },
    )

    assert steps[0]["from_node_ids"] == [steps[0]["from_node_ids"][0]]
    assert steps[0]["precursor_multiplicity"] == [
        {"molecule_node_id": steps[0]["from_node_ids"][0], "count": 2}
    ]
    assert steps[0]["stoichiometric_input_count"] == 2
    assert len({edge["edge_id"] for edge in graph_edges}) == len(graph_edges)


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

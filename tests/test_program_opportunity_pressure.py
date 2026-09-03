from __future__ import annotations

from copy import deepcopy

from cascade_planner.application.program_opportunity_pressure import (
    compile_program_opportunity_pressure,
    compile_program_review_pressure,
)


def _linear_graph() -> tuple[dict, dict]:
    smiles = [
        "CC(=O)C1CCCCC1",
        "CC(O)(O)C1CCCCC1",
        "CC(Cl)C1CCCCC1",
        "CC(O)C1CCCCC1",
    ]
    molecules = {
        f"m:{index}": {"canonical_smiles": value}
        for index, value in enumerate(smiles)
    }
    edges = {
        f"edge:{index}": {
            "precursor_molecule_ids": [f"m:{index}"],
            "product_molecule_id": f"m:{index + 1}",
            "innovation_boundary_proof_level": 1,
            "status": "materialized",
            "condition_predictions": [],
        }
        for index in range(len(smiles) - 1)
    }
    route = {
        "route_id": "route:fixture",
        "route_family_id": "family:fixture",
        "edge_ids": list(edges),
        "unproven_edge_ids": ["edge:1"],
        "length": len(edges),
        "physical_step_count": len(edges),
        "risk_score": 0.7,
        "reported_source_refs": ["doi:10.1000/anchor"],
    }
    return {"molecules": molecules, "edges": edges}, route


def _capability() -> dict:
    return {
        "capability_id": "fixture:cyclic-ketone-reduction",
        "enzyme": {"classes": ["ketoreductase"]},
        "match": {
            "net_motif_delta": {"carbonyl": -1, "hydroxyl": 1},
            "preserved_motifs": ["alkene", "ester"],
            "element_delta": {"C": 0, "O": 0},
            "min_scaffold_similarity": 0.3,
            "max_abs_heavy_atom_delta": 0,
            "min_substrate_carbons": 6,
            "min_substrate_rings": 1,
            "min_window_steps": 2,
            "max_window_steps": 4,
            "reject_unlisted_motif_changes": True,
        },
        "selectivity_objective": (
            "Reduce the cyclic ketone to the specified alcohol."
        ),
        "substrate_scope_basis": "fixture analog only",
        "precedent_refs": ["doi:10.1000/fixture"],
    }


def test_program_pressure_rewards_cost_match_selectivity_and_step_savings() -> None:
    graph, route = _linear_graph()

    pressure = compile_program_opportunity_pressure(
        graph,
        route,
        capabilities=[_capability()],
    )

    assert pressure["candidate_count"] == 1
    assert pressure["matched_capability_count"] == 1
    assert pressure["matched_capability_ids"] == [
        "fixture:cyclic-ketone-reduction"
    ]
    assert pressure["longest_contiguous_window_steps"] == 3
    assert pressure["maximum_step_savings"] == 2
    assert pressure["components"]["high_cost_contiguous_span"] > 0.0
    assert pressure["components"]["known_capability_match"] > 0.0
    assert pressure["components"]["selectivity_bottleneck"] > 0.0
    assert pressure["components"]["multi_step_replacement_gain"] == 0.5
    assert pressure["score"]["priority"] > 0.0
    assert pressure["legacy_priority"] > 280.0


def test_high_cost_route_remains_visible_without_a_matching_capability() -> None:
    graph, route = _linear_graph()
    matched = compile_program_opportunity_pressure(
        graph,
        route,
        capabilities=[_capability()],
    )
    unmatched = compile_program_opportunity_pressure(
        graph,
        route,
        capabilities=[],
    )

    assert unmatched["candidate_count"] == 0
    assert unmatched["components"]["high_cost_contiguous_span"] == matched[
        "components"
    ]["high_cost_contiguous_span"]
    assert unmatched["components"]["known_capability_match"] == 0.0
    assert unmatched["components"]["selectivity_bottleneck"] == 0.0
    assert matched["score"]["priority"] > unmatched["score"]["priority"]


def test_short_low_risk_route_keeps_the_fixed_baseline() -> None:
    graph, route = _linear_graph()
    route["edge_ids"] = ["edge:0"]
    route["unproven_edge_ids"] = []
    route["length"] = 1
    route["physical_step_count"] = 1
    route["risk_score"] = 0.0

    pressure = compile_program_opportunity_pressure(
        graph,
        route,
        capabilities=[],
    )

    assert pressure["pressure_total"] == 0.0
    assert pressure["components"] == {
        "high_cost_contiguous_span": 0.0,
        "known_capability_match": 0.0,
        "selectivity_bottleneck": 0.0,
        "multi_step_replacement_gain": 0.0,
        "mechanism_hypothesis_gain": 0.0,
    }
    assert pressure["legacy_priority"] == 280.0


def test_program_pressure_is_label_order_invariant_and_read_only() -> None:
    graph, route = _linear_graph()
    frozen_graph = deepcopy(graph)
    frozen_route = deepcopy(route)
    baseline = compile_program_opportunity_pressure(
        graph,
        route,
        capabilities=[_capability()],
    )
    labelled_graph = {
        **graph,
        "target_name": "opaque",
        "dataset_id": "held-out",
        "edges": dict(reversed(list(graph["edges"].items()))),
    }
    labelled_route = {
        **route,
        "objective_mode": "compatibility-only",
        "target_index": 19,
    }
    comparison = compile_program_opportunity_pressure(
        labelled_graph,
        labelled_route,
        capabilities=[_capability()],
    )

    assert baseline == comparison
    assert graph == frozen_graph
    assert route == frozen_route
    assert baseline["semantics"][
        "conventional_route_remains_the_explicit_fallback"
    ] is True


def test_program_review_aggregates_only_digest_valid_route_pressure() -> None:
    graph, route = _linear_graph()
    pressure = compile_program_opportunity_pressure(
        graph,
        route,
        capabilities=[_capability()],
    )
    tampered = deepcopy(pressure)
    tampered["pressure_total"] = 1.0

    review = compile_program_review_pressure([pressure, tampered])
    baseline = compile_program_review_pressure([])

    assert review["route_pressure_count"] == 1
    assert review["candidate_route_fraction"] == 1.0
    assert review["pressure_total"] > baseline["pressure_total"]
    assert review["score"]["priority"] > baseline["score"]["priority"]
    assert review["legacy_priority"] > 260.0
    assert review["semantics"]["conventional_fallbacks_are_retained"] is True

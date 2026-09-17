from __future__ import annotations

from typing import Any

from cascade_planner.application.proof_policy import ProofPolicy
from cascade_planner.application.route_candidate_builder import (
    CHEMICAL_CRITIC_RISK_VECTOR_SCHEMA,
    _route_critic_step_id_to_edge_id,
    build_route_candidate,
)
from cascade_planner.application.route_variants import RouteSubroute


def _route_candidate(
    *,
    assessments: list[dict[str, Any]],
    key_event_step_ids: tuple[str, ...] = (),
    route_level_risks: tuple[str, ...] = (),
    proposal_ids_by_edge: dict[str, str] | None = None,
) -> dict[str, Any]:
    edge_ids = tuple(f"edge:{index}" for index in range(1, 5))
    proposal_ids_by_edge = dict(proposal_ids_by_edge or {})
    graph = {
        "target_molecule_id": "molecule:target",
        "edges": {
            edge_id: {
                "product_molecule_id": (
                    "molecule:target"
                    if index == 1
                    else f"molecule:{index - 1}"
                ),
                "precursor_molecule_ids": [
                    "molecule:leaf" if index == 4 else f"molecule:{index}"
                ],
                "origin_records": (
                    [
                        {
                            "proposal_id": f"proposal:other:{index}",
                            "canonical_route_family_ids": ["family:other"],
                        },
                        {
                            "proposal_id": proposal_ids_by_edge[edge_id],
                            **(
                                {"route_family_id": "family:one"}
                                if index % 2 == 0
                                else {"canonical_route_family_ids": ["family:one"]}
                            ),
                        },
                    ]
                    if edge_id in proposal_ids_by_edge
                    else []
                ),
            }
            for index, edge_id in enumerate(edge_ids, start=1)
        },
    }
    edge_proofs = {
        edge_id: {
            "edge_id": edge_id,
            "achieved_level": 2,
            "accepted": True,
            "reaction_validated": True,
            "exact_source_bound": False,
            "procedure_record_ids": [],
            "independent_source_groups": [],
            "conflict_ids": [],
        }
        for edge_id in edge_ids
    }
    return build_route_candidate(
        graph,
        family_id="family:one",
        family={
            "strategy": "test strategy",
            "chemical_critic": {
                "status": "reject",
                "step_assessments": assessments,
                "route_level_risks": list(route_level_risks),
            },
            "key_event_critic_history": [
                {
                    "checkpoint_match": True,
                    "focus_step_id": step_id,
                }
                for step_id in key_event_step_ids
            ],
        },
        variant=RouteSubroute(
            edge_ids=frozenset(edge_ids),
            leaf_ids=frozenset({"molecule:leaf"}),
            module_selections=(),
        ),
        edge_proofs=edge_proofs,
        leaf_proof_cache={
            "molecule:leaf": {
                "molecule_id": "molecule:leaf",
                "accepted": True,
            }
        },
        policy=ProofPolicy(
            minimum_edge_proof_level=2,
            minimum_independent_source_groups=0,
            require_stock_for_every_selected_leaf=False,
            stock_boundary="benchmark_search",
        ),
    )


def test_route_risk_is_derived_from_bound_step_assessments() -> None:
    route = _route_candidate(
        assessments=[
            {"step_id": "edge:1", "verdict": "pass", "blocking_type": "none"},
            {
                "step_id": "edge:2",
                "verdict": "uncertain",
                "blocking_type": "none",
            },
            {
                "step_id": "edge:3",
                "verdict": "reject",
                "blocking_type": "mechanism",
            },
            {
                "step_id": "edge:4",
                "verdict": "uncertain",
                "blocking_type": "chemoselectivity",
            },
        ],
        key_event_step_ids=("edge:2", "edge:3"),
        route_level_risks=("prose risk one", "prose risk two"),
    )

    vector = route["chemical_critic_risk_vector"]
    assert vector["schema_version"] == CHEMICAL_CRITIC_RISK_VECTOR_SCHEMA
    assert vector["verdict_counts"] == {
        "pass": 1,
        "uncertain": 2,
        "reject": 1,
    }
    assert vector["key_event_verdict_counts"] == {
        "pass": 0,
        "uncertain": 1,
        "reject": 1,
    }
    assert vector["blocking_type_counts"] == {
        "chemoselectivity": 1,
        "mechanism": 1,
    }
    assert vector["score_components"] == {
        "reject_presence": 0.52,
        "reject_prevalence": 0.0325,
        "uncertainty_prevalence": 0.15,
        "key_event_uncertainty": 0.06,
        "typed_blocking_prevalence": 0.015,
    }
    assert vector["score"] == 0.7775
    assert route["risk_score"] == 0.8175
    assert vector["semantics"]["grants_no_route_admission"] is True


def test_review_coverage_is_diagnostic_not_an_extra_chemical_penalty() -> None:
    assessments = [
        {
            "step_id": "edge:2",
            "verdict": "uncertain",
            "blocking_type": "none",
        },
        {
            "step_id": "edge:outside",
            "verdict": "reject",
            "blocking_type": "mechanism",
        },
    ]
    without_prose_risks = _route_candidate(
        assessments=assessments,
        key_event_step_ids=("edge:2",),
    )
    with_prose_risks = _route_candidate(
        assessments=assessments,
        key_event_step_ids=("edge:2",),
        route_level_risks=("one", "two", "three", "four"),
    )

    vector = with_prose_risks["chemical_critic_risk_vector"]
    assert vector["reviewed_step_count"] == 1
    assert vector["unreviewed_step_count"] == 3
    assert vector["unmatched_assessment_count"] == 1
    assert vector["review_coverage_rate"] == 0.25
    assert vector["score"] == 0.42
    assert vector["semantics"]["review_coverage_is_not_chemical_risk"] is True
    assert with_prose_risks["risk_score"] == without_prose_risks["risk_score"]


def test_route_risk_binds_final_critic_proposal_ids_to_route_edges() -> None:
    route = _route_candidate(
        assessments=[
            {
                "step_id": "proposal:route:2",
                "verdict": "reject",
                "blocking_type": "chemoselectivity",
            },
            {
                "step_id": "proposal:route:3",
                "verdict": "pass",
                "blocking_type": "none",
            },
            {
                "step_id": "proposal:unbound",
                "verdict": "reject",
                "blocking_type": "mechanism",
            },
        ],
        key_event_step_ids=("proposal:route:2",),
        proposal_ids_by_edge={
            "edge:2": "proposal:route:2",
            "edge:3": "proposal:route:3",
        },
    )

    vector = route["chemical_critic_risk_vector"]
    assert vector["verdict_counts"] == {
        "pass": 1,
        "uncertain": 0,
        "reject": 1,
    }
    assert vector["unmatched_assessment_count"] == 1
    assert vector["key_event_verdict_counts"] == {
        "pass": 0,
        "uncertain": 0,
        "reject": 1,
    }
    assert vector["score_components"]["reject_presence"] == 0.52
    assert vector["blocking_type_counts"] == {"chemoselectivity": 1}


def test_route_critic_identity_does_not_guess_between_unbound_origins() -> None:
    bindings = _route_critic_step_id_to_edge_id(
        graph={
            "edges": {
                "edge:1": {
                    "origin_records": [
                        {
                            "proposal_id": "proposal:first",
                            "canonical_route_family_ids": ["family:other"],
                        },
                        {
                            "proposal_id": "proposal:second",
                            "route_family_id": "family:also-other",
                        },
                    ]
                }
            }
        },
        edge_ids=["edge:1"],
        route_family_id="family:one",
    )

    assert bindings == {"edge:1": "edge:1"}

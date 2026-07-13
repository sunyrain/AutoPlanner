from cascade_planner.baselines.chem_enzy_adapter import (
    audit_materialized_chem_enzy_route,
    route_candidates_from_chem_enzy_result,
)
from cascade_planner.baselines.chem_enzy_budget import (
    classify_chemenzy_attempt_outcome,
    resolve_chemenzy_budget,
)
from cascade_planner.baselines.route_contract import RouteStepCandidate
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteSearchConfig
from scripts.run_chem_enzy_plan_for_web import _web_payload_from_result


BAD_PRODUCT = "O=C(O)C(O)(CCO)C(=O)OCc1ccccc1"
BAD_REACTANTS = ["O=C(Cl)OCc1ccccc1", "O=C([O-])[O-]"]


def _raw_one_step_route(
    product: str,
    reactants: list[str],
    *,
    template_id: str,
) -> dict:
    return {
        "all_succ_dict_routes": [
            {
                "type": "mol",
                "smiles": product,
                "children": [
                    {
                        "type": "reaction",
                        "score": 0.99,
                        "template": {
                            "template_id": template_id,
                            "source": "native_graphfp",
                            "model_full_name": "graphfp_models.test",
                        },
                        "children": [
                            {"type": "mol", "smiles": reactant, "in_stock": True}
                            for reactant in reactants
                        ],
                    }
                ],
            }
        ],
        "iter": 1,
    }


def _step(product: str, reactants: list[str], index: int) -> RouteStepCandidate:
    return RouteStepCandidate(
        product_smiles=product,
        reactant_smiles=reactants,
        rxn_smiles=f"{'.'.join(reactants)}>>{product}",
        source_model="graphfp_models.test",
        score=0.9,
        raw_backend_metadata={
            "template": {
                "template_id": f"template-{index}",
                "source": "native_graphfp",
                "model_full_name": "graphfp_models.test",
            }
        },
    )


def test_flattened_bad_edge_is_diagnostic_raw_solved_but_never_host_solved() -> None:
    raw = _raw_one_step_route(
        BAD_PRODUCT,
        BAD_REACTANTS,
        template_id="bad-nirmatrelvir-edge",
    )

    routes = route_candidates_from_chem_enzy_result(raw, target_smiles=BAD_PRODUCT)

    assert len(routes) == 1
    route = routes[0]
    assert len(route.steps) == 1
    assert route.solved is False
    assert route.raw_backend_metadata["raw_solved"] is True
    assert route.raw_backend_metadata["route_outcome"] == "verification_rejected"
    admission = route.raw_backend_metadata["route_materialization_admission"]
    assert admission["accepted"] is False
    assert admission["route_outcome"] == "verification_rejected"
    assert admission["reasons"] == ["element_inventory_not_conserved"]
    assert admission["accepted_edge_count"] == 0
    assert admission["rejected_edge_count"] == 1
    rejection = admission["rejected_edges"][0]
    assert rejection["stage"] == "final_route_materialization"
    assert rejection["source"] == "native_graphfp"
    assert rejection["model"] == "graphfp_models.test"
    assert rejection["template"]["fields"]["template_id"] == "bad-nirmatrelvir-edge"
    assert len(rejection["edge_digest"]) == 64


def test_any_bad_edge_rejects_an_entire_fifteen_step_materialized_route() -> None:
    steps = [_step("CCO", ["CC=O"], index) for index in range(15)]
    steps[9] = _step(BAD_PRODUCT, BAD_REACTANTS, 9)

    admission = audit_materialized_chem_enzy_route(steps, route_index=4)

    assert admission["accepted"] is False
    assert admission["route_outcome"] == "verification_rejected"
    assert admission["raw_solved"] is True
    assert admission["host_admission_accepted"] is False
    assert admission["edge_count"] == 15
    assert admission["accepted_edge_count"] == 14
    assert admission["rejected_edge_count"] == 1
    assert admission["rejected_edges"][0]["candidate_index"] == 9
    assert admission["rejected_edges"][0]["reasons"] == [
        "element_inventory_not_conserved"
    ]


def test_valid_materialized_route_is_not_falsely_rejected() -> None:
    raw = _raw_one_step_route("CCO", ["CC=O"], template_id="valid-oxidation")

    routes = route_candidates_from_chem_enzy_result(raw, target_smiles="CCO")

    assert len(routes) == 1
    route = routes[0]
    assert route.solved is True
    assert route.raw_backend_metadata["raw_solved"] is True
    assert route.raw_backend_metadata["route_outcome"] == "admitted_raw_solved"
    admission = route.raw_backend_metadata["route_materialization_admission"]
    assert admission["accepted"] is True
    assert admission["accepted_edge_count"] == 1
    assert admission["rejected_edge_count"] == 0
    assert admission["reasons"] == []


def test_web_export_keeps_raw_rejection_diagnostic_without_solved_promotion() -> None:
    route = route_candidates_from_chem_enzy_result(
        _raw_one_step_route(
            BAD_PRODUCT,
            BAD_REACTANTS,
            template_id="bad-nirmatrelvir-edge",
        ),
        target_smiles=BAD_PRODUCT,
    )[0]
    result = BaselineRunResult(
        target_smiles=BAD_PRODUCT,
        backend="ChemEnzyRetroPlanner",
        routes=[route],
    )

    payload = _web_payload_from_result(
        result,
        {},
        RouteSearchConfig(target_smiles=BAD_PRODUCT),
        0.1,
    )

    assert payload["raw_solved"] is True
    assert payload["materialization_admission_solved"] is False
    assert payload["search_status"]["raw_solved"] is True
    assert payload["search_status"]["solved"] is False
    assert payload["search_status"]["status"] == "partial"
    assert payload["routes"][0]["metrics"]["raw_backend_solved_not_proof"] is True
    assert payload["routes"][0]["metrics"]["route_solved"] is False
    assert payload["routes"][0]["metrics"]["route_outcome"] == "verification_rejected"
    summary = payload["route_set_metrics"]["route_materialization_admission"]
    assert summary["rejected_route_count"] == 1


def test_explicit_raw_solved_diagnostic_cannot_override_rejected_host_verifier() -> None:
    resolution = resolve_chemenzy_budget(
        target_smiles=BAD_PRODUCT,
        action_kind="child",
        payload={},
        policy={},
        authority="host_profile",
        attempt_index=1,
    )
    outcome = classify_chemenzy_attempt_outcome(
        resolution,
        {
            "raw_solved": True,
            "verified_solved": False,
            "search_status": {"status": "partial", "solved": False},
            "routes": [{"raw_backend_metadata": {"route_outcome": "verification_rejected"}}],
        },
        verifier={
            "accepted": False,
            "route_status": "fake_closed_rejected",
            "reasons": ["element_inventory_not_conserved"],
            "accepted_route_count": 0,
        },
        verified_solved=False,
    )

    assert outcome["outcome"] == "verification_rejected"
    assert outcome["raw_solved"] is True
    assert outcome["verified_solved"] is False
    assert outcome["solved"] is False
    assert outcome["raw_search_status_is_authority"] is False

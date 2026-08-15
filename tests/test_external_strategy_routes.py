from __future__ import annotations

from pathlib import Path

import pytest

from cascade_planner.application.external_strategy_routes import (
    ExternalStrategyRouteError,
    compile_external_strategy_route_bundle,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.runtime.paths import RuntimePaths


def _bundle() -> dict:
    return {
        "schema_version": "external_strategy_route_bundle.v1",
        "provider": "example-strategy-planner",
        "target_smiles": "CCOC(C)=O",
        "routes": [
            {
                "id": "external-route-1",
                "strategy": "late acyl substitution",
                "solved": True,
                "feasibility": "good",
                "steps": [
                    {
                        "idx": 1,
                        "rxn_smiles": "CCO.CC(=O)Cl>>CCOC(C)=O",
                        "conditions": "Et3N, dichloromethane, 0 C",
                        "critic_verdict": "pass",
                        "reaxys_close_count": 17,
                    },
                    {"idx": 2, "rxn_smiles": "CC=O>>CCO"},
                ],
            }
        ],
    }


def _paths(tmp_path: Path) -> RuntimePaths:
    repository = tmp_path / "repository"
    repository.mkdir()
    return RuntimePaths.discover(
        repository_root=repository,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "index.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(tmp_path / "external"),
        },
    )


def test_compile_external_strategy_bundle_is_connected_and_advisory() -> None:
    first = compile_external_strategy_route_bundle(
        _bundle(), expected_target_smiles="CCOC(=O)C"
    )
    second = compile_external_strategy_route_bundle(
        _bundle(), expected_target_smiles="CCOC(C)=O"
    )

    assert first == second
    receipt = first["receipt"]
    assert receipt["route_count"] == 1
    assert receipt["step_count"] == 2
    assert receipt["authority"] == {
        "scope": "external_strategy_advisory_only",
        "self_reported_solved_grants_proof": False,
        "self_reported_feasibility_grants_validation": False,
        "raw_condition_text_grants_condition_completeness": False,
    }
    steps = first["global_plan"]["multi_step_skeletons"][0]["steps"]
    assert steps[0]["product_smiles"] == "CCOC(C)=O"
    assert steps[0]["condition_predictions"] == []
    metadata = steps[0]["provider_reaction_metadata"]
    assert metadata["raw_condition_text"].startswith("Et3N")
    assert metadata["external_route_claims"]["solved"] is True
    assert metadata["not_reaction_proof"] is True


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda value: value.update(target_smiles="CCO"),
            "external_strategy_target_mismatch",
        ),
        (
            lambda value: value["routes"][0]["steps"].append(
                {"rxn_smiles": "CCC>>CCCC"}
            ),
            "external_strategy_route_disconnected",
        ),
        (
            lambda value: value["routes"][0]["steps"][0].update(
                rxn_smiles="not a reaction"
            ),
            "external_strategy_reaction_smiles_invalid",
        ),
        (
            lambda value: value["routes"][0]["steps"][0].update(
                product_smiles="CCOC(C)=O",
                precursor_smiles=["CCO", "CC(=O)Cl"],
                rxn_smiles="CCO.CC(=O)Br>>CCOC(C)=O",
            ),
            "external_strategy_step_structure_conflict",
        ),
    ],
)
def test_compile_external_strategy_bundle_fails_closed(mutate, reason: str) -> None:
    value = _bundle()
    mutate(value)
    with pytest.raises(ExternalStrategyRouteError, match=reason):
        compile_external_strategy_route_bundle(
            value, expected_target_smiles="CCOC(C)=O"
        )


def test_gateway_import_scopes_strategy_to_experiment_closure(tmp_path: Path) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    gateway.create_run(
        target_name="example ester",
        target_smiles="CCOC(C)=O",
        run_id="external-import",
    )

    result = gateway.import_strategy_routes(
        run_id="external-import",
        bundle=_bundle(),
        materialize=True,
    )

    closure = result["strategy_to_experiment_closure"]
    assert closure["route_count"] == 1
    route = closure["routes"][0]
    assert route["strategy_structure"]["status"] == "complete"
    assert route["canonical_materialization"]["status"] == "complete"
    assert "blockers" not in route["canonical_materialization"]
    assert route["host_reaction_validation"]["status"] == "open"
    assert route["exact_source_evidence"]["status"] == "open"
    assert route["complete_exact_conditions"]["status"] == "open"
    assert route["stock_closure"]["status"] == "open"
    assert closure["semantics"]["external_claims_grant_no_host_authority"] is True
    assert "experimental_program_validation" in closure["next_required_capabilities"]

    service = gateway._open("external-import")
    graph = service.graph_store.load()
    origins = [
        origin
        for hypothesis in graph["hypotheses"].values()
        for origin in hypothesis["origin_records"]
    ]
    assert {origin["origin_kind"] for origin in origins} == {"external_strategy"}
    assert all(
        origin["origin_ref"].startswith("external_strategy:") for origin in origins
    )


def test_strategy_closure_explains_materialization_blockers(tmp_path: Path) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    gateway.create_run(
        target_name="protected alcohol",
        target_smiles="CCO[Si](C)(C)C(C)(C)C",
        run_id="external-blocked",
    )
    bundle = {
        "schema_version": "external_strategy_route_bundle.v1",
        "provider": "example-strategy-planner",
        "target_smiles": "CCO[Si](C)(C)C(C)(C)C",
        "routes": [
            {
                "id": "missing-reagent-route",
                "steps": [{"rxn_smiles": "CCO>>CCO[Si](C)(C)C(C)(C)C"}],
            }
        ],
    }

    result = gateway.import_strategy_routes(
        run_id="external-blocked",
        bundle=bundle,
        materialize=True,
    )

    materialization = result["strategy_to_experiment_closure"]["routes"][0][
        "canonical_materialization"
    ]
    assert materialization["status"] == "open"
    assert materialization["achieved"] == 0
    assert materialization["required"] == 1
    assert materialization["blockers"][0]["status"] == "admission_rejected"
    assert materialization["blockers"][0]["reasons"] == [
        "element_inventory_not_conserved"
    ]


def test_external_strategy_compiler_accepts_replayed_reactionjson() -> None:
    bundle = {
        "schema_version": "external_strategy_route_bundle.v1",
        "provider": "graph-edit-planner",
        "target_smiles": "[CH3][CH3]",
        "routes": [
            {
                "id": "edit-route",
                "steps": [
                    {
                        "mapped_product_smiles": "[CH3:1][CH3:2]",
                        "precursor_smiles": ["[CH3]", "[CH3]"],
                        "reactionjson": {
                            "operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}]
                        },
                    }
                ],
            }
        ],
    }

    compiled = compile_external_strategy_route_bundle(
        bundle, expected_target_smiles="CC"
    )

    step = compiled["global_plan"]["multi_step_skeletons"][0]["steps"][0]
    assert step["product_smiles"] == "CC"
    assert step["precursor_smiles"] == ["[CH3]", "[CH3]"]
    audit = step["provider_reaction_metadata"]["reactionjson_replay_audit"]
    assert audit["accepted"] is True
    assert audit["primitive_counts"]["break_bond"] == 1
    assert audit["semantics"]["replay_grants_no_reaction_proof"] is True


def test_external_strategy_reactionjson_must_match_expected_precursors() -> None:
    bundle = {
        "schema_version": "external_strategy_route_bundle.v1",
        "provider": "graph-edit-planner",
        "target_smiles": "CC",
        "routes": [
            {
                "steps": [
                    {
                        "mapped_product_smiles": "[CH3:1][CH3:2]",
                        "precursor_smiles": ["CC"],
                        "operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
                    }
                ]
            }
        ],
    }

    with pytest.raises(
        ExternalStrategyRouteError, match="reactionjson_expected_precursors_mismatch"
    ):
        compile_external_strategy_route_bundle(bundle, expected_target_smiles="CC")

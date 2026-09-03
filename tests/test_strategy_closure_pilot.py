from __future__ import annotations

import hashlib
import json

import pytest

from cascade_planner.application.external_strategy_routes import (
    compile_external_strategy_route_bundle,
)
from cascade_planner.eval.strategy_closure_pilot import (
    StrategyClosurePilotError,
    compile_strategy_closure_leakage_pack,
    compile_strategy_closure_pilot,
    external_bundle_for_case,
)


def _index() -> list[dict]:
    return [
        {
            "id": "named-a-s1",
            "name": "named target a",
            "target_smiles": "CCOC(=O)C",
            "total_steps": 1,
            "variants": 2,
        },
        {
            "id": "named-a-s2",
            "name": "named target a",
            "target_smiles": "CCOC(C)=O",
            "total_steps": 1,
            "variants": 2,
        },
        {
            "id": "named-b-s1",
            "name": "named target b",
            "target_smiles": "CCN",
            "total_steps": 1,
            "variants": 1,
        },
        {
            "id": "not-selected-s1",
            "name": "named target c",
            "target_smiles": "CCC",
            "total_steps": 1,
            "variants": 1,
        },
    ]


def _routes() -> dict[str, dict]:
    return {
        "named-a-s1": {
            "id": "named-a-s1",
            "name": "named target a",
            "target_smiles": "CCOC(C)=O",
            "steps": [{"idx": 0, "rxn_smiles": "CCO.CC(=O)Cl>>CCOC(C)=O"}],
        },
        "named-a-s2": {
            "id": "named-a-s2",
            "name": "named target a",
            "target_smiles": "CCOC(C)=O",
            "steps": [{"idx": 0, "rxn_smiles": "CCO.CC(=O)O>>CCOC(C)=O"}],
        },
        "named-b-s1": {
            "id": "named-b-s1",
            "name": "named target b",
            "target_smiles": "CCN",
            "steps": [{"idx": 0, "rxn_smiles": "CCCl.N>>CCN"}],
        },
    }


def _compile() -> dict:
    return compile_strategy_closure_pilot(
        index_rows=_index(),
        route_documents=_routes(),
        source_snapshot={
            "data_version": "fixture-v1",
            "manifest_sha256": "a" * 64,
            "index_sha256": "b" * 64,
        },
        target_count=2,
        frozen_at="2026-08-12T00:00:00+00:00",
        stock_binding={"index_sha256": "c" * 64, "member_count": 3},
    )


def test_pilot_freeze_separates_blind_manifest_from_evaluator_answers() -> None:
    result = _compile()

    manifest_text = json.dumps(result["target_manifest"], sort_keys=True)
    protocol_text = json.dumps(result["protocol"], sort_keys=True)
    assert len(result["target_manifest"]["cases"]) == 2
    assert "opaque benchmark target 001" in manifest_text
    assert "named target" not in manifest_text
    assert "named-a-s1" not in manifest_text
    assert "rxn_smiles" not in manifest_text
    assert "named-a-s1" not in protocol_text
    assert "CCOC" not in protocol_text
    assert result["protocol"]["scope"]["route_variant_count"] == 3
    assert result["protocol"]["preflight"]["host_c0_passed"] == 3
    assert result["protocol"]["status"] == "frozen_not_executed"
    assert result["protocol"]["arms"][0]["generation_cost_comparable"] is False

    evaluator = result["evaluator_pack"]
    assert [len(case["routes"]) for case in evaluator["cases"]] == [2, 1]
    assert evaluator["authority"]["planner_may_read_this_pack"] is False
    assert "named target a" in json.dumps(evaluator)


def test_evaluator_can_build_a_provider_neutral_external_bundle() -> None:
    result = _compile()
    case = result["evaluator_pack"]["cases"][0]

    bundle = external_bundle_for_case(case)
    compiled = compile_external_strategy_route_bundle(
        bundle, expected_target_smiles=case["target_smiles"]
    )

    assert compiled["receipt"]["route_count"] == 2
    assert compiled["receipt"]["step_count"] == 2
    assert (
        compiled["receipt"]["authority"]["self_reported_solved_grants_proof"] is False
    )


def test_leakage_pack_is_manifest_bound_and_evaluator_only() -> None:
    result = _compile()
    for index, case in enumerate(result["evaluator_pack"]["cases"]):
        case["key_intermediate_smiles"] = ["C" * (8 + index)]
    manifest_bytes = (json.dumps(result["target_manifest"]) + "\n").encode()

    leakage = compile_strategy_closure_leakage_pack(
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        evaluator_pack=result["evaluator_pack"],
    )

    case_ids = [case["case_id"] for case in result["target_manifest"]["cases"]]
    assert leakage["schema_version"] == "blind_leakage_audit_pack.v1"
    assert leakage["semantics"]["never_passed_to_planner_subprocess"] is True
    assert set(leakage["cases"]) == set(case_ids)
    assert leakage["cases"][case_ids[0]]["target_synonyms"] == ["named target a"]
    assert all(row["key_intermediate_smiles"] for row in leakage["cases"].values())


def test_freeze_fails_on_route_document_target_drift() -> None:
    routes = _routes()
    routes["named-a-s1"]["target_smiles"] = "CCC"

    with pytest.raises(
        StrategyClosurePilotError,
        match="strategy_pilot_route_document_target_mismatch:named-a-s1",
    ):
        compile_strategy_closure_pilot(
            index_rows=_index(),
            route_documents=routes,
            source_snapshot={"data_version": "fixture-v1"},
            target_count=2,
            frozen_at="2026-08-12T00:00:00+00:00",
        )


def test_route_quality_failure_is_retained_instead_of_filtered() -> None:
    routes = _routes()
    routes["named-a-s1"]["steps"].append({"idx": 1, "rxn_smiles": "CCO>>CCO"})

    result = compile_strategy_closure_pilot(
        index_rows=_index(),
        route_documents=routes,
        source_snapshot={"data_version": "fixture-v1"},
        target_count=2,
        frozen_at="2026-08-12T00:00:00+00:00",
    )

    assert result["protocol"]["scope"]["route_variant_count"] == 3
    assert result["protocol"]["preflight"]["host_c0_failed"] == 1
    assert result["protocol"]["preflight"]["failure_reasons"] == {
        "external_strategy_route_cycle": 1
    }

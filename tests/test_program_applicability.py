from __future__ import annotations

from copy import deepcopy

from cascade_planner.application.program_applicability import (
    APPLICABILITY_SEMANTICS,
    compile_program_applicability_model,
    program_experience_subject_key,
)
from cascade_planner.application.program_applicability_oracle import (
    program_applicability_model_oracle,
)
from cascade_planner.application.route_structure_matching import structure_transition
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def _candidate(
    *,
    domain: str = "biocatalytic",
    execution_domain: str = "",
    precursor: str = "CC(=O)C1CCCCC1",
    product: str = "CC(O)C1CCCCC1",
    extra: dict | None = None,
) -> dict:
    return {
        "candidate_id": "candidate:fixture",
        "candidate_kind": {
            "biocatalytic": "enzyme_window",
            "execution": "program_execution_window",
        }[domain],
        "capability_id": "capability:reduction",
        "execution_domain": execution_domain,
        "boundary": {"precursor_smiles": precursor, "product_smiles": product},
        **dict(extra or {}),
    }


def _record(
    experience_id: str,
    *,
    polarity: str = "positive",
    domain: str = "biocatalytic",
    execution_domain: str = "",
    precursor: str = "CC(=O)C1CCCCC1",
    product: str = "CC(O)C1CCCCC1",
) -> dict:
    observation_id = "claim:" + experience_id
    observation = {
        "claim_id": observation_id,
        "claim_sha256": "b" * 64,
        "polarity": polarity,
    }
    row = {
        "schema_version": "program_experience_record.v1",
        "experience_id": experience_id,
        "domain": domain,
        "subject_refs": {
            "capability_id": "capability:reduction",
            **(
                {"execution_domain": execution_domain}
                if domain == "execution"
                else {}
            ),
        },
        "strategy_signature_sha256": "",
        "exact_boundary": {
            "input_smiles": [precursor],
            "output_smiles": [product],
        },
        "structural_transition": structure_transition(precursor, product),
        "observations": {observation_id: observation},
    }
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def test_exact_and_analog_evidence_are_weighted_and_target_blind() -> None:
    record = _record("experience:cyclohexyl")
    exact_candidate = _candidate(
        extra={"target_name": "display alpha", "dataset_name": "hidden alpha"}
    )
    analog_candidate = _candidate(
        precursor="CC(=O)C1CCCC1",
        product="CC(O)C1CCCC1",
        extra={"target_name": "display beta", "dataset_name": "hidden beta"},
    )
    exact = compile_program_applicability_model(exact_candidate, [record])
    analog = compile_program_applicability_model(analog_candidate, [record])
    renamed = compile_program_applicability_model(
        {**analog_candidate, "target_name": "other", "dataset_name": "other"},
        [record],
    )

    assert analog == renamed
    assert exact["strongest_transfer_scope"] == "exact_boundary"
    assert exact["matches"][0]["transfer_weight"] == 1.0
    assert analog["strongest_transfer_scope"] == "structural_analog"
    assert 0.0 < analog["matches"][0]["transfer_weight"] < 1.0
    assert 0.0 < analog["priority_adjustment"] < exact["priority_adjustment"]
    assert analog["semantics"] == APPLICABILITY_SEMANTICS


def test_model_is_stable_under_record_order_and_preserves_conflict() -> None:
    candidate = _candidate()
    records = [
        _record("experience:positive", polarity="positive"),
        _record("experience:negative", polarity="negative"),
    ]
    forward = compile_program_applicability_model(candidate, records)
    reverse = compile_program_applicability_model(candidate, list(reversed(records)))

    assert forward == reverse
    assert forward["disposition"] == "conflicting"
    assert forward["applicability_score"] == 0.0
    assert forward["priority_adjustment"] == 0.0
    assert forward["uncertainty_score"] > forward["confidence_score"]
    assert "positive_negative_evidence_conflict" in forward["reasons"]


def test_execution_domains_cannot_share_applicability_evidence() -> None:
    whole_cell = _candidate(domain="execution", execution_domain="whole_cell")
    hybrid_record = _record(
        "experience:hybrid",
        domain="execution",
        execution_domain="hybrid",
    )
    whole_cell_record = _record(
        "experience:whole-cell",
        domain="execution",
        execution_domain="whole_cell",
    )

    assert compile_program_applicability_model(whole_cell, [hybrid_record]) == {}
    assert compile_program_applicability_model(whole_cell, [whole_cell_record])[
        "disposition"
    ] == "supported"
    assert program_experience_subject_key(
        "execution",
        {"capability_id": "capability:reduction", "execution_domain": "whole_cell"},
        "",
    ) != program_experience_subject_key(
        "execution",
        {"capability_id": "capability:reduction", "execution_domain": "hybrid"},
        "",
    )


def test_applicability_oracle_rejects_fresh_digest_tamper() -> None:
    candidate = _candidate()
    records = [_record("experience:oracle")]
    model = compile_program_applicability_model(candidate, records)
    tampered = deepcopy(model)
    tampered["applicability_score"] = -1.0
    tampered.pop("content_sha256")
    tampered["content_sha256"] = strict_canonical_json_sha256(tampered)

    assert program_applicability_model_oracle(candidate, records, model)["accepted"] is True
    assert (
        program_applicability_model_oracle(candidate, records, tampered)["accepted"]
        is False
    )

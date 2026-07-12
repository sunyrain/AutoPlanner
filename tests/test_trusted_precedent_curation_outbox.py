from __future__ import annotations

from cascade_planner.harness.source_detail_chain_builder import (
    build_trusted_precedent_curation_outbox,
)


def _chain_step(source_ref: str, index: int) -> dict:
    source_template_id = f"source_detail_exact_step:{index}"
    return {
        "step_index": index,
        "step_id": f"step:{index}",
        "product_smiles": "CC=O",
        "reactant_smiles": ["CCO"],
        "source_ref": source_ref,
        "source_template_id": source_template_id,
        "source_detail_exact_step": True,
        "relation_type": "exact",
        "evidence_refs": [f"evidence:{index}"],
        "source_evidence": [
            {
                "schema_version": "materialized_source_evidence.v1",
                "source_ref": source_ref,
                "document_id": f"document:{index}",
                "manifest_sha256": "c" * 64,
                "source_pdf_sha256": "a" * 64,
                "page_number": index,
                "image_sha256": "b" * 64,
            }
        ],
        "exact_step_validation": {
            "schema_version": "template_validation_report.v1",
            "accepted": True,
            "allowed_for_one_step_source": True,
            "source_template_id": source_template_id,
            "reasons": [],
        },
    }


def test_curation_outbox_is_deterministic_and_cannot_self_promote() -> None:
    steps = [
        _chain_step("doi:10.1000/two", 2),
        _chain_step("doi:10.1000/one", 1),
    ]

    first = build_trusted_precedent_curation_outbox(
        steps,
        case_id="fixture",
    )
    second = build_trusted_precedent_curation_outbox(
        list(reversed(steps)),
        case_id="fixture",
    )

    assert first == second
    assert first["candidate_count"] == 2
    assert first["production_write_blocked"] is True
    assert first["auto_promotion_allowed"] is False
    assert first["semantics"]["model_cannot_self_sign"] is True
    assert {row["independent_source_group"] for row in first["candidates"]} == {
        "doi:10.1000/one",
        "doi:10.1000/two",
    }
    assert all(row["promotion_allowed"] is False for row in first["candidates"])
    assert all(
        row["status"] == "pending_curator_or_deterministic_parser"
        for row in first["candidates"]
    )


def test_curation_outbox_drops_untraceable_or_unstructured_rows() -> None:
    outbox = build_trusted_precedent_curation_outbox(
        [
            {**_chain_step("not-a-source", 1)},
            {**_chain_step("doi:10.1000/valid", 2), "product_smiles": ""},
        ],
        case_id="fixture",
    )

    assert outbox["candidate_count"] == 0
    assert outbox["candidates"] == []
    assert outbox["production_write_blocked"] is True


def test_curation_outbox_rejects_failed_exact_gate_or_missing_materialized_evidence() -> None:
    rejected_validation = _chain_step("doi:10.1000/rejected", 1)
    rejected_validation["exact_step_validation"]["accepted"] = False
    missing_evidence = _chain_step("doi:10.1000/no-evidence", 2)
    missing_evidence["source_evidence"] = []
    not_exact = _chain_step("doi:10.1000/not-exact", 3)
    not_exact["source_detail_exact_step"] = False

    outbox = build_trusted_precedent_curation_outbox(
        [rejected_validation, missing_evidence, not_exact],
        case_id="fixture",
    )

    assert outbox["candidate_count"] == 0
    assert outbox["rejected_count"] == 3
    rejection_reasons = {
        reason
        for row in outbox["rejected"]
        for reason in row.get("reasons") or []
    }
    assert "exact_step_validation_not_accepted" in rejection_reasons
    assert "materialized_source_evidence_invalid" in rejection_reasons
    assert "source_detail_exact_step_required" in rejection_reasons


def test_curation_outbox_preserves_patent_family_independence() -> None:
    first = {
        **_chain_step("patent:US1234567A1", 1),
        "patent_family": "family-42",
    }
    second = {
        **_chain_step("patent:EP1234567A1", 2),
        "patent_family": "family-42",
    }

    outbox = build_trusted_precedent_curation_outbox(
        [first, second],
        case_id="fixture",
    )

    assert outbox["candidate_count"] == 2
    assert {
        row["independent_source_group"] for row in outbox["candidates"]
    } == {"patent_family:family-42"}
    assert all(
        row["source_identity"]["patent_family"] == "family-42"
        for row in outbox["candidates"]
    )


def test_curation_outbox_rejects_any_invalid_reactant_without_partial_digest() -> None:
    malformed = _chain_step("doi:10.1000/malformed", 1)
    malformed["reactant_smiles"] = ["CCO", "definitely-not-smiles"]

    outbox = build_trusted_precedent_curation_outbox(
        [malformed],
        case_id="fixture",
    )

    assert outbox["candidate_count"] == 0
    assert "reactant_smiles_invalid" in outbox["rejected"][0]["reasons"]

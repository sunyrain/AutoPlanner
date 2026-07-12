from __future__ import annotations

from cascade_planner.harness.source_detail_chain_builder import (
    build_trusted_precedent_curation_outbox,
)


def _chain_step(source_ref: str, index: int) -> dict:
    return {
        "step_index": index,
        "step_id": f"step:{index}",
        "product_smiles": "CC=O",
        "reactant_smiles": ["CCO"],
        "source_ref": source_ref,
        "evidence_refs": [f"evidence:{index}"],
        "source_evidence": [
            {
                "source_ref": source_ref,
                "document_id": f"document:{index}",
                "source_pdf_sha256": "a" * 64,
                "page_number": index,
                "image_sha256": "b" * 64,
            }
        ],
        "exact_step_validation": {
            "accepted": True,
            "allowed_for_one_step_source": True,
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

from __future__ import annotations

from copy import deepcopy

import cascade_planner.harness.codex_edge_verification as edge_verification


def _exact_row(source_ref: str, row_id: str, *, scope: str = "article") -> dict:
    return {
        "row_id": row_id,
        "product_smiles": "CC=O",
        "reactant_smiles": ["CCO"],
        "source_ref": source_ref,
        "content_scope": scope,
        "source_detail_exact_step": True,
        "conditions": [f"condition:{row_id}"],
        "source_evidence": [
            {
                "source_ref": source_ref,
                "document_id": f"document:{row_id}",
                "source_pdf_sha256": "a" * 64,
                "page_number": 1,
                "image_sha256": "b" * 64,
            }
        ],
    }


def _trusted_proof(row: dict, *, graph_and_stock_closed: bool = False) -> dict:
    del graph_and_stock_closed
    source_ref = str(row.get("source_ref") or "")
    trusted = "untrusted" not in source_ref
    digest = edge_verification.canonical_reaction_digest(
        str(row.get("product_smiles") or ""),
        list(row.get("reactant_smiles") or []),
    )
    return {
        "proof_level": "L3_precedent_supported" if trusted else "L2_reaction_validated",
        "checks": {"trusted_precedent_bound": trusted},
        "trusted_precedent_binding": (
            {
                "schema_version": "trusted_precedent_binding.v1",
                "accepted": True,
                "authority": "human_curator",
                "authority_id": "fixture",
                "binding_id": f"binding:{source_ref}",
                "reaction_digest": digest,
                "source_ref": source_ref,
            }
            if trusted
            else {}
        ),
    }


def test_exact_rows_by_signature_preserves_all_rows_stably() -> None:
    first = _exact_row("doi:10.1000/zeta", "row:z")
    second = _exact_row("doi:10.1000/alpha", "row:a")

    forward = edge_verification._exact_rows_by_signature([first, second, deepcopy(first)])
    reverse = edge_verification._exact_rows_by_signature([second, first])

    assert forward == reverse
    assert len(next(iter(forward.values()))) == 2


def test_binding_set_counts_only_independent_trusted_sources(monkeypatch) -> None:
    monkeypatch.setattr(edge_verification, "verify_reaction_step", _trusted_proof)
    rows = [
        _exact_row("doi:10.1000/one", "row:article"),
        _exact_row("doi:10.1000/one", "row:si", scope="supporting_information"),
        _exact_row("doi:10.1000/two", "row:independent"),
        _exact_row("doi:10.1000/untrusted", "row:untrusted"),
    ]

    binding_set = edge_verification._edge_evidence_binding_set(
        product="CC=O",
        reactants=["CCO"],
        exact_rows=rows,
    )

    assert binding_set["binding_count"] == 4
    assert binding_set["trusted_binding_count"] == 3
    assert binding_set["independent_trusted_source_groups"] == [
        "doi:10.1000/one",
        "doi:10.1000/two",
    ]
    assert binding_set["independent_trusted_source_group_count"] == 2
    assert binding_set["corroborated"] is True
    assert binding_set["semantics"]["proof_tier_is_orthogonal"] is True
    assert binding_set["content_sha256"] == edge_verification._digest(
        {key: value for key, value in binding_set.items() if key != "content_sha256"}
    )


def test_untrusted_rows_never_create_corroboration(monkeypatch) -> None:
    monkeypatch.setattr(edge_verification, "verify_reaction_step", _trusted_proof)
    rows = [
        _exact_row("doi:10.1000/untrusted-a", "row:a"),
        _exact_row("doi:10.1000/untrusted-b", "row:b"),
    ]

    binding_set = edge_verification._edge_evidence_binding_set(
        product="CC=O",
        reactants=["CCO"],
        exact_rows=deepcopy(rows),
    )

    assert binding_set["trusted_binding_count"] == 0
    assert binding_set["independent_trusted_source_groups"] == []
    assert binding_set["corroborated"] is False


def _verification_report(binding_set: dict, *, product: str = "CC=O", reactants: list[str] | None = None) -> dict:
    reactant_rows = list(reactants or ["CCO"])
    reaction_digest = edge_verification.canonical_reaction_digest(product, reactant_rows)
    step_proof = {
        "reaction_digest": reaction_digest,
        "validator_version": edge_verification.REACTION_STEP_VERIFIER_VERSION,
    }
    step_proof["proof_digest"] = edge_verification._digest(step_proof)
    candidate = {
        "product_smiles": product,
        "reactant_smiles": reactant_rows,
        "edge_evidence_binding_set": deepcopy(binding_set),
    }
    payload = {
        "schema_version": edge_verification.CODEX_EDGE_VERIFICATION_SCHEMA,
        "reaction_step_verifier_version": edge_verification.REACTION_STEP_VERIFIER_VERSION,
        "edge_count": 1,
        "edge_verifications": [
            {
                "step_id": "step:test",
                "product_smiles": product,
                "reactant_smiles": reactant_rows,
                "materialized_candidate": candidate,
                "step_proof": step_proof,
                "edge_evidence_binding_set": deepcopy(binding_set),
            }
        ],
    }
    payload["content_sha256"] = edge_verification._digest(payload)
    return payload


def test_invalid_original_reactant_rejects_exact_row_entirely() -> None:
    malformed = _exact_row("doi:10.1000/malformed", "row:malformed")
    malformed["reactant_smiles"] = ["CCO", "definitely-not-smiles"]

    assert edge_verification._exact_rows_by_signature([malformed]) == {}


def test_projection_binds_report_digest_and_edge_chemistry(monkeypatch) -> None:
    monkeypatch.setattr(edge_verification, "verify_reaction_step", _trusted_proof)
    binding_set = edge_verification._edge_evidence_binding_set(
        product="CC=O",
        reactants=["CCO"],
        exact_rows=[_exact_row("doi:10.1000/one", "row:one")],
    )
    valid_report = _verification_report(binding_set)

    projection = edge_verification.project_edge_evidence_binding_sets(valid_report)
    assert projection["accepted"] is True
    assert projection["source_verification_report_sha256"] == valid_report["content_sha256"]
    assert list(projection["by_reaction_digest"]) == [binding_set["reaction_digest"]]

    mismatched = _verification_report(binding_set, product="CCC=O")
    rejected = edge_verification.project_edge_evidence_binding_sets(mismatched)
    assert rejected["accepted"] is False
    assert rejected["by_reaction_digest"] == {}
    assert any(
        "edge_evidence_binding_set_expected_reaction_mismatch" in row["reasons"]
        for row in rejected["rejected"]
    )

    tampered_report = deepcopy(valid_report)
    tampered_report["edge_count"] = 2
    invalid_report = edge_verification.project_edge_evidence_binding_sets(tampered_report)
    assert invalid_report["accepted"] is False
    assert invalid_report["by_reaction_digest"] == {}
    assert invalid_report["rejected"][0]["reasons"] == [
        "edge_verification_report_digest_invalid"
    ]


def test_binding_revalidation_rejects_self_hashed_fake_authority(monkeypatch) -> None:
    monkeypatch.setattr(edge_verification, "verify_reaction_step", _trusted_proof)
    binding_set = edge_verification._edge_evidence_binding_set(
        product="CC=O",
        reactants=["CCO"],
        exact_rows=[_exact_row("doi:10.1000/one", "row:one")],
    )
    forged = deepcopy(binding_set)
    forged["bindings"][0]["authority"] = "model"
    forged["content_sha256"] = edge_verification._digest(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )

    validated, reasons = edge_verification.validate_edge_evidence_binding_set(
        forged,
        expected_reaction_digest=forged["reaction_digest"],
    )
    assert validated == {}
    assert "edge_evidence_binding:0:trusted_precedent_invalid" in reasons

from __future__ import annotations

from cascade_planner.routes.consensus import (
    consensus_to_blackboard_proposals,
    fuse_route_candidates,
    validate_retrosynthesis_report_payload,
)


def candidate(
    candidate_id: str,
    *,
    product: str = "CCO",
    precursors: list[str] | None = None,
    channel: str = "codex_strategy",
    evidence_level: str = "model_only",
    source_refs: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": candidate_id,
        "product_smiles": product,
        "precursor_smiles": precursors or ["CC=O"],
        "reaction_family": "carbonyl reduction",
        "transformation_rationale": "disconnect alcohol to aldehyde",
        "source_channel": channel,
        "source_refs": source_refs or [],
        "evidence_refs": [],
        "evidence_level": evidence_level,
        "confidence": "medium",
        "conditions": [],
        "catalyst": "",
        "enzyme": "",
        "limitations": ["requires forward validation"],
        "required_validation": ["forward_reconstruction"],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }


def test_model_children_share_one_correlation_group_and_never_become_executable() -> None:
    consensus = fuse_route_candidates(
        [
            candidate("strategy", channel="codex_strategy"),
            candidate("enzyme", channel="codex_chemoenzymatic"),
        ],
        case_id="ethanol",
        target_smiles="OCC",
    )

    assert consensus["accepted"]
    assert len(consensus["proposals"]) == 1
    proposal = consensus["proposals"][0]
    assert proposal["source_channel_count"] == 2
    assert proposal["independent_support_groups"] == ["codex_model"]
    assert proposal["status"] == "model_hypothesis"
    adapted = consensus_to_blackboard_proposals(consensus)[0]
    assert adapted["proposal_type"] == "strategic"
    assert adapted["executable"] is False


def test_target_mismatch_is_quarantined_and_stereo_is_not_collapsed() -> None:
    mismatch = fuse_route_candidates(
        [candidate("wrong", product="CCN")],
        target_smiles="CCO",
    )
    assert not mismatch["accepted"]
    assert mismatch["rejected_candidates"][0]["reasons"] == [
        "candidate_product_does_not_match_requested_target"
    ]

    stereo = fuse_route_candidates(
        [
            candidate("left", product="C[C@H](O)F", precursors=["CC(=O)F"]),
            candidate("right", product="C[C@@H](O)F", precursors=["CC(=O)F"]),
        ]
    )
    assert len(stereo["proposals"]) == 2
    assert len({row["consensus_id"] for row in stereo["proposals"]}) == 2


def test_model_claimed_exact_literature_is_downgraded_until_trusted_adapter() -> None:
    invalid = fuse_route_candidates(
        [candidate("bad_exact", channel="codex_literature", evidence_level="literature_exact")],
        target_smiles="CCO",
    )
    assert invalid["accepted"]
    assert invalid["proposals"][0]["status"] == "model_hypothesis"

    valid = fuse_route_candidates(
        [
            candidate(
                "exact",
                channel="codex_literature",
                evidence_level="literature_exact",
                source_refs=["doi:10.1000/example"],
            )
        ],
        target_smiles="CCO",
    )
    assert valid["proposals"][0]["status"] == "model_hypothesis"
    assert valid["proposals"][0]["independent_support_groups"] == ["codex_model"]
    adapted = consensus_to_blackboard_proposals(valid)[0]
    assert adapted["proposal_type"] == "strategic"
    assert adapted["executable"] is False

    explicitly_trusted_codex = fuse_route_candidates(
        [
            candidate(
                "still_model_only",
                channel="codex_literature",
                evidence_level="literature_exact",
                source_refs=["doi:10.1000/model-claim"],
            )
        ],
        target_smiles="CCO",
        allow_trusted_literature_exact_evidence=True,
    )
    assert explicitly_trusted_codex["proposals"][0]["status"] == "model_hypothesis"
    assert explicitly_trusted_codex["proposals"][0]["independent_support_groups"] == ["codex_model"]

    trusted = fuse_route_candidates(
        [
            candidate(
                "trusted_exact",
                channel="literature_exact",
                evidence_level="literature_exact",
                source_refs=["doi:10.1000/example"],
            )
        ],
        target_smiles="CCO",
        allow_trusted_literature_exact_evidence=True,
    )
    assert trusted["proposals"][0]["status"] == "evidence_backed_draft"


def test_invented_doi_from_codex_is_not_an_independent_support_group() -> None:
    consensus = fuse_route_candidates(
        [
            candidate("strategy", channel="codex_strategy"),
            candidate(
                "claimed_literature",
                channel="codex_literature",
                evidence_level="literature_exact",
                source_refs=["doi:10.1000/invented"],
            ),
        ],
        target_smiles="CCO",
    )

    proposal = consensus["proposals"][0]
    assert proposal["independent_support_groups"] == ["codex_model"]
    assert consensus["source_summary"]["multi_source_proposals"] == 0


def test_unattributed_legacy_hypothesis_is_not_an_independent_support_group() -> None:
    consensus = fuse_route_candidates(
        [
            candidate("strategy", channel="codex_strategy"),
            candidate("unbound", channel="other"),
        ],
        target_smiles="CCO",
    )

    proposal = consensus["proposals"][0]
    assert proposal["independent_support_groups"] == ["codex_model"]
    assert proposal["source_diversity"] == 1
    assert consensus["source_summary"]["multi_source_proposals"] == 0


def test_article_si_doi_pubmed_pmc_and_local_aliases_count_as_one_source() -> None:
    rows = [
        candidate(
            "doi",
            channel="literature_exact",
            evidence_level="literature_exact",
            source_refs=["DOI:10.1021/JA00083A066"],
        ),
        candidate(
            "doi-url-pmid-bridge",
            channel="literature_exact",
            evidence_level="literature_exact",
            source_refs=[
                "https://doi.org/10.1021/ja00083a066",
                "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            ],
        ),
        candidate(
            "pmid-pmc-bridge",
            channel="literature_exact",
            evidence_level="literature_exact",
            source_refs=["pmid:012345678", "PMC:PMC987654"],
        ),
        candidate(
            "pmc-local-bridge",
            channel="literature_exact",
            evidence_level="literature_exact",
            source_refs=[
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC987654/",
                "local_pdf:taxol_holton_article.pdf#page=3",
            ],
        ),
        candidate(
            "supporting-info-copy",
            channel="literature_exact",
            evidence_level="literature_exact",
            source_refs=["local_pdf:taxol_holton_article.pdf#supporting-information"],
        ),
    ]

    consensus = fuse_route_candidates(
        rows,
        target_smiles="CCO",
        allow_trusted_literature_exact_evidence=True,
    )

    proposal = consensus["proposals"][0]
    assert proposal["support_count"] == 5
    assert proposal["independent_support_groups"] == [
        "literature:doi:10.1021/ja00083a066"
    ]
    assert proposal["source_diversity"] == 1
    assert consensus["source_summary"]["multi_source_proposals"] == 0


def test_source_claimed_validation_never_becomes_executable() -> None:
    consensus = fuse_route_candidates(
        [candidate("self_validated", evidence_level="validated")],
        target_smiles="CCO",
    )

    assert consensus["accepted"] is False
    assert "untrusted_self_validated_evidence_claim" in consensus["rejected_candidates"][0]["reasons"]

    trusted = fuse_route_candidates(
        [candidate("deterministically_validated", evidence_level="validated")],
        target_smiles="CCO",
        allow_trusted_validated_evidence=True,
    )
    assert trusted["proposals"][0]["status"] == "validation_claimed_draft"
    adapted = consensus_to_blackboard_proposals(trusted)[0]
    assert adapted["proposal_type"] == "strategic"
    assert adapted["executable"] is False

    report = {
        "schema_version": "retrosynthesis_proposal_report.v1",
        "case_id": "case",
        "agent_role": "coordinator",
        "target_smiles": "CCO",
        "candidates": [candidate("model_self_validated", evidence_level="validated")],
        "evidence_refs": [],
        "limitations": [],
        "no_solved_claim": True,
    }
    assert (
        "proposal_report_candidate:0:untrusted_self_validated_evidence_claim"
        in validate_retrosynthesis_report_payload(report)
    )


def test_report_validator_rejects_solved_claim() -> None:
    payload = {
        "schema_version": "retrosynthesis_proposal_report.v1",
        "case_id": "case",
        "agent_role": "coordinator",
        "target_smiles": "CCO",
        "candidates": [candidate("one")],
        "evidence_refs": [],
        "limitations": [],
        "no_solved_claim": True,
        "route_status": "solved",
    }
    assert "proposal_report_direct_solved_claim" in validate_retrosynthesis_report_payload(payload)

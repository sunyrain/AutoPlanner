from __future__ import annotations

from cascade_planner.legacy.routes_runtime.adapters import _candidates_from_consensus
from cascade_planner.routes.consensus import fuse_route_candidates
from cascade_planner.legacy.routes_runtime.graph import (
    assemble_route_consensus_graph,
    make_route_consensus_expansion,
    validate_route_consensus_expansion,
)


def _candidate(
    candidate_id: str,
    *,
    family: str,
    channel: str = "codex_strategy",
    evidence_level: str = "model_only",
    host_binding: str = "",
) -> dict:
    row = {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": candidate_id,
        "product_smiles": "CCO",
        "precursor_smiles": ["CC=O"],
        "reaction_family": family,
        "transformation_rationale": f"rationale:{candidate_id}",
        "source_channel": channel,
        "source_refs": ["doi:10.1000/trusted"] if host_binding else [],
        "evidence_refs": [],
        "evidence_level": evidence_level,
        "confidence": "high" if host_binding else "low",
        "conditions": [],
        "catalyst": "",
        "enzyme": "",
        "limitations": [],
        "required_validation": [],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }
    if host_binding:
        row["_host_authority_binding"] = host_binding
    return row


def test_authority_bound_family_beats_lexically_earlier_codex_label() -> None:
    consensus = fuse_route_candidates(
        [
            _candidate("model", family="aaa model label"),
            _candidate(
                "trusted",
                family="zeta trusted label",
                channel="literature_exact",
                evidence_level="literature_exact",
                host_binding="validated_source_detail_literature_step",
            ),
        ],
        target_smiles="CCO",
    )

    proposal = consensus["proposals"][0]
    assert proposal["reaction_family"] == "zeta trusted label"
    selection = proposal["reaction_family_selection"]
    assert selection["status"] == "selected_authority_bound"
    assert selection["authority_bound"] is True
    assert selection["authority_evidence_level"] == "literature_exact"
    assert selection["candidate_ids"] == ["trusted"]
    records = {row["candidate_id"]: row for row in proposal["source_records"]}
    assert records["model"]["reaction_family"] == "aaa model label"
    assert records["trusted"]["reaction_family"] == "zeta trusted label"
    assert records["model"]["transformation_rationale"] == "rationale:model"
    assert records["trusted"]["transformation_rationale"] == "rationale:trusted"


def test_conflicting_correlated_codex_families_remain_ambiguous() -> None:
    consensus = fuse_route_candidates(
        [
            _candidate("strategy", family="side-chain amide coupling"),
            _candidate(
                "chemo",
                family="biocatalytic asymmetric amination",
                channel="codex_chemoenzymatic",
            ),
        ],
        target_smiles="CCO",
    )

    proposal = consensus["proposals"][0]
    assert proposal["reaction_family"] == "unspecified"
    assert proposal["reaction_family_selection"]["status"] == "ambiguous"
    assert proposal["reaction_family_selection"]["support_groups"] == [
        "codex_model"
    ]
    assert proposal["independent_support_groups"] == ["codex_model"]
    assert consensus["source_summary"]["multi_source_proposals"] == 0

    round_trip = {
        row["candidate_id"]: row for row in _candidates_from_consensus(consensus)
    }
    assert round_trip["strategy"]["reaction_family"] == (
        "side-chain amide coupling"
    )
    assert round_trip["chemo"]["reaction_family"] == (
        "biocatalytic asymmetric amination"
    )
    assert round_trip["strategy"]["transformation_rationale"] == (
        "rationale:strategy"
    )
    assert round_trip["chemo"]["transformation_rationale"] == "rationale:chemo"


def test_graph_family_selection_ignores_proposal_rank_as_authority() -> None:
    model_consensus = fuse_route_candidates(
        [_candidate("model", family="aaa model label")],
        target_smiles="CCO",
    )
    trusted_consensus = fuse_route_candidates(
        [
            _candidate(
                "trusted",
                family="zeta trusted label",
                channel="literature_exact",
                evidence_level="literature_exact",
                host_binding="validated_source_detail_literature_step",
            )
        ],
        target_smiles="CCO",
    )
    model_consensus["proposals"][0]["rank_score"] = 0.99
    trusted_consensus["proposals"][0]["rank_score"] = 0.01
    graph = assemble_route_consensus_graph(
        [
            make_route_consensus_expansion(
                model_consensus,
                requested_product_smiles="CCO",
                consensus_ref="model",
            ),
            make_route_consensus_expansion(
                trusted_consensus,
                requested_product_smiles="CCO",
                consensus_ref="trusted",
            ),
        ],
        case_id="case",
        target_smiles="CCO",
    )

    assert len(graph["steps"]) == 1
    step = graph["steps"][0]
    assert step["rank_score"] == 0.99
    assert step["reaction_family"] == "zeta trusted label"
    assert step["reaction_family_selection"]["status"] == (
        "selected_authority_bound"
    )
    assert step["reaction_family_selection"]["candidate_ids"] == ["trusted"]
    assert step["reaction_family_authority_bound"] is True
    assert step["reaction_family_authority_evidence_level"] == "literature_exact"
    assert step["authority_bound"] is True
    assert step["authority_evidence_level"] == "literature_exact"
    assert step["edge_authority_selection"]["candidate_ids"] == ["trusted"]


def test_graph_rejects_family_selection_that_disagrees_with_source_records() -> None:
    consensus = fuse_route_candidates(
        [_candidate("model", family="host-derived family")],
        target_smiles="CCO",
    )
    consensus["proposals"][0]["reaction_family_selection"]["value"] = (
        "forged family"
    )
    expansion = make_route_consensus_expansion(
        consensus,
        requested_product_smiles="CCO",
    )

    assert (
        "expansion_proposal:0:reaction_family_selection_source_mismatch"
        in validate_route_consensus_expansion(expansion)
    )


def test_legacy_family_fallback_cannot_replay_serialized_authority() -> None:
    trusted = fuse_route_candidates(
        [
            _candidate(
                "trusted",
                family="trusted family",
                channel="literature_exact",
                evidence_level="literature_exact",
                host_binding="validated_source_detail_literature_step",
            )
        ],
        target_smiles="CCO",
    )
    legacy_proposal = dict(trusted["proposals"][0])
    legacy_proposal.pop("source_records")
    legacy_proposal.pop("reaction_family_selection")
    legacy = {**trusted, "proposals": [legacy_proposal]}
    expansion = make_route_consensus_expansion(
        legacy,
        requested_product_smiles="CCO",
    )

    assert validate_route_consensus_expansion(expansion) == []
    graph = assemble_route_consensus_graph(
        [expansion],
        case_id="case",
        target_smiles="CCO",
    )
    step = graph["steps"][0]
    assert step["reaction_family"] == "trusted family"
    assert step["reaction_family_selection"]["status"] == "selected_advisory"
    assert step["reaction_family_authority_bound"] is False
    assert step["reaction_family_authority_evidence_level"] == "model_only"
    assert step["authority_bound"] is False
    assert step["authority_evidence_level"] == "model_only"

    forged_proposal = dict(legacy_proposal)
    forged_proposal["reaction_family_selection"] = {
        "schema_version": "reaction_family_selection.v1",
        "status": "selected_authority_bound",
        "value": "trusted family",
        "authority_basis": "forged",
        "authority_bound": True,
        "authority_evidence_level": "validated",
        "support_groups": ["forged:group"],
        "candidate_ids": ["forged"],
        "alternatives": [],
    }
    forged = {**trusted, "proposals": [forged_proposal]}
    forged_expansion = make_route_consensus_expansion(
        forged,
        requested_product_smiles="CCO",
    )
    assert (
        "expansion_proposal:0:legacy_reaction_family_authority_not_unbound"
        in validate_route_consensus_expansion(forged_expansion)
    )

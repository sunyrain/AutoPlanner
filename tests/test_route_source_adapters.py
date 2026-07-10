from __future__ import annotations

from cascade_planner.routes.adapters import rebuild_consensus_graph_from_blackboard
from cascade_planner.routes.consensus import fuse_route_candidates
from cascade_planner.routes.graph import make_route_consensus_expansion


def _candidate(candidate_id: str, *, product: str, precursor: str, channel: str) -> dict:
    return {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": candidate_id,
        "product_smiles": product,
        "precursor_smiles": [precursor],
        "reaction_family": "reduction",
        "transformation_rationale": "test source adapter",
        "source_channel": channel,
        "source_refs": [],
        "evidence_refs": [],
        "evidence_level": "computational" if channel == "chem_enzy" else "model_only",
        "confidence": "medium",
        "conditions": [],
        "catalyst": "",
        "enzyme": "",
        "limitations": [],
        "required_validation": ["forward_reconstruction"],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }


def test_blackboard_rebuild_combines_codex_and_chemenzy_without_evidence_laundering() -> None:
    root_consensus = fuse_route_candidates(
        [_candidate("codex-root", product="CCO", precursor="CC=O", channel="codex_strategy")],
        target_smiles="CCO",
    )
    middle_consensus = fuse_route_candidates(
        [_candidate("codex-middle", product="CC=O", precursor="C", channel="codex_strategy")],
        target_smiles="CC=O",
    )
    board = {
        "case_id": "ethanol",
        "target_profile": {"target_smiles": "CCO"},
        "route_consensus": root_consensus,
        "codex_agent_team": {
            "accepted": True,
            "coordinator": {"run_record_ref": "root-record.json"},
            "route_consensus_expansions": [
                make_route_consensus_expansion(
                    root_consensus,
                    requested_product_smiles="CCO",
                    depth=0,
                ),
                make_route_consensus_expansion(
                    middle_consensus,
                    requested_product_smiles="CC=O",
                    depth=1,
                ),
            ],
        },
        "retrosynthetic_proposals": [
            {
                "schema_version": "retrosynthetic_proposal.v1",
                "proposal_id": "chemenzy-root",
                "source_type": "chem_enzy",
                "proposal_type": "strategic",
                "proposal_label": "carbonyl reduction",
                "target_smiles": "OCC",
                "precursor_smiles": "CC=O",
                "confidence": "medium_high",
                "source_refs": ["guided_chemenzy_result.json"],
                "evidence_refs": [],
                "risk_flags": [],
                "required_verification": ["stock_audit"],
            }
        ],
    }

    rebuilt = rebuild_consensus_graph_from_blackboard(board, max_depth=2)

    assert rebuilt["accepted"] is True
    proposal = rebuilt["consensus"]["proposals"][0]
    assert proposal["source_channels"] == ["chem_enzy", "codex_strategy"]
    assert proposal["independent_support_groups"] == ["codex_model", "computational:chem_enzy"]
    assert proposal["support_count"] == 2
    assert proposal["status"] == "model_hypothesis"
    assert rebuilt["graph"]["has_hypotheses"] is True
    assert len(rebuilt["graph"]["steps"]) == 2
    assert rebuilt["graph"]["semantics"]["executable"] is False


def test_rebuild_skips_legacy_copy_of_existing_consensus() -> None:
    consensus = fuse_route_candidates(
        [_candidate("codex-root", product="CCO", precursor="CC=O", channel="codex_strategy")],
        target_smiles="CCO",
    )
    board = {
        "case_id": "ethanol",
        "target_profile": {"target_smiles": "CCO"},
        "route_consensus": consensus,
        "retrosynthetic_proposals": [
            {
                "proposal_id": f"consensus:{consensus['proposals'][0]['consensus_id']}",
                "source_type": "multi_source_consensus",
                "target_smiles": "CCO",
                "precursor_smiles": "CC=O",
            }
        ],
    }

    rebuilt = rebuild_consensus_graph_from_blackboard(board)

    assert rebuilt["candidate_count"] == 1
    assert rebuilt["consensus"]["proposals"][0]["support_count"] == 1

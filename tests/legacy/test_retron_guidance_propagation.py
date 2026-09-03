from __future__ import annotations

from cascade_planner.agent.chem_enzy_policy import (
    apply_chem_enzy_search_policy,
    chem_enzy_guidance_contract,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig
from cascade_planner.legacy.harness_runtime.agent_action_planner import (
    build_child_expansion_payload_from_blackboard,
    build_guided_chemenzy_payload_from_blackboard,
)
from cascade_planner.legacy.harness_runtime.retrosynthetic_proposals import (
    recursive_tasks_from_retrosynthetic_proposals,
)
from cascade_planner.routes.consensus import (
    consensus_to_blackboard_proposals,
    fuse_route_candidates,
)


def _codex_candidate(*, candidate_id: str, retron: str) -> dict:
    return {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": candidate_id,
        "product_smiles": "CCO",
        "precursor_smiles": ["CC=O"],
        "reaction_family": "carbonyl reduction",
        "product_retron_type": retron,
        "transformation_rationale": "disconnect the alcohol to an aldehyde",
        "source_channel": "codex_strategy",
        "source_refs": [],
        "evidence_refs": [],
        "evidence_level": "model_only",
        "confidence": "medium",
        "conditions": [],
        "catalyst": "",
        "enzyme": "",
        "limitations": ["requires deterministic validation"],
        "required_validation": ["forward_reconstruction"],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }


def test_codex_retron_survives_consensus_recursive_task_and_guided_policy() -> None:
    consensus = fuse_route_candidates(
        [_codex_candidate(candidate_id="codex:reduction", retron="secondary_alcohol_carbonyl")],
        case_id="ethanol",
        target_smiles="CCO",
    )
    proposal = consensus_to_blackboard_proposals(consensus)[0]

    assert proposal["reaction_family"] == "carbonyl reduction"
    assert proposal["product_retron_type"] == "secondary_alcohol_carbonyl"
    assert proposal["retron_authority"] == "producer_advisory_only"

    board = {
        "case_id": "ethanol",
        "target_profile": {"target_smiles": "CCO"},
        "current_belief": {"constraints": {}},
        "retrosynthetic_proposals": [proposal],
        "terminal_blacklist": [],
    }
    task = recursive_tasks_from_retrosynthetic_proposals(board, [proposal])[0]
    assert task["reaction_family"] == "carbonyl reduction"
    assert task["derived_from_retron"] == "secondary_alcohol_carbonyl"

    guided = build_guided_chemenzy_payload_from_blackboard(board)["search_policy"]
    assert "carbonyl reduction" in guided["source_budget"]["preferred_reaction_classes"]
    assert guided["source_budget"]["preferred_retrons"] == [
        "secondary_alcohol_carbonyl"
    ]

    guidance = chem_enzy_guidance_contract(guided)
    assert "carbonyl reduction" in guidance["preferred_reaction_classes"]
    assert guidance["preferred_retrons"] == ["secondary_alcohol_carbonyl"]
    assert guidance["raw_reaction_injection"] is False


def test_recursive_child_policy_keeps_reaction_and_retron_priors() -> None:
    consensus = fuse_route_candidates(
        [_codex_candidate(candidate_id="codex:child", retron="secondary_alcohol_carbonyl")],
        target_smiles="CCO",
    )
    proposal = consensus_to_blackboard_proposals(consensus)[0]
    board = {
        "case_id": "ethanol",
        "target_profile": {"target_smiles": "CCO"},
        "recursive_hypothesis_tasks": recursive_tasks_from_retrosynthetic_proposals(
            {"target_profile": {"target_smiles": "CCO"}},
            [proposal],
        ),
        "budget_state": {"child_target_runs": 0, "max_child_target_runs": 2},
        "terminal_blacklist": [],
        "action_history": [],
    }

    target = build_child_expansion_payload_from_blackboard(board)["subgoal_targets"][0]
    policy = target["chem_enzy_search_policy"]
    assert target["reaction_family"] == "carbonyl reduction"
    assert target["derived_from_retron"] == "secondary_alcohol_carbonyl"
    assert "carbonyl reduction" in policy["source_budget"]["preferred_reaction_classes"]
    assert policy["source_budget"]["preferred_retrons"] == [
        "secondary_alcohol_carbonyl"
    ]


def test_policy_projection_preserves_bounded_classification_priors_without_reaction_injection() -> None:
    policy = {
        "schema_version": "chem_enzy_search_policy.v1",
        "policy_id": "retron-prior",
        "operator_id": "test",
        "case_id": "test",
        "evidence_refs": ["test:proposal"],
        "terminal_blacklist": [],
        "anchor_whitelist": [],
        "preferred_subgoal": {},
        "source_budget": {
            "preferred_reaction_classes": ["amide coupling"],
            "preferred_retrons": ["amide_c_n_disconnection"],
        },
        "rerun_reason": "test bounded priors",
        "budget": {
            "max_reruns": 1,
            "max_iterations": 5,
            "max_depth": 3,
            "expansion_topk": 10,
        },
        "mode": "guided",
        "compiler_metadata": {"not_raw_reaction_injection": True},
    }

    config = apply_chem_enzy_search_policy(
        RouteSearchConfig(target_smiles="CCO"),
        policy,
    )
    guidance = config.search_flags["chem_enzy_guidance"]
    source_policy = config.search_flags["cascade_source_policy"]

    assert guidance["preferred_reaction_classes"] == ["amide coupling"]
    assert guidance["preferred_retrons"] == ["amide_c_n_disconnection"]
    assert source_policy["preferred_reaction_classes"] == ["amide coupling"]
    assert source_policy["preferred_retrons"] == ["amide_c_n_disconnection"]
    assert source_policy["reaction_and_retron_priors_are_advisory_only"] is True
    assert guidance["raw_reaction_injection"] is False

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from cascade_planner.application.frontier_ledger import (
    _normalize_graph,
    _reaction_graph_identity,
)
from cascade_planner.harness.agent_action_planner import (
    _canonical_graph_min_depths,
    _guided_canonical_graph_identity,
    build_guided_chemenzy_payload_from_blackboard,
    select_guided_retrosynthetic_proposals,
)


ROOT = "CCOC(N)=O"
OPEN_FRONTIER = "CCN"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_graph() -> dict:
    step = {
        "schema_version": "route_consensus_step.v1",
        "step_id": "step:root",
        "signature": f"{ROOT}<-{OPEN_FRONTIER}",
        "product_smiles": ROOT,
        "precursor_smiles": [OPEN_FRONTIER],
        "product_node_id": "mol:root",
        "precursor_node_ids": ["mol:frontier"],
        "proposal_ids": ["canonical:root"],
        "source_refs": [],
        "evidence_refs": [],
    }
    return {
        "schema_version": "route_consensus_graph.v1",
        "case_id": "guided-priority",
        "target_smiles": ROOT,
        "root_node_id": "mol:root",
        "nodes": [
            {"node_id": "mol:root", "smiles": ROOT, "min_depth": 0},
            {
                "node_id": "mol:frontier",
                "smiles": OPEN_FRONTIER,
                "min_depth": 1,
            },
        ],
        "steps": [step],
    }


def _graph_identity(graph: dict) -> str:
    steps = [
        {
            "step_id": row["step_id"],
            "signature": row["signature"],
            "product_smiles": row["product_smiles"],
            "precursor_smiles": sorted(row["precursor_smiles"]),
        }
        for row in graph["steps"]
    ]
    return _digest(
        {
            "schema_version": "route_consensus_graph.v1",
            "case_id": graph["case_id"],
            "target_smiles": graph["target_smiles"],
            "steps": sorted(
                steps,
                key=lambda row: (row["step_id"], row["signature"]),
            ),
        }
    )


def _ledger(graph: dict) -> dict:
    ledger = {
        "schema_version": "frontier_ledger.v1",
        "input_bindings": {
            "schema_version": "frontier_ledger_input_bindings.v1",
            "graph_identity_sha256": _graph_identity(graph),
            "frontier_queue_content_sha256": "a" * 64,
            "frontier_queue_revision": 3,
            "campaign_policy_sha256": "b" * 64,
        },
        "root": {"canonical_smiles": ROOT},
        "molecules": {
            ROOT: {
                "proposal": {"state": "expanded"},
                "work": {
                    "open": False,
                    "proposal_expansion_allowed": False,
                },
                "stock": {"closed": False},
            },
            OPEN_FRONTIER: {
                "proposal": {"state": "frontier"},
                "work": {
                    "open": True,
                    "proposal_expansion_allowed": True,
                },
                "stock": {"closed": False},
            },
        },
        "edges": {},
        "summary": {},
        "input_validation": {
            "graph": {"valid": True},
            "frontier_queue": {"valid": True},
            "reaction_proof_state": {"valid": True},
            "stock_authority": {
                "valid": True,
                "authority_boundary": "current_host_stock_provider_replay",
            },
        },
    }
    ledger["content_sha256"] = _digest(ledger)
    return ledger


def _proposal_rows() -> list[dict]:
    low_precursors = [
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CO",
        "CCO",
        "CCN",
        "CCCl",
        "CCF",
        "c1ccccc1",
        "C1CCCCC1",
        "CC(=O)O",
    ]
    rows = [
        {
            "proposal_id": f"low:{index:02d}",
            "target_smiles": ROOT,
            "precursor_smiles": precursor,
            "confidence": "high",
            "score": 10_000,
            "validated": True,
            "authority_bound": True,
        }
        for index, precursor in enumerate(low_precursors)
    ]
    rows.append(
        {
            "proposal_id": "codex:late-open-frontier",
            "target_smiles": OPEN_FRONTIER,
            "precursor_smiles": "CN",
            "confidence": "low",
            "score": -10_000,
            "validated": False,
            "authority_bound": False,
        }
    )
    return rows


def _board() -> dict:
    graph = _canonical_graph()
    ledger = _ledger(graph)
    return {
        "case_id": "guided-priority",
        "target_profile": {"target_smiles": ROOT},
        "current_belief": {"constraints": {}},
        "terminal_blacklist": [],
        "canonical_route_consensus_graph": graph,
        "frontier_ledger": ledger,
        "frontier_ledger_summary": {
            "schema_version": "frontier_ledger_summary.v1",
            "input_valid": True,
            "ledger_validation_accepted": True,
            "frontier_ledger_content_sha256": ledger["content_sha256"],
        },
        "retrosynthetic_proposals": _proposal_rows(),
    }


def test_late_open_frontier_proposal_enters_capacity_and_replaces_low_priority() -> None:
    board = _board()
    # Node annotations are a presentation projection and are not covered by
    # the graph identity.  Depth must be recomputed from the bound steps.
    board["canonical_route_consensus_graph"]["nodes"][1]["min_depth"] = 999

    selected, audit = select_guided_retrosynthetic_proposals(board, limit=12)

    selected_ids = [row["proposal_id"] for row in selected]
    assert selected_ids[0] == "codex:late-open-frontier"
    assert "codex:late-open-frontier" in selected_ids
    assert len(selected_ids) == 12
    assert len(set(audit["dropped_proposal_ids"]) & {f"low:{i:02d}" for i in range(12)}) == 1
    assert audit["authoritative_frontier_ledger"] is True
    assert audit["ranking"][0]["frontier_priority_tier"] == "open_expandable_frontier"
    assert audit["ranking"][0]["canonical_min_depth"] == 1

    payload = build_guided_chemenzy_payload_from_blackboard(board)
    source_budget = payload["search_policy"]["source_budget"]
    assert source_budget["retrosynthetic_proposals"][0]["proposal_id"] == (
        "codex:late-open-frontier"
    )
    assert source_budget["retrosynthetic_proposal_selection_audit"][
        "selected_proposal_ids"
    ] == selected_ids


def test_model_reported_authority_confidence_and_evidence_do_not_change_ranking() -> None:
    board = _board()
    first_rows, first_audit = select_guided_retrosynthetic_proposals(board, limit=5)
    tampered = deepcopy(board)
    for index, row in enumerate(tampered["retrosynthetic_proposals"]):
        row.update(
            {
                "confidence": "high" if index % 2 else "low",
                "score": 999_999 - index,
                "validated": index % 3 == 0,
                "authority_bound": index % 4 == 0,
                "evidence_refs": [f"self-reported:{index}"],
                "validation_tier": "L4" if index % 2 else "L0",
                "achieved_proof_level": 4 if index % 2 else 0,
            }
        )

    second_rows, second_audit = select_guided_retrosynthetic_proposals(
        tampered,
        limit=5,
    )

    assert [row["proposal_id"] for row in first_rows] == [
        row["proposal_id"] for row in second_rows
    ]
    assert first_audit["selected_proposal_ids"] == second_audit[
        "selected_proposal_ids"
    ]
    assert [row["proposal_id"] for row in first_audit["ranking"]] == [
        row["proposal_id"] for row in second_audit["ranking"]
    ]
    assert {
        "confidence",
        "score",
        "evidence_refs",
        "validated",
        "authority_bound",
    }.issubset(first_audit["ignored_self_reported_fields"])


def test_selection_is_deterministic_across_input_order_and_audits_diversity() -> None:
    board = _board()
    forward_rows, forward_audit = select_guided_retrosynthetic_proposals(
        board,
        limit=12,
    )
    reversed_board = deepcopy(board)
    reversed_board["retrosynthetic_proposals"].reverse()
    reverse_rows, reverse_audit = select_guided_retrosynthetic_proposals(
        reversed_board,
        limit=12,
    )

    assert [row["proposal_id"] for row in forward_rows] == [
        row["proposal_id"] for row in reverse_rows
    ]
    assert forward_audit["selected_proposal_ids"] == reverse_audit[
        "selected_proposal_ids"
    ]
    assert [row["proposal_id"] for row in forward_audit["ranking"]] == [
        row["proposal_id"] for row in reverse_audit["ranking"]
    ]
    assert all(
        any(reason.startswith("structural_diversity:") for reason in row["ranking_reasons"])
        for row in forward_audit["ranking"]
    )


def test_non_authoritative_ledger_is_fail_soft_and_cannot_set_priority() -> None:
    first = _board()
    first["frontier_ledger_summary"]["ledger_validation_accepted"] = False
    second = deepcopy(first)
    first["frontier_ledger"]["molecules"][ROOT]["work"]["open"] = True
    second["frontier_ledger"]["molecules"][OPEN_FRONTIER]["stock"]["closed"] = True
    # Keep both forged ledgers self-consistent.  Their apparent priorities must
    # still be ignored because the current host validation flag is negative.
    for board in (first, second):
        ledger = board["frontier_ledger"]
        ledger.pop("content_sha256")
        ledger["content_sha256"] = _digest(ledger)
        board["frontier_ledger_summary"]["frontier_ledger_content_sha256"] = ledger[
            "content_sha256"
        ]

    first_rows, first_audit = select_guided_retrosynthetic_proposals(first, limit=6)
    second_rows, second_audit = select_guided_retrosynthetic_proposals(second, limit=6)

    assert [row["proposal_id"] for row in first_rows] == [
        row["proposal_id"] for row in second_rows
    ]
    assert first_audit["authoritative_frontier_ledger"] is False
    assert second_audit["authoritative_frontier_ledger"] is False
    assert first_audit["selection_authority"] == (
        "stable_fail_soft_without_frontier_authority"
    )
    assert "frontier_ledger_summary_validation_not_accepted" in first_audit[
        "frontier_authority_reasons"
    ]
    assert all(
        row["frontier_priority_tier"] == "fail_soft_unranked_by_frontier"
        for row in first_audit["ranking"]
    )


def test_graph_identity_matches_ledger_contract_and_cycle_depth_is_recomputed() -> None:
    graph = {
        "schema_version": "route_consensus_graph.v1",
        "case_id": "identity-parity",
        "target_smiles": "CCOC(=O)N",
        "nodes": [
            {"node_id": "root", "smiles": "CCOC(N)=O", "min_depth": 800},
            {"node_id": "child", "smiles": "CCN", "min_depth": 900},
            {"node_id": "co", "smiles": "CCO", "min_depth": 901},
        ],
        "steps": [
            {
                "schema_version": "route_consensus_step.v1",
                "step_id": "step:forward",
                "signature": "CCOC(N)=O<-CCN.CCO",
                "product_smiles": "CCOC(=O)N",
                "precursor_smiles": ["CCO", "CCN"],
                "product_node_id": "root",
                "precursor_node_ids": ["co", "child"],
            },
            {
                "schema_version": "route_consensus_step.v1",
                "step_id": "step:return",
                "signature": "CCN<-CCOC(N)=O",
                "product_smiles": "CCN",
                "precursor_smiles": ["CCOC(=O)N"],
                "product_node_id": "child",
                "precursor_node_ids": ["root"],
            },
        ],
    }
    normalized, normalization_reasons = _normalize_graph(graph)
    selector_identity, selector_reasons = _guided_canonical_graph_identity(graph)

    assert normalization_reasons == []
    assert selector_reasons == []
    assert selector_identity == _reaction_graph_identity(normalized)
    expected_depths = {"CCOC(N)=O": 0, "CCN": 1, "CCO": 1}
    assert _canonical_graph_min_depths(graph) == expected_depths
    reversed_graph = deepcopy(graph)
    reversed_graph["steps"].reverse()
    assert _canonical_graph_min_depths(reversed_graph) == expected_depths

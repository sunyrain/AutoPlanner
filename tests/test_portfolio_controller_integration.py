from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

from cascade_planner.harness.agentic_blackboard_controller import (
    _portfolio_verifier_bundle,
    _refresh_multisource_route_consensus,
)
from cascade_planner.harness.tools import ToolExecutionState


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _verifier(marker: str, target: str) -> dict:
    return {
        "schema_version": "harness_route_verifier_report.v1",
        "marker": marker,
        "target_equivalence_audit": {
            "request_canonical_isomeric_smiles": target,
        },
    }


def test_controller_collects_parent_and_child_verifiers_into_deduplicated_bundle() -> None:
    parent = _verifier("parent", "CCO")
    child = _verifier("child", "CC")

    bundle = _portfolio_verifier_bundle(
        artifacts={
            "guided_chemenzy": {"raw_route_verifier": parent},
            "route_expansion_subgoal_search": {
                "subgoals": [{"verifier": child}, {"verifier": child}]
            },
        },
        parent_proof={
            "proof_evidence": {"parent_verifier_attempt": parent},
        },
        solved_parent_verifier={},
    )

    assert bundle["schema_version"] == "route_verifier_bundle.v1"
    assert bundle["input_report_count"] == 4
    assert bundle["report_count"] == 2
    assert bundle["duplicate_report_count"] == 2
    payload = dict(bundle)
    content_sha256 = payload.pop("content_sha256")
    assert content_sha256 == _digest(payload)


def test_consensus_refresh_keeps_hashed_portfolio_and_runtime_bindings_as_siblings(
    tmp_path,
) -> None:
    portfolio = {
        "schema_version": "route_portfolio.v1",
        "routes": [],
        "content_sha256": "a" * 64,
    }
    bindings = {
        "schema_version": "route_portfolio_bindings.v1",
        "stock_molecule_ids": [],
        "edge_proof_levels": {},
        "content_sha256": "b" * 64,
    }
    replacement_catalog = {
        "schema_version": "route_replacement_catalog.v1",
        "candidates": [],
        "content_sha256": "c" * 64,
    }
    rebuild = {
        "accepted": True,
        "consensus": {"schema_version": "route_consensus.v1", "proposals": []},
        "graph": {
            "schema_version": "route_consensus_graph.v1",
            "v2_overlay": {
                "schema_version": "route_hypergraph_overlay.v2",
                "root_molecule_id": "target",
                "validation": {"valid": True, "errors": []},
                "molecules": [],
                "reaction_hyperedges": [],
            },
        },
    }
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_smiles": "CCO"},
        preflight={"case_id": "case"},
    )

    with (
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "rebuild_consensus_graph_from_blackboard",
            return_value=rebuild,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "derive_portfolio_bindings",
            return_value=bindings,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller.solve_diverse_routes",
            return_value=SimpleNamespace(to_dict=lambda: dict(portfolio)),
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "validate_portfolio_replacements",
            return_value=replacement_catalog,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "consensus_to_blackboard_proposals",
            return_value=[],
        ),
    ):
        board = _refresh_multisource_route_consensus(
            state=state,
            blackboard={
                "case_id": "case",
                "target_profile": {"target_smiles": "CCO"},
            },
        )

    graph = board["route_consensus_graph"]
    assert graph["route_portfolio"] == portfolio
    assert "bindings" not in graph["route_portfolio"]
    assert graph["route_portfolio_bindings"] == bindings
    assert graph["route_replacement_catalog"] == replacement_catalog

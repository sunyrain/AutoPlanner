from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from cascade_planner.harness.route_forest_delivery import (
    DELIVERY_SCHEMA_VERSION,
    build_route_forest_delivery_payload,
    render_route_forest_html,
    route_forest_delivery_integrity_reasons,
)
from cascade_planner.harness.route_forest_layout import canonical_sha256


def _forest() -> dict:
    branch_id = "branch:main"
    step_id = "step:main"
    molecule_a = "mol:a"
    molecule_b = "mol:b"
    graph_a = f"graph:molecule:{molecule_a}"
    graph_b = f"graph:molecule:{molecule_b}"
    graph_r = f"graph:reaction:{step_id}"
    trust = {
        "proof_tier": "L2_reaction_validated",
        "identity": 1.0,
        "connectivity": 1.0,
    }
    return {
        "schema_version": "explored_route_forest.v1",
        "case_id": "compact-delivery",
        "target": {"name": "<Taxol>", "smiles": "CCO"},
        "counts": {"branches": 1, "steps": 1, "nodes": 2},
        "primary_branch_id": branch_id,
        "primary_selection": {
            "schema_version": "route_forest_primary_selection.v1",
            "primary_branch_id": branch_id,
            "status": "deterministically_verified",
            "proof_level": "parent_route_proof",
            "advisory_only": False,
        },
        "semantic_summary": {
            "schema_version": "route_forest_semantic_summary.v1",
            "agent_tasks": {"completed": 2, "total": 2},
        },
        "frontier_ledger": {
            "schema_version": "route_forest_frontier_ledger_view.v1",
            "source_schema_version": "frontier_ledger.v1",
            "authoritative": True,
            "content_sha256": "a" * 64,
            "validation_reasons": [],
            "counts": {
                "l0_break_suggestion_edges": 0,
                "expanded_work_molecules": 1,
                "l2_reaction_edges": 1,
                "l3_precedent_edges": 0,
                "stock_closed_leaves": 1,
                "reachable_leaves": 1,
            },
            "closure": {
                "any_benchmark_route_closed": True,
                "all_explored_benchmark_closed": True,
                "any_procurement_route_closed": False,
                "all_explored_procurement_closed": False,
                "any_route_closed": True,
                "all_explored_graph_closed": True,
                "l3_parent_solved": True,
                "l4_parent_route_proof_ready": False,
                "l4_procurement_ready": False,
            },
        },
        "display_policy": {
            "schema_version": "route_forest_display_policy.v1",
            "default_overview_top_k": 12,
            "default_group_visible_count": 5,
        },
        "nodes": [
            {
                "node_id": molecule_a,
                "label": "A",
                "structure_svg": "<svg><path d='a'/></svg>",
            },
            {
                "node_id": molecule_b,
                "label": "B",
                "structure_svg": "<svg><path d='b'/></svg>",
            },
        ],
        "steps": [
            {
                "step_id": step_id,
                "branch_id": branch_id,
                "label": "A to B",
                "from_node_ids": [molecule_a],
                "to_node_ids": [molecule_b],
                "trust_vector": trust,
                "source_refs": ["doi:10.1000/example"],
            }
        ],
        "branches": [
            {
                "branch_id": branch_id,
                "title": "Main route",
                "kind": "direct_verified_route",
                "step_ids": [step_id],
                "listed": True,
                "is_primary": True,
                "solved": True,
                "executable": True,
                "advisory_only": False,
                "not_parent_route_proof": False,
            }
        ],
        "dependency_graph": {
            "schema_version": "molecule_reaction_dependency_graph.v1",
            "nodes": [
                {
                    "graph_node_id": graph_a,
                    "node_type": "molecule",
                    "molecule_node_id": molecule_a,
                    "label": "A",
                    "structure_svg": "DUPLICATE-A",
                },
                {
                    "graph_node_id": graph_r,
                    "node_type": "reaction",
                    "reaction_step_id": step_id,
                    "branch_id": branch_id,
                    "label": "A to B",
                    "proof_tier": "L2_reaction_validated",
                    "trust_vector": trust,
                },
                {
                    "graph_node_id": graph_b,
                    "node_type": "molecule",
                    "molecule_node_id": molecule_b,
                    "label": "B",
                    "structure_svg": "DUPLICATE-B",
                },
            ],
            "molecule_nodes": [
                {"graph_node_id": graph_a, "structure_svg": "THIRD-COPY"}
            ],
            "reaction_nodes": [{"graph_node_id": graph_r}],
            "hyperedges": [{"hyperedge_id": "duplicate-not-needed"}],
            "edges": [
                {
                    "edge_id": "edge:1",
                    "source_graph_node_id": graph_a,
                    "target_graph_node_id": graph_r,
                    "reaction_step_id": step_id,
                    "branch_id": branch_id,
                    "trust_vector": trust,
                },
                {
                    "edge_id": "edge:2",
                    "source_graph_node_id": graph_r,
                    "target_graph_node_id": graph_b,
                    "reaction_step_id": step_id,
                    "branch_id": branch_id,
                    "trust_vector": trust,
                },
            ],
            "branch_views": [
                {
                    "branch_id": branch_id,
                    "step_ids": [step_id],
                    "topological_step_ids": [step_id],
                    "dependencies": [],
                    "acyclic": True,
                }
            ],
            "no_array_adjacency_edges": True,
            "proof_tier_legend": [],
        },
        "replacement_validation": {
            "schema_version": "route_replacement_validation.v1",
            "candidate_count": 1,
            "validated_count": 0,
            "rejected_count": 1,
            "records": [
                {
                    "replacement_id": "replacement:rejected",
                    "base_step_id": step_id,
                    "base_branch_id": branch_id,
                    "candidate_step_id": "",
                    "candidate_branch_id": "",
                    "revalidated_route_branch_id": "",
                    "accepted": False,
                    "validated": False,
                    "status": "rejected",
                    "preview_only": True,
                    "connectivity_revalidated": True,
                    "stock_closure_revalidated": True,
                    "reaction_proof_revalidated": True,
                }
            ],
            "interface_diagnostics": {
                "candidate_count": 2,
                "interface_compatible_count": 0,
                "records": [
                    {"diagnostic_id": "one", "payload": "x" * 10_000},
                    {"diagnostic_id": "two", "payload": "y" * 10_000},
                ],
                "authority": "diagnostics_only_not_replacement_validation",
            },
            "semantics": {"single_step_splicing_forbidden": True},
        },
        "artifact_revision": {
            "status": "committed",
            "committed": True,
            "revision_id": "revision-1",
        },
        "projection_coverage": {"complete": False},
    }


def _add_alternative_branch(forest: dict) -> None:
    branch_id = "branch:other"
    step_id = "step:other"
    graph_reaction_id = f"graph:reaction:{step_id}"
    molecule_a = "mol:a"
    molecule_b = "mol:b"
    graph_a = f"graph:molecule:{molecule_a}"
    graph_b = f"graph:molecule:{molecule_b}"
    trust = copy.deepcopy(forest["steps"][0]["trust_vector"])
    forest["steps"].append(
        {
            "step_id": step_id,
            "branch_id": branch_id,
            "label": "Alternative A to B",
            "from_node_ids": [molecule_a],
            "to_node_ids": [molecule_b],
            "trust_vector": trust,
            "source_refs": ["doi:10.1000/alternative"],
        }
    )
    forest["branches"].append(
        {
            "branch_id": branch_id,
            "title": "Alternative route",
            "kind": "alternative_route",
            "step_ids": [step_id],
            "listed": True,
        }
    )
    graph = forest["dependency_graph"]
    reaction_node = {
        "graph_node_id": graph_reaction_id,
        "node_type": "reaction",
        "reaction_step_id": step_id,
        "branch_id": branch_id,
        "label": "Alternative A to B",
    }
    graph["nodes"].append(reaction_node)
    if "reaction_nodes" in graph:
        graph["reaction_nodes"].append({"graph_node_id": graph_reaction_id})
    graph["edges"].extend(
        [
            {
                "edge_id": "edge:other:1",
                "edge_type": "molecule_to_reaction",
                "source_graph_node_id": graph_a,
                "target_graph_node_id": graph_reaction_id,
                "molecule_node_id": molecule_a,
                "reaction_step_id": step_id,
                "branch_id": branch_id,
            },
            {
                "edge_id": "edge:other:2",
                "edge_type": "reaction_to_molecule",
                "source_graph_node_id": graph_reaction_id,
                "target_graph_node_id": graph_b,
                "molecule_node_id": molecule_b,
                "reaction_step_id": step_id,
                "branch_id": branch_id,
            },
        ]
    )
    graph["branch_views"].append(
        {
            "branch_id": branch_id,
            "step_ids": [step_id],
            "topological_step_ids": [step_id],
            "dependencies": [],
            "acyclic": True,
        }
    )


def _forest_with_valid_replacement() -> dict:
    forest = _forest()
    _add_alternative_branch(forest)
    replacement_branch = forest["branches"][-1]
    replacement_branch.update(
        {
            "kind": "validated_replacement_route",
            "listed": False,
            "complete": True,
            "reaction_validated": True,
            "proof_eligible": True,
        }
    )
    validation = forest["replacement_validation"]
    validation["candidate_count"] = 2
    validation["validated_count"] = 1
    validation["records"].append(
        {
            "replacement_id": "replacement:validated",
            "base_step_id": "step:main",
            "base_branch_id": "branch:main",
            "candidate_step_id": "step:other",
            "candidate_branch_id": "branch:other",
            "revalidated_route_branch_id": "branch:other",
            "accepted": True,
            "validated": True,
            "status": "route_revalidated",
            "preview_only": True,
            "connectivity_revalidated": True,
            "stock_closure_revalidated": True,
            "reaction_proof_revalidated": True,
        }
    )
    return forest


def _mutate_replacement_case(forest: dict, case: str) -> None:
    records = forest["replacement_validation"]["records"]
    validated = next(record for record in records if record["validated"] is True)
    rejected = next(record for record in records if record["validated"] is False)
    replacement_branch = next(
        branch for branch in forest["branches"] if branch["branch_id"] == "branch:other"
    )
    if case == "base_step_missing":
        validated["base_step_id"] = ""
    elif case == "base_step_unknown":
        validated["base_step_id"] = "step:missing"
    elif case == "base_branch_missing":
        validated["base_branch_id"] = ""
    elif case == "base_branch_unknown":
        validated["base_branch_id"] = "branch:missing"
    elif case == "base_owner_mismatch":
        validated["base_branch_id"] = "branch:other"
    elif case == "candidate_step_missing":
        validated["candidate_step_id"] = ""
    elif case == "candidate_step_unknown":
        validated["candidate_step_id"] = "step:missing"
    elif case == "candidate_branch_missing":
        validated["candidate_branch_id"] = ""
    elif case == "candidate_branch_unknown":
        validated["candidate_branch_id"] = "branch:missing"
        validated["revalidated_route_branch_id"] = "branch:missing"
    elif case == "revalidated_branch_missing":
        validated["revalidated_route_branch_id"] = ""
    elif case == "candidate_branch_mismatch":
        validated["revalidated_route_branch_id"] = "branch:main"
    elif case == "candidate_owner_mismatch":
        validated["candidate_step_id"] = "step:main"
    elif case == "candidate_branch_wrong_kind":
        replacement_branch["kind"] = "alternative_route"
    elif case == "candidate_branch_listed":
        replacement_branch["listed"] = True
    elif case == "candidate_branch_incomplete":
        replacement_branch["complete"] = False
    elif case == "validated_status_rejected":
        validated["status"] = "rejected"
    elif case.startswith("flag_not_true:"):
        validated[case.split(":", 1)[1]] = False
    elif case == "rejected_preview_reference":
        rejected["candidate_step_id"] = "step:other"
        rejected["candidate_branch_id"] = "branch:other"
        rejected["revalidated_route_branch_id"] = "branch:other"
    elif case == "rejected_accepted":
        rejected["accepted"] = True
    else:  # pragma: no cover - parameter table is the authority
        raise AssertionError(f"unknown replacement test case: {case}")


_REPLACEMENT_FAILURE_CASES = (
    (
        "base_step_missing",
        "replacement_base_step_id_missing:replacement:validated",
    ),
    (
        "base_step_unknown",
        "replacement_base_step_id_unknown:replacement:validated:step:missing",
    ),
    (
        "base_branch_missing",
        "replacement_base_branch_id_missing:replacement:validated",
    ),
    (
        "base_branch_unknown",
        "replacement_base_branch_id_unknown:replacement:validated:branch:missing",
    ),
    (
        "base_owner_mismatch",
        "replacement_base_ownership_mismatch:"
        "replacement:validated:step:main:branch:other",
    ),
    (
        "candidate_step_missing",
        "replacement_candidate_step_id_missing:replacement:validated",
    ),
    (
        "candidate_step_unknown",
        "replacement_candidate_step_id_unknown:replacement:validated:step:missing",
    ),
    (
        "candidate_branch_missing",
        "replacement_candidate_branch_id_missing:replacement:validated",
    ),
    (
        "candidate_branch_unknown",
        "replacement_candidate_branch_id_unknown:replacement:validated:branch:missing",
    ),
    (
        "revalidated_branch_missing",
        "replacement_revalidated_branch_id_missing:replacement:validated",
    ),
    (
        "candidate_branch_mismatch",
        "replacement_candidate_branch_mismatch:"
        "replacement:validated:branch:other:branch:main",
    ),
    (
        "candidate_owner_mismatch",
        "replacement_candidate_ownership_mismatch:"
        "replacement:validated:step:main:branch:other",
    ),
    (
        "candidate_branch_wrong_kind",
        "replacement_candidate_branch_kind_invalid:replacement:validated:branch:other",
    ),
    (
        "candidate_branch_listed",
        "replacement_candidate_branch_not_hidden:replacement:validated:branch:other",
    ),
    (
        "candidate_branch_incomplete",
        "replacement_candidate_branch_incomplete:replacement:validated:branch:other",
    ),
    (
        "validated_status_rejected",
        "replacement_validated_status_invalid:replacement:validated",
    ),
    *(
        (
            f"flag_not_true:{flag}",
            f"replacement_revalidated_flag_not_true:replacement:validated:{flag}",
        )
        for flag in (
            "accepted",
            "connectivity_revalidated",
            "stock_closure_revalidated",
            "reaction_proof_revalidated",
        )
    ),
    (
        "rejected_preview_reference",
        "replacement_rejected_preview_reference:"
        "replacement:rejected:candidate_step_id:step:other",
    ),
    (
        "rejected_accepted",
        "replacement_rejected_record_accepted:replacement:rejected",
    ),
)


def _mutate_relational_case(forest: dict, case: str) -> None:
    graph = forest["dependency_graph"]
    if case == "empty_dependency_graph":
        graph.clear()
    elif case == "duplicate_top_node":
        forest["nodes"].append(copy.deepcopy(forest["nodes"][0]))
    elif case == "dangling_step_input":
        forest["steps"][0]["from_node_ids"] = ["mol:missing"]
    elif case == "dangling_step_output":
        forest["steps"][0]["to_node_ids"] = ["mol:missing"]
    elif case == "dangling_graph_molecule":
        graph["nodes"][0]["molecule_node_id"] = "mol:missing"
    elif case == "dangling_reaction_step":
        graph["nodes"][1]["reaction_step_id"] = "step:missing"
    elif case == "molecule_to_molecule":
        graph["edges"][0]["target_graph_node_id"] = graph["nodes"][2]["graph_node_id"]
    elif case == "missing_step_edge":
        graph["edges"].pop(0)
    else:
        _add_alternative_branch(forest)
        if case == "edge_wrong_reaction_step":
            graph["edges"][0]["reaction_step_id"] = "step:other"
            graph["edges"][0]["branch_id"] = "branch:other"
        elif case == "cross_branch_edge":
            graph["edges"][0]["branch_id"] = "branch:other"
        elif case == "cross_branch_reaction":
            graph["nodes"][1]["branch_id"] = "branch:other"
        elif case == "cross_branch_membership":
            forest["branches"][1]["step_ids"].append("step:main")
        elif case == "cross_branch_view":
            graph["branch_views"][1]["step_ids"].append("step:main")
            graph["branch_views"][1]["topological_step_ids"].append("step:main")
        else:  # pragma: no cover - parameter table is the authority
            raise AssertionError(f"unknown relational test case: {case}")


_RELATIONAL_FAILURE_CASES = (
    ("empty_dependency_graph", "dependency_graph_empty"),
    ("duplicate_top_node", "node_node_id_duplicate:mol:a"),
    (
        "dangling_step_input",
        "step_from_node_id_unknown:step:main:mol:missing",
    ),
    (
        "dangling_step_output",
        "step_to_node_id_unknown:step:main:mol:missing",
    ),
    (
        "dangling_graph_molecule",
        "dependency_graph_molecule_node_id_unknown:graph:molecule:mol:a:mol:missing",
    ),
    (
        "dangling_reaction_step",
        "dependency_graph_reaction_step_id_unknown:"
        "graph:reaction:step:main:step:missing",
    ),
    (
        "molecule_to_molecule",
        "dependency_graph_edge_not_bipartite:edge:1:molecule:molecule",
    ),
    (
        "missing_step_edge",
        "dependency_graph_edge_topology_missing:step:main:input:mol:a",
    ),
    (
        "edge_wrong_reaction_step",
        "dependency_graph_edge_reaction_topology_mismatch:edge:1:step:other:step:main",
    ),
    (
        "cross_branch_edge",
        "dependency_graph_edge_branch_mismatch:edge:1:branch:other:branch:main",
    ),
    (
        "cross_branch_reaction",
        "dependency_graph_reaction_branch_mismatch:"
        "graph:reaction:step:main:branch:other:branch:main",
    ),
    (
        "cross_branch_membership",
        "branch_step_owner_mismatch:branch:other:step:main:branch:main",
    ),
    (
        "cross_branch_view",
        "dependency_graph_branch_view_step_owner_mismatch:"
        "branch:other:step:main:branch:main",
    ),
)


def _safe_json(value: object) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _resign_payload(payload: dict) -> None:
    payload.pop("embedded_json_sha256", None)
    payload.pop("delivery_sha256", None)
    payload["delivery_sha256"] = canonical_sha256(payload)
    payload["embedded_json_sha256"] = hashlib.sha256(
        _safe_json(payload).encode("utf-8")
    ).hexdigest()


def test_delivery_payload_is_digest_bound_compact_and_non_mutating() -> None:
    forest = _forest()
    forest["canonical_route_consensus_graph"] = {
        "schema_version": "route_consensus_graph.v1",
        "case_id": "compact-delivery",
        "target_smiles": "CCO",
        "steps": [],
    }
    forest["canonical_route_consensus_graph_source"] = (
        "canonical_route_consensus_graph"
    )
    original = copy.deepcopy(forest)

    payload = build_route_forest_delivery_payload(forest)

    assert forest == original
    assert payload["schema_version"] == DELIVERY_SCHEMA_VERSION
    assert payload["source_schema_version"] == "explored_route_forest.v1"
    assert payload["source_forest_sha256"] == canonical_sha256(forest)
    assert len(payload["delivery_sha256"]) == 64
    assert len(payload["embedded_json_sha256"]) == 64
    assert "artifact_revision" not in payload
    assert payload["source_revision_context"] == forest["artifact_revision"]
    assert payload["semantic_summary"] == forest["semantic_summary"]
    assert payload["frontier_ledger"] == forest["frontier_ledger"]
    assert payload["canonical_route_consensus_graph"] == forest[
        "canonical_route_consensus_graph"
    ]
    assert payload["canonical_route_consensus_graph_source"] == (
        "canonical_route_consensus_graph"
    )
    assert payload["display_policy"] == forest["display_policy"]
    assert payload["delivery_semantics"]["branch_count"] == (
        "exploration_views_only_never_completion_authority"
    )
    assert payload["delivery_semantics"]["authority"] == (
        "none_byte_integrity_projection_only"
    )
    assert payload["delivery_semantics"][
        "digest_does_not_grant_closeout_authority"
    ] is True
    assert all(
        "structure_svg" not in row for row in payload["dependency_graph"]["nodes"]
    )
    assert all(
        "trust_vector" not in row and "visual_encoding" not in row
        for row in payload["dependency_graph"]["nodes"]
        if row.get("node_type") == "reaction"
    )
    assert all(
        "trust_vector" not in row and "visual_encoding" not in row
        for row in payload["dependency_graph"]["edges"]
    )
    assert payload["steps"][0]["trust_vector"] == forest["steps"][0]["trust_vector"]
    assert "molecule_nodes" not in payload["dependency_graph"]
    assert "reaction_nodes" not in payload["dependency_graph"]
    assert "hyperedges" not in payload["dependency_graph"]
    assert payload["nodes"][0]["structure_svg"].startswith(
        '<svg xmlns="http://www.w3.org/2000/svg"'
    )
    assert (
        payload["replacement_validation"]["records"]
        == forest["replacement_validation"]["records"]
    )
    assert (
        payload["replacement_validation"]["semantics"]
        == forest["replacement_validation"]["semantics"]
    )
    diagnostics = payload["replacement_validation"]["interface_diagnostics"]
    assert "records" not in diagnostics
    assert diagnostics["source_record_count"] == 2
    assert diagnostics["records_omitted_from_delivery"] is True
    assert len(json.dumps(payload)) < len(json.dumps(forest))
    assert route_forest_delivery_integrity_reasons(payload, source_forest=forest) == []

    tampered = copy.deepcopy(payload)
    tampered["target"]["name"] = "tampered"
    assert route_forest_delivery_integrity_reasons(tampered, source_forest=forest) == [
        "route_forest_delivery_sha256_mismatch"
    ]

    embedded_digest_tampered = copy.deepcopy(payload)
    embedded_digest_tampered["embedded_json_sha256"] = "0" * 64
    assert route_forest_delivery_integrity_reasons(embedded_digest_tampered) == [
        "route_forest_embedded_json_sha256_mismatch"
    ]


@pytest.mark.parametrize("not_parent_route_proof", [None, True])
def test_verified_primary_requires_explicit_parent_proof_binding(
    not_parent_route_proof: bool | None,
) -> None:
    forest = _forest()
    if not_parent_route_proof is None:
        forest["branches"][0].pop("not_parent_route_proof")
    else:
        forest["branches"][0]["not_parent_route_proof"] = not_parent_route_proof

    with pytest.raises(
        ValueError,
        match="route_forest_source_verified_primary_contract_invalid",
    ):
        build_route_forest_delivery_payload(forest)

    payload = build_route_forest_delivery_payload(_forest())
    if not_parent_route_proof is None:
        payload["branches"][0].pop("not_parent_route_proof")
    else:
        payload["branches"][0]["not_parent_route_proof"] = not_parent_route_proof
    _resign_payload(payload)

    assert "route_forest_delivery_verified_primary_contract_invalid" in (
        route_forest_delivery_integrity_reasons(payload)
    )


def test_renderer_escapes_script_data_and_binds_exact_source_digest() -> None:
    forest = _forest()
    forest["target"]["name"] = (
        "<Taxol> __SCRIPT__ __STYLES__ __DATA__ __TITLE__ \u2028 \u2029"
    )
    template = "<title>__TITLE__</title><style>__STYLES__</style><script id='forest-data' type='application/json'>__DATA__</script><script>__SCRIPT__</script>"

    rendered = render_route_forest_html(
        forest,
        template=template,
        styles='.ok::after{content:"__SCRIPT__"}',
        script='window.marker="__DATA__";',
    )

    assert "<title>&lt;Taxol&gt; __SCRIPT__ __STYLES__ __DATA__ __TITLE__" in rendered
    assert "\\u003cTaxol>" in rendered
    assert "\\u2028" in rendered
    assert "\\u2029" in rendered
    assert '.ok::after{content:"__SCRIPT__"}' in rendered
    assert 'window.marker="__DATA__";' in rendered
    assert "DUPLICATE-A" not in rendered
    embedded = re.search(
        r"<script id='forest-data' type='application/json'>(.*?)</script>", rendered
    ).group(1)
    payload = json.loads(embedded)
    assert payload["source_forest_sha256"] == canonical_sha256(forest)
    assert payload["target"]["name"] == forest["target"]["name"]
    unsigned_embedded = re.sub(
        r'"embedded_json_sha256":"[0-9a-f]{64}",',
        "",
        embedded,
        count=1,
    )
    assert unsigned_embedded != embedded
    assert (
        hashlib.sha256(unsigned_embedded.encode("utf-8")).hexdigest()
        == payload["embedded_json_sha256"]
    )


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        (
            "duplicate_node_id",
            "route_forest_source_dependency_graph_node_graph_node_id_duplicate",
        ),
        (
            "empty_node_id",
            "route_forest_source_dependency_graph_node_graph_node_id_missing",
        ),
        (
            "duplicate_edge_id",
            "route_forest_source_dependency_graph_edge_edge_id_duplicate",
        ),
        (
            "empty_edge_id",
            "route_forest_source_dependency_graph_edge_edge_id_missing",
        ),
        (
            "dangling_source",
            "route_forest_source_dependency_graph_edge_source_unknown",
        ),
        (
            "dangling_target",
            "route_forest_source_dependency_graph_edge_target_unknown",
        ),
        (
            "missing_step_binding",
            "route_forest_source_dependency_graph_edge_reaction_step_id_missing",
        ),
        (
            "unknown_step_binding",
            "route_forest_source_dependency_graph_edge_reaction_step_id_unknown",
        ),
        (
            "missing_branch_binding",
            "route_forest_source_dependency_graph_edge_branch_id_missing",
        ),
        (
            "unknown_branch_binding",
            "route_forest_source_dependency_graph_edge_branch_id_unknown",
        ),
    ],
)
def test_delivery_build_rejects_invalid_explicit_dependency_graph(
    case: str,
    reason: str,
) -> None:
    forest = _forest()
    graph = forest["dependency_graph"]
    if case == "duplicate_node_id":
        graph["nodes"].append(copy.deepcopy(graph["nodes"][0]))
    elif case == "empty_node_id":
        graph["nodes"][0]["graph_node_id"] = ""
    elif case == "duplicate_edge_id":
        graph["edges"].append(copy.deepcopy(graph["edges"][0]))
    elif case == "empty_edge_id":
        graph["edges"][0]["edge_id"] = ""
    elif case == "dangling_source":
        graph["edges"][0]["source_graph_node_id"] = "graph:missing"
    elif case == "dangling_target":
        graph["edges"][0]["target_graph_node_id"] = "graph:missing"
    elif case == "missing_step_binding":
        graph["edges"][0]["reaction_step_id"] = ""
    elif case == "unknown_step_binding":
        graph["edges"][0]["reaction_step_id"] = "step:missing"
    elif case == "missing_branch_binding":
        graph["edges"][0]["branch_id"] = ""
    elif case == "unknown_branch_binding":
        graph["edges"][0]["branch_id"] = "branch:missing"

    with pytest.raises(ValueError, match=reason):
        build_route_forest_delivery_payload(forest)


@pytest.mark.parametrize(
    ("case", "reason_suffix"),
    _RELATIONAL_FAILURE_CASES,
)
def test_source_build_fails_closed_for_relational_and_topology_corruption(
    case: str,
    reason_suffix: str,
) -> None:
    forest = _forest()
    _mutate_relational_case(forest, case)

    with pytest.raises(ValueError) as exc_info:
        build_route_forest_delivery_payload(forest)

    assert f"route_forest_source_{reason_suffix}" in str(exc_info.value)


@pytest.mark.parametrize(
    ("case", "reason_suffix"),
    _RELATIONAL_FAILURE_CASES,
)
def test_resigned_delivery_fails_closed_for_relational_and_topology_corruption(
    case: str,
    reason_suffix: str,
) -> None:
    payload = build_route_forest_delivery_payload(_forest())
    _mutate_relational_case(payload, case)
    _resign_payload(payload)

    reasons = route_forest_delivery_integrity_reasons(payload)

    assert f"route_forest_delivery_{reason_suffix}" in reasons


def test_validated_replacement_record_binds_complete_hidden_route() -> None:
    forest = _forest_with_valid_replacement()

    payload = build_route_forest_delivery_payload(forest)

    assert (
        route_forest_delivery_integrity_reasons(
            payload,
            source_forest=forest,
        )
        == []
    )
    assert (
        payload["replacement_validation"]["records"]
        == forest["replacement_validation"]["records"]
    )


@pytest.mark.parametrize(
    ("case", "reason_suffix"),
    _REPLACEMENT_FAILURE_CASES,
)
def test_source_build_fails_closed_for_replacement_record_corruption(
    case: str,
    reason_suffix: str,
) -> None:
    forest = _forest_with_valid_replacement()
    _mutate_replacement_case(forest, case)

    with pytest.raises(ValueError) as exc_info:
        build_route_forest_delivery_payload(forest)

    assert f"route_forest_source_{reason_suffix}" in str(exc_info.value)


@pytest.mark.parametrize(
    ("case", "reason_suffix"),
    _REPLACEMENT_FAILURE_CASES,
)
def test_resigned_delivery_fails_closed_for_replacement_record_corruption(
    case: str,
    reason_suffix: str,
) -> None:
    payload = build_route_forest_delivery_payload(_forest_with_valid_replacement())
    _mutate_replacement_case(payload, case)
    _resign_payload(payload)

    reasons = route_forest_delivery_integrity_reasons(payload)

    assert f"route_forest_delivery_{reason_suffix}" in reasons


def test_integrity_rejects_resigned_invalid_payload_and_source_graphs() -> None:
    forest = _forest()
    payload = build_route_forest_delivery_payload(forest)
    payload["dependency_graph"]["edges"][0]["source_graph_node_id"] = "graph:missing"
    _resign_payload(payload)

    reasons = route_forest_delivery_integrity_reasons(payload)

    assert reasons == [
        "route_forest_delivery_dependency_graph_edge_source_unknown:"
        "edge:1:graph:missing"
    ]

    forest = _forest()
    payload = build_route_forest_delivery_payload(forest)
    forest["dependency_graph"]["edges"][0]["target_graph_node_id"] = "graph:missing"
    payload["source_forest_sha256"] = canonical_sha256(forest)
    _resign_payload(payload)

    reasons = route_forest_delivery_integrity_reasons(
        payload,
        source_forest=forest,
    )

    assert reasons == [
        "route_forest_source_dependency_graph_edge_target_unknown:edge:1:graph:missing"
    ]


def test_integrity_rebuilds_source_projection_to_reject_resigned_ui_tampering() -> None:
    forest = _forest()
    payload = build_route_forest_delivery_payload(forest)
    payload["dependency_layout"]["nodes"][0]["layer"] = 999
    _resign_payload(payload)

    assert route_forest_delivery_integrity_reasons(
        payload,
        source_forest=forest,
    ) == ["route_forest_delivery_projection_mismatch"]


def test_structure_svg_is_allowlist_sanitized_without_breaking_rdkit_paths() -> None:
    forest = _forest()
    malicious_svg = """<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)">
      <script>alert(2)</script>
      <foreignObject><img xmlns="http://www.w3.org/1999/xhtml" src="x" onerror="alert(3)" /></foreignObject>
      <image href="data:text/html,bad" onerror="alert(4)" />
      <path d="M 0 0 L 10 10" onclick="alert(5)" style="fill:#fff;stroke:url(javascript:alert(6));stroke-width:2px" />
    </svg>"""
    forest["nodes"][0]["structure_svg"] = malicious_svg

    payload = build_route_forest_delivery_payload(forest)
    sanitized = payload["nodes"][0]["structure_svg"]
    lowered = sanitized.lower()

    assert sanitized.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert '<path d="M 0 0 L 10 10"' in sanitized
    assert "fill:#fff" in sanitized
    assert "stroke-width:2px" in sanitized
    for executable_fragment in (
        "<script",
        "foreignobject",
        "<img",
        "<image",
        "onerror",
        "onclick",
        "onload",
        "javascript:",
        "data:text",
        "alert(",
    ):
        assert executable_fragment not in lowered
    assert "<path" in payload["nodes"][1]["structure_svg"]
    assert (
        route_forest_delivery_integrity_reasons(
            payload,
            source_forest=forest,
        )
        == []
    )

    payload["nodes"][0]["structure_svg"] = malicious_svg
    _resign_payload(payload)
    assert route_forest_delivery_integrity_reasons(payload) == [
        "route_forest_delivery_structure_svg_unsafe:0"
    ]


@pytest.mark.parametrize(
    ("placeholder", "duplicate"),
    [
        ("__TITLE__", False),
        ("__STYLES__", False),
        ("__DATA__", False),
        ("__SCRIPT__", False),
        ("__TITLE__", True),
        ("__STYLES__", True),
        ("__DATA__", True),
        ("__SCRIPT__", True),
    ],
)
def test_renderer_requires_each_template_placeholder_exactly_once(
    placeholder: str,
    duplicate: bool,
) -> None:
    template = "__TITLE__ __STYLES__ __DATA__ __SCRIPT__"
    template = (
        template + placeholder if duplicate else template.replace(placeholder, "")
    )

    with pytest.raises(ValueError, match="invalid_route_forest_template_placeholders"):
        render_route_forest_html(
            _forest(),
            template=template,
            styles="",
            script="",
        )


def test_build_and_integrity_fail_closed_for_non_json_source_or_payload() -> None:
    forest = _forest()
    forest["target"]["score"] = float("nan")
    with pytest.raises(ValueError, match="route_forest_source_not_json_serializable"):
        build_route_forest_delivery_payload(forest)

    payload = build_route_forest_delivery_payload(_forest())
    payload["target"]["score"] = float("nan")
    reasons = route_forest_delivery_integrity_reasons(payload)
    assert "route_forest_delivery_not_json_serializable" in reasons
    assert "route_forest_delivery_sha256_mismatch" in reasons


def test_build_and_integrity_reject_wrong_projection_field_shapes() -> None:
    forest = _forest()
    forest["target"] = ["silently dropping this would change the projection"]
    with pytest.raises(ValueError, match="route_forest_source_target_not_object"):
        build_route_forest_delivery_payload(forest)

    payload = build_route_forest_delivery_payload(_forest())
    payload["dependency_layout"] = []
    _resign_payload(payload)
    assert route_forest_delivery_integrity_reasons(payload) == [
        "route_forest_delivery_dependency_layout_not_object"
    ]


def test_repository_delivery_assets_expose_required_dom_and_read_only_semantics() -> (
    None
):
    root = Path("cascade_planner/harness/route_forest_ui")
    template = (root / "template.html").read_text(encoding="utf-8")
    script = (root / "script.js").read_text(encoding="utf-8")
    required_ids = {
        "pageTitle",
        "verdictBadge",
        "overviewMetrics",
        "integrityStatus",
        "layoutPreset",
        "themeToggle",
        "navToggle",
        "inspectorToggle",
        "branchSearch",
        "stageFilterBar",
        "stageFilterStatus",
        "partialExpandedSummary",
        "branchFilterBar",
        "branchGroups",
        "evidenceStats",
        "graphTitle",
        "graphSubtitle",
        "graphVisibleCount",
        "closureStatusTitle",
        "ledgerAuthorityBadge",
        "ledgerProgressMetrics",
        "closureStatusGrid",
        "overviewToggle",
        "graphViewport",
        "mainRoute",
        "graphMinimap",
        "zoomReadout",
        "orientationSelect",
        "routeDirectionSelect",
        "auxiliarySelect",
        "densitySelect",
        "edgeStyleSelect",
        "labelModeSelect",
        "legendPopover",
        "inspectorTitle",
        "detailTabs",
        "detail",
    }
    for element_id in required_ids:
        assert f'id="{element_id}"' in template
    for action in ("fit", "zoom-in", "zoom-out", "reset"):
        assert f'data-graph-action="{action}"' in template
    for stage in ("all", "suggestion", "expanded", "reaction", "stock"):
        assert f'data-stage-filter="{stage}"' in template
    assert template.count('aria-controls="branchGroups"') >= 5
    assert "laneMatchesStage" in script
    assert "route_forest_branch_lanes.v2" in script
    assert "route_forest_branch_stage_evidence.v3" in script
    assert "stageMembershipIsAuthoritative" in script
    assert "partialExpansionProgress" in script
    assert "fully_expanded" in script
    assert "partial_expanded" in script
    assert "all_leaves_stock_bound" not in script
    assert "旧版数据、聚合 proof tier、步骤数量与库存别名" in script
    assert "ensureSelectedBranchMatchesFilters();" in script
    assert "clearReplacementPreviewForFilterChange" in script
    assert "state.selectedBranchId = next?.branch_id || '';" in script
    assert "当前筛选没有权威绑定路线" in script
    assert 'role="status" aria-live="polite"' in script
    for phrase in (
        "Consensus evidence audit",
        "独立信源组",
        "文献文档",
        "来源表示",
        "候选记录",
        "同分候选",
        "展示锚点",
        "Independent support groups",
        "Condition conflicts",
        "Codex roles are correlated",
        "No backend AND/OR-revalidated replacement is available",
        "never enable a single-step splice",
        "Array adjacency never creates an edge",
        "full AND/OR route re-solved",
        "no_stock_closed_reaction_validated_route",
        "Projection truncated",
        "Delivery bytes verified",
        "current closeout requires external manifest",
        "完整 portfolio",
        "全路径已展开",
        "仅部分展开，不计入本阶段",
        "不计入“全路径已展开”",
        "部分展开",
        "数量不代表完整路线数",
        "default_overview_top_k",
        "L0 断键边",
        "已展开 work",
        "L2 反应边",
        "L3 先例边",
        "ANY BENCHMARK ROUTE",
        "ALL BENCHMARK GRAPH",
        "ANY PROCUREMENT ROUTE",
        "ALL PROCUREMENT GRAPH",
        "L3 SELECTED ROUTES",
        "L4 PROCUREMENT",
        "结论 fail-closed",
        "搜索库存叶",
        "benchmark 不等于可采购",
        "交付字节完整性未验证",
        "benchmark 命中绝不冒充商业采购",
    ):
        assert phrase in script
    assert "deliveryIntegrityStatus === 'verified'" in script
    assert "deliveryBytesVerified && frontierLedger.authoritative === true" in script
    assert "交付仅字节完整" in script
    assert "external_closeout_authority: false" in script
    assert "get('embed') === '1'" in script
    assert "data-replacement-id" in script
    assert "node.structure_svg" in script
    assert "deterministic_adaptive_shelf_grid.v1" in script
    assert "Math.max(1, viewport.clientWidth" in script
    assert ".015" in script
    assert ".branch-card[data-branch-id]" in script
    assert "document.querySelectorAll('[data-branch-id]')" not in script
    assert "forest.primary_selection?.display_tiebreak_only && primaryId" in script
    assert "row.support_group || row.source_channel" in script
    assert "row.source_refs" in script
    assert "row.evidence_refs" in script
    assert "Array.isArray(row.values)" in script
    assert "row.source_group || row.source_ref" not in script


def test_delivery_stage_views_ignore_aggregate_hints_and_preserve_exact_authority() -> (
    None
):
    aggregate_only = _forest()
    aggregate_lane = build_route_forest_delivery_payload(aggregate_only)[
        "branch_lanes"
    ]["lanes"][0]

    assert aggregate_lane["proof_tier"] == "L2_reaction_validated"
    assert aggregate_lane["stage_memberships"] == []
    assert "stage_authority_missing" in aggregate_lane["stage_evidence"][
        "reaction"
    ]["reasons"]

    authoritative = _forest()
    authoritative["frontier_ledger"]["stage_authority"] = {
        "schema_version": "route_forest_stage_authority.v1",
        "authoritative": True,
        "molecules": [],
        "edges": [
            {
                "exact_edge_signature": "edge:main",
                "step_ids": ["step:main"],
                "product_smiles": "CCO",
                "precursor_smiles": ["CC"],
                "reaction_proof": {
                    "achieved_proof_level": 2,
                    "authority": "current_host_verifier_replay",
                    "current_host_reaction_validated": True,
                    "proof_request_ids": ["proof:main"],
                },
            }
        ],
        "reasons": [],
    }
    payload = build_route_forest_delivery_payload(authoritative)
    lane = payload["branch_lanes"]["lanes"][0]

    assert lane["stage_memberships"] == ["reaction"]
    assert lane["stage_evidence"]["reaction"]["matched_step_ids"] == [
        "step:main"
    ]
    assert route_forest_delivery_integrity_reasons(
        payload, source_forest=authoritative
    ) == []

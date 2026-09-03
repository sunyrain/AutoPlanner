from __future__ import annotations

import hashlib
import json

import pytest

from cascade_planner.legacy.application_runtime.frontier_ledger import project_frontier_ledger
from scripts.legacy.audit_architecture_v2 import _frontier_ledger_evidence, audit_run
from cascade_planner.legacy.runtime.artifact_revision import publish_closeout_revision


pytestmark = pytest.mark.legacy


def write(path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def with_content_hash(value: dict) -> dict:
    payload = dict(value)
    payload.pop("content_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def with_named_hash(value: dict, field: str) -> dict:
    payload = dict(value)
    payload.pop(field, None)
    payload[field] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def frontier_ledger_fixture(
    *,
    any_route_closed: bool,
    all_explored_graph_closed: bool,
    inputs_valid: bool = True,
    queue_override: dict | None = None,
) -> dict:
    graph = {
        "schema_version": "route_consensus_graph.v1",
        "case_id": "test",
        "target_smiles": "CCO",
        "nodes": [],
        "steps": [],
    }
    queue = dict(queue_override or {}) or with_content_hash(
        {
            "schema_version": "frontier_queue.v1",
            "run_id": "test",
            "revision": 1,
            "jobs": [],
        }
    )
    proof_state = with_content_hash(
        {
            "schema_version": "codex_retrosynthesis_reaction_proof_state.v1",
            "graph_identity_sha256": with_content_hash(
                {
                    "schema_version": graph["schema_version"],
                    "case_id": graph["case_id"],
                    "target_smiles": graph["target_smiles"],
                    "steps": [],
                }
            )["content_sha256"],
            "records": [],
        }
    )
    ledger = project_frontier_ledger(
        graph,
        queue,
        proof_state,
        campaign_policy_sha256=sha("fixture-policy"),
    )
    if any_route_closed or all_explored_graph_closed:
        # Deliberately forge a digest-valid positive claim without a stock
        # replay binding.  The current validator must reject it.
        for container in (
            ledger["root"]["closure"],
            ledger["molecules"]["CCO"]["closure"],
            ledger["summary"],
        ):
            container["any_benchmark_route_closed"] = any_route_closed
            container["all_explored_benchmark_closed"] = (
                all_explored_graph_closed
            )
            container["any_route_closed"] = any_route_closed
            container["all_explored_graph_closed"] = all_explored_graph_closed
        ledger["molecules"]["CCO"]["stock"]["closed"] = any_route_closed
        ledger["summary"]["stock_closed_molecule_count"] = int(any_route_closed)
    if not inputs_valid:
        for key in ("graph", "frontier_queue", "reaction_proof_state"):
            ledger["input_validation"][key]["valid"] = False
    return with_content_hash(ledger)


def _route_nodes(route: dict, edges_by_id: dict[str, dict]) -> tuple[set[str], set[str]]:
    products: set[str] = set()
    nodes: set[str] = set()
    for selection in route.get("selected_hyperedges") or []:
        product = str(selection.get("product_molecule_id") or "")
        edge = edges_by_id[str(selection.get("hyperedge_id") or "")]
        products.add(product)
        nodes.add(product)
        nodes.update(str(value) for value in edge.get("precursor_molecule_ids") or [])
    return nodes, nodes - products


def _route_acyclic(route: dict, edges_by_id: dict[str, dict]) -> bool:
    adjacency: dict[str, set[str]] = {}
    nodes, _ = _route_nodes(route, edges_by_id)
    for selection in route.get("selected_hyperedges") or []:
        edge = edges_by_id[str(selection.get("hyperedge_id") or "")]
        product = str(selection.get("product_molecule_id") or "")
        adjacency.setdefault(product, set())
        for precursor in edge.get("precursor_molecule_ids") or []:
            adjacency.setdefault(str(precursor), set()).add(product)
    indegree = {node: 0 for node in nodes}
    for targets in adjacency.values():
        for target in targets:
            indegree[target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in adjacency.get(node, set()):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited == len(nodes)


def test_architecture_audit_separates_engineering_from_run_acceptance(tmp_path) -> None:
    write(tmp_path / "target_input.json", {"target_name": "test", "target_smiles": "CCO"})
    write(tmp_path / "preflight.json", {"accepted": True, "inchi_key": "TEST"})
    write(
        tmp_path / "codex_retrosynthesis_team" / "team_report.json",
        {
            "accepted": True,
            "coordinator": {
                "required_child_roles": ["critic"],
                "observed_child_agents": [{"report_accepted": True}],
            },
            "runtime_summary": {"consistent": True},
            "campaign": {
                "proposal_graph_exhausted": True,
                "frontier_queue": {"jobs": []},
                "frontier_completeness": {"complete": False},
            },
        },
    )
    write(
        tmp_path / "route_consensus_graph_fused.json",
        {
            "v2_overlay": {
                "validation": {"valid": True, "errors": []},
                "molecules": [],
                "reaction_hyperedges": [],
            },
            "route_portfolio": {"routes": [], "reasons": ["no_route"]},
        },
    )
    write(
        tmp_path / "explored_route_forest.json",
        {
            "dependency_graph": {"schema_version": "route_dependency_graph.v1"},
            "projection_coverage": {"complete": True},
        },
    )
    write(tmp_path / "agent_blackboard.json", {})
    write(
        tmp_path / "final_verdict.json",
        {"verdict": "unresolved", "route_status": "unresolved", "solved": False},
    )

    report = audit_run(tmp_path)

    assert report["capability_surface"]["coverage_percent"] == 100.0
    assert "engineering" not in report
    assert "engineering-completion percentage" in report["capability_surface"]["semantics"]
    assert report["executable_contract_evidence"]["coverage_percent"] < 100
    assert report["run_acceptance"]["completion_percent"] < 100
    assert "deterministic_parent_route_solved" in report["remaining_gaps"]
    assert "proof_eligible_portfolio_valid" in report["remaining_gaps"]
    assert report["run_acceptance"]["gates"]["portfolio_routes_projected"] is False
    assert report["portfolio"]["forest_projection_evidence"]["reasons"] == [
        "no_proof_eligible_portfolio_routes"
    ]
    assert report["run_acceptance"]["gates"]["distinct_alternatives_at_least_2"] is False
    assert (
        report["run_acceptance"]["gates"]["backend_revalidated_replacement_available"]
        is False
    )
    assert report["completion_truth"] == {
        "schema_version": "architecture_completion_truth.v1",
        "ledger_required_reaction_proof_level": 0,
        "any_route_closed": False,
        "all_explored_graph_closed": False,
        "l3_parent_route_solved": False,
        "l4_procurement_route_ready": False,
        "l4_procurement_route_ids": [],
        "semantics": {
            "any_route_is_existential_hypergraph_closure": True,
            "all_explored_is_universal_hypergraph_closure": True,
            "l3_parent_requires_deterministic_parent_proof": True,
            "l4_procurement_requires_a_contract_valid_all_l4_route": True,
            "no_completion_field_implies_another_stronger_field": True,
        },
    }


def test_frontier_ledger_audit_checks_digest_and_fails_closed() -> None:
    ledger = frontier_ledger_fixture(
        any_route_closed=True,
        all_explored_graph_closed=True,
    )
    forged_positive = _frontier_ledger_evidence(
        ledger,
        expected_target_smiles="CCO",
    )

    assert forged_positive["schema_valid"] is True
    assert forged_positive["content_sha256_valid"] is True
    assert forged_positive["producer_fail_closed"] is True
    assert forged_positive["contract_valid"] is False
    assert forged_positive["authority_valid"] is False
    assert forged_positive["effective_any_route_closed"] is False
    assert forged_positive["effective_all_explored_graph_closed"] is False
    assert "current_host_stock_provider_replay_not_supplied" in forged_positive[
        "authority_blockers"
    ]

    structurally_valid = _frontier_ledger_evidence(
        frontier_ledger_fixture(
            any_route_closed=False,
            all_explored_graph_closed=False,
        ),
        expected_target_smiles="CCO",
    )
    assert structurally_valid["contract_valid"] is True
    assert structurally_valid["authority_valid"] is False
    assert structurally_valid["current_input_bindings_verified"] is False

    digest_tampered = json.loads(json.dumps(ledger))
    digest_tampered["summary"]["all_explored_graph_closed"] = False
    tampered = _frontier_ledger_evidence(
        digest_tampered,
        expected_target_smiles="CCO",
    )
    assert tampered["content_sha256_valid"] is False
    assert tampered["authority_valid"] is False
    assert tampered["effective_any_route_closed"] is False

    invalid_authority = frontier_ledger_fixture(
        any_route_closed=True,
        all_explored_graph_closed=True,
        inputs_valid=False,
    )
    failed_open = _frontier_ledger_evidence(
        invalid_authority,
        expected_target_smiles="CCO",
    )
    assert failed_open["producer_fail_closed"] is False
    assert failed_open["authority_valid"] is False
    assert failed_open["effective_any_route_closed"] is False

    malformed = with_content_hash({**ledger, "summary": ["not", "an", "object"]})
    malformed_evidence = _frontier_ledger_evidence(
        malformed,
        expected_target_smiles="CCO",
    )
    assert malformed_evidence["contract_valid"] is False
    assert malformed_evidence["effective_any_route_closed"] is False

    inconsistent = json.loads(json.dumps(ledger))
    inconsistent["summary"]["reachable_molecule_count"] = 2
    inconsistent = with_content_hash(inconsistent)
    inconsistent_evidence = _frontier_ledger_evidence(
        inconsistent,
        expected_target_smiles="CCO",
    )
    assert "frontier_ledger_reachable_molecule_count_mismatch" in inconsistent_evidence[
        "reasons"
    ]
    assert inconsistent_evidence["authority_valid"] is False


def _write_portfolio_run(
    tmp_path,
    *,
    routes: list[dict],
    edges: list[dict],
    proof_level: int = 3,
) -> None:
    named_proof_level = (
        "L4_procurement_ready" if proof_level == 4 else "L3_precedent_supported"
    )
    write(tmp_path / "target_input.json", {"target_name": "test", "target_smiles": "CCO"})
    write(tmp_path / "preflight.json", {"accepted": True, "inchi_key": "TEST"})
    write(
        tmp_path / "codex_retrosynthesis_team" / "team_report.json",
        {
            "schema_version": "codex_retrosynthesis_team_run.v1",
            "accepted": True,
            "coordinator": {
                "required_child_roles": ["critic"],
                "observed_child_agents": [{"report_accepted": True}],
            },
            "runtime_summary": {"consistent": True},
            "reasons": [],
            "campaign": {
                "schema_version": "codex_retrosynthesis_campaign.v1",
                "proposal_graph_exhausted": True,
                "frontier_queue": with_content_hash(
                    {
                        "schema_version": "frontier_queue.v1",
                        "run_id": "test",
                        "revision": 1,
                        "jobs": [],
                    }
                ),
                "frontier_completeness": {"complete": False},
            },
        },
    )
    edges_by_id = {str(edge["hyperedge_id"]): dict(edge) for edge in edges}
    molecule_ids = sorted(
        {
            str(edge.get("product_molecule_id") or "")
            for edge in edges
        }
        | {
            str(value)
            for edge in edges
            for value in edge.get("precursor_molecule_ids") or []
        }
        - {""}
    )
    smiles_by_id = {
        molecule_id: {
            "target": "CCO",
            "middle": "CC",
            "stock": "C",
            "side": "O",
            "stock-alt": "N",
        }.get(molecule_id, f"C{len(molecule_id)}")
        for molecule_id in molecule_ids
    }
    normalized_routes: list[dict] = []
    for raw_route in routes:
        route = dict(raw_route)
        nodes, leaves = _route_nodes(route, edges_by_id) if route.get("selected_hyperedges") else (set(), set())
        route.update(
            {
                "root_molecule_id": str(route.get("root_molecule_id") or "target"),
                "molecule_ids": sorted(nodes),
                "stock_terminal_ids": sorted(leaves),
                "source_channels": list(route.get("source_channels") or ["literature_exact"]),
                "independent_support_groups": list(
                    route.get("independent_support_groups") or ["literature:fixture"]
                ),
                "weakest_proof_level": int(
                    route.get("weakest_proof_level") or proof_level
                ),
                "procurement_ready": bool(
                    route.get("procurement_ready") is True or proof_level == 4
                ),
                "mean_edge_rank": float(route.get("mean_edge_rank") or 0.8),
                "base_score": float(route.get("base_score") or 0.8),
                "diversity_score": float(route.get("diversity_score") or 1.0),
                "portfolio_score": float(route.get("portfolio_score") or 0.8),
                "unresolved_frontiers": list(route.get("unresolved_frontiers") or []),
                "hyperedge_ids": [
                    str(row.get("hyperedge_id") or "")
                    for row in route.get("selected_hyperedges") or []
                ],
            }
        )
        normalized_routes.append(with_content_hash(route))
    routes = normalized_routes
    portfolio = with_content_hash(
        {
            "schema_version": "route_portfolio.v1",
            "root_molecule_id": "target",
            "routes": routes,
            "complete_candidate_count": len(routes),
            "enumerated_candidate_count": len(routes),
            "truncated": False,
            "reasons": [],
            "selection_policy": "and_or_closure_then_maximal_marginal_relevance",
            "requires_explicit_stock_and_reaction_proof": True,
        }
    )
    exact_bindings: dict[str, dict] = {}
    for edge in edges:
        edge_id = str(edge["hyperedge_id"])
        product_id = str(edge["product_molecule_id"])
        precursor_ids = sorted(str(value) for value in edge["precursor_molecule_ids"])
        signature = sha(
            json.dumps(
                {
                    "product_canonical_isomeric_smiles": smiles_by_id[product_id],
                    "reactant_canonical_isomeric_smiles": sorted(
                        smiles_by_id[value] for value in precursor_ids
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        exact_bindings[edge_id] = with_named_hash(
            {
                "schema_version": "exact_edge_proof_binding.v1",
                "hyperedge_id": edge_id,
                "product_molecule_id": product_id,
                "precursor_molecule_ids": precursor_ids,
                "structure_signature_sha256": signature,
                "proof_level": named_proof_level,
                "portfolio_proof_level": proof_level,
                "advisory": False,
                "proof_accepted": True,
                "proof_digest": sha(f"proof:{edge_id}"),
                "route_proof_digest": sha(f"route:{edge_id}"),
                "reaction_digest": signature,
                "trusted_precedent_sha256": sha(f"precedent:{edge_id}"),
                "validator_version": "fixture.verifier.v1",
                "proof_source": "route_proof_bank.v1",
                "proof_bank_entry_id": f"proof-entry:{edge_id}",
                "proof_bank_entry_sha256": sha(f"proof-entry:{edge_id}"),
            },
            "binding_sha256",
        )
    all_leaves = sorted(
        {
            leaf
            for route in routes
            for leaf in route.get("stock_terminal_ids") or []
        }
    )
    stock_bindings = {
        molecule_id: with_named_hash(
            {
                "schema_version": "exact_stock_binding.v1",
                "molecule_id": molecule_id,
                "canonical_isomeric_smiles": smiles_by_id[molecule_id],
                "catalog_id": f"catalog:{molecule_id}",
                "catalog_sha256": sha(f"catalog:{molecule_id}"),
                "lookup_basis": "exact_canonical_smiles",
                "boundary_type": (
                    "commercially_orderable"
                    if proof_level == 4
                    else "benchmark_stock"
                ),
                "benchmark_membership": proof_level != 4,
                "commercial_orderability_claimed": proof_level == 4,
                "snapshot_digest_replayed": proof_level == 4,
                "provider_trust_authority": (
                    "autoplanner_host_builtin_allowlist.v1"
                    if proof_level == 4
                    else ""
                ),
                "provider_descriptor_sha256": (
                    sha(f"provider:{molecule_id}") if proof_level == 4 else ""
                ),
                "stock_audit_sha256": sha(f"stock-audit:{molecule_id}"),
                "evidence_sha256": sha(f"stock-evidence:{molecule_id}"),
                "binding_authority": "strictly_replayed_route_proof_bank.v1",
                "proof_bank_authorities": [
                    {
                        "proof_bank_entry_id": f"proof-entry:{molecule_id}",
                        "proof_bank_entry_sha256": sha(f"proof-entry:{molecule_id}"),
                        "stock_evidence_binding_sha256": sha(
                            f"stock-evidence-binding:{molecule_id}"
                        ),
                    }
                ],
            },
            "binding_sha256",
        )
        for molecule_id in all_leaves
    }
    bindings = with_content_hash(
        {
            "schema_version": "route_portfolio_bindings.v1",
            "stock_molecule_ids": all_leaves,
            "edge_proof_levels": {
                edge_id: proof_level for edge_id in exact_bindings
            },
            "exact_edge_proof_bindings": exact_bindings,
            "stock_bindings": stock_bindings,
            "matched_edge_count": len(exact_bindings),
            "proof_step_count": len(exact_bindings),
            "matched_stock_terminal_count": len(stock_bindings),
            "materialized_terminal_count": len(stock_bindings),
            "unmatched_materialized_terminals": [],
            "stock_binding_valid": True,
            "all_materialized_terminals_proven": bool(stock_bindings),
            "stock_binding_source": "independent_stock_catalog_audit.terminal_evidence",
            "proof_binding_source": "strictly_replayed_route_proof_bank.v1",
            "proof_bank_present": True,
            "proof_bank_fail_closed": False,
            "replayed_proof_bank_entry_count": max(1, len(routes)),
            "binding_is_exact_structure_signature": True,
        }
    )
    branches: list[dict] = []
    branch_views: list[dict] = []
    for route in routes:
        route_id = str(route["route_id"])
        branch_id = f"branch:{route_id}"
        acyclic = _route_acyclic(route, edges_by_id)
        branches.append(
            {
                "branch_id": branch_id,
                "kind": "proof_eligible_portfolio_route",
                "listed": True,
                "proof_eligible": True,
                "portfolio_route_id": route_id,
                "root_molecule_node_id": "node:target",
                "stock_terminal_node_ids": [
                    f"node:{value}" for value in route["stock_terminal_ids"]
                ],
            }
        )
        branch_views.append(
            {
                "branch_id": branch_id,
                "portfolio_route_id": route_id,
                "acyclic": acyclic,
                "all_leaves_stock_bound": bool(route["stock_terminal_ids"] and acyclic),
                "stock_leaf_molecule_node_ids": [
                    f"node:{value}" for value in route["stock_terminal_ids"]
                ],
                "target_molecule_node_ids": ["node:target"],
            }
        )
    accepted_replacements: list[dict] = []
    replacement_records: list[dict] = []
    if len(routes) >= 2:
        base = routes[0]
        result_route = routes[1]
        base_map = {
            row["product_molecule_id"]: row["hyperedge_id"]
            for row in base["selected_hyperedges"]
        }
        result_map = {
            row["product_molecule_id"]: row["hyperedge_id"]
            for row in result_route["selected_hyperedges"]
        }
        changed = next(
            (
                product_id
                for product_id, edge_id in base_map.items()
                if result_map.get(product_id) and result_map[product_id] != edge_id
            ),
            "",
        )
        if changed:
            candidate_id = "replacement:fixture"
            accepted_replacements.append(
                {
                    "candidate_id": candidate_id,
                    "base_route_id": base["route_id"],
                    "product_molecule_id": changed,
                    "original_hyperedge_id": base_map[changed],
                    "replacement_hyperedge_id": result_map[changed],
                    "replacement_rank_score": 0.7,
                    "accepted": True,
                    "route": result_route,
                    "reasons": [],
                    "connectivity_revalidated": True,
                    "stock_closure_revalidated": True,
                    "reaction_proof_revalidated": True,
                }
            )
            replacement_branch_id = f"replacement-branch:{result_route['route_id']}"
            branches.append(
                {
                    "branch_id": replacement_branch_id,
                    "kind": "validated_replacement_route",
                    "listed": False,
                    "proof_eligible": True,
                    "portfolio_route_id": result_route["route_id"],
                    "root_molecule_node_id": "node:target",
                }
            )
            branch_views.append(
                {
                    "branch_id": replacement_branch_id,
                    "portfolio_route_id": result_route["route_id"],
                    "acyclic": True,
                    "all_leaves_stock_bound": True,
                    "target_molecule_node_ids": ["node:target"],
                }
            )
            replacement_records.append(
                {
                    "candidate_id": candidate_id,
                    "replacement_id": candidate_id,
                    "accepted": True,
                    "validated": True,
                    "revalidated_route_branch_id": replacement_branch_id,
                }
            )
    replacement_catalog = with_content_hash(
        {
            "schema_version": "route_replacement_catalog.v1",
            "portfolio_content_sha256": portfolio["content_sha256"],
            "portfolio_integrity_valid": True,
            "candidate_count": len(accepted_replacements),
            "available_candidate_count": len(accepted_replacements),
            "accepted_candidate_count": len(accepted_replacements),
            "rejected_candidate_count": 0,
            "max_candidates": 100,
            "truncated": False,
            "candidates": accepted_replacements,
            "reasons": [],
            "validation_policy": "full_and_or_resolve_with_stock_and_reaction_proof",
        }
    )
    write(
        tmp_path / "route_consensus_graph_fused.json",
        {
            "v2_overlay": {
                "schema_version": "route_hypergraph_overlay.v2",
                "root_molecule_id": "target",
                "validation": {"valid": True, "errors": []},
                "molecules": [
                    {
                        "molecule_id": molecule_id,
                        "canonical_isomeric_smiles": smiles_by_id[molecule_id],
                    }
                    for molecule_id in molecule_ids
                ],
                "reaction_hyperedges": edges,
            },
            "route_portfolio": portfolio,
            "route_portfolio_bindings": bindings,
            "route_replacement_catalog": replacement_catalog,
        },
    )
    write(
        tmp_path / "explored_route_forest.json",
        {
            "dependency_graph": {
                "schema_version": "molecule_reaction_dependency_graph.v1",
                "molecule_nodes": [],
                "reaction_nodes": [],
                "edges": [],
                "branch_views": branch_views,
                "acyclic": False,
                "cycle_graph_node_ids": ["explored:cycle"],
            },
            "branches": branches,
            "replacement_validation": {"records": replacement_records},
            "projection_coverage": {"complete": True},
        },
    )
    write(tmp_path / "agent_blackboard.json", {})
    write(
        tmp_path / "final_verdict.json",
        {"verdict": "unresolved", "route_status": "unresolved", "solved": False},
    )


def test_architecture_audit_validates_every_portfolio_route_and_allows_overlay_cycles(
    tmp_path,
) -> None:
    edges = [
        {
            "hyperedge_id": "edge:target",
            "product_molecule_id": "target",
            "precursor_molecule_ids": ["middle"],
        },
        {
            "hyperedge_id": "edge:middle",
            "product_molecule_id": "middle",
            "precursor_molecule_ids": ["stock"],
        },
    ]
    good_route = {
        "schema_version": "route_portfolio_item.v1",
        "route_id": "route:good",
        "complete": True,
        "reaction_validated": True,
        "selected_hyperedges": [
            {"product_molecule_id": "target", "hyperedge_id": "edge:target"},
            {"product_molecule_id": "middle", "hyperedge_id": "edge:middle"},
        ],
    }
    _write_portfolio_run(tmp_path, routes=[good_route], edges=edges)

    report = audit_run(tmp_path)

    assert report["run_acceptance"]["gates"]["proof_eligible_portfolio_valid"] is True
    assert report["run_acceptance"]["gates"]["selected_route_dags_acyclic"] is True
    assert report["portfolio"]["content_sha256_valid"] is True
    assert report["portfolio"]["selected_route_evidence"][0]["contract_valid"] is True
    assert report["projection"]["explored_overlay_acyclic"] is False
    assert report["projection"]["explored_overlay_cycles_are_allowed"] is True


def test_architecture_audit_reports_l4_procurement_separately_from_parent_proof(
    tmp_path,
) -> None:
    edges = [
        {
            "hyperedge_id": "edge:target",
            "product_molecule_id": "target",
            "precursor_molecule_ids": ["stock"],
        }
    ]
    route = {
        "schema_version": "route_portfolio_item.v1",
        "route_id": "route:l4",
        "complete": True,
        "reaction_validated": True,
        "selected_hyperedges": [
            {"product_molecule_id": "target", "hyperedge_id": "edge:target"}
        ],
    }
    _write_portfolio_run(
        tmp_path,
        routes=[route],
        edges=edges,
        proof_level=4,
    )

    report = audit_run(tmp_path)

    assert report["completion_truth"]["l4_procurement_route_ready"] is True
    assert report["completion_truth"]["l4_procurement_route_ids"] == ["route:l4"]
    assert report["completion_truth"]["l3_parent_route_solved"] is False

    graph_path = tmp_path / "route_consensus_graph_fused.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    bindings = graph["route_portfolio_bindings"]
    stock_binding = bindings["stock_bindings"]["stock"]
    stock_binding.update(
        {
            "boundary_type": "benchmark_stock",
            "benchmark_membership": True,
            "commercial_orderability_claimed": False,
            "snapshot_digest_replayed": False,
            "provider_trust_authority": "",
            "provider_descriptor_sha256": "",
        }
    )
    bindings["stock_bindings"]["stock"] = with_named_hash(
        stock_binding,
        "binding_sha256",
    )
    graph["route_portfolio_bindings"] = with_content_hash(bindings)
    write(graph_path, graph)

    benchmark_only = audit_run(tmp_path)
    assert benchmark_only["portfolio"]["selected_route_evidence"][0][
        "contract_valid"
    ] is True
    assert benchmark_only["completion_truth"]["l4_procurement_route_ready"] is False


def test_architecture_audit_rejects_one_invalid_or_cyclic_selected_route(tmp_path) -> None:
    edges = [
        {
            "hyperedge_id": "edge:target",
            "product_molecule_id": "target",
            "precursor_molecule_ids": ["middle"],
        },
        {
            "hyperedge_id": "edge:middle",
            "product_molecule_id": "middle",
            "precursor_molecule_ids": ["target"],
        },
    ]
    invalid_route = with_content_hash(
        {
            "schema_version": "route_portfolio_item.v1",
            "route_id": "route:cycle",
            "complete": True,
            "reaction_validated": False,
            "selected_hyperedges": [
                {"product_molecule_id": "target", "hyperedge_id": "edge:target"},
                {"product_molecule_id": "middle", "hyperedge_id": "edge:middle"},
            ],
        }
    )
    _write_portfolio_run(tmp_path, routes=[invalid_route], edges=edges)

    report = audit_run(tmp_path)
    evidence = report["portfolio"]["selected_route_evidence"][0]

    assert report["run_acceptance"]["gates"]["proof_eligible_portfolio_valid"] is False
    assert report["run_acceptance"]["gates"]["selected_route_dags_acyclic"] is False
    assert evidence["content_sha256_valid"] is True
    assert evidence["reaction_validated"] is False
    assert evidence["dag_acyclic"] is False
    assert {"route_not_reaction_validated", "selected_route_cycle_detected"} <= set(
        evidence["reasons"]
    )


def test_architecture_audit_rejects_stale_portfolio_content_hash(tmp_path) -> None:
    route = {
        "schema_version": "route_portfolio_item.v1",
        "route_id": "route:hash-bound",
        "complete": True,
        "reaction_validated": True,
        "selected_hyperedges": [
            {"product_molecule_id": "target", "hyperedge_id": "edge:target"}
        ],
    }
    edges = [
        {
            "hyperedge_id": "edge:target",
            "product_molecule_id": "target",
            "precursor_molecule_ids": ["stock"],
        }
    ]
    _write_portfolio_run(tmp_path, routes=[route], edges=edges)
    graph_path = tmp_path / "route_consensus_graph_fused.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["route_portfolio"]["complete_candidate_count"] = 99
    write(graph_path, graph)

    report = audit_run(tmp_path)

    assert report["portfolio"]["content_sha256_valid"] is False
    assert report["run_acceptance"]["gates"]["proof_eligible_portfolio_valid"] is False


def _two_route_fixture() -> tuple[list[dict], list[dict]]:
    edges = [
        {
            "hyperedge_id": "edge:main",
            "product_molecule_id": "target",
            "precursor_molecule_ids": ["middle"],
        },
        {
            "hyperedge_id": "edge:alt",
            "product_molecule_id": "target",
            "precursor_molecule_ids": ["side"],
        },
        {
            "hyperedge_id": "edge:middle",
            "product_molecule_id": "middle",
            "precursor_molecule_ids": ["stock"],
        },
        {
            "hyperedge_id": "edge:side",
            "product_molecule_id": "side",
            "precursor_molecule_ids": ["stock-alt"],
        },
    ]
    routes = [
        {
            "schema_version": "route_portfolio_item.v1",
            "route_id": "route:main",
            "complete": True,
            "reaction_validated": True,
            "selected_hyperedges": [
                {"product_molecule_id": "target", "hyperedge_id": "edge:main"},
                {"product_molecule_id": "middle", "hyperedge_id": "edge:middle"},
            ],
        },
        {
            "schema_version": "route_portfolio_item.v1",
            "route_id": "route:alt",
            "complete": True,
            "reaction_validated": True,
            "selected_hyperedges": [
                {"product_molecule_id": "target", "hyperedge_id": "edge:alt"},
                {"product_molecule_id": "side", "hyperedge_id": "edge:side"},
            ],
        },
    ]
    return routes, edges


def test_audit_requires_hashed_bindings_forest_projection_and_backend_replacement(
    tmp_path,
) -> None:
    routes, edges = _two_route_fixture()
    _write_portfolio_run(tmp_path, routes=routes, edges=edges)

    report = audit_run(tmp_path)
    gates = report["run_acceptance"]["gates"]

    assert gates["proof_eligible_portfolio_valid"] is True
    assert gates["portfolio_routes_projected"] is True
    assert gates["distinct_alternatives_at_least_2"] is True
    assert gates["backend_revalidated_replacement_available"] is True
    portfolio = report["portfolio"]
    assert portfolio["bindings_content_sha256_valid"] is True
    assert portfolio["distinct_valid_route_selection_count"] == 2
    assert portfolio["forest_projection_evidence"]["contract_valid"] is True
    replacement = portfolio["replacement_evidence"]
    assert replacement["catalog_integrity_valid"] is True
    assert replacement["accepted_candidates_valid"] is True
    assert replacement["candidate_evidence"][0]["triple_revalidated"] is True
    assert replacement["candidate_evidence"][0]["forest_projected"] is True


def test_audit_rejects_rehashed_mapping_level_escalation_and_stock_authority(
    tmp_path,
) -> None:
    routes, edges = _two_route_fixture()
    _write_portfolio_run(tmp_path, routes=routes, edges=edges)
    graph_path = tmp_path / "route_consensus_graph_fused.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    bindings = graph["route_portfolio_bindings"]
    binding = bindings["exact_edge_proof_bindings"]["edge:main"]
    # Begin with a mapping-only advisory binding, then simulate an attacker
    # changing every visible level/accepted field and recomputing both hashes.
    binding.update(
        {
            "proof_level": "L2_mapping_consistent",
            "portfolio_proof_level": 0,
            "proof_accepted": False,
            "advisory": True,
            "trusted_precedent_sha256": "",
        }
    )
    binding = with_named_hash(binding, "binding_sha256")
    binding.update(
        {
            "proof_level": "L2_reaction_validated",
            "portfolio_proof_level": 2,
            "proof_accepted": True,
            "advisory": False,
            "proof_source": "route_proof_bank.v1",
        }
    )
    bindings["edge_proof_levels"]["edge:main"] = 2
    bindings["exact_edge_proof_bindings"]["edge:main"] = with_named_hash(
        binding,
        "binding_sha256",
    )
    graph["route_portfolio_bindings"] = with_content_hash(bindings)
    write(graph_path, graph)

    escalated = audit_run(tmp_path)
    edge_evidence = escalated["portfolio"]["selected_route_evidence"][0][
        "edge_binding_evidence"
    ][0]
    assert edge_evidence["valid"] is False
    assert "exact_edge_binding_l2_transform_authority_invalid" in edge_evidence["reasons"]
    assert escalated["run_acceptance"]["gates"]["proof_eligible_portfolio_valid"] is False

    _write_portfolio_run(tmp_path, routes=routes, edges=edges)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    bindings = graph["route_portfolio_bindings"]
    stock_binding = bindings["stock_bindings"]["stock"]
    stock_binding["binding_authority"] = "model_claim"
    bindings["stock_bindings"]["stock"] = with_named_hash(
        stock_binding,
        "binding_sha256",
    )
    graph["route_portfolio_bindings"] = with_content_hash(bindings)
    write(graph_path, graph)

    stock_tampered = audit_run(tmp_path)
    assert stock_tampered["run_acceptance"]["gates"]["proof_eligible_portfolio_valid"] is False
    stock_evidence = stock_tampered["portfolio"]["selected_route_evidence"][0][
        "stock_binding_evidence"
    ][0]
    assert "exact_stock_binding_authority_untrusted" in stock_evidence["reasons"]


def test_audit_rejects_missing_forest_projection_and_rehashed_replacement_claim(
    tmp_path,
) -> None:
    routes, edges = _two_route_fixture()
    _write_portfolio_run(tmp_path, routes=routes, edges=edges)
    forest_path = tmp_path / "explored_route_forest.json"
    forest = json.loads(forest_path.read_text(encoding="utf-8"))
    forest["dependency_graph"]["branch_views"][0]["acyclic"] = False
    write(forest_path, forest)

    projection_tampered = audit_run(tmp_path)
    assert projection_tampered["run_acceptance"]["gates"]["portfolio_routes_projected"] is False

    _write_portfolio_run(tmp_path, routes=routes, edges=edges)
    graph_path = tmp_path / "route_consensus_graph_fused.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    candidate = graph["route_replacement_catalog"]["candidates"][0]
    candidate["reaction_proof_revalidated"] = False
    graph["route_replacement_catalog"] = with_content_hash(
        graph["route_replacement_catalog"]
    )
    write(graph_path, graph)

    replacement_tampered = audit_run(tmp_path)
    replacement = replacement_tampered["portfolio"]["replacement_evidence"]
    assert replacement["catalog_content_sha256_valid"] is True
    assert replacement["accepted_candidates_valid"] is False
    assert (
        replacement_tampered["run_acceptance"]["gates"]
        ["backend_revalidated_replacement_available"]
        is False
    )


def test_architecture_audit_prefers_cas_decision_and_forest_over_drifted_views(
    tmp_path,
) -> None:
    _write_portfolio_run(tmp_path, routes=[], edges=[])
    proof_snapshot_path = tmp_path / "parent_route_proof_snapshot.json"
    verdict_core_path = tmp_path / "final_verdict_core.json"
    frontier_ledger_path = tmp_path / "frontier_ledger.json"
    reconciliation_path = tmp_path / "codex_campaign_proof_reconciliation.json"
    reconciliation_queue = with_content_hash(
        {
            "schema_version": "frontier_queue.v1",
            "run_id": "test",
            "revision": 2,
            "jobs": [],
        }
    )
    proof_snapshot = {
        "schema_version": "parent_route_proof_snapshot.v1",
        "case_id": "test",
        "target_smiles": "CCO",
        "proof_schema_version": "missing",
        "solved": False,
        "authority": "deterministic_parent_route_proof",
        "proof": {},
    }
    verdict = {
        "case_id": "test",
        "verdict": "unresolved",
        "route_status": "unresolved",
        "solved": False,
        "stock_audit_passed": False,
        "reasons": ["no_deterministic_parent_route_proof"],
    }
    verdict_core = {
        "schema_version": "final_verdict_core.v1",
        "case_id": "test",
        "authority": "deterministic_parent_route_proof",
        "parent_route_proof_solved": False,
        "validation": {"accepted": True, "reasons": []},
        "verdict": verdict,
    }
    write(proof_snapshot_path, proof_snapshot)
    write(verdict_core_path, verdict_core)
    write(
        frontier_ledger_path,
        frontier_ledger_fixture(
            any_route_closed=False,
            all_explored_graph_closed=False,
            queue_override=reconciliation_queue,
        ),
    )
    write(
        reconciliation_path,
        {
            "schema_version": "codex_campaign_proof_reconciliation.v1",
            "accepted": True,
            "frontier_queue": reconciliation_queue,
            "frontier_completeness": {
                "complete": True,
                "authority_marker": "cas-reconciliation",
            },
            "proposal_graph_exhausted": False,
        },
    )
    publish_closeout_revision(
        tmp_path,
        artifacts={
            "route_consensus_graph": tmp_path / "route_consensus_graph_fused.json",
            "frontier_ledger": frontier_ledger_path,
            "codex_campaign_proof_reconciliation": reconciliation_path,
            "parent_route_proof_snapshot": proof_snapshot_path,
            "final_verdict_core": verdict_core_path,
            "explored_route_forest": tmp_path / "explored_route_forest.json",
        },
        dependencies={
            "codex_campaign_proof_reconciliation": (
                "route_consensus_graph",
            ),
            "frontier_ledger": (
                "route_consensus_graph",
                "codex_campaign_proof_reconciliation",
            ),
            "parent_route_proof_snapshot": ("route_consensus_graph",),
            "final_verdict_core": ("parent_route_proof_snapshot",),
            "explored_route_forest": (
                "route_consensus_graph",
                "parent_route_proof_snapshot",
                "final_verdict_core",
            ),
        },
        producer="architecture-audit-test",
        case_id="test",
    )

    # Drift only the mutable compatibility views after the CAS commit.
    write(
        tmp_path / "agent_blackboard.json",
        {"parent_route_proof": {"accepted": True, "solved": True, "forged": True}},
    )
    write(
        tmp_path / "final_verdict.json",
        {"verdict": "solved", "route_status": "solved", "solved": True},
    )
    write(
        tmp_path / "explored_route_forest.json",
        {
            "dependency_graph": {"schema_version": "drifted.v1"},
            "projection_coverage": {"complete": False},
        },
    )
    write(
        frontier_ledger_path,
        frontier_ledger_fixture(
            any_route_closed=True,
            all_explored_graph_closed=True,
        ),
    )
    write(
        reconciliation_path,
        {
            "schema_version": "codex_campaign_proof_reconciliation.v1",
            "accepted": True,
            "frontier_queue": with_content_hash(
                {
                    "schema_version": "frontier_queue.v1",
                    "run_id": "test",
                    "revision": 99,
                    "jobs": [],
                }
            ),
            "frontier_completeness": {
                "complete": False,
                "authority_marker": "drifted-compatibility-view",
            },
            "proposal_graph_exhausted": True,
        },
    )

    report = audit_run(tmp_path)

    assert report["closeout"]["accepted"] is True
    assert report["closeout"]["authority_source"] == "committed_cas_revision"
    assert report["run_acceptance"]["gates"]["closeout_parent_proof_bound"] is True
    assert report["run_acceptance"]["gates"]["closeout_final_verdict_bound"] is True
    assert report["run_acceptance"]["gates"]["closeout_frontier_ledger_bound"] is True
    assert report["frontier_ledger"]["authority_source"] == "committed_cas_revision"
    assert report["frontier_ledger"]["current_identities"][
        "frontier_queue_revision"
    ] == 2
    assert not {
        "frontier_ledger_queue_digest_binding_mismatch",
        "frontier_ledger_queue_revision_binding_mismatch",
    }.intersection(report["frontier_ledger"]["authority_blockers"])
    assert report["frontier_scheduler"]["authority_source"] == (
        "committed_cas_reconciliation"
    )
    assert report["frontier_scheduler"]["completeness"] == {
        "complete": True,
        "authority_marker": "cas-reconciliation",
    }
    assert report["frontier_scheduler"]["proposal_graph_exhausted"] is False
    assert (
        report["run_acceptance"]["gates"]
        ["closeout_codex_campaign_proof_reconciliation_bound"]
        is True
    )
    assert report["completion_truth"]["any_route_closed"] is False
    assert report["final_verdict"]["verdict"] == "unresolved"
    assert report["final_verdict"]["solved"] is False
    assert report["projection"]["projection_complete"] is True
    assert report["compatibility_projection"]["board_parent_proof_matches_cas"] is False
    assert report["compatibility_projection"]["final_verdict_semantics_match_cas"] is False
    assert report["compatibility_projection"]["forest_matches_cas"] is False
    assert report["compatibility_projection"]["frontier_ledger_matches_cas"] is False
    assert report["compatibility_projection"]["drift_is_diagnostic_only"] is True


def test_architecture_audit_fails_closed_when_cas_reconciliation_is_missing(
    tmp_path,
) -> None:
    _write_portfolio_run(tmp_path, routes=[], edges=[])
    proof_snapshot_path = tmp_path / "parent_route_proof_snapshot.json"
    verdict_core_path = tmp_path / "final_verdict_core.json"
    frontier_ledger_path = tmp_path / "frontier_ledger.json"
    compatibility_reconciliation_path = (
        tmp_path / "codex_campaign_proof_reconciliation.json"
    )
    proof_snapshot = {
        "schema_version": "parent_route_proof_snapshot.v1",
        "case_id": "test",
        "target_smiles": "CCO",
        "proof_schema_version": "missing",
        "solved": False,
        "authority": "deterministic_parent_route_proof",
        "proof": {},
    }
    verdict = {
        "case_id": "test",
        "verdict": "unresolved",
        "route_status": "unresolved",
        "solved": False,
        "stock_audit_passed": False,
        "reasons": ["no_deterministic_parent_route_proof"],
    }
    verdict_core = {
        "schema_version": "final_verdict_core.v1",
        "case_id": "test",
        "authority": "deterministic_parent_route_proof",
        "parent_route_proof_solved": False,
        "validation": {"accepted": True, "reasons": []},
        "verdict": verdict,
    }
    write(proof_snapshot_path, proof_snapshot)
    write(verdict_core_path, verdict_core)
    write(
        frontier_ledger_path,
        frontier_ledger_fixture(
            any_route_closed=False,
            all_explored_graph_closed=False,
        ),
    )
    # This mutable compatibility artifact must never fill a hole in a
    # committed CAS revision.
    write(
        compatibility_reconciliation_path,
        {
            "schema_version": "codex_campaign_proof_reconciliation.v1",
            "accepted": True,
            "frontier_queue": with_content_hash(
                {
                    "schema_version": "frontier_queue.v1",
                    "run_id": "test",
                    "revision": 77,
                    "jobs": [],
                }
            ),
            "frontier_completeness": {"complete": True},
            "proposal_graph_exhausted": True,
        },
    )
    publish_closeout_revision(
        tmp_path,
        artifacts={
            "route_consensus_graph": tmp_path / "route_consensus_graph_fused.json",
            "frontier_ledger": frontier_ledger_path,
            "parent_route_proof_snapshot": proof_snapshot_path,
            "final_verdict_core": verdict_core_path,
            "explored_route_forest": tmp_path / "explored_route_forest.json",
        },
        dependencies={
            "frontier_ledger": ("route_consensus_graph",),
            "parent_route_proof_snapshot": ("route_consensus_graph",),
            "final_verdict_core": ("parent_route_proof_snapshot",),
            "explored_route_forest": (
                "route_consensus_graph",
                "parent_route_proof_snapshot",
                "final_verdict_core",
            ),
        },
        producer="architecture-audit-missing-reconciliation-test",
        case_id="test",
    )

    report = audit_run(tmp_path)

    blocker = "cas_codex_campaign_proof_reconciliation_missing"
    assert report["closeout"]["accepted"] is True
    assert report["closeout"][
        "codex_campaign_proof_reconciliation_bound"
    ] is False
    assert blocker in report["closeout"][
        "codex_campaign_proof_reconciliation_blockers"
    ]
    assert report["frontier_scheduler"]["authority_source"] == (
        "invalid_or_missing_cas_reconciliation"
    )
    assert report["frontier_scheduler"]["job_count"] == 0
    assert report["frontier_scheduler"]["completeness"] == {}
    assert report["frontier_scheduler"]["proposal_graph_exhausted"] is False
    assert report["executable_contract_evidence"]["contracts"][
        "frontier_scheduler"
    ]["contract_valid"] is False
    assert report["run_acceptance"]["gates"][
        "closeout_codex_campaign_proof_reconciliation_bound"
    ] is False
    assert report["frontier_ledger"]["authority_valid"] is False
    assert blocker in report["frontier_ledger"]["authority_blockers"]

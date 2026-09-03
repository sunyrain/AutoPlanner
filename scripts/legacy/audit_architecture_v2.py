#!/usr/bin/env python3
"""Audit Architecture V2 capabilities and one run's chemistry acceptance gates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.legacy.application_runtime.frontier_ledger import (  # noqa: E402
    FRONTIER_LEDGER_SCHEMA,
    validate_frontier_ledger,
)
from cascade_planner.legacy.harness_runtime.parent_route_proof import (  # noqa: E402
    is_solved_parent_route_proof,
)
from cascade_planner.legacy.harness_runtime.route_forest import (  # noqa: E402
    frontier_ledger_current_authority_evidence,
)
from cascade_planner.providers.stock import (  # noqa: E402
    SnapshotStockProvider,
    build_trusted_stock_provider_instances,
)
from cascade_planner.legacy.runtime.artifact_revision import (  # noqa: E402
    ArtifactRevisionError,
    load_latest_closeout_artifact,
    load_latest_closeout_decision,
    validate_latest_closeout_revision,
)


SCHEMA_VERSION = "architecture_v2_audit.v1"


def audit_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    target = _load(root / "target_input.json")
    preflight = _load(root / "preflight.json")
    compatibility_board = _load(root / "agent_blackboard.json")
    team = _load(root / "codex_retrosynthesis_team" / "team_report.json")
    compatibility_graph = _load(root / "route_consensus_graph_fused.json")
    if not compatibility_graph:
        compatibility_graph = dict(team.get("route_consensus_graph") or {})
    compatibility_authority_graph = _load(
        root / "route_consensus_graph_canonical.json"
    )
    compatibility_reconciliation = _load(
        root / "codex_campaign_proof_reconciliation.json"
    )
    if not compatibility_authority_graph:
        compatibility_authority_graph = dict(
            compatibility_board.get("canonical_route_consensus_graph") or {}
        )
    if not compatibility_authority_graph:
        compatibility_authority_graph = dict(
            compatibility_reconciliation.get(
                "canonical_route_consensus_graph"
            )
            or {}
        )
    compatibility_ledger = _load(root / "frontier_ledger.json")
    compatibility_forest = _load(root / "explored_route_forest.json")
    compatibility_final = _load(root / "final_verdict.json")
    compatibility_proof = dict(compatibility_board.get("parent_route_proof") or {})
    closeout = validate_latest_closeout_revision(root)
    cas_closeout_mode = closeout.get("present") is True
    closeout_manifest_path = str(closeout.get("manifest_path") or "").strip()
    closeout_manifest = _load(Path(closeout_manifest_path)) if closeout_manifest_path else {}
    closeout_artifact_ids = {
        str(row.get("artifact_id") or "")
        for row in closeout_manifest.get("artifacts") or []
        if isinstance(row, Mapping)
    }
    cas_decision: dict[str, Any] = {}
    cas_graph: dict[str, Any] = {}
    cas_authority_graph: dict[str, Any] = {}
    cas_forest: dict[str, Any] = {}
    cas_ledger: dict[str, Any] = {}
    cas_reconciliation: dict[str, Any] = {}
    cas_load_errors: list[str] = []
    if closeout.get("accepted") is True:
        try:
            cas_decision = load_latest_closeout_decision(root)
        except (ArtifactRevisionError, OSError, ValueError) as exc:
            cas_load_errors.append(f"decision:{type(exc).__name__}:{exc}")
        artifact_destinations = [
            ("route_consensus_graph", "graph"),
            ("explored_route_forest", "forest"),
        ]
        if "canonical_route_consensus_graph" in closeout_artifact_ids:
            artifact_destinations.append(
                ("canonical_route_consensus_graph", "authority_graph")
            )
        if "frontier_ledger" in closeout_artifact_ids:
            artifact_destinations.append(("frontier_ledger", "ledger"))
        if "codex_campaign_proof_reconciliation" in closeout_artifact_ids:
            artifact_destinations.append(
                (
                    "codex_campaign_proof_reconciliation",
                    "reconciliation",
                )
            )
        for artifact_id, destination in artifact_destinations:
            try:
                value = load_latest_closeout_artifact(root, artifact_id)
            except (ArtifactRevisionError, OSError, ValueError) as exc:
                cas_load_errors.append(f"{artifact_id}:{type(exc).__name__}:{exc}")
                continue
            if destination == "graph":
                cas_graph = value
            elif destination == "authority_graph":
                cas_authority_graph = value
            elif destination == "ledger":
                cas_ledger = value
            elif destination == "reconciliation":
                cas_reconciliation = value
            else:
                cas_forest = value

    cas_reconciliation_authority_blockers: list[str] = []
    if cas_closeout_mode:
        if closeout.get("accepted") is not True:
            cas_reconciliation_authority_blockers.append(
                "cas_closeout_revision_invalid"
            )
        elif "codex_campaign_proof_reconciliation" not in closeout_artifact_ids:
            cas_reconciliation_authority_blockers.append(
                "cas_codex_campaign_proof_reconciliation_missing"
            )
        elif not cas_reconciliation:
            cas_reconciliation_authority_blockers.append(
                "cas_codex_campaign_proof_reconciliation_load_failed"
            )
        else:
            if (
                cas_reconciliation.get("schema_version")
                != "codex_campaign_proof_reconciliation.v1"
            ):
                cas_reconciliation_authority_blockers.append(
                    "cas_codex_campaign_proof_reconciliation_schema_invalid"
                )
            if cas_reconciliation.get("accepted") is not True:
                cas_reconciliation_authority_blockers.append(
                    "cas_codex_campaign_proof_reconciliation_rejected"
                )
            if not isinstance(
                cas_reconciliation.get("frontier_queue"), Mapping
            ):
                cas_reconciliation_authority_blockers.append(
                    "cas_codex_campaign_proof_reconciliation_frontier_queue_missing"
                )
    cas_reconciliation_bound = bool(
        cas_closeout_mode
        and not cas_reconciliation_authority_blockers
    )

    graph = cas_graph or compatibility_graph
    authority_graph = (
        cas_authority_graph
        or compatibility_authority_graph
        or graph
    )
    forest = cas_forest or compatibility_forest
    frontier_ledger = cas_ledger if cas_decision else compatibility_ledger
    proof = (
        dict(cas_decision.get("parent_route_proof") or {})
        if cas_decision
        else compatibility_proof
    )
    final = (
        dict(cas_decision.get("final_verdict") or {})
        if cas_decision
        else compatibility_final
    )
    authority_source = "committed_cas_revision" if cas_decision else "compatibility_projection"
    overlay = dict(graph.get("v2_overlay") or {})
    hyperedges = [
        dict(row)
        for row in overlay.get("reaction_hyperedges") or []
        if isinstance(row, Mapping)
    ]
    campaign = dict(team.get("campaign") or {})
    if cas_closeout_mode:
        reconciliation = (
            cas_reconciliation if cas_reconciliation_bound else {}
        )
        scheduler_facts = reconciliation
        scheduler_authority_source = (
            "committed_cas_reconciliation"
            if cas_reconciliation_bound
            else "invalid_or_missing_cas_reconciliation"
        )
    elif compatibility_reconciliation:
        reconciliation = compatibility_reconciliation
        scheduler_facts = reconciliation
        scheduler_authority_source = "compatibility_reconciliation"
    else:
        reconciliation = {}
        scheduler_facts = campaign
        scheduler_authority_source = "team_campaign_compatibility"
    queue = dict(scheduler_facts.get("frontier_queue") or {})
    jobs = [dict(row) for row in queue.get("jobs") or [] if isinstance(row, Mapping)]
    portfolio = dict(graph.get("route_portfolio") or {})
    portfolio_bindings = dict(graph.get("route_portfolio_bindings") or {})
    replacement_catalog = dict(graph.get("route_replacement_catalog") or {})
    portfolio_routes = [
        dict(row) for row in portfolio.get("routes") or [] if isinstance(row, Mapping)
    ]
    dependency_graph = dict(forest.get("dependency_graph") or {})
    forest_branches = [
        dict(row) for row in forest.get("branches") or [] if isinstance(row, Mapping)
    ]
    projection = dict(forest.get("projection_coverage") or {})
    target_smiles = str(target.get("target_smiles") or "")
    campaign_policy = _audit_campaign_policy(root, campaign=campaign, team=team)
    trusted_stock_providers, stock_replay_context_reasons = (
        _audit_trusted_stock_providers(
            root,
            campaign_policy=campaign_policy,
        )
    )
    frontier_ledger_evidence = _frontier_ledger_evidence(
        frontier_ledger,
        expected_target_smiles=str(
            authority_graph.get("target_smiles") or target_smiles
        ),
        route_consensus_graph=authority_graph,
        frontier_queue=queue,
        campaign_policy=campaign_policy,
        campaign_policy_sha256=str(
            campaign.get("campaign_policy_sha256")
            or team.get("campaign_policy_sha256")
            or ""
        ),
        trusted_stock_provider_instances=trusted_stock_providers,
    )
    frontier_ledger_evidence["stock_replay_context_reasons"] = list(
        stock_replay_context_reasons
    )
    frontier_ledger_evidence["authority_source"] = (
        "committed_cas_revision" if cas_decision else "compatibility_projection"
    )
    frontier_ledger_evidence[
        "codex_campaign_proof_reconciliation_bound"
    ] = cas_reconciliation_bound if cas_closeout_mode else None
    if cas_reconciliation_authority_blockers:
        frontier_ledger_evidence["authority_blockers"] = sorted(
            set(
                [
                    *frontier_ledger_evidence.get("authority_blockers", []),
                    *cas_reconciliation_authority_blockers,
                ]
            )
        )
        frontier_ledger_evidence["authority_valid"] = False
        frontier_ledger_evidence["effective_any_route_closed"] = False
        frontier_ledger_evidence[
            "effective_all_explored_graph_closed"
        ] = False
    cas_proof_snapshot = dict(cas_decision.get("proof_snapshot") or {})
    cas_verdict_core = dict(cas_decision.get("final_verdict_core") or {})
    cas_proof_solved = is_solved_parent_route_proof(
        proof,
        expected_target_smiles=target_smiles,
    )
    cas_proof_binding_valid = bool(
        cas_decision.get("schema_version") == "closeout_decision.v1"
        and cas_proof_snapshot.get("schema_version") == "parent_route_proof_snapshot.v1"
        and dict(cas_proof_snapshot.get("proof") or {}) == proof
        and cas_proof_snapshot.get("solved") is cas_proof_solved
    )
    cas_final_binding_valid = bool(
        cas_decision.get("schema_version") == "closeout_decision.v1"
        and cas_verdict_core.get("schema_version") == "final_verdict_core.v1"
        and dict(cas_verdict_core.get("verdict") or {}) == final
        and cas_verdict_core.get("parent_route_proof_solved") is cas_proof_solved
        and (final.get("solved") is True) is cas_proof_solved
    )
    compatibility_projection = {
        "schema_version": "architecture_compatibility_projection_drift.v1",
        "cas_authoritative": bool(cas_decision),
        "validation_reports_drift": closeout.get("compatibility_projection_drift") is True,
        "board_parent_proof_matches_cas": (
            compatibility_proof == proof if cas_decision else None
        ),
        "final_verdict_semantics_match_cas": (
            _verdict_semantics(compatibility_final) == _verdict_semantics(final)
            if cas_decision
            else None
        ),
        "forest_matches_cas": (
            compatibility_forest == forest if cas_forest else None
        ),
        "frontier_ledger_matches_cas": (
            compatibility_ledger == frontier_ledger if cas_ledger else None
        ),
        "canonical_authority_graph_matches_cas": (
            compatibility_authority_graph == authority_graph
            if cas_authority_graph
            else None
        ),
        "proof_reconciliation_matches_cas": (
            compatibility_reconciliation == cas_reconciliation
            if cas_reconciliation
            else None
        ),
        "caller_graph_is_display_portfolio_projection_only": bool(
            authority_graph is not graph or compatibility_authority_graph
        ),
        "drift_is_diagnostic_only": True,
    }

    required_children = list((team.get("coordinator") or {}).get("required_child_roles") or [])
    observed_children = list((team.get("coordinator") or {}).get("observed_child_agents") or [])
    children_complete = bool(required_children) and len(observed_children) == len(required_children)
    children_complete = children_complete and all(
        row.get("report_accepted") is True for row in observed_children if isinstance(row, Mapping)
    )
    codex_correlation_honest = all(
        not any(str(channel).startswith("codex_") for channel in edge.get("source_channels") or [])
        or {
            str(group)
            for group in edge.get("independent_support_groups") or []
            if str(group).startswith("codex")
        }
        in (set(), {"codex_model"})
        for edge in hyperedges
    )
    multi_source_edges = [
        edge for edge in hyperedges if len(set(edge.get("independent_support_groups") or [])) > 1
    ]
    trust_tiers = Counter(
        str((row.get("trust_vector") or {}).get("proof_tier") or "unassigned")
        for row in forest.get("steps") or []
        if isinstance(row, Mapping)
    )

    portfolio_route_evidence = [
        _portfolio_route_evidence(
            route,
            overlay=overlay,
            bindings=portfolio_bindings,
        )
        for route in portfolio_routes
    ]
    portfolio_hash_valid = _content_sha256_valid(portfolio)
    portfolio_bindings_hash_valid = _content_sha256_valid(portfolio_bindings)
    portfolio_bindings_valid = bool(
        portfolio_bindings.get("schema_version") == "route_portfolio_bindings.v1"
        and portfolio_bindings_hash_valid is True
    )
    portfolio_routes_valid = bool(portfolio_route_evidence) and all(
        row["contract_valid"] is True for row in portfolio_route_evidence
    )
    selected_route_dags_acyclic = bool(portfolio_route_evidence) and all(
        row["dag_acyclic"] is True and row["selection_valid"] is True
        for row in portfolio_route_evidence
    )
    portfolio_contract_valid = bool(
        portfolio.get("schema_version") == "route_portfolio.v1"
        and portfolio_hash_valid is True
        and portfolio_bindings_valid
        and portfolio_routes_valid
    )
    portfolio_projection_evidence = _portfolio_projection_evidence(
        portfolio_routes,
        forest_branches=forest_branches,
        branch_views=[
            dict(row)
            for row in dependency_graph.get("branch_views") or []
            if isinstance(row, Mapping)
        ],
    )
    portfolio_routes_projected = portfolio_projection_evidence["contract_valid"] is True
    distinct_alternative_count = len(
        {
            tuple(
                sorted(
                    (
                        str(row.get("product_molecule_id") or ""),
                        str(row.get("hyperedge_id") or ""),
                    )
                    for row in route.get("selected_hyperedges") or []
                    if isinstance(row, Mapping)
                )
            )
            for route, evidence in zip(portfolio_routes, portfolio_route_evidence)
            if evidence["contract_valid"] is True
        }
    )
    replacement_evidence = _replacement_catalog_evidence(
        replacement_catalog,
        portfolio=portfolio,
        portfolio_routes=portfolio_routes,
        overlay=overlay,
        bindings=portfolio_bindings,
        forest_branches=forest_branches,
        branch_views=[
            dict(row)
            for row in dependency_graph.get("branch_views") or []
            if isinstance(row, Mapping)
        ],
        replacement_records=[
            dict(row)
            for row in (forest.get("replacement_validation") or {}).get("records") or []
            if isinstance(row, Mapping)
        ],
    )
    l4_procurement_route_ids = sorted(
        str(route.get("route_id") or "")
        for route, evidence in zip(portfolio_routes, portfolio_route_evidence)
        if evidence["contract_valid"] is True
        and int(evidence.get("recomputed_weakest_proof_level") or -1) >= 4
        and bool(evidence.get("stock_binding_evidence"))
        and all(
            row.get("procurement_boundary_valid") is True
            for row in evidence.get("stock_binding_evidence") or []
        )
        and route.get("procurement_ready") is True
        and str(route.get("route_id") or "")
    )
    completion_truth = {
        "schema_version": "architecture_completion_truth.v1",
        "ledger_required_reaction_proof_level": frontier_ledger_evidence[
            "required_reaction_proof_level"
        ],
        "any_route_closed": frontier_ledger_evidence["effective_any_route_closed"],
        "all_explored_graph_closed": frontier_ledger_evidence[
            "effective_all_explored_graph_closed"
        ],
        "l3_parent_route_solved": cas_proof_solved,
        "l4_procurement_route_ready": bool(l4_procurement_route_ids),
        "l4_procurement_route_ids": l4_procurement_route_ids,
        "semantics": {
            "any_route_is_existential_hypergraph_closure": True,
            "all_explored_is_universal_hypergraph_closure": True,
            "l3_parent_requires_deterministic_parent_proof": True,
            "l4_procurement_requires_a_contract_valid_all_l4_route": True,
            "no_completion_field_implies_another_stronger_field": True,
        },
    }

    capability_gates = {
        "provider_spi": ROOT.joinpath("cascade_planner/providers/contracts.py").is_file(),
        "direct_codex_agent_backend": ROOT.joinpath(
            "cascade_planner/legacy/providers.py"
        ).is_file(),
        "per_intermediate_hypergraph_v2": ROOT.joinpath(
            "cascade_planner/routes/overlay.py"
        ).is_file(),
        "reaction_proof_ladder": ROOT.joinpath(
            "cascade_planner/harness/reaction_step_verifier.py"
        ).is_file(),
        "persistent_frontier_scheduler": ROOT.joinpath(
            "cascade_planner/legacy/application_runtime/frontier_scheduler.py"
        ).is_file(),
        "unified_frontier_ledger_projection": ROOT.joinpath(
            "cascade_planner/legacy/application_runtime/frontier_ledger.py"
        ).is_file(),
        "and_or_diverse_portfolio": ROOT.joinpath(
            "cascade_planner/legacy/application_runtime/route_portfolio.py"
        ).is_file(),
        "content_addressed_closeout": ROOT.joinpath(
            "cascade_planner/legacy/runtime/artifact_revision.py"
        ).is_file(),
        "global_dependency_dag": ROOT.joinpath(
            "cascade_planner/legacy/harness_runtime/route_forest.py"
        ).is_file(),
        "codex_edge_verification_bridge": ROOT.joinpath(
            "cascade_planner/legacy/harness_runtime/codex_edge_verification.py"
        ).is_file(),
    }
    executable_contracts = {
        "codex_team_report": _contract_evidence(
            artifact="codex_retrosynthesis_team/team_report.json",
            observed_schema=team.get("schema_version"),
            expected_schema="codex_retrosynthesis_team_run.v1",
            required_shapes=(
                isinstance(team.get("coordinator"), Mapping),
                isinstance(team.get("runtime_summary"), Mapping),
                isinstance(team.get("reasons"), list),
            ),
        ),
        "hypergraph_v2": _contract_evidence(
            artifact="route_consensus_graph_fused.json:v2_overlay",
            observed_schema=overlay.get("schema_version"),
            expected_schema="route_hypergraph_overlay.v2",
            required_shapes=(
                isinstance(overlay.get("molecules"), list),
                isinstance(overlay.get("reaction_hyperedges"), list),
                isinstance(overlay.get("validation"), Mapping),
            ),
        ),
        "frontier_scheduler": _contract_evidence(
            artifact=f"{scheduler_authority_source}:frontier_queue",
            observed_schema=queue.get("schema_version"),
            expected_schema="frontier_queue.v1",
            required_shapes=(
                (
                    cas_reconciliation_bound
                    if cas_closeout_mode
                    else (
                        reconciliation.get("schema_version")
                        == "codex_campaign_proof_reconciliation.v1"
                        if reconciliation
                        else campaign.get("schema_version")
                        == "codex_retrosynthesis_campaign.v1"
                    )
                ),
                isinstance(queue.get("jobs"), list),
                _content_sha256_valid(queue) is not False,
                all(row.get("schema_version") == "frontier_job.v1" for row in jobs),
            ),
        ),
        "frontier_ledger": _contract_evidence(
            artifact=(
                "committed_cas_revision:frontier_ledger"
                if cas_decision
                else "frontier_ledger.json"
            ),
            observed_schema=frontier_ledger.get("schema_version"),
            expected_schema=FRONTIER_LEDGER_SCHEMA,
            required_shapes=(frontier_ledger_evidence["contract_valid"],),
        ),
        "route_portfolio": _contract_evidence(
            artifact="route_consensus_graph_fused.json:route_portfolio",
            observed_schema=portfolio.get("schema_version"),
            expected_schema="route_portfolio.v1",
            required_shapes=(
                isinstance(portfolio.get("routes"), list),
                portfolio_hash_valid is True,
                portfolio_bindings_valid,
                all(row["contract_valid"] is True for row in portfolio_route_evidence),
            ),
            allow_empty_collection=True,
        ),
        "selected_route_dags": _contract_evidence(
            artifact="route_portfolio.routes:selected_hyperedges",
            observed_schema=(
                "selected_route_dag_evidence.v1" if portfolio_route_evidence else ""
            ),
            expected_schema="selected_route_dag_evidence.v1",
            required_shapes=(
                bool(portfolio_route_evidence),
                selected_route_dags_acyclic,
            ),
        ),
        "portfolio_forest_projection": _contract_evidence(
            artifact="explored_route_forest.json:proof_eligible_portfolio_branches",
            observed_schema=portfolio_projection_evidence.get("schema_version"),
            expected_schema="portfolio_forest_projection_evidence.v1",
            required_shapes=(portfolio_routes_projected,),
        ),
        "replacement_catalog": _contract_evidence(
            artifact="route_consensus_graph_fused.json:route_replacement_catalog",
            observed_schema=replacement_catalog.get("schema_version"),
            expected_schema="route_replacement_catalog.v1",
            required_shapes=(
                replacement_evidence["catalog_integrity_valid"] is True,
                replacement_evidence["accepted_candidates_valid"] is True,
            ),
        ),
        "route_forest_dependency_graph": _contract_evidence(
            artifact="explored_route_forest.json:dependency_graph",
            observed_schema=dependency_graph.get("schema_version"),
            expected_schema="molecule_reaction_dependency_graph.v1",
            required_shapes=(
                isinstance(dependency_graph.get("molecule_nodes"), list),
                isinstance(dependency_graph.get("reaction_nodes"), list),
                isinstance(dependency_graph.get("edges"), list),
                isinstance(dependency_graph.get("branch_views"), list),
            ),
        ),
        "parent_proof_attempt": _contract_evidence(
            artifact=f"{authority_source}:parent_route_proof",
            observed_schema=proof.get("schema_version"),
            expected_schema="stitched_parent_route_proof.v1",
            required_shapes=(
                isinstance(proof.get("proof_attempt"), Mapping),
                (proof.get("proof_attempt") or {}).get("schema_version")
                == "parent_route_proof_attempt.v1",
                isinstance(proof.get("proof_clauses"), Mapping),
            ),
        ),
        "closeout_validation": _contract_evidence(
            artifact=".autoplanner/closeout/latest.json",
            observed_schema=closeout.get("schema_version"),
            expected_schema="closeout_revision_validation.v1",
            required_shapes=(
                isinstance(closeout.get("accepted"), bool),
                isinstance(closeout.get("reasons"), list),
            ),
        ),
    }
    run_gates = {
        "target_identity_accepted": preflight.get("accepted") is True,
        "root_codex_team_accepted": team.get("accepted") is True,
        "required_child_agents_completed": children_complete,
        "hypergraph_v2_valid": overlay.get("validation", {}).get("valid") is True,
        "codex_source_correlation_honest": codex_correlation_honest,
        "true_independent_multi_source_edge_present": bool(multi_source_edges),
        "frontier_ledger_authority_valid": frontier_ledger_evidence[
            "authority_valid"
        ],
        "any_route_closed": completion_truth["any_route_closed"],
        "all_explored_graph_closed": completion_truth[
            "all_explored_graph_closed"
        ],
        "proof_eligible_portfolio_valid": portfolio_contract_valid,
        "selected_route_dags_acyclic": selected_route_dags_acyclic,
        "portfolio_routes_projected": portfolio_routes_projected,
        "distinct_alternatives_at_least_2": distinct_alternative_count >= 2,
        "backend_revalidated_replacement_available": replacement_evidence[
            "backend_revalidated_replacement_available"
        ]
        is True,
        "deterministic_parent_route_solved": is_solved_parent_route_proof(
            proof,
            expected_target_smiles=target_smiles,
        ),
        "global_projection_complete": projection.get("complete") is True,
        "closeout_revision_valid": closeout.get("accepted") is True,
        "closeout_codex_campaign_proof_reconciliation_bound": bool(
            cas_decision and cas_reconciliation_bound
        ),
        "closeout_frontier_ledger_bound": bool(
            "frontier_ledger" in closeout_artifact_ids
            and cas_ledger
            and frontier_ledger_evidence["contract_valid"] is True
        ),
        "closeout_parent_proof_bound": bool(
            "parent_route_proof_snapshot" in closeout_artifact_ids
            and cas_proof_binding_valid
        ),
        "closeout_final_verdict_bound": bool(
            "final_verdict_core" in closeout_artifact_ids
            and cas_final_binding_valid
        ),
    }
    capability_passed = sum(capability_gates.values())
    executable_passed = sum(
        row.get("contract_valid") is True for row in executable_contracts.values()
    )
    run_passed = sum(run_gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(root),
        "target": {
            "name": str(target.get("target_name") or ""),
            "smiles": target_smiles,
            "inchi_key": str(preflight.get("inchi_key") or ""),
        },
        "capability_surface": {
            "available": capability_passed,
            "declared": len(capability_gates),
            "coverage_percent": _percent(capability_passed, len(capability_gates)),
            "gates": capability_gates,
            "semantics": (
                "file-presence inventory only; this is neither executable contract evidence "
                "nor an engineering-completion percentage"
            ),
        },
        "executable_contract_evidence": {
            "passed": executable_passed,
            "total": len(executable_contracts),
            "coverage_percent": _percent(executable_passed, len(executable_contracts)),
            "contracts": executable_contracts,
        },
        "run_acceptance": {
            "passed": run_passed,
            "total": len(run_gates),
            "completion_percent": _percent(run_passed, len(run_gates)),
            "gates": run_gates,
        },
        "codex_team": {
            "accepted": team.get("accepted") is True,
            "required_child_count": len(required_children),
            "observed_child_count": len(observed_children),
            "runtime_consistent": (team.get("runtime_summary") or {}).get("consistent") is True,
        },
        "hypergraph": {
            "molecule_count": len(overlay.get("molecules") or []),
            "reaction_hyperedge_count": len(hyperedges),
            "route_neighborhood_count": len(overlay.get("route_neighborhoods") or []),
            "route_variant_count": len(overlay.get("route_variants") or []),
            "alternative_set_count": len(overlay.get("alternative_sets") or []),
            "true_multi_source_hyperedge_count": len(multi_source_edges),
            "validation": dict(overlay.get("validation") or {}),
        },
        "frontier_scheduler": {
            "authority_source": scheduler_authority_source,
            "job_count": len(jobs),
            "states": dict(Counter(str(row.get("state") or "") for row in jobs)),
            "stock_closed_count": sum(
                1 for row in jobs if row.get("closure_kind") == "stock_boundary"
            ),
            "completeness": dict(
                scheduler_facts.get("frontier_completeness") or {}
            ),
            "proposal_graph_exhausted": (
                scheduler_facts.get("proposal_graph_exhausted") is True
            ),
        },
        "frontier_ledger": frontier_ledger_evidence,
        "completion_truth": completion_truth,
        "portfolio": {
            "route_count": len(portfolio_routes),
            "complete_candidate_count": int(portfolio.get("complete_candidate_count") or 0),
            "truncated": bool(portfolio.get("truncated")),
            "reasons": [str(item) for item in portfolio.get("reasons") or []],
            "schema_valid": portfolio.get("schema_version") == "route_portfolio.v1",
            "content_sha256_valid": portfolio_hash_valid,
            "bindings_schema_valid": portfolio_bindings.get("schema_version")
            == "route_portfolio_bindings.v1",
            "bindings_content_sha256_valid": portfolio_bindings_hash_valid,
            "bindings_contract_valid": portfolio_bindings_valid,
            "all_routes_complete_and_reaction_validated": portfolio_routes_valid,
            "distinct_valid_route_selection_count": distinct_alternative_count,
            "selected_route_evidence": portfolio_route_evidence,
            "forest_projection_evidence": portfolio_projection_evidence,
            "replacement_evidence": replacement_evidence,
        },
        "projection": {
            "branch_count": len(forest.get("branches") or []),
            "molecule_node_count": len(dependency_graph.get("molecule_nodes") or []),
            "reaction_node_count": len(dependency_graph.get("reaction_nodes") or []),
            "dependency_edge_count": len(dependency_graph.get("edges") or []),
            "projection_complete": projection.get("complete") is True,
            "trust_tiers": dict(sorted(trust_tiers.items())),
            "explored_overlay_acyclic": dependency_graph.get("acyclic") is True,
            "explored_overlay_cycle_graph_node_ids": [
                str(value) for value in dependency_graph.get("cycle_graph_node_ids") or []
            ],
            "explored_overlay_cycles_are_allowed": True,
            "selected_route_dags_acyclic": selected_route_dags_acyclic,
            "portfolio_routes_projected": portfolio_routes_projected,
        },
        "closeout": {
            **closeout,
            "authority_source": authority_source,
            "artifact_ids": sorted(closeout_artifact_ids),
            "frontier_ledger_bound": bool(
                "frontier_ledger" in closeout_artifact_ids
                and cas_ledger
                and frontier_ledger_evidence["contract_valid"] is True
            ),
            "proof_bound": cas_proof_binding_valid,
            "final_verdict_bound": cas_final_binding_valid,
            "cas_decision_schema": str(cas_decision.get("schema_version") or ""),
            "cas_load_errors": cas_load_errors,
            "codex_campaign_proof_reconciliation_bound": (
                cas_reconciliation_bound
            ),
            "codex_campaign_proof_reconciliation_blockers": sorted(
                set(cas_reconciliation_authority_blockers)
            ),
        },
        "compatibility_projection": compatibility_projection,
        "final_verdict": {
            "verdict": str(final.get("verdict") or ""),
            "route_status": str(final.get("route_status") or ""),
            "solved": final.get("solved") is True,
            "reasons": [str(item) for item in final.get("reasons") or []],
        },
        "remaining_gaps": [name for name, accepted in run_gates.items() if not accepted],
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _verdict_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compare decision meaning without path/digest presentation fields."""
    return {
        "verdict": str(value.get("verdict") or ""),
        "route_status": str(value.get("route_status") or ""),
        "solved": value.get("solved") is True,
        "stock_audit_passed": value.get("stock_audit_passed") is True,
        "reasons": sorted(str(item) for item in value.get("reasons") or []),
    }


def _content_sha256_valid(value: Mapping[str, Any]) -> bool | None:
    """Validate an embedded canonical-JSON digest when the schema provides one."""
    expected = str(value.get("content_sha256") or "").strip().lower()
    if not expected:
        return None
    payload = dict(value)
    payload.pop("content_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return expected == hashlib.sha256(encoded).hexdigest()


def _audit_campaign_policy(
    root: Path,
    *,
    campaign: Mapping[str, Any],
    team: Mapping[str, Any],
) -> dict[str, Any]:
    for value in (campaign.get("campaign_policy"), team.get("campaign_policy")):
        if isinstance(value, Mapping) and value:
            return dict(value)
    for value in (
        campaign.get("campaign_policy_ref"),
        team.get("campaign_policy_ref"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        raw = Path(text).expanduser()
        candidates = [raw, root / raw, root / "codex_retrosynthesis_team" / raw.name]
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            policy = _load(resolved)
            if policy:
                return policy
    return {}


def _audit_trusted_stock_providers(
    root: Path,
    *,
    campaign_policy: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """Rebuild only host-owned provider state materialized by the policy."""

    policy = dict(campaign_policy or {})
    binding = dict(policy.get("stock_authority_binding") or {})
    descriptor = dict(binding.get("provider_descriptor") or {})
    provider_id = str(descriptor.get("provider_id") or "")
    if not provider_id:
        return None, ("campaign_policy_stock_provider_descriptor_missing",)
    if provider_id == "autoplanner.snapshot_stock":
        snapshot_hashes = [
            str(item)
            for item in binding.get("trusted_snapshot_sha256") or []
            if str(item)
        ]
        if snapshot_hashes:
            # Hashes identify snapshots but cannot reconstruct their contents.
            # Serialized queue envelopes are not a substitute for host trust.
            return None, ("current_host_stock_snapshot_material_missing",)
        provider = SnapshotStockProvider(trusted_snapshots=[])
        return {provider.descriptor.provider_id: provider}, ()
    if provider_id != "autoplanner.benchmark_catalog_stock":
        return None, ("campaign_policy_stock_provider_not_allowlisted",)

    artifact_text = str(binding.get("catalog_artifact") or "").strip()
    if not artifact_text:
        return None, ("campaign_policy_benchmark_catalog_artifact_missing",)
    raw_artifact = Path(artifact_text).expanduser()
    artifact = next(
        (
            candidate.resolve()
            for candidate in (raw_artifact, root / raw_artifact, ROOT / raw_artifact)
            if candidate.is_file()
        ),
        raw_artifact,
    )
    providers, reasons = build_trusted_stock_provider_instances(
        benchmark_catalog_artifact=artifact,
        benchmark_catalog_sha256=str(binding.get("catalog_sha256") or ""),
        benchmark_catalog_name=str(binding.get("catalog_name") or ""),
    )
    if reasons or provider_id not in providers:
        return None, tuple(
            sorted(
                set(
                    [
                        *reasons,
                        *( ["current_host_benchmark_provider_missing"] if provider_id not in providers else [] ),
                    ]
                )
            )
        )
    return providers, ()


def _frontier_ledger_evidence(
    value: Mapping[str, Any],
    *,
    expected_target_smiles: str,
    route_consensus_graph: Mapping[str, Any] | None = None,
    frontier_queue: Mapping[str, Any] | None = None,
    campaign_policy: Mapping[str, Any] | None = None,
    campaign_policy_sha256: str = "",
    trusted_stock_provider_instances: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit one serialized ledger without trusting its positive summary.

    A digest-valid ledger can still be an honest negative projection of
    invalid upstream authority.  ``contract_valid`` therefore covers schema,
    digest, shape, semantic consistency, and producer fail-closed behavior,
    while ``authority_valid`` additionally requires every joined input to be
    valid.  Effective closure claims require both.
    """

    row = dict(value) if isinstance(value, Mapping) else {}
    present = bool(row)
    if present:
        try:
            validator_reasons = validate_frontier_ledger(row)
        except (TypeError, ValueError, OverflowError) as exc:
            validator_reasons = [
                f"frontier_ledger_validator_exception:{type(exc).__name__}"
            ]
    else:
        validator_reasons = ["frontier_ledger_missing"]
    reasons = list(validator_reasons)
    schema_valid = row.get("schema_version") == FRONTIER_LEDGER_SCHEMA
    try:
        digest_valid = _content_sha256_valid(row) is True
    except (TypeError, ValueError, OverflowError):
        digest_valid = False
    summary = (
        dict(row["summary"])
        if isinstance(row.get("summary"), Mapping)
        else {}
    )
    root = dict(row["root"]) if isinstance(row.get("root"), Mapping) else {}
    molecules = (
        dict(row["molecules"])
        if isinstance(row.get("molecules"), Mapping)
        else {}
    )
    root_smiles = str(root.get("canonical_smiles") or "")

    claimed_any_raw = summary.get("any_route_closed")
    claimed_all_raw = summary.get("all_explored_graph_closed")
    claimed_any = claimed_any_raw is True
    claimed_all = claimed_all_raw is True
    if not isinstance(claimed_any_raw, bool):
        reasons.append("frontier_ledger_any_route_closed_not_boolean")
    if not isinstance(claimed_all_raw, bool):
        reasons.append("frontier_ledger_all_explored_graph_closed_not_boolean")
    if claimed_all and not claimed_any:
        reasons.append("frontier_ledger_all_closed_without_any_closed")

    count_fields = (
        "reachable_molecule_count",
        "reachable_edge_count",
        "reaction_proven_edge_count",
        "stock_closed_molecule_count",
        "proposal_pending_molecule_count",
        "proposal_expansion_eligible_molecule_count",
        "work_pending_molecule_count",
        "stock_pending_leaf_count",
        "reaction_proof_pending_edge_count",
        "dependency_pending_edge_count",
        "open_work_molecule_count",
        "fixed_point_iterations",
    )
    summary_counts: dict[str, int] = {}
    for field in count_fields:
        raw_count = summary.get(field)
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
        ):
            reasons.append(f"frontier_ledger_summary_count_invalid:{field}")
            continue
        summary_counts[field] = raw_count

    if summary_counts.get("reachable_molecule_count") != len(molecules):
        reasons.append("frontier_ledger_reachable_molecule_count_mismatch")
    raw_edges = row.get("edges")
    edge_count = len(raw_edges) if isinstance(raw_edges, Mapping) else -1
    if summary_counts.get("reachable_edge_count") != edge_count:
        reasons.append("frontier_ledger_reachable_edge_count_mismatch")
    count_limits = (
        ("reaction_proven_edge_count", "reachable_edge_count"),
        ("stock_closed_molecule_count", "reachable_molecule_count"),
        ("proposal_pending_molecule_count", "reachable_molecule_count"),
        (
            "proposal_expansion_eligible_molecule_count",
            "proposal_pending_molecule_count",
        ),
        ("work_pending_molecule_count", "reachable_molecule_count"),
        ("stock_pending_leaf_count", "reachable_molecule_count"),
        ("reaction_proof_pending_edge_count", "reachable_edge_count"),
        ("dependency_pending_edge_count", "reachable_edge_count"),
        ("open_work_molecule_count", "reachable_molecule_count"),
    )
    for field, limit_field in count_limits:
        if (
            field in summary_counts
            and limit_field in summary_counts
            and summary_counts[field] > summary_counts[limit_field]
        ):
            reasons.append(f"frontier_ledger_summary_count_exceeds_limit:{field}")
    if summary_counts.get("fixed_point_iterations", 0) < 1:
        reasons.append("frontier_ledger_fixed_point_iterations_invalid")
    if claimed_all and any(
        summary_counts.get(field, -1) != 0
        for field in (
            "proposal_pending_molecule_count",
            "stock_pending_leaf_count",
            "reaction_proof_pending_edge_count",
            "dependency_pending_edge_count",
        )
    ):
        reasons.append("frontier_ledger_all_closed_with_pending_graph_facts")

    raw_root_molecule = molecules.get(root_smiles)
    root_molecule = (
        dict(raw_root_molecule) if isinstance(raw_root_molecule, Mapping) else {}
    )
    root_closure = (
        dict(root_molecule["closure"])
        if isinstance(root_molecule.get("closure"), Mapping)
        else {}
    )
    root_closure_matches = bool(
        root_smiles
        and root_molecule
        and root_closure.get("any_route_closed") is claimed_any_raw
        and root_closure.get("all_explored_graph_closed") is claimed_all_raw
    )
    if not root_closure_matches:
        reasons.append("frontier_ledger_root_summary_mismatch")
    target_matches = bool(
        root_smiles
        and (
            not expected_target_smiles
            or root_smiles == str(expected_target_smiles)
        )
    )
    if not target_matches:
        reasons.append("frontier_ledger_target_mismatch")

    try:
        required_level = int(row.get("required_reaction_proof_level"))
    except (TypeError, ValueError):
        required_level = 0
    if required_level not in {2, 3, 4}:
        reasons.append("frontier_ledger_required_proof_level_invalid")

    input_validation = (
        dict(row["input_validation"])
        if isinstance(row.get("input_validation"), Mapping)
        else {}
    )
    input_names = ("graph", "frontier_queue", "reaction_proof_state")
    input_status = {
        name: bool(
            isinstance(input_validation.get(name), Mapping)
            and input_validation[name].get("valid") is True
        )
        for name in input_names
    }
    inputs_valid = all(input_status.values())
    invalid_inputs = [name for name, accepted in input_status.items() if not accepted]

    semantics = (
        dict(row["semantics"])
        if isinstance(row.get("semantics"), Mapping)
        else {}
    )
    semantics_declared = bool(
        semantics.get(
            "proposal_work_stock_reaction_proof_and_dependencies_are_orthogonal"
        )
        is True
        and semantics.get("invalid_authority_fails_closed") is True
        and semantics.get("route_hypotheses_are_not_consumed") is True
    )
    if not semantics_declared:
        reasons.append("frontier_ledger_semantics_incomplete")

    producer_fail_closed = bool(
        present
        and schema_valid
        and digest_valid
        and not (not inputs_valid and (claimed_any or claimed_all))
    )
    if not producer_fail_closed:
        reasons.append("frontier_ledger_invalid_authority_not_failed_closed")

    contract_reasons = sorted(set(reasons))
    contract_valid = not contract_reasons
    current_authority = frontier_ledger_current_authority_evidence(
        row,
        route_consensus_graph=route_consensus_graph,
        frontier_queue=frontier_queue,
        campaign_policy=campaign_policy,
        campaign_policy_sha256=campaign_policy_sha256,
        trusted_stock_provider_instances=trusted_stock_provider_instances,
    )
    authority_blockers = list(current_authority.get("blockers") or [])
    authority_valid = bool(
        contract_valid
        and inputs_valid
        and current_authority.get("authoritative") is True
    )
    return {
        "schema_version": "frontier_ledger_audit_evidence.v1",
        "artifact": "frontier_ledger.json",
        "present": present,
        "observed_schema": str(row.get("schema_version") or ""),
        "schema_valid": schema_valid,
        "content_sha256_valid": digest_valid,
        "validator_reasons": sorted(set(validator_reasons)),
        "root_canonical_smiles": root_smiles,
        "target_matches": target_matches,
        "root_summary_matches": root_closure_matches,
        "required_reaction_proof_level": required_level,
        "input_authority": input_status,
        "invalid_inputs": invalid_inputs,
        "all_input_authority_valid": inputs_valid,
        "historical_host_replay_recorded": current_authority.get(
            "historical_host_replay_recorded"
        )
        is True,
        "current_host_replay_verified": current_authority.get(
            "current_host_replay_verified"
        )
        is True,
        "current_input_bindings_verified": current_authority.get(
            "current_input_bindings_verified"
        )
        is True,
        "current_identities": dict(current_authority.get("current_identities") or {}),
        "input_bindings": dict(current_authority.get("input_bindings") or {}),
        "authority_blockers": authority_blockers,
        "producer_fail_closed": producer_fail_closed,
        "semantics_declared": semantics_declared,
        "claimed_any_route_closed": claimed_any,
        "claimed_all_explored_graph_closed": claimed_all,
        "summary_counts": summary_counts,
        "effective_any_route_closed": bool(authority_valid and claimed_any),
        "effective_all_explored_graph_closed": bool(
            authority_valid and claimed_all
        ),
        "contract_valid": contract_valid,
        "authority_valid": authority_valid,
        "reasons": sorted(set([*contract_reasons, *authority_blockers])),
        "contract_reasons": contract_reasons,
    }


def _contract_evidence(
    *,
    artifact: str,
    observed_schema: Any,
    expected_schema: str,
    required_shapes: tuple[bool, ...],
    allow_empty_collection: bool = False,
) -> dict[str, Any]:
    schema = str(observed_schema or "")
    shape_valid = all(value is True for value in required_shapes)
    reasons = []
    if schema != expected_schema:
        reasons.append("schema_mismatch_or_missing")
    if not shape_valid:
        reasons.append("required_contract_fields_invalid")
    return {
        "schema_version": "architecture_executable_contract_evidence.v1",
        "artifact": artifact,
        "observed_schema": schema,
        "expected_schema": expected_schema,
        "materialized": bool(schema),
        "shape_valid": shape_valid,
        "contract_valid": bool(schema == expected_schema and shape_valid),
        "empty_collection_can_still_prove_execution": bool(allow_empty_collection),
        "reasons": reasons,
    }


def _portfolio_route_evidence(
    route: Mapping[str, Any],
    *,
    overlay: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    edge_by_id = {
        str(row.get("hyperedge_id") or ""): dict(row)
        for row in overlay.get("reaction_hyperedges") or []
        if isinstance(row, Mapping) and str(row.get("hyperedge_id") or "")
    }
    selections = [
        dict(row)
        for row in route.get("selected_hyperedges") or []
        if isinstance(row, Mapping)
    ]
    reasons: list[str] = []
    products: set[str] = set()
    graph: dict[str, set[str]] = {}
    nodes: set[str] = set()
    edge_binding_evidence: list[dict[str, Any]] = []
    exact_edge_bindings = dict(bindings.get("exact_edge_proof_bindings") or {})
    edge_levels = dict(bindings.get("edge_proof_levels") or {})
    molecule_smiles = {
        str(row.get("molecule_id") or ""): str(
            row.get("canonical_isomeric_smiles") or ""
        )
        for row in overlay.get("molecules") or []
        if isinstance(row, Mapping) and str(row.get("molecule_id") or "")
    }
    for selection in selections:
        product_id = str(selection.get("product_molecule_id") or "")
        edge_id = str(selection.get("hyperedge_id") or "")
        edge = edge_by_id.get(edge_id)
        if not product_id or not edge_id:
            reasons.append("selected_hyperedge_identity_missing")
            continue
        if product_id in products:
            reasons.append(f"duplicate_product_selection:{product_id}")
        products.add(product_id)
        if edge is None:
            reasons.append(f"selected_hyperedge_missing_from_overlay:{edge_id}")
            continue
        if str(edge.get("product_molecule_id") or "") != product_id:
            reasons.append(f"selected_hyperedge_product_mismatch:{edge_id}")
            continue
        binding_evidence = _exact_edge_binding_evidence(
            edge,
            binding=exact_edge_bindings.get(edge_id),
            declared_level=edge_levels.get(edge_id),
            molecule_smiles=molecule_smiles,
        )
        edge_binding_evidence.append(binding_evidence)
        reasons.extend(
            f"edge_binding:{edge_id}:{reason}"
            for reason in binding_evidence["reasons"]
        )
        nodes.add(product_id)
        for precursor in edge.get("precursor_molecule_ids") or []:
            precursor_id = str(precursor or "")
            if not precursor_id:
                reasons.append(f"selected_hyperedge_precursor_missing:{edge_id}")
                continue
            nodes.add(precursor_id)
            graph.setdefault(precursor_id, set()).add(product_id)
            graph.setdefault(product_id, set())
    if not selections:
        reasons.append("selected_hyperedges_empty")
    route_id = str(route.get("route_id") or "")
    if not route_id:
        reasons.append("route_id_missing")
    selected_edge_ids = [str(row.get("hyperedge_id") or "") for row in selections]
    if list(route.get("hyperedge_ids") or []) != selected_edge_ids:
        reasons.append("route_hyperedge_ids_mismatch")
    selected_products = set(products)
    computed_leaves = sorted(nodes - selected_products)
    declared_stock = sorted(
        {str(value) for value in route.get("stock_terminal_ids") or [] if str(value)}
    )
    if computed_leaves != declared_stock:
        reasons.append("selected_route_stock_leaves_mismatch")
    if sorted(str(value) for value in route.get("molecule_ids") or []) != sorted(nodes):
        reasons.append("selected_route_molecule_ids_mismatch")
    stock_binding_evidence = [
        _stock_binding_evidence(
            molecule_id,
            binding=(bindings.get("stock_bindings") or {}).get(molecule_id),
        )
        for molecule_id in declared_stock
    ]
    reasons.extend(
        f"stock_binding:{row['molecule_id']}:{reason}"
        for row in stock_binding_evidence
        for reason in row["reasons"]
    )
    target_nodes = sorted(node for node in nodes if not graph.get(node))
    root_id = str(route.get("root_molecule_id") or "")
    unique_target = bool(target_nodes == [root_id] and root_id == overlay.get("root_molecule_id"))
    if not unique_target:
        reasons.append("selected_route_unique_target_invalid")
    dag_acyclic = _directed_graph_acyclic(nodes, graph)
    if not dag_acyclic:
        reasons.append("selected_route_cycle_detected")
    route_hash_valid = _content_sha256_valid(route)
    if route_hash_valid is not True:
        reasons.append("route_content_sha256_mismatch")
    if route.get("complete") is not True:
        reasons.append("route_not_complete")
    if route.get("reaction_validated") is not True:
        reasons.append("route_not_reaction_validated")
    if str(route.get("schema_version") or "") != "route_portfolio_item.v1":
        reasons.append("route_item_schema_invalid")
    selection_valid = not any(
        reason.startswith(
            (
                "selected_hyperedge_",
                "duplicate_product_selection:",
            )
        )
        for reason in reasons
    )
    bindings_hash_valid = _content_sha256_valid(bindings)
    if (
        bindings.get("schema_version") != "route_portfolio_bindings.v1"
        or bindings_hash_valid is not True
    ):
        reasons.append("route_portfolio_bindings_invalid")
    exact_bindings_valid = bool(edge_binding_evidence) and all(
        row["valid"] is True for row in edge_binding_evidence
    )
    selected_levels = [
        int(row.get("portfolio_proof_level") or -1) for row in edge_binding_evidence
    ]
    weakest_level = min(selected_levels, default=-1)
    if _safe_int(route.get("weakest_proof_level")) != weakest_level:
        reasons.append("route_weakest_proof_level_mismatch")
    stock_bindings_valid = bool(declared_stock) and all(
        row["valid"] is True for row in stock_binding_evidence
    )
    contract_valid = bool(
        route.get("complete") is True
        and bool(route_id)
        and route.get("reaction_validated") is True
        and str(route.get("schema_version") or "") == "route_portfolio_item.v1"
        and route_hash_valid is True
        and bindings_hash_valid is True
        and selection_valid
        and dag_acyclic
        and unique_target
        and computed_leaves == declared_stock
        and exact_bindings_valid
        and stock_bindings_valid
    )
    return {
        "schema_version": "selected_route_dag_evidence.v1",
        "route_id": route_id,
        "complete": route.get("complete") is True,
        "reaction_validated": route.get("reaction_validated") is True,
        "content_sha256_valid": route_hash_valid,
        "selection_valid": selection_valid,
        "bindings_content_sha256_valid": bindings_hash_valid,
        "exact_edge_bindings_valid": exact_bindings_valid,
        "stock_bindings_valid": stock_bindings_valid,
        "edge_binding_evidence": edge_binding_evidence,
        "recomputed_weakest_proof_level": weakest_level,
        "stock_binding_evidence": stock_binding_evidence,
        "selected_hyperedge_count": len(selections),
        "molecule_node_count": len(nodes),
        "computed_stock_leaf_molecule_ids": computed_leaves,
        "declared_stock_terminal_ids": declared_stock,
        "target_molecule_ids": target_nodes,
        "unique_target": unique_target,
        "dag_acyclic": dag_acyclic,
        "contract_valid": contract_valid,
        "reasons": sorted(set(reasons)),
    }


_PROOF_LEVELS = {
    "L0_materialized": 0,
    "L1_graph_and_stock_closed": 1,
    "L2_mapping_consistent": 0,
    "L2_reaction_validated": 2,
    "L3_precedent_supported": 3,
    "L4_procurement_ready": 4,
}


def _exact_edge_binding_evidence(
    edge: Mapping[str, Any],
    *,
    binding: Any,
    declared_level: Any,
    molecule_smiles: Mapping[str, str],
) -> dict[str, Any]:
    edge_id = str(edge.get("hyperedge_id") or "")
    reasons: list[str] = []
    if not isinstance(binding, Mapping):
        return {
            "schema_version": "exact_edge_binding_audit.v1",
            "hyperedge_id": edge_id,
            "valid": False,
            "reasons": ["exact_edge_binding_missing"],
        }
    row = dict(binding)
    if row.get("schema_version") != "exact_edge_proof_binding.v1":
        reasons.append("exact_edge_binding_schema_invalid")
    if _embedded_digest_valid(row, "binding_sha256") is not True:
        reasons.append("exact_edge_binding_sha256_invalid")
    product_id = str(edge.get("product_molecule_id") or "")
    precursor_ids = sorted(str(value) for value in edge.get("precursor_molecule_ids") or [])
    if (
        row.get("hyperedge_id") != edge_id
        or row.get("product_molecule_id") != product_id
        or sorted(str(value) for value in row.get("precursor_molecule_ids") or [])
        != precursor_ids
    ):
        reasons.append("exact_edge_binding_identity_mismatch")
    product_smiles = str(molecule_smiles.get(product_id) or "")
    reactant_smiles = sorted(str(molecule_smiles.get(value) or "") for value in precursor_ids)
    signature_digest = _canonical_digest(
        {
            "product_canonical_isomeric_smiles": product_smiles,
            "reactant_canonical_isomeric_smiles": reactant_smiles,
        }
    )
    if (
        not product_smiles
        or any(not value for value in reactant_smiles)
        or row.get("structure_signature_sha256") != signature_digest
        or row.get("reaction_digest") != signature_digest
    ):
        reasons.append("exact_edge_binding_structure_digest_mismatch")
    named_level = str(row.get("proof_level") or "")
    expected_level = _PROOF_LEVELS.get(named_level, -1)
    try:
        bound_level = int(row.get("portfolio_proof_level"))
        selected_level = int(declared_level)
    except (TypeError, ValueError):
        bound_level = -1
        selected_level = -1
    if expected_level != bound_level or bound_level != selected_level:
        reasons.append("proof_level_portfolio_level_mismatch")
    if (
        bound_level < 2
        or row.get("proof_accepted") is not True
        or row.get("advisory") is not False
    ):
        reasons.append("exact_edge_binding_not_proof_eligible")
    source = str(row.get("proof_source") or "")
    if source not in {
        "route_proof_bank.v1",
        "legacy_best_accepted_route",
        "deterministic_transform_reapply.v1",
    }:
        reasons.append("exact_edge_binding_source_untrusted")
    if named_level == "L2_reaction_validated" and (
        source != "deterministic_transform_reapply.v1"
        or not _is_sha256(row.get("trusted_transform_sha256"))
    ):
        # Mapping-only evidence can be relabelled and re-hashed by an attacker.
        # L2 therefore needs a separate deterministic transform-reapply
        # commitment; current route-bank precedent bindings legitimately use
        # L3/L4 instead.
        reasons.append("exact_edge_binding_l2_transform_authority_invalid")
    if source == "route_proof_bank.v1" and (
        not str(row.get("proof_bank_entry_id") or "")
        or not _is_sha256(row.get("proof_bank_entry_sha256"))
    ):
        reasons.append("exact_edge_binding_proof_bank_authority_invalid")
    required_digests = ("proof_digest", "route_proof_digest", "reaction_digest")
    if any(not _is_sha256(row.get(field)) for field in required_digests):
        reasons.append("exact_edge_binding_proof_digest_invalid")
    if named_level in {"L3_precedent_supported", "L4_procurement_ready"} and not _is_sha256(
        row.get("trusted_precedent_sha256")
    ):
        reasons.append("exact_edge_binding_precedent_digest_invalid")
    if not str(row.get("validator_version") or ""):
        reasons.append("exact_edge_binding_validator_missing")
    return {
        "schema_version": "exact_edge_binding_audit.v1",
        "hyperedge_id": edge_id,
        "proof_level": named_level,
        "portfolio_proof_level": bound_level,
        "declared_edge_proof_level": selected_level,
        "proof_source": source,
        "binding_sha256_valid": _embedded_digest_valid(row, "binding_sha256"),
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
    }


def _stock_binding_evidence(
    molecule_id: str,
    *,
    binding: Any,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(binding, Mapping):
        return {
            "schema_version": "exact_stock_binding_audit.v1",
            "molecule_id": molecule_id,
            "boundary_type": "",
            "benchmark_only": False,
            "procurement_boundary_valid": False,
            "valid": False,
            "reasons": ["exact_stock_binding_missing"],
        }
    row = dict(binding)
    if row.get("schema_version") != "exact_stock_binding.v1":
        reasons.append("exact_stock_binding_schema_invalid")
    if row.get("molecule_id") != molecule_id:
        reasons.append("exact_stock_binding_molecule_mismatch")
    if _embedded_digest_valid(row, "binding_sha256") is not True:
        reasons.append("exact_stock_binding_sha256_invalid")
    if any(
        not _is_sha256(row.get(field))
        for field in (
            "catalog_sha256",
            "stock_audit_sha256",
            "evidence_sha256",
        )
    ):
        reasons.append("exact_stock_binding_digest_invalid")
    if not all(
        str(row.get(field) or "")
        for field in (
            "canonical_isomeric_smiles",
            "catalog_id",
            "lookup_basis",
        )
    ):
        reasons.append("exact_stock_binding_identity_incomplete")
    authority = str(row.get("binding_authority") or "")
    if authority not in {
        "strictly_replayed_route_proof_bank.v1",
        "legacy_best_route_independent_stock_audit",
    }:
        reasons.append("exact_stock_binding_authority_untrusted")
    if authority == "strictly_replayed_route_proof_bank.v1":
        proof_authorities = row.get("proof_bank_authorities")
        if not isinstance(proof_authorities, list) or not proof_authorities:
            reasons.append("exact_stock_binding_proof_bank_authority_missing")
        elif any(
            not isinstance(item, Mapping)
            or not str(item.get("proof_bank_entry_id") or "")
            or not _is_sha256(item.get("proof_bank_entry_sha256"))
            or not _is_sha256(item.get("stock_evidence_binding_sha256"))
            for item in proof_authorities
        ):
            reasons.append("exact_stock_binding_proof_bank_authority_invalid")
    boundary_type = str(row.get("boundary_type") or "")
    benchmark_only = bool(
        boundary_type == "benchmark_stock"
        or row.get("benchmark_membership") is True
    )
    procurement_boundary_valid = bool(
        not reasons
        and not benchmark_only
        and boundary_type
        in {
            "commercially_orderable",
            "in_house_available",
            "common_commodity",
        }
        and (
            boundary_type != "commercially_orderable"
            or (
                row.get("commercial_orderability_claimed") is True
                and row.get("snapshot_digest_replayed") is True
                and row.get("provider_trust_authority")
                == "autoplanner_host_builtin_allowlist.v1"
                and _is_sha256(row.get("provider_descriptor_sha256"))
            )
        )
    )
    return {
        "schema_version": "exact_stock_binding_audit.v1",
        "molecule_id": molecule_id,
        "boundary_type": boundary_type,
        "benchmark_only": benchmark_only,
        "procurement_boundary_valid": procurement_boundary_valid,
        "binding_authority": authority,
        "binding_sha256_valid": _embedded_digest_valid(row, "binding_sha256"),
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
    }


def _portfolio_projection_evidence(
    portfolio_routes: list[dict[str, Any]],
    *,
    forest_branches: list[dict[str, Any]],
    branch_views: list[dict[str, Any]],
) -> dict[str, Any]:
    route_ids = [str(route.get("route_id") or "") for route in portfolio_routes]
    branches = [
        row
        for row in forest_branches
        if row.get("kind") == "proof_eligible_portfolio_route"
        and row.get("listed") is True
        and row.get("proof_eligible") is True
    ]
    branch_by_route: dict[str, list[dict[str, Any]]] = {}
    for branch in branches:
        branch_by_route.setdefault(str(branch.get("portfolio_route_id") or ""), []).append(branch)
    view_by_branch: dict[str, list[dict[str, Any]]] = {}
    for view in branch_views:
        view_by_branch.setdefault(str(view.get("branch_id") or ""), []).append(view)
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for route_id in route_ids:
        route_branches = branch_by_route.get(route_id, [])
        valid = len(route_branches) == 1
        row_reasons: list[str] = []
        if len(route_branches) != 1:
            row_reasons.append("portfolio_route_branch_not_one_to_one")
            view = {}
            branch = {}
        else:
            branch = route_branches[0]
            views = view_by_branch.get(str(branch.get("branch_id") or ""), [])
            if len(views) != 1:
                row_reasons.append("portfolio_route_branch_view_not_one_to_one")
                view = {}
            else:
                view = views[0]
        if view and (
            view.get("acyclic") is not True
            or view.get("all_leaves_stock_bound") is not True
            or len(view.get("target_molecule_node_ids") or []) != 1
            or (branch.get("root_molecule_node_id") and view.get("target_molecule_node_ids") != [branch.get("root_molecule_node_id")])
            or (
                branch.get("stock_terminal_node_ids") is not None
                and sorted(str(value) for value in view.get("stock_leaf_molecule_node_ids") or [])
                != sorted(str(value) for value in branch.get("stock_terminal_node_ids") or [])
            )
        ):
            row_reasons.append("portfolio_route_branch_view_closure_invalid")
        valid = bool(valid and view and not row_reasons)
        reasons.extend(f"route:{route_id}:{reason}" for reason in row_reasons)
        rows.append(
            {
                "route_id": route_id,
                "branch_id": str(branch.get("branch_id") or ""),
                "acyclic": view.get("acyclic") is True,
                "all_leaves_stock_bound": view.get("all_leaves_stock_bound") is True,
                "target_molecule_node_ids": list(view.get("target_molecule_node_ids") or []),
                "valid": valid,
                "reasons": row_reasons,
            }
        )
    extra_routes = sorted(set(branch_by_route) - set(route_ids) - {""})
    if extra_routes:
        reasons.append("forest_contains_extra_listed_portfolio_routes")
    if not route_ids:
        reasons.append("no_proof_eligible_portfolio_routes")
    contract_valid = bool(
        route_ids
        and len(route_ids) == len(set(route_ids))
        and len(branches) == len(route_ids)
        and all(row["valid"] for row in rows)
        and not extra_routes
    )
    if len(route_ids) != len(set(route_ids)):
        reasons.append("portfolio_route_ids_not_unique")
    return {
        "schema_version": "portfolio_forest_projection_evidence.v1",
        "portfolio_route_ids": route_ids,
        "listed_proof_eligible_branch_count": len(branches),
        "projected_route_count": sum(row["valid"] for row in rows),
        "routes": rows,
        "contract_valid": contract_valid,
        "reasons": sorted(set(reasons)),
    }


def _replacement_catalog_evidence(
    catalog: Mapping[str, Any],
    *,
    portfolio: Mapping[str, Any],
    portfolio_routes: list[dict[str, Any]],
    overlay: Mapping[str, Any],
    bindings: Mapping[str, Any],
    forest_branches: list[dict[str, Any]],
    branch_views: list[dict[str, Any]],
    replacement_records: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog_hash_valid = _content_sha256_valid(catalog)
    portfolio_hash = str(portfolio.get("content_sha256") or "")
    catalog_candidates = [
        dict(row) for row in catalog.get("candidates") or [] if isinstance(row, Mapping)
    ]
    catalog_counts_valid = bool(
        _safe_int(catalog.get("candidate_count")) == len(catalog_candidates)
        and _safe_int(catalog.get("accepted_candidate_count"))
        == sum(row.get("accepted") is True for row in catalog_candidates)
        and _safe_int(catalog.get("rejected_candidate_count"))
        == sum(row.get("accepted") is not True for row in catalog_candidates)
    )
    catalog_integrity_valid = bool(
        catalog.get("schema_version") == "route_replacement_catalog.v1"
        and catalog_hash_valid is True
        and _is_sha256(portfolio_hash)
        and catalog.get("portfolio_content_sha256") == portfolio_hash
        and catalog.get("portfolio_integrity_valid") is True
        and catalog_counts_valid
    )
    base_routes = {str(row.get("route_id") or ""): row for row in portfolio_routes}
    branch_by_id = {
        str(row.get("branch_id") or ""): row
        for row in forest_branches
        if str(row.get("branch_id") or "")
    }
    view_by_id = {
        str(row.get("branch_id") or ""): row
        for row in branch_views
        if str(row.get("branch_id") or "")
    }
    record_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for record in replacement_records:
        for identity in {
            str(record.get("candidate_id") or ""),
            str(record.get("replacement_id") or ""),
        }:
            if identity:
                record_by_candidate.setdefault(identity, []).append(record)
    candidate_evidence: list[dict[str, Any]] = []
    accepted_rows = [
        dict(row)
        for row in catalog_candidates
        if row.get("accepted") is True
    ]
    for candidate in accepted_rows:
        candidate_id = str(candidate.get("candidate_id") or "")
        reasons: list[str] = []
        triple_revalidated = all(
            candidate.get(field) is True
            for field in (
                "connectivity_revalidated",
                "stock_closure_revalidated",
                "reaction_proof_revalidated",
            )
        )
        if not triple_revalidated:
            reasons.append("replacement_candidate_not_triple_revalidated")
        base = base_routes.get(str(candidate.get("base_route_id") or ""))
        result_route = candidate.get("route")
        if base is None or not isinstance(result_route, Mapping):
            reasons.append("replacement_candidate_route_identity_missing")
            route_evidence = {"contract_valid": False, "reasons": ["route_missing"]}
        else:
            base_selections = {
                str(row.get("product_molecule_id") or ""): str(row.get("hyperedge_id") or "")
                for row in base.get("selected_hyperedges") or []
                if isinstance(row, Mapping)
            }
            result_selections = {
                str(row.get("product_molecule_id") or ""): str(row.get("hyperedge_id") or "")
                for row in result_route.get("selected_hyperedges") or []
                if isinstance(row, Mapping)
            }
            product_id = str(candidate.get("product_molecule_id") or "")
            if (
                base_selections.get(product_id)
                != str(candidate.get("original_hyperedge_id") or "")
                or result_selections.get(product_id)
                != str(candidate.get("replacement_hyperedge_id") or "")
                or result_selections.get(product_id) == base_selections.get(product_id)
            ):
                reasons.append("replacement_candidate_edge_substitution_invalid")
            route_evidence = _portfolio_route_evidence(
                result_route,
                overlay=overlay,
                bindings=bindings,
            )
            if route_evidence["contract_valid"] is not True:
                reasons.append("replacement_candidate_route_contract_invalid")
        records = record_by_candidate.get(candidate_id, [])
        if len(records) != 1:
            reasons.append("replacement_candidate_forest_record_not_one_to_one")
            record = {}
        else:
            record = records[0]
        branch_id = str(record.get("revalidated_route_branch_id") or "")
        branch = branch_by_id.get(branch_id, {})
        view = view_by_id.get(branch_id, {})
        projected = bool(
            record.get("validated") is True
            and record.get("accepted") is True
            and branch
            and branch.get("portfolio_route_id")
            == (result_route.get("route_id") if isinstance(result_route, Mapping) else None)
            and branch.get("kind")
            in {"validated_replacement_route", "proof_eligible_portfolio_route"}
            and view.get("acyclic") is True
            and view.get("all_leaves_stock_bound") is True
            and len(view.get("target_molecule_node_ids") or []) == 1
        )
        if not projected:
            reasons.append("replacement_candidate_forest_projection_invalid")
        candidate_evidence.append(
            {
                "candidate_id": candidate_id,
                "base_route_id": str(candidate.get("base_route_id") or ""),
                "result_route_id": str(
                    result_route.get("route_id") if isinstance(result_route, Mapping) else ""
                ),
                "triple_revalidated": triple_revalidated,
                "route_contract_valid": route_evidence["contract_valid"] is True,
                "forest_projected": projected,
                "valid": not reasons,
                "reasons": sorted(set(reasons)),
            }
        )
    accepted_candidates_valid = bool(accepted_rows) and all(
        row["valid"] for row in candidate_evidence
    )
    return {
        "schema_version": "replacement_catalog_audit_evidence.v1",
        "catalog_present": bool(catalog),
        "catalog_content_sha256_valid": catalog_hash_valid,
        "portfolio_hash_matches": bool(
            portfolio_hash and catalog.get("portfolio_content_sha256") == portfolio_hash
        ),
        "catalog_integrity_valid": catalog_integrity_valid,
        "catalog_counts_valid": catalog_counts_valid,
        "accepted_candidate_count": len(accepted_rows),
        "accepted_candidates_valid": accepted_candidates_valid,
        "candidate_evidence": candidate_evidence,
        "backend_revalidated_replacement_available": bool(
            catalog_integrity_valid and accepted_candidates_valid
        ),
        "reasons": (
            []
            if catalog_integrity_valid and accepted_candidates_valid
            else [
                *([] if catalog_integrity_valid else ["replacement_catalog_integrity_invalid"]),
                *(
                    []
                    if accepted_candidates_valid
                    else ["no_valid_backend_revalidated_replacement"]
                ),
            ]
        ),
    }


def _embedded_digest_valid(value: Mapping[str, Any], field: str) -> bool | None:
    expected = str(value.get(field) or "").lower()
    if not expected:
        return None
    payload = dict(value)
    payload.pop(field, None)
    return expected == _canonical_digest(payload)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return bool(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text)
    )


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _directed_graph_acyclic(
    nodes: set[str],
    adjacency: Mapping[str, set[str]],
) -> bool:
    indegree = {node: 0 for node in nodes}
    for source in nodes:
        for target in adjacency.get(source, set()):
            indegree[target] = indegree.get(target, 0) + 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node = ready.pop(0)
        visited += 1
        for target in sorted(adjacency.get(node, set())):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    return visited == len(indegree)


def _percent(passed: int, total: int) -> float:
    return round(100.0 * passed / total, 1) if total else 0.0


def _human(report: Mapping[str, Any]) -> str:
    capability = dict(report.get("capability_surface") or {})
    executable = dict(report.get("executable_contract_evidence") or {})
    acceptance = dict(report.get("run_acceptance") or {})
    ledger = dict(report.get("frontier_ledger") or {})
    completion = dict(report.get("completion_truth") or {})
    verdict = dict(report.get("final_verdict") or {})
    gaps = ", ".join(str(item) for item in report.get("remaining_gaps") or []) or "none"
    return "\n".join(
        [
            f"Capability surface: {capability.get('available')}/{capability.get('declared')} "
            f"({capability.get('coverage_percent')}%; file presence only)",
            f"Executable contracts: {executable.get('passed')}/{executable.get('total')} "
            f"({executable.get('coverage_percent')}%)",
            f"Run acceptance: {acceptance.get('passed')}/{acceptance.get('total')} "
            f"({acceptance.get('completion_percent')}%)",
            "Closure truth: "
            f"any-route={completion.get('any_route_closed')} / "
            f"all-explored={completion.get('all_explored_graph_closed')} / "
            f"L3-parent={completion.get('l3_parent_route_solved')} / "
            f"L4-procurement={completion.get('l4_procurement_route_ready')}",
            "Frontier ledger: "
            f"schema={ledger.get('schema_valid')} / "
            f"digest={ledger.get('content_sha256_valid')} / "
            f"fail-closed={ledger.get('producer_fail_closed')} / "
            f"authority={ledger.get('authority_valid')}",
            f"Final verdict: {verdict.get('verdict')} / {verdict.get('route_status')} / "
            f"solved={verdict.get('solved')}",
            f"Remaining gaps: {gaps}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--human", action="store_true")
    args = parser.parse_args(argv)
    report = audit_run(args.run_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(_human(report) if args.human else json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

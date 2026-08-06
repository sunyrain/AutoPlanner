"""Deterministic authority projection for retrosynthesis frontier closure.

The campaign queue, advisory route graph, and reaction proof state are three
different records of the same search.  This module joins them without making
any of those concerns impersonate another one:

* graph rows establish proposals and hypergraph dependencies;
* queue rows establish work state and terminal stock/route boundaries;
* exact, host-replayed proof rows establish reaction-edge proof.

Closure is derived from the complete target-reachable hypergraph.  It never
reads the bounded ``route_hypotheses`` presentation projection.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
from typing import Any, Iterable, Mapping

from rdkit import Chem, RDLogger

from cascade_planner.providers.contracts import validate_provider_result
from cascade_planner.providers.stock import (
    BenchmarkCatalogStockProvider,
    SnapshotStockProvider,
    replay_stock_provider_result,
    stock_provider_set_authority_binding,
    validate_stock_observation_state,
)
from cascade_planner.harness.reaction_step_verifier import (
    REACTION_STEP_VERIFIER_VERSION,
    verify_reaction_step,
)
from cascade_planner.legacy.routes_runtime.signatures import exact_edge_signature


RDLogger.DisableLog("rdApp.*")

FRONTIER_LEDGER_SCHEMA = "frontier_ledger.v1"
FRONTIER_LEDGER_INPUT_BINDINGS_SCHEMA = "frontier_ledger_input_bindings.v1"
FRONTIER_LEDGER_MOLECULE_SCHEMA = "frontier_ledger_molecule.v1"
FRONTIER_LEDGER_EDGE_SCHEMA = "frontier_ledger_edge.v1"
ROUTE_CONSENSUS_GRAPH_SCHEMA = "route_consensus_graph.v1"
FRONTIER_QUEUE_SCHEMA = "frontier_queue.v1"
FRONTIER_JOB_SCHEMA = "frontier_job.v1"
REACTION_PROOF_STATE_SCHEMA = "codex_retrosynthesis_reaction_proof_state.v1"
REACTION_PROOF_RECORD_SCHEMA = "codex_retrosynthesis_reaction_proof_record.v2"
REACTION_STEP_PROOF_SCHEMA = "reaction_step_proof.v1"

_OPEN_JOB_STATES = {"pending", "leased", "retry_wait"}
_JOB_STATES = {
    *_OPEN_JOB_STATES,
    "succeeded",
    "failed",
    "cancelled",
}
_REACTION_LEVELS = {
    "L2_reaction_validated": 2,
    "L3_precedent_supported": 3,
    "L4_procurement_ready": 4,
}
_STOCK_BOUNDARY_TYPES = {
    "benchmark_stock",
    "commercially_orderable",
    "in_house_available",
    "common_commodity",
}


def project_frontier_ledger(
    route_consensus_graph: Mapping[str, Any],
    frontier_queue: Mapping[str, Any] | None,
    reaction_proof_state: Mapping[str, Any] | None,
    *,
    required_reaction_proof_level: int = 2,
    trusted_stock_provider_instances: Mapping[str, Any] | None = None,
    campaign_policy_sha256: str = "",
    campaign_revision: int | None = None,
    campaign_revision_sha256: str = "",
) -> dict[str, Any]:
    """Project graph, queue, and proof records into ``frontier_ledger.v1``.

    ``frontier_queue`` may be either a complete ``frontier_queue.v1`` snapshot
    or a mapping from job id to ``frontier_job.v1`` dictionaries.  Snapshot
    digests and all nested authoritative proof/envelope digests are checked.
    Invalid authority is ignored rather than downgraded into a positive claim.
    """

    if not 2 <= int(required_reaction_proof_level) <= 4:
        raise ValueError("required_reaction_proof_level must be in [2, 4]")
    required_level = int(required_reaction_proof_level)

    graph, graph_errors = _normalize_graph(route_consensus_graph)
    jobs, queue_validation = _normalize_queue(
        frontier_queue,
        expected_run_id=str(graph.get("case_id") or ""),
        expected_target_smiles=str(graph.get("target_smiles") or ""),
    )
    proof_records, proof_validation = _normalize_proof_state(
        reaction_proof_state,
        graph=graph,
        required_level=required_level,
    )
    policy_candidates = {
        str(job["metadata"].get("campaign_policy_sha256") or "")
        for job in jobs
        if _valid_sha256(job["metadata"].get("campaign_policy_sha256"))
    }
    resolved_campaign_policy_sha256 = str(campaign_policy_sha256 or "")
    if not resolved_campaign_policy_sha256 and len(policy_candidates) == 1:
        resolved_campaign_policy_sha256 = next(iter(policy_candidates))
    queue_row = dict(frontier_queue or {}) if isinstance(frontier_queue, Mapping) else {}
    try:
        queue_revision = int(queue_row.get("revision") or 0)
    except (TypeError, ValueError):
        queue_revision = -1
    input_bindings = {
        "schema_version": FRONTIER_LEDGER_INPUT_BINDINGS_SCHEMA,
        "graph_identity_sha256": _reaction_graph_identity(graph),
        "frontier_queue_content_sha256": str(
            queue_row.get("content_sha256") or ""
        ),
        "frontier_queue_revision": queue_revision,
        "campaign_policy_sha256": resolved_campaign_policy_sha256,
    }
    if campaign_revision is not None or campaign_revision_sha256:
        if (
            type(campaign_revision) is not int
            or campaign_revision < 0
            or not _valid_sha256(campaign_revision_sha256)
        ):
            raise ValueError("campaign projection revision binding is invalid")
        input_bindings.update(
            {
                "campaign_revision": campaign_revision,
                "campaign_revision_sha256": str(campaign_revision_sha256),
            }
        )

    root = str(graph.get("target_smiles") or "")
    reachable_molecules, reachable_edges = _reachable_hypergraph(graph)
    edge_rows = {
        signature: graph["edges"][signature]
        for signature in sorted(reachable_edges)
    }
    jobs_by_smiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        jobs_by_smiles[str(job["frontier_smiles"])].append(job)
    for rows in jobs_by_smiles.values():
        rows.sort(key=lambda row: str(row["job_id"]))

    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    for signature, edge in edge_rows.items():
        outgoing[str(edge["product_smiles"])].append(signature)
        for precursor in edge["precursor_smiles"]:
            incoming[str(precursor)].append(signature)
    for values in [*outgoing.values(), *incoming.values()]:
        values[:] = sorted(set(values))

    molecule_stock: dict[str, dict[str, Any]] = {}
    molecule_terminal_closure: dict[str, dict[str, Any]] = {}
    for smiles in sorted(reachable_molecules):
        molecule_jobs = jobs_by_smiles.get(smiles, [])
        molecule_stock[smiles] = _stock_projection(
            smiles,
            molecule_jobs,
            trusted_stock_provider_instances=trusted_stock_provider_instances,
        )
        molecule_terminal_closure[smiles] = _terminal_reaction_closure_projection(
            molecule_jobs,
            required_level=required_level,
        )

    edge_proofs: dict[str, dict[str, Any]] = {}
    for signature, edge in edge_rows.items():
        edge_proofs[signature] = _edge_proof_projection(
            signature,
            edge=edge,
            records=proof_records.get(signature, []),
            required_level=required_level,
        )

    any_closed, all_closed, iterations = _closure_fixed_point(
        molecules=reachable_molecules,
        outgoing=outgoing,
        edges=edge_rows,
        proofs=edge_proofs,
        stock=molecule_stock,
        terminal_closure=molecule_terminal_closure,
    )
    procurement_stock = {
        smiles: {"closed": row["procurement_boundary_closed"] is True}
        for smiles, row in molecule_stock.items()
    }
    procurement_any_closed, procurement_all_closed, procurement_iterations = (
        _closure_fixed_point(
            molecules=reachable_molecules,
            outgoing=outgoing,
            edges=edge_rows,
            proofs=edge_proofs,
            stock=procurement_stock,
            terminal_closure=molecule_terminal_closure,
        )
    )

    molecules: dict[str, dict[str, Any]] = {}
    for smiles in sorted(reachable_molecules):
        molecule_jobs = jobs_by_smiles.get(smiles, [])
        edge_signatures = outgoing.get(smiles, [])
        proof_closed_count = sum(
            1 for signature in edge_signatures if edge_proofs[signature]["closed"]
        )
        work_states = sorted({str(row["state"]) for row in molecule_jobs})
        molecules[smiles] = {
            "schema_version": FRONTIER_LEDGER_MOLECULE_SCHEMA,
            "canonical_smiles": smiles,
            "node_ids": list(graph["node_ids_by_smiles"].get(smiles, [])),
            "proposal": {
                "state": "expanded" if edge_signatures else "frontier",
                "outgoing_edge_signatures": list(edge_signatures),
                "alternative_count": len(edge_signatures),
            },
            "work": {
                "job_ids": [str(row["job_id"]) for row in molecule_jobs],
                "states": work_states,
                "open": any(str(row["state"]) in _OPEN_JOB_STATES for row in molecule_jobs),
                "proposal_expansion_allowed": any(
                    str(row["state"]) in _OPEN_JOB_STATES
                    and row["metadata"].get("proposal_expansion_allowed") is not False
                    for row in molecule_jobs
                ),
                "proposal_expansion_succeeded": any(
                    row["state"] == "succeeded"
                    and row["closure_kind"] == "proposal_expansion"
                    for row in molecule_jobs
                ),
                "queue_dependency_ids": sorted(
                    {
                        str(dependency)
                        for row in molecule_jobs
                        for dependency in row["dependency_ids"]
                    }
                ),
            },
            "stock": molecule_stock[smiles],
            "reaction_proof": {
                "outgoing_edge_count": len(edge_signatures),
                "closed_outgoing_edge_count": proof_closed_count,
                "all_outgoing_edges_proven": bool(edge_signatures)
                and proof_closed_count == len(edge_signatures),
                "terminal_closure": molecule_terminal_closure[smiles],
            },
            "dependencies": {
                "outgoing_edge_signatures": list(edge_signatures),
                "incoming_edge_signatures": list(incoming.get(smiles, [])),
            },
            "closure": {
                "any_benchmark_route_closed": bool(
                    any_closed.get(smiles, False)
                ),
                "all_explored_benchmark_closed": bool(
                    all_closed.get(smiles, False)
                ),
                "any_procurement_route_closed": bool(
                    procurement_any_closed.get(smiles, False)
                ),
                "all_explored_procurement_closed": bool(
                    procurement_all_closed.get(smiles, False)
                ),
                # Compatibility aliases: generic closure is benchmark-search
                # closure, never an implicit procurement claim.
                "any_route_closed": bool(any_closed.get(smiles, False)),
                "all_explored_graph_closed": bool(all_closed.get(smiles, False)),
            },
        }

    edges: dict[str, dict[str, Any]] = {}
    for signature, edge in edge_rows.items():
        product = str(edge["product_smiles"])
        precursors = list(edge["precursor_smiles"])
        product_jobs = jobs_by_smiles.get(product, [])
        proof = edge_proofs[signature]
        edge_any_closed = bool(
            proof["closed"] and all(any_closed.get(item, False) for item in precursors)
        )
        edge_all_closed = bool(
            proof["closed"] and all(all_closed.get(item, False) for item in precursors)
        )
        edge_procurement_any_closed = bool(
            proof["closed"]
            and all(
                procurement_any_closed.get(item, False) for item in precursors
            )
        )
        edge_procurement_all_closed = bool(
            proof["closed"]
            and all(
                procurement_all_closed.get(item, False) for item in precursors
            )
        )
        edges[signature] = {
            "schema_version": FRONTIER_LEDGER_EDGE_SCHEMA,
            "exact_edge_signature": signature,
            "product_smiles": product,
            "precursor_smiles": precursors,
            "proposal": {
                "present": True,
                "step_ids": list(edge["step_ids"]),
                "proposal_ids": list(edge["proposal_ids"]),
                "source_refs": list(edge["source_refs"]),
                "evidence_refs": list(edge["evidence_refs"]),
            },
            "work": {
                "product_frontier_job_ids": [
                    str(row["job_id"]) for row in product_jobs
                ],
                "proof_request_ids": list(proof["proof_request_ids"]),
                "open_proof_work": not proof["closed"],
            },
            "stock": {
                "precursor_count": len(precursors),
                "stock_closed_precursor_count": sum(
                    1 for item in precursors if molecule_stock[item]["closed"]
                ),
                "all_precursors_stock_closed": bool(precursors)
                and all(molecule_stock[item]["closed"] for item in precursors),
            },
            "reaction_proof": proof,
            "dependencies": {
                "requires_molecule_smiles": precursors,
                "precursor_edge_options": {
                    item: list(outgoing.get(item, [])) for item in sorted(set(precursors))
                },
                "required_by_edge_signatures": list(incoming.get(product, [])),
            },
            "closure": {
                "any_benchmark_route_closed": edge_any_closed,
                "all_explored_benchmark_closed": edge_all_closed,
                "any_procurement_route_closed": edge_procurement_any_closed,
                "all_explored_procurement_closed": (
                    edge_procurement_all_closed
                ),
                "any_route_closed": edge_any_closed,
                "all_explored_graph_closed": edge_all_closed,
            },
        }

    unresolved_frontier_smiles = {
        smiles
        for smiles, row in molecules.items()
        if dict(row.get("proposal") or {}).get("state") == "frontier"
        and dict(row.get("stock") or {}).get("closed") is not True
        and dict(
            dict(row.get("reaction_proof") or {}).get("terminal_closure") or {}
        ).get("closed")
        is not True
    }
    graph_valid = not graph_errors
    queue_valid = bool(queue_validation["valid"])
    root_present = bool(root and root in reachable_molecules)
    any_route_closed = bool(
        graph_valid and queue_valid and root_present and any_closed.get(root, False)
    )
    all_explored_graph_closed = bool(
        graph_valid and queue_valid and root_present and all_closed.get(root, False)
    )
    any_procurement_route_closed = bool(
        graph_valid
        and queue_valid
        and root_present
        and procurement_any_closed.get(root, False)
    )
    all_explored_procurement_closed = bool(
        graph_valid
        and queue_valid
        and root_present
        and procurement_all_closed.get(root, False)
    )
    payload: dict[str, Any] = {
        "schema_version": FRONTIER_LEDGER_SCHEMA,
        "input_bindings": input_bindings,
        "root": {
            "canonical_smiles": root,
            "node_ids": list(graph["node_ids_by_smiles"].get(root, [])),
            "closure": {
                "any_benchmark_route_closed": any_route_closed,
                "all_explored_benchmark_closed": all_explored_graph_closed,
                "any_procurement_route_closed": any_procurement_route_closed,
                "all_explored_procurement_closed": (
                    all_explored_procurement_closed
                ),
                "any_route_closed": any_route_closed,
                "all_explored_graph_closed": all_explored_graph_closed,
            },
        },
        "required_reaction_proof_level": required_level,
        "molecules": molecules,
        "edges": edges,
        "summary": {
            "any_benchmark_route_closed": any_route_closed,
            "all_explored_benchmark_closed": all_explored_graph_closed,
            "any_procurement_route_closed": any_procurement_route_closed,
            "all_explored_procurement_closed": (
                all_explored_procurement_closed
            ),
            "any_route_closed": any_route_closed,
            "all_explored_graph_closed": all_explored_graph_closed,
            "reachable_molecule_count": len(reachable_molecules),
            "reachable_edge_count": len(edge_rows),
            "reaction_proven_edge_count": sum(
                1 for row in edge_proofs.values() if row["closed"]
            ),
            "stock_closed_molecule_count": sum(
                1 for row in molecule_stock.values() if row["closed"]
            ),
            "benchmark_membership_closed_molecule_count": sum(
                1
                for row in molecule_stock.values()
                if row["benchmark_membership_closed"]
            ),
            "procurement_boundary_closed_molecule_count": sum(
                1
                for row in molecule_stock.values()
                if row["procurement_boundary_closed"]
            ),
            # Proposal state describes the hypergraph, not scheduler
            # eligibility.  Depth/cycle boundaries therefore remain visible
            # as unexpanded proposal frontiers while the separate eligibility
            # counter says whether an open worker may consume them.
            "proposal_pending_molecule_count": len(unresolved_frontier_smiles),
            "proposal_expansion_eligible_molecule_count": sum(
                1
                for smiles in unresolved_frontier_smiles
                if dict(molecules[smiles].get("work") or {}).get(
                    "proposal_expansion_allowed"
                )
                is True
            ),
            "work_pending_molecule_count": sum(
                1
                for row in molecules.values()
                if dict(row.get("work") or {}).get("open") is True
            ),
            "stock_pending_leaf_count": len(unresolved_frontier_smiles),
            "reaction_proof_pending_edge_count": sum(
                1
                for row in edges.values()
                if dict(row.get("reaction_proof") or {}).get("closed") is not True
            ),
            "dependency_pending_edge_count": sum(
                1
                for row in edges.values()
                if any(
                    dict(molecules.get(str(smiles)) or {})
                    .get("closure", {})
                    .get("all_explored_graph_closed")
                    is not True
                    for smiles in dict(row.get("dependencies") or {}).get(
                        "requires_molecule_smiles", []
                    )
                )
            ),
            "open_work_molecule_count": sum(
                1
                for smiles in reachable_molecules
                if any(
                    str(row["state"]) in _OPEN_JOB_STATES
                    for row in jobs_by_smiles.get(smiles, [])
                )
            ),
            "fixed_point_iterations": iterations,
            "benchmark_fixed_point_iterations": iterations,
            "procurement_fixed_point_iterations": procurement_iterations,
        },
        "input_validation": {
            "graph": {"valid": graph_valid, "reasons": graph_errors},
            "frontier_queue": queue_validation,
            "reaction_proof_state": proof_validation,
            "stock_authority": {
                "valid": True,
                "positive_claim_count": sum(
                    len(row["closure_job_ids"])
                    + len(row["rejected_stock_job_ids"])
                    for row in molecule_stock.values()
                ),
                "host_replayed_claim_count": sum(
                    len(row["closure_job_ids"])
                    for row in molecule_stock.values()
                ),
                "rejected_claim_count": sum(
                    len(row["rejected_stock_job_ids"])
                    for row in molecule_stock.values()
                ),
                "rejection_reasons": sorted(
                    {
                        reason
                        for row in molecule_stock.values()
                        for reasons in row["replay_rejection_reasons"].values()
                        for reason in reasons
                    }
                ),
                "authority_boundary": "current_host_stock_provider_replay",
            },
        },
        "semantics": {
            "route_hypotheses_are_not_consumed": True,
            "proposal_work_stock_reaction_proof_and_dependencies_are_orthogonal": True,
            "reaction_proof_requires_exact_edge_signature": True,
            "any_route_closed_is_existential_and_or_hypergraph_fixed_point": True,
            "all_explored_graph_closed_is_universal_hypergraph_fixed_point": True,
            "all_explored_graph_requires_every_reachable_edge_and_leaf": True,
            "generic_any_all_mean_benchmark_search_closure": True,
            "procurement_closure_has_an_independent_fixed_point": True,
            "orthogonal_pending_counts_are_ledger_derived": True,
            "invalid_authority_fails_closed": True,
            "stock_envelopes_are_replayed_by_host_provider": True,
            "benchmark_membership_is_not_procurement_authority": True,
            "input_bindings_are_digest_bound": True,
        },
    }
    # Canonicalize before hashing and returning.  Provider/verifier dataclasses
    # can contain tuples even though the persisted JSON artifact contains
    # lists; returning the strict JSON value prevents in-memory and on-disk
    # ledgers from disagreeing while sharing a digest.
    payload = _json_roundtrip(payload)
    payload["content_sha256"] = _digest(payload)
    return _json_roundtrip(payload)


def validate_frontier_ledger(
    value: Any,
    *,
    trusted_stock_provider_instances: Mapping[str, Any] | None = None,
    expected_input_bindings: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return stable reason codes for a serialized ledger projection.

    The content digest detects accidental mutation, but it is not an
    authorization signature.  Recompute the exact-edge topology and both
    hypergraph fixed points as well, so a caller cannot make a self-consistent
    closure claim merely by editing booleans and recalculating the hash.
    """

    if not isinstance(value, Mapping):
        return ["frontier_ledger_not_object"]
    row = dict(value)
    reasons: list[str] = []
    if row.get("schema_version") != FRONTIER_LEDGER_SCHEMA:
        reasons.append("invalid_frontier_ledger_schema")
    supplied_digest = str(row.pop("content_sha256", ""))
    if not supplied_digest or supplied_digest != _digest(row):
        reasons.append("frontier_ledger_content_digest_invalid")
    if not isinstance(row.get("root"), Mapping):
        reasons.append("frontier_ledger_root_not_object")
    if not isinstance(row.get("summary"), Mapping):
        reasons.append("frontier_ledger_summary_not_object")
    reasons.extend(
        _frontier_ledger_input_binding_reasons(
            row.get("input_bindings"),
            expected=expected_input_bindings,
        )
    )
    molecules = row.get("molecules")
    if not isinstance(molecules, Mapping):
        reasons.append("frontier_ledger_molecules_not_object")
        molecules = {}
    edges = row.get("edges")
    if not isinstance(edges, Mapping):
        reasons.append("frontier_ledger_edges_not_object")
        edges = {}
    for key, raw in molecules.items():
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema_version") != FRONTIER_LEDGER_MOLECULE_SCHEMA
            or raw.get("canonical_smiles") != key
        ):
            reasons.append(f"invalid_frontier_ledger_molecule:{key}")
            continue
        for field in ("proposal", "work", "stock", "reaction_proof", "dependencies", "closure"):
            if not isinstance(raw.get(field), Mapping):
                reasons.append(f"frontier_ledger_molecule_missing_{field}:{key}")
    for key, raw in edges.items():
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema_version") != FRONTIER_LEDGER_EDGE_SCHEMA
            or raw.get("exact_edge_signature") != key
        ):
            reasons.append(f"invalid_frontier_ledger_edge:{key}")
            continue
        for field in ("proposal", "work", "stock", "reaction_proof", "dependencies", "closure"):
            if not isinstance(raw.get(field), Mapping):
                reasons.append(f"frontier_ledger_edge_missing_{field}:{key}")
    if not reasons:
        reasons.extend(
            _frontier_ledger_semantic_reasons(
                row,
                molecules=molecules,
                edges=edges,
                trusted_stock_provider_instances=(
                    trusted_stock_provider_instances
                ),
            )
        )
    return sorted(set(reasons))


def _frontier_ledger_input_binding_reasons(
    value: Any,
    *,
    expected: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["frontier_ledger_input_bindings_not_object"]
    row = dict(value)
    reasons: list[str] = []
    if row.get("schema_version") != FRONTIER_LEDGER_INPUT_BINDINGS_SCHEMA:
        reasons.append("invalid_frontier_ledger_input_bindings_schema")
    for field in (
        "graph_identity_sha256",
        "frontier_queue_content_sha256",
        "campaign_policy_sha256",
    ):
        if not _valid_sha256(row.get(field)):
            reasons.append(f"frontier_ledger_input_binding_invalid:{field}")
    revision = row.get("frontier_queue_revision")
    if type(revision) is not int or revision < 0:
        reasons.append(
            "frontier_ledger_input_binding_invalid:frontier_queue_revision"
        )
    campaign_revision_present = "campaign_revision" in row
    campaign_digest_present = "campaign_revision_sha256" in row
    if campaign_revision_present != campaign_digest_present:
        reasons.append("frontier_ledger_campaign_revision_binding_incomplete")
    elif campaign_revision_present:
        campaign_revision = row.get("campaign_revision")
        if type(campaign_revision) is not int or campaign_revision < 0:
            reasons.append(
                "frontier_ledger_input_binding_invalid:campaign_revision"
            )
        if not _valid_sha256(row.get("campaign_revision_sha256")):
            reasons.append(
                "frontier_ledger_input_binding_invalid:campaign_revision_sha256"
            )
    if expected is not None:
        expected_row = dict(expected)
        for field in (
            "graph_identity_sha256",
            "frontier_queue_content_sha256",
            "frontier_queue_revision",
            "campaign_policy_sha256",
            "campaign_revision",
            "campaign_revision_sha256",
        ):
            if field in row or field in expected_row:
                if row.get(field) == expected_row.get(field):
                    continue
                reasons.append(
                    f"frontier_ledger_input_binding_mismatch:{field}"
                )
    return reasons


def _frontier_ledger_semantic_reasons(
    ledger: Mapping[str, Any],
    *,
    molecules: Mapping[str, Any],
    edges: Mapping[str, Any],
    trusted_stock_provider_instances: Mapping[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    molecule_rows: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in molecules.items():
        key = str(raw_key or "")
        row = dict(raw_value) if isinstance(raw_value, Mapping) else {}
        if not row:
            continue
        canonical = _canonical_smiles(key)
        if not canonical or canonical != key or row.get("canonical_smiles") != key:
            reasons.append(f"frontier_ledger_molecule_identity_invalid:{key}")
            continue
        molecule_rows[key] = row

    edge_rows: dict[str, dict[str, Any]] = {}
    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    for raw_key, raw_value in edges.items():
        key = str(raw_key or "")
        row = dict(raw_value) if isinstance(raw_value, Mapping) else {}
        if not row:
            continue
        product = _canonical_smiles(row.get("product_smiles"))
        raw_precursors = row.get("precursor_smiles")
        precursors = (
            [_canonical_smiles(item) for item in raw_precursors]
            if isinstance(raw_precursors, list)
            else []
        )
        expected_signature = exact_edge_signature(product, precursors)
        if (
            not product
            or not precursors
            or any(not item for item in precursors)
            or product != str(row.get("product_smiles") or "")
            or precursors != list(raw_precursors or [])
            or expected_signature != key
        ):
            reasons.append(f"frontier_ledger_exact_edge_invalid:{key}")
            continue
        if product not in molecule_rows or any(
            precursor not in molecule_rows for precursor in precursors
        ):
            reasons.append(f"frontier_ledger_edge_molecule_missing:{key}")
            continue
        edge_rows[key] = {
            **row,
            "product_smiles": product,
            "precursor_smiles": precursors,
        }
        outgoing[product].append(key)
        for precursor in precursors:
            incoming[precursor].append(key)
    for values in [*outgoing.values(), *incoming.values()]:
        values[:] = sorted(set(values))

    for smiles, row in molecule_rows.items():
        expected_outgoing = list(outgoing.get(smiles, []))
        expected_incoming = list(incoming.get(smiles, []))
        proposal = dict(row.get("proposal") or {})
        dependencies = dict(row.get("dependencies") or {})
        if sorted(set(proposal.get("outgoing_edge_signatures") or [])) != expected_outgoing:
            reasons.append(f"frontier_ledger_molecule_proposal_topology_mismatch:{smiles}")
        expected_state = "expanded" if expected_outgoing else "frontier"
        try:
            alternative_count = int(proposal.get("alternative_count") or 0)
        except (TypeError, ValueError):
            alternative_count = -1
        if (
            proposal.get("state") != expected_state
            or alternative_count != len(expected_outgoing)
        ):
            reasons.append(f"frontier_ledger_molecule_proposal_state_mismatch:{smiles}")
        if (
            sorted(set(dependencies.get("outgoing_edge_signatures") or []))
            != expected_outgoing
            or sorted(set(dependencies.get("incoming_edge_signatures") or []))
            != expected_incoming
        ):
            reasons.append(f"frontier_ledger_molecule_dependencies_mismatch:{smiles}")
        reasons.extend(
            _serialized_stock_replay_reasons(
                smiles,
                stock=dict(row.get("stock") or {}),
                trusted_stock_provider_instances=(
                    trusted_stock_provider_instances
                ),
            )
        )

    for signature, row in edge_rows.items():
        precursors = list(row["precursor_smiles"])
        dependencies = dict(row.get("dependencies") or {})
        if list(dependencies.get("requires_molecule_smiles") or []) != precursors:
            reasons.append(f"frontier_ledger_edge_dependencies_mismatch:{signature}")
        options = dependencies.get("precursor_edge_options")
        if isinstance(options, Mapping):
            expected_options = {
                item: list(outgoing.get(item, [])) for item in sorted(set(precursors))
            }
            normalized_options = {
                str(key): sorted(set(value if isinstance(value, list) else []))
                for key, value in options.items()
            }
            if normalized_options != expected_options:
                reasons.append(
                    f"frontier_ledger_edge_precursor_options_mismatch:{signature}"
                )
        reasons.extend(
            _serialized_edge_proof_reasons(
                signature,
                edge=row,
                required_level=int(
                    ledger.get("required_reaction_proof_level") or 0
                ),
            )
        )

    if reasons:
        # Identity/topology failures make a fixed-point calculation ambiguous.
        return reasons

    molecule_keys = set(molecule_rows)
    stock = {
        smiles: {"closed": dict(row.get("stock") or {}).get("closed") is True}
        for smiles, row in molecule_rows.items()
    }
    procurement_stock = {
        smiles: {
            "closed": dict(row.get("stock") or {}).get(
                "procurement_boundary_closed"
            )
            is True
        }
        for smiles, row in molecule_rows.items()
    }
    terminal = {
        smiles: {
            "closed": dict(
                dict(row.get("reaction_proof") or {}).get("terminal_closure")
                or {}
            ).get("closed")
            is True
        }
        for smiles, row in molecule_rows.items()
    }
    proofs = {
        signature: {
            "closed": dict(row.get("reaction_proof") or {}).get("closed") is True
        }
        for signature, row in edge_rows.items()
    }
    any_closed, all_closed, iterations = _closure_fixed_point(
        molecules=molecule_keys,
        outgoing=outgoing,
        edges=edge_rows,
        proofs=proofs,
        stock=stock,
        terminal_closure=terminal,
    )
    procurement_any_closed, procurement_all_closed, procurement_iterations = (
        _closure_fixed_point(
            molecules=molecule_keys,
            outgoing=outgoing,
            edges=edge_rows,
            proofs=proofs,
            stock=procurement_stock,
            terminal_closure=terminal,
        )
    )
    for smiles, row in molecule_rows.items():
        closure = dict(row.get("closure") or {})
        expected_closure = {
            "any_benchmark_route_closed": bool(any_closed.get(smiles, False)),
            "all_explored_benchmark_closed": bool(
                all_closed.get(smiles, False)
            ),
            "any_procurement_route_closed": bool(
                procurement_any_closed.get(smiles, False)
            ),
            "all_explored_procurement_closed": bool(
                procurement_all_closed.get(smiles, False)
            ),
            "any_route_closed": bool(any_closed.get(smiles, False)),
            "all_explored_graph_closed": bool(all_closed.get(smiles, False)),
        }
        for field, expected in expected_closure.items():
            if closure.get(field) is not expected:
                reasons.append(
                    f"frontier_ledger_molecule_closure_mismatch:{smiles}:{field}"
                )
    for signature, row in edge_rows.items():
        precursors = list(row["precursor_smiles"])
        proof_closed = proofs[signature]["closed"]
        expected_any = bool(
            proof_closed and all(any_closed.get(item, False) for item in precursors)
        )
        expected_all = bool(
            proof_closed and all(all_closed.get(item, False) for item in precursors)
        )
        expected_procurement_any = bool(
            proof_closed
            and all(
                procurement_any_closed.get(item, False) for item in precursors
            )
        )
        expected_procurement_all = bool(
            proof_closed
            and all(
                procurement_all_closed.get(item, False) for item in precursors
            )
        )
        closure = dict(row.get("closure") or {})
        expected_edge_closure = {
            "any_benchmark_route_closed": expected_any,
            "all_explored_benchmark_closed": expected_all,
            "any_procurement_route_closed": expected_procurement_any,
            "all_explored_procurement_closed": expected_procurement_all,
            "any_route_closed": expected_any,
            "all_explored_graph_closed": expected_all,
        }
        for field, expected in expected_edge_closure.items():
            if closure.get(field) is not expected:
                reasons.append(
                    f"frontier_ledger_edge_closure_mismatch:{signature}:{field}"
                )

    root = dict(ledger.get("root") or {})
    root_smiles = str(root.get("canonical_smiles") or "")
    input_validation = dict(ledger.get("input_validation") or {})
    graph_valid = dict(input_validation.get("graph") or {}).get("valid") is True
    queue_valid = (
        dict(input_validation.get("frontier_queue") or {}).get("valid") is True
    )
    root_present = root_smiles in molecule_rows
    summary = dict(ledger.get("summary") or {})
    root_closure = dict(root.get("closure") or {})
    expected_root_closure = {
        "any_benchmark_route_closed": bool(
            graph_valid
            and queue_valid
            and root_present
            and any_closed.get(root_smiles, False)
        ),
        "all_explored_benchmark_closed": bool(
            graph_valid
            and queue_valid
            and root_present
            and all_closed.get(root_smiles, False)
        ),
        "any_procurement_route_closed": bool(
            graph_valid
            and queue_valid
            and root_present
            and procurement_any_closed.get(root_smiles, False)
        ),
        "all_explored_procurement_closed": bool(
            graph_valid
            and queue_valid
            and root_present
            and procurement_all_closed.get(root_smiles, False)
        ),
    }
    expected_root_closure["any_route_closed"] = expected_root_closure[
        "any_benchmark_route_closed"
    ]
    expected_root_closure["all_explored_graph_closed"] = expected_root_closure[
        "all_explored_benchmark_closed"
    ]
    for field, expected in expected_root_closure.items():
        if root_closure.get(field) is not expected:
            reasons.append(f"frontier_ledger_root_closure_mismatch:{field}")
    expected_summary: dict[str, Any] = {
        **expected_root_closure,
        "reachable_molecule_count": len(molecule_rows),
        "reachable_edge_count": len(edge_rows),
        "reaction_proven_edge_count": sum(
            1 for row in proofs.values() if row["closed"]
        ),
        "stock_closed_molecule_count": sum(
            1 for row in stock.values() if row["closed"]
        ),
        "benchmark_membership_closed_molecule_count": sum(
            1
            for row in molecule_rows.values()
            if dict(row.get("stock") or {}).get(
                "benchmark_membership_closed"
            )
            is True
        ),
        "procurement_boundary_closed_molecule_count": sum(
            1 for row in procurement_stock.values() if row["closed"]
        ),
        "open_work_molecule_count": sum(
            1
            for row in molecule_rows.values()
            if dict(row.get("work") or {}).get("open") is True
        ),
        "fixed_point_iterations": iterations,
        "benchmark_fixed_point_iterations": iterations,
        "procurement_fixed_point_iterations": procurement_iterations,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            reasons.append(f"frontier_ledger_summary_mismatch:{field}")
    if not root_present:
        reasons.append("frontier_ledger_root_molecule_missing")
    return reasons


def _serialized_edge_proof_reasons(
    signature: str,
    *,
    edge: Mapping[str, Any],
    required_level: int,
) -> list[str]:
    """Replay every positive serialized edge proof with the current host."""

    proof_projection = dict(edge.get("reaction_proof") or {})
    closed = proof_projection.get("closed") is True
    binding = proof_projection.get("host_replay_binding")
    binding_row = dict(binding) if isinstance(binding, Mapping) else {}
    reasons: list[str] = []
    if not closed:
        if binding_row:
            reasons.append(
                f"frontier_ledger_open_edge_carries_positive_binding:{signature}"
            )
        return reasons
    if (
        binding_row.get("schema_version")
        != "frontier_ledger_host_replay_binding.v1"
        or binding_row.get("proof_authority") != "current_host_verifier_replay"
    ):
        reasons.append(f"frontier_ledger_edge_host_binding_invalid:{signature}")
        return reasons
    materialized = binding_row.get("materialized_candidate")
    proof = binding_row.get("proof")
    if not isinstance(materialized, Mapping) or not isinstance(proof, Mapping):
        reasons.append(f"frontier_ledger_edge_host_binding_incomplete:{signature}")
        return reasons
    materialized_row = dict(materialized)
    proof_row = dict(proof)
    if str(binding_row.get("materialized_candidate_sha256") or "") != _digest(
        materialized_row
    ):
        reasons.append(
            f"frontier_ledger_edge_materialized_digest_invalid:{signature}"
        )
    if _canonical_smiles(materialized_row.get("product_smiles")) != edge.get(
        "product_smiles"
    ) or sorted(
        _canonical_smiles(item)
        for item in materialized_row.get("reactant_smiles") or []
    ) != sorted(edge.get("precursor_smiles") or []):
        reasons.append(
            f"frontier_ledger_edge_materialized_identity_mismatch:{signature}"
        )
    try:
        step_index = int(proof_row.get("step_index") or 0)
    except (TypeError, ValueError):
        step_index = 0
        reasons.append(f"frontier_ledger_edge_proof_step_index_invalid:{signature}")
    checks = (
        dict(proof_row.get("checks") or {})
        if isinstance(proof_row.get("checks"), Mapping)
        else {}
    )
    try:
        replayed = verify_reaction_step(
            materialized_row,
            step_index=step_index,
            graph_and_stock_closed=checks.get("graph_and_stock_closed") is True,
        )
    except Exception as exc:  # fail closed at the optional chemistry boundary
        replayed = {}
        reasons.append(
            f"frontier_ledger_edge_host_replay_error:{signature}:{type(exc).__name__}"
        )
    if not replayed or _digest(replayed) != _digest(proof_row):
        reasons.append(f"frontier_ledger_edge_host_replay_mismatch:{signature}")
    replay_level = _REACTION_LEVELS.get(
        str((replayed or {}).get("proof_level") or ""), 0
    )
    try:
        projected_level = int(proof_projection.get("achieved_proof_level") or 0)
    except (TypeError, ValueError):
        projected_level = -1
    if (
        not replayed
        or replayed.get("accepted") is not True
        or replay_level < required_level
        or projected_level != replay_level
    ):
        reasons.append(f"frontier_ledger_edge_proof_level_invalid:{signature}")
    if proof_projection.get("authority") != "current_host_verifier_replay":
        reasons.append(f"frontier_ledger_edge_proof_authority_invalid:{signature}")
    return reasons


def _normalize_graph(value: Any) -> tuple[dict[str, Any], list[str]]:
    row = dict(value) if isinstance(value, Mapping) else {}
    reasons: list[str] = []
    if row.get("schema_version") != ROUTE_CONSENSUS_GRAPH_SCHEMA:
        reasons.append("invalid_route_consensus_graph_schema")
    if not str(row.get("case_id") or "").strip():
        reasons.append("route_consensus_graph_case_id_missing")
    target = _canonical_smiles(row.get("target_smiles"))
    if not target:
        reasons.append("invalid_route_consensus_graph_target")
    edges: dict[str, dict[str, Any]] = {}
    node_ids: dict[str, set[str]] = defaultdict(set)
    raw_steps = row.get("steps")
    if not isinstance(raw_steps, list):
        reasons.append("route_consensus_graph_steps_not_list")
        raw_steps = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            reasons.append(f"route_consensus_graph_step_not_object:{index}")
            continue
        step = dict(raw)
        if step.get("schema_version") != "route_consensus_step.v1":
            reasons.append(f"invalid_route_consensus_step_schema:{index}")
        step_id = str(step.get("step_id") or "")
        product = _canonical_smiles(step.get("product_smiles"))
        raw_precursors = step.get("precursor_smiles")
        precursors = [
            _canonical_smiles(item) for item in raw_precursors
        ] if isinstance(raw_precursors, list) else []
        signature = exact_edge_signature(product, precursors)
        if not step_id:
            reasons.append(f"route_consensus_step_id_missing:{index}")
        if not product or not precursors or any(not item for item in precursors):
            reasons.append(f"route_consensus_step_structures_invalid:{index}")
            continue
        supplied_signature = str(step.get("signature") or "")
        if supplied_signature != _route_graph_edge_signature(product, precursors):
            reasons.append(f"route_consensus_step_signature_mismatch:{step_id or index}")
            continue
        if signature in edges:
            reasons.append(f"duplicate_route_consensus_edge_signature:{signature}")
            continue
        product_node_id = str(step.get("product_node_id") or "")
        precursor_node_ids = [str(item or "") for item in step.get("precursor_node_ids") or []]
        if product_node_id:
            node_ids[product].add(product_node_id)
        for smiles, node_id in zip(precursors, precursor_node_ids):
            if node_id:
                node_ids[smiles].add(node_id)
        edges[signature] = {
            "product_smiles": product,
            "precursor_smiles": sorted(precursors),
            "step_ids": [step_id],
            "source_graph_signatures": [supplied_signature],
            "proposal_ids": sorted(
                {str(item) for item in step.get("proposal_ids") or [] if str(item)}
            ),
            "source_refs": sorted(
                {str(item) for item in step.get("source_refs") or [] if str(item)}
            ),
            "evidence_refs": sorted(
                {str(item) for item in step.get("evidence_refs") or [] if str(item)}
            ),
        }
    for raw in row.get("nodes") or []:
        if not isinstance(raw, Mapping):
            continue
        smiles = _canonical_smiles(
            raw.get("canonical_isomeric_smiles") or raw.get("smiles")
        )
        node_id = str(raw.get("node_id") or "")
        if smiles and node_id:
            node_ids[smiles].add(node_id)
    if target and target not in {edge["product_smiles"] for edge in edges.values()}:
        # A target-only graph remains a valid stock frontier.  It is not a
        # reaction route until the target is independently closed by the queue.
        node_ids.setdefault(target, set())
    return (
        {
            "schema_version": str(row.get("schema_version") or ""),
            "case_id": str(row.get("case_id") or ""),
            "target_smiles": target,
            "edges": edges,
            "node_ids_by_smiles": {
                key: sorted(values) for key, values in sorted(node_ids.items())
            },
            "identity_steps": [
                {
                    "step_id": str(step.get("step_id") or ""),
                    "signature": str(step.get("signature") or ""),
                    "product_smiles": _canonical_smiles(step.get("product_smiles")),
                    "precursor_smiles": sorted(
                        _canonical_smiles(item)
                        for item in step.get("precursor_smiles") or []
                    ),
                }
                for step in raw_steps
                if isinstance(step, Mapping)
            ],
        },
        sorted(set(reasons)),
    )


def _normalize_queue(
    value: Any,
    *,
    expected_run_id: str,
    expected_target_smiles: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = dict(value) if isinstance(value, Mapping) else {}
    reasons: list[str] = []
    source_kind = "jobs_dict"
    raw_jobs: list[tuple[str, Any]] = []
    if row.get("schema_version") == FRONTIER_QUEUE_SCHEMA:
        source_kind = "frontier_queue_snapshot"
        digest_payload = dict(row)
        supplied_digest = str(digest_payload.pop("content_sha256", ""))
        if not supplied_digest or supplied_digest != _digest(digest_payload):
            reasons.append("frontier_queue_snapshot_digest_invalid")
        snapshot_run_id = str(row.get("run_id") or "")
        if not snapshot_run_id:
            reasons.append("frontier_queue_snapshot_run_id_missing")
        elif snapshot_run_id != expected_run_id:
            reasons.append("frontier_queue_snapshot_graph_case_mismatch")
        if not isinstance(row.get("jobs"), list):
            reasons.append("frontier_queue_snapshot_jobs_not_list")
        else:
            raw_jobs = [("", item) for item in row["jobs"]]
    else:
        reasons.append("frontier_jobs_dict_not_authoritative")
        if not isinstance(value, Mapping):
            reasons.append("frontier_jobs_not_mapping")
        else:
            raw_jobs = [(str(key), item) for key, item in row.items()]

    jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, (mapping_key, raw) in enumerate(raw_jobs):
        job = dict(raw) if isinstance(raw, Mapping) else {}
        job_id = str(job.get("job_id") or "")
        state = str(job.get("state") or "")
        smiles = _canonical_smiles(job.get("frontier_smiles"))
        job_reasons: list[str] = []
        if job.get("schema_version") != FRONTIER_JOB_SCHEMA:
            job_reasons.append("invalid_schema")
        if not job_id or job_id in seen_ids:
            job_reasons.append("missing_or_duplicate_job_id")
        if mapping_key and mapping_key != job_id:
            job_reasons.append("jobs_dict_key_mismatch")
        if not smiles or smiles != str(job.get("frontier_smiles") or ""):
            job_reasons.append("frontier_smiles_not_canonical")
        if state not in _JOB_STATES:
            job_reasons.append("invalid_state")
        try:
            achieved = int(job.get("achieved_proof_level") or 0)
        except (TypeError, ValueError):
            achieved = -1
        if not 0 <= achieved <= 4:
            job_reasons.append("invalid_achieved_proof_level")
        dependencies = job.get("dependency_ids")
        if not isinstance(dependencies, list):
            job_reasons.append("dependency_ids_not_list")
            dependencies = []
        metadata = job.get("metadata")
        if not isinstance(metadata, Mapping):
            job_reasons.append("metadata_not_object")
            metadata = {}
        if str(job.get("run_id") or "") != expected_run_id:
            job_reasons.append("run_id_graph_case_mismatch")
        campaign_root = _canonical_smiles(
            metadata.get("campaign_root_smiles")
        )
        if campaign_root != expected_target_smiles:
            job_reasons.append("campaign_root_graph_target_mismatch")
        if not _valid_sha256(metadata.get("campaign_identity_sha256")):
            job_reasons.append("campaign_identity_sha256_invalid")
        if job_reasons:
            reasons.extend(f"frontier_job:{job_id or index}:{reason}" for reason in job_reasons)
            continue
        seen_ids.add(job_id)
        jobs.append(
            {
                "job_id": job_id,
                "frontier_smiles": smiles,
                "state": state,
                "closure_kind": str(job.get("closure_kind") or ""),
                "achieved_proof_level": achieved,
                "result_ref": str(job.get("result_ref") or ""),
                "dependency_ids": sorted({str(item) for item in dependencies if str(item)}),
                "metadata": dict(metadata),
            }
        )
    if source_kind == "frontier_queue_snapshot" and reasons:
        # A snapshot is one atomic authority envelope.  Do not consume a
        # partially valid subset after its digest or nested schema failed.
        jobs = []
    jobs.sort(key=lambda item: str(item["job_id"]))
    return jobs, {
        "valid": not reasons,
        "source_kind": source_kind,
        "accepted_job_count": len(jobs),
        "reasons": sorted(set(reasons)),
    }


def _normalize_proof_state(
    value: Any,
    *,
    graph: Mapping[str, Any],
    required_level: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    row = dict(value) if isinstance(value, Mapping) else {}
    envelope_reasons: list[str] = []
    if row.get("schema_version") != REACTION_PROOF_STATE_SCHEMA:
        envelope_reasons.append("invalid_reaction_proof_state_schema")
    digest_payload = dict(row)
    supplied_digest = str(digest_payload.pop("content_sha256", ""))
    if not supplied_digest or supplied_digest != _digest(digest_payload):
        envelope_reasons.append("reaction_proof_state_digest_invalid")
    expected_graph_identity = _reaction_graph_identity(graph)
    if str(row.get("graph_identity_sha256") or "") != expected_graph_identity:
        envelope_reasons.append("reaction_proof_state_graph_identity_mismatch")
    if not isinstance(row.get("records"), list):
        envelope_reasons.append("reaction_proof_state_records_not_list")

    records_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_records: list[dict[str, Any]] = []
    if not envelope_reasons:
        edges = dict(graph.get("edges") or {})
        for index, raw in enumerate(row.get("records") or []):
            record = dict(raw) if isinstance(raw, Mapping) else {}
            signature = exact_edge_signature(
                record.get("product_smiles"),
                record.get("precursor_smiles") or [],
            )
            edge = edges.get(signature)
            reasons = _proof_record_reasons(
                record,
                edge=edge,
                required_level=required_level,
            )
            if reasons:
                rejected_records.append(
                    {
                        "index": index,
                        "signature": signature,
                        "reasons": reasons,
                    }
                )
                continue
            records_by_signature[signature].append(record)
    for records in records_by_signature.values():
        records.sort(key=lambda item: str(item.get("proof_request_id") or ""))
    return dict(records_by_signature), {
        "valid": not envelope_reasons and not rejected_records,
        "envelope_valid": not envelope_reasons,
        "records_valid": not rejected_records,
        "accepted_record_count": sum(len(rows) for rows in records_by_signature.values()),
        "rejected_record_count": len(rejected_records),
        "rejected_records": rejected_records,
        "reasons": sorted(set(envelope_reasons)),
    }


def _proof_record_reasons(
    record: Mapping[str, Any],
    *,
    edge: Mapping[str, Any] | None,
    required_level: int,
) -> list[str]:
    reasons: list[str] = []
    if record.get("schema_version") != REACTION_PROOF_RECORD_SCHEMA:
        reasons.append("invalid_record_schema")
    status = str(record.get("status") or "")
    if status not in {"pending", "rejected", "validated"}:
        reasons.append("invalid_record_status")
    if edge is None:
        reasons.append("record_edge_signature_not_in_graph")
        return sorted(set(reasons))
    if str(record.get("step_id") or "") not in edge.get("step_ids", []):
        reasons.append("record_step_id_mismatch")
    if str(record.get("signature") or "") not in edge.get(
        "source_graph_signatures", []
    ):
        reasons.append("record_source_graph_signature_mismatch")
    if _canonical_smiles(record.get("product_smiles")) != edge.get("product_smiles"):
        reasons.append("record_product_mismatch")
    record_precursors = sorted(
        _canonical_smiles(item) for item in record.get("precursor_smiles") or []
    )
    if record_precursors != edge.get("precursor_smiles"):
        reasons.append("record_precursors_mismatch")
    if status != "validated":
        try:
            achieved = int(record.get("achieved_proof_level") or 0)
        except (TypeError, ValueError):
            achieved = -1
        if achieved != 0:
            reasons.append("open_record_claims_achieved_proof")
        return sorted(set(reasons))
    if record.get("proof_authority") != "current_host_verifier_replay":
        reasons.append("record_not_current_host_replay")
    materialized = record.get("materialized_candidate")
    if not isinstance(materialized, Mapping) or not materialized:
        reasons.append("record_materialized_candidate_missing")
    else:
        if str(record.get("materialized_candidate_sha256") or "") != _digest(
            dict(materialized)
        ):
            reasons.append("record_materialized_candidate_digest_invalid")
        if str(materialized.get("step_id") or "") not in edge.get("step_ids", []):
            reasons.append("record_materialized_candidate_step_id_mismatch")
        if _canonical_smiles(materialized.get("product_smiles")) != edge.get(
            "product_smiles"
        ):
            reasons.append("record_materialized_candidate_product_mismatch")
        materialized_precursors = sorted(
            _canonical_smiles(item)
            for item in materialized.get("reactant_smiles") or []
        )
        if materialized_precursors != edge.get("precursor_smiles"):
            reasons.append("record_materialized_candidate_precursors_mismatch")
    proof = record.get("proof")
    if not isinstance(proof, Mapping):
        reasons.append("record_proof_missing")
        return sorted(set(reasons))
    proof_row = dict(proof)
    proof_digest = str(proof_row.pop("proof_digest", ""))
    if not proof_digest or proof_digest != _digest(proof_row):
        reasons.append("record_proof_digest_invalid")
    if proof.get("schema_version") != REACTION_STEP_PROOF_SCHEMA:
        reasons.append("record_proof_schema_invalid")
    if proof.get("validator_version") != REACTION_STEP_VERIFIER_VERSION:
        reasons.append("record_proof_verifier_version_invalid")
    if proof.get("accepted") is not True:
        reasons.append("record_proof_not_accepted")
    if str(proof.get("step_id") or "") not in edge.get("step_ids", []):
        reasons.append("record_proof_step_id_mismatch")
    level = _REACTION_LEVELS.get(str(proof.get("proof_level") or ""), 0)
    if level < required_level:
        reasons.append("record_proof_below_required_level")
    try:
        recorded_level = int(record.get("achieved_proof_level") or 0)
    except (TypeError, ValueError):
        recorded_level = 0
    # The durable reconciliation record historically stores the closure
    # threshold (L2) while the nested host proof can carry a stronger L3/L4
    # level.  The nested proof is the level authority; the record must only
    # attest that reaction validation was reached at all.
    if recorded_level < 2:
        reasons.append("record_achieved_level_mismatch")
    if _canonical_smiles(proof.get("product_smiles")) != edge.get("product_smiles"):
        reasons.append("record_proof_product_mismatch")
    proof_precursors = sorted(
        _canonical_smiles(item) for item in proof.get("reactant_smiles") or []
    )
    if proof_precursors != edge.get("precursor_smiles"):
        reasons.append("record_proof_precursors_mismatch")
    if isinstance(materialized, Mapping) and materialized and isinstance(proof, Mapping):
        try:
            replay_step_index = int(proof.get("step_index") or 0)
        except (TypeError, ValueError):
            replay_step_index = 0
            reasons.append("record_proof_step_index_invalid")
        supplied_checks = (
            dict(proof.get("checks") or {})
            if isinstance(proof.get("checks"), Mapping)
            else {}
        )
        try:
            replayed = verify_reaction_step(
                dict(materialized),
                step_index=replay_step_index,
                graph_and_stock_closed=(
                    supplied_checks.get("graph_and_stock_closed") is True
                ),
            )
        except Exception as exc:  # fail closed at the verifier boundary
            replayed = {}
            reasons.append(
                f"record_current_host_replay_error:{type(exc).__name__}"
            )
        if not replayed or _digest(dict(proof)) != _digest(replayed):
            reasons.append("record_proof_not_equal_to_current_host_replay")
    return sorted(set(reasons))


def _reachable_hypergraph(graph: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    root = str(graph.get("target_smiles") or "")
    by_product: dict[str, list[str]] = defaultdict(list)
    edges = dict(graph.get("edges") or {})
    for signature, edge in edges.items():
        by_product[str(edge["product_smiles"])].append(str(signature))
    molecules: set[str] = {root} if root else set()
    reachable_edges: set[str] = set()
    queue: deque[str] = deque([root] if root else [])
    visited: set[str] = set()
    while queue:
        molecule = queue.popleft()
        if molecule in visited:
            continue
        visited.add(molecule)
        for signature in sorted(by_product.get(molecule, [])):
            reachable_edges.add(signature)
            for precursor in edges[signature]["precursor_smiles"]:
                molecules.add(str(precursor))
                if precursor not in visited:
                    queue.append(str(precursor))
    return molecules, reachable_edges


def _stock_projection(
    smiles: str,
    jobs: list[dict[str, Any]],
    *,
    trusted_stock_provider_instances: Mapping[str, Any] | None,
) -> dict[str, Any]:
    accepted: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    accepted_observation_ids: set[str] = set()
    rejected_job_ids: list[str] = []
    rejection_reasons: dict[str, list[str]] = {}
    observed_job_ids: list[str] = []
    current_observation_ids: set[str] = set()
    history_observation_ids: set[str] = set()
    try:
        expected_provider_set = (
            stock_provider_set_authority_binding(
                trusted_stock_provider_instances or {}
            )
            if trusted_stock_provider_instances
            else {}
        )
    except (TypeError, ValueError):
        expected_provider_set = {}
    for job in jobs:
        job_id = str(job["job_id"])
        observation_state = job["metadata"].get("stock_observations")
        if not isinstance(observation_state, Mapping):
            if job["closure_kind"] == "stock_boundary":
                rejected_job_ids.append(job_id)
                rejection_reasons[job_id] = [
                    "legacy_stock_closure_missing_orthogonal_observations"
                ]
            continue
        state = dict(observation_state)
        observed_job_ids.append(job_id)
        current_rows = [
            dict(row)
            for row in state.get("current") or []
            if isinstance(row, Mapping)
        ]
        history_rows = [
            dict(row)
            for row in state.get("history") or []
            if isinstance(row, Mapping)
        ]
        current_observation_ids.update(
            str(row.get("observation_id") or "") for row in current_rows
        )
        history_observation_ids.update(
            str(row.get("observation_id") or "") for row in history_rows
        )
        state_reasons = validate_stock_observation_state(
            state,
            expected_smiles=smiles,
        )
        positive_rows = [
            row
            for row in current_rows
            if isinstance(row.get("provider_result"), Mapping)
            and row["provider_result"].get("accepted") is True
        ]
        if state_reasons:
            rejected_job_ids.append(job_id)
            rejection_reasons[job_id] = sorted(
                {f"stock_observation_state:{reason}" for reason in state_reasons}
            )
            continue
        if positive_rows and (
            not expected_provider_set
            or state.get("provider_set_binding") != expected_provider_set
        ):
            rejected_job_ids.append(job_id)
            rejection_reasons[job_id] = [
                "stock_observation_provider_set_not_current_host_policy"
            ]
            continue
        job_replay_reasons: list[str] = []
        for observation in positive_rows:
            audit = dict(observation.get("provider_result") or {})
            payload = dict(audit.get("payload") or {})
            envelope_reasons = validate_provider_result(audit)
            if (
                envelope_reasons
                or audit.get("provider_kind") != "stock"
                or audit.get("output_schema") != "stock_boundary.v1"
                or payload.get("schema_version") != "stock_boundary.v1"
                or payload.get("accepted") is not True
                or _canonical_smiles(payload.get("canonical_smiles")) != smiles
                or payload.get("boundary_type") not in _STOCK_BOUNDARY_TYPES
            ):
                job_replay_reasons.extend(
                    envelope_reasons
                    or ["stock_observation_positive_envelope_invalid"]
                )
                continue
            binding, replay_reasons = replay_stock_provider_result(
                audit,
                expected_smiles=smiles,
                trusted_provider_instances=trusted_stock_provider_instances,
            )
            if replay_reasons or not binding:
                job_replay_reasons.extend(replay_reasons)
                continue
            observation_id = str(observation.get("observation_id") or "")
            if observation_id in accepted_observation_ids:
                continue
            accepted_observation_ids.add(observation_id)
            boundary_type = str(payload.get("boundary_type") or "")
            proof_level = 0 if boundary_type == "benchmark_stock" else 4
            accepted.append(
                (
                    job,
                    payload,
                    {
                        **binding,
                        "job_id": job_id,
                        "observation_id": observation_id,
                        "achieved_proof_level": proof_level,
                        "boundary_type": boundary_type,
                    },
                )
            )
        if job_replay_reasons:
            rejected_job_ids.append(job_id)
            rejection_reasons[job_id] = sorted(set(job_replay_reasons))
    boundary_types = sorted(
        {str(payload["boundary_type"]) for _, payload, _ in accepted}
    )
    benchmark_membership_closed = "benchmark_stock" in boundary_types
    procurement_boundary_closed = any(
        payload["boundary_type"]
        in {
            "commercially_orderable",
            "in_house_available",
            "common_commodity",
        }
        for job, payload, _ in accepted
    )
    return {
        "closed": bool(accepted),
        "search_boundary_closed": bool(accepted),
        "benchmark_search_boundary_closed": bool(accepted),
        "closure_job_ids": sorted(
            {str(job["job_id"]) for job, _, _ in accepted}
        ),
        "observation_job_ids": sorted(set(observed_job_ids)),
        "current_observation_ids": sorted(current_observation_ids - {""}),
        "history_observation_count": len(history_observation_ids - {""}),
        "boundary_types": boundary_types,
        "benchmark_membership_closed": benchmark_membership_closed,
        "benchmark_only": bool(accepted) and benchmark_membership_closed
        and not procurement_boundary_closed,
        "procurement_boundary_closed": procurement_boundary_closed,
        "commercial_orderability_closed": (
            "commercially_orderable" in boundary_types
        ),
        "host_replay_verified": bool(accepted),
        "closure_replay_bindings": [binding for _, _, binding in accepted],
        "rejected_stock_job_ids": sorted(set(rejected_job_ids)),
        "replay_rejection_reasons": {
            job_id: rejection_reasons[job_id]
            for job_id in sorted(rejection_reasons)
        },
    }


def _serialized_stock_replay_reasons(
    smiles: str,
    *,
    stock: Mapping[str, Any],
    trusted_stock_provider_instances: Mapping[str, Any] | None,
) -> list[str]:
    """Validate persisted stock semantics and optionally replay host authority."""

    reasons: list[str] = []
    bindings = stock.get("closure_replay_bindings")
    if not isinstance(bindings, list):
        bindings = []
        if stock.get("closed") is True:
            reasons.append(f"frontier_ledger_stock_bindings_missing:{smiles}")
    accepted_bindings: list[dict[str, Any]] = []
    for index, raw_binding in enumerate(bindings):
        prefix = f"frontier_ledger_stock_binding:{smiles}:{index}"
        if not isinstance(raw_binding, Mapping):
            reasons.append(f"{prefix}:not_object")
            continue
        binding = dict(raw_binding)
        provider_result = binding.get("provider_result")
        replay_request = binding.get("replay_request")
        if (
            binding.get("schema_version")
            != "stock_provider_host_replay_binding.v1"
            or binding.get("authority")
            != "current_host_stock_provider_replay"
            or binding.get("canonical_smiles") != smiles
            or not isinstance(provider_result, Mapping)
            or not isinstance(replay_request, Mapping)
        ):
            reasons.append(f"{prefix}:identity_invalid")
            continue
        result = dict(provider_result)
        payload = result.get("payload")
        if not isinstance(payload, Mapping):
            reasons.append(f"{prefix}:payload_not_object")
            continue
        payload_row = dict(payload)
        boundary_type = str(payload_row.get("boundary_type") or "")
        provider_classes = {
            SnapshotStockProvider.descriptor.provider_id: SnapshotStockProvider,
            BenchmarkCatalogStockProvider.descriptor.provider_id: (
                BenchmarkCatalogStockProvider
            ),
        }
        provider_class = provider_classes.get(str(result.get("provider_id") or ""))
        descriptor = provider_class.descriptor if provider_class is not None else None
        try:
            achieved_level = int(binding.get("achieved_proof_level") or 0)
        except (TypeError, ValueError):
            achieved_level = -1
        expected_level_valid = (
            boundary_type == "benchmark_stock" and achieved_level == 0
        ) or (
            boundary_type
            in {
                "commercially_orderable",
                "in_house_available",
                "common_commodity",
            }
            and achieved_level == 4
        )
        binding_valid = bool(
            binding.get("boundary_type") == boundary_type
            and str(binding.get("provider_result_content_hash") or "")
            == str(result.get("content_hash") or "")
            and str(binding.get("replay_request_sha256") or "")
            == _digest(dict(replay_request))
            and descriptor is not None
            and binding.get("provider_version") == descriptor.version
            and binding.get("provider_descriptor_sha256")
            == _digest(descriptor.to_dict())
            and not validate_provider_result(result, descriptor=descriptor)
            and result.get("accepted") is True
            and payload_row.get("accepted") is True
            and _canonical_smiles(payload_row.get("canonical_smiles")) == smiles
            and boundary_type in _STOCK_BOUNDARY_TYPES
            and expected_level_valid
            and str(binding.get("job_id") or "")
        )
        if not binding_valid:
            reasons.append(f"{prefix}:content_invalid")
            continue
        if trusted_stock_provider_instances is not None:
            replayed_binding, replay_reasons = replay_stock_provider_result(
                result,
                expected_smiles=smiles,
                trusted_provider_instances=trusted_stock_provider_instances,
            )
            supplied_core = {
                key: value
                for key, value in binding.items()
                if key
                not in {
                    "job_id",
                    "observation_id",
                    "achieved_proof_level",
                    "boundary_type",
                }
            }
            if replay_reasons or replayed_binding != supplied_core:
                reasons.append(f"{prefix}:current_host_replay_failed")
                continue
        accepted_bindings.append(binding)

    boundary_types = sorted(
        {str(binding.get("boundary_type") or "") for binding in accepted_bindings}
    )
    job_ids = sorted(
        {str(binding.get("job_id") or "") for binding in accepted_bindings}
    )
    benchmark_closed = "benchmark_stock" in boundary_types
    procurement_closed = any(
        int(binding.get("achieved_proof_level") or 0) == 4
        and binding.get("boundary_type")
        in {
            "commercially_orderable",
            "in_house_available",
            "common_commodity",
        }
        for binding in accepted_bindings
    )
    expected_fields = {
        "closed": bool(accepted_bindings),
        "search_boundary_closed": bool(accepted_bindings),
        "benchmark_search_boundary_closed": bool(accepted_bindings),
        "closure_job_ids": job_ids,
        "boundary_types": boundary_types,
        "benchmark_membership_closed": benchmark_closed,
        "benchmark_only": bool(accepted_bindings)
        and benchmark_closed
        and not procurement_closed,
        "procurement_boundary_closed": procurement_closed,
        "commercial_orderability_closed": (
            "commercially_orderable" in boundary_types
        ),
        "host_replay_verified": bool(accepted_bindings),
    }
    for field, expected in expected_fields.items():
        supplied = stock.get(field)
        if supplied != expected:
            reasons.append(f"frontier_ledger_stock_semantics_mismatch:{smiles}:{field}")
    return reasons


def _terminal_reaction_closure_projection(
    jobs: list[dict[str, Any]],
    *,
    required_level: int,
) -> dict[str, Any]:
    legacy_claims = [
        job
        for job in jobs
        if job["state"] == "succeeded"
        and job["closure_kind"] in {"reaction_route", "verified_precedent"}
        and job["achieved_proof_level"] >= required_level
        and bool(job["result_ref"])
    ]
    return {
        "closed": False,
        "closure_job_ids": [],
        "best_proof_level": 0,
        "rejected_legacy_work_claim_job_ids": [
            str(job["job_id"]) for job in legacy_claims
        ],
        "reason": "queue_work_cannot_authorize_terminal_reaction_closure",
    }


def _edge_proof_projection(
    signature: str,
    *,
    edge: Mapping[str, Any],
    records: list[dict[str, Any]],
    required_level: int,
) -> dict[str, Any]:
    validated = [record for record in records if record.get("status") == "validated"]
    best = max(
        validated,
        key=lambda record: _REACTION_LEVELS.get(
            str((record.get("proof") or {}).get("proof_level") or ""), 0
        ),
        default=None,
    )
    best_level = (
        _REACTION_LEVELS.get(
            str((best.get("proof") or {}).get("proof_level") or ""), 0
        )
        if best
        else 0
    )
    return {
        "closed": bool(best and best_level >= required_level),
        "required_proof_level": required_level,
        "achieved_proof_level": best_level,
        "proof_level": str((best or {}).get("proof", {}).get("proof_level") or ""),
        "authority": str((best or {}).get("proof_authority") or "none"),
        "proof_request_ids": sorted(
            {
                str(record.get("proof_request_id") or "")
                for record in records
                if str(record.get("proof_request_id") or "")
            }
        ),
        "step_ids": list(edge["step_ids"]),
        "exact_edge_signature": signature,
        "host_replay_binding": (
            {
                "schema_version": "frontier_ledger_host_replay_binding.v1",
                "proof_request_id": str(best.get("proof_request_id") or ""),
                "materialized_candidate": dict(
                    best.get("materialized_candidate") or {}
                ),
                "materialized_candidate_sha256": str(
                    best.get("materialized_candidate_sha256") or ""
                ),
                "proof": dict(best.get("proof") or {}),
                "proof_authority": str(best.get("proof_authority") or ""),
            }
            if best
            else {}
        ),
    }


def _closure_fixed_point(
    *,
    molecules: set[str],
    outgoing: Mapping[str, list[str]],
    edges: Mapping[str, Mapping[str, Any]],
    proofs: Mapping[str, Mapping[str, Any]],
    stock: Mapping[str, Mapping[str, Any]],
    terminal_closure: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bool], dict[str, bool], int]:
    # Least fixed point: cycles without an independently closed boundary never
    # become true merely by referring back to themselves.
    any_closed = {
        molecule: bool(stock[molecule]["closed"] or terminal_closure[molecule]["closed"])
        for molecule in molecules
    }
    all_closed = {
        molecule: bool(
            not outgoing.get(molecule)
            and (stock[molecule]["closed"] or terminal_closure[molecule]["closed"])
        )
        for molecule in molecules
    }
    iterations = 0
    while True:
        iterations += 1
        next_any = dict(any_closed)
        next_all = dict(all_closed)
        for molecule in sorted(molecules):
            alternatives = outgoing.get(molecule, [])
            if alternatives:
                next_any[molecule] = bool(
                    any_closed[molecule]
                    or any(
                        proofs[signature]["closed"]
                        and all(
                            any_closed.get(str(precursor), False)
                            for precursor in edges[signature]["precursor_smiles"]
                        )
                        for signature in alternatives
                    )
                )
                next_all[molecule] = bool(
                    all(
                        proofs[signature]["closed"]
                        and all(
                            all_closed.get(str(precursor), False)
                            for precursor in edges[signature]["precursor_smiles"]
                        )
                        for signature in alternatives
                    )
                )
        if next_any == any_closed and next_all == all_closed:
            return next_any, next_all, iterations
        any_closed, all_closed = next_any, next_all


def _reaction_graph_identity(graph: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": str(graph.get("schema_version") or ""),
        "case_id": str(graph.get("case_id") or ""),
        "target_smiles": str(graph.get("target_smiles") or ""),
        "steps": sorted(
            list(graph.get("identity_steps") or []),
            key=lambda item: (str(item.get("step_id") or ""), str(item.get("signature") or "")),
        ),
    }
    return _digest(payload)


def _canonical_smiles(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _route_graph_edge_signature(product_smiles: Any, precursor_smiles: Iterable[Any]) -> str:
    """Reproduce the legacy graph-v1 signature for strict input validation."""

    product = _canonical_smiles(product_smiles)
    precursors = sorted(
        value
        for value in (_canonical_smiles(item) for item in precursor_smiles)
        if value
    )
    if not product or not precursors:
        return ""
    return f"{product}<-{'.'.join(precursors)}"


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_roundtrip(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)

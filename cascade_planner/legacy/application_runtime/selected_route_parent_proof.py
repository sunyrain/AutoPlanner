"""Deterministic parent proof compiled from the authoritative route graph.

This module deliberately does not consume persisted portfolio ``accepted`` or
``solved`` flags.  It re-solves the current hypergraph against exact edge and
stock bindings and emits content-addressed route witnesses.  Validation is a
full deterministic recompile and comparison, not a check of producer booleans.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import re
from typing import Any


SELECTED_ROUTE_PARENT_PROOF_SCHEMA = "selected_route_parent_proof.v1"
SELECTED_ROUTE_WITNESS_SCHEMA = "selected_route_witness.v1"
SELECTED_ROUTE_PROOF_COMPILER_VERSION = "selected-route-proof-compiler.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROOF_LEVELS = {
    "L0_materialized": 0,
    "L1_graph_and_stock_closed": 1,
    "L2_reaction_validated": 2,
    "L3_precedent_supported": 3,
    "L4_procurement_ready": 4,
}
_EDGE_PROOF_SOURCES = {
    "route_proof_bank.v1",
    "legacy_best_accepted_route",
    "supplemental_reaction_validation.v2_replayed",
}
_STOCK_AUTHORITIES = {
    "strictly_replayed_route_proof_bank.v1",
    "legacy_best_route_independent_stock_audit",
    "verified_stock_provider_envelope",
}


@dataclass(slots=True)
class _Candidate:
    selections: dict[str, str] = field(default_factory=dict)
    leaves: set[str] = field(default_factory=set)
    molecules: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _SearchState:
    max_routes: int
    max_depth: int
    truncated: bool = False
    depth_limited: bool = False


def compile_selected_route_parent_proof(
    route_hypergraph_overlay: Mapping[str, Any],
    route_portfolio_bindings: Mapping[str, Any],
    *,
    expected_overlay_sha256: str = "",
    expected_bindings_sha256: str = "",
    campaign_binding: Mapping[str, Any] | None = None,
    frontier_ledger: Mapping[str, Any] | None = None,
    expected_campaign_revision: int | None = None,
    minimum_complete_routes: int = 2,
    minimum_edge_proof_level: int = 3,
    max_depth: int = 32,
    max_enumerated_routes: int = 4096,
) -> dict[str, Any]:
    """Re-solve and compile a fail-closed selected-route parent proof.

    ``benchmark_solved`` means that at least ``minimum_complete_routes``
    distinct, non-zero-step, L3-or-better synthesis DAGs close at exact stock
    bindings.  ``procurement_ready`` is deliberately stronger: the same route
    count must be L4 on every edge and commercially orderable on every leaf.

    Optional campaign/frontier inputs are only revision/digest bindings.  Their
    persisted completion flags are never consulted.
    """

    overlay = dict(route_hypergraph_overlay or {})
    bindings = dict(route_portfolio_bindings or {})
    reasons: list[str] = []
    policy_reasons = _policy_reasons(
        minimum_complete_routes=minimum_complete_routes,
        minimum_edge_proof_level=minimum_edge_proof_level,
        max_depth=max_depth,
        max_enumerated_routes=max_enumerated_routes,
    )
    reasons.extend(policy_reasons)

    overlay_sha256, overlay_digest_reasons = _artifact_digest(
        overlay,
        label="route_hypergraph_overlay",
        require_supplied_digest=False,
    )
    bindings_sha256, bindings_digest_reasons = _artifact_digest(
        bindings,
        label="route_portfolio_bindings",
        require_supplied_digest=True,
    )
    reasons.extend(overlay_digest_reasons)
    reasons.extend(bindings_digest_reasons)
    reasons.extend(
        _expected_digest_reasons(
            expected_overlay_sha256,
            actual=overlay_sha256,
            label="route_hypergraph_overlay",
        )
    )
    reasons.extend(
        _expected_digest_reasons(
            expected_bindings_sha256,
            actual=bindings_sha256,
            label="route_portfolio_bindings",
        )
    )

    molecules, edges, overlay_reasons = _validate_overlay(overlay)
    reasons.extend(overlay_reasons)
    bindings_reasons = _validate_bindings_container(bindings)
    reasons.extend(bindings_reasons)

    campaign_record, campaign_reasons = _bind_campaign_inputs(
        campaign_binding=campaign_binding,
        frontier_ledger=frontier_ledger,
        overlay_sha256=overlay_sha256,
        bindings_sha256=bindings_sha256,
        expected_campaign_revision=expected_campaign_revision,
    )
    reasons.extend(campaign_reasons)

    edge_levels = dict(bindings.get("edge_proof_levels") or {})
    edge_bindings = dict(bindings.get("exact_edge_proof_bindings") or {})
    eligible_edge_bindings: dict[str, dict[str, Any]] = {}
    rejected_edge_bindings: list[dict[str, Any]] = []
    if not policy_reasons and not overlay_reasons and not bindings_reasons:
        for edge_id, edge in sorted(edges.items()):
            binding = edge_bindings.get(edge_id)
            binding_reasons = _edge_binding_reasons(
                binding,
                edge=edge,
                molecules=molecules,
                declared_level=edge_levels.get(edge_id),
                minimum_level=minimum_edge_proof_level,
            )
            if binding_reasons:
                rejected_edge_bindings.append(
                    {"hyperedge_id": edge_id, "reasons": binding_reasons}
                )
            else:
                eligible_edge_bindings[edge_id] = dict(binding)

    stock_bindings = dict(bindings.get("stock_bindings") or {})
    declared_stock_ids = {
        str(value)
        for value in bindings.get("stock_molecule_ids") or []
        if str(value)
    }
    eligible_stock_bindings: dict[str, dict[str, Any]] = {}
    rejected_stock_bindings: list[dict[str, Any]] = []
    if not policy_reasons and not overlay_reasons and not bindings_reasons:
        for molecule_id in sorted(declared_stock_ids | set(stock_bindings)):
            binding = stock_bindings.get(molecule_id)
            binding_reasons = _stock_binding_reasons(
                binding,
                molecule_id=molecule_id,
                molecule=molecules.get(molecule_id),
                declared=molecule_id in declared_stock_ids,
            )
            if binding_reasons:
                rejected_stock_bindings.append(
                    {"molecule_id": molecule_id, "reasons": binding_reasons}
                )
            else:
                eligible_stock_bindings[molecule_id] = dict(binding)

    root_id = str(overlay.get("root_molecule_id") or "")
    # A target stock hit is never a synthetic route witness.
    eligible_stock_bindings.pop(root_id, None)
    search_state = _SearchState(
        max_routes=max_enumerated_routes,
        max_depth=max_depth,
    )
    candidates: list[_Candidate] = []
    if not reasons:
        candidates = _solve_routes(
            root_id=root_id,
            edges=edges,
            eligible_edges=eligible_edge_bindings,
            eligible_stock=eligible_stock_bindings,
            state=search_state,
        )

    witnesses_by_edge_set: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        witness = _route_witness(
            root_id=root_id,
            candidate=candidate,
            molecules=molecules,
            edges=edges,
            edge_bindings=eligible_edge_bindings,
            stock_bindings=eligible_stock_bindings,
        )
        if not witness:
            continue
        witnesses_by_edge_set.setdefault(str(witness["edge_set_sha256"]), witness)
    witnesses = sorted(
        witnesses_by_edge_set.values(),
        key=lambda row: (len(row["selected_hyperedges"]), str(row["route_id"])),
    )

    if search_state.truncated:
        reasons.append("route_enumeration_truncated")
    if search_state.depth_limited:
        reasons.append("route_enumeration_depth_limited")
    if not witnesses:
        reasons.append("no_complete_l3_stock_closed_synthesis_route")
    elif len(witnesses) < minimum_complete_routes:
        reasons.append("insufficient_distinct_complete_route_edge_sets")

    reasons = sorted(set(reasons))
    input_contract_valid = not any(
        reason
        for reason in reasons
        if reason
        not in {
            "no_complete_l3_stock_closed_synthesis_route",
            "insufficient_distinct_complete_route_edge_sets",
            "route_enumeration_truncated",
            "route_enumeration_depth_limited",
        }
    )
    benchmark_solved = bool(
        not reasons and len(witnesses) >= minimum_complete_routes
    )
    procurement_routes = [
        row for row in witnesses if row.get("procurement_ready") is True
    ]
    procurement_ready = bool(
        benchmark_solved and len(procurement_routes) >= minimum_complete_routes
    )
    root_smiles = str(
        (molecules.get(root_id) or {}).get("canonical_isomeric_smiles") or ""
    )
    payload: dict[str, Any] = {
        "schema_version": SELECTED_ROUTE_PARENT_PROOF_SCHEMA,
        "compiler_version": SELECTED_ROUTE_PROOF_COMPILER_VERSION,
        "authority": "deterministic_selected_route_parent_proof",
        "proof_mode": "full_hypergraph_resolve_with_exact_bindings",
        "root_molecule_id": root_id,
        "target_canonical_isomeric_smiles": root_smiles,
        "input_bindings": {
            "route_hypergraph_overlay_sha256": overlay_sha256,
            "route_portfolio_bindings_sha256": bindings_sha256,
            "campaign": campaign_record,
        },
        "policy": {
            "minimum_complete_routes": minimum_complete_routes,
            "minimum_edge_proof_level": minimum_edge_proof_level,
            "max_depth": max_depth,
            "max_enumerated_routes": max_enumerated_routes,
            "target_stock_zero_step_forbidden": True,
            "distinctness_basis": "selected_hyperedge_id_set",
            "benchmark_requires_all_leaves_exactly_stock_bound": True,
            "procurement_requires_every_edge_l4_and_every_leaf_commercial": True,
        },
        "input_contract_valid": input_contract_valid,
        "enumeration_complete": not (
            search_state.truncated or search_state.depth_limited
        ),
        "benchmark_solved": benchmark_solved,
        "solved": benchmark_solved,
        "procurement_ready": procurement_ready,
        "any_procurement_route_ready": bool(procurement_routes),
        "distinct_complete_route_count": len(witnesses),
        "procurement_route_count": len(procurement_routes),
        "routes": witnesses,
        "route_edge_set_sha256s": sorted(witnesses_by_edge_set),
        "binding_audit": {
            "eligible_edge_ids": sorted(eligible_edge_bindings),
            "rejected_edges": rejected_edge_bindings,
            "eligible_stock_molecule_ids": sorted(eligible_stock_bindings),
            "rejected_stock": rejected_stock_bindings,
        },
        "reasons": reasons,
        "producer_completion_flags_ignored": True,
        "validator_requires_full_recompile": True,
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def validate_selected_route_parent_proof(
    proof: Mapping[str, Any],
    route_hypergraph_overlay: Mapping[str, Any],
    route_portfolio_bindings: Mapping[str, Any],
    **compile_options: Any,
) -> list[str]:
    """Return validation reasons after a full deterministic recompile."""

    reasons: list[str] = []
    candidate = dict(proof or {})
    if candidate.get("schema_version") != SELECTED_ROUTE_PARENT_PROOF_SCHEMA:
        reasons.append("invalid_selected_route_parent_proof_schema")
    supplied_sha256 = str(candidate.get("content_sha256") or "").lower()
    unsigned = dict(candidate)
    unsigned.pop("content_sha256", None)
    if not _valid_sha256(supplied_sha256) or supplied_sha256 != _digest(unsigned):
        reasons.append("selected_route_parent_proof_content_sha256_mismatch")
    expected = compile_selected_route_parent_proof(
        route_hypergraph_overlay,
        route_portfolio_bindings,
        **compile_options,
    )
    if _canonical_json(candidate) != _canonical_json(expected):
        reasons.append("selected_route_parent_proof_full_recompile_mismatch")
    return sorted(set(reasons))


def is_solved_selected_route_parent_proof(
    proof: Mapping[str, Any],
    route_hypergraph_overlay: Mapping[str, Any],
    route_portfolio_bindings: Mapping[str, Any],
    **compile_options: Any,
) -> bool:
    """Return true only for a valid, fully recompiled benchmark solution."""

    if validate_selected_route_parent_proof(
        proof,
        route_hypergraph_overlay,
        route_portfolio_bindings,
        **compile_options,
    ):
        return False
    row = dict(proof or {})
    policy = dict(row.get("policy") or {})
    try:
        required = int(policy.get("minimum_complete_routes") or 0)
        route_count = int(row.get("distinct_complete_route_count") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        row.get("input_contract_valid") is True
        and row.get("enumeration_complete") is True
        and row.get("benchmark_solved") is True
        and row.get("solved") is True
        and required >= 2
        and route_count >= required
        and len(row.get("routes") or []) == route_count
        and not row.get("reasons")
    )


# Short API spellings for callers that import this module as a service.
compile = compile_selected_route_parent_proof
validate = validate_selected_route_parent_proof
is_solved = is_solved_selected_route_parent_proof


def _policy_reasons(
    *,
    minimum_complete_routes: int,
    minimum_edge_proof_level: int,
    max_depth: int,
    max_enumerated_routes: int,
) -> list[str]:
    reasons: list[str] = []
    if minimum_complete_routes < 2:
        reasons.append("minimum_complete_routes_cannot_be_below_two")
    if minimum_edge_proof_level < 3 or minimum_edge_proof_level > 4:
        reasons.append("minimum_edge_proof_level_must_be_l3_or_l4")
    if max_depth < 1:
        reasons.append("max_depth_must_be_positive")
    if max_enumerated_routes < 2:
        reasons.append("max_enumerated_routes_must_allow_two_routes")
    return reasons


def _validate_overlay(
    overlay: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    reasons: list[str] = []
    if overlay.get("schema_version") != "route_hypergraph_overlay.v2":
        reasons.append("invalid_route_hypergraph_overlay_schema")
    root_id = str(overlay.get("root_molecule_id") or "")
    molecules: dict[str, dict[str, Any]] = {}
    smiles_seen: dict[str, str] = {}
    raw_molecules = overlay.get("molecules")
    if not isinstance(raw_molecules, Sequence) or isinstance(raw_molecules, (str, bytes)):
        raw_molecules = []
        reasons.append("route_hypergraph_overlay_molecules_not_a_list")
    for raw in raw_molecules:
        if not isinstance(raw, Mapping):
            reasons.append("route_hypergraph_overlay_molecule_not_an_object")
            continue
        row = dict(raw)
        molecule_id = str(row.get("molecule_id") or "")
        smiles = str(row.get("canonical_isomeric_smiles") or "")
        if not molecule_id or not smiles:
            reasons.append("route_hypergraph_overlay_molecule_identity_missing")
            continue
        if molecule_id in molecules:
            reasons.append(f"duplicate_overlay_molecule_id:{molecule_id}")
            continue
        if smiles in smiles_seen and smiles_seen[smiles] != molecule_id:
            reasons.append(f"duplicate_overlay_canonical_smiles:{smiles}")
            continue
        molecules[molecule_id] = row
        smiles_seen[smiles] = molecule_id
    if not root_id or root_id not in molecules:
        reasons.append("route_hypergraph_overlay_root_missing")

    edges: dict[str, dict[str, Any]] = {}
    raw_edges = overlay.get("reaction_hyperedges")
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        raw_edges = []
        reasons.append("route_hypergraph_overlay_edges_not_a_list")
    for raw in raw_edges:
        if not isinstance(raw, Mapping):
            reasons.append("route_hypergraph_overlay_edge_not_an_object")
            continue
        row = dict(raw)
        edge_id = str(row.get("hyperedge_id") or "")
        product_id = str(row.get("product_molecule_id") or "")
        precursor_ids = [str(value) for value in row.get("precursor_molecule_ids") or []]
        if not edge_id or not product_id or not precursor_ids or any(not value for value in precursor_ids):
            reasons.append(f"invalid_overlay_hyperedge_identity:{edge_id or '<missing>'}")
            continue
        if edge_id in edges:
            reasons.append(f"duplicate_overlay_hyperedge_id:{edge_id}")
            continue
        if len(precursor_ids) != len(set(precursor_ids)):
            reasons.append(f"duplicate_overlay_hyperedge_precursor:{edge_id}")
        if product_id not in molecules:
            reasons.append(f"overlay_hyperedge_product_missing:{edge_id}")
        for precursor_id in precursor_ids:
            if precursor_id not in molecules:
                reasons.append(f"overlay_hyperedge_precursor_missing:{edge_id}:{precursor_id}")
        edges[edge_id] = row
    if root_id and not any(
        str(edge.get("product_molecule_id") or "") == root_id
        for edge in edges.values()
    ):
        reasons.append("route_hypergraph_overlay_root_has_no_synthesis_edge")
    return molecules, edges, sorted(set(reasons))


def _validate_bindings_container(bindings: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if bindings.get("schema_version") != "route_portfolio_bindings.v1":
        reasons.append("invalid_route_portfolio_bindings_schema")
    if not isinstance(bindings.get("edge_proof_levels"), Mapping):
        reasons.append("route_portfolio_edge_proof_levels_not_an_object")
    if not isinstance(bindings.get("exact_edge_proof_bindings"), Mapping):
        reasons.append("route_portfolio_edge_bindings_not_an_object")
    if not isinstance(bindings.get("stock_bindings"), Mapping):
        reasons.append("route_portfolio_stock_bindings_not_an_object")
    stock_ids = bindings.get("stock_molecule_ids")
    if not isinstance(stock_ids, Sequence) or isinstance(stock_ids, (str, bytes)):
        reasons.append("route_portfolio_stock_molecule_ids_not_a_list")
    return reasons


def _edge_binding_reasons(
    value: Any,
    *,
    edge: Mapping[str, Any],
    molecules: Mapping[str, Mapping[str, Any]],
    declared_level: Any,
    minimum_level: int,
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["missing_exact_edge_proof_binding"]
    binding = dict(value)
    reasons: list[str] = []
    edge_id = str(edge.get("hyperedge_id") or "")
    product_id = str(edge.get("product_molecule_id") or "")
    precursor_ids = sorted(str(value) for value in edge.get("precursor_molecule_ids") or [])
    if binding.get("schema_version") != "exact_edge_proof_binding.v1":
        reasons.append("invalid_edge_binding_schema")
    if str(binding.get("hyperedge_id") or "") != edge_id:
        reasons.append("edge_binding_hyperedge_id_mismatch")
    if str(binding.get("product_molecule_id") or "") != product_id:
        reasons.append("edge_binding_product_id_mismatch")
    if sorted(str(item) for item in binding.get("precursor_molecule_ids") or []) != precursor_ids:
        reasons.append("edge_binding_precursor_ids_mismatch")
    if not _named_digest_valid(binding, "binding_sha256"):
        reasons.append("edge_binding_sha256_mismatch")

    signature = _digest(
        {
            "product_canonical_isomeric_smiles": str(
                (molecules.get(product_id) or {}).get("canonical_isomeric_smiles") or ""
            ),
            "reactant_canonical_isomeric_smiles": sorted(
                str((molecules.get(item) or {}).get("canonical_isomeric_smiles") or "")
                for item in precursor_ids
            ),
        }
    )
    if str(binding.get("structure_signature_sha256") or "").lower() != signature:
        reasons.append("edge_binding_structure_signature_mismatch")
    if str(binding.get("reaction_digest") or "").lower() != signature:
        reasons.append("edge_binding_reaction_digest_mismatch")
    level = _proof_level(binding.get("portfolio_proof_level"))
    if _proof_level(declared_level) != level:
        reasons.append("edge_binding_declared_level_mismatch")
    named_level = str(binding.get("proof_level") or "")
    if _PROOF_LEVELS.get(named_level, -1) != level:
        reasons.append("edge_binding_named_level_mismatch")
    if level < minimum_level:
        reasons.append("edge_binding_below_required_l3")
    if binding.get("proof_accepted") is not True:
        reasons.append("edge_binding_proof_not_accepted")
    if binding.get("advisory") is not False:
        reasons.append("edge_binding_is_advisory")
    for field_name in (
        "proof_digest",
        "route_proof_digest",
        "trusted_precedent_sha256",
        "verifier_source_sha256",
    ):
        if not _valid_sha256(str(binding.get(field_name) or "")):
            reasons.append(f"invalid_edge_binding_{field_name}")
    proof_source = str(binding.get("proof_source") or "")
    if proof_source not in _EDGE_PROOF_SOURCES:
        reasons.append("edge_binding_proof_source_not_authoritative")
    if proof_source == "route_proof_bank.v1" and (
        not str(binding.get("proof_bank_entry_id") or "")
        or not _valid_sha256(str(binding.get("proof_bank_entry_sha256") or ""))
    ):
        reasons.append("edge_binding_proof_bank_authority_invalid")
    if not str(binding.get("validator_version") or ""):
        reasons.append("edge_binding_validator_version_missing")
    return sorted(set(reasons))


def _stock_binding_reasons(
    value: Any,
    *,
    molecule_id: str,
    molecule: Mapping[str, Any] | None,
    declared: bool,
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["missing_exact_stock_binding"]
    binding = dict(value)
    reasons: list[str] = []
    if not declared:
        reasons.append("stock_binding_not_declared_in_stock_molecule_ids")
    if molecule is None:
        reasons.append("stock_binding_molecule_not_in_overlay")
        expected_smiles = ""
    else:
        expected_smiles = str(molecule.get("canonical_isomeric_smiles") or "")
    if binding.get("schema_version") != "exact_stock_binding.v1":
        reasons.append("invalid_stock_binding_schema")
    if str(binding.get("molecule_id") or "") != molecule_id:
        reasons.append("stock_binding_molecule_id_mismatch")
    if str(binding.get("canonical_isomeric_smiles") or "") != expected_smiles:
        reasons.append("stock_binding_canonical_smiles_mismatch")
    if not _named_digest_valid(binding, "binding_sha256"):
        reasons.append("stock_binding_sha256_mismatch")
    if not str(binding.get("catalog_id") or ""):
        reasons.append("stock_binding_catalog_id_missing")
    for field_name in ("catalog_sha256", "evidence_sha256"):
        if not _valid_sha256(str(binding.get(field_name) or "")):
            reasons.append(f"invalid_stock_binding_{field_name}")
    if not str(binding.get("lookup_basis") or ""):
        reasons.append("stock_binding_lookup_basis_missing")
    if str(binding.get("binding_authority") or "") not in _STOCK_AUTHORITIES:
        reasons.append("stock_binding_authority_not_replayable")

    boundary_type = str(binding.get("boundary_type") or "")
    if boundary_type == "benchmark_stock":
        if binding.get("benchmark_membership") is not True:
            reasons.append("benchmark_stock_membership_not_proven")
        if binding.get("commercial_orderability_claimed") is True:
            reasons.append("benchmark_stock_cannot_claim_commercial_orderability")
    elif boundary_type == "commercially_orderable":
        if binding.get("commercial_orderability_claimed") is not True:
            reasons.append("commercial_orderability_not_claimed")
        if binding.get("snapshot_digest_replayed") is not True:
            reasons.append("commercial_snapshot_digest_not_replayed")
        if binding.get("provider_trust_authority") != "autoplanner_host_builtin_allowlist.v1":
            reasons.append("commercial_stock_provider_not_host_trusted")
        if not str(binding.get("provider_id") or ""):
            reasons.append("commercial_stock_provider_id_missing")
        if not _valid_sha256(str(binding.get("provider_descriptor_sha256") or "")):
            reasons.append("commercial_stock_provider_descriptor_invalid")
    else:
        reasons.append("stock_binding_boundary_not_benchmark_or_commercial")
    return sorted(set(reasons))


def _solve_routes(
    *,
    root_id: str,
    edges: Mapping[str, Mapping[str, Any]],
    eligible_edges: Mapping[str, Mapping[str, Any]],
    eligible_stock: Mapping[str, Mapping[str, Any]],
    state: _SearchState,
) -> list[_Candidate]:
    by_product: dict[str, list[dict[str, Any]]] = {}
    for edge_id in sorted(eligible_edges):
        edge = dict(edges[edge_id])
        by_product.setdefault(str(edge.get("product_molecule_id") or ""), []).append(edge)
    for rows in by_product.values():
        rows.sort(key=lambda row: str(row.get("hyperedge_id") or ""))

    def expand(molecule_id: str, depth: int, ancestors: frozenset[str]) -> list[_Candidate]:
        if molecule_id in eligible_stock:
            return [_Candidate(leaves={molecule_id}, molecules={molecule_id})]
        if molecule_id in ancestors:
            return []
        if depth >= state.max_depth:
            if by_product.get(molecule_id):
                state.depth_limited = True
            return []
        results: list[_Candidate] = []
        for edge in by_product.get(molecule_id, []):
            precursor_ids = [str(value) for value in edge.get("precursor_molecule_ids") or []]
            branches = [
                expand(precursor_id, depth + 1, ancestors | {molecule_id})
                for precursor_id in precursor_ids
            ]
            if any(not branch for branch in branches):
                continue
            for combination in itertools.product(*branches):
                merged = _merge_candidates(
                    combination,
                    product_id=molecule_id,
                    edge_id=str(edge.get("hyperedge_id") or ""),
                )
                if merged is None or not _selection_is_acyclic(merged.selections, edges):
                    continue
                results.append(merged)
                results = _dedupe_candidates(results)
                if len(results) > state.max_routes:
                    state.truncated = True
                    return results[: state.max_routes]
        return _dedupe_candidates(results)

    return expand(root_id, 0, frozenset())


def _merge_candidates(
    rows: Sequence[_Candidate],
    *,
    product_id: str,
    edge_id: str,
) -> _Candidate | None:
    selections = {product_id: edge_id}
    leaves: set[str] = set()
    molecules = {product_id}
    for row in rows:
        for selected_product, selected_edge in row.selections.items():
            prior = selections.get(selected_product)
            if prior is not None and prior != selected_edge:
                return None
            selections[selected_product] = selected_edge
        leaves.update(row.leaves)
        molecules.update(row.molecules)
    if product_id in leaves:
        return None
    return _Candidate(selections=selections, leaves=leaves, molecules=molecules)


def _dedupe_candidates(rows: Sequence[_Candidate]) -> list[_Candidate]:
    by_identity: dict[str, _Candidate] = {}
    for row in rows:
        identity = _digest(
            {
                "selections": sorted(row.selections.items()),
                "leaves": sorted(row.leaves),
            }
        )
        by_identity.setdefault(identity, row)
    return [by_identity[key] for key in sorted(by_identity)]


def _selection_is_acyclic(
    selections: Mapping[str, str],
    edges: Mapping[str, Mapping[str, Any]],
) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(molecule_id: str) -> bool:
        if molecule_id in visiting:
            return False
        if molecule_id in visited or molecule_id not in selections:
            return True
        visiting.add(molecule_id)
        edge = edges.get(str(selections[molecule_id])) or {}
        if any(not visit(str(value)) for value in edge.get("precursor_molecule_ids") or []):
            return False
        visiting.remove(molecule_id)
        visited.add(molecule_id)
        return True

    return all(visit(product_id) for product_id in sorted(selections))


def _route_witness(
    *,
    root_id: str,
    candidate: _Candidate,
    molecules: Mapping[str, Mapping[str, Any]],
    edges: Mapping[str, Mapping[str, Any]],
    edge_bindings: Mapping[str, Mapping[str, Any]],
    stock_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not candidate.selections or root_id not in candidate.selections:
        return {}
    if not candidate.leaves or not _selection_is_acyclic(candidate.selections, edges):
        return {}
    selected_products = set(candidate.selections)
    reached_products: set[str] = set()
    reached_molecules: set[str] = set()
    ordered_edge_ids: list[str] = []

    def walk(molecule_id: str) -> bool:
        reached_molecules.add(molecule_id)
        edge_id = candidate.selections.get(molecule_id)
        if edge_id is None:
            return molecule_id in candidate.leaves
        if molecule_id in reached_products:
            return True
        reached_products.add(molecule_id)
        ordered_edge_ids.append(edge_id)
        edge = edges.get(edge_id)
        if edge is None:
            return False
        return all(walk(str(value)) for value in edge.get("precursor_molecule_ids") or [])

    if not walk(root_id) or reached_products != selected_products:
        return {}
    reached_leaves = reached_molecules - reached_products
    if reached_leaves != candidate.leaves:
        return {}

    selected_rows: list[dict[str, Any]] = []
    proof_levels: list[int] = []
    for edge_id in ordered_edge_ids:
        edge = dict(edges[edge_id])
        binding = dict(edge_bindings[edge_id])
        level = _proof_level(binding.get("portfolio_proof_level"))
        proof_levels.append(level)
        selected_rows.append(
            {
                "hyperedge_id": edge_id,
                "product_molecule_id": str(edge.get("product_molecule_id") or ""),
                "precursor_molecule_ids": [
                    str(value) for value in edge.get("precursor_molecule_ids") or []
                ],
                "portfolio_proof_level": level,
                "proof_level": str(binding.get("proof_level") or ""),
                "edge_binding_sha256": str(binding.get("binding_sha256") or ""),
                "structure_signature_sha256": str(
                    binding.get("structure_signature_sha256") or ""
                ),
                "trusted_precedent_sha256": str(
                    binding.get("trusted_precedent_sha256") or ""
                ),
            }
        )
    leaf_rows: list[dict[str, Any]] = []
    for molecule_id in sorted(candidate.leaves):
        binding = dict(stock_bindings[molecule_id])
        boundary_type = str(binding.get("boundary_type") or "")
        leaf_rows.append(
            {
                "molecule_id": molecule_id,
                "canonical_isomeric_smiles": str(
                    (molecules.get(molecule_id) or {}).get("canonical_isomeric_smiles")
                    or ""
                ),
                "stock_binding_sha256": str(binding.get("binding_sha256") or ""),
                "boundary_type": boundary_type,
                "catalog_id": str(binding.get("catalog_id") or ""),
                "catalog_sha256": str(binding.get("catalog_sha256") or ""),
                "benchmark_membership": binding.get("benchmark_membership") is True,
                "commercially_orderable": bool(
                    boundary_type == "commercially_orderable"
                    and binding.get("commercial_orderability_claimed") is True
                ),
            }
        )
    edge_ids = sorted(candidate.selections.values())
    edge_set_sha256 = _digest(edge_ids)
    all_commercial = all(row["commercially_orderable"] is True for row in leaf_rows)
    all_l4 = all(level >= 4 for level in proof_levels)
    payload: dict[str, Any] = {
        "schema_version": SELECTED_ROUTE_WITNESS_SCHEMA,
        "route_id": f"selected-route:{edge_set_sha256[:24]}",
        "root_molecule_id": root_id,
        "edge_set_sha256": edge_set_sha256,
        "selected_hyperedge_ids": edge_ids,
        "selected_hyperedges": selected_rows,
        "stock_terminal_ids": sorted(candidate.leaves),
        "stock_terminals": leaf_rows,
        "molecule_ids": sorted(reached_molecules),
        "weakest_edge_proof_level": min(proof_levels),
        "complete": True,
        "connected": True,
        "acyclic": True,
        "non_zero_step_synthesis": True,
        "benchmark_closed": True,
        "procurement_ready": bool(all_l4 and all_commercial),
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _bind_campaign_inputs(
    *,
    campaign_binding: Mapping[str, Any] | None,
    frontier_ledger: Mapping[str, Any] | None,
    overlay_sha256: str,
    bindings_sha256: str,
    expected_campaign_revision: int | None,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    if campaign_binding is None and frontier_ledger is None and expected_campaign_revision is None:
        return {}, reasons
    source = dict(campaign_binding or {})
    record: dict[str, Any] = {
        "campaign_id": str(source.get("campaign_id") or source.get("run_id") or ""),
        "revision": source.get("revision", source.get("campaign_revision")),
        "source_binding_content_sha256": "",
        "frontier_ledger_content_sha256": "",
    }
    if campaign_binding is None:
        reasons.append("campaign_binding_required_for_revision_bound_proof")
    else:
        source_sha256, source_reasons = _artifact_digest(
            source,
            label="campaign_binding",
            require_supplied_digest=True,
        )
        reasons.extend(source_reasons)
        record["source_binding_content_sha256"] = source_sha256
        bound_overlay = str(
            source.get("route_hypergraph_overlay_sha256")
            or source.get("overlay_content_sha256")
            or ""
        ).lower()
        bound_bindings = str(
            source.get("route_portfolio_bindings_sha256")
            or source.get("portfolio_bindings_content_sha256")
            or ""
        ).lower()
        if bound_overlay != overlay_sha256:
            reasons.append("campaign_binding_overlay_sha256_mismatch")
        if bound_bindings != bindings_sha256:
            reasons.append("campaign_binding_portfolio_bindings_sha256_mismatch")
    if expected_campaign_revision is not None:
        try:
            actual_revision = int(record["revision"])
        except (TypeError, ValueError):
            reasons.append("campaign_binding_revision_missing_or_invalid")
        else:
            if actual_revision != int(expected_campaign_revision):
                reasons.append("campaign_binding_revision_mismatch")
            record["revision"] = actual_revision
    if frontier_ledger is not None:
        ledger = dict(frontier_ledger)
        ledger_sha256, ledger_reasons = _artifact_digest(
            ledger,
            label="frontier_ledger",
            require_supplied_digest=True,
        )
        reasons.extend(ledger_reasons)
        record["frontier_ledger_content_sha256"] = ledger_sha256
        bound_ledger = str(source.get("frontier_ledger_content_sha256") or "").lower()
        if bound_ledger != ledger_sha256:
            reasons.append("campaign_binding_frontier_ledger_sha256_mismatch")
        source_revision = record.get("revision")
        ledger_revision = ledger.get("revision", ledger.get("campaign_revision"))
        if source_revision is not None and ledger_revision is not None:
            try:
                if int(source_revision) != int(ledger_revision):
                    reasons.append("campaign_frontier_revision_mismatch")
            except (TypeError, ValueError):
                reasons.append("campaign_frontier_revision_invalid")
    return record, sorted(set(reasons))


def _artifact_digest(
    value: Mapping[str, Any],
    *,
    label: str,
    require_supplied_digest: bool,
) -> tuple[str, list[str]]:
    row = dict(value or {})
    supplied = str(row.pop("content_sha256", "")).lower()
    computed = _digest(row)
    if supplied:
        if not _valid_sha256(supplied) or supplied != computed:
            return supplied or computed, [f"{label}_content_sha256_mismatch"]
        return supplied, []
    if require_supplied_digest:
        return computed, [f"{label}_content_sha256_missing"]
    return computed, []


def _expected_digest_reasons(expected: str, *, actual: str, label: str) -> list[str]:
    value = str(expected or "").lower()
    if not value:
        return []
    if not _valid_sha256(value):
        return [f"expected_{label}_sha256_invalid"]
    if value != actual:
        return [f"expected_{label}_sha256_mismatch"]
    return []


def _named_digest_valid(value: Mapping[str, Any], field_name: str) -> bool:
    row = dict(value or {})
    expected = str(row.pop(field_name, "")).lower()
    return bool(_valid_sha256(expected) and expected == _digest(row))


def _proof_level(value: Any) -> int:
    if isinstance(value, Mapping):
        value = value.get("portfolio_proof_level", value.get("level"))
    if isinstance(value, str) and not value.isdigit():
        return _PROOF_LEVELS.get(value, 0)
    try:
        return max(0, min(4, int(value or 0)))
    except (TypeError, ValueError):
        return 0


def _valid_sha256(value: str) -> bool:
    return _SHA256.fullmatch(str(value or "").lower()) is not None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "SELECTED_ROUTE_PARENT_PROOF_SCHEMA",
    "SELECTED_ROUTE_WITNESS_SCHEMA",
    "SELECTED_ROUTE_PROOF_COMPILER_VERSION",
    "compile_selected_route_parent_proof",
    "validate_selected_route_parent_proof",
    "is_solved_selected_route_parent_proof",
    "compile",
    "validate",
    "is_solved",
]

"""AND/OR route closure and diversity-aware portfolio selection.

The solver consumes the typed route-hypergraph overlay and explicit stock and
reaction-proof bindings.  It never treats a high model score as execution
proof.  Shared intermediates remain shared, so returned routes are DAGs rather
than duplicated path strings.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import itertools
import json
import re
from typing import Any, ClassVar, Mapping, Sequence

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutePortfolioItem:
    route_id: str
    root_molecule_id: str
    selected_hyperedges: tuple[tuple[str, str], ...]
    molecule_ids: tuple[str, ...]
    stock_terminal_ids: tuple[str, ...]
    source_channels: tuple[str, ...]
    independent_support_groups: tuple[str, ...]
    weakest_proof_level: int
    mean_edge_rank: float
    base_score: float
    diversity_score: float
    portfolio_score: float
    complete: bool
    reaction_validated: bool
    unresolved_frontiers: tuple[Mapping[str, Any], ...] = ()
    schema_version: ClassVar[str] = "route_portfolio_item.v1"

    @property
    def hyperedge_ids(self) -> tuple[str, ...]:
        return tuple(edge_id for _, edge_id in self.selected_hyperedges)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["schema_version"] = self.schema_version
        row["selected_hyperedges"] = [
            {"product_molecule_id": product_id, "hyperedge_id": edge_id}
            for product_id, edge_id in self.selected_hyperedges
        ]
        row["hyperedge_ids"] = list(self.hyperedge_ids)
        row["molecule_ids"] = list(self.molecule_ids)
        row["stock_terminal_ids"] = list(self.stock_terminal_ids)
        row["source_channels"] = list(self.source_channels)
        row["independent_support_groups"] = list(self.independent_support_groups)
        row["unresolved_frontiers"] = [dict(item) for item in self.unresolved_frontiers]
        row["content_sha256"] = _digest(row)
        return row


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutePortfolioReport:
    root_molecule_id: str
    routes: tuple[RoutePortfolioItem, ...]
    complete_candidate_count: int
    enumerated_candidate_count: int
    truncated: bool
    reasons: tuple[str, ...] = ()
    schema_version: ClassVar[str] = "route_portfolio.v1"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "root_molecule_id": self.root_molecule_id,
            "routes": [route.to_dict() for route in self.routes],
            "complete_candidate_count": self.complete_candidate_count,
            "enumerated_candidate_count": self.enumerated_candidate_count,
            "truncated": self.truncated,
            "reasons": list(self.reasons),
            "selection_policy": "and_or_closure_then_maximal_marginal_relevance",
            "requires_explicit_stock_and_reaction_proof": True,
        }
        payload["content_sha256"] = _digest(payload)
        return payload


@dataclass(slots=True)
class _Candidate:
    selection: dict[str, str] = field(default_factory=dict)
    molecules: set[str] = field(default_factory=set)
    stock: set[str] = field(default_factory=set)
    unresolved: list[dict[str, Any]] = field(default_factory=list)


def solve_diverse_routes(
    overlay: Mapping[str, Any],
    *,
    stock_molecule_ids: Sequence[str],
    edge_proof_levels: Mapping[str, int | Mapping[str, Any]],
    top_k: int = 5,
    min_reaction_proof_level: int = 2,
    max_depth: int = 20,
    max_enumerated_routes: int = 2000,
    diversity_weight: float = 0.25,
    fixed_selections: Mapping[str, str] | None = None,
) -> RoutePortfolioReport:
    """Return top-K complete, reaction-validated and structurally diverse DAGs."""

    if top_k < 1 or max_depth < 1 or max_enumerated_routes < 1:
        raise ValueError("route portfolio bounds must be positive")
    if not 0 <= diversity_weight <= 1:
        raise ValueError("diversity_weight must be in [0, 1]")
    root_id = str(overlay.get("root_molecule_id") or "")
    edges = {
        str(row.get("hyperedge_id") or ""): dict(row)
        for row in overlay.get("reaction_hyperedges") or []
        if isinstance(row, Mapping) and str(row.get("hyperedge_id") or "")
    }
    by_product: dict[str, list[dict[str, Any]]] = {}
    for edge in edges.values():
        by_product.setdefault(str(edge.get("product_molecule_id") or ""), []).append(edge)
    for product_edges in by_product.values():
        product_edges.sort(
            key=lambda row: (-float(row.get("rank_score") or 0.0), str(row["hyperedge_id"]))
        )
    fixed = {str(key): str(value) for key, value in dict(fixed_selections or {}).items()}
    reasons = _validate_inputs(
        overlay,
        root_id=root_id,
        edges=edges,
        fixed=fixed,
        edge_proof_levels=edge_proof_levels,
        min_reaction_proof_level=min_reaction_proof_level,
    )
    if reasons:
        return RoutePortfolioReport(
            root_molecule_id=root_id,
            routes=(),
            complete_candidate_count=0,
            enumerated_candidate_count=0,
            truncated=False,
            reasons=tuple(reasons),
        )

    stock = {str(item) for item in stock_molecule_ids if str(item)}
    truncated = [False]

    def expand(molecule_id: str, depth: int, ancestors: frozenset[str]) -> list[_Candidate]:
        if molecule_id in stock:
            return [_Candidate(molecules={molecule_id}, stock={molecule_id})]
        if molecule_id in ancestors:
            return [
                _Candidate(
                    molecules={molecule_id},
                    unresolved=[{"molecule_id": molecule_id, "reason": "cycle_detected"}],
                )
            ]
        if depth >= max_depth:
            return [
                _Candidate(
                    molecules={molecule_id},
                    unresolved=[{"molecule_id": molecule_id, "reason": "depth_limit"}],
                )
            ]
        choices = by_product.get(molecule_id, [])
        if molecule_id in fixed:
            choices = [row for row in choices if row["hyperedge_id"] == fixed[molecule_id]]
        choices = [
            row
            for row in choices
            if _proof_level(edge_proof_levels.get(str(row["hyperedge_id"])))
            >= min_reaction_proof_level
        ]
        if not choices:
            return [
                _Candidate(
                    molecules={molecule_id},
                    unresolved=[
                        {
                            "molecule_id": molecule_id,
                            "reason": (
                                "fixed_replacement_not_available"
                                if molecule_id in fixed
                                else "no_disconnection_or_stock_binding"
                            ),
                        }
                    ],
                )
            ]
        result: list[_Candidate] = []
        for edge in choices:
            if len(result) >= max_enumerated_routes:
                truncated[0] = True
                return _dedupe_candidates(result)
            edge_id = str(edge["hyperedge_id"])
            precursor_ids = [str(item) for item in edge.get("precursor_molecule_ids") or []]
            if not precursor_ids:
                result.append(
                    _Candidate(
                        selection={molecule_id: edge_id},
                        molecules={molecule_id},
                        unresolved=[
                            {"molecule_id": molecule_id, "reason": "hyperedge_has_no_precursors"}
                        ],
                    )
                )
                continue
            branches = [
                expand(precursor, depth + 1, ancestors | {molecule_id})
                for precursor in precursor_ids
            ]
            if any(not branch for branch in branches):
                continue
            for combination in itertools.product(*branches):
                merged = _merge_candidates(
                    combination,
                    product_id=molecule_id,
                    edge_id=edge_id,
                )
                if merged is None:
                    continue
                if len(result) >= max_enumerated_routes:
                    truncated[0] = True
                    return _dedupe_candidates(result)
                result.append(merged)
        return _dedupe_candidates(result)

    candidates = expand(root_id, 0, frozenset())
    complete_items: list[RoutePortfolioItem] = []
    for candidate in candidates:
        item = _portfolio_item(
            candidate,
            root_id=root_id,
            edges=edges,
            edge_proof_levels=edge_proof_levels,
            min_reaction_proof_level=min_reaction_proof_level,
        )
        if item.complete and item.reaction_validated:
            complete_items.append(item)
    complete_items.sort(key=lambda row: (-row.base_score, row.route_id))
    selected = _mmr_select(complete_items, top_k=top_k, diversity_weight=diversity_weight)
    report_reasons: list[str] = []
    if not complete_items:
        report_reasons.append("no_stock_closed_reaction_validated_route")
    if truncated[0]:
        report_reasons.append("route_enumeration_truncated")
    return RoutePortfolioReport(
        root_molecule_id=root_id,
        routes=tuple(selected),
        complete_candidate_count=len(complete_items),
        enumerated_candidate_count=len(candidates),
        truncated=truncated[0],
        reasons=tuple(report_reasons),
    )


def validate_route_replacement(
    overlay: Mapping[str, Any],
    *,
    stock_molecule_ids: Sequence[str],
    edge_proof_levels: Mapping[str, int | Mapping[str, Any]],
    base_selections: Mapping[str, str],
    product_molecule_id: str,
    replacement_hyperedge_id: str,
    min_reaction_proof_level: int = 2,
) -> dict[str, Any]:
    """Re-solve a replacement; never accept a UI-only edge splice."""

    selections = {str(key): str(value) for key, value in base_selections.items()}
    selections[str(product_molecule_id)] = str(replacement_hyperedge_id)
    report = solve_diverse_routes(
        overlay,
        stock_molecule_ids=stock_molecule_ids,
        edge_proof_levels=edge_proof_levels,
        top_k=1,
        min_reaction_proof_level=min_reaction_proof_level,
        fixed_selections=selections,
    )
    accepted = bool(report.routes)
    return {
        "schema_version": "route_replacement_validation.v1",
        "accepted": accepted,
        "product_molecule_id": str(product_molecule_id),
        "replacement_hyperedge_id": str(replacement_hyperedge_id),
        "route": report.routes[0].to_dict() if accepted else {},
        "reasons": [] if accepted else list(report.reasons),
        "connectivity_revalidated": True,
        "stock_closure_revalidated": True,
        "reaction_proof_revalidated": True,
    }


def validate_portfolio_replacements(
    overlay: Mapping[str, Any],
    *,
    portfolio: RoutePortfolioReport | Mapping[str, Any],
    stock_molecule_ids: Sequence[str],
    edge_proof_levels: Mapping[str, int | Mapping[str, Any]],
    min_reaction_proof_level: int = 2,
    max_candidates: int = 100,
) -> dict[str, Any]:
    """Build a bounded catalog of fully re-solved same-product replacements.

    A catalog row is never inferred by splicing an edge into a rendered route.
    Every row delegates to :func:`validate_route_replacement`, which re-runs
    AND/OR closure, stock closure, connectivity, and the reaction-proof floor.
    Candidate discovery is sorted and de-duplicated *before* the bound is
    applied, making truncation and the resulting digest reproducible.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    portfolio_payload = (
        portfolio.to_dict()
        if isinstance(portfolio, RoutePortfolioReport)
        else dict(portfolio or {})
    )
    supplied_portfolio_digest = str(portfolio_payload.get("content_sha256") or "")
    digest_payload = dict(portfolio_payload)
    digest_payload.pop("content_sha256", None)
    portfolio_integrity_valid = bool(
        supplied_portfolio_digest
        and supplied_portfolio_digest == _digest(digest_payload)
        and portfolio_payload.get("schema_version") == RoutePortfolioReport.schema_version
    )

    edges_by_product: dict[str, list[dict[str, Any]]] = {}
    for raw_edge in overlay.get("reaction_hyperedges") or []:
        if not isinstance(raw_edge, Mapping):
            continue
        edge = dict(raw_edge)
        edge_id = str(edge.get("hyperedge_id") or "")
        product_id = str(edge.get("product_molecule_id") or "")
        if edge_id and product_id:
            edges_by_product.setdefault(product_id, []).append(edge)
    for edges in edges_by_product.values():
        edges.sort(
            key=lambda row: (
                -float(row.get("rank_score") or 0.0),
                str(row.get("hyperedge_id") or ""),
            )
        )

    candidate_specs: dict[str, dict[str, Any]] = {}
    if portfolio_integrity_valid:
        for raw_route in portfolio_payload.get("routes") or []:
            if not isinstance(raw_route, Mapping):
                continue
            route = dict(raw_route)
            if route.get("complete") is not True or route.get("reaction_validated") is not True:
                continue
            selections = _selected_hyperedge_map(route.get("selected_hyperedges"))
            if not selections:
                continue
            base_route_id = str(route.get("route_id") or _digest(sorted(selections.items())))
            for product_id, original_edge_id in sorted(selections.items()):
                for edge in edges_by_product.get(product_id, []):
                    replacement_edge_id = str(edge.get("hyperedge_id") or "")
                    if not replacement_edge_id or replacement_edge_id == original_edge_id:
                        continue
                    identity = {
                        "base_selections": sorted(selections.items()),
                        "product_molecule_id": product_id,
                        "replacement_hyperedge_id": replacement_edge_id,
                    }
                    candidate_key = _digest(identity)
                    candidate_specs.setdefault(
                        candidate_key,
                        {
                            "candidate_key": candidate_key,
                            "base_route_id": base_route_id,
                            "base_selections": selections,
                            "product_molecule_id": product_id,
                            "original_hyperedge_id": original_edge_id,
                            "replacement_hyperedge_id": replacement_edge_id,
                            "replacement_rank_score": float(edge.get("rank_score") or 0.0),
                        },
                    )

    ordered_specs = sorted(
        candidate_specs.values(),
        key=lambda row: (
            str(row["base_route_id"]),
            str(row["product_molecule_id"]),
            -float(row["replacement_rank_score"]),
            str(row["replacement_hyperedge_id"]),
            str(row["candidate_key"]),
        ),
    )
    selected_specs = ordered_specs[:max_candidates]
    candidates: list[dict[str, Any]] = []
    for spec in selected_specs:
        validation = validate_route_replacement(
            overlay,
            stock_molecule_ids=stock_molecule_ids,
            edge_proof_levels=edge_proof_levels,
            base_selections=spec["base_selections"],
            product_molecule_id=str(spec["product_molecule_id"]),
            replacement_hyperedge_id=str(spec["replacement_hyperedge_id"]),
            min_reaction_proof_level=min_reaction_proof_level,
        )
        candidates.append(
            {
                "candidate_id": f"route-replacement:{spec['candidate_key'][:24]}",
                "base_route_id": str(spec["base_route_id"]),
                "product_molecule_id": str(spec["product_molecule_id"]),
                "original_hyperedge_id": str(spec["original_hyperedge_id"]),
                "replacement_hyperedge_id": str(spec["replacement_hyperedge_id"]),
                "replacement_rank_score": round(float(spec["replacement_rank_score"]), 8),
                "accepted": validation["accepted"],
                "route": validation["route"],
                "reasons": validation["reasons"],
                "connectivity_revalidated": validation["connectivity_revalidated"],
                "stock_closure_revalidated": validation["stock_closure_revalidated"],
                "reaction_proof_revalidated": validation["reaction_proof_revalidated"],
            }
        )

    payload = {
        "schema_version": "route_replacement_catalog.v1",
        "portfolio_content_sha256": supplied_portfolio_digest,
        "portfolio_integrity_valid": portfolio_integrity_valid,
        "candidate_count": len(candidates),
        "available_candidate_count": len(ordered_specs),
        "accepted_candidate_count": sum(row["accepted"] is True for row in candidates),
        "rejected_candidate_count": sum(row["accepted"] is not True for row in candidates),
        "max_candidates": max_candidates,
        "truncated": len(ordered_specs) > max_candidates,
        "candidates": candidates,
        "reasons": (
            []
            if portfolio_integrity_valid
            else ["invalid_or_digest_mismatched_route_portfolio"]
        ),
        "validation_policy": "full_and_or_resolve_with_stock_and_reaction_proof",
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def build_route_verifier_bundle(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create a deterministic, per-report content-addressed verifier bundle."""

    input_rows = [copy.deepcopy(dict(report)) for report in reports if isinstance(report, Mapping)]
    entries_by_source: dict[str, dict[str, Any]] = {}
    for report in input_rows:
        source_sha256 = _verifier_source_sha256(report)
        target = _verifier_report_target(report)
        entry = {
            "schema_version": "route_verifier_bundle_entry.v1",
            "report_id": f"verifier-report:{source_sha256[:24]}",
            "source_sha256": source_sha256,
            "target_smiles": target,
            "report_content_sha256": _digest(report),
            "verifier_report": report,
        }
        entry["content_sha256"] = _digest(entry)
        existing = entries_by_source.get(source_sha256)
        if existing is None or str(entry["content_sha256"]) < str(existing["content_sha256"]):
            entries_by_source[source_sha256] = entry
    entries = sorted(
        entries_by_source.values(),
        key=lambda row: (str(row["target_smiles"]), str(row["report_id"])),
    )
    payload = {
        "schema_version": "route_verifier_bundle.v1",
        "input_report_count": len(input_rows),
        "report_count": len(entries),
        "duplicate_report_count": len(input_rows) - len(entries),
        "reports": entries,
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def derive_portfolio_bindings(
    overlay: Mapping[str, Any],
    verifier: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind independently replayed root/child verifier proofs to v2 identities."""

    molecule_id_by_smiles = {
        _canonical_smiles(row.get("canonical_isomeric_smiles")): str(
            row.get("molecule_id") or ""
        )
        for row in overlay.get("molecules") or []
        if isinstance(row, Mapping)
        and _canonical_smiles(row.get("canonical_isomeric_smiles"))
        and str(row.get("molecule_id") or "")
    }
    molecule_smiles_by_id = {
        molecule_id: smiles for smiles, molecule_id in molecule_id_by_smiles.items()
    }
    root_smiles = molecule_smiles_by_id.get(str(overlay.get("root_molecule_id") or ""), "")
    candidates, bundle_meta = _verifier_report_candidates(
        verifier,
        single_report_target_smiles=root_smiles,
    )
    contexts: list[dict[str, Any]] = []
    report_audits: list[dict[str, Any]] = []
    for candidate in candidates:
        report = dict(candidate.get("verifier_report") or {})
        target = str(candidate.get("target_smiles") or "")
        reasons = [str(value) for value in candidate.get("reasons") or []]
        bank_present = "route_proof_bank" in report
        bank_entries: list[dict[str, Any]] = []
        legacy_valid = False
        if not reasons and bank_present:
            bank_entries = _replayable_proof_bank_entries(
                report,
                expected_target_smiles=target,
            )
            if not bank_entries:
                reasons.append("route_proof_bank_invalid_or_replay_failed")
        elif not reasons:
            legacy_valid = _strict_legacy_verifier_valid(
                report,
                expected_target_smiles=target,
            )
            if not legacy_valid:
                reasons.append("legacy_verifier_strict_replay_failed")
        accepted = not reasons and bool(bank_entries or legacy_valid)
        audit = {
            "report_id": str(candidate.get("report_id") or ""),
            "source_sha256": str(candidate.get("source_sha256") or ""),
            "target_smiles": target,
            "accepted": accepted,
            "proof_bank_present": bank_present,
            "replayed_proof_bank_entry_count": len(bank_entries),
            "reasons": sorted(set(reasons)),
        }
        report_audits.append(audit)
        if accepted:
            contexts.append(
                {
                    **audit,
                    "verifier_report": report,
                    "proof_bank_entries": bank_entries,
                    "legacy_valid": legacy_valid,
                }
            )

    proof_by_signature: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    materialized_terminals: set[str] = set()
    stock_evidence_candidates: dict[str, list[dict[str, Any]]] = {}
    replayed_entry_count = 0
    for context in contexts:
        report = dict(context["verifier_report"])
        bank_entries = list(context["proof_bank_entries"])
        replayed_entry_count += len(bank_entries)
        if bank_entries:
            proof_sources = [
                (
                    dict(entry.get("reaction_validation") or {}),
                    {
                        "proof_source": "route_proof_bank.v1",
                        "proof_bank_entry_id": str(entry.get("proof_id") or ""),
                        "proof_bank_entry_sha256": str(entry.get("content_hash") or ""),
                        "verifier_report_id": context["report_id"],
                        "verifier_source_sha256": context["source_sha256"],
                        "verifier_target_smiles": context["target_smiles"],
                    },
                )
                for entry in bank_entries
            ]
            bank_sha256 = str((report.get("route_proof_bank") or {}).get("content_hash") or "")
            for entry in bank_entries:
                steps = [
                    dict(row)
                    for row in (entry.get("materialized_route") or {}).get("steps") or []
                    if isinstance(row, Mapping)
                ]
                terminals = _materialized_terminal_smiles(steps)
                materialized_terminals.update(terminals)
                stock_binding = dict(entry.get("stock_terminal_evidence_binding") or {})
                evidence_rows = dict(stock_binding.get("terminal_evidence") or {})
                for terminal in sorted(terminals):
                    evidence = evidence_rows.get(terminal)
                    if isinstance(evidence, Mapping):
                        stock_evidence_candidates.setdefault(terminal, []).append(
                            {
                                "evidence": dict(evidence),
                                "authority": "strictly_replayed_route_proof_bank.v1",
                                "authority_sha256": bank_sha256,
                                "report_id": context["report_id"],
                                "verifier_source_sha256": context["source_sha256"],
                                "proof_bank_entry_id": str(entry.get("proof_id") or ""),
                                "proof_bank_entry_sha256": str(entry.get("content_hash") or ""),
                                "stock_evidence_binding_sha256": str(
                                    stock_binding.get("content_hash") or ""
                                ),
                            }
                        )
        else:
            proof_sources = [
                (
                    dict(report.get("reaction_validation") or {}),
                    {
                        "proof_source": "legacy_best_accepted_route",
                        "proof_bank_entry_id": "",
                        "proof_bank_entry_sha256": "",
                        "verifier_report_id": context["report_id"],
                        "verifier_source_sha256": context["source_sha256"],
                        "verifier_target_smiles": context["target_smiles"],
                    },
                )
            ]
            steps = [
                dict(row)
                for row in (report.get("accepted_route") or {}).get("steps") or []
                if isinstance(row, Mapping)
            ]
            terminals = _materialized_terminal_smiles(steps)
            materialized_terminals.update(terminals)
            stock_audit = dict(report.get("stock_catalog_audit") or {})
            evidence_rows = dict(stock_audit.get("terminal_evidence") or {})
            stock_audit_sha256 = _digest(stock_audit)
            for terminal in sorted(terminals):
                evidence = evidence_rows.get(terminal)
                if isinstance(evidence, Mapping):
                    stock_evidence_candidates.setdefault(terminal, []).append(
                        {
                            "evidence": dict(evidence),
                            "authority": "legacy_best_route_independent_stock_audit",
                            "authority_sha256": stock_audit_sha256,
                            "report_id": context["report_id"],
                            "verifier_source_sha256": context["source_sha256"],
                            "proof_bank_entry_id": "",
                            "proof_bank_entry_sha256": "",
                            "stock_evidence_binding_sha256": "",
                        }
                    )
        _merge_exact_proof_sources(proof_by_signature, proof_sources)

    edge_levels, exact_edge_bindings = _bind_proofs_to_overlay_edges(
        overlay,
        molecule_smiles_by_id=molecule_smiles_by_id,
        proof_by_signature=proof_by_signature,
    )
    stock_bindings: dict[str, dict[str, Any]] = {}
    evidence_by_smiles: dict[str, dict[str, Any]] = {}
    for terminal in sorted(materialized_terminals):
        molecule_id = molecule_id_by_smiles.get(terminal, "")
        eligible = [
            row
            for row in stock_evidence_candidates.get(terminal, [])
            if _valid_stock_evidence(dict(row.get("evidence") or {}))
        ]
        eligible.sort(
            key=lambda row: (
                0 if row["authority"] == "strictly_replayed_route_proof_bank.v1" else 1,
                _digest(row["evidence"]),
                str(row["report_id"]),
            )
        )
        if not molecule_id or not eligible:
            continue
        selected = eligible[0]
        evidence = dict(selected["evidence"])
        evidence_by_smiles[terminal] = evidence
        proof_bank_authorities = {
            (
                str(row["proof_bank_entry_id"]),
                str(row["proof_bank_entry_sha256"]),
                str(row["stock_evidence_binding_sha256"]),
            )
            for row in eligible
            if row["authority"] == "strictly_replayed_route_proof_bank.v1"
        }
        evidence_payload = {
            "canonical_isomeric_smiles": terminal,
            "terminal_evidence": evidence,
            "stock_audit_sha256": selected["authority_sha256"],
        }
        binding = {
            "schema_version": "exact_stock_binding.v1",
            "molecule_id": molecule_id,
            "canonical_isomeric_smiles": terminal,
            "catalog_id": str(evidence.get("catalog_id") or ""),
            "catalog_sha256": str(evidence.get("catalog_sha256") or "").lower(),
            "lookup_basis": str(evidence.get("lookup_basis") or ""),
            "stock_audit_sha256": str(selected["authority_sha256"]),
            "evidence_sha256": _digest(evidence_payload),
            "binding_authority": str(selected["authority"]),
            "verifier_report_id": str(selected["report_id"]),
            "verifier_source_sha256": str(selected["verifier_source_sha256"]),
            "proof_bank_authorities": [
                {
                    "proof_bank_entry_id": entry_id,
                    "proof_bank_entry_sha256": entry_sha256,
                    "stock_evidence_binding_sha256": stock_sha256,
                }
                for entry_id, entry_sha256, stock_sha256 in sorted(proof_bank_authorities)
            ],
        }
        binding["binding_sha256"] = _digest(binding)
        stock_bindings[molecule_id] = binding

    stock_ids = sorted(stock_bindings)
    unmatched_terminals = sorted(
        terminal
        for terminal in materialized_terminals
        if terminal not in molecule_id_by_smiles
        or terminal not in evidence_by_smiles
        or not _valid_stock_evidence(evidence_by_smiles[terminal])
    )
    accepted_count = sum(row["accepted"] is True for row in report_audits)
    rejected_count = len(report_audits) - accepted_count
    bundle_audit = {
        "schema_version": "route_verifier_bundle_audit.v1",
        "input_schema_version": bundle_meta["input_schema_version"],
        "bundle_content_sha256": bundle_meta["bundle_content_sha256"],
        "bundle_integrity_valid": bundle_meta["bundle_integrity_valid"],
        "report_count": len(report_audits),
        "accepted_report_count": accepted_count,
        "rejected_report_count": rejected_count,
        "duplicate_report_count": bundle_meta["duplicate_report_count"],
        "reports": sorted(report_audits, key=lambda row: row["report_id"]),
        "reasons": bundle_meta["reasons"],
    }
    bundle_audit["content_sha256"] = _digest(bundle_audit)
    proof_bank_present = any(row["proof_bank_present"] for row in report_audits)
    proof_bank_fail_closed = any(
        row["proof_bank_present"] and row["accepted"] is not True
        for row in report_audits
    )
    payload = {
        "schema_version": "route_portfolio_bindings.v1",
        "stock_molecule_ids": stock_ids,
        "edge_proof_levels": dict(sorted(edge_levels.items())),
        "exact_edge_proof_bindings": dict(sorted(exact_edge_bindings.items())),
        "stock_bindings": dict(sorted(stock_bindings.items())),
        "matched_edge_count": len(edge_levels),
        "proof_step_count": len(proof_by_signature),
        "matched_stock_terminal_count": len(stock_ids),
        "materialized_terminal_count": len(materialized_terminals),
        "unmatched_materialized_terminals": unmatched_terminals,
        "stock_binding_valid": bool(contexts),
        "all_materialized_terminals_proven": bool(
            materialized_terminals and not unmatched_terminals
        ),
        "stock_binding_source": "independent_stock_catalog_audit.terminal_evidence",
        "proof_binding_source": (
            "route_verifier_bundle.v1"
            if bundle_meta["input_schema_version"] == "route_verifier_bundle.v1"
            else (
                "strictly_replayed_route_proof_bank.v1"
                if contexts and contexts[0]["proof_bank_entries"]
                else (
                    "strictly_replayed_legacy_best_accepted_route"
                    if contexts
                    else (
                        "route_proof_bank_rejected_fail_closed"
                        if proof_bank_present
                        else "untrusted_legacy_verifier_rejected"
                    )
                )
            )
        ),
        "proof_bank_present": proof_bank_present,
        "proof_bank_fail_closed": proof_bank_fail_closed,
        "replayed_proof_bank_entry_count": replayed_entry_count,
        "accepted_verifier_report_count": accepted_count,
        "rejected_verifier_report_count": rejected_count,
        "rejected_verifier_report_reasons": {
            row["report_id"]: list(row["reasons"])
            for row in sorted(report_audits, key=lambda value: value["report_id"])
            if row["accepted"] is not True
        },
        "duplicate_verifier_report_count": bundle_meta["duplicate_report_count"],
        "verifier_bundle_content_sha256": bundle_meta["bundle_content_sha256"],
        "verifier_bundle_reasons": bundle_meta["reasons"],
        "verifier_bundle_audit": bundle_audit,
        "binding_is_exact_structure_signature": True,
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _verifier_report_target(report: Mapping[str, Any]) -> str:
    if "route_proof_bank" in report:
        bank = report.get("route_proof_bank")
        raw = bank.get("target_smiles") if isinstance(bank, Mapping) else ""
    else:
        audit = report.get("target_equivalence_audit")
        raw = (
            audit.get("request_canonical_isomeric_smiles")
            or audit.get("request_target_smiles")
            if isinstance(audit, Mapping)
            else ""
        )
    return _canonical_smiles(raw)


def _verifier_source_sha256(report: Mapping[str, Any]) -> str:
    if "route_proof_bank" in report:
        authority = {
            "kind": "route_proof_bank",
            "payload": report.get("route_proof_bank"),
        }
    else:
        authority = {"kind": "legacy_verifier_report", "payload": dict(report)}
    return _digest(authority)


def _verifier_report_candidates(
    verifier: Mapping[str, Any],
    *,
    single_report_target_smiles: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if verifier.get("schema_version") != "route_verifier_bundle.v1":
        report = dict(verifier)
        source_sha256 = _verifier_source_sha256(report)
        return (
            [
                {
                    "report_id": f"verifier-report:{source_sha256[:24]}",
                    "source_sha256": source_sha256,
                    "target_smiles": single_report_target_smiles,
                    "verifier_report": report,
                    "reasons": [],
                }
            ],
            {
                "input_schema_version": str(verifier.get("schema_version") or ""),
                "bundle_content_sha256": _digest(report),
                "bundle_integrity_valid": True,
                "duplicate_report_count": 0,
                "reasons": [],
            },
        )

    bundle = dict(verifier)
    supplied_hash = str(bundle.get("content_sha256") or "").lower()
    hash_payload = dict(bundle)
    hash_payload.pop("content_sha256", None)
    bundle_reasons: list[str] = []
    if re.fullmatch(r"[0-9a-f]{64}", supplied_hash) is None or supplied_hash != _digest(
        hash_payload
    ):
        bundle_reasons.append("route_verifier_bundle_content_sha256_mismatch")
    raw_entries = bundle.get("reports")
    if not isinstance(raw_entries, list):
        bundle_reasons.append("route_verifier_bundle_reports_not_list")
        raw_entries = []
    try:
        if int(bundle.get("report_count") or 0) != len(raw_entries):
            bundle_reasons.append("route_verifier_bundle_report_count_mismatch")
        declared_duplicates = max(0, int(bundle.get("duplicate_report_count") or 0))
    except (TypeError, ValueError):
        bundle_reasons.append("route_verifier_bundle_counts_invalid")
        declared_duplicates = 0

    candidates: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    observed_duplicates = 0
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            candidates.append(
                {
                    "report_id": f"invalid-bundle-entry:{index}",
                    "source_sha256": _digest({"index": index, "entry": raw_entry}),
                    "target_smiles": "",
                    "verifier_report": {},
                    "reasons": ["route_verifier_bundle_entry_not_object"],
                }
            )
            continue
        entry = dict(raw_entry)
        report = entry.get("verifier_report")
        report = dict(report) if isinstance(report, Mapping) else {}
        source_sha256 = str(entry.get("source_sha256") or "").lower()
        report_id = str(entry.get("report_id") or f"invalid-bundle-entry:{index}")
        target = str(entry.get("target_smiles") or "")
        reasons: list[str] = []
        entry_hash = str(entry.get("content_sha256") or "").lower()
        entry_payload = dict(entry)
        entry_payload.pop("content_sha256", None)
        if entry.get("schema_version") != "route_verifier_bundle_entry.v1":
            reasons.append("invalid_route_verifier_bundle_entry_schema")
        if re.fullmatch(r"[0-9a-f]{64}", entry_hash) is None or entry_hash != _digest(
            entry_payload
        ):
            reasons.append("route_verifier_bundle_entry_content_sha256_mismatch")
        report_sha256 = _digest(report)
        if str(entry.get("report_content_sha256") or "").lower() != report_sha256:
            reasons.append("route_verifier_report_content_sha256_mismatch")
        actual_source_sha256 = _verifier_source_sha256(report)
        if source_sha256 != actual_source_sha256:
            reasons.append("route_verifier_source_sha256_mismatch")
        if report_id != f"verifier-report:{source_sha256[:24]}":
            reasons.append("route_verifier_report_id_mismatch")
        actual_target = _verifier_report_target(report)
        if not target or target != actual_target:
            reasons.append("route_verifier_report_target_mismatch")
        duplicate_key = actual_source_sha256
        if duplicate_key in seen_sources:
            observed_duplicates += 1
            continue
        seen_sources.add(duplicate_key)
        candidates.append(
            {
                "report_id": report_id,
                "source_sha256": source_sha256,
                "target_smiles": target,
                "verifier_report": report,
                "reasons": sorted(set(reasons)),
            }
        )
    return (
        candidates,
        {
            "input_schema_version": "route_verifier_bundle.v1",
            "bundle_content_sha256": supplied_hash,
            "bundle_integrity_valid": not bundle_reasons,
            "duplicate_report_count": declared_duplicates + observed_duplicates,
            "reasons": sorted(set(bundle_reasons)),
        },
    )


def _merge_exact_proof_sources(
    proof_by_signature: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
    proof_sources: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    for reaction_validation, authority in proof_sources:
        route_proof_digest = _valid_reaction_route_digest(reaction_validation)
        for proof in reaction_validation.get("step_proofs") or []:
            if not isinstance(proof, Mapping):
                continue
            proof_row = dict(proof)
            raw_reactants = list(proof_row.get("reactant_smiles") or [])
            canonical_reactants = [_canonical_smiles(item) for item in raw_reactants]
            signature = (
                _canonical_smiles(proof_row.get("product_smiles")),
                tuple(sorted(value for value in canonical_reactants if value)),
            )
            if (
                not signature[0]
                or not raw_reactants
                or any(not value for value in canonical_reactants)
            ):
                continue
            binding = _exact_proof_binding(
                proof_row,
                signature=signature,
                route_proof_digest=route_proof_digest,
                authority=authority,
            )
            if not binding:
                continue
            existing = proof_by_signature.get(signature)
            if existing is None or (
                int(binding["portfolio_proof_level"]),
                int(binding["proof_source"] == "route_proof_bank.v1"),
                str(binding["proof_digest"]),
            ) > (
                int(existing["portfolio_proof_level"]),
                int(existing["proof_source"] == "route_proof_bank.v1"),
                str(existing["proof_digest"]),
            ):
                proof_by_signature[signature] = binding


def _bind_proofs_to_overlay_edges(
    overlay: Mapping[str, Any],
    *,
    molecule_smiles_by_id: Mapping[str, str],
    proof_by_signature: Mapping[tuple[str, tuple[str, ...]], Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    edge_levels: dict[str, int] = {}
    exact_edge_bindings: dict[str, dict[str, Any]] = {}
    for edge in overlay.get("reaction_hyperedges") or []:
        if not isinstance(edge, Mapping):
            continue
        edge_id = str(edge.get("hyperedge_id") or "")
        product_id = str(edge.get("product_molecule_id") or "")
        precursor_ids = tuple(
            sorted(str(item) for item in edge.get("precursor_molecule_ids") or [])
        )
        product_smiles = molecule_smiles_by_id.get(product_id, "")
        precursor_smiles = [
            molecule_smiles_by_id.get(precursor_id, "") for precursor_id in precursor_ids
        ]
        if (
            not edge_id
            or not product_smiles
            or not precursor_ids
            or any(not smiles for smiles in precursor_smiles)
        ):
            continue
        signature = (product_smiles, tuple(sorted(precursor_smiles)))
        proof_binding = proof_by_signature.get(signature)
        if proof_binding is None:
            continue
        exact_binding = {
            "schema_version": "exact_edge_proof_binding.v1",
            "hyperedge_id": edge_id,
            "product_molecule_id": product_id,
            "precursor_molecule_ids": list(precursor_ids),
            "structure_signature_sha256": _digest(
                {
                    "product_canonical_isomeric_smiles": signature[0],
                    "reactant_canonical_isomeric_smiles": list(signature[1]),
                }
            ),
            **dict(proof_binding),
        }
        exact_binding["binding_sha256"] = _digest(exact_binding)
        exact_edge_bindings[edge_id] = exact_binding
        level = int(proof_binding["portfolio_proof_level"])
        if level:
            edge_levels[edge_id] = level
    return edge_levels, exact_edge_bindings


def _selected_hyperedge_map(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {
            str(product_id): str(edge_id)
            for product_id, edge_id in value.items()
            if str(product_id) and str(edge_id)
        }
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    selected: dict[str, str] = {}
    for row in value:
        if isinstance(row, Mapping):
            product_id = str(row.get("product_molecule_id") or "")
            edge_id = str(row.get("hyperedge_id") or "")
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) == 2:
            product_id, edge_id = (str(row[0]), str(row[1]))
        else:
            continue
        if product_id and edge_id:
            selected[product_id] = edge_id
    return selected


def _replayable_proof_bank_entries(
    verifier: Mapping[str, Any],
    *,
    expected_target_smiles: str,
) -> list[dict[str, Any]]:
    bank = verifier.get("route_proof_bank")
    if not isinstance(bank, Mapping):
        return []
    try:
        from cascade_planner.harness.route_verifier import (
            replay_route_proof_bank_entry,
            validate_route_proof_bank,
        )
    except ImportError:
        return []
    target = str(expected_target_smiles or "")
    if not target or validate_route_proof_bank(bank, expected_target_smiles=target):
        return []
    raw_entries = [
        row for row in bank.get("entries") or [] if isinstance(row, Mapping)
    ]
    if not raw_entries:
        return []
    replayed: list[dict[str, Any]] = []
    for raw_entry in sorted(
        raw_entries,
        key=lambda row: (str(row.get("proof_id") or ""), int(row.get("route_rank") or 0)),
    ):
        entry = dict(raw_entry)
        replay = replay_route_proof_bank_entry(
            bank,
            proof_id=str(entry.get("proof_id") or ""),
            expected_target_smiles=target,
        )
        if replay.get("accepted") is not True:
            return []
        replayed.append(entry)
    return replayed


def _strict_legacy_verifier_valid(
    verifier: Mapping[str, Any],
    *,
    expected_target_smiles: str,
) -> bool:
    if not expected_target_smiles:
        return False
    try:
        from cascade_planner.harness.route_verifier import (
            is_accepted_route_verifier_report,
        )
    except ImportError:
        return False
    return is_accepted_route_verifier_report(
        dict(verifier),
        expected_target_smiles=expected_target_smiles,
    )


def _valid_reaction_route_digest(value: Mapping[str, Any]) -> str:
    row = dict(value or {})
    expected = str(row.pop("proof_digest", "")).lower()
    proofs = row.get("step_proofs")
    if (
        row.get("schema_version") != "reaction_route_validation.v1"
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or expected != _digest(row)
        or not isinstance(proofs, list)
    ):
        return ""
    try:
        step_count = int(row.get("step_count") or 0)
        validated_count = int(row.get("reaction_validated_step_count") or 0)
    except (TypeError, ValueError):
        return ""
    valid_step_proofs = [
        proof
        for proof in proofs
        if isinstance(proof, Mapping) and _valid_step_proof_digest(proof)
    ]
    actual_validated_count = sum(
        dict(proof).get("accepted") is True for proof in valid_step_proofs
    )
    if (
        step_count != len(proofs)
        or len(valid_step_proofs) != len(proofs)
        or validated_count != actual_validated_count
        or bool(row.get("accepted")) != bool(proofs and validated_count == step_count)
    ):
        return ""
    return expected


def _valid_step_proof_digest(value: Mapping[str, Any]) -> bool:
    row = dict(value or {})
    expected = str(row.pop("proof_digest", "")).lower()
    return bool(
        row.get("schema_version") == "reaction_step_proof.v1"
        and re.fullmatch(r"[0-9a-f]{64}", expected)
        and expected == _digest(row)
    )


def _exact_proof_binding(
    proof: Mapping[str, Any],
    *,
    signature: tuple[str, tuple[str, ...]],
    route_proof_digest: str,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    if not route_proof_digest or not _valid_step_proof_digest(proof):
        return {}
    row = dict(proof)
    product = _canonical_smiles(row.get("product_smiles"))
    raw_reactants = list(row.get("reactant_smiles") or [])
    canonical_reactants = [_canonical_smiles(item) for item in raw_reactants]
    if not raw_reactants or any(not item for item in canonical_reactants):
        return {}
    reactants = tuple(sorted(canonical_reactants))
    if (product, reactants) != signature:
        return {}
    reaction_digest = _reaction_signature_digest(product, reactants)
    if not reaction_digest or str(row.get("reaction_digest") or "").lower() != reaction_digest:
        return {}

    named_level = str(row.get("proof_level") or "")
    level = _named_proof_level(named_level)
    accepted = row.get("accepted") is True
    checks = dict(row.get("checks") or {})
    trusted_precedent = dict(row.get("trusted_precedent_binding") or {})
    strict_precedent = _strict_trusted_precedent_binding(
        trusted_precedent,
        reaction_digest=reaction_digest,
    )
    if named_level in {"L3_precedent_supported", "L4_procurement_ready"}:
        if (
            not accepted
            or not _mapping_consistency_checks_pass(checks)
            or checks.get("trusted_precedent_bound") is not True
            or not strict_precedent
        ):
            level = 0
    elif named_level == "L2_reaction_validated":
        if not accepted:
            level = 0
    elif named_level == "L2_mapping_consistent":
        # Atom-map consistency is useful advisory evidence, but it cannot
        # satisfy the executable portfolio proof floor.
        level = 0
    elif named_level == "L1_graph_and_stock_closed":
        level = 1
    else:
        level = 0
    if named_level == "L4_procurement_ready" and checks.get("procurement_bound") is not True:
        level = 0

    return {
        "proof_level": named_level,
        "portfolio_proof_level": level,
        "advisory": level < 2,
        "proof_accepted": accepted,
        "proof_digest": str(row.get("proof_digest") or "").lower(),
        "route_proof_digest": route_proof_digest,
        "reaction_digest": reaction_digest,
        "trusted_precedent_sha256": (
            _digest(trusted_precedent) if strict_precedent else ""
        ),
        "validator_version": str(row.get("validator_version") or ""),
        "proof_source": str(authority.get("proof_source") or ""),
        "proof_bank_entry_id": str(authority.get("proof_bank_entry_id") or ""),
        "proof_bank_entry_sha256": str(
            authority.get("proof_bank_entry_sha256") or ""
        ),
        "verifier_report_id": str(authority.get("verifier_report_id") or ""),
        "verifier_source_sha256": str(
            authority.get("verifier_source_sha256") or ""
        ),
        "verifier_target_smiles": str(
            authority.get("verifier_target_smiles") or ""
        ),
    }


def _mapping_consistency_checks_pass(checks: Mapping[str, Any]) -> bool:
    required = (
        "structures_materialized",
        "mapped_reaction_present",
        "mapped_product_matches",
        "mapped_reactants_match",
        "atom_maps_complete",
        "atom_maps_unique",
        "product_atoms_have_reactant_provenance",
        "mapped_elements_preserved",
        "mapped_reactant_components_contribute",
        "scaffold_continuity_plausible",
        "ring_change_plausible",
        "bond_change_present",
        "reaction_edit_budget_plausible",
        "stereochemical_product_matches",
    )
    return all(checks.get(key) is True for key in required)


def _strict_trusted_precedent_binding(
    value: Mapping[str, Any],
    *,
    reaction_digest: str,
) -> bool:
    authority = str(value.get("authority") or "")
    return bool(
        value.get("schema_version") == "trusted_precedent_binding.v1"
        and value.get("accepted") is True
        and authority in {"human_curator", "deterministic_structure_parser"}
        and str(value.get("authority_id") or "").strip()
        and str(value.get("binding_id") or "").strip()
        and str(value.get("source_ref") or "").strip()
        and str(value.get("reaction_digest") or "").lower() == reaction_digest
    )


def _reaction_signature_digest(product: str, reactants: Sequence[str]) -> str:
    if not product or not reactants:
        return ""
    return _digest(
        {
            "product_canonical_isomeric_smiles": product,
            "reactant_canonical_isomeric_smiles": sorted(reactants),
        }
    )


def _valid_stock_evidence(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("in_stock") is True
        and str(value.get("catalog_id") or "").strip()
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(value.get("catalog_sha256") or "").lower(),
        )
        and str(value.get("lookup_basis") or "").strip()
    )


def _materialized_terminal_smiles(steps: Sequence[Mapping[str, Any]]) -> set[str]:
    """Derive leaves from the accepted route graph, never from producer metrics."""

    products = {
        canonical
        for step in steps
        if (canonical := _canonical_smiles(_step_product_smiles(step)))
    }
    reactants = {
        canonical
        for step in steps
        for raw_smiles in _step_reactant_smiles(step)
        if (canonical := _canonical_smiles(raw_smiles))
    }
    return reactants - products


def _step_product_smiles(step: Mapping[str, Any]) -> str:
    return str(
        step.get("product")
        or step.get("product_smiles")
        or step.get("target_smiles")
        or ""
    )


def _step_reactant_smiles(step: Mapping[str, Any]) -> list[str]:
    value = (
        step.get("reactant_smiles")
        or step.get("reactants")
        or step.get("precursor_smiles")
        or step.get("precursors")
        or []
    )
    if isinstance(value, str):
        value = [item for item in value.split(".") if item]
    if not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _validate_inputs(
    overlay: Mapping[str, Any],
    *,
    root_id: str,
    edges: Mapping[str, Mapping[str, Any]],
    fixed: Mapping[str, str],
    edge_proof_levels: Mapping[str, int | Mapping[str, Any]],
    min_reaction_proof_level: int,
) -> list[str]:
    reasons: list[str] = []
    if str(overlay.get("schema_version") or "") != "route_hypergraph_overlay.v2":
        reasons.append("invalid_route_hypergraph_overlay_schema")
    if not root_id:
        reasons.append("missing_root_molecule_id")
    if not 0 <= min_reaction_proof_level <= 4:
        reasons.append("invalid_min_reaction_proof_level")
    if overlay.get("validation", {}).get("valid") is not True:
        reasons.append("route_hypergraph_overlay_invalid")
    for product_id, edge_id in fixed.items():
        edge = edges.get(edge_id)
        if edge is None:
            reasons.append(f"fixed_edge_not_found:{edge_id}")
        elif str(edge.get("product_molecule_id") or "") != product_id:
            reasons.append(f"fixed_edge_product_mismatch:{product_id}:{edge_id}")
        elif _proof_level(edge_proof_levels.get(edge_id)) < min_reaction_proof_level:
            reasons.append(
                f"fixed_edge_below_min_reaction_proof:{product_id}:{edge_id}"
            )
    return sorted(set(reasons))


def _merge_candidates(
    candidates: Sequence[_Candidate],
    *,
    product_id: str,
    edge_id: str,
) -> _Candidate | None:
    selection = {product_id: edge_id}
    molecules = {product_id}
    stock: set[str] = set()
    unresolved: list[dict[str, Any]] = []
    for candidate in candidates:
        for selected_product, selected_edge in candidate.selection.items():
            existing = selection.get(selected_product)
            if existing is not None and existing != selected_edge:
                return None
            selection[selected_product] = selected_edge
        molecules.update(candidate.molecules)
        stock.update(candidate.stock)
        unresolved.extend(candidate.unresolved)
    return _Candidate(
        selection=selection,
        molecules=molecules,
        stock=stock,
        unresolved=unresolved,
    )


def _dedupe_candidates(rows: Sequence[_Candidate]) -> list[_Candidate]:
    result: dict[str, _Candidate] = {}
    for row in rows:
        key = _digest(
            {
                "selection": sorted(row.selection.items()),
                "stock": sorted(row.stock),
                "unresolved": row.unresolved,
            }
        )
        result.setdefault(key, row)
    return list(result.values())


def _proof_level(value: int | Mapping[str, Any] | None) -> int:
    if isinstance(value, Mapping):
        raw = value.get("level")
        if raw is None:
            raw = value.get("level_index")
        if raw is None:
            raw = value.get("achieved_proof_level")
        if raw is None:
            raw = value.get("portfolio_proof_level")
        value = raw
    try:
        return max(0, min(4, int(value or 0)))
    except (TypeError, ValueError):
        return 0


def _named_proof_level(value: Any) -> int:
    return {
        "L0_materialized": 0,
        "L1_graph_and_stock_closed": 1,
        "L2_reaction_validated": 2,
        "L3_precedent_supported": 3,
        "L4_procurement_ready": 4,
    }.get(str(value or ""), 0)


def _portfolio_item(
    candidate: _Candidate,
    *,
    root_id: str,
    edges: Mapping[str, Mapping[str, Any]],
    edge_proof_levels: Mapping[str, int | Mapping[str, Any]],
    min_reaction_proof_level: int,
) -> RoutePortfolioItem:
    selections = tuple(sorted(candidate.selection.items()))
    selected_edges = [edges[edge_id] for _, edge_id in selections]
    proof_levels = [_proof_level(edge_proof_levels.get(edge_id)) for _, edge_id in selections]
    weakest = min(proof_levels, default=0)
    ranks = [float(edge.get("rank_score") or 0.0) for edge in selected_edges]
    mean_rank = sum(ranks) / len(ranks) if ranks else 0.0
    channels = sorted(
        {str(value) for edge in selected_edges for value in edge.get("source_channels") or []}
    )
    support_groups = sorted(
        {
            str(value)
            for edge in selected_edges
            for value in edge.get("independent_support_groups") or []
        }
    )
    complete = not candidate.unresolved and bool(selections or root_id in candidate.stock)
    reaction_validated = bool(selections) and all(
        level >= min_reaction_proof_level for level in proof_levels
    )
    if not selections and root_id in candidate.stock:
        reaction_validated = True
        weakest = 4
    proof_score = weakest / 4.0
    support_score = min(len(support_groups) / 3.0, 1.0)
    length_penalty = min(len(selections) / 50.0, 0.25)
    base_score = round(
        max(0.0, min(1.0, 0.45 * mean_rank + 0.4 * proof_score + 0.15 * support_score - length_penalty)),
        8,
    )
    identity = {
        "root_molecule_id": root_id,
        "selected_hyperedges": selections,
        "stock_terminal_ids": sorted(candidate.stock),
    }
    return RoutePortfolioItem(
        route_id=f"portfolio-route:{_digest(identity)[:24]}",
        root_molecule_id=root_id,
        selected_hyperedges=selections,
        molecule_ids=tuple(sorted(candidate.molecules)),
        stock_terminal_ids=tuple(sorted(candidate.stock)),
        source_channels=tuple(channels),
        independent_support_groups=tuple(support_groups),
        weakest_proof_level=weakest,
        mean_edge_rank=round(mean_rank, 8),
        base_score=base_score,
        diversity_score=0.0,
        portfolio_score=base_score,
        complete=complete,
        reaction_validated=reaction_validated,
        unresolved_frontiers=tuple(candidate.unresolved),
    )


def _mmr_select(
    candidates: Sequence[RoutePortfolioItem],
    *,
    top_k: int,
    diversity_weight: float,
) -> list[RoutePortfolioItem]:
    remaining = list(candidates)
    selected: list[RoutePortfolioItem] = []
    while remaining and len(selected) < top_k:
        scored: list[tuple[float, float, RoutePortfolioItem]] = []
        for candidate in remaining:
            diversity = (
                1.0
                if not selected
                else min(_route_distance(candidate, other) for other in selected)
            )
            score = (1.0 - diversity_weight) * candidate.base_score + diversity_weight * diversity
            scored.append((score, diversity, candidate))
        score, diversity, best = max(
            scored,
            key=lambda row: (round(row[0], 12), row[2].base_score, row[2].route_id),
        )
        selected.append(
            RoutePortfolioItem(
                **{
                    **asdict(best),
                    "diversity_score": round(diversity, 8),
                    "portfolio_score": round(score, 8),
                }
            )
        )
        remaining = [row for row in remaining if row.route_id != best.route_id]
    return selected


def _route_distance(left: RoutePortfolioItem, right: RoutePortfolioItem) -> float:
    left_edges = set(left.hyperedge_ids)
    right_edges = set(right.hyperedge_ids)
    union = left_edges | right_edges
    return 1.0 - (len(left_edges & right_edges) / len(union) if union else 1.0)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if molecule is not None
        else ""
    )

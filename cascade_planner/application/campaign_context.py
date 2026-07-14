"""Bounded global campaign context compiled from canonical run projections.

The context is a read-only reasoning view.  It deliberately preserves complete
graph topology and route-family structure while replacing raw documents and
large unchanged prose with digests and compact metadata.  It cannot mutate the
hypergraph, frontier, proof ledger, stock ledger, or RunKernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from cascade_planner.application.run_kernel import RunKernel, RunRevision


CAMPAIGN_CONTEXT_SCHEMA = "autoplanner_campaign_context.v1"
CAMPAIGN_CONTEXT_DELTA_SCHEMA = "autoplanner_campaign_context_delta.v1"
_RAW_CONTENT_KEYS = {
    "binary",
    "content",
    "full_text",
    "html",
    "image_bytes",
    "ocr_text",
    "pdf_bytes",
    "raw_document",
    "raw_html",
    "raw_text",
}
_TEXT_LIMIT = 640
_FAILURE_LIMIT = 48


class CampaignContextError(RuntimeError):
    """Base error for campaign context compilation."""


class CampaignContextTooLargeError(CampaignContextError):
    """Raised before a model call when complete topology cannot fit safely."""


@dataclass(frozen=True, slots=True)
class CampaignContextDelta:
    previous_context_sha256: str = ""
    changed_sections: tuple[str, ...] = ()
    section_digests: Mapping[str, str] = field(default_factory=dict)
    topology: Mapping[str, Any] = field(default_factory=dict)
    material_events: tuple[str, ...] = ()
    schema_version: str = CAMPAIGN_CONTEXT_DELTA_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "previous_context_sha256": self.previous_context_sha256,
            "changed_sections": list(self.changed_sections),
            "section_digests": dict(self.section_digests),
            "topology": dict(self.topology),
            "material_events": list(self.material_events),
        }


@dataclass(frozen=True, slots=True)
class CampaignContext:
    run_id: str
    target: Mapping[str, Any]
    revision: RunRevision
    topology: Mapping[str, Any]
    route_portfolio: Mapping[str, Any]
    evidence: Mapping[str, Any]
    stock: Mapping[str, Any]
    deficits: tuple[Mapping[str, Any], ...]
    proposal_history: tuple[Mapping[str, Any], ...]
    failure_history: tuple[Mapping[str, Any], ...]
    budget_state: Mapping[str, Any]
    acceptance_state: Mapping[str, Any]
    delta: CampaignContextDelta
    content_sha256: str = ""
    byte_count: int = 0
    schema_version: str = CAMPAIGN_CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        row = self._body()
        digest = _digest(row)
        payload_bytes = 0
        for _ in range(4):
            measured = len(
                _canonical_bytes(
                    {
                        **row,
                        "content_sha256": digest,
                        "byte_count": payload_bytes,
                    }
                )
            )
            if measured == payload_bytes:
                break
            payload_bytes = measured
        if self.content_sha256 and self.content_sha256 != digest:
            raise CampaignContextError("campaign_context_digest_invalid")
        if self.byte_count and self.byte_count != payload_bytes:
            raise CampaignContextError("campaign_context_byte_count_invalid")
        object.__setattr__(self, "content_sha256", digest)
        object.__setattr__(self, "byte_count", payload_bytes)

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "target": dict(self.target),
            "revision": self.revision.to_dict(),
            "topology": dict(self.topology),
            "route_portfolio": dict(self.route_portfolio),
            "evidence": dict(self.evidence),
            "stock": dict(self.stock),
            "deficits": [dict(row) for row in self.deficits],
            "proposal_history": [dict(row) for row in self.proposal_history],
            "failure_history": [dict(row) for row in self.failure_history],
            "budget_state": dict(self.budget_state),
            "acceptance_state": dict(self.acceptance_state),
            "delta": self.delta.to_dict(),
            "semantics": {
                "read_only_projection": True,
                "complete_topology_preserved": True,
                "raw_documents_excluded": True,
                "director_has_no_scientific_authority": True,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._body(),
            "content_sha256": self.content_sha256,
            "byte_count": self.byte_count,
        }


class CampaignContextCompiler:
    """Compile one digest-bound global view from current canonical revisions."""

    def __init__(self, *, max_context_bytes: int | None = None) -> None:
        if max_context_bytes is not None and int(max_context_bytes) <= 0:
            raise ValueError("max_context_bytes must be positive")
        self.max_context_bytes = (
            int(max_context_bytes) if max_context_bytes is not None else None
        )

    def compile(
        self,
        *,
        kernel: RunKernel,
        hypergraph: Mapping[str, Any] | None = None,
        route_portfolio: Mapping[str, Any] | None = None,
        evidence_ledger: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
        stock_ledger: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
        proposal_history: Iterable[Mapping[str, Any]] = (),
        failure_history: Iterable[Mapping[str, Any]] = (),
        material_events: Iterable[str] = (),
        previous: CampaignContext | Mapping[str, Any] | None = None,
    ) -> CampaignContext:
        state = kernel.state
        graph_input = dict(hypergraph or {})
        graph = (
            _compact_canonical_hypergraph(graph_input)
            if graph_input.get("schema_version")
            == "canonical_retrosynthesis_hypergraph.v1"
            else _compact(graph_input)
        )
        portfolio_input = dict(route_portfolio or {})
        portfolio = _deduplicate_routes(
            _compact_proof_portfolio(portfolio_input)
            if portfolio_input.get("schema_version")
            == "proof_stitched_route_portfolio.v1"
            else _compact(portfolio_input)
        )
        evidence = _compact_ledger(evidence_ledger)
        stock = _compact_ledger(stock_ledger)
        proposals = tuple(
            _compact(row) for row in proposal_history if isinstance(row, Mapping)
        )
        failures = tuple(
            _compact(row)
            for row in list(failure_history)[-_FAILURE_LIMIT:]
            if isinstance(row, Mapping)
        )
        deficits = tuple(_compact_deficit(row) for row in state.deficits)
        sections = {
            "topology": graph,
            "route_portfolio": portfolio,
            "evidence": evidence,
            "stock": stock,
            "deficits": deficits,
            "proposal_history": proposals,
            "failure_history": failures,
            "acceptance_state": _compact(state.acceptance_report),
        }
        delta = _compile_delta(
            sections,
            previous=previous,
            material_events=material_events,
        )
        context = CampaignContext(
            run_id=kernel.spec.run_id,
            target={
                "name": kernel.spec.target_name,
                "canonical_smiles": kernel.spec.target_smiles,
                "acceptance": kernel.spec.acceptance.to_dict(),
            },
            revision=kernel.revision,
            topology=graph,
            route_portfolio=portfolio,
            evidence=evidence,
            stock=stock,
            deficits=deficits,
            proposal_history=proposals,
            failure_history=failures,
            budget_state={
                "limits": kernel.spec.limits.to_dict(),
                "attempt_count": state.attempt_count,
                "accepted_expansion_count": state.accepted_expansion_count,
                "model_totals": dict(state.model_totals),
                "in_flight_task_count": len(state.in_flight_tasks),
                "stop_decision": kernel.decide_stop().to_dict(),
            },
            acceptance_state=_compact(state.acceptance_report),
            delta=delta,
        )
        limit = self.max_context_bytes
        if limit is None:
            limit = kernel.spec.limits.model.max_prompt_context_bytes
        if context.byte_count > limit:
            raise CampaignContextTooLargeError(
                "campaign_context_byte_budget_exceeded:"
                f"{context.byte_count}>{limit}"
            )
        return context


def _compile_delta(
    sections: Mapping[str, Any],
    *,
    previous: CampaignContext | Mapping[str, Any] | None,
    material_events: Iterable[str],
) -> CampaignContextDelta:
    current_digests = {key: _digest(value) for key, value in sections.items()}
    previous_row = previous.to_dict() if isinstance(previous, CampaignContext) else dict(previous or {})
    previous_digest = str(previous_row.get("content_sha256") or "")
    previous_sections = {
        key: previous_row.get(key, [] if isinstance(value, tuple) else {})
        for key, value in sections.items()
    }
    changed = tuple(
        key
        for key, digest in current_digests.items()
        if not previous_row or digest != _digest(previous_sections.get(key))
    )
    topology_delta = _topology_delta(
        dict(previous_row.get("topology") or {}),
        dict(sections.get("topology") or {}),
    )
    return CampaignContextDelta(
        previous_context_sha256=previous_digest,
        changed_sections=changed,
        section_digests=current_digests,
        topology=topology_delta,
        material_events=tuple(
            sorted({str(item) for item in material_events if str(item).strip()})
        ),
    )


def _topology_delta(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("molecules", "nodes", "reactions", "edges", "hyperedges", "route_families"):
        if key not in previous and key not in current:
            continue
        before = _identity_map(previous.get(key))
        after = _identity_map(current.get(key))
        result[key] = {
            "added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "changed": sorted(
                identity
                for identity in set(before) & set(after)
                if _digest(before[identity]) != _digest(after[identity])
            ),
        }
    return result


def _identity_map(value: Any) -> dict[str, Any]:
    rows: list[Any]
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    rows = list(value or []) if isinstance(value, (list, tuple)) else []
    result: dict[str, Any] = {}
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            identity = next(
                (
                    str(row.get(key))
                    for key in (
                        "id",
                        "molecule_id",
                        "edge_id",
                        "reaction_id",
                        "route_family_id",
                        "route_id",
                    )
                    if row.get(key)
                ),
                f"index:{index}",
            )
        else:
            identity = f"index:{index}"
        result[identity] = row
    return result


def _compact_ledger(
    value: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return _compact(value)
    return {"records": [_compact(row) for row in value if isinstance(row, Mapping)]}


def _compact_canonical_hypergraph(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve complete identities/connectivity while dropping verbose proofs."""

    return {
        "schema_version": str(graph.get("schema_version") or ""),
        "revision": int(graph.get("revision") or 0),
        "scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "topology_sha256": str(graph.get("topology_sha256") or ""),
        "target_molecule_id": str(graph.get("target_molecule_id") or ""),
        "molecules": {
            str(key): _fields(
                value,
                (
                    "canonical_smiles",
                    "incoming_edge_ids",
                    "is_leaf",
                    "outgoing_edge_ids",
                    "stock_closed",
                    "active_stock_observation_id",
                ),
            )
            for key, value in dict(graph.get("molecules") or {}).items()
        },
        "edges": {
            str(key): {
                **_fields(
                    value,
                    (
                        "edge_digest",
                        "exact_record_ids",
                        "independent_source_groups",
                        "precursor_molecule_ids",
                        "precursor_smiles",
                        "product_molecule_id",
                        "product_smiles",
                        "route_family_ids",
                        "source_binding_ids",
                        "status",
                    ),
                ),
                "reaction_proofs": [
                    {
                        **_fields(
                            proof,
                            ("accepted", "proof_level", "reasons"),
                        ),
                        "transform_family": str(
                            dict(
                                dict(proof).get("deterministic_transform_audit")
                                or {}
                            ).get("transform_family")
                            or ""
                        ),
                    }
                    for proof in list(dict(value).get("reaction_proofs") or [])[-2:]
                    if isinstance(proof, Mapping)
                ],
                "origins": [
                    _fields(
                        origin,
                        (
                            "origin_kind",
                            "origin_ref",
                            "route_family_id",
                            "skeleton_id",
                            "step_id",
                            "transformation_hypothesis",
                        ),
                    )
                    for origin in list(dict(value).get("origin_records") or [])[-4:]
                    if isinstance(origin, Mapping)
                ],
            }
            for key, value in dict(graph.get("edges") or {}).items()
            if isinstance(value, Mapping)
        },
        "hypotheses": {
            str(key): {
                **_fields(
                    value,
                    (
                    "frontier_priority",
                    "precursor_smiles",
                    "product_smiles",
                    "route_family_ids",
                    "status",
                    ),
                ),
                "origin_kinds": sorted(
                    {
                        str(origin.get("origin_kind") or "")
                        for origin in dict(value).get("origin_records") or []
                        if isinstance(origin, Mapping) and origin.get("origin_kind")
                    }
                ),
                "condition_prediction_count": len(
                    dict(value).get("condition_predictions") or []
                ),
            }
            for key, value in dict(graph.get("hypotheses") or {}).items()
        },
        "route_families": {
            str(key): _fields(
                value,
                (
                    "aliases",
                    "blocking_deficit_ids",
                    "closed",
                    "edge_ids",
                    "hypothesis_ids",
                    "leaf_molecule_ids",
                    "minimum_proof_level",
                    "selected",
                    "skeleton_ids",
                    "status",
                    "stock_closure_rate",
                    "strategy",
                    "unmaterialized_hypothesis_ids",
                ),
            )
            for key, value in dict(graph.get("route_families") or {}).items()
        },
        "source_bindings": {
            str(key): _fields(
                value,
                (
                    "acquisition_status",
                    "exact_row_count_observed",
                    "independence_group",
                    "proxy_request_id",
                    "source_group",
                    "source_kind",
                    "source_pdf_sha256",
                    "source_ref",
                    "title",
                    "visual_candidate_page_count",
                ),
            )
            for key, value in dict(graph.get("source_bindings") or {}).items()
        },
        "exact_records": {
            str(key): _fields(
                value,
                (
                    "edge_digest",
                    "independence_group",
                    "source_binding_id",
                    "source_location",
                ),
            )
            for key, value in dict(graph.get("exact_records") or {}).items()
        },
        "stock_observations": {
            str(key): _fields(
                value,
                ("accepted", "canonical_smiles", "reasons"),
            )
            for key, value in dict(graph.get("stock_observations") or {}).items()
        },
        "conflicts": _compact(graph.get("conflicts") or {}),
        "deficit_frontier": _compact_deficit_frontier(
            graph.get("deficit_frontier") or {}
        ),
        "portfolio_ranking": _compact(graph.get("portfolio_ranking") or []),
        "delta": _fields(
            graph.get("delta") or {},
            ("dirty_entity_ids", "rejected", "revision"),
        ),
        "semantics": {
            "complete_entity_identity_and_connectivity_preserved": True,
            "verbose_proof_and_provider_payloads_omitted": True,
        },
    }


def _compact_proof_portfolio(portfolio: Mapping[str, Any]) -> dict[str, Any]:
    route_fields = (
        "all_edges_proven",
        "all_leaves_stock_closed",
        "complete",
        "edge_ids",
        "independent_source_groups",
        "leaf_molecule_ids",
        "minimum_edge_proof_level",
        "open_leaf_molecule_ids",
        "reasons",
        "root_edge_ids",
        "route_family_id",
        "route_id",
        "selected",
        "stock_closure_rate",
        "unproven_edge_ids",
    )
    return {
        "schema_version": str(portfolio.get("schema_version") or ""),
        "graph_revision": int(portfolio.get("graph_revision") or 0),
        "graph_scientific_sha256": str(
            portfolio.get("graph_scientific_sha256") or ""
        ),
        "accepted": portfolio.get("accepted") is True,
        "proof_policy": _compact(portfolio.get("proof_policy") or {}),
        "edge_proofs": {
            str(key): _fields(
                value,
                (
                    "accepted",
                    "achieved_level",
                    "conflict_ids",
                    "edge_id",
                    "exact_source_bound",
                    "independent_source_groups",
                    "reaction_validated",
                    "reasons",
                    "required_level",
                ),
            )
            for key, value in dict(portfolio.get("edge_proofs") or {}).items()
        },
        "leaf_proofs": {
            str(key): _fields(
                value,
                ("accepted", "molecule_id", "reasons", "stock_boundary"),
            )
            for key, value in dict(portfolio.get("leaf_proofs") or {}).items()
        },
        "route_candidates": [
            _fields(value, route_fields)
            for value in portfolio.get("route_candidates") or []
            if isinstance(value, Mapping)
        ],
        "selected_routes": [
            _fields(value, route_fields)
            for value in portfolio.get("selected_routes") or []
            if isinstance(value, Mapping)
        ],
        "route_modules": [
            _fields(
                value,
                (
                    "alternatives",
                    "module_id",
                    "product_molecule_id",
                    "route_family_id",
                ),
            )
            for value in portfolio.get("route_modules") or []
            if isinstance(value, Mapping)
        ],
        "deficits": [
            _compact_deficit(value)
            for value in portfolio.get("deficits") or []
            if isinstance(value, Mapping)
        ],
        "metrics": _compact(portfolio.get("metrics") or {}),
        "closeout": _compact(portfolio.get("closeout") or {}),
        "semantics": {
            "all_route_edge_sets_preserved": True,
            "verbose_proof_payloads_omitted": True,
        },
    }


def _fields(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    row = dict(value or {}) if isinstance(value, Mapping) else {}
    return {name: _compact(row[name], key=name) for name in names if name in row}


def _compact_deficit_frontier(value: Any) -> dict[str, Any]:
    row = dict(value or {}) if isinstance(value, Mapping) else {}
    raw_items = row.get("items") or []
    items = (
        list(raw_items.values())
        if isinstance(raw_items, Mapping)
        else list(raw_items)
        if isinstance(raw_items, (list, tuple))
        else []
    )
    return {
        "schema_version": str(row.get("schema_version") or ""),
        "summary": _compact(row.get("summary") or {}),
        "items": [
            _compact_deficit(item)
            for item in items
            if isinstance(item, Mapping)
        ],
    }


def _compact_deficit(value: Any) -> dict[str, Any]:
    row = dict(value or {}) if isinstance(value, Mapping) else {}
    metadata = dict(row.get("metadata") or {})
    return {
        **_fields(
            row,
            (
                "deficit_id",
                "dependency_ids",
                "deterministic",
                "entity_ids",
                "entity_refs",
                "kind",
                "model_allowed",
                "object_id",
                "priority",
                "reason",
                "reasons",
                "route_family_ids",
                "score",
            ),
        ),
        "route_ids": _compact(metadata.get("route_ids") or []),
    }


def _compact(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        omitted: list[dict[str, Any]] = []
        for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = str(raw_key)
            if name.lower() in _RAW_CONTENT_KEYS:
                payload = _raw_content_bytes(item)
                omitted.append(
                    {
                        "field": name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "byte_count": len(payload),
                    }
                )
                continue
            result[name] = _compact(item, key=name)
        if omitted:
            result["omitted_raw_content"] = omitted
        return result
    if isinstance(value, (set, frozenset)):
        compacted = [_compact(item, key=key) for item in value]
        return sorted(compacted, key=_canonical_bytes)
    if isinstance(value, (list, tuple)):
        return [_compact(item, key=key) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CampaignContextError(f"campaign_context_non_finite:{key}")
        return value
    if isinstance(value, str) and len(value) > _TEXT_LIMIT:
        encoded = value.encode("utf-8")
        return {
            "summary": value[:_TEXT_LIMIT],
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "original_byte_count": len(encoded),
            "truncated": True,
        }
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _deduplicate_routes(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"value": value}
    result = dict(value)
    routes = result.get("routes")
    if not isinstance(routes, list):
        return result
    unique: list[Any] = []
    duplicate_ids: dict[str, list[str]] = {}
    signatures: dict[str, str] = {}
    for index, route in enumerate(routes):
        if not isinstance(route, Mapping):
            unique.append(route)
            continue
        route_id = str(route.get("route_id") or route.get("id") or f"index:{index}")
        edge_ids = route.get("edge_ids") or route.get("reaction_ids") or []
        if not edge_ids:
            unique.append(route)
            continue
        signature = _digest(sorted(str(item) for item in edge_ids))
        if signature in signatures:
            duplicate_ids.setdefault(signatures[signature], []).append(route_id)
            continue
        signatures[signature] = route_id
        unique.append(route)
    result["routes"] = unique
    if duplicate_ids:
        result["duplicate_route_ids_by_representative"] = duplicate_ids
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CampaignContextError(
            f"campaign_context_not_canonicalizable:{type(exc).__name__}"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_content_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    return _canonical_bytes(value)


__all__ = [
    "CAMPAIGN_CONTEXT_DELTA_SCHEMA",
    "CAMPAIGN_CONTEXT_SCHEMA",
    "CampaignContext",
    "CampaignContextCompiler",
    "CampaignContextDelta",
    "CampaignContextError",
    "CampaignContextTooLargeError",
]

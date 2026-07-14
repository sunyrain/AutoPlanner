"""Single incremental work frontier for the V4 canonical hypergraph.

The frontier is a deterministic projection, not a second queue or source of
scientific authority.  Every item points back to canonical graph entities and
can be rebuilt fully or only for a dirty dependency closure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
)


DEFICIT_FRONTIER_SCHEMA = "deficit_frontier.v1"
DEFICIT_FRONTIER_ITEM_SCHEMA = "deficit_frontier_item.v1"


class DeficitKind(str, Enum):
    MATERIALIZATION = "materialization"
    EVIDENCE = "evidence"
    VALIDATION = "validation"
    STOCK = "stock"
    CONFLICT = "conflict"
    EXPANSION = "expansion"
    DIVERSITY = "diversity"
    ROUTE_CLOSURE = "route_closure"


_KIND_ORDER = {
    DeficitKind.CONFLICT: 0,
    DeficitKind.VALIDATION: 1,
    DeficitKind.EVIDENCE: 2,
    DeficitKind.STOCK: 3,
    DeficitKind.EXPANSION: 4,
    DeficitKind.MATERIALIZATION: 5,
    DeficitKind.ROUTE_CLOSURE: 6,
    DeficitKind.DIVERSITY: 7,
}


@dataclass(frozen=True, slots=True)
class DeficitScore:
    expected_portfolio_gain: float = 0.0
    distance_to_closure: float = 0.0
    evidence_gain: float = 0.0
    source_independence_gain: float = 0.0
    route_diversity_gain: float = 0.0
    cost_penalty: float = 0.0
    failure_risk_penalty: float = 0.0
    prior_attempt_penalty: float = 0.0

    @property
    def priority(self) -> float:
        value = (
            320.0 * self.expected_portfolio_gain
            + 220.0 * self.distance_to_closure
            + 180.0 * self.evidence_gain
            + 120.0 * self.source_independence_gain
            + 100.0 * self.route_diversity_gain
            - 80.0 * self.cost_penalty
            - 140.0 * self.failure_risk_penalty
            - 40.0 * self.prior_attempt_penalty
        )
        return round(value, 6)

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("deficit score components must be within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "priority": self.priority}


@dataclass(frozen=True, slots=True)
class DeficitItem:
    deficit_id: str
    kind: DeficitKind
    object_id: str
    entity_ids: tuple[str, ...]
    route_family_ids: tuple[str, ...] = ()
    dependency_ids: tuple[str, ...] = ()
    deterministic: bool = True
    model_allowed: bool = False
    reason: str = ""
    score: DeficitScore = field(default_factory=DeficitScore)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DEFICIT_FRONTIER_ITEM_SCHEMA

    def __post_init__(self) -> None:
        if not self.deficit_id or not self.object_id or not self.reason:
            raise ValueError("deficit identity, object, and reason are required")
        if self.model_allowed and self.deterministic:
            raise ValueError("deterministic deficits cannot require model authority")

    def to_dict(self) -> dict[str, Any]:
        row = {
            "schema_version": self.schema_version,
            "deficit_id": self.deficit_id,
            "kind": self.kind.value,
            "object_id": self.object_id,
            "entity_ids": list(self.entity_ids),
            "route_family_ids": list(self.route_family_ids),
            "dependency_ids": list(self.dependency_ids),
            "deterministic": self.deterministic,
            "model_allowed": self.model_allowed,
            "reason": self.reason,
            "score": self.score.to_dict(),
            "priority": self.score.priority,
            "metadata": _json_value(dict(self.metadata)),
        }
        row["content_sha256"] = _digest(row)
        return row


def compile_deficit_frontier(
    graph: Mapping[str, Any],
    *,
    acceptance_spec: RetrosynthesisAcceptanceSpec | None = None,
    prior_attempts: Mapping[str, int] | None = None,
    previous_frontier: Mapping[str, Any] | None = None,
    dirty_entity_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compile all deficits or incrementally replace dirty-dependent items."""
    acceptance = acceptance_spec or RetrosynthesisAcceptanceSpec()
    attempts = {str(key): max(0, int(value)) for key, value in dict(prior_attempts or {}).items()}
    dirty = (
        None
        if dirty_entity_ids is None
        else {str(value) for value in dirty_entity_ids if str(value)}
    )
    route_index = dict(dict(graph.get("dependency_index") or {}).get("routes_by_entity") or {})
    selected_routes = {
        route_id
        for route_id, route in dict(graph.get("route_families") or {}).items()
        if isinstance(route, Mapping)
        and route.get("selected") is not False
        and route.get("status") != "dominated"
    }

    items: list[DeficitItem] = []
    recomputed_entities: set[str] = set()

    def affected(entity_id: str) -> bool:
        return dirty is None or entity_id in dirty

    for hypothesis_id, raw in sorted(dict(graph.get("hypotheses") or {}).items()):
        if not isinstance(raw, Mapping) or not affected(str(hypothesis_id)):
            continue
        hypothesis = dict(raw)
        recomputed_entities.add(str(hypothesis_id))
        if hypothesis.get("status") != "frontier_candidate":
            continue
        routes = _routes_for(route_index, hypothesis_id)
        items.append(
            _item(
                DeficitKind.MATERIALIZATION,
                str(hypothesis_id),
                entity_ids=(str(hypothesis_id),),
                route_family_ids=routes,
                deterministic=True,
                model_allowed=False,
                reason="accepted_hypothesis_requires_host_materialization",
                score=_score(
                    DeficitKind.MATERIALIZATION,
                    selected=bool(set(routes) & selected_routes),
                    attempts=attempts.get(str(hypothesis_id), 0),
                    route_diversity=float(hypothesis.get("route_diversity_gain") or 0.0),
                ),
            )
        )

    for edge_id, raw in sorted(dict(graph.get("edges") or {}).items()):
        if not isinstance(raw, Mapping) or not affected(str(edge_id)):
            continue
        edge = dict(raw)
        recomputed_entities.add(str(edge_id))
        routes = _routes_for(route_index, edge_id)
        selected = bool(set(routes) & selected_routes)
        proof_level = _edge_proof_level(edge)
        exact_groups = {
            str(value)
            for value in edge.get("independent_source_groups") or []
            if str(value)
        }
        if proof_level < 2:
            items.append(
                _item(
                    DeficitKind.VALIDATION,
                    str(edge_id),
                    entity_ids=(str(edge_id),),
                    route_family_ids=routes,
                    deterministic=True,
                    model_allowed=False,
                    reason="materialized_edge_requires_reaction_validation",
                    score=_score(
                        DeficitKind.VALIDATION,
                        selected=selected,
                        attempts=attempts.get(str(edge_id), 0),
                    ),
                )
            )
        required = int(acceptance.minimum_edge_proof_level)
        if required >= 3 and (proof_level < 3 or not exact_groups):
            items.append(
                _item(
                    DeficitKind.EVIDENCE,
                    str(edge_id),
                    entity_ids=(str(edge_id),),
                    route_family_ids=routes,
                    deterministic=True,
                    model_allowed=False,
                    reason="edge_requires_exact_source_binding",
                    score=_score(
                        DeficitKind.EVIDENCE,
                        selected=selected,
                        attempts=attempts.get(str(edge_id), 0),
                        source_groups=len(exact_groups),
                    ),
                )
            )

    source_aliases = dict(graph.get("source_aliases") or {})
    exact_binding_ids = {
        str(
            source_aliases.get(str(row.get("source_binding_id") or ""))
            or row.get("source_binding_id")
            or ""
        )
        for row in dict(graph.get("exact_records") or {}).values()
        if isinstance(row, Mapping)
    }
    for source_id, raw in sorted(dict(graph.get("source_bindings") or {}).items()):
        if not isinstance(raw, Mapping) or not affected(str(source_id)):
            continue
        source = dict(raw)
        recomputed_entities.add(str(source_id))
        if str(source_id) in exact_binding_ids:
            continue
        status = str(source.get("acquisition_status") or "discovered")
        if status == "queued_for_authorized_browser":
            reason = "source_waiting_authorized_pdf_acquisition"
            model_allowed = False
        elif int(source.get("visual_candidate_page_count") or 0) > 0:
            reason = "source_material_requires_structure_or_procedure_extraction"
            model_allowed = True
        else:
            reason = "source_requires_extractable_full_text"
            model_allowed = False
        items.append(
            _item(
                DeficitKind.EVIDENCE,
                str(source_id),
                entity_ids=(str(source_id),),
                route_family_ids=_routes_for(route_index, str(source_id)),
                deterministic=False,
                model_allowed=model_allowed,
                reason=reason,
                score=_score(
                    DeficitKind.EVIDENCE,
                    selected=True,
                    attempts=attempts.get(str(source_id), 0),
                ),
                metadata={
                    "source_ref": str(source.get("source_ref") or ""),
                    "acquisition_status": status,
                    "source_pdf_sha256": str(
                        source.get("source_pdf_sha256") or ""
                    ),
                    "proxy_request_id": str(source.get("proxy_request_id") or ""),
                    "visual_candidate_page_count": int(
                        source.get("visual_candidate_page_count") or 0
                    ),
                },
            )
        )

    for molecule_id, raw in sorted(dict(graph.get("molecules") or {}).items()):
        if not isinstance(raw, Mapping) or not affected(str(molecule_id)):
            continue
        molecule = dict(raw)
        recomputed_entities.add(str(molecule_id))
        if (
            molecule_id == graph.get("target_molecule_id")
            or molecule.get("stock_closed") is True
        ):
            continue
        routes = _routes_for(route_index, molecule_id)
        selected = bool(set(routes) & selected_routes)
        active_stock_id = str(molecule.get("active_stock_observation_id") or "")
        active_stock = dict(
            dict(graph.get("stock_observations") or {}).get(active_stock_id) or {}
        )
        if (
            molecule.get("provider_expansion_requested") is True
            and active_stock.get("accepted") is not True
        ):
            items.append(
                _item(
                    DeficitKind.EXPANSION,
                    str(molecule_id),
                    entity_ids=(str(molecule_id),),
                    route_family_ids=routes,
                    deterministic=False,
                    model_allowed=True,
                    reason="codex_selected_frontier_requires_local_generation",
                    score=_score(
                        DeficitKind.EXPANSION,
                        selected=selected,
                        attempts=attempts.get(str(molecule_id), 0),
                        route_diversity=min(
                            1.0,
                            float(molecule.get("provider_expansion_priority") or 0.0)
                            / 10.0,
                        ),
                    ),
                    metadata={
                        "frontier_smiles": str(molecule.get("canonical_smiles") or ""),
                        "provider_preferences": list(
                            molecule.get("provider_preferences") or ["chemenzy"]
                        ),
                        "retron_hints": list(molecule.get("provider_retron_hints") or []),
                        "provider_request_ids": list(
                            molecule.get("provider_request_ids") or []
                        ),
                        "provider_request_rationale": str(
                            molecule.get("provider_request_rationale") or ""
                        ),
                    },
                )
            )
        # Codex may deliberately delegate a shared intermediate that already
        # has one proposed upstream edge.  ChemEnzy's role is to add local
        # alternatives around that node, so provider expansion is independent
        # of leaf status.  Stock closure, by contrast, remains leaf-only.
        if molecule.get("is_leaf") is not True:
            continue
        if active_stock_id and active_stock.get("accepted") is not True:
            items.append(
                _item(
                    DeficitKind.EXPANSION,
                    str(molecule_id),
                    entity_ids=(str(molecule_id), active_stock_id),
                    route_family_ids=routes,
                    deterministic=False,
                    model_allowed=True,
                    reason="stock_rejected_leaf_requires_upstream_expansion",
                    score=_score(
                        DeficitKind.EXPANSION,
                        selected=selected,
                        attempts=attempts.get(str(molecule_id), 0),
                    ),
                    metadata={
                        "frontier_smiles": str(
                            molecule.get("canonical_smiles") or ""
                        ),
                        "provider_preferences": ["chemenzy", "codex_global_director"],
                        "stock_observation_id": active_stock_id,
                    },
                )
            )
            continue
        items.append(
            _item(
                DeficitKind.STOCK,
                str(molecule_id),
                entity_ids=(str(molecule_id),),
                route_family_ids=routes,
                deterministic=True,
                model_allowed=False,
                reason=(
                    "selected_leaf_requires_trusted_stock_audit"
                    if selected
                    else "reachable_leaf_requires_trusted_stock_audit"
                ),
                score=_score(
                    DeficitKind.STOCK,
                    selected=selected,
                    attempts=attempts.get(str(molecule_id), 0),
                ),
            )
        )

    for conflict_id, raw in sorted(dict(graph.get("conflicts") or {}).items()):
        if not isinstance(raw, Mapping) or not affected(str(conflict_id)):
            continue
        conflict = dict(raw)
        recomputed_entities.add(str(conflict_id))
        if conflict.get("status") == "resolved":
            continue
        subject_id = str(conflict.get("subject_id") or conflict_id)
        routes = _routes_for(route_index, subject_id)
        items.append(
            _item(
                DeficitKind.CONFLICT,
                str(conflict_id),
                entity_ids=(str(conflict_id), subject_id),
                route_family_ids=routes,
                deterministic=True,
                model_allowed=False,
                reason=str(conflict.get("conflict_kind") or "unresolved_source_conflict"),
                score=_score(
                    DeficitKind.CONFLICT,
                    selected=bool(set(routes) & selected_routes),
                    attempts=attempts.get(str(conflict_id), 0),
                ),
            )
        )

    route_dirty = dirty is None or bool(
        dirty
        and (
            set(graph.get("edges") or {})
            | set(graph.get("hypotheses") or {})
            | set(graph.get("route_families") or {})
        )
        & dirty
    )
    for route_id, raw in sorted(dict(graph.get("route_families") or {}).items()):
        if not isinstance(raw, Mapping) or not affected(str(route_id)):
            continue
        route_dirty = True
        route = dict(raw)
        recomputed_entities.add(str(route_id))
        if route.get("status") == "dominated" or route.get("selected") is False:
            continue
        if route.get("closed") is not True:
            items.append(
                _item(
                    DeficitKind.ROUTE_CLOSURE,
                    str(route_id),
                    entity_ids=tuple(
                        sorted(
                            {
                                str(route_id),
                                *(str(value) for value in route.get("edge_ids") or []),
                                *(str(value) for value in route.get("leaf_molecule_ids") or []),
                            }
                        )
                    ),
                    route_family_ids=(str(route_id),),
                    dependency_ids=tuple(
                        str(value) for value in route.get("blocking_deficit_ids") or []
                    ),
                    deterministic=True,
                    model_allowed=False,
                    reason="selected_route_family_not_closed",
                    score=_score(
                        DeficitKind.ROUTE_CLOSURE,
                        selected=True,
                        attempts=attempts.get(str(route_id), 0),
                    ),
                )
            )

    if route_dirty:
        complete_routes = sum(
            1
            for route_id, route in dict(graph.get("route_families") or {}).items()
            if route_id in selected_routes
            and isinstance(route, Mapping)
            and route.get("closed") is True
        )
        if complete_routes < acceptance.minimum_complete_routes:
            items.append(
                _item(
                    DeficitKind.DIVERSITY,
                    "selected-route-portfolio",
                    entity_ids=tuple(sorted(selected_routes)),
                    route_family_ids=tuple(sorted(selected_routes)),
                    deterministic=False,
                    model_allowed=True,
                    reason=(
                        f"closed_route_count_{complete_routes}_below_required_"
                        f"{acceptance.minimum_complete_routes}"
                    ),
                    score=_score(
                        DeficitKind.DIVERSITY,
                        selected=True,
                        attempts=attempts.get("selected-route-portfolio", 0),
                        route_diversity=1.0,
                    ),
                )
            )

    if dirty is not None and previous_frontier:
        for raw in previous_frontier.get("items") or previous_frontier.get("deficits") or []:
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            entity_ids = {str(value) for value in row.get("entity_ids") or []}
            if entity_ids & dirty:
                continue
            try:
                items.append(_item_from_dict(row))
            except (TypeError, ValueError):
                continue

    deduped = {item.deficit_id: item for item in items}
    ordered = sorted(
        deduped.values(),
        key=lambda item: (
            -item.score.priority,
            _KIND_ORDER[item.kind],
            item.object_id,
            item.deficit_id,
        ),
    )
    result = {
        "schema_version": DEFICIT_FRONTIER_SCHEMA,
        "graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "items": [item.to_dict() for item in ordered],
        "summary": {
            "total": len(ordered),
            "deterministic": sum(item.deterministic for item in ordered),
            "model_eligible": sum(item.model_allowed for item in ordered),
            "by_kind": {
                kind.value: sum(item.kind == kind for item in ordered)
                for kind in DeficitKind
            },
            "next_deficit_id": ordered[0].deficit_id if ordered else "",
            "next_kind": ordered[0].kind.value if ordered else "",
        },
        "incremental": dirty is not None,
        "dirty_entity_ids": sorted(dirty or []),
        "recomputed_entity_ids": sorted(recomputed_entities),
        "semantics": {
            "single_work_projection": True,
            "frontier_is_not_scientific_authority": True,
            "deterministic_work_precedes_model_work_by_score": True,
            "tie_breaking_is_deterministic": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def frontier_scientific_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return revision-independent frontier content for oracle comparison."""
    rows = []
    for raw in value.get("items") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        row.pop("content_sha256", None)
        rows.append(row)
    return {
        "schema_version": DEFICIT_FRONTIER_SCHEMA,
        "items": sorted(rows, key=lambda row: str(row.get("deficit_id") or "")),
    }


def compile_selected_route_deficits(
    selected_routes: Iterable[Mapping[str, Any]],
    *,
    edge_proofs: Mapping[str, Mapping[str, Any]],
    acceptance_spec: RetrosynthesisAcceptanceSpec,
) -> list[dict[str, Any]]:
    """Compile proof-stitched portfolio gaps with the canonical V4 taxonomy.

    Route selection adds variant-level membership, but it must not introduce a
    second deficit schema or queue.  The returned records are ordinary
    :class:`DeficitItem` rows and can be projected directly into ``RunKernel``.
    """
    grouped: dict[tuple[DeficitKind, str], dict[str, Any]] = {}

    def add(
        kind: DeficitKind,
        object_id: str,
        *,
        route: Mapping[str, Any] | None,
        entity_ids: Iterable[str],
        reason: str,
        deterministic: bool = True,
        model_allowed: bool = False,
        source_groups: int = 0,
        route_diversity: float = 0.0,
    ) -> None:
        key = (kind, object_id)
        current = grouped.setdefault(
            key,
            {
                "entity_ids": set(),
                "route_family_ids": set(),
                "route_ids": set(),
                "reasons": set(),
                "deterministic": deterministic,
                "model_allowed": model_allowed,
                "source_groups": source_groups,
                "route_diversity": route_diversity,
            },
        )
        current["entity_ids"].update(str(value) for value in entity_ids if str(value))
        current["reasons"].add(reason)
        current["deterministic"] = current["deterministic"] and deterministic
        current["model_allowed"] = current["model_allowed"] or model_allowed
        current["source_groups"] = max(current["source_groups"], source_groups)
        current["route_diversity"] = max(
            current["route_diversity"], route_diversity
        )
        if route:
            route_id = str(route.get("route_id") or "")
            family_id = str(route.get("route_family_id") or "")
            if route_id:
                current["route_ids"].add(route_id)
            if family_id:
                current["route_family_ids"].add(family_id)

    routes = [dict(value) for value in selected_routes]
    for route in routes:
        for edge_id in route.get("unproven_edge_ids") or []:
            proof = dict(edge_proofs.get(str(edge_id)) or {})
            if proof.get("reaction_validated") is not True:
                kind = DeficitKind.VALIDATION
                reason = "selected_edge_requires_reaction_validation"
            else:
                kind = DeficitKind.EVIDENCE
                reason = (
                    "selected_edge_requires_exact_source_binding"
                    if proof.get("exact_source_bound") is not True
                    else "selected_edge_proof_below_policy"
                )
            add(
                kind,
                str(edge_id),
                route=route,
                entity_ids=(str(edge_id),),
                reason=reason,
                source_groups=len(proof.get("independent_source_groups") or []),
            )
        for molecule_id in route.get("open_leaf_molecule_ids") or []:
            add(
                DeficitKind.STOCK,
                str(molecule_id),
                route=route,
                entity_ids=(str(molecule_id),),
                reason="selected_leaf_requires_trusted_stock_audit",
            )
        for conflict_id in route.get("conflict_ids") or []:
            add(
                DeficitKind.CONFLICT,
                str(conflict_id),
                route=route,
                entity_ids=(str(conflict_id),),
                reason="selected_route_contains_unresolved_conflict",
            )
        if route.get("source_independence_met") is not True:
            groups = len(route.get("independent_source_groups") or [])
            add(
                DeficitKind.EVIDENCE,
                str(route.get("route_id") or "selected-route"),
                route=route,
                entity_ids=tuple(route.get("edge_ids") or []),
                reason=(
                    f"independent_source_groups_{groups}_below_required_"
                    f"{acceptance_spec.minimum_independent_source_groups}"
                ),
                source_groups=groups,
            )
        if route.get("complete") is not True:
            add(
                DeficitKind.ROUTE_CLOSURE,
                str(route.get("route_id") or "selected-route"),
                route=route,
                entity_ids=(
                    *tuple(route.get("edge_ids") or []),
                    *tuple(route.get("leaf_molecule_ids") or []),
                ),
                reason="selected_proof_stitched_route_not_closed",
            )

    complete = sum(route.get("complete") is True for route in routes)
    if complete < acceptance_spec.minimum_complete_routes:
        add(
            DeficitKind.DIVERSITY,
            "selected-route-portfolio",
            route=None,
            entity_ids=tuple(route.get("route_id") or "" for route in routes),
            reason=(
                f"complete_route_count_{complete}_below_required_"
                f"{acceptance_spec.minimum_complete_routes}"
            ),
            deterministic=False,
            model_allowed=True,
            route_diversity=1.0,
        )

    items: list[DeficitItem] = []
    for (kind, object_id), value in grouped.items():
        reasons = sorted(value["reasons"])
        items.append(
            _item(
                kind,
                object_id,
                entity_ids=value["entity_ids"],
                route_family_ids=value["route_family_ids"],
                deterministic=value["deterministic"],
                model_allowed=value["model_allowed"],
                reason=reasons[0],
                score=_score(
                    kind,
                    selected=True,
                    attempts=0,
                    source_groups=value["source_groups"],
                    route_diversity=value["route_diversity"],
                ),
                metadata={
                    "route_ids": sorted(value["route_ids"]),
                    "reasons": reasons,
                    "proof_stitched": True,
                },
            )
        )
    return [
        item.to_dict()
        for item in sorted(
            items,
            key=lambda item: (
                -item.score.priority,
                _KIND_ORDER[item.kind],
                item.object_id,
                item.deficit_id,
            ),
        )
    ]


def _item(
    kind: DeficitKind,
    object_id: str,
    *,
    entity_ids: Iterable[str],
    route_family_ids: Iterable[str] = (),
    dependency_ids: Iterable[str] = (),
    deterministic: bool,
    model_allowed: bool,
    reason: str,
    score: DeficitScore,
    metadata: Mapping[str, Any] | None = None,
) -> DeficitItem:
    identity = f"{kind.value}\0{object_id}".encode("utf-8")
    return DeficitItem(
        deficit_id=f"deficit:{kind.value}:{hashlib.sha256(identity).hexdigest()}",
        kind=kind,
        object_id=object_id,
        entity_ids=tuple(sorted({str(value) for value in entity_ids if str(value)})),
        route_family_ids=tuple(
            sorted({str(value) for value in route_family_ids if str(value)})
        ),
        dependency_ids=tuple(
            sorted({str(value) for value in dependency_ids if str(value)})
        ),
        deterministic=deterministic,
        model_allowed=model_allowed,
        reason=reason,
        score=score,
        metadata=dict(metadata or {}),
    )


def _item_from_dict(value: Mapping[str, Any]) -> DeficitItem:
    row = dict(value)
    score = dict(row.get("score") or {})
    score.pop("priority", None)
    return DeficitItem(
        deficit_id=str(row.get("deficit_id") or ""),
        kind=DeficitKind(str(row.get("kind") or "")),
        object_id=str(row.get("object_id") or ""),
        entity_ids=tuple(str(value) for value in row.get("entity_ids") or []),
        route_family_ids=tuple(
            str(value) for value in row.get("route_family_ids") or []
        ),
        dependency_ids=tuple(str(value) for value in row.get("dependency_ids") or []),
        deterministic=row.get("deterministic") is True,
        model_allowed=row.get("model_allowed") is True,
        reason=str(row.get("reason") or ""),
        score=DeficitScore(**score),
        metadata=dict(row.get("metadata") or {}),
    )


def _score(
    kind: DeficitKind,
    *,
    selected: bool,
    attempts: int,
    source_groups: int = 0,
    route_diversity: float = 0.0,
) -> DeficitScore:
    defaults = {
        DeficitKind.MATERIALIZATION: (0.55, 0.45, 0.15, 0.10, 0.35, 0.15, 0.20),
        DeficitKind.EVIDENCE: (0.72, 0.70, 1.00, 0.85, 0.20, 0.35, 0.15),
        DeficitKind.VALIDATION: (0.85, 0.85, 0.55, 0.20, 0.15, 0.20, 0.18),
        DeficitKind.STOCK: (0.78, 0.90, 0.15, 0.10, 0.10, 0.10, 0.08),
        DeficitKind.EXPANSION: (0.70, 0.68, 0.10, 0.10, 0.45, 0.42, 0.38),
        DeficitKind.CONFLICT: (0.92, 0.80, 0.75, 0.65, 0.20, 0.30, 0.45),
        DeficitKind.DIVERSITY: (0.50, 0.20, 0.10, 0.15, 1.00, 0.75, 0.55),
        DeficitKind.ROUTE_CLOSURE: (0.88, 0.75, 0.30, 0.20, 0.35, 0.15, 0.12),
    }
    portfolio, distance, evidence, independence, diversity, cost, risk = defaults[kind]
    if selected:
        portfolio = min(1.0, portfolio + 0.12)
        distance = min(1.0, distance + 0.08)
    if kind == DeficitKind.EVIDENCE:
        independence = 1.0 if source_groups == 1 else 0.55 if source_groups == 0 else 0.15
    diversity = max(diversity, _unit(route_diversity))
    return DeficitScore(
        expected_portfolio_gain=portfolio,
        distance_to_closure=distance,
        evidence_gain=evidence,
        source_independence_gain=independence,
        route_diversity_gain=diversity,
        cost_penalty=cost,
        failure_risk_penalty=risk,
        prior_attempt_penalty=min(1.0, max(0, int(attempts)) / 5.0),
    )


def _routes_for(index: Mapping[str, Any], entity_id: Any) -> tuple[str, ...]:
    values = index.get(str(entity_id)) or []
    return tuple(sorted({str(value) for value in values if str(value)}))


def _edge_proof_level(edge: Mapping[str, Any]) -> int:
    level = 0
    for proof in edge.get("reaction_proofs") or []:
        if not isinstance(proof, Mapping):
            continue
        name = str(proof.get("proof_level") or "")
        if name == "L4_procurement_ready":
            level = max(level, 4)
        elif name == "L3_precedent_supported":
            level = max(level, 3)
        elif proof.get("accepted") is True or name == "L2_reaction_validated":
            level = max(level, 2)
    return level


def _unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

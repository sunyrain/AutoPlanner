"""One deterministic work projection for route-closing deficits.

Retrosynthesis previously exposed proposal frontiers, source capabilities,
structure tasks, bridge tasks, and proof requests as unrelated queues.  This
module projects those orthogonal records into one ordered list of *deficits*.
It does not replace the durable lease implementation yet; it replaces the
ambiguous question of which subsystem is allowed to request the next unit of
work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
)


ROUTE_DEFICIT_QUEUE_SCHEMA = "route_deficit_queue.v1"
ROUTE_DEFICIT_SCHEMA = "route_deficit.v1"


class RouteDeficitKind(str, Enum):
    EXACT_EVIDENCE = "exact_evidence"
    REACTION_VALIDATION = "reaction_validation"
    STOCK_AUDIT = "stock_audit"
    STRUCTURE_MATERIALIZATION = "structure_materialization"
    PROPOSAL_EXPANSION = "proposal_expansion"
    ROUTE_DIVERSITY = "route_diversity"


_KIND_ORDER = {
    RouteDeficitKind.EXACT_EVIDENCE: 0,
    RouteDeficitKind.REACTION_VALIDATION: 1,
    RouteDeficitKind.STOCK_AUDIT: 2,
    RouteDeficitKind.STRUCTURE_MATERIALIZATION: 3,
    RouteDeficitKind.PROPOSAL_EXPANSION: 4,
    RouteDeficitKind.ROUTE_DIVERSITY: 5,
}


@dataclass(frozen=True)
class RouteDeficit:
    deficit_id: str
    kind: RouteDeficitKind
    object_id: str
    route_ids: tuple[str, ...] = ()
    deterministic: bool = True
    model_allowed: bool = False
    priority: float = 0.0
    reason: str = ""
    source_refs: tuple[str, ...] = ()
    dependency_ids: tuple[str, ...] = ()
    schema_version: str = ROUTE_DEFICIT_SCHEMA

    def __post_init__(self) -> None:
        if not self.deficit_id or not self.object_id or not self.reason:
            raise ValueError("route deficit identity and reason are required")
        if self.model_allowed and self.deterministic:
            raise ValueError("deterministic deficit cannot require model authority")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["kind"] = self.kind.value
        row["route_ids"] = list(self.route_ids)
        row["source_refs"] = list(self.source_refs)
        row["dependency_ids"] = list(self.dependency_ids)
        return row


def compile_route_deficit_queue(
    *,
    frontier_ledger: Mapping[str, Any] | None,
    route_portfolio: Mapping[str, Any] | None = None,
    source_capability_queue: Mapping[str, Any] | None = None,
    acceptance_spec: RetrosynthesisAcceptanceSpec | None = None,
) -> dict[str, Any]:
    """Compile all currently actionable closure gaps into one ordered queue."""

    acceptance = acceptance_spec or RetrosynthesisAcceptanceSpec()
    ledger = dict(frontier_ledger or {})
    portfolio = dict(route_portfolio or {})
    capabilities = dict(source_capability_queue or {})
    deficits: list[RouteDeficit] = []

    selected_route_ids, edge_route_ids, molecule_route_ids = _route_membership(
        portfolio
    )

    def route_bonus(values: Iterable[str]) -> float:
        return 100.0 if set(values) & selected_route_ids else 0.0

    for signature, raw in sorted(dict(ledger.get("edges") or {}).items()):
        if not isinstance(raw, Mapping):
            continue
        edge = dict(raw)
        proof = dict(edge.get("reaction_proof") or {})
        route_ids = tuple(sorted(edge_route_ids.get(str(signature), ())))
        achieved = _proof_level(proof)
        if achieved < acceptance.minimum_edge_proof_level:
            kind = (
                RouteDeficitKind.REACTION_VALIDATION
                if achieved < 2
                else RouteDeficitKind.EXACT_EVIDENCE
            )
            deficits.append(
                _deficit(
                    kind,
                    object_id=str(signature),
                    route_ids=route_ids,
                    deterministic=True,
                    model_allowed=False,
                    priority=(
                        900.0
                        + route_bonus(route_ids)
                        + 10.0
                        * (acceptance.minimum_edge_proof_level - achieved)
                    ),
                    reason=(
                        f"edge_proof_level_{achieved}_below_required_"
                        f"{acceptance.minimum_edge_proof_level}"
                    ),
                    source_refs=tuple(
                        sorted(
                            str(item)
                            for item in dict(edge.get("proposal") or {}).get(
                                "source_refs"
                            )
                            or []
                            if str(item or "").strip()
                        )
                    ),
                )
            )

    for smiles, raw in sorted(dict(ledger.get("molecules") or {}).items()):
        if not isinstance(raw, Mapping):
            continue
        molecule = dict(raw)
        route_ids = tuple(sorted(molecule_route_ids.get(str(smiles), ())))
        stock = dict(molecule.get("stock") or {})
        proposal = dict(molecule.get("proposal") or {})
        work = dict(molecule.get("work") or {})
        is_leaf = not list(proposal.get("outgoing_edge_signatures") or [])
        stock_closed = _stock_closed(
            stock,
            boundary=acceptance.stock_boundary,
        )
        stock_audited = bool(
            stock.get("observation_job_ids")
            or stock.get("current_observation_ids")
            or stock.get("rejected_stock_job_ids")
        )
        if is_leaf and not stock_closed and not stock_audited:
            deficits.append(
                _deficit(
                    RouteDeficitKind.STOCK_AUDIT,
                    object_id=str(smiles),
                    route_ids=route_ids,
                    deterministic=True,
                    model_allowed=False,
                    priority=800.0 + route_bonus(route_ids),
                    reason=(
                        "selected_leaf_stock_boundary_open"
                        if route_bonus(route_ids)
                        else "reachable_leaf_stock_boundary_open"
                    ),
                )
            )
        if (
            is_leaf
            and not stock_closed
            and work.get("proposal_expansion_allowed") is True
        ):
            deficits.append(
                _deficit(
                    RouteDeficitKind.PROPOSAL_EXPANSION,
                    object_id=str(smiles),
                    route_ids=route_ids,
                    deterministic=False,
                    model_allowed=True,
                    priority=300.0 + route_bonus(route_ids),
                    reason="leaf_not_stock_closed_and_no_outgoing_reaction",
                    dependency_ids=(
                        ()
                        if stock_audited
                        else (
                            _stable_deficit_id(
                                RouteDeficitKind.STOCK_AUDIT,
                                str(smiles),
                            ),
                        )
                    ),
                )
            )

    deficits.extend(
        _source_capability_deficits(
            capabilities,
            selected_route_ids=selected_route_ids,
        )
    )

    complete_routes = _complete_route_count(portfolio)
    if complete_routes < acceptance.minimum_complete_routes:
        deficits.append(
            _deficit(
                RouteDeficitKind.ROUTE_DIVERSITY,
                object_id="selected-route-portfolio",
                route_ids=tuple(sorted(selected_route_ids)),
                deterministic=True,
                model_allowed=False,
                priority=100.0,
                reason=(
                    f"complete_route_count_{complete_routes}_below_required_"
                    f"{acceptance.minimum_complete_routes}"
                ),
            )
        )

    deduped = _deduplicate(deficits)
    ordered = sorted(
        deduped,
        key=lambda item: (
            -item.priority,
            _KIND_ORDER[item.kind],
            item.object_id,
            item.deficit_id,
        ),
    )
    queue: dict[str, Any] = {
        "schema_version": ROUTE_DEFICIT_QUEUE_SCHEMA,
        "acceptance_spec": acceptance.to_dict(),
        "deficits": [item.to_dict() for item in ordered],
        "summary": {
            "total": len(ordered),
            "deterministic": sum(1 for item in ordered if item.deterministic),
            "model_eligible": sum(1 for item in ordered if item.model_allowed),
            "by_kind": {
                kind.value: sum(1 for item in ordered if item.kind == kind)
                for kind in RouteDeficitKind
            },
            "next_deficit_id": ordered[0].deficit_id if ordered else "",
            "next_kind": ordered[0].kind.value if ordered else "",
        },
        "semantics": {
            "single_cross_subsystem_work_projection": True,
            "deterministic_route_closure_gaps_precede_model_expansion": True,
            "queue_is_not_chemistry_authority": True,
            "selected_route_gaps_receive_priority": True,
        },
    }
    queue["content_sha256"] = _digest(queue)
    return queue


def next_route_deficit(
    queue: Mapping[str, Any] | None,
    *,
    allow_model: bool,
) -> dict[str, Any]:
    for raw in dict(queue or {}).get("deficits") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if row.get("model_allowed") is True and not allow_model:
            continue
        return row
    return {}


def _route_membership(
    portfolio: Mapping[str, Any],
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    selected: set[str] = set()
    edge_routes: dict[str, set[str]] = {}
    molecule_routes: dict[str, set[str]] = {}
    rows = (
        portfolio.get("routes")
        or portfolio.get("items")
        or portfolio.get("portfolio")
        or []
    )
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            continue
        route = dict(raw)
        route_id = str(
            route.get("route_id")
            or route.get("portfolio_route_id")
            or f"route:{index}"
        )
        if route.get("selected") is not False:
            selected.add(route_id)
        edges = (
            route.get("edge_signatures")
            or route.get("selected_edge_signatures")
            or route.get("hyperedge_ids")
            or route.get("edge_ids")
            or []
        )
        for edge_id in edges:
            edge_routes.setdefault(str(edge_id), set()).add(route_id)
        molecules = (
            route.get("molecule_smiles")
            or route.get("molecule_ids")
            or route.get("leaf_smiles")
            or route.get("terminal_smiles")
            or []
        )
        for smiles in molecules:
            molecule_routes.setdefault(str(smiles), set()).add(route_id)
    return selected, edge_routes, molecule_routes


def _source_capability_deficits(
    queue: Mapping[str, Any],
    *,
    selected_route_ids: set[str],
) -> list[RouteDeficit]:
    out: list[RouteDeficit] = []
    kind_by_action = {
        "extract_pdf_literature_structures": (
            RouteDeficitKind.STRUCTURE_MATERIALIZATION
        ),
        "extract_visual_literature_chain": (
            RouteDeficitKind.STRUCTURE_MATERIALIZATION
        ),
        "resolve_literature_structure_task": (
            RouteDeficitKind.STRUCTURE_MATERIALIZATION
        ),
        "compile_exact_literature_rows": RouteDeficitKind.EXACT_EVIDENCE,
    }
    for index, raw in enumerate(queue.get("capabilities") or [], start=1):
        if not isinstance(raw, Mapping) or raw.get("eligible") is not True:
            continue
        row = dict(raw)
        action_type = str(row.get("action_type") or "")
        kind = kind_by_action.get(action_type)
        if kind is None:
            continue
        object_id = str(
            row.get("capability_id")
            or row.get("source_ref")
            or row.get("task_id")
            or f"source-capability:{index}"
        )
        route_ids = tuple(
            sorted(
                str(item)
                for item in row.get("route_ids") or []
                if str(item or "").strip()
            )
        )
        out.append(
            _deficit(
                kind,
                object_id=object_id,
                route_ids=route_ids,
                deterministic=True,
                model_allowed=False,
                priority=(
                    950.0
                    if kind == RouteDeficitKind.EXACT_EVIDENCE
                    else 700.0
                )
                + (100.0 if set(route_ids) & selected_route_ids else 0.0),
                reason=f"eligible_source_capability:{action_type}",
                source_refs=tuple(
                    [str(row.get("source_ref"))]
                    if str(row.get("source_ref") or "").strip()
                    else []
                ),
            )
        )
    return out


def _deficit(
    kind: RouteDeficitKind,
    *,
    object_id: str,
    route_ids: Iterable[str] = (),
    deterministic: bool,
    model_allowed: bool,
    priority: float,
    reason: str,
    source_refs: Iterable[str] = (),
    dependency_ids: Iterable[str] = (),
) -> RouteDeficit:
    return RouteDeficit(
        deficit_id=_stable_deficit_id(kind, object_id),
        kind=kind,
        object_id=object_id,
        route_ids=tuple(sorted(set(route_ids))),
        deterministic=deterministic,
        model_allowed=model_allowed,
        priority=float(priority),
        reason=reason,
        source_refs=tuple(sorted(set(source_refs))),
        dependency_ids=tuple(sorted(set(dependency_ids))),
    )


def _stable_deficit_id(kind: RouteDeficitKind, object_id: str) -> str:
    payload = f"{kind.value}\0{object_id}".encode("utf-8")
    return f"deficit:{kind.value}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _proof_level(proof: Mapping[str, Any]) -> int:
    for key in (
        "achieved_proof_level",
        "portfolio_proof_level",
        "level",
    ):
        try:
            if key in proof:
                return max(0, min(4, int(proof.get(key) or 0)))
        except (TypeError, ValueError):
            pass
    named = str(proof.get("proof_level") or "")
    for level in range(4, -1, -1):
        if named.startswith(f"L{level}"):
            return level
    return 0


def _stock_closed(stock: Mapping[str, Any], *, boundary: str) -> bool:
    if boundary == "procurement":
        return stock.get("procurement_boundary_closed") is True
    if boundary == "in_house":
        return stock.get("in_house_boundary_closed") is True
    return stock.get("closed") is True


def _complete_route_count(portfolio: Mapping[str, Any]) -> int:
    rows = (
        portfolio.get("routes")
        or portfolio.get("items")
        or portfolio.get("portfolio")
        or []
    )
    return sum(
        1
        for row in rows
        if isinstance(row, Mapping)
        and (
            row.get("complete") is True
            or row.get("proof_eligible") is True
            or row.get("closed") is True
        )
    )


def _deduplicate(values: Iterable[RouteDeficit]) -> list[RouteDeficit]:
    by_id: dict[str, RouteDeficit] = {}
    for value in values:
        prior = by_id.get(value.deficit_id)
        if prior is None or value.priority > prior.priority:
            by_id[value.deficit_id] = value
    return list(by_id.values())


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ROUTE_DEFICIT_QUEUE_SCHEMA",
    "ROUTE_DEFICIT_SCHEMA",
    "RouteDeficit",
    "RouteDeficitKind",
    "compile_route_deficit_queue",
    "next_route_deficit",
]

"""Route-family edge scope and guided-provider binding helpers.

A canonical edge may be retained for diagnosis without being a legitimate
continuation of every route family listed on it.  In particular, historical
guided-provider calls could be launched at a node that later materialized as
an internal strategic intermediate.  Route traversal must honor the explicit
family exclusion rather than treating route-family membership as sufficient
topology authority.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


_GUIDED_PROVIDER_KINDS = {"aizynthfinder", "chemenzy"}
_GUIDED_GROUP = re.compile(r"^((?:aizynthfinder|chemenzy):guided-[^:]+)")


def origin_condition_predictions(
    edge: Mapping[str, Any], origin: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Read the chosen implementation, including an explicitly empty choice.

    Shared edge annotations describe alternatives, not a branch's selection.
    Old records can use them only when there is at most one origin; otherwise
    missing attribution stays missing rather than borrowing a sibling's recipe.
    """

    if "condition_predictions" in origin:
        values = origin["condition_predictions"]
    elif len(edge.get("origin_records") or ()) <= 1:
        values = edge.get("condition_predictions")
    else:
        values = ()
    return [dict(value) for value in values or () if isinstance(value, Mapping)]


def route_family_bound_origin_records(
    edge: Mapping[str, Any],
    *,
    route_family_id: str,
) -> tuple[dict[str, Any], ...]:
    """Return origins explicitly bound to one route, without order guessing.

    A sole origin is an unambiguous legacy fallback.  Multiple origins with no
    explicit scalar or canonical-family binding remain unbound so callers can
    fall back to the canonical edge identity.
    """

    origins = tuple(
        dict(value)
        for value in edge.get("origin_records") or ()
        if isinstance(value, Mapping)
    )
    matches = tuple(
        origin
        for origin in origins
        if route_family_id
        and (
            str(origin.get("route_family_id") or "") == route_family_id
            or route_family_id
            in {
                str(value)
                for value in origin.get("canonical_route_family_ids") or ()
            }
        )
    )
    if matches:
        return matches
    return origins if len(origins) == 1 else ()


def guided_provider_group_ids(edge: Mapping[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()
    for raw in edge.get("origin_records") or []:
        if not isinstance(raw, Mapping):
            continue
        origin = dict(raw)
        if str(origin.get("origin_kind") or "") not in _GUIDED_PROVIDER_KINDS:
            continue
        metadata = dict(origin.get("provider_reaction_metadata") or {})
        binding = dict(metadata.get("short_tail_binding") or {})
        explicit = str(binding.get("provider_group_id") or "")
        if explicit:
            values.add(explicit)
            continue
        match = _GUIDED_GROUP.match(str(origin.get("origin_ref") or ""))
        if match:
            values.add(match.group(1))
    return tuple(sorted(values))


def route_family_scoped_edge_ids(
    graph: Mapping[str, Any],
    *,
    family: Mapping[str, Any],
) -> set[str]:
    """Return materialized edges allowed to participate in this family.

    Exclusion applies only to provider-only edges.  If the same canonical edge
    also has an independent Codex/template/literature origin, the chemistry
    remains available through that independent route authority.
    """

    edges = dict(graph.get("edges") or {})
    excluded = {
        str(value)
        for value in family.get("excluded_provider_group_ids") or []
        if str(value)
    }
    allowed: set[str] = set()
    for raw_edge_id in family.get("edge_ids") or []:
        edge_id = str(raw_edge_id)
        edge = dict(edges.get(edge_id) or {})
        if not edge:
            continue
        groups = set(guided_provider_group_ids(edge))
        has_independent_origin = any(
            isinstance(origin, Mapping)
            and not (
                str(origin.get("origin_kind") or "") in _GUIDED_PROVIDER_KINDS
                and bool(_GUIDED_GROUP.match(str(origin.get("origin_ref") or "")))
            )
            for origin in edge.get("origin_records") or []
        )
        if groups and groups <= excluded and not has_independent_origin:
            continue
        allowed.add(edge_id)
    return allowed


__all__ = [
    "guided_provider_group_ids",
    "origin_condition_predictions",
    "route_family_bound_origin_records",
    "route_family_scoped_edge_ids",
]

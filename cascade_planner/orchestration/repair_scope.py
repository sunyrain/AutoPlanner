"""Host-owned repair component selection and minimal connected graph interventions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from cascade_planner.application.routejson_compiler import RouteJSONCompiler
from cascade_planner.application.reactionjson_replay import ReactionJsonReplayError


def minimum_connected_repair_indices(
    step_ids: Sequence[str], parent_indices: Sequence[int | None], change_step_ids: Iterable[str],
) -> frozenset[int]:
    """Unique smallest connected subtree containing the requested route occurrences.

    The input is the compiler's occurrence tree, not an arbitrary molecular
    graph. Least-common-ancestor paths avoid deleting interleaved siblings.
    At most one root may be involved; target identity and exact cut chemistry
    are still checked by the ordinary route replay and suffix reconnection.
    """
    if len(step_ids) != len(parent_indices) or len(set(step_ids)) != len(step_ids):
        raise ValueError("path_repair_occurrence_tree_invalid")
    positions = {step_id: index for index, step_id in enumerate(step_ids)}
    requested = set(change_step_ids)
    if not requested:
        raise ValueError("path_repair_change_step_ids_missing")
    if requested - positions.keys():
        raise ValueError("path_repair_change_step_not_found")
    paths = []
    for step_id in requested:
        path = []
        cursor = positions[step_id]
        while cursor is not None:
            if cursor in path or not 0 <= cursor < len(step_ids):
                raise ValueError("path_repair_occurrence_tree_invalid")
            path.append(cursor)
            cursor = parent_indices[cursor]
        paths.append(path)
    common = set(paths[0]).intersection(*map(set, paths[1:]))
    if not common:
        raise ValueError("path_repair_change_steps_disconnected")
    ancestor = next(index for index in paths[0] if index in common)
    return frozenset(index for path in paths for index in path[:path.index(ancestor) + 1])


@dataclass(frozen=True, slots=True)
class _PathRepairBlockerScope:
    """One topology- or chemistry-coupled component for a repair transaction."""

    selected_step_ids: tuple[str, ...]
    deferred_step_ids: tuple[str, ...]
    component_step_ids: tuple[tuple[str, ...], ...]


def _select_path_repair_blocker_scope(
    *,
    current_steps: Iterable[Mapping[str, Any]],
    mapped_target_smiles: str,
    blocking_steps: Iterable[Mapping[str, Any]],
) -> tuple[_PathRepairBlockerScope | None, dict[str, Any]]:
    """Select one chemically coherent blocker component for one transaction.

    Host dependency ancestry and the Critic's explicit ``coupled_step_ids``
    are the only grouping authorities.  The latter joins topological siblings
    only when their remedies share an inseparable functional-state or sequence
    dependency; prose keywords never grant repair scope.
    """

    rows = [dict(row) for row in current_steps if isinstance(row, Mapping)]
    if not rows:
        return None, {"reason": "path_repair_current_route_empty"}
    compiler = RouteJSONCompiler()
    try:
        state = compiler.compile_route_graph_state(
            mapped_target_smiles=str(mapped_target_smiles or ""),
            steps=rows,
            minimum_depth=1,
        )
    except ReactionJsonReplayError as exc:
        return None, {
            "reason": "path_repair_current_route_not_replayable",
            "compiler_error": str(exc),
        }
    host_rows = compiler.assemble_route(state.reactions, metadata=rows)
    step_ids = [str(row.get("step_id") or "") for row in host_rows]
    if any(not value for value in step_ids) or len(set(step_ids)) != len(step_ids):
        return None, {"reason": "path_repair_current_step_ids_invalid"}
    requested_ids = list(
        dict.fromkeys(
            str(row.get("step_id") or "").strip()
            for row in blocking_steps
            if isinstance(row, Mapping) and str(row.get("step_id") or "").strip()
        )
    )
    if not requested_ids:
        return None, {"reason": "path_repair_blocking_step_ids_missing"}
    missing_ids = sorted(set(requested_ids) - set(step_ids))
    if missing_ids:
        return None, {
            "reason": "path_repair_blocking_step_not_found",
            "blocking_step_ids": missing_ids,
        }

    parent_indices = tuple(state.parent_step_indices)

    def descends_from(index: int, ancestor: int) -> bool:
        cursor: int | None = index
        while cursor is not None:
            if cursor == ancestor:
                return True
            cursor = parent_indices[cursor]
        return False

    blocker_indices = {step_ids.index(step_id) for step_id in requested_ids}
    adjacency = {index: set() for index in blocker_indices}
    for left in blocker_indices:
        for right in blocker_indices:
            if left >= right:
                continue
            if descends_from(left, right) or descends_from(right, left):
                adjacency[left].add(right)
                adjacency[right].add(left)
    requested_set = set(requested_ids)
    for raw in blocking_steps:
        if not isinstance(raw, Mapping):
            continue
        source_id = str(raw.get("step_id") or "").strip()
        if source_id not in requested_set:
            continue
        assessment = dict(raw.get("critic_assessment") or {})
        coupled_ids = {
            str(value).strip()
            for value in assessment.get("coupled_step_ids") or raw.get("coupled_step_ids") or ()
            if str(value).strip() in requested_set
        }
        source_index = step_ids.index(source_id)
        for coupled_id in coupled_ids:
            coupled_index = step_ids.index(coupled_id)
            if coupled_index == source_index:
                continue
            adjacency[source_index].add(coupled_index)
            adjacency[coupled_index].add(source_index)

    unassigned = set(blocker_indices)
    components: list[tuple[int, ...]] = []
    while unassigned:
        component: set[int] = set()
        frontier = [min(unassigned)]
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            frontier.extend(adjacency[current] - component)
        unassigned -= component
        components.append(tuple(sorted(component)))
    components.sort(key=lambda value: value[0])
    component_step_ids = tuple(
        tuple(step_ids[index] for index in component) for component in components
    )
    selected_step_ids = component_step_ids[0]
    selected_set = set(selected_step_ids)
    deferred_step_ids = tuple(step_id for step_id in requested_ids if step_id not in selected_set)
    return (
        _PathRepairBlockerScope(
            selected_step_ids=selected_step_ids,
            deferred_step_ids=deferred_step_ids,
            component_step_ids=component_step_ids,
        ),
        {},
    )


def _path_repair_component_recritic_result(
    pending: Mapping[str, Any],
    blocking_steps: Iterable[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Accept a rebuilt component only when all remaining blockers were deferred."""

    current_ids = {
        str(row.get("step_id") or "").strip() for row in blocking_steps if isinstance(row, Mapping)
    }
    deferred_ids = {
        str(value).strip()
        for value in pending.get("deferred_blocker_step_ids") or ()
        if str(value).strip()
    }
    unexpected_ids = sorted(current_ids - deferred_ids)
    diagnostic = {
        "selected_blocker_step_ids": [
            str(value) for value in pending.get("selected_blocker_step_ids") or () if str(value)
        ],
        "deferred_blocker_step_ids": sorted(deferred_ids),
        "current_blocker_step_ids": sorted(current_ids),
        "unexpected_blocker_step_ids": unexpected_ids,
    }
    return not unexpected_ids, diagnostic

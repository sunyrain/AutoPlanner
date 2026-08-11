"""AutoPlanner-owned bounded MCTS loop for ChemEnzyRetroPlanner."""
from __future__ import annotations

from copy import deepcopy
import logging
import math
import os
import time
from typing import Any, Callable


def bounded_mol_planner(
    target_mol: str,
    target_mol_id: int,
    starting_mols: set[str],
    expand_fn: Callable[..., Any],
    iterations: int,
    max_depth: int = 10,
    viz: bool = False,
    exclude_target: bool = True,
    viz_dir: str | None = None,
    value_fn: Callable[[str], float] = lambda _value: 0.0,
    keep_search: bool = False,
    cascade_cost_model: Any = None,
    cascade_search_context: Any = None,
    max_success_routes: int | None = None,
) -> tuple[bool, tuple[Any, ...]]:
    """Run ChemEnzy MCTS with an iteration cap and a success-reserve stop."""

    from retro_planner.search_frame.mcts_star.mol_tree import MolTree

    started = time.time()
    exclude_flag = False
    starting_mols = deepcopy(starting_mols)
    if exclude_target and target_mol in starting_mols:
        exclude_flag = True
        starting_mols.discard(target_mol)
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn,
        max_depth=max_depth,
        cascade_cost_model=cascade_cost_model,
        cascade_search_context=cascade_search_context,
    )

    iteration_index = -1
    first_success_time = float("inf")
    stop_reason = "iteration_limit"
    observed_success_route_count = 0
    if mol_tree.succ:
        stop_reason = "target_already_in_stock"
    else:
        for iteration_index in range(int(iterations)):
            open_nodes = [node for node in mol_tree.mol_nodes if node.open]
            if not open_nodes:
                logging.info("No open nodes!")
                stop_reason = "no_open_nodes"
                break
            m_next = min(open_nodes, key=_node_selection_key)
            mol_tree.search_status = m_next.v_target()
            try:
                result = mol_tree.call_expand_fn(expand_fn, m_next)
            except Exception:
                result = None

            if result is None or not list(result.get("scores") or []):
                mol_tree.expand(m_next, None, None, None)
                logging.info("Expansion fails on %s!", m_next.mol)
                continue

            reactants, costs, templates, annotations = _prepare_expansion_with_trace(
                mol_tree,
                m_next.mol,
                result,
                use_result_costs=True,
                parent_depth=m_next.depth,
                cascade_context=mol_tree.cascade_context_for_mol(m_next),
            )
            success, _ = mol_tree.expand(
                m_next,
                reactants,
                costs,
                templates,
                max_depth=max_depth,
                cascade_annotations=annotations,
            )
            if success:
                if first_success_time == float("inf"):
                    first_success_time = time.time() - started
                if not keep_search:
                    stop_reason = "first_success"
                    break

            current_depth = m_next.depth
            rollout_depth = current_depth
            rollout_success = False
            while (
                not m_next.is_terminal()
                and not rollout_success
                and rollout_depth - current_depth < 6
            ):
                grandchild = _select_promising_grandchild(m_next)
                if grandchild is None:
                    m_next.go_back = True
                    break
                if grandchild.go_back:
                    m_next = grandchild
                    continue
                success, grandchild_success = _rollout_with_trace(
                    mol_tree,
                    grandchild,
                    expand_fn=expand_fn,
                    max_depth=max_depth,
                )
                rollout_depth = grandchild.depth
                m_next = grandchild
                rollout_success = grandchild_success if keep_search else success

            if mol_tree.root.succ_value <= mol_tree.search_status and not keep_search:
                stop_reason = "optimal_route_found"
                break
            if mol_tree.succ:
                if first_success_time == float("inf"):
                    first_success_time = time.time() - started
                if keep_search and max_success_routes:
                    observed_success_route_count = count_success_routes_capped(
                        mol_tree.root,
                        int(max_success_routes),
                    )
                    if observed_success_route_count >= int(max_success_routes):
                        stop_reason = "success_route_limit_reached"
                        break

    best_route = None
    all_routes = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        all_routes = mol_tree.extract_all_succ_routes()
        observed_success_route_count = len(all_routes or [])

    if viz:
        if viz_dir and not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        if mol_tree.succ and best_route is not None:
            suffix = "_optimal" if best_route.optimal else ""
            best_route.viz_route(f"{viz_dir}/mol_{target_mol_id}_route{suffix}")
        mol_tree.viz_search_tree(f"{viz_dir}/mol_{target_mol_id}_search_tree")
    if exclude_flag:
        starting_mols.add(target_mol)

    executed_iterations = iteration_index + 1
    search_stop = {
        "reason": stop_reason,
        "configured_iteration_limit": int(iterations),
        "executed_iterations": int(executed_iterations),
        "configured_success_route_limit": (
            int(max_success_routes) if max_success_routes else None
        ),
        "observed_success_route_count": int(observed_success_route_count),
        "stopped_early": int(executed_iterations) < int(iterations),
    }
    return mol_tree.succ, (
        best_route,
        executed_iterations,
        all_routes,
        first_success_time,
        mol_tree.cascade_expansion_trace,
        search_stop,
    )


def _select_promising_grandchild(mol_node: Any) -> Any | None:
    """Select a rollout node without process-address-dependent set ordering.

    The vendor implementation deduplicates successful siblings with
    ``list(set(nodes))``. ``MolNode`` uses the default object hash, so equal
    searches in independent processes can choose different tied siblings.
    Preserve first-seen topology for deduplication and use the stable tree node
    ID as the explicit tie-breaker.
    """

    children = list(getattr(mol_node, "children", []) or [])
    if not children and bool(getattr(mol_node, "open", False)):
        return None
    if int(getattr(mol_node, "depth", 0)) == int(
        getattr(mol_node, "max_depth", 0)
    ) - 1:
        mol_node.go_back = True
        return mol_node

    grandchildren: list[Any] = []
    successful_siblings: list[Any] = []
    seen_siblings: set[tuple[int, int, str]] = set()
    for reaction in children:
        if reaction is None:
            continue
        for grandchild in list(getattr(reaction, "children", []) or []):
            grandchildren.append(grandchild)
            if not (
                bool(getattr(grandchild, "succ", False))
                and getattr(grandchild, "sibling", None)
            ):
                continue
            for sibling in list(grandchild.sibling):
                identity = _node_identity_key(sibling)
                if identity in seen_siblings:
                    continue
                seen_siblings.add(identity)
                successful_siblings.append(sibling)

    candidates = successful_siblings or grandchildren
    if not candidates:
        return None
    return min(candidates, key=_node_selection_key)


def _node_identity_key(node: Any) -> tuple[int, int, str]:
    return (
        int(getattr(node, "id", -1)),
        int(getattr(node, "depth", -1)),
        str(getattr(node, "mol", "")),
    )


def _node_selection_key(node: Any) -> tuple[float, int, int, str]:
    terminal = getattr(node, "is_terminal", None)
    if callable(terminal) and terminal():
        value = math.inf
    else:
        try:
            value = float(node.v_target())
        except (AttributeError, TypeError, ValueError):
            value = math.inf
        if math.isnan(value):
            value = math.inf
    node_id, depth, mol = _node_identity_key(node)
    return value, node_id, depth, mol


def _rollout_with_trace(
    mol_tree: Any,
    mol_node: Any,
    *,
    expand_fn: Callable[..., Any],
    max_depth: int,
) -> tuple[bool, bool]:
    """Mirror the vendor rollout while retaining raw proposal candidates."""

    try:
        result = mol_tree.call_expand_fn(expand_fn, mol_node)
    except Exception:
        result = None
    if result is None or not list(result.get("scores") or []):
        mol_tree.expand(mol_node, None, None, None)
        logging.info("Rollout expansion fails on %s!", mol_node.mol)
        return False, False

    reactants, costs, templates, annotations = _prepare_expansion_with_trace(
        mol_tree,
        mol_node.mol,
        result,
        use_result_costs=mol_tree.cascade_cost_model is not None,
        parent_depth=mol_node.depth,
        cascade_context=mol_tree.cascade_context_for_mol(mol_node),
    )
    return mol_tree.expand(
        mol_node,
        reactants,
        costs,
        templates,
        max_depth=max_depth,
        cascade_annotations=annotations,
    )


def _prepare_expansion_with_trace(
    mol_tree: Any,
    parent_mol: str,
    result: dict[str, Any],
    **kwargs: Any,
) -> tuple[Any, Any, Any, Any]:
    """Capture proposal candidates even when no cascade cost model is active."""

    trace = mol_tree.cascade_expansion_trace
    before = len(trace)
    prepared = mol_tree.prepare_expansion(parent_mol, result, **kwargs)
    reactants, costs, templates, annotations = prepared
    if len(trace) != before:
        return prepared

    scores = list(result.get("scores") or [])
    raw_costs = list(result.get("costs") or [])
    for index, reactant_list in enumerate(reactants):
        annotation = annotations[index] if index < len(annotations) else None
        annotation = annotation if isinstance(annotation, dict) else {}
        template = templates[index] if index < len(templates) else None
        trace.append(
            {
                "event": "cascade_expansion_candidate",
                "parent_mol": parent_mol,
                "parent_depth": kwargs.get("parent_depth"),
                "candidate_index": index,
                "reactants": list(reactant_list or []),
                "template": _json_safe(template),
                "source_model": _source_model(annotation, template),
                "reaction_domain": annotation.get("reaction_domain"),
                "base_score": _safe_number(scores[index] if index < len(scores) else None),
                "base_cost": _safe_number(raw_costs[index] if index < len(raw_costs) else None),
                "cascade_adjustment": _safe_number(annotation.get("cascade_adjustment")),
                "total_cost": _safe_number(costs[index] if index < len(costs) else None),
                "components": annotation.get("components") or {},
                "context_features": annotation.get("context_features") or {},
                "source_policy_decision": annotation.get("source_policy_decision"),
                "action_value_score": _safe_number(annotation.get("action_value_score")),
                "active_failure_modes": annotation.get("active_failure_modes") or [],
            }
        )
    return prepared


def _source_model(annotation: dict[str, Any], template: Any) -> str:
    source = annotation.get("source_model")
    if source:
        return str(source)
    if isinstance(template, dict):
        return str(
            template.get("model_full_name")
            or template.get("source_model")
            or template.get("source")
            or "ChemEnzyRetroPlanner"
        )
    return "ChemEnzyRetroPlanner"


def _safe_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def count_success_routes_capped(node: Any, limit: int) -> int:
    """Count successful AND/OR route combinations without enumerating them."""

    cap = max(1, int(limit))
    if not getattr(node, "succ", False):
        return 0
    children = [
        child
        for child in getattr(node, "children", [])
        if getattr(child, "succ", False)
    ]
    if hasattr(node, "mol"):
        if not children:
            return 1
        total = 0
        for child in children:
            total += count_success_routes_capped(child, cap)
            if total >= cap:
                return cap
        return total
    if not children:
        return 0
    total = 1
    for child in children:
        count = count_success_routes_capped(child, cap)
        if count <= 0:
            return 0
        total *= count
        if total >= cap:
            return cap
    return total


__all__ = ["bounded_mol_planner", "count_success_routes_capped"]

"""Native ChemEnzy route stock and bounded-selection contracts."""
from __future__ import annotations

from typing import Any


def chem_route_stock_closed(route: dict[str, Any]) -> bool:
    products = {
        step.get("product_smiles")
        for step in route.get("steps") or []
        if step.get("product_smiles")
    }
    terminal_flags = []
    for step in route.get("steps") or []:
        status = step.get("stock_status") or {}
        for smiles in step.get("reactant_smiles") or []:
            if smiles and smiles not in products:
                terminal_flags.append(bool(status.get(smiles)))
    return bool(terminal_flags) and all(terminal_flags)


def select_chem_routes(
    routes: list[dict[str, Any]],
    *,
    topk: int | None,
    selection: str,
) -> list[dict[str, Any]]:
    annotated = []
    for rank, route in enumerate(routes, 1):
        item = dict(route)
        item["_native_rank"] = rank
        annotated.append(item)
    if selection == "rank":
        selected = annotated
    elif selection == "stock_first":
        selected = sorted(
            annotated,
            key=lambda route: (
                not chem_route_stock_closed(route),
                route["_native_rank"],
            ),
        )
    elif selection == "rank_plus_stock":
        if topk is None:
            selected = annotated
        else:
            limit = max(0, int(topk))
            selected = list(annotated[:limit])
            if limit > 0 and not any(
                chem_route_stock_closed(route) for route in selected
            ):
                stock_route = next(
                    (
                        route
                        for route in annotated
                        if chem_route_stock_closed(route)
                    ),
                    None,
                )
                if stock_route is not None:
                    selected = [*selected[: limit - 1], stock_route]
    else:
        raise ValueError(f"unsupported native selection mode: {selection}")
    if topk is None:
        return selected
    return selected[: max(0, int(topk))]


__all__ = ["chem_route_stock_closed", "select_chem_routes"]

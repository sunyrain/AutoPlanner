"""Conservative common-commodity stock helpers."""
from __future__ import annotations

from collections.abc import Callable

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


COMMON_COMMODITY_SMILES = (
    "N",
    "O",
    "O=O",
    "N#N",
    "O=C=O",
    "[H][H]",
    "ClCl",
    "BrBr",
)

COMMON_COMMODITY_CANONICAL = frozenset(
    can for smi in COMMON_COMMODITY_SMILES if (can := canonical_smiles(smi))
)


def is_common_commodity_stock(smiles: str | None) -> bool:
    """Return true for very common non-catalog commodity molecules."""
    if not smiles:
        return False
    key = canonical_smiles(str(smiles))
    return bool(key and key in COMMON_COMMODITY_CANONICAL)


def wrap_with_common_commodity_stock(
    stock_checker: Callable[[str], bool] | None,
) -> Callable[[str], bool]:
    """Add common-commodity hits on top of an existing stock checker."""

    def checker(smiles: str) -> bool:
        if is_common_commodity_stock(smiles):
            return True
        if stock_checker is None:
            return False
        return bool(stock_checker(smiles))

    return checker

"""Source-supported chemical anchor stock helpers."""
from __future__ import annotations

from collections.abc import Callable

from cascade_planner.baselines.chemical_anchor_rescue import known_chemical_anchor_precursor_record


def is_chemical_anchor_stock(smiles: str | None) -> bool:
    """Return true for curated, source-supported chemical anchor precursors."""
    if not smiles:
        return False
    return bool(known_chemical_anchor_precursor_record(str(smiles)))


def wrap_with_chemical_anchor_stock(
    stock_checker: Callable[[str], bool] | None,
) -> Callable[[str], bool]:
    """Add curated chemical anchor precursor hits on top of an existing stock checker."""

    def checker(smiles: str) -> bool:
        if is_chemical_anchor_stock(smiles):
            return True
        if stock_checker is None:
            return False
        return bool(stock_checker(smiles))

    return checker

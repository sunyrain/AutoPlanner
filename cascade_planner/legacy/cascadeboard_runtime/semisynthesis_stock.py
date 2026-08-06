"""Source-supported advanced precursor stock helpers."""
from __future__ import annotations

from collections.abc import Callable

from cascade_planner.baselines.semisynthesis_rescue import known_advanced_precursor_record


def is_semisynthesis_stock(smiles: str | None) -> bool:
    """Return true for curated, source-supported semisynthesis precursors."""
    if not smiles:
        return False
    return bool(known_advanced_precursor_record(str(smiles)))


def wrap_with_semisynthesis_stock(
    stock_checker: Callable[[str], bool] | None,
) -> Callable[[str], bool]:
    """Add curated semisynthesis precursor hits on top of an existing stock checker."""

    def checker(smiles: str) -> bool:
        if is_semisynthesis_stock(smiles):
            return True
        if stock_checker is None:
            return False
        return bool(stock_checker(smiles))

    return checker

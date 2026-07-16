"""Small deterministic Pareto primitives shared by route optimizers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


MAXIMIZE = "maximize"
MINIMIZE = "minimize"


def dominates(
    left: Sequence[float],
    right: Sequence[float],
    *,
    directions: Sequence[str] | None = None,
) -> bool:
    """Return whether ``left`` is no worse on every axis and better on one."""

    if len(left) != len(right):
        raise ValueError("pareto_vector_length_mismatch")
    modes = tuple(directions or (MAXIMIZE,) * len(left))
    if len(modes) != len(left) or any(mode not in {MAXIMIZE, MINIMIZE} for mode in modes):
        raise ValueError("pareto_directions_invalid")
    pairs = tuple(zip(_numbers(left), _numbers(right), modes, strict=True))
    no_worse = all(a >= b if mode == MAXIMIZE else a <= b for a, b, mode in pairs)
    better = any(a > b if mode == MAXIMIZE else a < b for a, b, mode in pairs)
    return no_worse and better


def pareto_layers(
    vectors: Mapping[str, Sequence[float]],
    *,
    directions: Sequence[str],
) -> list[list[str]]:
    """Return stable non-dominated fronts covering every candidate exactly once."""

    remaining = {str(key): tuple(_numbers(value)) for key, value in vectors.items()}
    if not remaining:
        return []
    widths = {len(value) for value in remaining.values()}
    if len(widths) != 1 or widths != {len(directions)}:
        raise ValueError("pareto_vector_shape_invalid")
    layers: list[list[str]] = []
    while remaining:
        front = sorted(
            candidate_id
            for candidate_id, vector in remaining.items()
            if not any(
                dominates(other, vector, directions=directions)
                for other_id, other in remaining.items()
                if other_id != candidate_id
            )
        )
        if not front:
            raise ValueError("pareto_front_empty")
        layers.append(front)
        for candidate_id in front:
            remaining.pop(candidate_id)
    return layers


def _numbers(values: Sequence[float]) -> tuple[float, ...]:
    numbers = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("pareto_vector_not_finite")
    return numbers


__all__ = ["MAXIMIZE", "MINIMIZE", "dominates", "pareto_layers"]

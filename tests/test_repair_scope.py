from itertools import combinations

import pytest

from cascade_planner.orchestration.repair_scope import minimum_connected_repair_indices


def test_minimum_repair_agrees_with_exhaustive_connected_subsets():
    parents = (None, 0, 0, 1, 2, 1)
    ids = tuple(f"step-{i}" for i in range(len(parents)))
    adjacency = {i: set() for i in range(len(parents))}
    for i, parent in enumerate(parents):
        if parent is not None:
            adjacency[i].add(parent)
            adjacency[parent].add(i)
    # Independent brute-force oracle over every nonempty request on a small
    # branched occurrence tree, including interleaved sibling steps.
    subsets = [set(c) for n in range(1, 7) for c in combinations(range(6), n)]
    for requested in subsets:
        admissible = []
        for candidate in subsets:
            if not requested <= candidate:
                continue
            reached, frontier = set(), {min(candidate)}
            while frontier:
                reached |= frontier
                frontier = set.union(*(adjacency[i] for i in frontier)) & candidate - reached
            if reached == candidate:
                admissible.append(candidate)
        expected = min(admissible, key=len)
        actual = minimum_connected_repair_indices(ids, parents, [ids[i] for i in requested])
        assert actual == expected


@pytest.mark.parametrize("ids,parents,changes,reason", [
    (("a", "b"), (None, None), ("a", "b"), "disconnected"),
    (("a", "b"), (None, 0), ("missing",), "not_found"),
    (("a", "a"), (None, 0), ("a",), "tree_invalid"),
    (("a", "b"), (1, 0), ("b",), "tree_invalid"),
    (("a",), (None,), (), "missing"),
])
def test_invalid_graph_interventions_are_explicit(ids, parents, changes, reason):
    with pytest.raises(ValueError, match=reason):
        minimum_connected_repair_indices(ids, parents, changes)

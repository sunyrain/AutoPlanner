from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from cascade_planner.baselines.chem_enzy_bounded_mcts import (
    _select_promising_grandchild,
)


@dataclass(eq=False)
class _Node:
    id: int
    mol: str
    score: float
    hash_value: int
    depth: int = 1
    max_depth: int = 10
    succ: bool = False
    open: bool = False
    go_back: bool = False
    terminal: bool = False
    children: list[object] = field(default_factory=list)
    sibling: list["_Node"] = field(default_factory=list)

    def __hash__(self) -> int:
        return self.hash_value

    def is_terminal(self) -> bool:
        return self.terminal

    def v_target(self) -> float:
        return self.score


def _parent_with_successful_siblings(*siblings: _Node) -> _Node:
    known = _Node(99, "known", 0.0, 99, succ=True, terminal=True)
    known.sibling = list(siblings)
    parent = _Node(1, "parent", 0.0, 1, depth=0)
    parent.children = [SimpleNamespace(children=[known])]
    return parent


def test_successful_sibling_tie_break_ignores_process_object_hash() -> None:
    lower_id = _Node(2, "CC", 1.0, 1000)
    higher_id = _Node(7, "CN", 1.0, -1000)
    first = _parent_with_successful_siblings(higher_id, lower_id)

    lower_id_replay = _Node(2, "CC", 1.0, -2000)
    higher_id_replay = _Node(7, "CN", 1.0, 2000)
    replay = _parent_with_successful_siblings(lower_id_replay, higher_id_replay)

    assert _select_promising_grandchild(first).id == 2
    assert _select_promising_grandchild(replay).id == 2


def test_rollout_selection_prefers_value_then_stable_node_id() -> None:
    later = _Node(8, "CO", 0.25, 1)
    earlier = _Node(3, "CCO", 0.25, 2)
    worse = _Node(2, "CCC", 0.5, 3)
    parent = _Node(1, "parent", 0.0, 1, depth=0)
    parent.children = [SimpleNamespace(children=[later, worse, earlier])]

    assert _select_promising_grandchild(parent) is earlier


def test_rollout_selection_marks_depth_boundary_for_backtracking() -> None:
    parent = _Node(1, "parent", 0.0, 1, depth=9, max_depth=10, open=True)
    parent.children = [SimpleNamespace(children=[])]

    assert _select_promising_grandchild(parent) is parent
    assert parent.go_back is True

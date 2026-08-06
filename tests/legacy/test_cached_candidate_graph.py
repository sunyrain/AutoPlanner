from __future__ import annotations

from cascade_planner.legacy.eval_runtime.cascadeboard_cached_candidate_graph import (
    StrictCachedGraph,
)


def test_strict_cached_graph_is_cache_only_and_records_misses() -> None:
    graph = StrictCachedGraph(
        {"CCO": [{"main_reactant": "CC", "score": 0.8, "source": "cache"}]},
        stock_checker=lambda _: False,
        max_depth=1,
    )
    graph.build("CCO")
    assert graph.root is not None
    assert graph.root.children
    assert graph.cache_misses == set()

    missing = StrictCachedGraph({}, stock_checker=lambda _: False, max_depth=1)
    missing.build("CCO")
    assert missing.root is not None
    assert missing.root.children == []
    assert missing.cache_misses == {"CCO"}

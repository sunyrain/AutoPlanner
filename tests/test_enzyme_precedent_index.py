from pathlib import Path

from cascade_planner.cascadeboard.enzyme_precedent_retrieval import (
    _limited_index_path,
)


def test_limited_precedent_index_cannot_replace_full_index() -> None:
    full = Path("data/bridge_pack_v0/enzyme_precedent_index_v2.joblib")

    limited = _limited_index_path(full, 12_000)

    assert limited != full
    assert limited.name == "enzyme_precedent_index_v2.limit-12000.joblib"

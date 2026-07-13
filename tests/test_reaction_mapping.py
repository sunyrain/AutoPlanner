from __future__ import annotations

import pytest

from cascade_planner.application import reaction_mapping
from cascade_planner.application.reaction_mapping import (
    ReactionMappingConfig,
    map_reactions_locally,
)


def test_local_mapping_batches_deduplicates_and_preserves_failures() -> None:
    calls: list[list[str]] = []

    def mapper(values: list[str]) -> list[dict[str, str]]:
        calls.append(values)
        return [
            {"mapped_rxn": "[CH3:1][OH:2]>>[CH2:1]=[O:2]"}
            if value == "CO>>C=O"
            else {"mapped_rxn": ""}
            for value in values
        ]

    report = map_reactions_locally(
        ["CO>>C=O", "CO>>C=O", "CC>>C=C"],
        mapper=mapper,
        config=ReactionMappingConfig(batch_size=1, max_reactions=4),
    )

    assert calls == [["CO>>C=O"], ["CC>>C=C"]]
    assert report["requested_count"] == 2
    assert report["mapped_count"] == 1
    assert report["failure_count"] == 1
    assert report["semantics"]["hosted_model_calls"] == 0


def test_local_mapping_hard_cap_is_reported() -> None:
    report = map_reactions_locally(
        ["CO>>C=O", "CCO>>CC=O"],
        mapper=lambda values: ["[CH3:1][OH:2]>>[CH2:1]=[O:2]"] * len(values),
        config=ReactionMappingConfig(max_reactions=1),
    )
    assert report["truncated"] is True
    assert report["mapped_count"] == 1


def test_empty_mapping_batch_does_not_initialize_rxnmapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_mapper() -> object:
        raise AssertionError("empty batch must not initialize RXNMapper")

    monkeypatch.setattr(reaction_mapping, "_rxnmapper", unexpected_mapper)
    report = map_reactions_locally([])

    assert report["requested_count"] == 0
    assert report["mapped_count"] == 0
    assert report["elapsed_s"] == 0.0

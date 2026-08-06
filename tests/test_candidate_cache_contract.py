from __future__ import annotations

from cascade_planner.cascadeboard import live_retro
from cascade_planner.cascadeboard.candidate_cache import (
    cache_summary,
    canon_set,
    merge_candidate_caches,
)


def test_candidate_cache_contract_normalizes_deduplicates_and_summarizes() -> None:
    first = {
        "CCO": [
            {
                "main_reactant": "CC",
                "aux_reactants": ["O"],
                "source": "enzexpand",
                "score": 0.2,
            }
        ]
    }
    second = {
        "OCC": [
            {
                "main_reactant": "CC",
                "aux_reactants": ["O"],
                "source": "enzexpand",
                "score": 0.1,
            },
            {
                "main_reactant": "CC",
                "aux_reactants": ["O"],
                "source": "retrochimera",
                "score": 0.9,
            },
        ]
    }

    merged = merge_candidate_caches(first, second)

    assert canon_set("O.CC.O") == frozenset({"O", "CC"})
    assert list(merged) == ["CCO"]
    assert [row["source"] for row in merged["CCO"]] == [
        "retrochimera",
        "enzexpand",
    ]
    assert cache_summary(merged) == {
        "n_products": 1,
        "n_products_nonempty": 1,
        "n_candidates": 2,
        "source_counts": {"enzexpand": 1, "retrochimera": 1},
    }


def test_live_retro_engine_does_not_publish_deleted_expand_sources(monkeypatch) -> None:
    for name in (
        "_retrorules_enabled",
        "_chemical_templates_enabled",
        "_semisynthesis_rescue_enabled",
        "_chemical_anchor_rescue_enabled",
        "_retrochimera_enabled",
        "_chem_enzy_onestep_enabled",
        "_chem_enzy_graphfp_fusion_enabled",
        "_template_relevance_enabled",
        "_chem_enzy_bionav_enabled",
    ):
        monkeypatch.setattr(live_retro, name, lambda: False)

    engine = live_retro.build_live_retro_engine()

    assert "enzexpand" not in engine
    assert "enzyformer" not in engine

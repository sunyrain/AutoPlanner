from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.eval.cache_uspto190_targets import cache_uspto190_targets
from cascade_planner.eval.syntharena_uspto190 import (
    pagination_pages,
    parse_target_page,
    target_paths,
)
from cascade_planner.interfaces.live_stock import FrozenBenchmarkStockIndex
from scripts.prepare_retrostar190_benchmark import build_stock_index


def test_retrostar_stock_builder_canonicalizes_and_freezes_membership(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inventory.csv"
    source.write_text(
        ",mol\n"
        "0,OCC\n"
        "1,CN\n"
        "2,not-smiles\n",
        encoding="utf-8",
    )
    index = tmp_path / "stock.sqlite3"

    result = build_stock_index(
        source,
        index,
        catalog_name="fixture-retrostar-stock",
        workers=1,
        batch_size=2,
    )
    catalog = FrozenBenchmarkStockIndex(
        index,
        expected_sha256=result["index_sha256"],
    )(["CCO", "CCN"])

    assert result["member_count"] == 2
    assert [row["canonical_smiles"] for row in catalog["members"]] == ["CCO"]
    assert [row["canonical_smiles"] for row in catalog["misses"]] == ["CCN"]


def test_syntharena_target_cache_discovery_is_stable_and_resumable(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "syntharena"
    cache_dir.mkdir()
    (cache_dir / "uspto190_index.html").write_text(
        '<a href="targets/a">A</a><a href="targets/a">duplicate</a>'
        '<a href="targets/b">B</a><a href="?page=3">last</a>',
        encoding="utf-8",
    )
    (cache_dir / "uspto190_a.html").write_text("cached", encoding="utf-8")
    (cache_dir / "uspto190_b.html").write_text("cached", encoding="utf-8")

    report = cache_uspto190_targets(
        cache_dir=cache_dir,
        limit=2,
        fetch=False,
    )

    assert target_paths("targets/a targets/a targets/b") == ["targets/a", "targets/b"]
    assert pagination_pages("?page=3") == [2, 3]
    assert report["target_paths_discovered"] == 2
    assert report["selected_targets"] == 2
    assert report["ready_for_selected_window"] is True


def test_syntharena_target_page_parser_preserves_reference_route(
    tmp_path: Path,
) -> None:
    path = tmp_path / "uspto190_target-1.html"
    path.write_text(
        json.dumps(
            {
                "route": {},
                "target": {
                    "targetId": "target-1",
                    "routeLength": 2,
                    "molecule": {"smiles": "CCO"},
                },
                "rootNode": {
                    "reactionStep": {"id": "r1"},
                    "molecule": {"smiles": "CCO"},
                    "children": [
                        {
                            "reactionStep": None,
                            "molecule": {"smiles": "CC"},
                            "children": [],
                        },
                        {
                            "reactionStep": None,
                            "molecule": {"smiles": "O"},
                            "children": [],
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    row = parse_target_page(path)

    assert row is not None
    assert row["cascade_id"] == "target-1"
    assert row["target_smiles"] == "CCO"
    assert row["depth"] == 2
    assert row["gt_route"] == [
        {
            "rxn_smiles": "CC.O>>CCO",
            "transformation": "other",
            "step_role": "external_acceptable_route",
        }
    ]

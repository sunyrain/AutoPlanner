from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.application.blind_benchmark_contract import load_blind_manifest
from cascade_planner.eval.build_classic_multistep_benchmark import (
    DepthStratum,
    build_classic_multistep_benchmark,
    reference_route_metrics,
)


def _route(target: str, depth: int, *, branch: bool = False) -> dict:
    def chain(level: int, label: str) -> dict:
        molecule = {
            "type": "mol",
            "smiles": "C" if level else target,
            "in_stock": level == depth,
            "children": [],
        }
        if level >= depth:
            return molecule
        children = [chain(level + 1, label)]
        if branch and level == 0:
            children.append(
                {
                    "type": "mol",
                    "smiles": "O",
                    "in_stock": False,
                    "children": [
                        {
                            "type": "reaction",
                            "metadata": {"smiles": "[OH2:91]>>[OH2:92]"},
                            "children": [
                                {
                                    "type": "mol",
                                    "smiles": "O",
                                    "in_stock": True,
                                    "children": [],
                                }
                            ],
                        }
                    ],
                }
            )
        molecule["children"] = [
            {
                "type": "reaction",
                "metadata": {"smiles": f"[CH3:{level + 1}]>>[CH4:{level + 1}]"},
                "children": children,
            }
        ]
        return molecule

    return chain(0, "main")


def _write_source(root: Path, split: str) -> None:
    targets = []
    references = []
    for depth, prefix in ((3, "C"), (4, "N")):
        for offset in range(3):
            target = prefix + "C" * (offset + 1)
            targets.append(target)
            references.append(_route(target, depth, branch=offset == 0))
    (root / f"targets_{split}.txt").write_text(
        "\n".join(targets) + "\n", encoding="utf-8"
    )
    (root / f"ref_routes_{split}.json").write_text(
        json.dumps(references), encoding="utf-8"
    )


def test_reference_route_metrics_tracks_llr_stock_and_convergence() -> None:
    metrics = reference_route_metrics(_route("CCC", 4, branch=True))

    assert metrics == {
        "reaction_count": 5,
        "longest_linear_depth": 4,
        "leaf_count": 2,
        "stock_leaf_count": 2,
        "reference_stock_closed": True,
        "convergent_reaction_count": 1,
        "convergent": True,
    }


def test_builder_keeps_reference_answers_out_of_blind_manifest(tmp_path: Path) -> None:
    source = tmp_path / "paroutes"
    source.mkdir()
    _write_source(source, "n1")
    _write_source(source, "n5")
    manifest = tmp_path / "manifest.json"
    references = tmp_path / "references.json"
    protocol = tmp_path / "protocol.json"
    search_benchmarks = tmp_path / "search"

    result = build_classic_multistep_benchmark(
        paroutes_root=source,
        manifest_output=manifest,
        reference_output=references,
        protocol_output=protocol,
        search_benchmark_output_dir=search_benchmarks,
        seed="fixed-test-seed",
        strata=(
            DepthStratum("llr_3", 3, 3, 1),
            DepthStratum("llr_4", 4, 4, 1),
        ),
    )

    cases = load_blind_manifest(manifest)
    manifest_text = manifest.read_text(encoding="utf-8")
    reference_pack = json.loads(references.read_text(encoding="utf-8"))
    assert result["target_count"] == 4
    assert len(cases) == 4
    assert "gt_route" not in manifest_text
    assert "source_index" not in manifest_text
    assert all(row["gt_route"] for row in reference_pack["cases"])
    n1_search = json.loads(
        (search_benchmarks / "paroutes_n1_multistep.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(n1_search["targets"]) == 2
    assert all(row["split"] == "n1" for row in n1_search["targets"])
    assert json.loads(protocol.read_text(encoding="utf-8"))["blindness"][
        "planner_receives_reference_routes"
    ] is False


def test_builder_selection_is_reproducible_and_cross_split_unique(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paroutes"
    source.mkdir()
    _write_source(source, "n1")
    _write_source(source, "n5")
    outputs = []
    for suffix in ("a", "b"):
        manifest = tmp_path / f"manifest-{suffix}.json"
        build_classic_multistep_benchmark(
            paroutes_root=source,
            manifest_output=manifest,
            reference_output=tmp_path / f"references-{suffix}.json",
            seed="fixed-test-seed",
            strata=(DepthStratum("llr_3", 3, 3, 1),),
        )
        outputs.append(json.loads(manifest.read_text(encoding="utf-8")))

    assert outputs[0] == outputs[1]
    smiles = [row["target_smiles"] for row in outputs[0]["cases"]]
    assert len(smiles) == len(set(smiles))

import json

from scripts.analyze_bufotalin_proposal_frontiers import analyze_proposal_frontiers


def test_analyze_proposal_frontiers_groups_rejected_frontiers(tmp_path):
    cycle = tmp_path / "cycle_001"
    cycle.mkdir()
    core = "C1CC[C@H]2[C@@H](C1)CC[C@H]1[C@@H]2CC[C@]2(C)[C@@H](O)C[C@H](O)C[C@]12O"
    prenyl = "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CO"
    (cycle / "web_payload.json").write_text(
        json.dumps(
            {
                "proposal_gate": {
                    "dropped": [
                        {
                            "route_rank": 2,
                            "n_steps": 4,
                            "score": 0.2,
                            "frontier": {
                                "smiles": core,
                                "reason": "unexplained_complex_core_growth",
                                "proposal_reasons": ["unexplained_complex_core_growth"],
                            },
                        },
                        {
                            "route_rank": 3,
                            "n_steps": 4,
                            "score": 0.1,
                            "frontier": {
                                "smiles": core,
                                "reason": "large_unexplained_heavy_atom_gain",
                                "proposal_reasons": ["large_unexplained_heavy_atom_gain"],
                            },
                        },
                        {
                            "route_rank": 4,
                            "n_steps": 4,
                            "score": 0.05,
                            "frontier": {
                                "smiles": prenyl,
                                "reason": "unsupported_biosynthetic_prenyl_terminal",
                                "proposal_reasons": ["unsupported_biosynthetic_prenyl_terminal"],
                            },
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    report = analyze_proposal_frontiers(tmp_path)

    assert report["summary"]["dropped_rows_with_frontier"] == 3
    assert report["summary"]["unique_frontiers"] == 2
    assert report["top_frontiers"][0]["smiles"] == core
    assert report["top_frontiers"][0]["count"] == 2
    assert report["top_complex_core_frontiers"]
    assert report["top_unsupported_prenyl_frontiers"][0]["smiles"] == prenyl

import json

from scripts.probe_bufotalin_frontier_proposals import probe_frontier_proposals


class FakeProvider:
    load_error = ""

    def predict(self, product, top_k=10):
        return [
            {
                "rank": 1,
                "score": 0.8,
                "source": "fake",
                "main_reactant": "CCO",
                "reaction_smiles": f"CCO>>{product}",
                "proposal_gate": {"decision": "keep", "hard_reasons": []},
            },
            {
                "rank": 2,
                "score": 0.2,
                "source": "fake",
                "main_reactant": "C",
                "reaction_smiles": f"C>>{product}",
                "proposal_gate": {"decision": "reject", "hard_reasons": ["large_unexplained_heavy_atom_gain"]},
            },
        ][:top_k]


def test_probe_frontier_proposals_counts_gate_decisions(tmp_path):
    (tmp_path / "proposal_frontier_analysis.json").write_text(
        json.dumps(
            {
                "top_frontiers": [
                    {
                        "smiles": "CC=O",
                        "count": 3,
                        "reason_counts": {"large_unexplained_heavy_atom_gain": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = probe_frontier_proposals(tmp_path, provider=FakeProvider(), top_frontiers=1, top_k=2)

    assert report["summary"]["frontier_count"] == 1
    assert report["summary"]["proposal_count"] == 2
    assert report["summary"]["gate_keep_count"] == 1
    assert report["summary"]["gate_reject_count"] == 1
    assert report["frontiers"][0]["top_proposals"][0]["gate_decision"] == "keep"

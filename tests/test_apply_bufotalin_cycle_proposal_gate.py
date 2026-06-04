import json

from scripts.apply_bufotalin_cycle_proposal_gate import apply_cycle_proposal_gate_to_root


def test_apply_cycle_proposal_gate_to_root_writes_backup_and_summary(tmp_path):
    cycle = tmp_path / "cycle_001"
    cycle.mkdir()
    prenyl_terminal = (
        "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/"
        "CC/C(C)=C/CC/C(C)=C/CO"
    )
    payload = {
        "n_results": 1,
        "routes": [
            {
                "score": 0.1,
                "n_steps": 1,
                "route_rank": 0,
                "metrics": {"terminal_reactants": [prenyl_terminal]},
                "steps": [
                    {
                        "index": 0,
                        "product": "CC",
                        "main_reactant": "C",
                        "reaction_smiles": "C>>CC",
                        "condition_predictions": [{"condition_label": "RCR model prediction"}],
                    }
                ],
            }
        ],
        "route_set_metrics": {},
        "ui_metadata": {},
        "search_status": {},
        "failure_diagnosis": [],
        "failure_analysis": {"failure_categories": []},
    }
    (cycle / "web_payload.json").write_text(json.dumps(payload), encoding="utf-8")
    figures = cycle / "figures"
    figures.mkdir()
    (figures / "manifest.json").write_text(json.dumps({"figures": [{"svg": "scheme_route_01.svg"}]}), encoding="utf-8")
    (figures / "scheme_route_01.svg").write_text("<svg/>", encoding="utf-8")

    report = apply_cycle_proposal_gate_to_root(tmp_path)
    updated = json.loads((cycle / "web_payload.json").read_text(encoding="utf-8"))

    assert report["payload_count"] == 1
    assert report["input_routes"] == 1
    assert report["kept_routes"] == 0
    assert report["dropped_routes"] == 1
    assert report["repaired_routes"] == 0
    assert report["repair_reason_counts"] == {}
    assert "unsupported_biosynthetic_prenyl_terminal" in report["reason_counts"]
    assert updated["routes"] == []
    assert (cycle / "web_payload_pre_proposal_gate.json").exists()
    assert (tmp_path / "cycle_proposal_gate_retrofit_summary.json").exists()
    assert not (figures / "manifest.json").exists()
    assert not (figures / "scheme_route_01.svg").exists()
    assert report["rows"][0]["cleared_figure_files"] == 2
    assert report["rows"][0]["repaired_routes"] == 0

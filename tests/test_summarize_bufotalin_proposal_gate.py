import json

from scripts.summarize_bufotalin_proposal_gate import summarize_existing_proposal_gates


def test_summarize_existing_proposal_gates_aggregates_payload_reports(tmp_path):
    _write_payload(
        tmp_path / "anchor",
        {
            "mode": "hard_reject",
            "input_routes": 1,
            "kept_routes": 1,
            "dropped_routes": 0,
            "repaired_routes": 0,
            "reason_counts": {},
            "repair_reason_counts": {},
            "frontiers": [],
            "dropped": [],
        },
    )
    _write_payload(
        tmp_path / "cycle_001",
        {
            "mode": "hard_reject",
            "input_routes": 5,
            "kept_routes": 3,
            "dropped_routes": 2,
            "repaired_routes": 1,
            "reason_counts": {"large_unexplained_heavy_atom_gain": 2},
            "repair_reason_counts": {"late_stage_tbs_silylation": 1},
            "frontiers": [{"smiles": "CCO"}],
            "dropped": [{"route_rank": 2}, {"route_rank": 4}],
        },
    )

    report = summarize_existing_proposal_gates(tmp_path)

    assert report["schema_version"] == "bufotalin_cycle_proposal_gate_summary.v1"
    assert report["mode"] == "hard_reject"
    assert report["payload_count"] == 2
    assert report["input_routes"] == 6
    assert report["kept_routes"] == 4
    assert report["dropped_routes"] == 2
    assert report["repaired_routes"] == 1
    assert report["reason_counts"]["large_unexplained_heavy_atom_gain"] == 2
    assert report["repair_reason_counts"]["late_stage_tbs_silylation"] == 1
    assert report["rows"][1]["frontier_count"] == 1
    assert report["rows"][1]["dropped_row_count"] == 2


def _write_payload(directory, proposal_gate):
    directory.mkdir()
    (directory / "web_payload.json").write_text(
        json.dumps({"proposal_gate": proposal_gate}),
        encoding="utf-8",
    )

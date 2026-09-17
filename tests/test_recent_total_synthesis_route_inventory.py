from __future__ import annotations

import copy
import json

from scripts.recent_total_synthesis_route_inventory import build_inventory, structured_counts


def test_shared_source_steps_are_not_counted_again_for_different_target_ids():
    base = {"paper_id": "p", "steps": [{"step_id": "target-a-1", "precursor_labels": ["1"], "product_label": "2", "conditions": {"solvent": "water"}}]}
    other = copy.deepcopy(base)
    other["steps"][0]["step_id"] = "target-b-1"
    counts = structured_counts([base, other])
    assert counts["route_operation_instances"] == 2
    assert counts["unique_paper_bound_source_steps"] == 1
    assert counts["structured_route_papers"] == 1
    other["steps"][0]["conditions"]["solvent"] = "ethanol"
    assert structured_counts([base, other])["unique_paper_bound_source_steps"] == 2


def test_inventory_includes_p1_targets_even_without_sources_or_evidence(tmp_path):
    p1 = tmp_path / "curation_candidates/p1_scope"
    p1.mkdir(parents=True)
    target = {"target_slot_id": "p1-target", "paper_id": "p1-paper", "target_name": "Example", "doi": "10.1/example", "slot_class": "primary_candidate"}
    (p1 / "candidate-target-slots.jsonl").write_text(json.dumps(target) + "\n", encoding="utf-8")
    rows, counts = build_inventory(tmp_path)
    assert counts["candidate_targets"] == counts["candidate_papers"] == 1
    assert rows[0]["cohort"] == "P1"
    assert "source_package_missing" in rows[0]["blockers"]
    assert "ordered_route_not_extracted" in rows[0]["blockers"]
    assert counts["complete_ordered_route_candidates"] == 0

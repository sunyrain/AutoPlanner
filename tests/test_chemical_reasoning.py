from cascade_planner.orchestration.chemical_reasoning import bind_chemical_dependencies


def test_causal_links_bind_passing_preparations_without_creating_verdicts():
    bindings = [{"review_slot": f"review-{i}", "step_id": f"canonical:{i}"} for i in range(3)]
    links, diagnostics = bind_chemical_dependencies([{
        "consumer_review_slot": "review-0", "prerequisite_review_slots": ["review-1", "review-2"],
        "requirement": "The supplied configuration must permit the following displacement.",
    }], bindings)
    assert diagnostics == []
    assert links == [{"consumer_step_id": "canonical:0", "prerequisite_step_ids": ["canonical:1", "canonical:2"],
                      "requirement": "The supplied configuration must permit the following displacement."}]


def test_malformed_dependency_does_not_discard_other_valid_links():
    bindings = [{"review_slot": str(i), "step_id": f"s{i}"} for i in range(2)]
    links, diagnostics = bind_chemical_dependencies([
        {"consumer_review_slot": "0", "prerequisite_review_slots": ["unknown"], "requirement": "bad"},
        {"consumer_review_slot": "0", "prerequisite_review_slots": ["0"], "requirement": "self"},
        {"consumer_review_slot": "0", "prerequisite_review_slots": ["1"], "requirement": "protect first"},
    ], bindings)
    assert len(links) == 1 and links[0]["requirement"] == "protect first"
    assert len(diagnostics) == 2
    assert bind_chemical_dependencies(None, bindings) == ([], [])

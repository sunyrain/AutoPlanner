import json
from pathlib import Path

from scripts.export_bufotalin_final_candidates import classify_route, export_final_candidates
from scripts.run_bufotalin_12h_iteration import BUFOTALIN_TARGET


def test_classify_source_supported_semisynthesis_as_high_confidence():
    route = {
        "n_steps": 1,
        "score": 0.9,
        "steps": [
            {
                "main_reactant": "CCO",
                "aux_reactants": ["CC(=O)OC(C)=O"],
                "condition_predictions": [{"condition_label": "Ac2O, DMAP"}],
            }
        ],
        "metrics": {
            "source_supported_semisynthesis": True,
            "native_returned_route": False,
            "terminal_reactants": ["CCO", "CC(=O)OC(C)=O"],
            "cascade_verifier": {"feasible": True},
        },
        "raw_backend_metadata": {
            "advanced_precursor_record": {
                "name": "Deacetylbufotalin",
                "cas": "465-19-0",
            }
        },
    }

    classification = classify_route(route, target_smiles=BUFOTALIN_TARGET)

    assert classification["reportable"]
    assert classification["presentation_ready"]
    assert classification["confidence_tier"] == "high_confidence_source_supported"
    assert classification["exclusion_reasons"] == []


def test_classify_native_route_with_target_terminal_as_excluded():
    route = {
        "n_steps": 2,
        "steps": [
            {"condition_predictions": [{"condition_label": "RCR model prediction"}]},
            {"condition_predictions": [{"condition_label": "RCR model prediction"}]},
        ],
        "metrics": {
            "source_supported_semisynthesis": False,
            "native_returned_route": True,
            "terminal_reactants": [BUFOTALIN_TARGET, "CC(C)(C)[Si](C)(C)Cl"],
            "cascade_verifier": {"feasible": True},
        },
    }

    classification = classify_route(route, target_smiles=BUFOTALIN_TARGET)

    assert not classification["reportable"]
    assert classification["confidence_tier"] == "excluded"
    assert "terminal_reactants_include_target" in classification["exclusion_reasons"]


def test_classify_stitched_semisynthesis_upstream_as_review_only():
    route = {
        "n_steps": 4,
        "score": 0.01,
        "steps": [
            {"reaction_smiles": "A>>B", "condition_predictions": [{"condition_label": "RCR model prediction"}]},
            {"reaction_smiles": "B>>C", "condition_predictions": [{"condition_label": "RCR model prediction"}]},
            {"reaction_smiles": "C>>D", "condition_predictions": [{"condition_label": "RCR model prediction"}]},
            {"reaction_smiles": "D>>E", "condition_predictions": [{"condition_label": "Ac2O, DMAP"}]},
        ],
        "metrics": {
            "source_supported_semisynthesis": False,
            "stitched_semisynthesis": True,
            "native_returned_route": False,
            "terminal_reactants": ["A", "CC(=O)OC(C)=O"],
            "cascade_verifier": {"feasible": True},
        },
        "raw_backend_metadata": {"route_class_hint": "stitched_semisynthesis_upstream"},
    }

    classification = classify_route(route, target_smiles=BUFOTALIN_TARGET)

    assert classification["reportable"]
    assert not classification["presentation_ready"]
    assert classification["confidence_tier"] == "stitched_semisynthesis_upstream_review_only"
    assert classification["exclusion_reasons"] == []


def test_classify_unsupported_biosynthetic_prenyl_terminal_as_excluded():
    prenyl_terminal = (
        "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/"
        "CC/C(C)=C/CC/C(C)=C/CO"
    )
    route = {
        "n_steps": 4,
        "score": 0.2,
        "steps": [
            {"reaction_smiles": "A>>B", "condition_predictions": [{"condition_label": "RCR model prediction"}]},
            {"reaction_smiles": "B>>C", "condition_predictions": [{"condition_label": "RCR model prediction"}]},
            {"reaction_smiles": "C>>D", "condition_predictions": [{"condition_label": "RCR model prediction"}]},
            {"reaction_smiles": "D>>E", "condition_predictions": [{"condition_label": "RCR model prediction"}]},
        ],
        "metrics": {
            "source_supported_semisynthesis": False,
            "native_returned_route": True,
            "terminal_reactants": [prenyl_terminal],
            "cascade_verifier": {"feasible": True},
        },
    }

    classification = classify_route(route, target_smiles=BUFOTALIN_TARGET)

    assert not classification["reportable"]
    assert classification["confidence_tier"] == "excluded"
    assert "unsupported_biosynthetic_prenyl_terminal" in classification["warnings"]
    assert "unsupported_biosynthetic_prenyl_terminal" in classification["exclusion_reasons"]


def test_classify_enzyme_supported_prenyl_terminal_remains_review_only():
    prenyl_terminal = (
        "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/"
        "CC/C(C)=C/CC/C(C)=C/CO"
    )
    route = {
        "n_steps": 4,
        "score": 0.2,
        "steps": [
            {
                "reaction_smiles": "A>>B",
                "ec": "2.5.1.21",
                "is_enzymatic": True,
                "condition_predictions": [{"condition_label": "RCR model prediction"}],
            },
            {"reaction_smiles": "B>>C", "condition_predictions": [{"condition_label": "RCR model prediction"}]},
            {"reaction_smiles": "C>>D", "condition_predictions": [{"condition_label": "RCR model prediction"}]},
            {"reaction_smiles": "D>>E", "condition_predictions": [{"condition_label": "RCR model prediction"}]},
        ],
        "metrics": {
            "source_supported_semisynthesis": False,
            "native_returned_route": True,
            "terminal_reactants": [prenyl_terminal],
            "cascade_verifier": {"feasible": True},
        },
    }

    classification = classify_route(route, target_smiles=BUFOTALIN_TARGET)

    assert classification["reportable"]
    assert classification["confidence_tier"] == "native_model_candidate_review_only"
    assert "unsupported_biosynthetic_prenyl_terminal" not in classification["warnings"]
    assert classification["exclusion_reasons"] == []


def test_export_final_candidates_writes_conservative_package(tmp_path):
    cycle = tmp_path / "cycle_001"
    cycle.mkdir()
    (cycle / "web_payload.json").write_text(
        json.dumps(
            {
                "target_smiles": BUFOTALIN_TARGET,
                "ok": True,
                "n_results": 2,
                "search_status": {"status": "solved"},
                "routes": [
                    {
                        "n_steps": 1,
                        "score": 0.9,
                        "steps": [
                            {
                                "reaction_smiles": "CCO.CC(=O)OC(C)=O>>CCOC(C)=O",
                                "main_reactant": "CCO",
                                "aux_reactants": ["CC(=O)OC(C)=O"],
                                "condition_predictions": [{"condition_label": "Ac2O, DMAP"}],
                            }
                        ],
                        "metrics": {
                            "source_supported_semisynthesis": True,
                            "semisynthesis_anchor": True,
                            "native_returned_route": False,
                            "route_solved": True,
                            "terminal_reactants": ["CCO", "CC(=O)OC(C)=O"],
                            "cascade_verifier": {"feasible": True},
                        },
                        "raw_backend_metadata": {
                            "advanced_precursor_record": {
                                "name": "Deacetylbufotalin",
                                "cas": "465-19-0",
                            }
                        },
                    },
                    {
                        "n_steps": 4,
                        "score": 0.2,
                        "steps": [
                            {
                                "reaction_smiles": "A>>B",
                                "condition_predictions": [{"condition_label": "RCR model prediction"}],
                            },
                            {
                                "reaction_smiles": "B>>C",
                                "condition_predictions": [{"condition_label": "RCR model prediction"}],
                            },
                            {
                                "reaction_smiles": "C>>D",
                                "condition_predictions": [{"condition_label": "RCR model prediction"}],
                            },
                            {
                                "reaction_smiles": "D>>E",
                                "condition_predictions": [{"condition_label": "Ac2O, DMAP"}],
                            },
                        ],
                        "metrics": {
                            "source_supported_semisynthesis": False,
                            "stitched_semisynthesis": True,
                            "native_returned_route": False,
                            "route_solved": True,
                            "terminal_reactants": ["A", "CC(=O)OC(C)=O"],
                            "cascade_verifier": {"feasible": True},
                        },
                        "raw_backend_metadata": {"route_class_hint": "stitched_semisynthesis_upstream"},
                    },
                    {
                        "n_steps": 2,
                        "score": 0.1,
                        "steps": [
                            {
                                "reaction_smiles": "A>>B",
                                "condition_predictions": [{"condition_label": "RCR model prediction"}],
                            },
                            {
                                "reaction_smiles": "B>>C",
                                "condition_predictions": [{"condition_label": "RCR model prediction"}],
                            },
                        ],
                        "metrics": {
                            "source_supported_semisynthesis": False,
                            "native_returned_route": True,
                            "route_solved": True,
                            "terminal_reactants": [BUFOTALIN_TARGET],
                            "cascade_verifier": {"feasible": True},
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = export_final_candidates(tmp_path, render=False)
    data = json.loads((tmp_path / "final_candidates" / "final_candidates.json").read_text(encoding="utf-8"))

    assert report["high_confidence_count"] == 1
    assert report["stitched_review_only_count"] == 1
    assert report["native_review_only_count"] == 0
    assert report["selected_count"] == 2
    assert data["excluded_route_count"] == 1
    assert data["stitched_review_only_count"] == 1


def test_export_stitched_review_candidates_exclude_high_condition_risk(tmp_path):
    cycle = tmp_path / "cycle_001"
    cycle.mkdir()
    routes = [
        _stitched_route(
            n_steps=5,
            score=0.5,
            source="high_risk",
            conditions=[
                {"condition_label": "RCR model prediction", "Temperature": -60, "Reagent": "[Li+].[AlH4-]"},
                {"condition_label": "RCR model prediction"},
                {"condition_label": "RCR model prediction"},
                {"condition_label": "RCR model prediction"},
                {"condition_label": "Ac2O, DMAP"},
            ],
        ),
        _stitched_route(
            n_steps=4,
            score=0.01,
            source="low_risk",
            conditions=[
                {"condition_label": "RCR model prediction"},
                {"condition_label": "RCR model prediction"},
                {"condition_label": "RCR model prediction"},
                {"condition_label": "Ac2O, DMAP"},
            ],
        ),
    ]
    (cycle / "web_payload.json").write_text(
        json.dumps({"target_smiles": BUFOTALIN_TARGET, "ok": True, "routes": routes}),
        encoding="utf-8",
    )

    report = export_final_candidates(tmp_path, top_stitched=2, top_native=0, render=False)
    data = json.loads(Path(report["final_candidates_json"]).read_text(encoding="utf-8"))
    stitched = [
        row
        for row in data["selected"]
        if row["confidence_tier"] == "stitched_semisynthesis_upstream_review_only"
    ]

    assert [row["source_route_index"] for row in stitched] == [2]
    assert all("strong_hydride_reagent_predicted" not in row["warnings"] for row in stitched)
    assert any(
        "strong_hydride_reagent_predicted" in row["exclusion_reasons"]
        for row in data["excluded_sample"]
    )


def test_export_review_candidates_prefer_three_or_more_steps_before_short_routes(tmp_path):
    cycle = tmp_path / "cycle_001"
    cycle.mkdir()
    routes = [
        _stitched_route(
            n_steps=2,
            score=0.9,
            source="short",
            conditions=[
                {"condition_label": "RCR model prediction"},
                {"condition_label": "Ac2O, DMAP"},
            ],
        ),
        _stitched_route(
            n_steps=3,
            score=0.01,
            source="longer",
            conditions=[
                {"condition_label": "RCR model prediction"},
                {"condition_label": "RCR model prediction"},
                {"condition_label": "Ac2O, DMAP"},
            ],
        ),
    ]
    (cycle / "web_payload.json").write_text(
        json.dumps({"target_smiles": BUFOTALIN_TARGET, "ok": True, "routes": routes}),
        encoding="utf-8",
    )

    report = export_final_candidates(tmp_path, top_stitched=1, top_native=0, render=False)
    data = json.loads(Path(report["final_candidates_json"]).read_text(encoding="utf-8"))
    stitched = [
        row
        for row in data["selected"]
        if row["confidence_tier"] == "stitched_semisynthesis_upstream_review_only"
    ]

    assert len(stitched) == 1
    assert stitched[0]["n_steps"] == 3


def _stitched_route(*, n_steps: int, score: float, source: str, conditions: list[dict]):
    steps = []
    for idx in range(n_steps):
        steps.append(
            {
                "reaction_smiles": f"{source}_{idx}>>{source}_{idx + 1}",
                "main_reactant": f"{source}_{idx}",
                "condition_predictions": [conditions[idx]],
            }
        )
    return {
        "n_steps": n_steps,
        "score": score,
        "steps": steps,
        "metrics": {
            "source_supported_semisynthesis": False,
            "stitched_semisynthesis": True,
            "native_returned_route": False,
            "route_solved": True,
            "terminal_reactants": [f"{source}_0", "CC(=O)OC(C)=O"],
            "cascade_verifier": {"feasible": True},
        },
        "raw_backend_metadata": {"route_class_hint": "stitched_semisynthesis_upstream"},
    }

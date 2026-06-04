from cascade_planner.baselines.chem_enzy_native_enzyme_plugin import PLUGIN_MODEL_FULL_NAME
from cascade_planner.baselines.chem_enzy_adapter import route_candidates_from_chem_enzy_result
from cascade_planner.baselines.enzyme_step_audit import (
    audit_baseline_results,
    proposal_domain_for_step,
    summarize_enzyme_step_audit,
)
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteCandidate, RouteStepCandidate
from scripts.export_chem_enzy_enzyme_step_audit import (
    comparison_delta,
    plugin_stats_summary,
    render_comparison_markdown,
)


def test_audit_separates_posthoc_ec_from_enzyme_like_source():
    step = RouteStepCandidate(
        product_smiles="CCO",
        reactant_smiles=["CC=O"],
        rxn_smiles="CC=O>>CCO",
        source_model="graphfp_models.USPTO-full_remapped",
        enzyme_ec_annotations=[
            {"rank": "Top-1", "ec_number": "1.1.1.1", "confidence": 0.82},
        ],
        raw_backend_metadata={
            "rxn_attribute": {
                "organic_enzyme_rxn_classification": {
                    "Reaction Type": {"0": "Enzymatic Reaction"},
                    "Confidence": {"0": 0.91},
                }
            }
        },
    )
    rows = audit_baseline_results([_result([step])])

    assert len(rows) == 1
    row = rows[0]
    assert row["proposal_domain"] == "chemical"
    assert row["generated_by_enzyme_like_source"] is False
    assert row["posthoc_classified_enzymatic"] is True
    assert row["ec_top1"] == "1.1.1.1"
    assert "posthoc_enzymatic_on_chemical_source" in row["weakness_flags"]


def test_audit_marks_native_bionav_as_enzyme_like_source_without_ec():
    step = RouteStepCandidate(
        product_smiles="CCO",
        reactant_smiles=["CC=O"],
        rxn_smiles="CC=O>>CCO",
        source_model="onmt_models.bionav_one_step",
        raw_backend_metadata={
            "rxn_attribute": {
                "organic_enzyme_rxn_classification": {
                    "Reaction Type": {"0": "Organic Reaction"},
                    "Confidence": {"0": 0.71},
                }
            }
        },
    )
    rows = audit_baseline_results([_result([step])])

    assert proposal_domain_for_step(step) == "enzymatic"
    assert rows[0]["proposal_source_kind"] == "native_bionav"
    assert rows[0]["posthoc_classified_organic"] is True
    assert "enzyme_like_source_without_ec" in rows[0]["weakness_flags"]
    assert "enzyme_like_source_classified_organic" in rows[0]["weakness_flags"]


def test_audit_extracts_plugin_sp_v1_and_material_failure():
    step = RouteStepCandidate(
        product_smiles="CCCCCCCCCCCCCCCCCC",
        reactant_smiles=["CCC"],
        rxn_smiles="CCC>>CCCCCCCCCCCCCCCCCC",
        source_model=PLUGIN_MODEL_FULL_NAME,
        enzyme_ec_annotations=[
            {"rank": "Top-1", "ec_number": "2.5.1.1", "confidence": 0.73},
        ],
        raw_backend_metadata={
            "template": {
                "model_full_name": PLUGIN_MODEL_FULL_NAME,
                "ec": "2.5.1.1",
                "autoplanner_native_enzyme_plugin": True,
                "evidence": {"transition_signature": "prenyl_transfer"},
                "enzyme_sp_verifier_v1": {
                    "score": 0.77,
                    "threshold": 0.35,
                    "accepted": True,
                    "ec_numbers": ["2.5.1.1"],
                },
                "autoplanner_enzyme_quality_v1": {
                    "quality_score": 0.31,
                    "decision": "reject",
                    "flags": ["material_sanity_failed"],
                    "material_sanity": {"passed": False, "reasons": ["large_unexplained_carbon_gain"]},
                },
            }
        },
    )
    rows = audit_baseline_results([_result([step])])

    row = rows[0]
    assert row["plugin_injected"] is True
    assert row["sp_v1_accepted"] is True
    assert row["enzyme_quality_decision"] == "reject"
    assert row["enzyme_quality_score"] == 0.31
    assert row["template_transition_signature"] == "prenyl_transfer"
    assert row["material_audit_passed"] is False
    assert "plugin_injected_material_failed" in row["weakness_flags"]
    assert "search_time_enzyme_quality_rejected" in row["weakness_flags"]


def test_summary_counts_core_error_modes():
    chemical = RouteStepCandidate(
        product_smiles="CCO",
        reactant_smiles=["CC=O"],
        rxn_smiles="CC=O>>CCO",
        source_model="graphfp_models.USPTO-full_remapped",
        enzyme_ec_annotations=[{"rank": "Top-1", "ec_number": "1.1.1.1", "confidence": 0.2}],
        raw_backend_metadata={
            "rxn_attribute": {
                "organic_enzyme_rxn_classification": {"Reaction Type": {"0": "Enzymatic Reaction"}}
            }
        },
    )
    enzyme = RouteStepCandidate(
        product_smiles="CCO",
        reactant_smiles=["CC=O"],
        rxn_smiles="CC=O>>CCO",
        source_model="onmt_models.bionav_one_step",
    )
    rows = audit_baseline_results([_result([chemical, enzyme])])
    summary = summarize_enzyme_step_audit(rows)

    assert summary["steps"] == 2
    assert summary["enzyme_like_source_steps"] == 1
    assert summary["posthoc_enzymatic_steps"] == 1
    assert summary["posthoc_enzymatic_on_chemical_source_steps"] == 1
    assert summary["enzyme_source_without_ec_steps"] == 1
    assert summary["weakness_flag_counts"]["low_top1_ec_confidence"] == 1


def test_audit_derives_visible_quality_for_native_bionav_step():
    step = RouteStepCandidate(
        product_smiles="CCO",
        reactant_smiles=["CC=O"],
        rxn_smiles="CC=O>>CCO",
        source_model="onmt_models.bionav_one_step",
        enzyme_ec_annotations=[{"rank": "Top-1", "ec_number": "1.1.1.1", "confidence": 0.73}],
        raw_backend_metadata={
            "cascade_cost": {
                "total_cost": 1.2,
                "cascade_adjustment": 0.7,
                "components": {"enzyme_evidence": 0.7, "material_sanity": 0.0},
                "material_sanity": {"passed": True, "reasons": []},
            }
        },
    )
    rows = audit_baseline_results([_result([step])])
    summary = summarize_enzyme_step_audit(rows)

    row = rows[0]
    assert row["enzyme_quality_origin"] == "derived"
    assert row["enzyme_quality_decision"] == "warn"
    assert row["enzyme_quality_score"] == 0.6
    assert "missing_sp_v1" in row["enzyme_quality_flags"]
    assert "native_or_posthoc_derived_quality" in row["enzyme_quality_flags"]
    assert summary["enzyme_quality_scored_steps"] == 1
    assert summary["derived_quality_scored_steps"] == 1
    assert summary["search_time_quality_scored_steps"] == 0


def test_audit_exports_step_enhancement_opportunity(monkeypatch):
    monkeypatch.setattr(
        "cascade_planner.baselines.enzyme_step_enhancement.retrieve_enzyme_precedents",
        lambda *args, **kwargs: [
            {
                "main_reactant": "CCO",
                "aux_reactants": [],
                "rxn_smiles": "CCO>>CC=O",
                "source": "enzyme_precedent",
                "score": 0.9,
                "ec": "1.1.1.1",
                "type": "enzyme_precedent_retrieval",
                "evidence": {
                    "source_db": "fixture",
                    "reaction_id": "rxn1",
                    "product_similarity": 1.0,
                    "transition_signature": {"transition_quality_score": 0.9, "transition_flags": []},
                    "occurrences": 20,
                },
            }
        ],
    )

    class FakeSP:
        def score_action(self, *, product, action):
            return type(
                "Score",
                (),
                {
                    "to_dict": lambda self: {
                        "score": 0.91,
                        "threshold": 0.3,
                        "accepted": True,
                        "ec_numbers": [action.ec],
                    }
                },
            )()

    step = RouteStepCandidate(
        product_smiles="CC=O",
        reactant_smiles=["CCCO"],
        rxn_smiles="CCCO>>CC=O",
        source_model="graphfp_models.USPTO-full_remapped",
    )
    rows = audit_baseline_results([_result([step])], enable_step_enhancement=True, step_enhancement_scorer=FakeSP())
    summary = summarize_enzyme_step_audit(rows)

    assert rows[0]["enzyme_step_enhancement_kind"] == "missing_enzyme_step"
    assert rows[0]["enzyme_step_enhancement_best_main_reactant"] == "CCO"
    assert summary["missing_enzyme_step_opportunities"] == 1
    assert summary["step_enhancement_viable_candidates"] == 1


def test_strengthening_comparison_report_includes_plugin_delta():
    native_plugin = plugin_stats_summary([])
    strengthened_plugin = plugin_stats_summary(
        [
            BaselineRunResult(
                target_smiles="CCO",
                backend="ChemEnzyRetroPlanner",
                raw_backend_metadata={
                    "native_enzyme_plugin": {
                        "enabled": True,
                        "calls": 1,
                        "bridge_hit_calls": 1,
                        "retrieved_candidates": 4,
                        "quality_scored": 2,
                        "quality_passed": 2,
                        "added_candidates": 2,
                    }
                },
            )
        ]
    )
    comparison = {
        "native": {"routes": 1, "steps": 1},
        "strengthened": {"routes": 2, "steps": 2},
        "delta": comparison_delta({"routes": 1, "steps": 1}, {"routes": 2, "steps": 2}),
        "native_plugin": native_plugin,
        "strengthened_plugin": strengthened_plugin,
        "plugin_delta": comparison_delta(native_plugin, strengthened_plugin),
    }

    md = render_comparison_markdown(comparison)

    assert strengthened_plugin["quality_scored"] == 2
    assert "| routes | 1 | 2 | +1 |" in md
    assert "## Search-Time Plugin Delta" in md
    assert "| plugin calls | 0 | 1 | +1 |" in md
    assert "| added candidates | 0 | 2 | +2 |" in md


def test_route_conversion_preserves_cascade_cost_source_model_for_audit():
    raw = {
        "all_succ_dict_routes": [
            {
                "smiles": "CCO",
                "type": "mol",
                "children": [
                    {
                        "type": "reaction",
                        "template": None,
                        "cascade_cost": {
                            "source_model": "onmt_models.bionav_one_step",
                            "reaction_domain": "enzymatic",
                        },
                        "children": [{"smiles": "CC=O", "type": "mol", "in_stock": True}],
                    }
                ],
            }
        ],
    }

    route = route_candidates_from_chem_enzy_result(raw, target_smiles="CCO")[0]
    step = route.steps[0]

    assert step.source_model == "onmt_models.bionav_one_step"
    assert proposal_domain_for_step(step) == "enzymatic"


def _result(steps):
    route = RouteCandidate(
        target_smiles="CCO",
        steps=list(steps),
        backend="ChemEnzyRetroPlanner",
        solved=True,
        route_rank=0,
    )
    return BaselineRunResult(
        target_smiles="CCO",
        backend="ChemEnzyRetroPlanner",
        routes=[route],
    )

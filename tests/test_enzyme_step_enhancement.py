from types import SimpleNamespace

from cascade_planner.baselines.enzyme_step_enhancement import (
    EnzymeStepEnhancementConfig,
    evaluate_step_enhancement,
)
from cascade_planner.baselines.route_contract import RouteStepCandidate


class FakeAcceptedSP:
    def score_action(self, *, product, action):
        return SimpleNamespace(
            to_dict=lambda: {
                "score": 0.91,
                "threshold": 0.30,
                "accepted": True,
                "ec_numbers": [action.ec],
            }
        )


def test_enhancement_finds_missing_enzyme_step(monkeypatch):
    monkeypatch.setattr(
        "cascade_planner.baselines.enzyme_step_enhancement.retrieve_enzyme_precedents",
        lambda *args, **kwargs: [_precedent("CCO", "1.1.1.1")],
    )
    step = RouteStepCandidate(
        product_smiles="CC=O",
        reactant_smiles=["CCCO"],
        rxn_smiles="CCCO>>CC=O",
        source_model="graphfp_models.USPTO-full_remapped",
    )

    result = evaluate_step_enhancement(
        step,
        scorer=FakeAcceptedSP(),
        config=EnzymeStepEnhancementConfig(min_efficiency_score=0.45),
    )

    assert result["available"] is True
    assert result["recommended_kind"] == "missing_enzyme_step"
    assert result["best_candidate"]["main_reactant"] == "CCO"
    assert result["best_candidate"]["sp_v1_accepted"] is True


def test_enhancement_replaces_weak_native_enzyme_step(monkeypatch):
    monkeypatch.setattr(
        "cascade_planner.baselines.enzyme_step_enhancement.retrieve_enzyme_precedents",
        lambda *args, **kwargs: [_precedent("CCO", "1.1.1.1")],
    )
    step = RouteStepCandidate(
        product_smiles="CC=O",
        reactant_smiles=["CCCCCCCC"],
        rxn_smiles="CCCCCCCC>>CC=O",
        source_model="onmt_models.bionav_one_step",
        enzyme_ec_annotations=[{"ec_number": "1.1.1.1", "confidence": 0.2}],
    )

    result = evaluate_step_enhancement(
        step,
        scorer=FakeAcceptedSP(),
        config=EnzymeStepEnhancementConfig(min_efficiency_score=0.45),
    )

    assert result["recommended_kind"] == "wrong_enzyme_step_replacement"
    assert "selected_enzyme_step_lacks_required_evidence" in result["reasons"]


def test_posthoc_ec_on_chemical_step_is_missing_search_time_enzyme_step(monkeypatch):
    monkeypatch.setattr(
        "cascade_planner.baselines.enzyme_step_enhancement.retrieve_enzyme_precedents",
        lambda *args, **kwargs: [_precedent("CCO", "1.1.1.1")],
    )
    step = RouteStepCandidate(
        product_smiles="CC=O",
        reactant_smiles=["CCCO"],
        rxn_smiles="CCCO>>CC=O",
        source_model="graphfp_models.USPTO-full_remapped",
        enzyme_ec_annotations=[{"ec_number": "1.1.1.1", "confidence": 0.9}],
    )

    result = evaluate_step_enhancement(
        step,
        scorer=FakeAcceptedSP(),
        config=EnzymeStepEnhancementConfig(min_efficiency_score=0.45),
    )

    assert result["current"]["has_search_time_enzyme_source"] is False
    assert result["current"]["has_posthoc_ec_annotation"] is True
    assert result["recommended_kind"] == "missing_enzyme_step"
    assert "selected_step_only_has_posthoc_ec_annotation" in result["reasons"]


def _precedent(main_reactant, ec):
    return {
        "main_reactant": main_reactant,
        "aux_reactants": [],
        "rxn_smiles": f"{main_reactant}>>CC=O",
        "source": "enzyme_precedent",
        "score": 0.95,
        "ec": ec,
        "type": "enzyme_precedent_retrieval",
        "evidence": {
            "source_db": "fixture",
            "reaction_id": "rxn1",
            "product_similarity": 1.0,
            "transition_signature": {
                "transition_quality_score": 0.92,
                "transition_flags": [],
            },
            "occurrences": 25,
            "example_ids": ["rxn1:example"],
        },
    }

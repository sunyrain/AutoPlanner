from types import SimpleNamespace
from unittest.mock import patch

from cascade_planner.baselines.chem_enzy_adapter import route_candidates_from_chem_enzy_result
from cascade_planner.baselines.chem_enzy_native_enzyme_plugin import (
    NativeEnzymeOneStepWrapper,
    NativeEnzymePluginConfig,
    NativeEnzymePluginState,
    PLUGIN_MODEL_FULL_NAME,
)


class FakeOneStep:
    one_step_models = {"graphfp_models.fake": object()}

    def run(self, target, *args, **kwargs):
        return {
            "reactants": ["C"],
            "scores": [0.2],
            "template": [{"model_full_name": "graphfp_models.fake"}],
            "model_full_name": ["graphfp_models.fake"],
            "weight": [1.0],
        }


class FakeBridgeRetriever:
    def __init__(self, *args, **kwargs):
        pass

    def retrieve(self, *args, **kwargs):
        return [SimpleNamespace(enzyme_ec_sample=("1.1.1.1",))]


class EmptyBridgeRetriever(FakeBridgeRetriever):
    def retrieve(self, *args, **kwargs):
        return []


class FakeAcceptedSP:
    def __init__(self, *args, **kwargs):
        pass

    def score_action(self, *, product, action):
        return SimpleNamespace(
            accepted=True,
            to_dict=lambda: {
                "score": 0.9,
                "threshold": 0.3,
                "accepted": True,
                "ec_numbers": ["1.1.1.1"],
            },
        )


class FakeRejectedSP(FakeAcceptedSP):
    def score_action(self, *, product, action):
        return SimpleNamespace(
            accepted=False,
            to_dict=lambda: {
                "score": 0.1,
                "threshold": 0.3,
                "accepted": False,
                "ec_numbers": ["1.1.1.1"],
            },
        )


def _enzyme_rows(*args, **kwargs):
    return [
        {
            "main_reactant": "CC",
            "aux_reactants": [],
            "rxn_smiles": "CC>>CCO",
            "source": "enzyme_precedent",
            "score": 0.7,
            "ec": "1.1.1.1",
            "type": "enzyme_precedent_retrieval",
            "evidence": {"ec_numbers": ["1.1.1.1"]},
        }
    ]


def _material_jump_rows(*args, **kwargs):
    product = "CCCCCCCCCCCCCCCCCC"
    return [
        {
            "main_reactant": "CCC",
            "aux_reactants": [],
            "rxn_smiles": f"CCC>>{product}",
            "source": "enzyme_precedent",
            "score": 0.7,
            "ec": "2.5.1.1",
            "type": "enzyme_precedent_retrieval",
            "evidence": {"transition_signature": "prenyl_transfer", "ec_numbers": ["2.5.1.1"]},
        }
    ]


def test_native_enzyme_wrapper_appends_bridge_gated_sp_accepted_candidate():
    config = NativeEnzymePluginConfig(enabled=True, top_k=1, max_added=1)
    state = NativeEnzymePluginState(config=config)
    wrapper = NativeEnzymeOneStepWrapper(FakeOneStep(), config=config, state=state)

    with (
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin._make_bridge_retriever", lambda *_: FakeBridgeRetriever()),
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin._make_sp_v1_scorer", lambda: FakeAcceptedSP()),
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin.retrieve_enzyme_precedents", _enzyme_rows),
    ):
        result = wrapper.run("CCO")

    assert result["reactants"] == ["C", "CC"]
    assert result["model_full_name"][-1] == PLUGIN_MODEL_FULL_NAME
    assert result["template"][-1]["autoplanner_native_enzyme_plugin"] is True
    assert result["template"][-1]["enzyme_sp_verifier_v1"]["accepted"] is True
    assert result["template"][-1]["autoplanner_enzyme_quality_v1"]["decision"] == "pass"
    assert result["reaction_domains"] == [None, "enzymatic"]
    assert result["enzyme_evidence_confidences"][-1] > 0.5
    assert state.calls == 1
    assert state.bridge_hit_calls == 1
    assert state.added_candidates == 1
    assert state.sp_v1_accepted == 1
    assert state.quality_passed == 1


def test_native_enzyme_wrapper_skips_without_bridge_hit_by_default():
    config = NativeEnzymePluginConfig(enabled=True, top_k=1, max_added=1)
    state = NativeEnzymePluginState(config=config)
    wrapper = NativeEnzymeOneStepWrapper(FakeOneStep(), config=config, state=state)

    with (
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin._make_bridge_retriever", lambda *_: EmptyBridgeRetriever()),
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin.retrieve_enzyme_precedents", _enzyme_rows),
    ):
        result = wrapper.run("CCO")

    assert result["reactants"] == ["C"]
    assert state.skipped_no_bridge == 1
    assert state.added_candidates == 0


def test_native_enzyme_wrapper_hard_gate_rejects_low_sp_v1_candidate():
    config = NativeEnzymePluginConfig(enabled=True, top_k=1, max_added=1, sp_v1_hard_gate=True)
    state = NativeEnzymePluginState(config=config)
    wrapper = NativeEnzymeOneStepWrapper(FakeOneStep(), config=config, state=state)

    with (
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin._make_bridge_retriever", lambda *_: FakeBridgeRetriever()),
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin._make_sp_v1_scorer", lambda: FakeRejectedSP()),
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin.retrieve_enzyme_precedents", _enzyme_rows),
    ):
        result = wrapper.run("CCO")

    assert result["reactants"] == ["C"]
    assert state.sp_v1_rejected == 1
    assert state.added_candidates == 0


def test_native_enzyme_wrapper_rejects_material_failed_enzyme_candidate():
    config = NativeEnzymePluginConfig(enabled=True, top_k=1, max_added=1, require_material_sanity=True)
    state = NativeEnzymePluginState(config=config)
    wrapper = NativeEnzymeOneStepWrapper(FakeOneStep(), config=config, state=state)

    with (
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin._make_bridge_retriever", lambda *_: FakeBridgeRetriever()),
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin._make_sp_v1_scorer", lambda: FakeAcceptedSP()),
        patch("cascade_planner.baselines.chem_enzy_native_enzyme_plugin.retrieve_enzyme_precedents", _material_jump_rows),
    ):
        result = wrapper.run("CCCCCCCCCCCCCCCCCC")

    assert result["reactants"] == ["C"]
    assert state.quality_scored == 1
    assert state.material_rejected == 1
    assert state.added_candidates == 0


def test_route_conversion_marks_native_plugin_template_as_enzymatic():
    raw = {
        "all_succ_dict_routes": [
            {
                "smiles": "CCO",
                "type": "mol",
                "children": [
                    {
                        "type": "reaction",
                        "template": {
                            "model_full_name": PLUGIN_MODEL_FULL_NAME,
                            "source": "enzyme_precedent",
                            "ec": "1.1.1.1",
                            "autoplanner_native_enzyme_plugin": True,
                            "enzyme_sp_verifier_v1": {
                                "score": 0.9,
                                "threshold": 0.3,
                                "accepted": True,
                                "ec_numbers": ["1.1.1.1"],
                            },
                        },
                        "children": [{"smiles": "CC", "type": "mol", "in_stock": True}],
                    }
                ],
            }
        ],
    }

    routes = route_candidates_from_chem_enzy_result(raw, target_smiles="CCO")

    assert routes[0].enzymatic_step_present is True
    assert routes[0].steps[0].source_model == PLUGIN_MODEL_FULL_NAME
    assert routes[0].steps[0].enzyme_ec_annotations[0]["ec_number"] == "1.1.1.1"

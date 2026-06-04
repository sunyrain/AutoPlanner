from types import SimpleNamespace

from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    _configure_native_autoplanner_plugins,
    route_candidates_from_chem_enzy_result,
)
from cascade_planner.baselines.chem_enzy_native_chemical_plugin import (
    NativeChemicalOneStepWrapper,
    NativeChemicalPluginConfig,
    NativeChemicalPluginState,
    PLUGIN_MODEL_FULL_NAME,
)
from cascade_planner.baselines.chem_enzy_native_enzyme_plugin import (
    NativeEnzymePluginConfig,
    PLUGIN_MODEL_FULL_NAME as ENZYME_PLUGIN_MODEL_FULL_NAME,
)


class FakeGraphFPOneStep:
    one_step_models = {"graphfp_models.USPTO-full_remapped": object()}

    def run(self, target, *args, **kwargs):
        return {
            "reactants": ["CC.C"],
            "scores": [0.9],
            "template": [{"model_full_name": "graphfp_models.USPTO-full_remapped"}],
            "model_full_name": ["graphfp_models.USPTO-full_remapped"],
            "weight": [1.0],
        }


class FakeFusion:
    def dual_rows(self, product):
        return [
            {
                "reactant_smiles": ["CC"],
                "rxn_smiles": f"CC>>{product}",
                "reaction_smiles": f"CC>>{product}",
                "source": "autoplanner_dualtower",
                "score": 2.0,
                "rank": 1,
                "template": "dual_template",
                "template_id": 7,
                "template_rank": 3,
                "model_full_name": "autoplanner_dualtower.enhanced_v2_e8_ft",
            }
        ]


def test_native_chemical_wrapper_appends_dualtower_tail_candidate(monkeypatch):
    config = NativeChemicalPluginConfig(enabled=True, top_k=4, max_added=2, require_proposal_gate=True)
    state = NativeChemicalPluginState(config=config)
    wrapper = NativeChemicalOneStepWrapper(FakeGraphFPOneStep(), config=config, state=state)
    monkeypatch.setattr(
        "cascade_planner.baselines.chem_enzy_native_chemical_plugin._make_graphfp_dualtower_fusion",
        lambda *_: FakeFusion(),
    )

    result = wrapper.run("CCC")

    assert result["reactants"] == ["CC.C", "CC"]
    assert result["model_full_name"][-1] == PLUGIN_MODEL_FULL_NAME
    assert result["template"][-1]["autoplanner_native_chemical_plugin"] is True
    assert result["template"][-1]["proposal_gate"]["decision"] == "keep"
    assert result["reaction_domains"] == [None, "organic"]
    assert state.calls == 1
    assert state.graphfp_base_candidates == 1
    assert state.dual_candidates == 1
    assert state.added_candidates == 1
    assert state.proposal_gate_kept == 1


def test_route_conversion_marks_native_chemical_plugin_as_organic_source():
    raw = {
        "all_succ_dict_routes": [
            {
                "smiles": "CCC",
                "type": "mol",
                "children": [
                    {
                        "type": "reaction",
                        "template": {
                            "model_full_name": PLUGIN_MODEL_FULL_NAME,
                            "source": "autoplanner_dualtower",
                            "autoplanner_native_chemical_plugin": True,
                        },
                        "children": [{"smiles": "CC", "type": "mol", "in_stock": True}],
                    }
                ],
            }
        ],
    }

    routes = route_candidates_from_chem_enzy_result(raw, target_smiles="CCC")

    assert routes[0].enzymatic_step_present is False
    assert routes[0].steps[0].source_model == PLUGIN_MODEL_FULL_NAME
    assert routes[0].steps[0].enzyme_ec_annotations == []


def test_adapter_combined_plugin_patch_wraps_chemical_then_enzyme(monkeypatch):
    calls = []

    def original_prepare(one_step):
        calls.append(one_step)
        return SimpleNamespace(one_step=one_step)

    api = SimpleNamespace(prepare_molstar_planner=original_prepare)
    enzyme_config = NativeEnzymePluginConfig(enabled=True)
    chemical_config = NativeChemicalPluginConfig(enabled=True)

    enzyme_state, chemical_state = _configure_native_autoplanner_plugins(
        api,
        enzyme_config=enzyme_config,
        chemical_config=chemical_config,
    )
    api.prepare_molstar_planner(FakeGraphFPOneStep())

    assert enzyme_state is not None
    assert chemical_state is not None
    assert calls
    assert calls[0].one_step_models[PLUGIN_MODEL_FULL_NAME] is calls[0].one_step
    assert calls[0].one_step_models[ENZYME_PLUGIN_MODEL_FULL_NAME] is calls[0]


def test_adapter_no_route_result_keeps_plugin_stats():
    adapter = ChemEnzyBackendAdapter()
    state_config = NativeChemicalPluginConfig(enabled=True)
    state = NativeChemicalPluginState(config=state_config)

    def plan(_target):
        state.added_candidates = 2
        return None

    planner = SimpleNamespace(plan=plan)
    config = SimpleNamespace(
        target_smiles="CCC",
        search_flags={},
    )
    planner._autoplanner_native_chemical_plugin_state = state

    result = adapter._run_with_planner(planner, config)

    assert result.solved is False
    assert result.raw_backend_metadata["native_chemical_plugin"]["added_candidates"] == 2

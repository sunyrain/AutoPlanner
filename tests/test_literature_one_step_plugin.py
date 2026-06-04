from types import SimpleNamespace

from cascade_planner.agent.literature_templates import default_literature_template_cards
from cascade_planner.baselines.chem_enzy_adapter import (
    _configure_native_autoplanner_plugins,
    route_candidates_from_chem_enzy_result,
)
from cascade_planner.baselines.chem_enzy_native_chemical_plugin import NativeChemicalPluginConfig
from cascade_planner.baselines.chem_enzy_native_enzyme_plugin import NativeEnzymePluginConfig
from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider
from cascade_planner.baselines.literature_one_step_plugin import (
    LiteratureOneStepPlugin,
    LiteratureOneStepPluginConfig,
    LiteratureOneStepPluginState,
    LiteratureTemplateOneStepWrapper,
    PLUGIN_MODEL_FULL_NAME,
)
from cascade_planner.route_tree.source_gate import source_group, source_policy_group


PHENOLIC_O_GLYCOSIDE = "Oc1ccccc1OC1COC(O)C(O)C1O"
TAXANE = "CC(=O)OC1CC(O)C2(C)C(OC(=O)c3ccccc3)C3OC3C(O)C12"


class FakeOneStep:
    one_step_models = {"graphfp_models.fake": object()}

    def run(self, product, topk=10):
        return {
            "reactants": ["CC.O"],
            "scores": [0.8],
            "template": [{"model_full_name": "graphfp_models.fake"}],
            "model_full_name": ["graphfp_models.fake"],
            "weight": [1.0],
        }


def test_literature_plugin_run_returns_chemenzy_compatible_validated_candidate():
    plugin = LiteratureOneStepPlugin()

    result = plugin.run(PHENOLIC_O_GLYCOSIDE, topk=4)

    assert result["reactants"]
    assert result["model_full_name"][0] == PLUGIN_MODEL_FULL_NAME
    assert result["template"][0]["source"] == "literature_template_plugin"
    assert result["template"][0]["evidence_refs"]
    assert result["template"][0]["not_lab_procedure"] is True
    assert result["template"][0]["requires_audit"] is True
    assert result["template"][0]["template_validation_report"]["allowed_for_one_step_source"] is True


def test_literature_wrapper_appends_candidate_without_breaking_native_rows():
    config = LiteratureOneStepPluginConfig(enabled=True, top_k=4, max_added=2)
    state = LiteratureOneStepPluginState(config=config)
    wrapper = LiteratureTemplateOneStepWrapper(FakeOneStep(), config=config, state=state)

    result = wrapper.run(TAXANE, topk=4)

    assert result["reactants"][0] == "CC.O"
    assert result["model_full_name"][0] == "graphfp_models.fake"
    assert PLUGIN_MODEL_FULL_NAME in result["model_full_name"]
    idx = result["model_full_name"].index(PLUGIN_MODEL_FULL_NAME)
    assert result["template"][idx]["source"] == "literature_template_plugin"
    assert state.added_candidates == 1


def test_chem_enzy_onestep_provider_preserves_literature_metadata():
    plugin = LiteratureOneStepPlugin()
    provider = ChemEnzyOneStepProposalProvider(one_step=plugin)

    rows = provider.predict(PHENOLIC_O_GLYCOSIDE, top_k=2)

    assert rows[0]["source"] == "literature_template_plugin"
    assert rows[0]["source_model"] == "literature_template_plugin"
    assert rows[0]["proposal_type"] == "literature_template_plugin"
    assert rows[0]["template_validation_report"]["allowed_for_one_step_source"] is True
    assert rows[0]["evidence_refs"]
    assert rows[0]["requires_audit"] is True


def test_adapter_wraps_literature_plugin_and_route_conversion_marks_source():
    calls = []

    def original_prepare(one_step):
        calls.append(one_step)
        return SimpleNamespace(one_step=one_step)

    api = SimpleNamespace(prepare_molstar_planner=original_prepare)
    enzyme_state, chemical_state, literature_state = _configure_native_autoplanner_plugins(
        api,
        enzyme_config=NativeEnzymePluginConfig(enabled=False),
        chemical_config=NativeChemicalPluginConfig(enabled=False),
        literature_config=LiteratureOneStepPluginConfig(enabled=True),
    )
    api.prepare_molstar_planner(FakeOneStep())

    assert enzyme_state is None
    assert chemical_state is None
    assert literature_state is not None
    assert calls[0].one_step_models[PLUGIN_MODEL_FULL_NAME] is calls[0].plugin

    raw = {
        "all_succ_dict_routes": [
            {
                "smiles": TAXANE,
                "type": "mol",
                "children": [
                    {
                        "type": "reaction",
                        "template": {
                            "model_full_name": PLUGIN_MODEL_FULL_NAME,
                            "source": "literature_template_plugin",
                            "source_model": "literature_template_plugin",
                        },
                        "children": [{"smiles": "CC", "type": "mol", "in_stock": False}],
                    }
                ],
            }
        ],
    }
    routes = route_candidates_from_chem_enzy_result(raw, target_smiles=TAXANE)

    assert routes[0].steps[0].source_model == "literature_template_plugin"


def test_literature_plugin_source_groups_are_chemical_template_policy_visible():
    assert source_group("literature_template_plugin") == "chemical"
    assert source_policy_group("literature_template_plugin") == "chemical"

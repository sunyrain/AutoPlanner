from types import SimpleNamespace

from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    _configure_native_autoplanner_plugins,
    route_candidates_from_chem_enzy_result,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig
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
from scripts.run_chem_enzy_plan_for_web import _route_config_from_payload


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


def test_literature_wrapper_preserves_template_alias_lengths_for_vendor_mcts():
    class TemplateOnlyBase:
        one_step_models = {"graphfp_models.fake": object()}

        def run(self, product, topk=10):
            del product, topk
            return {
                "reactants": ["CC.O"],
                "scores": [0.8],
                "template": [{"model_full_name": "graphfp_models.fake"}],
                "model_full_name": ["graphfp_models.fake"],
                "weight": [1.0],
            }

    row = _compiled_row(product_smiles="CCO", reactants="CCN")
    config = LiteratureOneStepPluginConfig.from_raw(
        {
            "enabled": True,
            "max_added": 2,
            "template_cards": [],
            "one_step_rows": [row],
        }
    )
    state = LiteratureOneStepPluginState(config=config)
    wrapper = LiteratureTemplateOneStepWrapper(TemplateOnlyBase(), config=config, state=state)

    result = wrapper.run("CCO", topk=4)

    assert len(result["scores"]) == 2
    assert len(result["costs"]) == 2
    assert len(result["template"]) == 2
    assert len(result["templates"]) == 2
    assert result["templates"][0]["model_full_name"] == "graphfp_models.fake"
    assert result["templates"][1]["source"] == "literature_template_plugin"


def test_literature_plugin_consumes_compiled_one_step_rows_for_matching_product():
    row = _compiled_row(product_smiles="CCO", reactants="CC.O")
    config = LiteratureOneStepPluginConfig.from_raw(
        {
            "enabled": True,
            "max_added": 2,
            "template_cards": [],
            "one_step_rows": [row],
        }
    )
    plugin = LiteratureOneStepPlugin(config=config)

    result = plugin.run("CCO", topk=4)
    mismatch = plugin.run("CCN", topk=4)

    assert result["reactants"] == ["CC.O"]
    assert result["model_full_name"][0] == PLUGIN_MODEL_FULL_NAME
    assert result["template"][0]["source"] == "literature_template_plugin"
    assert result["template"][0]["requires_audit"] is True
    assert result["template"][0]["no_solved_claim"] is True
    assert mismatch["reactants"] == []
    assert plugin.state.added_candidates == 1


def test_literature_plugin_respects_explicit_empty_template_cards():
    row = _compiled_row(product_smiles="CCO", reactants="CC.O")
    config = LiteratureOneStepPluginConfig.from_raw(
        {
            "enabled": True,
            "max_added": 4,
            "template_cards": [],
            "one_step_rows": [row],
        }
    )
    plugin = LiteratureOneStepPlugin(config=config)

    result = plugin.run("CCO", topk=4)

    assert result["reactants"] == ["CC.O"]
    assert plugin.state.candidate_templates == 0
    assert config.use_default_template_cards is False


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


def test_adapter_dry_run_preserves_compiled_one_step_rows_in_search_config():
    row = _compiled_row(product_smiles="CCO", reactants="CC.O")
    config = RouteSearchConfig(
        target_smiles="CCO",
        search_flags={
            "literature_template_plugin": {
                "enabled": True,
                "max_added": 2,
                "one_step_rows": [row],
                "template_cards": [],
            }
        },
    )
    adapter = ChemEnzyBackendAdapter(vendor_root="/missing/vendor/root")

    result = adapter.run_target(config, dry_run=True)

    plugin_flags = result.raw_backend_metadata["search_config"]["search_flags"]["literature_template_plugin"]
    assert plugin_flags["enabled"] is True
    assert plugin_flags["one_step_rows"][0]["reactants"] == "CC.O"
    assert plugin_flags["one_step_rows"][0]["template"]["requires_audit"] is True


def test_web_runner_carries_literature_plugin_payload_into_search_flags():
    row = _compiled_row(product_smiles="CCO", reactants="CC.O")
    config = _route_config_from_payload(
        {
            "target_smiles": "CCO",
            "search_preset": "quick",
            "max_steps": 3,
            "chem_enzy_iterations": 2,
            "chem_enzy_expansion_topk": 4,
            "literature_template_plugin": {
                "enabled": True,
                "max_added": 2,
                "one_step_rows": [row],
                "template_cards": [],
            },
        },
        gpu=-1,
    )

    plugin_flags = config.search_flags["literature_template_plugin"]
    assert plugin_flags["enabled"] is True
    assert plugin_flags["one_step_rows"][0]["reactants"] == "CC.O"


def _compiled_row(*, product_smiles: str, reactants: str) -> dict:
    return {
        "reactants": reactants,
        "scores": 0.7,
        "costs": None,
        "template": {
            "model_full_name": PLUGIN_MODEL_FULL_NAME,
            "source": "literature_template_plugin",
            "source_model": "literature_template_plugin",
            "template_id": "compiled_segment_step",
            "evidence_refs": ["ev1"],
            "not_lab_procedure": True,
            "requires_audit": True,
            "no_solved_claim": True,
            "template_validation_report": {
                "allowed_for_one_step_source": True,
                "accepted": True,
                "reasons": [],
            },
            "template_applicability_report": {
                "target_smiles": product_smiles,
                "frontier_smiles": product_smiles,
            },
            "literature_template_trace": {
                "structured_segment_step": True,
                "requires_audit": True,
                "no_solved_claim": True,
            },
        },
        "templates": {},
        "model_full_name": PLUGIN_MODEL_FULL_NAME,
        "weight": 1.0,
        "reaction_domains": "literature_chemical",
        "literature_template_trace": {
            "structured_segment_step": True,
            "requires_audit": True,
            "no_solved_claim": True,
        },
        "source_policy_decision": "enabled_literature_template_plugin",
    }

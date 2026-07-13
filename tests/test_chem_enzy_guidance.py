import math
from types import SimpleNamespace

from cascade_planner.agent.chem_enzy_policy import apply_chem_enzy_search_policy
from cascade_planner.baselines.chem_enzy_guidance import (
    ChemEnzyGuidanceConfig,
    ChemEnzyGuidanceState,
    ChemEnzyGuidedOneStepWrapper,
    exclude_guided_terminal_blacklist,
    install_canonical_ancestor_cycle_filter,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig
from cascade_planner.baselines.literature_one_step_plugin import (
    LiteratureOneStepPluginConfig,
    LiteratureOneStepPluginState,
    LiteratureTemplateOneStepWrapper,
)
from cascade_planner.harness.tools import (
    _guided_policy_runtime_diagnostics,
    _literature_template_plugin_runtime_diagnostics,
)


def _policy(*, preferred: list[str], blacklist: list[str] | None = None) -> dict:
    return {
        "schema_version": "chem_enzy_search_policy.v1",
        "policy_id": "guided-test",
        "operator_id": "test",
        "case_id": "test-case",
        "evidence_refs": ["test:evidence"],
        "terminal_blacklist": list(blacklist or []),
        "anchor_whitelist": [],
        "preferred_subgoal": {
            "target": {"name": "target", "smiles": "CCOC"},
            "preferred_subgoals": preferred,
            "hypothetical_precursor_targets": [
                {
                    "smiles": preferred[0],
                    "precursor_set_smiles": preferred[0],
                    "allowed_use": "guided_search_subgoal_hint_only",
                    "not_parent_route_proof": True,
                }
            ],
        },
        "source_budget": {"preferred_precursor_smiles": preferred},
        "rerun_reason": "test executable guidance",
        "budget": {
            "max_reruns": 1,
            "max_iterations": 5,
            "max_depth": 3,
            "expansion_topk": 10,
        },
        "mode": "guided",
        "compiler_metadata": {"not_raw_reaction_injection": True},
    }


class _OneStep:
    one_step_models = {"stub": object()}

    def __init__(self, result: dict):
        self.result = result

    def run(self, _target: str, *args, **kwargs) -> dict:
        del args, kwargs
        return dict(self.result)


def test_apply_policy_compiles_executable_precursor_guidance_without_target_or_reaction_injection():
    config = apply_chem_enzy_search_policy(
        RouteSearchConfig(target_smiles="CCOC", max_iterations=20, max_depth=8, expansion_topk=50),
        _policy(preferred=["CCO.C"], blacklist=["CCC"]),
    )

    guidance = config.search_flags["chem_enzy_guidance"]
    assert guidance["enabled"] is True
    assert set(guidance["preferred_smiles"]) == {"C", "CCO"}
    assert guidance["preferred_precursor_sets"] == [["C", "CCO"]]
    assert "CCOC" not in guidance["preferred_smiles"]
    assert guidance["raw_reaction_injection"] is False
    assert config.search_flags["starting_molecule_exclusions"] == ["CCC"]
    assert config.search_flags["cascade_search_context"]["guidance_contract"]["raw_reaction_injection"] is False


def test_guided_wrapper_consumes_hint_and_changes_native_search_cost_order():
    config = ChemEnzyGuidanceConfig.from_flags(
        apply_chem_enzy_search_policy(RouteSearchConfig(target_smiles="CCOC"), _policy(preferred=["CCO.C"])).search_flags
    )
    state = ChemEnzyGuidanceState(config=config)
    wrapper = ChemEnzyGuidedOneStepWrapper(
        _OneStep(
            {
                "reactants": ["CCCO", "CCO.C"],
                "scores": [0.90, 0.10],
                "costs": [-math.log(0.90), -math.log(0.10)],
                "template": ["native-high", "native-hint"],
            }
        ),
        config=config,
        state=state,
    )

    result = wrapper.run("CCOC")
    stats = state.to_dict()

    assert result["reactants"][0] == "CCO.C"
    assert result["template"][0] == "native-hint"
    assert result["chem_enzy_guidance"][0]["exact_precursor_set_match"] is True
    assert result["costs"][0] < result["costs"][1]
    assert stats["hint_comparison_executed"] is True
    assert stats["ranking_signal_applied"] is True
    assert stats["exact_set_matches"] == 1
    assert stats["reranked_calls"] == 1
    assert stats["raw_reaction_injection"] is False


def test_guided_wrapper_hard_filters_self_loop_element_inventory_and_large_atom_jump():
    flags = apply_chem_enzy_search_policy(
        RouteSearchConfig(target_smiles="CCCCCCCCCCCCCCCCCC"),
        _policy(preferred=["CCCCCCCCCCCCCCCC.CC"]),
    ).search_flags
    config = ChemEnzyGuidanceConfig.from_flags(flags)
    state = ChemEnzyGuidanceState(config=config)
    wrapper = ChemEnzyGuidedOneStepWrapper(
        _OneStep(
            {
                "reactants": [
                    "CCCCCCCCCCCCCCCCCC",
                    "NNN",
                    "CC",
                    "CCCCCCCCCCCCCCCC.CC",
                ],
                "scores": [0.99, 0.98, 0.97, 0.20],
                "template": ["loop", "transmutation", "jump", "valid"],
            }
        ),
        config=config,
        state=state,
    )

    result = wrapper.run("CCCCCCCCCCCCCCCCCC")
    stats = state.to_dict()

    assert result["reactants"] == ["CCCCCCCCCCCCCCCC.CC"]
    assert stats["candidates_rejected"] == 3
    assert stats["rejected_by_reason"]["target_or_current_node_self_loop"] == 1
    assert stats["rejected_by_reason"]["element_inventory_not_conserved"] == 2
    assert stats["rejected_by_reason"]["large_atom_jump"] == 2


def test_disabled_guidance_still_runs_shared_structural_admission_without_reranking():
    base = _OneStep(
        {
            "reactants": ["CC=O", "C"],
            "scores": [0.4, 0.9],
            "costs": [0.8, 0.1],
            "template": ["balanced", "unbalanced"],
        }
    )
    config = ChemEnzyGuidanceConfig(enabled=False)
    state = ChemEnzyGuidanceState(config=config)
    wrapped = ChemEnzyGuidedOneStepWrapper(base, config=config, state=state)

    result = wrapped.run("CCO")

    assert result["reactants"] == ["CC=O"]
    assert result["template"] == ["balanced"]
    stats = state.to_dict()
    assert stats["enabled"] is False
    assert stats["hard_filter_executed"] is True
    assert stats["reranked_calls"] == 0
    assert stats["rejected_by_reason"] == {"element_inventory_not_conserved": 1}


def test_nirmatrelvir_bad_edge_is_rejected_before_moltree_with_bound_provenance():
    product = "O=C(O)C(O)(CCO)C(=O)OCc1ccccc1"
    reactants = "O=C(Cl)OCc1ccccc1.O=C([O-])[O-]"
    config = ChemEnzyGuidanceConfig(enabled=False)
    state = ChemEnzyGuidanceState(config=config)
    wrapped = ChemEnzyGuidedOneStepWrapper(
        _OneStep(
            {
                "reactants": [reactants],
                "scores": [0.99],
                "model_full_name": ["graphfp_models.test"],
                "source": ["native_graphfp"],
                "template": [
                    {
                        "template_id": "bad-nirmatrelvir-edge",
                        "model_full_name": "graphfp_models.test",
                        "source": "native_graphfp",
                    }
                ],
            }
        ),
        config=config,
        state=state,
    )

    result = wrapped.run(product)
    stats = state.to_dict()

    assert result["reactants"] == []
    assert stats["rejected_by_reason"] == {"element_inventory_not_conserved": 1}
    record = stats["rejected_candidate_audits"][0]
    assert record["stage"] == "pre_moltree_one_step"
    assert record["source"] == "native_graphfp"
    assert record["model"] == "graphfp_models.test"
    assert record["template"]["fields"]["template_id"] == "bad-nirmatrelvir-edge"
    assert len(record["edge_digest"]) == 64
    assert record["precursor_smiles_multiset"] == sorted(
        ["O=C(Cl)OCc1ccccc1", "O=C([O-])[O-]"]
    )


def test_reaction_family_and_retron_prior_only_reranks_matching_native_candidate():
    config = ChemEnzyGuidanceConfig(
        enabled=True,
        preferred_reaction_classes=("late stage esterification",),
        preferred_retrons=("alpha hydroxy acid",),
    )
    state = ChemEnzyGuidanceState(config=config)
    wrapped = ChemEnzyGuidedOneStepWrapper(
        _OneStep(
            {
                "reactants": ["CC=O", "OCCO"],
                "scores": [0.90, 0.80],
                "costs": [-math.log(0.90), -math.log(0.80)],
                "template": [
                    {"reaction_class": "oxidation"},
                    {
                        "reaction_class": "late_stage_esterification",
                        "product_retron": {"retron_type": "alpha-hydroxy-acid"},
                    },
                ],
            }
        ),
        config=config,
        state=state,
    )

    result = wrapped.run("CCO")
    stats = state.to_dict()

    assert result["reactants"][0] == "OCCO"
    prior = result["chem_enzy_guidance"][0]["reaction_prior"]
    assert prior["reaction_class_matches"] == ["late stage esterification"]
    assert prior["retron_matches"] == ["alpha hydroxy acid"]
    assert prior["applied"] is True
    assert prior["raw_reaction_injection"] is False
    assert stats["reaction_class_matches"] == 1
    assert stats["retron_matches"] == 1
    assert stats["reaction_prior_score_adjusted"] == 1
    assert stats["reaction_prior_applied"] is True


def test_unmatched_reaction_prior_is_observable_and_never_claimed_applied():
    config = ChemEnzyGuidanceConfig(
        enabled=True,
        preferred_reaction_classes=("esterification",),
        preferred_retrons=("alpha hydroxy acid",),
    )
    state = ChemEnzyGuidanceState(config=config)
    wrapped = ChemEnzyGuidedOneStepWrapper(
        _OneStep(
            {
                "reactants": ["CC=O", "OCCO"],
                "scores": [0.90, 0.80],
                "costs": [-math.log(0.90), -math.log(0.80)],
            }
        ),
        config=config,
        state=state,
    )

    result = wrapped.run("CCO")
    stats = state.to_dict()

    assert result["reactants"] == ["CC=O", "OCCO"]
    assert all(
        row["reaction_prior"]["applied"] is False
        for row in result["chem_enzy_guidance"]
    )
    assert stats["requested_reaction_prior_count"] == 2
    assert stats["reaction_prior_metadata_observed"] is False
    assert stats["reaction_prior_score_adjusted"] == 0
    assert stats["reaction_prior_applied"] is False


def test_repeated_precursors_preserve_stoichiometry_for_homocoupling_inventory():
    config = ChemEnzyGuidanceConfig.from_flags(
        {
            "chem_enzy_guidance": {
                "enabled": True,
                "preferred_smiles": ["CC"],
                "preferred_precursor_sets": ["CC.CC"],
            }
        }
    )
    state = ChemEnzyGuidanceState(config=config)
    wrapper = ChemEnzyGuidedOneStepWrapper(
        _OneStep(
            {
                "reactants": ["CC.CC"],
                "scores": [0.5],
                "template": ["native-homocoupling"],
            }
        ),
        config=config,
        state=state,
    )

    result = wrapper.run("CCCC")

    assert result["reactants"] == ["CC.CC"]
    assert result["chem_enzy_guidance"][0]["reactants"] == ["CC", "CC"]
    assert result["chem_enzy_guidance"][0]["exact_precursor_set_match"] is True
    assert state.candidates_rejected == 0


def test_terminal_blacklist_removes_only_matching_stock_terminal_and_never_adds_hint():
    flags = apply_chem_enzy_search_policy(
        RouteSearchConfig(target_smiles="CCOC"),
        _policy(preferred=["CCO.C"], blacklist=["CCC"]),
    ).search_flags
    config = ChemEnzyGuidanceConfig.from_flags(flags)
    state = ChemEnzyGuidanceState(config=config)

    filtered = exclude_guided_terminal_blacklist({"CCC", "CC", "O"}, state=state)

    assert filtered == {"CC", "O"}
    assert "CCO" not in filtered
    assert state.terminal_stock_exclusions_requested == 1
    assert state.terminal_stock_exclusions_removed == 1


def test_vendor_ancestor_filter_uses_canonical_identity_and_emits_observable_trace():
    class FakeTree:
        def __init__(self):
            self.cascade_expansion_trace = []
            self.added = []

        def _add_reaction_and_mol_nodes(
            self, cost, mols, parent, template, ancestors, cascade_annotation=None
        ):
            self.added.append((cost, list(mols), template, set(ancestors), cascade_annotation))
            return "added"

    module = SimpleNamespace(MolTree=FakeTree)
    install_canonical_ancestor_cycle_filter(module)
    tree = FakeTree()
    parent = SimpleNamespace(mol="CCN")

    blocked = tree._add_reaction_and_mol_nodes(
        1.0, ["C(C)O"], parent, "tpl", {"CCO"}, cascade_annotation={}
    )
    allowed = tree._add_reaction_and_mol_nodes(
        1.0, ["CCCN"], parent, "tpl", {"CCO"}, cascade_annotation={}
    )

    assert blocked is None
    assert allowed == "added"
    assert len(tree.added) == 1
    assert "ancestor_or_target_cycle" in tree.cascade_expansion_trace[0]["reasons"]
    assert tree.cascade_expansion_trace[0]["raw_reaction_injection"] is False


def test_vendor_insertion_gate_rejects_bad_edge_even_if_one_step_wrapper_is_bypassed():
    class FakeTree:
        def __init__(self):
            self.cascade_expansion_trace = []
            self.added = []

        def _add_reaction_and_mol_nodes(
            self, cost, mols, parent, template, ancestors, cascade_annotation=None
        ):
            self.added.append((cost, list(mols), template, set(ancestors)))
            return "added"

    module = SimpleNamespace(MolTree=FakeTree)
    install_canonical_ancestor_cycle_filter(module)
    tree = FakeTree()
    parent = SimpleNamespace(mol="O=C(O)C(O)(CCO)C(=O)OCc1ccccc1")

    blocked = tree._add_reaction_and_mol_nodes(
        0.01,
        ["O=C(Cl)OCc1ccccc1", "O=C([O-])[O-]"],
        parent,
        {
            "template_id": "bypassed-high-score-template",
            "source": "native_model",
            "model_full_name": "graphfp_models.test",
        },
        set(),
        cascade_annotation={"source": "native_model"},
    )

    assert blocked is None
    assert tree.added == []
    trace = tree.cascade_expansion_trace[0]
    assert trace["stage"] == "pre_moltree_insert"
    assert trace["reasons"] == ["element_inventory_not_conserved"]
    assert trace["source"] == "native_model"
    assert trace["model"] == "graphfp_models.test"
    assert trace["admission_record"]["decision"] == "rejected"
    assert len(trace["edge_digest"]) == 64


def test_vendor_prepare_expansion_preserves_exact_precursor_multiplicity():
    class FakeTree:
        def prepare_expansion(self, _parent, result, **_kwargs):
            # Reproduce the historical vendor behavior that discarded
            # multiplicity before insertion.
            reactants = [list(set(row.split("."))) for row in result["reactants"]]
            return reactants, result["costs"], result["template"], [None]

        def _add_reaction_and_mol_nodes(
            self, cost, mols, parent, template, ancestors, cascade_annotation=None
        ):
            return (cost, mols, parent, template, ancestors, cascade_annotation)

    module = SimpleNamespace(MolTree=FakeTree)
    install_canonical_ancestor_cycle_filter(module)
    tree = FakeTree()

    reactants, _costs, _templates, _annotations = tree.prepare_expansion(
        "CC",
        {
            "reactants": ["C.C"],
            "scores": [0.5],
            "costs": [0.7],
            "template": ["homocoupling"],
        },
    )

    assert reactants == [["C", "C"]]


def test_runtime_diagnostic_refuses_guided_label_when_adapter_or_plugin_was_not_loaded():
    request = {
        "chem_enzy_search_policy": _policy(preferred=["CCO.C"]),
        "literature_template_plugin": {
            "enabled": True,
            "one_step_rows": [{"reactants": "CCO.C"}],
        },
    }
    result = {"raw_backend_metadata": {}}
    plugin = _literature_template_plugin_runtime_diagnostics(result, request)
    diagnostic = _guided_policy_runtime_diagnostics(result, request, plugin_runtime=plugin)

    assert plugin["reasons"] == ["literature_template_plugin_backend_stats_missing"]
    assert diagnostic["guided_execution_confirmed"] is False
    assert diagnostic["execution_mode"] == "unguided_fallback"
    assert "guided_policy_adapter_not_loaded" in diagnostic["reasons"]
    assert "guided_execution_not_confirmed" in diagnostic["reasons"]


def test_literature_wrapper_records_production_invocation_for_runtime_diagnostics():
    config = LiteratureOneStepPluginConfig(
        enabled=True,
        use_default_template_cards=False,
        template_cards=(),
        one_step_rows=(),
    )
    state = LiteratureOneStepPluginState(config=config)
    wrapper = LiteratureTemplateOneStepWrapper(
        _OneStep({"reactants": [], "scores": [], "template": []}),
        config=config,
        state=state,
    )

    wrapper.run("CCOC")

    assert state.calls == 1


def test_runtime_diagnostic_confirms_observed_hint_comparisons():
    request = {"chem_enzy_search_policy": _policy(preferred=["CCO.C"])}
    result = {
        "raw_backend_metadata": {
            "chem_enzy_guidance": {
                "enabled": True,
                "requested_hint_count": 2,
                "calls": 1,
                "hint_comparisons": 4,
                "ranking_signal_applied": True,
                "hard_filter_executed": True,
            }
        }
    }

    diagnostic = _guided_policy_runtime_diagnostics(result, request, plugin_runtime={})

    assert diagnostic["guided_execution_confirmed"] is True
    assert diagnostic["execution_mode"] == "guided"
    assert diagnostic["ranking_guidance_confirmed"] is True
    assert diagnostic["reasons"] == []


def test_runtime_diagnostic_does_not_substitute_template_calls_for_precursor_hint_consumption():
    request = {
        "chem_enzy_search_policy": _policy(preferred=["CCO.C"]),
        "literature_template_plugin": {
            "enabled": True,
            "one_step_rows": [{"reactants": "CCO.C"}],
        },
    }
    result = {
        "raw_backend_metadata": {
            "chem_enzy_guidance": {
                "enabled": True,
                "requested_hint_count": 2,
                "calls": 0,
                "hint_comparisons": 0,
            },
            "literature_template_plugin": {
                "calls": 1,
                "added_candidates": 1,
            },
        }
    }
    plugin = _literature_template_plugin_runtime_diagnostics(result, request)

    diagnostic = _guided_policy_runtime_diagnostics(
        result,
        request,
        plugin_runtime=plugin,
    )

    assert diagnostic["template_plugin_guidance_confirmed"] is True
    assert diagnostic["ranking_guidance_confirmed"] is False
    assert diagnostic["guided_execution_confirmed"] is False
    assert diagnostic["execution_mode"] == "unguided_fallback"
    assert "preferred_precursor_hints_not_consumed" in diagnostic["reasons"]


def test_runtime_diagnostic_names_one_step_model_initialization_failure():
    request = {"chem_enzy_search_policy": _policy(preferred=["CCO.C"])}
    result = {
        "failures": [
            {
                "category": "backend_initialization_failed",
                "message": "ONMT checkpoint model path is missing",
            }
        ]
    }

    diagnostic = _guided_policy_runtime_diagnostics(result, request, plugin_runtime={})

    assert diagnostic["guided_execution_confirmed"] is False
    assert "guided_backend_initialization_failed" in diagnostic["reasons"]
    assert "guided_one_step_model_runtime_unavailable" in diagnostic["reasons"]

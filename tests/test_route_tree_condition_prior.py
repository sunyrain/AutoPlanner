import json

from cascade_planner.cascadeboard.route_export import slot_to_dict
from cascade_planner.route_tree.condition_prior import (
    BRENDA_CONDITION_PRIOR_CACHE_ENV,
    BRENDA_CONDITION_PRIOR_ENV,
    brenda_condition_prior_from_env,
    clear_brenda_condition_prior_cache,
)
from cascade_planner.route_tree.proposals import ProposalContext
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState
from cascade_planner.route_tree.search import NeuralGuidedAOSearch


def test_brenda_condition_prior_uses_exact_organism_when_enabled(tmp_path, monkeypatch):
    cache = tmp_path / "brenda.json"
    cache.write_text(
        json.dumps(
            {
                "1.13.11.34||": {"T_opt": 32.0, "pH_opt": 7.4},
                "1.13.11.34||Aspergillus niger": {"T_opt": 28.0, "pH_opt": 6.6},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(BRENDA_CONDITION_PRIOR_ENV, "1")
    monkeypatch.setenv(BRENDA_CONDITION_PRIOR_CACHE_ENV, str(cache))
    clear_brenda_condition_prior_cache()

    prior = brenda_condition_prior_from_env(
        ec="1.13.11.34",
        metadata={"evidence": {"organism": "Aspergillus niger"}},
    )

    assert prior is not None
    assert prior["temperature_c"] == 28.0
    assert prior["ph"] == 6.6
    assert prior["temperature_source"] == "brenda_ec4_organism"
    assert prior["ph_source"] == "brenda_ec4_organism"


def test_brenda_condition_prior_is_opt_in(tmp_path, monkeypatch):
    cache = tmp_path / "brenda.json"
    cache.write_text(json.dumps({"1.1.1.1||": {"T_opt": 30.0, "pH_opt": 7.0}}), encoding="utf-8")
    monkeypatch.setenv(BRENDA_CONDITION_PRIOR_ENV, "0")
    monkeypatch.setenv(BRENDA_CONDITION_PRIOR_CACHE_ENV, str(cache))
    clear_brenda_condition_prior_cache()

    assert brenda_condition_prior_from_env(ec="1.1.1.1") is None


def test_route_tree_contextualize_action_exports_brenda_condition_prior(tmp_path, monkeypatch):
    cache = tmp_path / "brenda.json"
    cache.write_text(json.dumps({"1.13.11.34||": {"T_opt": 31.0, "pH_opt": 7.2}}), encoding="utf-8")
    monkeypatch.setenv(BRENDA_CONDITION_PRIOR_ENV, "1")
    monkeypatch.setenv(BRENDA_CONDITION_PRIOR_CACHE_ENV, str(cache))
    clear_brenda_condition_prior_cache()

    search = NeuralGuidedAOSearch(retro_engine=None, controller=None, enzyme_sp_verifier=None)
    action = CandidateAction(
        product="CC=O",
        reactants=("CCO",),
        main_reactant="CCO",
        rxn_smiles="CCO>>CC=O",
        source="enzyme_precedent",
        ec="1.13.11.34",
    )

    updated = search._contextualize_action(action, ProposalContext(depth=0))

    assert updated.T == 31.0
    assert updated.pH == 7.2
    assert updated.metadata["condition_prior"]["source"] == "brenda_condition_prior"
    state = RouteTreeState.initial("CC=O").advance(
        leaf="CC=O",
        action=updated,
        next_open_leaves=(),
        score_delta=0.0,
    )
    slot = state.to_board().slots[0]
    exported = slot_to_dict(slot)
    assert exported["T"] == 31.0
    assert exported["pH"] == 7.2
    assert exported["condition_prior"]["temperature_source"] == "brenda_ec4_median"
    assert exported["condition_predictions"][0]["source"] == "brenda_condition_prior"

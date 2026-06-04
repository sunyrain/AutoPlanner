from cascade_planner.route_tree.proposals import ProposalContext
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState
from cascade_planner.route_tree.search import NeuralGuidedAOSearch
from cascade_planner.cascadeboard.route_export import route_result_to_dict


class _Score:
    def __init__(self, accepted: bool, score: float):
        self.accepted = accepted
        self.score = score
        self.threshold = 0.5

    def to_dict(self):
        return {
            "accepted": self.accepted,
            "score": self.score,
            "threshold": self.threshold,
            "schema_version": "fake",
        }


class _FakeEnzymeSPVerifier:
    def __init__(self, accepted: bool):
        self.accepted = accepted
        self.calls = []

    def score_action(self, *, product, action):
        self.calls.append((product, action.source, action.ec))
        return _Score(self.accepted, 0.9 if self.accepted else 0.1)


def test_enzyme_sp_v1_gate_rejects_low_score_enzymatic_action():
    product = "CCCCCCCC"
    state = RouteTreeState.initial(product)
    chemical = CandidateAction.from_candidate(
        product,
        {
            "main_reactant": "CCCC",
            "rxn_smiles": "CCCC>>CCCCCCCC",
            "score": 0.5,
            "source": "retrochimera",
        },
        source="retrochimera",
    )
    enzyme = CandidateAction.from_candidate(
        product,
        {
            "main_reactant": "CCCC",
            "rxn_smiles": "CCCC>>CCCCCCCC",
            "score": 0.99,
            "source": "enzyformer",
            "ec": "1.1.1.1",
        },
        source="enzyformer",
    )
    verifier = _FakeEnzymeSPVerifier(accepted=False)
    planner = NeuralGuidedAOSearch(retro_engine={}, controller=None, enzyme_sp_verifier=verifier)

    actions, contract_pruned, invalid_pruned = planner._filter_actions(
        state,
        product,
        [chemical, enzyme],
        ProposalContext(),
    )

    assert [action.source for action in actions] == ["retrochimera"]
    assert contract_pruned == 0
    assert invalid_pruned == 1
    assert verifier.calls == [(product, "enzyformer", "1.1.1.1")]
    assert planner.stats.enzyme_sp_verifier_scored == 1
    assert planner.stats.enzyme_sp_verifier_rejections == 1


def test_enzyme_sp_v1_gate_allows_high_score_bridge_supported_enzyme_action():
    product = "CCCCCCCC"
    state = RouteTreeState.initial(product)
    enzyme = CandidateAction.from_candidate(
        product,
        {
            "main_reactant": "CCCC",
            "rxn_smiles": "CCCC>>CCCCCCCC",
            "score": 0.99,
            "source": "v3_retrieval",
            "ec": "2.7.1.1",
        },
        source="v3_retrieval",
    )
    enzyme.metadata["source_gate"] = {
        "policy_reason": "bridge_gate_hits",
        "molecule_flags": {"bridge_gate_checked": True, "bridge_gate_hits": 2},
    }
    verifier = _FakeEnzymeSPVerifier(accepted=True)
    planner = NeuralGuidedAOSearch(retro_engine={}, controller=None, enzyme_sp_verifier=verifier)

    actions, contract_pruned, invalid_pruned = planner._filter_actions(
        state,
        product,
        [enzyme],
        ProposalContext(),
    )

    assert len(actions) == 1
    assert contract_pruned == 0
    assert invalid_pruned == 0
    assert actions[0].metadata["enzyme_sp_verifier_v1"]["score"] == 0.9
    assert planner.stats.enzyme_sp_verifier_scored == 1


def test_enzyme_sp_v1_gate_applies_to_enzyme_precedent_source():
    product = "CC=O"
    state = RouteTreeState.initial(product)
    enzyme = CandidateAction.from_candidate(
        product,
        {
            "main_reactant": "CCO",
            "rxn_smiles": "CCO>>CC=O",
            "score": 1.0,
            "source": "enzyme_precedent",
            "ec": "1.1.1.1",
        },
        source="enzyme_precedent",
    )
    verifier = _FakeEnzymeSPVerifier(accepted=False)
    planner = NeuralGuidedAOSearch(retro_engine={}, controller=None, enzyme_sp_verifier=verifier)

    actions, _contract_pruned, invalid_pruned = planner._filter_actions(
        state,
        product,
        [enzyme],
        ProposalContext(),
    )

    assert actions == []
    assert invalid_pruned == 1
    assert verifier.calls == [(product, "enzyme_precedent", "1.1.1.1")]
    assert planner.stats.enzyme_sp_verifier_scored == 1
    assert planner.stats.enzyme_sp_verifier_rejections == 1


class _AcceptedEnzymeRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        return [
            {
                "main_reactant": "CCO",
                "rxn_smiles": "CCO>>CC=O",
                "score": 1.0,
                "source": "enzyme_precedent",
                "ec": "1.1.1.1",
            }
        ]


def test_route_export_includes_enzyme_sp_v1_payload_for_selected_enzyme_action():
    verifier = _FakeEnzymeSPVerifier(accepted=True)
    planner = NeuralGuidedAOSearch(
        retro_engine={"enzyme_precedent": _AcceptedEnzymeRetro()},
        stock_checker=lambda smi: smi == "CCO",
        max_depth=1,
        branch_factor=1,
        expansion_budget=2,
        controller=None,
        enzyme_sp_verifier=verifier,
    )

    results = planner.search("CC=O", n_results=1)

    assert len(results) == 1
    payload = route_result_to_dict(results[0], stock_checker=lambda smi: smi == "CCO")
    step = payload["steps"][0]
    assert step["source"] == "enzyme_precedent"
    assert step["enzyme_sp_verifier_v1"]["accepted"] is True
    assert step["evidence"]["enzyme_sp_verifier_v1"]["score"] == 0.9

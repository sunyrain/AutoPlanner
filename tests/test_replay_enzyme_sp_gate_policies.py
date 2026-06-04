from scripts.replay_enzyme_sp_gate_policies import collect_actions, replay_target


class _Score:
    def __init__(self, accepted, score):
        self.accepted = accepted
        self.score = score


class _FakeScorer:
    def score_action(self, *, product, action):
        del product
        return _Score(accepted=action.ec == "1.1.1.1", score=0.9 if action.ec == "1.1.1.1" else 0.1)


def test_replay_sp_v1_hard_rejects_bad_enzyme_action():
    target = {
        "target_smiles": "CCO",
        "target_canonical": "CCO",
        "label": 1,
        "label_source": "fixture",
        "contexts": [
            {
                "context_id": "root_no_ec",
                "ec1": 0,
                "source_results": {
                    "retrochimera": {
                        "actions": [
                            {
                                "main_reactant": "CC",
                                "rxn_smiles": "CC>>CCO",
                                "source": "retrochimera",
                                "raw_score": 0.4,
                                "rank": 1,
                            }
                        ]
                    },
                    "enzyformer": {
                        "actions": [
                            {
                                "main_reactant": "CO",
                                "rxn_smiles": "CO>>CCO",
                                "source": "enzyformer",
                                "raw_score": 0.99,
                                "rank": 1,
                                "ec": "9.9.9.9",
                            }
                        ]
                    },
                },
            }
        ],
    }

    actions = collect_actions(target)
    ungated = replay_target(
        target,
        actions,
        policy="ungated_all",
        bridge_hit_count=1,
        scorer=_FakeScorer(),
        top_k=1,
        sp_penalty=0.35,
    )
    hard = replay_target(
        target,
        actions,
        policy="bridge_gate_v0_sp_v1_hard",
        bridge_hit_count=1,
        scorer=_FakeScorer(),
        top_k=1,
        sp_penalty=0.35,
    )

    assert ungated["selected_enzyme_target"] is True
    assert hard["selected_enzyme_target"] is False
    assert hard["sp_v1_rejections"] == 1


def test_replay_bridge_gate_rejects_enzyme_when_no_bridge_hit():
    target = {
        "target_smiles": "CCO",
        "label": 0,
        "contexts": [
            {
                "context_id": "root_no_ec",
                "ec1": 0,
                "source_results": {
                    "enzyformer": {
                        "actions": [
                            {
                                "main_reactant": "CO",
                                "rxn_smiles": "CO>>CCO",
                                "source": "enzyformer",
                                "raw_score": 0.99,
                                "rank": 1,
                                "ec": "1.1.1.1",
                            }
                        ]
                    }
                },
            }
        ],
    }

    actions = collect_actions(target)
    gated = replay_target(
        target,
        actions,
        policy="bridge_gate_v0",
        bridge_hit_count=0,
        scorer=_FakeScorer(),
        top_k=1,
        sp_penalty=0.35,
    )

    assert gated["selected_enzyme_target"] is False
    assert gated["rejected_actions"] == 1
    assert gated["rejections"][0]["reason"] == "no_bridge_hit"

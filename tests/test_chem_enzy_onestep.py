import cascade_planner.baselines.chem_enzy_onestep as onestep
from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider
from cascade_planner.baselines.graphfp_dualtower_fusion import fuse_graphfp_dualtower_rows


TBS_DEACETYLBUFOTALIN = "CC(C)(C)[Si](C)(C)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](O)C[C@]32O)C1"


class FakeOneStep:
    def run(self, product, topk=10):
        return {
            "reactants": ["CCO", "CC"],
            "scores": [0.1, 0.2],
            "model_full_name": ["graphfp_models.fake", "graphfp_models.fake"],
        }


class FakeMixedOneStep:
    def run(self, product, topk=10):
        return {
            "reactants": ["CCO", "CCN", "CO", "CN"],
            "scores": [0.9, 0.8, 0.7, 0.6],
            "model_full_name": [
                "graphfp_models.fake",
                "graphfp_models.fake",
                "onmt_models.bionav_one_step",
                "onmt_models.bionav_one_step",
            ],
        }


class FakeFusion:
    graphfp_topk = 2

    def __init__(self):
        self.fused_base_rows = []

    def dual_rows(self, product):
        return []

    def fuse(self, product, base_rows, dual_rows, *, output_k):
        self.fused_base_rows = list(base_rows)
        return list(base_rows)


def test_onestep_provider_prepends_source_supported_semisynthesis_rescue():
    provider = ChemEnzyOneStepProposalProvider(one_step=FakeOneStep())

    rows = provider.predict(TBS_DEACETYLBUFOTALIN, top_k=2)

    assert rows[0]["source"] == "autoplanner_semisynthesis_rescue"
    assert rows[0]["proposal_type"] == "source_supported_derivatization"
    assert rows[0]["proposal_gate"]["decision"] == "keep"
    assert "CC(C)(C)[Si](C)(C)Cl" in rows[0]["reactant_smiles"]
    assert len(rows) == 2


def test_graphfp_fusion_keeps_bionav_rows_in_mixed_provider(monkeypatch):
    fusion = FakeFusion()
    monkeypatch.setattr(onestep, "_graphfp_dualtower_fusion_from_env", lambda: fusion)
    monkeypatch.setenv(onestep.GRAPHFP_FUSION_PROTECTED_TOPK_ENV, "2")
    provider = ChemEnzyOneStepProposalProvider(one_step=FakeMixedOneStep())

    rows = provider.predict("CC=O", top_k=4)

    assert [row["source"] for row in fusion.fused_base_rows] == ["chem_enzy_graphfp", "chem_enzy_graphfp"]
    assert any(row["source"] == "chem_enzy_onmt" for row in rows)
    assert len(rows) == 4


def test_graphfp_dualtower_fusion_dedupes_and_promotes_agreement():
    base_rows = [
        {
            "reactant_smiles": ["CCO"],
            "reaction_smiles": "CCO>>CC=O",
            "rank": 1,
            "source": "chem_enzy_graphfp",
            "score": 0.9,
        },
        {
            "reactant_smiles": ["CCN"],
            "reaction_smiles": "CCN>>CC=O",
            "rank": 2,
            "source": "chem_enzy_graphfp",
            "score": 0.8,
        },
    ]
    dual_rows = [
        {
            "reactant_smiles": ["CCN"],
            "reaction_smiles": "CCN>>CC=O",
            "rank": 1,
            "source": "autoplanner_dualtower",
            "score": 0.7,
        }
    ]

    rows = fuse_graphfp_dualtower_rows(
        product="CC=O",
        base_rows=base_rows,
        dual_rows=dual_rows,
        output_k=2,
        mode="rrf",
    )

    assert rows[0]["reactant_smiles"] == ["CCN"]
    assert rows[0]["graphfp_rank"] == 2
    assert rows[0]["dualtower_rank"] == 1
    assert "autoplanner_dualtower" in rows[0]["fusion_sources"]

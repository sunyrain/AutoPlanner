from __future__ import annotations

from cascade_planner.cascade_search import enzyme_coverage_sidecar as sidecar_module


class _EmptyBridgeRetriever:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def retrieve(self, *_args, **_kwargs) -> list:
        return []


def test_missing_sp_model_keeps_advisory_enzyme_precedents(monkeypatch) -> None:
    def missing_scorer():
        raise FileNotFoundError("missing verifier fixture")

    monkeypatch.setattr(sidecar_module, "BridgeRetrieverV0", _EmptyBridgeRetriever)
    monkeypatch.setattr(sidecar_module, "EnzymeSPVerifierV1Scorer", missing_scorer)
    monkeypatch.setattr(
        sidecar_module,
        "retrieve_enzyme_precedents",
        lambda *_args, **_kwargs: [
            {
                "main_reactant": "CC",
                "reaction_smiles": "CC>>CCO",
                "source": "enzyme_precedent_fixture",
                "score": 0.73,
                "ec": "1.1.1.1",
            }
        ],
    )

    report = sidecar_module.build_enzyme_coverage_sidecar("CCO")

    assert report["error"] == ""
    assert report["candidate_count"] == 1
    assert report["sp_v1_requested"] is True
    assert report["sp_v1_available"] is False
    assert report["sp_v1_error"] == "FileNotFoundError: missing verifier fixture"
    assert report["sp_v1_accepted_count"] == 0
    assert report["top_advisory_candidates"][0]["main_reactant"] == "CC"
    assert report["top_advisory_candidates"][0]["enzyme_sp_verifier_v1"] is None
    assert report["semantics"][
        "missing_learned_verifier_does_not_erase_enzyme_precedent"
    ] is True

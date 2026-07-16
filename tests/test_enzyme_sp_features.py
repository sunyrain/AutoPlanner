from __future__ import annotations

import numpy as np

from cascade_planner.cascade_search import enzyme_sp_verifier_v1 as verifier_module


class _FixtureModel:
    def __init__(self) -> None:
        self.shape = None

    def predict_proba(self, matrix):
        self.shape = matrix.shape
        return np.asarray([[0.12, 0.88]], dtype=float)


def test_runtime_verifier_uses_dependency_light_feature_projection(monkeypatch) -> None:
    model = _FixtureModel()
    monkeypatch.setattr(
        verifier_module.joblib,
        "load",
        lambda _path: {"model": model, "threshold": {"threshold": 0.5}},
    )

    scorer = verifier_module.EnzymeSPVerifierV1Scorer("fixture.joblib")
    score = scorer.score_tuple(
        substrate_smiles="CC",
        product_smiles="CCO",
        ec_numbers=("1.1.1.1",),
    )

    assert model.shape == (1, 6175)
    assert score.score == 0.88
    assert score.threshold == 0.5
    assert score.accepted is True

"""Runtime scorer for enzyme substrate-product verifier v1.

The model scores a concrete enzymatic transformation tuple:

    substrate side + product side + EC evidence

During retrosynthesis search, ``CandidateAction.reactants`` are treated as the
forward substrate side and ``CandidateAction.product`` as the forward product.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


DEFAULT_ENZYME_SP_VERIFIER_V1_MODEL = Path(
    "results/shared/enzyme_sp_verifier_v1_20260528/enzyme_sp_verifier_v1_lgbm.joblib"
)
DEFAULT_ENZYME_SP_VERIFIER_V1_THRESHOLD = 0.36331207712759417


@dataclass(frozen=True)
class EnzymeSPVerifierV1Score:
    substrate_smiles: str
    product_smiles: str
    ec_numbers: tuple[str, ...]
    score: float
    threshold: float
    accepted: bool
    model_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "enzyme_sp_verifier_v1.runtime_score.v1",
            "substrate_smiles": self.substrate_smiles,
            "product_smiles": self.product_smiles,
            "ec_numbers": list(self.ec_numbers),
            "score": float(self.score),
            "threshold": float(self.threshold),
            "accepted": bool(self.accepted),
            "model_path": self.model_path,
        }


class EnzymeSPVerifierV1Scorer:
    """LightGBM runtime wrapper for the v1 enzyme-step verifier."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_ENZYME_SP_VERIFIER_V1_MODEL,
        *,
        threshold: float | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        model_threshold = _artifact_threshold(artifact)
        self.threshold = float(threshold if threshold is not None else model_threshold)

        from scripts.train_enzyme_sp_verifier_v1 import SideFeatureCache, build_matrix

        self._cache = SideFeatureCache()
        self._build_matrix = build_matrix

    def score_action(self, *, product: str, action: Any) -> EnzymeSPVerifierV1Score:
        substrate = ".".join(str(smi) for smi in getattr(action, "reactants", ()) if smi)
        product_smiles = str(product or getattr(action, "product", "") or "")
        ecs = _action_ec_numbers(action)
        return self.score_tuple(substrate_smiles=substrate, product_smiles=product_smiles, ec_numbers=ecs)

    def score_tuple(
        self,
        *,
        substrate_smiles: str,
        product_smiles: str,
        ec_numbers: list[str] | tuple[str, ...] | None = None,
    ) -> EnzymeSPVerifierV1Score:
        ecs = tuple(str(ec).strip() for ec in (ec_numbers or ()) if str(ec or "").strip())
        row = {
            "row_id": "runtime",
            "substrate_smiles": str(substrate_smiles or ""),
            "product_smiles": str(product_smiles or ""),
            "ec_numbers_json": json.dumps(list(ecs), ensure_ascii=False),
            "ec1": _ec1_from_ecs(ecs),
            "ec_count": len(set(ecs)),
            "label": 0,
            "label_type": "runtime_candidate",
            "label_weight": 1.0,
        }
        matrix, *_ = self._build_matrix([row], self._cache)
        score = float(self.model.predict_proba(matrix)[0, 1])
        return EnzymeSPVerifierV1Score(
            substrate_smiles=row["substrate_smiles"],
            product_smiles=row["product_smiles"],
            ec_numbers=ecs,
            score=score,
            threshold=self.threshold,
            accepted=bool(score >= self.threshold),
            model_path=str(self.model_path),
        )


def _artifact_threshold(artifact: dict[str, Any]) -> float:
    threshold = artifact.get("threshold")
    if isinstance(threshold, dict):
        value = threshold.get("threshold")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return DEFAULT_ENZYME_SP_VERIFIER_V1_THRESHOLD


def _action_ec_numbers(action: Any) -> tuple[str, ...]:
    values: list[str] = []
    ec = str(getattr(action, "ec", "") or "").strip()
    if ec:
        values.append(ec)
    metadata = getattr(action, "metadata", {}) or {}
    for key in ("ec_numbers", "enzyme_ec_sample", "enzyme_ec_numbers"):
        raw = metadata.get(key)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = [raw]
        else:
            parsed = raw
        if isinstance(parsed, (list, tuple, set)):
            values.extend(str(item).strip() for item in parsed if str(item or "").strip())
    return tuple(dict.fromkeys(values))


def _ec1_from_ecs(ecs: tuple[str, ...] | list[str]) -> str:
    for ec in ecs:
        head = str(ec or "").split(".", 1)[0]
        if head in {"1", "2", "3", "4", "5", "6", "7"}:
            return head
    return "unknown"

"""Bridge retriever and verifier gate for chemo-enzymatic search.

This module wraps the P0/P1 bridge pack into a small runtime component. It is
not a route generator by itself; it answers whether a chemical frontier molecule
has evidence-supported entry points into enzyme substrate/product space, and it
can gate those candidates with the verifier v0 model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

from cascade_planner.cascadeboard.route_recovery import canonical_smiles

RDLogger.DisableLog("rdApp.*")

DEFAULT_BRIDGE_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_BRIDGE_VERIFIER_MODEL = Path("results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_lgbm.joblib")
DEFAULT_BRIDGE_VERIFIER_THRESHOLD = 0.8409896871324669


@dataclass(frozen=True)
class BridgeCandidate:
    chemical_smiles: str
    enzyme_smiles: str
    chemical_inchikey: str
    enzyme_inchikey: str
    bridge_direction: str
    confidence_tier: str = ""
    source: str = ""
    tanimoto: float = 0.0
    enzyme_ec_sample: tuple[str, ...] = ()
    verifier_score: float | None = None
    verifier_pass: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chemical_smiles": self.chemical_smiles,
            "enzyme_smiles": self.enzyme_smiles,
            "chemical_inchikey": self.chemical_inchikey,
            "enzyme_inchikey": self.enzyme_inchikey,
            "bridge_direction": self.bridge_direction,
            "confidence_tier": self.confidence_tier,
            "source": self.source,
            "tanimoto": float(self.tanimoto),
            "enzyme_ec_sample": list(self.enzyme_ec_sample),
            "verifier_score": self.verifier_score,
            "verifier_pass": self.verifier_pass,
            "metadata": dict(self.metadata),
        }

    def to_scoring_row(self) -> dict[str, Any]:
        return {
            "chemical_smiles": self.chemical_smiles,
            "enzyme_smiles": self.enzyme_smiles,
            "chemical_inchikey": self.chemical_inchikey,
            "enzyme_inchikey": self.enzyme_inchikey,
            "bridge_direction": self.bridge_direction,
            "enzyme_ec_sample_json": json.dumps(list(self.enzyme_ec_sample)),
            "label": 0,
            "label_type": "candidate",
            "label_weight": 1.0,
            "tanimoto": float(self.tanimoto),
        }


class BridgeVerifierV0Scorer:
    """Runtime scorer for bridge verifier v0."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_BRIDGE_VERIFIER_MODEL,
        *,
        threshold: float = DEFAULT_BRIDGE_VERIFIER_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        from scripts.train_bridge_verifier_v0 import MoleculeCache, build_matrix

        self._cache = MoleculeCache()
        self._build_matrix = build_matrix

    def score_candidates(self, candidates: list[BridgeCandidate]) -> list[BridgeCandidate]:
        if not candidates:
            return []
        rows = [candidate.to_scoring_row() for candidate in candidates]
        matrix, *_ = self._build_matrix(rows, self._cache)
        scores = self.model.predict_proba(matrix)[:, 1]
        scored: list[BridgeCandidate] = []
        for candidate, score in zip(candidates, scores):
            scored.append(
                BridgeCandidate(
                    chemical_smiles=candidate.chemical_smiles,
                    enzyme_smiles=candidate.enzyme_smiles,
                    chemical_inchikey=candidate.chemical_inchikey,
                    enzyme_inchikey=candidate.enzyme_inchikey,
                    bridge_direction=candidate.bridge_direction,
                    confidence_tier=candidate.confidence_tier,
                    source=candidate.source,
                    tanimoto=candidate.tanimoto,
                    enzyme_ec_sample=candidate.enzyme_ec_sample,
                    verifier_score=float(score),
                    verifier_pass=bool(float(score) >= self.threshold),
                    metadata=dict(candidate.metadata),
                )
            )
        return scored


class BridgeRetrieverV0:
    """Exact/similarity bridge retriever backed by ``bridge_pack_v0``."""

    def __init__(
        self,
        pack_dir: str | Path = DEFAULT_BRIDGE_PACK_DIR,
        *,
        scorer: BridgeVerifierV0Scorer | None = None,
        scored_cache_path: str | Path | None = None,
        prefer_scored_cache: bool = True,
    ) -> None:
        self.pack_dir = Path(pack_dir)
        self.scorer = scorer
        self.scored_cache_path = Path(scored_cache_path) if scored_cache_path is not None else self.pack_dir / "bridge_candidates_scored.parquet"
        self.prefer_scored_cache = bool(prefer_scored_cache)
        self._exact_by_chemical: dict[str, list[BridgeCandidate]] = {}
        self._similar_by_chemical: dict[str, list[BridgeCandidate]] = {}
        self._load()

    def retrieve(
        self,
        smiles: str,
        *,
        top_k: int = 16,
        min_tanimoto: float = 0.0,
        require_verifier_pass: bool = False,
        include_exact: bool = True,
        include_similarity: bool = True,
    ) -> list[BridgeCandidate]:
        inchikey = inchikey_from_smiles(smiles)
        if not inchikey:
            return []
        candidates: list[BridgeCandidate] = []
        if include_exact:
            candidates.extend(self._exact_by_chemical.get(inchikey) or [])
        if include_similarity:
            candidates.extend(self._similar_by_chemical.get(inchikey) or [])
        if min_tanimoto > 0:
            candidates = [candidate for candidate in candidates if float(candidate.tanimoto) >= float(min_tanimoto)]
        candidates = _dedupe_candidates(candidates)
        if self.scorer is not None and any(candidate.verifier_score is None for candidate in candidates):
            candidates = self.scorer.score_candidates(candidates)
        if require_verifier_pass:
            candidates = [candidate for candidate in candidates if bool(candidate.verifier_pass)]
        candidates.sort(key=_candidate_sort_key, reverse=True)
        return candidates[: max(0, int(top_k or 0))]

    def retrieve_dicts(self, smiles: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [candidate.to_dict() for candidate in self.retrieve(smiles, **kwargs)]

    def _load(self) -> None:
        if self.prefer_scored_cache and self.scored_cache_path.exists():
            loaded = self._load_scored_cache(self.scored_cache_path)
            if loaded:
                return
        exact_path = self.pack_dir / "exact_bridge_strict.parquet"
        if exact_path.exists():
            for row in pq.read_table(exact_path).to_pylist():
                candidate = _exact_candidate(row)
                if candidate is not None:
                    self._exact_by_chemical.setdefault(candidate.chemical_inchikey, []).append(candidate)
        similarity_path = self.pack_dir / "similarity_bridge_filtered.parquet"
        if similarity_path.exists():
            for row in pq.read_table(similarity_path).to_pylist():
                candidate = _similarity_candidate(row)
                if candidate is not None:
                    self._similar_by_chemical.setdefault(candidate.chemical_inchikey, []).append(candidate)

    def _load_scored_cache(self, path: Path) -> int:
        loaded = 0
        for row in pq.read_table(path).to_pylist():
            candidate = _scored_candidate(row)
            if candidate is None:
                continue
            if candidate.source == "exact_bridge_strict":
                self._exact_by_chemical.setdefault(candidate.chemical_inchikey, []).append(candidate)
            else:
                self._similar_by_chemical.setdefault(candidate.chemical_inchikey, []).append(candidate)
            loaded += 1
        return loaded


def inchikey_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    try:
        return MolToInchiKey(mol)
    except Exception:
        return ""


def ec_sample(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item)
    if not value:
        return ()
    try:
        parsed = json.loads(str(value))
    except Exception:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed if item)


def _exact_candidate(row: dict[str, Any]) -> BridgeCandidate | None:
    smiles = canonical_smiles(str(row.get("canonical_smiles") or ""))
    inchikey = str(row.get("inchikey") or "")
    if not smiles or not inchikey:
        return None
    return BridgeCandidate(
        chemical_smiles=smiles,
        enzyme_smiles=smiles,
        chemical_inchikey=inchikey,
        enzyme_inchikey=inchikey,
        bridge_direction=str(row.get("bridge_direction") or ""),
        confidence_tier=str(row.get("confidence_tier") or ""),
        source="exact_bridge_strict",
        tanimoto=1.0,
        enzyme_ec_sample=ec_sample(row.get("enzyme_ec_sample_json")),
        metadata={
            "chemical_occurrences": int(row.get("chemical_occurrences") or 0),
            "enzyme_occurrences": int(row.get("enzyme_occurrences") or 0),
            "enzyme_ec_unique": int(row.get("enzyme_ec_unique") or 0),
            "bridge_flags_json": row.get("bridge_flags_json") or "[]",
        },
    )


def _similarity_candidate(row: dict[str, Any]) -> BridgeCandidate | None:
    chemical = canonical_smiles(str(row.get("chemical_smiles") or ""))
    enzyme = canonical_smiles(str(row.get("enzyme_smiles") or ""))
    chemical_key = str(row.get("chemical_inchikey") or "")
    enzyme_key = str(row.get("enzyme_inchikey") or "")
    if not chemical or not enzyme or not chemical_key or not enzyme_key:
        return None
    tanimoto = row.get("tanimoto")
    if tanimoto is None:
        tanimoto = _tanimoto(chemical, enzyme)
    return BridgeCandidate(
        chemical_smiles=chemical,
        enzyme_smiles=enzyme,
        chemical_inchikey=chemical_key,
        enzyme_inchikey=enzyme_key,
        bridge_direction=str(row.get("bridge_direction") or ""),
        confidence_tier=str(row.get("confidence_tier") or ""),
        source="similarity_bridge_filtered",
        tanimoto=float(tanimoto or 0.0),
        enzyme_ec_sample=ec_sample(row.get("enzyme_ec_sample_json")),
        metadata={
            "chemical_occurrences": int(row.get("chemical_occurrences") or 0),
            "enzyme_occurrences": int(row.get("enzyme_occurrences") or 0),
            "enzyme_ec_unique": int(row.get("enzyme_ec_unique") or 0),
        },
    )


def _scored_candidate(row: dict[str, Any]) -> BridgeCandidate | None:
    chemical = str(row.get("chemical_smiles") or "")
    enzyme = str(row.get("enzyme_smiles") or "")
    chemical_key = str(row.get("chemical_inchikey") or "")
    enzyme_key = str(row.get("enzyme_inchikey") or "")
    if not chemical or not enzyme or not chemical_key or not enzyme_key:
        return None
    metadata = {}
    metadata_json = row.get("metadata_json")
    if metadata_json:
        try:
            parsed = json.loads(str(metadata_json))
            if isinstance(parsed, dict):
                metadata = parsed
        except Exception:
            metadata = {}
    score = row.get("verifier_score")
    pass_value = row.get("verifier_pass")
    if pass_value is None and score is not None:
        pass_value = bool(float(score) >= DEFAULT_BRIDGE_VERIFIER_THRESHOLD)
    return BridgeCandidate(
        chemical_smiles=chemical,
        enzyme_smiles=enzyme,
        chemical_inchikey=chemical_key,
        enzyme_inchikey=enzyme_key,
        bridge_direction=str(row.get("bridge_direction") or ""),
        confidence_tier=str(row.get("confidence_tier") or ""),
        source=str(row.get("source") or ""),
        tanimoto=float(row.get("tanimoto") or 0.0),
        enzyme_ec_sample=ec_sample(row.get("enzyme_ec_sample_json") or row.get("enzyme_ec_sample")),
        verifier_score=float(score) if score is not None else None,
        verifier_pass=bool(pass_value) if pass_value is not None else None,
        metadata=metadata,
    )


def _tanimoto(left: str, right: str) -> float:
    from rdkit.Chem import AllChem

    left_mol = Chem.MolFromSmiles(left)
    right_mol = Chem.MolFromSmiles(right)
    if left_mol is None or right_mol is None:
        return 0.0
    left_fp = AllChem.GetMorganFingerprintAsBitVect(left_mol, 2, nBits=2048)
    right_fp = AllChem.GetMorganFingerprintAsBitVect(right_mol, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def _dedupe_candidates(candidates: list[BridgeCandidate]) -> list[BridgeCandidate]:
    out: list[BridgeCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (candidate.chemical_inchikey, candidate.enzyme_inchikey, candidate.bridge_direction)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _candidate_sort_key(candidate: BridgeCandidate) -> tuple[float, float, float, float]:
    exact_bonus = 1.0 if candidate.source == "exact_bridge_strict" else 0.0
    substrate_bonus = 1.0 if "substrate" in candidate.bridge_direction else 0.0
    verifier = candidate.verifier_score if candidate.verifier_score is not None else 0.0
    return (exact_bonus, verifier, substrate_bonus, float(candidate.tanimoto))

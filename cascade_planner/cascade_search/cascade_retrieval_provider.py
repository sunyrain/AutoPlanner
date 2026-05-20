"""v4 train-backed cascade transition retrieval provider.

This provider is intentionally narrow: it does not try to replace ChemEnzy as a
generic generator. It retrieves literature-supported cascade transitions/blocks
from the v4 train split so search can receive sparse cascade-specific proposal
supplements when the native route pool misses the cascade core.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascade_search.ids import stable_id
from cascade_planner.cascade_search.state import CascadeAction, CascadeActionType, StepAnnotation
from cascade_planner.cascadeboard.route_recovery import canonical_smiles


RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class CascadeRetrievalHit:
    """One retrieved v4 cascade transition/block precedent."""

    hit_id: str
    mode: str
    similarity: float
    rxn_smiles: str
    product_smiles: str
    reactants: tuple[str, ...]
    main_reactant: str
    transformation_superclass: str
    downstream_transformation_superclass: str
    transform_pair: str
    doi: str
    cascade_id: str
    program_id: str
    transition_id: str
    downstream_transition_id: str
    condition_tokens: tuple[str, ...]
    catalyst_classes: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit_id": self.hit_id,
            "mode": self.mode,
            "similarity": round(float(self.similarity), 6),
            "rxn_smiles": self.rxn_smiles,
            "product_smiles": self.product_smiles,
            "reactants": list(self.reactants),
            "main_reactant": self.main_reactant,
            "transformation_superclass": self.transformation_superclass,
            "downstream_transformation_superclass": self.downstream_transformation_superclass,
            "transform_pair": self.transform_pair,
            "doi": self.doi,
            "cascade_id": self.cascade_id,
            "program_id": self.program_id,
            "transition_id": self.transition_id,
            "downstream_transition_id": self.downstream_transition_id,
            "condition_tokens": list(self.condition_tokens),
            "catalyst_classes": list(self.catalyst_classes),
            "evidence": self.evidence,
        }

    def to_action(self, *, target_leaf: str = "") -> CascadeAction:
        step = StepAnnotation(
            product_smiles=self.product_smiles,
            reactant_smiles=list(self.reactants),
            rxn_smiles=self.rxn_smiles,
            source_model="cascade_retrieval_provider",
            score=float(self.similarity),
            reaction_type=self.transformation_superclass,
            evidence_confidence=float(self.similarity),
            raw_metadata={"cascade_retrieval_hit": self.to_dict()},
        )
        return CascadeAction(
            action_type=CascadeActionType.RETROSYNTHETIC_STEP,
            target_leaf=target_leaf or self.product_smiles,
            step=step,
            source="cascade_retrieval_provider",
            metadata={"cascade_retrieval": self.to_dict()},
        )


class CascadeRetrievalProvider:
    """Nearest-neighbor provider over v4 train transition/block evidence."""

    def __init__(self, program_manifest: str | Path):
        self.program_manifest = str(program_manifest)
        self._index = _load_index(self.program_manifest)

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self._index.get("summary") or {})

    def retrieve_for_product(
        self,
        product_smiles: str,
        *,
        mode: str = "step_product",
        limit: int = 10,
        min_similarity: float = 0.35,
        required_transform: str | None = None,
        required_downstream_transform: str | None = None,
        exclude_program_ids: set[str] | None = None,
    ) -> list[CascadeRetrievalHit]:
        """Retrieve v4 train transitions by product/open-leaf similarity."""
        query_fp = _fp(product_smiles)
        if query_fp is None:
            return []
        records = self._records_for_mode(mode)
        return _rank_records(
            query_fp,
            records,
            mode=mode,
            limit=limit,
            min_similarity=min_similarity,
            required_transform=required_transform,
            required_downstream_transform=required_downstream_transform,
            exclude_program_ids=exclude_program_ids or set(),
        )

    def retrieve_for_transition(
        self,
        product_smiles: str,
        main_reactant: str,
        *,
        mode: str = "transition",
        limit: int = 10,
        min_similarity: float = 0.35,
        required_transform: str | None = None,
        required_downstream_transform: str | None = None,
        exclude_program_ids: set[str] | None = None,
    ) -> list[CascadeRetrievalHit]:
        """Retrieve v4 train transitions by product+main-reactant similarity."""
        query_fp = _transition_fp(product_smiles, main_reactant)
        if query_fp is None:
            return []
        records = self._records_for_mode(mode)
        return _rank_records(
            query_fp,
            records,
            mode=mode,
            limit=limit,
            min_similarity=min_similarity,
            required_transform=required_transform,
            required_downstream_transform=required_downstream_transform,
            exclude_program_ids=exclude_program_ids or set(),
        )

    def _records_for_mode(self, mode: str) -> list[dict[str, Any]]:
        mode = str(mode or "step_product")
        if mode == "step_product":
            return self._index["step_product_records"]
        if mode == "transition":
            return self._index["transition_records"]
        if mode == "block_downstream_product":
            return self._index["block_downstream_product_records"]
        if mode == "block_downstream_transition":
            return self._index["block_downstream_transition_records"]
        raise ValueError(f"unknown cascade retrieval mode: {mode}")


def cascade_retrieval_provider_from_manifest(program_manifest: str | Path) -> CascadeRetrievalProvider:
    return CascadeRetrievalProvider(program_manifest)


@lru_cache(maxsize=4)
def _load_index(program_manifest: str) -> dict[str, Any]:
    manifest = json.loads(Path(program_manifest).read_text(encoding="utf-8"))
    train_path = Path((manifest.get("outputs") or {})["train"])
    rows = _read_jsonl(train_path)
    step_product_records = []
    transition_records = []
    block_downstream_product_records = []
    block_downstream_transition_records = []
    transform_counts: dict[str, int] = {}
    block_count = 0
    for program in rows:
        steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
        for idx, step in enumerate(steps):
            downstream = steps[idx + 1] if idx + 1 < len(steps) else None
            record = _record_from_step(program, step, downstream)
            if record is None:
                continue
            transform_counts[record["transformation_superclass"]] = transform_counts.get(record["transformation_superclass"], 0) + 1
            step_product_records.append(_with_fp(record, _fp(record.get("product_smiles"))))
            transition_records.append(_with_fp(record, _transition_fp(record.get("product_smiles"), record.get("main_reactant"))))
        for upstream, downstream in zip(steps, steps[1:]):
            record = _record_from_step(program, upstream, downstream)
            if record is None:
                continue
            block_count += 1
            block_downstream_product_records.append(_with_fp(record, _fp(downstream.get("product_smiles"))))
            block_downstream_transition_records.append(
                _with_fp(record, _transition_fp(downstream.get("product_smiles"), downstream.get("main_reactant")))
            )
    for records in (step_product_records, transition_records, block_downstream_product_records, block_downstream_transition_records):
        records[:] = [row for row in records if row.get("_fp") is not None]
    return {
        "step_product_records": step_product_records,
        "transition_records": transition_records,
        "block_downstream_product_records": block_downstream_product_records,
        "block_downstream_transition_records": block_downstream_transition_records,
        "summary": {
            "program_manifest": program_manifest,
            "train_path": str(train_path),
            "train_programs": len(rows),
            "step_product_records": len(step_product_records),
            "transition_records": len(transition_records),
            "block_downstream_product_records": len(block_downstream_product_records),
            "block_downstream_transition_records": len(block_downstream_transition_records),
            "train_blocks": block_count,
            "transform_counts": dict(sorted(transform_counts.items())),
        },
    }


def _record_from_step(program: dict[str, Any], step: dict[str, Any], downstream: dict[str, Any] | None) -> dict[str, Any] | None:
    product = canonical_smiles(str(step.get("product_smiles") or "")) or str(step.get("product_smiles") or "")
    reactants = tuple(canonical_smiles(str(value)) or str(value) for value in step.get("reactants") or [] if value)
    rxn = str(step.get("rxn_smiles") or "")
    if not product or not reactants or not rxn:
        return None
    transform = _norm_transform(step.get("transformation_superclass"))
    downstream_transform = _norm_transform((downstream or {}).get("transformation_superclass"))
    return {
        "hit_id": stable_id("cascade_retrieval", program.get("program_id"), step.get("transition_id"), (downstream or {}).get("transition_id")),
        "program_id": str(program.get("program_id") or ""),
        "doi": str(program.get("doi") or ""),
        "cascade_id": str(program.get("cascade_id") or ""),
        "transition_id": str(step.get("transition_id") or ""),
        "downstream_transition_id": str((downstream or {}).get("transition_id") or ""),
        "rxn_smiles": rxn,
        "product_smiles": product,
        "reactants": reactants,
        "main_reactant": canonical_smiles(str(step.get("main_reactant") or "")) or str(step.get("main_reactant") or ""),
        "transformation_superclass": transform,
        "downstream_transformation_superclass": downstream_transform,
        "transform_pair": f"{transform}->{downstream_transform}" if downstream else f"{transform}->",
        "condition_tokens": tuple(str(value) for value in step.get("condition_tokens") or []),
        "catalyst_classes": tuple(str(value) for value in step.get("catalyst_classes") or []),
        "evidence": {
            "source": "dataset_v4_release_train",
            "doi": program.get("doi"),
            "cascade_id": program.get("cascade_id"),
            "program_id": program.get("program_id"),
            "transition_id": step.get("transition_id"),
            "downstream_transition_id": (downstream or {}).get("transition_id"),
            "cascade_type": program.get("cascade_type"),
            "quality_tier": program.get("quality_tier"),
            "condition_tokens": step.get("condition_tokens") or [],
            "catalyst_classes": step.get("catalyst_classes") or [],
        },
    }


def _with_fp(record: dict[str, Any], fp: Any) -> dict[str, Any]:
    out = dict(record)
    out["_fp"] = fp
    return out


def _rank_records(
    query_fp: Any,
    records: list[dict[str, Any]],
    *,
    mode: str,
    limit: int,
    min_similarity: float,
    required_transform: str | None,
    required_downstream_transform: str | None,
    exclude_program_ids: set[str],
) -> list[CascadeRetrievalHit]:
    filtered = [
        row
        for row in records
        if row.get("_fp") is not None
        and row.get("program_id") not in exclude_program_ids
        and _required_transform_ok(row, required_transform, required_downstream_transform)
    ]
    if not filtered:
        return []
    scores = DataStructs.BulkTanimotoSimilarity(query_fp, [row["_fp"] for row in filtered])
    hits = []
    for row, score in zip(filtered, scores):
        if float(score) < min_similarity:
            continue
        hits.append(_hit_from_record(row, mode=mode, similarity=float(score)))
    hits.sort(key=lambda hit: (hit.similarity, bool(hit.downstream_transition_id), hit.transform_pair), reverse=True)
    return hits[: max(0, int(limit))]


def _hit_from_record(row: dict[str, Any], *, mode: str, similarity: float) -> CascadeRetrievalHit:
    evidence = dict(row.get("evidence") or {})
    evidence["retrieval_mode"] = mode
    evidence["similarity"] = float(similarity)
    return CascadeRetrievalHit(
        hit_id=str(row.get("hit_id") or ""),
        mode=mode,
        similarity=float(similarity),
        rxn_smiles=str(row.get("rxn_smiles") or ""),
        product_smiles=str(row.get("product_smiles") or ""),
        reactants=tuple(str(value) for value in row.get("reactants") or ()),
        main_reactant=str(row.get("main_reactant") or ""),
        transformation_superclass=str(row.get("transformation_superclass") or "unknown"),
        downstream_transformation_superclass=str(row.get("downstream_transformation_superclass") or "unknown"),
        transform_pair=str(row.get("transform_pair") or ""),
        doi=str(row.get("doi") or ""),
        cascade_id=str(row.get("cascade_id") or ""),
        program_id=str(row.get("program_id") or ""),
        transition_id=str(row.get("transition_id") or ""),
        downstream_transition_id=str(row.get("downstream_transition_id") or ""),
        condition_tokens=tuple(str(value) for value in row.get("condition_tokens") or ()),
        catalyst_classes=tuple(str(value) for value in row.get("catalyst_classes") or ()),
        evidence=evidence,
    )


def _required_transform_ok(
    row: dict[str, Any],
    required_transform: str | None,
    required_downstream_transform: str | None,
) -> bool:
    required = _norm_transform(required_transform)
    if required and required != "unknown" and _norm_transform(row.get("transformation_superclass")) != required:
        return False
    downstream = _norm_transform(required_downstream_transform)
    if downstream and downstream != "unknown" and _norm_transform(row.get("downstream_transformation_superclass")) != downstream:
        return False
    return True


def _transition_fp(product_smiles: Any, main_reactant: Any):
    product_fp = _fp(product_smiles)
    reactant_fp = _fp(main_reactant)
    if product_fp is None and reactant_fp is None:
        return None
    if product_fp is None:
        return reactant_fp
    if reactant_fp is None:
        return product_fp
    arr_product = np.zeros((2048,), dtype=np.int8)
    arr_reactant = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(product_fp, arr_product)
    DataStructs.ConvertToNumpyArray(reactant_fp, arr_reactant)
    arr = np.maximum(arr_product, arr_reactant)
    fp = DataStructs.ExplicitBitVect(2048)
    for bit in np.nonzero(arr)[0]:
        fp.SetBit(int(bit))
    return fp


def _fp(smiles: Any):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _norm_transform(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows
